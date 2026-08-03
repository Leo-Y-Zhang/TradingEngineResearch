"""Run the pre-registered PEAD sleeve, once, and print what it actually measured.

Registered design: `research/sleeves/pead_prereg.md`. Three holding horizons are
pre-declared and all three are printed regardless of what they say.

    .venv/Scripts/python.exe scripts/run_pead_sleeve.py

The bottom-decile book is printed as a NON-REGISTERED diagnostic. It exists to answer
"does SUE order returns at all", cannot change the verdict, and is labelled as such.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.capacity_panel import DEV_CUTOFF, load_prices  # noqa: E402
from research.sleeves import pead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("pead")

OUT_DIR = REPO / "research" / "sleeves" / "_pead_output"


def _select_decile(sue: pd.DataFrame, quantile: float, top: bool) -> pd.DataFrame:
    """Filings on the far side of a point-in-time decile breakpoint."""
    breakpoints = pead.decile_breakpoints(sue, quantile=quantile)
    month_start = sue["datekey"].dt.to_period("M").dt.to_timestamp()
    aligned = breakpoints.reindex(
        breakpoints.index.union(month_start.unique())
    ).sort_index().ffill().reindex(month_start.to_numpy())
    threshold = pd.Series(aligned.to_numpy(), index=sue.index)
    keep = threshold.notna() & ((sue["sue"] >= threshold) if top
                                else (sue["sue"] <= threshold))
    selected = sue[keep].copy()
    selected["threshold"] = threshold[keep]
    return selected


def _return_on_invested_capital(equity: pd.Series, exposure: pd.Series) -> pd.Series:
    """Daily return per unit of capital actually AT RISK.

    The registered sizing rule caps a position at 0.5% of equity, and with ~475 entries
    a year the book never finds enough signals to spend its cash. The whole-book return
    is therefore the return of a partly-invested account, which measures the sizing rule
    as much as the signal. Dividing the day's P&L by the exposure carried into that day
    isolates the sleeve. It is a decomposition of the recorded book, not a re-run with
    different parameters, and it is reported as a SECONDARY statistic.
    """
    # exposure_start[d] is the notional held at the close of d-1, which is exactly the
    # capital that earned equity[d] - equity[d-1]. No shift is needed.
    #
    # The guard matters: on the first days of the sample the book holds a handful of
    # dollars, and dividing a normal P&L by a near-zero denominator produced an
    # annualised volatility of 2e9 on the first attempt. Days below 1% of starting
    # capital carry no information about the sleeve and are dropped.
    floor = 0.01 * pead.START_CAPITAL
    usable = exposure >= floor
    pnl = equity.diff()
    return (pnl / exposure.where(usable))[usable].dropna()


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading DEV price panel")
    prices = load_prices()
    calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    logger.info("%s bars, %s trading days, %s .. %s", f"{len(prices):,}", len(calendar),
                calendar[0].date(), calendar[-1].date())
    assert calendar[-1] <= DEV_CUTOFF, "DEV guard breached"

    logger.info("building SUE from SF1 ARQ")
    sf1 = pead.load_sf1_arq()
    sue = pead.build_sue(sf1)
    logger.info("%s filings with a usable SUE, %s tickers",
                f"{len(sue):,}", sue["ticker"].nunique())

    # Global day index of the last trading day at or on the filing date. The entry is
    # the first bar STRICTLY after this.
    filing_day = np.searchsorted(calendar.to_numpy(), sue["datekey"].to_numpy(),
                                 side="right") - 1
    sue = sue.assign(filing_day=filing_day)
    sue = sue[sue["filing_day"] >= 0].reset_index(drop=True)

    books = {
        "top_decile": _select_decile(sue, pead.TOP_DECILE, top=True),
        "bottom_decile_DIAGNOSTIC": _select_decile(sue, 1.0 - pead.TOP_DECILE,
                                                   top=False),
    }
    for name, frame in books.items():
        logger.info("%s: %s candidate filings", name, f"{len(frame):,}")

    needed = set(books["top_decile"]["ticker"]) | set(
        books["bottom_decile_DIAGNOSTIC"]["ticker"])
    logger.info("building per-ticker bar arrays for %s tickers", f"{len(needed):,}")
    bars = pead.build_ticker_bars(prices[prices["ticker"].isin(needed)], calendar)
    del prices

    delistings = pd.read_parquet(pead.PANEL_DIR / "delistings.parquet")
    terminal = {row.ticker: (row.date, float(row.terminal_return))
                for row in delistings.itertuples(index=False)}

    results: list[dict] = []
    benchmark_cache: dict[tuple, pd.Series] = {}

    for book_name, candidates in books.items():
        candidates = candidates.sort_values("filing_day").reset_index(drop=True)
        for horizon in pead.HOLDING_HORIZONS:
            logger.info("--- %s, %sd hold ---", book_name, horizon)
            positions, rejects = pead.build_positions(
                candidates, bars, calendar, terminal, horizon)
            if len(positions) == 0:
                logger.warning("no positions")
                continue

            net = pead.simulate_book(positions, calendar, costs_on=True)
            gross = pead.simulate_book(positions, calendar, costs_on=False)

            # Restrict the equity curve to the period the book is actually live.
            first = int(np.min(positions.entry_day))
            last = int(np.max(positions.exit_day))
            span = calendar[first:last + 1]
            net_equity = net.equity.loc[span]
            gross_equity = gross.equity.loc[span]

            net_monthly = pead.monthly_from_equity(net_equity)
            gross_monthly = pead.monthly_from_equity(gross_equity)

            key = (span[0], span[-1])
            if key not in benchmark_cache:
                benchmark_cache[key] = pead.universe_benchmark(
                    set(bars), span[0], span[-1])
            benchmark_monthly = benchmark_cache[key]

            shared = net_monthly.index.intersection(benchmark_monthly.index)
            net_stats = pead.summarise(net_monthly.loc[shared], net_equity)
            gross_stats = pead.summarise(gross_monthly.loc[shared], gross_equity)
            bench_stats = pead.summarise(benchmark_monthly.loc[shared])
            assert len(shared) > 150, f"benchmark alignment collapsed: {len(shared)}"

            # SECONDARY, clearly labelled: the same book expressed per unit of capital
            # at risk, which strips out the idle cash the sizing rule left behind.
            invested_daily = _return_on_invested_capital(
                net.equity.loc[span], net.exposure_start.loc[span])
            invested_curve = (1.0 + invested_daily).cumprod()
            invested_monthly = pead.monthly_from_equity(invested_curve)
            invested_shared = invested_monthly.index.intersection(shared)
            invested_stats = pead.summarise(invested_monthly.loc[invested_shared],
                                            invested_curve)

            years = len(shared) / 12.0
            mean_equity = float(net_equity.mean())
            turnover = ((net.total_bought + net.total_sold)
                        / (2.0 * mean_equity * years))
            cost_drag = gross_stats.annual_return - net_stats.annual_return

            row = {
                "book": book_name,
                "horizon_days": horizon,
                "entries": len(positions),
                "entries_per_year": len(positions) / years,
                "entry_days_per_year": net.entry_days / years,
                "mean_concurrent_positions": net.mean_concurrent,
                "mean_position_notional": net.mean_position_notional,
                "mean_cash_weight": net.mean_cash_weight,
                "turnover_annual": turnover,
                "gross_return_annual": gross_stats.annual_return,
                "gross_sharpe": gross_stats.sharpe,
                "net_return_annual": net_stats.annual_return,
                "net_volatility": net_stats.annual_volatility,
                "net_sharpe": net_stats.sharpe,
                "max_drawdown": net_stats.max_drawdown,
                "benchmark_return_annual": bench_stats.annual_return,
                "benchmark_volatility": bench_stats.annual_volatility,
                "benchmark_sharpe": bench_stats.sharpe,
                "excess_over_benchmark_annual": (net_stats.annual_return
                                                 - bench_stats.annual_return),
                "cost_drag_annual": cost_drag,
                "invested_return_annual": invested_stats.annual_return,
                "invested_volatility": invested_stats.annual_volatility,
                "invested_sharpe": invested_stats.sharpe,
                "invested_excess_annual": (invested_stats.annual_return
                                           - bench_stats.annual_return),
                "mean_position_gross_return": float(np.mean(positions.gross_return)),
                # Per-bet accounting, which is the statement of the finding that is
                # immune to how much cash the sizing rule left idle.
                "benchmark_return_over_horizon": (
                    (1.0 + bench_stats.annual_return) ** (horizon / 252.0) - 1.0),
                "gross_alpha_per_bet": (
                    float(np.mean(positions.gross_return))
                    - ((1.0 + bench_stats.annual_return) ** (horizon / 252.0) - 1.0)),
                "cost_per_roundtrip_bps": (
                    1e4 * net.total_cost / net.total_bought
                    if net.total_bought > 0 else float("nan")),
                "median_spread_bps": float(np.median(positions.spread) * 1e4),
                "median_dollar_volume": float(np.median(
                    positions.median_dollar_volume)),
                "delisted_positions": int(np.sum(positions.delisted)),
                "truncated_positions": int(np.sum(positions.truncated)),
                "open_at_end": net.open_at_end,
                "months": len(shared),
                "start": str(span[0].date()),
                "end": str(span[-1].date()),
                "rejects": rejects,
            }
            results.append(row)
            logger.info(
                "entries=%s/yr concurrent=%.0f net=%.2f%% bench=%.2f%% "
                "excess=%.2f%% sharpe=%.2f cost=%.2f%% turnover=%.1fx",
                f"{row['entries_per_year']:.0f}", row["mean_concurrent_positions"],
                100 * row["net_return_annual"], 100 * row["benchmark_return_annual"],
                100 * row["excess_over_benchmark_annual"], row["net_sharpe"],
                100 * row["cost_drag_annual"], row["turnover_annual"])

            pead.PositionSet.to_frame(positions).to_parquet(
                OUT_DIR / f"positions_{book_name}_{horizon}d.parquet", index=False)
            net_equity.to_frame("equity").to_parquet(
                OUT_DIR / f"equity_{book_name}_{horizon}d.parquet")

    frame = pd.DataFrame(results)
    frame.drop(columns=["rejects"]).to_csv(OUT_DIR / "pead_results.csv", index=False)
    (OUT_DIR / "pead_results.json").write_text(json.dumps(results, indent=2, default=str))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n=== PEAD SLEEVE — registered single run (PRIMARY: whole book) ===")
    print(frame[["book", "horizon_days", "entries_per_year", "entry_days_per_year",
                 "mean_concurrent_positions", "mean_cash_weight", "turnover_annual",
                 "gross_sharpe", "net_return_annual", "net_volatility", "net_sharpe",
                 "max_drawdown", "benchmark_return_annual",
                 "excess_over_benchmark_annual",
                 "cost_drag_annual"]].to_string(index=False))
    print("\n--- SECONDARY (decomposition, not a re-run): per unit of invested capital")
    print(frame[["book", "horizon_days", "invested_return_annual",
                 "invested_volatility", "invested_sharpe", "invested_excess_annual",
                 "mean_position_notional", "median_spread_bps",
                 "median_dollar_volume"]].to_string(index=False))
    print("\n--- PER-BET accounting (immune to the cash weight)")
    per_bet = frame[["book", "horizon_days", "mean_position_gross_return",
                     "benchmark_return_over_horizon", "gross_alpha_per_bet",
                     "cost_per_roundtrip_bps"]].copy()
    per_bet["gross_alpha_bps"] = 1e4 * per_bet["gross_alpha_per_bet"]
    per_bet["net_alpha_bps"] = (per_bet["gross_alpha_bps"]
                                - per_bet["cost_per_roundtrip_bps"])
    print(per_bet.to_string(index=False))
    print("\nrejection reasons (top decile, 20d):")
    for row in results:
        if row["book"] == "top_decile" and row["horizon_days"] == 20:
            for reason, num in sorted(row["rejects"].items(),
                                      key=lambda item: -item[1]):
                print(f"  {reason:<24} {num:>9,}")
    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
