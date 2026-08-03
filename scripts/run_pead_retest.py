"""Run the pre-registered PEAD RE-TEST on the corrected universe, once.

Registered design: `research/sleeves/pead_retest_prereg.md` (written before this ran).
Three horizons pre-declared, both cost bounds reported, headline verdict at 40 days.

    .venv/Scripts/python.exe scripts/run_pead_retest.py

Writes derived statistics only. No Sharadar row leaves `_data/`.
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
from research.sleeves import pead, pead_retest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("pead_retest")

OUT_DIR = REPO / "research" / "sleeves" / "_pead_retest_output"

BOUNDS = ("conservative", "realistic")


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


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading DEV price panel")
    prices = load_prices()
    calendar = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    assert calendar[-1] <= DEV_CUTOFF, "DEV guard breached"
    logger.info("%s bars, %s trading days, %s .. %s", f"{len(prices):,}", len(calendar),
                calendar[0].date(), calendar[-1].date())

    logger.info("building SUE from SF1 ARQ")
    sf1 = pead.load_sf1_arq()
    sue = pead.build_sue(sf1)
    logger.info("%s filings with a usable SUE, %s tickers",
                f"{len(sue):,}", sue["ticker"].nunique())

    filing_day = np.searchsorted(calendar.to_numpy(), sue["datekey"].to_numpy(),
                                 side="right") - 1
    sue = sue.assign(filing_day=filing_day)
    sue = sue[sue["filing_day"] >= 0].reset_index(drop=True)

    books = {
        "top_decile": _select_decile(sue, pead.TOP_DECILE, top=True),
        "bottom_decile_DIAGNOSTIC": _select_decile(sue, 1.0 - pead.TOP_DECILE, top=False),
    }
    for name, frame in books.items():
        logger.info("%s: %s candidate filings", name, f"{len(frame):,}")

    needed = set().union(*(set(f["ticker"]) for f in books.values()))
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
        logger.info("screening %s candidates for %s", f"{len(candidates):,}", book_name)
        screens, screen_rejects = pead_retest.screen_all(candidates, bars, calendar)
        passed = sum(1 for s in screens.values() if s.passed)
        logger.info("  %s/%s distinct (ticker, filing) screens passed",
                    f"{passed:,}", f"{len(screens):,}")

        horizons = (pead_retest.HOLDING_HORIZONS if book_name == "top_decile"
                    else (pead_retest.HEADLINE_HORIZON,))

        for horizon in horizons:
            logger.info("--- %s, %sd hold ---", book_name, horizon)
            positions, rejects = pead_retest.build_positions(
                candidates, bars, calendar, terminal, horizon, screens)
            if len(positions) == 0:
                logger.warning("no positions")
                continue

            spread_cons = np.asarray(positions.spread_conservative, dtype=np.float64)
            spread_real = np.asarray(positions.spread_realistic, dtype=np.float64)

            gross_book = pead_retest.simulate_book(positions, calendar, spread=None)
            books_by_bound = {
                "conservative": pead_retest.simulate_book(positions, calendar,
                                                          spread=spread_cons),
                "realistic": pead_retest.simulate_book(positions, calendar,
                                                       spread=spread_real),
            }

            first = int(np.min(positions.entry_day))
            last = int(np.max(positions.exit_day))
            span = calendar[first:last + 1]

            key = (span[0], span[-1])
            if key not in benchmark_cache:
                benchmark_cache[key] = pead_retest.universe_benchmark(
                    set(bars), span[0], span[-1])
            benchmark_monthly = benchmark_cache[key]

            gross_equity = gross_book.equity.loc[span]
            gross_monthly = pead.monthly_from_equity(gross_equity)
            shared = gross_monthly.index.intersection(benchmark_monthly.index)
            assert len(shared) > 150, f"benchmark alignment collapsed: {len(shared)}"
            gross_stats = pead.summarise(gross_monthly.loc[shared], gross_equity)
            bench_stats = pead.summarise(benchmark_monthly.loc[shared])

            years = len(shared) / 12.0
            bench_over_horizon = ((1.0 + bench_stats.annual_return) ** (horizon / 252.0)
                                  - 1.0)
            mean_gross_bet = float(np.mean(positions.gross_return))
            gross_alpha_bps = 1e4 * (mean_gross_bet - bench_over_horizon)

            regimes = pd.Series(positions.regime)
            row = {
                "book": book_name,
                "horizon_days": horizon,
                "entries": len(positions),
                "entries_per_year": len(positions) / years,
                "entry_days_per_year": gross_book.entry_days / years,
                "mean_concurrent_positions": gross_book.mean_concurrent,
                "months": len(shared),
                "years": years,
                "start": str(span[0].date()),
                "end": str(span[-1].date()),
                "gross_return_annual": gross_stats.annual_return,
                "gross_volatility": gross_stats.annual_volatility,
                "gross_sharpe": gross_stats.sharpe,
                "benchmark_return_annual": bench_stats.annual_return,
                "benchmark_volatility": bench_stats.annual_volatility,
                "benchmark_sharpe": bench_stats.sharpe,
                "mean_position_gross_return": mean_gross_bet,
                "benchmark_return_over_horizon": bench_over_horizon,
                "gross_alpha_bps_per_bet": gross_alpha_bps,
                "median_spread_conservative_bps": float(np.median(spread_cons) * 1e4),
                "median_spread_realistic_bps": float(np.median(spread_real) * 1e4),
                "median_dollar_volume": float(np.median(
                    positions.median_dollar_volume)),
                "share_entries_upper_bound": float(
                    (regimes == "upper_bound").mean()),
                "share_entries_measured": float((regimes == "measured").mean()),
                "delisted_positions": int(np.sum(positions.delisted)),
                "truncated_positions": int(np.sum(positions.truncated)),
                "open_at_end": gross_book.open_at_end,
                "mean_position_notional": gross_book.mean_position_notional,
                "median_position_notional": gross_book.median_position_notional,
                "rejects": rejects,
                "screen_rejects": screen_rejects,
            }

            for bound in BOUNDS:
                book = books_by_bound[bound]
                equity = book.equity.loc[span]
                monthly = pead.monthly_from_equity(equity)
                stats = pead.summarise(monthly.loc[shared], equity)
                mean_equity = float(equity.mean())
                turnover = (book.total_bought + book.total_sold) / (2.0 * mean_equity
                                                                    * years)
                cost_rt_bps = (1e4 * book.total_cost / book.total_bought
                               if book.total_bought > 0 else float("nan"))
                concentration = pead_retest.pnl_concentration(positions, book, calendar)
                prefix = f"{bound}_"
                row.update({
                    prefix + "net_return_annual": stats.annual_return,
                    prefix + "net_volatility": stats.annual_volatility,
                    prefix + "net_sharpe": stats.sharpe,
                    prefix + "max_drawdown": stats.max_drawdown,
                    prefix + "excess_over_benchmark_annual": (stats.annual_return
                                                              - bench_stats.annual_return),
                    prefix + "cost_drag_annual": (gross_stats.annual_return
                                                  - stats.annual_return),
                    prefix + "turnover_annual": turnover,
                    prefix + "cost_per_roundtrip_bps": cost_rt_bps,
                    prefix + "cover_ratio": (gross_alpha_bps / cost_rt_bps
                                             if cost_rt_bps > 0 else float("nan")),
                    prefix + "net_alpha_bps_per_bet": gross_alpha_bps - cost_rt_bps,
                    prefix + "bets_per_unit_turnover": ((len(positions) / years)
                                                        / turnover if turnover > 0
                                                        else float("nan")),
                    prefix + "mean_cash_weight": book.mean_cash_weight,
                    prefix + "total_cost": book.total_cost,
                    prefix + "cost_share_spread": (book.total_spread_cost
                                                   / book.total_cost
                                                   if book.total_cost > 0 else float("nan")),
                    prefix + "cost_share_impact": (book.total_impact_cost
                                                   / book.total_cost
                                                   if book.total_cost > 0 else float("nan")),
                    prefix + "cost_share_commission": (book.total_commission
                                                       / book.total_cost
                                                       if book.total_cost > 0
                                                       else float("nan")),
                    prefix + "orders": book.orders,
                    prefix + "orders_at_035_minimum": book.orders_at_minimum,
                    prefix + "share_orders_at_035_minimum": (book.orders_at_minimum
                                                             / book.orders
                                                             if book.orders else 0.0),
                    prefix + "sharpe_by_decade": pead_retest.sharpe_by_decade(
                        monthly.loc[shared]),
                    prefix + "pnl_concentration": concentration,
                })

                equity.to_frame("equity").to_parquet(
                    OUT_DIR / f"equity_{book_name}_{horizon}d_{bound}.parquet")
                daily = equity.pct_change().dropna()
                daily.to_frame("net_return").to_parquet(
                    OUT_DIR / f"daily_net_returns_{book_name}_{horizon}d_{bound}.parquet")

            if book_name == "top_decile":
                verdict, inverted = pead_retest.registered_verdict(
                    row["conservative_excess_over_benchmark_annual"],
                    row["conservative_net_sharpe"],
                    row["realistic_excess_over_benchmark_annual"],
                    row["realistic_net_sharpe"],
                )
                row["registered_verdict"] = verdict
                row["bounds_sharpe_inverted"] = inverted
            else:
                row["registered_verdict"] = "DIAGNOSTIC_NOT_GATE_ELIGIBLE"
                row["bounds_sharpe_inverted"] = False

            results.append(row)
            pead_retest.PositionSet.to_frame(positions).to_parquet(
                OUT_DIR / f"positions_{book_name}_{horizon}d.parquet", index=False)

            logger.info(
                "entries=%.0f/yr gross_sharpe=%.2f | (a) net=%.2f%% sh=%.2f exc=%.2f%% "
                "cost=%.0fbp cover=%.2f | (b) net=%.2f%% sh=%.2f exc=%.2f%% cost=%.0fbp "
                "cover=%.2f",
                row["entries_per_year"], row["gross_sharpe"],
                100 * row["conservative_net_return_annual"],
                row["conservative_net_sharpe"],
                100 * row["conservative_excess_over_benchmark_annual"],
                row["conservative_cost_per_roundtrip_bps"], row["conservative_cover_ratio"],
                100 * row["realistic_net_return_annual"], row["realistic_net_sharpe"],
                100 * row["realistic_excess_over_benchmark_annual"],
                row["realistic_cost_per_roundtrip_bps"], row["realistic_cover_ratio"])

    frame = pd.DataFrame(results)
    drop = ["rejects", "screen_rejects", "conservative_sharpe_by_decade",
            "realistic_sharpe_by_decade", "conservative_pnl_concentration",
            "realistic_pnl_concentration"]
    frame.drop(columns=[c for c in drop if c in frame]).to_csv(
        OUT_DIR / "pead_retest_results.csv", index=False)
    (OUT_DIR / "pead_retest_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    _report(results)
    print(f"\nelapsed {time.time() - started:.0f}s")
    return 0


def _report(results: list[dict]) -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    frame = pd.DataFrame(results)
    top = frame[frame["book"] == "top_decile"]

    print("\n" + "=" * 96)
    print("PEAD RE-TEST on the CORRECTED universe -- registered single run, n_trials 33")
    print("=" * 96)

    headline = top[top["horizon_days"] == pead_retest.HEADLINE_HORIZON]
    if not headline.empty:
        row = headline.iloc[0]
        print(f"\nHEADLINE ({pead_retest.HEADLINE_HORIZON}d hold, declared in advance): "
              f"{row['registered_verdict']}")
        for bound in BOUNDS:
            print(f"  ({bound[:1]}) {bound:<13} net {100 * row[bound + '_net_return_annual']:>7.2f}%/yr  "
                  f"vol {100 * row[bound + '_net_volatility']:>6.2f}%  "
                  f"Sharpe {row[bound + '_net_sharpe']:>6.3f}  "
                  f"excess {100 * row[bound + '_excess_over_benchmark_annual']:>7.2f}%/yr  "
                  f"cover {row[bound + '_cover_ratio']:>5.2f}")
        print(f"      benchmark {100 * row['benchmark_return_annual']:.2f}%/yr "
              f"Sharpe {row['benchmark_sharpe']:.3f} | gross Sharpe {row['gross_sharpe']:.3f}")

    print("\n--- ALL HORIZONS, both bounds (nothing selected on)")
    columns = ["book", "horizon_days", "entries_per_year", "gross_sharpe",
               "gross_alpha_bps_per_bet",
               "conservative_cost_per_roundtrip_bps", "conservative_cover_ratio",
               "conservative_net_sharpe", "conservative_excess_over_benchmark_annual",
               "realistic_cost_per_roundtrip_bps", "realistic_cover_ratio",
               "realistic_net_sharpe", "realistic_excess_over_benchmark_annual",
               "registered_verdict"]
    print(frame[columns].to_string(index=False))

    print("\n--- TRADED UNIVERSE CHARACTER (H2: median traded spread below 60bps?)")
    print(frame[["book", "horizon_days", "median_spread_conservative_bps",
                 "median_spread_realistic_bps", "median_dollar_volume",
                 "share_entries_upper_bound", "mean_position_notional",
                 "median_position_notional"]].to_string(index=False))

    print("\n--- COST DECOMPOSITION and the IBKR $0.35 order minimum")
    for row in results:
        for bound in BOUNDS:
            print(f"  {row['book']:<26} {row['horizon_days']:>3}d {bound:<13} "
                  f"cost {row[bound + '_cost_per_roundtrip_bps']:>7.1f}bp/rt  "
                  f"spread {100 * row[bound + '_cost_share_spread']:>5.1f}%  "
                  f"impact {100 * row[bound + '_cost_share_impact']:>5.1f}%  "
                  f"commission {100 * row[bound + '_cost_share_commission']:>5.1f}%  "
                  f"orders at $0.35 floor {100 * row[bound + '_share_orders_at_035_minimum']:>5.1f}%")

    print("\n--- SHARPE PER DECADE (net). A decade under 24 months is not evidence.")
    for row in results:
        for bound in BOUNDS:
            decades = row[bound + "_sharpe_by_decade"]
            parts = [f"{k}: {v['sharpe']:+.2f} ({v['months']}m"
                     f"{', THIN' if v['thin'] else ''})" for k, v in decades.items()]
            print(f"  {row['book']:<26} {row['horizon_days']:>3}d {bound:<13} "
                  + "  ".join(parts))

    print("\n--- P&L CONCENTRATION (alarm at 3% of total net P&L in one name-month)")
    for row in results:
        for bound in BOUNDS:
            conc = row[bound + "_pnl_concentration"]
            flag = ("  *** ALARM: CONCENTRATION-DRIVEN ***" if conc["exceeds_alarm"]
                    else "")
            print(f"  {row['book']:<26} {row['horizon_days']:>3}d {bound:<13} "
                  f"max name-month = {100 * conc['max_name_month_share_of_total']:>6.2f}% "
                  f"of total net P&L ({conc['name_months_over_alarm']} over 3%)"
                  f"{flag}")
    headline_rows = [r for r in results
                     if r["book"] == "top_decile"
                     and r["horizon_days"] == pead_retest.HEADLINE_HORIZON]
    if headline_rows:
        conc = headline_rows[0]["realistic_pnl_concentration"]
        print("\n  top 10 (name, month) by |share of total net P&L|, headline hold, "
              "realistic bound:")
        for item in conc["top"]:
            print(f"    {item['ticker']:<8} {item['month']}  "
                  f"{100 * item['share_of_total']:>7.2f}%  ${item['pnl']:>12,.0f}")

    print("\n--- REJECTION REASONS (top decile screen, distinct ticker-filing pairs)")
    for row in results:
        if row["book"] == "top_decile" and row["horizon_days"] == 20:
            for reason, num in sorted(row["screen_rejects"].items(),
                                      key=lambda item: -item[1]):
                print(f"  {reason:<26} {num:>9,}")


if __name__ == "__main__":
    sys.exit(main())
