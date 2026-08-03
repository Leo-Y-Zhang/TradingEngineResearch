"""Run the short-horizon reversal RE-TEST, once, exactly as pre-registered.

Registered design: `research/sleeves/reversal_retest_prereg.md` (frozen before this ran).

Two universe cuts x three rebalance frequencies x two mandatory cost bounds (plus the
declared, physically-impossible zero-cost ceiling), all reported, none selected.

    .venv/Scripts/python.exe -m scripts.run_reversal_retest
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sleeves.reversal_retest import (  # noqa: E402
    COST_TREATMENTS,
    FREQUENCIES,
    UNIVERSE_CUTS,
    PanelMatrices,
    RetestConfig,
    build_matrices,
    build_selections,
    month_row_for,
    precompute_periods,
    rebalance_grid,
    run_leg,
)
from research.validation import deflated_sharpe_ratio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reversal_retest")

OUT_DIR = Path(__file__).resolve().parents[1] / "research" / "sleeves" / "_reversal_retest"

# Registered promotion gate and trial count (prereg 10, 11).
PROMOTION_GATE = 0.75
N_TRIALS = 34

# Registered era splits (prereg 9.4).
ERAS = (("1998-2006", 1998, 2006), ("2007-2015", 2007, 2015))
SUB_ERAS = (("1998-2001", 1998, 2001), ("2002-2007", 2002, 2007),
            ("2008-2011", 2008, 2011), ("2012-2015", 2012, 2015))


@dataclass
class Stats:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def summarise(returns: np.ndarray, periods_per_year: float) -> Stats:
    """Geometric annual return, annualised vol, Sharpe vs a zero risk-free rate, max DD."""
    if returns.size < 2:
        return Stats(float("nan"), float("nan"), float("nan"), float("nan"))
    years = returns.size / periods_per_year
    equity = np.cumprod(1.0 + returns)
    annual_return = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    sd = float(np.std(returns, ddof=1))
    annual_vol = sd * float(np.sqrt(periods_per_year))
    sharpe = float(np.mean(returns) / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(1.0 - equity / peak))
    return Stats(annual_return, annual_vol, sharpe, max_drawdown)


def era_sharpes(dates: pd.DatetimeIndex, returns: np.ndarray, ppy: float,
                eras: tuple) -> dict:
    out = {}
    years = dates.year.to_numpy()
    for label, lo, hi in eras:
        mask = (years >= lo) & (years <= hi)
        block = returns[mask]
        s = summarise(block, ppy)
        out[label] = {"n": int(block.size), "sharpe": s.sharpe,
                      "annual_return": s.annual_return}
    return out


def information_coefficient(panel: PanelMatrices, config: RetestConfig,
                            selections: dict, exec_idx: np.ndarray,
                            period_returns: list[np.ndarray]) -> dict:
    """Mean cross-sectional rank IC of the z-scored signal against realised holding return.

    Measured over the FULL tradable universe, not the traded deciles.
    """
    per_date = []
    for k in range(len(exec_idx) - 1):
        universe = selections["universe"][k]
        if universe.size < 30:
            continue
        realised = period_returns[k]
        signal = selections["zsignal"][k][universe]
        outcome = realised[universe]
        ok = np.isfinite(signal) & np.isfinite(outcome)
        if int(ok.sum()) < 30:
            continue
        rho = stats.spearmanr(signal[ok], outcome[ok]).statistic
        if np.isfinite(rho):
            per_date.append(float(rho))
    series = np.asarray(per_date)
    mean_ic = float(series.mean()) if series.size else float("nan")
    t_stat = (float(series.mean() / series.std(ddof=1) * np.sqrt(series.size))
              if series.size > 1 else float("nan"))
    return {"mean": mean_ic, "t_stat": t_stat, "n_dates": int(series.size),
            "hit_rate": float(np.mean(series > 0)) if series.size else float("nan")}


def concentration(pnl_by_ticker: np.ndarray, tickers: np.ndarray,
                  max_cell: float, max_cell_label: str) -> dict:
    """Largest single-ticker and single (ticker, period) share of total gross P&L (prereg 9.5)."""
    total_abs = float(np.abs(pnl_by_ticker).sum())
    if total_abs <= 0.0:
        return {"total_abs_pnl": 0.0, "top_ticker": None, "top_ticker_share": float("nan"),
                "max_cell_share": float("nan"), "max_cell": max_cell_label}
    j = int(np.argmax(np.abs(pnl_by_ticker)))
    return {
        "total_abs_pnl": total_abs,
        "top_ticker": str(tickers[j]),
        "top_ticker_share": float(abs(pnl_by_ticker[j]) / total_abs),
        "max_cell_share": float(abs(max_cell) / total_abs),
        "max_cell": max_cell_label,
    }


def run_one(panel: PanelMatrices, config: RetestConfig, cut_name: str,
            frequency: str) -> dict:
    """One (universe cut, frequency) cell: gross once, costs under every registered bound."""
    threshold = UNIVERSE_CUTS[cut_name]
    signal_idx, exec_idx = rebalance_grid(panel.dates, frequency)
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    signal_idx, exec_idx, month_rows = signal_idx[keep], exec_idx[keep], month_rows[keep]

    span_years = float((panel.dates[signal_idx[-1]] - panel.dates[signal_idx[0]]).days) / 365.25
    # REALISED periods per year, not the nominal grid frequency (prereg 5).
    ppy = (len(signal_idx) - 1) / span_years

    selections = build_selections(panel, config, threshold, signal_idx, exec_idx, month_rows)
    periods = precompute_periods(panel, config, signal_idx, exec_idx)

    empty = np.array([], dtype=int)
    neutral_ok = [s.size > 0 for s in selections["short"]]
    ls_long_selection = [lg if ok else empty
                         for lg, ok in zip(selections["long"], neutral_ok)]

    trim = slice(0, len(signal_idx) - 1)
    result: dict = {
        "universe_cut": cut_name,
        "frequency": frequency,
        "n_periods": int(len(signal_idx) - 1),
        "periods_per_year_realised": ppy,
        "span_years": span_years,
        "start": str(panel.dates[signal_idx[0]].date()),
        "end": str(panel.dates[signal_idx[-1]].date()),
        "books": {},
    }

    universe_sizes, long_sizes, short_sizes = [], [], []
    for k in range(len(signal_idx) - 1):
        universe_sizes.append(selections["universe"][k].size)
        long_sizes.append(selections["long"][k].size)
        short_sizes.append(selections["short"][k].size)
    result["mean_universe_names"] = float(np.mean(universe_sizes))
    result["mean_long_names"] = float(np.mean(long_sizes))
    result["mean_short_names"] = float(np.mean(short_sizes))
    result["periods_with_short"] = int(np.sum(np.asarray(short_sizes) > 0))

    result["ic"] = information_coefficient(panel, config, selections, exec_idx,
                                           periods.returns)

    series_out: dict[str, np.ndarray] = {}

    for treatment in COST_TREATMENTS:
        basis = panel.spread_basis[treatment]
        charge_impact = treatment != "zero_cost"
        borrow = 0.0 if treatment == "zero_cost" else config.short_borrow_annual

        ls_long_book = run_leg(panel, config, ls_long_selection, signal_idx, exec_idx,
                               month_rows, periods, config.long_leg_exposure, basis, ppy,
                               charge_impact=charge_impact)
        long_book = run_leg(panel, config, selections["long"], signal_idx, exec_idx,
                            month_rows, periods, config.long_leg_exposure, basis, ppy,
                            charge_impact=charge_impact)
        short_book = run_leg(panel, config, selections["short"], signal_idx, exec_idx,
                             month_rows, periods, config.short_leg_exposure, basis, ppy,
                             charge_impact=charge_impact, borrow_annual=borrow)
        universe_book = run_leg(panel, config, selections["universe"], signal_idx,
                                exec_idx, month_rows, periods, 1.0, basis, ppy,
                                charge_impact=charge_impact)

        bench = summarise(universe_book.gross_return[trim], ppy)
        dates = universe_book.dates[trim]

        books = {}
        for name in ("long_short", "long_only"):
            if name == "long_short":
                gross = ls_long_book.gross_return[trim] - short_book.gross_return[trim]
                cost = ls_long_book.cost[trim] + short_book.cost[trim]
                turn = ls_long_book.turnover[trim] + short_book.turnover[trim]
                spread_c = ls_long_book.spread_cost[trim] + short_book.spread_cost[trim]
                impact_c = ls_long_book.impact_cost[trim] + short_book.impact_cost[trim]
                comm_c = ls_long_book.commission_cost[trim] + short_book.commission_cost[trim]
                borrow_c = short_book.borrow_cost[trim]
                pnl_ticker = ls_long_book.pnl_by_ticker - short_book.pnl_by_ticker
                max_cell, max_label = ((ls_long_book.max_cell_pnl,
                                        ls_long_book.max_cell_label)
                                       if abs(ls_long_book.max_cell_pnl)
                                       >= abs(short_book.max_cell_pnl)
                                       else (short_book.max_cell_pnl,
                                             short_book.max_cell_label))
            else:
                gross = long_book.gross_return[trim]
                cost = long_book.cost[trim]
                turn = long_book.turnover[trim]
                spread_c = long_book.spread_cost[trim]
                impact_c = long_book.impact_cost[trim]
                comm_c = long_book.commission_cost[trim]
                borrow_c = np.zeros_like(cost)
                pnl_ticker = long_book.pnl_by_ticker
                max_cell, max_label = long_book.max_cell_pnl, long_book.max_cell_label

            net = gross - cost
            g, n = summarise(gross, ppy), summarise(net, ppy)
            turnover_annual = float(np.mean(turn) * ppy)
            cost_annual = float(np.mean(cost) * ppy)

            entry = {
                "gross": g.__dict__,
                "net": n.__dict__,
                "turnover_annual": turnover_annual,
                "cost_drag_annual": cost_annual,
                "excess_annual": n.annual_return - bench.annual_return,
                "gross_excess_annual": g.annual_return - bench.annual_return,
                "cost_per_round_trip_bps": (1e4 * cost_annual / turnover_annual
                                            if turnover_annual > 0 else float("nan")),
                "gross_alpha_per_round_trip_bps": (1e4 * g.annual_return / turnover_annual
                                                   if turnover_annual > 0 else float("nan")),
                "cost_decomposition_annual": {
                    "spread": float(np.mean(spread_c) * ppy),
                    "impact": float(np.mean(impact_c) * ppy),
                    "commission": float(np.mean(comm_c) * ppy),
                    "borrow": float(np.mean(borrow_c) * ppy),
                },
                "ladder": {f"{m:g}x": summarise(gross - m * cost, ppy).sharpe
                           for m in config.cost_ladder},
                "era_sharpe": era_sharpes(dates, net, ppy, ERAS),
                "sub_era_sharpe": era_sharpes(dates, net, ppy, SUB_ERAS),
                "gross_sub_era_sharpe": era_sharpes(dates, gross, ppy, SUB_ERAS),
                "pnl_concentration": concentration(pnl_ticker, panel.tickers,
                                                   max_cell, max_label),
                "dsr_n34": float(deflated_sharpe_ratio(net, n_trials=N_TRIALS)),
                "clears_gate": bool(np.isfinite(n.sharpe) and n.sharpe >= PROMOTION_GATE
                                    and n.annual_return > bench.annual_return),
            }
            entry["cover_ratio"] = (
                entry["gross_alpha_per_round_trip_bps"] / entry["cost_per_round_trip_bps"]
                if entry["cost_per_round_trip_bps"] and
                np.isfinite(entry["cost_per_round_trip_bps"]) and
                entry["cost_per_round_trip_bps"] > 0 else float("nan")
            )
            books[name] = entry
            series_out[f"{treatment}__{name}"] = net

        result["books"][treatment] = {
            "benchmark_universe_ew": bench.__dict__,
            **books,
        }

    result["_series_dates"] = universe_book.dates[trim]
    result["_series"] = series_out
    return result


def persist_series(result: dict, cut: str, frequency: str) -> str:
    """Prereg 9.9: every reported configuration persists its net return series."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(result.pop("_series"), index=result.pop("_series_dates"))
    frame.index.name = "date"
    path = OUT_DIR / f"net_returns_{cut}_{frequency}.parquet"
    frame.to_parquet(path)
    return str(path)


def render(result: dict) -> str:
    lines: list[str] = []
    add = lines.append
    head = f"{result['universe_cut']}  /  {result['frequency']}"
    add(f"\n{'=' * 82}\n{head}\n{'=' * 82}")
    add(f"window {result['start']} .. {result['end']}   {result['n_periods']} periods   "
        f"realised {result['periods_per_year_realised']:.2f}/yr over "
        f"{result['span_years']:.1f} yr")
    add(f"universe {result['mean_universe_names']:.0f} names/rebalance   "
        f"long {result['mean_long_names']:.0f}   short {result['mean_short_names']:.0f}   "
        f"({result['periods_with_short']} periods with a short leg)")
    ic = result["ic"]
    add(f"IC (rank, z-scored signal, full universe): mean {ic['mean']:+.4f}  "
        f"t {ic['t_stat']:+.2f}  n {ic['n_dates']}  positive on {ic['hit_rate']:.1%}")

    for treatment in COST_TREATMENTS:
        block = result["books"][treatment]
        bench = block["benchmark_universe_ew"]
        add(f"\n-- cost bound: {treatment.upper()} "
            f"{'(diagnostic ceiling, NOT gate-eligible)' if treatment == 'zero_cost' else ''}")
        add(f"   benchmark EW own universe: {bench['annual_return']:+.2%}/yr  "
            f"vol {bench['annual_vol']:.2%}  Sharpe {bench['sharpe']:.2f}")
        for name in ("long_short", "long_only"):
            b = block[name]
            add(f"   {name:11s} gross {b['gross']['annual_return']:+7.2%}/yr "
                f"Sh {b['gross']['sharpe']:+.2f} | net {b['net']['annual_return']:+8.2%}/yr "
                f"vol {b['net']['annual_vol']:.2%} Sh {b['net']['sharpe']:+7.2f} | "
                f"excess {b['excess_annual']:+8.2%}/yr")
            add(f"               turnover {b['turnover_annual']:5.1f}x  "
                f"cost {b['cost_drag_annual']:6.2%}/yr  "
                f"= {b['cost_per_round_trip_bps']:6.1f}bp/RT vs alpha "
                f"{b['gross_alpha_per_round_trip_bps']:6.1f}bp/RT  "
                f"cover {b['cover_ratio']:.2f}  gate {b['clears_gate']}")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    config = RetestConfig()
    panel = build_matrices(config)
    logger.info("matrices built in %.1fs", time.time() - t0)

    results: dict[str, dict] = {}
    for cut in UNIVERSE_CUTS:
        for frequency in FREQUENCIES:
            started = time.time()
            key = f"{cut}__{frequency}"
            res = run_one(panel, config, cut, frequency)
            res["series_path"] = persist_series(res, cut, frequency)
            results[key] = res
            logger.info("[%s] done in %.1fs", key, time.time() - started)
            print(render(res))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "reversal_retest_result.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten: {out}")

    # The registered deliverable: net Sharpe as a function of rebalance frequency.
    print(f"\n{'=' * 82}\nFREQUENCY CURVE - net Sharpe by rebalance frequency\n{'=' * 82}")
    for cut in UNIVERSE_CUTS:
        for treatment in ("conservative", "realistic", "zero_cost"):
            for book in ("long_short", "long_only"):
                row = []
                for frequency in FREQUENCIES:
                    r = results[f"{cut}__{frequency}"]
                    b = r["books"][treatment][book]
                    row.append(f"{frequency[:4]} f={r['periods_per_year_realised']:.0f} "
                               f"S={b['net']['sharpe']:+.2f}")
                print(f"{cut:14s} {treatment:12s} {book:11s} " + "   ".join(row))

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
