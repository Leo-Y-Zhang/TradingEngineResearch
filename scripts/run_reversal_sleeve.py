"""Run the short-horizon cross-sectional reversal sleeve, once, as registered.

Registered design: `research/sleeves/short_horizon_reversal.py` module docstring.
Both the PRIMARY (measured-spread-only) and the declared SECONDARY diagnostic
(measured + upper_bound, costed at the upper bound) are run and reported. No selection
between them; no parameter is adjusted after seeing a number.

    .venv/Scripts/python.exe -m scripts.run_reversal_sleeve
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sleeves.short_horizon_reversal import (  # noqa: E402
    SECONDARY_CONFIG,
    PanelMatrices,
    ReversalConfig,
    _holding_returns,
    _run_leg,
    build_matrices,
    build_selections,
    month_row_for,
    weekly_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reversal")


@dataclass
class Stats:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float


def summarise(returns: np.ndarray, periods_per_year: float) -> Stats:
    """Geometric annual return, annualised vol, Sharpe vs zero, and max drawdown.

    Sharpe is measured against a zero risk-free rate. For the dollar-neutral book that
    is roughly right (broker financing on the long leg approximately offsets the cash
    yield); for the long-only book it FLATTERS by the risk-free rate of the era, which
    averaged ~2-3% over 1998-2015. Stated here rather than silently absorbed.
    """
    if returns.size < 2:
        return Stats(np.nan, np.nan, np.nan, np.nan)
    years = returns.size / periods_per_year
    equity = np.cumprod(1.0 + returns)
    annual_return = float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.max(1.0 - equity / peak))
    return Stats(annual_return, annual_vol, sharpe, max_drawdown)


def information_coefficient(
    panel: PanelMatrices,
    config: ReversalConfig,
    selections: dict,
    exec_idx: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Mean cross-sectional rank IC between signal and realised holding return.

    This is the IC of Grinold's law, measured over the FULL tradable universe rather
    than the traded deciles -- the deciles are a construction on top of it, and their
    performance already appears in the P&L.
    """
    per_date = []
    for k in range(len(exec_idx) - 1):
        universe = selections["universe"][k]
        if universe.size < 30:
            continue
        realised = _holding_returns(panel, exec_idx[k], exec_idx[k + 1], config)
        signal = selections["signal"][k][universe]
        outcome = realised[universe]
        ok = np.isfinite(signal) & np.isfinite(outcome)
        if ok.sum() < 30:
            continue
        rho = stats.spearmanr(signal[ok], outcome[ok]).statistic
        if np.isfinite(rho):
            per_date.append(float(rho))
    series = np.asarray(per_date)
    mean_ic = float(series.mean()) if series.size else np.nan
    t_stat = float(series.mean() / series.std(ddof=1) * np.sqrt(series.size)) if series.size > 1 else np.nan
    return mean_ic, t_stat, series


def run_config(config: ReversalConfig, panel: PanelMatrices) -> dict:
    """Execute one registered configuration end to end and return its measured numbers."""
    signal_idx, exec_idx = weekly_grid(panel.dates)
    # A signal date needs the lookback window behind it and a published monthly row.
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    signal_idx, exec_idx, month_rows = signal_idx[keep], exec_idx[keep], month_rows[keep]
    logger.info("[%s] %d weekly rebalances, %s to %s", config.label, len(signal_idx),
                panel.dates[signal_idx[0]].date(), panel.dates[signal_idx[-1]].date())

    selections = build_selections(panel, config, signal_idx, exec_idx, month_rows)

    # A dollar-neutral book cannot be formed on a week when the shortable subset is too
    # thin to yield a decile of at least `min_names_per_leg`. Rather than let the
    # long/short book silently become a long-only book on those weeks -- which would
    # smuggle market beta into a "market-neutral" result -- it goes flat, and pays the
    # cost of getting flat. The registered fallback for un-shortable weeks is the
    # separate long-only book, which is reported alongside it.
    empty = np.array([], dtype=int)
    neutral_ok = [s.size > 0 for s in selections["short"]]
    ls_long_selection = [
        long_leg if ok else empty
        for long_leg, ok in zip(selections["long"], neutral_ok)
    ]

    ls_long_book = _run_leg(panel, config, ls_long_selection, signal_idx, exec_idx,
                            month_rows, config.long_leg_exposure)
    long_book = _run_leg(panel, config, selections["long"], signal_idx, exec_idx,
                         month_rows, config.long_leg_exposure)
    short_book = _run_leg(panel, config, selections["short"], signal_idx, exec_idx,
                          month_rows, config.short_leg_exposure,
                          borrow_annual=config.short_borrow_annual)
    universe_book = _run_leg(panel, config, selections["universe"], signal_idx, exec_idx,
                             month_rows, 1.0)

    # The final period's exit is clamped to the end of the panel, so it is a ragged
    # holding window and is dropped rather than diluted into the annualisation.
    trim = slice(0, len(signal_idx) - 1)

    long_gross = long_book.gross_return[trim]
    long_cost = long_book.cost[trim]
    short_gross = short_book.gross_return[trim]
    short_cost = short_book.cost[trim]
    bench_gross = universe_book.gross_return[trim]

    ls_gross = ls_long_book.gross_return[trim] - short_gross
    ls_cost = ls_long_book.cost[trim] + short_cost
    lo_gross = long_gross
    lo_cost = long_cost

    result: dict = {
        "label": config.label,
        "n_periods": int(len(ls_gross)),
        "start": str(panel.dates[signal_idx[0]].date()),
        "end": str(panel.dates[signal_idx[-1]].date()),
        "mean_universe_names": float(np.mean(universe_book.n_names[trim])),
        "mean_long_names": float(np.mean(long_book.n_names[trim])),
        "mean_short_names": float(np.mean(short_book.n_names[trim])),
        "periods_with_short": int(np.sum(short_book.n_names[trim] > 0)),
    }

    ppy = config.periods_per_year
    for name, gross, cost, turnover, book in (
        ("long_short", ls_gross, ls_cost, ls_long_book.turnover[trim] + short_book.turnover[trim], None),
        ("long_only", lo_gross, lo_cost, long_book.turnover[trim], long_book),
    ):
        g = summarise(gross, ppy)
        n = summarise(gross - cost, ppy)
        result[name] = {
            "gross": g.__dict__,
            "net": n.__dict__,
            "turnover_annual": float(np.mean(turnover) * ppy),
            "cost_drag_annual": float(np.mean(cost) * ppy),
            "ladder": {
                f"{m:g}x": summarise(gross - m * cost, ppy).sharpe
                for m in config.cost_ladder
            },
        }

    bench = summarise(bench_gross, ppy)
    result["benchmark_universe_ew"] = bench.__dict__
    result["long_short"]["excess_annual"] = (
        result["long_short"]["net"]["annual_return"] - bench.annual_return
    )
    result["long_only"]["excess_annual"] = (
        result["long_only"]["net"]["annual_return"] - bench.annual_return
    )

    # Cost decomposition, annualised, for the long/short book.
    result["cost_decomposition_annual"] = {
        "spread": float(np.mean(ls_long_book.spread_cost[trim] + short_book.spread_cost[trim]) * ppy),
        "impact": float(np.mean(ls_long_book.impact_cost[trim] + short_book.impact_cost[trim]) * ppy),
        "commission": float(np.mean(ls_long_book.commission_cost[trim] + short_book.commission_cost[trim]) * ppy),
        "borrow": float(config.short_borrow_annual),
    }
    result["long_only_cost_decomposition_annual"] = {
        "spread": float(np.mean(long_book.spread_cost[trim]) * ppy),
        "impact": float(np.mean(long_book.impact_cost[trim]) * ppy),
        "commission": float(np.mean(long_book.commission_cost[trim]) * ppy),
    }

    mean_ic, ic_t, ic_series = information_coefficient(panel, config, selections, exec_idx)
    result["ic"] = {"mean": mean_ic, "t_stat": ic_t, "n_dates": int(ic_series.size),
                    "hit_rate": float(np.mean(ic_series > 0)) if ic_series.size else np.nan}

    # Breadth, three ways. The naive count is what a decile sort nominally provides;
    # the Grinold inversion is what the realised numbers actually imply, and the gap
    # between them is the cross-sectional correlation the naive count ignores.
    gross_ir = result["long_short"]["gross"]["sharpe"]
    implied_br = float((gross_ir / mean_ic) ** 2) if mean_ic and np.isfinite(mean_ic) and mean_ic != 0 else np.nan
    result["breadth"] = {
        "rebalances_per_year": ppy,
        "mean_names_traded_per_rebalance": result["mean_long_names"] + result["mean_short_names"],
        "naive_bets_per_year": ppy * (result["mean_long_names"] + result["mean_short_names"]),
        "grinold_implied_bets_per_year": implied_br,
    }
    return result


def render(result: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"\n{'=' * 78}\n{result['label']}\n{'=' * 78}")
    add(f"window {result['start']} .. {result['end']}   "
        f"{result['n_periods']} weekly periods")
    add(f"universe {result['mean_universe_names']:.0f} names/rebalance   "
        f"long {result['mean_long_names']:.0f}   short {result['mean_short_names']:.0f} "
        f"({result['periods_with_short']} periods with a short leg)")
    bench = result["benchmark_universe_ew"]
    add("\nBENCHMARK  equal-weight buy-and-hold of this universe, gross of costs:")
    add(f"  return {bench['annual_return']:+.2%}/yr   vol {bench['annual_vol']:.2%}   "
        f"Sharpe {bench['sharpe']:.2f}   maxDD {bench['max_drawdown']:.1%}")

    for book in ("long_short", "long_only"):
        b = result[book]
        add(f"\n{book.upper().replace('_', '/')}")
        add(f"  gross  return {b['gross']['annual_return']:+.2%}/yr  "
            f"vol {b['gross']['annual_vol']:.2%}  Sharpe {b['gross']['sharpe']:.2f}")
        add(f"  net    return {b['net']['annual_return']:+.2%}/yr  "
            f"vol {b['net']['annual_vol']:.2%}  Sharpe {b['net']['sharpe']:.2f}  "
            f"maxDD {b['net']['max_drawdown']:.1%}")
        add(f"  excess over universe EW: {b['excess_annual']:+.2%}/yr")
        add(f"  turnover {b['turnover_annual']:.0f}x/yr   "
            f"cost drag {b['cost_drag_annual']:.2%}/yr")
        ladder = "  ".join(f"{k} {v:+.2f}" for k, v in b["ladder"].items())
        add(f"  net Sharpe ladder: {ladder}")

    dec = result["cost_decomposition_annual"]
    add(f"\ncost decomposition (long/short, annual): spread {dec['spread']:.2%}  "
        f"impact {dec['impact']:.2%}  commission {dec['commission']:.2%}  "
        f"borrow {dec['borrow']:.2%}")
    lod = result["long_only_cost_decomposition_annual"]
    add(f"cost decomposition (long-only, annual):  spread {lod['spread']:.2%}  "
        f"impact {lod['impact']:.2%}  commission {lod['commission']:.2%}")
    ic = result["ic"]
    add(f"IC (rank, full universe): mean {ic['mean']:+.4f}  t {ic['t_stat']:+.2f}  "
        f"n {ic['n_dates']}  positive on {ic['hit_rate']:.1%} of dates")
    br = result["breadth"]
    add(f"breadth: {br['rebalances_per_year']:.0f} rebalances/yr, "
        f"{br['mean_names_traded_per_rebalance']:.0f} names/rebalance, "
        f"naive {br['naive_bets_per_year']:,.0f} bets/yr, "
        f"Grinold-implied {br['grinold_implied_bets_per_year']:,.0f} bets/yr")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    primary = ReversalConfig()
    configs = (primary, SECONDARY_CONFIG)
    panels = build_matrices(configs)
    logger.info("matrices built in %.1fs", time.time() - t0)

    results = {}
    for config in configs:
        started = time.time()
        results[config.label] = run_config(config, panels[config.label])
        logger.info("[%s] done in %.1fs", config.label, time.time() - started)
        print(render(results[config.label]))

    out = Path(__file__).resolve().parents[1] / "reports" / "reversal_sleeve_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
