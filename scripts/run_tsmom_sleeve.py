"""Run the multi-timeframe time-series-momentum sleeve ONCE, as pre-specified.

The pre-specification lives in `research/sleeves/tsmom_multitimeframe.py`'s module
docstring, together with its two errata. Three cost/universe treatments are declared
there -- PRIMARY (rule 3 read literally), PRIMARY-STICKY (erratum 1, rule-3 compliant,
artefact-corrected) and SENSITIVITY-B (a liquid-name schedule, explicitly NOT
gate-eligible). The signal, sizing and rebalance rule are identical across all three.
All are run once and all are reported regardless of outcome.

    .venv/Scripts/python.exe -m scripts.run_tsmom_sleeve
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR
from research.sleeves.tsmom_multitimeframe import (
    DECADES,
    LIQUID_SPREAD_SUBSTITUTE,
    LOOKBACKS,
    SMALL_ACCOUNT_EQUITY,
    STARTING_EQUITY,
    TARGET_VOLS,
    UNIVERSE_SIZE,
    annualised,
    build_daily_matrix,
    build_universe,
    effective_instruments,
    load_inputs,
    max_drawdown,
    monthly_retention,
    plan_book,
    simulate,
)

CONFIGURATIONS: tuple[tuple[str, str, bool], ...] = (
    ("PRIMARY (rule 3 literal: hold only while the spread resolves)",
     "measured_only", True),
    ("PRIMARY-STICKY (erratum 1, rule-3 compliant: last measured spread, <=1yr old)",
     "sticky_measured", True),
    ("SENSITIVITY-B (flat 20bps liquid-name schedule; NOT gate-eligible)",
     "liquid_schedule", False),
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("tsmom")

OUT_DIR = Path(__file__).resolve().parents[1] / "research" / "sleeves" / "_out"


def load_prices(tickers: set[str]) -> pd.DataFrame:
    """Daily bars for the named tickers only, DEV window enforced."""
    path = PANEL_DIR / "prices_to_2015-12-31.parquet"
    reader = pq.ParquetFile(path)
    frames = []
    for group in range(reader.num_row_groups):
        chunk = reader.read_row_group(
            group, columns=["ticker", "date", "close", "closeadj"]
        ).to_pandas()
        frames.append(chunk[chunk["ticker"].isin(tickers)])
    prices = pd.concat(frames, ignore_index=True)
    if pd.Timestamp(prices["date"].max()) > DEV_CUTOFF:
        raise ValueError("price panel leaked past the DEV cutoff")
    return prices


def window_stats(dates: pd.DatetimeIndex, returns: np.ndarray,
                 start: str, end: str) -> dict[str, float]:
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    cagr, vol, sharpe = annualised(returns[mask])
    return {"return": cagr, "vol": vol, "sharpe": sharpe}


def run_configuration(
    label: str,
    mode: str,
    gate_eligible: bool,
    panel: pd.DataFrame,
    delistings: pd.DataFrame,
    sector_of: dict[str, str],
    prices: pd.DataFrame,
) -> dict:
    logger.info("=== %s ===", label)
    universe = build_universe(panel, sector_of, mode=mode)
    tickers = {t for members in universe.members.values() for t in members}
    retention = monthly_retention(universe)
    logger.info(
        "%s: %d rebalances, %d distinct tickers, %d measured / %d substituted cells, "
        "month-to-month retention %.1f%%",
        label, len(universe.dates), len(tickers), universe.n_measured,
        universe.n_upper_bound_costed, retention * 100,
    )
    prices = prices[prices["ticker"].isin(tickers)]

    matrix = build_daily_matrix(prices, tickers, delistings)
    logger.info(
        "matrix %s, %d daily returns capped at +/-100%%, %d names carry a terminal return",
        matrix.returns.shape, matrix.n_returns_capped, matrix.n_terminal_names,
    )

    started = time.time()
    plan = plan_book(matrix, universe)
    benchmark_plan = plan_book(matrix, universe, long_only_equal_weight=True)
    logger.info("planned %d rebalances in %.0fs", len(plan.rebalances),
                time.time() - started)

    n_eff = effective_instruments(matrix, universe)
    flips_total = sum(plan.flips_per_year.values())
    breadth = n_eff * flips_total

    benchmark_net = simulate(matrix, benchmark_plan, None, charge_costs=True)
    benchmark_gross = simulate(matrix, benchmark_plan, None, charge_costs=False)
    bench_cagr, bench_vol, bench_sharpe = annualised(benchmark_net.returns)
    bench_gross_cagr, _, _ = annualised(benchmark_gross.returns)
    logger.info(
        "benchmark (EW long-only, same universe): net %.2f%%/yr, gross %.2f%%/yr, "
        "vol %.1f%%, Sharpe %.2f, turnover %.2fx",
        bench_cagr * 100, bench_gross_cagr * 100, bench_vol * 100, bench_sharpe,
        benchmark_net.turnover_per_year,
    )

    results = []
    for target in TARGET_VOLS:
        net = simulate(matrix, plan, target, charge_costs=True)
        gross = simulate(matrix, plan, target, charge_costs=False)
        small = simulate(matrix, plan, target, charge_costs=True,
                         starting_equity=SMALL_ACCOUNT_EQUITY)
        net_cagr, net_vol, net_sharpe = annualised(net.returns)
        gross_cagr, gross_vol, gross_sharpe = annualised(gross.returns)
        small_cagr, _, small_sharpe = annualised(small.returns)
        entry = {
            "target_vol": target,
            "gross_return": gross_cagr,
            "gross_vol": gross_vol,
            "gross_sharpe": gross_sharpe,
            "net_return": net_cagr,
            "net_vol": net_vol,
            "net_sharpe": net_sharpe,
            "max_drawdown": max_drawdown(net.returns),
            "turnover_per_year": net.turnover_per_year,
            "cost_drag": gross_cagr - net_cagr,
            "excess_vs_net_benchmark": net_cagr - bench_cagr,
            "excess_vs_gross_benchmark": net_cagr - bench_gross_cagr,
            "small_account_net_return": small_cagr,
            "small_account_net_sharpe": small_sharpe,
            "mean_leverage": net.mean_leverage,
            "mean_gross_exposure": net.mean_gross_exposure,
            "mean_net_exposure": net.mean_net_exposure,
            "terminal_returns_booked": net.n_terminal_booked,
            "decades": {
                name: window_stats(net.dates, net.returns, start, end)
                for name, start, end in DECADES
            },
            "decades_gross": {
                name: window_stats(gross.dates, gross.returns, start, end)
                for name, start, end in DECADES
            },
        }
        results.append(entry)
        logger.info(
            "target %.0f%%: net %.2f%%/yr vol %.1f%% Sharpe %.2f (gross Sharpe %.2f) "
            "maxDD %.1f%% turnover %.1fx cost %.2f%%/yr excess %.2f%%/yr "
            "[$100k account: net %.2f%%/yr Sharpe %.2f]",
            target * 100, net_cagr * 100, net_vol * 100, net_sharpe, gross_sharpe,
            entry["max_drawdown"] * 100, net.turnover_per_year,
            entry["cost_drag"] * 100, entry["excess_vs_net_benchmark"] * 100,
            small_cagr * 100, small_sharpe,
        )
        for name, _, _ in DECADES:
            decade = entry["decades"][name]
            logger.info(
                "    %s: net %.2f%%/yr vol %.1f%% Sharpe %.2f (gross Sharpe %.2f)",
                name, decade["return"] * 100, decade["vol"] * 100, decade["sharpe"],
                entry["decades_gross"][name]["sharpe"],
            )

    return {
        "label": label,
        "mode": mode,
        "gate_eligible": gate_eligible,
        "monthly_universe_retention": retention,
        "n_rebalances": len(universe.dates),
        "first_rebalance": str(universe.dates[0].date()),
        "last_rebalance": str(universe.dates[-1].date()),
        "n_distinct_tickers": len(tickers),
        "universe_size_target": UNIVERSE_SIZE,
        "mean_instruments": plan.mean_instruments,
        "mean_single_names": plan.mean_singles,
        "mean_sector_baskets": plan.mean_baskets,
        "n_cells_measured_spread": universe.n_measured,
        "n_cells_substituted_spread": universe.n_upper_bound_costed,
        "median_spread_bps": float(
            np.median([v for v in universe.spread.values() if np.isfinite(v)]) * 1e4
        ),
        "daily_returns_capped": matrix.n_returns_capped,
        "names_with_terminal_return": matrix.n_terminal_names,
        "breadth": {
            "nominal_instruments": plan.mean_instruments,
            "effective_independent_instruments": n_eff,
            "signal_flips_per_year": {str(k): v for k, v in plan.flips_per_year.items()},
            "flips_per_year_all_timeframes": flips_total,
            "bets_per_year": breadth,
            "naive_bets_per_year": plan.mean_instruments * len(LOOKBACKS) * 12,
        },
        "benchmark": {
            "net_return": bench_cagr,
            "gross_return": bench_gross_cagr,
            "vol": bench_vol,
            "sharpe": bench_sharpe,
            "max_drawdown": max_drawdown(benchmark_net.returns),
            "turnover_per_year": benchmark_net.turnover_per_year,
            "decades": {
                name: window_stats(benchmark_net.dates, benchmark_net.returns, start, end)
                for name, start, end in DECADES
            },
        },
        "targets": results,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, delistings, sector_of = load_inputs()
    logger.info("panel %s rows, DEV max date %s", f"{len(panel):,}",
                panel["date"].max().date())

    # The three universes overlap heavily; load the union once so the expensive parquet
    # read happens a single time.
    union: set[str] = set()
    for _label, mode, _gate in CONFIGURATIONS:
        universe = build_universe(panel, sector_of, mode=mode)
        union |= {t for m in universe.members.values() for t in m}
    logger.info("loading prices for %d tickers", len(union))
    started = time.time()
    prices = load_prices(union)
    logger.info("loaded %s price rows in %.0fs", f"{len(prices):,}",
                time.time() - started)

    output = {
        "sleeve": "time_series_momentum_multi_timeframe",
        "dev_cutoff": str(DEV_CUTOFF.date()),
        "lookbacks": list(LOOKBACKS),
        "target_vols": list(TARGET_VOLS),
        "liquid_spread_substitute_bps": LIQUID_SPREAD_SUBSTITUTE * 1e4,
        "starting_equity": STARTING_EQUITY,
        "small_account_equity": SMALL_ACCOUNT_EQUITY,
        "configurations": [],
    }
    for label, mode, gate_eligible in CONFIGURATIONS:
        output["configurations"].append(
            run_configuration(label, mode, gate_eligible, panel, delistings,
                              sector_of, prices)
        )

    destination = OUT_DIR / "tsmom_multitimeframe_result.json"
    destination.write_text(json.dumps(output, indent=2, default=float))
    logger.info("wrote %s", destination)


if __name__ == "__main__":
    main()
