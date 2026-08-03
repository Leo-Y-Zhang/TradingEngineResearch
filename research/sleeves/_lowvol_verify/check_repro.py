"""ATTACK 0 -- REPRODUCTION. Nothing below is worth reading until this passes.

Re-runs `lowvol_retest.run_band` + `evaluate_band` from the committed code, compares every
headline in `lowvol_retest_result.json` against the fresh run, and then compares the
independent instrumented re-implementation against the original book bar by bar.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.check_repro
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "research" / "sleeves" / "lowvol_retest_result.json"

HEADLINES = [
    ("gross", "sharpe"), ("gross", "tstat"), ("gross", "dsr"),
    ("benchmark", "sharpe"), ("benchmark", "annual_arithmetic"),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    published = json.loads(RESULT.read_text(encoding="utf-8"))
    by_band = {b["band"]: b for b in published["bands"]}

    print("=" * 100)
    print("ATTACK 0 - REPRODUCTION OF THE PUBLISHED RESULT")
    print("=" * 100)
    worst = 0.0
    for band in LV.BAND_ORDER:
        books = LV.run_band(merged, band, delistings)
        if books is None:
            continue
        fresh = LV.evaluate_band(books, n_trials=LV.N_TRIALS)
        old = by_band[band]
        deltas = []
        for section, key in HEADLINES:
            a, b = fresh[section][key], old[section][key]
            deltas.append((f"{section}.{key}", a, b, a - b))
        for bound in ("conservative", "realistic"):
            for key in ("sharpe", "annual_arithmetic", "volatility", "tstat", "dsr",
                        "max_drawdown"):
                a = fresh["bounds"][bound]["net"][key]
                b = old["bounds"][bound]["net"][key]
                deltas.append((f"{bound}.net.{key}", a, b, a - b))
            for key in ("vol_matched_active_annual", "vol_matched_active_tstat",
                        "benchmark_scale_factor", "raw_active_annual"):
                a = fresh["bounds"][bound]["vol_matched"][key]
                b = old["bounds"][bound]["vol_matched"][key]
                deltas.append((f"{bound}.vm.{key}", a, b, a - b))
            for key in ("cost_one_way_bps", "cost_annual_total"):
                a = fresh["bounds"][bound][key]
                b = old["bounds"][bound][key]
                deltas.append((f"{bound}.{key}", a, b, a - b))
        for key in ("turnover_annual", "forced_exit_share", "delisting_drag_annual",
                    "dsr_sharpe_bar", "deployable_capital", "position_value",
                    "upper_bound_share_held", "n_months"):
            deltas.append((key, fresh[key], old[key], fresh[key] - old[key]))
        band_worst = max(abs(d) for *_x, d in deltas)
        worst = max(worst, band_worst)
        print(f"{band:>13}  max |fresh - published| = {band_worst:.3e}  "
              f"({'IDENTICAL' if band_worst < 1e-9 else 'DIFFERS'})")
        if band_worst >= 1e-9:
            for name, a, b, d in deltas:
                if abs(d) >= 1e-9:
                    print(f"     {name:>40} fresh {a:.10g}  published {b:.10g}  d {d:+.3e}")

    print(f"\n  worst discrepancy across all bands: {worst:.3e}")
    print(f"  published verdict: {published['verdict']}   n_trials {published['n_trials']}")

    print("\n" + "=" * 100)
    print("INDEPENDENT RE-IMPLEMENTATION vs THE COMMITTED ONE (same rules, separate code)")
    print("=" * 100)
    for band in LV.BAND_ORDER:
        books = LV.run_band(merged, band, delistings)
        mine = INS.run(merged, band, delistings)
        if books is None or mine is None:
            continue
        d_gross = float(np.max(np.abs(books.gross - mine.gross)))
        d_cost = float(np.max(np.abs(books.cost_conservative - mine.cost["conservative"])))
        d_bench = float(np.max(np.abs(books.benchmark - mine.benchmark)))
        print(f"{band:>13}  max|d gross| {d_gross:.3e}   max|d cost| {d_cost:.3e}   "
              f"max|d bench| {d_bench:.3e}   legs {books.legs_traded} vs {mine.legs_traded}"
              f"   rebalances {books.n_rebalances} vs {mine.n_rebalances}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
