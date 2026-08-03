"""Measure the universe bias iteration 1 was carrying, on the real DEV panel.

Iteration 1 costed only ``measured`` names and DELETED every ``upper_bound`` one. This
script re-prices the same panel under the two-bound model
(`research/spread_estimation.py`) and reports the difference:

  * how many (name, month) cells become tradable -- that count IS the size of the bias;
  * what the newly-admitted names look like next to the ones already admitted, which is
    the test of whether the deleted names really were the cheap, liquid ones;
  * the spread bracket, conservative vs realistic, band by band.

It writes derived statistics only (counts, medians, ratios). No Sharadar row ever leaves
`_data/`, per the licence.

Run:  .venv/Scripts/python.exe scripts/measure_spread_universe_bias.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.spread_estimation import bounds_from_estimate  # noqa: E402

PANEL = REPO / "_data" / "sharadar" / "panel" / "monthly_panel_dev.parquet"
OUT = REPO / "reports" / "spread_universe_bias.json"

BANDS: tuple[tuple[str, float, float], ...] = (
    ("<$350k", 0.0, 3.5e5),
    ("$350k-$1.5M", 3.5e5, 1.5e6),
    ("$1.5M-$5M", 1.5e6, 5e6),
    ("$5M-$20M", 5e6, 2e7),
    (">$20M", 2e7, np.inf),
)


def band_of(dollar_volume: float) -> str:
    for label, low, high in BANDS:
        if low <= dollar_volume < high:
            return label
    return "unbanded"


def main() -> int:
    if not PANEL.exists():
        print(f"ERROR: {PANEL} not found. Run scripts/build_capacity_panel.py first.")
        return 2

    panel = pd.read_parquet(PANEL)
    eligible = panel[panel["spread_regime"] != "ineligible"].copy()

    conservative = np.full(len(eligible), np.nan)
    realistic = np.full(len(eligible), np.nan)
    tradable = np.zeros(len(eligible), dtype=bool)

    columns = ["spread", "spread_regime", "median_dollar_volume", "close", "date"]
    for i, row in enumerate(eligible[columns].itertuples(index=False)):
        bounds = bounds_from_estimate(
            row.spread, row.spread_regime, row.median_dollar_volume,
            price=row.close, when=row.date,
        )
        conservative[i] = bounds.conservative
        realistic[i] = bounds.realistic
        tradable[i] = bounds.tradable

    eligible["conservative"] = conservative
    eligible["realistic"] = realistic
    eligible["tradable"] = tradable
    eligible["band"] = eligible["median_dollar_volume"].map(band_of)

    before = eligible[eligible["spread_regime"] == "measured"]
    added = eligible[eligible["tradable"] & (eligible["spread_regime"] != "measured")]
    after = eligible[eligible["tradable"]]

    print("=" * 78)
    print("UNIVERSE BIAS -- (name, month) cells, DEV panel "
          f"{eligible['date'].min().date()} to {eligible['date'].max().date()}")
    print("=" * 78)
    print(f"  eligible cells                    {len(eligible):>10,}")
    print(f"  tradable under iteration 1        {len(before):>10,}  (measured only)")
    print(f"  tradable under the two-bound model{len(after):>10,}")
    print(f"  EXTRA CELLS NOW TRADABLE          {len(added):>10,}  "
          f"(+{len(added) / len(before):.1%})")
    still_out = int((~eligible["tradable"]).sum())
    print(f"  still excluded (unmeasurable)     {still_out:>10,}")

    print("\n" + "=" * 78)
    print("WERE THE DELETED NAMES THE CHEAP ONES? (the bias, tested directly)")
    print("=" * 78)
    print(f"{'':>34} {'kept by iter 1':>16} {'deleted by iter 1':>18}")
    rows = (
        ("median dollar volume / day",
         before["median_dollar_volume"].median(),
         added["median_dollar_volume"].median(), "{:>15,.0f}"),
        ("median share price",
         before["close"].median(), added["close"].median(), "{:>15,.2f}"),
        ("median spread, bound (a) bps",
         before["conservative"].median() * 1e4,
         added["conservative"].median() * 1e4, "{:>15,.1f}"),
        ("median spread, bound (b) bps",
         before["realistic"].median() * 1e4,
         added["realistic"].median() * 1e4, "{:>15,.1f}"),
    )
    for label, kept, deleted, fmt in rows:
        print(f"{label:>34} {fmt.format(kept)} {fmt.format(deleted):>18}")

    print("\n" + "=" * 78)
    print("THE BRACKET, BY LIQUIDITY BAND (median bps of full effective spread)")
    print("=" * 78)
    print(f"{'band':>14} {'cells':>9} {'added':>9} {'(a) cons':>10} {'(b) real':>10} "
          f"{'(a)/(b)':>9}")
    by_band = {}
    for label, _, _ in BANDS:
        cells = after[after["band"] == label]
        if cells.empty:
            continue
        new_cells = added[added["band"] == label]
        cons = float(cells["conservative"].median()) * 1e4
        real = float(cells["realistic"].median()) * 1e4
        ratio = cons / real if real > 0 else float("nan")
        by_band[label] = {
            "tradable_cells": int(len(cells)),
            "cells_added": int(len(new_cells)),
            "median_conservative_bps": round(cons, 2),
            "median_realistic_bps": round(real, 2),
        }
        print(f"{label:>14} {len(cells):>9,} {len(new_cells):>9,} {cons:>9.1f}b "
              f"{real:>9.1f}b {ratio:>8.2f}x")

    inverted = int((after["realistic"] > after["conservative"] + 1e-12).sum())
    print(f"\n  cells where the bracket inverted: {inverted}  (must be 0)")

    payload = {
        "panel": PANEL.name,
        "dev_window": [str(eligible["date"].min().date()),
                       str(eligible["date"].max().date())],
        "eligible_cells": int(len(eligible)),
        "tradable_iteration_1": int(len(before)),
        "tradable_two_bound": int(len(after)),
        "extra_cells_tradable": int(len(added)),
        "expansion_ratio": round(len(after) / len(before), 4),
        "still_excluded_unmeasurable": still_out,
        "kept_by_iteration_1": {
            "median_dollar_volume": round(float(
                before["median_dollar_volume"].median()), 2),
            "median_price": round(float(before["close"].median()), 4),
            "median_conservative_bps": round(float(
                before["conservative"].median()) * 1e4, 2),
            "median_realistic_bps": round(float(
                before["realistic"].median()) * 1e4, 2),
        },
        "deleted_by_iteration_1": {
            "median_dollar_volume": round(float(
                added["median_dollar_volume"].median()), 2),
            "median_price": round(float(added["close"].median()), 4),
            "median_conservative_bps": round(float(
                added["conservative"].median()) * 1e4, 2),
            "median_realistic_bps": round(float(
                added["realistic"].median()) * 1e4, 2),
        },
        "by_band": by_band,
        "bracket_inversions": inverted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(REPO)}")
    return 0 if inverted == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
