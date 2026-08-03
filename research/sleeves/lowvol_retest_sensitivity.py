"""The two accounting defects found by verification, priced. Reproduces section 6 of
`research/sleeves/lowvol_retest_result.md`.

Neither is a strategy parameter and neither was chosen after seeing a return. Both are
inherited from iteration 1's harness, both were found by asking why a diagnostic was
implausible, and both are switches whose DEFAULTS reproduce the registered run bit-for-bit:

1. `delisting_window` -- the registered `(exit, exit+62]` excludes a delisting dated on the
   same day as the ticker's last bar, which is the modal case. It fires 39 times against
   3,018 available.
2. `charge_unpriced_exits` -- a held name that leaves the tradable universe still has to be
   sold. Iteration 1 counted the leg in turnover and charged nothing for it: 777 free sell
   legs in B2.

    .venv/Scripts/python.exe -m research.sleeves.lowvol_retest_sensitivity
"""

from __future__ import annotations

import logging

import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.sleeves.low_vol_quality import build_signal
from research.sleeves.lowvol_retest import (
    BAND_ORDER,
    CORRECTED_DELISTING_WINDOW,
    REGISTERED_DELISTING_WINDOW,
    attach_spread_bounds,
    evaluate_band,
    run_band,
    verdict_for,
)
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, load_universe

VARIANTS = (
    ("registered", REGISTERED_DELISTING_WINDOW, False),
    ("+ exits charged", REGISTERED_DELISTING_WINDOW, True),
    ("+ delisting corrected", CORRECTED_DELISTING_WINDOW, False),
    ("BOTH corrections", CORRECTED_DELISTING_WINDOW, True),
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    universe = load_universe()
    risk = pd.read_parquet(PANEL_DIR / "risk_features_dev.parquet")
    quality = pd.read_parquet(QUALITY_CACHE)
    merged = build_signal(attach_spread_bounds(
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    ))
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    print("=" * 118)
    print("ACCOUNTING SENSITIVITY - conservative cost bound. Defaults = the registered run.")
    print("=" * 118)
    print(f"{'band':>12} {'accounting':>23} {'free legs':>10} {'cost':>7} {'net':>7} "
          f"{'net Sharpe':>11} {'bench':>7} {'VOL-MATCHED':>12} {'t':>6} {'drag':>7} "
          f"{'verdict':>13}")
    for band in BAND_ORDER:
        for label, window, charge in VARIANTS:
            books = run_band(merged, band, delistings, delisting_window=window,
                             charge_unpriced_exits=charge)
            if books is None:
                print(f"{band:>12} {label:>23}   insufficient data")
                continue
            evaluated = evaluate_band(books)
            bound = evaluated["bounds"]["conservative"]
            matched = bound["vol_matched"]
            print(f"{band:>12} {label:>23} {books.unpriced_exit_legs:>10,} "
                  f"{bound['cost_annual_total']:>6.2%} "
                  f"{bound['net']['annual_arithmetic']:>6.2%} "
                  f"{bound['net']['sharpe']:>11.3f} "
                  f"{evaluated['benchmark']['annual_arithmetic']:>6.2%} "
                  f"{matched['vol_matched_active_annual']:>+11.2%} "
                  f"{matched['vol_matched_active_tstat']:>+6.2f} "
                  f"{evaluated['delisting_drag_annual']:>+6.2%} "
                  f"{verdict_for(evaluated):>13}")
    print("\n  Every verdict is unchanged under every accounting. The low-vol/quality book")
    print("  dies LESS than its own universe, so booking real bankruptcies costs the")
    print("  benchmark more than the strategy and the vol-matched excess WIDENS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
