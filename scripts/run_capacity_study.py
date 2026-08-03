"""Run the capacity-curve study on the DEV window and print the registered verdict.

Design + errata: `research/medallion_style_alpha_search/capacity_curve_prereg.md`.
Prerequisite: `python scripts/build_capacity_panel.py`.

Reads nothing after 2015-12-31. The 2016+ confirmation window stays unfired.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.sharadar_ingestion import build_panel, load_sf1  # noqa: E402
from research.capacity_panel import BANDS, DATA_DIR, DEV_CUTOFF, PANEL_DIR  # noqa: E402
from research.capacity_study import (  # noqa: E402
    FACTOR_SIGNS,
    capacity_statistic,
    composite_signal,
    run_band,
)
from research.fundamental_features import compute_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("study")

FEATURE_CACHE = PANEL_DIR / "features_dev.parquet"


def build_features(panel: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not force:
        log.info("loading cached features")
        return pd.read_parquet(FEATURE_CACHE)

    log.info("loading SF1 (as-reported quarterly, PIT on filing datekey)")
    sf1 = load_sf1(DATA_DIR / "SF1.csv")
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    sf1 = sf1[sf1["datekey"] <= DEV_CUTOFF]
    log.info("SF1: %s filings, %s tickers", f"{len(sf1):,}",
             f"{sf1['ticker'].nunique():,}")

    # build_panel expects a price frame; give it exactly the monthly rebalance grid so
    # fundamentals are attached only at the dates the strategy actually acts on.
    grid = panel[["ticker", "date", "close"]].copy()
    grid["volume"] = 1.0

    log.info("attaching latest-known filing to each rebalance date (merge_asof)")
    merged = build_panel(sf1, grid)
    log.info("computing the 14 registered factors, normalised strictly per date")
    features = compute_features(merged, method="zscore")
    features.to_parquet(FEATURE_CACHE, index=False)
    return features


def main() -> int:
    panel_path = PANEL_DIR / "monthly_panel_dev.parquet"
    if not panel_path.exists():
        log.error("panel not found; run scripts/build_capacity_panel.py first")
        return 2

    panel = pd.read_parquet(panel_path)
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    panel = panel[panel["spread_regime"] != "ineligible"].copy()
    assert panel["date"].max() <= DEV_CUTOFF, "confirmation window leaked into DEV"
    log.info("panel: %s eligible cells, %s tickers, %s -> %s",
             f"{len(panel):,}", f"{panel['ticker'].nunique():,}",
             panel["date"].min().date(), panel["date"].max().date())

    features = build_features(panel)
    merged = panel.merge(features, on=["ticker", "date"], how="left")
    merged["signal"] = composite_signal(merged)
    covered = merged["signal"].notna().sum()
    log.info("signal computed for %s of %s cells (%.1f%%)", f"{covered:,}",
             f"{len(merged):,}", 100 * covered / max(len(merged), 1))

    print("\n" + "=" * 92)
    print("CAPACITY-CURVE STUDY - DEV WINDOW (<= 2015-12-31)")
    print("registered: research/medallion_style_alpha_search/capacity_curve_prereg.md")
    print("=" * 92)
    print(f"{'band':>16} {'capital':>11} {'net ret':>9} {'net vol':>9} "
          f"{'SHARPE':>8} {'maxDD':>8} {'bench':>8} {'cost':>8} {'turn':>7} "
          f"{'meas%':>7}")

    results = []
    for label, _, _ in BANDS:
        result = run_band(merged, label, delistings)
        if result is None:
            print(f"{label:>16} {'-':>11} {'insufficient measurable data':>60}")
            continue
        results.append(result)
        print(f"{label:>16} ${result.deployable_capital / 1e3:>9.0f}k "
              f"{result.net_return_annual:>8.1%} {result.net_volatility:>8.1%} "
              f"{result.net_sharpe:>8.2f} {result.max_drawdown:>7.1%} "
              f"{result.benchmark_return_annual:>7.1%} "
              f"{result.cost_drag_annual:>7.1%} {result.turnover_annual:>6.1f}x "
              f"{result.measured_share:>6.0%}")

    print("\n" + "=" * 92)
    print("PRIMARY TEST (one registered statistic, one trial)")
    print("=" * 92)
    statistic = capacity_statistic(results)
    print(f"  Spearman rho (deployable capital vs net Sharpe): "
          f"{statistic['rho']:>7.3f}")
    print(f"  one-sided permutation p-value:                   "
          f"{statistic['p_value']:>7.4f}")
    print(f"  bands with a usable measurement:                 "
          f"{statistic['n_bands']:>7}")
    print(f"\n  VERDICT: {statistic['verdict']}")

    excess = capacity_statistic(results, use_excess=True)
    print("\n" + "-" * 92)
    print("SECONDARY (declared, NOT the registered primary): the same test on EXCESS")
    print("over each band's own equal-weight buy-and-hold. Raw returns fall with")
    print("capacity partly because the small-cap premium does, so the primary statistic")
    print("alone cannot separate a capacity effect from a size effect.")
    print(f"  Spearman rho (capital vs excess Sharpe): {excess['rho']:>16.3f}")
    print(f"  one-sided permutation p-value:           {excess['p_value']:>16.4f}")
    print("\n  Band-by-band excess over buy-and-hold (annualised):")
    for band_result in results:
        delta = band_result.net_return_annual - band_result.benchmark_return_annual
        verdict = "BEATS" if delta > 0 else "LOSES TO"
        print(f"    {band_result.band:>16} {delta:>+8.1%}  {verdict} its own benchmark")

    print("\n" + "=" * 92)
    print("HONEST CAVEATS (registered, not added after seeing the numbers)")
    print("=" * 92)
    print("  - This tests the SHAPE of the capacity curve, not deployability. Any band")
    print("    claiming a deployable edge must separately clear selection_rule:")
    print("    rank-IC > 0.01, DSR >= 0.95, PBO, stability, leakage, regime floor.")
    print("  - 'meas%' is the share of band cells whose spread is genuinely MEASURED")
    print("    rather than floor-bounded. A low share means the band's cost model rests")
    print("    on few names and its Sharpe should be read with suspicion.")
    print("  - Costs are one-way per traded name: half-spread + sqrt impact +")
    print("    IBKR commission incl. the $0.35 per-order minimum + FX.")
    print("  - Delisting returns are applied by ACTIONS event type; acquisitions are")
    print("    booked flat, which UNDERSTATES the return to being acquired.")
    print(f"  - Signal is a fixed equal-weight composite of {len(FACTOR_SIGNS)} factors.")
    print("    Nothing was fitted, so no in-sample selection was possible.")
    print("  - Cumulative n_trials: 23 prior + 3 this study = 26.")

    for result in results:
        for note in result.notes:
            print(f"  ! {result.band}: {note}")

    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
