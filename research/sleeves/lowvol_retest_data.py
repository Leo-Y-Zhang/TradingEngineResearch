"""Inputs for the LOW-VOL / QUALITY RE-TEST on the CORRECTED universe.

Split out from the sleeve itself because it is the only expensive step and because it is
the step the correction actually changes. Iteration 1 ranked names only inside the
``spread_regime == "measured"`` universe, so `quality_art_dev.parquet` was built on that
grid and covers 302,538 cells. The corrected universe also carries every ``upper_bound``
name -- 498,803 further (ticker, month) cells in bands B2..B6 -- and those cells have no
quality row at all. Re-using the old cache would silently make every added name
un-rankable, which would reinstate exactly the bias this re-test exists to remove.

Nothing here reads a forward return, a signal or a bar after the DEV cutoff.

    .venv/Scripts/python.exe -m research.sleeves.lowvol_retest_data
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.sharadar_ingestion import build_panel, load_sf1
from research.capacity_panel import DATA_DIR, DEV_CUTOFF, PANEL_DIR
from research.sleeves.low_vol_quality import QUALITY_LEGS, band_group

logger = logging.getLogger(__name__)

__all__ = ["QUALITY_CACHE", "SF1_DIMENSION", "build_quality_cache", "load_universe"]

# Distinct filename: the iteration-1 cache is a DIFFERENT (smaller) grid and both must
# survive so the two runs stay independently reproducible.
QUALITY_CACHE = PANEL_DIR / "quality_art_dev_lowvol_retest.parquet"

# Unchanged from iteration 1: As-Reported, Trailing Twelve Months. AR* is the original
# filing and is never restated; TTM because gross profit is a flow.
SF1_DIMENSION = "ART"

# Only the six fundamentals the three registered quality legs need. Restricting the read
# is a speed decision on a 2.4 GB CSV, not a construction change -- the same loader and
# the same point-in-time merge are used.
SF1_VALUE_COLUMNS = ("gp", "assets", "debt", "equity", "netinc", "ncfo")

# The regimes that MAY be traded. `unmeasurable` stays excluded (there is no honest cost
# for a name that barely traded); `ineligible` failed the panel's own price/volume
# filters before a spread was ever estimated. `upper_bound` is INCLUDED -- that is the
# correction. See research/spread_estimation.spread_cost_bounds.
TRADABLE_REGIMES = ("measured", "upper_bound")


def load_universe(panel_path: Path | None = None) -> pd.DataFrame:
    """The CORRECTED band universe: measured AND upper_bound cells, bands B2..B6."""
    path = panel_path or (PANEL_DIR / "monthly_panel_dev.parquet")
    panel = pd.read_parquet(path)
    if panel["date"].max() > DEV_CUTOFF:
        raise ValueError("confirmation window leaked into the monthly panel")
    panel["band_group"] = panel["band"].map(band_group)
    universe = panel[
        panel["spread_regime"].isin(TRADABLE_REGIMES) & panel["band_group"].notna()
    ].copy()
    return universe.reset_index(drop=True)


def build_quality_cache(grid: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Point-in-time gp/assets, debt/equity and accruals on ``grid``'s (ticker, date)."""
    if QUALITY_CACHE.exists() and not force:
        logger.info("loading cached quality features from %s", QUALITY_CACHE.name)
        return pd.read_parquet(QUALITY_CACHE)

    logger.info("loading SF1 dimension %s (as-reported TTM, PIT on filing datekey)",
                SF1_DIMENSION)
    sf1 = load_sf1(DATA_DIR / "SF1.csv", dimension=SF1_DIMENSION,
                   value_columns=SF1_VALUE_COLUMNS)
    sf1 = sf1[sf1["datekey"] <= DEV_CUTOFF]
    logger.info("SF1: %s filings, %s tickers, datekey %s -> %s", f"{len(sf1):,}",
                f"{sf1['ticker'].nunique():,}", sf1["datekey"].min().date(),
                sf1["datekey"].max().date())

    merged = build_panel(sf1, grid, dimension=SF1_DIMENSION,
                         value_columns=SF1_VALUE_COLUMNS)
    del sf1

    # datekey <= date is what makes this point-in-time. Assert it rather than assume it.
    filed = merged["filed_datekey"]
    leaked = int((filed.notna() & (filed > merged["date"])).sum())
    if leaked:
        raise ValueError(f"{leaked} rows carry a filing published after the rebalance date")

    assets = merged["assets"].replace(0.0, np.nan)
    equity = merged["equity"].replace(0.0, np.nan)
    out = pd.DataFrame({
        "ticker": merged["ticker"],
        "date": merged["date"],
        "gross_profitability": merged["gp"] / assets,
        "debt_to_equity": merged["debt"] / equity,
        "accruals": (merged["netinc"] - merged["ncfo"]) / assets,
    }).replace([np.inf, -np.inf], np.nan)

    out.to_parquet(QUALITY_CACHE, index=False)
    for column in QUALITY_LEGS:
        logger.info("  %s: %s of %s cells populated", column,
                    f"{out[column].notna().sum():,}", f"{len(out):,}")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    universe = load_universe()
    logger.info("corrected universe: %s cells (%s measured, %s upper_bound), %s tickers",
                f"{len(universe):,}",
                f"{int((universe['spread_regime'] == 'measured').sum()):,}",
                f"{int((universe['spread_regime'] == 'upper_bound').sum()):,}",
                f"{universe['ticker'].nunique():,}")
    grid = universe[["ticker", "date", "close"]].copy()
    grid["volume"] = 1.0  # build_panel wants a price frame; only ticker/date are used
    build_quality_cache(grid, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
