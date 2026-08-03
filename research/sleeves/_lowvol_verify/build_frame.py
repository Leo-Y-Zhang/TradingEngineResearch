"""Stage 0 of the adversarial verification: build the EXACT frame the re-test ran on.

Caches to a scratch directory OUTSIDE the repo (no Sharadar rows are ever written into
the working tree). Every later check loads this cache so the eight attacks are all run
against one frame that is provably the same one `lowvol_retest_run` used.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.build_frame
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR
from research.sleeves.low_vol_quality import build_signal
from research.sleeves.lowvol_retest import attach_spread_bounds
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, load_universe

log = logging.getLogger("verify.build")

SCRATCH = Path(os.environ.get("ENGINE_VERIFY_SCRATCH", Path.home() / ".engine_lowvol_verify"))
MERGED_CACHE = SCRATCH / "lowvol_retest_merged.parquet"

KEEP = [
    "ticker", "date", "close", "closeadj", "median_dollar_volume", "trading_fraction",
    "spread", "spread_regime", "band", "band_group", "forward_return",
    "realised_vol", "beta", "risk_n_obs", "zero_return_fraction",
    "gross_profitability", "debt_to_equity", "accruals",
    "spread_conservative", "spread_realistic",
    "leg_low_vol", "leg_low_beta", "leg_quality", "signal",
]


def build(force: bool = False) -> pd.DataFrame:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if MERGED_CACHE.exists() and not force:
        log.info("loading cached merged frame from %s", MERGED_CACHE)
        return pd.read_parquet(MERGED_CACHE)

    universe = load_universe()
    if universe["date"].max() > DEV_CUTOFF:
        raise ValueError("confirmation window leaked into the universe")
    risk = pd.read_parquet(PANEL_DIR / "risk_features_dev.parquet")
    quality = pd.read_parquet(QUALITY_CACHE)
    merged = (
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    )
    log.info("universe %s cells, %s tickers", f"{len(merged):,}",
             f"{merged['ticker'].nunique():,}")
    merged = attach_spread_bounds(merged)
    merged = build_signal(merged)
    keep = [c for c in KEEP if c in merged.columns]
    merged = merged[keep]
    merged.to_parquet(MERGED_CACHE, index=False)
    log.info("cached %s rows to %s", f"{len(merged):,}", MERGED_CACHE)
    return merged


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    frame = build(force=True)
    print(frame.shape)
    print(frame["spread_regime"].value_counts())
    print("signal defined:", int(frame["signal"].notna().sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
