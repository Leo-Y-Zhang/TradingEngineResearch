"""Cache the SF1 ARQ slice the PEAD sleeve runs on.

SF1.csv is 2.4 GB and the sleeve only needs eight columns of one dimension. Extracting
once to parquet turns a 20-second read into a 0.5-second one, which is the difference
between a study that gets re-run and one that does not.

    .venv/Scripts/python.exe scripts/build_pead_inputs.py

The output is derived row-for-row from licensed Sharadar Data and is covered by the
purge obligation (`scripts/purge_sharadar_data.py` globs ``*.parquet``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.capacity_panel import DATA_DIR, PANEL_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_pead_inputs")

# `datekey` is the SEC filing date and the only announcement proxy used. `calendardate`
# is the quarter end and must never be joined on directly — it precedes publication by
# a month and a half.
COLUMNS = ["ticker", "dimension", "calendardate", "datekey", "reportperiod",
           "eps", "netinc", "shareswa"]


def main() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for chunk in pd.read_csv(DATA_DIR / "SF1.csv", usecols=COLUMNS,
                             chunksize=2_000_000, low_memory=False):
        # ARQ is the as-reported quarterly dimension. ART (trailing twelve months)
        # would smear a quarter's surprise across four of them.
        frames.append(chunk[chunk["dimension"] == "ARQ"])
    sf1 = pd.concat(frames, ignore_index=True)
    for column in ("calendardate", "datekey", "reportperiod"):
        sf1[column] = pd.to_datetime(sf1[column])

    out = PANEL_DIR / "sf1_arq_raw.parquet"
    sf1.to_parquet(out, index=False)
    logger.info("wrote %s ARQ filings to %s (datekey %s .. %s)",
                f"{len(sf1):,}", out.name, sf1["datekey"].min().date(),
                sf1["datekey"].max().date())


if __name__ == "__main__":
    main()
