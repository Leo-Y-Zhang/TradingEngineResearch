"""Aggregate SF3 (13F institutional holdings) and SF1 share counts into PIT caches.

SF3 is 79,190,744 rows / 2.9 GB of (ticker, investor, quarter) holdings. No code in this
repo has ever read it. This script reduces it, in one streaming pass, to one row per
(ticker, calendardate): total institutional shares held and the number of distinct
institutions holding. That reduction is the only thing the sleeve needs, and it is cached
so the study can be re-run without re-reading 2.9 GB.

**The point-in-time rule that governs this file.** SF3 carries `calendardate` (the quarter
end the holdings are as of) and NO filing date. 13F-HR is due 45 days after quarter end,
so holdings for 2015-06-30 were not public until 2015-08-14. Any study that joins SF3 on
calendardate is reading a six-week lookahead and is worthless. The availability date is
therefore computed here, once, as calendardate + 45 days, and every consumer joins on
THAT. Nothing downstream is permitted to see `calendardate` as a usable timestamp.

Two securitytype filters matter and are pre-specified:

* Only ``SHR`` rows are summed. SF3 also carries ``PUT``/``CLL`` (option positions) and
  ``FND``/``DBT``/``WNT``. An option position is not ownership of the float, and summing
  a put's "units" into shares held would count a bearish position as accumulation --
  precisely backwards for a flow signal.
* The DENOMINATOR is quarter-end dollar market capitalisation, NOT an SF1 share count.
  SF1 restates shares onto today's split basis while SF3 units are as-reported, so the
  two cannot be divided; see `build_quarter_end_marketcap` for the full argument and the
  validation. `build_sf1_shares` is retained because it is the diagnostic that exposed
  the mismatch, and because the mismatch is invisible until you look at a mega-cap.

All caches are derived row-for-row from licensed Sharadar Data and are covered by the
purge obligation (`scripts/purge_sharadar_data.py` globs ``*.parquet``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pa_csv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.capacity_panel import DATA_DIR, DEV_CUTOFF, PANEL_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# 13F-HR deadline: 45 calendar days after the quarter end. This is the whole PIT
# argument for the sleeve, so it lives in one named constant rather than inline.
FILING_LAG_DAYS = 45

# Only equity share positions count as ownership of the float.
OWNERSHIP_SECURITY_TYPE = "SHR"

SF3_CACHE = PANEL_DIR / "sf3_ownership_dev.parquet"
SF1_CACHE = PANEL_DIR / "sf1_shares_dev.parquet"
MARKETCAP_CACHE = PANEL_DIR / "quarter_end_marketcap_dev.parquet"

# How far back to look for the last trading day at or before a calendar quarter end.
# 2013-06-30 was a Sunday; a three-day Easter weekend is the worst realistic case.
QUARTER_END_LOOKBACK_DAYS = 7


def build_sf3_ownership(force: bool = False) -> pd.DataFrame:
    """One row per (ticker, calendardate): summed institutional shares and holder count.

    Streams the CSV in Arrow record batches rather than loading it: 79M rows at full
    width does not fit comfortably in memory, and the aggregation is associative so a
    per-batch groupby followed by a final groupby is exact, not an approximation.
    """
    if SF3_CACHE.exists() and not force:
        logger.info("loading cached SF3 ownership from %s", SF3_CACHE.name)
        return pd.read_parquet(SF3_CACHE)

    source = DATA_DIR / "SF3.csv"
    logger.info("streaming %s (%.1f GB)", source.name,
                source.stat().st_size / 1e9)

    convert = pa_csv.ConvertOptions(
        include_columns=["ticker", "securitytype", "calendardate", "units", "value"],
        column_types={
            "ticker": pa.string(),
            "securitytype": pa.string(),
            "calendardate": pa.string(),
            "units": pa.float64(),
            "value": pa.float64(),
        },
    )
    read = pa_csv.ReadOptions(block_size=64 << 20)

    partials: list[pd.DataFrame] = []
    rows_seen = 0
    earliest_any = None
    with pa_csv.open_csv(source, read_options=read, convert_options=convert) as reader:
        for batch in reader:
            rows_seen += batch.num_rows
            frame = batch.to_pandas()
            if earliest_any is None:
                earliest_any = frame["calendardate"].min()
            else:
                earliest_any = min(earliest_any, frame["calendardate"].min())

            # String comparison is safe on ISO dates and avoids parsing 79M timestamps
            # only to throw most of them away.
            frame = frame[
                (frame["securitytype"] == OWNERSHIP_SECURITY_TYPE)
                & (frame["calendardate"] <= DEV_CUTOFF.strftime("%Y-%m-%d"))
                & frame["units"].notna()
                & (frame["units"] > 0)
            ]
            if frame.empty:
                continue
            grouped = frame.groupby(["ticker", "calendardate"], sort=False).agg(
                inst_shares=("units", "sum"),
                inst_value_musd=("value", "sum"),
                n_holders=("units", "size"),
            )
            partials.append(grouped.reset_index())

    logger.info("read %s rows; earliest calendardate in file: %s",
                f"{rows_seen:,}", earliest_any)

    combined = pd.concat(partials, ignore_index=True)
    ownership = combined.groupby(["ticker", "calendardate"], sort=False).agg(
        inst_shares=("inst_shares", "sum"),
        inst_value_musd=("inst_value_musd", "sum"),
        n_holders=("n_holders", "sum"),
    ).reset_index()

    ownership["calendardate"] = pd.to_datetime(ownership["calendardate"])
    # The only timestamp downstream code is allowed to join on.
    ownership["available_date"] = ownership["calendardate"] + pd.Timedelta(
        days=FILING_LAG_DAYS
    )
    ownership = ownership.sort_values(["ticker", "calendardate"]).reset_index(drop=True)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    ownership.to_parquet(SF3_CACHE, index=False)
    logger.info("cached %s (ticker, quarter) cells to %s",
                f"{len(ownership):,}", SF3_CACHE.name)
    return ownership


def build_sf1_shares(force: bool = False) -> pd.DataFrame:
    """Point-in-time shares outstanding from SF1 ARQ, keyed by filing date.

    ``sharesbas * sharefactor`` is Sharadar's convention for total shares across all
    classes: `sharesbas` counts the class the filing reports on, and `sharefactor`
    scales it to the whole company. Using `sharesbas` alone would understate the float
    for dual-class names and inflate their computed institutional ownership.

    ``datekey`` is the SEC filing date, so this table is point-in-time as it stands;
    consumers filter ``datekey <= decision_date`` and take the latest row.
    """
    if SF1_CACHE.exists() and not force:
        logger.info("loading cached SF1 shares from %s", SF1_CACHE.name)
        return pd.read_parquet(SF1_CACHE)

    source = DATA_DIR / "SF1.csv"
    logger.info("streaming %s (%.1f GB)", source.name, source.stat().st_size / 1e9)

    convert = pa_csv.ConvertOptions(
        include_columns=["ticker", "dimension", "calendardate", "datekey",
                         "sharesbas", "sharefactor"],
        column_types={
            "ticker": pa.string(),
            "dimension": pa.string(),
            "calendardate": pa.string(),
            "datekey": pa.string(),
            "sharesbas": pa.float64(),
            "sharefactor": pa.float64(),
        },
    )
    read = pa_csv.ReadOptions(block_size=64 << 20)

    kept: list[pd.DataFrame] = []
    with pa_csv.open_csv(source, read_options=read, convert_options=convert) as reader:
        for batch in reader:
            frame = batch.to_pandas()
            frame = frame[
                (frame["dimension"] == "ARQ")
                & (frame["datekey"].notna())
                & (frame["datekey"] <= DEV_CUTOFF.strftime("%Y-%m-%d"))
                & frame["sharesbas"].notna()
                & (frame["sharesbas"] > 0)
            ]
            if not frame.empty:
                kept.append(frame)

    shares = pd.concat(kept, ignore_index=True)
    shares["calendardate"] = pd.to_datetime(shares["calendardate"])
    shares["datekey"] = pd.to_datetime(shares["datekey"])
    # A missing sharefactor means "one class", not "zero shares".
    factor = shares["sharefactor"].fillna(1.0).replace(0.0, 1.0)
    shares["shares_outstanding"] = shares["sharesbas"] * factor
    shares = shares[np.isfinite(shares["shares_outstanding"])
                    & (shares["shares_outstanding"] > 0)]

    # A restated quarter can appear twice; the LAST filing known at any decision date is
    # what a live system would have used, and dedup on (ticker, calendardate, datekey)
    # keeps every distinct filing so that choice stays with the consumer.
    shares = shares.drop_duplicates(["ticker", "calendardate", "datekey"], keep="last")
    shares = shares[["ticker", "calendardate", "datekey", "shares_outstanding"]]
    shares = shares.sort_values(["ticker", "datekey"]).reset_index(drop=True)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    shares.to_parquet(SF1_CACHE, index=False)
    logger.info("cached %s SF1 ARQ rows to %s", f"{len(shares):,}", SF1_CACHE.name)
    return shares


def build_quarter_end_marketcap(quarters: pd.DatetimeIndex,
                                force: bool = False) -> pd.DataFrame:
    """Dollar market capitalisation on the last trading day of each SF3 quarter.

    **This exists because `build_sf1_shares` cannot be used to scale SF3 holdings, and
    the reason is not obvious.** Sharadar SF1 restates share counts onto TODAY's split
    basis: AAPL's 2015-09-30 `sharesbas` reads 22.3 billion, which is its real 5.575
    billion multiplied by the 4:1 split of August 2020. SF3 `units`, by contrast, are as
    reported in the 13F at the time. Dividing one by the other gives AAPL a 15%
    institutional ownership where the truth is 58%, and the error is *per name*, scaled
    by each stock's own post-sample split history -- so it does not cancel in a
    cross-sectional z-score and it corrupts the quarter-on-quarter difference for any
    name that split.

    Dollar values are immune to all of this. SF3 `value` is the holding's market value in
    USD millions at the quarter-end price; SF1/DAILY `marketcap` is the company's market
    value in USD millions on the same day. Their ratio is the institutional ownership
    fraction, computed with no share count anywhere:

        ownership = sum(SF3.value) / DAILY.marketcap

    Validated on 2015-09-30: AAPL 57.6%, MSFT 72.3%, XOM 50.2%, JPM 74.9%, KO 63.7% --
    all correct to the published figures, against 0.01-0.08% via the SF1 share route.
    """
    if MARKETCAP_CACHE.exists() and not force:
        logger.info("loading cached quarter-end marketcap from %s",
                    MARKETCAP_CACHE.name)
        return pd.read_parquet(MARKETCAP_CACHE)

    quarters = pd.DatetimeIndex(sorted(pd.unique(quarters)))
    candidates = {
        (quarter - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        for quarter in quarters
        for back in range(QUARTER_END_LOOKBACK_DAYS + 1)
    }

    source = DATA_DIR / "DAILY.csv"
    logger.info("streaming %s (%.1f GB) for %d candidate dates",
                source.name, source.stat().st_size / 1e9, len(candidates))
    convert = pa_csv.ConvertOptions(
        include_columns=["ticker", "date", "marketcap"],
        column_types={"ticker": pa.string(), "date": pa.string(),
                      "marketcap": pa.float64()},
    )
    kept: list[pd.DataFrame] = []
    with pa_csv.open_csv(source,
                         read_options=pa_csv.ReadOptions(block_size=64 << 20),
                         convert_options=convert) as reader:
        for batch in reader:
            frame = batch.to_pandas()
            frame = frame[frame["date"].isin(candidates)]
            if not frame.empty:
                kept.append(frame)

    daily = pd.concat(kept, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["marketcap"].notna() & (daily["marketcap"] > 0)]
    daily = daily.sort_values(["ticker", "date"])

    # Last observation at or before each quarter end, per ticker.
    frames = []
    for quarter in quarters:
        window = daily[(daily["date"] <= quarter)
                       & (daily["date"] > quarter
                          - pd.Timedelta(days=QUARTER_END_LOOKBACK_DAYS + 1))]
        if window.empty:
            continue
        latest = window.groupby("ticker", sort=False).tail(1).copy()
        latest["calendardate"] = quarter
        frames.append(latest[["ticker", "calendardate", "date", "marketcap"]])

    marketcap = pd.concat(frames, ignore_index=True)
    marketcap = marketcap.rename(columns={"date": "marketcap_date"})
    marketcap = marketcap.sort_values(["ticker", "calendardate"]).reset_index(drop=True)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    marketcap.to_parquet(MARKETCAP_CACHE, index=False)
    logger.info("cached %s quarter-end marketcaps to %s",
                f"{len(marketcap):,}", MARKETCAP_CACHE.name)
    return marketcap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild caches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ownership = build_sf3_ownership(force=args.force)
    print("\nSF3 institutional ownership (DEV window only)")
    print(f"  cells                {len(ownership):>12,}")
    print(f"  tickers              {ownership['ticker'].nunique():>12,}")
    print(f"  quarters             {ownership['calendardate'].nunique():>12,}")
    print(f"  first calendardate   {ownership['calendardate'].min().date()!s:>12}")
    print(f"  last calendardate    {ownership['calendardate'].max().date()!s:>12}")
    print(f"  median holders/name  {ownership['n_holders'].median():>12,.0f}")

    shares = build_sf1_shares(force=args.force)
    print("\nSF1 ARQ shares outstanding (DEV window only)")
    print("  RETAINED FOR DIAGNOSIS ONLY - split-restated, unusable against SF3 units")
    print(f"  rows                 {len(shares):>12,}")
    print(f"  tickers              {shares['ticker'].nunique():>12,}")
    print(f"  first datekey        {shares['datekey'].min().date()!s:>12}")
    print(f"  last datekey         {shares['datekey'].max().date()!s:>12}")

    marketcap = build_quarter_end_marketcap(
        pd.DatetimeIndex(ownership["calendardate"].unique()), force=args.force)
    print("\nQuarter-end market capitalisation (the denominator actually used)")
    print(f"  rows                 {len(marketcap):>12,}")
    print(f"  tickers              {marketcap['ticker'].nunique():>12,}")

    check = ownership.merge(marketcap, on=["ticker", "calendardate"], how="inner")
    check["ownership"] = check["inst_value_musd"] / check["marketcap"]
    sample = check[check["calendardate"] == check["calendardate"].max()]
    print("  sanity, latest quarter:")
    for ticker in ("AAPL", "MSFT", "XOM", "JPM", "KO"):
        row = sample[sample["ticker"] == ticker]
        if not row.empty:
            print(f"    {ticker:<6} institutional ownership "
                  f"{row['ownership'].iloc[0]:>6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
