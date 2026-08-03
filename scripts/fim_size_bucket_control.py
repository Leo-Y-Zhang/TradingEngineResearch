"""Control for WHERE the published cost benchmarks sit on our liquidity axis.

Every cost check in this repo compares a modelled number against Frazzini, Israel &
Moskowitz (2018), "Trading Costs" -- $1.7tn of live institutional executions, Aug 1998 -
Jun 2016, average trade 0.9% of daily volume. Table II Panel A gives a median all-in
one-way cost of 5.54bps for large caps and 13.53bps for small caps. Those comparisons are
only worth anything if the buckets are placed at the right liquidity, and one of them was
not: `scripts/impact_positive_control.py` mapped "FIM small cap" onto $10M-$50M/day on the
strength of an argument rather than a measurement, and the internal research log iteration 4 then
used that mapping to explain away a 3.6-4.4x discrepancy in the spread schedule.

This script decides that question with data instead. Three checks:

  A. THE MAPPING IS READ OFF THE PAPER, NOT GUESSED. FIM define the split by benchmark:
     "The distinction between large and small cap is based on the portfolio's benchmark
     (e.g. for the US large cap is the Russell 1000 and small cap is below the Russell
     1000 in market cap, typically within the Russell 2000 universe)" (Table II note, and
     again in §II.A). That is a market-cap RANK, so it is reproduced here as a market-cap
     rank -- ranks 1-1000 and 1001-3000 of the US domestic-common-stock cross-section --
     and the resulting dollar volumes are MEASURED rather than asserted.

  B. THE REPRODUCTION IS VALIDATED AGAINST TWO PUBLISHED NUMBERS (out of sample). FIM
     Table IX Panel A publishes the average daily dollar volume of the S&P 500 ($662.83M)
     and of the Russell 2000 ($14.76M). Both are capitalisation-weighted averages over
     the index. Our rank buckets must reproduce both. Nothing in this repo is calibrated
     on Table IX, so if the rank mapping were wrong this check would say so.

  C. THE SCHEDULE'S ANCHORS ARE AGK'S OWN QUINTILES. `AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME`
     claims to be the dollar volume of each MARKET-CAP quintile of the unscreened
     CRSP-TAQ universe that Ardia-Guidotti-Kroencke (2024) Table 4 Panel C quintiles. It
     is re-measured here and must match. The SUPERSEDED anchors must fail the same test --
     a drift detector that only the shipped constants can pass is not a detector.

Exit codes: 0 all checks pass, 1 a check failed, 2 the data is not available.

Runtime is a few minutes: it reads market capitalisation from DAILY and dollar volume
from SEP, both in full. It touches no signal, no strategy and no forward return -- it
measures how liquid a rank bucket is -- which is why it is allowed to read the 2016-2020
bars that AGK's own sample covers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.spread_estimation import (  # noqa: E402
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED,
    FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME,
    FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME,
    FIM_SMALL_CAP_DOLLAR_VOLUME_IQR,
    FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME,
    FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME,
    FIM_SMALL_CAP_RANK_RANGE,
)

SHARADAR = REPO / "_data" / "sharadar"
SEP = SHARADAR / "SEP.csv"
DAILY = SHARADAR / "DAILY.csv"
TICKERS = SHARADAR / "TICKERS.csv"

# AGK's sample is 1993-2020 and this tape starts in 1998; FIM's is Aug 1998 - Jun 2016.
# Each constant is measured on the window that matches ITS source, and the windows are
# named here rather than buried in the code.
AGK_WINDOW = ("1998-01-01", "2020-12-31")
FIM_WINDOW = ("1998-01-01", "2015-12-31")
# FIM Table IX is calibrated on fund sizes "in 2016", so its two published dollar volumes
# describe the end of the sample, not its average.
FIM_INDEX_WINDOW = ("2011-01-01", "2015-12-31")

# AGK §3.1 keep "all NYSE, AMEX, and NASDAQ stocks with CRSP share codes of 10 or 11
# (i.e., U.S. common shares)" and do nothing else to the sample. This is that filter,
# expressed in Sharadar's vocabulary.
EXCHANGES = ("NYSE", "NASDAQ", "NYSEMKT", "NYSEARCA", "AMEX")
CATEGORY = "Domestic Common Stock"
MIN_BARS_PER_MONTH = 15

# Tolerances, registered before any number is printed.
INDEX_TOLERANCE_RELATIVE = 0.15   # check B: our reproduction vs FIM's published figures
ANCHOR_TOLERANCE_RELATIVE = 0.05  # check C: shipped anchors vs a fresh measurement
# The mapping check A refutes. If the measured median were above this, explanation (a) --
# "FIM's small cap is simply more liquid than $1M-$10M/day" -- would have been right.
REGISTERED_SMALL_CAP_FLOOR = 1.0e7


def _eligible_tickers() -> set[str]:
    frame = pd.read_csv(TICKERS, usecols=["table", "ticker", "exchange", "category"])
    frame = frame[frame["table"] == "SEP"]
    frame = frame[frame["category"].astype(str).str.contains(CATEGORY, na=False)]
    frame = frame[frame["exchange"].isin(EXCHANGES)]
    return set(frame["ticker"].dropna().unique())


def _month_end_market_cap(start: str, end: str) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(DAILY, usecols=["ticker", "date", "marketcap"],
                             chunksize=5_000_000, parse_dates=["date"]):
        chunk = chunk[(chunk["date"] >= start) & (chunk["date"] <= end)]
        chunk = chunk[chunk["marketcap"] > 0]
        if chunk.empty:
            continue
        chunk["month"] = chunk["date"].values.astype("datetime64[M]")
        chunk = chunk.sort_values(["ticker", "month", "date"], kind="stable")
        frames.append(chunk.groupby(["ticker", "month"], observed=True,
                                    as_index=False).last())
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["ticker", "month", "date"], kind="stable")
    return data.groupby(["ticker", "month"], observed=True, as_index=False).last()[
        ["ticker", "month", "marketcap"]]


def _monthly_dollar_volume(start: str, end: str) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(SEP, usecols=["ticker", "date", "close", "volume"],
                             chunksize=5_000_000, parse_dates=["date"]):
        chunk = chunk[(chunk["date"] >= start) & (chunk["date"] <= end)]
        if chunk.empty:
            continue
        chunk["dollar_volume"] = chunk["close"] * chunk["volume"]
        chunk["month"] = chunk["date"].values.astype("datetime64[M]")
        frames.append(chunk.groupby(["ticker", "month"],
                                    observed=True)["dollar_volume"].agg(
            ["sum", "size"]).reset_index())
    data = pd.concat(frames, ignore_index=True)
    data = data.groupby(["ticker", "month"], observed=True, as_index=False).sum()
    # The mean of daily dollar volume within the month. A per-day median would need the
    # daily rows kept in memory for 29 million bars; the mean is what the sum/count gives
    # and it is the LARGER of the two on a right-skewed distribution, so using it makes
    # every anchor dearer rather than cheaper.
    data["dollar_volume"] = data["sum"] / data["size"]
    return data[(data["size"] >= MIN_BARS_PER_MONTH)
                & (data["dollar_volume"] > 0)][["ticker", "month", "dollar_volume"]]


def build_panel() -> pd.DataFrame:
    """(ticker, month, market cap, dollar volume, within-month market-cap rank)."""
    start, end = AGK_WINDOW
    print(f"  reading market capitalisation from DAILY ({start}..{end})...", flush=True)
    market_cap = _month_end_market_cap(start, end)
    print(f"    {len(market_cap):,} month-end observations", flush=True)
    print(f"  reading dollar volume from SEP ({start}..{end})...", flush=True)
    volume = _monthly_dollar_volume(start, end)
    print(f"    {len(volume):,} (name, month) observations", flush=True)

    panel = market_cap.merge(volume, on=["ticker", "month"])
    panel = panel[panel["ticker"].isin(_eligible_tickers())].reset_index(drop=True)
    panel["rank"] = panel.groupby("month")["marketcap"].rank(ascending=False,
                                                             method="first")
    print(f"  panel: {len(panel):,} (name, month) cells, "
          f"{panel['ticker'].nunique():,} names, median "
          f"{panel.groupby('month').size().median():,.0f} names per month", flush=True)
    return panel


def _window(panel: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    start, end = window
    return panel[(panel["month"] >= start) & (panel["month"] <= end)]


def _cap_weighted_dollar_volume(frame: pd.DataFrame) -> float:
    """Median across months of the cap-weighted mean dollar volume -- FIM's ``dtv``."""
    def weighted(group: pd.DataFrame) -> float:
        weights = group["marketcap"] / group["marketcap"].sum()
        return float((weights * group["dollar_volume"]).sum())

    return float(frame.groupby("month")[["marketcap", "dollar_volume"]].apply(
        weighted).median())


def check_a_mapping(panel: pd.DataFrame) -> bool:
    print("=" * 78)
    print("A. FIM'S SIZE BUCKETS, READ OFF THE PAPER AND MEASURED ON THIS TAPE")
    print("=" * 78)
    low, high = FIM_SMALL_CAP_RANK_RANGE
    start, end = FIM_WINDOW
    frame = _window(panel, FIM_WINDOW)
    print('  FIM: "small cap is below the Russell 1000 in market cap, typically within')
    print(f'  the Russell 2000 universe" -> market-cap ranks {low}-{high}, {start}..{end}')
    print(f"  ({len(frame):,} cells; FIM's own sample is Aug 1998 - Jun 2016)\n")

    small = frame[(frame["rank"] >= low) & (frame["rank"] <= high)]
    large = frame[frame["rank"] < low]
    monthly_small = small.groupby("month")["dollar_volume"]
    monthly_large = large.groupby("month")["dollar_volume"]

    print(f"{'':>4}{'bucket':<28}{'cells':>10}{'p25':>13}{'median':>13}{'p75':>13}")
    for label, monthly in (("small cap (rank 1001-3000)", monthly_small),
                           ("large cap (rank 1-1000)", monthly_large)):
        print(f"    {label:<28}{int(monthly.size().sum()):>10,}"
              f"{monthly.quantile(0.25).median() / 1e6:>12,.2f}M"
              f"{monthly.median().median() / 1e6:>12,.2f}M"
              f"{monthly.quantile(0.75).median() / 1e6:>12,.2f}M")

    measured_small = float(monthly_small.median().median())
    measured_large = float(monthly_large.median().median())

    print(f"\n{'':>4}{'quantity':<44}{'frozen':>14}{'measured now':>16}   verdict")
    frozen_ok = True
    for label, frozen, measured in (
        ("FIM small cap, median $/day", FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME,
         measured_small),
        ("FIM large cap, median $/day", FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME,
         measured_large),
        ("FIM small cap, 25th percentile", FIM_SMALL_CAP_DOLLAR_VOLUME_IQR[0],
         float(monthly_small.quantile(0.25).median())),
        ("FIM small cap, 75th percentile", FIM_SMALL_CAP_DOLLAR_VOLUME_IQR[1],
         float(monthly_small.quantile(0.75).median())),
    ):
        drift = abs(measured - frozen) / frozen
        ok = drift <= ANCHOR_TOLERANCE_RELATIVE
        frozen_ok &= ok
        print(f"    {label:<44}{frozen:>14,.0f}{measured:>16,.0f}   "
              f"{'ok' if ok else f'DRIFTED {drift:.1%}'}")

    # The falsifiable part. If the median small-cap name really did trade above $10M/day,
    # the mapping the impact control registered would have been right and the spread
    # schedule would have had nothing to answer for.
    refutes = measured_small < REGISTERED_SMALL_CAP_FLOOR
    print(f"\n    the registered mapping said FIM small cap = "
          f"${REGISTERED_SMALL_CAP_FLOOR / 1e6:.0f}M-$50M/day;")
    print(f"    the median constituent measures ${measured_small / 1e6:.2f}M/day, "
          f"which is BELOW that floor: {refutes}")
    print(f"    for scale, ${REGISTERED_SMALL_CAP_FLOOR / 1e6:.0f}M-$50M/day spans the "
          f"BOTTOM HALF of the Russell 1000 on this tape")
    print(f"    (rank 1-1000: p25 "
          f"${float(monthly_large.quantile(0.25).median()) / 1e6:,.1f}M, median "
          f"${measured_large / 1e6:,.1f}M).")

    passed = bool(frozen_ok and refutes)
    print(f"\n  CHECK A: {'PASS' if passed else 'FAIL'}\n")
    return passed


def check_b_published_index_volumes(panel: pd.DataFrame) -> bool:
    print("=" * 78)
    print("B. THE RANK MAPPING REPRODUCES TWO PUBLISHED INDEX DOLLAR VOLUMES (OUT OF "
          "SAMPLE)")
    print("=" * 78)
    start, end = FIM_INDEX_WINDOW
    frame = _window(panel, FIM_INDEX_WINDOW)
    low, high = FIM_SMALL_CAP_RANK_RANGE
    print("  FIM Table IX Panel A publishes each index's capitalisation-weighted average")
    print(f"  daily dollar volume for 2016. Reproduced here over {start}..{end}.")
    print("  Nothing in this repo is calibrated on Table IX.\n")
    print(f"{'':>4}{'index':<32}{'published':>14}{'measured':>14}{'error':>9}   verdict")

    passed = True
    for label, published, selected in (
        ("S&P 500 (top 500 by market cap)", FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME,
         frame[frame["rank"] <= 500]),
        (f"Russell 2000 (rank {low}-{high})", FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME,
         frame[(frame["rank"] >= low) & (frame["rank"] <= high)]),
    ):
        measured = _cap_weighted_dollar_volume(selected)
        error = (measured - published) / published
        ok = abs(error) <= INDEX_TOLERANCE_RELATIVE
        passed &= ok
        print(f"    {label:<32}{published / 1e6:>13,.1f}M{measured / 1e6:>13,.1f}M"
              f"{error:>+8.1%}   {'ok' if ok else 'MAPPING DOES NOT REPRODUCE THE PAPER'}")

    print(f"\n    Required: both within {INDEX_TOLERANCE_RELATIVE:.0%}. If the rank")
    print("    mapping were not FIM's universe, these two numbers would not land.")
    print(f"\n  CHECK B: {'PASS' if passed else 'FAIL'}\n")
    return bool(passed)


def check_c_schedule_anchors(panel: pd.DataFrame) -> bool:
    print("=" * 78)
    print("C. THE SCHEDULE'S ANCHORS ARE AGK'S MARKET-CAP QUINTILES (+ CONTROL OF THE "
          "CONTROL)")
    print("=" * 78)
    start, end = AGK_WINDOW
    frame = _window(panel, AGK_WINDOW).copy()
    frame["quintile"] = frame.groupby("month")["marketcap"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 5,
                               labels=[1, 2, 3, 4, 5]).astype(int))
    print("  AGK Table 4 Panel C quintiles the CRSP-TAQ universe by market cap over")
    print(f"  1993-2020 (1,626,448 cells, ~4,840 names/month). Reproduced over "
          f"{start}..{end}:")
    print(f"  {len(frame):,} cells, median "
          f"{frame.groupby('month').size().median():,.0f} names/month.\n")

    print(f"{'':>4}{'quintile':<10}{'med mcap':>12}{'shipped anchor':>17}"
          f"{'measured':>14}{'error':>9}{'superseded':>14}{'error':>9}")
    shipped_ok, superseded_errors = True, []
    for index in (1, 2, 3, 4, 5):
        selected = frame[frame["quintile"] == index]
        measured = float(selected.groupby("month")["dollar_volume"].median().median())
        shipped = AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME[index - 1]
        superseded = AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED[index - 1]
        drift = abs(measured - shipped) / measured
        superseded_drift = abs(measured - superseded) / measured
        superseded_errors.append(superseded_drift)
        shipped_ok &= drift <= ANCHOR_TOLERANCE_RELATIVE
        print(f"    Q{index:<9}{selected['marketcap'].median():>10,.0f}m"
              f"{shipped:>16,.0f}{measured:>14,.0f}{drift:>+8.1%}"
              f"{superseded:>14,.0f}{superseded_drift:>+8.1%}")

    detected = max(superseded_errors) > ANCHOR_TOLERANCE_RELATIVE
    print(f"\n    shipped anchors match a fresh measurement within "
          f"{ANCHOR_TOLERANCE_RELATIVE:.0%}: {shipped_ok}")
    print(f"    the superseded anchors do NOT (worst {max(superseded_errors):.0%}), so "
          f"this check has teeth: {detected}")
    print("    The superseded set was the study's own eligible-universe DOLLAR-VOLUME")
    print("    quintiles. AGK quintile an unscreened universe by MARKET CAP, so quintile")
    print("    k of the screened set is a strictly more liquid group of names and every")
    print("    spread level was pinned too far to the right.")

    passed = bool(shipped_ok and detected)
    print(f"\n  CHECK C: {'PASS' if passed else 'FAIL'}\n")
    return passed


def main() -> int:
    for path in (SEP, DAILY, TICKERS):
        if not path.exists():
            print(f"ERROR: {path} not found. "
                  f"Run scripts/download_sharadar_data.py first.")
            return 2

    print("Measuring the tape (this reads DAILY and SEP in full)...", flush=True)
    panel = build_panel()
    print()

    results = {
        "A FIM buckets measured, not guessed": check_a_mapping(panel),
        "B published index volumes reproduced": check_b_published_index_volumes(panel),
        "C schedule anchors are AGK's quintiles": check_c_schedule_anchors(panel),
    }

    print("=" * 78)
    for label, ok in results.items():
        print(f"  {label:<42} {'PASS' if ok else 'FAIL'}")
    passed = all(results.values())
    print()
    if passed:
        print("  CONTROL PASSED - FIM's small-cap bucket really does sit in $1M-$10M/day,")
        print("  so the discrepancy recorded in internal research log iteration 4 was NOT a bucket")
        print("  mismatch, and `AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME` is the liquidity AGK")
        print("  actually measured their quintile spreads on.")
    else:
        print("  CONTROL FAILED - the liquidity mapping underneath every published-cost")
        print("  comparison in this repo does not hold. No cost number benchmarked against")
        print("  FIM or AGK may be reported until it does.")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
