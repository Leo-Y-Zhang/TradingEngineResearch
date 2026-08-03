"""
TradingEngineResearch — tests for the insider-transaction monthly feature panels
(``research.insider_features``). Offline, deterministic, NO network.

The properties under test:

  1. PIT month assignment: a filing enters the month whose month-end is
     >= filing_date + 1 BUSINESS day — a filing ON the month-end day lands in the
     NEXT month; a filing whose availability falls after that month's PANEL date
     (holiday month-end) also lands in the next month.
  2. No-lookahead property: adding a transaction filed AFTER a panel date cannot
     change any feature value at or before that date.
  3. net_buy_ratio_6m / net_buy_value_6m arithmetic (count vs dollar-value based),
     officer+director-only filtering, amendment exclusion.
  4. cluster_buying_3m counts DISTINCT owner CIKs, respecting the 3-month window.
  5. Routine-insider detection (Cohen-Malloy-Pomorski simplified): positive case
     (same-calendar-month purchases in each of the 3 preceding years) is stripped
     from opportunistic_buy_6m; negative case (only 2 preceding years) is not.
  6. Trailing-window edges (6 calendar months inclusive) and buy_intensity sums.
  7. Per-date cross-sectional z-score normalization; no-activity symbols stay NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.insider_features import (
    INSIDER_FEATURES,
    compute_insider_features,
    raw_insider_features,
)

# Calendar month-end panel dates (2020; the runner passes trading month-ends).
DATES_2020 = list(pd.date_range("2020-01-31", "2020-12-31", freq="ME"))


def _txn(ticker: str, filing: str, code: str = "P", shares: float = 100.0,
         price: float = 10.0, owner: str = "1", rel: str = "OFFICER",
         amendment: bool = False, accession: str | None = None) -> dict[str, object]:
    filing_ts = pd.Timestamp(filing)
    # Default accession is unique per (issuer, filing, owner, transaction) — separate
    # filings by distinct owners each carry their own accession, as on EDGAR. Pass the
    # SAME accession explicitly to model a multi-owner fan-out of ONE filing.
    acc = accession or f"{ticker}|{filing}|{owner}|{code}|{shares}|{price}"
    return {
        "ticker": ticker, "filing_date": filing_ts,
        "trans_date": filing_ts - pd.Timedelta(days=2), "trans_code": code,
        "shares": shares, "price": price, "owner_cik": owner, "relationship": rel,
        "direct_indirect": "D", "is_amendment": amendment,
        "shrs_owned_after": 1000.0, "accession": acc,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _cell(tidy: pd.DataFrame, date: str, ticker: str, feature: str) -> float:
    sel = tidy[(tidy["date"] == pd.Timestamp(date)) & (tidy["ticker"] == ticker)]
    assert len(sel) == 1, f"expected exactly one ({date}, {ticker}) row"
    return float(sel.iloc[0][feature])


# --------------------------------------------------------------------------- #
# 1. PIT month assignment
# --------------------------------------------------------------------------- #
def test_filing_on_month_end_day_enters_next_month() -> None:
    # 2020-01-31 is a Friday: availability = Mon 2020-02-03 -> February bucket.
    txns = _frame([_txn("AAA", "2020-01-31")])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA"])
    assert np.isnan(_cell(tidy, "2020-01-31", "AAA", "net_buy_ratio_6m"))
    assert _cell(tidy, "2020-02-29", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0)


def test_mid_month_filing_enters_its_own_month() -> None:
    txns = _frame([_txn("AAA", "2020-01-15")])   # Wed; avail Thu 16 Jan -> January
    tidy = raw_insider_features(txns, DATES_2020, ["AAA"])
    assert _cell(tidy, "2020-01-31", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0)


def test_holiday_month_end_availability_after_panel_date_shifts_forward() -> None:
    # Panel date Fri 2021-05-28 (Memorial Day made the 31st a non-trading month end).
    # Filing Fri 2021-05-28 -> avail Mon 2021-05-31 (a BUSINESS day) which is AFTER
    # the May panel date -> must land in JUNE, not May (no 1-day leak).
    dates = [pd.Timestamp("2021-04-30"), pd.Timestamp("2021-05-28"),
             pd.Timestamp("2021-06-30")]
    txns = _frame([_txn("AAA", "2021-05-28")])
    tidy = raw_insider_features(txns, dates, ["AAA"])
    assert np.isnan(_cell(tidy, "2021-05-28", "AAA", "net_buy_ratio_6m"))
    assert _cell(tidy, "2021-06-30", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. No-lookahead property
# --------------------------------------------------------------------------- #
def test_late_filing_cannot_change_earlier_panel_rows() -> None:
    base_rows = [
        _txn("AAA", "2020-02-10", code="P"),
        _txn("AAA", "2020-03-12", code="S", owner="2"),
        _txn("BBB", "2020-04-14", code="P", owner="3"),
    ]
    late = _txn("AAA", "2020-07-06", code="S", owner="4")   # filed after 2020-06-30
    base = raw_insider_features(_frame(base_rows), DATES_2020, ["AAA", "BBB"])
    with_late = raw_insider_features(_frame(base_rows + [late]), DATES_2020,
                                     ["AAA", "BBB"])
    cutoff = pd.Timestamp("2020-06-30")
    pd.testing.assert_frame_equal(
        base[base["date"] <= cutoff].reset_index(drop=True),
        with_late[with_late["date"] <= cutoff].reset_index(drop=True),
    )
    # Sanity: the late filing DOES matter later (otherwise the test proves nothing).
    assert (_cell(base, "2020-07-31", "AAA", "net_buy_ratio_6m")
            != _cell(with_late, "2020-07-31", "AAA", "net_buy_ratio_6m"))


# --------------------------------------------------------------------------- #
# 3. net_buy_ratio / net_buy_value arithmetic + filters
# --------------------------------------------------------------------------- #
def test_net_buy_ratio_count_based_officers_directors_only() -> None:
    txns = _frame([
        _txn("AAA", "2020-03-03", code="P", owner="1", rel="OFFICER"),
        _txn("AAA", "2020-03-04", code="P", owner="2", rel="DIRECTOR"),
        _txn("AAA", "2020-04-06", code="P", owner="3", rel="DIRECTOR,OFFICER"),
        _txn("AAA", "2020-04-07", code="S", owner="1", rel="OFFICER"),
        _txn("AAA", "2020-04-08", code="P", owner="9", rel="TENPERCENTOWNER"),  # excluded
        _txn("AAA", "2020-04-09", code="P", owner="8", rel="OFFICER", amendment=True),  # excluded
        _txn("AAA", "2020-04-10", code="A", owner="7", rel="OFFICER"),  # award: not P/S
    ])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA"])
    # Window at 2020-04-30 (6m): 3 officer/director buys, 1 sell -> (3-1)/4 = 0.5
    assert _cell(tidy, "2020-04-30", "AAA", "net_buy_ratio_6m") == pytest.approx(0.5)


def test_net_buy_value_dollar_weighted_and_nan_price_excluded_from_value() -> None:
    txns = _frame([
        _txn("AAA", "2020-03-03", code="P", shares=100.0, price=10.0),   # $1000 buy
        _txn("AAA", "2020-03-05", code="S", shares=100.0, price=30.0),   # $3000 sell
        _txn("AAA", "2020-03-09", code="P", shares=50.0, price=np.nan),  # counts, no value
    ])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA"])
    # Counts: (2-1)/3 ; value: (1000-3000)/4000 = -0.5
    assert _cell(tidy, "2020-03-31", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0 / 3.0)
    assert _cell(tidy, "2020-03-31", "AAA", "net_buy_value_6m") == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# 4. cluster_buying_3m distinctness + window
# --------------------------------------------------------------------------- #
def test_cluster_counts_distinct_owner_ciks() -> None:
    txns = _frame([
        _txn("AAA", "2020-04-06", code="P", owner="1"),
        _txn("AAA", "2020-05-06", code="P", owner="1"),        # same owner again -> 1
        _txn("AAA", "2020-06-08", code="P", owner="2"),        # second distinct owner
        _txn("AAA", "2020-01-08", code="P", owner="5"),        # outside 3m at June
        _txn("BBB", "2020-06-09", code="S", owner="6"),        # activity, no buys
    ])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA", "BBB", "CCC"])
    assert _cell(tidy, "2020-06-30", "AAA", "cluster_buying_3m") == pytest.approx(2.0)
    assert _cell(tidy, "2020-05-31", "AAA", "cluster_buying_3m") == pytest.approx(1.0)
    # Activity but zero buyers -> 0; NO activity at all -> NaN.
    assert _cell(tidy, "2020-06-30", "BBB", "cluster_buying_3m") == pytest.approx(0.0)
    assert np.isnan(_cell(tidy, "2020-06-30", "CCC", "cluster_buying_3m"))


# --------------------------------------------------------------------------- #
# 5. Routine-insider detection (positive + negative)
# --------------------------------------------------------------------------- #
def _march_buys(owner: str, years: list[int]) -> list[dict[str, object]]:
    return [_txn("AAA", f"{y}-03-16", code="P", owner=owner) for y in years]


def test_routine_insider_stripped_from_opportunistic() -> None:
    dates = list(pd.date_range("2019-06-30", "2020-12-31", freq="ME"))
    # Owner 1 bought AAA every March 2017/2018/2019 -> the March-2020 buy is ROUTINE.
    rows = _march_buys("1", [2017, 2018, 2019, 2020])
    tidy = raw_insider_features(_frame(rows), dates, ["AAA"])
    # net_buy_ratio still sees the (routine) buy...
    assert _cell(tidy, "2020-03-31", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0)
    # ...opportunistic strips it: no non-routine trades left -> NaN (undefined ratio).
    assert np.isnan(_cell(tidy, "2020-03-31", "AAA", "opportunistic_buy_6m"))


def test_two_prior_years_is_not_routine() -> None:
    dates = list(pd.date_range("2019-06-30", "2020-12-31", freq="ME"))
    rows = _march_buys("1", [2018, 2019, 2020])     # only 2 preceding years -> NOT routine
    tidy = raw_insider_features(_frame(rows), dates, ["AAA"])
    assert _cell(tidy, "2020-03-31", "AAA", "opportunistic_buy_6m") == pytest.approx(1.0)


def test_routine_detection_is_per_owner_and_month() -> None:
    dates = list(pd.date_range("2019-06-30", "2020-12-31", freq="ME"))
    rows = _march_buys("1", [2017, 2018, 2019])
    # A DIFFERENT owner buying in March 2020 is NOT routine (no such history).
    rows += [_txn("AAA", "2020-03-16", code="P", owner="2")]
    tidy = raw_insider_features(_frame(rows), dates, ["AAA"])
    assert _cell(tidy, "2020-03-31", "AAA", "opportunistic_buy_6m") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 6. Trailing-window edges + buy_intensity
# --------------------------------------------------------------------------- #
def test_six_month_window_is_inclusive_of_t_minus_5_periods() -> None:
    txns = _frame([
        _txn("AAA", "2020-01-08", code="P"),       # Jan -> IN the Jan..Jun window
        _txn("BBB", "2019-12-10", code="P"),       # Dec -> OUT at 2020-06-30
    ])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA", "BBB"])
    assert _cell(tidy, "2020-06-30", "AAA", "net_buy_ratio_6m") == pytest.approx(1.0)
    assert np.isnan(_cell(tidy, "2020-06-30", "BBB", "net_buy_ratio_6m"))


def test_buy_intensity_sums_dollar_value() -> None:
    txns = _frame([
        _txn("AAA", "2020-02-05", code="P", shares=100.0, price=10.0),  # $1000
        _txn("AAA", "2020-03-05", code="P", shares=200.0, price=5.0),   # $1000
        _txn("BBB", "2020-03-06", code="S", shares=10.0, price=1.0),    # sells only
    ])
    tidy = raw_insider_features(txns, DATES_2020, ["AAA", "BBB"])
    assert _cell(tidy, "2020-03-31", "AAA", "buy_intensity_6m") == pytest.approx(2000.0)
    assert _cell(tidy, "2020-03-31", "BBB", "buy_intensity_6m") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 7. Normalization + grid shape
# --------------------------------------------------------------------------- #
def test_compute_features_zscores_per_date_and_keeps_nan() -> None:
    txns = _frame([
        _txn("AAA", "2020-03-03", code="P"),
        _txn("BBB", "2020-03-04", code="S", owner="2"),
        _txn("CCC", "2020-03-05", code="P", owner="3"),
        _txn("CCC", "2020-03-06", code="S", owner="3"),
    ])
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    tidy = compute_insider_features(txns, DATES_2020, symbols)
    assert list(tidy.columns) == ["ticker", "date", *INSIDER_FEATURES]
    assert len(tidy) == len(DATES_2020) * len(symbols)        # full grid

    at = tidy[tidy["date"] == pd.Timestamp("2020-03-31")].set_index("ticker")
    vals = at.loc[["AAA", "BBB", "CCC"], "net_buy_ratio_6m"].to_numpy(dtype=float)
    assert np.all(np.isfinite(vals))
    assert vals.mean() == pytest.approx(0.0, abs=1e-12)       # per-date z-score
    assert vals.std(ddof=0) == pytest.approx(1.0, rel=1e-9)
    assert np.isnan(at.loc["DDD", "net_buy_ratio_6m"])        # no activity stays NaN


def test_empty_transactions_yield_all_nan_grid() -> None:
    txns = _frame([_txn("AAA", "2020-01-15")]).iloc[0:0]
    tidy = raw_insider_features(txns, DATES_2020, ["AAA", "BBB"])
    assert len(tidy) == len(DATES_2020) * 2
    assert tidy[INSIDER_FEATURES].isna().all().all()


# --------------------------------------------------------------------------- #
# 8. Accession-level value dedup (multi-owner fan-out)
# --------------------------------------------------------------------------- #
def test_multi_owner_fanout_counts_value_once_but_owners_separately() -> None:
    """ONE filing co-reported by two owners (same accession): its dollar value must
    enter the value features ONCE, while cluster_buying still sees two distinct
    buyers and the count features keep their registered per-owner-row semantics.
    The adversarial review measured 41% of purchase-leg dollar value duplicated
    this way in the real data."""
    fanout = [
        _txn("AAA", "2020-03-03", code="P", shares=100.0, price=10.0,
             owner="1", accession="ACC-1"),
        _txn("AAA", "2020-03-03", code="P", shares=100.0, price=10.0,
             owner="2", accession="ACC-1"),                     # same filing, 2nd owner
        _txn("BBB", "2020-03-04", code="S", owner="3"),
    ]
    grid = raw_insider_features(_frame(fanout), DATES_2020, ["AAA", "BBB"])
    at = grid[grid["date"] == pd.Timestamp("2020-03-31")].set_index("ticker")
    assert at.loc["AAA", "buy_intensity_6m"] == pytest.approx(1000.0)   # once, not 2000
    assert at.loc["AAA", "cluster_buying_3m"] == pytest.approx(2.0)     # both owners
    assert at.loc["AAA", "net_buy_ratio_6m"] == pytest.approx(1.0)


def test_separate_filings_by_distinct_owners_both_count_value() -> None:
    """Two DIFFERENT filings (distinct accessions) with identical economics are two
    real transactions — dedup must NOT collapse them."""
    two = [
        _txn("AAA", "2020-03-03", code="P", shares=100.0, price=10.0,
             owner="1", accession="ACC-A"),
        _txn("AAA", "2020-03-03", code="P", shares=100.0, price=10.0,
             owner="2", accession="ACC-B"),
    ]
    grid = raw_insider_features(_frame(two), DATES_2020, ["AAA"])
    at = grid[grid["date"] == pd.Timestamp("2020-03-31")].set_index("ticker")
    assert at.loc["AAA", "buy_intensity_6m"] == pytest.approx(2000.0)


def test_missing_accession_column_fails_loudly() -> None:
    rows = [_txn("AAA", "2020-03-03")]
    frame = _frame(rows).drop(columns=["accession"])
    with pytest.raises(ValueError, match="accession"):
        raw_insider_features(frame, DATES_2020, ["AAA"])
