"""Tests for the long-history multi-asset panel primitives.

What is actually at risk here is the YIELD→RETURN conversion. Every other series in
the panel is a price and its return is a ratio; ``^TNX``/``^TYX``/``^FVX``/``^IRX``
are yields, and a wrong conversion is not noisy — it is SIGN-INVERTED, which would
turn a bond bear market into a bond bull market and no downstream Sharpe would look
odd. So ``par_bond_total_return`` is pinned against closed-form identities that hold
independently of the implementation:

* unchanged yield ⇒ return is exactly the accrued coupon;
* a yield RISE ⇒ a NEGATIVE return, monotonically bigger for longer maturity;
* the capital leg ≈ −modified duration × Δy for small Δy;
* a Friday→Monday bar accrues three days of carry, not one.

The rest guards the accounting the programme has already paid for: chronological
sorting, duplicate handling, undefined ratios on non-positive prices (WTI 2020),
long-gap nulling, and the month-end panel refusing to present a stub as a month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.instruments import (
    INSTRUMENTS,
    QUARANTINE,
    VALIDATION_PAIRS,
    by_key,
    panel_instruments,
    tickers,
)
from research.multiasset.panel import (
    apply_quarantine,
    bill_cash_return,
    clean_levels,
    coverage_row,
    day_of_month_signature,
    dsr_sharpe_bar,
    flag_extreme_returns,
    gap_report,
    monthly_last,
    monthly_returns,
    par_bond_total_return,
    simple_returns,
    wide_panel,
)


def _yield_series(values: list[float], dates: list[str]) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))


# ── par_bond_total_return: the conversion that would be silently sign-wrong ────

def test_unchanged_yield_returns_exactly_the_accrued_coupon():
    """Flat yield ⇒ the bond earns carry and nothing else."""
    s = _yield_series([5.0, 5.0], ["2020-01-02", "2020-01-03"])
    ret = par_bond_total_return(s, 10.0)
    # One calendar day at 5% semi-annual-compounded: (1.025)^(2/365) - 1.
    expected = (1.025) ** (2.0 / 365.0) - 1.0
    assert ret.iloc[0] != ret.iloc[0]          # first bar is NaN
    assert ret.iloc[1] == pytest.approx(expected, rel=1e-12)


def test_weekend_bar_accrues_three_days_of_carry_not_one():
    """ACT/365 on the actual gap — a 1/252 day count would under-accrue by 2 days."""
    fri_mon = par_bond_total_return(
        _yield_series([4.0, 4.0], ["2021-01-08", "2021-01-11"]), 10.0).iloc[1]
    one_day = par_bond_total_return(
        _yield_series([4.0, 4.0], ["2021-01-11", "2021-01-12"]), 10.0).iloc[1]
    # Exactly three days of compounded carry at 4% semi-annual.
    assert fri_mon == pytest.approx((1.02) ** (6.0 / 365.0) - 1.0, rel=1e-12)
    # Just over 3x the one-day bar — exceeding exactly 3x only by compounding (~1e-4 rel).
    assert 3.0 * one_day < fri_mon < 3.001 * one_day


def test_yield_rise_gives_a_negative_return_and_longer_maturity_loses_more():
    """The capital leg must move OPPOSITE the yield. pct_change would get this backwards."""
    s = _yield_series([4.00, 4.20], ["2021-06-01", "2021-06-02"])
    r5 = par_bond_total_return(s, 5.0).iloc[1]
    r10 = par_bond_total_return(s, 10.0).iloc[1]
    r30 = par_bond_total_return(s, 30.0).iloc[1]
    assert r30 < r10 < r5 < 0.0
    # A naive pct_change on the yield would report +5%: right magnitude class, wrong sign.
    assert s.pct_change().iloc[1] > 0.0


def test_capital_leg_matches_modified_duration_for_a_small_yield_move():
    """Independent check: −D_mod·Δy, with D_mod from the textbook par-bond formula."""
    y0, dy = 0.05, 0.0001
    s = _yield_series([100 * y0, 100 * (y0 + dy)], ["2021-03-01", "2021-03-02"])
    ret = par_bond_total_return(s, 10.0).iloc[1]
    carry = (1.0 + y0 / 2.0) ** (2.0 / 365.0) - 1.0

    n, m = 20, 2.0                                   # 10y, semi-annual
    macaulay_periods = (1.0 + y0 / m) / (y0 / m) * (
        1.0 - 1.0 / (1.0 + y0 / m) ** n
    )                                                # par-bond Macaulay duration, in periods
    d_mod = (macaulay_periods / m) / (1.0 + y0 / m)
    assert (ret - carry) == pytest.approx(-d_mod * dy, rel=2e-3)


def test_thirty_year_par_bond_duration_is_in_the_right_ballpark():
    """A 30y par bond at 5% has modified duration ~15.4y; a 1bp rise loses ~15bp."""
    s = _yield_series([5.00, 5.01], ["2021-03-01", "2021-03-02"])
    capital = par_bond_total_return(s, 30.0).iloc[1] - ((1.025) ** (2.0 / 365.0) - 1.0)
    assert -0.0017 < capital < -0.0014


def test_zero_yield_is_handled_and_gives_zero_carry():
    """v == 1 makes the annuity factor 0/0; the limit is n, and a 0% par bond earns 0."""
    s = _yield_series([0.0, 0.0], ["2021-01-04", "2021-01-05"])
    ret = par_bond_total_return(s, 5.0)
    assert np.isfinite(ret.iloc[1])
    assert ret.iloc[1] == pytest.approx(0.0, abs=1e-15)


def test_long_gap_and_first_bar_are_nulled_not_fabricated():
    s = _yield_series([3.0, 3.0, 3.0], ["2020-01-02", "2020-01-03", "2020-04-01"])
    ret = par_bond_total_return(s, 10.0)
    assert ret.isna().tolist() == [True, False, True]


def test_par_bond_rejects_nonsense_maturities():
    s = _yield_series([3.0, 3.0], ["2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError):
        par_bond_total_return(s, 0.0)
    with pytest.raises(ValueError):
        par_bond_total_return(s, 10.0, coupons_per_year=0)


# ── bill_cash_return ──────────────────────────────────────────────────────────

def test_bill_discount_is_converted_to_bond_equivalent_yield_before_accruing():
    """BEY = 365d/(360 − 91d) exceeds the discount rate; accruing d raw understates cash."""
    s = _yield_series([5.0, 5.0], ["2020-01-02", "2020-01-03"])
    ret = bill_cash_return(s).iloc[1]
    d = 0.05
    bey = 365.0 * d / (360.0 - d * 91.0)
    assert ret == pytest.approx(bey / 365.0, rel=1e-12)
    assert bey > d                                   # the conversion matters, ~+9bps here


def test_bill_accrual_uses_the_previous_days_rate():
    """You earn the rate you bought at — using today's rate would be a lookahead."""
    s = _yield_series([1.0, 9.0], ["2020-01-02", "2020-01-03"])
    ret = bill_cash_return(s).iloc[1]
    d = 0.01
    assert ret == pytest.approx((365.0 * d / (360.0 - d * 91.0)) / 365.0, rel=1e-12)


def test_bill_accrual_is_never_negative_for_a_positive_rate_and_scales_with_the_gap():
    s = _yield_series([2.0, 2.0, 2.0], ["2021-01-08", "2021-01-11", "2021-01-12"])
    ret = bill_cash_return(s)
    assert ret.iloc[1] == pytest.approx(3.0 * ret.iloc[2], rel=1e-12)
    assert ret.iloc[2] > 0.0


# ── clean_levels ──────────────────────────────────────────────────────────────

def test_clean_levels_sorts_dedupes_and_reports_what_it_did():
    raw = pd.Series(
        [3.0, 1.0, 2.0, 2.5, np.nan],
        index=pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02",
                                "2020-01-02", "2020-01-06"]),
    )
    clean, stats = clean_levels(raw)
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert clean.tolist() == [1.0, 2.5, 3.0]         # duplicate keeps the LAST print
    assert stats["n_duplicate_dates_dropped"] == 1
    assert stats["n_nonfinite_dropped"] == 1
    assert stats["was_already_sorted"] == 0
    assert stats["n_clean"] == 3


def test_clean_levels_makes_a_tz_aware_index_naive_and_midnight_normalised():
    raw = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(
        ["2020-01-01 21:30", "2020-01-02 21:30"]).tz_localize("UTC"))
    clean, _ = clean_levels(raw)
    assert clean.index.tz is None
    assert (clean.index == clean.index.normalize()).all()


# ── simple_returns ────────────────────────────────────────────────────────────

def test_negative_price_makes_the_return_undefined_on_both_affected_bars():
    """The WTI April-2020 case. A ratio through a negative price is not a return."""
    s = pd.Series([18.27, -37.63, 10.01],
                  index=pd.DatetimeIndex(["2020-04-17", "2020-04-20", "2020-04-21"]))
    ret, stats = simple_returns(s)
    assert ret.isna().all()                          # bar 1 and bar 2 both undefined
    assert stats["n_nonpositive_level_bars"] == 1
    assert stats["n_returns_nulled_nonpositive"] == 2
    # Left alone, pandas reports a -306% "return" and then compounds it.
    assert s.pct_change().iloc[1] < -3.0


def test_inverted_quote_flips_the_sign_of_the_position_return():
    """JPY=X falls ⇒ a long-JPY position GAINS."""
    s = pd.Series([100.0, 90.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    plain, _ = simple_returns(s)
    inverted, _ = simple_returns(s, invert=True)
    assert plain.iloc[1] == pytest.approx(-0.10)
    assert inverted.iloc[1] == pytest.approx(100.0 / 90.0 - 1.0)
    assert inverted.iloc[1] > 0.0


def test_returns_spanning_a_long_gap_are_nulled_and_counted():
    s = pd.Series([100.0, 101.0, 150.0],
                  index=pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-03-02"]))
    ret, stats = simple_returns(s, max_gap_days=15)
    assert ret.iloc[1] == pytest.approx(0.01)
    assert np.isnan(ret.iloc[2])
    assert stats["n_returns_nulled_long_gap"] == 1
    kept, _ = simple_returns(s, max_gap_days=None)
    assert kept.iloc[2] == pytest.approx(150.0 / 101.0 - 1.0)


# ── panels ────────────────────────────────────────────────────────────────────

def test_monthly_returns_compound_daily_and_refuse_to_present_a_stub_as_a_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-05")
    daily = pd.DataFrame({"A": 0.001, "B": 0.002}, index=idx)
    monthly = monthly_returns(daily, min_obs=5)

    jan = daily.loc["2020-01"]
    assert monthly.loc["2020-01-31", "A"] == pytest.approx((1.001 ** len(jan)) - 1.0)
    # March has 4 business days of data ⇒ below min_obs AND a trailing partial month.
    assert pd.Timestamp("2020-03-31") not in monthly.index
    assert monthly.index.tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]


def test_a_month_below_min_obs_is_nan_for_that_instrument_only():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"]                      # 2 obs ⇒ stub
                           + list(pd.bdate_range("2020-02-03", "2020-02-10"))  # 6 obs ⇒ kept
                           + list(pd.bdate_range("2020-03-02", "2020-03-31")))
    daily = pd.DataFrame({"A": 0.01}, index=idx)
    monthly = monthly_returns(daily, min_obs=5)
    assert np.isnan(monthly.loc["2020-01-31", "A"])
    assert monthly.loc["2020-02-29", "A"] == pytest.approx(1.01 ** 6 - 1.0)


def test_monthly_last_takes_the_final_observation_of_each_calendar_month():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-31", "2020-02-27"])
    levels = pd.DataFrame({"Y": [1.0, 2.0, 3.0]}, index=idx)
    out = monthly_last(levels)
    assert out["Y"].tolist() == [2.0, 3.0]
    assert out.index.tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]


def test_wide_panel_aligns_disjoint_calendars_on_a_sorted_union_index():
    a = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-06"]))
    b = pd.Series([3.0, 4.0], index=pd.DatetimeIndex(["2020-01-03", "2020-01-06"]))
    panel = wide_panel({"A": a, "B": b})
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert len(panel) == 3
    assert np.isnan(panel.loc["2020-01-02", "B"])


# ── integrity reporting ───────────────────────────────────────────────────────

def test_flag_extreme_returns_finds_every_breach_and_ranks_by_magnitude():
    idx = pd.bdate_range("2020-01-01", periods=6)
    returns = pd.DataFrame({
        "A": [np.nan, 0.01, 0.60, -0.02, 0.00, 0.01],
        "B": [np.nan, 0.01, 0.02, -0.99, 0.00, 0.01],
    }, index=idx)
    hits = flag_extreme_returns(returns, threshold=0.50)
    assert len(hits) == 2
    assert hits.iloc[0]["key"] == "B"                # ranked by |ret|
    assert hits.iloc[0]["ret"] == pytest.approx(-0.99)
    assert set(hits["key"]) == {"A", "B"}


def test_flag_extreme_returns_is_empty_when_nothing_breaches():
    idx = pd.bdate_range("2020-01-01", periods=4)
    quiet = pd.DataFrame({"A": [np.nan, 0.01, -0.02, 0.03]}, index=idx)
    assert flag_extreme_returns(quiet, threshold=0.50).empty


def test_gap_report_counts_the_holes_and_measures_business_day_coverage():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-02-20", "2020-02-21"])
    rep = gap_report(idx)
    assert rep["max_gap_days"] == 48.0
    assert rep["n_gaps_gt_5d"] == 1 and rep["n_gaps_gt_30d"] == 1
    assert 0.0 < rep["bday_coverage_pct"] < 20.0

    dense = pd.bdate_range("2020-01-01", "2020-01-31")
    assert gap_report(dense)["bday_coverage_pct"] == pytest.approx(100.0)


def test_coverage_row_reports_span_and_missing_percent():
    idx = pd.bdate_range("2010-01-04", "2019-12-31")
    row = coverage_row("X", pd.Series(1.0, index=idx))
    assert row["first_date"] == "2010-01-04" and row["last_date"] == "2019-12-31"
    assert row["n_obs"] == len(idx)
    assert row["years"] == pytest.approx(9.99, abs=0.02)
    assert row["pct_missing_vs_bdays"] == pytest.approx(0.0, abs=0.01)


# ── quarantine ────────────────────────────────────────────────────────────────

def test_quarantine_drops_the_close_and_leaves_one_valid_two_day_return():
    """The point of dropping the LEVEL: the true move across the bad print survives."""
    idx = pd.DatetimeIndex(["2008-12-05", "2008-12-08", "2008-12-09", "2008-12-10"])
    levels = pd.Series([1.00, 1.17, 1.0135, 1.02], index=idx)     # 12-08 is corrupt
    before, _ = simple_returns(levels)
    assert before.iloc[1] == pytest.approx(0.17, abs=1e-9)        # fabricated spike
    assert before.iloc[2] < -0.13                                 # fabricated reversal

    cleaned, audit = apply_quarantine({"EURUSD": levels},
                                      (("EURUSD", "2008-12-08", "corrupt close"),))
    after, _ = simple_returns(cleaned["EURUSD"])
    assert len(cleaned["EURUSD"]) == 3
    assert audit == [{"key": "EURUSD", "date": "2008-12-08",
                      "reason": "corrupt close", "matched": True}]
    # 12-05 -> 12-09 is now ONE bar carrying the genuine +1.35%, not a spike and a crash.
    assert after.iloc[1] == pytest.approx(0.0135, abs=1e-9)
    assert after.abs().max() < 0.02


def test_quarantine_reports_entries_that_matched_nothing():
    """A stale exclusion that silently stops matching is an invisible one."""
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"]))
    cleaned, audit = apply_quarantine({"X": s}, (("X", "1999-01-01", "not present"),
                                                 ("MISSING_KEY", "2020-01-02", "no series")))
    assert len(cleaned["X"]) == 2
    assert [a["matched"] for a in audit] == [False, False]


def test_quarantine_does_not_mutate_the_caller_series():
    s = pd.Series([1.0, 2.0, 3.0],
                  index=pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"]))
    original = {"X": s}
    apply_quarantine(original, (("X", "2020-01-03", "drop"),))
    assert len(original["X"]) == 3


def test_the_shipped_quarantine_list_is_well_formed_and_narrow():
    """Every entry names a real key and a parseable date, and the list stays small."""
    known = {i.key for i in INSTRUMENTS}
    for key, date_str, reason in QUARANTINE:
        assert key in known
        assert pd.Timestamp(date_str).year == 2008          # the evidenced defect window
        assert reason
    assert len(QUARANTINE) <= 12, "a growing quarantine list is data mining, not cleaning"


def test_day_of_month_signature_detects_a_planted_vendor_defect():
    """The test that identified the 2008 FX corruption, on synthetic data."""
    idx = pd.bdate_range("2010-01-01", periods=1500)
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)
    clean_sig = day_of_month_signature(r)
    r_bad = r.copy()
    r_bad[[d for d in idx if d.day == 8][:6]] = 0.25         # a defect on the 8th
    bad_sig = day_of_month_signature(r_bad)
    assert bad_sig["modal_day"] == 8
    assert bad_sig["n_of_top"] >= 6
    assert bad_sig["lift"] > 3.0
    assert clean_sig["n_of_top"] < bad_sig["n_of_top"]


# ── DSR bar ───────────────────────────────────────────────────────────────────

def test_dsr_bar_reproduces_both_recorded_anchors_exactly():
    """1.488 at 7 years and 0.597 at 40, n_trials=32 — the finding this study rests on."""
    assert dsr_sharpe_bar(7.0, n_trials=32) == pytest.approx(1.488, abs=0.001)
    assert dsr_sharpe_bar(40.0, n_trials=32) == pytest.approx(0.597, abs=0.001)


def test_dsr_bar_falls_with_sample_length_and_rises_with_trials():
    assert dsr_sharpe_bar(98.0) < dsr_sharpe_bar(42.0) < dsr_sharpe_bar(20.0)
    assert dsr_sharpe_bar(20.0, n_trials=64) > dsr_sharpe_bar(20.0, n_trials=32)


def test_dsr_bar_agrees_with_the_repos_own_deflated_sharpe():
    """Cross-check the inversion against research.validation, not just against itself."""
    from research.validation import deflated_sharpe_ratio

    rng = np.random.default_rng(0)
    for years in (7.0, 20.0, 42.0):
        ann = dsr_sharpe_bar(years, n_trials=32)
        per_period = ann / np.sqrt(12.0)
        T = int(round(years * 12))
        z = rng.standard_normal(T)
        r = (z - z.mean()) / z.std(ddof=1) + per_period      # exact per-period Sharpe
        assert deflated_sharpe_ratio(r, n_trials=32) == pytest.approx(0.95, abs=0.01)


def test_dsr_bar_rejects_a_sample_too_short_to_deflate():
    with pytest.raises(ValueError):
        dsr_sharpe_bar(0.1)


# ── registry ──────────────────────────────────────────────────────────────────

def test_registry_keys_and_tickers_are_unique():
    keys = [i.key for i in INSTRUMENTS]
    assert len(keys) == len(set(keys))
    assert len(tickers()) == len(set(tickers()))


def test_every_instrument_declares_a_known_return_method_and_role():
    allowed = {"price_return", "inverse_price_return", "par_bond_total_return",
               "bill_cash_accrual", "none"}
    for inst in INSTRUMENTS:
        assert inst.return_method in allowed, inst.key
        assert inst.role in {"panel", "cash", "validation"}, inst.key
        if inst.return_method == "par_bond_total_return":
            assert inst.maturity_years and inst.maturity_years > 0, inst.key


def test_validation_instruments_are_excluded_from_the_tradable_panel():
    """A validation instrument leaking into the panel would be double counting."""
    panel_keys = {i.key for i in panel_instruments()}
    assert {"BIL", "IEI", "SLV"}.isdisjoint(panel_keys)
    assert by_key("US_CASH_13W").role == "cash"
    assert by_key("US_CASH_13W").key not in panel_keys


def test_every_validation_pair_references_real_registry_keys():
    known = {i.key for i in INSTRUMENTS}
    for constructed, benchmark, _ in VALIDATION_PAIRS:
        assert constructed in known and benchmark in known


def test_the_yield_series_are_never_treated_as_prices():
    """^TNX/^TYX/^FVX/^IRX must not be on the plain price-return path."""
    for ticker in ("^TNX", "^TYX", "^FVX", "^IRX"):
        inst = next(i for i in INSTRUMENTS if i.ticker == ticker)
        assert inst.return_method in {"par_bond_total_return", "bill_cash_accrual"}, ticker
