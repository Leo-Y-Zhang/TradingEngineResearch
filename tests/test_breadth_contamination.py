"""The roll-splice contamination detectors, and the monthly reconciliation.

These are the checks that decide whether an instrument's data can be trusted at all --
``roll_contamination`` is the test that condemned NATGAS_F (65.7% of |r|>15% bars in
days 24-31 against a 24.0% base rate, a 2.74x lift). A detector that quietly stopped
detecting would admit spliced price series into the universe and every downstream
result would inherit them, so its arithmetic is pinned here on synthetic series where
the right answer is known by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.multiasset.breadth_build import (
    ROLL_EXTREME_THRESHOLD,
    TRADING_DAYS,
    dom_variance_share,
    max_window_lift,
    monthly_reconciliation,
    roll_contamination,
    validation_stats,
)

WINDOW = (24, 31)


def _daily(days: int = 400, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2015-01-01", periods=days)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, size=days), index=idx)


# ── roll_contamination ────────────────────────────────────────────────────────

def test_a_clean_series_with_no_extreme_bars_reports_no_lift():
    """Nothing over the threshold means there is nothing to attribute to the roll."""
    out = roll_contamination(_daily(), WINDOW)

    assert out["n_extreme"] == 0
    assert math.isnan(out["lift"])
    assert out["base_rate_pct"] > 0.0        # the window itself is still measured


def test_extremes_falling_entirely_inside_the_roll_window_lift_by_one_over_the_base_rate():
    """Constructed so the answer is forced: every extreme sits in the window.

    pct_in_window is then 100%, and the lift must be exactly 1 / base_rate.
    """
    s = _daily(seed=1)
    in_window = pd.DatetimeIndex(s.index).day >= WINDOW[0]
    s[in_window] = 0.30                       # well over the 15% threshold

    out = roll_contamination(s, WINDOW)

    assert out["n_extreme"] == int(in_window.sum())
    assert out["pct_in_window"] == pytest.approx(100.0)
    assert out["lift"] == pytest.approx(round(1.0 / (out["base_rate_pct"] / 100.0), 3))
    assert out["lift"] > 1.0


def test_extremes_spread_evenly_across_the_month_give_a_lift_near_one():
    """The null case: a real price move has no opinion about the calendar."""
    s = _daily(days=1200, seed=2)
    s.iloc[::7] = 0.30                        # every seventh bar, ignoring day of month

    out = roll_contamination(s, WINDOW)

    assert out["lift"] == pytest.approx(1.0, abs=0.25)


def test_the_threshold_decides_what_counts_as_extreme():
    s = _daily(seed=3)
    s.iloc[10] = ROLL_EXTREME_THRESHOLD * 0.9     # just under: not extreme
    s.iloc[20] = ROLL_EXTREME_THRESHOLD * 1.1     # just over: extreme

    assert roll_contamination(s, WINDOW)["n_extreme"] == 1


def test_sign_does_not_matter_only_magnitude():
    up, down = _daily(seed=4), _daily(seed=4)
    up.iloc[30] = 0.30
    down.iloc[30] = -0.30

    assert roll_contamination(up, WINDOW)["n_extreme"] == \
        roll_contamination(down, WINDOW)["n_extreme"]


def test_an_empty_series_reports_nan_rather_than_raising():
    out = roll_contamination(pd.Series(dtype=float), WINDOW)
    assert out["n_extreme"] == 0
    assert math.isnan(out["lift"])


# ── dom_variance_share ────────────────────────────────────────────────────────

def test_variance_share_exceeds_bar_share_when_the_window_holds_the_big_moves():
    """The stronger test: a splice need not clear 15% to dominate the variance."""
    s = _daily(seed=5)
    in_window = pd.DatetimeIndex(s.index).day >= WINDOW[0]
    s[in_window] = s[in_window] * 8.0         # same bars, much larger magnitude

    out = dom_variance_share(s, WINDOW)

    assert out["variance_share_pct"] > out["bar_share_pct"]
    assert out["ratio"] > 1.0
    assert out["mean_abs_in_pct"] > out["mean_abs_out_pct"]


def test_a_homogeneous_series_has_a_variance_share_close_to_its_bar_share():
    out = dom_variance_share(_daily(days=1200, seed=6), WINDOW)
    assert out["ratio"] == pytest.approx(1.0, abs=0.35)


def test_bar_share_is_the_fraction_of_bars_inside_the_window():
    s = _daily(seed=7)
    expected = float((pd.DatetimeIndex(s.index).day >= WINDOW[0]).mean())

    assert dom_variance_share(s, WINDOW)["bar_share_pct"] == pytest.approx(
        round(100.0 * expected, 2)
    )


def test_dom_variance_share_on_an_empty_series_is_all_nan():
    out = dom_variance_share(pd.Series(dtype=float), WINDOW)
    assert all(math.isnan(v) for v in out.values())


# ── max_window_lift ───────────────────────────────────────────────────────────

def test_max_window_lift_finds_the_window_the_extremes_actually_sit_in():
    s = _daily(days=1200, seed=8)
    days = pd.DatetimeIndex(s.index).day
    s[(days >= 10) & (days <= 16)] = 0.30     # a 7-day block, matching the default width

    out = max_window_lift(s)

    assert out["best_window"] == (10, 16)
    assert out["lift"] > 1.0


def test_too_few_extremes_declines_to_name_a_window():
    """Under five extreme bars is not evidence of anything."""
    s = _daily(seed=9)
    s.iloc[5] = 0.30
    s.iloc[50] = 0.30

    out = max_window_lift(s)

    assert out["best_window"] is None
    assert math.isnan(out["lift"])


# ── validation_stats ──────────────────────────────────────────────────────────

def test_validation_stats_declines_on_a_short_overlap():
    a = _daily(days=59, seed=10)
    out = validation_stats(a, a)
    assert out["n_overlap"] == 59
    assert math.isnan(out["corr"])


def test_a_series_against_itself_correlates_perfectly_with_no_drift_gap():
    a = _daily(days=200, seed=11)

    out = validation_stats(a, a)

    assert out["corr"] == pytest.approx(1.0)
    assert out["drift_gap_pct_yr"] == pytest.approx(0.0, abs=1e-9)
    assert out["vol_ratio"] == pytest.approx(1.0)


def test_the_drift_gap_is_the_benchmark_minus_the_construction_annualised():
    a = _daily(days=200, seed=12)
    b = a + 0.0001                            # benchmark drifts one bp a day faster

    out = validation_stats(a, b)

    assert out["drift_gap_pct_yr"] == pytest.approx(
        round(0.0001 * TRADING_DAYS * 100.0, 3)
    )


# ── monthly_reconciliation ────────────────────────────────────────────────────

def test_reconciliation_is_exact_when_the_monthly_panel_really_is_the_compound():
    idx = pd.bdate_range("2015-01-01", periods=200)
    daily = pd.DataFrame({"X": np.linspace(0.001, 0.002, 200)}, index=idx)
    period = idx.to_period("M")
    rows = {}
    for per, chunk in daily["X"].groupby(period):
        rows[per.to_timestamp(how="end").normalize()] = float(np.prod(1.0 + chunk)) - 1.0
    monthly = pd.DataFrame({"X": rows})

    out = monthly_reconciliation(daily, monthly)

    assert out["n_cells_checked"] > 0
    assert out["max_abs_discrepancy"] == pytest.approx(0.0, abs=1e-12)


def test_reconciliation_reports_the_worst_cell_when_a_month_disagrees():
    idx = pd.bdate_range("2015-01-01", periods=200)
    daily = pd.DataFrame({"X": np.linspace(0.001, 0.002, 200)}, index=idx)
    period = idx.to_period("M")
    rows = {}
    for per, chunk in daily["X"].groupby(period):
        rows[per.to_timestamp(how="end").normalize()] = float(np.prod(1.0 + chunk)) - 1.0
    monthly = pd.DataFrame({"X": rows})
    corrupted = monthly.index[1]
    monthly.at[corrupted, "X"] = monthly.at[corrupted, "X"] + 0.01

    out = monthly_reconciliation(daily, monthly)

    assert out["max_abs_discrepancy"] == pytest.approx(0.01, abs=1e-9)
