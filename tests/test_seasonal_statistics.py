"""The seasonal sleeve's reporting statistics.

The test that matters here is the vol-matched t-stat identity. ``active_report``'s
docstring says statistic C -- the benchmark levered to the strategy's own volatility --
is the one that decides, precisely BECAUSE its t-stat is identical to the equivalent
form that scales the strategy down instead, unlike the raw arithmetic active t-stat
which the carry study measured to be a pure leverage dial.

That is a mathematical claim, and the whole choice of deciding statistic rests on it,
so it is asserted rather than trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves.multiasset_seasonal import (
    MONTHS,
    NW_LAG,
    active_report,
    era_split,
    geometric_annual,
    to_monthly,
)
from research.sleeves.multiasset_trend import newey_west_tstat


def _monthly(n: int, seed: int = 0, mu: float = 0.006, sd: float = 0.03) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(mu, sd, size=n), index=idx)


# ── to_monthly ────────────────────────────────────────────────────────────────

def test_daily_returns_compound_exactly_within_a_calendar_month():
    """+10% then -10% in the same month is -1%, not 0%."""
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07"])
    out = to_monthly(pd.Series([0.10, -0.10], index=idx))

    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(1.10 * 0.90 - 1.0)


def test_each_calendar_month_becomes_one_observation_stamped_at_month_end():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-31", "2020-02-03"])
    out = to_monthly(pd.Series([0.01, 0.02, 0.03], index=idx))

    assert list(out.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    assert out.iloc[0] == pytest.approx(1.01 * 1.02 - 1.0)
    assert out.iloc[1] == pytest.approx(0.03)
    assert out.index.name == "date"


def test_to_monthly_on_an_empty_series_returns_empty_rather_than_raising():
    assert to_monthly(pd.Series(dtype=float)).empty


# ── geometric_annual ──────────────────────────────────────────────────────────

def test_geometric_annual_compounds_a_constant_monthly_return():
    r = pd.Series([0.01] * 24)
    assert geometric_annual(r) == pytest.approx(1.01 ** MONTHS - 1.0)


def test_geometric_annual_is_below_the_arithmetic_mean_when_returns_vary():
    """Volatility drag: the geometric mean of a varying series is the smaller one."""
    r = pd.Series([0.10, -0.08] * 12)
    assert geometric_annual(r) < float(r.mean()) * MONTHS


def test_geometric_annual_of_an_empty_series_is_nan():
    assert math.isnan(geometric_annual(pd.Series(dtype=float)))


# ── active_report: the identity the deciding statistic rests on ───────────────

def test_the_volmatched_tstat_equals_the_form_that_scales_the_strategy_down():
    """The documented reason statistic C decides.

    C levers the BENCHMARK to the strategy's vol:      a - b*k,  k = sd_s/sd_b
    The equivalent form scales the STRATEGY down:      a/k - b
    and the two differ by the positive factor k, which a t-stat is invariant to.

    If this ever stopped holding, the sleeve's deciding statistic would silently
    become direction-dependent -- exactly the defect the carry study found in the
    raw arithmetic active t-stat.
    """
    strat, bench = _monthly(120, seed=1), _monthly(120, seed=2, mu=0.004)

    rep = active_report(strat, bench)

    k = rep["bench_leverage_applied"]
    scaled_down = strat / k - bench
    assert rep["volmatched_active_tstat"] == pytest.approx(
        newey_west_tstat(scaled_down, NW_LAG)
    )


def test_the_benchmark_leverage_is_the_ratio_of_the_two_volatilities():
    strat, bench = _monthly(120, seed=3, sd=0.04), _monthly(120, seed=4, sd=0.02)

    rep = active_report(strat, bench)

    a, b = strat.align(bench, join="inner")
    assert rep["bench_leverage_applied"] == pytest.approx(a.std(ddof=1) / b.std(ddof=1))


def test_variance_drag_is_the_annualised_half_difference_of_variances():
    strat, bench = _monthly(120, seed=5, sd=0.05), _monthly(120, seed=6, sd=0.02)

    rep = active_report(strat, bench)

    a, b = strat.align(bench, join="inner")
    expected = (a.var(ddof=1) - b.var(ddof=1)) / 2.0 * MONTHS
    assert rep["variance_drag_annual"] == pytest.approx(expected)
    assert rep["variance_drag_annual"] > 0.0      # the more volatile book drags more


def test_a_strategy_identical_to_its_benchmark_has_unit_beta_and_no_alpha():
    x = _monthly(120, seed=7)

    rep = active_report(x, x)

    assert rep["jensen_beta"] == pytest.approx(1.0)
    assert rep["jensen_alpha_annual"] == pytest.approx(0.0, abs=1e-12)
    assert rep["arith_active_annual"] == pytest.approx(0.0, abs=1e-12)
    assert rep["bench_leverage_applied"] == pytest.approx(1.0)


def test_a_sample_under_twelve_months_reports_only_its_length():
    """Too short to say anything, so it declines to rather than reporting noise."""
    strat, bench = _monthly(11, seed=8), _monthly(11, seed=9)

    rep = active_report(strat, bench)

    assert rep == {"months": 11}


def test_active_report_uses_only_the_overlapping_months():
    idx_a = pd.date_range("2000-01-31", periods=36, freq="ME")
    idx_b = idx_a[12:]                      # benchmark starts a year later
    strat = pd.Series(np.linspace(0.01, 0.02, 36), index=idx_a)
    bench = pd.Series(np.linspace(0.01, 0.02, 24), index=idx_b)

    assert active_report(strat, bench)["months"] == 24


# ── era_split ─────────────────────────────────────────────────────────────────

def test_the_split_year_itself_counts_as_post_publication():
    idx = pd.date_range("2008-01-31", periods=48, freq="ME")   # 2008-2011
    x = pd.Series(0.01, index=idx)

    out = era_split(x, 2010)

    assert out["split_year"] == 2010
    assert out["pre_months"] == 24          # 2008 and 2009
    assert out["post_months"] == 24         # 2010 and 2011


def test_survival_requires_keeping_at_least_half_the_pre_publication_sharpe():
    """The stated rule: post_sharpe >= 0.5 * pre_sharpe."""
    idx = pd.date_range("2005-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(11)
    strong = rng.normal(0.02, 0.02, size=24)      # healthy pre-period
    weak = rng.normal(0.0005, 0.02, size=24)      # decayed post-period
    decayed = era_split(pd.Series(np.concatenate([strong, weak]), index=idx), 2007)

    assert decayed["pre_sharpe"] > decayed["post_sharpe"]
    assert decayed["survived_publication"] is False
    assert decayed["decay_sharpe_points"] == pytest.approx(
        decayed["post_sharpe"] - decayed["pre_sharpe"]
    )

    steady = era_split(pd.Series(np.concatenate([strong, strong]), index=idx), 2007)
    assert steady["survived_publication"] is True


def test_an_era_with_too_few_months_reports_nan_rather_than_a_number():
    idx = pd.date_range("2009-08-31", periods=12, freq="ME")   # only 5 months pre-2010
    out = era_split(pd.Series(0.01, index=idx), 2010)

    assert out["pre_months"] == 5
    assert math.isnan(out["pre_tstat"])
