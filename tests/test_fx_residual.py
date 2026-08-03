"""Offline tests for the FX residual decomposition.

The load-bearing test is `test_injected_depository_margin_is_recovered_exactly`: it
builds a synthetic trust whose margin is known by construction and requires the
decomposition to hand that margin back. If the algebra in `fx_residual.decompose` is
wrong in any term, that test fails — no market data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.fx_residual import (
    CONSTRUCTIONS,
    MONTHS_PER_YEAR,
    SPONSOR_FEE,
    annualise,
    decompose,
    earned_rate,
    evaluate_null_control,
    evaluate_sign_discipline,
    regime_split,
)


def _index(n: int = 240) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-31", periods=n, freq="ME")


# --------------------------------------------------------------------------- earned_rate


def test_published_construction_passes_the_overnight_rate_through() -> None:
    o = pd.Series([-0.005, 0.0, 0.03])
    pd.testing.assert_series_equal(earned_rate(o, construction="published"), o)


def test_zero_floored_construction_removes_negative_deposit_rates() -> None:
    o = pd.Series([-0.005, 0.0, 0.03])
    got = earned_rate(o, construction="zero_floored")
    assert list(got) == [0.0, 0.0, 0.03]


def test_fee_first_construction_floors_the_net_credit_at_zero() -> None:
    o = pd.Series([-0.005, 0.001, 0.03])
    got = earned_rate(o, construction="fee_first", fee=SPONSOR_FEE)
    # net = earned - fee must never be negative under this construction
    assert ((got - SPONSOR_FEE) >= -1e-15).all()
    assert got.iloc[-1] == pytest.approx(0.03)


def test_unregistered_construction_is_refused() -> None:
    with pytest.raises(ValueError, match="unregistered construction"):
        earned_rate(pd.Series([0.01]), construction="whatever_fits")


# ----------------------------------------------------------------------------- decompose


def _synthetic(margin: float, *, i3m_f: float, on_f: float, i3m_us: float,
               bill: float, n: int = 240) -> tuple[pd.Series, dict]:
    """A trust whose depository margin is `margin` by construction.

    Rates are held constant so the one-month contract lag upstream of `diff` cannot
    smuggle a difference into the identity being tested.
    """
    idx = _index(n)
    rng = np.random.default_rng(11)
    spot = pd.Series(rng.normal(0.0, 0.02, n), index=idx)
    cash = pd.Series(bill / MONTHS_PER_YEAR, index=idx)

    earned = max(0.0, on_f)                       # the zero_floored construction
    net = earned - SPONSOR_FEE - margin
    etf_ret = spot + net / MONTHS_PER_YEAR
    fx_excess = spot + (i3m_f - i3m_us) / MONTHS_PER_YEAR

    diff = fx_excess - (etf_ret - cash)
    inputs = {
        "i3m_foreign": pd.Series(i3m_f, index=idx),
        "overnight_foreign": pd.Series(on_f, index=idx),
        "i3m_us": pd.Series(i3m_us, index=idx),
        "cash": cash,
    }
    return diff, inputs


def test_injected_depository_margin_is_recovered_exactly() -> None:
    margin = 0.0027
    diff, inputs = _synthetic(margin, i3m_f=0.035, on_f=0.030, i3m_us=0.040, bill=0.037)
    frame = decompose(diff, construction="zero_floored", **inputs)
    assert annualise(frame["remainder"]) == pytest.approx(margin, abs=1e-12)


@pytest.mark.parametrize("margin", [0.0, 0.001, 0.005])
@pytest.mark.parametrize("on_f", [-0.004, 0.0, 0.05])
def test_margin_recovery_holds_across_rate_regimes(margin: float, on_f: float) -> None:
    """Including negative and zero foreign overnight rates, which is where the refuted
    hypothesis went wrong."""
    diff, inputs = _synthetic(margin, i3m_f=max(on_f, 0.0) + 0.004, on_f=on_f,
                              i3m_us=0.02, bill=0.018)
    frame = decompose(diff, construction="zero_floored", **inputs)
    assert annualise(frame["remainder"]) == pytest.approx(margin, abs=1e-12)


def _synthetic_time_varying(margin: float, n: int = 240) -> tuple[pd.Series, dict, dict]:
    """Rates that MOVE, so the one-month timing convention is load-bearing.

    `fx_excess` credits the differential contracted at ``t-1``; the trust accrues the
    overnight rate prevailing during ``t``. Returns ``diff`` plus both the correctly
    lagged inputs and the naively contemporaneous ones.
    """
    idx = _index(n)
    rng = np.random.default_rng(5)
    spot = pd.Series(rng.normal(0.0, 0.02, n), index=idx)
    # a genuine rate cycle, not noise: 0% to 6% and back
    i3m_f = pd.Series(0.03 + 0.03 * np.sin(np.linspace(0, 4 * np.pi, n)), index=idx)
    on_f = i3m_f - 0.004
    i3m_us = pd.Series(0.035 + 0.025 * np.cos(np.linspace(0, 3 * np.pi, n)), index=idx)
    bill = i3m_us - 0.003
    cash = bill / MONTHS_PER_YEAR

    earned = on_f.clip(lower=0.0)                              # contemporaneous accrual
    etf_ret = spot + (earned - SPONSOR_FEE - margin) / MONTHS_PER_YEAR
    fx_excess = spot + (i3m_f - i3m_us).shift(1) / MONTHS_PER_YEAR   # lagged contract
    diff = fx_excess - (etf_ret - cash)

    lagged = {"i3m_foreign": i3m_f.shift(1), "overnight_foreign": earned,
              "i3m_us": i3m_us.shift(1), "cash": cash}
    contemporaneous = {"i3m_foreign": i3m_f, "overnight_foreign": earned,
                       "i3m_us": i3m_us, "cash": cash}
    return diff, lagged, contemporaneous


def test_margin_is_recovered_only_under_the_runners_lag_convention() -> None:
    margin = 0.0031
    diff, lagged, contemporaneous = _synthetic_time_varying(margin)

    right = decompose(diff, construction="published", **lagged)
    assert annualise(right["remainder"]) == pytest.approx(margin, abs=1e-12)

    # and the test discriminates: getting the lag wrong does not merely add noise
    wrong = decompose(diff, construction="published", **contemporaneous)
    assert abs(annualise(wrong["remainder"]) - margin) > 1e-6


@pytest.mark.parametrize("construction", CONSTRUCTIONS)
def test_identity_holds_for_every_construction(construction: str) -> None:
    diff, inputs = _synthetic(0.002, i3m_f=0.03, on_f=0.025, i3m_us=0.04, bill=0.037)
    frame = decompose(diff, construction=construction, **inputs)
    pd.testing.assert_series_equal(
        frame["predicted"] + frame["remainder"], frame["diff"], check_names=False,
    )


def test_the_us_leg_enters_with_a_minus_sign() -> None:
    """A wider US TED spread must SHRINK the predicted residual, not grow it.

    This pins the sign that the refuted hypothesis never considered.
    """
    diff, inputs = _synthetic(0.002, i3m_f=0.03, on_f=0.025, i3m_us=0.04, bill=0.037)
    base = decompose(diff, construction="zero_floored", **inputs)
    wider = dict(inputs)
    wider["cash"] = inputs["cash"] - 0.01 / MONTHS_PER_YEAR   # bill 100bp below interbank
    widened = decompose(diff, construction="zero_floored", **wider)
    assert annualise(widened["predicted"]) < annualise(base["predicted"])


def test_decompose_drops_months_where_any_input_is_missing() -> None:
    diff, inputs = _synthetic(0.002, i3m_f=0.03, on_f=0.025, i3m_us=0.04, bill=0.037, n=60)
    inputs["overnight_foreign"] = inputs["overnight_foreign"].copy()
    inputs["overnight_foreign"].iloc[:10] = np.nan
    frame = decompose(diff, construction="zero_floored", **inputs)
    assert len(frame) == 50


# -------------------------------------------------------------------------- regime split


def test_regime_split_reports_the_asymmetry_between_rate_regimes() -> None:
    idx = _index(120)
    i3m = pd.Series([0.001] * 60 + [0.04] * 60, index=idx)
    # 0.5%/yr in low-rate months, 1.5%/yr in normal ones -> asymmetry 1.0 pp
    col = pd.Series([0.005 / MONTHS_PER_YEAR] * 60 + [0.015 / MONTHS_PER_YEAR] * 60,
                    index=idx)
    out = regime_split(pd.DataFrame({"i3m_foreign": i3m, "x": col}), "x")
    assert out["n_low_rate_months"] == 60
    assert out["gap_low_pct_yr"] == pytest.approx(0.5, abs=1e-6)
    assert out["gap_high_pct_yr"] == pytest.approx(1.5, abs=1e-6)
    assert out["asymmetry_pct_yr"] == pytest.approx(1.0, abs=1e-6)


def test_regime_split_refuses_to_report_a_bucket_thinner_than_a_year() -> None:
    idx = _index(120)
    i3m = pd.Series([0.001] * 6 + [0.04] * 114, index=idx)
    out = regime_split(pd.DataFrame({"i3m_foreign": i3m, "x": pd.Series(0.001, index=idx)}),
                       "x")
    assert out["gap_low_pct_yr"] is None
    assert out["asymmetry_pct_yr"] is None


# ------------------------------------------------------------------------------ P5 / P6


def test_sign_discipline_fails_when_a_three_month_rate_sits_below_overnight() -> None:
    assert evaluate_sign_discipline({"EZ": 0.002, "GB": 0.001}).passed
    bad = evaluate_sign_discipline({"EZ": 0.002, "JP": -0.0004})
    assert not bad.passed
    assert "JP" in bad.detail["negative_offenders"]


def test_null_control_fails_on_a_reproduction_error_above_a_hundredth_of_a_point() -> None:
    published = {"EURUSD": 0.94, "GBPUSD": 0.86, "JPYUSD": 0.496}
    assert evaluate_null_control(dict(published), published).passed
    drifted = dict(published, EURUSD=0.96)
    assert not evaluate_null_control(drifted, published).passed


def test_null_control_fails_when_nothing_was_reproduced_at_all() -> None:
    assert not evaluate_null_control({}, {"EURUSD": 0.94}).passed
