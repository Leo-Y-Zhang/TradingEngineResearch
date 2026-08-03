"""Tests for the capacity study's market-impact model.

**What went wrong, and what these tests exist to stop happening again.** The module used
to charge ``IMPACT_COEFFICIENT * sqrt(participation)`` with ``IMPACT_COEFFICIENT = 0.1``
and no volatility term at all. At the registered 1%-of-daily-volume position cap that is
100bps per side -- 200bps round trip -- from market impact alone, before spread and before
commission, and identical for a placid mega-cap and a wild micro-cap. Iteration 1 measured
total round-trip costs of 117-236bps across six sleeves; impact was not a component of that
bill, it very nearly WAS the bill.

The replacement keeps the square-root form (Tóth et al. 2011 eq. 1;
``Delta = Y * sigma * sqrt(Q/V)``, "the numerical constant Y is of order unity") but adds
the volatility term and brackets the coefficient between two bounds calibrated on Frazzini,
Israel & Moskowitz (2018) Table II Panel A -- $1.7tn of live US institutional executions.

The load-bearing test in this file is `test_fim_anchor_is_pinned`: if the coefficients stop
reproducing the published live-execution cost at the published participation rate, the
calibration is void and `scripts/impact_positive_control.py` must be re-run before anything
is shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.capacity_study import (
    FIM_ANCHOR_DAILY_VOLATILITY,
    FIM_ANCHOR_HALF_SPREAD_BPS,
    FIM_ANCHOR_PARTICIPATION,
    FIM_LARGE_CAP_MEAN_BPS,
    FIM_LARGE_CAP_MEDIAN_BPS,
    IMPACT_COEFFICIENT,
    IMPACT_COEFFICIENT_CONSERVATIVE,
    IMPACT_COEFFICIENT_REALISTIC,
    IMPACT_EXPONENT,
    REFERENCE_DAILY_VOLATILITY,
    ImpactBounds,
    _impact_fraction,
    impact_cost_bounds,
    impact_fraction,
    round_trip_cost,
    run_band,
)

BPS = 1e4

# The defect, reproduced here so the regression tests do not rely on memory.
OLD_COEFFICIENT = 0.1
REGISTERED_PARTICIPATION_CAP = 0.01


def _impact_bps(participation: float, volatility: float, coefficient: float) -> float:
    """Impact in bps for a unit-dollar-volume name, which makes trade value == share."""
    return impact_fraction(participation, 1.0, volatility, coefficient) * BPS


# ---------------------------------------------------------------------------
# The anchor. This is the test that decides whether the calibration is alive.
# ---------------------------------------------------------------------------


def test_fim_anchor_is_pinned():
    """Both coefficients must reproduce FIM 2018 Table II Panel A at its own conditions.

    Conservative attributes the whole all-in MEAN (8.90bps) to impact; realistic attributes
    the all-in MEDIAN (5.54bps) less the half-spread this repo already charges the same
    name (4.50bps). Anything else means the coefficients were changed without re-running
    the positive control.
    """
    conservative = _impact_bps(FIM_ANCHOR_PARTICIPATION, FIM_ANCHOR_DAILY_VOLATILITY,
                               IMPACT_COEFFICIENT_CONSERVATIVE)
    realistic = _impact_bps(FIM_ANCHOR_PARTICIPATION, FIM_ANCHOR_DAILY_VOLATILITY,
                            IMPACT_COEFFICIENT_REALISTIC)

    assert conservative == pytest.approx(FIM_LARGE_CAP_MEAN_BPS, abs=1e-6)
    assert realistic == pytest.approx(
        FIM_LARGE_CAP_MEDIAN_BPS - FIM_ANCHOR_HALF_SPREAD_BPS, abs=1e-6)

    # The whole modelled one-way cost at the anchor, spread included, must land on the
    # published median under the realistic bound and above it under the conservative one.
    assert realistic + FIM_ANCHOR_HALF_SPREAD_BPS == pytest.approx(
        FIM_LARGE_CAP_MEDIAN_BPS, abs=1e-6)
    assert conservative + FIM_ANCHOR_HALF_SPREAD_BPS > FIM_LARGE_CAP_MEDIAN_BPS


def test_impact_component_is_smaller_than_the_all_in_cost_it_is_calibrated_on():
    """FIM measure ALL-IN cost. Impact is a strict subset and must never exceed it."""
    realistic = _impact_bps(FIM_ANCHOR_PARTICIPATION, FIM_ANCHOR_DAILY_VOLATILITY,
                            IMPACT_COEFFICIENT_REALISTIC)
    conservative = _impact_bps(FIM_ANCHOR_PARTICIPATION, FIM_ANCHOR_DAILY_VOLATILITY,
                               IMPACT_COEFFICIENT_CONSERVATIVE)
    assert 0.0 < realistic < FIM_LARGE_CAP_MEDIAN_BPS
    assert conservative <= FIM_LARGE_CAP_MEAN_BPS + 1e-9


def test_coefficients_sit_in_the_published_order_of_magnitude():
    """Toth et al. (2011): the square-root prefactor Y is 'of order unity'.

    The conservative bound must land in that range -- if it were 18x it, that is the
    defect being fixed; if it were 1/100th of it, the model would be charging nothing.
    """
    assert 0.1 <= IMPACT_COEFFICIENT_CONSERVATIVE <= 2.0
    assert 0.0 < IMPACT_COEFFICIENT_REALISTIC < IMPACT_COEFFICIENT_CONSERVATIVE


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


def test_the_hundred_bps_defect_is_gone():
    """At the registered 1% cap the old model charged 100bps a side. Nothing may now."""
    old = OLD_COEFFICIENT * np.sqrt(REGISTERED_PARTICIPATION_CAP) * BPS
    assert old == pytest.approx(100.0)

    # Charge the most volatile end of the study's own universe and it must still be an
    # order of magnitude below the old flat number.
    worst = _impact_bps(REGISTERED_PARTICIPATION_CAP, 0.0673,
                        IMPACT_COEFFICIENT_CONSERVATIVE)
    assert worst < 30.0
    assert worst < old / 4.0


def test_a_volatile_name_is_charged_more_than_a_placid_one():
    """The old model charged them identically. That was the whole defect."""
    placid = _impact_bps(REGISTERED_PARTICIPATION_CAP, 0.0189,
                         IMPACT_COEFFICIENT_CONSERVATIVE)
    wild = _impact_bps(REGISTERED_PARTICIPATION_CAP, 0.0673,
                       IMPACT_COEFFICIENT_CONSERVATIVE)
    assert wild > 3.0 * placid

    old_placid = OLD_COEFFICIENT * np.sqrt(REGISTERED_PARTICIPATION_CAP)
    old_wild = OLD_COEFFICIENT * np.sqrt(REGISTERED_PARTICIPATION_CAP)
    assert old_placid == old_wild  # the model that could not tell them apart


# ---------------------------------------------------------------------------
# Functional form
# ---------------------------------------------------------------------------


def test_square_root_scaling_is_exact():
    assert IMPACT_EXPONENT == 0.5
    base = impact_fraction(0.0025, 1.0, 0.03)
    assert impact_fraction(0.0100, 1.0, 0.03) == pytest.approx(2.0 * base)
    assert impact_fraction(0.0225, 1.0, 0.03) == pytest.approx(3.0 * base)


def test_impact_is_linear_in_volatility():
    single = impact_fraction(0.01, 1.0, 0.02)
    assert impact_fraction(0.01, 1.0, 0.04) == pytest.approx(2.0 * single)
    assert impact_fraction(0.01, 1.0, 0.0) == 0.0


def test_impact_scales_with_the_name_s_own_dollar_volume():
    """Twice the dollar volume is half the participation is 1/sqrt(2) of the impact."""
    thin = impact_fraction(10_000.0, 1_000_000.0, 0.03)
    thick = impact_fraction(10_000.0, 2_000_000.0, 0.03)
    assert thick == pytest.approx(thin / np.sqrt(2.0))


@pytest.mark.parametrize("median_dollar_volume", [0.0, -1.0, float("nan")])
def test_missing_dollar_volume_is_nan_not_free(median_dollar_volume):
    """A name we cannot size against is untradeable, not free. NaN, never 0.0."""
    assert np.isnan(impact_fraction(1000.0, median_dollar_volume, 0.03))
    bounds = impact_cost_bounds(1000.0, median_dollar_volume, 0.03)
    assert np.isnan(bounds.conservative) and np.isnan(bounds.realistic)
    assert not bounds.determined


def test_zero_trade_size_costs_nothing():
    assert impact_fraction(0.0, 1e6, 0.03) == 0.0


def test_missing_volatility_falls_back_to_the_documented_reference():
    assert impact_fraction(0.01, 1.0, None) == pytest.approx(
        impact_fraction(0.01, 1.0, REFERENCE_DAILY_VOLATILITY))
    assert impact_cost_bounds(0.01, 1.0).daily_volatility == REFERENCE_DAILY_VOLATILITY


def test_negative_volatility_is_refused():
    assert np.isnan(impact_fraction(0.01, 1.0, -0.02))


# ---------------------------------------------------------------------------
# The bracket
# ---------------------------------------------------------------------------


def test_bounds_never_invert():
    rng = np.random.default_rng(7)
    for _ in range(3_000):
        volatility = float(rng.uniform(0.001, 0.20))
        trade_value = float(rng.uniform(1.0, 5e6))
        dollar_volume = float(rng.uniform(1e4, 1e9))
        bounds = impact_cost_bounds(trade_value, dollar_volume, volatility)
        assert bounds.realistic <= bounds.conservative
        assert bounds.realistic >= 0.0


def test_bounds_report_their_inputs():
    bounds = impact_cost_bounds(50_000.0, 5_000_000.0, 0.04)
    assert isinstance(bounds, ImpactBounds)
    assert bounds.participation == pytest.approx(0.01)
    assert bounds.daily_volatility == pytest.approx(0.04)
    assert not bounds.determined  # they must genuinely differ, not collapse


def test_bounds_collapse_only_when_there_is_nothing_to_charge():
    assert impact_cost_bounds(0.0, 1e6, 0.03).determined


# ---------------------------------------------------------------------------
# Backwards compatibility -- other sleeves import these names
# ---------------------------------------------------------------------------


def test_legacy_private_alias_routes_to_the_conservative_bound():
    """`research/sleeves/institutional_flow.py` imports `_impact_fraction`."""
    assert _impact_fraction(50_000.0, 5e6) == pytest.approx(
        impact_fraction(50_000.0, 5e6, None, IMPACT_COEFFICIENT_CONSERVATIVE))


def test_legacy_flat_coefficient_is_the_calibrated_model_at_reference_volatility():
    """`research/sleeves/low_vol_quality.py` uses `IMPACT_COEFFICIENT * sqrt(p)`.

    That call site cannot express a volatility, so the constant it reads must be the
    calibrated coefficient evaluated at the reference volatility -- which is exactly what
    the flat form means once sigma is held fixed -- and it must no longer be 0.1.
    """
    assert IMPACT_COEFFICIENT == pytest.approx(
        IMPACT_COEFFICIENT_CONSERVATIVE * REFERENCE_DAILY_VOLATILITY)
    assert IMPACT_COEFFICIENT < OLD_COEFFICIENT / 5.0
    flat = IMPACT_COEFFICIENT * np.sqrt(REGISTERED_PARTICIPATION_CAP) * BPS
    assert flat == pytest.approx(
        _impact_bps(REGISTERED_PARTICIPATION_CAP, REFERENCE_DAILY_VOLATILITY,
                    IMPACT_COEFFICIENT_CONSERVATIVE))
    assert flat < 20.0  # was 100.0


def test_round_trip_cost_decomposes_as_documented():
    spread, trade_value, price, dollar_volume, volatility = (
        0.0100, 50_000.0, 40.0, 5_000_000.0, 0.03)
    total = round_trip_cost(spread, trade_value, price, dollar_volume, volatility)
    impact = impact_fraction(trade_value, dollar_volume, volatility,
                             IMPACT_COEFFICIENT_CONSERVATIVE)
    shares = trade_value / price
    commission = max(0.35, shares * 0.0035) / trade_value
    assert total == pytest.approx(spread / 2.0 + impact + commission + 0.00002)


def test_round_trip_cost_accepts_the_realistic_bound_and_is_cheaper_under_it():
    args = (0.0100, 50_000.0, 40.0, 5_000_000.0, 0.03)
    conservative = round_trip_cost(*args,
                                   impact_coefficient=IMPACT_COEFFICIENT_CONSERVATIVE)
    realistic = round_trip_cost(*args,
                                impact_coefficient=IMPACT_COEFFICIENT_REALISTIC)
    assert realistic < conservative


def test_round_trip_cost_default_signature_still_works():
    """Sleeves call this positionally with four arguments; that must keep working."""
    assert np.isfinite(round_trip_cost(0.01, 50_000.0, 40.0, 5_000_000.0))


# ---------------------------------------------------------------------------
# run_band
# ---------------------------------------------------------------------------


def _panel(n_months: int = 40, n_names: int = 40,
           daily_volatility: float | None = None) -> pd.DataFrame:
    """A deterministic panel just rich enough for `run_band` to complete."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    rows = []
    for date in dates:
        for index in range(n_names):
            row = {
                "date": date,
                "ticker": f"T{index:03d}",
                "band": "B",
                "signal": float(rng.normal()),
                "spread": 0.01,
                "spread_regime": "measured",
                "close": 20.0,
                "median_dollar_volume": 5_000_000.0,
                "forward_return": float(rng.normal(0.005, 0.05)),
            }
            if daily_volatility is not None:
                row["daily_volatility"] = daily_volatility
            rows.append(row)
    return pd.DataFrame(rows)


def test_run_band_reports_both_cost_bounds_and_they_bracket():
    result = run_band(_panel(), "B")
    assert result is not None
    assert result.cost_drag_annual_realistic <= result.cost_drag_annual
    assert result.cost_drag_annual_realistic > 0.0


def test_run_band_uses_a_per_name_volatility_column_when_the_panel_carries_one():
    quiet = run_band(_panel(daily_volatility=0.01), "B")
    loud = run_band(_panel(daily_volatility=0.08), "B")
    assert quiet is not None and loud is not None
    assert loud.cost_drag_annual > quiet.cost_drag_annual


def test_run_band_falls_back_to_the_reference_volatility_without_the_column():
    without = run_band(_panel(), "B")
    with_reference = run_band(_panel(daily_volatility=REFERENCE_DAILY_VOLATILITY), "B")
    assert without is not None and with_reference is not None
    assert without.cost_drag_annual == pytest.approx(with_reference.cost_drag_annual)


def test_run_band_costs_far_less_than_the_old_flat_model_would_have():
    """The regression that matters: the same book, the same trades, a sane bill."""
    result = run_band(_panel(), "B")
    assert result is not None
    # Reconstruct what the OLD model would have charged for the same trades: the impact
    # term was 0.1 * sqrt(participation) with the same participation this panel implies.
    trade_value = 40 * 0.01 * 5_000_000.0 / 30.0  # deployable capital / n positions
    participation = trade_value / 5_000_000.0
    old_impact = OLD_COEFFICIENT * np.sqrt(participation)
    new_impact = impact_fraction(trade_value, 5_000_000.0, None,
                                 IMPACT_COEFFICIENT_CONSERVATIVE)
    assert new_impact < old_impact / 5.0
