"""Tests for the third registered FX-residual test.

The load-bearing ones are the null's own properties: a circular shift must not
change the series, only its alignment; an injected effect must move the statistic;
and the decision rule must implement the pre-registered thresholds exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.fx_shape_reality import (
    LOW_RATE_THRESHOLD,
    _circular_shift,
    _stationary_bootstrap_index,
    asymmetry,
    block_bootstrap_null,
    circular_shift_null,
    p_value,
    pooled_statistic,
    regime_mask,
    verdict,
)


def test_circular_shift_is_a_permutation_of_the_same_values():
    v = np.arange(10.0)
    for tau in range(1, 10):
        out = _circular_shift(v, tau)
        assert len(out) == len(v)
        assert sorted(out) == sorted(v)


def test_circular_shift_actually_moves_the_series():
    v = np.arange(10.0)
    assert not np.array_equal(_circular_shift(v, 3), v)
    assert np.array_equal(_circular_shift(v, 10), v)      # full turn is identity


def test_asymmetry_is_annualised_high_minus_low_in_percent():
    low = np.array([True, True, False, False])
    v = np.array([0.0, 0.0, 0.01, 0.01])                  # 1% per month in high months
    assert asymmetry(v, low) == pytest.approx(0.01 * 12 * 100.0)


def test_asymmetry_is_nan_when_one_regime_is_empty():
    v = np.array([0.1, 0.2, 0.3])
    assert np.isnan(asymmetry(v, np.array([True, True, True])))
    assert np.isnan(asymmetry(v, np.array([False, False, False])))


def test_pooled_statistic_takes_absolute_values():
    assert pooled_statistic({"a": -1.0, "b": 3.0}) == pytest.approx(2.0)


def test_pooled_statistic_ignores_nan_legs():
    assert pooled_statistic({"a": float("nan"), "b": 2.0}) == pytest.approx(2.0)


def test_regime_mask_uses_the_committed_three_month_rule():
    frame = pd.DataFrame({"i3m_foreign": [0.001, 0.005, 0.02]})
    assert list(regime_mask(frame)) == [True, True, False]
    assert LOW_RATE_THRESHOLD == 0.005


def test_verdict_implements_the_registered_thresholds():
    assert verdict(0.01, 0.02) == "REAL"
    assert verdict(0.20, 0.30) == "ARTEFACT"
    assert verdict(0.01, 0.30) == "UNDETERMINED"          # nulls disagree
    assert verdict(0.07, 0.07) == "UNDETERMINED"          # between the bands


def test_p_value_uses_the_plus_one_correction_and_is_bounded():
    null = np.zeros(99)
    assert p_value(1.0, null) == pytest.approx(1 / 100)
    assert p_value(-1.0, null) == pytest.approx(100 / 100)


def test_stationary_bootstrap_indices_stay_in_range():
    rng = np.random.default_rng(0)
    idx = _stationary_bootstrap_index(50, 6.0, rng)
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def _legs(n=240, effect=0.0, seed=0, block=40):
    """Synthetic leg with BLOCKY regime labels, as real rate regimes are.

    Rates are persistent, so low-rate months arrive in long runs. Alternating
    labels are pathological here -- see the degeneracy test below.
    """
    rng = np.random.default_rng(seed)
    low = np.array([(i // block) % 2 == 0 for i in range(n)])
    v = rng.normal(0.0, 0.004, size=n)
    v[~low] += effect
    return {"X": (v, low)}


def test_circular_shift_null_is_degenerate_for_periodic_labels():
    """A known limitation, pinned so nobody rediscovers it as a bug.

    Under strictly alternating regime labels a circular shift either preserves the
    alternation or flips it, so |asymmetry| is invariant and the null collapses to a
    single value. Real rate regimes are long runs, not alternations, so this does not
    affect the study -- but a future reuse on periodic labels would be meaningless.
    """
    n = 100
    rng = np.random.default_rng(1)
    alternating = np.array([i % 2 == 0 for i in range(n)])
    legs = {"X": (rng.normal(0.0, 0.004, size=n), alternating)}
    null = circular_shift_null(legs, draws=100, seed=1)
    assert np.allclose(null, null[0]), "expected the degenerate collapse"


def test_nulls_are_deterministic_under_a_fixed_seed():
    legs = _legs()
    a = circular_shift_null(legs, draws=200, seed=7)
    b = circular_shift_null(legs, draws=200, seed=7)
    assert np.array_equal(a, b)
    c = block_bootstrap_null(legs, draws=200, seed=7)
    d = block_bootstrap_null(legs, draws=200, seed=7)
    assert np.array_equal(c, d)


def test_a_large_injected_effect_is_detected_by_the_sharper_null():
    """Sanity that the machinery can see something huge.

    This is deliberately far larger than the 0.5%/yr the real C1 control injects --
    C1 FAILED on the real data, and that failure is the published result. This test
    pins that the shift null is not broken, not that the study was powered.
    """
    legs = _legs(effect=0.02, seed=3)                     # ~24%/yr, unmissable
    obs = pooled_statistic({k: asymmetry(v, low) for k, (v, low) in legs.items()})
    null = circular_shift_null(legs, draws=500, seed=3)
    assert p_value(obs, null) < 0.05


def test_a_pure_noise_series_is_not_flagged():
    legs = _legs(effect=0.0, seed=11)
    obs = pooled_statistic({k: asymmetry(v, low) for k, (v, low) in legs.items()})
    null = circular_shift_null(legs, draws=500, seed=11)
    assert p_value(obs, null) > 0.05
