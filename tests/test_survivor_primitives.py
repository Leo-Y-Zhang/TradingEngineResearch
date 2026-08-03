"""The survivor-verification primitives, and one cross-module identity.

``_survivor/survivor_verification.py`` produced iteration 22's constant-leverage
ladder, so its arithmetic sits under published ceiling numbers, and it had no direct
test. Expected values are worked out from the docstrings by hand.

The identity test is the one worth reading: ``_convexity/convexity.py`` states in its
module docstring that its leverage convention is "the identical convention as
research/sleeves/_survivor/survivor_verification.py::levered_total". That claim is
load-bearing -- the convexity study exists to re-derive the survivor ladder's ceiling
-- and it is asserted here rather than trusted, because two copies of one formula
drift.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves._convexity.convexity import levered_total as convexity_levered
from research.sleeves._survivor.survivor_verification import (
    MPY,
    cagr,
    circular_blocks,
    inv_vol_weights,
    is_ruined,
    levered_total,
    max_dd,
    sharpe,
)


# ── The leverage convention ───────────────────────────────────────────────────

def test_levered_total_charges_the_spread_only_on_the_borrowed_units():
    """L units funded by 1 of equity and (L-1) borrowed at bill + spread.

    At L=2 with a 1.5% spread: 2 x 0.01 excess, minus one borrowed unit at
    0.015/12, plus the bill of 0.002.
    """
    excess = np.array([0.01])
    cash = np.array([0.002])

    out = levered_total(excess, cash, 2.0, 0.015)

    assert out == pytest.approx(0.02 - 0.015 / 12.0 + 0.002)


def test_no_spread_is_charged_at_or_below_unit_leverage():
    excess, cash = np.array([0.01]), np.array([0.002])

    assert levered_total(excess, cash, 1.0, 0.03) == pytest.approx(0.01 + 0.002)
    assert levered_total(excess, cash, 0.5, 0.03) == pytest.approx(0.005 + 0.002)


def test_the_bill_is_added_because_the_input_is_an_excess_return():
    excess, cash = np.array([0.0]), np.array([0.004])
    assert levered_total(excess, cash, 1.0, 0.0) == pytest.approx(0.004)


def test_convexity_uses_the_identical_leverage_convention_it_claims_to():
    """Pins the cross-module claim made in convexity.py's module docstring.

    Two independent copies of the same formula are exactly the thing that drifts
    apart under maintenance, and a divergence here would mean the convexity study
    was re-deriving the survivor ceiling on a different funding assumption without
    anyone noticing.
    """
    rng = np.random.default_rng(20260728)
    excess = rng.normal(0.004, 0.03, size=240)
    cash = np.full(240, 0.0025)

    for lev in (0.25, 1.0, 1.5, 2.0, 3.75, 8.0):
        for spread in (0.0050, 0.0150, 0.0300):
            assert levered_total(excess, cash, lev, spread) == pytest.approx(
                convexity_levered(excess, cash, lev, spread)
            ), f"leverage conventions diverged at lev={lev}, spread={spread}"


# ── Ruin ──────────────────────────────────────────────────────────────────────

def test_ruin_is_a_loss_of_the_whole_unit_or_worse():
    assert is_ruined(np.array([0.1, -1.0])) is True
    assert is_ruined(np.array([0.1, -1.5])) is True
    assert is_ruined(np.array([0.1, -0.999])) is False


def test_a_ruined_path_reports_minus_one_rather_than_compounding_through_zero():
    ruined = np.array([0.05, -1.0, 10.0])

    assert cagr(ruined) == -1.0
    assert max_dd(ruined) == -1.0


def test_cagr_compounds_and_annualises_at_twelve_periods_a_year():
    r = np.array([0.10, -0.20, 0.05])          # 1.10 x 0.80 x 1.05 = 0.924

    assert cagr(r) == pytest.approx(0.924 ** (MPY / 3) - 1.0)


def test_max_dd_is_negative_here_unlike_the_tsmom_module():
    """1.10 -> 0.88 against a running peak of 1.10 is -20%.

    Third data point on the sign question: this module and ``riskparity`` report
    drawdown as NEGATIVE, ``tsmom_multitimeframe`` reports it POSITIVE. All three are
    pinned so a comparison across modules cannot silently flip.
    """
    dd = max_dd(np.array([0.10, -0.20, 0.05]))

    assert dd == pytest.approx(-0.20)
    assert dd < 0.0


# ── Weights ───────────────────────────────────────────────────────────────────

def test_inverse_vol_weights_are_proportional_to_one_over_the_standard_deviation():
    idx = pd.date_range("2000-01-31", periods=9, freq="ME")
    # column B is built with exactly twice A's dispersion
    a = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]) * 0.01
    frame = pd.DataFrame({"A": a, "B": a * 2.0}, index=idx)

    w = inv_vol_weights(frame)

    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(2.0 / 3.0)
    assert w[1] == pytest.approx(1.0 / 3.0)
    assert w[0] / w[1] == pytest.approx(2.0)


# ── Sharpe threshold ──────────────────────────────────────────────────────────

def test_sharpe_refuses_a_sample_shorter_than_eight_observations():
    """A deliberate floor: too short a sample has no usable standard error."""
    rng = np.random.default_rng(1)
    short = pd.Series(rng.normal(0.01, 0.02, size=7))
    long = pd.Series(rng.normal(0.01, 0.02, size=8))

    assert math.isnan(sharpe(short))
    assert not math.isnan(sharpe(long))


def test_the_zero_dispersion_guard_does_not_fire_on_a_constant_series():
    """The SECOND module found with this, so it is a pattern rather than a one-off.

    The guard is ``a.std(ddof=1) == 0``, an exact comparison, and pandas' ddof=1
    standard deviation of a constant series is floating-point noise near 1e-18 rather
    than exactly zero. The guard therefore misses and the Sharpe comes back around
    2e16. ``tsmom_multitimeframe.annualised`` has the identical shape of bug with
    ``vol > 0.0``.

    Harmless in practice -- it needs a perfectly constant non-zero series, which no
    real return stream is -- and left unchanged for the same reason as in tsmom:
    altering a published code path for no measured benefit. Recorded here so that if
    a degenerate series ever does reach one of these, the number is recognised for
    what it is instead of believed.
    """
    result = sharpe(pd.Series([0.01] * 12))

    assert not math.isnan(result)
    assert result > 1e10


def test_sharpe_annualises_by_the_square_root_of_twelve():
    x = pd.Series([0.02, -0.01] * 6)
    a = x.dropna()
    expected = a.mean() / a.std(ddof=1) * math.sqrt(MPY)

    assert sharpe(x) == pytest.approx(expected)


# ── Circular block bootstrap ──────────────────────────────────────────────────

def test_circular_blocks_returns_one_row_per_replicate_of_the_original_length():
    idx = circular_blocks(n=10, block=3, rng=np.random.default_rng(0), reps=5)

    assert idx.shape == (5, 10)
    assert idx.min() >= 0
    assert idx.max() <= 9


def test_blocks_are_contiguous_and_wrap_around_the_end_of_the_sample():
    """Contiguity is the whole point: it is what preserves autocorrelation."""
    n, block = 10, 3
    idx = circular_blocks(n=n, block=block, rng=np.random.default_rng(7), reps=20)

    for row in idx:
        # the trailing partial block is truncated by [:, :n], so check whole blocks
        for start in range(0, (len(row) // block) * block, block):
            chunk = row[start:start + block]
            for offset in range(1, block):
                assert chunk[offset] == (chunk[0] + offset) % n


def test_the_same_seed_reproduces_the_same_blocks():
    a = circular_blocks(12, 4, np.random.default_rng(20260728), 3)
    b = circular_blocks(12, 4, np.random.default_rng(20260728), 3)
    assert np.array_equal(a, b)
