"""The standard error behind every "is book A better than book B" claim.

``memmel_se`` is the Jobson-Korkie statistic with Memmel's correction, and it decides
whether a Sharpe gap between two books is real. Getting it wrong in the optimistic
direction manufactures significance, so its closed form is checked against properties
that follow from the formula rather than against recorded numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from research._gate_review.sharpe_difference_power import (
    MPY,
    ann_sharpe,
    ann_vol,
    memmel_se,
    unpaired_se,
)


# ── The paired standard error ─────────────────────────────────────────────────

def test_two_identical_perfectly_correlated_books_have_no_uncertainty_in_their_gap():
    """The strongest property the formula has.

    At rho = 1 with equal Sharpes the variance term collapses to exactly zero:
    2(1-1) + 0.5(a^2 + a^2 - 2 a a) = 0. A book compared against itself has a gap of
    zero with no standard error, and if that ever stopped holding the correction term
    would be wrong.
    """
    assert memmel_se(1.2, 1.2, 1.0, 240) == pytest.approx(0.0, abs=1e-12)


def test_correlation_shrinks_the_standard_error_of_the_gap():
    """Two books that share their noise are compared more precisely than two that do
    not, which is the entire reason a PAIRED test is used here."""
    ses = [memmel_se(1.0, 0.8, rho, 240) for rho in (0.0, 0.3, 0.6, 0.9)]

    assert ses == sorted(ses, reverse=True)
    assert ses[-1] < ses[0]


def test_the_unpaired_standard_error_is_the_zero_correlation_case():
    """Stated in the docstring, so it is asserted rather than assumed."""
    for sa, sb, T in ((1.0, 0.5, 120), (0.3, 0.31, 600), (2.0, -0.4, 240)):
        assert unpaired_se(sa, sb, T) == pytest.approx(memmel_se(sa, sb, 0.0, T))


def test_treating_correlated_books_as_independent_overstates_the_error():
    """The unpaired SE is the conservative one, which is the safe direction: it makes
    a gap harder to call significant, never easier."""
    paired = memmel_se(1.0, 0.7, 0.75, 240)
    unpaired = unpaired_se(1.0, 0.7, 240)

    assert unpaired > paired


def test_the_standard_error_falls_with_the_square_root_of_the_sample():
    short = memmel_se(1.0, 0.6, 0.4, 120)
    long = memmel_se(1.0, 0.6, 0.4, 480)

    assert long == pytest.approx(short / 2.0)


def test_the_standard_error_is_never_negative_even_at_extreme_inputs():
    for rho in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for sa, sb in ((3.0, -3.0), (0.0, 0.0), (5.0, 5.0)):
            assert memmel_se(sa, sb, rho, 60) >= 0.0


def test_the_standard_error_is_symmetric_in_the_two_books():
    assert memmel_se(1.4, 0.2, 0.5, 200) == pytest.approx(memmel_se(0.2, 1.4, 0.5, 200))


def test_it_matches_the_formula_worked_through_by_hand():
    """One fully independent evaluation of the documented expression.

    Var = (1/T)[2(1-rho) + 0.5(a^2 + b^2 - 2 rho^2 a b)] in PER-PERIOD Sharpes,
    annualised by multiplying the variance by MPY.
    """
    sa, sb, rho, T = 1.0, 0.5, 0.4, 240
    a, b = sa / math.sqrt(MPY), sb / math.sqrt(MPY)
    var = (2.0 * (1.0 - rho) + 0.5 * (a * a + b * b - 2.0 * rho * rho * a * b)) / T
    expected = math.sqrt(var * MPY)

    assert memmel_se(sa, sb, rho, T) == pytest.approx(expected)


# ── The annualisers ───────────────────────────────────────────────────────────

def test_ann_sharpe_annualises_by_the_square_root_of_twelve():
    rng = np.random.default_rng(3)
    x = rng.normal(0.01, 0.03, size=240)
    per_period = float(np.mean(x) / np.std(x, ddof=1))

    assert ann_sharpe(x) == pytest.approx(per_period * math.sqrt(MPY))


def test_ann_vol_annualises_the_sample_standard_deviation():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 0.05, size=240)

    assert ann_vol(x) == pytest.approx(float(np.std(x, ddof=1)) * math.sqrt(MPY))


def test_scaling_every_return_leaves_the_sharpe_unchanged_but_doubles_the_vol():
    """Sharpe is scale-free; volatility is not. A leverage change must not move one."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.01, 0.03, size=240)

    assert ann_sharpe(2.0 * x) == pytest.approx(ann_sharpe(x))
    assert ann_vol(2.0 * x) == pytest.approx(2.0 * ann_vol(x))
