"""Multi-sleeve combination, and the reachability check on Kelly growth.

Two claims are pinned here, both of which the module states in prose.

``combined_sharpe`` reports the brief's formula AND the exact equal-risk answer, on
the stated grounds that the exact form collapses to the brief's one when its two
assumptions hold -- equal Sharpes and equal pairwise correlations. That collapse is a
mathematical identity, so it is asserted, and the disagreement is shown to appear
exactly when an assumption is broken.

``kelly_reality`` exists because of a standing rule the programme adopted on
2026-07-28: bare ``g = 3S^2/8`` is correct arithmetic and misleading as a deployable
number, because it silently assumes running at sigma = S/2. The test pins the chain
that turns it back into something reachable.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves.multiasset_defensive import combined_sharpe, kelly_reality


def _corr(names: list[str], rho: float) -> pd.DataFrame:
    n = len(names)
    c = np.full((n, n), rho)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=names, columns=names)


# ── combined_sharpe ───────────────────────────────────────────────────────────

def test_the_exact_form_collapses_to_the_brief_formula_when_its_assumptions_hold():
    """Equal Sharpes and equal pairwise correlations: the two must agree exactly."""
    names = ["a", "b", "c", "d"]
    sharpes = dict.fromkeys(names, 0.55)

    for rho in (0.0, 0.15, 0.4, 0.75):
        out = combined_sharpe(sharpes, _corr(names, rho))
        assert out["exact_equal_risk_sharpe"] == pytest.approx(out["formula_sharpe"]), \
            f"the identity failed at rho={rho}"


def test_the_two_reported_figures_are_algebraically_the_same_number_always():
    """A finding, not just a check: the "approximation error" here is identically zero.

    The docstring says the brief's formula assumes equal Sharpes and equal pairwise
    correlations, and that both figures are reported "so the approximation error is
    visible rather than assumed away". For an EQUAL-RISK portfolio the two cannot
    differ, because the quadratic form sees nothing but the mean correlation:

        rho_bar = mean of the off-diagonal entries
        1'C1    = n + 2*(sum of upper off-diagonals) = n(1 + (n-1)*rho_bar)   ALWAYS

        exact   = sum(s) / sqrt(1'C1)     = sum(s) / sqrt(n(1 + (n-1)rho_bar))
        approx  = s_bar * sqrt(n/(1 + (n-1)rho_bar))
                = (sum(s)/n) * sqrt(n) / sqrt(1 + (n-1)rho_bar)
                = sum(s) / sqrt(n(1 + (n-1)rho_bar))

    Neither assumption is needed. Both the mean-Sharpe and the sqrt(n) cancel, so the
    fields agree for any Sharpes and any correlation structure. Asserted below over
    unequal Sharpes AND a deliberately non-uniform correlation matrix.

    Nothing reported is WRONG -- both numbers are the correct equal-risk Sharpe. But
    the comparison between them cannot detect anything, so it should not be read as
    evidence that the brief's approximation was checked and held. Measuring a real
    approximation error would need a different comparator, e.g. the Sharpe of the
    actually-weighted portfolio rather than the equal-risk closed form.
    """
    names = ["a", "b", "c"]
    lumpy = pd.DataFrame(
        [[1.00, 0.90, 0.05],
         [0.90, 1.00, 0.05],
         [0.05, 0.05, 1.00]],
        index=names, columns=names,
    )

    for sharpes in (dict.fromkeys(names, 0.6), {"a": 1.2, "b": 0.3, "c": 0.2}):
        for corr in (_corr(names, 0.3), lumpy):
            out = combined_sharpe(sharpes, corr)
            assert out["exact_equal_risk_sharpe"] == pytest.approx(
                out["formula_sharpe"]
            ), "the two fields are the same expression; they cannot disagree"


def test_independent_sleeves_combine_by_the_square_root_of_their_number():
    names = ["a", "b", "c", "d"]
    out = combined_sharpe(dict.fromkeys(names, 0.5), _corr(names, 0.0))

    assert out["formula_sharpe"] == pytest.approx(0.5 * math.sqrt(4))
    assert out["mean_pairwise_corr"] == pytest.approx(0.0)


def test_perfectly_correlated_sleeves_add_no_diversification():
    names = ["a", "b", "c"]
    out = combined_sharpe(dict.fromkeys(names, 0.7), _corr(names, 1.0))

    assert out["formula_sharpe"] == pytest.approx(0.7)
    assert out["exact_equal_risk_sharpe"] == pytest.approx(0.7)


def test_correlation_reduces_the_combined_sharpe_monotonically():
    names = ["a", "b", "c"]
    combined = [combined_sharpe(dict.fromkeys(names, 0.6), _corr(names, r))["exact_equal_risk_sharpe"]
                for r in (0.0, 0.25, 0.5, 0.9)]

    assert combined == sorted(combined, reverse=True)


def test_a_single_sleeve_combines_to_itself():
    out = combined_sharpe({"only": 0.42}, _corr(["only"], 0.0))

    assert out["formula_sharpe"] == pytest.approx(0.42)
    assert out["exact_equal_risk_sharpe"] == pytest.approx(0.42)
    assert out["mean_pairwise_corr"] == 0.0      # no pairs exist


def test_sleeves_absent_from_the_correlation_matrix_are_dropped_not_guessed_at():
    names = ["a", "b"]
    out = combined_sharpe({"a": 0.5, "b": 0.5, "missing": 9.9}, _corr(names, 0.2))

    assert out["sleeves"] == names
    assert out["mean_sharpe"] == pytest.approx(0.5)


def test_half_kelly_growth_is_three_eighths_of_the_squared_sharpe():
    names = ["a", "b"]
    out = combined_sharpe(dict.fromkeys(names, 0.8), _corr(names, 0.1))

    assert out["half_kelly_growth_formula"] == pytest.approx(
        3.0 * out["formula_sharpe"] ** 2 / 8.0)
    assert out["half_kelly_growth_exact"] == pytest.approx(
        3.0 * out["exact_equal_risk_sharpe"] ** 2 / 8.0)


# ── kelly_reality ─────────────────────────────────────────────────────────────

def _series(mu: float, sd: float, n: int = 240, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, size=n), index=idx)


def test_the_reported_chain_is_internally_consistent():
    """required vol = S/2, leverage = required/own, implied dd = measured dd x leverage."""
    out = kelly_reality(_series(0.008, 0.02, seed=1))

    assert out["required_vol"] == pytest.approx(out["sharpe"] / 2.0)
    assert out["leverage_on_own_vol"] == pytest.approx(
        out["required_vol"] / out["own_vol"])
    assert out["implied_max_drawdown_at_that_leverage"] == pytest.approx(
        out["measured_max_drawdown"] * out["leverage_on_own_vol"])
    assert out["half_kelly_growth"] == pytest.approx(3.0 * out["sharpe"] ** 2 / 8.0)


def test_a_low_volatility_book_needs_leverage_to_reach_its_kelly_growth():
    """The point of the whole function: the growth number assumes a vol the book
    does not run at, and the leverage that closes the gap is what makes it real."""
    out = kelly_reality(_series(0.010, 0.012, seed=2))

    assert out["sharpe"] > 1.0
    assert out["leverage_on_own_vol"] > 1.0
    assert out["required_vol"] > out["own_vol"]


def test_survivability_is_decided_at_a_sixty_percent_implied_drawdown():
    out = kelly_reality(_series(0.010, 0.012, seed=3))

    assert out["survivable"] is bool(
        out["implied_max_drawdown_at_that_leverage"] > -0.60)


def test_a_series_with_no_dispersion_or_no_sharpe_reports_nothing_rather_than_guessing():
    flat = pd.Series(0.0, index=pd.date_range("2000-01-31", periods=60, freq="ME"))
    assert kelly_reality(flat) == {}


def test_the_measured_drawdown_is_negative_so_the_implied_one_is_too():
    """Sign discipline: this module inherits multiasset_trend's NEGATIVE convention."""
    out = kelly_reality(_series(0.008, 0.03, seed=4))

    assert out["measured_max_drawdown"] < 0.0
    assert out["implied_max_drawdown_at_that_leverage"] < 0.0


# ── optimal_sharpe: the comparator that replaced the vacuous one ──────────────
#
# The test above proves `formula_sharpe` and `exact_equal_risk_sharpe` CANNOT disagree.
# `optimal_sharpe` was added 2026-08-01 so the module has a comparison that can fail.
# These tests exist to show it is not vacuous in the same way -- one asserts a STRICT
# inequality on the exact input where the old pair is provably identical.

def test_the_optimum_is_never_worse_than_equal_risk():
    """Equal risk is one feasible weighting, so the optimum is an upper bound on it."""
    names = ["a", "b", "c"]
    lumpy = pd.DataFrame(
        [[1.00, 0.90, 0.05],
         [0.90, 1.00, 0.05],
         [0.05, 0.05, 1.00]],
        index=names, columns=names,
    )

    for sharpes in (dict.fromkeys(names, 0.6), {"a": 1.2, "b": 0.3, "c": 0.2}):
        for corr in (_corr(names, 0.0), _corr(names, 0.4), lumpy):
            out = combined_sharpe(sharpes, corr)
            assert out["optimal_sharpe"] >= out["exact_equal_risk_sharpe"] - 1e-12


def test_the_optimum_equals_equal_risk_exactly_when_the_briefs_assumptions_hold():
    """Equal Sharpes AND equal correlations: equal-risk weighting IS the optimum.

    This is what makes the new field a control rather than merely a second number. It
    agrees precisely in the symmetric case the brief assumes, so any gap it reports is
    caused by asymmetry rather than by having swapped in a different formula.
    """
    names = ["a", "b", "c", "d"]
    for rho in (0.0, 0.25, 0.6):
        out = combined_sharpe(dict.fromkeys(names, 0.55), _corr(names, rho))
        assert out["optimal_sharpe"] == pytest.approx(out["exact_equal_risk_sharpe"])
        assert out["sharpe_dispersion"] == pytest.approx(0.0)
        assert out["corr_dispersion"] == pytest.approx(0.0)


def test_the_new_comparator_disagrees_where_the_old_pair_provably_cannot():
    """The whole point. Same input as the vacuity test; this comparison moves.

    On a lumpy correlation matrix with unequal Sharpes the two old fields are still
    bit-for-bit identical, while `optimal_sharpe` is strictly larger -- the gap being
    what equal-risk weighting costs when the sleeves are not interchangeable.
    """
    names = ["a", "b", "c"]
    lumpy = pd.DataFrame(
        [[1.00, 0.90, 0.05],
         [0.90, 1.00, 0.05],
         [0.05, 0.05, 1.00]],
        index=names, columns=names,
    )
    out = combined_sharpe({"a": 1.2, "b": 0.3, "c": 0.2}, lumpy)

    assert out["formula_sharpe"] == pytest.approx(out["exact_equal_risk_sharpe"])
    assert out["optimal_sharpe"] > out["exact_equal_risk_sharpe"] + 1e-6
    assert out["sharpe_dispersion"] > 0.0
    assert out["corr_dispersion"] > 0.0


def test_two_uncorrelated_sleeves_combine_in_quadrature():
    """Closed form: with rho=0 the optimum is sqrt(s1^2 + s2^2)."""
    out = combined_sharpe({"a": 0.3, "b": 0.4}, _corr(["a", "b"], 0.0))

    assert out["optimal_sharpe"] == pytest.approx(math.hypot(0.3, 0.4))


def test_a_singular_correlation_matrix_reports_nothing_rather_than_an_unholdable_number():
    """Perfectly correlated sleeves: C is singular and the optimum is a blown-up hedge.

    `multiasset_value.combined_sharpe_optimal` returns NaN at |rho| >= 1 for the same
    reason. NaN keeps an unreachable figure out of a result file.
    """
    out = combined_sharpe({"a": 0.7, "b": 0.4}, _corr(["a", "b"], 1.0))

    assert math.isnan(out["optimal_sharpe"])
    # The equal-risk figure is still perfectly well defined and still reported.
    assert math.isfinite(out["exact_equal_risk_sharpe"])
