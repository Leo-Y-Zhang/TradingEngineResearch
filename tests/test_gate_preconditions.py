"""Regression tests for GATE-1, GATE-2 and GATE-3 — preconditions C9 of the 2026-07-28
benchmark-relative gate review (docs/project-control/specs/, section 1.4).

GATE-1: ``evaluate_factor`` deflated the real DSR at the FOLD count
(``n_trials = max(len(splits), 1)``), which made the criterion ~0.14 Sharpe too lenient
at 8 folds. Fixed: the default deflation is the programme's cumulative trial ledger.

GATE-2: ``ValidationResult.deflated_sharpe_ratio`` defaulted to 1.0 — a default-ALLOW
on a criterion documented as default-deny. Fixed: the default is 0.0 (default-DENY).

GATE-3: ``selection_rule`` never checked ``pbo_proxy`` at all, although the specs listed
PBO as a standing safeguard, ``ValidationResult`` carried the field, and the CSCV
estimator was implemented and tested. Both the 2026-07-27 and 2026-07-28 reviews recorded
it as still live. Fixed 2026-07-31: condition 6 rejects ``pbo_proxy >= PBO_MAX``, where
the bar is the failure point the estimator's own docstring names rather than a tuned one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import research.alpha_factory as af
from research.trial_ledger import cumulative_trials
from research.validation import (
    PBO_MAX,
    PurgedWalkForwardSplitter,
    ValidationResult,
    probability_of_backtest_overfitting,
    selection_rule,
)

# Values that clear every OTHER absolute criterion of selection_rule (mirrors the
# long-standing _passing_result helper in tests/test_phase2.py).
_PASSING_FIELDS = dict(
    mean_ic=0.05, mean_rank_ic=0.04, sharpe_net=1.2,
    turnover=0.02, hit_rate=0.55, max_drawdown=-0.05,
    pbo_proxy=0.15, deflated_sharpe_proxy=0.50,
    cost_drag_bps=5.0, stability_score=0.70,
    regime_breakdown={"trending": {"sharpe": 0.9, "ic": 0.03}},
    leakage_flags=[],
)


# ── GATE-2: the default is deny ───────────────────────────────────────────────

def test_gate2_default_dsr_is_deny() -> None:
    result = ValidationResult(**_PASSING_FIELDS)
    assert result.deflated_sharpe_ratio == 0.0, (
        "GATE-2 regression: deflated_sharpe_ratio must default to 0.0 (default-DENY), "
        f"got {result.deflated_sharpe_ratio}"
    )


def test_gate2_result_without_explicit_dsr_fails_gate() -> None:
    # Everything else passes; the DSR was simply never computed. The gate must DENY.
    result = ValidationResult(**_PASSING_FIELDS)
    assert selection_rule(result) is False, (
        "GATE-2 regression: a result that never computed its real DSR passed the gate"
    )


def test_gate2_explicit_dsr_still_passes() -> None:
    result = ValidationResult(**_PASSING_FIELDS, deflated_sharpe_ratio=0.99)
    assert selection_rule(result) is True


# ── GATE-1: default deflation is the trial LEDGER, never the fold count ───────

def _edge_panel(seed: int, n: int = 300, k: int = 3) -> tuple[pd.Series, pd.DataFrame]:
    """A synthetic factor that genuinely predicts the forward market return, with
    enough noise that the DSR is interior (sensitive to n_trials), not saturated."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = pd.DataFrame(rng.normal(0.0, 0.01, (n, k)), index=idx,
                       columns=[f"s{i}" for i in range(k)])
    h = 2
    fwd = ret.mean(axis=1).rolling(h).sum().shift(-h)
    factor = (fwd + rng.normal(0.0, 2e-3, n)).fillna(0.0)
    return factor, ret


_SPLITTER = PurgedWalkForwardSplitter(
    train_size=60, valid_size=30, test_size=30, embargo_size=2, label_horizon=2
)


def test_gate1_default_deflation_equals_ledger_not_folds() -> None:
    factor, ret = _edge_panel(seed=7)
    default = af.evaluate_factor(factor, ret, splitter=_SPLITTER)
    at_ledger = af.evaluate_factor(factor, ret, splitter=_SPLITTER,
                                   n_trials=cumulative_trials())
    assert default.deflated_sharpe_ratio == at_ledger.deflated_sharpe_ratio, (
        "GATE-1 regression: the default deflation must equal the cumulative trial "
        "ledger, not the fold count"
    )


def test_gate1_dsr_non_increasing_in_trials() -> None:
    factor, ret = _edge_panel(seed=7)
    dsr_1 = af.evaluate_factor(factor, ret, splitter=_SPLITTER, n_trials=1)
    dsr_many = af.evaluate_factor(factor, ret, splitter=_SPLITTER, n_trials=10_000)
    assert dsr_1.deflated_sharpe_ratio >= dsr_many.deflated_sharpe_ratio, (
        "DSR must be non-increasing in the trial count"
    )


def test_gate1_junk_still_denied_at_ledger_deflation() -> None:
    # Stronger deflation must not weaken the junk rejection (sanity direction check).
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=300)
    ret = pd.DataFrame(rng.normal(0.0, 0.01, (300, 3)), index=idx,
                       columns=["a", "b", "c"])
    junk = pd.Series(rng.normal(0.0, 1.0, 300), index=idx)
    result = af.evaluate_factor(junk, ret, splitter=_SPLITTER)
    assert selection_rule(result) is False


# ── GATE-3: PBO is actually checked, and the check has teeth ──────────────────
# GATE-3 (recorded open by BOTH the 2026-07-27 and 2026-07-28 reviews): the specs
# described PBO as a standing safeguard, ValidationResult carried a `pbo_proxy` field,
# validation.py implemented the CSCV estimator -- and `selection_rule` never looked at
# it. Closed 2026-07-31 by adding condition 6 against `PBO_MAX`.

def _result(**overrides: object) -> ValidationResult:
    """A result that clears every OTHER criterion, so one field decides the verdict."""
    return ValidationResult(**{**_PASSING_FIELDS, "deflated_sharpe_ratio": 0.99,
                               **overrides})


def test_gate3_pbo_bar_is_the_estimators_own_documented_failure_point() -> None:
    """The bar must not be a number invented to make something pass or fail."""
    assert PBO_MAX == 0.50, (
        "GATE-3: the bar is the failure point the estimator's own docstring names "
        "('PBO near 0.5+ => the research process is overfitting'), not a tuned value"
    )


def test_gate3_overfit_selection_is_rejected() -> None:
    assert selection_rule(_result(pbo_proxy=0.55)) is False, (
        "GATE-3 regression: a result with PBO above the bar must NOT be promotable"
    )


def test_gate3_clean_selection_still_passes() -> None:
    assert selection_rule(_result(pbo_proxy=0.15)) is True


def test_gate3_bar_is_exclusive_not_inclusive() -> None:
    """Exactly at the bar is a FAIL. A gate that admits its own threshold is not a gate."""
    assert selection_rule(_result(pbo_proxy=PBO_MAX)) is False
    assert selection_rule(_result(pbo_proxy=PBO_MAX - 1e-9)) is True


def test_gate3_pbo_is_the_only_thing_separating_these_two() -> None:
    """Everything else identical, so the verdict flip is attributable to PBO alone."""
    assert selection_rule(_result(pbo_proxy=0.10)) is True
    assert selection_rule(_result(pbo_proxy=0.90)) is False


# ── the estimator itself: a positive control, not just a threshold ────────────

def test_pbo_estimator_separates_noise_selection_from_a_real_edge() -> None:
    """POSITIVE CONTROL. Selecting the best of N pure-noise configs is exactly what PBO
    exists to catch, and a genuinely dominant config is what it must NOT flag. If the
    estimator cannot tell these apart, the gate condition above is decoration."""
    rng = np.random.default_rng(20260731)
    noise = rng.normal(0.0, 1.0, size=(240, 8))
    pbo_noise = probability_of_backtest_overfitting(noise)

    with_edge = noise.copy()
    with_edge[:, 0] += 0.5                      # one config genuinely dominates
    pbo_edge = probability_of_backtest_overfitting(with_edge)

    assert pbo_edge < pbo_noise, (
        f"PBO must fall when a real edge exists: noise={pbo_noise:.3f} "
        f"edge={pbo_edge:.3f}")
    assert pbo_edge < PBO_MAX, (
        f"a genuinely dominant config must CLEAR the gate, got PBO={pbo_edge:.3f}")
    assert pbo_noise > 0.25, (
        f"selecting the best of 8 noise configs should look overfit, got {pbo_noise:.3f}")
