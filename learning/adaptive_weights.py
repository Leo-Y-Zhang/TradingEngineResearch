"""
TradingEngineResearch — Adaptive Sleeve Weights
===================================
Validation-gated sleeve weight management (Part 18.2).

No sleeve weight change is ever applied without passing the same governance gate
used for live promotion:

  1. Run `PurgedWalkForwardSplitter` over the recent (≈90-day) window.
  2. Run `selection_rule()` on the proposed allocation's `ValidationResult`.
  3. If it passes, apply the new weights; otherwise retain the existing weights
     and log the rejection reason.

`frozen` mode is an operator-level circuit breaker: while frozen, every update
is rejected regardless of validation outcome. Freezing/unfreezing is logged and
fully reversible.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from research.validation import (
    PurgedWalkForwardSplitter,
    ValidationResult,
    selection_rule,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveWeights",
    "get_adaptive_weights",
    "reset_adaptive_weights",
]


class AdaptiveWeights:
    """Holds the live sleeve weights and gates every proposed change."""

    def __init__(
        self,
        initial_weights: Optional[dict] = None,
        frozen: bool = False,
    ) -> None:
        self.weights: dict = dict(initial_weights or {})
        self.frozen: bool = frozen

    # ── operator freeze controls ───────────────────────────────────────────────

    def freeze(self, reason: str) -> None:
        self.frozen = True
        logger.warning("ADAPTIVE_WEIGHTS frozen: %s", reason)

    def unfreeze(self, reason: str) -> None:
        self.frozen = False
        logger.warning("ADAPTIVE_WEIGHTS unfrozen: %s", reason)

    # ── the gate ───────────────────────────────────────────────────────────────

    def propose_and_validate(
        self,
        new_weights: dict,
        validation_result: ValidationResult,
        *,
        timestamps: Optional[list] = None,
    ) -> dict:
        """
        Validate and (if it passes) apply a proposed sleeve reweighting.

        ``timestamps`` is the recent (≈90-day) validation window and is REQUIRED:
        the purged walk-forward split runs on it before any change is applied. If
        it is omitted (or yields no folds), the change is rejected and the current
        weights are retained — the split is never bypassed.

        Returns a dict with ``applied_weights`` (the weights now in force),
        ``validation_result``, ``accepted`` and a human-readable ``reason``.
        ``n_folds`` reports how many purged walk-forward folds were evaluated.
        """
        if self.frozen:
            logger.warning("ADAPTIVE_WEIGHTS update rejected: instance is FROZEN")
            return self._result(validation_result=None, accepted=False,
                                 reason="FROZEN", n_folds=0)

        for sleeve, weight in new_weights.items():
            if float(weight) < 0.0:
                raise ValueError(f"sleeve weights must be non-negative, got {sleeve}={weight}")

        # The purged walk-forward split is MANDATORY before any weight change —
        # it is never skippable. Without a validation window we cannot run it, so
        # the change is rejected and the existing weights are retained.
        if timestamps is None:
            logger.warning("ADAPTIVE_WEIGHTS update rejected: no walk-forward window supplied")
            return self._result(validation_result=validation_result, accepted=False,
                                reason="VALIDATION_FAILED: walk-forward window required", n_folds=0)
        try:
            n_folds = len(self._run_purged_split(timestamps))
        except ValueError as exc:
            logger.warning("ADAPTIVE_WEIGHTS update rejected: walk-forward split failed: %s", exc)
            return self._result(validation_result=validation_result, accepted=False,
                                reason=f"VALIDATION_FAILED: {exc}", n_folds=0)
        if n_folds < 1:
            return self._result(validation_result=validation_result, accepted=False,
                                reason="VALIDATION_FAILED: no walk-forward folds", n_folds=0)

        if selection_rule(validation_result):
            self.weights = dict(new_weights)
            logger.info("ADAPTIVE_WEIGHTS applied new weights: %s", self.weights)
            return self._result(validation_result=validation_result, accepted=True,
                                reason="ACCEPTED", n_folds=n_folds)

        logger.warning("ADAPTIVE_WEIGHTS update rejected: selection_rule failed; retaining weights")
        return self._result(validation_result=validation_result, accepted=False,
                            reason="VALIDATION_FAILED: selection_rule not satisfied", n_folds=n_folds)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _run_purged_split(self, timestamps: list) -> list:
        """Run a purged, embargoed walk-forward split over the recent window."""
        index = pd.DatetimeIndex(timestamps)
        n = len(index)
        if n < 4:
            raise ValueError(f"insufficient history for walk-forward validation: {n} observations")
        train = max(int(n * 0.50), 2)
        valid = max(int(n * 0.20), 1)
        test = max(int(n * 0.20), 1)
        while train + valid + test > n and train > 2:
            train -= 1
        embargo = max(int(n * 0.05), 0)
        label_horizon = max(test, 1)
        splitter = PurgedWalkForwardSplitter(train, valid, test, embargo, label_horizon)
        return splitter.split(index)

    def _result(self, *, validation_result: Optional[ValidationResult],
                accepted: bool, reason: str, n_folds: int) -> dict:
        return {
            "applied_weights": dict(self.weights),
            "validation_result": validation_result,
            "accepted": accepted,
            "reason": reason,
            "n_folds": n_folds,
        }


# ── Module-level singleton ───────────────────────────────────────────────────

_ADAPTIVE: Optional[AdaptiveWeights] = None


def get_adaptive_weights() -> AdaptiveWeights:
    """Process-wide AdaptiveWeights singleton."""
    global _ADAPTIVE
    if _ADAPTIVE is None:
        _ADAPTIVE = AdaptiveWeights()
    return _ADAPTIVE


def reset_adaptive_weights() -> None:
    global _ADAPTIVE
    _ADAPTIVE = None
