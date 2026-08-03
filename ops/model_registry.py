"""
TradingEngineResearch — Model Registry
===========================
The system of record for every model that reaches PAPER/LIVE. Each promotion to
live is a deliberate, audited event with a `ModelRecord`; each rollback restores
the previously-live model exactly (Part 19.1).

Governance posture:
  • Nothing trades live without a registered `ModelRecord`.
  • Promotion demotes the incumbent and remembers it, so a rollback is always
    reversible to the last-known-good model.
  • The shadow/challenger workflow is first-class: a registered-but-not-promoted
    model is the pending challenger surfaced by `latest_shadow()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from research.validation import ValidationResult, selection_rule

logger = logging.getLogger(__name__)

__all__ = [
    "ModelRecord",
    "ModelRegistry",
    "get_model_registry",
    "reset_model_registry",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── 19.1 ModelRecord ─────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    """A registered model and its full provenance / validation trail."""

    model_id: str
    model_type: str
    training_window: tuple[datetime, datetime]
    feature_schema_version: str
    hyperparameters: dict
    validation_metrics: ValidationResult
    calibration_metrics: dict
    drift_baseline: dict
    regime_breakdown: dict[str, dict]
    artifact_path: str
    promoted_to_live: bool = False
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None


# ── Registry ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """In-process registry of `ModelRecord`s with promote / rollback semantics."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._order: list[str] = []                 # registration order
        self._live_id: Optional[str] = None
        self._live_history: list[str] = []          # stack of prior live ids

    def register(self, record: ModelRecord) -> str:
        """Register a new model. Returns its id. Duplicate ids are refused."""
        if record.model_id in self._records:
            raise ValueError(f"model_id already registered: {record.model_id!r}")
        self._records[record.model_id] = record
        self._order.append(record.model_id)
        logger.info("MODEL_REGISTRY registered %s (%s)", record.model_id, record.model_type)
        return record.model_id

    def promote(self, model_id: str) -> None:
        """Promote a registered model to live, demoting and remembering the
        incumbent. Validation-gated (golden rule 5): a record whose
        ``validation_metrics`` fails ``selection_rule()`` can never go live."""
        if model_id not in self._records:
            raise ValueError(f"unknown model_id: {model_id!r}")
        if self._live_id == model_id:
            return  # idempotent
        if not selection_rule(self._records[model_id].validation_metrics):
            raise ValueError(
                f"refusing to promote {model_id!r}: validation_metrics fail selection_rule()"
            )

        if self._live_id is not None:
            incumbent = self._records[self._live_id]
            incumbent.promoted_to_live = False
            incumbent.retired_at = _now()
            self._live_history.append(self._live_id)

        record = self._records[model_id]
        record.promoted_to_live = True
        record.promoted_at = _now()
        record.retired_at = None
        self._live_id = model_id
        logger.info("MODEL_REGISTRY promoted %s to LIVE", model_id)

    def rollback(self, reason: str, expect_current: Optional[str] = None) -> Optional[str]:
        """Restore the previously-live model and return its id. Raises if there
        is none. Pass ``expect_current`` (the model you intend to demote) to make
        operator retries idempotent: if the live model is already someone else,
        the rollback has evidently happened and this becomes a logged no-op."""
        if expect_current is not None and self._live_id != expect_current:
            logger.info(
                "MODEL_REGISTRY rollback no-op: live is %r, expected %r (%s)",
                self._live_id, expect_current, reason,
            )
            return None
        if not self._live_history:
            raise ValueError("no previous live model to roll back to")

        if self._live_id is not None:
            current = self._records[self._live_id]
            current.promoted_to_live = False
            current.retired_at = _now()

        previous_id = self._live_history.pop()
        previous = self._records[previous_id]
        previous.promoted_to_live = True
        previous.promoted_at = _now()
        previous.retired_at = None
        self._live_id = previous_id
        logger.warning("MODEL_REGISTRY rollback to %s: %s", previous_id, reason)
        return previous_id

    def promotion_candidate(self) -> Optional[ModelRecord]:
        """The current shadow model IF its validation passes ``selection_rule()``
        — i.e. it is eligible for a (manual, never automatic) promotion."""
        shadow = self.latest_shadow()
        if shadow is not None and selection_rule(shadow.validation_metrics):
            return shadow
        return None

    def latest_live(self) -> Optional[ModelRecord]:
        """The current live model, or None."""
        return self._records[self._live_id] if self._live_id is not None else None

    def latest_shadow(self) -> Optional[ModelRecord]:
        """The most recently registered model that is neither live nor retired."""
        for model_id in reversed(self._order):
            record = self._records[model_id]
            if not record.promoted_to_live and record.retired_at is None:
                return record
        return None

    def get(self, model_id: str) -> ModelRecord:
        """Look up a record by id."""
        if model_id not in self._records:
            raise ValueError(f"unknown model_id: {model_id!r}")
        return self._records[model_id]


# ── Module-level singleton ───────────────────────────────────────────────────

_REGISTRY: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Process-wide ModelRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ModelRegistry()
    return _REGISTRY


def reset_model_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
