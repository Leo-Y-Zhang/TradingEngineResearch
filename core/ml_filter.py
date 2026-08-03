"""
TradingEngineResearch — ML Direction Filter
===============================
A lightweight directional classifier that decides whether a candidate is a BUY,
a SELL, or should stay FLAT, and then defers to the order-flow gate so the model
never trades into a wall of opposing size.

Feature set (master prompt Part 12.1): ``ofi_signal``, ``insider_flow_age_days``,
``news_age_minutes``, ``realized_vol_5d``, ``realized_vol_20d``, ``spread_bps``,
``adv_ratio``, ``correlation_to_portfolio``, plus each freshness flag as its own
binary column. The model abstains (FLAT) whenever it is unfitted, errors, or the
microstructure gate vetoes the direction — it never raises into the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.engine.microstructure import ofi_filter_gate

logger = logging.getLogger(__name__)

__all__ = ["BASE_FEATURE_NAMES", "MLFilter", "get_filter", "reset_filter"]

BASE_FEATURE_NAMES: list[str] = [
    "ofi_signal",
    "insider_flow_age_days",
    "news_age_minutes",
    "realized_vol_5d",
    "realized_vol_20d",
    "spread_bps",
    "adv_ratio",
    "correlation_to_portfolio",
]

_BUY_THRESHOLD = 0.55      # P(up) above which we go BUY
_SELL_THRESHOLD = 0.45     # P(up) below which we go SELL


class MLFilter:
    """Directional classifier with an order-flow veto."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._model: Optional[RandomForestClassifier] = None
        self._freshness_keys: list[str] = []
        self._fitted = False

    # ── vectorisation ───────────────────────────────────────────────────────

    def _vectorize(self, features: dict) -> np.ndarray:
        base = [float(features.get(name, 0.0)) for name in BASE_FEATURE_NAMES]
        flags = features.get("freshness_flags", {}) or {}
        fresh = [1.0 if flags.get(key) else 0.0 for key in self._freshness_keys]
        return np.array(base + fresh, dtype=float)

    # ── fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        feature_dicts: list[dict],
        y_direction,
        freshness_keys: Optional[list[str]] = None,
    ) -> None:
        """Fit the classifier. ``y_direction`` is treated as up (BUY) when > 0."""
        self._freshness_keys = list(freshness_keys) if freshness_keys else []
        if len(feature_dicts) < 10:
            raise ValueError("MLFilter.fit needs >= 10 samples.")

        x = np.vstack([self._vectorize(f) for f in feature_dicts])
        y_up = (np.asarray(y_direction, dtype=float).ravel() > 0).astype(int)
        if np.unique(y_up).size < 2:
            raise ValueError("MLFilter.fit needs both up and down examples.")

        self._model = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state
        )
        self._model.fit(x, y_up)
        self._fitted = True
        logger.info("MLFilter fitted on %d samples (%d features).", len(feature_dicts), x.shape[1])

    # ── prediction ──────────────────────────────────────────────────────────

    def prob_up(self, features: dict) -> float:
        """Calibrated P(up) for a single feature dict (0.5 if unfitted/errored)."""
        if not self._fitted or self._model is None:
            return 0.5
        try:
            x = self._vectorize(features).reshape(1, -1)
            classes = list(self._model.classes_)
            proba = self._model.predict_proba(x)[0]
            return float(proba[classes.index(1)]) if 1 in classes else 0.0
        except Exception as exc:  # noqa: BLE001 — abstain on any error
            logger.warning("MLFilter.prob_up failed (%s); returning neutral 0.5.", exc)
            return 0.5

    def predict_direction(
        self, features: dict, ofi_norm: Optional[float] = None
    ) -> tuple[str, float]:
        """
        Return ``(direction, confidence)`` with ``direction`` in
        {"BUY", "SELL", "FLAT"}. The order-flow gate can veto a directional call
        to FLAT; an unfitted model abstains with ("FLAT", 0.0).
        """
        if not self._fitted or self._model is None:
            return ("FLAT", 0.0)

        p_up = self.prob_up(features)
        if p_up >= _BUY_THRESHOLD:
            direction = "BUY"
        elif p_up <= _SELL_THRESHOLD:
            direction = "SELL"
        else:
            return ("FLAT", float(abs(p_up - 0.5) * 2.0))

        confidence = float(abs(p_up - 0.5) * 2.0)

        ofi = features.get("ofi_signal", 0.0) if ofi_norm is None else ofi_norm
        if not ofi_filter_gate(direction, float(ofi)):
            logger.debug("MLFilter: %s vetoed by OFI gate (ofi=%.3f).", direction, float(ofi))
            return ("FLAT", confidence)

        return (direction, confidence)


# ── Module-level singleton ──────────────────────────────────────────────────────

_FILTER: Optional[MLFilter] = None


def get_filter() -> MLFilter:
    """Return the process-wide MLFilter singleton (created on first use)."""
    global _FILTER
    if _FILTER is None:
        _FILTER = MLFilter()
    return _FILTER


def reset_filter() -> None:
    """Drop the singleton (tests / restarts)."""
    global _FILTER
    _FILTER = None
