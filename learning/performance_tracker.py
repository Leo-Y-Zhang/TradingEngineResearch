"""
TradingEngineResearch — Performance Tracker
===============================
Multi-horizon outcome tracking and the post-trade learning loop (Part 18.1).

For every prediction it resolves cost-adjusted realised returns at +1d / +5d /
+10d / +20d once the horizon has elapsed and price data exists, then feeds those
outcomes back into the learning subsystems:

  • ml_return_model.get_model().record_outcome(predicted, actual)
  • ml_return_model.get_model().record_training_example(features, actual) at the
    1d horizon when the prediction-time features were captured (the refit loop)
  • optimizer.get_view_tracker().record(source, predicted, actual)
  • calibration diagnostics (Brier score on p_positive vs realised sign)

Outcomes are tracked along every dimension the spec requires — symbol, strategy
sleeve, model version, regime, and execution regime — and resolution is
idempotent (a horizon is recorded once). Where a fill exists, realised returns
are fill-cost adjusted; otherwise the gross market return is used.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core import ml_return_model
from core.engine import optimizer
from data.data_contracts import FillEvent, PredictionRow

logger = logging.getLogger(__name__)

__all__ = [
    "HORIZONS",
    "PerformanceTracker",
    "get_performance_tracker",
    "reset_performance_tracker",
]

HORIZONS: tuple[int, ...] = (1, 5, 10, 20)   # resolution horizons, in days


@dataclass
class _PredictionRecord:
    prediction: PredictionRow
    source: str
    sleeve: str
    regime: str
    execution_regime: str
    resolved_horizons: set[int] = field(default_factory=set)
    features: Optional[dict] = None      # features at prediction time (training loop)


class PerformanceTracker:
    """Resolves multi-horizon outcomes and feeds the learning loop."""

    def __init__(self) -> None:
        self._predictions: dict[str, list[_PredictionRecord]] = {}
        self._price_ts: dict[str, list[datetime]] = {}
        self._price_val: dict[str, list[float]] = {}
        self._fills: dict[str, list[FillEvent]] = {}
        self._outcomes: list[dict] = []
        self._brier_sum: float = 0.0
        self._brier_n: int = 0

    # ── ingestion ────────────────────────────────────────────────────────────

    def record_prediction(
        self,
        prediction: PredictionRow,
        *,
        source: str = "ml",
        sleeve: str = "blended",
        regime: str = "unknown",
        execution_regime: str = "normal_exec",
        features: Optional[dict] = None,
    ) -> None:
        if prediction.asof_timestamp is None:
            raise ValueError("PredictionRow.asof_timestamp is required to track outcomes")
        self._predictions.setdefault(prediction.symbol, []).append(
            _PredictionRecord(prediction, source, sleeve, regime, execution_regime,
                              features=features)
        )

    def record_price(self, symbol: str, timestamp: datetime, price: float) -> None:
        ts = self._price_ts.setdefault(symbol, [])
        val = self._price_val.setdefault(symbol, [])
        # keep ts sorted (prices usually arrive in order; bisect handles the rest)
        idx = bisect_right(ts, timestamp)
        ts.insert(idx, timestamp)
        val.insert(idx, float(price))

    def record_fill(self, fill: FillEvent) -> None:
        self._fills.setdefault(fill.symbol, []).append(fill)

    # ── price helpers ──────────────────────────────────────────────────────────

    def _price_asof(self, symbol: str, ts: datetime) -> Optional[float]:
        """Most recent price with timestamp <= ts (point-in-time safe)."""
        stamps = self._price_ts.get(symbol)
        if not stamps:
            return None
        idx = bisect_right(stamps, ts) - 1
        if idx < 0:
            return None
        return self._price_val[symbol][idx]

    def _has_elapsed(self, symbol: str, ts: datetime) -> bool:
        """True once a price at or after ts exists (the horizon has elapsed)."""
        stamps = self._price_ts.get(symbol)
        if not stamps:
            return False
        return stamps[-1] >= ts

    def _entry_cost_bps(self, symbol: str, anchor: datetime) -> float:
        """Slippage of the fill nearest the prediction time, else 0 (no fill)."""
        fills = self._fills.get(symbol)
        if not fills:
            return 0.0
        nearest = min(fills, key=lambda f: abs((f.fill_timestamp - anchor).total_seconds()))
        return abs(float(nearest.slippage_bps))

    # ── outcome resolution ──────────────────────────────────────────────────────

    def evaluate_signal(self, symbol: str, timestamp: datetime) -> None:
        """
        Resolve every elapsed, unresolved horizon for the symbol's predictions
        dated at or before ``timestamp`` and feed them into the learning loop.
        """
        model = ml_return_model.get_model()
        tracker = optimizer.get_view_tracker()

        for record in self._predictions.get(symbol, []):
            pred = record.prediction
            anchor = pred.asof_timestamp
            if anchor is None or anchor > timestamp:
                continue
            base_price = self._price_asof(symbol, anchor)
            if base_price is None or base_price <= 0.0:
                continue
            cost_bps = self._entry_cost_bps(symbol, anchor)

            for horizon in HORIZONS:
                if horizon in record.resolved_horizons:
                    continue
                target = anchor + timedelta(days=horizon)
                if not self._has_elapsed(symbol, target):
                    continue
                future_price = self._price_asof(symbol, target)
                if future_price is None:
                    continue

                raw_return = (future_price - base_price) / base_price
                actual_return = raw_return - cost_bps / 10_000.0
                predicted = float(pred.expected_return)

                model.record_outcome(predicted, actual_return)
                tracker.record(record.source, predicted, actual_return)
                self._update_brier(float(pred.p_positive), actual_return)
                # The 1d horizon is the model's prediction target: when the features
                # at prediction time were captured, this resolved outcome becomes a
                # training example (the real refit loop).
                if horizon == 1 and record.features:
                    model.record_training_example(record.features, actual_return)

                record.resolved_horizons.add(horizon)
                self._outcomes.append({
                    "symbol": symbol,
                    "sleeve": record.sleeve,
                    "model_version": pred.model_version,
                    "regime": record.regime,
                    "execution_regime": record.execution_regime,
                    "horizon": horizon,
                    "predicted_return": predicted,
                    "actual_return": actual_return,
                    "raw_return": raw_return,
                    "cost_bps": cost_bps,
                    "resolved_at": target,
                })
                logger.debug(
                    "PERF_OUTCOME %s h=%dd predicted=%.4f actual=%.4f",
                    symbol, horizon, predicted, actual_return,
                )

    def _update_brier(self, p_positive: float, actual_return: float) -> None:
        outcome = 1.0 if actual_return > 0.0 else 0.0
        self._brier_sum += (p_positive - outcome) ** 2
        self._brier_n += 1

    # ── reporting ────────────────────────────────────────────────────────────

    def outcomes(self) -> list[dict]:
        return list(self._outcomes)

    def calibration_report(self) -> dict:
        brier = self._brier_sum / self._brier_n if self._brier_n else 0.0
        return {
            "brier_score": float(brier),
            "n_samples": self._brier_n,
            "n_outcomes": len(self._outcomes),
        }


# ── Module-level singleton ───────────────────────────────────────────────────

_TRACKER: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Process-wide PerformanceTracker singleton."""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = PerformanceTracker()
    return _TRACKER


def reset_performance_tracker() -> None:
    global _TRACKER
    _TRACKER = None
