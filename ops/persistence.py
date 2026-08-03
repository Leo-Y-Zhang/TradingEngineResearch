"""
TradingEngineResearch — State Persistence
=============================
Durable JSON snapshots of the learning-loop singletons (ROADMAP Phase 4): the
model registry (records, live pointer, rollback history) and the performance
tracker (open predictions with their features, price history, fills, resolved
outcomes, calibration sums). `save_state` writes one self-contained file;
`restore_state` repopulates the live singletons in place.

Retention is **deterministic**: the cutoff is ``newest data timestamp −
retention_days`` (never the wall clock), so saving a replayed state prunes
identically every time. Default 90 days, per the spec's retention requirement.

This is the durability layer the singletons were missing. ``dump_payload`` /
``load_payload`` are the backend-agnostic core; ``ops/state_store.py`` adds a
pluggable backend (JSON here, SQLAlchemy/SQLite in ``SqlStateStore``) selected
via ``core.config.make_state_store`` (ROADMAP Phase 6 item 2).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from data.data_contracts import FillEvent, PredictionRow, to_aware_utc
from learning import performance_tracker as pt
from ops import model_registry as reg
from research.validation import ValidationResult

logger = logging.getLogger(__name__)

__all__ = ["RETENTION_DAYS", "dump_payload", "load_payload", "save_state", "restore_state"]

RETENTION_DAYS = 90
_STATE_VERSION = 1


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


def _parse(ts: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(ts) if ts else None


# ── registry ─────────────────────────────────────────────────────────────────────

def _dump_registry(registry: reg.ModelRegistry) -> dict:
    records = {}
    for model_id, r in registry._records.items():
        records[model_id] = {
            "model_id": r.model_id,
            "model_type": r.model_type,
            "training_window": [_iso(r.training_window[0]), _iso(r.training_window[1])],
            "feature_schema_version": r.feature_schema_version,
            "hyperparameters": r.hyperparameters,
            "validation_metrics": vars(r.validation_metrics),
            "calibration_metrics": r.calibration_metrics,
            "drift_baseline": r.drift_baseline,
            "regime_breakdown": r.regime_breakdown,
            "artifact_path": r.artifact_path,
            "promoted_to_live": r.promoted_to_live,
            "promoted_at": _iso(r.promoted_at),
            "retired_at": _iso(r.retired_at),
        }
    return {
        "records": records,
        "order": list(registry._order),
        "live_id": registry._live_id,
        "live_history": list(registry._live_history),
    }


def _load_registry(data: dict, registry: reg.ModelRegistry) -> None:
    registry._records = {}
    for model_id, d in data.get("records", {}).items():
        start, end = d["training_window"]
        registry._records[model_id] = reg.ModelRecord(
            model_id=d["model_id"],
            model_type=d["model_type"],
            training_window=(_parse(start), _parse(end)),   # type: ignore[arg-type]
            feature_schema_version=d["feature_schema_version"],
            hyperparameters=d["hyperparameters"],
            validation_metrics=ValidationResult(**d["validation_metrics"]),
            calibration_metrics=d["calibration_metrics"],
            drift_baseline=d["drift_baseline"],
            regime_breakdown=d["regime_breakdown"],
            artifact_path=d["artifact_path"],
            promoted_to_live=bool(d["promoted_to_live"]),
            promoted_at=_parse(d["promoted_at"]),
            retired_at=_parse(d["retired_at"]),
        )
    registry._order = list(data.get("order", []))
    registry._live_id = data.get("live_id")
    registry._live_history = list(data.get("live_history", []))


# ── tracker ──────────────────────────────────────────────────────────────────────

def _newest_timestamp(tracker: pt.PerformanceTracker) -> Optional[datetime]:
    candidates: list[datetime] = []
    for stamps in tracker._price_ts.values():
        if stamps:
            candidates.append(stamps[-1])
    for records in tracker._predictions.values():
        for r in records:
            if r.prediction.asof_timestamp is not None:
                candidates.append(r.prediction.asof_timestamp)
    for fills in tracker._fills.values():
        candidates.extend(f.fill_timestamp for f in fills if f.fill_timestamp is not None)
    for o in tracker._outcomes:
        if isinstance(o.get("resolved_at"), datetime):
            candidates.append(o["resolved_at"])
    # tz-robust: this aggregates timestamps from heterogeneous sources (price data, prediction
    # asofs, broker fills, resolved outcomes) which can be a naive/aware MIX in LIVE — normalise to
    # aware UTC before comparing, else max() raises "can't compare offset-naive and offset-aware".
    return max(to_aware_utc(c) for c in candidates) if candidates else None


def _dump_tracker(tracker: pt.PerformanceTracker, retention_days: int) -> dict:
    newest = _newest_timestamp(tracker)
    cutoff = newest - timedelta(days=retention_days) if newest is not None else None

    def keep(ts: Optional[datetime]) -> bool:
        return ts is None or cutoff is None or to_aware_utc(ts) >= cutoff   # cutoff is aware UTC

    predictions: dict[str, list[dict]] = {}
    for symbol, records in tracker._predictions.items():
        kept = [
            {
                "prediction": r.prediction.model_dump(mode="json"),
                "source": r.source,
                "sleeve": r.sleeve,
                "regime": r.regime,
                "execution_regime": r.execution_regime,
                "resolved_horizons": sorted(r.resolved_horizons),
                "features": r.features,
            }
            for r in records if keep(r.prediction.asof_timestamp)
        ]
        if kept:
            predictions[symbol] = kept

    prices: dict[str, list[list]] = {}
    for symbol, stamps in tracker._price_ts.items():
        vals = tracker._price_val[symbol]
        kept_pairs = [[_iso(t), v] for t, v in zip(stamps, vals) if keep(t)]
        if kept_pairs:
            prices[symbol] = kept_pairs

    fills = {
        symbol: [f.model_dump(mode="json") for f in fs if keep(f.fill_timestamp)]
        for symbol, fs in tracker._fills.items()
    }
    fills = {s: fs for s, fs in fills.items() if fs}

    outcomes = []
    for o in tracker._outcomes:
        resolved_at = o.get("resolved_at")
        if isinstance(resolved_at, datetime) and not keep(resolved_at):
            continue
        outcomes.append({**o, "resolved_at": _iso(resolved_at)
                         if isinstance(resolved_at, datetime) else resolved_at})

    return {
        "predictions": predictions,
        "prices": prices,
        "fills": fills,
        "outcomes": outcomes,
        "brier_sum": tracker._brier_sum,
        "brier_n": tracker._brier_n,
    }


def _load_tracker(data: dict, tracker: pt.PerformanceTracker) -> None:
    tracker._predictions = {}
    for symbol, records in data.get("predictions", {}).items():
        tracker._predictions[symbol] = [
            pt._PredictionRecord(
                prediction=PredictionRow.model_validate(d["prediction"]),
                source=d["source"],
                sleeve=d["sleeve"],
                regime=d["regime"],
                execution_regime=d["execution_regime"],
                resolved_horizons=set(d["resolved_horizons"]),
                features=d.get("features"),
            )
            for d in records
        ]
    tracker._price_ts = {}
    tracker._price_val = {}
    for symbol, pairs in data.get("prices", {}).items():
        tracker._price_ts[symbol] = [_parse(t) for t, _ in pairs]   # type: ignore[misc]
        tracker._price_val[symbol] = [float(v) for _, v in pairs]
    tracker._fills = {
        symbol: [FillEvent.model_validate(d) for d in fs]
        for symbol, fs in data.get("fills", {}).items()
    }
    tracker._outcomes = [
        {**o, "resolved_at": _parse(o["resolved_at"])
         if isinstance(o.get("resolved_at"), str) else o.get("resolved_at")}
        for o in data.get("outcomes", [])
    ]
    tracker._brier_sum = float(data.get("brier_sum", 0.0))
    tracker._brier_n = int(data.get("brier_n", 0))


# ── public API ───────────────────────────────────────────────────────────────────

def dump_payload(retention_days: int = RETENTION_DAYS) -> dict[str, Any]:
    """Backend-agnostic snapshot of the live registry + tracker singletons.

    The deterministic retention prune (cutoff = newest data timestamp −
    ``retention_days``, never wall clock) is applied here, once, so every storage
    backend persists an identically-pruned payload."""
    return {
        "version": _STATE_VERSION,
        "registry": _dump_registry(reg.get_model_registry()),
        "tracker": _dump_tracker(pt.get_performance_tracker(), retention_days),
    }


def load_payload(payload: dict[str, Any]) -> None:
    """Repopulate the live registry + tracker singletons from a ``dump_payload``
    dict, in place. Raises ``ValueError`` on an unsupported state version."""
    version = payload.get("version")
    if version != _STATE_VERSION:
        raise ValueError(f"unsupported state version: {version!r}")
    _load_registry(payload.get("registry", {}), reg.get_model_registry())
    _load_tracker(payload.get("tracker", {}), pt.get_performance_tracker())


def save_state(path: str | Path, retention_days: int = RETENTION_DAYS) -> None:
    """Snapshot the singletons to a JSON file ``path`` (atomic write-then-replace).

    The JSON convenience wrapper; for a config-selected backend (JSON or SQL) use
    ``core.config.make_state_store``."""
    payload = dump_payload(retention_days)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    logger.info("persistence: state saved to %s", path)


def restore_state(path: str | Path) -> None:
    """Repopulate the singletons from a ``save_state`` JSON file. Raises
    ``FileNotFoundError`` / ``ValueError`` on a missing or corrupt file — the
    caller decides whether a cold start is acceptable."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    load_payload(payload)
    logger.info("persistence: state restored from %s", path)
