"""
TradingEngineResearch — Pluggable State-Store Backends
===========================================
A backend abstraction over the durable learning-loop state (model registry +
performance tracker). Both backends operate on the *live* singletons through the
backend-agnostic ``ops.persistence.dump_payload`` / ``load_payload`` core, so the
choice of storage never changes what is persisted — only where (ROADMAP Phase 6
item 2).

  • ``JsonStateStore``  — the default. Atomic JSON file; zero extra dependencies;
    used by tests and RESEARCH (preserves determinism and the existing path).
  • ``SqlStateStore``   — SQLAlchemy/SQLite (Postgres-swappable via the URL) for
    PAPER/LIVE durability. SQLAlchemy is imported lazily, so this module imports
    cleanly without the ``persistence`` extra installed.

``core.config.make_state_store(settings)`` selects the backend from config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ops import persistence

logger = logging.getLogger(__name__)

__all__ = ["StateStore", "JsonStateStore", "SqlStateStore"]


@runtime_checkable
class StateStore(Protocol):
    """Persist / restore the live registry + tracker singletons."""

    def save(self, retention_days: int = persistence.RETENTION_DAYS) -> None: ...

    def restore(self) -> None: ...


class JsonStateStore:
    """JSON-file backend (the default; wraps ``persistence.save_state``)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def save(self, retention_days: int = persistence.RETENTION_DAYS) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        persistence.save_state(self._path, retention_days)

    def restore(self) -> None:
        persistence.restore_state(self._path)


class SqlStateStore:
    """SQLAlchemy backend. SQLAlchemy is imported lazily on first use.

    ``save`` writes the retention-pruned payload in a single transaction
    (replace-state: delete-all + insert — correct and simple because retention
    bounds the data and the snapshot is the source of truth); a failure rolls
    back, leaving the prior persisted state intact. ``restore`` reconstructs the
    payload dict from the tables and feeds it to ``persistence.load_payload``.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine: Any = None

    def _setup(self) -> tuple[Any, Any]:
        """Create the engine + schema on first use; return (engine, sql_models)."""
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:  # pragma: no cover — only without the extra
            raise ImportError(
                "The SQL state store requires SQLAlchemy. "
                "Install it with: pip install tradingengineresearch[persistence]"
            ) from exc
        from ops import sql_models

        if self._engine is None:
            self._engine = create_engine(self._url)
            sql_models.Base.metadata.create_all(self._engine)
        return self._engine, sql_models

    def save(self, retention_days: int = persistence.RETENTION_DAYS) -> None:
        from sqlalchemy import delete, insert

        payload = persistence.dump_payload(retention_days)
        engine, m = self._setup()
        registry = payload.get("registry", {})
        tracker = payload.get("tracker", {})
        with engine.begin() as conn:
            for table in (
                m.SchemaMeta, m.ModelRecord, m.RegistryMeta, m.Prediction,
                m.Price, m.Fill, m.Outcome, m.TrackerMeta,
            ):
                conn.execute(delete(table))

            conn.execute(insert(m.SchemaMeta).values(id=1, version=payload.get("version")))

            for model_id, rec in registry.get("records", {}).items():
                conn.execute(insert(m.ModelRecord).values(
                    model_id=model_id,
                    promoted_to_live=bool(rec.get("promoted_to_live")),
                    data=rec,
                ))
            conn.execute(insert(m.RegistryMeta).values(
                id=1,
                record_order=registry.get("order", []),
                live_id=registry.get("live_id"),
                live_history=registry.get("live_history", []),
            ))

            for symbol, records in tracker.get("predictions", {}).items():
                for r in records:
                    asof = r.get("prediction", {}).get("asof_timestamp")
                    conn.execute(insert(m.Prediction).values(
                        symbol=symbol, asof_timestamp=asof, data=r))
            for symbol, pairs in tracker.get("prices", {}).items():
                for ts, value in pairs:
                    conn.execute(insert(m.Price).values(
                        symbol=symbol, ts=ts, value=value))
            for symbol, fills in tracker.get("fills", {}).items():
                for f in fills:
                    conn.execute(insert(m.Fill).values(
                        symbol=symbol, fill_timestamp=f.get("fill_timestamp"), data=f))
            for o in tracker.get("outcomes", []):
                resolved = o.get("resolved_at")
                conn.execute(insert(m.Outcome).values(
                    resolved_at=resolved if isinstance(resolved, str) else None, data=o))

            conn.execute(insert(m.TrackerMeta).values(
                id=1,
                brier_sum=float(tracker.get("brier_sum", 0.0)),
                brier_n=int(tracker.get("brier_n", 0)),
            ))
        logger.info("persistence: state saved to %s", self._url)

    def restore(self) -> None:
        from sqlalchemy import select

        engine, m = self._setup()
        with engine.connect() as conn:
            smeta = conn.execute(select(m.SchemaMeta)).first()
            version = smeta.version if smeta is not None else None

            records: dict[str, Any] = {}
            for row in conn.execute(select(m.ModelRecord)):
                records[row.model_id] = row.data
            rmeta = conn.execute(select(m.RegistryMeta)).first()
            registry = {
                "records": records,
                "order": list(rmeta.record_order) if rmeta is not None else [],
                "live_id": rmeta.live_id if rmeta is not None else None,
                "live_history": list(rmeta.live_history) if rmeta is not None else [],
            }

            predictions: dict[str, list[Any]] = {}
            for row in conn.execute(select(m.Prediction).order_by(m.Prediction.id)):
                predictions.setdefault(row.symbol, []).append(row.data)
            prices: dict[str, list[Any]] = {}
            for row in conn.execute(select(m.Price).order_by(m.Price.id)):
                prices.setdefault(row.symbol, []).append([row.ts, row.value])
            fills: dict[str, list[Any]] = {}
            for row in conn.execute(select(m.Fill).order_by(m.Fill.id)):
                fills.setdefault(row.symbol, []).append(row.data)
            outcomes = [row.data for row in conn.execute(select(m.Outcome).order_by(m.Outcome.id))]
            tmeta = conn.execute(select(m.TrackerMeta)).first()
            tracker = {
                "predictions": predictions,
                "prices": prices,
                "fills": fills,
                "outcomes": outcomes,
                "brier_sum": tmeta.brier_sum if tmeta is not None else 0.0,
                "brier_n": tmeta.brier_n if tmeta is not None else 0,
            }

        payload: dict[str, Any] = {"version": version, "registry": registry, "tracker": tracker}
        persistence.load_payload(payload)
        logger.info("persistence: state restored from %s", self._url)
