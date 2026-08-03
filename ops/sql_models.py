"""
TradingEngineResearch — Persistence SQL Schema (SQLAlchemy 2.0)
===================================================
The relational schema behind ``ops.state_store.SqlStateStore`` (ROADMAP Phase 6
item 2). One row per persisted entity (model record, prediction, price point,
fill, outcome) plus singleton meta rows; the already-Pydantic-validated nested
contracts are stored in ``JSON`` columns rather than over-normalised — the rows
are real and queryable while the JSON column guarantees an exact round-trip with
the backend-agnostic payload from ``ops.persistence.dump_payload``.

This module imports SQLAlchemy at top level and is therefore imported **lazily**
(only inside ``SqlStateStore``), so the core platform and the default JSON
backend need no SQL dependency — the same pattern as ``cryptography`` in the
vault and ``ib-insync`` in the broker. Install via ``pip install
tradingengineresearch[persistence]``.

``Base.metadata`` is the single source of truth for the schema: the SQL store
calls ``create_all`` and the initial Alembic migration binds to this same
metadata, so the two can never drift.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import Boolean, Float, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "Base",
    "SchemaMeta",
    "ModelRecord",
    "RegistryMeta",
    "Prediction",
    "Price",
    "Fill",
    "Outcome",
    "TrackerMeta",
]


class Base(DeclarativeBase):
    """Declarative base; ``Base.metadata`` is the authoritative schema."""


class SchemaMeta(Base):
    """Single-row state-format version guard (mirrors ``_STATE_VERSION``)."""

    __tablename__ = "schema_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer)


class ModelRecord(Base):
    """One row per registry model record; full record dict in ``data``."""

    __tablename__ = "model_record"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    promoted_to_live: Mapped[bool] = mapped_column(Boolean, default=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class RegistryMeta(Base):
    """Single-row registry scalars: ordering + live pointer + rollback history."""

    __tablename__ = "registry_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_order: Mapped[list[Any]] = mapped_column(JSON)
    live_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    live_history: Mapped[list[Any]] = mapped_column(JSON)


class Prediction(Base):
    """One row per open prediction record (with its features) in the tracker."""

    __tablename__ = "prediction"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    asof_timestamp: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class Price(Base):
    """One row per (symbol, timestamp, value) price observation."""

    __tablename__ = "price"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    ts: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    value: Mapped[float] = mapped_column(Float)


class Fill(Base):
    """One row per fill event; full ``FillEvent`` dump in ``data``."""

    __tablename__ = "fill"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    fill_timestamp: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class Outcome(Base):
    """One row per resolved outcome; full outcome dict in ``data``."""

    __tablename__ = "outcome"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resolved_at: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON)


class TrackerMeta(Base):
    """Single-row tracker scalars: rolling Brier sum + count."""

    __tablename__ = "tracker_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brier_sum: Mapped[float] = mapped_column(Float, default=0.0)
    brier_n: Mapped[int] = mapped_column(Integer, default=0)
