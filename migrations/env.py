"""Alembic migration environment for TradingEngineResearch persistence (Phase 6 item 2).

The target schema is ``ops.sql_models.Base.metadata`` (the same metadata the SQL
state store creates), and the database URL is resolved from the TradingEngineResearch
settings so migrations and the running platform always agree on the location.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ops.sql_models import Base

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False: running a migration in-process must not
    # silence the platform's already-created loggers (fileConfig's default
    # disables them, which suppressed e.g. data.feature_store drift warnings).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _resolve_url() -> str:
    """Explicit env var → alembic.ini → TradingEngineResearch settings default."""
    url = os.environ.get("ENGINE_PERSISTENCE__DATABASE_URL") or config.get_main_option(
        "sqlalchemy.url"
    )
    if url:
        return url
    try:
        from core.config import get_settings

        ps = get_settings().persistence
        return ps.database_url or f"sqlite:///{(ps.state_dir / 'tradingengineresearch.db').as_posix()}"
    except Exception:  # pragma: no cover — fall back to the documented default
        return "sqlite:///state/tradingengineresearch.db"


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
