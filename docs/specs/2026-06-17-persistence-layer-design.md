# Persistence Layer (SQLAlchemy + SQLite) — Design

**Date:** 2026-06-17 · **ROADMAP:** Phase 6, item 2 ·
**Status:** approved (design), pre-implementation

## Purpose

The learning-loop singletons — the model registry (`ops/model_registry.py`) and
the performance tracker (`learning/performance_tracker.py`) — gained durable JSON
snapshots in Phase 4 (`ops/persistence.py`). That module already states the next
step: *"the full database persistence (SQLAlchemy/Alembic) … can replace the
storage backend behind these same call sites."* This item delivers exactly that:
a SQL-backed, crash-safe persistence backend for PAPER/LIVE, while keeping the
JSON backend as the **default** for tests and RESEARCH (preserving determinism
and a zero-dependency path).

Nothing calls persistence on a schedule today (the engine's only "persistence"
reference is the markdown cycle-audit). The scheduled run-loop (Phase 6 item 3)
will persist via the configured `StateStore` (`make_state_store`); this item
provides the swappable durable backend that run-loop needs.

## Scope (explicit)

**In scope — "minimal swap":** persist the *same* state as today (registry +
tracker) through a pluggable backend, add a SQLAlchemy/SQLite implementation, and
wire Alembic for schema versioning.

**Deferred (noted follow-ups, NOT in this item):**
- Persisting `learning/adaptive_weights.py` sleeve weights — currently **not**
  persisted at all (a real durability gap, but out of this minimal scope).
- A queryable per-cycle run/audit table (today's cycle audit is markdown via
  `ops/audit_log.py`).
- Postgres-specific features and concurrent multi-process access.

These are the "swap + close gaps" / "full productionization" options that were
considered and not chosen.

## Approaches considered

- **A (chosen): pluggable backend behind the existing facade.** `save_state` /
  `restore_state` keep their signatures and gain a backend chosen by config;
  JSON stays the default, a SQLAlchemy/SQLite backend is added. Smallest change,
  reuses the existing dict (de)serialization and retention logic, and is the path
  `ops/persistence.py` already documents.
- **B: replace JSON entirely.** Rejected — discards a working, tested,
  deterministic path; forces every test and every RESEARCH replay onto a database
  (slower, non-deterministic ordering risks, more setup) for no benefit.
- **C: full repository-per-entity normalization.** Rejected — over-normalizes
  contracts that Pydantic already validates (`PredictionRow`, `FillEvent`, …);
  that level of modelling is the declined "full productionization" scope and adds
  schema churn with no payoff until an external SQL reader exists.

## Components

### `ops/persistence.py` (refactor — backend-agnostic core)

Extract the existing `_dump_*` / `_load_*` into two public, backend-agnostic
functions operating on the live singletons:

- `dump_payload(retention_days: int) -> dict` — the current snapshot dict
  (registry + tracker), with the **deterministic** retention prune (cutoff =
  newest data timestamp − N, never wall clock) computed once, here, so both
  backends prune identically.
- `load_payload(payload: dict) -> None` — repopulate the live registry + tracker
  singletons (today's `_load_*`).

`save_state(path, retention_days)` / `restore_state(path)` are retained
unchanged (default to the JSON backend) so existing call sites and tests keep
working.

### `ops/state_store.py` (new — the backend abstraction)

- `StateStore` `Protocol`: `save(retention_days: int) -> None`,
  `restore() -> None` (both act on the live singletons via `dump_payload` /
  `load_payload`).
- `JsonStateStore(path)` — wraps today's atomic write/replace behavior.
- `SqlStateStore(url)` — SQLAlchemy backend (see below).
- `make_state_store(settings) -> StateStore` lives in `core/config.py` (it needs
  settings); `state_store.py` stays import-cycle-free w.r.t. config.

### `ops/sql_models.py` (new — schema, lazy-imported)

SQLAlchemy 2.0 typed `DeclarativeBase` + the tables below. The whole module is
imported lazily inside `SqlStateStore` (the `cryptography`-in-vault /
`ib-insync`-in-broker pattern) so the core platform and the JSON path need no SQL
dependency.

### `alembic/` + `alembic.ini` (new, under `TradingEngineResearch/`)

One initial migration that creates the schema. Tests use
`Base.metadata.create_all` against a throwaway SQLite DB; Alembic governs
real-deployment schema evolution. Alembic reads the database URL from settings.

### `core/config.py` (extend)

`PersistenceSettings` gains:
- `backend: Literal["json", "sqlite"] = "json"` (default-deny toward the simple,
  deterministic path).
- `database_url: Optional[str] = None`, defaulting to
  `sqlite:///{state_dir}/tradingengineresearch.db` when `backend == "sqlite"`.

Plus `make_state_store(settings) -> StateStore`. **SQLite by default,
Postgres-swappable** via the URL — SQLAlchemy abstracts the dialect, honoring
"SQLite for lightweight" now without foreclosing "PostgreSQL for serious" later.

### Packaging

A dedicated `persistence` extra (`sqlalchemy>=2,<3`, `alembic>=1.13,<2`),
mirroring the `vault` extra; SQLAlchemy is already pinned in `constraints.txt`
(2.0.36) and `alembic` gets pinned there. Both are lazy-imported, so they stay
optional. `state/` (the default DB directory) is gitignored (`*.db` already is;
add `state/`).

## Schema (lean; mirrors the current payload)

JSON columns are used for the already-Pydantic-validated nested contracts — a
deliberate choice, not under-modelling.

| Table | Columns (abbreviated) |
|-------|-----------------------|
| `model_record` | `model_id` PK, `model_type`, `training_window_start/end`, `feature_schema_version`, `hyperparameters` JSON, `validation_metrics` JSON, `calibration_metrics` JSON, `drift_baseline` JSON, `regime_breakdown` JSON, `artifact_path`, `promoted_to_live`, `promoted_at`, `retired_at` |
| `registry_meta` | single row: `order` JSON, `live_id`, `live_history` JSON |
| `prediction` | `id` PK, `symbol` (idx), `prediction` JSON (`PredictionRow` dump), `source`, `sleeve`, `regime`, `execution_regime`, `resolved_horizons` JSON, `features` JSON, `asof_timestamp` (idx, for retention) |
| `price` | `id` PK, `symbol`+`ts` (idx), `value` |
| `fill` | `id` PK, `symbol` (idx), `fill` JSON (`FillEvent` dump), `fill_timestamp` |
| `outcome` | `id` PK, `payload` JSON, `resolved_at` |
| `tracker_meta` | single row: `brier_sum`, `brier_n` |
| `schema_meta` | `version` (matches `_STATE_VERSION`; forward-compat guard) |

## Data flow

- **save:** `dump_payload(retention_days)` (prunes by retention once) → backend
  writes. The SQL backend writes the pruned payload in **one transaction**
  (replace-state: delete-all + insert — correct and simple because retention
  bounds the data and the snapshot is the source of truth).
- **restore:** backend reads → `load_payload(payload)` into the live singletons
  (identical to today). The SQL backend reconstructs the payload dict from the
  tables, so `load_payload` is reused verbatim.
- **mode/backend selection:** config picks the backend; JSON is the default, so
  RESEARCH, determinism tests, and existing call sites are byte-for-byte
  unchanged. PAPER/LIVE may select `sqlite`.
- **no engine wiring here:** the engine does not auto-persist in this item; the
  run-loop (item 3) will persist via the configured `StateStore`
  (`make_state_store`). The path-based `save_state`/`restore_state` remain as the
  JSON convenience wrappers for existing callers and tests.

## Error handling

- `SqlStateStore` lazy-imports SQLAlchemy with an actionable message
  (`pip install tradingengineresearch[persistence]`) — the platform runs without the extra.
- `save` is atomic via a single transaction; a failure rolls back, leaving the
  prior persisted state intact (parity with the JSON write-then-replace
  guarantee).
- `restore` raises `FileNotFoundError` / `ValueError` (or the SQL equivalent) on
  a missing or corrupt store — the caller decides whether a cold start is
  acceptable (unchanged contract).
- `schema_meta.version` is checked on restore (mismatch → `ValueError`), matching
  the JSON `version` guard.

## Testing (`tests/test_persistence_sql.py`)

The JSON path is unchanged, so existing `ops/persistence.py` tests stay green.
New tests, all on `tmp_path`/in-memory SQLite (no network, deterministic):

- **SQL round-trip:** populate registry + tracker, `SqlStateStore.save()`, reset
  singletons, `.restore()` → state equals the original.
- **Retention parity:** SQL prunes identically to JSON for the same data and
  `retention_days` (cutoff = newest − N).
- **Backend parity:** from one `dump_payload`, JSON and SQL restore to equivalent
  singleton state.
- **Schema/migration:** `Base.metadata.create_all` builds the schema; the initial
  Alembic migration applies cleanly and produces the same tables.
- **Config:** `make_state_store(settings)` returns `JsonStateStore` by default and
  `SqlStateStore` when `backend="sqlite"`, with the derived default URL.
- **Version guard:** restoring a `schema_meta.version` mismatch raises.

## Backward compatibility

`save_state(path)` / `restore_state(path)` keep working and default to JSON; no
existing call site changes. The SQL backend is purely additive and optional.
