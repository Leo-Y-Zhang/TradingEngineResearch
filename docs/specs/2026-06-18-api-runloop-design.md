# Design — API + Scheduled Run-Loop (ROADMAP Phase 6, item 3)

**Status:** accepted · **Date:** 2026-06-18

## Problem

Everything needed to *run* TradingEngineResearch exists — `TradingEngine.run_cycle`, the
mode-aware `make_broker`, the pluggable `make_state_store`, the encrypted vault,
and the central `EngineSettings`. But **nothing calls the engine on a
schedule**, and there is **no way to observe a running platform**. The engine is
a pure function of `CycleInputs`; it has no clock, no data feed, no persistence
trigger, and no surface. This item supplies the *composition root* and the
*operational surface* without changing any quant logic.

The offline analog already exists: `backtesting/harness.Backtester` replays the
engine over a fixed price history, builds PIT-safe `CycleInputs`, carries the
book, and books net-of-cost returns. The run-loop is the **online** analog —
same "build inputs → run cycle → carry book" spine, but it pulls fresh data,
persists after every cycle, reconciles against the broker, and exposes state.

## Goals

1. A **scheduled run-loop** that builds engine + broker + state-store from
   `EngineSettings`, restores persisted learning state on start, runs one
   cycle per tick, persists after each, and carries the book across ticks **and
   across restarts**.
2. A read-mostly **HTTP API** (FastAPI) to observe a running platform: health,
   status, latest monitoring snapshot + alerts, current book, latest cycle
   summary; plus a guarded on-demand single-cycle trigger.
3. **Mode discipline preserved end-to-end** (golden rule 1): RESEARCH plans no
   orders; PAPER submits zero live orders; only LIVE may reach the broker — the
   engine already enforces this, and the loop must not weaken it.
4. Net-new code only in the operational layer (`ops/`). No edits to the 13-step
   pipeline, contracts, optimiser, or risk math.

## Non-goals

- A frontend (the `frontend/` scaffold stays empty this item).
- Real-time streaming / websockets.
- Authn/z on the API (it binds loopback; deployment hardening is item 6).
- Auto-correcting broker divergence (we *surface* divergence; correcting the
  book from the broker is a deliberate risk decision, out of scope here).

## Design

### Placement
- `ops/run_loop.py` — `EngineService`, `LoopState`, `build_cycle_inputs`,
  `run_forever`, and a `python -m ops.run_loop` operator entry.
- `ops/api.py` — `create_app(service) -> FastAPI` factory (lazy `fastapi`
  import; FastAPI lives in the `app` extra).

Both land in `ops/` — already packaged (`tool.setuptools.packages.find`),
mypy-checked, and coverage-tracked — so there is **no packaging churn** and no
new top-level package to register.

### `EngineService` (the composition root)
Constructed from a `EngineSettings`:
- `engine = TradingEngine(**engine_kwargs(settings))` with
  `broker = make_broker(settings, vault)` injected (vault opened only when LIVE
  needs a secret).
- `state_store = make_state_store(settings)`.
- `symbols` = `settings.universe` (new field) or an explicit constructor arg;
  a non-empty universe is required (fail-closed: refuse to run blind).
- `price_provider: Callable[[datetime, list[str]], pd.DataFrame]` — injected.
  Default wraps `data.price_ingestion.fetch_prices` (network → `pragma: no
  cover`); tests inject an in-memory frame. Returns a PIT history (index ≤ asof,
  columns = symbols).

State held: `last_result`, `last_snapshot`, `last_alerts`, `cycle_count`,
`last_asof`, `current_book` (weights), `live_orders_total`.

Lifecycle:
- `start()` — idempotent. `state_store.restore()` (re-hydrate the registry +
  performance-tracker singletons) and load `LoopState` (book + counters) from
  `{state_dir}/loop_state.json`. Connect the broker if LIVE.
- `run_once(asof) -> CycleResult`:
  1. `prices = price_provider(asof, symbols)`; `build_cycle_inputs(...)` with
     `current_weights = current_book` (carried, durable).
  2. `result = engine.run_cycle(inputs)`.
  3. New book: `achieved_weights` (PAPER/LIVE) else `target_weights`
     (RESEARCH); on `result.blocked`, carry the existing book unchanged. Mirrors
     the harness exactly.
  4. `state_store.save()` then persist `LoopState` (atomic write-then-replace).
  5. If a broker is present: `bs = broker.account_state(asof)`; convert the
     internal book to expected shares (`w * nav / price`) and call
     `position_divergence` vs `bs.positions`; any divergence → a reconciliation
     alert appended to `last_alerts` (surfaced, never auto-applied).
  6. Update `last_*`, `cycle_count += 1`, `live_orders_total +=
     result.live_orders_submitted`.
- `stop()` — persist final state; disconnect broker.

`LoopState` durability matters: a restarted PAPER deployment would otherwise see
an empty in-memory `PaperBroker` and lose its book. The persisted `LoopState` is
the authoritative `current_weights`; the broker view is used only for the
divergence *check*. Kept as a small atomic JSON sidecar so the SQL/JSON state
store stays focused on registry + tracker (no schema churn).

### `run_forever(service, *, interval_seconds, clock, max_cycles=None, should_stop=None)`
The thin scheduling shell: compute `asof = clock()`, `service.run_once(asof)`,
sleep `interval_seconds`, repeat. `clock`, `max_cycles`, and `should_stop` are
injected so the loop **body** is unit-tested; only the literal `time.sleep` and
the unbounded operator loop are `pragma: no cover`. A per-cycle exception is
logged loudly and the loop continues (a transient data error must not kill a
long-running platform) — except `KeyboardInterrupt`/`SystemExit`, which stop it.

### `ops/api.py` — `create_app(service)`
FastAPI app, lazy import. Endpoints (read-mostly):
- `GET /health` → `{status, mode, cycle_count, last_asof, broker_connected}`
- `GET /status` → mode, capital, universe, cycle_count, last_asof, blocked,
  live_orders_total
- `GET /monitoring` → `last_snapshot` (4 sections) + `last_alerts`
- `GET /book` → `current_book` + last `target_weights`/`achieved_weights`
- `GET /cycle/latest` → sanitized summary of `last_result` (no raw objects)
- `POST /cycle/run` → trigger one `run_once(clock())`; returns the summary.
  Guarded: 409 if a cycle is already running (a simple re-entrancy flag).

Responses are plain JSON-able dicts (a `_summarize_result` helper coerces the
`CycleResult` dataclass to primitives) — no leaking of live objects, numpy
scalars, or timestamps-as-objects.

### Config additions (`core/config.py`)
- `universe: list[str] = []` — the traded symbol set (empty ⇒ must be supplied
  explicitly; `EngineService` fails closed on an empty universe).
- `cycle_interval_seconds: float = 86_400.0` (> 0) — default cadence for
  `run_forever` (daily); overridable via `ENGINE_CYCLE_INTERVAL_SECONDS`.

## Testing (TDD)
`tests/test_run_loop.py`:
- `run_once` in RESEARCH builds inputs from an injected provider, runs the
  engine, books the target weights, persists `LoopState`, places **zero** live
  orders.
- Book carries across two `run_once` calls; a blocked cycle carries unchanged.
- `LoopState` round-trips across a fresh `EngineService` (restart durability).
- `state_store.save` is called once per cycle (spy/temp dir).
- PAPER `run_once` against a `PaperBroker` keeps `live_orders_total == 0`;
  divergence check runs and surfaces an alert when positions disagree.
- `run_forever` runs exactly `max_cycles` cycles with an injected clock; a
  per-cycle exception is swallowed and the loop continues.
- Empty universe ⇒ `EngineService` refuses (fail-closed).

`tests/test_api.py` (skips if `fastapi`/`httpx` missing — both installed):
- Each GET endpoint returns 200 with the documented keys after a cycle.
- `POST /cycle/run` runs a cycle and increments `cycle_count`.
- `/health` reflects mode and broker connectivity.

Gates: `python -m pytest tests/`, `ruff check`, `mypy`. Then an adversarial
review workflow over the uncommitted diff before commit.

## Risks / mitigations
- **Network in tests** → all live data/broker paths are injected or
  `pragma: no cover`; the suite never hits the network (matches existing
  fixtures-only policy).
- **Mode safety regression** → the loop never calls `broker.submit`; only the
  engine does, and only in LIVE for `risk_approved` orders. A test asserts
  `live_orders_total == 0` in RESEARCH/PAPER.
- **Coverage** → fastapi/httpx are installed, so `ops/api.py` is exercised;
  `ops/run_loop.py` wall-clock/network lines are `pragma: no cover`.
