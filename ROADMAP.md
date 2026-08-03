# TradingEngineResearch — Improvement Roadmap

Each item is a
named improvement; tags are `(impact / effort)`. Ordered for **highest risk-adjusted
returns as a quant engine**: first make returns *measurable*, then protect them with
correct risk math, then add orthogonal alpha, then close the learning loop, then ship.

Derived from a full 8-subsystem audit (70 findings) on 2026-06-08. Check items off as done.
Correctness items should be **verify-then-fix** (adversarial check first — single-pass audit).

---

## Phase 0 — Foundation (fast; unblocks safe iteration) ✅ DONE 2026-06-09
- [x] **pyproject** — make TradingEngineResearch an installable package; fix the `python -m pytest` collision `(high/med)`
- [x] **Dep lockfile** — pin exact validated versions + upper bounds on numpy/scipy/sklearn `(high/small)`
- [x] **Test config + coverage** — `pyproject` pytest config, add `pytest-cov` with a floor `(med/small)`
- [x] **CI + pre-commit** — run ruff + mypy + pytest on push `(med/small)`

## Phase 1 — Risk & sizing correctness (protects returns / drawdowns) ✅ DONE 2026-06-09
- [x] **Enforce CVaR limit** (not flag-only) in the optimiser objective `(high/med)`
- [x] **Fix no-history CVaR=0.0** silent fallback → conservative non-zero `(high/small)`
- [x] **PSD projection** after covariance blend + RMT denoise `(med/small)`
- [x] **Out-of-sample calibration** — fix in-sample Brier/ECE/isotonic fit `(high/med)`
- [x] **Wire `cross_sectional_prior`** (currently 0.0 → shrinks μ to zero) `(high/small)`
- [x] **Apply `size_multiplier` before optimisation** (spec STEP 8, not after STEP 10) `(high/med)`
- [x] **Crisis composite tightens CVaR limit + vol target** (use the continuous severity) `(med/med)`
- [x] **Cornish-Fisher CVaR sanity floor** vs Gaussian `(med/small)`

## Phase 2 — Make returns measurable (highest-leverage: can't improve what you can't measure) ✅ DONE 2026-06-10
- [x] **Backtest / walk-forward harness** — replay the engine over history, net-of-cost, purged splits `(high/large)` ✅ 2026-06-09 (`backtesting/`)
- [x] **Wire `evaluate_factor()` end-to-end** + `selection_rule()` promotion gating `(high/med)` ✅ 2026-06-10 (`promote_candidates` + live factor library)
- [x] **Real feature ingestion** behind `get_features` + a recorded (non-synthetic) data fixture `(high/large)` ✅ 2026-06-10 (`data/price_ingestion.py` + yfinance fixture)
- [x] **Determinism / reproducibility test** for RESEARCH mode `(med/small)` ✅ 2026-06-09
- [x] **Property-based tests** for numeric kernels (OFI, vol_ratio, CVaR, TCA monotonicity) `(high/med)` ✅ 2026-06-09 (surfaced + fixed 2 latent bugs)

## Phase 3 — Expand & sharpen alpha (more orthogonal return sources) ✅ DONE 2026-06-10
- [x] **Carry sleeve** — implement + wire into STEP 4 `(high/large)` ✅ 2026-06-10 (`strategies/carry.py`, dividend-yield carry + dividend ingestion)
- [x] **Volatility-overlay sleeve** — implement + wire into STEP 4 `(high/large)` ✅ 2026-06-10 (`strategies/volatility_overlay.py`, risk-off overlay)
- [x] **Wire NLP sentiment end-to-end** — FinBERT+aggregator → ML `sentiment_score` + event sleeve (today 0.0) `(high/med)` ✅ 2026-06-10 (`nlp/sentiment_pipeline.py` + `strategies/sentiment.py` + engine STEP 4/6 wiring + `fetch_news`)
- [x] **Signal-health gates on a `ValidationResult`** (not this-cycle confidence) `(med/med)` ✅ 2026-06-10 (per-sleeve validation registry → STEP 5)
- [x] **OFI L2 partial-data robustness** + level-count handling `(med/small)` ✅ 2026-06-10 (skip bad levels, fall to L1)
- [x] **Tail calibration from live outcomes** — use `record_outcome(tail_event=...)` `(med/med)` ✅ 2026-06-10 (live tail-rate vs base → refit trigger)

## Phase 4 — Close the learning loop (compounding improvement) ✅ DONE 2026-06-10
- [x] **Real background-refit hook** so `needs_refit` actually retrains `(high/med)` ✅ 2026-06-10 (training buffer + `refit()` + initial-fit bootstrap; prices/features now feed the tracker)
- [x] **Reconcile held book from achieved fills** (fix multi-cycle delta accounting) `(high/med)` ✅ 2026-06-10 (`achieved_weights` + delta-sized orders + explicit exits for dropped names)
- [x] **Populate monitoring MODEL/TRADING/HEALTH** sections from the engine `(high/med)` ✅ 2026-06-10 (STEP 13 `_monitoring_state`: model reports + fills/book + ingest health)
- [x] **Persist registry + performance tracker + per-cycle audit** (durability; 90-day retention) `(high/med)` ✅ 2026-06-10 (`ops/persistence.py` JSON snapshots + deterministic retention)
- [x] **Complete shadow/challenger lifecycle** + rollback safety/idempotency `(med/small)` ✅ 2026-06-10 (validation-gated promote, idempotent rollback, STEP-13 candidate surfacing)
- [x] **Persist audit to `DECISIONS.md`** per cycle (spec STEP 13) `(high/small)` ✅ 2026-06-10 (opt-in `ops/audit_log.py` cycle trail — separate file, see DECISIONS.md)

## Phase 5 — Data contracts & store hardening ✅ DONE 2026-06-10
- [x] **`PortfolioState` / `BrokerState` contracts** (positions + NAV) `(high/med)` ✅ 2026-06-10 (contracts 10+11, LIVE fail-closed NAV/connectivity, `position_divergence` reconciliation primitive)
- [x] **Cross-validate store schema vs model `FEATURE_NAMES`** `(high/small)` ✅ 2026-06-10 (`validate_schema_against_model`; all 18 model features now have freshness+imputation metadata, conservative risk imputes)
- [x] **Versioned PIT retrieval** + `schema_hash` drift warnings `(med/med)` ✅ 2026-06-10 (`get_features(schema_version=)`; registration warns when a version's feature set grows)
- [x] **Mode-aware missing-symbol** (fail-closed in LIVE) + boundary tz validation `(med/small)` ✅ 2026-06-10 (LIVE raises on missing symbol / naive asof_time; off-LIVE degrades loudly)
- [x] **Stronger train/serve parity** (distributional shift, not just min/max) `(med/med)` ✅ 2026-06-10 (PSI over train-quantile bins; >0.25 fails parity)

## Phase 6 — Productionization / go-live (run it for real) ✅ DONE 2026-06-18
- [x] **Central config (`pydantic-settings`) + encrypted secrets vault** `(high/med)` ✅ 2026-06-11 (`core/config.py` + `core/vault.py`; LIVE double-armed via `confirm_live`; vault CLI; **adversarial-review hardened**: crash-safe `rotate`, KDF bounds, `validate_assignment` + money-boundary re-check, `secrets/` default-deny)
- [x] **Persistence layer** (SQLAlchemy + Alembic) behind the singletons `(high/med)` ✅ 2026-06-17 (`ops/state_store.py` pluggable `StateStore`: JSON default + `SqlStateStore` SQLite/Postgres; `ops/sql_models.py`; `migrations/` Alembic; `make_state_store` config seam; same registry+tracker state — adaptive_weights + per-cycle audit table noted as follow-ups)
- [x] **API + scheduled run-loop** wiring `TradingEngine` `(high/large)` ✅ 2026-06-18 (`ops/run_loop.py` `EngineService` composition root + `LoopState` durable book/counters + `run_forever` driver; `ops/api.py` FastAPI observe/control surface; `core/config.py` `universe`/`cycle_interval_seconds`; **adversarial-review hardened** pre-commit: LIVE broker connect fail-closed, PAPER reconciliation skip, service-level cycle lock, LIVE on-demand trigger disabled, durable fsync'd atomic writes)
- [x] **Broker `Protocol` + real `broker/ibkr.py`** (PaperBroker for PAPER; ib-insync) `(high/large)` ✅ 2026-06-10 (`broker/{protocol,paper,ibkr}.py`; fail-closed connectivity; translation unit-tested, gateway paths supervised-only)
- [x] **Logging/metrics/alerting sink** (alerts are computed but go nowhere) `(med/med)` ✅ 2026-06-18 (`ops/observability.py`: `AlertSink` Protocol + Logging/Jsonl/Composite/Null sinks, `MetricsRegistry` with Prometheus exposition; `core/config.py` `AlertingSettings` + `make_alert_sink`; wired into `EngineService.run_once` fail-soft + API `/metrics`,`/metrics/prometheus`)
- [x] **Docker / compose / run scripts** for reproducible deploy `(med/med)` ✅ 2026-06-18 (multi-stage `Dockerfile` non-root + pinned constraints, `docker-compose.yml` loopback-only API + optional Postgres profile, `scripts/entrypoint.sh` combined/loop/api modes, `.dockerignore`, `.env.example`, `docs/DEPLOY.md`; `serve_combined` one-process loop+API + `create_app_from_settings` ASGI factory) — **Phase 6 COMPLETE**

## Phase 7 — Engineering polish
- [ ] **TypedDicts** for engine inter-step payloads (machine-checked step contracts) `(med/med)`
- [ ] **Dedup helpers** (`_side_sign`, `_now`, `_safe`) + strategy-sleeve base `(low/small)`
- [ ] **Stricter error handling** — named exceptions / `strict` mode so regressions don't hide behind `_safe` `(low/small)`
- [x] **Fuzz the Pydantic contract validators** across their constraint space `(low/small)` ✅ 2026-06-18 (`tests/test_contract_fuzzing.py`, hypothesis). **Surfaced + fixed a systemic fail-open**: the original `<0`/`<=0`/`==0` validators let NaN/±inf through (NaN comparisons are False; inf passes `<=0`) on FillEvent/PredictionRow/OrderIntent and others — a shared `_FiniteContract(allow_inf_nan=False)` base now rejects non-finite for every float field (scalar + dict-valued) at parse time, fail-closed)

---

**Recommended order for return impact:** Phase 0 → 1 → 2 → 3 → 4, then 5/6/7 as you move to live.
Phases 2 and 3 are where returns actually improve; Phase 1 keeps them from being given back in drawdowns.
