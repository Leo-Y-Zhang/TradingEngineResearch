# Architecture

The architecture reference for this repository: what the engine is, the
non-negotiable safety rules it is built around, and where every responsibility
lives.

## What the engine is

A **systematic trading platform** that runs in three explicit
modes — **RESEARCH**, **PAPER/SHADOW**, and **LIVE** — and aims to maximise robust,
net-of-cost, risk-adjusted returns. The mathematical specification of the quant
core lives in `upgrade-spec/`.

## Golden rules (non-negotiable)

1. **Mode is explicit, never inferred.** Every order path checks `TRADING_MODE`.
   RESEARCH places **no** orders; PAPER/SHADOW runs the full pipeline but submits
   **zero** live orders; LIVE is the only mode that may reach a broker, and only
   for `risk_approved` orders. `data_contracts.normalize_mode()` is default-deny
   (an unknown mode is treated as LIVE and rejected).
2. **The pre-trade risk gate (engine STEP 10) fails CLOSED.** A kill switch or a
   KILL-level drawdown halts all new orders.
3. **Point-in-time safety.** All feature access goes through `data/feature_store.py`;
   no value may use information unavailable at `asof_time` (`asof_timestamp <= asof_time`).
4. **No silent exception swallowing** in risk, data, or execution modules. Failures
   are raised or logged loudly; the engine's integration layer degrades gracefully
   but always logs the degradation to the cycle audit.
5. **No live promotion without validation.** Sleeve-weight and model changes pass
   `PurgedWalkForwardSplitter` + `selection_rule()` first.

## Architecture — module map

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `core/config.py` | `EngineSettings` (pydantic-settings, `ENGINE_*` env / `.env`; LIVE requires `confirm_live` + `audit_log_path`) + `get_settings`, `engine_kwargs`, `make_broker`, `load_vault` |
| Secrets | `core/vault.py` | encrypted vault (Fernet + scrypt) → `secrets/vault.enc` + `vault_meta.json` (gitignored); operator CLI `python -m core.vault` |
| Contracts | `data/data_contracts.py` | 11 Pydantic v2 models (MarketBar, QuoteSnapshot, NewsEvent, InsiderEvent, FeatureRow, PredictionRow, OrderIntent, FillEvent, RiskEvent, PortfolioState, BrokerState) + `normalize_mode`, `position_divergence` |
| Features | `data/feature_store.py` | PIT-safe `get_features`, `feature_freshness_report`, train/serve parity |
| Research | `research/validation.py` | `PurgedWalkForwardSplitter`, `leakage_guard`, `selection_rule` |
| Research | `research/alpha_factory.py` | `SignalOutput`, factor evaluation, `apply_signal_health` |
| Vol/regime | `strategies/volatility_model.py` | GJR-GARCH + HAR-RV `fit`/`forecast_vol`, `vol_ratio_current`, `rmt_denoise_cov` |
| Regime | `core/regime_engine.py` | HMM `detect_with_probs`, `infer_execution_regime` |
| Crisis | `core/crisis_manager.py` | 7 detectors → composite `CrisisStatus` |
| Signals | `strategies/{momentum,mean_reversion,stat_arb,volatility_overlay,carry,sentiment,black_scholes}.py` | sleeve `generate_signals` → `list[SignalOutput]` |
| Microstructure | `core/engine/microstructure.py` | `compute_ofi`, `ofi_filter_gate` |
| NLP | `nlp/{finbert_scorer,sentiment_aggregator,sentiment_pipeline}.py` | news → per-symbol sentiment (FinBERT lazy-loaded, lexicon fallback) |
| ML | `core/ml_return_model.py` | 18-feature ensemble `predict` → 5-tuple, calibration, drift |
| ML | `core/meta_labeler.py` | `compute` → `TradeDecision` (admission + sizing) |
| Optimizer | `core/engine/optimizer.py` | Black-Litterman + exact CVaR `optimise_portfolio` |
| Risk | `core/risk_manager.py` | `check_pretrade` → `RiskSnapshot`, 10 kill switches, drawdown governors |
| Execution | `execution/{execution_engine,tca,capacity_model,slippage_model}.py` | order state machine, child scheduling, TCA, capacity |
| Learning | `learning/{performance_tracker,adaptive_weights}.py` | multi-horizon outcomes, validation-gated weights |
| Ops | `ops/{model_registry,monitoring}.py` | registry promote/rollback, 4-section snapshot + alerts |
| Ops | `ops/{persistence,state_store,sql_models}.py` | durable state (JSON default / SQLAlchemy) behind `save`/`restore` |
| Run-loop | `ops/run_loop.py` | `EngineService` (composition root: engine+broker+store from config), `LoopState` (durable book/counters), `run_forever` driver, service-level cycle lock |
| API | `ops/api.py` | `create_app(service)` — FastAPI observe/control surface (lazy `fastapi`, `app` extra) + `/metrics` |
| API security | `ops/api_security.py` | `SecurityLog` (per-request + `auth_failed` + `rate_limited` events), `attach_security_log_file` (full JSONL trail, size-capped, may roll) + `attach_security_alert_file` (WARNING+ only, 30-day time-based retention, repeat-aggregated so a flood cannot erase it), `cap_field` (every attacker-controlled log field), and `RateLimiter`/`RateLimitPolicy` (read vs control budgets, keyed on caller identity, bounded LRU). **`ops.api.api_uvicorn_kwargs()` (`proxy_headers=False`) is a security control — both entry points must serve uvicorn with it, see SEC-8.** |
| Observability | `ops/observability.py` | alert sinks (Logging/Jsonl/Composite/Null) + `MetricsRegistry` (Prometheus); built via `core.config.make_alert_sink` |
| **Engine** | **`core/engine/engine.py`** | **the integrated 13-step `_run_cycle()` pipeline** |

## The decision pipeline — `core/engine/engine.py`

`TradingEngine(mode, capital_gbp, broker, stale_threshold_seconds).run_cycle(CycleInputs)`
executes the 13 steps in exact order (they are never merged or reordered):

1. **Ingest & validate** — validate every input contract for the mode; freshness;
   LIVE stale data blocks new risk-taking (fail-closed).
2. **Build market state** — regime (`detect_with_probs`), crisis (`assess`),
   execution regime (`infer_execution_regime`).
3. **Volatility & risk forecasts** — `fit` → `forecast_vol` (1d/5d), `vol_ratio_current`, `rmt_denoise_cov`.
4. **Raw signals** — every sleeve's `generate_signals`.
5. **Signal-health filter** — `apply_signal_health` + OFI veto → per-symbol scores.
6. **Build features** — `feature_store.get_features` (mode-aware).
7. **ML prediction** — `ml_return_model.predict` (safe fallback if unfitted; background refit hook if `needs_refit`).
8. **Meta-label admission** — `tca.ex_ante_cost_model` → `meta_labeler.compute`.
9. **Portfolio optimization** — `optimise_portfolio` (9-key diagnostics).
10. **Pre-trade risk gate** — `risk_manager.check_pretrade`; kill switch halts; drawdown governor scales. **Fails closed.**
11. **Execution planning** — `schedule_order` (regime-aware; no market orders except URGENT_DERISK). RESEARCH plans none.
12. **Execute & TCA** — submit (LIVE only) / simulate (PAPER) / skip (RESEARCH); `ex_post_cost_analysis`; `update_cost_priors`.
13. **Post-trade learning** — `performance_tracker.evaluate_signal`, `monitoring.snapshot` (4 sections) + `alert_list`.

Each step appends one entry to `CycleResult.audit`; the result also carries the
regime/crisis state, predictions, decisions, optimizer output, risk snapshot,
orders, fills, monitoring snapshot, and `live_orders_submitted` (which **must** be
0 unless `mode == "LIVE"`).

## Conventions

- **Tests:** `python -m pytest tests/`. One `tests/test_phaseN.py` per phase.
- **Static gates:** `python -m mypy --ignore-missing-imports <files>` and
  `python -m ruff check <files>`. Public functions carry full type hints.
- **Singletons** are obtained via `get_*()` and cleared via `reset_*()` (tests
  reset them for isolation).
- **Determinism:** RESEARCH-mode RNG uses fixed seeds; the engine timestamps the
  audit from `inputs.asof_time`, never wall-clock.
- **Git:** never commit anything under `secrets/`; scope each commit to the
  paths it actually changes.

## Status

Phases 1–9 complete; ROADMAP Phases 0–6 complete. The platform is now *runnable and
deployable*: `ops/run_loop.py` (`EngineService` + `run_forever`) and `ops/api.py` wire
`TradingEngine` to a config-built broker + persistence on a schedule, observability
routes alerts + metrics, and `Dockerfile`/`docker-compose.yml`/`scripts/entrypoint.sh`
give a reproducible deploy (`docs/DEPLOY.md`). Phase 7 in progress (item 5 ✅:
contracts fail closed on non-finite via the `_FiniteContract` base). **Returns fix
(2026-06-19): long-biased baseline deployment — when ML admits nothing, STEP 9 deploys
the CAPM-equilibrium prior tilted by the validated sleeves (config `baseline_deploy_enabled`)
instead of sitting in cash. Honest real-data headline (recomputed 2026-06-27, net of cost
INCL. financing on levered notional): ann 18.4%, Sharpe 1.15, maxDD 17.1% vs EW benchmark
ann 18.0% / Sharpe 1.12 (8 large-caps, monthly, 2016–2024, 2× max leverage). The earlier
"Sharpe 1.32" was optimistic — it omitted borrow/financing cost and predated the
independent STEP-10 concentration enforcement; the true edge over passive is thin.**
mypy + ruff clean (run `python -m pytest tests/` for the live count). See `DECISIONS.md`.

Next: ROADMAP Phase 7 items 1/2/4 (TypedDicts, dedup helpers, stricter errors).
`broker/ibkr.py` exists; its live gateway paths are validated only in supervised
paper sessions.
