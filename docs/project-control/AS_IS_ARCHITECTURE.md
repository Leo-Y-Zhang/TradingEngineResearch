# TradingEngineResearch — As-Is Architecture & Incumbent Capability Inventory

| Field | Value |
|-------|-------|
| **Purpose** | As-is architecture + incumbent capability inventory + strengths/weaknesses. The incumbent is the **champion**; this is the baseline to preserve and improve against. (`§N` cites the internal improvement directive, not part of this repository.) |
| **Status** | PARTIAL — quant core mapped from code + `docs/ARCHITECTURE.md`; `backend/`/`frontend/` baseline app + live performance reconstruction still pending. |
| **Last verified** | 2026-06-26 · **Source** `docs/ARCHITECTURE.md`, `core/engine/engine.py`, module sources, audit findings. |

## 1. As-is runtime shape

**Type:** modular, single-process, synchronous **13-step decision cycle** over an injected `CycleInputs`, driven on a schedule by `ops/run_loop.EngineService` + `run_forever`, observable via `ops/api.py` (FastAPI). Three explicit modes — **RESEARCH** (no orders) · **PAPER/SHADOW** (full pipeline, zero live orders) · **LIVE** (disabled by default, double-armed). Deployable via Docker/compose. State persisted JSON (default) or SQLAlchemy.

**Order/data flow (single cycle, `core/engine/engine.py._run_cycle`):**
```
CycleInputs(asof, prices, news, betas, …)
 1 ingest+validate (PIT, fail-closed on LIVE stale)
 2 market state  → regime(HMM) + crisis(7 detectors) + exec-regime
 3 vol/risk      → GJR-GARCH/HAR-RV forecast, vol_ratio, RMT-denoise cov
 4 raw signals   → 7 sleeves (momentum, mean_rev, stat_arb, vol_overlay, carry, sentiment, black_scholes)
 5 signal health → apply_signal_health + OFI veto         [DEFECT SIGNALS-5: trusts unvalidated sleeves]
 6 features      → feature_store.get_features (PIT)        [DEFECT DATA-1: absent-feature impute bypass]
 7 ML predict    → ml_return_model (safe fallback if cold; bg refit)
 8 meta-label    → ex_ante cost gate → meta_labeler admission
 9 optimise      → Black-Litterman + exact-CVaR; baseline-deploy CAPM tilt when ML admits nothing
10 pre-trade RISK gate → check_pretrade; kill-switch/drawdown halt  [DEFECT RISK-1/ENGINE-1: limits not independently enforced]
11 exec planning → schedule_order (regime-aware child orders); RESEARCH plans none
12 execute + TCA → LIVE submit / PAPER simulate / RESEARCH skip; ex_post TCA; update cost priors
13 post-trade learning → performance_tracker, monitoring snapshot (4 sections) + alerts
→ CycleResult(audit[], regime, crisis, predictions, decisions, opt, risk snapshot, orders, fills, monitoring, live_orders_submitted)
```
Invariant: `live_orders_submitted == 0` unless `mode == "LIVE"`.

**Trust zones (as-is):** RESEARCH / PAPER / LIVE enforced in-process by mode checks + `normalize_mode` default-deny. *Not yet* separated by credentials/accounts/DBs/queues as directive §14 requires (Phase 2/8 gap).

## 2. Incumbent capability inventory

| Capability | Module(s) | Maturity | Notes |
|------------|-----------|----------|-------|
| Data contracts (11 Pydantic v2 models, finite-validated) | `data/data_contracts.py` | **Strong** | Fail-closed on non-finite; `position_divergence` reconciliation primitive. |
| PIT feature store | `data/feature_store.py` | Medium | PIT scan good; **no durable persistence (DATA-2)**; **absent-feature impute bypass (DATA-1)**. |
| Price/news ingestion | `data/price_ingestion.py` | Medium | yfinance; only ~6/18 features produced; survivorship in research scripts. |
| Research validation | `research/validation.py` | **Weak/blocking** | PurgedWalkForwardSplitter + selection_rule exist but **leakage_guard brick-walls promotion (SIGNALS-1)** + **calendar-vs-bizday purge leak (SIGNALS-2)**. |
| Factor/alpha factory | `research/alpha_factory.py` | Weak | decay/IC functions defective (SIGNALS-3/4); promotion loop dead in practice. |
| 7 signal sleeves | `strategies/*` | Medium | Individually plausible; **no sleeve has validated independent alpha** (proven on 30-name + broad universes). |
| Vol / regime / crisis | `strategies/volatility_model.py`, `core/regime_engine.py`, `core/crisis_manager.py` | **Strong** | GJR-GARCH/HAR-RV, HMM, 7-detector crisis composite; numerically guarded. *(Not yet audited this round.)* |
| ML return model + meta-labeler | `core/ml_return_model.py`, `core/meta_labeler.py` | Medium | Calibration/drift/abstention present; cold-starts to no-admission → engine runs on baseline beta. |
| Microstructure (OFI) + NLP sentiment | `core/engine/microstructure.py`, `nlp/*` | Medium | OFI partial-data robust; FinBERT lazy + lexicon fallback. |
| Portfolio optimizer | `core/engine/optimizer.py` | **Strong (central to returns)** | Black-Litterman + exact CVaR LP + vol-target/leverage scaler + baseline-deploy. **Not yet audited this round** — the leverage scaler (`:672`) is the main return lever. |
| Risk manager | `core/risk_manager.py` | Medium | Kill switches + drawdown governor + stress battery, BUT **STEP-10 doesn't independently enforce CVaR/vol/concentration/leverage (RISK-1)** and **kill switch not latched (RISK-6)**. Directive §16 wants this fully independent + authoritative. |
| Execution / TCA / capacity / slippage | `execution/*` | Medium | Order state machine, child scheduling, ex-post TCA. **Cost realism not yet audited** (attribution flags frictionless fills + no financing). |
| Brokers | `broker/{protocol,paper,ibkr}.py` | Medium | Paper deterministic; IBKR adapter fail-closed, LIVE-only submit. **Gateway paths only validated in supervised paper** — directive §15 state-machine invariants (idempotency, unknown-state, reconnect resync) need full coverage. |
| Learning | `learning/{performance_tracker,adaptive_weights}.py` | Medium | Multi-horizon outcomes; validation-gated weights. |
| Ops / registry / monitoring / persistence / run-loop / API / observability | `ops/*` | Medium | Runnable + deployable. **observability + run-loop serve_combined + deploy built inline, owed adversarial review** (ops audit not yet run). |
| Config + encrypted vault | `core/config.py`, `core/vault.py` | **Strong** | Default-deny mode, double-armed LIVE, Fernet+scrypt vault, crash-safe rotate. |
| Backtest harness | `backtesting/*` | Medium | PIT replay, metrics; **but headline scripts never wire the splitter (METH-2)** and costs are frictionless (METH-4). |
| Baseline web app | `backend/`, `frontend/`, `launcher/` | **UNKNOWN** | Baseline trees kept outside this repository; relationship to quant core unmapped. Directive §19 operator-UI is largely unbuilt against real APIs. |

## 3. Incumbent strengths (preserve — directive §2)
- Disciplined **mode separation + fail-closed defaults**; encrypted vault; finite-validated contracts.
- **Risk-management machinery** (vol-targeting, CVaR de-lever, crisis tightening, drawdown governor) — this, not alpha, earns the Sharpe edge over buy-and-hold and is the genuinely defensible component.
- Clean PIT discipline in the replay path; deterministic RESEARCH mode; broad unit-test coverage (~607 tests).
- Runnable/deployable end-to-end skeleton (run-loop + API + Docker).

## 4. Incumbent weaknesses (improve — gated by audit + three-layer verification)
- **No demonstrated independent alpha**; headline return is leverage × smart-beta on a survivor universe (METH-1..5).
- **Research-validation gates are broken** (SIGNALS-1/2/3/4) → no honest factor promotion is currently possible — this blocks all credible alpha work and is the #1 fix.
- **Risk gate is not an independent authoritative limit-enforcer** (RISK-1/ENGINE-1, RISK-6).
- **Feature persistence + absent-feature imputation** undermine reproducibility/safety (DATA-1/2).
- **Cost/financing realism** missing for levered tiers; **no real OOS/walk-forward** behind the headline.
- **No immutable order/cash/P&L ledger or reconciliation** vs broker/bank (directive §17) — required for paper/shadow readiness, largely absent.
- **Broker state-machine** §15 invariants and **operator UI** §19 not yet built to spec.

## 5. Not yet reconstructed (Phase 1 remaining)
- Independent **performance reconstruction** of the ~24% claim — note: the system has **no live trading history**, so "reconstruction" = honest re-backtest on a survivorship-free PIT universe with realistic costs + walk-forward (there are no order/fill/commission events to reconcile yet).
- Mapping `backend/`/`frontend/` baseline vs target architecture (§14/§19).
- As-is dependency graph, data-flow, trust-boundary, order-path sequence diagrams (§6).
