# DECISIONS.md — TradingEngineResearch Architecture Decision Record

Significant decisions made while building TradingEngineResearch v6, newest first. Each entry:
context → decision → rationale.

---

## 2026-06-17 — ROADMAP Phase 6: persistence layer (pluggable backend + why `migrations/` not `alembic/`)

**Pluggable backend, JSON default.** The learning-loop singletons already had
durable JSON snapshots (`ops/persistence.py`, Phase 4). Rather than replace that
working, deterministic path, the SQL layer is added *behind* it: `dump_payload` /
`load_payload` became the backend-agnostic core (shared deterministic retention),
and `ops/state_store.py` provides `JsonStateStore` (the default — tests/RESEARCH
stay byte-identical and dependency-free) and `SqlStateStore` (SQLAlchemy/SQLite,
Postgres-swappable via the URL, lazy-imported like the vault's `cryptography`).
`core.config.make_state_store` selects the backend. Rejected alternatives:
replacing JSON wholesale (throws away a tested/deterministic path, forces every
test onto a DB) and full per-entity normalisation (over-models Pydantic-validated
contracts — the declined "full productionization" scope).

**JSON columns, not over-normalised.** `ops/sql_models.py` stores one real row
per entity (model record, prediction, price, fill, outcome) but keeps the nested,
already-Pydantic-validated contracts in `JSON` columns. The rows are queryable
while the JSON column guarantees an exact round-trip with `dump_payload`. Save is
a single-transaction replace-state (delete-all + insert), correct because
retention bounds the data and the snapshot is the source of truth.

**Migrations dir is `migrations/`, not `alembic/` (deliberate).** TradingEngineResearch runs
with `pythonpath=["."]`, so a top-level `alembic/` directory would shadow the
installed `alembic` package for `import alembic` — breaking both the test
`importorskip` and the real `alembic` CLI when run from `TradingEngineResearch/`. The dir is
named `migrations/` (a common convention) and `alembic.ini` points `script_location`
at it. The initial migration binds to `ops.sql_models.Base.metadata` (the same
metadata `SqlStateStore` creates via `create_all`), so the migration can never
drift from the ORM schema.

## 2026-06-11 — ROADMAP Phase 6: central config + encrypted vault (and why vault code is NOT in `secrets/`)

**Vault location (spec deviation, deliberate).** The master prompt places the vault
code at `secrets/vault.py`. Importing it would require `secrets` to be a Python
package on the project root — which, under the repo's `pythonpath=["."]` (pytest)
and editable installs, would shadow the **stdlib `secrets` module** platform-wide.
Decision: the code lives in `core/vault.py`; `secrets/` remains exactly what the
spec generates into it — the data directory holding `vault.enc` + `vault_meta.json`
(both gitignored, verified).

**Vault format.** Fernet (AES-128-CBC + HMAC-SHA256, authenticated) over a JSON
secret map; key derived from a master passphrase via scrypt (default n=2**17, r=8,
p=1 — OWASP interactive minimum; params live in `vault_meta.json` so they are
tunable without a format change). A wrong passphrase and a tampered file are
indistinguishable by design and both raise `VaultAuthError`. Saves are atomic
(write-tmp-then-replace, the `ops/persistence.py` pattern). The passphrase comes
from `ENGINE_VAULT__PASSPHRASE` or an interactive prompt; secret values are
prompted via `getpass` — never argv, never logs. `cryptography` stays an optional
extra (`pip install tradingengineresearch[vault]`), lazy-imported like ib-insync.

**LIVE must be armed twice.** `EngineSettings` refuses `mode=LIVE` unless
`confirm_live=True` AND `audit_log_path` are also set: a lone `ENGINE_MODE=LIVE`
environment variable can never arm real-money trading, and a LIVE engine is never
unaudited. Broker-credential completeness is enforced in `make_broker` (the LIVE
account id must come from settings or the vault — no anonymous LIVE broker).

**Hardening from the adversarial review (same day).** A multi-lens security
review of the uncommitted config + vault surfaced (and an independent skeptic
confirmed) several real defects, all fixed before the commit:

- **`rotate()` is now crash-safe across the two-file boundary.** Re-keying
  changes two coupled files (new salt in `vault_meta.json` + ciphertext
  re-encrypted under the new key in `vault.enc`). Previously a crash *between*
  those writes left a new-salt meta with old-key ciphertext that NEITHER the old
  nor the new passphrase could open — a permanently bricked vault. Now `rotate`
  first snapshots the known-good pair to `*.bak`; if it dies mid-rotate, the next
  `open()` rolls back to the pre-rotation state (the rotation simply did not take
  effect). The naive "write enc before meta" ordering does **not** fix this — the
  new ciphertext requires the new salt that lives only in the new meta — so a
  backup/rollback (or a single-file format) is the only correct approach.
- **KDF parameters are bounds-checked before use.** `vault_meta.json` is
  plaintext and not covered by Fernet's HMAC, so a tampered/corrupt `n` could
  drive scrypt to an unbounded allocation (OOM/DoS at `open`) or be a
  non-power-of-two that escaped as a raw `ValueError`. `_validate_kdf_params`
  now requires `n` a power of two in `[2**12, 2**22]`, `r∈[1,32]`, `p∈[1,16]`,
  salt = 16 bytes, raising `VaultError` (fail closed) before any derivation.
- **LIVE is gated twice, by value not by trust.** `validate_assignment=True`
  makes a post-construction `settings.mode = "LIVE"` re-run the fail-closed
  validator (a lone mode flip cannot arm money), and `make_broker` independently
  re-asserts `confirm_live`/`audit_log_path` at the money boundary so even an
  unvalidated (e.g. `model_construct`'d) settings object cannot build a
  real-money broker.
- **`secrets/` is default-deny in `.gitignore`.** The previous exact-filename
  allowlist (`secrets/vault.enc`, `secrets/vault_meta.json`) left transient
  `.tmp`/`.bak` siblings and any stray file committable. Now `secrets/*` is
  ignored (only `secrets/.gitkeep` tracked), `.env.*`/`*.env` variants are
  ignored, and a root-anchored `.gitignore` covers the repo root as a safety net.

## 2026-06-10 — ROADMAP Phase 4: durability + challenger lifecycle + cycle audit

**Cycle audit (spec deviation, deliberate).** Spec STEP 13 says "append cycle summary
to DECISIONS.md audit trail" — but DECISIONS.md is the human architecture record, and
per-cycle machine rows would bury it; unconditional disk writes would also break
backtest determinism and hammer I/O across hundreds of replay cycles. Decision: a
dedicated append-only trail via `ops/audit_log.py`, **opt-in** through the engine's
`audit_log_path` (the PAPER/LIVE run-loop enables it; replays/tests stay I/O-free).
Rows are stamped from `asof_time`, never wall clock.

**Persistence.** `ops/persistence.py` snapshots the registry + tracker singletons to
one JSON file with **deterministic retention**: cutoff = newest data timestamp −
retention_days (default 90), never the wall clock, so replayed saves prune
identically. The Phase 6 SQLAlchemy layer can later replace the backend behind the
same call sites.

**Challenger lifecycle.** `promote()` now refuses any record whose
`validation_metrics` fail `selection_rule()` (golden rule 5 enforced at the registry,
not just by convention). `rollback(reason, expect_current=...)` makes operator retries
idempotent (a second rollback for the same incident no-ops instead of double-popping
history). STEP 13 surfaces a validated shadow as an INFO alert for a HUMAN decision —
promotion is never automatic.

## 2026-06-10 — ROADMAP Phase 4: book reconciliation (delta sizing, explicit exits, achieved weights)

**Context.** Three multi-cycle accounting holes: (1) `schedule_order`'s fallback sized
child orders from `|target_weight| × capital` — holding 30% with a 10% target re-traded
the whole 10% position instead of the 20% delta, over-trading massively on every
rebalance; (2) STEP 11 iterated only the target book, so a held name dropped by the
optimizer never received a SELL — it would linger forever in LIVE while paper books
silently exited it "for free"; (3) nothing reconciled what was actually filled — the
backtester booked the *target* as held, so partial fills corrupted every later delta.

**Decision.** STEP 11 trades the UNION of target and held books (dropped names get an
explicit SELL-to-zero) and passes an explicit delta-sized `target_qty` to the
scheduler. STEP 12 reconciles `CycleResult.achieved_weights = held + signed fill
notional / capital` (dust < 1bp of capital dropped so spread residuals never churn
exits). The backtester carries the achieved book in PAPER/LIVE and falls back to the
target book in RESEARCH (no orders are planned there — the replay book is
hypothetical by construction) or when a result lacks the field (back-compat).

## 2026-06-10 — ROADMAP Phase 4: real refit loop (training buffer + mode-aware execution)

**Context.** `needs_refit` only logged "deferring" — the model had no `refit` hook, no
training data, and the engine never recorded prices into the performance tracker, so no
prediction horizon could ever elapse. The learning loop was open at three points.

**Decision.**
1. The model owns a **rolling training buffer** (`record_training_example`, cap 2000):
   the tracker feeds it the **1-day horizon** outcome only (the model's prediction
   target — mixing horizons in one regression would corrupt the label definition),
   paired with the features captured at prediction time (STEP 13 threads them in).
2. `refit()` retrains from the buffer (min 20 rows; failures keep serving the previous
   model). An UNFITTED model bootstraps its first fit at 40 resolved outcomes
   (`ready_for_initial_fit`) — without this the loop could never start, since
   `needs_refit` requires `_fitted`.
3. **Mode-aware execution:** LIVE refits on a background daemon thread (a refit must
   never block a live cycle — spec STEP 7); RESEARCH and PAPER refit **synchronously**,
   because backtests/replays must stay deterministic (golden rule) and a background
   thread mutating the model mid-replay would make results timing-dependent.

**Also fixed (latent).** `get_features` returns an `asof_timestamp` column;
`engine._feature_row`'s `float(v)` crashed on it — i.e. ANY cycle with real rows in the
feature store would have crashed STEP 7. Engine tests had only ever run with an empty
store. `_feature_row` now keeps numeric values only.

## 2026-06-10 — ROADMAP Phase 3: sentiment wired end-to-end (live-only alpha)

**Context.** `sentiment_score` was hardcoded 0.0; FinBERT scorer + aggregator existed
but nothing produced news or consumed their output.

**Decision.** News items are caller-supplied (`CycleInputs.news_items`); sentiment is
computed once per cycle and used twice — a 6th fast-decay event sleeve (deadband 0.15)
and the real `sentiment_score` ML feature. Free feeds (yfinance) serve only *recent*
headlines, so this alpha is **live/paper-only**: it cannot be backfilled into the
historical backtest, and no-news cycles are bit-identical to the pre-wiring engine
(determinism preserved). PIT safety: undated or future-dated news items are skipped at
ingestion rather than given a fabricated age.

## 2026-06-10 — ROADMAP Phase 3: carry sleeve (returns-first; unlock the data)

**Context.** The carry sleeve was initially deferred as "data-blocked" (needs dividends;
the sleeve contract passes only prices). The user then set **returns as the top
priority** — so the right move was to *unlock the data*, not defer the return source.

**Decisions.**
- **Equity carry = dividend yield.** The carry of holding a stock at an unchanged price
  is its dividend yield, so the sleeve is a cross-sectional trailing-dividend-yield tilt
  (`raw = tanh(0.7·z)`, z = standardised yield) — long high-yield, short below-average.
  A real, well-known equity return factor, computable from dividends + prices.
- **Fetch the missing data via yfinance** (already authorised): added dividend ingestion
  to `price_ingestion` (`fetch_dividends`/`load_dividends`/`trailing_dividend_yields`,
  PIT-safe) + a committed dividend fixture. Real cross-section (3 payers, 2 non-payers).
- **Thread yields with a minimal, localised change**, not a contract overhaul: added
  `CycleInputs.dividend_yields` and a STEP-4 *special-case* for carry (it needs the
  yields, every other sleeve still takes `prices` only). Avoids forcing `**kwargs` onto
  the other four sleeves.
- **Alpha is a hypothesis**: implemented + integrated, to be validated through the
  backtest harness and gated by the validation framework before being trusted.

**Rationale.** Returns lead: rather than defer a genuine return factor because the
price-only pipeline lacked dividends, fetch the dividends (a small, authorised data
expansion) and wire the sleeve cleanly. Yields look right (JPM 2.5%, MSFT 0.8%, AAPL
0.5%, GOOG/AMZN 0%), giving real cross-sectional carry.

## 2026-06-10 — ROADMAP Phase 3: volatility-overlay sleeve

**Context.** The first speculative new-alpha sleeve. The universe is equity/price-only
(real ingestion is price-derived), so the sleeve must be computable from prices.

**Decisions.**
- **Risk-off / low-vol overlay as the alpha.** `raw = -tanh(k·(recent_vol/baseline_vol − 1))`:
  defensive (SELL/reduce) when an asset's realised vol expands above its baseline,
  constructive (BUY) when it compresses. This harvests the volatility-timing / low-vol
  premium and de-risks names whose vol is spiking — a recognised, price-derivable,
  PIT-safe signal that targets *risk-adjusted* return, chosen over an equity "carry"
  proxy (which needs dividend/fundamental data the price feed doesn't provide).
- **Follow the existing sleeve contract exactly** (`generate_signals(prices, asof) →
  list[SignalOutput]`, deadband → direction, `_asof`/`_flat` helpers), wired as a 4th
  entry in the engine's `_SLEEVES`. The existing engine + harness tests now exercise it,
  giving integration coverage for free.
- **Treat the alpha as unproven.** Per the end-goal, new alpha is a research bet: the
  sleeve is implemented and integrated, but its edge must be *validated through the
  backtest harness* (and ultimately gated by the factor-promotion / signal-health
  framework) before being trusted — not assumed from plausibility.

**Rationale.** Add an orthogonal, defensible return/​risk source that the real
price-ingestion path can actually compute, using the proven sleeve plumbing, while being
honest that its profitability is a hypothesis to test, not a given.

## 2026-06-10 — ROADMAP Phase 3: validation-driven signal health

**Context.** STEP 5 weighted each sleeve by this-cycle mean confidence and passed
`validation=None` to `apply_signal_health`, so the validated-quality gate (which the
function already supports) was never applied — at odds with golden rule #5.

**Decisions.**
- **Honor a persistent per-sleeve `ValidationResult`, registered out-of-band.** A small
  registry in `alpha_factory` (`register/get/reset_sleeve_validation`, same convention as
  the live factor library) holds each sleeve's validated quality. STEP 5 looks it up and,
  when present, uses the validated `stability_score` and passes the result so a failed
  `selection_rule` disables the sleeve.
- **Backward-compatible soft default.** Un-validated sleeves keep the old this-cycle
  behaviour rather than being hard-disabled (which would zero the pipeline) — sleeves get
  gated as they are validated, with no regression today. The gate logic itself is the
  existing `apply_signal_health`/`selection_rule`; this only feeds it.
- **Warning-free as a robustness bar.** While here, fixed a benign statsmodels
  divide-by-zero (`fit_har_rv`'s `model.rsquared` on a zero-variance target, read after
  the `np.errstate` guard and from a background refit thread): the lazy `rsquared` is now
  read inside the guard and sanitised. The full suite emits zero warnings.

**Rationale.** The validation framework existed end-to-end except for the last hop into
the per-cycle signal weighting; wiring it (without forcing a regression on un-validated
sleeves) makes signal health quality-driven, which is the point of validation.

## 2026-06-10 — ROADMAP Phase 2 (item 3): real feature ingestion (Phase 2 complete)

**Context.** Replace synthetic feature data with real, price-derived features through the
PIT-safe feature store, with a committed offline fixture. Full design in
`docs/specs/2026-06-10-feature-ingestion-design.md`.

**Decisions.**
- **Upgrade yfinance, don't switch sources.** The pinned `yfinance 0.2.44` returns empty
  (Yahoo changed their API); `1.4.1` fetches cleanly. Upgraded + repinned
  (`constraints.txt`, `ingestion` extra). The network/fetch were verified live before
  designing; the direct Yahoo chart API also worked as a fallback but was not needed.
- **Ingest only the price-derivable subset; leave the rest imputed.** Most model features
  need sources yfinance does not provide (insider→SEC, sentiment→news, OFI→L2,
  earnings→calendar, engine_expected_return→a model). "Real ingestion" computes the six
  price-derivable features and registers them; the feature store's existing deterministic
  imputation continues to fill the rest. Honest about what real price data can and cannot
  supply, rather than fabricating the unobtainable features.
- **All feature windows are trailing → PIT-safe by construction**, and a property test
  asserts it (a feature at date t is identical computed on full vs t-truncated history).
- **Committed offline fixture; live fetch out of the suite.** `tests/fixtures/prices_sample.csv`
  (5 liquid symbols × 2 years) is fetched once and committed; tests run on it offline
  (deterministic, no network in CI). `fetch_prices` is `# pragma: no cover` (network I/O,
  validated only when recording the fixture). A small real-price fixture stays within the
  proprietary repo and is not redistributed.

**Rationale.** Make `get_features` return genuinely real values for everything price data
can support, with a reproducible offline test path — without overreaching into data
sources that aren't wired, and without ever coupling the test suite to the network.

## 2026-06-10 — ROADMAP Phase 2 (item 2): factor-promotion pipeline

**Context.** `evaluate_factor` and `promote_factor` existed and worked, but nothing
connected them, and the module-level `_LIVE_FACTOR_MATRIX`/`_LIVE_FACTOR_NAMES` were
declared and never updated — the research→live factor loop was open.

**Decisions.**
- **Orchestrate, don't re-implement.** Added a thin `promote_candidates()` that runs the
  existing evaluate→gate→promote pieces in sequence and *maintains the live library*
  (appending each promoted factor's column so later candidates are gated for
  correlation/cluster-diversity against the already-promoted set). The gate logic itself
  (`selection_rule`, the 0.80 correlation / 0.50 cluster-distance thresholds) is unchanged.
- **Make the dead globals a real library via `get_*`/`reset_*` accessors**, matching the
  codebase's singleton convention — rather than introducing a new class. Minimal churn,
  inspectable, test-resettable.
- **Research-mode scope.** Factor promotion is an offline research activity, not a
  trading-cycle step (the per-cycle signal gate is the existing STEP-5
  `apply_signal_health`). The library accumulates within one consistent time index; a
  differently-sized universe starts a fresh library (`promote_factor` already guards the
  length mismatch) — documented, not engineered around (YAGNI).

**Rationale.** The validation + diversity gates were already correct and tested; the only
missing piece was the loop that uses them and remembers what was promoted. Keeping it a
thin orchestrator over the proven primitives is lower-risk than a new subsystem.

## 2026-06-09 — ROADMAP Phase 2 (items 4, 5): determinism + property tests (2 bugs fixed)

**Context.** Hardening the measurement foundation: a reproducibility guarantee for
RESEARCH mode and property-based (`hypothesis`) invariants for the numeric kernels
the spec flags as correctness-critical (OFI, vol_ratio, CVaR, TCA).

**Decisions.**
- **Assert only provable invariants; reject the plausible-but-false.** Property tests
  encode invariants that hold for ALL valid in-domain inputs. Two tempting invariants
  were deliberately NOT asserted because they flag correct code: (a) "CVaR ≥ 0" on the
  exact-LP path — the Rockafellar-Uryasev CVaR is a *signed* tail expectation and is
  genuinely negative when the worst (1−c) outcomes are net gains; (b) "exact-LP CVaR ≥
  Gaussian CVaR" — `_gaussian_cvar` uses the 1.65 VaR multiplier (not the 2.063 ES
  multiplier) and drops the mean, so the LP falls below it under positive drift. We
  assert positive homogeneity, Gaussian-path non-negativity, and the CF≥Gaussian floor
  (cornish_fisher branch only) instead.
- **Verify-then-fix: two latent bugs the property tests surfaced were fixed, not
  `assume()`-d around.** (1) `compute_ofi` propagated NaN from a bad tick
  (`np.clip(nan)=nan`); added a `_bounded` clip-and-nan-guard so the contract ("[-1,1],
  never an exception") holds. (2) `fit_har_rv` raised `ValueError: expected 4, got 3`
  on constant-magnitude returns: collinear realised-variance regressors made
  `sm.add_constant` skip the intercept, so OLS returned 3 params. Fixed at the root with
  `has_constant="add"` (always 4 params; pinv handles the rank deficiency), plus a
  defensive try/except → 1.0 in `vol_ratio_current` (matching its "degenerate → 1.0"
  contract) so the per-cycle risk kernel can never crash the optimiser.
- **Numerical-resolvability scoping, not weakening.** The homogeneity property is bounded
  away from sub-1e-6 scales where both sides fall below the HiGHS LP solver's absolute
  resolution (~1e-9); the exact `s=0` edge is covered by a separate deterministic test.
- **Determinism via clean-slate reset.** A RESEARCH cycle is bit-reproducible after
  resetting the stateful singletons (the backtester's `_reset_engine_state` was corrected
  to the full canonical set incl. `risk_manager`/`tca`/`model_registry`).

**Rationale.** Property tests are only as good as the truth of their invariants — a
false invariant is worse than none (it erodes trust and flags good code). Encoding the
exact, justified invariants both locks in correctness and, here, exposed two real
robustness holes in kernels the engine runs every cycle.

## 2026-06-09 — ROADMAP Phase 2 (item 1): backtest / walk-forward harness

**Context.** "Can't improve what you can't measure." Phase 2's first sub-project is a
harness that makes net-of-cost, risk-adjusted returns measurable. Full design in
`docs/specs/2026-06-09-backtest-harness-design.md`.

**Decisions.**
- **PAPER-mode engine replay**, not an optimiser-only or vectorised backtest. The
  harness steps the *real* `TradingEngine` over the price history so the measured
  returns reflect the actual 13-step pipeline (regime, crisis tightening, optimiser +
  CVaR enforcement, the fail-closed risk gate + drawdown governor, TCA). A parallel
  vectorised implementation would drift from the code that trades.
- **`CycleResult.target_weights` as the book source.** Added a backward-compatible
  (default-valued) field carrying the risk-approved book, rather than reconstructing
  positions from `order_intents` — which omit exited and untraded-hold names and would
  silently mis-state the book. On a blocked cycle the harness carries the current book
  (no new risk), matching the gate's "halt new orders" semantics.
- **Determinism by construction.** `run()` resets the engine's stateful singletons
  (view tracker, ML model, regime/crisis/perf/registry) and seeds RNG; timestamps come
  from the data, never wall-clock. Two runs on the same input are bit-identical — this
  also seeds the separate "determinism test" Phase-2 item.
- **Net-of-cost = realised turnover × a bps rate** (default 10 bps/unit turnover),
  charged at each rebalance. A deliberately simple, transparent cost model that can be
  swapped for full per-order TCA later without changing the harness interface.
- **Pure `metrics.py`.** Performance functions are pure and side-effect-free (degenerate
  inputs → 0.0, never NaN/inf), so they are the natural target for the upcoming
  property-based-tests item and are independently unit-tested with known answers.
- **Scope (YAGNI).** No plotting, no multi-asset-class roll, no parameter search, no
  parallelism. Rebalance = the last actual trading day per `rebalance` period after a
  warmup. Runs on synthetic prices now; swaps to a recorded yfinance fixture when that
  sub-project lands.

**Rationale.** A faithful, deterministic, net-of-cost replay of the real engine on
purged walk-forward splits is the measurement foundation every later alpha/learning
change will be judged against. Keeping the cost model and data source simple now (with
clean seams) avoids over-building before there is real data to measure on.

## 2026-06-09 — ROADMAP Phase 1: risk & sizing correctness (verify-then-fix)

**Context.** An 8-subsystem audit flagged 8 risk/sizing correctness gaps. Each was
re-verified against the *current* code before any fix (7 confirmed real, 1 —
PSD — confirmed as a latent invariant gap, not an active bug, since the 3-way
blend of PSD components is provably PSD today). Every fix was TDD red→green.

**Decisions.**
- **CVaR limit is enforced, not penalised.** ITEM 1 implements the upgrade-spec's
  "iteratively enforced before final allocation" as a post-solve iterative
  de-lever (`_enforce_cvar_limit`) rather than the master-prompt 14.5 `λ_cvar`
  objective penalty (which needs an LP solve per SLSQP iteration — slow and
  convergence-fragile). NOTE: because the vol target already de-levers any book
  whose annualised vol exceeds the target, CVaR structurally cannot exceed the
  limit after vol-targeting for realistic inputs — so enforcement rarely *binds*.
  Its value is converting a silent flag into a hard guarantee on the returned
  weights (defence-in-depth; it WILL bind on the no-history / degenerate paths).
- **Crisis tightening uses the upgrade-spec P4 continuous bands, not the master
  prompt's binary `crisis_mode`.** The two specs conflict; the ROADMAP item says
  "use the continuous severity", so P4 wins. To avoid a protection *regression*
  (P4's Elevated/Defensive bands are looser than the legacy binary crisis floor),
  the optimiser takes the *tighter* of the P4 scalar and the legacy crisis floor
  whenever `crisis_mode` is set — protection is monotone, never looser than before.
  The scalar bands key off the raw severity at P4 thresholds (0.35/0.60/0.80),
  deliberately distinct from the `CrisisLevel` thresholds (0.20/0.50/0.75).
  `crisis_severity` is an additive optional arg (default None → no-op), so the
  master-prompt-shaped boolean callers and tests are unaffected.
- **Calibration is out-of-sample.** ITEM 4 fits the isotonic calibrators on a
  purged, time-ordered tail fold (30%, 1-row embargo) and scores Brier/ECE there.
  In-sample calibration reported ~0.004 Brier on pure noise; honest OOS is ~0.25.
  When there are too few rows for an honest split the metrics are `None` (not a
  fabricated optimistic number); the point return/vol models stay full-sample.
- **The cross-sectional prior is the universe grand mean.** ITEM 5 anchors the
  adaptive-shrinkage prior at the mean of the *raw* ensemble views across the
  cycle's universe (James-Stein), computed in `predict_batch` and wired into
  engine STEP 7. A single-symbol universe ends up un-shrunk (prior == its own raw
  view) — an accepted, sensible behaviour change.
- **`size_multiplier` is applied once, before the risk gate.** ITEM 6 scales the
  optimiser's weights by the meta-label `size_multiplier` at STEP 9 (so STEP 10
  evaluates the conviction-sized book) and removes the old post-gate multiply at
  STEP 11. The architecture has no "initial target weights" prior to optimisation,
  so scaling the optimiser output before the gate is the faithful realisation of
  spec STEP 8. `size_multiplier ∈ [0, 1]`, so it only de-levers and cannot breach
  the enforced CVaR limit. This supersedes the Phase-9 note (DECISIONS 2026-06-08)
  that deliberately placed it at STEP 11.
- **PSD projection / CF floor are silent, shape-preserving guards.** `_project_psd`
  clips eigenvalues to a small *positive* floor (PD, strictly convex QP), applied
  after the trace-rescale; the CF CVaR `max(cf, gaussian)` floor subsumes the old
  `abs()`. Neither changes the public return shapes.

**Rationale.** Make the risk math actually bind on outputs (CVaR, sizing, crisis
tightening), make the learning signals honest (OOS calibration, non-zero prior),
and guarantee the numerical invariants the optimiser assumes (PSD, CVaR ≥ Gaussian)
— all while keeping every change additive/backward-compatible so the 327-test
baseline stayed green (now 342).

## 2026-06-09 — ROADMAP Phase 0: packaging & the `pytest` collision fix

**Context.** The quant core (Phases 1–9) was importable only by running tests from
the project root with `python -m pytest tests/`; bare `pytest` collided with a
second `tests` package (the extracted baseline at
`codebase/extracted/TradingEngineResearch-master/backend/tests`). There was no `pyproject.toml`,
no dependency lockfile, and no CI/lint gates.

**Decision.**
- **Flat-package packaging, no umbrella refactor.** `pyproject.toml` ships the
  existing flat top-level packages (`core`, `data`, …) via `setuptools.packages.find`
  with explicit include/exclude. We did **not** rename everything under a `tradingengineresearch.`
  namespace — that touches every intra-project import and is deferred (it belongs
  with the Phase-7 repo split).
- **Dependencies modelled as required-core + optional extras.** `[project].dependencies`
  is only what the 327-test quant core needs (numpy/pandas/scipy/scikit-learn/pydantic/
  arch/statsmodels/hmmlearn, with ABI upper bounds). Everything else is an extra
  (`nlp`, `boost`, `shrinkage`, `app`, `brokers`, `ingestion`, `vault`, `dev`, `all`),
  including the two undeclared lazy fallbacks found in the audit (`lightgbm`→`boost`,
  `nlshrink`→`shrinkage`). `constraints.txt` pins the exact validated versions.
- **Collision fixed with `--import-mode=importlib`** (in `addopts`) + `testpaths=["tests"]`
  + `pythonpath=["."]`, so both bare `pytest` and `python -m pytest` collect only the
  quant-core suite. `import_mode`/`importmode` are *not* valid ini keys (they warn).
- **Coverage floor 80%** (measured 85% with branch coverage); `--cov` is kept out of
  default `addopts` so bare `pytest` stays cov-optional.
- **CI/pre-commit live at the repository root** (`.github/workflows/ci.yml`,
  `.pre-commit-config.yaml`), self-contained. (When this entry was written the
  project lived inside a private multi-project tree and CI was dormant; CI now
  runs on every push.) pre-commit uses **local** hooks invoking the pinned
  ruff/mypy.

**Rationale.** Make the engine installable and the test/lint/type gates reproducible
*without* a risky import-namespace rewrite.
Strict required/optional separation keeps a core install lean and honestly reflects
that the baseline app, brokers, and ingestion are not yet wired.

## 2026-06-08 — Phase 9: the 13-step engine is a thin integration layer

**Context.** `core/engine/engine.py` must wire Phases 1–8 into the Part-20
pipeline without re-implementing any quant logic.

**Decision.** The engine owns no quant logic. Each of the 13 steps delegates to
the module built for it and threads the result forward via a `CycleInputs` →
`CycleResult` pair. Steps are explicit, ordered methods (`_step1…_step13`) and are
never merged or reordered. The audit trail records one entry per step.

**Rationale.** Keeps the quant modules independently testable and the pipeline
auditable; matches the spec's "documented internal boundaries" requirement and
lets STEP 1 / STEP 10 / STEP 13 be unit-tested in isolation.

## 2026-06-08 — Mode discipline is enforced at the execution boundary

**Decision.** RESEARCH plans no orders at all (STEP 11 short-circuits); PAPER
simulates fills locally and submits **zero** live orders; LIVE submits only via an
injected `broker` and only for `risk_approved` orders. A LIVE cycle with no broker
submits nothing (fail-safe). `CycleResult.live_orders_submitted` is asserted 0
outside LIVE.

**Rationale.** "PAPER/SHADOW produces zero live orders" is a hard compliance
requirement; making the broker an injected dependency keeps LIVE submission
explicit and test-isolated.

## 2026-06-08 — The pre-trade risk gate fails CLOSED

**Decision.** STEP 10 treats any kill switch, KILL-level drawdown, or an exception
inside the gate itself as a halt: no new orders, weights dropped, `RISK_EVENT RED`
logged. The drawdown governor scales surviving exposure (SOFT 0.80 / MEDIUM 0.70 /
HARD 0.40 / KILL 0.0).

**Rationale.** A risk gate that fails open is worse than no gate. Errors must stop
trading, not be swallowed.

## 2026-06-08 — Phase 7/8 reconciliation: two bugs fixed before committing

Phases 7 & 8 were found already-implemented-but-uncommitted from a prior working pass.
A 6-dimension adversarial audit against the acceptance criteria found two real
bugs (green tests had not caught them because the same pass wrote the tests):

- **CRITICAL — adaptive weights skipped mandatory validation.**
  `adaptive_weights.propose_and_validate` gated `PurgedWalkForwardSplitter` behind
  an optional `timestamps` argument, so the default path applied weight changes on
  `selection_rule()` alone. **Decision:** the walk-forward window is now required;
  absent/empty-fold windows reject the change and retain existing weights. The
  split is unconditional before any accepted change.

- **MAJOR — TCA sign-flipped SELL costs.** `tca.ex_post_cost_analysis` matched
  decisions to fills by `order_id`, but `OrderIntent` has no `order_id` — only
  `FillEvent` does. The join always missed, defaulting the side sign to BUY, so
  every SELL's realised cost was reported as a gain (corrupting `passive_fill_ratio`
  and the `observed_k1/k2` fed into the cost priors). **Decision:** match by
  `symbol` (the field both carry); added a SELL-side regression test.

## 2026-06-08 — feature_store skips PIT-unsafe rows

**Context.** `FeatureRow.asof_timestamp` is `Optional`; the PIT scan dereferenced
it without a guard (a latent crash and a mypy error surfaced once the engine
imported the module under mypy).

**Decision.** A candidate with no `asof_timestamp` is skipped (`continue`) in both
`get_features` and `feature_freshness_report` — it can never be PIT-eligible.

## 2026-06-08 — NLP is optional and lazy in the engine

**Decision.** The engine does not import `nlp/*` at module load and runs the three
core sleeves (momentum, mean-reversion, stat-arb) directly. Sentiment is an
optional, lazily-attempted overlay.

**Rationale.** Keeps the engine importable and testable without the transformers
stack; FinBERT already lazy-loads with an offline lexicon fallback.

## 2026-06-08 — Phase 9 engine: two bugs caught by adversarial review, fixed

A 3-lens adversarial review (mode-safety / step-fidelity / robustness) of the
uncommitted engine confirmed the mode gating and fail-closed risk gate are sound,
and found two real bugs the green tests had missed:

- **STEP 8 cost-model arg mismatch.** `tca.ex_ante_cost_model`'s 5th parameter is
  `volatility`, but the engine passed the share **price** there, inflating the
  modelled impact and skewing meta-label admission. **Fix:** pass `volatility`
  (defaulting to the model's predicted sigma) and bind all args by keyword.
- **Cross-cycle state corruption.** The engine overwrote an internal
  `_prev_weights` with target weights even on blocked/RESEARCH cycles (phantom
  positions) and never read `inputs.current_weights`. **Fix:** the engine is now
  stateless w.r.t. positions — the held book is supplied per cycle via
  `inputs.current_weights`, used for both the optimizer's `w_prev` and STEP 11
  deltas. The meta-label `size_multiplier` now actually scales the order target.

Both are locked by regression tests (cost price-invariance; deltas vs the current
book; cross-cycle statelessness).

---

## Earlier (Phases 1–6) — key standing decisions

- **2026-06-05 — Build from Version 4, follow Version 5's 9-phase plan.** Version 4
  is the only fully self-contained spec; Versions 1–3 are historical patches.
- **Project renamed (2026-06-05).** The original working name was crowded in
  fintech; a collision-free replacement was vetted and adopted. (The project has
  been renamed again since; earlier names are not used in this repository.)
- **Exact CVaR via `scipy.optimize.linprog(method="highs")`** when `T >= 30`, with a
  Gaussian/Cornish-Fisher fallback when `T < 30`.
- **Regime-aware Black-Litterman τ** (0.05 / 0.10 / 0.02 by regime; crisis override
  `min(τ, 0.02)`), ML view confidence decayed by realised vol.
- **Three-way covariance blend** (Ledoit-Wolf / stress / EWMA) with RMT denoising;
  crisis-mode blend weights differ from normal.
- **Volatility:** GJR-GARCH(1,1,1) Student-t via `arch`, HAR-RV OLS via
  `statsmodels`, ensemble-switched by sample size, with a rolling-std fallback on
  non-convergence.
- **ML safe fallback** `(0.0, 0.15, 0.50, 0.10, 0.0)` whenever the model is not
  ready — the pipeline never crashes on an unfitted model (it simply admits no
  trades).
- **nonlinear Ledoit-Wolf** uses `nlshrink` if installed, else linear LW; the
  primary covariance path is the 3-way blend regardless.
