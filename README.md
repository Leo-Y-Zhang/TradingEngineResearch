# TradingEngineResearch

<!-- Coverage is enforced in CI (80% floor) but no badge service is wired, so no
     coverage badge is shown. -->
[![CI](https://github.com/GreenPandaTech/TradingEngineResearch/actions/workflows/ci.yml/badge.svg)](https://github.com/GreenPandaTech/TradingEngineResearch/actions/workflows/ci.yml)

TradingEngineResearch is a systematic-trading research platform built to answer one question
honestly: can a disciplined retail researcher find deployable equity alpha in public
data? Over a pre-registered, month-long programme it searched five data sources —
prices and volumes, Fama–French loadings, SEC EDGAR fundamentals, Form 4 insider
filings, and paid survivorship-free fundamentals — through a default-deny validation
gate. The answer was no: nine studies, every one banked NOT-DEPLOYABLE. What makes the
project unusual is what that answer took: a validation pipeline designed so that its
own operator cannot fool it, and a fail-closed engine that, with no validated signal
to trade, ran a live-mode paper session flawlessly and correctly traded nothing.

## Read this first

The document this repository exists to support is the research write-up:
**[docs/RESEARCH_WRITEUP.md](docs/RESEARCH_WRITEUP.md)** — *A Pre-Registered Search
for Equity Alpha on a Retail Budget*. It describes the platform, the validation gate,
all nine studies, the two measurement artefacts the programme caught in its own best
results, and why a rigorous negative result is worth publishing. The primary study
artefacts (pre-registrations, errata, result files, dev logs) live in
[research/medallion_style_alpha_search/](research/medallion_style_alpha_search/).

## Headline numbers

| | |
|---|---|
| Automated tests | 1,813 passing (unit, property-based and stateful; Hypothesis fuzzing of order-state invariants) |
| CI | ruff + mypy + pytest with an 80% coverage floor (measured baseline 88%) + gitleaks secret scan |
| Engine baseline | 18.4% annualised, Sharpe 1.15, max drawdown 17.1%, net of costs including financing (8 large caps, monthly, 2016–2024) |
| Baseline replication | rerun on a second, survivorship-free data vendor: 18.37% / 1.15 / 17.09%; daily-return correlation 1.000000 on the overlapping sample |
| Alpha studies | 9 studies across 5 data sources — all banked NOT-DEPLOYABLE |
| Strongest result | Deflated Sharpe Ratio 0.905 on 21,916 survivorship-free names — refused by the gate, then diagnosed as a micro-cap artefact class |
| Trial ledger | 23 cumulative trials charged against the closing study's deflation |
| Live-path safety | 15-test proof that RESEARCH and PAPER cycles can never reach a broker |

The baseline deserves its own honesty note: against an equal-weight buy-and-hold
benchmark at 18.0% and Sharpe 1.12, the edge is thin, and the project's own documents
attribute it to market beta with cost-honest drawdown control — not alpha — on a
hand-picked survivor universe (a disclosed remaining optimism). A modest headline that
replicates to two decimal places across two independent data vendors is worth more
than a spectacular one that does not.

## Architecture

A modular, single-process Python engine running a synchronous **13-step decision
cycle**: point-in-time data validation → market state (HMM regime + a 7-detector
crisis composite) → volatility and covariance forecasting (GJR-GARCH/HAR-RV,
RMT-denoised covariance) → seven signal sleeves → signal-health gate → feature
construction → ML prediction → meta-label admission with an ex-ante cost gate →
Black–Litterman optimisation with an exact CVaR linear programme → pre-trade risk
gate → execution planning → execution with transaction-cost analysis → post-trade
learning. Three modes — RESEARCH, PAPER/SHADOW, LIVE — share one code path; LIVE is
disabled by default and double-armed (an explicit `confirm_live=True` plus an audit
log path). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/SYSTEMS.md](docs/SYSTEMS.md).

The engine is **fail-closed against its own research**:

- **No unvalidated signal drives real money.** Under rule SIGNALS-5, a sleeve without
  a registered validation result is capped at 0.25 weight, tightened to 0.0 in LIVE.
  This is why the live-mode paper session placed zero orders: every sleeve was
  unvalidated, so every sleeve was weighted zero — recorded as working as designed.
- **Hash-chained event ledger.** Every financial event (orders, fills, cash,
  reconciliations) is appended to a tamper-evident log in which each event's SHA-256
  hash includes the previous event's hash, chaining from a genesis value. Mistakes are
  corrected by appending explicit reversing events, never by editing; caller-supplied
  timestamps make replays bit-reproducible (`ops/ledger.py`).
- **An order lifecycle that treats broker reality honestly.** A submission timeout is
  not a rejection — `SUBMISSION_UNCERTAIN` is a first-class state that blocks
  resubmission; a cancel request is not a cancellation; duplicate fills are idempotent
  no-ops; filled quantity is clamped so it can never exceed approved quantity
  (`execution/order_lifecycle.py`, fuzzed by a Hypothesis state machine).
- **A latched kill switch.** A hard stop engages a durable latch that halts every
  subsequent cycle, survives restart, and clears only by an audited operator reset.
- **A no-live-path proof.** A single auditable module of 15 tests
  (`tests/test_safety_no_live_path.py`) proves that unknown modes are rejected, that
  a bare `mode=LIVE` assignment fails without both arming conditions, that the broker
  factory refuses to build a real-money broker from an unconfirmed config even when
  constructed to bypass validation, and — with a spy broker and a LIVE positive
  control — that full RESEARCH and PAPER cycles never reach a broker at all.

The execution path was validated end-to-end with a real fill on a funded paper
account: one share, round-tripped and auto-flattened.

## The validation gate

All studies from the learned-combination stage onward ran through one pipeline.
`research/validation.py` is the only permitted cross-validation implementation in the
project — standard k-fold and plain train/test splits are banned from research paths —
and it implements purged walk-forward with embargo, purging by bar count rather than
calendar days (a recorded defect fix: calendar purging under-purges a business-day
index).

Promotion is decided by an all-or-nothing, default-deny rule enforcing **seven
checks**: mean rank-IC > 0.01, net-of-cost Sharpe > 0.75, IC stability > 0.60, a
deflated-Sharpe proxy > 0.25, the full **Deflated Sharpe Ratio ≥ 0.95** (Bailey and
López de Prado, 2014), zero leakage flags, and no single regime with Sharpe below
−0.50. Any one failure blocks promotion. Overfitting probability is computed by
Combinatorial Symmetric Cross-Validation (Bailey, Borwein, López de Prado and Zhu,
2017). Trial counts are honest: `n_trials` is the number of configurations tried
during the research, not folds, and a superseded defective run still counts as a look.

A gate that only ever says no is untrustworthy, so the gate is itself under test.
`tests/test_edge_recovery_proof.py` runs four controls on seeded synthetic data: it
recovers a clearly planted edge; it passes a genuine edge tuned to land *near* the
threshold (DSR ≈ 0.965–0.987), proving discrimination rather than saturation; it
denies pure noise; and it denies the best of 50 pure-noise configurations selected
in-sample — a candidate whose in-sample Sharpe would clear a naive gate.

## The research programme

Nine studies, 18 June – 14 July 2026, in the order the failures posed the next
question. Full details, sources and per-study trial accounting are in the
[write-up](docs/RESEARCH_WRITEUP.md) §4–5.

| # | Study (data) | Headline | Verdict |
|---|---|---|---|
| 0 | Price/volume signal battery — large caps, full S&P 500, crypto | classic signals carry no robust alpha; engine returns attributed to smart-beta and risk management | no robust alpha |
| 1 | Single Fama–French factor loadings (30 names, 84 months) | best single tilt (size) DSR 0.83 | FAIL |
| 2 | Naive six-signal composite + EDGAR ROE/ROA (30 names) | net Sharpe −0.29 — worse than the best single signal | FAIL |
| 3 | Learned ridge over 8 point-in-time-safe features (30 names) | net Sharpe 0.92, DSR 0.467 — beats the naive composite, fails deflation | FAIL |
| 4 | Breadth run, factors and price/volume only (473 names) | IC fell from +0.11 to +0.04, DSR 0.254 — the fundamentals had carried the signal | FAIL |
| 5 | Free EDGAR fundamentals, 14 factors (140 names, 2006–2026) | DSR 0.543 despite a universe bias favouring the signal | FAIL |
| 6 | SEC Form 4 insider filings, pre-registered (140 names, two runs) | defective ticker join caught by adversarial review, corrected via an audited CIK bridge; corrected DSR 0.007 — and the fix made the signal *weaker* | FAIL, banked |
| 7 | Paid survivorship-free fundamentals (21,916 names, 1999–2026) | DSR 0.905, PBO 0.00 — the gate still refused it, and re-running variants to cross 0.95 was banned as selection bias | FAIL |
| 8 | Dev/confirm programme + six-run liquidity ladder | dev-window DSR 1.000 diagnosed as an untradeable micro-cap artefact (the plain rank-IC criterion caught what DSR and PBO both missed); tradable net Sharpe ≤ 0 | closed with the confirmation shot unfired |

What survived is narrow and real: a rank-IC of roughly +0.013 in clean,
survivorship-free fundamentals — reproducible, and unmonetisable by this construction
at honest retail costs. The programme's terminal act was declining to spend a
pre-registered confirmation test it was entitled to fire, because nothing could pass
the development-side gate. Folding on schedule is the discipline the whole apparatus
exists to enforce.

## How this was built

This project was built with AI assistance, and the write-up's §8 says so plainly. I
directed the programme — the question, the standards (default-deny gate,
pre-registration, review-before-run, honest trial accounting) and every go/no-go
decision — while AI coding agents did most of the hands-on implementation and were
also deployed adversarially against their own output, with every claimed defect
required to survive an active attempt to refute it before being acted on. Both of the
measurement artefacts described in the write-up were introduced by AI-written code and
caught by AI-driven review.

## Quickstart

Requires Python ≥ 3.11 (validated on CPython 3.13).

```bash
# Install (dev extras = tests, lint, types), pinned to the validated environment
pip install -e ".[dev]" -c constraints.txt

# Run the test suite (1,813 tests; a few skip when optional local data is absent)
pytest -q

# Run the offline research self-tests — each builds a deterministic synthetic
# dataset, plants an edge, and proves the pipeline recovers it and denies noise.
# No network, no paid data.
python scripts/research_free_alpha.py --selftest
python scripts/research_insider_alpha.py --selftest
python scripts/research_sharadar_alpha.py --selftest
```

Optionally, the FastAPI control surface (RESEARCH/PAPER modes; LIVE stays disabled):

```bash
pip install -e ".[dev,app]" -c constraints.txt
# --no-proxy-headers is required, not optional: uvicorn trusts X-Forwarded-For from
# loopback by default and rewrites the client address before the app sees it.
uvicorn --no-proxy-headers --factory ops.api:create_app_from_settings
```

## Repository map

```
core/          engine, config, optimiser, risk manager, regime/crisis models
data/          data contracts + point-in-time feature store
research/      the validation gate (purged walk-forward CV, selection rule)
research/medallion_style_alpha_search/   pre-registrations, errata, results
strategies/    signal sleeves and volatility models
learning/      ML prediction, meta-labelling, performance tracking
nlp/           sentiment features (FinBERT with a lexicon fallback)
execution/     order lifecycle, child scheduling, cost models, TCA
broker/        broker adapters (paper deterministic; live-only submit paths)
ops/           run loop, FastAPI control API, hash-chained ledger
backtesting/   backtest engine and bootstrap analysis
tests/         1,813 tests, incl. the no-live-path and edge-recovery proofs
scripts/       study runners (each with an offline --selftest)
docs/          RESEARCH_WRITEUP.md, ARCHITECTURE.md, SYSTEMS.md, project-control/
```

## Provenance

This repository is the initial public release of work developed privately; its public
history starts at a single commit. The research and engineering records inside it ship as
they were written, which has two consequences worth naming. Commit hashes quoted in
research documents (pre-registration anchors, fix records) refer to the private
development history, not to this repository's public history. And citations of the form
"§N", "standards §N" or "the internal research log" refer to internal project documents
that are not included here; where a record leans on one, the surrounding text states the
claim it relies on.

## Licence

Source-available; **all rights reserved**. Copyright © 2026 Leo Y. Zhang. You may
read the code and run it locally to evaluate or verify it (test suite included);
no licence is granted to reuse, copy, modify or redistribute it. See
[LICENSE](LICENSE).

## Disclaimer

Nothing in this repository is investment advice. The platform has no live trading
history; LIVE mode is disabled by design and the research conclusion of the project
is a negative result. The numbers above are backtest and paper-trading measurements,
reported net of modelled costs, with their limitations documented in the write-up.
