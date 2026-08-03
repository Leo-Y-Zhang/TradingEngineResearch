# PRE-REGISTRATION — SEC Form 4 insider-transactions alpha study (2026-07-11)

**Status: registered BEFORE any real-data run.** This document fixes the hypothesis,
feature set, universe, dates, gate and decision rule for the insider study. The real-data
run (`scripts/research_insider_alpha.py`, verdict to `insider_alpha_result.md`) happens
only AFTER an adversarial review of the build. Nothing below may be changed after seeing
results.

## Why this study (context)

Price/volume, Fama-French loadings and EDGAR XBRL fundamentals are exhausted and banked
NOT-DEPLOYABLE (`free_richer_fundamentals_result.md`, `stage_b_ff_loadings_result.md`,
`learned_combination_result.md`). Insider transactions are a genuinely NEW data modality
for this repo — filings-based, event-driven, never tested here — and one of the few
free datasets with a credible published out-of-sample record.

## Hypothesis

Insider **net purchases** — especially **opportunistic** (non-routine) and **clustered**
(multiple distinct officers/directors buying) — predict 1-month cross-sectional returns.

**Realistic, literature-based expectation:** ~2-5%/yr long-tilt alpha post-publication
decay (Lakonishok & Lee 2001; Jeng, Metrick & Zeckhauser 2003 find ~6%/yr gross on
purchases pre-decay; Cohen, Malloy & Pomorski 2012 find opportunistic trades carry the
predictive content; post-2001 (SOX 2-day filing) and post-publication decay plausibly
halves headline estimates). This is **NOT a 30%/yr candidate** — the 30% figure remains
a hypothesis about *stacked* edges, not a promise, and this study cannot deliver it alone.
Direction: net buying is bullish; insider *sales* are a much weaker (liquidity-driven)
signal — the composite features net them but the buy side carries the hypothesis.

## Fixed feature set (5 — FINAL, no post-hoc additions)

All features: monthly panels, officers+directors only, as-filed Form 4 only (no `4/A`),
availability = `FILING_DATE + 1 business day`, per-date cross-sectional winsorize+z-score
(`research.fundamental_features` conventions), NaN when no qualifying activity
(neutral-filled 0.0 only at the combination layer).

1. `net_buy_ratio_6m` — (n_buys - n_sells) / (n_buys + n_sells), trailing 6 months,
   count-based, open-market P/S codes only.
2. `net_buy_value_6m` — same, dollar-value weighted (shares x price).
3. `cluster_buying_3m` — number of DISTINCT officer/director owner CIKs with a `P`
   purchase in the trailing 3 months.
4. `opportunistic_buy_6m` — `net_buy_ratio_6m` EXCLUDING routine insiders
   (Cohen-Malloy-Pomorski simplified: a trade is routine if the same owner filed a `P`
   purchase for the same issuer in the SAME calendar month in EACH of the 3 preceding
   years; computed PIT — only strictly earlier years are queried).
5. `buy_intensity_6m` — total `P`-purchase dollar value over the trailing 6 months
   (per-date z-score does the scaling; no shares-outstanding data needed).

## Universe, dates, data

- **Universe:** the free runner's 140-name current-listed `DEFAULT_UNIVERSE`
  (`scripts/research_free_alpha.py`), overridable via `--tickers` / `--universe-file`.
  No larger curated list exists in this repo; reusing it keeps results comparable with
  the banked fundamentals studies.
- **Dates:** monthly rebalances **2007-01 onward** (2006 is burn-in for the trailing
  6-month windows; the routine filter needs 3 years of history and is simply NaN-sparse
  early — accepted a priori). Forward return: t to t+1 month-end, no look-ahead.
- **Insider data:** SEC quarterly `form345` TSV ZIPs, 2006q1-2026q2 (82 quarters),
  `data.insider_ingestion` (as-filed Form 4 only; the ONLY availability timestamp is
  FILING_DATE).
- **Prices:** free yfinance monthly adjusted closes — **CURRENT-LISTED, an OPTIMISTIC
  survivorship bias that HELPS the signal** (insider buying predicts best in small/
  distressed names most likely to have delisted). Registered consequence: **a FAIL is
  robust; a PASS is provisional** and requires survivorship-free prices before any
  deployment decision.
- **Costs:** 10 bps per monthly rebalance subtracted from OOS returns.

## Gate (pre-registered, default-deny)

- `research.alpha_factory.learn_signal_weights` (ridge, `PurgedWalkForwardSplitter`
  with embargo, label_horizon=1; same construction as `research_free_alpha`).
- `research.validation.selection_rule`: **Deflated Sharpe Ratio >= 0.95** (Bailey &
  Lopez de Prado) plus its rank-IC / net-Sharpe / stability checks — the binding gate.
- **PBO (CSCV)** reported as an overfitting diagnostic across the 5 single features.
- **n_trials = 8, counted honestly:** 5 pre-registered features + naive equal-weight
  composite + learned ridge combination + 1 runner configuration (the single set of
  windows/filters fixed in this document — nothing was tuned against returns data).
  If ANY additional configuration is ever tried, n_trials must be raised accordingly.

## Decision rule (pre-registered)

- **Any FAIL of `selection_rule` = banked NOT-DEPLOYABLE.** No post-hoc feature
  additions, no window re-tuning, no universe swaps, no "just one more variant".
- A PASS is **provisional only**: it must be replicated on survivorship-free prices
  before any deployment decision (the registered bias direction above).
- Either outcome is written to `insider_alpha_result.md` in the standard verdict format
  and banked.

## Verification before the run

`python scripts/research_insider_alpha.py --selftest` (offline, synthetic ZIPs) proves:
PIT lag (a month-end filing cannot influence its own month — a leak flips the measured
payoff sign), routine-stripping, end-to-end gate wiring (planted edge PASSES), and that
pure noise is DENIED. Unit suites: `tests/test_insider_ingestion.py`,
`tests/test_insider_features.py`.

---

## ERRATUM & PROTOCOL RECORD (appended 2026-07-13 — nothing above was edited)

1. **Protocol deviation (recorded honestly):** the registered order was *adversarial
   review → run*. The 2026-07-11 session ran the study after only a partial review (it
   did catch and fix the month-label defect first, commit `24905bc`). The full 4-lens
   review ran 2026-07-13 (44 agents, findings adversarially verified) and confirmed a
   **P1 measurement defect** in that first run: the join matched insider filings to the
   universe by the *as-filed free-text ticker*, silently losing ~22% of matched rows
   across issuer renames (GOOG→GOOGL, FB→META, UTX→RTX, PCLN→BKNG, WLP/ANTM→ELV,
   MHP/MHFI→SPGI, FPL→NEE, KFT→MDLZ, pre-2009 `(KO)`-style symbols) — Alphabet was
   effectively absent for the whole sample. The first run's FAIL is therefore **not a
   valid measurement of the registered universe** and is superseded.
2. **Correction (measurement fix, NOT tuning):** the join is now rename-safe by issuer
   CIK (`research.insider_universe`, audited map); a multi-owner filing's dollar value
   counts once (accession dedup; counts stay per owner-row as registered); a trailing
   partial-month price bar is dropped. Hypothesis, features, universe, dates, gate and
   decision rule are UNCHANGED.
3. **Honest trial accounting:** the seen-then-superseded first run counts as a trial —
   `n_trials` is raised 8 → **9** for the corrected re-run.
4. **Universe count:** the registered constant `DEFAULT_UNIVERSE` has **141** names
   ("140-name" above was a prose mis-count; the registered object was always the named
   constant). The first run used 140 after MMC's price download failed.
5. **Interpretation constraint (registered before the corrected re-run):** at ~92 OOS
   months, DSR ≥ 0.95 needs an observed annualized Sharpe ≈ 1.1; power for the
   registered realistic 2–5%/yr effect is ~1–30%. Any FAIL must therefore be banked as
   **"cannot certify a deployable edge"**, not as evidence the literature effect is
   absent. The routine strip fires on ~0.05% of events on this universe, so
   `opportunistic_buy_6m` degenerates to `net_buy_ratio_6m` — the CMP non-routine
   channel is NOT independently discriminated here (the "NaN-sparse early" note above
   was wrong: the feature is a near-duplicate, not sparse).
6. **Decision rule for the corrected re-run:** unchanged — any FAIL = banked
   NOT-DEPLOYABLE (with the framing of point 5); a PASS is provisional pending
   survivorship-free prices. **This is the final run of this study on free data;** no
   further variants regardless of outcome.
