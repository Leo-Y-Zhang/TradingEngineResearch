# PRE-REGISTRATION — Sharadar dev/confirm fundamentals program (2026-07-13)

**Status: registered BEFORE any development work.** This document fixes the design,
data split, gate, trial accounting and decision rule for the one legitimate follow-up
to `sharadar_alpha_result.md` (DSR 0.905 vs 0.95 → banked NOT-DEPLOYABLE). Nothing in
§§2-6 may change after development begins, except via a dated amendment appended BEFORE
the confirmation run and covered by the pre-run adversarial review.

## 1. Why this design (context)

The 2026-07-13 full-sample study measured real structure (net Sharpe 0.52, PBO 0.00)
that fails certification at the registered 0.95 DSR bar. Iterating variants against the
same full-sample gate until it crosses would be selection bias — banked as forbidden.
The honest route to "improve until deployable" is a **development/confirmation split**:
unlimited iteration on an early window that never touches the confirmation years, then
ONE pre-registered test on the held-out recent window.

**Contamination disclosure (registered up front):** the 2026-07-13 run's walk-forward
OOS folds covered the confirmation years, and its aggregate result (one model's OOS
performance) is known to the developers. The confirmation window is therefore *mildly*
compromised — one prior look, disclosed and counted in `n_trials` — not virgin. This is
the best evidence still available from this dataset and is disclosed rather than hidden.

## 2. Data split (physical, enforced by separate files)

- **DEV extract:** SF1 rows with `datekey <= 2015-12-31`; SEP rows with
  `date <= 2015-12-31`. Development rebalances end 2015-11 (labels stay inside the
  extract). All iteration — features, horizons, weighting schemes, cost/liquidity
  modelling, universe filters — happens ONLY against this extract.
- **CONFIRM extract:** the full dataset; confirmation rebalances run 2016-01 →
  end-of-data. Untouched by every development tool; opened exactly once, by the
  confirmation runner, after the final model is frozen.
- The extracts are separate parquet files; development code paths take the DEV path
  only. The pre-run review must verify no development artifact read the confirm years.

## 3. Development phase (2026-07-13 → freeze)

Anything goes ON DEV DATA, provided it is honest engineering: factor refinements and
additions, holding horizons (1-12m), sector/size neutralization, delisting-return
handling, calibrated transaction-cost and liquidity models (Track C feeds these),
regularization and weighting schemes. Dev-set walk-forward + DSR/PBO are used freely as
*guides*. None of this confers deployability — dev results are exploration by
definition and will be labeled as such wherever reported.

## 4. Freeze + confirmation (ONE shot)

1. A dated **FREEZE amendment** is appended here specifying the single final model
   completely: features and exact definitions, horizon, rebalance calendar, weighting,
   universe/liquidity rules, cost model (calibrated values frozen as constants), and
   the confirmation runner invocation.
2. A **4-lens adversarial review** of the frozen build runs BEFORE the confirmation
   (the insider-study protocol; review-then-run is mandatory).
3. The confirmation runner executes ONCE on the CONFIRM extract, rebalances 2016-01
   onward.
4. **Gate:** `research.validation.selection_rule` with Deflated Sharpe ≥ 0.95 and
   **n_trials = 17 + d**, where 17 = the 16 trials of the 2026-07-13 full-sample study
   (they saw the confirmation years) + this one frozen model, and d = any additional
   confirm-window looks that occur for any reason before the run (target: d = 0). PBO
   reported. The freeze amendment must restate the final n_trials before the run.

## 5. Decision rule (pre-registered)

- **PASS →** the model is *provisionally validated*: wired through the engine's gated
  path (`learning/adaptive_weights.py`), PAPER trading first, live remains disabled
  behind the standing operator gate (§23 / no-live-path). Deployability language must
  say "certified on a disclosed once-compromised holdout".
- **FAIL → final.** No further fundamentals gate attempts on this dataset, ever. The
  program ends with the calibration gains (Track C) and the banked knowledge.
- Either way the verdict document reports the full dev-phase iteration honestly
  (what was tried, what the dev-set showed) alongside the single confirmation number.
- **Deadline:** the confirmation must run before the subscription's data-retention
  window closes (subscription cancelled 2026-07; purge due within 30 days of period
  end). Target freeze: on or before 2026-08-05.

## 6. Track C (parallel, always-honest — no gate implications)

Using the same paid data, independent of the study: recalibrate the engine's honest
baseline (the yfinance 8-name Sharpe 1.15 headline) on survivorship-free prices, and
calibrate cost/liquidity/delisting models. These improve the engine regardless of the
confirmation outcome and consume no statistical validity; calibrated constants may be
frozen into the final model spec (§4.1).

## 7. Expectations (registered a priori)

The dev window (1999-2015) differs in regime from the confirm window (2016-2026:
low-rate bull, COVID, 2022 rate shock); a model tuned to dev may degrade OOS — that is
the test working. Prior probability of a PASS is judged well under 50%. A certified
pass, if it comes, validates a MODEST edge (the 2026-07-13 run's honest scale: ~0.5
net Sharpe); it is the foundation for compounding toward the 30%/yr *hypothesis*, not
its achievement.

---

## PROGRAM CLOSURE (2026-07-14, operator-accepted)

**Outcome: CLOSED WITHOUT FIRING THE CONFIRMATION SHOT — no freeze-eligible model
found.** The dev phase (entries 1-4 of `sharadar_dev_log.md`, 23 ledger trials)
established that inside any tradable universe the 1-month fundamentals L/S family has
a real but unmonetizable ordering edge (rank-IC ≈ +0.013; net Sharpe ≤ 0 at honest
costs), far below the registered `sharpe_net > 0.75` gate. Firing the single
pre-registered confirmation at a model that cannot pass the dev-side gate would have
wasted it. The confirmation window therefore remains unconsumed by this program;
any future program requires a new pre-registration and honest cumulative trial
accounting. Track C deliverables (engine headline verified data-source-proof;
return-cap QA standard; closeadj artifact findings) are banked and survive the
licence purge as Derived Data.
