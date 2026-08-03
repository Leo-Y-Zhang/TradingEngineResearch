# Sharadar survivorship-free fundamentals study — real-data result (2026-07-13)

> The **definitive** run of the fundamentals hypothesis: full-history (1999-2026),
> survivorship-free (delisted names included), 21,916-ticker universe, point-in-time
> as-reported (`ARQ`, filing `datekey`) — the clean version of the question every free
> study could only approximate. Data: paid Sharadar SFA export (one-month subscription,
> 2026-07-13; raw data purged post-termination per licence — see
> `sharadar_purge_record.md`; these Derived-Data results are owned outright under §6.2).

**Question:** does a learned combination of 14 PIT-safe fundamental factors carry a
deflation-surviving 1-month cross-sectional edge on clean, survivorship-free data?

**Method:** `scripts/research_sharadar_alpha.py` (pipeline adversarially reviewed
2026-06-30 BEFORE any real-data run; trust proof `tests/test_edge_recovery_proof.py`
passes a real near-threshold edge and denies noise/overfit) → `load_panel` (SF1 `ARQ`
on filing `datekey` + SEP `closeadj`) → 14 factors → `learn_signal_weights` (ridge,
purged walk-forward) → `selection_rule` (Deflated Sharpe ≥ 0.95) + PBO (CSCV); net of
10 bps per monthly rebalance. Export verified pre-run: SEP 46,172,487 rows / 21,915
tickers / 1997-12-31→2026-07-10 with delisted names present (ENRNQ, LEHMQ, WCOEQ,
SHLDQ); SF1 676,796 ARQ filings.

- **Universe:** 21,916 names (every ticker in the export — zero selection), monthly
  1999-02-26 → 2026-06-30, **329 rebalances**, walk-forward test window 65 per fold,
  DSR deflated for **16 trials** (14 factors + naive composite + learned combination).

## Result

| Combination | OOS IC | rank-IC | net Sharpe | stability | **DSR** | PBO | `selection_rule` |
|---|---|---|---|---|---|---|---|
| **Learned ridge** | +0.0059 | −0.0373 | **0.52** | 0.56 | **0.905** | **0.00** | **FAIL (default-deny)** |
| Naive equal-weight | — | — | −0.38 | — | 0.000 | — | — |

Learned weights: earnings_yield=−0.017, book_to_price=+0.005, sales_to_price=+0.012,
roe=+0.004, roa=+0.000, gross_profitability=+0.004, operating_margin=−0.001,
revenue_growth=−0.001, earnings_growth=+0.007, asset_growth=−0.005,
net_share_issuance=+0.005, accruals=−0.002, debt_to_equity=−0.003, momentum_12_1=+0.002

**VERDICT: NOT-DEPLOYABLE (default-deny).**

## Honest interpretation — read all of it

1. **This is the strongest honest result this project has ever measured.** Net Sharpe
   0.52 out-of-sample on a survivorship-free 21,916-name universe net of costs, with
   PBO 0.00 (the CSCV diagnostic finds no evidence the combination was overfit) and
   DSR 0.905 — versus DSR ≤ 0.54 for every free-data study. The learned combination
   decisively beats the naive composite (−0.38). Real predictive structure exists in
   clean fundamentals at breadth.
2. **And the gate still says no — that is the system working, not a technicality.**
   DSR 0.905 means: after deflating for 16 trials, there remains a ~9.5% probability
   that a Sharpe this large arises with no true edge. The cutoff is 0.95. Worse, the
   registered n_trials **undercounts** the true search space (noted in
   `DATA_EDGE_PLAN.md` §deferred BEFORE this run) — factor definitions, the ridge
   spec, windows and the gate itself embed design choices tested across this
   project's history, so the honest deflation is harsher than 16.
3. **Do NOT chase the threshold.** Re-running variants (windows, universes, factor
   tweaks, l2, cost assumptions) until 0.905 crosses 0.95 is textbook selection bias —
   precisely what the gate exists to refuse. Any future attempt must be a NEW
   pre-registered study (new prereg doc, review-before-run, n_trials honestly raised
   by every variant tried), on materially new information (different horizon, new
   data, structural refinements from the deferred list) — not a re-roll of this one.
4. **Oddity recorded:** rank-IC is negative (−0.0373) while Pearson IC is positive and
   the portfolio Sharpe is solidly positive — the signal's payoff is carried by
   magnitude in the tails rather than uniform ordering. Recorded as a caveat, not
   investigated post-hoc (that would be tuning).
5. **Decision consequence:** the engine stays fail-closed (SIGNALS-5); no sleeve is
   validated for LIVE. The fundamentals question now has its definitive, clean-data
   answer under this project's registered standard of evidence: *real structure,
   not certifiable at the 0.95 bar with honest deflation.*

## Run record

Single run, 2026-07-13 21:45→22:5x local, exit code 0, detached (Task Scheduler),
log `study_run2.log`. No re-runs, no parameter changes, no variants. First and only
real-data execution of this pipeline.

---

## ADDENDUM 2026-07-13 (same day, after dev-window diagnosis — honesty update)

The dev/confirm program's diagnosis (`sharadar_dev_log.md` entry 2, exact replication
of the gate arithmetic) showed the dev-window version of this construction earns its
Sharpe almost entirely in untradeable micro-caps (least-liquid quintile = 112.7% of
gross P&L; ±100% return cap flips the sign; single penny-stock name-months dominate),
and is NEGATIVE within top-1500/3000 liquidity universes. This study used the same
unfiltered 21,916-name universe and linear construction, so the "real predictive
structure" interpretation in §Honest-interpretation-1 is **DOWNGRADED**: the 0.905
DSR / 0.52 Sharpe are likely dominated by the same untradeable-tail artifact. The
NOT-DEPLOYABLE verdict stands with a stronger rationale than deflation alone. The
negative rank-IC (−0.037) was the honest tell, and the rank-IC gate criterion was
right. Tradable-universe measurement is the dev/confirm program's job.
