# Insider (SEC Form 4) transactions alpha study — real-data result (2026-07-13)

> **SURVIVORSHIP NOTE (header, by design):** prices are CURRENT-LISTED yfinance closes —
> an OPTIMISTIC bias that *helps* the signal (insider buying predicts best in the small /
> distressed names most likely to have delisted). A **FAIL below is therefore robust**;
> a PASS would be provisional and would require survivorship-free prices before any
> deployment.

**Question:** do PIT-safe SEC Form 4 insider-transaction features (net buying, clustered
buying, opportunistic i.e. non-routine buying) carry a deflation-surviving 1-month
cross-sectional edge? Pre-registered BEFORE this run in `insider_study_prereg.md`
(fixed 5-feature set, gate, decision rule — no post-hoc feature additions).

**Method:** `scripts/research_insider_alpha.py` -> `research.alpha_factory.learn_signal_weights`
(ridge, purged walk-forward) gated by `research.validation.selection_rule`
(Deflated Sharpe >= 0.95) + PBO (CSCV) diagnostic; net of
10 bps per monthly rebalance.

- **Universe:** 140 current-listed names, 2007-01-31 -> 2026-05-31,
  **233 monthly rebalances**, walk-forward test window 46
  per fold, DSR deflated for **9 trials** (5 pre-registered features +
  naive composite + learned combination + 1 runner configuration + the seen-then-corrected
  2026-07-11 first run; see the prereg erratum).
- **Join:** transactions matched to the universe RENAME-SAFELY by issuer CIK
  (`research.insider_universe`; the 2026-07-11 first run's as-filed-ticker join silently
  lost ~22% of matched rows across renames — GOOG/FB/UTX/PCLN/WLP/MHP/FPL/KFT and
  pre-2009 parenthesized symbols).
- **Features (5, fixed a priori):** net_buy_ratio_6m, net_buy_value_6m, cluster_buying_3m, opportunistic_buy_6m, buy_intensity_6m. Multi-owner filings
  contribute dollar value once (accession dedup); counts stay per reporting owner.
- **PIT discipline:** availability = FILING_DATE + 1 business day; as-filed Form 4 only
  (4/A amendments excluded); month-end panel assignment proven leak-free by `--selftest`;
  a trailing partial-month price bar is dropped (no mislabeled short forward return).

## Result

| Combination | OOS IC | rank-IC | net Sharpe | stability | **DSR** | PBO | `selection_rule` |
|---|---|---|---|---|---|---|---|
| **Learned ridge** | -0.0012 | +0.0018 | -0.35 | 0.49 | **0.007** | 0.50 | **FAIL (default-deny)** |
| Naive equal-weight | — | — | -0.09 | — | 0.028 | — | — |

Learned weights: net_buy_ratio_6m=-0.006, net_buy_value_6m=+0.003, cluster_buying_3m=+0.000, opportunistic_buy_6m=+0.003, buy_intensity_6m=-0.000

**VERDICT: NOT-DEPLOYABLE**

## Decision rule (pre-registered)

Any FAIL = banked NOT-DEPLOYABLE — no feature additions, no window re-tuning, no
universe swaps after seeing this table. A PASS is provisional until replicated on
survivorship-free prices (the current-listed bias above is in the signal's favour).

## Interpretation caveats (registered from the 2026-07 adversarial review, verdict-independent)

1. **Power:** at ~92 OOS months and this trial count, DSR >= 0.95 requires an
   OBSERVED annualized Sharpe of ~1.1; simulated power for the pre-registered realistic
   effect size (~2-5%/yr long-tilt) is ~1-30%. A FAIL is therefore **"cannot certify a
   deployable edge"** — it is NOT evidence the (small) literature effect is absent.
2. **Scope:** on this mega-cap universe insiders overwhelmingly sell (~97% of qualifying
   events), so the ratio features are mostly binary (69% of finite cells pinned at -1)
   and the FAIL generalizes to *insider net-buy indicators on ~140 mega-caps*, not to
   insider signals on richer cross-sections (small caps / breadth).
3. **Opportunistic channel not discriminated:** the Cohen-Malloy-Pomorski routine strip
   fires on ~0.05% of events here (mega-cap insiders almost never buy on a fixed annual
   schedule), so `opportunistic_buy_6m` is a near-duplicate of `net_buy_ratio_6m`; the
   CMP non-routine hypothesis was NOT independently tested on this universe.
4. **OOS window:** the walk-forward verdict rests on the LAST ~92 months only (2 folds);
   earlier history is training-only. The learner's 46-month validation block between
   train and test is computed but unused (fixed l2) — a harness inefficiency for future
   studies, bounded here by the staleness-free naive composite failing independently.
5. The naive-composite DSR row is an in-sample full-length statistic (T=233),
   not an OOS one; it is a control, not a second candidate.

## Run history (study record)

| Run | Join | Rows matched | Rebalances | OOS IC | net Sharpe | DSR | Status |
|---|---|---|---|---|---|---|---|
| 2026-07-11 | as-filed ticker (defective) | 507,680 | 234 (incl. a partial July bar) | +0.0046 | -0.16 | 0.029 (8 trials) | **SUPERSEDED** (P1 join loss; prereg erratum pt 1) |
| 2026-07-13 | audited CIK bridge | 639,552 (+26%) | 233 | -0.0012 | -0.35 | 0.007 (9 trials) | **BANKED** |

The corrected join recovered +131,872 rows (Alphabet's full 2006-2026 history via the
audited Google-Inc predecessor CIK; FB/UTX/PCLN/WLP/MHP/FPL/KFT/(KO)-era renames) and
REMOVED ~4.3k pollution rows the naive join had wrongly counted (First Commonwealth
filed as "F" inside Ford's history, Genesis Energy's "GE:", AirXpanders' "AXP", ...).
Fixing the measurement made the signal WEAKER — a cleaner zero, not a suppressed edge.
Evidence table: `insider_universe_cik_audit.csv` (158 audited ticker-CIK pairs).

**CONCLUSION (banked): free SEC Form-4 insider indicators on this mega-cap universe
cannot be certified as a deployable edge — consistent with every prior free-data
modality (price/volume, FF loadings, EDGAR fundamentals). This was the final run of
this study on free data.** The engine correctly continues to trade nothing (fail-closed,
SIGNALS-5); the honest path to a validated edge remains richer data (Sharadar SFA)
and/or the mapped remaining free PIT modalities (FINRA short interest 2018+, 13F
2013Q2+, FTD 2004+), each under its own pre-registration.
