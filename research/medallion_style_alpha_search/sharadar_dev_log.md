# Sharadar dev-window EXPLORATION log (prereg: sharadar_confirmatory_prereg.md)

> Every number in this file is DEV-WINDOW (≤2015-12-31) EXPLORATION. Nothing here
> confers deployability; the single confirmation shot (2016+, frozen model, DSR ≥ 0.95
> at n_trials = 17) decides. This log exists so the freeze amendment can report the
> full iteration history honestly.

## Entry 1 — 2026-07-13: registered-model baseline on DEV

`scripts/research_sharadar_dev.py` (hard-cut verified), single run, exit 0.

| Window | Names | Rebals | OOS IC | rank-IC | net Sharpe | stability | DSR | PBO | Gate |
|---|---|---|---|---|---|---|---|---|---|
| DEV 1999-02..2015-11 | 14,591 | 202 | +0.0240 | **−0.0155** | **1.33** | 0.65 | 1.000 | 0.00 | **FAIL** |

- **Binding failure: `mean_rank_ic ≤ 0.01` — every other criterion passes** (Sharpe
  1.33 > 0.75, stability 0.65 > 0.60, DSR 1.000 ≥ 0.95, PBO 0.00, no leakage flags).
- Same Pearson-positive / Spearman-negative divergence as the full-sample run
  (+0.0059/−0.0373): the dollar-neutral portfolio profits from signal MAGNITUDE in the
  tails while average rank ordering across the (mostly micro-cap) cross-section is
  slightly inverted. Hypotheses to test (dev-only): tail/liquidity concentration of
  the P&L, long-leg vs short-leg attribution, monotonicity by signal decile,
  micro-cap noise drowning the ranks, flat-10bps cost unrealism on small names.
- Context for expectations: dev-era (pre-2016) fundamentals were far stronger
  (Sharpe 1.33) than the full sample (0.52) — consistent with post-publication factor
  decay. The 2016+ confirmation window is the hard regime; iterate accordingly
  (regime-robustness > dev-Sharpe-maximization).
- Weights (dev baseline): earnings_yield −0.025, book_to_price +0.013,
  sales_to_price +0.015, roe +0.002, roa +0.007, gross_profitability +0.004,
  operating_margin +0.002, revenue_growth −0.005, earnings_growth +0.007,
  asset_growth −0.010, net_share_issuance +0.019, accruals −0.003,
  debt_to_equity −0.004, momentum_12_1 +0.006.

## Entry 2 — 2026-07-13: diagnosis of the rank-IC divergence (3-agent workflow, exact replication)

The empirical agent reproduced the gate's own arithmetic to the digit (same folds, same
80 OOS months 2009-04..2015-11, mean IC/rank-IC/Sharpe identical), so these are facts
about the real signal, not an approximation:

1. **The dev Sharpe 1.33 is an untradeable-microcap artifact.** Mean forward return by
   signal decile is flat noise D1-D9 and explodes only in D10 (+4.21%, top-1% +11.62%);
   D10 alone is >100% of P&L. The least-liquid dollar-volume quintile carries **112.7%**
   of gross P&L. D10 median price $3.61 (31% sub-$1), median 63d dollar volume $51.5k
   (36% below $10k/day). Within the top-1500 or top-3000 liquid names the learned signal
   is NEGATIVE on every metric (Sharpe −0.45 / −0.37, rank-IC −0.011 / −0.008).
2. **Robustness kill-shots:** capping monthly returns at ±100% flips Sharpe +1.33 →
   **−0.43**; excluding sub-$1 names → +0.51. A single name-month (ILXRQ 2009-05,
   +9,900% at $0 volume) is 13% of total P&L; AWHL 2009-08 (+33,650%) looks like a
   closeadj data artifact (QA item). The short leg LOSES money (hit rate 38%).
3. **The rank-IC criterion failed CORRECTLY.** Portfolio return is linear in the signal
   (the Pearson numerator); Spearman weighs all ~5.5k names equally — the humble
   rank-IC caught the artifact that DSR=1.000 and PBO=0.00 completely missed. Also
   proved: Spearman is invariant to monotone transforms of the combined signal —
   cosmetic fixes are impossible; only universe/content changes move it.
4. **Consequence for the banked full-sample result:** same construction, same unfiltered
   universe → the 0.905 near-miss is very likely the same artifact class. Interpretation
   DOWNGRADED (addendum added to `sharadar_alpha_result.md`). The honest tradable-
   universe starting point is roughly Sharpe −0.4..+0.5, not 1.33/0.52.

**Iteration plan (each run counted in the n_trials ledger below):**
- DEV-1: `build_liquidity_universe(sep, rebal_dates, top_n, window=63, min_obs=42)`
  (monthly-reconstituted top-N by trailing MEDIAN dollar volume) + `universe` mask
  passthrough (mask forward returns; learner/composite/PBO inherit via isfinite).
  Run top-1000 @10bps. Success criterion: the IC/rank-IC divergence COLLAPSES (same
  sign, gap < ~0.01) — not merely rank-IC > 0.01.
- DEV-2: N=500/1500 + absolute floors ($5M median dv, $5 price) — liquidity phenomenon
  vs N-artifact check.
- DEV-3: cost sensitivity 20bps on the best variant (10bps-only passes are not
  freeze-eligible); forward-return cap ±100% as a permanent dev-QA guard + closeadj
  QA on the flagged prints; feature-coverage mask (≥8/14 non-missing pre-fillna).
- DEV-4 (if divergence collapses but rank-IC < 0.01): 3m horizon, size-neutralization,
  rank-based portfolio construction; the D10 distressed-shell cluster re-measured
  INSIDE the tradable universe (possible real but capacity-limited effect).

**n_trials ledger (for the freeze amendment):** full-sample run (16) + dev baseline (1)
+ each DEV run above as executed.

## Entry 3 — 2026-07-14 (overnight): the liquidity ladder — divergence collapsed, edge unmonetizable

Six runs (schtasks ladder, each exit 0; machinery `8002b71`, defaults pinned identical):

| Run | Universe | Cap | Cost | OOS IC | rank-IC | net Sharpe | DSR | PBO | naive Sharpe/DSR |
|---|---|---|---|---|---|---|---|---|---|
| baseline (E1) | unfiltered 14,591 | — | 10bp | +0.0240 | −0.0155 | **+1.33** | 1.000 | 0.00 | −0.27 / 0.001 |
| dev1 | top-1000 | — | 10bp | +0.0074 | +0.0100 | −0.18 | 0.012 | 0.68 | +0.13 / 0.102 |
| dev1b | top-1000 | ±100% | 10bp | +0.0112 | +0.0130 | −0.05 | 0.026 | 0.44 | +0.18 / 0.139 |
| dev2a | top-500 | ±100% | 10bp | +0.0087 | +0.0179 | −0.13 | 0.016 | 0.57 | +0.18 / 0.142 |
| dev2b | top-1500 | ±100% | 10bp | +0.0094 | +0.0118 | −0.10 | 0.020 | 0.35 | +0.23 / 0.186 |
| dev2c | $5M dv floor | ±100% | 10bp | +0.0106 | +0.0113 | −0.02 | 0.032 | 0.56 | +0.22 / 0.180 |
| dev3 | top-1000 | ±100% | 20bp | +0.0112 | +0.0130 | −0.28 | 0.006 | 0.44 | +0.03 / 0.048 |

**Findings (consistent across every tradable variant):**
1. **The diagnosis was right.** Inside tradable universes the Pearson/Spearman
   divergence disappears (both small-positive; rank-IC +0.010..+0.018 would now clear
   the gate's rank criterion). The unfiltered +1.33 was pure micro-cap artifact.
2. **The real ordering edge is ~+0.013 rank-IC — and it is unmonetizable by this
   construction.** Net Sharpe ≤ 0 in ALL tradable variants at 10bps; doubling costs to
   a still-optimistic 20bps costs another ~0.23 Sharpe (monthly turnover of a linear
   signal is far too high for an edge this thin).
3. **The learned ridge UNDERPERFORMS the naive equal-weight composite in liquid
   universes** (naive +0.13..+0.23 vs learned ≤ −0.02): the learner's tail-weighted
   fit was chasing the micro-cap artifact; with it masked, its weights add negative
   value. Naive's best (+0.23, DSR 0.186) is itself nowhere near the gate.
4. **Ceiling analysis (freeze-or-fold input):** with rank-IC ≈ +0.013 on ~1000 names,
   the gross monthly-rebalance IR ceiling sits far below the gate's `sharpe_net > 0.75`
   at any honest cost. DEV-4 levers (3m horizon = ~1/3 the cost drag; rank-based
   construction; size-neutralization) could plausibly lift net Sharpe to ~+0.2-0.4
   best-case — still not 0.75, and each attempt spends ledger trials.

**Assessment: RECOMMEND FOLD on the 1-month fundamentals L/S hypothesis.** Do NOT burn
the one pre-registered confirmation shot on a model family that cannot pass the DEV-side
gate: the prereg's shot is only worth firing at a dev-passing model, and none is in
reach. Legitimate residual threads if the operator wants them (each: dev-only, ledger-
counted, expectations set low): the distress-cluster long-only overlay measured inside
the tradable universe (capacity-limited, different sleeve semantics than the L/S gate),
and the 3m-horizon variant as LEARNING about turnover economics, not as a gate path.
Track C (engine calibration on clean data) remains fully alive and is the program's
concrete deliverable either way.

**n_trials ledger: 16 (full-sample) + 1 (dev baseline) + 6 (ladder) = 23.**

### Entry 3b — closeadj QA on the flagged prints (Track C data-QA)

- **ILXRQ 2009**: $0.0001 flat for months with ZERO median volume, then indicative
  quotes step 0.0001→0.01→0.03→0.08 (still zero volume). The +9,900% "return" that was
  13% of the unfiltered baseline's P&L is a no-trade quote reprice on a bankrupt shell —
  unrealizable by construction.
- **AWHL 2009-09**: $0.60 → $202.50 in one month, price SUSTAINED at $200-324 after —
  a large reverse split that `closeadj` did NOT adjust: a genuine Sharadar data artifact
  (adjusted series broken for this name).
- Consequence: the `fwd_return_cap` guard stays ON for all dev work, and any future
  study on this data should carry it (or a delisting/return-integrity filter) as
  standard equipment. Liquidity masks alone do not protect against vendor adjustment
  breaks in thin names.

## Entry 4 — 2026-07-14 (overnight): Track C — the engine headline is data-source-proof

`scripts/backtest_real_sharadar.py` (identical config to the banked yfinance headline;
only the price source changes; offline, chunked read of the raw export; yfinance
auto-adjust semantics reproduced via closeadj/close OHLC scaling; run twice,
bit-identical under seed 42):

| Metric | yfinance (banked) | Sharadar | Δ |
|---|---|---|---|
| ann return (net) | 18.4% | **18.37%** | ~0 |
| Sharpe | 1.15 | **1.15** | 0 |
| max drawdown | 17.1% | **17.09%** | ~0 |
| EW benchmark | 18.0% / 1.12 | 18.00% / 1.12 | 0 |

Daily-return correlation on the overlapping committed fixture (AAPL/MSFT/JPM,
2022-23): 1.000000, max |Δ| ≈ 1e-4/day; no divergence years; the 8-megacap Sharadar
matrix is pristine (0 NaN/zero-volume/outlier days). **Conclusion: the engine's honest
market-beta headline (thin but positive edge over EW, superior drawdown control) is
robust to the price source.** This closes the "is the yfinance headline a data
artifact?" question for the incumbent config — the remaining known optimism is the
hand-picked 8-survivor universe (a future Track C slice could measure that with a
PIT top-N large-cap basket; delisted coverage now exists to do it honestly).
