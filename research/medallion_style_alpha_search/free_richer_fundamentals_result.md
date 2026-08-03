# Free richer-fundamentals attempt — real-data result (2026-06-30)

**Question:** the prior 30-name study (`learned_combination_result.md`) got *close* (DSR 0.467) and
failed deflation purely on sample size, pointing at "**richer fundamental features at moderate
breadth**" as the lever — but that was blocked for free by (a) EDGAR being too slow per-name and
(b) no free survivorship-free universe. Using the new **14-factor library** + a **`companyfacts`
bulk-per-company** EDGAR path (one request per company), does the free-data edge now cross the
deflation bar?

**Method:** `scripts/research_free_alpha.py` → `research.alpha_factory.learn_signal_weights` (ridge,
purged walk-forward). FREE data only: SEC EDGAR `companyfacts` fundamentals + yfinance monthly prices.

- **Universe:** 140 current US large/mid-caps (1 skipped: no CIK), 2006-03 → 2026-05, **243 monthly
  rebalances** (vs 84 before), walk-forward test window 48, DSR deflated for 16 trials.
- **Features (14):** earnings_yield, book_to_price, sales_to_price, roe, roa, gross_profitability,
  operating_margin, revenue_growth, earnings_growth, asset_growth, net_share_issuance, accruals,
  debt_to_equity, momentum_12_1.
- **Data correctness:** the EDGAR XBRL bridge was adversarially reviewed and fixed BEFORE this run —
  flow concepts collapsed to a consistent quarterly period (no 3M/YTD/annual mixing), multi-class
  shares summed, ASC-606 revenue concept-switch stitched, debt = interest-bearing (not total
  liabilities). So the result below is a real signal, not an XBRL artefact.

## Result

| Combination | OOS IC | rank-IC | net Sharpe | stability | **DSR** | PBO | `selection_rule` |
|---|---|---|---|---|---|---|---|
| **Learned ridge** | +0.036 | +0.028 | 0.67 | 0.60 | **0.543** | 0.19 | **FAIL (default-deny)** |
| Naive equal-weight | — | — | 0.11 | — | 0.096 | — | FAIL |

**VERDICT: NOT-DEPLOYABLE** — no robust edge survives deflation.

## Interpretation (honest)

1. **More data tightened the DSR but not enough** (0.467 → **0.543**, cutoff 0.95). The 140-name ×
   20-year panel added the cross-sectional power the 30-name study lacked — but the OOS IC also fell
   (+0.113 → +0.036): breadth diluted the signal that the 30 mega-caps carried. Net effect: still
   fails, and not close.
2. **This free result is OPTIMISTICALLY BIASED** — the universe is current-listed *and*
   winner-selected (no delisted/bankrupt names; today's survivors). Removing that bias (a real
   survivorship-free universe) would generally make the measured edge **weaker, not stronger**. So a
   free test biased *in its own favour* still fails deflation.
3. **The XBRL pitfalls were real** and would have faked the result if unfixed (period mixing alone
   flipped net income up to ~4×). The adversarial review caught them; the FAIL is trustworthy.

## Conclusion (banked)

The free-data lever is now **exhausted**: across the 461-name price/volume study, the 30-name
fundamentals study, the 473-name breadth study, and this **140-name × 20-year × 14-factor** study,
the answer is consistent — **no deflation-robust edge in free data**, and this last one was biased in
its own favour and still failed.

**Implication for the paid (Sharadar) decision — honest:** Sharadar would give the *definitive*
survivorship-free answer and removes the bias that flatters this study. But because that bias was
*helping* the free result and it still failed at DSR 0.543, the evidence points toward **"no robust
deployable edge from these fundamental factors alone,"** rather than "an edge is hiding behind the
bias." A reliable **30%/yr** target via this fundamental-factor route is **not supported by the
evidence.** Sharadar is worth it for a clean, final verdict (and richer data: delisted names,
insiders/13F, daily metrics) — but go in expecting a rigorous *test*, not a likely *yes*.

The engine's honest standing value is unchanged: market-beta with superior, cost-honest drawdown
control (Sharpe ~1.15 net). The durable win remains the **rigorous pipeline that refuses to deploy
overfit noise** — which it just demonstrated again, on free data, the right way.
