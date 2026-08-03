# Learned cross-sectional combination — real-data result (2026-06-28)

**Question:** does a *learned* (regularised, OOS-validated) combination of richer features
carry a deflation-surviving edge, where naive equal-weight composites did not?

**Method:** `scripts/research_learned_alpha.py` → `research.alpha_factory.learn_signal_weights`
(ridge, purged walk-forward). PIT-safe cross-sectional feature panel over 30 US large-caps,
monthly, 2015–2025:

- **Factor loadings** (Fama-French, live Ken French data 1926–2026): `value` (β_HML),
  `size` (β_SMB), `low_beta` (−β_MKT).
- **Price/volume:** `momentum` (12-1), `reversal` (1m), `low_vol` (−63d σ).
- **Fundamentals** (SEC EDGAR, filing-date PIT, 16,771 facts / 30 names): `roe`, `roa`.

Forward returns: next-month. Net of 10 bps. DSR deflated for 10 trials (8 features +
naive composite + learned config).

## Result

| Combination | net Sharpe | mean IC | rank IC | stability | **DSR** | `selection_rule` |
|---|---|---|---|---|---|---|
| **Learned ridge** | 0.92 | **+0.113** | +0.073 | 0.69 | **0.467** | **FAIL (default-deny)** |
| Naive equal-weight | −0.70 | — | — | — | 0.001 | FAIL |

## Interpretation (honest)

1. **The learned combiner clearly beats the naive composite** (Sharpe 0.92 vs −0.70;
   DSR 0.467 vs 0.001) and shows a **healthy positive cross-sectional IC (+0.11)**. It
   passes 5 of the 6 classic `selection_rule` conditions (rank-IC, Sharpe>0.75,
   stability>0.60, DSR-proxy>0.25, no leakage). So the "alpha is in the *combination*"
   thesis has real substance — learning adds value over equal-weight.
2. **But it fails deflation** (DSR 0.467 < 0.95). On only 84 monthly observations with 10
   trials, the Deflated Sharpe Ratio cannot rule out that the 0.92 Sharpe is luck/overfit.
   The gate **correctly default-denies it** — it is NOT deployable.
3. **This refines, not contradicts, the prior** ("no easy robust price/volume alpha"): a
   *learned* combination of *richer* (factor + fundamental) features gets meaningfully
   closer to a robust edge than singles or naive composites did. The binding constraint is
   now **statistical power / sample length**, not the absence of any signal.

## What would move the needle (next, honest)

- **More observations** to tighten the DSR: longer history and/or a **much broader
  universe** (100s of names → far more cross-sectional obs per date) — the single biggest
  lever on deflation. Needs a **survivorship-free PIT universe** (METH-1; historical index
  membership is not free via yfinance — caveat).
- **Better features** (earnings revisions, accruals, richer fundamentals) — the IC is
  already positive, so incremental signal could push DSR over the line.
- Keep every candidate **DSR/PBO-gated** (`learn_signal_weights` already does this) — only
  a gated pass wires through `learning/adaptive_weights.py` to replace the equal-weight blend.

**Bottom line:** the learned-combination vehicle works and shows promise; the data on a
30-name large-cap sample is not yet enough for a *deployable* (deflation-robust) edge.
Honest >benchmark alpha remains a hypothesis — but a better-supported one than before.
Survivorship caveat (METH-1): current large caps; a first read, not deployable.

## Breadth follow-up (same day) — `python scripts/research_learned_alpha.py --broad`

Re-ran on **473 current S&P 500 names** (survivorship-biased — caveat), **factor + price/volume
features only** (EDGAR fundamentals skipped — per-name fetch is too slow at 500 names).

| Run | features | net Sharpe | mean IC | DSR | gate |
|---|---|---|---|---|---|
| 30-name | factor + price/vol + **fundamentals** | 0.92 | +0.113 | 0.467 | FAIL |
| 473-name | factor + price/vol (no fundamentals) | 0.48 | +0.036 | 0.254 | FAIL |

**Decisive read:** breadth alone did **not** tighten the DSR — it *lowered* it. The extra
cross-sectional power was more than offset by **dropping the fundamental features**, whose
absence cut the IC from +0.11 to +0.04. So **the fundamentals (ROE/ROA), not price/volume
or factor loadings, carried most of the 30-name signal.** Both runs beat the naive
composite; neither survives deflation.

**Conclusion (banked, honest):** the learned, DSR/PBO-gated combiner is built and works; a
*deployable* edge is not reachable with the free data tested. The most promising lever is
**richer fundamental features at moderate breadth** (not raw breadth, and not price/volume) —
but that needs (a) fundamentals fetched for 100s of names (EDGAR is slow/rate-limited) and
(b) a **survivorship-free PIT universe** (METH-1; historical membership not free via
yfinance). Until that data exists, the engine's honest value stands: market-beta with
superior, cost-honest drawdown control (Sharpe ~1.15 net). Do NOT keep re-running breadth
chase-experiments — across 30-name, 473-name, and the prior 461-name price/volume study,
the answer is consistent.
