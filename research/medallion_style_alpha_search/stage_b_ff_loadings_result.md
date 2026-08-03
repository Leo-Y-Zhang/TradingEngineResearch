# Stage B — first richer-data result: FF factor loadings (honest)

**Date:** 2026-06-26 · **Script:** `scripts/research_ff_factors.py` · **Validator:** real Deflated Sharpe (`research.validation.deflated_sharpe_ratio`)

## Test
Universe: 30 current large caps (⚠️ survivorship, METH-1), 2015–2025, monthly rebalance.
Each month: trailing 126-day **PIT-safe Fama-French loadings** per stock → cross-sectional
rank → long top tercile / short bottom tercile, ~10bps cost → strategy monthly returns →
**Deflated Sharpe Ratio**, deflated for the **4 factors tried** (honest multiple-testing).

## Result
| factor | n | ann_ret | ann_vol | Sharpe | DSR | verdict |
|---|---:|---:|---:|---:|---:|---|
| value (β_HML) | 84 | −10.5% | 23.1% | −0.45 | 0.01 | no edge |
| size (β_SMB) | 84 | +12.7% | 17.3% | **0.73** | **0.83** | **no robust edge** |
| low-beta (−β_MKT) | 84 | −22.2% | 23.9% | −0.93 | 0.00 | no edge |
| momentum 12-1 | 84 | −0.1% | 20.0% | −0.01 | 0.14 | no edge |

## Verdict — honest
**No single cross-sectional FF-loading (or price-momentum) factor carries a deflation-surviving edge** on this universe/period. The size tilt's 0.73 Sharpe is the only positive, but **DSR 0.83 < 0.95** → it does **not** survive multiple-testing/non-normality deflation, so it is **not trustworthy alpha**. The hardened validator correctly prevented us from promoting noise.

## Why this is the *right* outcome (and what it implies)
- It confirms the audit's honest finding at a new layer: easy single factors don't give robust alpha. This matches the literature — Medallion-style returns come from **many weak, independent signals combined**, not one factor.
- The pipeline now WORKS end-to-end: richer free data → PIT-safe cross-sectional features → rigorous DSR-deflated validation → trustworthy verdict. That machinery is the durable asset.
- **Next credible steps** (each a Stage-B increment): (1) survivorship-free PIT universe (removes METH-1 bias); (2) richer per-stock *fundamentals* from SEC EDGAR (value/quality/profitability — stronger cross-sectional signals than FF loadings); (3) a **combination** of many weak signals with turnover-aware portfolio construction, validated by DSR + PBO; (4) FRED macro for regime conditioning. A credible >30% (if it exists) lives in the *combination*, not any single factor.

**Bottom line:** the honest answer so far is "no easy edge" — exactly what a trustworthy validator should tell us. We now have the rigorous machinery to keep searching without fooling ourselves.

---

## #2 — combining the signals (`scripts/research_combined_alpha.py`)

Added **SEC EDGAR fundamentals** (ROE, ROA — PIT-safe on filing date) to the FF loadings + momentum, and tested an **equal-weight z-score composite** (value + size + low-beta + ROE + ROA + momentum), same long-short / cost / DSR setup (deflated for 7 trials).

| signal | ann_ret | Sharpe | DSR |
|---|---:|---:|---:|
| size | +12.7% | 0.73 | 0.73 |
| roa | +6.3% | 0.38 | 0.35 |
| value | −10.5% | −0.45 | 0.01 |
| roe | −4.5% | −0.25 | 0.02 |
| low_beta | −22.2% | −0.93 | 0.00 |
| momentum | −0.1% | −0.01 | 0.08 |
| **COMPOSITE (equal-weight)** | **−5.0%** | **−0.29** | **0.02** |

**Finding:** the naive composite is *worse* than the best single signal — because equal-weighting blindly includes signals that were strongly NEGATIVE on this universe/period (value, low-beta). **No combination survives deflation.**

**The real lesson (not a failure):**
1. "Combine many weak signals" does **not** mean naive equal-weighting with assumed signs — the bad signals drag it down.
2. A genuine combination needs **sign-aware, validated weighting** (e.g. learn weights on a train fold, validate OOS with DSR/PBO) — but that must be done *without* overfitting, which is exactly what the DSR/PBO machinery guards.
3. On **30 survivorship-biased large caps over a growth-dominated decade**, there is no easy edge — and the validator honestly says so at every step.
4. A credible edge would need: a **survivorship-free, broad universe** (hundreds of names), **more/better signals** (alt-data, finer fundamentals, microstructure), and a **learned, regularised, OOS-validated combination** — a serious multi-week research program with uncertain payoff. The honest >30% remains a *hypothesis*, not a result.

**Durable asset delivered:** the full rigorous pipeline now exists and works — richer free data (FF + EDGAR, PIT-safe) → cross-sectional signals → composite → DSR/PBO-deflated validation. That machinery is what lets the search continue *without self-deception*.
