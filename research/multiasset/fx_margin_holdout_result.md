# RESULT — the holdout passed the registered rule, and I do not believe it

**Pre-registration:** `fx_margin_holdout_prereg.md`, written after the fit step and before
any holdout residual existed, with `k = 0.345139` frozen from EUR/GBP/JPY.

**Registered verdict: SUPPORTED. Final verdict: NOT SUPPORTED.**

The registered decision rule passed on every count. It is overturned by an adversarial
comparator that I ran *because* it passed, and the reason is recorded here rather than the
verdict quietly changed. The registered rule was too weak, and saying so is the finding.

**Nothing here moves the corrected book Sharpe of 0.7834.**

---

## 1. What the registered test found

Holdout: FXF (Swiss franc), FXA (Australian dollar), FXC (Canadian dollar) — same sponsor,
**fee verified at 0.40%/yr on all three** (the registered invalidating condition did not
fire), n = 239 months each. All figures %/yr; errors in pp/yr.

| leg | measured | pred(frozen k) | pred(k=0) | err(k) | err(null) | k̂ | 95% CI | rate sd |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| CHFUSD / FXF | +0.300 | +0.035 | −0.084 | 0.265 | 0.384 | 0.304 | [−1.066, 1.673] | 1.01 pp |
| AUDUSD / FXA | +1.078 | +1.354 | +0.295 | 0.276 | 0.784 | 0.211 | [−0.077, 0.500] | 1.88 pp |
| CADUSD / FXC | +0.778 | +0.654 | +0.044 | 0.124 | 0.734 | 0.356 | [−0.017, 0.729] | 1.52 pp |

**H1 PASS** (every leg within 0.35 pp). **H2 PASS** (MAE 0.222 vs null 0.634 — error cut
roughly threefold). **H3 PASS** (k̂ positive on AUD and CAD, both CIs overlapping the frozen
0.345). **H4 PASS** (every k̂ in (0,1)). Registered rule ⇒ **SUPPORTED**.

Taken at face value that is a striking result: a parameter fitted on three currencies
predicted three untouched trusts to within 0.12–0.28 pp/yr.

---

## 2. Why it does not survive contact with a fair comparator

Two things were wrong with the registered rule, and both are my fault for registering it.

**The null was a straw man.** `k = 0` is not "no rate-proportionality" — it is **no
depository margin at all**. The proportional model carries a margin fitted on the original
three; the null carries none. H2 therefore tested "is there a margin?", which was never in
doubt, rather than "is the margin proportional to the rate?", which was the whole question.

**The fair comparator is a CONSTANT margin fitted on the same information.** Mean of the
original three remainders ⇒ **m = 0.483%/yr**, using exactly the data the frozen `k` used.

| model | CHF err | AUD err | CAD err | **MAE** |
|---|---:|---:|---:|---:|
| proportional, k = 0.345 | 0.265 | 0.276 | 0.124 | **0.2215** |
| **constant, m = 0.483** | 0.099 | 0.301 | 0.251 | **0.2170** |

**The constant is marginally better.** Out of sample, on three untouched trusts, a single
fixed number beats the rate-proportional model. H2's threefold improvement was real but it
was improvement over having no margin at all — it does not discriminate between the two.

### The dimension that does discriminate, and it goes the wrong way

A constant margin shifts both rate regimes equally, so it **cannot** change the regime
asymmetry — "constant" and "none" are identical there by construction. Regime shape is
therefore the *only* dimension separating the models, and it is precisely what the
registered rule failed to test.

| leg | no margin | proportional | constant |
|---|---:|---:|---:|
| CHF | 0.610 | **0.131** | 0.610 |
| AUD | 0.181 | **0.943** | 0.181 |
| CAD | 0.272 | **0.373** | 0.272 |
| **mean \|asymmetry\|** | **0.354** | **0.482** | **0.354** |

**The proportional correction makes the regime asymmetry worse than doing nothing** — and
worst on **AUD and CAD**, the two legs I registered in advance as the only ones with enough
rate variation to identify `k`. On AUD it over-corrects so hard that the high-rate remainder
flips negative (−0.370). It helps only CHF, the leg registered in advance as unidentifiable.

That is the opposite of what a real rate-proportional margin would do.

---

## 3. The verdict, and what the confidence intervals were already saying

**NOT SUPPORTED.** A rate-proportional depository margin joins the constant margin as a
mechanism that does not explain the residual's rate-dependent shape.

The wide intervals were the tell, and they were visible in the registered output: every
holdout k̂ has a 95% CI containing **zero** (AUD [−0.077, 0.500], CAD [−0.017, 0.729],
CHF [−1.066, 1.673]). H3 as I wrote it — "positive, and CI overlaps 0.345" — is a test a
sufficiently noisy estimate passes automatically, because a wide interval overlaps
everything. **A CI that contains both the hypothesis and its null discriminates nothing.**
That is a lesson about the test, not about the data.

---

## 4. What IS established, and it is not nothing

Stripping out what failed, three things survive and are worth carrying forward:

1. **A depository margin of roughly 0.2–0.5%/yr exists across all six CurrencyShares
   trusts, and it transfers.** Predicting it from the original three cut level error on
   three untouched trusts from 0.634 to ~0.22 pp/yr. That is a genuine out-of-sample
   result and it holds for either functional form.
2. **The magnitude is now pinned across six instruments** rather than three, on trusts
   spanning negative rates (CHF) to 7% rates (AUD, CAD).
3. **Both natural margin models are now eliminated as explanations of the SHAPE.** The
   constant cannot produce regime dependence at all; the proportional produces it with the
   wrong sign and magnitude on the identified legs.

**This does revise the framing of `fx_residual_result.md`.** That document ruled out a
constant margin on the strength of the regime shape, and that remains correct — but it is
now clear the constant explains the *level* better than the proportional model does. The
honest decomposition is: **level ≈ a fee-plus-margin story that works; shape ≈ unexplained
by any depository-margin model tested so far.**

---

## 5. Honest limits

* The post-hoc comparator in §2 is exactly that — post-hoc. It is legitimate because it can
  only *weaken* the verdict, never strengthen it; adding analyses that could only help
  would be the abuse. It is labelled as such in the JSON and here.
* `m = 0.483%/yr` is the unweighted mean of three remainders, not a fitted optimum. A
  properly fitted constant would do slightly better still, which strengthens the conclusion
  rather than weakening it.
* Holdout samples start 2006-06 (all three ETFs launched together), n = 239. Nothing here
  speaks to earlier periods.
* ETF returns remain market-price, not NAV; premium/discount noise is unmodelled and is a
  live candidate for part of the unexplained shape.
* The trial count for this programme is now **two registered tests** on the same residual
  (`fx_residual`, `fx_margin_holdout`), both negative. A third should be registered only
  with a genuinely new mechanism, not a third parameterisation of the same one.

**Next candidate, named and not tested:** the shape may not be a margin at all. The
remaining rate-dependence could be **tracking error in the trusts' own creation/redemption
and premium/discount behaviour**, which widens when rates — and therefore the cost of
carrying inventory — are high. Testing that needs NAV series rather than market prices, and
it should be pre-registered with a control that can fail.
