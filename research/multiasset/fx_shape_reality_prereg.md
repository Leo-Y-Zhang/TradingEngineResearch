# PRE-REGISTRATION — is the regime asymmetry REAL, or an alignment artefact?

**Written 2026-07-31, BEFORE any statistic in §3 has been computed.** No circular
shift, no block bootstrap, no p-value exists at the time of writing. That is the
whole design, and the commit history is the evidence.

Governing instruction: `fx_margin_holdout_result.md` closes with *"a third should be
registered only with a genuinely NEW mechanism, not a third parameterisation of the
same one."* This is that test.

---

## 0. The design, in one line

Both previous tests assumed the shape is real and hunted for its cause. **This one
asks whether there is a shape at all** — whether the regime asymmetry is
distinguishable from an autocorrelated, near-zero-drift residual that happens to line
up with the rate regime.

## 1. Why this is a NEW mechanism and not a third margin

A constant margin and a rate-proportional margin are both *causes of a real effect*.
They share a premise. This hypothesis rejects the premise: that the asymmetry is a
sampling artefact of serial dependence, and there is nothing to explain.

It is also falsifiable in the **opposite direction** to its predecessors. They
predicted a residual would shrink; this predicts an observed statistic lies INSIDE a
null band. A real effect pushes it outside. That asymmetry of direction is deliberate
— it cannot be passed by the same noise that passed the last two.

**It also avoids the failure mode the register named.** The holdout lesson was that
"every holdout k-hat had a 95% CI containing ZERO... a CI containing both the
hypothesis and its null discriminates nothing." Here the null is an explicit
distribution and the hypothesis is a point outside it; there is no interval that can
contain both.

## 2. What is already known, disclosed so this cannot claim credit for it

From `fx_residual_result.md` and `fx_margin_holdout_result.md`, both committed:

- Remainders, headline `zero_floored`, %/yr: **EUR 0.743, GBP 0.490, JPY 0.216**.
- Low-rate months: GBP remainder falls to **0.003%/yr**, JPY 0.506 -> 0.148.
  High-rate months: all three keep **0.45-1.12%/yr**.
- Mean |asymmetry| reported as **0.354** (no correction and constant margin alike),
  **0.482** under the proportional correction.
- `k` is weakly identified; every holdout CI contained zero.
- The published 0.40%/yr fee is confirmed; the US TED term enters negatively.

**Also established today, and it is why this test exists.** NAV series are **not
obtainable from this machine**: Invesco serves a JavaScript shell on every product and
NAV-history path tried, the Yahoo quote endpoint returns 401, Stooq is behind a JS
challenge. And NAV would not have been independent evidence anyway — a CurrencyShares
trust holds a plain deposit, so NAV per share is (deposit per share) x spot with the
deposit evolving by (interest earned - fee), which is *exactly* the model whose
residual is under test. Substituting NAV for market price returns the depository
margin **by construction**, the same unfalsifiable quantity already ruled out of
bounds. Premium/discount needs the price-minus-NAV *difference*, which is bounded and
mean-reverting and therefore contributes about (P_end - P_start)/T to an annualised
mean — near zero over 239 months — unless the premium is itself rate-correlated.

**NOT known at writing:** any null distribution, any p-value, any verdict.

## 3. The statistic

Over the committed monthly decomposition (`decompose`, headline construction, the
committed `_convention/convention_repair.json` inputs), for each leg
L in {EURUSD, GBPUSD, JPYUSD}:

    A_L = 12 * [ mean(remainder_t | high-rate) - mean(remainder_t | low-rate) ]   (%/yr)

`high-rate` / `low-rate` use the **committed** threshold already in the module.
No variant threshold is tried.

> **AMENDMENT, recorded before any statistic was computed.** The parenthetical in
> the first version of this line said "foreign overnight rate >= 0.5%". That is a
> **wrong description** of the committed rule: `regime_split` splits on
> `frame["i3m_foreign"] <= LOW_RATE_THRESHOLD` with `LOW_RATE_THRESHOLD = 0.005`,
> i.e. the foreign **3-month** rate at 0.5%. The binding instruction was always
> "the committed threshold already in the module", so the module governs and
> nothing about the test changes. Corrected in the open rather than silently, and
> before the run, so the record shows which of the two was actually used.

    Pooled statistic  S = mean over L of |A_L|

## 4. The null distributions

**N1 — circular shift.** Shift the remainder series by a uniform random offset
tau in [1, T-1] with wrap-around, hold the regime labels fixed, recompute S.
B = 10,000. This preserves the residual's autocorrelation **exactly** while
destroying its alignment with the regime.

**N2 — stationary block bootstrap**, mean block length 6 months, B = 10,000. A second
opinion with different assumptions.

    p = fraction of null draws with S_null >= S_observed

## 5. The decision rule, fixed now

- **REAL** — p < 0.05 under **both** nulls. The shape survives; the thread stays open
  and a mechanism is still owed.
- **ARTEFACT** — p >= 0.10 under **both** nulls. The asymmetry is not distinguishable
  from accidental alignment. The thread **closes**: the residual is reported as a
  level (a depository margin of ~0.2-0.5%/yr, which already transfers out of sample)
  with **no regime structure to explain**.
- **UNDETERMINED** — anything else, including the two nulls disagreeing. No
  conclusion is drawn, the thread stays open, and this trial is spent regardless.

## 6. Controls, all required to pass before the verdict is read

- **C1 POWER.** Inject a synthetic rate-proportional effect of 0.5%/yr into the
  remainder and rerun. Must return REAL. A test that cannot see a real effect is
  toothless, and reporting ARTEFACT from a toothless test would be worse than useless.
- **C2 SIZE.** Replace the remainder with i.i.d. Gaussian noise of matched variance.
  Must return ARTEFACT, at approximately the nominal rate.
- **C3 DETERMINISM.** Fixed seed; two runs agree exactly.
- **C4 NO VARIANT SHOPPING.** The construction, the fee, and the regime threshold are
  the committed ones. If the answer is read and then a threshold is changed, this
  document is void.

## 7. What would make me wrong

A circular shift under-disperses when the series has strong low-frequency, near
unit-root structure, which would make ARTEFACT too easy to reach. **N2 is the guard**,
and disagreement between the nulls yields UNDETERMINED rather than a convenient pick.
The honest failure mode of this design is concluding "no shape" from a test with too
little power against a small true effect; C1 sets a floor on that, and the result
document will report the smallest effect the test can see, not merely the verdict.

## 8. Trial accounting

This is the **third** registered test on this residual. It consumes one trial whatever
the outcome. Nothing here touches the panel, the ledger's selection path, the live
path, or 0.7834.
