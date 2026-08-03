# PRE-REGISTRATION — the rate-proportional margin, tested on trusts the panel has never used

**Written 2026-07-31, AFTER the fit step and BEFORE any holdout residual was computed.**
The `k` frozen below, and everything known at writing time, is stated in §1 so this cannot
later be read as having predicted what it already knew. **No `diff`, remainder, or
regression has been computed for FXF, FXA or FXC at the time of writing.** That is the
whole design.

Governing finding: `fx_residual_result.md` §2 — what remains after the published fee and
the measured tenor and TED spreads is **rate-proportional**, so a constant depository
margin is ruled out. That document named this hypothesis and deliberately refused to fit
it to the same data. This is that test, done properly.

---

## 0. The design, in one line

**Fit on A, test on B.** `k` is estimated on EUR/GBP/JPY — where the answer is already
known, so the fit proves nothing — then **frozen** and used to predict the residuals of
three CurrencyShares trusts the panel has never touched.

The holdout is not arbitrary. **FXF (Swiss franc), FXA (Australian dollar) and FXC
(Canadian dollar) are the same sponsor, the same published 0.40%/yr fee, and the same
two-account JPMorgan London deposit structure** as FXE/FXB/FXY. If the depository keeps a
fraction of the rate under one contract, it should keep it under the others. And they add
what the original three could not supply: **genuine rate variation.**

---

## 1. What is already known — disclosed so this document cannot claim credit for it

The fit step (`scripts/fit_fx_margin_k.py`, output `_fx_residual/margin_fit.json`) has
been run. Its results, in full:

| leg | k alone | se | t | 95% CI | overnight sd |
|---|---:|---:|---:|---|---:|
| EURUSD | 0.5864 | 0.2934 | 2.00 | [0.011, 1.161] | 1.613 pp |
| GBPUSD | 0.2113 | 0.1690 | 1.25 | [−0.120, 0.543] | 2.017 pp |
| JPYUSD | 1.8010 | 3.0318 | 0.59 | [−4.141, 7.743] | 0.204 pp |

**Pooled, no intercept: k = 0.345139**, raw R² **0.0054**, R² about mean **0.0024**,
n = 709 months. Free-intercept diagnostic: slope 0.2896, intercept 0.1964%/yr.

**The honest reading, fixed now:** `k` is **weakly identified**, not wildly inconsistent.
Only EUR reaches t = 2.00; GBP's interval includes zero; JPY is effectively unidentified
because its overnight rate barely moves (sd 0.204 pp), which is why its point estimate of
1.80 — a depository keeping 180% of the rate, which is impossible — is small-denominator
noise rather than evidence. Monthly R² is negligible throughout. The pooled estimate rests
almost entirely on EUR.

**This is exactly why the holdout is worth running**, and it is the reason it emphasises
AUD and CAD: both carried 4–7% policy rates before 2008, so they can identify `k` where
JPY could not. Registering that motivation here, in advance, rather than discovering it
afterwards.

---

## 2. The model, frozen

    earned_t = max(0, overnight_t) * (1 - k),      k = 0.345139   [FROZEN]

giving, on the same identity the result doc used,

    predicted_t = (i3m_foreign - earned_t)/12 + fee/12 - (i3m_US/12 - cash_t)

with `fee = 0.40%/yr` published, the same lag convention
(panel-side rates lagged, benchmark-side contemporaneous), and the same `zero_floored`
flooring. **`k` is not re-estimated on the holdout for H1 or H2.** Where the holdout is
used to estimate `k` at all, that is H3 and it is a separate, labelled question.

**Null model to beat:** the constant model already reported, i.e. **k = 0** — the
`zero_floored` decomposition of `fx_residual_result.md`. The proportional model adds one
parameter; it must earn it.

---

## 3. Registered predictions

**H1 — point accuracy (primary).** With `k` frozen, the predicted annualised residual for
each of CHF, AUD and CAD lands within **0.35 pp/yr** of measured. That bar is the
cross-currency consistency tolerance already registered in the first prereg (P3), reused
rather than newly chosen.

**H2 — it must beat the null (decisive).** Mean absolute prediction error across the three
holdout trusts must be **strictly lower** under the frozen-k model than under k = 0. If
the extra parameter does not reduce error out of sample, it has earned nothing and the
hypothesis is dead regardless of H1.

**H3 — identification.** `k` estimated independently on **AUD and CAD** (the two legs with
real rate variation) must be **positive** and its 95% interval must **overlap the frozen
0.345**. CHF is reported but excluded from this test, registered now and for a stated
reason: Swiss rates were negative for much of the sample, so `max(0, overnight)` is near
zero and `k` is unidentifiable there — the same defect JPY has. Excluding it after seeing
its result would be indefensible; excluding it now, for a reason visible in advance, is not.

**H4 — physical plausibility.** Every per-leg `k̂` must lie in **(0, 1)**. A depository
cannot keep a negative share of the rate, nor more than all of it. Violations are reported,
and a violation on an *identified* leg (AUD, CAD) counts against the model; a violation on
an unidentified leg (CHF) is reported as noise, consistent with how JPY is treated above.

**H5 — do no harm.** Applying the frozen-k model must not push any of the ORIGINAL three
legs outside the 0.75%/yr budget they currently sit inside.

### Decision rule, fixed now

The rate-proportional margin is **SUPPORTED** only if **H2 holds** (beats the null out of
sample) **and H3 holds for both AUD and CAD**. H1 and H4 are reported and inform the
write-up but cannot rescue a failure of H2 or H3.

If H2 fails, the honest conclusion is that **a rate-proportional margin joins the constant
margin as eliminated**, and the residual's rate-scaling shape remains unexplained by any
depository-margin story. Given the weak identification recorded in §1, that is a plausible
outcome and will be reported as the headline if it occurs, not buried.

---

## 4. Data, and the one thing that could invalidate the comparison

* **ETFs**: FXF, FXA, FXC via the repo's own `fetch_one` (`auto_adjust=True`, so Close is
  total-return), cleaned by the same `clean_levels` / `simple_returns` the panel uses.
  Probed 2026-07-31: all three return 5,056 daily rows, 2006-06-26 → 2026-07-31.
* **Spot**: `FX_CHF`, `FX_AUD`, `FX_CAD` from `_data/carry/fx_spot_returns_monthly.parquet`
  (274 / 242 / 274 monthly observations) — already built, already quarantine-screened, and
  already in the long-foreign convention the carry module expects.
* **3-month rates**: `CH`, `AU`, `CA` from the existing `short_rates_monthly.parquet`
  (324 / 702 / 846 observations).
* **Overnight**: OECD `IRSTCI` for `CHE`, `AUS`, `CAN` — probed 2026-07-31 and present
  (−0.2 / 3.6 / 2.250676 for 2026-01). Same dataflow, same measure, same transport as the
  amended first run.

**The invalidating condition, registered now:** these three trusts must be verified to
carry the **same 0.40%/yr sponsor's fee** as FXE/FXB/FXY. If any differs, its fee term is
wrong and that leg must be dropped from H1/H2 **and said so**, not quietly re-parameterised.

---

## 5. What this can and cannot change

Unchanged from the first prereg: **no panel series, no strategy, no gate, no headline
number.** Nothing here can move the corrected book Sharpe of **0.7834**. This is still a
test of the yardstick.

Its only possible outcomes are: the residual's rate-scaling component gets a mechanism, or
the most natural remaining mechanism is eliminated and the thread stays open with one
fewer candidate. Both are reportable; only the second requires resisting the temptation to
keep re-parameterising until something fits, which is why H2 is decisive and why `k` is
frozen before the holdout is touched.
