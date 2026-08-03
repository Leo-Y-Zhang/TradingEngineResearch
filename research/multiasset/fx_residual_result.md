# RESULT — the EUR/GBP FX residual is STILL NOT EXPLAINED

**Pre-registration:** `fx_residual_prereg.md`, written and committed (`06e6e34`) with no
results attached, before any decomposition existed. Its predictions, tolerances, bracket
and decision rule are quoted here unchanged. The transport amendment in that document was
also recorded before any remainder existed.

**Verdict: NOT EXPLAINED**, on two independent grounds. The registered mechanism is real,
measurable and removes a meaningful part of the residual — but it fails the primary
prediction, and the three registered constructions disagree by more than the tolerance.

**Nothing here moves the corrected book Sharpe of 0.7834.** This tested the yardstick,
not the ruler, exactly as registered.

---

## 0. The gates that had to pass first, and did

| gate | result |
|---|---|
| **P0 source identity** | **PASS.** The OECD pull reproduces the pre-block FRED series on the registered window: GBR n=582 1978-01→2026-06 last 3.7298; JPN n=492 1985-07→2026-06 last 0.841; USA n=864 1954-07→2026-06 last 3.625172 — all exact. EA20 truncated at 2026-01 gives n=385 last 1.931671, matching. |
| **P5 null control** | **PASS.** With the fee and both rate spreads zeroed the pipeline returns **0.9356 / 0.8580 / 0.4960**, reproducing Control C's committed residuals exactly. The decomposition is measuring the same quantity Control C measured. |
| **P6 sign discipline** | **PASS.** `mean(i3m − overnight) ≥ 0` for EZ, GB, JP and US. The two series are paired correctly. |

**Transport cross-check.** OECD's own `IR3TIB` 3-month series against the repo's existing
FRED-sourced `short_rates_monthly.parquet`: **GB, JP and US byte-identical**
(max |diff| = 0.000e+00 over 481 / 290 / 744 months); EZ max 1.244e-04, mean 1.035e-06.
FRED is a mirror of OECD, and swapping the pipe did not swap the data.

---

## 1. The decomposition, headline construction (`zero_floored`)

All figures %/yr. `A1` = foreign 3-month minus overnight, `B` = the published 0.40%
sponsor's fee, `C` = the US TED spread, which is **subtracted**.

| pair | measured residual | A1 | B | −C | predicted | **remainder** |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD vs FXE | +0.936 | +0.070 | +0.400 | −0.278 | +0.193 | **+0.743** |
| GBPUSD vs FXB | +0.858 | +0.247 | +0.400 | −0.280 | +0.367 | **+0.490** |
| JPYUSD vs FXY | +0.496 | +0.155 | +0.400 | −0.275 | +0.280 | **+0.216** |

The fee lands at exactly +0.400 on all three because it is **published, not fitted**. The
US TED term is real and material at ~0.28%/yr, and it enters negatively — **the US leg
works against the hypothesis**, shrinking the prediction most in exactly the high-rate
months where the residual is largest. That was visible before the run and is confirmed by it.

---

## 2. Why it fails — P1, the primary prediction

Registered: after adjustment the regime asymmetry `|gap_high − gap_low|` must fall to
**≤ 0.25 pp/yr** for EUR and GBP.

| pair | asymmetry before | asymmetry after | bar | verdict |
|---|---:|---:|---:|---|
| EURUSD | 1.0732 | **0.7639** | 0.25 | **FAIL** |
| GBPUSD | 0.6108 | **0.5728** | 0.25 | **FAIL** |
| JPYUSD *(control)* | 0.0466 | **0.3045** | — | **worse** |

Neither asymmetry *widened*, so P1 is falsified rather than refuted outright — but neither
comes close to the bar. And the control leg moved the wrong way: JPY had almost no regime
asymmetry (0.047) and the adjustment **created** one (0.305).

### The regime detail is the actual finding

| pair | before: low / high | after: low / high |
|---|---|---|
| EURUSD | 0.399 / 1.472 | **0.361 / 1.125** |
| GBPUSD | 0.338 / 0.949 | **0.003 / 0.576** |
| JPYUSD | 0.506 / 0.460 | **0.148 / 0.453** |

**In low-rate months the registered story essentially works.** GBP's remainder falls to
**0.003%/yr** — the fee, the tenor premium and the TED spread account for the entire gap.
JPY falls from 0.506 to 0.148.

**In high-rate months it does not.** All three keep 0.45–1.12%/yr, and the fall there is
small. So what is still missing is **proportional to the rate level**, not a constant.

That falsifies my own registered framing as well as the earlier one. The prereg treated
the depository margin as an unobservable but essentially **contractual constant**, which
is why it was left as a bare remainder. A constant cannot produce this shape, so a
constant-margin model cannot close this gap no matter what value is chosen for it.

---

## 3. The bracket disagrees, which is a second independent failure

Prereg §4: *"If they disagree by more than 0.25 pp/yr the disagreement itself is the
finding and the residual is declared bracketed, not explained."*

| construction | EURUSD | GBPUSD | JPYUSD |
|---|---:|---:|---:|
| `published` | +0.5952 | +0.4905 | +0.2000 |
| `zero_floored` *(headline)* | +0.7427 | +0.4905 | +0.2164 |
| `fee_first` | +0.9349 | +0.5329 | +0.5112 |
| **range** | **0.3397** | 0.0424 | 0.3112 |

Widest range **0.3397 pp** against a 0.25 pp tolerance ⇒ **disagree**. GBP is
well-determined (0.042 pp); EUR and JPY are not, because both spent long spells at or
below zero where the three flooring rules diverge most. Under `fee_first`, `A1` even goes
negative for EUR (−0.122) and JPY (−0.140) — that construction credits the trust more than
the 3-month rate when rates are near zero.

### P2 passed, and should not be leaned on

P2 (remainder inside the 0.75%/yr budget) passed on the headline: EUR **0.743**, GBP
**0.490**. But EUR passes by **0.007 pp/yr**, and it **fails under `fee_first` at 0.9349**,
which is an equally registered construction. A pass that depends on which of three
pre-registered constructions you pick is not a pass, and it is reported here as one that
does not survive its own bracket. P4 (do no harm: JPY inside budget) passed at 0.216 on
every construction.

**Decision rule, applied as written:** EXPLAINED required P1 **and** P2 **and** P4. P1
failed. The verdict is NOT EXPLAINED, and P3 (cross-currency consistency) is not permitted
to rescue it.

---

## 4. Honest limits of this run

1. **The "before" asymmetries here are not the published ones** (1.073 / 0.611 / 0.047 vs
   the published 0.916 / 0.358 / 0.079). Two causes, both benign: this split uses the
   **lagged** foreign rate — what `fx_excess` actually credited — while Control C split on
   the contemporaneous one; and the sample is the intersection with the overnight series.
   The residual **levels** reproduce exactly (P5), so this is a difference in how the
   regime split is defined, not a disagreement about the residual itself. P1's bar was an
   absolute one on the *after* value, so the comparison it gates is unaffected.
2. **ETF returns are market-price, not NAV.** Trust premium/discount noise sits inside
   `diff` and is not modelled. Disclosed in the prereg; it inflates dispersion and no
   figure here should be read as if it were absent.
3. **EZ transport difference** of at most 1.24e-04 (mean 1.04e-06) between OECD-direct and
   FRED, i.e. ~0.0001 pp/yr — four orders of magnitude below the quantity measured.
4. The sample starts when each ETF launched (EUR n=242 from 2005-12, GBP n=235 from
   2006-07, JPY n=232 from 2007-02), so none of this speaks to the pre-2005 panel.

---

## 5. What this changes, and what is now open

**It changes no panel series, no strategy, no gate and no headline number**, exactly as
registered. The FX correction's verification still stands on the JPY leg alone.

What is genuinely gained:

* The residual is **better bounded and better characterised** than it was. It is no longer
  "unexplained ~0.9%/yr"; it is "≈0.2–0.5%/yr in low-rate months, rising to 0.45–1.1%/yr
  in high-rate months, with the published fee and the measured tenor and TED spreads
  already removed."
* **A constant depository margin is ruled out** as the explanation. That is a real
  elimination, and it was the most natural remaining candidate.
* The **fee term is confirmed exactly** at the published 0.40%/yr, and the **US TED term is
  confirmed real** at ~0.28%/yr with the sign that works against the hypothesis.

**The next hypothesis, named but deliberately NOT fitted here:** a **rate-proportional**
depository margin — a bank that keeps a fraction of the rate rather than a fixed spread
would produce precisely the regime shape left behind. Fitting that to this same data after
seeing this result is the overfitting this repo exists to avoid, and it would be an
unregistered trial. It should be pre-registered on its own terms, with its own control,
and it must beat the null that the remainder is simply tracking error plus the unmodelled
premium/discount noise of three small currency trusts.

Until then the honest statement is unchanged from the convention repair, only sharper:
**the FX leg's verification rests on JPY, and EUR/GBP carry a residual whose rate-scaling
component nobody has yet explained.**
