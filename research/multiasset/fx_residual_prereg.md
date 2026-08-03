# PRE-REGISTRATION — explaining the EUR/GBP FX residual left open by the convention repair

**Written 2026-07-31, BEFORE any decomposition was computed and before any adjusted
residual existed.** Every source, window, tolerance and pass/fail rule below was fixed in
advance. The measured numbers in `convention_repair_result.md` §1 (0.94 / 0.86 / 0.496
%/yr and the six regime cells) are the only quantities known at writing time; they are
quoted here so that this document cannot later be re-read as if it had predicted them.

Governing instruction: `convention_repair_result.md` §7 item 3 — *"The EUR and GBP FX
residual (~0.9%/yr) is unexplained and its registered explanation was refuted."*
This document is the attempt to explain it.

Standing method rules obeyed: **rule 9** (build the control that the old model must
fail), **rule 10** (bracket, do not point-estimate, where the decomposition is
unobservable), **rule 11** (persist dated series, declare what the index means).

---

## 0. What is open, in one paragraph

After the convention repair, `EURUSD` and `GBPUSD` sit **0.94%/yr** and **0.86%/yr**
above their currency-ETF benchmarks, outside the registered 0.75%/yr budget; `JPYUSD`
sits at 0.496%/yr, inside it. The registered explanation was that a currency-deposit ETF
earns nothing in a zero-rate era while its fee keeps accruing, so the residual should
concentrate in **low**-rate months. Measured, the residual is **larger in normal-rate
months** (EUR 1.394% vs 0.477%; GBP 0.910% vs 0.552%; JPY 0.557% vs 0.478%). That
hypothesis is refuted and stays refuted. This document registers a different one.

---

## 1. The claim — the refuted hypothesis was mis-specified, not merely wrong

The old hypothesis bundled the sponsor's fee (constant in every regime) together with
the interest the trust fails to earn (which can only exist when there **is** a rate to
earn), and then predicted the *sum* would concentrate in low-rate months. Only the second
component behaves that way, and it enters with the opposite sign to the one assumed.

Write the measured residual exactly as the code computes it
(`run_convention_repair.py`, Control C):

```
diff_t = fx_excess_t − (ETF_ret_t − cash_t)
fx_excess_t = spot_t + (i3m_foreign − i3m_US)_{t−1} / 12
```

and substitute what a CurrencyShares trust actually returns,
`ETF_ret_t = spot_t + earned_foreign_t − fee/12`:

```
diff_t = [ i3m_foreign/12 − earned_foreign_t ]      (A) foreign leg over-credit
       + fee/12                                     (B) sponsor's fee
       − [ i3m_US/12 − cash_t ]                     (C) US leg: minus the TED spread
```

**The residual is a benchmark-construction artefact, and its three parts are separately
identifiable.** The panel credits a **3-month interbank** differential on both legs — by
design, so that no maturity or basis is mixed *inside* the differential
(`build_carry_inputs.py` docstring). The benchmark breaks that symmetry: its foreign leg
earns an **overnight deposit rate less an unpublished margin**, and its US leg is a
**government bill**. Terms (A) and (C) are the two halves of that broken symmetry, and
both scale with the rate level — which is precisely why the residual is larger in
normal-rate months and why the old fee-only story could not produce that.

### What is published, what is measured, what is left over

| term | status | value |
|---|---|---|
| **(B)** sponsor's fee | **published**, not fitted | 0.40%/yr for FXE, FXB, FXY; accrued daily, paid out of interest |
| **(A₁)** foreign tenor/credit premium | **measured** from OECD data | `IR3TIB01 − IRSTCI01` per currency |
| **(C)** US TED spread | **measured** | `IR3TIB01_US − cash`, the panel's own bill accrual |
| **(A₂)** depository margin + zero floor | **unobservable — deliberately not published** | reported as the **remainder**, never as an input |

The 10-K language confirms (A₂) exists and confirms it is not disclosed: the Depository
"may earn a spread or margin over the rate of interest it pays to the Trust", and FXB's
rate in effect at 2021-12-31 was **0.00%** — the zero floor is real, not assumed. Because
its magnitude is unpublished, **no value for (A₂) is assumed anywhere in this test.** It
is what is left after (A₁), (B) and (C) are removed. A test that fitted (A₂) could not
fail, and would be worthless.

---

## 2. Sources — probed 2026-07-31, before this document was finished

| series | id | coverage probed |
|---|---|---|
| overnight, euro area | `IRSTCI01EZM156N` | 385 obs, 1994-01 → 2026-01 |
| overnight, UK | `IRSTCI01GBM156N` | 582 obs, 1978-01 → 2026-06 |
| overnight, Japan | `IRSTCI01JPM156N` | 492 obs, 1985-07 → 2026-06 |
| overnight, US | `IRSTCI01USM156N` | 864 obs, 1954-07 → 2026-06 |
| 3-month interbank | `IR3TIB01*M156N` | already in `_data/carry/short_rates_monthly.parquet` |

Free, keyless, same FRED/OECD family and the same fetch path the repo already uses
(`fetch_fred`). **No new vendor, no paid feed, no new cleaning code.** The overnight
family is the exact tenor counterpart of the 3-month family already in use, which is why
it was chosen over EONIA/€STR/SONIA/TONA splices — those would mix sources inside a
spread and reintroduce the error this test exists to measure.

---

### AMENDMENT 2026-07-31, made BEFORE any decomposition was run — transport, not source

FRED became **unreachable from this machine** partway through: every path on
`fred.stlouisfed.org` times out, including the site root, while the TLS handshake
succeeds — an Akamai IP-level block triggered by a burst of requests, not a data problem.
Browser-realistic headers did not lift it.

The overnight series are therefore taken from **OECD directly** (`sdmx.oecd.org`,
dataflow `OECD.SDD.STES,DSD_STES@DF_FINMARK`, `MEASURE=IRSTCI`, *"Immediate interest
rates, call money, interbank rate"*, `UNIT_MEASURE=PA`, `FREQ=M`) for `EA20`, `GBR`,
`JPN`, `USA`. **OECD is the publisher of these series; FRED only mirrors them.** The
registered scientific choice — the OECD immediate-rates family as the exact tenor
counterpart of OECD `IR3TIB01` — is unchanged. What changed is which pipe it arrives
through, and that is not a research degree of freedom.

**It is verified rather than asserted.** Against the FRED values recorded here before the
block: `GBR` n=582 1978-01→2026-06 last 3.7298, `JPN` n=492 1985-07→2026-06 last 0.841,
`USA` n=864 1954-07→2026-06 last 3.625172 — **all three match exactly**. `EA20` returns
390 months to 2026-06 against FRED's 385 to 2026-01; **390 − 5 = 385**, i.e. OECD is five
months fresher and identical on the overlap (2026-01 = 1.931671 both ways).

Independently, OECD's **`IR3TIB` 3-month** series was compared against the repo's existing
FRED-sourced `short_rates_monthly.parquet`: **GB, JP and US are byte-identical
(max |diff| = 0.000e+00)**; EZ differs by at most **1.24e-04** with mean **1.04e-06**,
which is precision/revision noise. The 3-month legs therefore keep coming from the repo's
existing parquet so that `diff` still reproduces Control C exactly (P5), and only the
overnight leg is new. For EZ that leaves at most ~0.0001 pp/yr of transport contamination
in the spread, against a residual of ~0.9 pp/yr — four orders of magnitude below the
quantity being measured, and disclosed rather than buried.

This amendment is recorded **before any remainder, asymmetry or verdict existed**, so it
cannot have been chosen to suit an outcome. Nothing in §3 below is altered.

## 3. Registered predictions — fixed now, in this order of authority

Let `remainder_t = diff_t − (A₁ + B − C)_t`, annualised the same way Control C
annualises, over the identical windows Control C used.

**P1 — the regime asymmetry must collapse. (PRIMARY.)** This is the direct repair of the
refuted diagnostic and the reason to run at all. Let `A = |gap_high − gap_low|`.
Currently **EUR 0.917 pp, GBP 0.358 pp, JPY 0.079 pp**. Registered: after adjustment,
**A ≤ 0.25 pp/yr for EUR and for GBP**. The bar is set at roughly the scale of the leg
that already passes, not at a level chosen to be reachable. Falsified if either exceeds
0.25 pp, and **refuted outright if either asymmetry widens**.

**P2 — the level must come inside the registered budget.** The adjusted remainder for
EUR and for GBP each falls inside **TOL_FX = 0.75%/yr** (currently 0.94 and 0.86, both
outside).

**P3 — cross-currency consistency.** The three adjusted remainders lie within
**0.35 pp/yr** of one another. A single depository operating one contractual structure
across three trusts should leave a similar margin on each. *Informative, not decisive* —
it is the weakest link because (A₂) genuinely may differ per currency.

**P4 — do no harm. (CONTROL.)** JPY's adjusted remainder must stay **inside 0.75%/yr**.
JPY is the leg that already passes; an adjustment that pushes it out is over-correcting,
and that is a failure of this model, not a discovery.

**P5 — null control.** With `fee = 0` and both rate spreads forced to zero, the pipeline
must reproduce the published residuals **0.940 / 0.860 / 0.496** to within 0.01 pp. If it
does not, the decomposition is not measuring the same quantity Control C measured and
nothing else in this document may be believed.

**P6 — sign discipline. (INTEGRITY GATE.)** `mean(i3m − i_overnight) ≥ 0` for each of EZ,
GB, JP, US over the sample. A 3-month interbank rate below overnight on average would
mean the two series have been paired wrongly. If P6 fails the run is **void** — report
the pairing error, change nothing else.

### Decision rule, fixed now

The residual is declared **explained** only if **P1 and P2 and P4** all hold, with P5 and
P6 having passed as preconditions. P3 is reported either way and cannot rescue a failure.
If P1 holds but P2 does not, the honest finding is **"the mechanism is identified but does
not close the gap"** — that is a partial result and will be written as one, not rounded up.

---

## 4. Bracket, because (A₂) is unobservable — method rule 10

A point estimate of the remainder would imply a precision the data does not support. The
reported remainder is bracketed over three defensible constructions of the foreign leg,
all registered now:

1. **Overnight, as published** — `IRSTCI01` unmodified.
2. **Zero-floored** — `max(0, IRSTCI01)`, since the trust cannot be credited a negative
   deposit rate it did not pay (FXB at 0.00% is the evidence this case is real).
3. **Fee-first** — the sponsor's fee is paid out of interest before the holder sees any,
   so the trust's net credit is `max(0, IRSTCI01 − fee)`.

The headline is construction 2. All three are reported. If they disagree by more than
0.25 pp/yr the disagreement itself is the finding and the residual is declared bracketed,
not explained.

---

## 5. What this can and cannot change — fixed in advance

**It changes no panel series, no strategy, no gate and no headline number.** The panel
already credits the interest differential; this document tests whether the *benchmark*
used to verify it is a fair yardstick. Nothing here can move the corrected book Sharpe of
**0.7834**, and any later text implying it did is wrong.

* **If explained**: the FX correction's verification extends from the JPY leg alone to
  all three legs, and `convention_repair_result.md` §7 item 3 stops being an open thread.
* **If not explained**: the residual stays open and is better bounded than it was. That
  is still a result and will be reported as the primary outcome, not buried under P3.
* **Either way**: the cross-currency basis disclosed in `carry.fx_excess_returns`
  (10–50 bps/yr, post-2008) remains a genuine panel-side error term that this test does
  **not** address and does not claim to have retired.

One further honest limit, registered now: the ETF total returns are market-price returns,
not NAV returns, so trust premium/discount noise sits inside `diff` and is not modelled.
It is mean-reverting and should not bias an annualised mean over 200+ months, but it
inflates dispersion and no confidence interval below should be read as if it were absent.
