# RESULT — is the FX residual's regime asymmetry real? **INCONCLUSIVE BY ITS OWN CONTROLS.**
## But the power analysis closes the thread anyway, and that is the finding.

Pre-registration: `fx_shape_reality_prereg.md`, committed at `f8a10a2` with **no
statistic attached**, plus one openly-recorded threshold correction at `403f8b6`,
also before any run. Reproduce with:

    python scripts/run_fx_shape_reality.py --use-cache --draws 10000

Result JSON: `_fx_residual/fx_shape_reality.json`. **Nothing here touches the panel,
the ledger's selection path, the live path, or 0.7834.**

---

## 1. The registered verdict is NOT READABLE. Say that first.

The decision rule returned **ARTEFACT** (p = 0.231 circular shift, p = 0.490 block
bootstrap, both >= 0.10). **It is withheld**, because the pre-registration made the
controls a gate on reading it, and **C1 POWER FAILED**:

| control | result | passed |
|---|---|---|
| **C0** rebuild reproduces the committed remainders | worst diff **0.0005 pp** vs 0.02 tolerance | ✅ |
| **C1 POWER** inject a real 0.5%/yr effect, must return REAL | returned **UNDETERMINED** | ❌ |
| **C2 SIZE** matched-variance noise, must not return REAL | ARTEFACT | ✅ |
| **C3 DETERMINISM** two runs identical | identical | ✅ |

The prereg said it plainly: *"A test that cannot see a real effect is toothless, and
reporting ARTEFACT from a toothless test would be worse than useless."* An injected,
genuinely real 0.5%/yr asymmetry raises S to 1.0471 and **still fails to clear both
nulls.** So ARTEFACT here means "this test cannot tell", not "there is no effect".
**The trial is spent and the registered hypothesis is undecided.**

## 2. Why it is underpowered — and why that IS the result

The measurement floor is bigger than the thing being measured.

| quantity | %/yr |
|---|---|
| **observed** pooled asymmetry S | **0.5471** |
| circular-shift null, **median** | **0.4201** |
| circular-shift null, 95th percentile | **0.7039** |
| block-bootstrap null, 95th percentile | 1.0329 |

Per leg: EUR **0.7639**, GBP **0.5728**, JPY **0.3045** (n = 242 / 235 / 232 months).

**Take the same residual series and slide it against the regime labels at a random
offset, and it still shows a typical asymmetry of 0.42%/yr.** The observed 0.547 is
about 1.3x that, and sits *below* the null's own 95th percentile. Sliding a series
changes nothing about it except which months are called "high-rate" — so a
substantial part of the shape that two registered tests set out to explain is
produced by the alignment alone.

This is a statement about the data, not about any hypothesis, and it needs no
verdict: **an asymmetry of this size is not resolvable in 239 months of this
residual, by this test or by any mechanism-fitting test on the same series.** That is
why a constant margin and a rate-proportional margin both failed to explain "the
shape". There may be no measurable shape to explain.

## 3. What this does and does not license

**Licensed.** Stop spending registered trials on the *shape* of this residual until
there is more data or a lower-variance construction. The two prior negatives are now
explained without needing either mechanism to be wrong: the target was under the
noise floor the whole time.

**NOT licensed.** This does **not** say the asymmetry is zero, and it must never be
quoted as "the residual has no regime structure". A test that cannot detect a real
0.5%/yr effect cannot rule one out either. The honest statement is *undetermined,
and undetectable at this sample size.*

**Unchanged.** The residual's **level** still stands and still transfers: a depository
margin of ~0.2-0.5%/yr across all six trusts, out-of-sample level error 0.634 -> ~0.22
pp. `convention_repair_result.md` §7 item 3 stays open as a level, not a shape.

## 4. Why the test came out underpowered, recorded so it is not repeated

The two registered nulls are not equally sharp, and pairing them cost the test its
teeth:

- **N1 circular shift** is the right null. It preserves the series exactly and
  destroys only the alignment, which is precisely the hypothesis.
- **N2 stationary block bootstrap** resamples **with replacement**, so it destroys
  alignment *and* re-composes the series. Its null is far wider (95th percentile
  1.0329 vs 0.7039), and requiring agreement between the two made the joint test
  roughly as weak as the weaker one.

Registering "both must agree" looked conservative and was, in the direction that
mattered least. **Lesson for the next registration: pair nulls only when they test
the same thing at comparable sharpness; otherwise pre-commit which is primary and
report the other as a diagnostic.**

**A limitation of the shift null, found by its own test and pinned in the suite.**
If the regime labels were *periodic*, a circular shift would either preserve the
period or flip it, `|asymmetry|` would be invariant, and the null would collapse to a
single value — a meaningless test that would look like overwhelming significance.
Real rate regimes are long runs, not alternations, so the study is unaffected; but
`test_circular_shift_null_is_degenerate_for_periodic_labels` records it so a future
reuse on periodic labels cannot rediscover it as a mystery. It is also the reason the
long low-rate run of 2009-2021 makes this null wide rather than narrow: shifting a
series against long regime blocks moves a lot of mass between regimes.

## 5. Integrity notes

- **C0 is the loader's own null control.** The frames are rebuilt here rather than
  refactored out of reviewed research code, so a loader mistake would surface as a
  reproduction failure rather than a wrong verdict. It reproduces the committed
  remainders (EUR 0.743 / GBP 0.490 / JPY 0.216) to **0.0005 pp**, and the per-leg
  asymmetries match `fx_residual_result.md`'s post-correction 0.764 / 0.573 exactly.
- The threshold description in the prereg was corrected **before** the run and in the
  open (`403f8b6`): the committed rule splits on the foreign **3-month** rate at 0.5%,
  not the overnight rate. The binding text was always "the committed threshold already
  in the module", so nothing about the test changed.
- **NAV was never obtainable**, and would not have helped: for a currency trust NAV is
  (deposit per share) x spot with the deposit evolving by (interest earned - fee),
  which is the model under test, so it returns the margin by construction. The
  premium/discount candidate needs price-minus-NAV, which is bounded and
  mean-reverting and contributes about (P_end - P_start)/T to an annualised mean.

## 6. Trial accounting

**Third registered test on this residual. Spent, undecided.** Recommendation: do not
register a fourth on the shape. Register one on the *level* only if a new trust family
or a longer sample becomes available.
