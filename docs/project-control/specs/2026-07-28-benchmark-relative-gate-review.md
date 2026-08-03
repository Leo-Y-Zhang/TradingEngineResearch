# Benchmark-relative selection criterion — ADVERSARIAL REVIEW

**Date:** 2026-07-28 · **Type:** REVIEW, not an implementation. No code was changed.
**Subject:** the absence of any benchmark-relative criterion in
`research/validation.py::selection_rule`.
**Prior art:** `specs/2026-07-27-portfolio-gate-amendment-REVIEW.md` (verdict: REJECTED —
the amendment targeted a non-binding constraint and calibrated a threshold below its own
estimator's standard error). This review is written to be refutable on the same grounds and
therefore leads with the evidence that would refute it.

**VERDICT: ADOPT-WITH-CONDITIONS**, and with an honest headline that cuts against the
proposal: **on this programme's actual measured results, the criterion flips ZERO outcomes
of the real seven-criterion gate.** It is worth adopting anyway, for a reason that is
measured rather than asserted (§3.6), but anyone who wants to reject it on
"non-binding-constraint" grounds has a real case and it is stated in full in §2.4.

New measurement written for this review: `research/_gate_review/sharpe_difference_power.py`
→ `research/_gate_review/sharpe_difference_power.json` (seed 20260728). Every number below
tagged **[M]** was produced by that script on this machine today. Numbers tagged **[R]** are
quoted from committed run artefacts with file:line. Nothing here is estimated.

---

## 1. THE DEFECT, ESTABLISHED AT SOURCE

### 1.1 The code

`research/validation.py`, `def selection_rule(result: ValidationResult) -> bool` at **line
410**, body lines **425–476**.

> *Correction to the task brief and to `internal research log:2193`: the range is 408–476 (section
> comment at 408, `def` at 410, final `return True` at 476), not 401–467. The "401–467"
> figure is stale by ~9 lines and should stop being propagated.*

The seven checks, verbatim from source:

```python
    if result.mean_rank_ic <= 0.01:                       # 427
    if result.sharpe_net <= 0.75:                         # 432
    if result.stability_score <= 0.60:                    # 437
    if result.deflated_sharpe_proxy <= 0.25:              # 442
    if result.deflated_sharpe_ratio < 0.95:               # 448
    if result.leakage_flags:                              # 454
    for regime, metrics in result.regime_breakdown.items():   # 460
        if regime_sharpe < -0.50:                         # 462
```

**All seven compare a candidate statistic to a hard-coded constant.** Not one of them
references a benchmark, a passive alternative, an opportunity cost, or the candidate's own
universe. `ValidationResult` (lines 29–49) has no field that could carry a benchmark: its
ten scalars and two containers are all own-strategy quantities. The gate is therefore
**structurally incapable** of expressing "better than doing nothing" — this is not a
calibration problem, it is a missing input.

Two incidental defects found while establishing this, both live:

- **The docstring is wrong.** Lines 414 and 417 say "ALL six conditions" / "Conditions (from
  build spec Part 6.3)" and then list six. The code applies **seven** (the real DSR at 448 is
  not in the list). The same error is replicated in the internal build-instructions document
  ("returns `False` correctly when any of the six conditions fails").
- The regime check at 460 iterates `result.regime_breakdown.items()`, so an **empty**
  `regime_breakdown` passes vacuously. This is the same class of defect as GATE-6 in the
  prior review and is unchanged.

### 1.2 `sr_benchmark` is accepted and never supplied — verified exhaustively

`deflated_sharpe_ratio(returns, n_trials=1, sr_benchmark=None)` at **line 315** takes a
benchmark Sharpe. When `sr_benchmark is None` (line 357) it **manufactures one from the
multiple-testing deflation** (lines 358–365): `SR* = sigma * [(1-gamma)*z1 + gamma*z2]`. That
substitute benchmark is a *noise threshold*, not an *alternative investment*.

I grepped every occurrence of `deflated_sharpe_ratio(` in the repository, excluding the
definition itself and this review's own measurement script. **34 call sites at HEAD
`e0d517d`. Zero pass `sr_benchmark`** — all 34 pass `n_trials=` only, verified by grepping
the two together (the only two lines in the repo containing both tokens are the `def` at
`validation.py:315` and a dict key in this review's script). The 34 span
`research/alpha_factory.py:244,863`, seven `scripts/research_*.py`, thirteen
`research/sleeves/**`, and the tests. **There is no code path anywhere in this repository
that gives this function a real benchmark.**

Consequently `selection_rule` cannot supply it even in principle: it never *calls*
`deflated_sharpe_ratio`. It reads a **pre-computed float field**
(`result.deflated_sharpe_ratio`) at line 448. Wiring a benchmark into the gate is not a
matter of passing an argument — it requires a new field on `ValidationResult` and a new
producer for it.

### 1.3 Every caller of `selection_rule`

| # | Call site | What it gates | Live? |
|---|---|---|---|
| 1 | `research/alpha_factory.py:506` (`promote_factor`) | factor promotion into `_LIVE_FACTOR_MATRIX` | research → live library |
| 2 | `research/alpha_factory.py:630` (`promote_candidates`) | batch promotion; sets `PromotionOutcome.passed_selection_rule` | research → live library |
| 3 | `ops/model_registry.py:88` (`promote`) | **raises** if a model fails; the golden-rule-5 barrier to LIVE | **LIVE** |
| 4 | `ops/model_registry.py:138` (`promotion_candidate`) | whether the shadow model is eligible for promotion | **LIVE** |
| 5 | `learning/adaptive_weights.py:109` (`propose_and_validate`) | whether new **sleeve weights** are applied | **LIVE** |

The prior review's GATE-4 finding stands and I confirm it independently: sites 3–5 are live
paths, and a change to `selection_rule` propagates to all of them. **A benchmark-relative
criterion added naively to `selection_rule` would apply to model promotion and to sleeve
reweighting, where "the benchmark" is undefined.** In `adaptive_weights` the object being
gated is *a weight vector*, not a return stream; there is no candidate series to compare to
anything. Adding a criterion that requires a benchmark to a rule that is also asked about
weight vectors is a type error waiting to be a silent one — because
`ValidationResult.deflated_sharpe_ratio` **defaults to 1.0** (line 45), any new
benchmark-relative field with a permissive default would *default to pass* on exactly these
paths (this is GATE-2, still unfixed as of `e0d517d`).

### 1.4 Status of the three preconditions the prior review demanded

Checked at HEAD `e0d517d`, not assumed:

| Defect | Status | Evidence |
|---|---|---|
| GATE-1 `n_trials` = fold count | **STILL LIVE** | `research/alpha_factory.py:233` `n_trials = max(len(splits), 1)`, feeding both the proxy (`:234`) and the real DSR (`:244`) |
| GATE-2 `deflated_sharpe_ratio` defaults to 1.0 = pass | **STILL LIVE** | `research/validation.py:45` |
| GATE-3 PBO never checked | **CLOSED 2026-07-31** | `selection_rule` condition 6 now rejects `pbo_proxy >= PBO_MAX` (0.50 — the failure point the CSCV estimator's own docstring names, not a tuned value). Regressions plus an estimator positive control (best-of-8-noise vs a genuinely dominant config) in `tests/test_gate_preconditions.py` |

### 1.5 The finding that reframes everything: the gate was never run

`ValidationResult(` is constructed in exactly **7 files**: `research/alpha_factory.py`,
`ops/persistence.py`, and five test modules. **No sleeve study built one.** No script in
`research/sleeves/**` or `scripts/research_*.py` calls `selection_rule`.

So the claim "the gate PASSED trend" is, strictly, false. What the overnight run did was
apply **criterion 5 alone, in inverted form** — solving DSR = 0.95 for the Sharpe that
achieves it ("the DSR bar") — plus a *human-specified* matched-volatility comparison that
exists in no gate. The other six criteria were never evaluated because the quantities they
need (rank IC, stability score, leakage flags, regime breakdown) were never computed for any
sleeve.

This matters for §2 and I do not want it buried: **the near-misses catalogued below are
near-misses of one criterion applied by hand, not of `selection_rule`.**

---

## 2. THE CONSEQUENCE, QUANTIFIED ON THIS PROGRAMME'S ACTUAL RESULTS

### 2.1 The table

Every sleeve/arm measured in the 2026-07-28 run. "DSR-crit" = criterion 5 as the run applied
it. "Beats bench?" = vol-matched active > 0, the run's standing rule
(`internal research log:579-581`), which is algebraically identical to the Sharpe comparison
(`volmatched_active = bench_vol × Sharpe_gap`, verified to 1e-16 in the run,
`2026-07-28_RESEARCH_RUN_ACCOUNTING.md:332-334`). All figures **[R]** unless marked.

| # | Sleeve / arm | T (mo) | Net SR | DSR bar (n) | DSR-crit | Benchmark | Bench SR | Bench DSR | VM active %/yr (t) | Beats bench? | **Flip?** |
|---|---|---:|---:|---:|:--:|---|---:|---:|---:|:--:|:--:|
| 1 | multi-asset trend, 20% vol, 10bps | 738 | 0.6116 | 0.4863 (36) | **PASS** | passive EW, same 18 | **0.6691** | 0.9964 (n47) | −0.52 (−0.31) | **NO** | **YES** |
| 2 | risk parity, naive | 738 | 0.6483 | 0.4988 (46) | **PASS** | equal weight, same 18 | **0.6678** | clears | −1.35 (−2.67) | **NO** | **YES** |
| 3 | risk parity, bucketed | 738 | 0.5513 | 0.4988 (46) | **PASS** | equal weight, same 18 | **0.6678** | clears | (negative) | **NO** | **YES** |
| 4 | seasonal Halloween, 10bps | 737 | 0.5868 | 0.4808 (32) | **PASS** | passive daily EW | **0.7021** | clears | −2.53 (−0.95) | **NO** | **YES** |
| 5 | seasonal turn-of-month, **2bps** | 737 | 0.6455 | 0.4808 (32) | **PASS** | passive daily EW | 0.6996 | clears | −1.07 (−0.39) | **NO** | **YES** |
| 6 | seasonal composite, **2bps** | 736 | 0.6231 | 0.4808 (32) | **PASS** | passive daily EW | **0.7067** | 0.7065 clears | −2.06 (−0.64) | **NO** | **YES** |
| 7 | seasonal turn-of-month, 10bps | 737 | 0.2455 | 0.4808 | FAIL | passive daily EW | 0.6993 | clears | −8.98 (−3.29) | NO | no |
| 8 | seasonal composite, 10bps | 736 | 0.4680 | 0.4808 | FAIL | passive daily EW | 0.7065 | clears | −5.86 (−1.84) | NO | no |
| 9 | seasonal January, 10bps | 737 | 0.3511 | 0.4808 | FAIL | passive daily EW | 0.7199 | clears | −13.04 (−2.26) | NO | no |
| 10 | defensive / BAB, 20% vol, 10bps | 629 | 0.1136 | 0.5303 (38) | FAIL (DSR **0.0887**) | own EW universe | 0.6898 | **0.9942** | −5.22 (−2.63) | NO | no |
| 11 | cross-asset value, 10bps | 533 | −0.0824 | ~0.57 | FAIL | own long-only | **0.7551** | — | −8.99 (−3.84) | NO | no |
| 12 | cross-asset carry, 3bps | 269 | 0.4301 | 0.8135 (36) | FAIL | own EW, 13 instr | 0.0293 | — | +1.60 (+1.22) | **YES** | no¹ |
| 13 | low-vol B2 (registered) | 213 | 0.8779 | 0.9234 (38) | FAIL (DSR 0.8736) | own EW universe | 0.3744 | 0.2637 | +7.37 (+2.64) | **YES** | no¹ |
| 14 | low-vol B3 / B4 / B5 | 213 | 0.680 / 0.682 / 0.712 | 0.9234 | FAIL | own EW universe | ~0.36 | ~0.24 | +4.1 / +4.1 / +3.8 (t<1.6) | yes (weak) | no¹ |
| 15 | PEAD re-test 40d, realistic | 212 | 0.4227 | 0.911 (33) | FAIL | own EW universe | 0.1492 | — | −0.17 (−0.03) | NO | no |
| 16 | reversal re-test, weekly | 921w | −0.8535 | 0.917 (34) | FAIL | own EW top decile | 0.2716 | — | −29.4 | NO | no |
| 17 | TSMoM multi-timeframe | 213 | −0.3682 | — | FAIL | own EW universe | 0.1755 | — | −8.79 | NO | no |
| 18 | lowvol+trend pair, corrected v2 | 213 | 0.9212 | 0.9443 (47) | FAIL (DSR 0.9333) | passive monthly | 0.4667 | 0.3588 | +5.39 (+1.62)² | yes² | no¹ |
| 19 | **trend + passive (SURVIVOR)** | 738 | **0.9033** | 0.4988 (46) | **PASS** | passive monthly | 0.6691 | 0.9965 | **+2.11 (+2.34)** | **YES** | **no** |

¹ Fails the absolute criterion, so a *conjunctive* relative criterion changes nothing. Would
flip if the relative criterion **replaced** the absolute one — see §3.5.
² Point estimate only. Excluding both bear markets the same book measures **−1.55%/yr,
t −0.38**, and passive beats it 1.1299 vs 1.0031 (`pair_deflation.json`
`corrected_v2.ex_both_bears`).

### 2.2 How many outcomes flip

**Against criterion 5 as the run actually applied it: 6 of 19 arms flip** (rows 1–6), all in
the **tightening** direction, all PASS → FAIL. **Zero loosen.** Distinct sleeves affected:
trend, risk parity (2 variants), seasonal (3 arms). Every one of those six is a book that
would have been carried forward as "clears the deflated-Sharpe gate" while losing to buying
and holding the same instruments.

The survivor (row 19) is the only book that passes both the absolute and the relative test.
That is the single strongest argument for the criterion: it is the difference between
"one of seven books survived" and "one of seven books survived and the other six were
already known to be worse than nothing."

### 2.3 Against the *real* gate, the answer is zero

Now the honest version. `selection_rule` also requires `sharpe_net > 0.75` (line 432). Apply
it to the six flippers:

| Row | Net Sharpe | > 0.75? |
|---|---:|:--:|
| 1 trend | 0.6116 | **no** |
| 2 risk parity naive | 0.6483 | **no** |
| 3 risk parity bucketed | 0.5513 | **no** |
| 4 seasonal Halloween | 0.5868 | **no** |
| 5 seasonal TOM 2bps | 0.6455 | **no** |
| 6 seasonal composite 2bps | 0.6231 | **no** |

**All six are already rejected by criterion 2.** The survivor (0.9033) and the pair-as-
claimed (1.2166) clear criterion 2 *and* beat their benchmarks. Therefore:

> **Under the full seven-criterion `selection_rule`, a benchmark-relative criterion would
> have flipped ZERO of the nineteen outcomes measured in this programme.**

This is precisely the failure mode that killed the 2026-07-27 amendment. I am reporting it
against my own proposal because it is true and because a reviewer would find it.

### 2.4 Why I still recommend adoption — and the measurement that justifies it

The obvious rebuttal is "criterion 2 already does the job". It does not; it *coincides* with
the job on this sample, and the coincidence is measurably fragile.

`sharpe_net > 0.75` is a **fixed constant**. It catches these six sleeves only because the
programme's passive benchmark happens to score **0.669** over 1965–2026 — below 0.75. The
run's own subperiod table (`internal research log:497-500`) measures the same benchmark on
deployable-era subsamples:

| era | benchmark Sharpe | above the 0.75 constant? |
|---|---:|:--:|
| full 61.5y | 0.669 | no |
| pre-2009 (44.0y) | 0.627 | no |
| **2009+ (17.5y)** | **0.777** | **YES** |
| **2015+ (11.5y)** | 0.739 | borderline |
| rolling 10y to 2026-06 | **0.811** | **YES** |

On the 2009+ window — *the deployable era, and the one any future study will use* — a
candidate measuring Sharpe 0.78 clears criterion 2 while losing to buy-and-hold at 0.777.
Criterion 2's protection is an accident of the 1965-start sample and it has **already
expired** on the recent data. The daily-rebalanced passive book scores **0.7065**
(`pair_deflation_result.md:132`), 4.4 basis points of Sharpe from breaching the constant on
the *full* sample.

That is the case for adoption: not "it changes outcomes today" (it does not), but "the thing
currently doing this job is a constant that the benchmark has already overtaken on the most
relevant subsample."

The defensive sleeve (row 10) is the clean diagnostic in the other direction: the gate
**rejected** it at DSR 0.0887 while its own passive benchmark scores DSR **0.9942**
(`_defensive/result.json:476,498`). The gate got the *decision* right and the *reason*
wrong — it asked "is this Sharpe distinguishable from luck", never "is this better than the
alternative". A gate that is right by accident twice in a row is still a gate that is not
measuring what the programme cares about.

---

## 3. THE PROPOSAL, AND THE ATTACK ON IT

### 3.1 Form A (the obvious one) — REFUTED BY MEASUREMENT

> *"The candidate's DSR must exceed the benchmark's DSR computed identically."*

**Do not adopt this form.** Two independent measurements kill it.

**A1. The multiple-testing deflation cancels exactly, so the comparison inherits none of
it. [M]**

`DSR = Phi((SR − sigma·k(n))/sigma) = Phi(SR/sigma − k(n))`. When both legs are deflated at
the same `n`, `k(n)` is common and drops out under the monotone `Phi`. So
`DSR_a > DSR_b  ⟺  SR_a/sigma_a > SR_b/sigma_b` — a comparison of two *unpaired,
moment-adjusted t-statistics*, with **zero** deflation.

Measured on the real trend/passive pair using the repo's own `deflated_sharpe_ratio`
(`Q2_dsr_deflation_cancels`): the sign of `DSR_candidate − DSR_benchmark` is **identical at
n_trials = 1, 2, 26, 34, 46, 47, 100, 281, 1000 and 10000** — `sign_invariant_across_all_n:
true`. Raising the trial count from 1 to 10,000 moves the *margin* from −4.5e-07 to −0.0366
but never the *verdict*. A criterion built on this ordering is therefore immune to the trial
ledger, which is the programme's principal defence against selection bias. The prior review
rejected the last amendment partly because its statistic "carries no deflation and accrues
no trials". Form A has exactly that property, and I have now measured it rather than argued
it.

**A2. A strictly worse strategy can win the DSR comparison on return shape alone. [M]**

`sigma_SR^2 = (1 − g3·SR + (g4−1)/4·SR^2)/(T−1)`. Positive skew (`g3 > 0`) *shrinks*
`sigma_SR` and *raises* DSR. So a positively skewed weak candidate can out-DSR a negatively
skewed strong benchmark. Constructed and measured at T = 738 (`Q3_dsr_shape_reversal`):

| | annualised Sharpe | DSR (n=47) |
|---|---:|---:|
| candidate (positive skew) | **0.6728** | **0.99999** |
| benchmark (negative skew) | **0.6928** | 0.9742 |

The candidate is **worse by 0.020 of Sharpe** and **wins the DSR comparison by 0.026 of
probability**. Form A would promote it. A criterion whose stated purpose is "must beat the
benchmark" and which can be passed by a strategy that does not beat the benchmark is not fit
for the purpose.

**A3. It is confounded by sample length.** `SR/sigma` scales with `sqrt(T−1)`. Comparing a
213-month candidate to a 738-month benchmark rewards the benchmark for existing longer.
Measured in the run's own artefacts: passive scores DSR **0.9965** on its 738 months
(`portfolio_longhistory_books.json:713`) and DSR **0.3588** on the pair's 213-month window
(`pair_deflation_result.md:130`) — *the same benchmark, a 0.64 swing in "quality", purely
from window length*. Any DSR-comparison form must force identical windows, and if it does,
it is no longer measuring what its name suggests.

### 3.2 Form B (the defensible one)

> **REGISTERED CRITERION (proposed wording, §5 has the final text).** A candidate is
> benchmark-eligible only if, on the **identical observation window** as its
> **pre-registered** benchmark: (a) its net Sharpe strictly exceeds the benchmark's net
> Sharpe, computed by the same estimator with the same cost bracket and the same
> excess-return convention; **and** (b) the one-sided 90% lower confidence bound on the
> Sharpe *difference*, from a **paired** stationary bootstrap that preserves the
> candidate-benchmark dependence, is `>= 0`.

Note what (b) is *not*: it is not a threshold on a point estimate. The 2026-07-27 amendment
was rejected for setting a 0.10 hurdle against an estimator with SE 0.14–0.30. A confidence
bound cannot commit that error by construction — the noise is inside the statistic, not
compared to it afterwards. That is the single design decision that distinguishes this
proposal from the rejected one, and it is deliberate.

### 3.3 Its standard error, at the sample lengths this programme actually has [M]

Jobson-Korkie with Memmel's correction, annualised:
`Var(SR_a − SR_b) = (M/T)[2(1−rho) + 0.5(sr_a² + sr_b² − 2rho²·sr_a·sr_b)]`, `sr` per-period.
Verified against 20,000 Monte-Carlo draws under the null (`Q4_null_false_pass`): analytic SE
vs empirical SD agree to **3 decimal places at every (T, rho) tested** — e.g. T=738 rho=0.5,
analytic 0.12929 vs empirical 0.12891.

Measured on the three real candidate/benchmark pairs available on disk:

| pair | T (mo) | SR cand | SR bench | gap | **rho** | paired SE | SE if rho ignored | ratio | t | bootstrap SE | boot 95% CI of gap | one-sided p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| trend vs passive | 738 | 0.6116 | 0.6691 | **−0.0575** | **0.0051** | 0.1814 | 0.1819 | 1.00 | **−0.317** | 0.1742 | [−0.396, +0.286] | 0.625 |
| **trend+passive vs passive** | 738 | 0.9033 | 0.6691 | **+0.2342** | **0.7089** | **0.0996** | 0.1827 | **1.84** | **+2.353** | 0.1060 | **[+0.032, +0.445]** | **0.016** |
| low-vol B2, total-return | 213 | 0.6138 | 0.2306 | +0.3833 | 0.8019 | 0.1513 | 0.3372 | 2.23 | +2.533 | 0.2160 | [+0.016, +0.859] | 0.047 |
| low-vol B2, **excess of bill** | 213 | 0.4853 | 0.1478 | +0.3375 | 0.8032 | 0.1502 | 0.3366 | 2.24 | +2.247 | 0.2089 | **[−0.030, +0.788]** | **0.059** |

Three things to take from this, none of them flattering to a naive version of the proposal.

**B1. Pairing is what makes the test usable — and only sometimes.** Where candidate and
benchmark are correlated the paired SE is **1.8×–2.2× smaller** than the unpaired one. But
`rho(trend, passive) = 0.0051` **[M]** — essentially zero — so for a genuinely orthogonal
sleeve the pairing buys **nothing** (ratio 1.00) and the test reverts to the brutal
independent-samples power in B2. The programme's own most-wanted object (an uncorrelated
sleeve) is exactly the case where the relative test is weakest.

**B2. The test has almost no power against small true gaps.** Minimum detectable annualised
Sharpe gap at alpha = 0.05 one-sided, 80% power (`Q4_minimum_detectable_gap`) **[M]**:

| T (years) | rho=0.0 | rho=0.25 | rho=0.50 | rho=0.71 | rho=0.80 | rho=0.90 |
|---|---:|---:|---:|---:|---:|---:|
| 17.75 (213 mo) | **0.842** | 0.731 | 0.598 | 0.457 | 0.378 | 0.269 |
| 22.42 (269 mo) | 0.750 | 0.651 | 0.533 | 0.407 | 0.336 | 0.239 |
| 61.50 (738 mo) | **0.453** | 0.393 | 0.322 | **0.246** | 0.203 | 0.144 |

At 17.75 years against an uncorrelated benchmark the test cannot reliably detect a Sharpe
gap smaller than **0.84** — larger than most of the programme's *absolute* Sharpes. Even at
61.5 years and rho = 0, it needs **0.45**. The survivor's measured gap of **0.2342** sits
essentially *at* the 80%-power threshold for its own correlation (0.246 at rho = 0.71) — it
is detectable by the narrowest possible margin, which is why its bootstrap CI lower bound is
**+0.032**, three percent of a Sharpe point from zero.

**Honest consequence: condition (b) will return "cannot distinguish" for most candidates.**
That is not a defect to be tuned away — it is the true information content of 17–61 years of
monthly data. It must be registered as a three-way verdict (beats / loses / undetermined),
exactly like the existing `bracket_verdict` convention for cost bounds. Treating
"undetermined" as "pass" recreates the rejected amendment. Treating it as "fail" makes the
gate unpassable for orthogonal sleeves, which is the thing the programme most wants to find.
**Registered resolution: "undetermined" fails (b) but the sleeve may be carried as a
research candidate; it may not be promoted.** Conservative, and it costs the programme
nothing today because zero current sleeves are in that state.

**B3. The convention flips the verdict at short samples.** Low-vol B2 on total returns:
bootstrap 95% CI **[+0.016, +0.859]**, p = 0.047 → passes. The *same series* on excess-of-
bill returns — which is the convention
`research/sleeves/_portfolio/portfolio_correlation_v2.py:59-71` marks for this sleeve
(`"total"`, meaning cash is subtracted before comparison): CI **[−0.030, +0.788]**, p =
0.059 → **fails**. A 0.012 change in p-value across a bookkeeping choice, at T = 213. Any
adopted criterion must **register the excess-return convention** alongside the benchmark, or
it hands back the degree of freedom it was introduced to remove.

### 3.4 Does it introduce a new degree of freedom? Yes. Quantified.

**Yes, and it is the largest single objection.** The choice of benchmark is a free parameter
with a measured effect. From `portfolio_window_control.json` **[R]**, the same benchmark
family measured two defensible ways over the same instruments:

- passive **monthly** equal weight: Sharpe **0.6691**
- passive **daily** equal weight: Sharpe **0.7065**

A researcher who may pick either moves the bar by **0.037 of Sharpe** with a rebalancing-
frequency choice — and the run already recorded a mislabelling incident on exactly this pair
(`internal research log:1798-1803`: "Anyone quoting 0.7065 must label it daily-rebalanced equal
weight; the monthly figure is 0.668"). Worse, the *window* choice dominates the benchmark
choice: the same passive monthly book scores **0.6691** on 61.5 years and **0.4667** on the
low-vol 213-month window (`portfolio_window_control.json`) — a **0.20** swing, five times
the rebalancing effect, available to anyone who picks the study window after seeing results.

**Could a weak benchmark be chosen to make a candidate pass? Demonstrably yes.** Low-vol B2
beats its own-universe equal weight at +7.37%/yr, t +2.64 **[R]**. Against the *passive
monthly* book on the same 213 months (Sharpe 0.4667) the same sleeve's advantage collapses,
and the corrected pair built on it measures **−1.55%/yr, t −0.38** once both bear markets are
excluded **[R]**. Benchmark selection is not a rounding error here; it is the whole verdict.

This is why §4 makes pre-registration a **precondition, not a recommendation**. A
benchmark-relative criterion with a post-hoc benchmark is strictly worse than no criterion,
because it converts a missing test into a passed one.

### 3.5 Conjunctive or replacement?

**Conjunctive only.** If the relative criterion *replaced* criterion 5, rows 12, 13, 14 and
18 of §2.1 (carry, low-vol B2/B3/B4/B5, the corrected pair) would flip **FAIL → PASS** — five
to eight books admitted on the strength of beating a benchmark they are not statistically
distinguishable from (carry t = +1.22; B3/B4/B5 all t < 1.6). That is a large, one-directional
loosening of the gate and it is exactly the ratchet the prior review warned about. Adding the
criterion can only tighten; replacing anything is out of scope and should be explicitly
forbidden in the registered wording.

### 3.6 Steel-manning the case against adoption

The strongest argument for **DO-NOT-ADOPT** is §2.3: zero outcomes flip, the constraint is
not binding, and the 2026-07-27 amendment was rejected for exactly that. I take it
seriously. Three things distinguish this case, and I claim only these three:

1. **Direction.** That amendment *loosened* the gate; this one can only tighten it. A
   non-binding tightening has a worst case of "no effect"; a non-binding loosening has a
   worst case of "admits junk once the binding constraint moves."
2. **The incumbent protection has an expiry date that has already passed.** §2.4: the
   benchmark scores **0.777** on 2009+ and **0.811** on the trailing 10 years, both above the
   0.75 constant that is currently doing this work by coincidence.
3. **It converts a human rule into a machine rule.** Every one of the six catches in §2.2 was
   made by a person applying a matched-volatility comparison that exists in no code path.
   The programme's own log says so (`internal research log:511-513`). Rules enforced only by the
   diligence of whoever is awake at 3am are not controls.

If a reviewer rejects the criterion on ground §2.3 alone, that is a defensible reading and
the correct fallback is stated in §5.

---

## 4. WHAT WOULD HAVE TO BE TRUE FOR THIS TO BE SAFE

Nine conditions. They are preconditions, not aspirations; if any is unmet the change is not
safe to make.

**C1. Callers 3, 4 and 5 must be excluded.** `ops/model_registry.py:88,138` and
`learning/adaptive_weights.py:109` must continue to evaluate the seven absolute criteria and
nothing else. `adaptive_weights` gates a *weight vector* and has no candidate return series;
`model_registry` gates a model whose benchmark is the incumbent model, not a passive index —
a different comparison entirely. Mechanism: the relative criterion must live in a
**separate function** (e.g. `benchmark_relative_rule(result, benchmark) -> Verdict`) called
by the sleeve-promotion path only. It must **not** be added inside `selection_rule`, because
that function is shared. Any implementation that edits the body of `selection_rule` should be
rejected on sight.

**C2. No permissive default.** Whatever field carries the benchmark comparison must be
**required**, not defaulted. `ValidationResult.deflated_sharpe_ratio: float = 1.0`
(`validation.py:45`) is the cautionary example: it is a default-*allow* on a criterion the
documentation describes as default-deny, and it is still live. A `benchmark_verdict` field
defaulting to anything other than "fail" reproduces GATE-2 in a new place.

**C3. The benchmark must be REGISTERED BEFORE THE STUDY RUNS.** A committed, hash-anchored
record containing, at minimum: the benchmark's construction rule, its **rebalancing
frequency** (the 0.669-vs-0.7065 problem), its **cost bracket**, its **excess-return
convention** (the 0.047-vs-0.059 problem in §3.3-B3), its **exact observation window**, and
the **instrument set**. Registration must precede the first look at candidate results, and
the git timestamp is the evidence. The prior review verified the last amendment's
pre-registration timeline by commit order; the same standard applies here.

**C4. Identical windows, enforced not assumed.** Candidate and benchmark must be computed on
the *same* index. §3.1-A3 measured a 0.64 DSR swing on the same benchmark from window length
alone. This should be an assertion that raises, not a convention.

**C5. The test must be paired.** Ignoring `rho` inflates the SE by up to **2.24×** **[M]**
and would reject genuinely-better correlated candidates. It must also be *reported*: a
result quoted without `rho(candidate, benchmark)` is not checkable.

**C6. Three-way verdict, with "undetermined" failing promotion** (§3.3-B2), mirroring the
existing `bracket_verdict` convention. No candidate may be promoted on an undetermined
relative verdict, and no result may be reported without stating which of the three it is.

**C7. A benchmark-shopping detector, run every time.** The specific test: **recompute the
verdict against a registered PANEL of benchmarks, not one.** Minimum panel for this
programme: (i) own-universe equal weight, (ii) passive monthly EW of the 18 instruments,
(iii) passive daily EW of the same. Report the verdict against **all three**. A candidate
that beats its nominated benchmark but loses to any other member of the panel is flagged
`BENCHMARK_SENSITIVE` and cannot be promoted without an explicit written justification for
why the nominated one is the right opportunity cost. This is cheap — every one of these
series already exists on disk — and it catches the exact failure demonstrated in §3.4, where
low-vol B2 beats its own universe but the book built on it loses to passive monthly once
bear markets are excluded.

**C8. A falsification replay before adoption.** Replay all nineteen arms in §2.1 plus the
junk negative control through the proposed criterion and check the flip table reproduces
6/19 against criterion 5 and 0/19 against full `selection_rule`. If any junk control passes,
the criterion is refuted on the programme's own evidence — the same standard item 9 of the
prior review imposed.

**C9. GATE-1 and GATE-2 fixed first.** Both are still live (§1.4). GATE-1 makes the DSR
criterion ~0.14 Sharpe too lenient at 8 folds; adding a relative criterion on top of a
mis-deflated absolute one produces a gate whose behaviour nobody can predict. GATE-3 (PBO
absent) should be fixed too but is not strictly a precondition for *this* change.

---

## 5. RECOMMENDATION

### **ADOPT-WITH-CONDITIONS.**

Conditional on **C1–C9**, all nine. If C1, C3 or C9 cannot be met, the recommendation
downgrades to **DO-NOT-ADOPT** — an unregistered benchmark or a criterion wired into the
shared `selection_rule` is worse than the status quo.

**Adopt Form B. Do not adopt Form A** (`DSR_candidate > DSR_benchmark`) in any variant: it
is refuted by two measurements in §3.1 and would promote a strategy measured at 0.020 of
Sharpe *below* its benchmark.

### Exact registered wording

> **BENCHMARK-RELATIVE ELIGIBILITY (registered 2026-07-28).**
>
> A strategy sleeve is *benchmark-eligible* only if all of the following hold against a
> benchmark registered, in a committed file, **before any candidate result for that study
> was inspected** — the registration fixing the benchmark's construction rule, instrument
> set, rebalancing frequency, cost bracket, excess-return convention and observation window:
>
> 1. **Identical window.** Candidate and benchmark returns are computed on the same
>    observation index. Assert, do not assume.
> 2. **Direction.** The candidate's net Sharpe strictly exceeds the benchmark's net Sharpe,
>    computed by the same estimator under the same cost bracket. Equivalently, vol-matched
>    active return is strictly positive.
> 3. **Significance.** The one-sided 90% lower confidence bound on the Sharpe *difference*,
>    from a paired stationary bootstrap (expected block length 6 months, B >= 10,000) that
>    resamples both series on a shared index, is `>= 0`. Report `rho(candidate, benchmark)`
>    with every such result.
> 4. **Verdict.** The outcome is one of `BEATS` / `LOSES` / `UNDETERMINED`. Only `BEATS`
>    permits promotion. `UNDETERMINED` permits continued research and forbids promotion.
> 5. **Panel check.** The verdict is recomputed against every benchmark in the registered
>    panel. Any candidate that is `BEATS` against its nominated benchmark and not `BEATS`
>    against another panel member is marked `BENCHMARK_SENSITIVE` and may not be promoted.
> 6. **Scope.** This criterion applies to the **sleeve-promotion path only**. It is
>    implemented as a separate function and is **not** added to `selection_rule`, whose
>    behaviour at `ops/model_registry.py:88,138` and `learning/adaptive_weights.py:109` is
>    unchanged.
> 7. **Conjunctive only.** This criterion is added to the existing seven. It **replaces
>    nothing**, relaxes nothing, and may not be cited as grounds for relaxing any absolute
>    criterion. Cumulative `n_trials` and the `DSR >= 0.95` threshold are not amended by
>    this change and are not amendable by reference to it.

### What adoption buys, stated without inflation

- **Today: nothing measurable.** Zero of nineteen measured outcomes change under the real
  gate (§2.3). Anyone who reports otherwise is wrong.
- **Against criterion 5 in isolation — the form the run actually used — six of nineteen
  arms flip from pass to fail** (§2.2), every one of them a book that loses to buying and
  holding the same instruments.
- **Structurally:** it replaces a coincidence (a fixed 0.75 constant that the benchmark has
  already overtaken at 0.777 on 2009+) with a comparison, and converts a human 3am rule into
  a machine one.
- **It does not** rescue any rejected sleeve, does not raise any Sharpe, does not move the
  programme closer to 0.894, and has too little power (§3.3-B2) to certify small edges. It
  is a control, not a lever.

### Fallback if the criterion is rejected

If a reviewer rejects on the non-binding-constraint ground of §2.3, the minimum defensible
substitute is **not** silence: register the standing matched-volatility rule
(`internal research log:579-581`) as a **reporting** obligation — no sleeve result may be quoted
without its benchmark's Sharpe on the same window beside it, which is already the run's own
lesson (`ACCOUNTING.md:354`: "Never quote a DSR without the benchmark's DSR beside it"). That
costs nothing, cannot loosen anything, and preserves the evidence trail that made this review
possible.

---

## 6. PROVENANCE

- Code read at HEAD `e0d517d`. **No code was modified by this review.**
  `research/validation.py` is untouched.
- New measurement: `research/_gate_review/sharpe_difference_power.py` →
  `sharpe_difference_power.json` (seed 20260728). Bootstrap B = 20,000, expected block
  length 6 months; Monte-Carlo null 20,000 draws. Analytic SE validated against the
  simulation to 3 d.p. at every (T, rho).
- Only summary statistics are written to disk. No return series, no row-level data, and
  nothing derived from the Sharadar subscription appears in this document or in the JSON.
- Figures marked **[R]** are quoted from committed artefacts under
  `research/sleeves/**`, the internal research log and
  `research/medallion_style_alpha_search/2026-07-28_RESEARCH_RUN_ACCOUNTING.md`.
- **Numbers deliberately NOT used**, per the run's own discrepancy register: "trend DSR
  0.612 / passive DSR 0.669" (those are **Sharpes**; the true DSRs are 0.9944 and 0.9964 at
  n=47) and any unqualified "passive 0.7065" (that is the **daily-rebalanced** book; monthly
  is 0.6691).
