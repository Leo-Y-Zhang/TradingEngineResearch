# PRE-REGISTRATION — `trend + passive` on the REPAIRED panel

**Written 2026-07-31, before any gate was run on the repaired panel.** The gates named
in §3 have not been executed. Every prediction in §4 is a number committed to in advance,
and §6 states in advance what result kills this.

Panel: `_data/multiasset/returns_monthly_corrected_central.parquet` and the two bracket
bounds beside it, built and controlled by
`research/multiasset/convention_repair_{prereg,result}.md`. Sleeve:
`research/sleeves/multiasset_trend.py`, unchanged and not re-tuned.

---

## 0. THE HONESTY PROBLEM WITH THIS DOCUMENT, STATED FIRST

**A pre-registration written after the panel exists cannot pre-register the panel's own
statistics.** Building the repaired panel required running the book on it — that was the
repair's acceptance evidence. So the corrected Sharpe, the leg Sharpes, the vol-matched
active, the decade table and the bill+150bp ladder **are already observed**, and calling
them "predictions" here would be a lie.

They are therefore listed in §1 as **INPUTS**, not forecasts. What §4 registers is
strictly the set of quantities that **have not been computed at the time of writing**.
This is a weaker document than a true ex-ante registration and it says so rather than
borrowing the authority of one.

| status | quantities |
|---|---|
| **OBSERVED — cannot be predicted, quoted as inputs** | corrected book Sharpe 0.7834 and the bracket 0.7499 / 0.8464; trend 0.5708 and passive 0.5078; vol-matched active +2.44%/yr t 2.68; bootstrap CI [0.539, 1.035]; P(S<0.894)=80.4%; P(S<0.75)=39.0%; the decade table; book Sharpe since 2010 = 0.7074; the **bill+150bp** ladder (1.95× → 14.32% after the 0.877 factor) |
| **NOT YET COMPUTED — registered as predictions in §4** | the DSR at the true trial count; the DSR *bar* at this sample length and count; `benchmark_relative_rule` and its whole panel; the **bill+300bp** ladder; the benchmark's own levered return (rule 13); the post-2010 **active** verdict on the repaired panel; whether the two bracket bounds change any verdict |

---

## 1. Inputs — fixed, and not re-derived here

| | central | conservative | realistic |
|---|---:|---:|---:|
| book Sharpe, 738 months | 0.7834 | 0.7499 | 0.8464 |
| trend leg / passive leg | 0.5708 / 0.5078 | 0.5469 / 0.4788 | 0.6125 / 0.5616 |
| vol-matched active vs passive | +2.44%/yr, t 2.68 | +2.38%, t 2.62 | +2.54%, t 2.79 |
| capital weights trend / passive | 0.280 / 0.720 | 0.280 / 0.720 | 0.281 / 0.719 |

Cost bracket **10bps** throughout, on both legs, which is the registration duty C3 imposes
before any benchmark comparison is legitimate.

---

## 2. The trial count — fixed now, and it is not 47

Deflation must charge **every** trial that could have produced this candidate, not the
ledger's own bookkeeping.

| component | count | why it counts |
|---|---:|---|
| trial ledger, cumulative | **47** | the programme's own registered count |
| the portfolio-combination search that FOUND this candidate | **234** | 58 subsets × 4 weighting schemes; recorded in the overnight accounting as *"found by a search over 234 configurations and not pre-registered"* |
| **registered deflation count** | **281** | the sum. The primary figure. |

The DSR is also reported at **47** (ledger only) and at **32** (the panel study's
convention) so the sensitivity to this choice is visible rather than hidden. **281 is the
one that decides.**

---

## 3. The gates, named before they are run

1. **DSR ≥ 0.95** at n_trials = 281, monthly returns, on the **conservative** bound —
   the harshest bound must clear, not the friendliest. `research.validation.
   deflated_sharpe_ratio`, cross-checked against `research.multiasset.panel.
   dsr_sharpe_bar` inverted at the same count.
2. **`benchmark_relative_rule` returns `promotable=True`** — `BEATS` against the
   nominated benchmark **and** against every registered panel member, paired stationary
   bootstrap, B = 10,000, expected block 6 months, ρ always reported.
   `UNDETERMINED` forbids promotion.
3. **Leverage solved against the bootstrap 95th-percentile drawdown** (rule 5), at
   **bill+150bp AND bill+300bp**, at DD ≤ 50% and DD ≤ 35%.
4. **Every gate is run on all three bracket bounds** (rule 10). A verdict that holds only
   at the realistic bound is UNDETERMINED, not a pass.

### 3a. The benchmark and the panel — and a registered deviation

**Nominated benchmark: corrected passive alone**, on the same repaired panel, same cost
bracket, same window. Not the old panel's passive — that comparator no longer exists.

Registered panel (C7, the benchmark-shopping detector):

| member | construction |
|---|---|
| (i) own-universe equal weight | as registered |
| (ii) passive monthly EW of the 18 instruments | as registered |
| (iii) **equal-RISK passive (inverse-vol)** | **substitution — see below** |
| (iv) **60/40 equity/rates**, monthly rebalanced | **addition — see below** |

> **REGISTERED DEVIATION.** C7's registered minimum panel names a **daily** passive EW
> member. **The repaired panel is monthly by construction** — the dividend, carry and
> bill corrections are monthly quantities — so that member cannot be built without a
> second repair at daily frequency, which is not done. Rather than drop a panel member
> and weaken the shopping detector, it is **replaced by an equal-RISK passive and a 60/40
> book are added**. Both are genuinely different opportunity costs rather than a
> re-frequencied duplicate, so this substitution makes the panel **harder** to pass, not
> easier. If the daily member is later judged mandatory, the daily repair must be built
> first and this document re-run.

---

## 4. THE PREDICTIONS — committed before running

Every interval below is a falsifiable commitment. Being wrong is the point of writing
them down.

| # | quantity | **prediction** | reasoning fixed in advance |
|---|---|---|---|
| P1 | DSR at n=281, central book | **≥ 0.95, and specifically 0.97–1.000** | 738 months is a very long sample; σ_SR ≈ 0.039 monthly, so even a 281-trial haircut moves SR* to ≈0.112 monthly against an observed ≈0.226 |
| P2 | DSR at n=281, **conservative** bound | **≥ 0.95** | the bound is 0.7499, still far above the bar in P3 |
| P3 | DSR *bar* (annual Sharpe) at 61.5 yr, n=281 | **0.55 – 0.65** | inverting the same formulation; the panel's own recorded anchors are 1.488 at 7 yr and 0.597 at 40 yr for n=32 |
| P4 | `benchmark_relative_rule`, nominated | **BEATS** | Sharpe gap +0.276 on a paired bootstrap; on the OLD panel the Sharpe-gain CI was [+0.032, +0.441] with P(gain≤0)=1.2%, and the corrected gap is **wider** |
| P5 | `benchmark_relative_rule`, all four panel members | **BEATS on all four → `promotable=True`** | members (i) and (ii) are near-identical to the nominated benchmark; (iii) and (iv) are the real tests |
| P6 | ρ(book, nominated benchmark) | **> 0.90** | the book is 72% passive by capital |
| P7 | ladder at **bill+300bp**, DD ≤ 50%, central | leverage **1.85–1.95×**, CAGR after the 0.877 factor **12.5–13.5%** | +150bp of financing on ≈0.95 of levered notional ≈ −1.4%/yr against the 14.32% measured at bill+150bp |
| P8 | **the benchmark's own** levered return (rule 13) — corrected passive alone, DD ≤ 50%, bill+150bp | leverage **1.30–1.45×**, CAGR after the factor **8.0–9.5%** | the survivor's constant-charge comparator was 9.44%; the repaired passive leg is **worse** than that (0.5078 vs 0.5773) |
| P9 | incremental over the benchmark | **+4.5 to +6.5 pp/yr** | 14.32 − P8 |
| P10 | post-2010 vol-matched active, repaired panel | **0.0 to +1.5%/yr, \|t\| < 2 — NOT significant** | the old panel read −0.13%/yr t −0.09; correction hurts the long-only leg more, so the gap should turn mildly positive without becoming significant |
| P11 | do the bracket bounds change any **verdict**? | **No** — same verdict at all three bounds | the bounds span 0.75–0.85, comfortably above the P3 bar |

**Rule 13 is satisfied by P8 and P9: the benchmark is forecast, not just the strategy.**
The value's failure in this programme came from predicting the active return without ever
predicting what the benchmark would earn.

---

## 5. What is NOT being claimed

1. **No re-tuning.** The sleeve, its lookbacks, its vol target and its universe are
   exactly as registered in `multiasset_trend_prereg.md`. Nothing is searched here.
2. **No new trial.** Running a registered gate on an existing candidate adds no trial;
   the deflation count charges the 234 that *did* produce it.
3. **The panel is still an upper bound** — 21.2% of live cells (commodity roll, USDX)
   remain uncorrected, and the EUR/GBP FX residual is unexplained.
4. **Nothing here is a deployment decision.** See §6.

---

## 6. THE STOPPING RULE — and the pre-committed decision even on a full pass

**Fails the gate if:** DSR < 0.95 at n=281 on the conservative bound; **or**
`benchmark_relative_rule` returns anything other than `BEATS` on the nominated benchmark
and all four panel members; **or** `benchmark_sensitive=True`; **or** the verdict differs
across the three bracket bounds.

**And this is registered in advance, before the gates run:**

> **EVEN IF EVERY GATE PASSES, THIS SLEEVE IS NOT PROMOTED TO ANY LIVE OR PAPER PATH.**
>
> The survivor verification already established two things that a gate pass cannot
> undo: the full-sample significance **rests on the 2000s** (leave that decade out and
> the active falls to +1.29%/yr at t 1.39), and **since 2010 the book adds nothing over
> passive alone**. A statistical gate that judges the full sample cannot see either.
> A pass here means *"the measured full-sample premium is not an artefact of multiple
> testing"* — which is a statement about the past, not a reason to trade it.
>
> The only outcome that would change this is a **post-2010 result that is itself
> significant**, and P10 predicts in advance that it will not be. If P10 is wrong in the
> favourable direction, that is a genuinely new finding and gets its own verification —
> not a promotion.

**Whatever the gates return, the honest ceiling stands where the repair left it:
≈14.3%/yr at a 50% drawdown, against a corrected passive comparator, on a panel that is
still an upper bound. Not 30%.**
