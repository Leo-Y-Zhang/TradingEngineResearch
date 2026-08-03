# RESULT — The long-history multi-asset iteration: 98.6 years of data, two sleeves, and a priced route that is still 0.24 Sharpe short

**Run:** 2026-07-28. Panel built (`7785186`), two sleeves pre-registered and run once
(`f9c9201` trend, `4dddae4` carry), then adversarially verified. This document is the
synthesis across all three.

**Governing documents:**
`docs/project-control/specs/2026-07-27-the-dsr-sample-length-finding.md`,
`docs/project-control/specs/2026-07-28-the-breadth-lever.md`,
`research/multiasset/data_integrity.md`.

**Every number in this document is reproduced by
`scripts/synthesise_multiasset_longhistory.py`**, which re-reads the sleeve artefacts,
re-runs the arithmetic, and additionally runs five checks the sleeve studies did not:
the DSR bar as a curve over achievable sample lengths, a Fisher interval on the
trend–carry correlation, leave-one-INSTRUMENT-out on carry, a 200-seed negative control,
and the carry accrual/price decomposition by decade. Machine-readable output:
`research/medallion_style_alpha_search/_multiasset_longhistory/synthesis.json`.

---

## Verdict in one line

**The two motivating findings both held, both stopped being the binding constraint, and
the constraint they exposed underneath is worse: two sleeves that are genuinely
uncorrelated combine to Sharpe 0.655 (half-Kelly 16.1%/yr) against the 0.894 that 30%/yr
needs — and neither sleeve is individually deployable, so 0.655 is arithmetic, not a
strategy.**

---

## 1. History actually obtained, and what it did to the DSR bar

### 1.1 The headline is one instrument, and the report says so

The panel spans **1927-12-30 → 2026-07-27**, **98.6 years**, 27 tradable instruments plus
a cash leg and 3 validation-only series, 0 fetch failures. But **98.6 years is ^GSPC
alone.** Breadth and length trade off directly, and the trade-off is measured, not
assumed (`breadth_vs_length` in the synthesis JSON, computed from the shipped monthly
panel):

| instruments available | from | years to 2026-06 | DSR bar, n_trials 36 | n_trials 32 |
|---:|---|---:|---:|---:|
| 1 | 1928-01 | 98.4 | **0.384** | 0.379 |
| 2 | 1962-01 | 64.4 | 0.475 | 0.469 |
| 4 | 1965-01 | 61.4 | 0.487 | 0.480 |
| 8 | 1984-01 | 42.4 | 0.587 | 0.580 |
| 12 | 1992-11 | 33.6 | 0.661 | 0.653 |
| 16 | 2000-09 | 25.7 | 0.757 | 0.748 |
| 20 | 2001-09 | 24.7 | 0.773 | 0.763 |
| 24 | 2003-12 | 22.5 | 0.812 | 0.801 |
| **27 (all)** | **2006-02** | **20.3** | **0.856** | 0.844 |

You may have length or breadth. You may not have both.

### 1.2 The bar fell by a factor of three, and that is a real result

The bar function (`research.multiasset.panel.dsr_sharpe_bar`) reproduces **both**
programme anchors exactly — **1.4881 at 7 years** and **0.5971 at 40 years**, n_trials 32
— and the test suite asserts both. Against that calibration:

| sample length | context | bar (n=32) | bar (n=36) | bar (n=38) |
|---:|---|---:|---:|---:|
| 7.0 yr | every study before this iteration | **1.488** | 1.509 | 1.518 |
| 17.0 yr | longest prior study | 0.927 | 0.939 | 0.944 |
| 22.4 yr | carry, actual | 0.803 | **0.813** | 0.818 |
| 47.4 yr | trend reference, actual | 0.548 | 0.555 | 0.558 |
| 61.5 yr | trend, actual | 0.480 | **0.486** | 0.489 |
| 98.6 yr | SPX alone | 0.378 | 0.383 | 0.385 |

**Was it long enough? For trend, decisively yes. For carry, no — and the reason is the
finding.** Trend ran on 61.5 years and cleared its 0.486 bar at net Sharpe 0.612: **the
first sleeve in the programme's history to clear DSR.** Carry could not collect the same
discount because carry's *inputs* — FX spot and OECD short rates — begin in 2003, not
1927. **Sample length is not a dial you turn on a strategy; it is a property of the data
that strategy needs.** That refines the 2026-07-27 finding rather than refuting it.

### 1.3 And clearing the bar turned out not to matter

Trend cleared 0.486 at 0.612. **Its own passive benchmark scored 0.669 and cleared the
same bar by more.** The DSR gate asks whether a Sharpe is distinguishable from luck; it
never asks whether the Sharpe beats owning the assets. It would have promoted a strategy
measurably worse than doing nothing. Lowering the bar is worth exactly nothing until the
gate is also run against the benchmark — recorded as DEFECT 2 in the log, still unpatched
because gate changes need their own adversarial review.

---

## 2. Results

Both sleeves pre-registered, run once, no tuning. Costs are round-trip.

| | **multi-asset trend** | **cross-asset carry** |
|---|---:|---:|
| instruments | 18 | 13 |
| sample | 1965-01 → 2026-06, **61.5 yr** (738 mo) | 2004-02 → 2026-06, **22.4 yr** (269 mo) |
| cost bracket | 10 bps | 3 bps |
| **gross Sharpe** | **0.672** | **0.438** |
| **net Sharpe** | **0.612** | **0.430** |
| own-universe benchmark Sharpe | **0.669** — beats the strategy | 0.029 — strategy beats it |
| **arithmetic active, raw** | **+8.07%/yr, t = +2.54** | +1.53%/yr, t = +0.88 |
| **arithmetic active, VOL-MATCHED** | **−0.51%/yr, t = −0.31** | **+1.60%/yr, t = +1.22** |
| strategy's own arithmetic mean | +13.95%/yr, t = +4.63 | +1.72%/yr, t = +2.05 |
| OLS alpha vs own universe | +13.87%/yr, t = +4.63, β = 0.01 | +1.74%/yr, t = +2.17, β = −0.13 |
| net vol | 22.80% | 3.99% |
| max drawdown | −52.90% | −6.22% |
| turnover / cost drag | 27.5×/yr, 1.38%/yr | 1.97×/yr, **0.030%/yr** |
| **DSR bar at its own length (n=36)** | 0.486 | 0.813 |
| **clears DSR** | **YES** (0.612 > 0.486) | **NO** (0.430 < 0.813) |
| **half-Kelly reachable return** | **14.03%/yr** | **6.94%/yr** |
| breadth, nominal → effective | 135.6 → **57.5** bets/yr (eff. N 4.79) | 156 nominal → **1.34 sign flips/yr** |
| **verdict** | **DEAD** | **MARGINAL, not deployable** |

### Per-decade net Sharpe — the deployable era fails in both

| decade | trend | trend's benchmark | carry |
|---|---:|---:|---:|
| 1960s (60 mo) | 0.65 | 0.27 | — |
| 1970s | 0.48 | 0.49 | — |
| 1980s | 0.88 | 0.95 | — |
| 1990s | 0.77 | 0.97 | — |
| 2000s | 0.91 | 0.39 | **0.12** |
| **2010s** | **0.05** | 0.61 | **0.86** |
| **2020s (78 mo)** | 0.39 | 0.79 | **0.08** |

Trend is positive in all seven decades and still loses: post-2009 it scores **0.180
against a benchmark of 0.777**, i.e. **−12.50%/yr vs the vol-matched benchmark (t =
−1.84)**. Carry is one decade: leave-one-decade-out gives +0.567 / +0.554 / **+0.100**
when the 2000s / 2020s / **2010s** are dropped.

### The two illusions, both measured, both refused

The active-return test is **not scale-invariant**, and this iteration found the second
half of a trap the programme had only seen one side of.

> **PEAD faked a positive GEOMETRIC excess by running at LOWER vol than its benchmark.**
> **Trend faked a positive ARITHMETIC active return by running at HIGHER vol.**
> **Same trap, opposite sign.**

The signature is measured: as trend's vol target rises 10 → 120%, its active t-stat runs
**0.63 / 2.54 / 3.64 / 3.85 / 3.99** while its own t-stat is flat at 4.6–4.7. At high
leverage the "active" test stops comparing and only asks whether the return is positive.
Carry sits on the *other* side (3.99% vol vs a 6.51% benchmark), so vol-matching *helps*
it — its raw active t of 0.88 becomes 1.22, invariant across a 10× leverage sweep.
**Standing rule now in force: compare at matched volatility, against the benchmark levered
to the strategy's own vol. Neither raw arithmetic nor raw geometric excess is safe alone.**

Jensen alpha is no defence either: trend's alpha t-stat of 4.63 at β = 0.01 simply *is*
its own Sharpe t-stat (0.612 × √61.5 = 4.80).

---

## 3. THE PORTFOLIO TEST

Both sleeves ran. Overlap **2004-02 → 2026-06, 269 months**. Carry scores **0.4301** on
the overlap; the real 61.5-year trend sleeve scores **0.4751** on the same months —
*below* its 0.612 full-sample figure, so computing the combination on the overlap
penalises trend rather than flattering it.

### 3.1 The correlation, with its uncertainty carried

**ρ(carry, trend) = −0.0441** over 269 months (recomputed from the two saved return
streams; discrepancy against the carry study's in-run figure is **exactly 0.0**).

**Fisher 95% CI: [−0.163, +0.076]. p = 0.472 against zero.**

**The sign is noise.** These are the first two sleeves in the programme that are not
variants of the same equity cross-section, and *that* is the solid finding — every prior
pair was correlated by construction. But "negative ρ, therefore no finite ceiling" is not
supported by 269 months. The study also measured this correlation twice: its
**pre-registered** comparator (its own in-study trend reference, on which prediction P3
was scored) gave **ρ = +0.0747**; **−0.0441** comes from a comparator introduced *after*
the run. The headline promoted the post-hoc, more favourable of the two. This document
reports the interval and treats **ρ = 0** as the honest working value.

### 3.2 The combined Sharpe

| combination | Sharpe | t | half-Kelly compound |
|---|---:|---:|---:|
| carry alone | 0.430 | 2.05 | 6.94%/yr |
| trend alone (overlap) | 0.475 | — | 8.46%/yr |
| **equal-DOLLAR two-sleeve** | **0.546** | 2.59 | **11.18%/yr** |
| **equal-RISK two-sleeve** | **0.655** | 3.10 | **16.07%/yr** |
| formula `S = s√(N/(1+(N−1)ρ))`, s = 0.4526, N = 2, ρ = −0.044 | 0.655 | — | 16.07%/yr |
| **required for 30%/yr at half Kelly** | **0.894** | | **30%** |

**The formula and the measured equal-risk number agree to 2.2 × 10⁻¹⁶ — and that is
algebra, not validation.** For two sleeves scaled to equal volatility the measured Sharpe
*is* (s₁+s₂)/√(2(1+ρ)), which is the formula at N = 2. The informative comparison is the
other one: **equal-DOLLAR 0.546 vs equal-RISK 0.655.** The sleeves differ 5.7× in
volatility, so dollar-weighting throws away **0.109 of Sharpe** — worth 4.9 percentage
points of compound return — for free.

### 3.3 The gap, stated plainly

**0.655 against 0.894. The shortfall is 0.240 of Sharpe, which is 16.1%/yr against 30%/yr
— the two sleeves together reach just over half the target.**

And 0.655 is generous on three counts, all measured:

1. **It does not clear its own gate.** The DSR bar at 22.4 years and n_trials 36 is
   **0.813**. The combination scores 0.655. It fails.
2. **Both constituents are individually not deployable.** Trend clears DSR and loses to a
   vol-matched hold of its own universe (−0.51%/yr, t = −0.31). Carry beats its benchmark
   and cannot clear DSR, is one decade, and — see §3.5 — is substantially one position.
   **Combining two sleeves that individually fail does not produce one that passes.**
3. **The diversification is weakest in the deployable era.** Equal-weight combination by
   decade: **2000s 1.14 / 2010s 0.21 / 2020s 0.41.** The 2000s carry it, and the 2000s
   are trend's decade, not carry's.

### 3.4 The price of the target, as a requirement on the next sleeve

Orthogonal equal-risk sleeves add in quadrature (S² = Σsᵢ²). Held: S² = 0.411. Required:
S² = 0.800.

| how many more sleeves | each must score (net, uncorrelated) |
|---:|---:|
| **1** | **0.624** — above anything the programme has ever produced |
| **2** | **0.441** — exactly the quality already achieved |
| 3 | 0.360 |

Equivalently, at the measured mean sleeve quality of **0.4526**:

| N sleeves | S at ρ = 0 | half-Kelly | S at ρ = −0.044 | half-Kelly |
|---:|---:|---:|---:|---:|
| 2 | 0.640 | 15.4%/yr | 0.655 | 16.1%/yr |
| 3 | 0.784 | 23.0%/yr | 0.821 | 25.3%/yr |
| **4** | **0.905** | **30.7%/yr** | 0.972 | 35.4%/yr |
| 5 | 1.012 | 38.4%/yr | 1.115 | 46.6%/yr |

**Four sleeves at the quality already demonstrated, mutually uncorrelated, hits 30%/yr.**
Sleeve-count sensitivity to the correlation interval: **2.78 sleeves at ρ = −0.163, 3.48
at −0.044, 3.91 at 0, 5.13 at +0.076.** The route exists. It runs through sleeve count,
and every sleeve costs a trial that raises the bar for all of them.

### 3.5 What adversarial verification took away

Three of the carry study's supporting claims did not survive re-measurement. All three
corrections are reproduced by this synthesis, independently of the reviewer.

**(a) The negative control was a four-seed artefact.** Reported: permuted books score
−0.312 ± 0.122, live +0.430, **"+6.1 sd"**. Re-run with **200 seeds** through the sleeve's
own `permute_seed` control: mean **−0.0934**, sd **0.2229**, 5th/50th/95th percentile
−0.449 / −0.084 / +0.311. Live +0.4301 is **+2.35 sd**, empirical **p = 0.010** (1 of 200
permutations beat it). Still a pass — but 2.35 sd, not 6.1.

**(b) Leave-one-INSTRUMENT-out was never run, and it breaks the verdict.** The study
tested leave-one-*cell*-out only. At the registered floor:

| dropped | Sharpe | | dropped | Sharpe |
|---|---:|---|---|---:|
| **FX_JPY** | **0.267** ⚠ | | FX_NOK | 0.448 |
| **SPY_EQ** | **0.337** ⚠ | | FX_CHF | 0.472 |
| FX_SEK | 0.396 | | US30Y_TR | 0.500 |
| FX_GBP / FX_EUR / FX_CAD | 0.425–0.428 | | US10Y_TR | 0.515 |
| FX_NZD / FX_AUD | 0.430–0.432 | | US5Y_TR | **0.557** |

Dropping **FX_JPY** — a short-yen position held in essentially every month, 27.3% of P&L
— takes the sleeve to **0.267, below its own pre-registered DEAD threshold of 0.35.**
Dropping any individual bond *raises* the Sharpe. Leave-one-class-out: drop rates → 0.409,
drop equity → 0.337, **drop FX → −0.099**. The honest composition statement is not "no
class is significant alone, the sleeve is the combination" — **the sleeve is the FX leg,
and the FX leg is substantially one perpetual short-yen carry position.**

**(c) The "ZIRP regime" diagnosis is contradicted by the sleeve's own decomposition.**
The study reported the accrual/price split full-sample only. By decade:

| decade | accrual | price | price vol |
|---|---:|---:|---:|
| 2000s | **+2.161%/yr** | −1.577%/yr | 4.53% |
| 2010s | +1.581%/yr | **+1.785%/yr** | 3.87% |
| 2020s | +1.520%/yr | −1.204%/yr | 3.53% |

The accrual — the deterministic carry premium, the quantity a steep curve and sub-zero
funding rates would actually inflate — was **highest in the 2000s** and flat thereafter.
**100% of the decade dispersion is the price leg**, whose full-sample mean is +0.031%/yr
at t = +0.04. So the real statement is harsher than "the edge is the ZIRP regime": in two
of three decades spot offset 73–79% of the accrual, exactly as the expectations hypothesis
predicts. **Carry's headline finding — that the EH fails — is itself a one-decade result.**

---

## 4. Trial accounting

Cumulative `n_trials` entering this iteration: **34.** One per sleeve run: **trend (+1),
carry (+1)**. **New total: 36.**

Two honest qualifications:

- **The sleeves' own accounting is higher.** Trend counted PRIMARY + SENSITIVITY-B as 2;
  carry counted PRIMARY + its in-study trend reference as 2. Both wrote `n_trials = 36`
  independently from a base of 34 — i.e. the register is being double-counted across
  parallel agents and needs one owner.
- **It does not matter.** The bar is almost flat in n over this range:

| sample length | n = 32 | **n = 36** | n = 38 |
|---|---:|---:|---:|
| 22.4 yr (carry) | 0.803 | **0.813** | 0.818 |
| 61.5 yr (trend) | 0.480 | **0.486** | 0.489 |
| 22.4 yr (the two-sleeve combination) | 0.803 | **0.813** | 0.818 |

Six extra trials move the bar by 0.015. **Sample length dominates trial count by an order
of magnitude, and neither is what is binding.** Resulting bars at n = 36: **carry 0.813
(fails at 0.430), trend 0.486 (clears at 0.612), the combination 0.813 (fails at 0.655).**

A caveat that raises every figure above: `dsr_sharpe_bar` is Gaussian by construction.
Real returns are skewed and fat-tailed, which raises the true bar. **Treat every bar in
this document as a floor.**

---

## 5. Did the two motivating findings hold?

### Finding 1 — "the DSR bar depends on sample length". **HELD, then stopped mattering.**

Confirmed quantitatively: the bar fell from **1.488** (7 years, where every prior study
lived) to **0.486** (61.5 years) — a factor of 3.1 — and trend became the first sleeve
ever to clear it. Three measured qualifications, in descending order of importance:

1. **The benchmark clears the same bar, by more** (0.669 vs 0.486). A gate with no
   benchmark-relative criterion cannot distinguish skill from beta, so lowering it bought
   nothing.
2. **Length and breadth trade off directly** (§1.1). The full 27-instrument cross-section
   exists only from 2006 — 20.3 years, bar 0.856 — which is *worse* than the 22.4-year
   bar carry actually faced. You cannot buy the discount and the breadth together.
3. **The discount is only collectable by strategies whose inputs are old.** Carry's are
   not, and no amount of wanting changes that.

### Finding 2 — "cheap instruments let breadth pay". **HALF HELD. Costs died; breadth did not arrive.**

**The cost half is emphatically confirmed and closes an argument that ran for twelve
studies.** Trend turns over 27.5× a year and pays **1.38%/yr at 10bps**; carry pays
**0.030%/yr at 3bps** — 3bps cost it **0.008 of Sharpe**. Against the **117–236bps** round
trip that killed every equity sleeve, the bill has simply ceased to exist as a constraint.

**The breadth half failed, and the measurement says why.** The Fundamental Law needs
*independent* bets, and cross-asset instruments are far more correlated than single names:

- Trend's nominal breadth is **135.6 bets/yr**; its **effective N from the correlation
  eigenvalues is 4.79** (2007+), i.e. **57.5 effective bets/yr**. Four FX pairs that are
  all the dollar are not four bets.
- Carry's nominal breadth is 156 bets/yr; its **measured sign-flip rate is 1.34/yr**. It
  is a very slow signal, confirmed by a factor of 30 against its own pre-registration.

And the decisive number: **trend's breakeven round-trip cost — the level at which its
Sharpe merely equals its benchmark's — is 0.52bps.** That is below the cheapest execution
available anywhere on earth. **Cheap instruments did not fail to pay for breadth; there
was no gross edge over passive ownership for them to pay for.** Fifteen studies now agree
on that diagnosis.

---

## 6. The single most promising direction remaining

**Sleeve count, weighted by risk rather than dollars, and gated on the vol-matched
benchmark test rather than DSR alone.** Grounded in the measured numbers:

1. **The arithmetic is the only thing that has ever produced a number in the target's
   neighbourhood.** Two uncorrelated sleeves at the demonstrated quality give 0.655;
   **four give 0.905 → 30.7%/yr** (§3.4). Nothing else in fifteen studies has closed even
   half the gap. This is the only route with a measured, non-zero prior.
2. **The cheapest 0.109 of Sharpe in the programme is free and untaken.** Equal-risk
   weighting beats equal-dollar by 0.546 → 0.655 — 4.9 percentage points of compound
   return — with no new signal, no new data, and no new trial. It must be *pre-registered*
   before the next combination is scored, because choosing it after seeing both is
   selection.
3. **The next sleeve has a hard, pre-computable spec.** To reach target as sleeve #3 it
   must score **0.624 net, uncorrelated** — above anything ever produced here, so
   realistically the plan is **two more at 0.441 each**, which is exactly the quality
   already demonstrated twice. That is a target a study can be registered against.
4. **The next sleeve must not be a price transformation.** Trend is a function of past
   returns; value was too, and its apparent diversification collapsed from −0.164 to
   −0.013 once the overlapping windows were removed. Carry is not — and carry's *price*
   leg earns **+0.031%/yr at t = +0.04 over 22 years**. The measured statement is that on
   this panel, at monthly frequency, **nothing yet found predicts the price**; carry earns
   only its accrual. A third sleeve built on returns will be correlated with trend and
   will earn nothing extra. Candidates conditioning on something else — second moments
   (a defensive/vol sleeve), calendar structure, or term-structure levels beyond the front
   point — are the ones with a real prior. Two such studies (`multiasset_defensive`,
   `multiasset_seasonal`) are already in flight from sibling agents; §3.4 is the bar they
   have to clear to matter.
5. **Every future sleeve must be gated twice.** DSR *and* the vol-matched active-return
   test against its own universe levered to equal volatility. Trend passed one and failed
   the other; carry did the reverse. A sleeve that fails either is not a sleeve.

**What is explicitly closed.** Do not re-run trend on other lookbacks, instruments,
rebalance frequencies or vol targets — the vol-target axis is proven inert (Sharpe
0.590–0.612 across a 12× leverage range) and every lookback variant is dead post-2009
(best 0.244 vs a benchmark of 0.777). Do not re-run carry with different weightings or
windows — 3bps cost it 0.008 Sharpe points; the signal is not underpaid, it is small, and
it is substantially one short-yen position.

---

## 7. Disclosed biases and what was taken on trust

Inherited from `research/multiasset/data_integrity.md`, all pre-registered, all
**common-mode with the benchmark** (which is defined on the identical convention):

- **7 equity indices are PRICE returns**, dividends excluded (measured at 1.95%/yr on SPX
  vs SPY); **DAX is total-return** and is not comparable to the other seven. Any signal
  built on long-horizon returns ranks high-dividend markets as permanently cheap —
  measured symptom in the value study: FTSE100 long in 93% of months.
- **Par-bond proxy omits roll-down**, understating bond total return by ~0.5%/yr at 5y and
  10y (0.09%/yr at 30y) — *against* the sleeves that hold bonds long.
- **FX spot excludes the interest differential**, so no carry is expressible in the panel
  itself; the carry sleeve sources it separately from FRED OECD 3-month interbank rates
  and **assumes CIP**. Stress: a 50bps/yr cross-currency basis takes carry to 0.344,
  100bps to 0.259. A second, unmodelled charge of the same class exists — the book
  rebalances monthly but prices the FX accrual off a **3-month** differential, a
  maturity/holding-period mismatch plausibly worth 10–30bps/yr, which lands inside the
  already-run stress band.
- **NATGAS_F is roll-contaminated** (65.7% of extreme bars in the roll window vs a 24.0%
  base, lift 2.74×); its 16.54%/yr headline is substantially manufactured.
- **Futures settle after the equity close**: corr(GOLD_F today, GLD tomorrow) = **+0.115**.
  A genuine daily-frequency lookahead for any signal mixing futures with ETFs. Both
  sleeves use the month-end panel for this reason — which is also why neither could test
  the daily breadth axis.
- **Survivorship**: yfinance returns survivors only. Indices/futures/FX/rates are
  near-immune by construction; the bias is confined to 7 ETFs and is small. Index
  *constitution* change is survivorship by another name and flatters long-horizon equity
  series.
- **Carry's FX universe is the modern G10 and nothing else.** Every classic carry-crash
  currency (TRY, ZAR, MXN, BRL) is absent, so the canonical negative-skew event is
  **excluded by universe construction**. Carry's most attractive property — skew +0.14, max
  drawdown −6.22%, "it didn't crash" — has an unmentioned co-explanation. (To its credit,
  the CHF 2015 de-peg *is* in sample and shows as the 7th-largest loss cell.)
- **Local currency is not converted to USD** — near-correct for futures, wrong for cash
  indices.

**Taken on trust, not re-verified here:** the raw cached yfinance/FRED vendor data (no
re-fetch), and the panel's par-bond and cash-accrual construction beyond the validation
already recorded (corr 0.947 vs IEF, 0.940 vs TLT/IEI; cash accrual 32.92% vs BIL's 29.81%
over 19.2 years, a 0.125%/yr gap).

**Reproducibility.** The carry run is byte-identical on re-run. `rho` recomputed from the
saved return streams matches the study's in-run figure to **0.0**. The accrual/price
identity reproduces `decompose_pnl` to < 1e-9 before slicing. Full suite **1389 passed / 1
skipped**; ruff clean.

**Hard limits respected throughout:** no live trading, no broker path, no Sharadar row
committed, nothing public, no account actions, no financial advice. Every number here came
from a run that executed.
