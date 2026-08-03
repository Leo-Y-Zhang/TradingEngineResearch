# RESULT — Cross-asset CARRY on the long-history panel: **MARGINAL**, and not a route to 30%

Pre-registration: `research/sleeves/multiasset_carry_prereg.md` (written first, unchanged
after the run). Run once: `scripts/run_multiasset_carry.py`. Adversarial verification:
`scripts/verify_multiasset_carry.py`. Machine-readable:
`research/sleeves/_carry_output/multiasset_carry_result.json` and
`…_verification.json`. `n_trials` 34 → **36**. Re-run is **byte-identical**.

---

## 1. Headline

**Cross-asset carry, 13 instruments, 2004-02 → 2026-06 (22.42 years, 269 months), monthly.**

| | |
|---|---|
| gross Sharpe | **+0.438** |
| net Sharpe @ 3 bps round trip (realistic) | **+0.430** |
| net Sharpe @ 10 bps round trip (conservative) | **+0.412** |
| net volatility | **3.99%** |
| **A — arithmetic active return** | **+1.717%/yr, t = +2.048** (Newey–West, 4 lags) |
| B — alpha vs own-universe basket | +1.742%/yr, t = +2.165, beta −0.131 |
| C — arithmetic difference vs that basket | +1.527%/yr, t = +0.877 |
| max drawdown | −6.22% (skew **+0.14**, worst month −3.36%) |
| DSR bar, n_trials 36, 22.42 yr | **0.813 — NOT cleared** |
| half-Kelly compound return at this Sharpe | **6.94%/yr** |
| Sharpe needed for 30%/yr at half Kelly | 0.894 |

**Verdict by the pre-registered rule: MARGINAL.** `S_net = 0.430 ≥ 0.35` and
`t(A) = 2.048 ≥ 1.5`, so it is not DEAD; it fails PROMISING because it does not clear the
DSR bar. It is **not deployable standalone and it is not a route to 30%/yr.**

**Predictions scored:** P1 **CONFIRMED** (gross 0.438 ∈ [0.40, 0.90]). P2 **CONFIRMED**
(costs took 0.008 Sharpe points). P3 **CONFIRMED** (ρ = +0.075). P4 **FAILED** — max
drawdown 6.22% not >15%, skew **+0.14** not negative: *this carry book did not crash*, which
is itself a finding (see §4). P5 **CONFIRMED**, emphatically — 1.3 sign flips/yr. P6
**CONFIRMED** and then some (accrual 98.2%). P7 **CONFIRMED** (0.430 ≪ 0.894).

---

## 2. What the sleeve actually is

Universe (13): 3 US par-bond points (`US5Y_TR`, `US10Y_TR`, `US30Y_TR`), 9 CIP-consistent
FX excess returns (EUR, GBP, JPY, AUD, NZD, CAD, CHF, SEK, NOK), and `SPY_EQ`.
**Commodities were excluded, not substituted** — the free data has no futures curve, and
putting momentum in the carry slot would have been a different strategy wearing carry's name.

Carry = the return if prices do not move: term spread for bonds, 3-month interbank
differential for FX (FRED OECD `IR3TIB01*`, free and keyless), trailing 12-month **realised**
dividend yield minus the bill for equity. Rank on carry/σ, KMP rank weights, inverse-vol
sized, monthly, 1.24 mean gross notional.

**New data built for this: nine FX spot series and ten short-rate series.** The panel's
*already-published* corrupt-close criterion (8th/9th of a month in 2008, |r| > 5%, round
trip < 2.5%) was applied mechanically to the new FX and **reproduced the panel's eight
quarantined closes exactly** — 5 EURUSD + 3 JPYUSD, no more, no fewer — while correctly
refusing all seven reversal bars and AUD 2008-10-08 (a real −5.83% move). That is an
independent confirmation of the panel's cleaning decision, obtained for free.

---

## 3. The decomposition that explains everything

| leg | annual return | annual vol | t |
|---|---:|---:|---:|
| carry accrual | **+1.716%** | **0.150%** | — (near-deterministic) |
| price move | **+0.031%** | **3.973%** | **+0.04** |

**All of the return is the accrual; all of the risk is the price leg; the price leg's
expected return is zero to three decimal places.** That is the carry premium in its purest
measured form — the expectations hypothesis fails, spot does not move to offset the yield
differential — and it is worth **1.72%/yr against 4% of volatility.** Sharpe 0.43 is that
ratio, nothing more.

This is a real result and a small one. It also means the sleeve is an accounting spread
that an unmodelled financing cost can erase, so that was priced:

| unmodelled CIP basis / forward drag on FX notional | net Sharpe | arithmetic active | t |
|---:|---:|---:|---:|
| 0 bps/yr (as reported) | +0.430 | +1.717% | +2.05 |
| 25 bps/yr | +0.387 | +1.546% | +1.85 |
| **50 bps/yr** (routine post-2008 EUR/JPY basis) | **+0.344** | +1.375% | **+1.64** |
| 100 bps/yr | +0.259 | +1.033% | +1.23 |

---

## 4. Everything the adversarial pass found

1. **No lookahead.** Positions through 2015-12-31 are **identical** when every later
   observation is deleted.
2. **Negative control passes by 6.1 sd.** Per-date permutation of the carry scores over 4
   fixed seeds: −0.312 ± 0.122 against a live +0.430.
3. **Concentration is CLEAN on the meaningful measure.** The worst cell (FX_CHF,
   2008-12) is −6.07% of *net* P&L but only **0.39% of gross absolute P&L** — a degenerate
   denominator, since 22 years of net P&L is 0.39 on a 1.24-notional book. Deleting the 1 /
   5 / 10 largest cells moves the Sharpe to **+0.457 / +0.432 / +0.516**: the result is
   *hurt* by its outliers, not manufactured by them.
4. **It is one decade.** Leave-one-decade-out: drop the 2000s → **+0.567**; drop the 2020s
   → **+0.554**; **drop the 2010s → +0.100.** Decade Sharpes are 2000s +0.12, 2010s +0.86,
   2020s +0.08. The edge lives in the ZIRP decade, when the US curve was steep and the
   funding currencies (JPY, CHF, EUR, SEK) were at or below zero. **That is a monetary
   regime, not a permanent premium.**
5. **A third of it is duration and a third is the dollar.** Single-factor alphas all
   survive (t 1.94–2.62), but the **joint 4-factor alpha is +0.569%/yr at t = +1.16** —
   *not* distinguishable from zero — with betas duration **+0.33** and dollar **+0.33**.
6. **The static/dynamic split, both ways, honestly.** Against a hindsight-fixed book at the
   full-sample mean position, the dynamic residual is +0.663%/yr at **t = 1.63** (not
   significant). Against a **point-in-time expanding-mean** book — the fair comparator — the
   residual is +0.970%/yr at **t = 2.31** (significant). So the time variation does carry
   content; the criticism is real but it is not fatal, and both numbers are reported.
7. **P4 failed and it matters.** Carry is supposed to crash. This book's skew is **+0.14**
   with a 6.22% max drawdown. The reason is visible in §3: with 98% of the return in a
   deterministic accrual and the book risk-balanced across three asset classes, the
   classic 2008 carry unwind was absorbed — the worst single cell is FX_CHF 2008-12 at
   −2.4% and the 2008-11 bond leg (+2.4%, +1.7%) paid for it. **Diversifying carry across
   classes removes its crash signature.** That is the most interesting positive finding here.
8. **Structural tilt, stated plainly.** Long bonds 87–97% of months; short JPY 100%, CHF
   100%, SEK 100%, EUR 99.6%. Mean net position: rates **+0.476**, FX **−0.376**.
   Top contributors: FX_JPY +27.3%, US10Y_TR +19.4%, US30Y_TR +18.4%, SPY_EQ +12.8%,
   FX_SEK +12.4%, FX_EUR +10.6%. Worst: FX_NZD −5.1%.

---

## 5. Secondaries (declared in advance, reported unconditionally)

| | span | net Sharpe | note |
|---|---|---:|---|
| **S1 bonds only** (N≥3) | 1979-02 → 2026-06, **47.4 yr** | **−0.148** | negative in the 1990s, 2000s, 2010s and 2020s; only the 1980s positive (+0.19). **US curve-shape carry is dead over half a century.** |
| **S2 FX only** (9 ccy) | 2005-12 → 2026-06, 20.6 yr | **+0.131** | 2000s −0.26, 2010s +0.14, 2020s **+0.45**. The classic carry trade, and it is weak. |
| **S3 vol-targeted overlay** | as primary | +0.465 | +0.035 Sharpe for a second untested bet and a −18.7% drawdown. Not worth it. |
| **S4 unscreened** | as primary | **+0.430** | the cleaning decision changes the Sharpe by 0.0000 and the return by 0.005pp. **The result does not depend on the quarantine.** |

**The composition puzzle resolved:** rates contribute +46% of P&L inside the global
cross-section while a bonds-only sleeve *loses* money. The profitable rates bet is
therefore **cross-class** — owning duration when the US term spread is wide relative to
world FX differentials — not a curve-shape bet. Per-class legs: rates +0.805%/yr (t 1.15),
FX +0.719%/yr (t 1.00), equity +0.223%/yr (t 1.26). **No single class is significant on its
own; the sleeve exists only as the combination.**

---

## 6. The trend reference (trial 36, a REFERENCE not a verdict)

12-month time-series momentum, 27 instruments, **1979-02 → 2026-06, 47.4 years**:
gross **+0.461**, net(3bp) **+0.455**, net(10bp) **+0.442**; arithmetic active
**+2.787%/yr, t = +3.03**; alpha vs own universe +2.829%/yr, t = +3.08; max drawdown
−17.07%, skew −0.26; 22.2 sign flips/yr; concentration 0.23% of gross absolute P&L.
DSR bar at 47.4 years and n_trials 36 is **0.555** — **trend does not clear it either.**

Decade Sharpes: 1980s +0.53, 1990s +0.52, 2000s +0.64, **2010s +0.15, 2020s +0.04.**
**Trend has decayed too**, and over the 2004–2026 overlap with carry it earns only +0.297.

---

## 7. THE TWO-SLEEVE TEST — the thesis is confirmed and the arithmetic still refuses

269 overlapping months, 2004-02 → 2026-06.

**ρ(carry, trend) = +0.0747.** The sleeves are genuinely uncorrelated — the first time in
this programme that two sleeves have not been variants of one another.

| combination | Sharpe | half-Kelly compound |
|---|---:|---:|
| carry alone (overlap) | +0.430 | 6.94%/yr |
| trend alone (overlap) | +0.297 | 3.31%/yr |
| formula `S = s√(N/(1+(N−1)ρ))`, N=2 | +0.496 | 9.21%/yr |
| **measured 50/50 equal weight** | **+0.485** (t 2.13) | **8.82%/yr** |
| measured point-in-time risk parity | +0.454 (t 1.92) | 7.74%/yr |

The formula over-predicts the measured combination by 0.011–0.042 Sharpe points — it
assumes equal sleeve Sharpes and these are 0.430 vs 0.297 — so **the brief's arithmetic is
right to within 2–8%, and the measured number is the one to use.** The combination's
drawdown is −8.53% and its decade profile is 2000s +0.46, 2010s +0.65, 2020s +0.10.

### The number this iteration exists to produce

Inverting `S = s√(N/(1+(N−1)ρ))` at ρ = 0.075 for S = 0.894:

| if each sleeve is… | sleeves needed for 30%/yr | ceiling as N → ∞ |
|---|---:|---:|
| Sharpe 0.430 (carry as reported) | **5.9** | 1.57 |
| Sharpe 0.344 (carry after a 50 bps basis) | **12.6** | 1.26 |
| Sharpe 0.297 (trend over the overlap) | **26.3** | 1.08 |
| Sharpe 0.100 (carry ex-2010s) | **impossible** | 0.37 |

**The multi-sleeve thesis is not refuted — it is priced.** At ρ ≈ 0.075 the route to
30%/yr requires roughly **six** independent sleeves of this quality, or a dozen at the
honest post-cost number. Two sleeves buy 8.8%/yr. There is no leverage fix: 0.894 is a
Sharpe, and the ceiling at infinite breadth of *this* quality is 1.57, so the target is
inside the space — but only through **sleeve count**, and each sleeve costs a trial that
raises the DSR bar for all of them.

---

## 7b. ADDENDUM — the real trend sleeve, and the test that killed it applied to carry

Written after the registered run, using two things produced in parallel in this repo:
`research/sleeves/_multiasset_trend/` (a full 61.5-year trend study with its own
pre-registration) and its finding that **the arithmetic active-return test has a
HIGH-volatility twin of the variance-drag trap.** Script:
`scripts/synthesise_carry_trend.py`; output `…/carry_trend_synthesis.json`.

### The vol-matched active return — carry is on the *other* side of the trap

PEAD faked a positive **geometric** excess by running at **lower** volatility than its
benchmark. Trend faked a positive **arithmetic active return** by running at **higher**
volatility: differencing two streams at different volatilities compares leverage, not
skill. The defence is to scale the benchmark to the strategy's own realised volatility
before differencing (`research.multiasset.carry.vol_matched_active`, now tested).

Carry runs at **3.99%** against a **6.51%** benchmark, so the correction goes the other way:

| statistic | value | t |
|---|---:|---:|
| raw active vs own universe | +1.527%/yr | +0.877 |
| **vol-matched active vs own universe** | **+1.600%/yr** | **+1.216** |
| sleeve's own arithmetic mean | +1.717%/yr | +2.048 |

**The leverage sweep proves which statistic to trust.** Levering the sleeve to 4/10/20/40%
target volatility:

| target vol | 4% | 10% | 20% | 40% |
|---|---:|---:|---:|---:|
| raw active t | +0.878 | +1.481 | +1.771 | +1.918 |
| **vol-matched active t** | **+1.216** | **+1.216** | **+1.216** | **+1.216** |
| sleeve's own t | +2.048 | +2.048 | +2.048 | +2.048 |

The raw active t-statistic is a **leverage dial**; the vol-matched one is invariant. So the
honest own-universe number for carry is **+1.600%/yr at t = +1.216 — positive, and NOT
significant at conventional levels.** Carry survives the test that killed trend (its
vol-matched active is positive where trend's was −0.51%/yr at t = −0.31), but it survives
it weakly. **This supersedes statistic C as the own-universe verdict.**

### The correlation, against the real sleeve

`ρ(my trend reference, the real trend sleeve) = +0.766` over 569 common months — the
reference built inside this study is a faithful stand-in, which is worth knowing.

| combination, 269 overlapping months 2004-02 → 2026-06 | ρ | measured EW Sharpe | measured RP Sharpe | half-Kelly (EW / RP) |
|---|---:|---:|---:|---:|
| carry × **real trend sleeve** | **−0.0441** | **+0.546** (t 2.42) | **+0.614** | **11.18% / 14.12%** |
| carry × my trend reference | +0.0747 | +0.485 (t 2.13) | +0.454 | 8.82% / 7.74% |

**Against the real sleeve the correlation is slightly NEGATIVE.** The multi-sleeve thesis is
confirmed more strongly than the registered run showed. The combination's decade profile is
2000s +1.14, 2010s +0.21, 2020s +0.41, and its drawdown is −26.56% (the trend sleeve runs at
20% target volatility). It still does **not** clear the 0.813 DSR bar.

**Revised price of the target.** At ρ = −0.044 and a mean sleeve Sharpe of 0.453,
**≈ 3.5 sleeves of this quality reach Sharpe 0.894 (30%/yr at half Kelly)** — and with a
negative ρ there is no finite ceiling, so the constraint really is sleeve count alone.

**The caveat that must travel with this.** The trend sleeve's own study judged it **DEAD**:
it clears DSR but loses to a levered buy-and-hold of its own universe (−0.51%/yr at
t = −0.31 vol-matched). Combining a MARGINAL sleeve with a DEAD one produces a number, not
a strategy. The correlation result is solid; the combined Sharpe should be read as *what
the arithmetic would give if both sleeves were real*, and only carry currently is.

---

## 8. Honest limitations

1. **CIP is assumed.** The FX leg's return uses interbank differentials, not forward
   points; the post-2008 cross-currency basis is real, unmodelled, and worth 0.04–0.17
   Sharpe points (§3).
2. **The bond accrual is approximate.** `carry_t/12` uses the month-end yield, while the
   realised coupon accrues at intra-month yields; the residual falls into the price leg.
   Exact for FX, approximate for bonds.
3. **22.4 years is short** for a 13-instrument cross-section, and the binding start is FX
   spot availability on Yahoo (2003-12), not anything about carry.
4. **Equity carry is one instrument.** No other index in the panel has a price/total-return
   pair, so there is no equity cross-section — `SPY_EQ` is a single directional line.
5. **The panel's own disclosed biases carry through:** the par-bond proxy omits roll-down
   (~0.5%/yr against the sleeve at 5y/10y), and 7 of 27 panel instruments are ETFs with
   modest survivorship flattery.
6. **The trend reference is a reference.** One lookback, one sizing rule, no tuning — it
   exists to make ρ measurable. A real trend study should be run separately.

---

## 9. Do not re-run this hypothesis with different parameters

The signal is not the problem and the cost model is not the problem — costs took 0.008
Sharpe points out of 0.438. The problem is that **the cross-asset carry premium available
in free data is worth about 1.7%/yr against 4% of volatility, and most of that lived in
one decade.** Changing the lookback, the rank weighting, the vol window or the instrument
list will move the third decimal and spend a trial. The productive direction is **more
uncorrelated sleeves**, not a better version of this one.
