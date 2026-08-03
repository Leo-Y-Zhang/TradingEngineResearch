# RESULT — Cross-asset VALUE on the long-history multi-asset panel

**Sleeve 3 of the multi-asset programme. Pre-registered in
`research/sleeves/multiasset_value_prereg.md`, run ONCE on 2026-07-28, no tuning.**
Code `research/sleeves/multiasset_value.py`, verification
`research/sleeves/multiasset_value_verify.py`, receipts `research/sleeves/_value/`.

---

## VERDICT

| question | answer |
|---|---|
| **Standalone** | **DEAD.** Net Sharpe **-0.082** at 10bps over 44.4 years. Arithmetic active return **-9.84%/yr, t = -2.80**. Fails every falsification criterion. |
| **Diversification** | **MECHANICAL.** Correlation to trend is **-0.164** full-sample (**-0.281** on the 2004+ overlap) — but removing the 12 months that overlap the trend window takes it to **-0.013**. The negative correlation is the reversal window containing the momentum window, not an economic effect. |
| **Does adding it help the portfolio?** | **NO.** Equal-risk trend+carry = **0.655**; adding value = **0.598**. The mean-variance optimal risk share for value is **3.5%**, worth **+0.0004** of Sharpe. |
| **Is the machinery sound?** | **YES.** Perfect-foresight control on the same pipeline returns net Sharpe **5.59**; the signal peaks/troughs on the exact historical dates it should; the DSR bar reproduces both recorded anchors. This is a real negative, not a bug. |

**The sleeve is not merely weak — its largest block is significantly ANTI-predictive.**
The 5-year reversal on equity indices has a rank IC of **-0.068 with t = -3.14** over 605
months, measured directly from the panel with no book, no sizing and no cost model. Cheap
country indices kept losing to expensive ones. That is a finding in its own right and it is
the opposite of what AMP 2013 report for equities using genuine book-to-market.

---

## 1. THE HEADLINE DELIVERABLE — correlation, because that is why this sleeve existed

The pre-registration fixed the ordering: *"the headline deliverable of this study is the
correlation to the trend sleeve, and the standalone Sharpe is secondary."* So it is reported
first, and it is reported with the diagnostic that undermines it.

| pair | months | window | correlation |
|---|---:|---|---:|
| **value vs trend** | 533 | 1982-02 → 2026-06 | **-0.164** (Spearman -0.091) |
| **value vs trend, 3-sleeve overlap** | 269 | 2004-02 → 2026-06 | **-0.281** |
| **value (skip-12m, D3) vs trend** | 533 | 1982-02 → 2026-06 | **-0.013** |
| value vs carry | 269 | 2004-02 → 2026-06 | +0.016 |
| value *rates block* vs carry | 269 | 2004-02 → 2026-06 | -0.012 |
| trend vs carry | 269 | 2004-02 → 2026-06 | -0.044 |

**The AMP negative correlation reproduced — and then dissolved.** The value score is
`-(trailing 5-year return)`, so it contains the 1/3/6/12-month windows the trend sleeve
trades. D3 removes the most recent 12 months and nothing else; the correlation goes from
-0.164 to **-0.013**, i.e. **92% of the diversification was the two signals looking at the
same twelve months with opposite signs.** The prereg named this in advance as the second
honest failure mode and pre-registered D3 specifically to size it. It is sized: the
diversification is **MECHANICAL**.

That is not a claim that AMP are wrong. It is a claim about *this* implementation: with a
price-reversal proxy standing in for book-to-market, "value" is largely just inverted
momentum, and inverted momentum is not an independent bet.

## 2. The portfolio arithmetic, which is the whole game

`S = s*sqrt(N/(1+(N-1)*rho))`, and max compound growth `= 3S²/8` at half Kelly, so
**30%/yr needs S = 0.894**. Measured on the sleeves that exist:

| portfolio | Sharpe | half-Kelly compound |
|---|---:|---:|
| trend alone (1982+ overlap, net 10bps) | 0.628 | 14.8%/yr |
| trend alone (2004+ overlap) | 0.475 | 8.5%/yr |
| carry alone (2004+ overlap) | 0.430 | 6.9%/yr |
| **value alone (1982+)** | **-0.082** | 0 |
| trend + carry, equal risk (2004+) | **0.655** | **16.1%/yr** |
| trend + carry + **value**, equal risk (2004+) | **0.598** | 13.4%/yr |
| value + trend, MV-optimal weights (1982+) | 0.628 | 14.8%/yr |

**These are Sharpe ratios, not verdicts.** Sleeves 1 and 2 returned their own verdicts
independently and neither is a validated strategy: trend is **DEAD** (it clears the DSR bar
but loses to a vol-matched long-only hold of its own universe) and carry is **DEAD** (it
beats its benchmark but cannot clear the DSR bar on 22.4 years). The Kelly ceiling `3S²/8`
depends only on the Sharpe, so the table above is the right arithmetic for "how much
compound growth is reachable" — but nothing in it should be read as a deployable book.

**Adding this sleeve at equal risk destroys 8.6% of the portfolio Sharpe.** The in-sample
optimal weight is 3.5% of risk and lifts value+trend from 0.6278 to 0.6282. Both statements
say the same thing: a negatively-correlated sleeve still has to pay its own way, and this
one does not.

**What it would have had to deliver.** Inverting
`S_c = sqrt((S1² + S2² - 2*rho*S1*S2)/(1-rho²))` for the value Sharpe that reaches 0.894:

| paired with | its Sharpe | rho | value Sharpe REQUIRED for 30%/yr |
|---|---:|---:|---:|
| trend (1982+) | 0.628 | -0.164 | **0.524** |
| trend (2004+) | 0.475 | -0.281 | **0.594** |
| trend+carry (2004+) | 0.655 | -0.281 | **0.401** |

Even at the friendliest pairing the sleeve needed **+0.40**. It produced **-0.08**.

## 3. What was built, and the one thing that honestly could not be

| block | value score | n | first tradable |
|---|---|---:|---|
| Equity index | `-(trailing 60m cumulative log excess return)` | 7 | 1976-01 |
| Rates (par-bond TR) | term spread `y - y_13w` minus its own **expanding** mean | 3 | 1982-01 |
| Commodity futures | `-(trailing 60m cumulative log excess return)` | 4 | 2005-08 |

**FX was EXCLUDED and no FX value claim is made.** The value measure for a currency is the
deviation from long-run PPP — the nominal rate deflated by the two countries' price levels.
**This panel contains no price-level series for any country**: no CPI, no deflator, no
inflation forecast, and nothing from which one can be derived. A nominal 5-year change in
spot is a *different signal*, and substituting it would have been dressing one thing as
another. Cost of the exclusion, stated plainly: 4 of 18 instruments and one entire asset
class of diversification are absent, which is part of why effective N is only 4.7.

**Real yields could not be built for the same reason**, so the brief's stated alternative —
term spread versus its own long-run average — was used for bonds.

**Sample:** 1982-02 → 2026-06, **533 months = 44.4 years**, mean 10.9 tradable instruments
(min 6, max 14), **effective N = 4.69**. Zero interior null cells. The unscreened panel is
**bit-identical** on this universe (the 2008 quarantine touches only EURUSD and JPYUSD,
both excluded) — asserted programmatically, D6.

## 4. Headline numbers

All net of cost, monthly rebalance, benchmark = equal-weight **long-only** buy-and-hold of
exactly the same tradable set, paying the same cost schedule.

| vol target | gross Sharpe | net Sharpe 2bps | net Sharpe 10bps | turnover/yr | mean gross lev | cap binds |
|---|---:|---:|---:|---:|---:|---:|
| 10% | -0.018 | -0.027 | -0.064 | 10.8x | 4.31x | 0.0% |
| **20%** | **-0.038** | **-0.047** | **-0.082** | 18.6x | 7.88x | 9.5% |
| 40% | -0.067 | -0.076 | -0.109 | 22.9x | 9.94x | 41.2% |

Sharpe is near-invariant to the target as pre-registered; the drift is the 10x leverage cap
binding, which is the only channel through which the targets can differ.

**At 20% vol (the reporting reference):**

| | strategy net 10bps | benchmark net 10bps |
|---|---:|---:|
| mean annual (excess) | **-1.73%** | **+8.11%** |
| volatility | 20.96% | 10.74% |
| Sharpe | **-0.082** | **+0.755** |
| t-stat (Newey-West, lag 6) | -0.58 | +4.75 |
| max drawdown | **-94.6%** | -34.3% |
| worst month | -25.3% | -16.2% |
| skew / excess kurtosis | +0.48 / 2.67 | -0.53 / 1.93 |

**Costs.** Turnover is 18.6x/yr — high, because a rank-weighted long/short book re-trades
when a *rank* moves, not only when a *sign* flips. Cost drag is 0.19%/yr at 2bps and
0.93%/yr at 10bps (exactly 5x, verified). The benchmark turns over 0.06x/yr, so its cost is
0.003%/yr. **Costs are not what killed this: it is negative gross.**

## 5. Arithmetic active return, and the variance-drag trap it was pre-registered to avoid

| measure (net 10bps) | value | t-stat (NW lag 6) |
|---|---:|---:|
| **Arithmetic active return (PRIMARY)** | **-9.84%/yr** | **-2.80** |
| Jensen alpha (beta = 0.033) | -1.99%/yr | -0.66 |
| Vol-matched active | -8.99%/yr | -3.84 |
| Geometric excess | -11.63%/yr | — |
| Variance drag `(var_s - var_b)/2` | **+1.62%/yr** | — |

Identity `geometric excess = arithmetic active - variance drag` reproduces to 0.0017 (the
residual is the arithmetic/geometric second-order term, not a defect).

**Read this honestly in both directions.** The headline arithmetic active is -9.84% with
t = -2.80, but a dollar-neutral book has a beta of 0.033 to a long-only benchmark that
itself earned +8.11%/yr, so most of that -9.84% is *the strategy not owning beta* rather
than negative skill. The beta-adjusted number — Jensen alpha **-1.99%/yr, t = -0.66** — is
statistically indistinguishable from zero. **The verdict does not depend on which one you
prefer: one is significantly negative, the other is indistinguishable from zero, and neither
is an edge.** The pre-registered headline is the arithmetic number and it is reported as
such, with the reason it is harsh stated rather than used as a hedge.

Note the direction of the variance-drag term here: it is **+1.62%/yr in the strategy's
favour**, so the geometric comparison would have flattered the strategy relative to the
arithmetic one had the sign of the result been marginal. It was not, so the trap did not
have to be sprung — but it was checked mechanically, which is the point.

## 6. Sharpe per decade — no era carries it

| decade | months | strategy net 10bps | benchmark net 10bps |
|---|---:|---:|---:|
| 1980s (from 1982-02) | 95 | **-0.124** | +1.239 |
| 1990s | 120 | **-0.258** | +0.944 |
| 2000s | 120 | +0.091 | +0.314 |
| 2010s | 120 | **-0.178** | +0.654 |
| 2020s (to 2026-06) | 78 | +0.071 | +0.844 |

Negative in 3 of 5 decades; the two positive decades are +0.09 and +0.07, i.e. noise. There
is no era in which this worked. The pre-2009 / post-2009 split agrees: -0.116 and -0.030.

## 7. Deflated Sharpe, and Kelly

`dsr_sharpe_bar` at 44.42 years (verified against both recorded anchors: 1.4881 at 7yr and
0.5971 at 40yr, n_trials=32):

| n_trials | DSR>=0.95 bar | net Sharpe 10bps | pass? |
|---:|---:|---:|---|
| 32 (anchor convention) | 0.566 | -0.082 | **NO** |
| 40 (honest cumulative incl. this study) | 0.580 | -0.082 | **NO** |
| 48 (pessimistic) | 0.591 | -0.082 | **NO** |

The verdict is insensitive to the trial-count accounting, which is why it was reported at
three counts.

**Half Kelly.** `g = 3S²/8` is undefined in any useful sense on a negative Sharpe (the
optimal allocation is zero, or a short, which is addressed in §10). For reference, 30%/yr
needs **S = 0.894**; the best portfolio measured anywhere in this programme is trend+carry
at **0.655 → 16.1%/yr**.

## 8. Concentration, structural tilt, and where the leverage actually sits

**No single cell or year dominates.** Because total P&L is negative, signed shares are
uninterpretable, so absolute shares are reported: the largest single (instrument, month) is
**0.32%** of gross absolute P&L (N225, 1982-11) and the largest calendar year is **7.5%**
(1990). This is the one pre-registered failure mode the sleeve did *not* have.

**Two persistent structural positions were found, and they are conventions, not value.**

| instrument | mean weight / mean \|weight\| | % of live months long |
|---|---:|---:|
| **FTSE100** | **+0.90** | **93.1%** |
| **US10Y_TR** | **+0.90** | **95.1%** |
| NASDAQ | -0.45 | 25.0% |
| GOLD_F | -0.32 | 32.0% |
| everything else | -0.24 … +0.29 | 36–68% |

FTSE100 is long in 93% of months. **This is the dividend-convention bias the prereg flagged
in advance:** six of the seven equity indices are price indices, and the UK's dividend yield
(~3–4%/yr) is roughly double the US's, so over a 5-year window the FTSE's price return is
mechanically ~10% lower than a like-for-like total return. A 5-year *return* score therefore
ranks it cheap permanently. **That is a data convention being traded as if it were a
valuation**, and it is one reason the equity block is anti-predictive.

**Gross leverage is dominated by the bonds.** Inverse-vol sizing hands the low-volatility 5y
bond an enormous notional:

| instrument | share of mean gross leverage |
|---|---:|
| **US5Y_TR** | **42.7%** |
| US30Y_TR | 14.8% |
| US10Y_TR | 7.9% |
| all 7 equity indices combined | 27.4% |
| all 4 commodities combined | 7.3% |

**65.4% of the book's gross notional is three US Treasury points.** Mean gross leverage is
7.88x and the 10x cap binds in 9.5% of months at the 20% target. Any deployment of a book
like this would be a rates book with an equity overlay, whatever the label says.

## 9. Diagnostics — all pre-registered, none promotable

**D2 — per-block sub-books (20% vol, net 10bps).** This is where the result actually lives:

| block | months | gross Sharpe | net Sharpe 10bps | t-stat | corr to PRIMARY |
|---|---:|---:|---:|---:|---:|
| **Equity** | 605 | -0.343 | **-0.360** | **-2.79** | +0.68 |
| Rates | 533 | +0.140 | +0.077 | +0.65 | +0.42 |
| Commodity | 250 | +0.352 | +0.338 | +1.50 | +0.63 |

The equity block is **significantly negative**. Commodity is the only block with a positive
sign and it is not significant on 250 months. Confirmed independently by the raw rank IC in
§11 (V2), which uses no book at all.

**D3 — skip the most recent 12 months.** Net Sharpe -0.071 (vs -0.082), correlation to
PRIMARY +0.74 — the standalone result is unchanged. **Correlation to trend collapses from
-0.164 to -0.013.** This is the single most informative number in the study.

**D4 — bonds scored by 5-year reversal instead of the term spread.** Net Sharpe **-0.186**,
worse than PRIMARY. The term-spread-vs-own-mean signal is the better of the two bond
choices, though "better" here means less bad.

**D1 — negative control (signal signs randomised, 8 seeds).** Control Sharpes: -0.188,
+0.099, -0.121, +0.008, -0.274, -0.166, -0.303, +0.016. Mean **-0.116**, SD **0.145**.
**The live sleeve sits at z = +0.23 inside that distribution.** A randomly-signed book is
statistically indistinguishable from the real one. This is the cleanest available statement
that there is no signal here.

**D5 — sub-period split.** Pre-2009: Sharpe -0.116, active -10.11%/yr (t = -2.02). Post-2009:
Sharpe -0.030, active -9.42%/yr (t = -2.10). Consistently bad; not an era effect.

**D6 — unscreened panel.** Bit-identical on this universe, as predicted. Asserted in code.

## 10. The tempting inversion, and why it is not a result

The equity block's IC is significantly *negative*, so an inverted sleeve — long 5-year
winners, short 5-year losers, i.e. **long-horizon momentum** — would show a positive Sharpe
on this sample. **That is not being claimed and must not be promoted.** Three reasons:

1. It is a **post-hoc sign flip on a pre-registered signal**, which is the purest form of
   the mistake the whole pre-registration discipline exists to prevent. It would need its
   own prereg and its own trial count.
2. It is **not a diversifier**. Inverting the sleeve inverts the correlation: -0.164 becomes
   **+0.164** to trend, +0.281 on the modern overlap. The programme's binding constraint is
   `rho`, and a second momentum sleeve makes `rho` worse, not better.
3. Most of it would be the **FTSE dividend-convention tilt** (§8) running the other way,
   which is a data artefact regardless of its sign.

Recorded because it is the honest reading of the IC, and flagged because acting on it would
be curve-fitting.

## 11. Verification — the negative was attacked before it was accepted

`research/sleeves/multiasset_value_verify.py` → `research/sleeves/_value/verification.json`.

**V1 — the signal says the obvious things.** Sign convention confirmed on three cases whose
answer is known in advance:

| series | dearest (score minimum) | cheapest (score maximum) |
|---|---|---|
| NASDAQ | **2000-02** (the exact dot-com top) | 2005-02 |
| SPX | 1937-05 (after the 1932–37 rally) | **1934-08** (after the 1929–32 crash) |
| GOLD_F | **2011-08** (the exact gold peak) | 2016-11 |

A sign error would have shown here and did not.

**V2 — raw predictive content, no book, no sizing, no costs.** Cross-sectional Spearman IC
of the value score against the *next* month's return:

| block | months | mean IC | t-stat (NW) | % months positive |
|---|---:|---:|---:|---:|
| **Equity** | 605 | **-0.068** | **-3.14** | 44.1% |
| Rates | 533 | +0.028 | +0.73 | 51.2% |
| Commodity | 250 | +0.084 | +2.16 | 54.0% |

The negative equity result is in the **signal**, not the machinery.

**V3 — perfect-foresight positive control.** The value score replaced by next month's
*actual* return, everything else identical: gross Sharpe **5.85**, net Sharpe **5.59**. The
pipeline can express an edge; it did not find one.

**V4 — headline numbers recomputed from the written CSV** by an independent path: net Sharpe
-0.08240 (matches to 13 significant figures), active -9.835%/yr, NW t -2.804, iid t -2.802,
corr to trend -0.16448.

**V5** — leverage concentration (§8). **V6** — DSR anchors reproduce (1.4881 / 0.5971 vs
1.488 / 0.597). **V7** — cost arithmetic exactly 5x between brackets.

## 12. Pre-registration scorecard — what the forecast got right and wrong

| quantity | predicted (80% interval) | realised | inside? |
|---|---|---:|---|
| Net Sharpe at 2bps | 0.30 (-0.10 … 0.65) | -0.047 | **no**, below |
| **Net Sharpe at 10bps** | 0.25 (-0.15 … 0.60) | **-0.082** | yes, bottom edge |
| Arithmetic active return | +0.5%/yr (-3% … +4%) | **-9.84%/yr** | **no**, far below |
| t-stat of active return | 0.6 (-1.2 … 2.0) | **-2.80** | **no**, far below |
| **Correlation to trend** | -0.25 (-0.55 … +0.10) | **-0.164** | **yes** |
| Sharpe positive in every decade | no | no | **yes** |
| P(clears DSR bar) = 15% | — | did not clear | consistent |
| P(correlation negative) = 70% | — | negative | consistent |

**The forecast was well calibrated on the thing it was run for (correlation, and the DSR
failure) and badly calibrated on the active return.** The reason is specific and worth
recording: the prediction did not account for the benchmark's own strength. An equal-weight
long-only book on this universe earned **8.11%/yr excess at Sharpe 0.755**, so a
dollar-neutral strategy is charged that entire amount as negative active return before any
skill is measured. The **Jensen alpha of -1.99%/yr (t = -0.66) is inside the predicted
interval**; the arithmetic active is not. Lesson for the next sleeve: when the book is
market-neutral, forecast the *alpha*, not the raw active return, or forecast the benchmark
too.

## 13. What this changes for the programme

1. **Cross-asset value, as constructible from free price data, is dead.** Not marginal, not
   undetermined — negative gross, negative in 3 of 5 decades, indistinguishable from a
   randomly-signed book, and anti-predictive in its largest block. **Thirteen studies have
   now failed.**
2. **A negatively-correlated sleeve still has to pay its own way.** This sleeve delivered the
   documented negative correlation (-0.16, -0.28 on modern data) and it *still* subtracted
   from the portfolio, because `S_c` depends on both `rho` and `s`. The required standalone
   Sharpe was 0.40–0.59 depending on the pairing. **rho is necessary, not sufficient.**
3. **Check whether a diversification benefit is mechanical before banking it.** 92% of this
   sleeve's negative correlation to trend was the two signals reading the same twelve months.
   Any future sleeve claiming low correlation should be run through the D3 treatment —
   remove the overlapping window and re-measure — before the correlation is used in a
   portfolio calculation.
4. **Price-index conventions leak into long-horizon signals.** A 5-year *return* score on
   dividend-excluding price indices with different national dividend yields produces a
   permanent long-FTSE tilt (93% of months). Any signal with a multi-year lookback on this
   panel needs a total-return series or an explicit dividend correction. Short-horizon
   signals such as trend are far less exposed.
5. **The programme's best measured portfolio remains trend + carry at Sharpe 0.655
   (half-Kelly 16.1%/yr) — and both of its components carry a DEAD verdict of their own**
   (trend loses to its own vol-matched universe; carry cannot clear the DSR bar on 22.4
   years). 30%/yr needs 0.894. The gap is still open, value did not close any of it, and the
   0.655 is a ceiling on an unvalidated pair rather than a route.

---

**Trial accounting:** this study adds **1** trial (PRIMARY). Diagnostics D1–D6 are
non-promotable by pre-registration and are not counted. DSR bars are reported at n_trials
32 / 40 / 48 so the verdict does not rest on the accounting.

**Run once. No tuning. No second look.**
