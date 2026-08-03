# Breadth expansion — the effective-N lever, measured

**Built** 2026-07-28 by `research/multiasset/breadth_build.py`,
`research/sleeves/breadth_neff.py`, `research/sleeves/breadth_ladder.py`.
**Source** yfinance, free tier, 25 new tickers, 0 fetch failures. Raw data stays in
`_data/multiasset/breadth/` — gitignored. Everything in this file is a derived statistic.

Iteration 12 identified breadth as the only lever left that can move the growth ceiling,
and named the number to beat: a correlation-effective **N_eff of 5.26**, against a
requirement of **~13** for a measured 30%/yr. This is the measurement.

---

## 0. The answer

**The ceiling does not move. 30%/yr remains unreachable.**

N_eff *did* rise — from **5.26 to 8.38** — and that is a real, matched-window increase.
It bought **nothing**, because the ceiling is `S²/2` with `S = s·√N_eff`, and the added
bets are independent **and worse**. Measured on the same window, per-bet Sharpe **fell
from 0.279 to 0.169** while N_eff rose 63%. The product went **down**.

| | iteration 11 (18 instruments) | best expanded variant | change |
|---|---:|---:|---:|
| **Highest compound at max DD ≤ 50%** (coarse rungs, as recorded) | **12.30%** | **12.43%** | **+0.13pp** |
| Highest compound at max DD ≤ 50% (fine grid) | 12.59% | 13.31% | +0.72pp |
| **Peak compound at any leverage** | **15.83%** | **16.05%** | **+0.22pp** |
| Peak compound at RETAIL financing (bill+300bp) | 11.13% | 11.24% | +0.11pp |
| Portfolio Sharpe, equal weight, net 10bps | 0.6678 | 0.6713 | +0.0035 |

The target is 30%. The best free-data breadth expansion available moved the ceiling by
**0.22 percentage points**, and the variant that did it (`expanded_long_history_only`)
did so by *excluding* most of the additions. The headline expanded panel — all 37
instruments — moved the peak **down**, from 15.83% to 15.59%.

Sharpe required for 30%/yr at half Kelly is **0.894**. The best measured is **0.6713**.

---

## 1. Both controls reproduced before anything else was reported

Neither number below is comparable unless the harness reproduces the old one, so both
runners assert it in code and refuse to print anything else on failure.

| control | recorded (iteration 11) | measured here | error |
|---|---:|---:|---:|
| N_eff, original 18, 1996+ | 5.2602648349 | 5.2602648349 | 1.6e-11 |
| compound at DD ≤ 50% | 12.29548756% | 12.29548756% | < 1e-6 |
| its max drawdown | −47.2873856% | −47.2873856% | < 1e-6 |
| peak compound, any leverage | 15.83% | 15.8281% | 0.0019pp |

The ladder imports `build_book`, `levered`, `ladder`, `drawdown_report` and
`weight_concentration` from `research/sleeves/riskparity.py` **unmodified**, and
`effective_n` from `multiasset_trend.py` unmodified. The only thing that differs between
iteration 11's numbers and these is the universe.

Repo suite after this work: **1389 passed, 1 skipped**.

---

## 2. The definition, stated explicitly

Let `C` be the Pearson correlation matrix of month-end **excess** returns and
`λ₁…λₙ` its eigenvalues:

```
N_eff = (Σ λᵢ)² / Σ λᵢ²
```

the participation ratio of the eigenvalue spectrum. For `n` uncorrelated instruments
every eigenvalue is 1 and `N_eff = n`; for `n` instruments that are one bet, one
eigenvalue is `n` and the rest are 0, so `N_eff = 1`.

Iteration 11's convention — pandas' default **pairwise-complete** correlations on the
**1996-onward** slice — is what reproduces 5.26, so it is what is used throughout.
Complete-case (`dropna(how="any")`) values are reported alongside, because with staggered
start dates the pairwise matrix need not be positive semi-definite and the two can
disagree.

---

## 3. What was built, and what could NOT be built

19 tradable additions (18 fetched + 1 synthetic), 6 validation-only series. Daily panel
9,362 × 19; month-end 438 × 19. Monthly panel reconciled cell-for-cell by an independent
code path: **max discrepancy 0.0 over 4,746 cells**. 0 `inf`, 0 all-NaN columns.

| block | keys | first month |
|---|---|---|
| agriculture (7) | CORN_F ZC=F, WHEAT_F ZW=F, SOYBEAN_F ZS=F, SUGAR_F SB=F, COFFEE_F KC=F, COTTON_F CT=F, COCOA_F CC=F | 2000-01 … 2000-09 |
| livestock (2) | CATTLE_F LE=F, HOGS_F HE=F | 2000-12, 2001-03 |
| volatility (1) | VIX_ETF **VIXY** | 2011-01 |
| credit (3) | HYG, LQD, CREDIT_SPREAD = HYG − IEF | 2002-08 … 2007-04 |
| foreign sovereign (3) | GILT_ETF IGLT.L, BUND_ETF EXX6.DE, JGB_ETF 1482.T | 2008-01, 2008-01, 2016-06 |
| real assets (3) | REIT VNQ, FREIGHT BDRY, CARBON KRBN | 2004-10, 2018-03, 2020-08 |

**Volatility: which proxy and why.** `^VIX` is an index level, not an instrument — there
is no spot VIX position, so its return is not earnable by anyone. It is carried
validation-only. The tradable proxy is **VIXY**, not VXX: Yahoo's VXX history starts
**2018-01** because the original iPath ETN matured and the Series B replaced it, whereas
VIXY runs continuously from **2011-01**. Measured, `VIX_ETF ~ VIX_SPOT` correlate 0.897
and the spot index out-drifts the tradable proxy by **+123.7%/yr** — that gap *is* the
VIX futures roll. VIXY's own CAGR over 15.5 years is **−48.5%/yr**. Volatility is not an
asset you can hold; it is an insurance premium you pay.

**Stated honestly: what could not be constructed.** Probed and measured, not assumed:

- **No non-US sovereign yield series exists on free Yahoo.** `^GDBR10`, `^GDBR2`,
  `^GDBR30`, `GDBR10.EX`, `^GBGB10`, `^GBGB10Y`, `^JP10YT`, `JP10Y-JP`, `GB10Y-GB`,
  `DE10Y-DE` all return empty. So the par-bond total-return construction that gives the
  US rates block **64 years** of history cannot be replicated for Bunds, gilts or JGBs.
- **No non-US sovereign futures either.** `FGBL=F`, `FGBM=F`, `FGBS=F` (Bund/Bobl/Schatz),
  `GG=F`, `R=F` (gilt), `JGB=F` all return empty.
- Consequence: non-US sovereigns are available **only as funded ETFs from 2008** (2016 for
  JGBs), in local currency, which is 16 years of history against the US block's 64.
- **No free freight index.** `^BDI` and `BADI` are empty; `BDI` returns an unrelated
  453-observation equity series. Only **BDRY (2018-03)** exists — 8 years.
- **No free carbon futures.** `ECF=F`, `CFI2Z5.NYM`, `CARB` empty. Only **KRBN (2020-07)**
  — 6 years.

---

## 4. Integrity: the ag/livestock block is roll-contaminated, and worse than NATGAS_F

The original builder condemned `NATGAS_F` because 65.7% of its |r|>15% bars landed in
days 24–31 against a 24.0% base rate — a **2.74× lift**. The same discipline is applied
here, strengthened in two ways: a **variance-share** test (a splice does not have to clear
a 15% threshold to dominate a series' variance), and a **declared roll window** taken from
each contract's own last-trading-day rule rather than fitted.

**The noise floor first.** The same single-day scan on the original panel's four futures
that §6a *cleared*:

| series | best single day | ratio |
|---|---|---:|
| GOLD_F | d19 | 1.46 |
| WTI_F | d22 | 2.01 |
| SILVER_F | d30 | 2.08 |
| COPPER_F | d31 | 2.19 |
| *NATGAS_F (condemned)* | *d29* | *2.91* |

Against that floor:

| series | declared window | bars | variance | ratio | worst single day | ratio |
|---|---|---:|---:|---:|---|---:|
| **HOGS_F** | 12–18 | 23.2% | **64.9%** | **2.80** | **d15** | **10.11** |
| **CATTLE_F** | 1–2 | 6.3% | **29.4%** | **4.67** | **d1** | **7.55** |
| **CORN_F** | 10–16 | 23.4% | 35.3% | 1.51 | **d15** | **4.03** |
| **SUGAR_F** | 1–2 | 6.3% | 15.0% | 2.39 | **d1** | **3.71** |
| **SOYBEAN_F** | 10–16 | 23.4% | 36.0% | 1.54 | **d15** | **3.52** |
| COCOA_F | 12–22 | 36.5% | 41.7% | 1.14 | d16 | 2.24 |
| COTTON_F | 3–12 | 33.3% | 37.5% | 1.13 | d10 | 2.08 |
| WHEAT_F | 10–16 | 23.4% | 26.5% | 1.13 | d15 | 1.92 |
| COFFEE_F | 18–28 | 35.8% | 38.4% | 1.07 | d20 | 1.69 |

**Five of nine are more contaminated than the series the programme already condemned.**
Lean hogs at 10.11 is 3.5× worse than NATGAS_F. The declared window and the blind
single-day scan land on the **same day** in every case — day 15 for the grains and hogs,
day 1 for cattle and sugar — which is exactly the contract calendar, so this is not a
multiple-testing artefact.

**A correction to my own reasoning, disclosed.** The first version of the roll table put
SUGAR_F and CATTLE_F at days 25–31 because both contracts expire on a *last business day*.
That was wrong about where the splice lands: a contract that trades to the last business
day is still the front month on that day, so the spliced bar is the **first bar of the
following month**. CATTLE_F scored 0.77 (nothing) on the wrong window and 7.55 on the
right one. Both were moved to (1, 2).

**The grains' clustering is the roll, not the USDA.** Grain extremes clustering on a
calendar day has an innocent explanation — WASDE reports are released the 9th–12th — so
the two windows were separated:

| series | roll window d14–15 (6.6% of bars) | WASDE window d9–12 (13.5% of bars) |
|---|---:|---:|
| CORN_F | 15.8% of variance, **ratio 2.38** | 14.6%, ratio 1.08 |
| SOYBEAN_F | 14.6% of variance, **ratio 2.20** | 16.8%, ratio 1.24 |
| WHEAT_F | 8.6% of variance, ratio 1.29 | 13.7%, ratio 1.01 |

The scheduled-news window shows **essentially nothing**. The splice window shows the
effect. Attribution is measured, not asserted.

**Corroborated against roll-managed funds.** The GOLD_F/GLD test, applied here. Gold and
silver front-month track their physical ETFs to within −0.40%/yr and −0.31%/yr. These do
not:

| pair | corr | benchmark − front-month | vol ratio |
|---|---:|---:|---:|
| WHEAT_F ~ WEAT | 0.906 | **−12.23%/yr** | 1.15 |
| CORN_F ~ CORN | 0.829 | **−5.29%/yr** | 1.24 |
| SUGAR_F ~ CANE | 0.750 | −2.99%/yr | 1.17 |
| COCOA_F ~ NIB | 0.887 | −2.19%/yr | 0.98 |
| SOYBEAN_F ~ SOYB | 0.803 | +0.50%/yr | 1.17 |
| **CATTLE_F ~ COW** | **0.457** | **−11.63%/yr** | 1.03 |
| **HOGS_F ~ COW** | **0.420** | **−19.36%/yr** | **2.47** |

The un-back-adjusted front-month series **manufacture return**: in a carry (contango)
market the splice from the expiring cheap front to the dearer next contract prints as a
price rise that no holder earned. Teucrium's ~1%/yr expense ratio accounts for one point
of these gaps, not twelve.

**Both biases run in the flattering direction, which is what makes the null answer
robust.** Splice noise is idiosyncratic by construction, so it inflates *apparent*
independence (N_eff) *and* inflates return. The headline expanded panel therefore uses the
front-month series **as fetched** — the most favourable possible treatment — and the
decontaminated variants are reported as sensitivities. If even the flattered version does
not reach the target, the conclusion does not depend on the cleaning decision.

**Other guards, all clean.** Chronological sort/dedupe on all 25 series (0 unsorted, 0
duplicate dates). |return| > 50%: **0 in the shipped panel, 0 in a naive build** — this
block contains no yield series, so the failure mode that produced 241 prints in the
original naive build cannot occur. Non-positive price guard: 0 triggers. Long-gap nulling:
1 bar. Day-of-month top-10 signature run on every column and reported in
`integrity.json`.

---

## 5. THE DELIVERABLE — N_eff, always on a matched window

Comparing 5.26 (measured from 1996) against an expanded number measured from 2008 would
confound breadth with window, because correlations rise in crises and the modern window is
crisis-dense. Every row below re-measures the **original 18 on the same window**.

| window | months | original 18 | expanded 37 | **gain** |
|---|---:|---:|---:|---:|
| 1996+ | 366 | 5.26 / 4.92 | 8.65 / 7.01 | **+3.39** |
| 2001+ | 303 | 5.01 / 4.92 | 8.45 / 7.01 | +3.44 |
| 2008+ | 221 | 4.76 / 4.76 | 7.96 / 7.01 | +3.20 |
| **2011+** | **185** | **5.13 / 5.13** | **8.38 / 7.01** | **+3.25** |
| 2016+ | 120 | 4.86 / 4.86 | 7.98 / 7.01 | +3.12 |
| 2020+ | 70 | 4.18 / 4.18 | 6.95 / 6.95 | +2.77 |

*(pairwise / complete-case. The complete-case column is pinned at 7.01 for every window
before 2020 because CARBON starts 2020-08, so only 71 months have every series present.)*

**The gain is real and stable at about +3.2 to +3.4 effective bets**, from 19 added
instruments. That is 0.17 effective bets per instrument added.

### Which additions actually bought independence (2011+)

| block | n | N_eff of the block alone | marginal N_eff added | per instrument |
|---|---:|---:|---:|---:|
| **agriculture** | 7 | **5.12** | **+2.90** | 0.41 |
| livestock | 2 | 1.96 | +0.92 | 0.46 |
| real assets | 3 | 2.76 | +0.72 | 0.24 |
| foreign sovereign | 3 | **1.40** | **+0.08** | **0.03** |
| volatility | 1 | — | −0.07 | −0.07 |
| credit | 3 | **1.81** | **−0.27** | **−0.09** |

**Agriculture is the only block that bought material independence.** Seven crops behave
like 5.12 independent bets on their own, and add 2.90 to the panel — weather and acreage
really are not macro factors.

**The most intuitive idea in the brief was the emptiest.** "Non-US sovereigns under
different central banks" sounds like three new bets. Measured, gilts + Bunds + JGBs are
**1.40 effective bets between them**, and adding all three to the panel raises N_eff by
**0.08**. Global duration is one trade regardless of which central bank prices it — the
same failure the US Treasury block already showed at 1.17 for three maturities.

**Credit is negative.** HYG, LQD and HYG−IEF are 1.81 effective bets, and adding them
*reduces* N_eff by 0.27: high-yield credit is equity risk and duration recombined, both
of which the panel already owns.

**Roughly half the gain is splice noise.** Dropping the five series whose contamination
exceeds NATGAS_F takes N_eff from 8.38 to **6.76**, i.e. the matched gain falls from +3.25
to +1.63. Half of the measured independence came from bars that are not returns.

---

## 6. THE LADDER — the two numbers that are the entire result

Equal weight, monthly, long-only, both cost brackets, both financing rates, the same code
iteration 11 ran. Every drawdown is on the compounded **total-return** path.

### Headline (10bps, bill + 150bp)

| universe | n | yrs | **DD ≤ 50%** (fine) | its DD | **peak, any leverage** | its DD | EW Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| **original_18** (control) | 18 | 61.5 | **12.59%** | −49.9% | **15.83%** | −87.8% | 0.6678 |
| expanded_37 | 37 | 61.5 | 12.88% | −49.6% | 15.59% | −87.1% | 0.6464 |
| expanded_no_livestock | 35 | 61.5 | 12.47% | −47.9% | 15.37% | −85.9% | 0.6319 |
| expanded_no_rollcontam | 32 | 61.5 | 11.78% | −48.2% | 14.78% | −89.9% | 0.6365 |
| expanded_no_vol | 36 | 61.5 | 13.23% | −49.6% | 15.92% | −82.9% | 0.6665 |
| **expanded_long_history_only** | 33 | 61.5 | **13.31%** | −49.6% | **16.05%** | −82.9% | 0.6713 |
| expanded_roll_managed | 37 | 61.5 | 11.16% | −48.6% | 13.13% | −90.3% | 0.6227 |

At the coarse pre-registered rungs, the DD ≤ 50% figures are 12.30% (original) against a
best of 12.43%. Peak leverage is τ ≈ 0.38–0.42, mean gross ≈ 4.8–5.5×.

### Full equal-weight ladder, compound / max DD

**10bps, bill + 150bp**

| universe | τ=10% | τ=15% | τ=20% | τ=25% | τ=30% | τ=40% |
|---|---|---|---|---|---|---|
| original_18 | 10.56 / −33 | **12.30 / −47** | 13.64 / −59 | 14.61 / −69 | 15.26 / −77 | 15.81 / −89 |
| expanded_37 | 10.41 / −30 | 12.07 / −43 | 13.36 / −54 | 14.28 / −64 | 14.90 / −72 | 15.55 / −85 |
| expanded_no_rollcontam | 10.22 / −34 | 11.78 / −48 | 12.95 / −60 | 13.74 / −70 | 14.21 / −78 | 14.78 / −90 |
| expanded_long_history_only | 10.65 / −30 | **12.43 / −43** | 13.82 / −54 | 14.83 / −64 | 15.50 / −72 | 16.00 / −85 |
| expanded_roll_managed | 9.82 / −34 | 11.16 / −49 | 12.11 / −61 | 12.69 / −70 | 12.94 / −80 | 13.07 / −92 |

**10bps, RETAIL bill + 300bp** — iteration 11's largest single lever, unchanged

| universe | τ=10% | τ=15% | τ=20% | τ=25% | τ=30% | τ=40% |
|---|---|---|---|---|---|---|
| original_18 | 10.11 / −33 | 10.83 / −48 | 11.12 / −61 | 11.02 / −71 | 10.59 / −79 | 9.08 / −93 |
| expanded_37 | 9.84 / −30 | 10.45 / −43 | 10.62 / −55 | 10.43 / −67 | 9.93 / −79 | 8.44 / −93 |
| expanded_long_history_only | 10.16 / −30 | 10.92 / −43 | 11.23 / −55 | 11.15 / −65 | 10.73 / −77 | 9.11 / −93 |

The retail ladder still **inverts** past τ ≈ 20%. Breadth did not change that: the peak at
retail financing moves from 11.13% to 11.24%. **The borrowing rate remains a bigger lever
than every strategy decision in 24 studies combined**, and it is not a research variable.

At the 2bps bracket the DD ≤ 50% rung is 12.33% (original) against 12.46% (best expanded)
— a 0.13pp difference, i.e. the cost bracket is not what is binding either.

---

## 7. Why N_eff rose 63% and the ceiling did not move

Iteration 12's arithmetic held per-bet Sharpe `s` **fixed** at 0.2913 and solved for
N_eff. Applied to the new N_eff it predicts a lot:

> N_eff 8.38 ⟹ S = 0.843 ⟹ half-Kelly 26.7% ⟹ expected measured peak **25.2%**

**That prediction is refuted by measurement.** On the same 2011+ window the measured peak
is **6.52%**. The error is not in the ceiling model — it is in the constant-`s`
assumption.

| 2011+, matched window | N_eff | portfolio Sharpe | **implied per-bet s** | mean instrument Sharpe |
|---|---:|---:|---:|---:|
| original 18 | 5.14 | 0.632 | **0.279** | 0.284 |
| expanded 37 | 8.39 | 0.489 | **0.169** | 0.192 |

N_eff rose **63%** (which alone would lift S by 27.7%, to 0.807). Per-bet Sharpe fell
**39%**. `S = s·√N_eff` therefore fell from 0.632 to 0.489, and `S²/2` fell with it.

The per-instrument Sharpes say it plainly (2011+):

- **original 18, mean +0.284** — NASDAQ +0.95, SPX +0.89, N225 +0.78, DAX +0.58, GOLD_F +0.49 …
- **additions, mean +0.105** — CARBON +0.60, HYG +0.47, REIT +0.43, CATTLE_F +0.41, LQD +0.31 … then GILT_ETF +0.01, BUND_ETF −0.00, COTTON_F −0.01, SUGAR_F −0.05, **JGB_ETF −0.72, VIX_ETF −0.76**

**This is the same mechanical failure iteration 11 found in bucketed risk parity, in a new
place.** Inverse-vol sizing is only valid if Sharpes are equal across assets; *equal
weighting a wider universe is only a gain if the added bets are as good as the existing
ones*. Here they are not, so equal weight dilutes.

**The friendliest test available makes it worse, not better.** On 61.5 years the additions
are only live near the end, so the same books were evaluated on windows where the
additions actually carry weight (eligibility and vol estimates still computed on full
history — the result series is sliced, never the input):

| evaluated from | yrs | mean eligible | EW Sharpe | DSR bar (n=46) | DD ≤ 50% | peak |
|---|---:|---:|---:|---:|---:|---:|
| 2004, original 18 | 22.5 | 17.7 | 0.729 | 0.833 | 10.47% | 14.22% |
| 2004, expanded 37 | 22.5 | 32.6 | 0.683 | 0.833 | 10.60% | 13.78% |
| **2011, original 18** | 15.5 | 18.0 | **0.632** | 1.011 | 10.02% | **10.35%** |
| **2011, expanded 37** | 15.5 | 34.8 | **0.489** | 1.011 | 6.52% | **6.52%** |

Doubling the eligible instrument count (18 → 34.8) **cut** the peak compound return by a
third. And the DSR bar on a 15.5-year sample is **1.011** — nothing here clears it.

**The ceiling model itself is sound.** On the full 61.5-year sample, `0.71 × S²/2` predicts
14.83% for the expanded panel against a measured 15.59% — a 0.76pp error. What was wrong
was the input, not the framework.

---

## 8. What N_eff would be required, and can free data supply it?

At the original panel's per-bet Sharpe of **0.2912** and iteration 11's measured
efficiency of **0.710**:

| basis | N_eff for a 30%/yr compound |
|---|---:|
| full Kelly, idealised | 7.08 |
| half Kelly, idealised | 9.44 |
| **half Kelly × measured efficiency — the honest number** | **13.29** |

Achieved: **8.38**. Shortfall: **4.91 effective bets**, on the *flattering* panel.

But that requirement is computed at `s = 0.2912`, which the measurement has now refuted.
At the expanded panel's **measured** per-bet Sharpe of **0.169**, the same target needs

```
N_eff = (8 × 0.30 / 3 / 0.710) / 0.169²  =  39.5
```

**about 40 independent bets** — against 8.38 achieved from 37 instruments spanning every
free asset class that exists. At the observed yield of **0.171 effective bets per
instrument added**, closing that gap needs a further **~182 instruments**, every one of
them as good as the ones already in the panel. There is no such free universe. The
candidates the brief named were the good ones, and they have now been measured:
**agriculture alone accounts for +2.90 of the +3.25 total, and the other twelve additions
are worth +0.35 between them** (their individual marginals sum to +4.28, and the
difference is how much they overlap each other). Half of even that is splice artefact.

**No plausible free-data universe supplies it.** The measured constraint is not the number
of tickers. It is that the bets which are genuinely independent of a global macro book are
also the ones nobody is paid much to hold.

---

## 9. Honesty requirements

**Survivorship — bias direction UPWARD, and this block is worse than the original panel.**
yfinance returns tickers that still exist. The original panel's exposure was confined to
7 of 27 instruments because index, futures, FX and rate series are near-immune. That
argument is **weaker here**:

- The **futures** additions (9) are still structurally immune — a contract expires by
  design. The defect there is roll, not survivorship, and §4 quantifies it.
- The **ETF/ETN** additions (10 of 19) carry real selection bias. This is not
  hypothetical: **four of the six validation instruments used here were delisted** — NIB,
  JO, BAL and COW all stop on 2023-07-21. A person selecting commodity ETNs in 2010 faced
  a menu that included products which no longer exist, and the survivors are the ones that
  worked. The panel can only see the survivors.
- **VXX is the cleanest single example.** Its Yahoo history begins 2018 not because
  volatility exposure began then, but because the original ETN matured. The instrument's
  own death is invisible in the series.

So the expanded panel's additions are *more* survivorship-flattered than the original 18,
and the result is still negative. That direction is what makes the conclusion safe.

**Length vs breadth — the price is stated, and it is steep.** The additions start between
2000 and 2020. On the full 61.5-year ladder they are eligible for only part of the sample
(mean eligible instruments 11.3 → 16.8, but 18.0 → 34.8 over the last 15 years). Running
the comparison on the window where they *are* live costs 46 of 61.5 years, and iteration
3c established that sample length is a property the strategy needs, not a lever:

| evaluation window | years | DSR bar, n_trials=46 | best measured Sharpe |
|---|---:|---:|---:|
| full | 61.5 | 0.499 | 0.671 ✓ clears |
| from 2004 | 22.5 | 0.833 | 0.749 ✗ fails |
| from 2011 | 15.5 | 1.011 | 0.632 ✗ fails |

**The trade is strictly bad.** The full-sample expanded panel clears its DSR bar but does
not move the ceiling; the short-sample panel where breadth is real fails its DSR bar *and*
has a lower ceiling. There is no window in which the expansion both matters and survives
the gate.

**Cumulative n_trials.** This study adds 1 (a single pre-specified breadth test, run
once). **46 → 47.** The bar at 61.5 years is unchanged at 0.4988; nothing here approaches
the 0.894 that 30%/yr requires.

**What is flattered, and by how much.** Front-month splice inflates both N_eff and return
(§4); the ETF additions are survivorship-selected (above); the panel prices no impact, no
borrow availability on 5× gross, and no tax. The `expanded_roll_managed` run — every
front-month ag/livestock series replaced by the roll-managed fund a person could actually
have bought — peaks at **13.13%**, *below* iteration 11's 15.83%. That is the closest
thing here to a holdable number, and it is worse than the starting point.

**Local currency.** GILT_ETF, BUND_ETF and JGB_ETF returns are GBP, EUR and JPY, matching
the panel's existing convention for FTSE/DAX/N225/HSI/ASX. A USD investor's return differs
by the FX move, which the panel already carries separately.

---

## 10. Verdict

**30%/yr is not reachable on the expanded panel, and breadth is not the missing lever.**

Iteration 11 established that 30% lies above the maximum of the leverage-return curve.
Iteration 12 identified effective breadth as the only quantity that could move that
maximum, and it was right that it is the only *candidate*. It is now measured, and the
candidate fails:

1. N_eff rose from **5.26 to 8.38** — genuine, matched-window, and about half of it is
   roll artefact.
2. Peak compound return moved from **15.83% to 16.05%** at best, and **down to 15.59%** on
   the headline expanded panel.
3. The reason is measured, not speculated: per-bet Sharpe **fell from 0.279 to 0.169**
   because the independent bets available for free are also the low-Sharpe ones.
4. The honest requirement was N_eff ≈ 13.3 at the old per-bet Sharpe. At the *measured*
   per-bet Sharpe it is ≈ **40**, which no free universe supplies.

Twenty-three studies attacked the signal and failed. The twenty-fourth attacked breadth
and failed. **The ceiling is where iteration 11 measured it, and the free-data universe
does not contain the bets required to lift it.** That is a complete answer.

---

## 11. Files and reproduction

```
.venv/Scripts/python.exe -m research.multiasset.breadth_build      # panel + integrity
.venv/Scripts/python.exe -m research.sleeves.breadth_neff          # N_eff (control first)
.venv/Scripts/python.exe -m research.sleeves.breadth_ladder        # ladder (control first)
```

| path | contents |
|---|---|
| `research/multiasset/breadth_instruments.py` | registry; reuses the original `Instrument` dataclass |
| `research/multiasset/breadth_build.py` | fetch, clean, guards, roll tests, panels |
| `research/sleeves/breadth_universe.py` | combined excess panel, identical conventions |
| `research/sleeves/breadth_neff.py` | the N_eff study, control-gated |
| `research/sleeves/breadth_ladder.py` | the leverage ladder, control-gated |
| `research/sleeves/_breadth/neff.json` | every N_eff number in this file |
| `research/sleeves/_breadth/ladder.json` | every ladder number in this file |
| `_data/multiasset/breadth/` | panels, coverage, integrity.json — **gitignored** |

Nothing in `research/multiasset/{instruments,panel}.py`, `scripts/build_multiasset_panel.py`,
`research/sleeves/riskparity*.py`, `spread_estimation.py` or `capacity_study.py` was
modified. No raw vendor rows are committed.
