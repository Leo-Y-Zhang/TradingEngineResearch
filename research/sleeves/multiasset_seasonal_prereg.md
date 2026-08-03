# PRE-REGISTRATION — CALENDAR SEASONALITY on the long-history multi-asset panel

**Written 2026-07-28, BEFORE any strategy code was written and before any backtest was
run.** Nothing in this file may be changed after the run. The run happens ONCE. If the
result is bad it is banked as bad. No re-tuning of windows, universes, weighting schemes,
thresholds or cost assumptions afterwards.

**The single largest risk in this study is not overfitting a parameter — it is searching the
calendar.** With 12 months x 18 instruments x two directions there is guaranteed to be a
window that "worked". Therefore every effect tested below is named, dated and cited from the
literature in this file, before the data was touched, and **no effect discovered in this
panel may be reported as a finding.** The count of calendar hypotheses is fixed at **3** and
is stated in the result, together with a deflation for the much larger space the literature
itself searched to produce them.

---

## 0. Why seasonality, and the honest version of the reason it was chosen

The brief's stated reason: seasonality's correlation to trend and carry is near zero **by
construction** rather than by hope, because a signal that depends only on the DATE cannot
share an estimation window with a momentum or a yield signal. That is precisely the failure
that made the cross-asset value sleeve's diversification illusory (removing the 12 months in
which the 5-year reversal window contained the 12-month momentum window took rho from -0.164
to -0.013).

**That reasoning is correct about the SIGNAL and wrong about the RETURNS, and this is
recorded here in advance so it cannot be presented as a discovery afterwards.** A
long-the-favourable-window / flat-otherwise seasonal sleeve is **net long the market
whenever it is on**. The multi-asset trend sleeve is also net long a majority of the time.
Two net-long books share market beta regardless of how orthogonal their signals are.
So:

> **Pre-registered expectation: rho(seasonal, trend) will be materially POSITIVE, in the
> region of +0.15 to +0.45, NOT zero — and the "uncorrelated by construction" argument
> applies to the signal, not to the P&L.**

If the measured rho comes in near zero, that is a genuine result. If it comes in at +0.3,
the sleeve's diversification value is much smaller than the brief hopes, and the honest
conclusion is that orthogonal signals do not imply orthogonal returns when both books carry
directional exposure. Either way the number is reported.

The long-short variant (S2 below) is declared for exactly this reason: it is closer to
dollar-neutral over a year and should show the lower correlation. It is a **secondary** and
may not be promoted to headline whatever it says.

**Success is a Sharpe number.** Half-Kelly growth = 3S^2/8. **30%/yr <=> S = 0.894.**

---

## 1. The three effects — pre-specified, cited, and NOT chosen from this data

| # | effect | definition | first published | citation |
|---|---|---|---|---|
| **E1** | **Turn-of-the-month** | long on the **last business day of the calendar month and the first three business days of the next**; flat on all other days | **1987** | Ariel, "A monthly effect in stock returns", *JFE* 18 (1987); Lakonishok & Smidt, *RFS* 1 (1988) |
| **E2** | **Halloween / Sell-in-May** | long in **November-April**; flat in **May-October** | **2002** | Bouman & Jacobsen, "The Halloween Indicator, Sell in May and Go Away", *AER* 92 (2002) |
| **E3** | **January effect (equity indices only)** | long the **equity block in January**; flat otherwise | **1976** | Rozeff & Kinney, *JFE* 3 (1976); Keim, *JFE* 12 (1983) |

Windows are fixed at the values in the original papers. **No neighbouring window is tested.**
No month is added to or removed from any window. No instrument subset is searched for a
stronger version. E3 is restricted to the equity block because that is where the paper
documents it; that restriction is a limitation stated in advance, not a selection.

**E4 (the sleeve headline) = an equal-risk composite of E1, E2, E3.** Composite weights are
fixed at equal risk (1/3 of book risk each after each leg is scaled to a common unit vol);
they are not optimised, and no alternative weighting is tested.

### 1a. Effects deliberately NOT tested, listed so the absence is auditable

Monday/weekend effect, holiday effect, day-of-week effects, the "Santa Claus rally", the
FOMC-cycle effect, the intra-month semi-month effect (Ariel's other result), the September
effect, quarter-end and index-rebalance effects, tax-loss-selling windows other than E3.
Each is a documented calendar anomaly and any one of them could have been added after seeing
E1-E3 fail. **They are excluded here so that they cannot be.**

---

## 2. Universe and data

**Universe = the multi-asset trend sleeve's `PRIMARY_UNIVERSE`, unchanged, 18 instruments:**

| block | instruments |
|---|---|
| equity (7) | SPX, NASDAQ, FTSE100, N225, DAX, HSI, ASX200 |
| rates (3) | US5Y_TR, US10Y_TR, US30Y_TR |
| commodity (4) | GOLD_F, WTI_F, SILVER_F, COPPER_F |
| fx (4) | USDX, EURUSD, GBPUSD, JPYUSD |

Reused **exactly** rather than re-chosen, for two reasons: it is already integrity-vetted
(NATGAS_F excluded as a roll-splice artefact; SPY/GLD/IEF/TLT excluded as ETF duplicates;
DJIA excluded as an SPX duplicate), and holding the universe fixed is what makes
rho(seasonal, trend) attributable to the SIGNAL rather than to a universe difference.

**Data.** `_data/multiasset/returns_daily.parquet` (business-day grid, 25,305 rows, no
weekend rows), `_data/multiasset/returns_monthly.parquet`, `_data/multiasset/cash_daily.parquet`,
`_data/multiasset/cash_monthly.parquet`. All gitignored; **no row of this data is committed,
quoted or pasted anywhere.** Quarantined (screened) panel is the primary; the unscreened
panel is secondary S4.

**Excess returns.** `US5Y_TR`, `US10Y_TR`, `US30Y_TR` are USD total returns and have the
13-week bill accrual subtracted (daily accrual for the daily grid, monthly for the monthly
grid). Every other series is a price / futures / spot return and is already an excess
return. Identical to the trend sleeve's `CASH_SUBTRACTED` convention.

### 2a. Why the DAILY panel may be used here, when the integrity report says not to

`research/multiasset/data_integrity.md` §10.1 says to use the month-end panel for
cross-asset work because the daily panel has a ~1-hour futures/equity session overlap that
is a **real lookahead at daily frequency**. That warning is about signals that read a
same-day price. **A calendar signal reads no price at all** — the entire position schedule
for 2026 was knowable in 1965 — so the overlap cannot leak information into the E1 signal.

What the overlap *does* still do is misattribute part of one instrument's session to the
neighbouring date label. Consequences, disclosed in advance:

1. It can shift a fraction of a day's return across a TOM window boundary in either
   direction. This is **noise, not bias**, because the calendar boundary is fixed and
   independent of the returns.
2. Cross-instrument daily correlations are attenuated. **No daily cross-sectional
   correlation is used anywhere in this study**: instrument volatilities come from the
   MONTHLY panel (§4), and every correlation reported is between monthly series.

### 2b. Point-in-time definition of "business day" — the one place a calendar can cheat

The TOM window is defined on the **pure Monday-Friday `pd.bdate_range` grid**, which
requires no holiday knowledge and no data knowledge whatsoever, and is therefore
unambiguously ex ante. It is **not** defined on the instruments' observed trading days,
because an observed-trading-day calendar is contaminated by data gaps: "the last day on
which this series happens to have a print" is partly a fact about the vendor, not the
exchange.

The cost of that choice is stated in advance: when the last Mon-Fri day of a month, or one
of the first three of the next, is an exchange holiday, no return accrues that day and the
sleeve is simply flat. This **attenuates** the measured effect. It cannot manufacture one.
The observed-trading-day variant is declared as secondary **S3** and reported unconditionally.

---

## 3. The signal — date-only indicators, no free parameters

For instrument `i` and business day `d`, `sig_i(d) in {0, 1}` (primary) is a function of the
calendar date alone:

```
E1  sig_i(d) = 1  if d is the last bday of its month, or one of the first 3 bdays of its month
E2  sig_i(d) = 1  if month(d) in {Nov, Dec, Jan, Feb, Mar, Apr}
E3  sig_i(d) = 1  if month(d) == Jan and i is in the equity block
```

No instrument, no return, no volatility and no price enters any of these three lines.

---

## 4. Position construction — the trend sleeve's machinery, unchanged, so only the signal differs

Every constant below is copied from `research/sleeves/multiasset_trend.py`. **Nothing is
re-tuned.** This is deliberate: if the sizing machinery is identical and only the signal
differs, the difference in outcome is attributable to the signal.

1. **Instrument volatility.** `sigma_i,t` = trailing **36-month** std of monthly excess
   returns through month-end `t`, annualised (x sqrt 12), min **24** observations.
2. **Eligibility.** >= **36** monthly observations of history, finite `sigma > 0`. Book is
   off in any month with fewer than **3** eligible instruments.
3. **Sizing.** `n_i(d) = sig_i(d) * (0.10 / sigma_i,t)` where `t` is the month-end
   **strictly before** the month containing `d`. Sigma is therefore lagged one full month,
   exactly as in the trend sleeve.
4. **Book vol scaler.** `k_t = min(vol_target / sigmahat_book,t , 10 / gross_unit_t)`, where
   `sigmahat_book,t` is the trailing **36-month** std of the unscaled book's MONTHLY returns
   through `t` (min 12), annualised. `k` is applied to the following month. Gross cap 10x
   book equity, and the number of months in which the cap binds is reported.
5. **Rebalance.** Weights change on the daily grid only when `sig` changes or at each
   month-end when `sigma` and `k` refresh. Turnover is measured on the daily grid as
   `sum_i |w_i(d) - w_i(d-1)|`.
6. **Vol targets reported: 10%, 20%, 40%.** Sharpe is invariant to the target up to the
   gross cap and to cost drag; both effects are visible in the table, which is why all three
   are reported rather than one.

---

## 5. Costs — reported as a bracket, never as a single number

Charged on notional traded, one-way = half the round-trip:
`cost(d) = 0.5 * c_roundtrip * turnover(d)`.

| bound | round trip | rationale |
|---|---|---|
| realistic | **2 bps** | index futures / FX forwards / liquid commodity futures trade at 1-5 bps round trip |
| conservative | **10 bps** | 5x the realistic bound, covering slippage, roll and a small book |

Costs matter **disproportionately** here: E1 does a full round trip **every month** in every
instrument, ~12 round trips/yr/instrument against trend's 4-12 total bets/yr. Therefore
additionally, and pre-registered as a headline statistic:

> **Breakeven round-trip cost** `c* = 2 * mean(gross_d) / mean(turnover_d)`, in bps,
> reported for every effect at every vol target. If `c*` is below 2 bps the sleeve is
> **dead on costs alone** and no other statistic can save it.

---

## 6. Benchmark — levered to the strategy's own volatility, and put through the same gate

**Benchmark = equal-weight LONG-ONLY ownership of the same eligible instruments, held every
day**, rebalanced monthly, then **levered by a constant factor so that its full-sample
realised volatility equals the strategy's own.** That is the comparison the brief mandates
and it is the correct one: a strategy that is in the market 19% of the time has ~44% of the
volatility of one that is always in it, so a raw geometric comparison flatters it exactly the
way it flattered PEAD.

Three active statistics, all with Newey-West (6 lag) t-statistics, **all three reported
together, never one alone**:

- **A. Geometric excess** = `geo(strategy) - geo(benchmark)`, annualised.
- **B. Arithmetic active** = `12 * mean(strategy - benchmark)` monthly, with t-stat.
- **C. Vol-matched active** = `12 * mean(strategy - benchmark * sd_s/sd_b)`, with t-stat.

The measured fact from the carry study is recorded here so it is not rediscovered: the
**raw arithmetic active t-stat is a leverage dial** (it moved from +0.88 to +1.92 purely by
levering carry from 4% to 40% vol) while the **vol-matched active t-stat is invariant**.
**Statistic C is the one that decides.** The variance-drag identity
`geo excess = arith active - (var_s - var_b)/2` is reported so the size of the illusion is
visible.

**The benchmark is run through the DSR gate too**, at the same sample length and trial count.
A strategy that clears DSR while its own levered universe also clears DSR has not
demonstrated anything.

---

## 7. Everything that will be measured — declared in advance, reported unconditionally

1. **Sharpe per decade** for every effect and the composite (1960s ... 2020s).
2. **Pre-publication vs post-publication Sharpe**, split at each effect's OWN publication
   year: **E1 at 1987, E2 at 2002, E3 at 1976.** Both eras' Sharpe, mean return, months, and
   the difference. **This is the single most informative test in the study** and its result
   is reported whatever it says. A seasonal effect that survived being written about is
   worth far more than one that did not.
3. **Correlation of monthly net returns to the real trend sleeve**
   (`research/sleeves/_multiasset_trend/primary_20pct_monthly.csv`, `net_10bps`, 738 months
   from 1965-01) **and to the real carry sleeve**
   (`research/sleeves/_carry_output/carry_primary_net_monthly.parquet`, 269 months from
   2004-02), each on its own overlap, with the overlap length stated.
4. **Combined portfolio Sharpe** three ways: (a) the brief's formula
   `S = s * sqrt(N/(1+(N-1)rho))` with **measured** rho and N=3; (b) the directly measured
   equal-risk trend+carry+seasonal combination; (c) trend+carry alone (0.655) as the
   incumbent to beat. **(b) is the number that counts**; (a) is reported as the check,
   because it assumes equal sleeve Sharpes which is known to be false here.
5. **Half-Kelly reachable compound return** `3S^2/8` at every headline Sharpe.
6. **DSR bar** at the realised sample length via `research.multiasset.panel.dsr_sharpe_bar`,
   reported at **four** trial counts (§8).
7. **P&L concentration**: largest single (instrument, month) share of net P&L and of gross
   absolute P&L. Alarm at 3%.
8. **Per-instrument and per-block P&L attribution.**
9. **Turnover, gross leverage, cap-binding months, and days-in-market fraction.**
10. **Max drawdown, monthly skew, worst month.**
11. **Negative control**: the TOM window shifted to a fixed, pre-declared placebo — the
    **10th, 11th, 12th and 13th business days of the month** (the mid-month interior, which
    no cited paper claims anything about). One placebo, declared now, not a search. The live
    E1 Sharpe must stand clear of it.
12. **Days-in-market**: fraction of business days with a non-zero book, per effect. A
    "Sharpe" earned on 19% of days is a different object from one earned on 100% and the
    ratio is reported so the reader can see it.

---

## 8. Trial accounting and the calendar-space deflation

This study spends **3 trials** (E1, E2, E3). The composite E4 is a fixed equal-risk
combination of the three with no free parameter and is **not** counted as a fourth.
Secondaries S1-S4 and the negative control are non-promotable by this pre-registration and
are not counted.

The programme's honest cumulative count entering this study is **40** (32 anchor convention
-> 36 after carry -> 40 after value). Other sleeves are running concurrently, so:

| n_trials | meaning |
|---:|---|
| 32 | the recorded-anchor convention, for comparability with every prior study |
| 44 | honest cumulative: 40 + this study's 3, + 1 for concurrent work |
| 56 | pessimistic: allows for a full concurrent study having landed unseen |
| **304** | **calendar-space deflation** (§8a) |

**§8a. The calendar-space deflation, and why it is the honest bar.** I tested 3 hypotheses.
The literature that produced those 3 searched a much larger space, and inheriting a published
window is inheriting its selection bias. The space is enumerable:

- contiguous month-of-year windows: 12 starts x 11 lengths = **132**, x 2 directions = **264**
- contiguous day-of-month windows on a ~21-day month, capped at length 5:
  21 starts x 5 lengths = **105**, but the papers only ever proposed windows at the month
  boundary, so the honest count is the ~**21** single-day-anchored windows, x 2 directions = **42**

`264 - 4` (the two directions of E2's and E3's own windows, already counted) `+ 42 - 2`
(E1's) `= 300`, plus this study's 4 = **304**. The DSR bar at n_trials = 304 is reported for
every effect. **A result that clears only the n_trials=44 bar and not the n_trials=304 bar is
reported as "clears the honest own-search bar, fails the inherited-search bar", never as a
pass.**

---

## 9. Secondaries — declared here, reported unconditionally, never promoted

| # | secondary | why it exists |
|---|---|---|
| **S1** | **Equity-block-only** version of E1 and E2 | the papers document these on equities; the cross-asset extension is mine, and if it fails while the equity version works that is informative |
| **S2** | **Long-short**: long the favourable window, **short** the unfavourable one, same sizing | tests §0's prediction that the long-flat form's correlation to trend is beta, not signal overlap |
| **S3** | **Observed-trading-day calendar** for E1 instead of the Mon-Fri grid | measures the attenuation admitted in §2b |
| **S4** | **Unscreened panel** sensitivity | the quarantine decision |

None of S1-S4 may become the headline. The headline is E4 (composite) with E1/E2/E3 beside it.

---

## 10. Falsifiable predictions, recorded before the run

| # | prediction |
|---|---|
| **P1** | **E4 composite gross Sharpe lands in [0.30, 0.65]. Point prediction: 0.42.** |
| **P2** | E1 gross Sharpe **[0.30, 0.80]** full sample; E2 **[0.25, 0.60]**; E3 **[0.00, 0.35]** (the January effect is a small-cap phenomenon and should be weak-to-absent on capitalisation-weighted indices). |
| **P3** | **Every effect's post-publication Sharpe is LOWER than its pre-publication Sharpe.** This is the prediction I most expect to be right and it is the reason the split is pre-registered. |
| **P4** | **rho(seasonal, trend) is POSITIVE, +0.15 to +0.45** — beta, not signal overlap (§0). rho(seasonal, carry) is smaller in magnitude, \|rho\| <= 0.25. |
| **P5** | **Vol-matched active return (statistic C) vs the levered long-only benchmark is NOT significant at t >= 2.0 for any effect.** Being long a third of the year is a market-timing bet and I do not expect it to beat owning the market at matched risk. |
| **P6** | Breakeven round-trip cost for **E1 is under 25 bps** and for E2/E3 **over 100 bps** — E1 trades 12x more often, so costs bind on it and essentially not on the others. |
| **P7** | **The seasonal sleeve alone does NOT reach Sharpe 0.894.** |
| **P8** | **Adding seasonal to trend+carry moves the 3-sleeve Sharpe by less than +0.10 above 0.655**, because of P4: a positively-correlated sleeve does not buy what an uncorrelated one would. |

---

## 11. Verdict rule, fixed in advance

Let `S_net` = composite (E4) net Sharpe at the **realistic** 2 bps bound, `S_cons` at the
conservative 10 bps bound, `bar_44` and `bar_304` the DSR bars at the realised sample length.

- **PROMISING** — `S_net >= bar_44` AND `S_cons > 0` AND **statistic C (vol-matched active)
  t-stat >= 2.0** AND no calendar decade with a negative Sharpe AND the post-publication
  Sharpe is at least half the pre-publication Sharpe AND max (instrument, month) net-P&L
  share < 3% AND the live E1 Sharpe exceeds the placebo control.
- **MARGINAL** — `S_net >= 0.35` and statistic C t-stat >= 1.5, but at least one PROMISING
  condition fails.
- **DEAD** — otherwise.

Separately and regardless of tier, reported as a plain yes/no: does the sleeve, or the
three-sleeve combination, reach **S = 0.894** (30%/yr at half Kelly)?

**A clean DEAD stated honestly is the expected and acceptable outcome. One run. No tuning.
No second look.**
