# PRE-REGISTRATION — Cross-asset VALUE on the long-history multi-asset panel

**Written 2026-07-28, BEFORE any backtest of this sleeve was run.** Everything below was
fixed in advance and is executed exactly once. Nothing here was chosen by looking at a
strategy return. The only things inspected before writing this file were the **coverage
receipts** (`research/multiasset/data_integrity.md`, first/last dates, observation counts,
yield availability) — never a return of any candidate construction.

Data: `_data/multiasset/returns_monthly.parquet`, `yields_monthly.parquet`,
`cash_monthly.parquet`, built and audited in `research/multiasset/data_integrity.md`.
Code: `research/sleeves/multiasset_value.py`. Output: `research/sleeves/_value/`.

---

## 0. Why this sleeve exists — and why its Sharpe is not the point

The portfolio arithmetic is the whole game:

```
S_portfolio = s * sqrt( N / (1 + (N-1) * rho) )        max compound growth = 3*S^2/8 (half Kelly)
```

30%/yr compound needs `S ~ 0.894`. No single sleeve in this programme has produced that, and
twelve US-equity cross-sectional studies have produced nothing at all. The remaining route is
**combination**, and combination is governed by `rho`, not by any individual `s`.

This sleeve was chosen for one documented reason: **cross-asset value is negatively
correlated to cross-asset momentum** (Asness, Moskowitz & Pedersen, *Value and Momentum
Everywhere*, Journal of Finance 68(3), 2013 — hereafter AMP). A sleeve at `s = 0.35` with
`rho = -0.3` to an existing sleeve is worth more than one at `s = 0.55` with `rho = +0.6`.
**Therefore the headline deliverable of this study is the correlation to the trend sleeve,
and the standalone Sharpe is secondary.** That ordering is fixed now so it cannot be
re-ordered after the numbers arrive.

**The honest failure mode to watch for, stated in advance:** cross-asset value has performed
badly since roughly 2010, and a 5-year-reversal proxy is a much weaker instrument than the
book-to-market data AMP had for equities. If the sleeve works only pre-2010 it is not
deployable, and the per-decade table is what will say so.

**The second honest failure mode:** the value signal below is `-(trailing 5-year return)`,
which *contains* the 12-month window the trend sleeve trades. Part of any negative
correlation to trend will therefore be **mechanical, not economic**. This is disclosed now,
and DIAGNOSTIC D3 (§8) is pre-registered specifically to size it.

---

## 1. Universe — 14 instruments in 3 blocks, and the block that is honestly excluded

Selection rule, applied before any backtest: one instrument per distinct underlying risk,
prefer the longest history, exclude series with a known structural defect, exclude ETF
duplicates of a series already present. This is the same rule the trend sleeve used, and it
is applied here to the same receipt.

| block | instruments | n |
|---|---|---:|
| Equity index | SPX, NASDAQ, FTSE100, N225, DAX, HSI, ASX200 | 7 |
| Rates (par-bond TR) | US5Y_TR, US10Y_TR, US30Y_TR | 3 |
| Commodity futures | GOLD_F, WTI_F, SILVER_F, COPPER_F | 4 |

### FX is EXCLUDED, and this is a data limitation, not a modelling choice

The value measure for currencies is the **deviation from long-run PPP** — the real exchange
rate, i.e. the nominal rate deflated by the ratio of the two countries' price levels (AMP §I.B
constructs it from 5-year changes in spot rates *adjusted for inflation differentials*).

**This panel contains no price-level series for any country.** There is no CPI, no GDP
deflator, no inflation forecast, and no way to derive one from `EURUSD=X`, `GBPUSD=X`,
`JPY=X` or `DX-Y.NYB`, which are nominal spot quotes and nothing else. A nominal 5-year
change in spot is **not** a PPP deviation; using it and calling it FX value would be
substituting a different signal for the one that has the evidence behind it.

So the four FX instruments are excluded and **no FX value claim is made**. The cost is
stated plainly: the sleeve loses 4 of 18 instruments and one entire asset class of
diversification, which lowers its breadth and raises its correlation to the equity block.
That is a worse sleeve than AMP's, and it is the honest one available from free data.

### The other exclusions, with reasons fixed in advance

| excluded | reason |
|---|---|
| `USDX`, `EURUSD`, `GBPUSD`, `JPYUSD` | PPP not constructible — see above. |
| `NATGAS_F` | Roll-contaminated: 65.7% of its \|r\|>15% bars fall in days 24–31 vs a 24.0% base rate (data_integrity §6a). Not a price series, so a 5-year price reversal on it is not a valuation. |
| `DJIA` | Redundant with SPX and starts only 1992. |
| `SPY`, `TLT`, `IEF`, `GLD` | ETF duplicates of SPX / US30Y_TR / US10Y_TR / GOLD_F, each with strictly shorter history. |
| `DBC`, `EFA`, `EEM` | Baskets that overlap instruments already included; all ETF-era only, and ETFs are the sole block carrying real survivorship bias (data_integrity §7). |
| `BIL`, `IEI`, `SLV` | Validation-only by construction. |
| `US_CASH_13W` | The risk-free leg, not a risk asset. |

### Eligibility, and the sample it implies

Instrument `i` is **eligible** at month-end `t` when its value score and its volatility
estimate both exist at `t` (§3, §4) — which requires **60 non-null monthly returns** for
equities and commodities, and **60 monthly observations of its term spread** for rates.

A **block is live** when **>= 3 of its instruments are eligible**. Three is the minimum at
which a "cross-section" has a top and a bottom that are not the same pair; a two-name block
is a single pair trade wearing a cross-sectional label.

The **book trades only when >= 2 blocks are live**, because a one-block book is not
cross-asset value.

Expected timeline, read off the coverage receipt (dates of the 60th observation), **not off
any return**:

| block | 3rd instrument eligible |
|---|---|
| Equity | 1976-01 (SPX 1932-12, N225 1969-12, NASDAQ 1976-01) |
| Rates | 1982-01 (US5Y and US10Y 1966-12, US30Y 1982-01) |
| Commodity | 2005-08 (WTI 2005-07, GOLD/SILVER/COPPER 2005-08) |

So the book is expected to start **1982-01** and run to 2026-06 — **~44.5 years**, with two
blocks (equity + rates) until 2005-08 and three thereafter. Sample length is therefore
roughly *half* the panel's 98.6-year headline, and that is a direct cost of demanding a
60-month lookback on 14 instruments. It is not hidden: every reported statistic is on 44.5
years, and the DSR bar is computed at that length.

**Measured before the run and recorded here: the value universe contains 0 interior null
cells.** The trend sleeve's interior-null amendment therefore has nothing to act on in this
sleeve, and the rule is carried over unchanged only so the two are identical in convention:
an interior null is a zero return with no position held.

---

## 2. Return convention — identical to the trend sleeve, deliberately

The strategy is a fully-collateralised futures book, so every reported stream is an **excess
return over cash**.

| series | treatment | justification |
|---|---|---|
| 7 equity price indices, 4 commodity futures | **used raw** | A futures price return already embeds financing; cash is the collateral, not the position. |
| US5Y_TR, US10Y_TR, US30Y_TR | **minus `US_CASH_13W`** | These are par-bond **total** returns (cash + excess); subtracting the bill rate recovers the futures-equivalent excess return. |

This is byte-for-byte the trend sleeve's convention (`research/sleeves/multiasset_trend.py::load_excess_panel`),
and that is a requirement rather than a convenience: **a correlation between two sleeves
computed on two different return conventions is not a correlation between two sleeves.**

Disclosed conventions this does not fix (all in data_integrity §3, §4, §6), carried over:

- Price indices exclude dividends (SPX vs SPY: measured **1.95%/yr**; UK/AU higher). For a
  futures excess return the correct object is `price return + dividend yield - local
  risk-free`, so the raw price return carries an error of `(local rf - dividend yield)`.
  **In a long/short book this bias is largely common-mode and cancels**, far more completely
  than it does in the long-only trend book; the residual is scaled by net exposure, and mean
  net exposure is reported so it can be sized.
- **DAX is a total-return index while the other six equity indices are price-only.** Not
  corrected (no EUR cash series). This is a *level* difference of ~2–3%/yr, and the value
  signal is a 5-year **return** — so DAX will look systematically "expensive" versus its
  peers by roughly 10–15% of cumulative 5-year return. **This is a known signal
  contamination on 1 of 7 equity names**, disclosed now, and the equity-block sub-book (D2)
  plus P&L concentration will show whether it matters.
- Local currency is not converted to USD; second-order at monthly frequency for a futures
  position.
- Bond series omit roll-down, understating bond total return by ~0.5%/yr at 5y and 10y. This
  biases *against* the rates block's long leg and *for* its short leg, and is common-mode
  across the three bonds so it largely cancels in a within-block long/short.
- **Monthly panel only.** The daily panel has a ~1-hour futures/equity session overlap that
  is a genuine lookahead at daily frequency (data_integrity §6b).

---

## 3. The value score — stated per block, with the units problem solved by ranking

Signs are fixed so that **high score = CHEAP = long**.

### 3a. Equity indices and commodities — 5-year reversal

```
v(i,t) = - sum_{k = t-59}^{t} log(1 + x(i,k))
```

the negative of the trailing 60-month cumulative **log** excess return. Requires 60 non-null
monthly returns ending at `t`. Log, not simple, because the object AMP uses is
`log(P_{t-5y} / P_t)` and because compounding 60 simple returns makes the score's scale
depend on its own sign.

This is the standard proxy where fundamental data is absent, and it is exactly what AMP use
for commodities (*"the log of the spot price 5 years ago divided by the current spot
price"*). For equity **indices** AMP use index book-to-market; **this panel has no book
value for any index**, so the reversal proxy is substituted, and that substitution is a
genuine weakening of the instrument, recorded here rather than glossed.

**No skip.** The window ends at `t`, so it overlaps the trend sleeve's 1/3/6/12-month
signals. AMP make the same choice. The consequence — part of the value/momentum negative
correlation is mechanical — is measured by D3, not asserted away.

### 3b. Rates — term spread versus its own long-run average

Real yield is not constructible: **the panel contains no inflation series**, so
`nominal yield - expected inflation` cannot be formed. The brief's alternative is used:

```
spread(i,t) = y_i(t) - y_13w(t)                       (yields panel, decimal)
v(i,t)      = spread(i,t) - mean( spread(i,s) : s <= t )
```

an **expanding-window** mean (min 60 observations), so it is causal and uses no future
information. High = the curve is steeper at this maturity than its own history = the bond is
cheap = long.

**Disclosed overlap with carry.** The *level* of the term spread is the classic bond **carry**
signal, and sleeve 2 is a carry sleeve. What makes this a *value* signal is the subtraction of
the instrument's own long-run mean: it trades the **deviation**, not the level. The two are
not the same object, but they are not orthogonal either, and the correlation to the carry
sleeve will be reported if that sleeve's return series is on disk when this one finishes.

### 3c. The units problem, and why ranking solves it

A 5-year log return and a term-spread deviation in decimal are not comparable numbers, and
pooling them into one cross-section would be meaningless arithmetic. AMP's own construction
avoids this by ranking **within** asset class and combining the resulting portfolios. The
same is done here (§4), so **no cross-block comparison of raw scores ever occurs**.

---

## 4. Sizing — rank-weighted within block, inverse-vol, equal risk across blocks

**Step 1 — cross-sectional rank spread within each live block.** For block `b` with `N_b`
eligible instruments at `t`, rank ascending on `v` (average ranks on ties):

```
d(i,t) = rank(i,t) - (N_b + 1)/2
u(i,t) = d(i,t) / sum_j |d(j,t)|            =>   sum_i u = 0,  sum_i |u| = 1
```

Rank weighting (rather than a top/bottom-tercile dummy) is AMP's own weighting scheme and is
fixed for that reason. It is long the cheap and short the expensive, dollar-neutral **within
block** by construction.

**Step 2 — inverse volatility.** `sigma(i,t)` = annualised standard deviation of instrument
`i`'s monthly excess returns over the **trailing 36 months ending at `t`** (min 24 obs):

```
n_b(i,t) = u(i,t) * ( 0.10 / sigma(i,t) )
```

0.10 is an arbitrary per-instrument vol unit; it cancels in step 4 and is not a free
parameter. Identical to the trend sleeve.

**Step 3 — equal risk across live blocks.** `n(i,t) = n_b(i,t) / (number of live blocks at t)`.
Each live block therefore contributes the same gross unit notional. This is a *choice*: it is
made because the blocks are 7/3/4 instruments and a flat pool would hand 54% of the book to
equities. It is fixed now and there is no flat-pool alternative run as a headline.

**Step 4 — scale the book to a target volatility.** Raw book excess return
`b(t+1) = sum_i n(i,t) * x(i,t+1)`. Book vol `sigmaHat_b(t)` = annualised stdev of `b` over
the trailing 36 months ending at `t` (min 12):

```
r_strat(t+1) = k(t) * b(t+1),     k(t) = sigma_target / sigmaHat_b(t)
```

`sigmaHat_b(t)` uses only book returns realised at or before `t`, so `k(t)` is causal.

**Volatility targets reported: 10%, 20%, 40%.** With strictly proportional costs, gross
return, cost and volatility all scale linearly in `k`, so **net Sharpe is invariant to the
vol target** except through the leverage cap. The three targets are **one trial, not three**;
they exist to translate a Sharpe into a compound return.

**Leverage cap:** gross notional `sum_i |k(t) n(i,t)|` capped at **10x book equity**, `k`
scaled down when it binds. Cap-binding frequency is reported at every target — it is the only
channel through which the three targets can differ.

---

## 5. Costs

```
turnover(t) = sum_i | w(i,t) - w(i,t-1) |          (w = k(t) * n(i,t), fraction of equity)
cost(t)     = 0.5 * c_roundtrip * turnover(t)
```

The 0.5 is because `sum |delta w|` counts a round trip as **two** units of notional change
while `c_roundtrip` prices it once.

| bracket | value | rationale |
|---|---:|---|
| (a) optimistic | **2bps** round trip | E-mini S&P / major index and metals futures: commission ~0.2bp plus half-spread ~0.5bp each way. Liquid futures genuinely trade here at research size. |
| (b) conservative | **10bps** round trip | Covers the illiquid tail of this universe (copper, ASX/HSI futures, the 30y), plus slippage and modest impact. Broad-ETF execution is 1–5bps, so 10bps is deliberately above the honest range. |

A result that passes at (a) and fails at (b) is **UNDETERMINED**, not a pass. Both are always
reported side by side.

**The benchmark pays the same schedule** on its own rebalancing turnover. A costless
benchmark against a costed strategy is a rigged comparison.

**Note in advance:** a long/short rank-weighted book turns over more than a long-only trend
book, because a rank can move without a sign changing. Turnover per year is reported, and if
the 2bps/10bps gap is large that is a finding, not a footnote.

---

## 6. Benchmark and the active-return test

**Benchmark:** equal-weight **long-only** holding of exactly the instruments eligible at `t`,
on the identical excess-return convention, rebalanced monthly:
`r_bench(t+1) = (1/N_t) * sum_i x(i,t+1)`.

The brief says "buy-and-hold". On an **unbalanced** panel — instruments enter as they become
eligible and the set grows from 6 to 14 — a literal buy-and-hold is undefined, so
equal-weight-over-the-eligible-set rebalanced monthly is used, **and it pays cost on its own
turnover**. That turnover is small (it changes only when an instrument enters), so the two
readings differ by a few basis points a year; the number is reported.

Three active measures, all reported:

1. **PRIMARY — arithmetic active return.** `a(t) = r_strat(t) - r_bench(t)`, reported as
   `12 * mean(a)` with a **Newey–West t-statistic (lag 6)**. This is the headline.
2. **Jensen alpha.** OLS `r_strat = alpha + beta * r_bench + e`; annualised alpha with a
   Newey–West t-stat.
3. **Vol-matched active return.** Strategy scaled by the full-sample constant
   `sigma_bench / sigma_strat`, then differenced. Disclosed: that scalar uses full-sample
   information; it is a positive constant and therefore cannot create alpha or flip a sign.

**The variance-drag identity is computed and printed explicitly:**

```
geometric excess = arithmetic active - (var_strat - var_bench)/2
```

A market-neutral book has a *much* lower variance than a long-only benchmark, so the second
term is large and positive here and will manufacture a **fake** positive geometric excess.
This is the exact illusion that killed the PEAD sleeve. **The headline is the arithmetic
number with its t-stat. A geometric excess unaccompanied by an arithmetic one is reported as
variance drag and nothing else.**

---

## 7. What will be reported — every item, pass or fail

- Gross and net Sharpe at **both** cost brackets and **all three** vol targets.
- **Arithmetic active return with Newey–West t-stat** (primary), Jensen alpha, vol-matched
  active, and the variance-drag decomposition.
- **Sharpe per decade** (1980s partial, 90s, 2000s, 2010s, 2020s). A full-sample number
  carried by one era is declared **not deployable** in advance.
- **DSR bar** via `research.multiasset.panel.dsr_sharpe_bar` at the realised sample length,
  at `n_trials=32` (the recorded anchor convention) **and** at the honest cumulative count
  including this study, with an explicit pass/fail on the **net** Sharpe.
- **Half-Kelly reachable return `g = 3S^2/8`** on the net Sharpe, and the implied book
  volatility `sigma* = S/2`.
- Max drawdown, worst month, skew, kurtosis, turnover/yr, gross leverage, cap-binding
  frequency, mean net exposure.
- **P&L concentration**: share of total P&L from the single largest (instrument, month), from
  the largest instrument, and from the largest calendar year. One name-month was once 13% of
  a study's P&L.
- **Breadth**: mean eligible instruments and **effective N** from the eigenvalues of the
  instrument correlation matrix.
- **CORRELATION TO THE TREND SLEEVE** (`research/sleeves/_multiasset_trend/primary_20pct_monthly.csv`)
  on the overlapping months, net of 10bps on both sides, plus the correlation to the carry
  sleeve if its series is on disk. **Reported first in the result document.**
- **The combined two-sleeve Sharpe** at equal risk weight, `S_c = (S1 + S2)/sqrt(2 + 2*rho)`,
  and at the optimal weight, `S_c* = sqrt((S1^2 + S2^2 - 2*rho*S1*S2)/(1 - rho^2))`, with the
  half-Kelly compound return each implies. This is the number the programme actually needs.

---

## 8. Diagnostics declared in advance — none of them can become the headline

Each is reported whatever it says. **This study makes exactly ONE strategy claim (PRIMARY).**
No diagnostic below may be promoted to a headline, and none is counted as a trial.

- **D1 — Negative control.** The sign of `u(i,t)` randomised per (instrument, month) at 8
  fixed seeds. Expected net Sharpe ~0. If PRIMARY is not several standard deviations outside
  this distribution, it is machinery, not edge.
- **D2 — Per-block sub-books.** Equity-only, rates-only, commodity-only at 20% vol, to show
  whether one block carries everything.
- **D3 — Skip-12m reversal.** `v = -(cumulative log excess return from t-59 to t-12)` for
  equities and commodities, rates unchanged. **This is the test of whether the sleeve is
  merely short 12-month momentum**, and it is the most important diagnostic here given §0.
- **D4 — Uniform-signal variant.** Rates use the same 5-year reversal as everything else,
  removing the term-spread signal entirely. Tests whether the bond signal choice matters and
  whether the carry overlap is load-bearing.
- **D5 — Sub-period split**, pre-2009 vs 2009+.
- **D6 — Unscreened-panel invariance.** The quarantine (data_integrity §5b) touches only
  `EURUSD` and `JPYUSD`, both of which this sleeve excludes. The two panels are therefore
  expected to be **identical on this universe**; this is asserted programmatically rather
  than assumed, and the assertion result is reported.

## 9. Trial accounting

This study adds **1 trial** (PRIMARY). Following the trend sleeve's accounting (34 -> 36) and
allowing for the carry sleeve running concurrently, the DSR bar is reported at `n_trials=32`
(the anchor convention), at **40**, and the sensitivity of the bar to that count is shown so
the verdict does not depend on an accounting guess.

## 10. Expected result — committed in advance, before any number exists

Stated so that the outcome can embarrass the prediction:

| quantity | point estimate | 80% interval |
|---|---:|---|
| Net **gross-of-nothing** Sharpe at 2bps | 0.30 | -0.10 to 0.65 |
| **Net Sharpe at 10bps (the headline)** | **0.25** | **-0.15 to 0.60** |
| Arithmetic active return vs long-only benchmark | +0.5%/yr | -3% to +4%/yr |
| t-stat of that active return | 0.6 | -1.2 to 2.0 |
| **Correlation to the trend sleeve** | **-0.25** | **-0.55 to +0.10** |
| Sharpe positive in every decade | no | — |

**Probability this clears the DSR bar (~0.58 at 44.5 years): ~15%.**
**Probability the correlation to trend is negative: ~70%.**

Reasoning, fixed now: AMP report value-everywhere Sharpes around 0.5 gross, but on a far
richer instrument set with genuine book-to-market data for equities, and value has been weak
since 2010. Fourteen instruments, three blocks, an effective N likely near 4–6, a proxy
signal for the largest block, and no FX at all — this should be a weak sleeve. **It is being
run for its correlation, and the pre-registered expectation is that its standalone Sharpe
does not clear the bar.**

## 11. Falsification criteria — fixed now

| verdict | condition |
|---|---|
| **DEAD** | Arithmetic active return <= 0, **or** its t-stat < 2, **or** net Sharpe at 10bps fails the DSR bar. |
| **UNDETERMINED** | Passes at 2bps and fails at 10bps. **Or** clears full-sample but the per-decade table confines the edge to one era. |
| **PROMISING** | Net Sharpe at 10bps clears the DSR bar at the honest trial count, **and** arithmetic active return > 0 with t >= 2, **and** Sharpe positive in **every** decade, **and** no single (instrument, month) above 5% of P&L. |

Separately and independently of the above, the **DIVERSIFICATION verdict** is reported:

| verdict | condition |
|---|---|
| **DIVERSIFYING** | Correlation to the trend sleeve <= 0.0 **and** the combined equal-risk two-sleeve Sharpe exceeds the trend sleeve's own Sharpe by >= 10%. |
| **MECHANICAL** | Correlation is negative but D3 (skip-12m) removes most of it — the diversification is the reversal window overlapping the momentum window, not an economic effect. |
| **REDUNDANT** | Correlation > +0.3. |

**A sleeve can be DEAD standalone and still DIVERSIFYING, and that combination is a useful
result.** It is exactly the case the portfolio arithmetic predicts, and reporting it
requires the standalone verdict to be given honestly rather than softened.

**30%/yr at half Kelly requires a portfolio Sharpe of 0.894.** Anything less gets reported as
the compound return it actually supports.

**Run once. No tuning. No second look.**
