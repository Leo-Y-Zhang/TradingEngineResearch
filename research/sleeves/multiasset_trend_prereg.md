# PRE-REGISTRATION — Multi-asset time-series momentum (trend)

**Written 2026-07-28, BEFORE any backtest of this sleeve was run.** Everything below was
fixed in advance and is executed exactly once. Nothing here was chosen by looking at a
result. If a number disagrees with a hope, the number is what gets reported.

Data: `_data/multiasset/returns_monthly.parquet` + `cash_monthly.parquet`, built and
audited by `research/multiasset/data_integrity.md`. Code: `research/sleeves/multiasset_trend.py`.

---

## 0. Why this sleeve, and what would make it different from the twelve that died

Twelve studies on the US equity cross-section failed. Two measured findings redirect here:

1. **The DSR bar falls with sample length.** DSR>=0.95 at `n_trials=32` needs annual Sharpe
   **1.488 on 7 years** but only **0.597 on 40**. Every prior study ran on 7–17 years.
2. **Breadth converts skill into gross Sharpe as the Fundamental Law predicts** (predicted
   IR 0.457 vs realised 0.530) — **costs** ate it, at 117–236bps round trip. Index futures
   and broad ETFs cost 1–5bps, which is 25–100x cheaper.

This sleeve is the intersection: the longest honest history available, on the cheapest
instruments that exist, with breadth from both instruments and lookbacks.

**It is also the highest-prior anomaly in the programme** — time-series momentum is the
most robustly evidenced effect in finance across the widest instrument set and the longest
history. That cuts both ways and is stated now: a high prior means a *negative* result here
is strong evidence, and a positive result is the *least* surprising thing this programme
could find, therefore the most likely to be a rediscovery of something already arbitraged.

**The honest failure mode to watch for:** trend-following's public track record largely
predates 2009. If the edge is confined to pre-2009 decades, it is not deployable, and the
per-decade table below is the instrument that will say so.

---

## 1. Universe — decided now, from the data receipt, not from any return

**Selection rule (applied before any backtest):** one instrument per distinct underlying
risk, preferring the **longest history**; exclude series with a known structural defect;
exclude ETF duplicates of a series already present.

**PRIMARY = 18 instruments**, unbalanced panel (an instrument enters when it becomes
eligible and never leaves):

| block | instruments | n |
|---|---|---:|
| Equity index | SPX, NASDAQ, FTSE100, N225, DAX, HSI, ASX200 | 7 |
| Rates (par-bond TR) | US5Y_TR, US10Y_TR, US30Y_TR | 3 |
| Commodity futures | GOLD_F, WTI_F, SILVER_F, COPPER_F | 4 |
| FX spot | USDX, EURUSD, GBPUSD, JPYUSD | 4 |

**Excluded, with reasons fixed in advance:**

| excluded | reason |
|---|---|
| `NATGAS_F` | Roll-contaminated. 65.7% of its \|r\|>15% bars fall in days 24–31 vs a 24.0% base rate (2.74x lift); its 16.54%/yr headline is substantially a splice artefact (data_integrity §6a). Not a price series. |
| `DJIA` | Redundant with SPX and only starts 1992 — adds a second US large-cap without adding history. |
| `SPY`, `TLT`, `IEF`, `GLD` | ETF duplicates of SPX / US30Y_TR / US10Y_TR / GOLD_F, each with strictly shorter history. |
| `DBC` | Broad commodity basket that overlaps all four commodity futures already included. |
| `EFA` | Developed-ex-US basket that overlaps FTSE100 + N225 + DAX. |
| `EEM` | The one debatable exclusion: emerging markets is a genuinely distinct risk with no substitute in the panel. Excluded because it is ETF-era only (2003+) and ETFs are the sole block carrying real survivorship bias (data_integrity §7). Recorded here as a judgement call, not an oversight. |
| `BIL`, `IEI`, `SLV` | Validation-only by construction; asserted disjoint from the tradable panel. |
| `US_CASH_13W` | The risk-free leg, not a risk asset. |

**Eligibility:** instrument `i` is eligible at month-end `t` if it has **>= 36 non-null
monthly returns ending at or before `t`**. The book trades only when **>= 3 instruments are
eligible**, which excludes a 30-year single-instrument prefix that would be an SPX study
wearing a multi-asset label. Expected start ~1965–66, expected span ~60 years.

Expected eligible count by year (from the coverage receipt, not from returns): 1966:3,
1968:4, 1974:6, 1980:7, 1987:8, 1991:10, 1996:11, 1999:12, 2003:16, 2007:18.

---

## 2. Return convention — and the one place cash is subtracted

The strategy is a **fully-collateralised futures book**, so every reported return stream is
an **excess return over cash**. The panel's series are not all on the same footing, so:

| series | treatment | justification |
|---|---|---|
| 7 equity price indices, 4 commodity futures, 4 FX spot | **used raw** as the excess return | A futures price return already embeds financing; the cash leg is the collateral, not the position. |
| US5Y_TR, US10Y_TR, US30Y_TR | **minus `US_CASH_13W`** | These are par-bond **total** returns in USD, i.e. cash + excess. Subtracting the bill rate recovers the futures-equivalent excess return. |

**Disclosed conventions that this does not fix** (all in data_integrity §3, §4, §6):

- Price indices exclude dividends (SPX vs SPY: measured **1.95%/yr**; UK/AU higher). For a
  cash index that is a downward bias; for a *futures* excess return the correct object is
  `price return + dividend yield - local risk-free`, so using the raw price return carries
  an error of `(local rf - dividend yield)`, historically of order **+1 to 2%/yr in the
  strategy's favour on a long position** in high-rate eras. **This bias is common-mode with
  the long-only benchmark**, which is precisely why the benchmark is defined as it is; the
  residual is the part scaled by (strategy net exposure - 1), and the strategy's average net
  long exposure is reported so the residual can be sized.
- **DAX is a total-return index while the other six equity indices are price-only.** Not
  corrected (no EUR cash series in the panel). Present in strategy *and* benchmark.
- Local currency is not converted to USD. For a **futures** position this is close to
  correct: USD return ~= local return x (1 + fx return), and the cross-term is second-order
  at monthly frequency. For a cash index position it would be wrong. Disclosed, not fixed.
- Bond series omit roll-down and therefore **understate** bond total return by ~0.51%/yr at
  5y and ~0.55%/yr at 10y (measured vs IEI/IEF). This biases *against* the bond block.
- FX spot excludes the interest differential, so **no carry is available here** and none is
  claimed. This is a pure trend study.

**Monthly panel only.** The daily panel has a ~1-hour futures/equity session overlap that is
a genuine lookahead at daily frequency (data_integrity §6b); it is negligible monthly.

---

## 3. Signal

At month-end `t`, for eligible instrument `i`, for each lookback `L` in **{1, 3, 6, 12}**
months:

```
s_L(i,t) = sign( sum of the instrument's last L monthly excess returns, ending at t )
```

Composite signal, equally weighted across the four lookbacks:

```
S(i,t) = (1/4) * [ s_1 + s_3 + s_6 + s_12 ]      in {-1, -0.75, ..., +0.75, +1}
```

All four lookbacks must exist (>= 12 months of returns), which the 36-month eligibility rule
already guarantees. `sign(0) = 0`.

**Amendment, written before the run — interior missing months.** The primary universe
contains exactly **2** interior null cells (measured, not estimated): `EURUSD` and `JPYUSD`
in **2008-08**, both produced by the panel's >15-calendar-day gap nulling. Treating a null
as "signal undefined" would delete the following 12 months of two instruments over two
cells. Rule fixed now: **an interior null is treated as a zero return** inside the trailing
sum and the volatility window, and the instrument **holds no position in that month**
(zero P&L, zero turnover). Leading nulls, before an instrument's first observation, remain
ineligible as specified. The count is 2 cells out of ~7,000, so this rule cannot carry a
result; it is recorded because it was a choice.

**The four lookbacks are the second source of breadth** and are quasi-independent by
construction: a 1-month and a 12-month signal on the same instrument disagree often. The
realised correlation between the four single-lookback books is **measured and reported**,
not assumed.

**No lookback, no weighting, no threshold, no smoothing, no signal cap will be tuned.**
1/3/6/12 equal-weight is the canonical Moskowitz–Ooi–Pedersen specification and is fixed.

---

## 4. Sizing

**Step 1 — inverse volatility per instrument.** `sigma(i,t)` = annualised standard deviation
of instrument `i`'s excess returns over the **trailing 36 months ending at `t`** (minimum 24
observations, else ineligible). Position notional:

```
n(i,t) = S(i,t) * ( 0.10 / sigma(i,t) )
```

The 0.10 is an arbitrary per-instrument vol unit; it cancels in step 2 and is not a free
parameter.

**Step 2 — scale the whole book to a target volatility.** Raw book excess return
`b(t+1) = sum_i n(i,t) * x(i,t+1)`. Book vol estimate `sigmaHat_b(t)` = annualised stdev of
`b` over the **trailing 36 months ending at `t`** (expanding window while fewer than 36 are
available, minimum 12). Realised strategy excess return:

```
r_strat(t+1) = k(t) * b(t+1),      k(t) = sigma_target / sigmaHat_b(t)
```

`sigmaHat_b(t)` uses only book returns realised at or before `t`, so `k(t)` is causal.

**Volatility targets reported: 10%, 20%, 40%, 60%.**

**Stated in advance so it cannot be spun later:** with strictly proportional costs, gross
return, cost and volatility all scale linearly in `k`, so **net Sharpe is invariant to the
vol target** except through the leverage cap below. The four targets are therefore **one
trial, not four**, and they exist to translate a Sharpe into a compound return, not to search.

**Leverage cap (pre-registered).** Gross notional `sum_i |k(t) * n(i,t)|` is capped at
**10x book equity**; `k(t)` is scaled down when it binds. An uncapped 60%-vol book is not
implementable, and reporting a number that no futures margin regime permits would be a
number I did not measure. **The fraction of months where the cap binds is reported at every
vol target**, and this is the only channel through which the four targets can differ.

---

## 5. Costs

```
turnover(t) = sum_i | w(i,t) - w(i,t-1) |          (w = k(t)*n(i,t), fraction of equity)
cost(t)     = 0.5 * c_roundtrip * turnover(t)
```

The 0.5 is because `sum |delta w|` counts a full round trip as **two** units of notional
change while `c_roundtrip` prices it once.

**Two brackets, both reported for every result:**

| bracket | value | rationale |
|---|---:|---|
| (a) optimistic | **2bps** round trip | E-mini S&P / major index futures: commission ~0.2bp + half-spread ~0.5bp each way. Liquid futures genuinely trade here at research size. |
| (b) conservative | **10bps** round trip | Covers the illiquid tail of this universe (copper, silver, ASX/HSI futures), plus slippage and modest market impact. Broad-ETF execution sits at 1–5bps, so 10bps is deliberately above the honest range. |

A result that passes at (a) and fails at (b) is **UNDETERMINED**, not a pass. Both are
reported side by side, always.

**The benchmark pays the same cost schedule** on its own rebalancing turnover. A costless
benchmark against a costed strategy is a rigged comparison.

---

## 6. Benchmark and the active-return test

**Benchmark:** equal-weight, **long-only**, monthly-rebalanced holding of exactly the
instruments eligible at `t`, on the identical excess-return convention:
`r_bench(t+1) = (1/N_t) * sum_i x(i,t+1)`.

Three active measures, all reported:

1. **PRIMARY — arithmetic active return.** `a(t) = r_strat(t) - r_bench(t)`, reported as
   `12 * mean(a)` with a **Newey–West t-statistic** (lag 6). This is the headline.
2. **Jensen alpha.** OLS `r_strat = alpha + beta*r_bench + e`; annualised `alpha` with a
   Newey–West t-stat. Beta-adjusted, so it is immune to the vol-mismatch objection.
3. **Vol-matched active return.** Strategy scaled by the constant `sigma_bench/sigma_strat`
   over the full sample, then differenced. Disclosed: the scalar uses full-sample
   information. It is a positive constant and therefore **cannot create alpha or change a
   sign**; it only makes the comparison legible.

**The variance-drag identity is computed and printed explicitly:**

```
geometric excess = arithmetic active - (var_strat - var_bench)/2
```

A positive geometric excess produced entirely by the second term is **not alpha**. This is
the exact illusion that killed the PEAD result; it is checked mechanically here rather than
being left to judgement.

---

## 7. What will be reported — every item, pass or fail

- Gross and net Sharpe at **both** cost brackets and **all four** vol targets.
- **Arithmetic active return with Newey–West t-stat** (primary), Jensen alpha, vol-matched
  active return, and the variance-drag decomposition.
- **Sharpe per decade** (1960s partial, 1970s, 80s, 90s, 2000s, 2010s, 2020s). A full-sample
  number carried by the 1970s–80s is explicitly declared **not deployable** in advance.
- **DSR bar** via `research.multiasset.panel.dsr_sharpe_bar` at the realised sample length,
  reported at `n_trials=32` (the recorded anchor convention) **and** at the honest cumulative
  count including this study, with a pass/fail on the net Sharpe.
- **Half-Kelly reachable return `g = 3S^2/8`** on the net Sharpe, with the implied book
  volatility `sigma* = S/2` at half Kelly.
- Max drawdown, worst month, skew, kurtosis.
- **P&L concentration**: share of total P&L from the single largest (instrument, month) and
  from the largest instrument. One name-month was once 13% of a study's P&L.
- **Breadth**: mean eligible instruments, bets/year, and **effective N** from the eigenvalues
  of the instrument correlation matrix — because 4 FX pairs that are all the dollar are not
  4 bets.
- Turnover per year, gross leverage distribution, cap-binding frequency, average net long
  exposure (needed to size the dividend-convention residual of §2).
- Correlation matrix of the four single-lookback sub-books.

## 8. Robustness runs declared in advance (not tuning)

Each is reported whatever it says. **Only SENSITIVITY-B is a second strategy claim; the rest
are diagnostics that cannot be promoted to a headline.**

- **SENSITIVITY-B — block risk parity.** Equal risk to each of the four blocks (equity,
  rates, commodity, FX) instead of flat inverse-vol, to test whether the result is an
  artefact of 7 equity indices and 4 dollar proxies dominating the risk budget. This is a
  genuine second configuration and is **counted as a trial**.
- **Unscreened panel.** Re-run on `returns_monthly_unscreened.parquet` (retaining the 8
  quarantined 2008 FX closes) to test the cleaning decision rather than trust it.
- **Negative control.** Signal sign randomised per (instrument, month) at 8 fixed seeds.
  Expected net Sharpe ~0. If the live result is not several standard deviations outside this
  distribution, it is machinery, not edge.
- **Sub-period split.** Pre-2009 vs 2009+, stated in advance because trend-following's public
  record is concentrated pre-2009.
- **Single-lookback books.** 1, 3, 6 and 12 reported separately — to show whether the
  composite is genuinely aggregating four signals or riding one.

## 9. Trial accounting

This study adds **2 trials** (PRIMARY + SENSITIVITY-B). Cumulative `n_trials` goes
**34 -> 36**. Diagnostics in §8 are not counted because none of them can become the headline.
The DSR bar is reported at both 32 and 36 so the comparison to the recorded anchors stays
legible.

## 10. Falsification criteria — fixed now

| verdict | condition |
|---|---|
| **DEAD** | Arithmetic active return <= 0, or its t-stat < 2, or the net Sharpe at 10bps fails the DSR bar. |
| **UNDETERMINED** | Passes at 2bps, fails at 10bps. Or clears full-sample but the per-decade table shows the edge confined to one era. |
| **PROMISING** | Net Sharpe at **10bps** clears the DSR bar at the honest trial count, arithmetic active return positive with t >= 2, Sharpe positive in **every** decade, and no single (instrument, month) exceeding 5% of P&L. |

**30%/yr at half Kelly requires net Sharpe 0.894.** That is the number this has to produce.
Anything less gets reported as the compound return it actually supports.

**Run once. No tuning. No second look.**
