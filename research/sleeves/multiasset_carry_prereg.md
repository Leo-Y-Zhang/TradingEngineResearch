# PRE-REGISTRATION — Cross-asset CARRY on the long-history multi-asset panel

**Written 2026-07-28, BEFORE any strategy code was written or any backtest was run.**
Nothing in this file may be changed after the run. The run happens ONCE. If the result is
bad, it is banked as bad. No re-tuning of lookbacks, universes, weighting schemes,
thresholds or cost assumptions afterwards — that is the selection bias the apparatus
exists to refuse.

Trial counter: cumulative `n_trials` **34 → 36** (this study spends **2**: the carry
primary, and the trend reference series constructed for the correlation test). Every
Sharpe in the result is judged against the DSR bar at n_trials = 36 and the realised
sample length.

---

## 0. Why carry, and why now

Twelve studies on the US equity cross-section have failed. The last two failed **gross** —
the cost excuse is gone and the signals had nothing. Two findings redirect the search:

1. The DSR bar falls with sample length (1.488 at 7 years, 0.597 at 40, n_trials 32).
2. Breadth converts skill into gross Sharpe exactly as the Fundamental Law predicts; it
   was **cost** that ate it, at 117–236 bps round trip. Index futures, FX forwards and
   broad ETFs cost **1–5 bps** round trip, so breadth can actually be paid for there.

Carry is the second-most-evidenced systematic style after trend and is **known to be
low-correlated to trend**. That is the point of this iteration: the portfolio arithmetic
needs uncorrelated sleeves, and every sleeve tested so far has been an equity
cross-section variant correlated with all the others.

**Success is a Sharpe number.** Max compound growth = S²/2 (full Kelly), 3S²/8 (half).
30%/yr ⇔ S ≈ 0.894 at half Kelly. Every result below is reported as the compound return it
supports.

---

## 1. What can and cannot be built from free data — decided BEFORE the run

The panel's own integrity report (`research/multiasset/data_integrity.md` §4) states that
**FX spot excludes the interest differential — i.e. FX carry itself** — and that the
futures curve is not available. So the honest inventory is:

| Asset class | Carry constructible? | How |
|---|---|---|
| **Rates (US curve)** | **YES** | carry = constant-maturity yield − 13-week bill yield. Both are already in the panel's `yields_monthly.parquet`. |
| **FX** | **YES, with one free addition** | carry = foreign 3-month interbank rate − US 3-month interbank rate. The rates are **not** in the panel; they come from FRED (free, no API key, OECD `IR3TIB01*M156N` family). |
| **Equity** | **YES, for the S&P 500 only** | carry = trailing 12-month **realised** dividend yield − 13-week bill yield. The dividend yield is recovered from the panel's own SPX (price index) vs SPY (total return) pair — the integrity report measures the gap at 1.95%/yr, which is exactly the dividend yield. No other index in the panel has a price/total-return pair, so **no other equity index can carry a signal**. |
| **Commodities** | **NO — EXCLUDED** | The futures curve is not free. Yahoo's `=F` series are front-month splices with no second contract, so a term-structure carry cannot be formed. **Commodities are excluded entirely rather than substituted with momentum**, which would not be carry. |

**Three further exclusions, decided in advance:**

- `USDX` is excluded from the carry cross-section: it is a fixed-weight basket of the same
  currencies already in the universe, so including it double-counts them.
- `NATGAS_F` is excluded everywhere (integrity report §6a: its return is substantially a
  roll splice artefact). Moot for carry since commodities are out; it binds on the trend
  reference.
- `SPY`, `GLD`, `IEF`, `TLT` are excluded from the **trend reference** universe as explicit
  ETF duplicates of series already present (SPX, GOLD_F, US10Y_TR, US30Y_TR).

---

## 2. Data

### 2.1 From the existing panel (`_data/multiasset/`, gitignored, already integrity-proven)

- `returns_monthly.parquet` — month-end compounded returns, quarantine applied.
- `yields_monthly.parquet` — `US5Y_YLD`, `US10Y_YLD`, `US30Y_YLD`, `US13W_YLD`, decimals.
- `cash_monthly.parquet` — the `^IRX` risk-free accrual, for excess returns.

The **month-end** panel is used, not the daily one, exactly as the integrity report §10.1
instructs: the daily panel has a ~1-hour futures/equity session overlap that is a real
lookahead at daily frequency.

### 2.2 New free inputs fetched by `scripts/build_carry_inputs.py`

**FX spot**, yfinance, `auto_adjust=True`, daily, cached under `_data/multiasset/raw/`:

| key | ticker | quote convention | long position gains when |
|---|---|---|---|
| FX_EUR | `EURUSD=X` | USD per EUR | quote rises |
| FX_GBP | `GBPUSD=X` | USD per GBP | quote rises |
| FX_JPY | `JPY=X` | JPY per USD | quote **falls** (inverted) |
| FX_AUD | `AUDUSD=X` | USD per AUD | quote rises |
| FX_NZD | `NZDUSD=X` | USD per NZD | quote rises |
| FX_CAD | `CAD=X` | CAD per USD | quote **falls** (inverted) |
| FX_CHF | `CHF=X` | CHF per USD | quote **falls** (inverted) |
| FX_SEK | `SEK=X` | SEK per USD | quote **falls** (inverted) |
| FX_NOK | `NOK=X` | NOK per USD | quote **falls** (inverted) |

**Short rates**, FRED CSV endpoint (free, keyless), OECD 3-month interbank, monthly, in
percent: `IR3TIB01{US,EZ,GB,JP,AU,NZ,CA,CH,SE,NO}M156N`. One family for every country
including the US, so no maturity or basis is mixed across the differential.

### 2.3 Cleaning applied to the new FX series — the SAME uniform rule, no new judgement

`research.multiasset.panel.clean_levels` (sort, dedupe, drop non-finite) and
`simple_returns` (null a return whose endpoint level is non-positive or whose bar spans
>15 calendar days) are reused unchanged.

The panel's quarantine of eight corrupt 2008 FX closes was admitted by a stated uniform
criterion: **(a) the 8th or 9th of a month in 2008, (b) |return| > 5%, (c) dropping the
close leaves a two-day return under 2.5% in magnitude.** That identical criterion — not a
new one, not a tuned one — is applied mechanically to the seven newly fetched FX series,
and **every** admission is printed with its numbers so the list is auditable. No other
observation is removed from any series for any reason.

### 2.4 Point-in-time safety of the FRED rates

The OECD monthly series is the **average of the month's daily fixings**. A month-`t`
average is fully observable by the end of month `t` (it is an aggregate of quotes that
were public as they printed), so using the month-`t` value as the signal that sets a
position held over month `t+1` is point-in-time safe. It is *stale*, not forward-looking:
if anything it degrades the signal. Signals are lagged one full month everywhere.

---

## 3. The instrument returns — stated exactly, because the classes do not share a convention

Everything below is an **excess return over USD cash**, so the three classes are directly
comparable and the portfolio is self-financing.

**Rates (3 instruments: `US5Y_TR`, `US10Y_TR`, `US30Y_TR`).**
`r_i,t = par_bond_total_return_i,t − cash_t`. Both legs already exist in the panel and the
bond conversion is validated against IEF/TLT/IEI at daily correlation 0.94–0.95.
*Disclosed bias:* the par-bond proxy omits roll-down and therefore understates bond total
return by ~0.5%/yr at 5y and 10y and ~0.1%/yr at 30y. The bias is **against** the sleeve at
the short end.

**FX (9 instruments).** Under covered interest parity, the return to a fully-collateralised
long-foreign / short-USD forward position is the spot move plus the interest differential:

```
r_i,t = spot_return_i,t + (r3m_i,t-1 − r3m_US,t-1) / 12
```

The differential uses the **previous** month's rate — you earn the rate you contracted at,
which also makes it point-in-time safe.
*Disclosed bias:* CIP is assumed. Post-2008 cross-currency basis deviations are of the
order of 10–50 bps/yr in some pairs; that is a real error term this construction cannot
see. It is stated, not hidden, and it is small relative to the differentials being ranked.

**Equity (1 instrument: `SPY_EQ`).** `r_t = SPY_t − cash_t`. SPY is dividend-adjusted, so
this is a genuine total excess return.

**Total universe: 13 instruments.** Nothing else is added later.

---

## 4. The carry signal — one definition, applied to all three classes

Carry is **the annualised return the position earns if prices do not move.**

| class | carry_i,t |
|---|---|
| rates | `yield_i,t − US13W_YLD_t` |
| FX | `r3m_i,t − r3m_US,t` |
| equity | `divyield_t − US13W_YLD_t`, where `divyield_t = (Π₁₂(1+SPY) / Π₁₂(1+SPX)) − 1` over the trailing 12 months |

All three are decimals per year, all observable at month-end `t`, all used to set the
position held over month `t+1`.

---

## 5. Position construction — fixed now, no free parameters left

1. **Volatility.** `σ_i,t` = trailing **36-month** standard deviation of `r_i` through `t`,
   annualised (×√12). An instrument is **eligible** at `t` only with ≥ **24** non-missing
   monthly returns in that window and a non-missing carry.
2. **Score.** `score_i,t = carry_i,t / σ_i,t`. Ranking on carry-per-unit-risk is what makes
   a 1.5% bond term spread comparable to a 5% FX differential once positions are risk-scaled.
3. **Cross-sectional rank.** Rank the `N_t` eligible instruments by `score`, 1 = lowest.
   Require **`N_t ≥ 6`**; months with fewer are not traded.
4. **Rank weights (Koijen–Moskowitz–Pedersen form).**
   `a_i,t = rank_i,t − (N_t+1)/2`, then `w_i,t = a_i,t / Σ_j |a_j,t|`, so `Σ|w| = 1` and
   `Σw = 0` — long the high-carry half, short the low-carry half, with the extremes
   carrying the most weight.
5. **Risk scaling.** `pos_i,t = w_i,t × (0.10 / σ_i,t)`. Every instrument is sized to a
   common 10%-vol reference so a 4%-vol bond and a 12%-vol currency contribute comparably.
   **Neutrality, stated precisely** (written before the run, after the property was
   checked in the test suite; the formula above is unchanged): step 4 is notional-neutral,
   but dividing by each instrument's own volatility in step 5 breaks that. What is
   conserved is `Σ(pos_i·σ_i) = 0.10·Σw_i = 0` — the book carries **equal risk long and
   short**, not equal notional. That is the intended property for a cross-asset book, and
   the residual notional imbalance is reported with the result rather than hidden.
6. **No portfolio-level volatility timing in the primary.** A trailing-vol overlay is a
   separate strategy and would be a second untested bet; the primary is reported at its
   natural volatility, and since Sharpe is invariant to a constant scale factor the headline
   is unaffected. Presentation-only rescaling to 10% ex-post vol is applied to the *equity
   curve*, never to the Sharpe.
7. **Monthly rebalance**, positions set at month-end `t` from information through `t`,
   earning `r_{t+1}`.

## 6. Costs

Turnover-based, charged on notional traded:

```
turnover_t = Σ_i |pos_i,t − pos_i,t-1 × (1 + r_i,t)|
cost_t     = turnover_t × one_way_bps / 10000
```

Two bounds, both reported, **never one alone**:

| bound | round trip | rationale |
|---|---|---|
| realistic | **3.0 bps** | FX forwards 0.2–1 bp, Treasury futures 0.5–1 bp, SPY 1 bp — this is already above the top of the observed range |
| conservative | **10.0 bps** | 3.3× the realistic bound, covering slippage, roll and a small book |

If it fails at 10 bps it is undetermined; if it fails at 3 bps it is **dead**.

---

## 7. Benchmark, and the statistic that killed the last result

The sleeve is **dollar-neutral and self-financing**, so its own return is already an
active, excess-of-cash return. Three statistics are reported, all arithmetic, all with
Newey–West (4 lag) t-statistics:

- **A. `12 × mean(monthly sleeve return)`** — the headline arithmetic active return. This is
  the number reported to the orchestrator.
- **B. OLS alpha of the sleeve on the own-universe benchmark**, annualised — the test of
  whether the sleeve is disguised beta.
- **C. `12 × mean(sleeve − benchmark)`** — the literal own-universe difference.

**Own-universe benchmark:** equal-risk long-only ownership of the same 13 instruments,
`pos_i = (1/N_t) × (0.10 / σ_i,t)` — "own everything at equal risk".

**Geometric excess is not reported as evidence.** Geometric excess = arithmetic excess −
(σ²_strat − σ²_bench)/2, so a lower-volatility strategy shows a fake positive excess. That
illusion killed the PEAD result and it will not be repeated here.

---

## 8. Everything that must be measured, declared in advance

1. **Sharpe per decade** (2000s / 2010s / 2020s, and per-decade for the long-history
   secondaries). A full-sample number carried by one era is not deployable.
2. **P&L concentration.** Largest single (instrument, month) share of total net P&L and of
   gross absolute P&L. Alarm at 3%.
3. **Per-instrument P&L attribution**, so a single currency carrying the sleeve is visible.
4. **Carry-accrual vs price-move decomposition.** How much of gross P&L is the deterministic
   differential/coupon accrual and how much is spot/price movement. A carry sleeve whose
   entire P&L is accrual and whose price leg is equally negative has found nothing.
5. **Breadth.** Position sign flips per year, summed across instruments = bets/year.
6. **Max drawdown, skewness, worst month.** Carry is known to crash.
7. **Negative control.** Per-date permutation of the carry scores across eligible
   instruments, 4 fixed seeds. The live Sharpe must stand clear of that distribution.
8. **DSR bar** at the realised sample length, n_trials = 36, via
   `research.multiasset.panel.dsr_sharpe_bar`.
9. **Half-Kelly compound return** `3S²/8` at the net Sharpe.
10. **Sensitivity to the cleaning decision** — the whole study re-run on the unscreened
    panel and unquarantined new FX.

### 8b. Secondaries — declared here, reported unconditionally, never selected between

These exist to show WHERE the primary's result comes from. The primary is the primary
whatever they say; none of them may be promoted to headline afterwards.

| # | secondary |
|---|---|
| **S1** | **Bonds only** (the 3 US curve points). A three-instrument class cannot satisfy `N_t ≥ 6`, so the eligibility floor for this secondary alone is `N_t ≥ 3`. The primary is unaffected. |
| **S2** | **FX only** (the 9 currencies), floor `N_t ≥ 6` as in the primary. |
| **S3** | **Trailing-volatility overlay** on the primary: scale by `0.10 / σ̂_36m`, lagged one month, clipped to [0.25, 4]. A second bet, kept out of the primary deliberately. |
| **S4** | **Unscreened sensitivity** — the whole study on the unquarantined panel and unquarantined new FX. |

---

## 9. The trend reference and the two-sleeve test (the actual point of this iteration)

No multi-asset trend sleeve exists on disk yet. One is therefore **constructed here as a
reference series**, with a standard, untuned specification, spending trial 36:

- Universe: the 27 tradable panel instruments **minus** `NATGAS_F`, `SPY`, `GLD`, `IEF`,
  `TLT` (roll contamination and ETF duplicates) **minus** the panel's four spot-only FX
  keys `USDX`, `EURUSD`, `GBPUSD`, `JPYUSD` → 18, **plus** the nine CIP-consistent FX
  excess-return instruments built in §3 → **27 instruments**.
  *Why the FX swap* (decided before the run): a spot-only return is not the return of any
  tradable FX position, and using the same FX construction in both sleeves is what makes
  the correlation between them mean anything. `USDX` is dropped for the same
  double-counting reason as in the carry universe.
  *Disclosed:* equity index returns are local-currency and the price indices exclude
  dividends (~2-4%/yr); for a sign-based signal that is a small distortion, but it is a
  distortion, and it is one more reason this is a reference rather than a verdict.
- Signal: `sign(trailing 12-month compounded excess return)`, month-end, no lag beyond the
  one-month holding convention.
- Sizing: `pos_i,t = sign_i,t × (0.10 / σ_i,t) / N_t`, same 36-month σ, same eligibility
  rule, same monthly rebalance, same two cost bounds.
- This is a **reference**, not a verdict on trend. Whoever runs trend properly should treat
  it as a separate, fuller study.

**The test:**

1. `ρ` = correlation of the carry sleeve's monthly net returns with the trend reference's,
   over their overlapping months, with the number of overlapping months reported.
2. If they are materially uncorrelated, the two-sleeve portfolio Sharpe is computed **two
   ways** and both are reported:
   - the brief's formula `S = s·√(N/(1+(N−1)ρ))` with `N = 2`, which assumes **equal**
     sleeve Sharpes and equal weights;
   - the **directly measured** Sharpe of the realised 50/50 equal-risk combination.
   The measured number is the one that counts; the formula is reported as the check.
3. The half-Kelly compound return `3S²/8` the combination supports.

---

## 10. Falsifiable predictions, recorded before the run

| # | prediction |
|---|---|
| **P1** | Primary gross Sharpe lands in **[0.40, 0.90]**. |
| **P2** | Costs remove **< 0.10 Sharpe points** under the realistic bound (low turnover × cheap instruments). |
| **P3** | \|ρ(carry, trend)\| ≤ **0.30**. |
| **P4** | Max drawdown **> 15%** and monthly skewness **< 0** — carry crashes. |
| **P5** | Breadth **< 40 bets/yr**. Carry is a slow signal; the Fundamental Law then caps the IR, and this is the sleeve's structural weakness. |
| **P6** | The accrual leg is **> 60%** of gross P&L. |
| **P7** | The primary alone does **not** reach Sharpe 0.894 (30%/yr at half Kelly). The multi-sleeve combination is the only route the arithmetic leaves open. |

---

## 11. Verdict rule, fixed in advance

Let `S_net` = net Sharpe under the **realistic** bound, `S_net_cons` under the conservative
bound, `bar` = the DSR bar at the realised length and n_trials = 36.

- **PROMISING** — `S_net ≥ bar` AND `S_net_cons > 0` AND t-stat of statistic **A** ≥ 2.0
  AND no calendar decade with a negative Sharpe AND max (instrument, month) P&L share < 3%
  AND the negative control is ≥ 2 sd below the live Sharpe.
- **MARGINAL** — `S_net ≥ 0.35` and t(A) ≥ 1.5, but at least one PROMISING condition fails.
- **DEAD** — otherwise.

Separately and regardless of the tier: whether the sleeve, or the two-sleeve combination,
reaches **S = 0.894** (30%/yr at half Kelly) is reported as a plain yes/no.

**One run. No second run with different parameters. Whatever comes out is the result.**
