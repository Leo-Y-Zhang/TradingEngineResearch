# PRE-REGISTRATION — RISK PARITY on the long-history multi-asset panel

**Written and committed BEFORE `research/sleeves/riskparity.py` exists.** Run ONCE. No
tuning, no second look, no parameter moved after the fact. Every number in §10 is a
forecast recorded before any risk-parity book has been computed.

---

## 1. Why this study exists

Four studies on this panel (trend, value, carry, calendar seasonality) have now died
against the **same** statistic: the equal-weight long-only benchmark drawn from their own
universe. Measured over 61.33 years on 18 instruments it returns **Sharpe 0.7065** and
**clears the DSR ≥ 0.95 bar at n_trials = 304**. It beats trend (0.61), carry (0.43) and
seasonal (0.47) individually and sits within 0.06 of the entire three-sleeve stack (0.714).

The question nobody has asked directly: **if diversified passive is the best thing this
programme can measure, what does it deliver when risk-budgeted and levered properly?**
That is risk parity — Bridgewater All Weather, AQR Risk Parity — a documented, deployable
allocation with **no signal to overfit**. The hypothesis under test is not "we found alpha".
It is "the programme's answer was always *own everything, sized by risk, levered*."

**The real deliverable is not the Sharpe.** A correction recorded at 04:15 on 2026-07-28
established that half-Kelly growth figures are unreachable: a strategy that needs 3× leverage
to hit its half-Kelly return dies on its own historical drawdown first. So the pre-registered
headline of this study is:

> **What is the highest compound return achievable at a maximum drawdown the account
> actually survives (≤ 50%)?**

That number, not the Sharpe, is the honest answer to "how close to 30%/yr can this get".

## 1a. What is deliberately NOT tested (so its absence is auditable)

Risk parity has a large parameter space and this study fixes all of it in advance. The
following are named here so that a later appearance of any of them is visibly a second look:

- vol windows other than **36 months** (the trend sleeve's, inherited unchanged — 12m, 24m,
  60m, EWMA and blended estimators are NOT tried);
- covariance-aware sizing (equal *risk contribution* solved on the full covariance matrix,
  minimum-variance, maximum-diversification, hierarchical risk parity) — only the two
  diagonal schemes in §4 are run;
- any tilt, momentum overlay, drawdown control, vol-of-vol adjustment, or dynamic vol target;
- any rebalance frequency other than **monthly**;
- any universe other than the 18 pre-registered instruments and the single pre-registered
  rates-excluded variant of §7.4c;
- any leverage cap other than the trend sleeve's **10× gross**.

## 2. Data and universe — fixed, no choices left

- Panel: `_data/multiasset/returns_monthly.parquet` (screened), loaded through
  `research.sleeves.multiasset_trend.load_excess_panel()` unchanged, so the return
  convention is bit-identical to the sleeves this is compared against.
- Universe: `multiasset_trend.PRIMARY_UNIVERSE` — the **same 18 instruments** the trend and
  seasonal sleeves used, so results are directly comparable:
  - **equity (7)** SPX, NASDAQ, FTSE100, N225, DAX, HSI, ASX200
  - **rates (3)** US5Y_TR, US10Y_TR, US30Y_TR
  - **commodity (4)** GOLD_F, WTI_F, SILVER_F, COPPER_F
  - **fx (4)** USDX, EURUSD, GBPUSD, JPYUSD
- Returns are **excess** returns. The three `*_TR` bond series are USD total returns and have
  the 13-week bill subtracted; every other series is a price / futures / spot return, which
  is already a futures-equivalent excess return.
- Cash / financing leg: `_data/multiasset/cash_monthly.parquet` → `US_CASH_13W`.
- Interior nulls are treated as a zero return with no position held (the trend prereg's
  amendment, inherited).

**Known biases in this panel, stated in advance and not repaired:**

| bias | direction on the reported result |
|---|---|
| 18 instruments are **hindsight-selected surviving major markets** (no Nikkei-that-vanished, no defaulted sovereign, no delisted exchange) | **UPWARD — the single largest bias here.** Every reported number is an upper bound on what was investable ex ante. |
| equity legs are **price-only** (dividends excluded, ≈2–4%/yr depending on era; DAX is the sole total-return index) | downward |
| bond legs are constant-maturity par-bond repricings with **no roll-down** (≈0.5%/yr) | downward |
| commodity legs are **front-month continuous, roll gaps not back-adjusted** | ambiguous; NATGAS_F excluded from the universe already |
| coverage grows from 3 instruments in 1965 to 18 by the 1990s | the early sample is *far* less diversified than the headline "18 instruments" implies — see §7.5 |

## 3. Sample

The book runs from the first month at least **3** instruments are eligible to the last
complete calendar month in the panel. On the trend sleeve's eligibility rule that is
expected to be **1965-03-31 → 2026-06-30, 736 months, 61.33 years** (the seasonal study's
recorded span; if the realised span differs it is reported, not adjusted).

## 4. The three books — long-only, weights sum to 1, decided at t, held over t+1

All three share the same eligibility rule, inherited unchanged from the trend sleeve:
instrument *i* is eligible at decision time *t* iff it has ≥ **36** non-null monthly
observations through *t* **and** its trailing volatility exists and is > 0. Trailing
volatility is `σ_i,t = std(r_i, t−35..t, ddof=1) × √12`, window **36 months, min 24 obs**,
strictly point-in-time. The book is flat if fewer than **3** instruments are eligible.

- **W0 — EQUAL WEIGHT (the benchmark).** `w_i = 1 / N_eligible`. This is the statistic that
  killed four studies; it is reproduced here inside this study's own pipeline as a
  correctness control (§8).
- **W1 — RISK PARITY, naive.** `w_i ∝ 1/σ_i,t`, normalised to sum to 1 over the eligible set.
- **W2 — RISK PARITY, bucketed** (the All Weather construction — "so bonds do not dominate",
  two-level):
  1. within block *b*: `u_i ∝ 1/σ_i,t`, normalised to sum to 1 over eligible names in *b*;
  2. the block sub-portfolio return `r_b,t+1 = Σ_i u_i,t · r_i,t+1` is accumulated
     point-in-time, and `σ_b,t` is its trailing 36-month annualised std (min 12 obs);
  3. block weights `W_b ∝ 1/σ_b,t` over **live** blocks, normalised to 1; if `σ_b,t` is not
     yet estimable the live blocks are weighted equally;
  4. `w_i = W_b · u_i`.

Rebalance **monthly** at month-end. Weights decided from information through *t* are held
during *t+1* — the trend sleeve's shift convention, so no lookahead is possible.

## 5. Leverage ladder, financing, and costs

**Leverage.** For each volatility target τ ∈ **{10%, 15%, 20%, 25%, 30%, 40%}**:
`k_t = τ / σ̂_book,t`, where `σ̂_book,t` is the trailing **36-month** annualised std
(min 12 obs) of that book's own **unlevered** excess-return series through *t*, strictly
point-in-time. `k_t` is capped at **10× gross** (the trend sleeve's `GROSS_CAP`); the
frequency with which the cap binds is reported, because a binding cap is an unintended
volatility limiter and has previously been mistaken for an edge.

**Financing — leverage is NOT free.** The book holds excess returns, so 1× is already
financed at the bill rate. Any notional above 1× is charged explicitly:

`R_t = cash_t + k_{t−1}·(book excess)_t − trading_costs_t − (spread/12)·max(k_{t−1} − 1, 0)`

| financing case | spread over 13-week bill | rationale |
|---|---|---|
| **PRIMARY** | **+1.50%/yr** | portfolio-margin / futures-carry realistic for a funded account |
| optimistic | +0.50%/yr | institutional futures financing |
| retail | +3.00%/yr | retail margin |
| legacy | **flat 6.00%/yr all-in** on `max(k−1,0)`, replacing bill+spread | reproduces what prior work in this repo charged, for comparability |

Mean bill rate on this sample is ≈4.6%/yr, so PRIMARY ≈ the legacy flat 6% *on average* but
correctly tracks the rate regime — which matters, because the 1970s–80s financed at 8–15%.

**Trading costs.** Round-trip bracket **2 bps** (realistic for these instruments) and
**10 bps** (conservative), charged on the *levered* book's turnover:
`cost_t = 0.5 · c · Σ_i |Δw^L_i|`. **The conservative 10 bps bound is the headline.** Both
are always reported.

## 6. Metrics — reported for every (book × τ × cost × financing) cell

1. **compound (geometric) annual return** of the total-return series;
2. annualised volatility;
3. **Sharpe** on the net excess series `R − cash`;
4. **maximum drawdown** on the compounded total-return path;
5. **time-to-recover**: months from the worst drawdown's trough back to the prior peak
   (`NOT RECOVERED` if it never does inside the sample), plus the full underwater duration
   peak → recovery;
6. mean and max gross leverage, and cap-binding frequency;
7. turnover ×/yr.

**Compound return and max drawdown are reported side by side in the same row, always.**
Reporting a growth rate without the drawdown that reaches it is the specific error this
study exists to avoid.

## 7. Mandatory tests — each of these killed a prior result

**7.1 Matched volatility.** RP is compared to EW **levered to the same target**, never to an
unlevered benchmark. Reported per τ: vol-matched active return with a Newey-West(6) t-stat,
Jensen α and β, geometric excess, and the variance-drag identity — via the trend sleeve's
`active_report`, unchanged.

**7.2 The benchmark goes through the DSR gate too.** `dsr_sharpe_bar(years, n_trials)` at
**n = 32 / 46 / 56 / 304** for RP *and* EW. n = 46 is the honest cumulative count
(44 after the seasonal study, + 2 spent here — W1 and W2; the leverage ladder is a scaling,
not a search, and the EW benchmark is not a fitted strategy). n = 304 is the
inherited-search bar the seasonal study applied to the benchmark, carried over for
comparability.

**7.3 Sharpe per decade** for RP and EW, plus the mean number of eligible instruments per
decade, because 61 years spans very different rate regimes and the early sample is thin.

**7.4 THE BOND BULL MARKET — the single most likely artefact here.** A 40-year bond bull
market flatters *any* bond-heavy risk-parity book, and it has killed real funds. Four tests,
all pre-specified:

- **(a) EXCLUDE it.** Drop **1981-10 → 2021-12** from the return series and recompute
  Sharpe, compound return and drawdown on what is left (1965-03→1981-09 and 2022-01→2026-06).
- **(b) Inside it.** The same statistics on 1981-10 → 2021-12 alone.
- **(c) Rates removed.** Re-run the *entire* pipeline on the 15 non-rates instruments, full
  sample. If the result depends on owning bonds, this is where it shows.
- **(d) The rate shock.** 2022-01 → 2026-06 alone — the period that broke live risk-parity
  funds.

**7.5 Gross-notional concentration.** Inverse-vol sizing put 65% into 3 instruments in
another sleeve; risk parity will do this **more**. Reported per book: mean/max share of the
top instrument and the **top 3**, the weight-vector effective N (`1/Σw²`), block shares
(equity / rates / commodity / fx), and the mean eligible count — each as a time series
summary and by decade.

**7.6 P&L concentration.** Top (instrument, month) cell as a share of net P&L (**3% alarm**,
inherited) and top instrument share, via the trend sleeve's `concentration`.

**7.7 Cost sensitivity.** Both brackets everywhere, plus the round-trip **breakeven cost**
at which the RP-vs-EW vol-matched active return goes to zero.

## 8. Verification controls — run before the result is read

| control | pass condition |
|---|---|
| EW reproduction | this pipeline's EW Sharpe reproduces the recorded **0.7065 ± 0.03** |
| leverage invariance | gross (pre-cost, pre-financing) Sharpe is identical across all six τ wherever the 10× cap does not bind |
| point-in-time audit | weights for a sample of month-ends rebuilt from a panel **truncated at that month-end** are identical to the full-sample weights for that row |
| noise control | weights are a function of lagged volatility only — rebuilding them on a sign-permuted panel must leave the *weights* unchanged in structure and drive the book's Sharpe to ≈0 |
| determinism | a second run produces a byte-identical `result.json` |
| financing arithmetic | at k ≡ 1 the financing charge is exactly 0, and `R − cash` equals the unlevered book's excess return net of costs to 1e−12 |

## 9. Verdict rule — fixed now, applied mechanically

Let `S` = RP (better of W1/W2 as pre-declared by higher net-10bps Sharpe) net Sharpe at
**10 bps**, PRIMARY financing.

**PROMOTABLE AS ALPHA** requires **all** of:
1. `S ≥ dsr_sharpe_bar(years, n_trials=46)`;
2. vol-matched active return vs EW at matched vol **> 0 with NW t ≥ +2.0**;
3. no decade with a negative Sharpe;
4. Sharpe excluding the bond bull market (§7.4a) is **> 0 and ≥ 0.50 × full-sample Sharpe**;
5. mean top-3 gross-notional share **< 70%**;
6. top (instrument, month) P&L cell **< 3%**.

**DEPLOYABLE AS BETA** — a deliberately separate, weaker category, because risk parity is an
*allocation*, not an alpha claim, and holding it to an alpha bar would be the wrong test —
requires all of:
1. `S ≥ dsr_sharpe_bar(years, n_trials=46)`;
2. survives §7.4a (Sharpe outside the bond bull market > 0);
3. **some** τ on the ladder delivers compound ≥ **12%/yr** at max drawdown ≤ **50%**, net of
   10 bps and PRIMARY financing.

**ANSWERS THE 30% QUESTION** iff some τ delivers compound ≥ **30%/yr** at max drawdown
≤ **50%**, net of 10 bps and PRIMARY financing. Pre-registered expectation: **NO** (P7).

**DEAD** if neither category is met.

## 10. Predictions — recorded before the run, scored honestly afterwards

| # | prediction | point | band |
|---|---|---:|---|
| **P1** | W1 risk-parity naive, net 10 bps Sharpe | **0.80** | [0.65, 0.95] |
| **P2** | W2 risk-parity bucketed, net 10 bps Sharpe | **0.82** | [0.65, 1.00] |
| **P3** | W0 equal weight reproduces the recorded benchmark | **0.70** | [0.68, 0.73] |
| **P4** | RP vs EW vol-matched active is **positive but not significant** | **+1.5%/yr, t ≈ +1.2** | t < +2.0 |
| **P5** | **max drawdown at τ = 20%** (best RP, 10 bps, PRIMARY financing) | **−42%** | [−30%, −60%] |
| **P6** | **max drawdown at τ = 40%** | **−75%** | [−55%, −95%] |
| **P7** | **highest compound return at max drawdown ≤ 50%** — *the headline* | **15%/yr** | [9%, 21%] — and explicitly **NOT ≥ 30%** |
| **P8** | mean top-3 gross-notional share, W1 | **55%** | [40%, 70%] |
| **P9** | RP Sharpe **excluding** the bond bull market | **0.45** | [0.10, 0.75] — a fall of ≥ 0.20 from full sample |
| **P10** | RP Sharpe on the **rates-excluded** 15-instrument universe | **0.65** | [0.45, 0.85] |
| **P11** | RP clears the DSR bar at n = 46 but **does not reach 0.894** (the Sharpe 30%/yr needs) | clears / fails | — |
| **P12** | the τ = 40% rung is **unsurvivable** (drawdown worse than −60%) even though its compound return is the highest on the ladder | unsurvivable | — |
| **P13** | 2022-01→2026-06 (§7.4d) RP Sharpe is **negative** | **−0.30** | [−1.00, +0.30] |

**The prediction I most expect to be wrong, stated in advance:** P4. If risk parity's whole
mechanism is that it fixes equal weight's implicit over-allocation to volatile assets, the
vol-matched active return could well be significant — in which case the interesting finding
is that the *sizing*, not the signal, was the missing piece all along. Conversely if P9 comes
in near the bottom of its band, risk parity on this panel is a 40-year bond bull market
wearing a diversification coat, and should be recorded as such.

## 11. Discipline

One run. All parameters above are fixed. No window is moved, no instrument is added or
removed outside the single pre-declared rates-excluded variant, no leverage level is added,
no financing spread is changed after seeing a result. Secondary analyses in §7 are
non-promotable by construction. Trials spent: **2** (W1, W2) → cumulative **46**.

**Hard limits acknowledged:** research only; no live trading, no broker path, no account
action; no Sharadar row-level data is involved in this study at all (this panel is the
public-market long-history panel); nothing goes public; no financial advice. No number will
be reported that was not measured by the run.
