# PRE-REGISTRATION — Short-horizon reversal, RE-TESTED on the corrected universe

**Written 2026-07-28, BEFORE any number in this study was computed.** One registered
configuration, one run. Two universe cuts, three rebalance frequencies and two cost
bounds are all declared here and ALL are reported, whatever they say. Nothing below may
be changed after a result is seen; a second attempt requires a new pre-registration at a
higher trial count.

Supersedes nothing: `research/sleeves/short_horizon_reversal.py` (iteration 1) stands as
the historical record and is not edited. This is a **new sleeve module** with a **new
universe**, so it is a new trial, not a re-cut of the old one.

---

## 0. Why this sleeve is being re-run at all

Iteration 1 gave short-horizon reversal the study's **worst** net Sharpe (−3.62) at the
study's **highest** frequency-driven breadth (577 bets/yr, 45.3× annual turnover). Its
cost drag was **71.81%/yr**, of which **62.10 points were spread** — 86% of the bill.

Iteration 2 then measured and repaired a bias in the cost model itself
(the internal research log, iteration 2): the rule "exclude `upper_bound`"
deleted **525,933 of 922,652** eligible (name, month) cells, and the deleted half had
**6.4× the dollar volume** and **0.24× the spread** of the half that was kept. A
45×-turnover strategy is the construction that rule punished hardest, because its bill is
linear in the spread it was mispriced at.

**That makes this sleeve the sharpest available test of the repair.** It is the one sleeve
where the correction is worth tens of percent a year, and it is the one sleeve iteration 1
claimed could not be rescued by any cost fix. Exactly one of those two statements can
survive this run.

Iteration 1's counter-claim, quoted so this run can falsify it:

> "With spreads set to zero it still nets −2.96%/yr; with spread, impact *and* borrow all
> zero — IBKR commissions alone, a physically impossible cost model — its net Sharpe caps
> at **0.41**, below the 0.75 promotion gate. No cost-model improvement can rescue a
> 45×-turnover sleeve."

That ceiling was measured **on iteration 1's universe** — the expensive, illiquid,
`measured`-only tail. This study runs a **different universe**, so the 0.41 ceiling does
not transfer, and re-establishing the zero-cost ceiling on the new universe is a mandatory
reported diagnostic (§9.7).

---

## 1. Hypotheses

**H1 (headline).** On the most-liquid cut of the US cross-section, priced under the
corrected two-bound cost model, a weekly-rebalanced long/short book formed on the negative
trailing 5-day return earns a **positive net-of-cost excess** over an equal-weight
buy-and-hold of **its own universe**, and a net Sharpe at or above the **0.75** promotion
gate.

**H2 (the frequency optimum — the reason this sleeve is worth a trial even if H1 fails).**
Cost scales **linearly** with rebalance frequency *f* while gross Sharpe scales with
**√breadth ∝ √f**, so net Sharpe as a function of *f* has an interior maximum at

    f* = ( sigma * IC * sqrt(N) / (2c) )^2

where *c* is the cost per round trip. H2 is falsified if the measured curve is **monotone**
over {12, 26, 52} — which is what happens when realised gross annual return grows
**linearly** in *f* rather than as √*f*, i.e. when per-round-trip alpha is roughly constant
in the holding period instead of Grinold-constant in IC. **The curve is reported in full
and is the deliverable even if every point on it is negative.**

**Declared in advance:** H1 support requires positive excess AND net Sharpe ≥ 0.75. High
breadth at zero IC is still zero.

---

## 2. THE UNIVERSE — liquidity-first, and it INCLUDES the cheap liquid names

This is the single change from iteration 1 that this study exists to test.

Eligibility of a (name, month) cell in `monthly_panel_dev.parquet`:

1. `spread_regime ∈ {measured, upper_bound}`. **`upper_bound` is now INCLUDED** — it means
   the true spread lies BELOW the estimate, i.e. the name is CHEAP, and deleting those
   names was the iteration-1 bias. Only `unmeasurable` is still excluded: the schedule
   prices cheap names, not absent ones.
2. `close ≥ $2.00`.
3. `trading_fraction ≥ 0.90` (non-zero volume and a genuine high>low range on ≥90% of the
   trailing 63 bars).
4. `spread` finite.

**Then the liquidity-first restriction, which is the point:** within each calendar month's
eligible cross-section, rank on `median_dollar_volume` (trailing 63-day median) and keep

* **PRIMARY — top DECILE** (rank percentile > 0.90). ≈ 433 names/month, 1,763 distinct
  tickers over the window.
* **SECONDARY — top QUINTILE** (rank percentile > 0.80). ≈ 866 names/month, 3,438 tickers.

Both are pre-registered, both are reported, **neither is selected**. They are declared
together because they trade breadth against cost in opposite directions and the pair is
informative in a way either alone is not. The PRIMARY is the governing headline.

The rank is **cross-sectional within the month**, not an absolute dollar threshold, so the
universe is not a secular time series of the market's growth — in 1998 the decile cut sits
near $17M/day and in 2015 near $96M/day. An absolute threshold would have made the sleeve
a 1998-weighted small-cap study and a 2015-weighted mega-cap study.

**Point-in-time:** the monthly row used at signal date *t* is the **previous calendar
month's**, never the current one — each ticker's monthly row is stamped with its own last
trading day of that month, so the current month is not knowable until it is over. Costs up
to eight weeks of staleness in liquidity and spread; the conservative direction.

**Registered in advance, because it is not neutral:** 74.2% of the top-decile cells carry
regime `upper_bound`. Iteration 1 would have deleted three quarters of this universe. That
is why this is a genuinely different universe and a genuinely new trial.

---

## 3. The signal

    ret5_t   = adj_close_t / adj_close_{t-5} - 1
    signal_t = -ret5_t                      # negative trailing 5-day return
    z_t      = (signal_t - mean_x(signal_t)) / sd_x(signal_t)   # cross-sectional, per date

Adjusted closes only (Sharadar `closeadj`; the open is put on the same basis by the
`closeadj/close` factor), so a split cannot manufacture a 50% reversal.

**The z-score is registered as a diagnostic, not a selector.** A decile sort is
rank-invariant, so cross-sectional z-scoring cannot change which names are picked. It is
computed and reported (mean/sd of the cross-section, and the IC is measured on it) so that
the registered spec is honoured literally and so a future weighted variant has the
quantity it needs. Stated here so nobody later reads the z-score as a free parameter.

---

## 4. Construction

* **Long leg:** top decile of `signal` within the universe. Equal-weighted.
* **Short leg:** bottom decile of `signal` among universe names with
  `median_dollar_volume ≥ $25,000,000/day` — the registered borrow-plausibility floor.
  Below that a short leg is a fiction.
* Legs are capped at **100 names** and require at least **20**; the shortable subset must
  contain at least **60** names for a short decile to be formed at all.
* Exposure **100% long / 100% short** (Reg-T retail maximum for a market-neutral book).
* **When no short leg can be formed the long/short book goes FLAT that period and pays the
  cost of getting flat.** It never silently becomes a long-only book — that would smuggle
  market beta into a "market-neutral" number. The long-only book is reported separately,
  in full, as the registered fallback where borrow is implausible.

**Execution.** Signal from closes through *t*; **execution at the OPEN of t+1.** Never the
close the signal was computed from. Short-horizon reversal is the construction most
exposed to that shortcut and it is the standard way this effect is manufactured.

**Turnover** is measured against the weights the previous period **drifted** to, not
against the previous target: a name that stays and merely drifts is not re-bought.

---

## 5. THE FREQUENCY CURVE — three grids, all reported

| grid | signal dates | nominal f |
|---|---|---:|
| **weekly** | last trading day of each ISO week | 52 |
| **fortnightly** | every 2nd weekly signal date | 26 |
| **monthly** | last trading day of each calendar month | 12 |

The signal is **identical** at every frequency (trailing 5-day return). Only the holding
period changes. Annualisation uses the **realised** periods-per-year, `n_periods /
(calendar span in years)`, not the nominal number, so a ragged grid cannot inflate a Sharpe.

All three are run under both universe cuts and both cost bounds. **No frequency is
selected**; the curve is the reported object and its argmax is a measurement.

---

## 6. COSTS — both bounds, per name, never flat

One-way cost = `spread/2 + impact + commission`; the short book additionally pays borrow.

    half_spread  = spread_bound / 2
    impact       = 1.0 * sigma_daily_21d * sqrt(notional / median_dollar_volume)
    commission   = min( max(0.0035 * shares, 0.35), 0.01 * notional )     # IBKR
    borrow       = 100 bps/yr flat on short notional, NO credit on short proceeds

`spread_bound` is taken from `research.spread_estimation.bounds_from_estimate`, per
(name, month), and **the entire backtest is run twice**:

* **(a) CONSERVATIVE** — charges the EDGE estimate itself (floored at the minimum legal
  tick). The truth is below it, so this **overstates** cost. **A result that passes here is
  REAL.**
* **(b) REALISTIC** — charges the Ardia-Guidotti-Kroencke Table 4 documented liquid-name
  schedule keyed on median dollar volume, era-scaled (factor floored at 1.0), capped at
  (a), floored at the minimum legal tick ($0.125 → $0.0625 from 1997-06-24 → $0.01 from
  2001-04-09). **A result that fails here is DEAD.**

Between them the verdict is **UNDETERMINED** (`spread_estimation.bracket_verdict`).
`realistic ≤ conservative` holds by construction, so a cheaper (b) can only ever move a
verdict from dead to undetermined, never to real.

**The pre-decimalisation tick floor is load-bearing here and points the expensive way.**
Half this window predates decimalisation; in 1999 a $20 stock could not trade inside 31bps
however liquid it was, and a 45×-turnover book pays that 45 times a year.

**Registered vectorisation check:** the two bound matrices are built with a vectorised
reimplementation of `bounds_from_estimate` for speed over ~185,000 cells. It is asserted
cell-for-cell against the reference scalar function on a fixed-seed random sample of 4,000
cells **before any return is computed**, and the run aborts on any mismatch.

**Robustness ladder** (reported in full, no rung selected): net Sharpe at 0.5×, 1×, 1.5×,
2×, 3× the estimated cost. The 1× rung is the registered answer.

---

## 7. Artefact and accounting rules (mandatory, identical to the PEAD prereg)

1. **Price floor** $2.00, re-asserted at execution on the raw open.
2. **Return cap** ±100% per name per holding period. A prior study booked +9,900% on a
   zero-volume bankrupt shell that was 13% of its P&L.
3. **Delisting by date, and removal from holdings.** A terminal return is booked only if
   the delisting date falls inside the position's own holding window extended by **62
   days**, and only for names that actually stopped printing prices. The book is re-formed
   from the current universe every period, so a terminal name cannot be re-booked. These
   two defects together once produced −112%/yr.
4. **Exits are costed.** The spread and dollar-volume matrices used to *price* a trade are
   forward-filled; the matrices used to decide *eligibility* are not. A name that leaves
   the universe must still be sold, and reading the current month's NaN there would hand
   the strategy a free liquidation.
5. **A traded name with no priceable spread on any prior month raises**, and never
   defaults to zero.
6. **DEV window only.** `load_prices` refuses bars after 2015-12-31 and this study never
   asks it to. The 2016+ confirmation window stays UNFIRED.

---

## 8. Benchmark

Equal-weight buy-and-hold of **this sleeve's own universe**, on the **same** rebalance
grid, **gross of costs** — which makes it harder to beat, the conservative direction.
Not the S&P.

**The reported statistic is EXCESS over this benchmark.** A positive raw return with a
negative excess is not an edge.

---

## 9. Mandatory reported diagnostics

1. **Both cost bounds** for every headline number, and the `bracket_verdict`.
2. **Excess over own universe**, per book, per frequency, per bound.
3. **Cover ratio** = gross alpha per round trip ÷ cost per round trip. Iteration 1 measured
   0.10 for this sleeve (15.3bp vs 158.6bp).
4. **Per-decade / per-era Sharpe.** Two splits, both registered in advance:
   halves **1998–2006** and **2007–2015**; and the four iteration-1 eras **1998–2001**,
   **2002–2007**, **2008–2011**, **2012–2015**. Iteration 1 found this sleeve's gross edge
   concentrated in 2008–2011 (+24.6%/yr there against −1.1%/yr in 1998–2002); if that
   recurs, the sleeve is a crisis trade, not an edge, and must be reported as one.
5. **P&L concentration.** Largest single (ticker) share and largest single (ticker, period)
   share of total gross P&L. A single name-month was once 13% of a study's P&L.
6. **Breadth**, three ways: rebalances/yr, names traded/rebalance, naive bets/yr, and the
   Grinold inversion BR = (IR/IC)² — the last flagged as **circular** and not evidence for
   the law, per `breadth_sleeve_hunt_result.md` §3.2.
7. **The zero-cost ceiling on the NEW universe** — net Sharpe with spread, impact and
   borrow all set to zero (commissions only). This re-establishes on the corrected universe
   the number iteration 1 used to declare the sleeve unrescuable (0.41).
8. **IC** (rank, full universe, per date) with its t-statistic and hit rate.
9. **Daily/periodic net return series persisted to disk** for every reported configuration,
   so cross-sleeve correlation is computable later without re-running everything. Recorded
   as a process defect in `breadth_sleeve_hunt_result.md` §4 that five of six sleeves
   failed this.
10. **Cost decomposition** — spread / impact / commission / borrow, annualised.

---

## 10. Falsifiable predictions, stated BEFORE the run

| # | prediction | falsified if |
|---|---|---|
| **P1** | Cost per round trip falls from iteration 1's **158.6bps** to **20–50bps** under bound (b) and **50–90bps** under bound (a). | measured outside those ranges |
| **P2** | Gross alpha per round trip in the top-liquidity decile at weekly frequency is **at or below** iteration 1's 15.3bps — liquid names are more efficient, and reversal is documented as competed away post-decimalisation. | measured above 15.3bps |
| **P3** | Weekly net Sharpe under bound (b) improves from **−3.62** to between **−1.5 and 0.0**, with excess still negative. | outside that band |
| **P4** | The frequency curve is **monotone increasing as f falls**, i.e. the argmax over {52, 26, 12} is at **monthly** — a corner, not the interior optimum H2 predicts — because per-round-trip alpha will prove roughly constant in the holding period rather than Grinold-constant in IC. | argmax at 52 or 26 |
| **P5** | The sleeve **does not** clear the 0.75 net Sharpe gate under **either** bound at **any** frequency. Verdict: **DEAD**. | any registered configuration clears 0.75 with positive excess |

**If P5 fails under bound (a) the result is REAL and must be promoted, not re-cut. If it
fails only under bound (b) the verdict is UNDETERMINED and must be reported as such.**
If the corrected cost model leaves the sleeve dead, that kills the "our cost model was the
problem" defence for every frequency-driven sleeve in the programme — which is a result
worth a trial on its own, and it is the outcome this pre-registration expects.

---

## 11. Trial accounting

Cumulative `n_trials` before this study: **32** (`breadth_sleeve_hunt_result.md` §5).
This study spends **1** trial: one signal, one construction, two pre-declared universe
cuts × three pre-declared frequencies × two mandatory cost bounds, all reported jointly and
none selected. Cumulative becomes **33** (**34+** if other sleeves are re-registered in the
same iteration; the DSR bar is reported at n=34 so the count cannot flatter the result).

At n=34 and a ~17.7-year sample the DSR ≥ 0.95 bar is ≈ **0.91** standalone annual Sharpe.
It is computed exactly, from `research/validation.py`, in the result document.
