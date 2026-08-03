# PRE-REGISTRATION — LOW-VOLATILITY / QUALITY, RE-TESTED ON THE CORRECTED COST MODEL

**Written and committed BEFORE the run. One configuration. Run once. No tuning.**
If it fails, the failure is the result and it is reported with its numbers.

Registered 2026-07-28 (unattended run). Window: DEV only, no bar after **2015-12-31**.
Cumulative `n_trials` **36 → 37** (a concurrent seasonality study may take it to 38; the
gate below is computed at **38**, the stricter of the two).

---

## 1. Why this is being run at all

Sixteen studies, zero deployable. Low-vol/quality is the only sleeve that has ever reached
positive excess, and it has **never been run with the two cost corrections that have since
landed**. Iteration 1 measured it as DEAD (net Sharpe 0.324, excess −5.54%/yr in B2), and
that measurement was taken under two defects that have both been fixed and validated
against positive controls since:

1. **Universe bias** (iteration 2a, `f5b9dfc`). The run kept only
   `spread_regime == "measured"` and deleted every `upper_bound` name. `upper_bound` means
   the true spread lies BELOW the estimator's resolution floor — the name is **cheap**, not
   unknown. 525,933 of 922,652 eligible DEV cells were discarded, carrying 6.4× the dollar
   volume at 0.24× the spread of the cells kept.
2. **Impact overcharge** (iteration 4, `32aaa78`). `IMPACT_COEFFICIENT = 0.1` with
   `impact = 0.1·√participation` charged a flat **100bps/side** at the registered 1%
   participation cap, with no volatility term at all — 17.9× FIM (2018)'s measured all-in
   of 5.54bps, and identical for a placid mega-cap and a wild micro-cap.

Iteration 4 **re-priced** the iteration-1 book arithmetically (cost 119.5 → 59.6/49.9bps
one-way; net Sharpe 0.324 → 0.715/0.779; excess −5.54%/yr → −0.13%/+0.75%). **That was
arithmetic, not a run.** The books were never re-run, the universe bias was still baked
into which names were held, and the impact model was still fed a reference volatility
rather than each name's own. This document registers the real run.

## 2. What is UNCHANGED from iteration 1 (this is not a new hypothesis)

The signal is **not** redesigned. Verbatim from `research/sleeves/low_vol_quality.py`:

- **Legs**, 2%-winsorised and z-scored within (band, month), equal-weight composite, all
  three required: `z(−realised_vol_252)`, `z(−beta_252)` vs an equal-weight market proxy,
  and quality = mean of `z(gp/assets)`, `z(−debt/equity)`, `z(−accruals)` (SF1 **ART**, ≥2
  of 3 required).
- **Construction**: long-only, top **30** by composite, **equal weight**, **monthly**
  rebalance, no no-trade band, no rebalance in a month with <60 rankable names.
- **Point-in-time**: SF1 attached by `datekey`, never `calendardate`. Risk features use
  only bars up to and including the month-end.
- **Artefact filters**: price floor **$2**, `high>low` and `volume>0` on ≥90% of the
  trailing 63 days (both from the panel), ≥200 valid daily returns in the 252-day window,
  <50% of them exactly zero, realised vol strictly positive, returns clipped to **±100%**
  for strategy and benchmark alike.
- **Delistings**: terminal return applied only if the delisting date falls in
  `(exit_date, exit_date + 62 days]`, and the name is **removed from holdings** the moment
  its exit is booked.
- **Capital**: `deployable = 30 × 1% × median dollar volume of the band`;
  position value = `deployable / 30`. Run separately in **each** band, so the capacity
  curve is read directly.

## 3. What CHANGES, and only this

1. **Universe.** Cells with `spread_regime ∈ {measured, upper_bound}` are all eligible.
   Only `unmeasurable` and `ineligible` stay out. This roughly doubles the cross-section
   (B2: 114,095 → 200,088 cells).
2. **Spread cost.** `research.spread_estimation.bounds_from_estimate` (the vectorised
   entry point of `spread_cost_bounds` for a pre-computed panel) gives **both** bounds per
   name-month. Every headline is reported twice — conservative and realistic — and the
   verdict uses `bracket_verdict`.
3. **Impact cost.** `research.capacity_study.impact_cost_bounds`, fed **the name's own
   daily volatility** (the 252-day realised vol the signal already computes) rather than
   the reference fallback. Both bounds.
4. **Quality coverage.** The SF1 quality cache is rebuilt on the expanded grid
   (`research/sleeves/lowvol_retest_data.py`). Re-using the old measured-only cache would
   make every added name un-rankable and silently reinstate the bias being removed.
5. **Reporting.** The mandatory tests in §5 below, none of which iteration 1 ran.

Nothing else. No band edges, horizons, weights, filters or holding counts will be adjusted
after seeing a number.

## 4. FORECASTS — recorded before the run

A prior prereg forecast the strategy without forecasting its benchmark and was badly wrong
as a result. Both sides are forecast here. Central estimate first, then the range I would
not be surprised by. **Band B2 ($200k–$1M/day)** is the one being forecast, because it is
the band iteration 1 came closest in; the others are expected to be worse.

| quantity | central forecast | range |
|---|---:|---|
| **Benchmark** arithmetic return (equal-weight buy-and-hold, corrected universe) | **9.5%/yr** | 8.0% – 11.5% |
| **Benchmark** volatility | **21%** | 19% – 23% |
| **Benchmark** Sharpe | **0.45** | 0.37 – 0.55 |
| **Strategy** net Sharpe, CONSERVATIVE bound | **0.74** | 0.55 – 0.95 |
| **Strategy** net Sharpe, REALISTIC bound | **0.79** | 0.60 – 1.00 |
| Strategy net volatility | **13.5%** | 12% – 16% |
| One-way cost, conservative bound | **~40bps** | 30 – 60bps |
| **Raw arithmetic excess** (net − benchmark) | **+0.5%/yr** | −3% – +3% |
| **Geometric excess** (net CAGR − benchmark CAGR) | **+2.5%/yr** | 0% – +5% |
| **VOL-MATCHED active return** (the deciding number) | **+3.9%/yr** | −1% – +7% |
| Vol-matched active NW t-stat | **+1.6** | +0.5 – +2.6 |
| Strategy DSR at n_trials 38 | **0.55** | 0.2 – 0.9 |

**Reasoning behind the forecast, so it can be judged rather than admired.** The
iteration-1 B2 book paid 119.4bps one-way (46.4 spread + 70.8 impact + 2.3 commission) on
9.06 turnovers/yr = 10.82%/yr of cost drag against 15.3%/yr gross. The impact fix alone
takes 70.8 → ~8.5bps at the reference volatility under the conservative bound; feeding the
holdings' **own** volatility (33% annualised vs the universe's 56%, i.e. ~2.08%/day vs the
3.35%/day fallback) takes it to ~5.3bps. The universe correction should cut the spread leg
too, because the added names are cheap by construction — call it 46.4 → ~33bps. That is
~40bps one-way, ~3.6%/yr of drag, and a net return near 10.4%/yr on gross around 14%
(gross forecast slightly BELOW iteration 1's 15.3% because the added names are the more
liquid ones and should carry less illiquidity premium).

**The benchmark is forecast slightly DOWN** from iteration 1's 10.04%/yr for the same
reason: the corrected universe adds more liquid, less volatile names to an equal-weight
average.

**Why the vol-matched number is forecast so much higher than the raw excess, and why that
is a warning rather than a result.** `vol_matched_active` scales the benchmark by
`k = σ_strategy / σ_benchmark`, so its mean is exactly `(Sharpe_s − Sharpe_b) × σ_s`. At
σ_s ≈ 13.5% and σ_b ≈ 21%, k ≈ 0.64: the benchmark is **de-levered**, and the strategy
gets credit for running quiet. That is the correct comparison — it is the only one
invariant to leverage — but it means **this sleeve's headline is structurally flattered by
the same mechanism that killed PEAD, only pointing the other way.** Both the raw and the
vol-matched numbers are therefore reported side by side, and the sleeve is judged on
whether its **Sharpe** beats the benchmark's Sharpe, which is the same statement without
the units.

**I expect this to FAIL the DSR gate even if it passes the excess gate.** The bar at 17.75
years and n_trials 38 is **0.9234**; the forecast net Sharpe is 0.74. A pass on excess and
a fail on DSR is the single most likely outcome and is registered here as such.

## 5. MANDATORY REPORTS — every one of these killed a prior result

1. **Both spread bounds and both impact bounds**, `upper_bound` names INCLUDED.
2. **Matched-volatility comparison.** Benchmark = equal-weight buy-and-hold of the SAME
   band, levered to the strategy's own full-sample volatility. Report geometric excess,
   arithmetic active return, and vol-matched active return, each with a Newey–West t-stat.
3. **The benchmark goes through the DSR gate too**, and both numbers are reported. (DSR
   passed trend at 0.612 while passive scored 0.669 — the gate does not by itself say a
   strategy is better than doing nothing.)
4. **Sharpe per decade.**
5. **Delisting returns by date, holdings removed after booking the exit.**
6. **±100% return cap, $2 price floor, volume filter.**
7. **P&L concentration** (largest single name-month share of total P&L, and top-10 share)
   **and gross-notional concentration** (largest share of gross notional in one name, and
   in the top 3). Equal weighting makes the notional test near-trivial by construction and
   it is reported anyway, because inverse-vol sizing put 65% of gross notional into 3
   instruments in another sleeve and the check is cheap.
8. **Capacity curve**: every band, reported.

## 6. DECISION RULE — pre-committed, in this order

Evaluated on the **CONSERVATIVE** bound (spread conservative + impact conservative). The
realistic bound is reported alongside and is used only to separate DEAD from UNDETERMINED.

- **PROMOTE** — a route worth taking further. Requires **all** of:
  (i) vol-matched active return **> +2.0%/yr**;
  (ii) its Newey–West t-stat **> 2.0**;
  (iii) net Sharpe **≥ 0.9234** (the DSR≥0.95 bar at this sample length, n_trials 38);
  (iv) strategy DSR **>** benchmark DSR.
- **MARGINAL** — (i) and (ii) hold but (iii) or (iv) fails.
- **UNDETERMINED** — (i) fails on the conservative bound but holds on the realistic one
  (`bracket_verdict`).
- **DEAD** — vol-matched active return **≤ 0** in every band under the **realistic**
  bound, i.e. it fails even where the cost model is generous to it.

A positive raw geometric excess with a non-positive vol-matched active return is **DEAD**,
not marginal. That trap is the specific reason this rule is written down in advance.

## 7. Pre-committed threats to the result

Named now so they cannot be discovered afterwards and explained away:

- **The forced-exit share was 70% in iteration 1.** Most selling was caused by a name
  ceasing to be rankable, not by its signal rank falling. If that share stays high, the
  measured turnover is a property of the measurement panel and the cost estimate inherits
  it. It will be reported.
- **The corrected universe changes the benchmark as well as the strategy.** The comparison
  is only honest if both are recomputed on the same universe, which they are — but it means
  the excess is not comparable, name for name, to iteration 1's.
- **A holding that leaves the tradable universe** is exited by the strategy and may be
  charged a terminal return, while the benchmark simply stops including it. This runs
  AGAINST the strategy and is left uncorrected, as in iteration 1.
- **17.75 years is one sample.** The 2016+ confirmation window stays unfired regardless of
  the outcome.
