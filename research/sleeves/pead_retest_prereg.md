# PRE-REGISTRATION — PEAD RE-TEST on the corrected (unbiased) universe

**Written 2026-07-28, BEFORE any number from this configuration was computed.** One
configuration, one run, three pre-declared horizons, both cost bounds. Nothing here may be
changed after a result is seen. A second attempt requires a new pre-registration at a
higher trial count.

**Trial accounting.** Cumulative `n_trials` before this study: **32**
(`breadth_sleeve_hunt_result.md` §5). This study spends **1** trial → **n_trials = 33**.
At n=33 and a 17.7-year sample the DSR≥0.95 bar is ≈0.91 annual Sharpe; the promotion gate
is 0.75. Both are recorded here so neither can be moved later.

**Why this re-run is legitimate rather than tuning.** The three conditions in
`breadth_sleeve_hunt_result.md` §6 are met: (a) the cost estimator was repaired and
validated against ground truth (`scripts/spread_positive_control.py`, 4/4 pass, mega-cap
realistic bound 4.50bps/side inside the registered 1–5bps window) with **no strategy run
during its selection** — iteration 2 of the internal research log spent zero trials; (b) this is a
new pre-registration written before the re-run; (c) the falsification condition in §6 below
is stated in advance and will be reported whichever way it lands.

---

## 1. What changed from `pead_prereg.md`, and NOTHING else changed

Exactly two things change. Both are corrections to the measurement apparatus, not to the
strategy.

**Change 1 — the universe.** Iteration 1 §4.4 excluded every name whose EDGE spread regime
was `upper_bound`. That was the measured universe bias: `upper_bound` means the true spread
lies **below** the estimator's resolution floor, i.e. the name is CHEAP, and excluding those
names deleted 525,933 of 922,652 eligible (name, month) cells at 6.4× the dollar volume and
0.24× the spread of the ones kept (`scripts/measure_spread_universe_bias.py`). Under this
pre-registration `upper_bound` names are **admitted**. Only `unmeasurable` (too few genuine
trading days) is still excluded — the schedule prices cheap names, not absent ones.

**Change 2 — the cost is now a bracket, never a single number.** Every position is priced
twice, by `research.spread_estimation.spread_cost_bounds`:

* **(a) CONSERVATIVE** — charge the EDGE estimate itself. The truth is below it, so this
  OVERSTATES cost. A result that passes here is REAL.
* **(b) REALISTIC** — charge the Ardia-Guidotti-Kroencke Table 4 liquid-name schedule keyed
  on median dollar volume, era-scaled (floored at 1.0), capped at (a), floored at the legal
  minimum tick of the day. A result that fails here is DEAD.

`realistic ≤ conservative` holds by construction, so the bracket cannot invert.

**Everything else is byte-identical to `pead_prereg.md`**: the SUE definition, the
seasonal-gap and filing-lag sanity bands, the $0.01 denominator floor, the point-in-time
12-month decile breakpoint, the $2 price floor, the 90% trading-fraction test, the $50k/day
dollar-volume floor, the ±100% return cap, $1,000,000 capital, the 0.5%-of-equity position
cap, the pro-rata scale-down, 0% on idle cash, the IBKR commission schedule, the
square-root impact term with coefficient 1.0, the 62-day delisting rule, and the horizons
20/40/60. **The impact coefficient is NOT touched**, even though the internal research log
iteration 2 records it as probably an order of magnitude too high — correcting it in the
same run that corrects the spread would make the result uninterpretable, and it would point
the flattering way.

## 2. Hypothesis, stated as a falsifiable number BEFORE the run

**H1.** On the corrected universe, a long-only book buying the top SUE decile at the close
of `datekey + 1` and holding 40 trading days earns a **positive excess return over an
equal-weight zero-cost buy-and-hold of its own (corrected) universe**, and a net annual
Sharpe of **0.70–0.80** under bound (b).

That number is not a guess; it is iteration 1's arithmetic
(`breadth_sleeve_hunt_result.md` §6.3) carried forward: measured round-trip cost 219.1bps
against 256.0bps of gross alpha per bet (cover 1.17); if the traded-universe median spread
falls from 126bps to 5–20bps the round-trip bill falls to ≈50–90bps, net alpha per bet goes
from +37bps to ≈+170–200bps and net Sharpe from 0.342 toward 0.7–0.8.

**H2 (auxiliary, reported either way).** The median spread of the TRADED names falls from
126.4bps (iteration 1) to below 60bps under bound (b).

**The way H1 most plausibly fails, stated in advance so it cannot be explained away
afterwards: the cost falls but the gross alpha falls at least as much.** PEAD is documented
to be weakest in the most liquid, most-followed names, which are precisely the names this
correction admits. If gross alpha per bet drops from 256bps toward the new (lower) cost, the
cover ratio does not improve and the sleeve is dead on a *signal* ground rather than a cost
ground. That outcome is a real result and will be reported as such, not re-cut by
re-introducing a liquidity filter.

## 3. Entry rule and point-in-time discipline (unchanged, restated so it is auditable)

- Signal: SF1 dimension `ARQ`. `eps`, falling back to `netinc / shareswa`.
  `SUE = (eps_q − eps_{q−4}) / stdev(prior 8 seasonal differences)`, ≥6 non-null priors,
  denominator ≥ $0.01, seasonal gap in [330, 400] days, `datekey − calendardate` in
  [0, 180] days, duplicate `calendardate` resolved to the EARLIEST `datekey` (original
  filing, not restatement).
- **Entry at the CLOSE of the first trading bar strictly after `datekey`. NEVER `datekey`
  itself** — a filing can be accepted after the close, so acting on the filing date is a
  look-ahead. Entering at the close of t+1 also forgoes the announcement jump, which is the
  conservative direction.
- Every screening quantity (price, trading fraction, median dollar volume, spread bounds,
  volatility) is computed on the 63 bars **ending at the last bar on or before `datekey`**.
- Decile breakpoint applied in month *m* = the 90th percentile of SUE over all filings with
  `datekey` in the 12 months ending at the close of month *m−1*.
- `load_prices` refuses any bar after 2015-12-31. **The 2016+ confirmation window stays
  unfired.**
- Era factor and tick regime for the cost schedule are keyed on the **entry date** — the
  date the cost is actually paid.

## 4. Universe (the one substantive change)

Measured on the 63 bars ending at or before `datekey`:

1. Close ≥ $2.00.
2. Genuine `high > low` range and non-zero volume on ≥ 90% of those 63 bars.
3. Median dollar volume ≥ $50,000.
4. **Spread regime ∈ {`measured`, `upper_bound`}. Only `unmeasurable` is excluded.**
5. Realised holding-period returns capped at ±100%, applied by rescaling the whole daily
   path so the marked-to-market curve compounds to the capped figure.

## 5. Costs — per name, both sides, both bounds, and the order minimum

    half_spread_(a) = conservative_bound / 2
    half_spread_(b) = realistic_bound  / 2
    impact          = daily_vol * sqrt(notional / median_dollar_volume)    (coef 1.0)
    commission      = min(max(0.0035 * shares, 0.35), 0.01 * notional)     (IBKR)

Charged on entry and on exit; the exit uses the entry-date spread bounds, which is
point-in-time safe. The two bounds differ ONLY in the spread term — impact and commission
are identical — so the difference between the two runs is attributable to the spread and
nothing else.

**POSITION SIZE, stated because the $0.35 order minimum is what makes small accounts
different.** Capital $1,000,000; maximum 0.5% of equity per position ⇒ **$5,000 per
position** at full size, less when concurrent signals exceed available cash. At $5,000 the
$0.35 minimum is **0.70bps per side, 1.40bps round trip** — negligible. The minimum binds
whenever a ticket is under 100 shares, i.e. under `$100 × price` of notional; at a $30 share
that is $3,000. **The report must state: the realised mean and median ticket, the fraction
of orders on which the $0.35 floor bound, and the commission in bps at that ticket, plus the
account size at which the floor would start to bind for the median ticket.** A $10,000
account running the same 0.5% rule trades $50 tickets and pays 70bps per side in commission
alone — that is a different strategy and this study does not claim anything about it.

## 6. Decision rule — FIXED BEFORE THE RUN

**Headline horizon is 40 days**, declared now, because that is where iteration 1 measured
cover 1.17. All three horizons are reported. If 20d or 60d does better, that is reported as
an observation and **does not change the headline verdict** — horizon is not selected on.

Primary gate **G** at the headline horizon:

    G  ≡  (excess over own universe > 0)  AND  (net annual Sharpe ≥ 0.75)

| verdict | condition |
|---|---|
| **PROMISING** | G passes under bound (a) CONSERVATIVE |
| **UNDETERMINED** | G passes under (b) REALISTIC but not under (a) |
| **MARGINAL** | under (b): excess > 0 but net Sharpe < 0.75 |
| **DEAD** | under (b): excess ≤ 0 |

**PROMISING is not a promotion.** Promotion additionally requires clearing the DSR bar of
≈0.91 for a 17.7-year sample at n=33, and the 2016+ confirmation window may only be fired at
a model that has already passed the DEV gate.

**Falsification of H1 is recorded explicitly**: if the verdict is MARGINAL or DEAD, the
0.70–0.80 prediction in §2 is FALSIFIED and must be reported as such. If bound (b)'s traded
median spread comes back above 60bps, H2 is falsified too and the whole cost-bias
explanation of iteration 1 is weaker than claimed.

## 7. Benchmark

Equal-weight, monthly-rebalanced, **zero-cost** buy-and-hold of the sleeve's OWN corrected
universe: every (ticker, month-end) cell in `monthly_panel_dev.parquet` whose
`spread_regime ∈ {measured, upper_bound}` and whose ticker has SF1 ARQ coverage — i.e. the
identical eligibility test the strategy uses. Delisting terminal returns booked on the same
62-day rule, so the benchmark carries the same survivorship treatment. Costing the benchmark
at zero makes it harder to beat, which is the conservative direction.

**The benchmark MOVES in this study and that is the point.** Admitting the liquid half of
the tape changes what "the same names" means, so the iteration-1 benchmark return is not
comparable and the excess must be recomputed, not carried over.

## 8. Delistings and holdings hygiene (both prior defects guarded)

If a position's price series ends before its horizon it exits at the last available bar. The
terminal return is booked **only if the delisting date falls within 62 days after that exit
date**, and the position is then **removed from the book** so it can never be marked again.
Counts of delisted and truncated positions, and the number of positions still open at the
end of the sample, are reported. These are the two defects that produced −60%/yr and
−112%/yr previously.

## 9. Mandatory reported diagnostics

Reported whatever they say:

1. **Cover ratio per hold** = gross alpha per bet (bps) ÷ cost per round trip (bps), under
   BOTH bounds, for 20/40/60.
2. **Sharpe per decade** on net monthly returns, under both bounds, with month counts, for
   the calendar decades the sample spans (1990s partial, 2000s, 2010s partial). A decade
   with fewer than 24 months is labelled as such and not read as evidence.
3. **P&L concentration.** Every position's realised dollar P&L is attributed to
   (ticker, calendar month of exit). **If any single name-month exceeds 3% of total net
   P&L, that is stated loudly at the top of the result and the headline is treated as
   concentration-driven.** The top 10 name-months are listed.
4. Breadth: entries/yr, distinct entry days/yr, mean concurrent positions, bets per unit of
   annual turnover.
5. Traded-universe character: median spread under both bounds, median dollar volume,
   share of entries that are newly-admitted `upper_bound` names.
6. Ticket-size statistics per §5.
7. The daily net return series is persisted to parquet for both bounds, so cross-sleeve
   correlation is computable later without re-running (the process defect recorded in
   `breadth_sleeve_hunt_result.md` §4).

**Non-registered diagnostic, cannot change the verdict:** the BOTTOM SUE decile at the 40d
hold, to test whether SUE orders returns at all on the corrected universe. Labelled
DIAGNOSTIC everywhere it appears.

## 10. Data hygiene

DEV window only; no bar after 2015-12-31 is read. No raw Sharadar row is written outside
`_data/`. Only derived statistics are committed. No live trading path is touched.
