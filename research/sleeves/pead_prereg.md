# PRE-REGISTRATION — Sleeve: Post-Earnings-Announcement Drift (PEAD)

**Written 2026-07-28, BEFORE any result was computed.** One configuration, one run.
Three holding horizons are declared here and ALL THREE are reported, whatever they say.
No threshold, decile, universe rule, cost constant or horizon may be changed after a
number is seen. A second attempt requires a new pre-registration at a higher trial count.

Data: Sharadar SF1 dimension `ARQ` (444,572 filings inside the DEV window, 12,795
tickers) joined to the SEP daily bars. `datekey` is the SEC filing date and is the only
announcement proxy used.

---

## 1. Hypothesis

**H1 (headline).** Prices under-react to earnings surprises. A long-only book that buys
the top decile of standardised unexpected earnings (SUE) one day after the filing date
and holds it for a fixed horizon earns a positive return **in excess of an equal-weight
buy-and-hold of its own eligible universe**, net of per-name spread, impact and
commission.

**H2 (breadth, the reason this sleeve was chosen).** Because entries are triggered by
filings arriving continuously rather than by a rebalance grid, realised breadth is an
order of magnitude above the 4–12 bets/year of every prior study in this programme
(`docs/project-control/specs/2026-07-28-the-breadth-lever.md`). H2 is a measurement, not
a performance claim: **high breadth at zero IC is still zero.**

**Declared in advance:** H1 support is a necessary but not sufficient condition for
deployability. Excess return must be positive AND net Sharpe must clear 0.75 for this to
go near the promotion gate.

## 2. Point-in-time rule

- Surprise for fiscal quarter *q* becomes visible at `datekey(q)`, the SEC filing date.
- **Entry is at the CLOSE of the first trading day strictly after `datekey`.** Never
  `datekey` itself: a filing can land after the close, so acting on the filing date is a
  look-ahead. Entering at the *close* of t+1 additionally forgoes the whole
  announcement-window jump, which is the conservative direction.
- Every screening quantity (price, trading fraction, spread, dollar volume, volatility)
  is computed on the 63 trading bars **ending at the last bar on or before `datekey`**.
- Decile breakpoints are computed from filings already public: for an entry in calendar
  month *m*, the breakpoint is the 90th percentile of SUE over all filings with `datekey`
  inside the 12 calendar months ending at the close of month *m−1*.
- `load_prices` refuses any bar after 2015-12-31. The 2016+ confirmation window stays
  unfired.

## 3. The signal

Per ticker, ARQ rows sorted by `calendardate`; on a duplicated `calendardate` the row
with the **earliest** `datekey` is kept (the original filing, not a restatement).

    eps_q      = eps, falling back to netinc / shareswa where eps is null
    d_q        = eps_q(t) - eps_q(t-4)        # t-4 must be 330-400 calendar days back
    SUE_q      = d_q / stdev(d_{t-1} .. d_{t-8})

Requirements: at least 6 non-null prior seasonal differences; the trailing standard
deviation must be at least $0.01 (below that the denominator is noise, not dispersion,
and the top decile fills with division artefacts). Filings with `datekey - calendardate`
outside [0, 180] days are dropped as data errors.

SUE is scale-invariant per ticker (numerator and denominator carry the same units), so
Sharadar's retroactive split adjustment of per-share fields cancels exactly and cannot
manufacture a surprise.

## 4. Universe and artefact filters (fixed in advance)

Measured on the 63 bars ending at or before `datekey`:

1. Close >= $2.00.
2. Non-zero volume and a genuine high>low range on >= 90% of those 63 bars.
3. Median dollar volume >= $50,000 (the lowest band floor in `capacity_panel.BANDS`).
4. `spread_with_resolution` regime == `measured`. Regimes `upper_bound` and
   `unmeasurable` are **excluded from the universe**, never costed at the floor.
5. Realised holding-period returns capped at +/-100%.

## 5. Construction

Long-only. Capital $1,000,000. Maximum 0.5% of equity per position (f = 1/200); when
concurrent signals exceed available cash, that day's new positions are scaled down
pro-rata rather than skipped, so **no signal is ever dropped and realised breadth is not
silently truncated**. Idle cash earns 0% (conservative: T-bills paid ~2% over the window).

Horizons, all three pre-specified and all three reported: **20, 40 and 60 trading days.**
A position opened at the close of t0 exits at the close of t0+H.

## 6. Costs — mandatory, per name, both sides

    half_spread   = EDGE spread / 2                      (regime 'measured' only)
    impact        = daily_vol * sqrt(notional / median_dollar_volume)   (sqrt law, coef 1.0)
    commission    = min(max(0.0035 * shares, 0.35), 0.01 * notional)    (IBKR)

Charged on entry and on exit. The exit uses the entry-date spread estimate, which is
point-in-time safe.

## 7. Delistings

If a position's price series ends before its horizon, it is exited at the last available
bar. The terminal return is booked **only if the delisting date falls within 62 days
after that exit date**, and the position is then removed from the book. This is the exact
pair of defects recorded in `capacity_curve_result.md` §4; both are guarded by tests.

## 8. Benchmark

Equal-weight, monthly-rebalanced, **zero-cost** buy-and-hold of the sleeve's OWN eligible
universe: every (ticker, month-end) cell in `monthly_panel_dev.parquet` with
`spread_regime == 'measured'` whose ticker has SF1 ARQ coverage. Delisting terminal
returns are booked into the benchmark on the same 62-day rule, so the benchmark is not
survivorship-flattered. Costing the benchmark at zero makes it harder to beat, which is
the conservative direction for H1.

**The reported statistic is EXCESS over this benchmark.** A raw return that loses to
passive ownership of the same names is not an edge.

## 9. Breadth accounting

Reported: entry events per year, distinct entry DAYS per year, and mean concurrent
positions. Entry events are the headline breadth figure. They are **not** fully
independent — filings cluster in four earnings seasons and share market beta — so the
distinct-entry-days figure is reported alongside as a crude lower bound on independence.

## 10. Trial accounting

Cumulative n_trials before this study: 26 (`capacity_curve_result.md` §6).
This study spends **1** trial: one signal, one universe, one construction, three
pre-declared horizons reported jointly. Horizon is not selected on; if the three
disagree, that disagreement is the result.
