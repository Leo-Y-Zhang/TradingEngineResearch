# PRE-REGISTRATION — Sleeve: Institutional Ownership Flow (SF3)

**Written 2026-07-28, BEFORE any result was computed.** One configuration, one run.
If it fails, the failure is the result. No band, horizon or threshold may be changed
after seeing a number; a second attempt requires a new pre-registration at a higher
trial count.

Data source: Sharadar SF3 (Core US Institutional Investors), 79,190,744 rows, never read
by any code in this repository before today.

---

## 1. Hypothesis

**H1 (headline).** The quarter-on-quarter CHANGE in institutional ownership, measured as
a fraction of shares outstanding and cross-sectionally z-scored, predicts subsequent
returns positively. Accumulation by 13F filers is informed, and it is persistent enough
that the *next* quarter's return is still positive after the filing becomes public.

**H2 (mechanism, reported separately and NOT gate-eligible).** The effect concentrates in
the LOWEST tercile of institutional ownership level — limits to arbitrage. If H1 holds
only where ownership is already high, the mechanism claimed for it is wrong even if the
headline number is positive.

## 2. The point-in-time rule, stated first because it decides the study

SF3 has **no filing date**. It carries `calendardate`, the quarter end the holdings are
as of. Form 13F-HR is due **45 calendar days after quarter end**. A study that joins SF3
on `calendardate` is reading holdings six weeks before they were public and is worthless.

    availability_date = calendardate + 45 days

Every join in this sleeve is on `availability_date`, computed once in
`scripts/build_sf3_ownership.py`. The rebalance date is the **first month-end on or after
the availability date**, which fixes the schedule at the **February, May, August and
November month-ends**. Shares outstanding come from SF1 `ARQ` joined on `datekey` (the
actual SEC filing date), latest row with `datekey <= rebalance_date`.

## 3. Universe (fixed in advance)

From `_data/sharadar/panel/monthly_panel_dev.parquet`, at the four rebalance month-ends:

1. `spread_regime == "measured"`. Names at the EDGE resolution floor (`upper_bound`) or
   with no resolvable spread (`unmeasurable`) are **excluded, never costed at the floor**.
   This filter already implies close >= $2.00, non-zero trading on >= 90% of the trailing
   63 days, and an assigned liquidity band, because `build_monthly_panel` only estimates a
   spread for cells that passed those tests.
2. `median_dollar_volume >= $500,000`, so a $5,000 position never exceeds 1% of a name's
   median daily volume. This is the participation cap, expressed as a universe filter.
3. Institutional ownership present for BOTH the signal quarter `q` and the immediately
   preceding quarter `q-1` (a gap disqualifies), with `availability_date <= rebalance`.
4. Shares outstanding present for both quarters, `datekey <= rebalance`.
5. Ownership ratio in `(0, 1.5]` for both quarters. Summed 13F holdings can slightly
   exceed shares outstanding through reporting overlap; above 150% it is a data error.
6. Forward returns clipped to +/-100%. A prior study booked +9,900% on a zero-volume
   bankrupt shell that was 13% of its P&L.

## 4. Signal (fixed in advance)

    own_q     = inst_shares(q)   / shares_outstanding(q)
    own_{q-1} = inst_shares(q-1) / shares_outstanding(q-1)
    delta_own = own_q - own_{q-1}

`inst_shares` is the sum of `units` over SF3 rows with `securitytype == "SHR"` only.
`PUT` and `CLL` rows are option positions, not ownership; summing a put into "shares
held" would score a bearish position as accumulation.

Differencing the RATIO rather than the raw share count is deliberate: a stock split
doubles both institutional shares and shares outstanding, so a raw share-count difference
would read a 2:1 split as enormous accumulation, and splits are not randomly distributed
across the cross-section.

`delta_own` is winsorised at the 1st/99th cross-sectional percentile, then z-scored
cross-sectionally. That z-score is the signal.

## 5. Construction (fixed in advance, inherited not invented)

The portfolio construction, cost model and delisting accounting are taken **unchanged**
from `research/capacity_study.py`, which is the repository's registered construction.
Reusing it means any accounting defect is shared with an already-audited implementation
rather than freshly introduced.

* Long only, equal weight, **N = 50** positions.
* Entry: top decile of the signal. No-trade band: a held name is sold only when it falls
  out of the top 30%. (Capacity-study values, copied, not chosen.)
* **Quarterly** rebalance at the four dates in §2; monthly return accrual in between.
* Book size **$250,000** fixed notional ($5,000 per position) — retail scale, which is
  the only scale at which this repository's cost model produces a tradable universe.
* Costs, charged one way per name entering and one way per name leaving:
  half of the name's own EDGE-measured spread + square-root impact
  `0.1 * sqrt(trade_value / median_dollar_volume)` + IBKR commission ($0.0035/share,
  $0.35 per-order minimum, capped at 1% of trade value) + 0.2bps FX each way.
* Delistings: `terminal_return` applied **only** if the delisting date falls in
  `(exit_date, exit_date + 62 days]`, and the name is **removed from holdings** the
  moment its exit is booked.

## 6. Benchmark

Equal-weight buy-and-hold of **this sleeve's own universe**: the identical portfolio
engine, holding every eligible name instead of the top 50, reconstituted at the same four
quarterly dates, with identical delisting accounting and **zero costs**. Zero costs
flatters the benchmark, which is the conservative direction for the strategy.

**The reported headline is EXCESS over that benchmark.** A strategy that loses to passive
ownership of its own universe has no edge whatever its raw return.

## 7. Statistics reported

* Net return, volatility, Sharpe, max drawdown, annual turnover, annual cost drag.
* **Excess over the universe benchmark** — the number that decides the verdict.
* **Breadth: independent bets per year.** Four rebalances per year against one
  cross-section is **4 independent bets/yr**. Grinold gives `IR ~= IC * sqrt(BR)`, so at
  BR = 4 an information ratio of 1.0 demands a per-bet IC of 0.5, which does not exist in
  equity cross-sections. **This sleeve is breadth-poor by construction and is expected to
  fail on that axis regardless of signal quality.** Stated here, in advance, so that a
  weak result is not mistaken for a surprise.
* Information coefficient: per-cross-section Spearman rank correlation between signal and
  realised 3-month forward return; mean, standard error, t-statistic, count.
* Cost decomposition (spread / impact / commission) as a diagnostic.

## 8. Mechanism test (H2, secondary, not gate-eligible)

Each rebalance cross-section is split into terciles by ownership LEVEL `own_q`. Within
each tercile: the same long-only construction at **N = 20**, benchmarked against that
tercile's own equal-weight buy-and-hold, plus that tercile's IC. H2 predicts the largest
excess and IC in the LOW tercile.

Also reported, explicitly **NOT deployable** (small-cap borrow is neither available nor
costed here): the long/short top-decile-minus-bottom-decile spread, as a second read on
whether the signal orders returns at all.

## 9. Verdict mapping (fixed in advance)

| verdict | condition |
|---|---|
| PROMISING | excess > +2%/yr AND net Sharpe >= 0.75 AND IC t-stat >= 2.0 |
| MARGINAL | excess > 0 but any gate above fails |
| DEAD | excess <= 0 |

## ERRATUM 1 — the denominator (written after run 1, before run 2)

**Registered:** `own_q = inst_shares(q) / shares_outstanding(q)`, shares from SF1 `ARQ`
`sharesbas`. **Corrected to:** `own_q = sum(SF3.value) / marketcap(q)`, both in USD
millions struck at the same quarter-end price.

**Why.** Sharadar SF1 restates share counts onto TODAY's split basis. AAPL's 2015-09-30
`sharesbas` reads 22,301,324,000 — its true 5.575 billion multiplied by the 4:1 split of
August 2020. SF3 `units` are as reported in the 13F at the time. Dividing one by the
other gave AAPL 0.01% institutional ownership against a true 58%, and mean ownership
across the whole universe came out at 0.0–1.4% by tercile, which is impossible. The
distortion is **per name**, scaled by each stock's own post-sample split history, so it
does not cancel in a cross-sectional z-score, and it corrupts the quarter-on-quarter
difference for precisely the names that split — which are the ones that rose.

The corrected ratio uses no share count at all, so no split adjustment can enter it.
Validated at 2015-09-30 against published figures: AAPL 58.6%, MSFT 72.6%, XOM 50.5%,
JPM 74.7%, KO 64.7%.

**Run 1 is void and its numbers are recorded here so that a discarded run is visible:**
net −22.10%/yr, Sharpe −1.32, excess −22.82%/yr, IC(3m) −0.0032 (t −0.21), verdict DEAD.
The correction was forced by an impossible diagnostic, not by an unfavourable result —
run 1 had already failed, and the fix was made without knowing which way it would move
the number. It **does not** count as a second trial: a defective denominator is a bug,
not a configuration.

A second defect was fixed at the same point, also from an impossible output.
`build_monthly_panel` stamps each name's *last bar of the month*, which for a name that
stopped trading mid-month is a mid-month date. Taking the raw set of panel dates as the
month-end grid put rebalances on 2014-02-14, 2014-05-16 and so on, where the
"cross-section" contained the one or two names that delisted that day. The grid is now
the market-wide last trading day of each calendar month.

## 10. Trial accounting

Cumulative n_trials: 26 (prior programme, per `capacity_curve_result.md` §6) + **1** for
this sleeve = **27**. This sleeve is one trial: one hypothesis, one construction, one run.
The mechanism test is a decomposition of the same run and buys no additional selection
freedom because it cannot change the headline verdict.
