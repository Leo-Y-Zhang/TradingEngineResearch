# PRE-REGISTRATION — The Capacity Curve of the Fundamental Ordering Edge

**Status:** REGISTERED, NOT YET RUN
**Written:** 2026-07-27 (before any study code was executed against the data)
**Supersedes nothing.** Runs alongside `sharadar_confirmatory_prereg.md`, whose
confirmation window remains **unfired** and is preserved by this design.

---

## 0. Why this is a new hypothesis and not threshold-chasing

`sharadar_alpha_result.md` bans re-runs, variants and window/universe tweaks on the
1-month fundamentals long/short hypothesis. That ban is honoured. This study is a
different hypothesis with a different economic mechanism, a different universe, a
different construction, a different side, and four data tables that no code in this
repository has ever read.

| | Banned prior hypothesis | This study |
|---|---|---|
| Mechanism | fundamentals predict returns | **limits to arbitrage** — the edge survives where institutional capital cannot go |
| Universe | top-500/1000/1500, $5M dollar-volume floor | a pre-specified **ladder spanning $50k to >$200M** median dollar volume |
| Side | long/short, dollar-neutral | **long-only** (see §5) |
| Rebalance | monthly | **quarterly with a no-trade band** |
| Costs | flat 10 / 20 bps | **per-name estimated spread** + impact + commission (see §6) |
| Delisting | dropped (registered limitation) | **modelled from ACTIONS** |
| Tables | SF1, SEP | SF1, SEP, **DAILY, SF2, SF3, ACTIONS, TICKERS** |

The prior programme measured this edge only at sizes where institutional capital
operates and found net Sharpe ≤ 0. It never measured the region between an untradeable
$51k/day shell and its $5M/day floor — two orders of magnitude of unexamined space.
That gap is the subject of this study.

## 1. Hypothesis (falsifiable, single primary claim)

> **H1.** The net-of-cost performance of the fundamental ordering edge is monotonically
> decreasing in deployable capital.

**Primary statistic:** the Spearman rank correlation `rho_cap` between band index
(ordered by deployable capital, ascending) and band net Sharpe, across the six
pre-specified bands of §3. One number, one test.

- **H1 supported** if `rho_cap < 0` with a one-sided permutation p < 0.05.
- **H1 refuted** if `rho_cap >= 0`. A refutation closes the capacity thesis and no
  further capacity variants will be run on this dataset.

Supporting H1 is **not** a deployability finding. Deployability is decided separately
and only by the gate in §8.

**Secondary (declared in advance as NOT gate-eligible, reported for interpretation
only):** per-band net Sharpe, rank-IC, turnover, realised cost drag, and the
institutional-ownership mechanism test of §2.

## 2. Mechanism test (the reason SF3 matters)

If H1 holds because of limits to arbitrage, the edge should concentrate where
institutions are absent — not merely where stocks are small. SF3 (institutional 13F
holdings, 79,190,744 rows) permits a direct test that no prior study could run.

> **H2.** Within a fixed liquidity band, the edge is stronger in the low
> institutional-ownership tercile than in the high tercile.

H2 is secondary and not gate-eligible. Its value is diagnostic: H1 supported **and** H2
supported is a mechanism; H1 supported **and** H2 refuted means the capacity effect is
probably a size or data-quality artefact and must be treated with suspicion.

## 3. Universe — pre-specified bands (fixed before any run)

Bands are defined on **median trailing-63-day dollar volume** measured point-in-time at
each rebalance date. Deployable capital is defined as
`n_positions x 0.01 x median_dollar_volume` — a position no larger than 1% of a name's
median daily dollar volume, i.e. exitable within roughly one session at 1%
participation. With `n_positions = 30`:

| Band | Median dollar volume | Implied deployable capital | Role |
|---|---|---|---|
| B1 | $50k – $200k | ~$37k | **artefact control** — expected to show a mirage |
| B2 | $200k – $1M | ~$150k | untested |
| B3 | $1M – $5M | ~$750k | untested |
| B4 | $5M – $25M | ~$3.8M | partially measured (prior $5M floor) |
| B5 | $25M – $200M | ~$23M | measured, expected ~0 |
| B6 | > $200M | ~$150M | measured, known ~0 — **negative control** |

B1 and B6 are controls, registered in advance. **B1 is expected to look good and to be
false.** The prior programme's micro-cap mirage (ILXRQ, +9,900% on zero volume, 13% of
total P&L) lived in this band. If B1 does not show inflated raw performance that the
artefact filters of §7 then destroy, the filters are not working and the run is void.

## 4. Data and provenance

Re-exported 2026-07-25..27 via `scripts/download_sharadar_data.py`; SHA-256, row counts
and vendor snapshot times are recorded in `_data/sharadar/download_manifest.json`.

| Table | Rows | Vendor snapshot (UTC) |
|---|---|---|
| SEP | 46,235,528 | 2026-07-27 20:13:19 |
| SF1 | 3,200,111 | 2026-07-25 13:43:57 |
| SF3 | 79,190,744 | 2026-07-27 19:04:42 |
| SF2 | 11,822,993 | 2026-07-25 17:19:01 |
| DAILY | 39,973,270 | 2026-07-26 15:27:56 |
| ACTIONS | 671,240 | 2026-07-25 14:13:30 |
| TICKERS | 78,883 | 2026-07-27 02:40:15 |

Point-in-time discipline is unchanged from `data/sharadar_ingestion.py`: SF1 dimension
`ARQ` (as-reported), joined on filing `datekey`, `direction="backward"`, delisted
tickers retained.

**SF3 carries no filing date** — only `calendardate`, the quarter-end. 13F filings are
due up to **45 calendar days after** quarter-end, so joining on `calendardate` would
give the strategy holdings data six weeks before anyone could have seen it. This is a
serious and easy lookahead bias, and it is registered here as a fixed rule:
**SF3 availability date = `calendardate` + 45 calendar days.** Late and amended filings
mean even 45 days is mildly optimistic; that residual optimism is disclosed rather than
modelled, and it biases *toward* finding an edge, so an SF3-driven failure is robust and
an SF3-driven success must be discounted.

SF2 (insider transactions) joins on its own filing date, which the table does carry. Any
row whose availability date cannot be established is **dropped, never imputed forward**.

## 5. Construction (fixed)

- **Long-only.** Borrow in bands B1–B3 is typically unavailable or costs 10–50%/yr.
  The prior long/short construction was never executable at these sizes; pricing it as
  if it were would be the single largest source of false optimism in this study.
- **30 equal-weighted positions**, top decile of the combined signal within the band.
  Equal weight, not cap weight: cap weighting concentrates the book in the largest names
  in each band, which is precisely where the hypothesis predicts no edge.
- **Quarterly rebalance** with a **no-trade band**: a held name is sold only when it
  leaves the top 30% of the ranking. Turnover, not signal strength, killed the prior
  study; this is the mechanism for controlling it and it is fixed in advance.
- **Benchmark: equal-weight buy-and-hold of the same band**, rebalanced on the same
  dates. Alpha is measured against this, never against the S&P 500. Otherwise the
  small-cap premium would be booked as skill.

## 6. Cost model (the part most likely to decide the outcome)

Flat 10bps is meaningless below $5M/day. Per side:

1. **Spread — Corwin & Schultz (2012)** high-low estimator on SEP `high`/`low`,
   computed per name over a trailing 63-day window; negative estimates (a known
   property of the estimator) are floored at a price-bucket minimum, documented in the
   result. Half-spread is charged on entry and on exit.
2. **Impact** — square-root law, reusing the existing `EXEC-4` implementation, with
   participation capped at 1% of median daily dollar volume (consistent with §3).
3. **Commission** — Interactive Brokers tiered US equities: $0.0035/share, **minimum
   $0.35 per order**, capped at 1% of trade value. The per-order minimum is material at
   the position sizes B1–B2 imply and is not optional.
4. **Borrow** — not applicable (long-only).
5. **FX** — 0.002% per conversion (IBKR spot), charged once per entry and exit.

**Positive control, run before anything else.** The spread estimator is applied to
mega-caps of known spread (AAPL, MSFT, JPM). If AAPL's estimated effective spread does
not land in roughly 1–5 bps, **the cost model is void and no result is reported.** A
cost model that cannot price a stock whose spread is known cannot be trusted to price
one whose spread is not.

## 7. Artefact filters (all applied, all fixed in advance)

Every one of these exists because the prior programme was fooled by the thing it
catches:

- Forward return capped at ±100% per period (standard equipment since the ILXRQ find).
- Minimum price $2.00 at rebalance.
- Non-zero volume on ≥ 90% of the trailing 63 sessions (kills the stale-quote class).
- **`closeadj` return cross-checked against ACTIONS split/dividend events**; a name-month
  whose adjusted return is inconsistent with its recorded corporate actions is dropped
  and counted. This catches the AWHL class (an unadjusted reverse split, a genuine
  vendor bug) automatically rather than by inspection.
- **Delisting returns applied from ACTIONS** rather than dropped, and **by event type**.
  ACTIONS (1997-12-31 → 2026-07-27) resolves the taxonomy the prior studies could not
  see: `delisted` (19,196), `bankruptcyliquidation` (3,347), `regulatorydelisting` (880),
  `voluntarydelisting` (375), `acquisitionby`/`acquisitionof` (8,247 each),
  `mergerto`/`mergerfrom` (134 each). This distinction is load-bearing: a bankruptcy is
  approximately −100%, an acquisition is typically a *premium*. Dropping all delistings
  (every prior study) biases returns **up**; assigning −100% to all of them biases
  **down**. Registered treatment: bankruptcy/regulatory/involuntary delisting → −100%;
  acquisition/merger → the last observed price with the deal consideration applied where
  ACTIONS records it, otherwise the last traded price with the assumption disclosed;
  ambiguous or unmapped codes → the name-month is dropped **and counted** in the result.
  In micro-caps the net of this is expected to be a *downward* correction that may on its
  own destroy the edge. That outcome is a result, not a failure.
- `tickerchangefrom`/`tickerchangeto` (13,424 pairs) are applied to bridge renames. This
  is the vendor-resolved version of the defect that invalidated the free insider study's
  first run (an as-filed-ticker join that silently lost ~22% of rows, with Alphabet
  absent from the entire sample). Here it is data, not a hand-audited bridge.
- Every filter reports how many rows it removed. Silent filtering is banned.

## 8. Gate and trial accounting

The promotion gate of `research/validation.py::selection_rule` applies **unchanged**:
mean rank-IC > 0.01, stability > 0.60, deflated-Sharpe proxy > 0.25, **DSR ≥ 0.95**,
zero leakage flags, no regime Sharpe < −0.50. Default-deny; any single failure blocks.

The standalone `sharpe_net > 0.75` criterion is reported **both** as-is and against the
portfolio-marginal criterion registered separately in
`docs/project-control/specs/2026-07-27-portfolio-gate-amendment.md`. That amendment is
registered **before** this study runs, precisely so it cannot be introduced afterwards
to rescue a disappointing number. If the amendment is not approved and reviewed before
this study executes, the standalone criterion governs.

**Cumulative trial ledger: 23 (prior programme) + 3 (this study: feature set,
construction, monotonicity test) = 26.** As recorded in `DATA_EDGE_PLAN.md` L92–93 and
`sharadar_alpha_result.md` L52–54, this count still under-states the true search space,
so deflation here remains slightly lenient. No additional configuration may be run
under this registration; a further variant requires a new registration at a higher
count.

## 9. Development / confirmation split

All work in this registration is confined to the **DEV window (≤ 2015-12-31)**. The
2016+ confirmation window from `sharadar_confirmatory_prereg.md` **remains unfired** and
no tool built under this registration may read it. One confirmation shot may be fired if
and only if a frozen model passes the DEV-side gate, under a four-lens review conducted
before the run.

## 10. What would refute this, and what a failure means

- `rho_cap >= 0` → H1 refuted; the capacity thesis is closed on this dataset.
- Spread positive control fails → no result reported at all.
- Delisting-adjusted returns flip the sign of any band's edge → that band's prior
  appearance of edge was survivorship, and it is reported as such.
- B1's raw performance survives the artefact filters → the filters are inadequate; the
  run is void and reported as void.

A negative result is a publishable measurement of where retail's capacity advantage
does and does not exist, which no prior study in this programme could produce. It is
not a reason to run a seventh band.

## 11. Deviations

Any deviation from this registration is recorded in a dated **erratum appended below**,
never by editing anything above this line.

---

## Errata

### ERRATUM 1 — 2026-07-27: the registered cost model is VOID; the study is BLOCKED

The §6 positive control was run before any study code touched forward returns. **It
failed, and the cost model registered in §6 may not be used.** Per §10 ("Spread positive
control fails → no result reported at all"), the capacity study does not run until a
cost model passes. Nothing about H1 has been measured, and no trials have been consumed
— the control never looked at a return, only at prices whose spreads are known
independently.

**Corwin & Schultz (2012) is retired.** Two disqualifying properties, both measured
against synthetic data with known ground truth and realistic overnight gaps
(`scripts/spread_positive_control.py`, `tests/test_spread_estimation.py`):

1. A **noise floor of ~11.2 bps per 1% of daily volatility** (linear to within 0.5%
   across daily vols of 1–8%). This is rectification bias — the estimator floors a
   negative sample mean at zero, turning pure noise into a positive "spread" whenever
   there is no real spread to measure. AAPL's 41.5 bps estimate for 2016–2026 was
   essentially all floor, against a true effective spread of 1–2 bps.
2. It also **understates genuinely large spreads**: with 50% of variance overnight, a
   true 300 bps spread was recovered as 208 bps and a true 100 bps as 73 bps.

Together these compress every name toward ~50–100 bps regardless of truth. Critically,
the *direction* is hostile to this study: overstating liquid-band costs while
understating illiquid ones biases the capacity curve **toward H1**. Had the study run on
it, a "confirmation" of H1 would have been substantially an artefact of the cost model.

**Abdi & Ranaldo (2017) was adopted, then also failed.** On synthetic data it is close
to unbiased in the range that matters — true 100/200/400 bps recovered as 97/198/399 at
2% daily vol, degrading to 78/187/393 at 6%. But on the real SEP tape it does not track
trading costs at all:

| median dollar volume | AR median estimate | true effective spread |
|---|---|---|
| $50k–$200k | 754 bps | plausibly 200–1000 bps |
| $200k–$1M | 374 bps | plausibly 100–500 bps |
| $1M–$5M | 533 bps | — |
| $5M–$25M | 489 bps | — |
| $25M–$200M | 346 bps | — |
| **>$200M** | **301 bps** | **~1–3 bps** |

Estimates are **non-monotone in liquidity** and overstate mega-cap costs by roughly two
orders of magnitude. An estimator that cannot order the cross-section by liquidity
cannot be used to measure a capacity curve, whose entire content is an ordering by
liquidity. The synthetic/real divergence is itself the finding: both estimators are
validated in the literature on eras of much wider spreads, and neither survives contact
with the modern tape.

**Consequences, registered now rather than after seeing any result:**

- §6 items 1 (spread) and its positive control are void. §6 items 2–5 (impact,
  commission, borrow, FX) are unaffected.
- The band ladder of §3, the construction of §5 and the artefact filters of §7 are
  unaffected and remain registered.
- **Cumulative n_trials remains 23.** The control consumed none: it examined no forward
  returns, fitted nothing, and tested no strategy. Selecting a measurement instrument by
  calibration against known ground truth is not a research trial, and would only become
  one if an instrument were chosen by which produced the better strategy result. No such
  selection was made or may be made.

**Unblocking requires one of the following, each needing its own registration:**

(a) Implement **EDGE** (Ardia, Guidotti & Kroencke 2024, *Journal of Financial
Economics*), the current state of the art, built specifically because CS and AR fail on
modern data. It must pass all three checks A/B/C before use.
(b) Abandon per-name OHLC estimation and cost trades from a **published empirical
schedule** keyed on price and dollar volume, with the schedule's source cited and its
own positive control.
(c) Conclude that H1 cannot be honestly tested at the required cost precision with the
data available, and bank that as the result.

Option (a) is preferred and (c) is a legitimate outcome. What is **not** permitted is
running the study on a cost model known to be non-monotone in the very variable the
hypothesis is about.

### ERRATUM 2 — 2026-07-27: EDGE adopted, control PASSED, study unblocked

**First, a correction to erratum 1.** Its claim that Abdi-Ranaldo is "non-monotone in
liquidity" on real data was **wrong, and the fault was mine.** The cross-sectional check
read a raw chunk of `SEP.csv` and never sorted it. The Sharadar export is *not* in
chronological order — dates run backwards within a ticker — and every estimator here
compares consecutive bars, so the check was fed reverse-chronological input and its
output was meaningless rather than merely noisy. The AR cross-sectional table in erratum
1 (301bps for >$200M/day names, non-monotone) should be disregarded. Nothing was
concluded from it that survives; AR was superseded on its *volatility* degradation,
which was measured on synthetic data and is unaffected.

The Corwin-Schultz retirement **stands**: its noise floor and its ~30% understatement of
genuinely large spreads were both measured against synthetic ground truth, independent of
the sorting defect.

**EDGE (Ardia, Guidotti & Kroencke 2024) is adopted and the control PASSES all three
checks.**

| check | result |
|---|---|
| A — accuracy on known spreads (synthetic) | worst relative error **1.8%** at 100–400bps across 2/4/6% daily vol (AR: 18–22% at 4–6%) |
| B — honest degradation on mega-caps (real) | all five classified `upper_bound`; the estimator declines to claim a measurement it cannot make |
| C — cross-sectional monotonicity (real) | **279 → 133 → 133 → 105 → 72 → 69 bps** as liquidity rises — monotone |

**The noise floor is recalibrated against real data, and this mattered.** The idealised
simulation gave 11.8 bps per 1% of daily volatility. Measured on eight mega-caps whose
true spreads sat at 1–3bps across 2010–2025 while their volatility varied by more than
2x, the real figure is **26.2 bps per 1% daily vol** (25 excluding the 2020 COVID
outlier; ~15% year-to-year dispersion). Real tapes have volatility clustering and a
U-shaped intraday volatility profile that a constant-volatility random walk does not
reproduce. Had the simulated constant been used, AAPL's 56bps estimate would have been
classified a genuine measurement.

**Registered cost model (replacing §6 item 1).** Per name, per rebalance, on a trailing
63-day window:

- `spread_with_resolution` returns `(value, regime)`.
- **`measured`** (estimate > 1.5x floor): use the EDGE estimate. The multiple is set from
  the synthetic evidence that accuracy returns once a true spread is comparable to the
  floor (true 100bps recovered as 98.2 against a 71bps floor), while a true 20bps against
  the same floor returned 76.6 and is pure noise.
- **`upper_bound`**: the true spread lies below the returned value. The name is
  **excluded from spread-costed trading** rather than costed at its floor. Inflating an
  upper bound to the floor would manufacture cost out of an absence of information, and
  would do so hardest for volatile names — precisely the population under study.
- **`unmeasurable`**: too few genuine trading days. Untradeable, not free.
- **Coverage must be reported per band.** Measured share falls from 62% in the smallest
  band to ~0% in the liquid ones. This is expected — liquid names' true spreads are below
  what daily bars can resolve — but it means the capacity curve is measured where spreads
  are large and bounded where they are small, and any result must say so.

**Cumulative n_trials remains 23.** Three estimators were evaluated against known ground
truth; none was selected by its effect on any strategy result, and no forward return has
yet been examined.

§6 items 2–5 (impact, commission, borrow, FX), the bands of §3, the construction of §5
and the artefact filters of §7 are unchanged. **The study is unblocked.**

---

### ERRATUM 3 — 2026-07-28: the `upper_bound` EXCLUSION was a universe bias; two bounds replace it

**Erratum 2 got the classification right and the CONSEQUENCE wrong.** It correctly
established that a liquid name's true spread sits below what daily bars can resolve, then
instructed that such names be **excluded from spread-costed trading**. That instruction is
withdrawn. `upper_bound` states that the true spread is BELOW the returned value — the
name is CHEAP, not unknown. Excluding it deletes the cheapest names in the market.

**Measured size of the bias** (`scripts/measure_spread_universe_bias.py`, DEV panel
1998-04-30 to 2015-12-31, derived statistics only): of 922,652 eligible (name, month)
cells, **525,933 — 57% — were being deleted**, and they carried **6.4x the median dollar
volume** of the cells kept ($5,408,200/day vs $850,300/day) at **0.24x the spread**
(36.2bps vs 153.1bps under the realistic bound). The tradable universe was 396,719 cells;
it is 922,652 under this erratum, **+132.6%**. Five of the six iteration-1 sleeves flagged
this independently before it was measured.

The estimator's authors documented the phenomenon: Ardia, Guidotti & Kroencke (2024) §4,
"the spreads for mid and large caps have become too small to be reliably estimated from a
monthly sample of daily data". Non-resolution in the liquid cross-section is expected
behaviour, never evidence about a name's tradability.

**Registered cost model (replacing the erratum-2 model).** Per name, per rebalance, on a
trailing 63-day window, via `research.spread_estimation.spread_cost_bounds`, which returns
BOTH of:

- **(a) CONSERVATIVE** — `upper_bound` names are charged their own EDGE estimate. The true
  spread is below it, so this OVERSTATES cost. **A result that passes under (a) is REAL.**
- **(b) REALISTIC** — `upper_bound` names are charged `liquid_name_spread`: the median TAQ
  effective spread for their liquidity quintile from **Ardia, Guidotti & Kroencke (2024),
  *JFE* 161, 103916, Table 4 Panel C** (3.14 / 2.09 / 1.08 / 0.30 / 0.09 percent), keyed
  on median daily dollar volume at breakpoints $350k / $1.5M / $5M / $20M (which split
  this study's own eligible universe into near-exact quintiles), log-interpolated between
  quintile anchors and **CLAMPED — never extrapolated** beyond the table's support; scaled
  by the Table 4 Panel B era factor **floored at 1.0** (a discount is never granted,
  because that table's post-2003 compression is concentrated in large caps and passing it
  through uniformly would cheapen modern small caps); **capped at the (a) figure**, since
  `upper_bound` genuinely bounds the truth from above; and **floored at the minimum legal
  tick** ($0.125, then $0.0625 from 1997-06-24, then $0.01 from 2001-04-09 — half the DEV
  window predates decimalisation, where a $20 stock could not trade inside 31bps).
  **A result that fails under (b) is DEAD.**
- **Between (a) and (b) the result is UNDETERMINED** and must be reported as such.
  `bracket_verdict` enforces the three-way outcome and raises if the pair inverts.
  `realistic <= conservative` holds by construction (0 inversions in 922,652 cells), so a
  cheaper (b) can only move a verdict from "dead" to "undetermined", never to "real".
- **`measured` names are unchanged.** Both bounds equal the EDGE estimate, so every prior
  result stays comparable. **`unmeasurable` names remain untradable** — the schedule
  prices CHEAP names, not ABSENT ones.
- **BOTH bounds must be reported for every result. A single-bound number is not a result.**

**Positive control extended, and it gates this erratum.** `scripts/spread_positive_control.py`
check D requires AAPL/MSFT/JPM/XOM/KO over 2016-2026 to cost **1-5bps per side** under (b)
with (a) at least 2x that. Measured: all five at **4.50bps per side** under (b) against
20.3-29.3bps under (a). Checks A-D all PASS. The check fails closed — outside that window
the schedule is declared wrong and nothing may be re-run. It is the only check permitted to
read post-2015 bars, and it touches no strategy, no signal and no forward return.

**Cumulative n_trials remains 32.** No strategy was run and nothing was fitted to any
return; the schedule is a published table and a liquidity ranking, both fixed before any
re-run. §6 items 2-5, the bands of §3, the construction of §5 and the artefact filters of
§7 are unchanged.
