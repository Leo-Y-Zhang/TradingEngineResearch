# RESULT — Short-horizon reversal, RE-TESTED on the corrected liquidity-first universe

**Run:** 2026-07-28, once, exactly as pre-registered in
`research/sleeves/reversal_retest_prereg.md` (written and frozen before the run).
**Governing documents:** the internal research log iterations 1–2,
`research/medallion_style_alpha_search/breadth_sleeve_hunt_result.md`,
`research/spread_estimation.py`.

**VERDICT: DEAD, under both cost bounds, at every registered frequency, on both universe
cuts. `bracket_verdict(False, False) == "dead"`.** Nothing here is promotable and the
2016+ confirmation window stays UNFIRED.

**But the sleeve did not die the way iteration 1 said it would, and the reason is the
result.** The corrected cost model moved net Sharpe from **−3.62 to −0.85** and the gross
edge is real, stable across all four eras and 4.1 standard deviations above its own
placebo. What kills it is now a single exact number rather than a vague "costs too high":

> **The top-decile weekly long/short book must trade at ≤ 4.70 bps per round trip to reach
> a net Sharpe of 0.75. IBKR commissions alone cost it 3.06 bps. The entire budget for
> spread, impact and borrow is therefore 1.64 bps per round trip — a full effective spread
> narrower than one cent on any share priced below $61, which is 75.7% of this universe.
> Its measured realistic cost is 57.8 bps: 12.3× over budget.**
>
> **Ten of the twelve registered configurations have a NEGATIVE cost budget — they cannot
> reach the gate at literally zero cost.**

---

## 1. Headline results

PRIMARY is the top decile by median dollar volume, weekly, long/short. Benchmark is an
equal-weight, zero-cost buy-and-hold of the sleeve's own universe.

| | gross | net (a) CONSERVATIVE | net (b) REALISTIC | zero-cost ceiling |
|---|---:|---:|---:|---:|
| annual return | **+24.88%** | −52.59% | −25.76% | +21.51% |
| annual vol | 29.76% | 29.86% | 29.73% | 29.76% |
| **Sharpe** | **+0.891** | **−2.333** | **−0.854** | +0.799 |
| excess vs own universe | +21.20% | **−56.27%** | **−29.44%** | +17.83% |
| cost per round trip | — | 107.1 bp | 57.8 bp | 3.1 bp |
| cover ratio | — | 0.26 | 0.48 | 9.05 |

Benchmark: **+3.68%/yr, vol 24.46%, Sharpe 0.272, maxDD 65.3%.** Universe 431 names/rebalance, long leg
43, short leg 42, short leg formed in all 921 periods. Turnover **89.8×/yr** (≈45× per
leg). Window 1998-05-01 → 2015-12-24, 921 weekly periods, 17.6 years.

**All twelve registered cells** (net Sharpe / excess over own universe):

| universe | book | f | gross Sh | net (a) cons | excess (a) | net (b) real | excess (b) | zero-cost Sh |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decile | L/S | 52.2 | 0.891 | −2.333 | −56.27% | −0.854 | −29.44% | +0.799 |
| decile | L/S | 26.1 | 0.396 | −1.394 | −37.81% | −0.580 | −21.14% | +0.346 |
| decile | L/S | 12.0 | 0.626 | −0.273 | −13.56% | **+0.130** | **−3.50%** | +0.601 |
| decile | long-only | 52.2 | 0.462 | −0.746 | −35.52% | −0.184 | −18.02% | +0.426 |
| decile | long-only | 26.1 | 0.245 | −0.379 | −23.63% | −0.090 | −13.87% | +0.226 |
| decile | long-only | 12.0 | 0.275 | −0.035 | −11.61% | +0.109 | −6.51% | +0.266 |
| quintile | L/S | 52.2 | 0.770 | −2.741 | −60.17% | −1.202 | −35.59% | +0.664 |
| quintile | L/S | 26.1 | 0.329 | −1.626 | −40.53% | −0.780 | −24.60% | +0.271 |
| quintile | L/S | 12.0 | 0.538 | −0.521 | −17.44% | −0.068 | −7.85% | +0.507 |
| quintile | long-only | 52.2 | 0.395 | −0.899 | −39.19% | −0.341 | −23.01% | +0.354 |
| quintile | long-only | 26.1 | 0.221 | −0.439 | −25.55% | −0.156 | −16.26% | +0.200 |
| quintile | long-only | 12.0 | 0.243 | −0.092 | −13.40% | +0.051 | −8.51% | +0.232 |

**Every excess in the table is negative. The best net Sharpe in the study is +0.130,
against a promotion gate of 0.75 and a DSR bar of 0.917.** Best DSR at n=34: **0.058**.

---

## 2. THE FREQUENCY CURVE — the registered deliverable

Net Sharpe against rebalance frequency, top decile, long/short, realised f:

| | f = 52.2 (weekly) | f = 26.1 (fortnightly) | f = 12.0 (monthly) |
|---|---:|---:|---:|
| **(a) conservative** | −2.333 | −1.394 | **−0.273** |
| **(b) realistic** | −0.854 | −0.580 | **+0.130** |
| zero-cost ceiling | **+0.799** | +0.346 | +0.601 |
| gross Sharpe | **0.891** | 0.396 | 0.626 |
| gross annual | +24.88% | +7.54% | +13.98% |
| turnover/yr | 89.8× | 45.3× | 21.0× |
| realistic cost/RT | 57.8 bp | 59.5 bp | 61.9 bp |
| **gross alpha/RT** | **27.7 bp** | 16.6 bp | **66.6 bp** |
| **cover ratio (b)** | 0.48 | 0.28 | **1.08** |

**The optimum is a CORNER, at the lowest frequency tested. H2's interior optimum is
falsified, and P4 is confirmed.** Net Sharpe rises monotonically as *f* falls under both
bounds; nothing in the registered range turns over.

**Why there is no interior optimum, stated precisely.** Grinold's IR = IC·√BR with BR ∝ f
gives gross annual return ∝ √f, which against a cost linear in f produces
f\* = (σ·IC·√N / 2c)². That is not what the tape does. From the monthly point, √f scaling
predicts a weekly gross Sharpe of 0.626·√(52.2/12) = **1.31**; measured **0.891**. From the
weekly point it predicts a monthly gross Sharpe of 0.891/√4.35 = **0.427**; measured
**0.626**. **Neither direction fits, because the gross curve is not monotone at all** —
fortnightly (0.396) is a genuine trough between weekly (0.891) and monthly (0.626).

That trough is not a phase artefact. The registered fortnightly grid takes weekly signal
dates `[0::2]`; the other phase `[1::2]` was run as a declared post-hoc diagnostic and
returns gross Sharpe **0.434** (net realistic −0.402), i.e. the same trough. The mean IC
tells the same story: **+0.0258 (t = 4.58)** at one week, **+0.0096 (t = 1.21)** at two
weeks, **+0.0243 (t = 2.26)** at one month.

**Reading:** the 5-day signal is not one effect decaying smoothly through horizon. It is
consistent with *two* separately documented anomalies — 1-week reversal and 1-month
reversal — with a genuine dead zone between them. **Any model that treats rebalance
frequency as a smooth breadth knob is wrong on this signal.** That is the transferable
finding, and it holds even though every point on the curve fails.

---

## 3. THE DECISIVE ARITHMETIC — break-even cost per round trip

The gate is a Sharpe, so its budget is computed from the arithmetic mean
(`gross_sharpe × gross_vol`), not the geometric return — at ~30% vol the two differ by
~4.5%/yr of variance drag, which is larger than the whole budget being solved for. Net
Sharpe is linear in cost (the reported ladder steps by −0.87, −0.87, −0.85 per 0.5× rung).

| configuration | actual (b) | needed for **Sharpe 0.75** | needed for **zero excess** | commissions alone |
|---|---:|---:|---:|---:|
| **decile / weekly / L/S** | 57.8 bp | **4.70 bp** | 25.0 bp | 3.06 bp |
| quintile / weekly / L/S | 62.8 bp | **0.65 bp** | 23.1 bp | 3.40 bp |
| decile / monthly / L/S | 61.9 bp | −15.24 bp | **45.3 bp** | 3.12 bp |
| decile / weekly / long-only | 58.0 bp | −25.78 bp | 17.7 bp | 3.21 bp |
| decile / fortnightly / L/S | 59.5 bp | −21.34 bp | 12.8 bp | 3.08 bp |
| quintile / monthly / L/S | 67.3 bp | −23.15 bp | 29.7 bp | 3.44 bp |
| *(remaining six)* | 58–66 bp | −31 to −238 bp | −3 to −15 bp | 3.2–3.7 bp |

**Ten of twelve configurations have a negative gate budget: no cost model, including one
that charges literally nothing, can bring them to 0.75.** The two that are not
arithmetically impossible need 4.70 bp and 0.65 bp per round trip against commissions of
3.06 bp and 3.40 bp — so both need spread + impact + borrow to fit inside **1.64 bp** and
**−2.75 bp** respectively. The second is already impossible. The first needs a full
effective spread of 1.64 bp, which is below the minimum legal tick for any share priced
under **$60.98** — and **75.7%** of this universe's cells trade below that (median share
price $38.69, lower quartile $23.96). It is not merely expensive, it is not quotable.

**This is a strictly stronger kill than iteration 1's.** Iteration 1 said "its zero-cost
ceiling is 0.41, below the gate". On the corrected universe the zero-cost ceiling is
**0.799 — above the gate** — so iteration 1's specific claim was universe-dependent and
**does not survive**. The sleeve dies anyway, on an exact budget that commissions alone
consume two-thirds of.

---

## 4. What the cost fix actually bought, and what it did not

| | iteration 1 (>$5M/day, `measured` only) | iteration 2 (top decile, both regimes) |
|---|---:|---:|
| net Sharpe, weekly L/S | **−3.619** | **−0.854** (realistic) / −2.333 (conservative) |
| cost per round trip | 158.6 bp | **57.8 bp** (realistic) / 107.1 bp (conservative) |
| gross alpha per round trip | 15.3 bp | **27.7 bp** |
| cover ratio | 0.10 | **0.48** |
| gross Sharpe | 0.519 | **0.891** |
| universe | 480 names | 431 names, 74.2% of cells previously deleted |

**The repair is large and real: 64% of the round-trip bill is gone, and the sleeve moved
2.77 Sharpe points. It is still not close.**

**P1 FAILED, and the reason is the most useful thing in this document.** Realistic cost per
round trip came in at **57.8–67.3 bp** against a registered prediction of 20–50 bp;
conservative at **107–117 bp** against 50–90 bp. Both are worse than predicted. The
`>$20M/day` band table in internal research log iteration 2 (realistic 14.7 bp) implied far more.
The measured decomposition of the top-decile universe says exactly why:

| regime | share of cells | median $vol/day | median (a) | median (b) | mean (b) |
|---|---:|---:|---:|---:|---:|
| `upper_bound` | **74.2%** | $121.4M | 50.0 bp | **9.0 bp** | 14.3 bp |
| `measured` | **25.8%** | $120.0M | **90.4 bp** | **90.4 bp** | 114.8 bp |
| all cells | 100% | — | 77.5 bp (mean) | — | **40.2 bp** |

**26% of the cells carry 74% of the cost.** Both bounds price a `measured` name
identically — by design, because the schedule has no business overriding a measurement —
and at *statistically identical liquidity* ($120.0M vs $121.4M median dollar volume) a
liquid name resolves **precisely when its estimate is wide**. That is the iteration-1
selection effect, alive and dominant *inside* the corrected universe: the two-bound fix
repaired **which names are admitted** but not **how the estimator selects which of them it
prices**.

**This is not a licence to reprice them.** Refusing to let the schedule override a
measurement is the discipline that keeps bound (a) meaningful. But it is now the largest
identified cost defect in the programme, it points the expensive way, and it should get its
own calibration and its own control — the same treatment the universe bias got, and
explicitly **not** a strategy re-run. Recorded for the log alongside
`capacity_study.IMPACT_COEFFICIENT`.

---

## 5. Pre-registered predictions, scored

| # | prediction | outcome |
|---|---|---|
| **P1** | cost/RT: 20–50 bp realistic, 50–90 bp conservative | **FAILED** — 57.8–67.3 and 107–117. Worse than predicted, for the measured reason in §4. |
| **P2** | gross alpha/RT ≤ iteration 1's 15.3 bp (liquid names more efficient) | **FAILED** — 27.7 bp weekly, 66.6 bp monthly. The liquid decile has **more** gross alpha per round trip, not less. |
| **P3** | weekly net Sharpe (b) between −1.5 and 0.0, excess still negative | **CONFIRMED** — −0.854, excess −29.44%. |
| **P4** | frequency curve monotone, argmax at monthly (corner, not interior) | **CONFIRMED** — monotone in net under both bounds; argmax monthly. H2 falsified. |
| **P5** | nothing clears 0.75 with positive excess under either bound at any frequency | **CONFIRMED** — best net Sharpe 0.130, best excess −3.50%. **Verdict DEAD.** |

**Two of five predictions failed and both failed in the same direction: the market's liquid
decile is *more* expensive to trade AND *more* profitable to trade than registered.** They
very nearly cancel — cover ratio 0.48 where iteration 1 had 0.10 — and the sleeve dies in
the gap.

---

## 6. Verification — everything that was run to try to break this

**Negative control PASSES.** Fixed-seed per-date permutation of the signal within the
universe, top decile weekly, gross: seeds 11/22/33/44 give **+0.217, +0.066, −0.231,
−0.243**, mean **−0.048 ± 0.227**, against a live gross Sharpe of **0.891**. The live
result is **4.1 sd above its own placebo**; the harness is not manufacturing alpha, which
is what makes the negative verdict trustworthy.

**Cost-bound vectorisation verified against the reference.** The two bound matrices are
built by a vectorised reimplementation of `spread_estimation.bounds_from_estimate` over
922,652 cells; it is asserted cell-for-cell against the reference scalar function on a
fixed-seed sample of **4,000 real cells before any return is computed**, and the run aborts
on a mismatch. `realistic ≤ conservative` holds on every cell — 0 inversions.

**P&L concentration is clean.** Largest single ticker **0.50%** of total gross P&L
(STI1); largest single (ticker, period) **0.20%** (HIG @ 2009-03-09). The long-only book:
0.48% and 0.26%. No name-month is anywhere near the 13% that once dominated a study.

**Era stability — and this is a genuine improvement over iteration 1.** Gross Sharpe by
era, top-decile weekly L/S: **1998–2001 +0.80, 2002–2007 +1.06, 2008–2011 +1.31,
2012–2015 +0.44.** Iteration 1's gross edge lived in one crisis (2008–2011 +24.6%/yr
against −1.1%/yr in 1998–2002). On the liquid decile the gross edge is present in **all
four eras**, decaying but never absent. Net realistic by era: −1.06 / −1.27 / −0.12 /
−1.52 — negative in all four. Halves: gross 1998–2006 vs 2007–2015 both positive; net
−1.07 and −0.60.

**Delisting accounting audited.** 1,266 terminal returns booked across all weekly
periods, each only when the delisting date fell inside the position's own window extended
by 62 days *and* the name actually stopped printing prices; the book is re-formed from the
current universe every period so a terminal name cannot be re-booked. 919 name-periods hit
the ±100% cap out of ~3.2M (0.03%), all clipped.

**A real defect was found and fixed by a test, before the reported run.**
`Timestamp.to_datetime64()` returns whatever *resolution* the timestamp carries, while
`delist_date` is nanoseconds. On a seconds-resolution calendar every delisting-window
comparison would silently evaluate false, booking **no delistings at all** — a survivorship
flatter that raises nothing. Production reads nanoseconds from parquet so it was correct,
proven three ways: the 1,266 bookings above, an explicit regression test on a
seconds-resolution index, and a **byte-identical re-run of the whole study** after the
cast was made explicit. Recorded because this is the eleventh-plus accounting defect in the
programme found by a check rather than by a symptom.

**Post-hoc, unregistered, changes no verdict:** the fortnightly grid's other phase (§2).

**Data hygiene.** DEV window only; the maximum bar date touched is 2015-12-31 and the
matrix builder raises if the panel contains anything later. Nothing was downloaded. No raw
Sharadar row appears in this document or in any committed artefact — only derived
statistics. The 2016+ confirmation window remains **UNFIRED**.

**Persisted (prereg 9.9):** per-period net return series for all 6 configurations × 3 cost
treatments in `research/sleeves/_reversal_retest/net_returns_*.parquet`, so cross-sleeve
correlation is computable later without re-running anything. This is the process defect
`breadth_sleeve_hunt_result.md` §4 recorded when five of six sleeves persisted nothing.

---

## 7. Trial accounting

**This study spends 1 trial.** Cumulative before: 32. After: **33** (34 if the concurrent
PEAD re-registration is counted; the bar is reported at n=34 so the count cannot flatter
the result).

At **n = 34** over this sleeve's **17.6-year** sample, DSR ≥ 0.95 demands a standalone
annual Sharpe of **0.917**. Computed with the same algebra as
`research/validation.py::deflated_sharpe_ratio` under normal monthly returns; it reproduces
the recorded anchors exactly (7 yr / n=32 → 1.488 vs recorded 1.488; 40 yr / n=32 → 0.597
vs recorded 0.597).

| | gross Sharpe | vs bar 0.917 | net Sharpe (b) | vs bar |
|---|---:|---|---:|---|
| decile / weekly / L/S | 0.891 | fail (by 0.026) | −0.854 | fail |
| decile / monthly / L/S | 0.626 | fail | +0.130 | fail |
| quintile / weekly / L/S | 0.770 | fail | −1.202 | fail |

**Not even the GROSS Sharpe clears the deflated bar** — the best in the study misses by
0.026. Iteration 1's reversal gross Sharpe was 0.519 against a bar of 0.91; the liquid
decile raises it to 0.891 and still fails. Six additional trials cost only +0.007 Sharpe at
this sample length; the sample-length term dominates completely, exactly as the DSR spec
found.

---

## 8. What this establishes for the programme

1. **The cost-model defence is now closed for frequency-driven breadth.** Iteration 1
   left open the possibility that six negative verdicts were artefacts of a cost model that
   could only see the expensive half of the tape. On the sleeve where that criticism was
   worth the most — 45× turnover, the construction the bias punished hardest — the repair
   delivers 64% of the bill and 2.77 Sharpe points, and the sleeve still loses to passive
   ownership of its own names by 3.5–29.4%/yr. **"Our costs were wrong" can no longer
   explain away a high-turnover result.**
2. **Breadth bought from frequency is still priced at the round-trip cost, and now the
   price is exact.** The corner optimum at the lowest frequency tested is the same
   conclusion `breadth_sleeve_hunt_result.md` §3.3 reached (12.7 bets per unit of turnover
   for frequency-driven breadth against ~200 for event-driven), reached independently and
   with a break-even budget attached.
3. **The gross signal is real and is not the problem.** IC +0.0258 at t = 4.58 over 921
   weekly cross-sections, 4.1 sd above its placebo, positive gross Sharpe in all four eras,
   +24.88%/yr gross at 29.8% vol on the most liquid decile of the market. **It fails
   because one round trip costs 57.8 bp and it needs 4.70.**
4. **Rebalance frequency is not a smooth knob on this signal.** The gross curve has a
   genuine trough at two weeks, confirmed in both phases. Any future study that treats
   f as a continuous breadth lever must measure the horizon profile first.
5. **The next cost defect is identified and sized:** within the liquid decile, `measured`
   cells are 25.8% of the universe and 74% of the cost, at the same dollar volume as the
   `upper_bound` cells they sit beside. That is a selection effect in *which names the
   estimator resolves*, distinct from the universe bias already fixed. It needs its own
   ground-truth calibration and its own positive control — **not** a strategy re-run.

**Do not re-run this hypothesis with adjusted lookbacks, universes, decile widths or
frequencies. That is the selection bias the apparatus exists to refuse.** Both universe
cuts, all three frequencies and both bounds were declared in advance and are all reported
above.

**Reproduce with:**
`.venv/Scripts/python.exe -m scripts.run_reversal_retest` then
`.venv/Scripts/python.exe -m scripts.verify_reversal_retest`
(13 s and ~40 s respectively; artefacts in `research/sleeves/_reversal_retest/`).
