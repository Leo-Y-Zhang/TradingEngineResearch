# RESULT — PEAD RE-TESTED on the corrected universe

**Run:** 2026-07-28, once, under `research/sleeves/pead_retest_prereg.md` (written before
the run). `n_trials` 32 → **33**. DEV window only; the maximum bar date touched is
2015-12-31 and the 2016+ confirmation window remains **UNFIRED**.

**Reproduce:** `.venv/Scripts/python.exe scripts/run_pead_retest.py`
Artefacts: `research/sleeves/_pead_retest_output/` (results JSON/CSV, per-position frames,
daily net return series for both bounds — the process defect recorded in
`breadth_sleeve_hunt_result.md` §4 is closed for this sleeve).

---

## 1. VERDICT — **MARGINAL**

At the pre-declared 40-day headline horizon:

| bound | net return | net vol | **net Sharpe** | max DD | **excess vs own universe** | cover ratio |
|---|---:|---:|---:|---:|---:|---:|
| (a) conservative | 4.54%/yr | 12.41% | **0.366** | 46.4% | **+1.14%/yr** | 1.73 |
| (b) realistic | 5.25%/yr | 12.42% | **0.423** | 45.7% | **+1.85%/yr** | 1.99 |

Benchmark (equal-weight, zero-cost, same corrected universe): 3.40%/yr, Sharpe 0.149.
Gross Sharpe 0.847.

**The excess is POSITIVE under both bounds — the first positive excess in this
programme.** It is also far too small to matter: the gate is 0.75 net Sharpe and the
result is 0.42, so the registered verdict is MARGINAL, not a pass. **PEAD is not
deployable and the 2016+ window may not be fired at it.**

**AND THE POSITIVE EXCESS IS WEAKER THAN IT LOOKS — read §3b before quoting it.** The
registered excess statistic is a difference of COMPOUND annual returns. On the arithmetic
(mean monthly) measure, which is immune to volatility drag, the sleeve's active return is
**−0.17%/yr with t = −0.030 and p = 0.976** — indistinguishable from zero and marginally
negative. The whole +1.85pp is the benchmark compounding worse because it is twice as
volatile.

## 2. The registered prediction was HALF right, and the half that failed is the half that decided it

**H1 (net Sharpe 0.70–0.80 under bound (b)) is FALSIFIED. Measured: 0.423.**

**H2 (traded-universe median spread below 60bps) is CONFIRMED. Measured: 55.0bps under
(b)**, down from 125.9bps in iteration 1. Under (a) it is 74.9bps.

Every intermediate step of the predicted mechanism happened, in the predicted direction,
at close to the predicted size:

| 40-day hold | iteration 1 (biased universe) | (a) conservative | (b) realistic |
|---|---:|---:|---:|
| entries per year | 476.9 | 1,222.2 | 1,222.2 |
| median traded spread | 125.9 bps | 74.9 bps | **55.0 bps** |
| median traded dollar volume | $1.98M | $6.95M | $6.95M |
| **cost per round trip** | **219.1 bps** | **132.4 bps** | **115.2 bps** |
| gross alpha per bet | 256.3 bps | 229.2 bps | 229.2 bps |
| **cover ratio** | **1.17** | **1.73** | **1.99** |
| net alpha per bet | +37.2 bps | +96.8 bps | **+114.0 bps** |
| net Sharpe | 0.342 | 0.366 | **0.423** |
| excess vs own universe | **−2.97%/yr** | **+1.14%/yr** | **+1.85%/yr** |

The bill halved. Gross alpha per bet fell only 10% (256.3 → 229.2bps), so the feared
failure mode — *cost falls but the liquid names have no drift* — did **not** happen. Cover
went 1.17 → 1.99. Net alpha per bet tripled. Everything the pre-registration said would
happen, happened.

**So why did the Sharpe not follow? Because iteration 1's 0.342 was the Sharpe of a book
that was 63.8% in cash.** The corrected universe supplies 2.56× more signals, so the same
0.5%-of-equity sizing rule now funds 172.7 concurrent positions instead of 73.8 and the
cash weight falls to 37.3%. Return roughly doubled (2.46% → 5.25%) and **volatility roughly
doubled with it (7.18% → 12.42%)**. The ratio barely moved.

Iteration 1 published the number that proves this, as its own SECONDARY statistic and
without knowing what it would later mean: its **return per unit of capital actually at
risk had a Sharpe of −0.232**, against a whole-book 0.342. The whole-book figure was
mostly idle cash. On the corrected universe the sleeve is genuinely invested and the
whole-book Sharpe is 0.423 — a real improvement in the underlying sleeve, and nowhere near
0.75.

**The forecasting error is now identifiable and is worth more than the verdict: a Sharpe
prediction was made by dividing a predicted return change by an UNCHANGED volatility, when
the same correction that raised the return necessarily raised the volatility too, by
un-idling the cash.** Any future prediction of this shape must state what it assumes about
volatility and cash weight.

## 3. HONEST DECOMPOSITION OF THE SIGN FLIP — 42% of it is the benchmark, not the strategy

The excess moved +4.81pp, from −2.97%/yr to +1.85%/yr. That splits as:

| source | contribution |
|---|---:|
| strategy net return rose 2.46% → 5.25% | **+2.79pp** |
| benchmark return FELL 5.42% → 3.40% | **+2.02pp** |

The benchmark falls because the names iteration 1 deleted really are the liquid ones, and
over 1998–2015 the liquid half of the tape had *lower* equal-weight returns than the
illiquid tail. Measured directly on the DEV panel over the same span, equal-weight and
zero-cost:

| universe | cells | annual return | Sharpe |
|---|---:|---:|---:|
| `measured` only (iteration 1's universe) | 394,904 | 1.48%/yr | 0.066 |
| `upper_bound` only (newly admitted) | 522,695 | **−0.39%/yr** | −0.016 |
| both (the corrected universe) | 917,599 | 0.49%/yr | 0.021 |

(Those are over all panel names; restricted to the SF1-ARQ names this sleeve can trade,
the run measured 5.42% → 3.40%.) The benchmark move is registered, correct and required —
a strategy must be benchmarked against the universe it trades — but **a reader who takes
"the excess turned positive" as "the strategy got better" would be overstating it by
almost half.**

## 3b. THE POSITIVE EXCESS IS VARIANCE DRAG, NOT RETURN — and it is not distinguishable from zero

A decomposition of the recorded run, not a re-run and not a change to the registered
statistic. It can only weaken the result, which is the safe direction and the reason it is
here.

The registered excess is a difference of **compound** annual returns, and compounding
penalises the more volatile series. The benchmark runs at **22.80%** annual volatility; the
sleeve runs at **12.42%**, because the registered 0.5%-of-equity sizing rule leaves it
37.3% in cash. Measured exactly on the same 212 shared months:

| 40d hold, bound (b) | arithmetic (mean monthly × 12) | compound (registered) | volatility |
|---|---:|---:|---:|
| strategy | **+5.898%/yr** | +5.250%/yr | 12.42% |
| benchmark | **+6.072%/yr** | +3.401%/yr | 22.80% |
| **active** | **−0.173%/yr** | **+1.849%/yr** | TE 24.52%/yr |

**The two series earn the same average return. Every basis point of the registered +1.85pp
excess is the benchmark's extra variance drag.** On the arithmetic active return the
t-statistic is **−0.030 (p = 0.976)** and the information ratio is **−0.007**. The tracking
error is 24.52%/yr, because a 63%-invested book against a fully-invested benchmark is
mostly an asset-allocation difference, so the active return is measured with almost no
precision: this test could not have detected a real effect of the size being claimed.

**On the same arithmetic measure the correction is still a large, real improvement — it
just lands on zero rather than above it.** Iteration 1's arithmetic active return was
**−4.800%/yr (t = −0.970)**; it is now **−0.173%/yr**. The universe correction is worth
about **+4.6 percentage points a year of active return**, and that takes PEAD from clearly
losing to its own universe, to exactly matching it.

**So the honest one-line reading of the headline is: the corrected sleeve MATCHES an
equal-weight buy-and-hold of its own universe on return, and beats it on compound growth
only by holding a third of its book in cash.** Its Sharpe of 0.423 does genuinely exceed
the benchmark's 0.149, so the risk-adjusted improvement is real — and 0.423 is still less
than half the 0.911 DSR bar, so nothing about the verdict changes.

## 4. ALL THREE HORIZONS — nothing was selected on

| hold | gross alpha/bet | (a) cost | (a) cover | (a) Sharpe | (a) excess | (b) cost | (b) cover | (b) Sharpe | (b) excess | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20d | 116.9 bp | 137.0 bp | 0.85 | −0.085 | −4.14%/yr | 120.0 bp | 0.97 | 0.007 | −3.34%/yr | **DEAD** |
| **40d** | **229.2 bp** | 132.4 bp | 1.73 | 0.366 | +1.14%/yr | 115.2 bp | **1.99** | **0.423** | **+1.85%/yr** | **MARGINAL** |
| 60d | 192.9 bp | 130.3 bp | 1.48 | 0.331 | +1.46%/yr | 113.1 bp | 1.71 | 0.371 | +2.07%/yr | MARGINAL |

40d was declared the headline in advance and remains it. 60d shows a marginally larger
excess (+2.07% vs +1.85%) at a lower Sharpe (0.371 vs 0.423); the two horizons agree on
the verdict, so nothing turns on the choice. **20d does not clear its own bill under either
bound** — the drift is not there yet at one month.

## 5. DSR — the sleeve now fails the deflated bar on the GROSS side too

At `n_trials = 33` on a 17.67-year sample, DSR ≥ 0.95 demands a standalone annual Sharpe of
**0.911** (closed form reproduced against the recorded anchors: 7yr/n=32 → 1.488, matching
the internal research log).

| | iteration 1 | iteration 3 |
|---|---:|---:|
| gross Sharpe | 1.075 — **PASS** | 0.847 — **FAIL** |
| net Sharpe (b) | 0.342 — fail | 0.423 — fail |

**Correcting the universe removed PEAD's only DSR pass.** Iteration 1's gross 1.075 was
itself a property of the expensive tail. This is the second finding of the run: the one
sleeve in the programme that cleared a deflated bar cleared it on a biased universe, and
does not clear it on the honest one.

## 6. P&L CONCENTRATION — the headline is CLEAN, and one alarm is a false one that must be read correctly

**Headline (40d):** the largest single (name, month) is **−1.01%** of total net P&L under
(a) and **−0.92%** under (b); as a share of gross absolute P&L it is **0.10%**. **ZERO
name-months exceed the 3% alarm.** The top ten all sit between −0.92% and +0.81%, five of
them are LOSSES, and they span 2012–2015. The 40-day result is not carried by any name.

**LOUD CAVEAT, because the raw statistic screams and the scream is wrong.** At the **20-day**
hold the same statistic reads **−50.57% of total net P&L in one name-month, with 7,503
name-months over 3%**. That is a **degenerate denominator, not concentration**: the 20d
book's total net P&L over 17.7 years is **$11,088 on $1,000,000 of capital**, so dividing
by it explodes every share. Its share of *gross absolute* P&L is **0.08% — the least
concentrated book in the study**. Reported here in full because an automated read of that
alarm would condemn the right book for the wrong reason, and because the opposite mistake
(suppressing an alarm that looked wrong) is the one this programme cannot afford.

## 7. Breadth, and what it cost

| | iteration 1 | iteration 3 |
|---|---:|---:|
| entries per year | 476.9 | **1,222.2** |
| distinct entry days per year | 119.5 | 172.9 |
| mean concurrent positions | 73.8 | 172.7 |
| annual turnover | 2.39× | 4.16× |
| **bets per unit of turnover** | **199.6** | **294.0** |

PEAD's defining property — breadth bought from discrete events rather than from frequency —
strengthened rather than weakened: 294 bets per unit of annual turnover against 12.7 for
the frequency-driven reversal sleeve. **Breadth was never the binding constraint here and
it still is not.** 2.56× the breadth delivered +0.08 of net Sharpe.

## 8. Costs, position size, and why a small account is a different strategy

Cost decomposition at the headline hold, bound (b): **spread 66.2%, impact 30.2%,
commission 3.6%.** Under (a): 71.3% / 25.5% / 3.1%.

**Position size, stated because the IBKR minimum is what separates account sizes.** Capital
$1,000,000, cap 0.5% of *current* equity: nominal $5,000 at inception, realised **mean
$10,864 / median $8,814** (the cap tracks equity, which compounds). At the median ticket the
$0.35 order minimum is **0.40bps per side**. It still bound on **26.7% of orders**, because
it binds on any ticket under 100 shares — under $2,572 at the median entry price of $25.72.

**The same rule at small account size, computed at the median entry price:**

| account | ticket | commission per side | round trip |
|---:|---:|---:|---:|
| $10,000 | $50 | **70.0 bps** | **140.0 bps** |
| $25,000 | $125 | 28.0 bps | 56.0 bps |
| $100,000 | $500 | 7.0 bps | 14.0 bps |
| $250,000 | $1,250 | 2.8 bps | 5.6 bps |
| $1,000,000 | $5,000 | 1.4 bps | 2.7 bps |

**At $10,000 the commission alone is 140bps round trip, which exceeds the entire 114bps of
net alpha per bet this sleeve produced.** The floor stops binding for the median ticket at
roughly a **$514,000** account. Everything in this document is a statement about a
$1,000,000 account and about nothing smaller.

The impact term is the second-largest component at 30.2% and it is **known to be suspect** —
the internal research log iteration 2 records `IMPACT_COEFFICIENT` as probably an order of
magnitude too high against Frazzini-Israel-Moskowitz (2018). **It was deliberately NOT
touched here**: correcting two cost terms in one run would make the result
uninterpretable, and it would point the flattering way. It is the obvious next calibration,
and it must get its own positive control before it is pointed at any strategy.

## 9. Accounting hygiene — every prior defect checked by direct count

- **Delisting returns applied BY DATE:** 87 of 21,593 positions booked a terminal return,
  each gated on the event falling within 62 days after the forced exit. 119 positions were
  truncated; the 32 truncated-but-not-delisted names correctly booked no terminal return.
- **Names removed from holdings after booking their exit:** `open_at_end = 0` on every
  book. A long-only book cannot end holding a position it already sold, and the −112%/yr
  defect is absent by construction and by test.
- **Forward returns capped at ±100%**, with the whole daily path uniformly rescaled so the
  marked-to-market curve compounds to the capped figure (test:
  `test_return_cap_binds_and_the_daily_path_compounds_to_it`).
- **Price floor $2.00**, 90% trading-fraction test, $50k/day dollar-volume floor — all on
  the 63 bars ending at or before `datekey`.
- **Entry at the close of `datekey + 1`, never `datekey`** (test:
  `test_entry_is_strictly_after_the_filing_date`, asserts entry day 101 for a filing on day
  100).
- **Spread bracket never inverted:** 0 of 21,593 positions had `realistic > conservative`.
- **Screen is point-in-time:** corrupting every bar after the filing day leaves both spread
  bounds and the volatility unchanged (test).
- Screen rejections, distinct (ticker, filing) pairs: thin_trading 3,299; price_floor
  1,550; illiquid 850; insufficient_history 259; no_bar_after_filing 163; no_price_data 56.
  **`spread_upper_bound` is gone, and it was iteration 1's largest rejection reason by
  far — 13,248 filings, 67.9% of all its rejections and 47.4% of every candidate it
  screened.** That is the whole point of the run. 60.98% of the positions actually taken
  here are newly-admitted `upper_bound` names.

## 10. Sharpe per decade (net), and the diagnostic

| book | hold | bound | 1990s (20m, THIN) | 2000s (120m) | 2010s (72m) |
|---|---:|---|---:|---:|---:|
| top decile | 20d | (a) | −0.30 | −0.04 | −0.16 |
| top decile | 20d | (b) | −0.24 | +0.06 | −0.07 |
| **top decile** | **40d** | **(a)** | **+0.85** | **+0.34** | **+0.37** |
| **top decile** | **40d** | **(b)** | **+0.89** | **+0.40** | **+0.43** |
| top decile | 60d | (a) | +0.67 | +0.20 | +0.59 |
| top decile | 60d | (b) | +0.69 | +0.24 | +0.63 |
| bottom decile (DIAG) | 40d | (b) | +0.61 | −0.35 | −0.36 |

The 1990s column is 20 months and is **not evidence**. The two real decades agree: the 40d
sleeve is positive and small in both (+0.40 and +0.43 under (b)), which is the one
encouraging structural feature of the result — it does not live in a single crisis, unlike
the reversal sleeve's +24.6%/yr 2008–2011 concentration. It is simply too small.

**Non-registered diagnostic — the BOTTOM SUE decile at 40d, which cannot change the
verdict.** Gross alpha per bet **72.8bps against the top decile's 229.2bps**; net Sharpe
−0.305, excess **−7.86%/yr** under (b). **SUE still orders returns on the corrected
universe**, and the ordering is therefore not an artefact of the illiquid tail iteration 1
was confined to. The signal is real. It is the size of the signal relative to the cost floor
that is the problem, and it always was.

## 11. What this run establishes, and what it does not

**Establishes:**

1. **The universe bias was real, was the size claimed, and its correction does exactly what
   iteration 2 measured it would do to costs.** Round-trip cost 219.1 → 115.2bps, traded
   median spread 125.9 → 55.0bps, cover 1.17 → 1.99, compound excess −2.97% → +1.85%/yr,
   and — on the drag-immune measure — arithmetic active return **−4.80%/yr → −0.17%/yr**,
   worth about **+4.6pp a year**.
2. **PEAD's signal survives the correction.** Gross alpha per bet fell only 10%, and the
   top-vs-bottom-decile ordering holds on the corrected universe.
3. **And it still is not enough.** 0.42 net Sharpe against a 0.75 gate and a 0.911 DSR bar;
   42% of the excess sign-flip is the benchmark falling rather than the strategy rising;
   and **the entire remaining +1.85pp is variance drag — the arithmetic active return is
   −0.17%/yr at t = −0.030 (§3b).** The sleeve matches its universe. It does not beat it.
4. **The programme's one gross-side DSR pass was an artefact of the biased universe.**
   PEAD's gross Sharpe falls from 1.075 to 0.847 and now fails the bar it used to clear.
5. **A methodological correction that generalises:** predicting a Sharpe from a predicted
   return while holding volatility fixed is invalid whenever the same change alters how much
   capital is deployed. Iteration 1's 0.342 was 63.8% cash; that number should never have
   been extrapolated.

**Does NOT establish:** anything about a corrected impact coefficient (untouched, and
suspected 10× too high — if it is, the 30.2% impact share is mostly fictitious and the true
cover ratio is higher than 1.99); anything about accounts below ~$500,000; anything about
2016+, which stays unfired; and anything about a short leg, which was never registered.

**Do not re-run this hypothesis with a different decile, horizon, universe filter or
weighting scheme.** That is the selection bias the whole apparatus exists to refuse. The
legitimate next step is the impact-coefficient calibration in §8 — done as a measurement
against ground truth, at zero trial cost, *before* it is pointed at any strategy — after
which PEAD's cover ratio can be re-read from the numbers already in this document without
spending another trial.
