# RESULT — LOW-VOLATILITY / QUALITY, RE-TESTED WITH BOTH COST FIXES APPLIED

**Verdict: MARGINAL.** Pre-registration `lowvol_retest_prereg.md`, committed at `0b12f93`
**before any run code existed**. One configuration, run once, nothing tuned afterwards.

DEV window only, 1998-04-30 → 2015-12-31, 213 months. No bar after 2015-12-31 was read.
`n_trials` 36 → 37 (gate evaluated at 38). Suite **1389 passed, 1 skipped**. Reproduce:
`.venv/Scripts/python.exe -m research.sleeves.lowvol_retest_run`, then
`... lowvol_retest_verify` and `... lowvol_retest_sensitivity` for §6–§7.

---

## 1. THE NUMBER THAT DECIDES IT

> **Vol-matched active return, band B2, conservative cost bound: +7.37%/yr, Newey–West
> t = +2.64.** Net Sharpe **0.878** against a benchmark Sharpe of **0.374**.
> The DSR≥0.95 bar at 17.75 years and n_trials 38 is **0.9234**.
> **It misses the bar by 0.046 of Sharpe.**
>
> **After the two accounting defects this study itself found (§6), that becomes net Sharpe
> 0.677 and vol-matched +6.52%/yr at t = +2.30 — still MARGINAL, and now 0.246 short of
> the bar.**

This is the first sleeve in the programme to produce a positive, statistically significant
excess over its own universe on the statistic that cannot be gamed by leverage — and it
does not clear the promotion gate under any accounting tested. Under the pre-committed rule
(prereg §6) that is **MARGINAL**, not PROMOTE.

Prior "alive" results died on one of four tests. This survives all four:

| test that killed a prior sleeve | what it did here |
|---|---|
| vol-matched active (killed trend, flattered PEAD) | **+7.37%/yr, t +2.64** — survives |
| benchmark through the DSR gate (passive beat trend 0.669 vs 0.612) | strategy **0.874** vs passive **0.264** |
| edge lives in the 2008–2011 crisis | **excluding** the crisis it is STRONGER: +9.37%/yr, t +3.08 |
| P&L concentration (one name-month was 13% elsewhere) | largest name-month = **0.32%** of gross P&L |

And it fails the one that matters most: **DSR**.

## 2. HEADLINE TABLE — the registered run

Holdings are identical under both cost bounds (the signal cannot see costs), so only the
cost stream differs; both books are accumulated in a single pass.

**CONSERVATIVE bound** — spread conservative + impact conservative. A result that passes
here is REAL.

| band | capital | gross | cost | net | CAGR | vol | **net Sharpe** | maxDD | DSR | bench | bench vol | bench Sharpe | bench DSR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 $200k–1M | $138k | 16.58% | 3.73% | **12.85%** | 12.43% | 14.64% | **0.878** | 49.5% | 0.874 | 8.34% | 22.27% | 0.374 | 0.264 |
| B3 $1M–5M | $685k | 10.96% | 2.25% | 8.71% | 8.17% | 12.80% | 0.680 | 55.7% | 0.675 | 8.10% | 22.60% | 0.358 | 0.242 |
| B4 $5M–25M | $3,149k | 10.18% | 1.52% | 8.66% | 8.13% | 12.70% | 0.682 | 40.9% | 0.687 | 7.95% | 21.97% | 0.362 | 0.246 |
| B5 >$25M | $19,645k | 8.71% | 1.09% | 7.63% | 7.28% | 10.71% | 0.712 | 31.8% | 0.688 | 7.71% | 21.37% | 0.361 | 0.242 |

**REALISTIC bound** — a result that fails here is DEAD:

| band | cost | net | net Sharpe | DSR | vol-matched | t |
|---|---:|---:|---:|---:|---:|---:|
| B2 | 3.30% | 13.28% | **0.907** | 0.893 | +7.80% | +2.80 |
| B3 | 1.90% | 9.05% | 0.708 | 0.710 | +4.47% | +1.51 |
| B4 | 1.11% | 9.07% | 0.715 | 0.729 | +4.48% | +1.59 |
| B5 | 0.64% | 8.07% | 0.753 | 0.734 | +4.20% | +1.69 |

Even the realistic bound's best net Sharpe (0.907, B2) is **below the 0.9234 bar**.

> **Note on the realistic bound.** Between the first execution and the final one, another
> workstream corrected the `AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME` anchors in
> `research/spread_estimation.py` (`b3c2f5a`). That schedule is used **only** by the
> realistic bound, so the realistic numbers above moved (B2 one-way 48.7 → 48.5bps, B3
> 33.5 → 33.1bps) while **every conservative number — the headline and the gate — is
> bit-identical across the change.** The numbers here are on the current committed code.
> Repeat runs otherwise agree to ~1e-9 relative; the residual noise is float summation
> order from `set` iteration and touches no reported digit.

## 3. THE MATCHED-VOLATILITY COMPARISON — and the trap it is exposing

Benchmark = equal-weight buy-and-hold of the **same band's own universe**, reported GROSS
of costs (conservative for the strategy), scaled by `k = σ_strategy / σ_benchmark`.

| band | bound | geometric excess | arithmetic active | t | **VOL-MATCHED** | **t** | k | vs rankable | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | conservative | +6.44% | +4.52% | +1.20 | **+7.37%** | **+2.64** | 0.658 | +7.40% | +2.76 |
| B2 | realistic | +6.92% | +4.94% | +1.32 | +7.80% | +2.80 | 0.658 | +7.83% | +2.93 |
| B3 | conservative | +2.52% | +0.61% | +0.15 | +4.12% | +1.39 | 0.566 | +3.73% | +1.35 |
| B4 | conservative | +2.49% | +0.71% | +0.17 | +4.06% | +1.44 | 0.578 | +3.74% | +1.40 |
| B5 | conservative | +1.76% | −0.08% | −0.02 | +3.76% | +1.51 | 0.501 | +3.55% | +1.45 |

**The three statistics disagree, and the disagreement is the finding.** Only B2 has a
significant vol-matched active return; on the *raw arithmetic* active return **no band is
significant** (t between −0.02 and +1.20) and B5's is *negative*. The gap between geometric
(+6.44%) and arithmetic (+4.52%) in B2 is pure variance drag — the mechanism that flattered
PEAD — and the gap between arithmetic (+4.52%) and vol-matched (+7.37%) is the strategy
being credited for running at 0.658× the benchmark's volatility.

**This was registered in advance as the trap this sleeve is structurally in**, which is why
the pre-committed rule reads on the vol-matched number: it equals
`(Sharpe_strategy − Sharpe_benchmark) × σ_strategy` exactly, so it is the only one of the
three invariant to leverage. The honest one-line statement is therefore **not** "+7.37%/yr"
but **"net Sharpe 0.878 versus a passive 0.374 on the same names"**.

**A correction that runs AGAINST the result.** De-levering the benchmark to 0.658× parks
34.2% in T-bills, which earn. At a 2.0%/yr risk-free rate (≈ the 1998–2015 average 3-month
bill) the vol-matched active falls by `(1−k)×rf`:

| band | vol-matched (rf = 0) | correction | vol-matched (rf = 2%) | still > +2%? |
|---|---:|---:|---:|---|
| B2 | +7.37% | −0.68% | **+6.69%** | yes |
| B3 | +4.12% | −0.87% | +3.25% | yes |
| B4 | +4.06% | −0.84% | +3.22% | yes |
| B5 | +3.76% | −1.00% | +2.77% | yes |

The programme prices at rf = 0 throughout, so the headline stays there for comparability;
the correction is recorded because it is real and adverse.

## 4. THE CAPACITY CURVE — H1's prediction, confirmed

H1 (registered in iteration 1) predicted the excess would be **larger in the less liquid
bands**, because the low-vol anomaly is an arbitrage-cost story. Vol-matched active by band,
least to most liquid: **+7.37%, +4.12%, +4.06%, +3.76%** — monotone decline. Confirmed on
the direct read; no significance is claimed on four points.

But the capacity is derisory: **B2 supports $138k of deployable capital.** The band where
the edge is significant is the band that cannot hold money. At $19.6M (B5) the vol-matched
active is +3.76%/yr at t = +1.51 — not significant.

## 5. COST DECOMPOSITION — what the two fixes were actually worth

| band | bound | spread | impact | comm+fx | total | one-way | turnover | forced exits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B2 | conservative | 3.07% | 0.47% | 0.19% | 3.73% | 54.8bp | 6.8× | 46% |
| B2 | realistic | 3.06% | 0.05% | 0.19% | 3.30% | 48.5bp | 6.8× | 46% |
| B3 | conservative | 1.81% | 0.33% | 0.11% | 2.25% | 39.1bp | 5.8× | 41% |
| B4 | conservative | 1.20% | 0.25% | 0.07% | 1.52% | 31.2bp | 4.9× | 34% |
| B5 | conservative | 0.87% | 0.17% | 0.04% | 1.09% | 31.9bp | 3.4× | 14% |

Iteration 1's B2 book paid **119.4bps one-way** (46.4 spread + 70.8 impact + 2.3
commission) on 9.06 turnovers = **10.82%/yr** of drag. This run pays **54.8bps** on 6.8
turnovers = **3.73%/yr**. Measured sources:

- **Impact**: 70.8bp → **6.9bp**. The old flat 100bps/side had no volatility term; this book
  holds names at 32.3% annualised vol against a universe at 57.8%, so feeding each name's
  own volatility cuts the charge below what the reference fallback would give. Under the
  realistic bound impact is **0.7bp** — essentially free. The fallback volatility was never
  used: **0 legs** in any band.
- **Spread**: `upper_bound` names cost **75bp** against `measured` names' **176bp** in B2,
  and the book holds 37% of them.
- **Turnover fell 9.06× → 6.8× and forced exits 70% → 46%**, because a name whose spread
  stops resolving no longer drops out of the universe. A large part of iteration 1's
  measured cost was an artefact of the measurement panel, exactly as that study warned.

## 6. TWO ACCOUNTING DEFECTS FOUND BY VERIFICATION — and the result survives both

Both are inherited from iteration 1's harness and **affect every sleeve in the programme.**

### 6a. The delisting window fires 39 times out of 3,018

`delisting_drag_annual` came back as **exactly 0.000** in three of four bands. A long-only
microcap book that never books a bankruptcy is a symptom, not a result.

The registered rule is `exit < delisting_date <= exit + 62 days`. **The ACTIONS delisting
date is typically the SAME DAY as the ticker's last SEP bar — median gap 0 days** — and the
strict `<` excludes exactly that case.

| window | last-observation cells matched | of which total losses |
|---|---:|---:|
| **registered** `(exit, exit+62]` | **39** | **14** |
| corrected `[exit, exit+62]` | **3,018** | **1,574** |

Across B2–B5 there are 7,580 last-observation cells, 6,322 carrying a delisting record whose
median terminal return is **−1.00**. The strategy held a name through its last bar 58 / 46 /
36 / 39 times (B2–B5) and booked a terminal return on **zero** of them.

### 6b. Exit legs that leave the universe are counted in turnover but charged nothing

A held name whose price falls through the $2 floor, whose dollar volume leaves the band, or
whose spread stops resolving, disappears from the tradable cross-section. Iteration 1's code
skips the cost of selling it (`if ticker not in priced.index: continue`) while still counting
the leg in turnover. **777 / 560 / 392 / 97 sell legs (B2–B5) were free.** They are now
priceable at the name's last observed cost inputs — which is still too cheap, because a name
that has just fallen out of the universe trades worse, not better.

### The sensitivity — all four accountings, conservative bound

| band | accounting | cost | net | net Sharpe | bench | **VOL-MATCHED** | **t** | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B2 | **registered** | 3.73% | 12.85% | **0.878** | 8.34% | **+7.37%** | **+2.64** | MARGINAL |
| B2 | + exits charged | 4.75% | 11.83% | 0.807 | 8.34% | +6.34% | +2.26 | MARGINAL |
| B2 | + delisting corrected | 3.73% | 11.16% | 0.746 | 5.40% | +7.55% | +2.68 | MARGINAL |
| B2 | **BOTH corrections** | 4.75% | 10.14% | **0.677** | 5.40% | **+6.52%** | **+2.30** | MARGINAL |
| B3 | BOTH | 2.74% | 7.09% | 0.546 | 5.60% | +3.87% | +1.35 | UNDETERMINED |
| B4 | BOTH | 1.79% | 8.01% | 0.634 | 5.95% | +4.60% | +1.63 | UNDETERMINED |
| B5 | BOTH | 1.17% | 6.98% | 0.648 | 6.08% | +3.94% | +1.57 | UNDETERMINED |

**Every verdict is unchanged.** Booking real bankruptcies costs the strategy 1.69%/yr and
the equal-weight benchmark 2.94%/yr — the low-vol/quality book dies *less* than its universe,
which is what the hypothesis says it should. Charging the free sell legs costs 1.02%/yr.
Net of both, B2's Sharpe falls to **0.677** and the vol-matched excess stays at **+6.52%,
t +2.30**.

Both windows and the exit-charging switch are parameters of
`research/sleeves/lowvol_retest.py`; the defaults reproduce the registered run bit-for-bit.
**The corrected accounting is the honest one and should be the default from here.**

## 7. VERIFICATION — what was checked before this was believed

1. **The harness reproduces iteration 1 exactly.** Run on iteration 1's universe (`measured`
   only) and its delisting window, this pipeline returns benchmark **10.04%/yr at 21.69%
   vol** for B2 — against iteration 1's recorded 0.10041 / 0.21694 — and 9.44% / 8.92% /
   7.98% for B3 / B4 / B5 against its 0.094376 / 0.089210 / 0.079764. Four bands, four
   decimals. The benchmark's fall to 8.34% is **entirely** the universe correction
   (−1.70%/yr from adding cheaper, marginally more liquid names to an equal-weight average);
   a further −2.94%/yr comes from the delisting fix.
2. **No return is ever invented.** All **6,210 held name-months in every band** take their
   return from the panel's own `realised_return`. The fabricated-exit path fired **0 times**.
   This answers the "forced exits" attack directly: a forced exit sells at an observed
   close, and the name's final return was already booked from the panel the month before.
3. **The added names carry no return artefact.** Mean forward return of `upper_bound` vs
   `measured` cells in B2: **7.78% vs 7.80%/yr**. They are not better stocks; they are the
   same stocks at **75bp instead of 176bp**. The correction is a *cost* correction.
4. **Sharpe per decade** (net conservative / benchmark): B2 **+0.14/+0.27** (21 mo stub),
   **+0.74/+0.33** (2000s), **+1.37/+0.54** (2010s). Not a one-regime result, and its best
   decade is the most recent.
5. **Crisis exclusion.** Dropping 2008-01…2011-12 (165 months left) *raises* B2's net Sharpe
   to **1.244** vs benchmark 0.443, and the vol-matched active to **+9.37%/yr, t +3.08**.
   Every prior sleeve went the other way.
6. **Concentration.** Largest single name-month = **0.32%** of gross |P&L| (B2); the largest
   by magnitude is a *loss* worth −1.8% of net P&L; the top 10 by magnitude are +10.6% of
   net P&L in B2 and **negative** in B3/B4/B5 — the profit comes from breadth, not from a
   handful of names. Gross notional is 3.33% max per name and 10.0% in the top 3, by
   construction, and was measured anyway.
7. **Panel forward returns.** 96.919% of consecutive panel rows are exactly one month apart;
   11,705 span more. Those cells' forward returns span the gap, both sides read them
   identically, and the property is inherited from the panel.
8. **Bracket integrity.** 0 inversions of `realistic > conservative` across all 801,341 cells.

## 8. THE UNCOMFORTABLE PARTS

- **The drawdowns are not investable at this size.** B2 max drawdown **49.5%**, worst rolling
  12 months **−43.1%**, longest stretch below its running peak of cumulative active return
  **76 months** (B3, B4: **160 months**; B5: **209 of 213**). A 6-year underperformance run
  on a 17.75-year sample is most of the sample.
- **Only B2 is significant, and B2 holds $138k.** The three bands that could hold real money
  have vol-matched t-statistics of 1.39 / 1.44 / 1.51.
- **Monthly win rate against the benchmark is a coin flip** — 54.5% / 48.4% / 50.2% / 50.2%.
  The excess comes from losing less in bad months, which is what a low-vol book does, and is
  exactly why the raw arithmetic t-statistics are insignificant.
- **17.75 years is one sample**, and the 2016+ confirmation window remains unfired.

## 9. FORECAST vs OUTCOME — the pre-registration graded

Nine of thirteen registered forecasts landed inside their stated range. **All four misses
were in the strategy's favour**, which is the direction that should attract suspicion, so
they are listed first.

| quantity | forecast | range | actual | |
|---|---:|---|---:|---|
| raw arithmetic excess | +0.5% | −3% … +3% | **+4.52%** | **MISSED (high)** |
| geometric excess | +2.5% | 0% … +5% | **+6.44%** | **MISSED (high)** |
| vol-matched active | +3.9% | −1% … +7% | **+7.37%** | **MISSED (high)** |
| vol-matched t-stat | +1.6 | +0.5 … +2.6 | **+2.64** | **MISSED (high)** |
| benchmark arithmetic | 9.5% | 8.0 … 11.5% | 8.34% | ok |
| benchmark volatility | 21% | 19 … 23% | 22.27% | ok |
| benchmark Sharpe | 0.45 | 0.37 … 0.55 | 0.374 | ok (lower edge) |
| net Sharpe, conservative | 0.74 | 0.55 … 0.95 | 0.878 | ok |
| net Sharpe, realistic | 0.79 | 0.60 … 1.00 | 0.907 | ok |
| net volatility | 13.5% | 12 … 16% | 14.64% | ok |
| one-way cost, conservative | ~40bp | 30 … 60bp | 54.8bp | ok |
| strategy DSR | 0.55 | 0.2 … 0.9 | 0.874 | ok |

**The registered "most likely outcome" was exactly right**: *"a pass on excess and a fail on
DSR is the single most likely outcome and is registered here as such."* That is what
happened. Note that under the corrected accounting of §6 the four misses shrink to roughly
their forecast values (vol-matched +6.52% vs +3.9% forecast; net Sharpe 0.677 vs 0.74) —
i.e. **the pre-registration was closer to the corrected truth than the registered run was.**

The one forecast wrong in kind rather than degree: **gross return was forecast to FALL** to
~14% because the added names are more liquid and should carry less illiquidity premium. It
**rose** to 16.58%. The added names return the same (§7.3); what changed is that the book
now selects 30 names from a median cross-section of **743** instead of **392**.

## 10. VERDICT

| band | verdict | bracket | excess | t | DSR bar | beats bench DSR |
|---|---|---|---|---|---|---|
| B2 | **MARGINAL** | real | PASS | PASS | **fail** (0.878 < 0.9234) | PASS |
| B3 | UNDETERMINED | dead | PASS | fail | fail | PASS |
| B4 | UNDETERMINED | dead | PASS | fail | fail | PASS |
| B5 | UNDETERMINED | dead | PASS | fail | fail | PASS |

**OVERALL: MARGINAL.** Not deployable, not dead.

**What it is worth, stated the way `b3c2f5a` requires** — half-Kelly growth reported with
the volatility it demands, the leverage that implies on this sleeve's natural vol, and the
measured drawdown scaled by that leverage:

| accounting | net Sharpe | half-Kelly growth | requires vol | leverage | 49.5% DD becomes | at a survivable 20% vol |
|---|---:|---:|---:|---:|---:|---|
| registered | 0.878 | 28.9%/yr | 43.9% | **3.00×** | **~87%** | 1.37×, ~15.6%/yr, DD ~61% |
| fully corrected | 0.677 | 17.2%/yr | 33.9% | **2.26×** | **~79%** | 1.34×, ~11.5%/yr, DD ~60% |

**The 28.9% is arithmetic, not a reachable return.** Reaching it requires 3× leverage on a
book that already realised a 49.5% drawdown once in 17.75 years, which is a wipeout. At a
survivable 20% volatility this sleeve compounds at roughly **11–16%/yr**, on **$138k** of
capacity. It is not a route to 30%/yr and never could be alone.

**Where its value actually is: as a third, uncorrelated sleeve.** Iteration 7 quantified the
bar exactly — trend+carry measures Sharpe 0.6546, and a third sleeve must bring
**0.883 at defensive's measured correlations, or 0.621 if genuinely uncorrelated to both**,
or it SUBTRACTS (all three of trend+carry+defensive measure 0.542, worse than two).

This sleeve is the only candidate that clears the uncorrelated bar under any accounting:
**0.878 registered, 0.677 fully corrected, against 0.621.** It does not clear 0.883 once
corrected. So its value depends entirely on a correlation that **has not been measured** —
and it is the one sleeve where low correlation is plausible on economics rather than
mechanics, because it is **US equity cross-sectional** while trend, carry and defensive are
all **futures/macro**.

**That measurement is the highest-value next step and needs no new data.** It must use the
iteration-7 discriminator (re-estimate inputs on non-overlapping lookbacks) that caught
value's correlation as an artefact, and it must be run on the **fully corrected** book
(Sharpe 0.677), not the registered one.

**Registered position: MARGINAL. Net Sharpe 0.878 as registered, 0.677 once the two
accounting defects found here are corrected — against a bar of 0.9234.**
