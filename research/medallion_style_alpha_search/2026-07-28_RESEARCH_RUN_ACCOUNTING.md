# THE HONEST ACCOUNTING — unattended research run, 2026-07-27/28

**Status: FINAL. This is the deliverable of the run, and it is a negative result.**

Private research for one person. Nothing here is financial advice, a recommendation, or an
offer. No live trading was performed or enabled at any point. No raw vendor data appears in
this document — only derived statistics, which are ours under the licence's §6.2.

**Scope.** Commits span 2026-07-27 19:59 → 2026-07-28 08:35 (**12h 36m**). The
internal mission brief was committed at 00:04 and the numbered iterations run
00:40 → 08:35 (**7h 55m**). 57 commits. Source of record for the narrative is
the internal research log, iterations 1–17.

**How this document was built.** Every number quoted below was cross-checked against the
underlying result file in `research/sleeves/`, `research/multiasset/` and
`research/medallion_style_alpha_search/`. **Where the log and a result file disagree, the
result file wins**; every such disagreement is recorded in §8. 47 + 93 + 47 = 187 individual
claims were checked; 12 came back discrepant. In addition, all 25 DSR bars quoted across the
run were independently recomputed from the shipped `sigma_SR` formula
(`research/validation.py:315-357`) and reproduce to within **0.0005**.

---

## 1. THE ANSWER, PLAINLY

**30%/yr compound is not reachable on this data.**

This is not "we did not find it". The ceiling was measured, the one remaining lever was
tested and moved it by 0.22 percentage points, and two independent constructions that
disagreed on method reached the same verdict:

- **Iteration 16** (`research/sleeves/_portfolio/portfolio_attack_lowvol_regime.json`,
  `portfolio_attack_trend_passive.json`) attacked the candidate pair on a regime argument.
- **Iteration 17** (`research/sleeves/_pair_deflation/pair_deflation.json`) attacked it on a
  deflation-and-leverage argument, on a different construction, and stated so explicitly:
  *"Two constructions, same verdict, arrived at separately. This is corroboration, not
  inheritance."*

### 1.1 Compound return is CONCAVE in leverage, and 30% is above the maximum

This is the single most important structural fact the run established
(`research/sleeves/_riskparity/result.json`, 738 months / 61.5 years, 18 instruments):

| leverage constraint | compound return | max drawdown |
|---|---:|---:|
| DD ≤ 35% | 10.56% | −32.99% |
| **DD ≤ 50%** | **12.30%** | **−47.29%** |
| DD ≤ 60% | 13.64% | −59.25% |
| any leverage (τ = 39%) | **15.83%** — the ceiling | −87.77% |
| τ = 60% | 9.48% | **−99.60%** |

Past τ ≈ 39% the curve turns **down**. Levering from a 20% to a 40% volatility target
doubles gross excess (12.99% → 26.43%) and buys **+2.17pp** of compound return, because
financing takes 3.70pp and variance drag takes 7.53pp. Iteration 14's breadth expansion
moved the peak to **16.05%** at best and **15.59%** on the headline 37-instrument panel.

**30%/yr therefore does not sit at an unacceptable point on the leverage curve. It sits
above the maximum of the curve.** No amount of leverage reaches it, because leverage is
already past its own optimum well below the target.

### 1.2 The measured ceiling numbers, with their conditions

All figures are compound annual return with financing charged, leverage solved by bisection
against the **95th percentile of drawdown across a 12-month-block bootstrap** (2,000
resamples) — not against the single observed maximum, which systematically over-levers and
flatters the answer by 7–9pp. All are after the **×0.877** engine-reconciliation factor
(a 12.3% haircut, validated against iteration 11 rather than assumed —
`pair_deflation.json attack4_engine_validation.known_optimism_factor`).

| book | financing | DD cap | leverage | compound |
|---|---|---:|---:|---:|
| **corrected lowvol+trend pair** | bill + 150bp | **50%** | 1.89× | **18.05%** |
| corrected pair | bill + 150bp | 35% | 1.30× | **13.69%** |
| **corrected pair** | **retail bill + 300bp** | **50%** | 1.82× | **16.28%** |
| corrected pair | retail bill + 300bp | 35% | 1.28× | **13.09%** |
| **trend + passive (the one survivor)** | bill + 150bp | 50% | 2.10× | **17.42%** |
| trend + passive | bill + 150bp | 35% | 1.45× | 13.98% |
| passive alone, its own 61.5 years | bill + 150bp | 50% | 1.45× | **10.80%** |
| passive alone, the pair's 213 months only | bill + 150bp | 50% | 1.15× | **5.80%** |
| passive alone (iteration-11 vol-targeted ladder) | — | 50% | ~1.9× | **12.30%** |

**Unconstrained peak: 60.82%/yr at 7.45× leverage on a −91.12% drawdown.** That is account
death, and it is quoted only to be discarded under standing rule 7. **Note that this figure
belongs to the pair *as originally claimed*, not to the corrected pair** — see §8, D-11. The
*corrected* pair's unconstrained peak is 40.44%/yr at 6.25× on a −80.61% drawdown.

**The highest survivable number in the table, 18.05%, comes from a book that FAILS its own
deflation gate.** The book that passes — trend+passive — gives **17.42%**. This ordering
matters and is stated first deliberately: the best-looking number is the one attached to the
dead book.

### 1.3 The target expressed as what it actually is

30%/yr at half Kelly is a **Sharpe number**: `3S²/8 = 0.30` ⟹ **S = 0.894**. 60%/yr ⟹
S = 1.265. No leverage decision changes this. Against 0.894 the programme measured:

| book | Sharpe | sample |
|---|---:|---|
| trend + passive (survivor) | **0.9033** | 738 mo / 61.5 yr |
| corrected lowvol + trend pair | 0.9212 | 213 mo / 17.75 yr — **fails deflation at every trial count** |
| passive equal-weight, monthly | 0.6691 | 738 mo / 61.5 yr |
| trend + carry, equal risk | 0.6546 | 269 mo / 22.4 yr |
| trend alone, net 10bps | 0.6116 | 738 mo |
| corrected low-vol B2 | 0.6138 | 213 mo |
| carry alone, net 3bps | 0.4301 | 269 mo |
| defensive / BAB, net 10bps | 0.1136 | 629 mo / 52.4 yr |

Two books nominally exceed 0.894. One fails its deflated bar. The other clears everything —
and is **72.2% passive by risk weight** (§3).

---

## 2. EVERY STUDY, WITH VERDICT

### 2.1 The night: 19 trial-spending studies

Sample lengths are the study's own. "Headline" is the statistic the study was gated on.

| # | study | data | sample | headline | verdict | what killed it |
|---|---|---|---|---|---|---|
| 1 | Short-horizon reversal, weekly L/S | Sharadar SEP, DEV | 17.7 yr, 921 weekly cross-sections | net Sharpe **−3.6189**, excess **−52.81%** | DEAD | 45× turnover; universe bias; 158.6bp cost vs 15.3bp alpha |
| 2 | Insider clustering (SF2) | Sharadar SF2, DEV | 7.7 yr, 92 mo | **−0.1200**, excess −8.69% | DEAD | 235.5bp cost vs 93.5bp alpha. The *clustering* hypothesis is specifically refuted: ≥2 buyers minus 1 buyer is +1.05%/yr at t 0.57; the whole step is 0 → 1 buyer (+5.57%/yr, t 3.49) |
| 3 | PEAD, SF1 ARQ SUE 40d | Sharadar SF1+SEP, DEV | 17.7 yr, 212 mo | **+0.3422**, excess −2.97% | DEAD | 219.1bp cost vs 256.0bp alpha; cover 1.17 |
| 4 | Time-series momentum, multi-timeframe | Sharadar SEP, DEV | 17.0 yr, 205 rebals | **−0.3682** (gate-eligible), excess −8.79% | DEAD | universe N_eff **9.29 of 206 names**; nominal breadth 7,408/yr collapses to 98/yr |
| 5 | Institutional flow (SF3 13F QoQ) | Sharadar SF3, DEV | **2.2 yr, 9 rebals** | **−0.4447**, excess −6.54% | DEAD | 4 bets/yr; sample too short to gate at all |
| 6 | Low-vol / quality composite, B2 | Sharadar SEP+SF1, DEV | 17.8 yr, 207 rebals | **+0.3244**, excess −5.54% | DEAD | 119.5bp cost vs 58.2bp alpha |
| 7 | **PEAD re-test**, corrected universe | same, DEV | 17.67 yr, 212 mo | net **0.4227** (realistic bound), excess **+1.85%** | DEAD | the positive excess is **variance drag**. Arithmetic active **−0.173%/yr, t −0.030, p 0.976** |
| 8 | **Reversal re-test**, corrected universe | same, DEV | 17.6 yr | net **−0.854** realistic / −2.333 conservative | DEAD | gate budget **4.70bps per round trip**; IBKR commission alone is 3.06bps; measured 57.8bps = 12.3× over. 10 of 12 configurations have a *negative* budget |
| 9 | **Multi-asset trend** | free long-history panel, 18 instruments | **738 mo / 61.5 yr** | net10 **0.6116**; **first sleeve ever to clear DSR** (bar 0.486) | DEAD | loses to its own universe levered to equal vol: **−0.51%/yr, t −0.31**. Post-2009 Sharpe **0.180 vs benchmark 0.777** |
| 10 | **Cross-asset carry** | free FX spot + FRED OECD rates, 13 instruments | 269 mo / 22.4 yr | net3 **0.4301**, vol-matched active **+1.60%/yr, t 1.22** | MARGINAL, not deployable | DSR bar **0.813 not cleared**. Leave-one-instrument-out: dropping **FX_JPY → 0.267**, below its own DEAD line. It is one perpetual short-yen position |
| 11 | Cross-asset value | same panel | 605 mo / 44.4 yr | net10 **−0.0824**, arithmetic active **−9.84%/yr, t −2.80** | DEAD | anti-predictive on equity indices (rank IC −0.0684, t −3.14). Its negative correlation to trend was **construction overlap**: −0.1645 → **−0.013** once the overlapping 12 months are removed |
| 12 | **Low-vol / quality re-test** | Sharadar, DEV | 213 mo / 17.75 yr | registered net **0.8779**; corrected **0.6138** | MARGINAL | misses the **0.9234** DSR bar. Verification found it is a **bear-market payoff**: excluding both bears takes it to +4.87%/yr at t 1.45, below its own gate |
| 13 | Calendar seasonality (3 effects + composite) | free panel, 18 instruments | 736 mo / 61.33 yr | composite net10 **0.4680** vs benchmark **0.7067** | DEAD | vol-matched active negative in **every** cell; geometric excess negative too — the two statistics that fail in opposite directions agree |
| 14 | Defensive / betting-against-beta | free panel | 629 mo / 52.4 yr | net10 **0.1136**, DSR **0.089** vs bar 0.530 | DEAD | it *is* levered bonds plus long dollar (53.6% of gross in bonds, 24.3% USDX, correlation 0.631 to a plain bond book, alpha over levered bonds −0.77%/yr t −0.27) |
| 15 | **Risk parity + the leverage ladder** | free panel | 738 mo / 61.5 yr | RP **0.6483** vs equal-weight **0.6678** | DEAD | RP loses to equal weight at matched vol, **−1.35%/yr t −2.67**, at zero trading cost. Outside the 1981–2021 bond bull RP falls 0.648 → **0.160** |
| 16 | **Breadth expansion** (37 instruments) | free, +19 series | full sample + 2011+ | N_eff **5.13 → 8.38**; peak compound **16.05%** best, **15.59%** headline | ROUTE REFUTED | per-bet Sharpe **fell 0.279 → 0.169**, so S went 0.632 → 0.489. The assets that add independence are the ones with worse signal |
| 17 | Portfolio combination search v1 | 5 sleeves + passive | 143-mo common window | 234 configurations searched, best claimed **1.163** | OVERFIT | the window quadruples defensive's Sharpe (0.114 → 0.534); the DSR bar at 11.9 yr is **1.138**, before deflating for 234 |
| 18 | **Portfolio combination v2 + regime attack** | same, realigned | 213 / 738 mo | lowvol+trend **0.9260** vs bar **0.9422** | DEAD | low-vol's diversification is the **1998–2016 stock/bond regime**, not the sleeve. A long-history equity proxy reproduces the entire benefit on that window and loses it off it |
| 19 | **Pair deflation** | same | 213 mo | corrected pair **0.9212** vs bars 0.9443 / 1.0872 / 1.1022 | DEAD | fails all three, including at n = 47 alone. The dominant defect was a **stale input**, not the alignment bug |

### 2.2 The night: 6 measurement-apparatus and verification studies (no trial spent)

| # | study | scale | outcome |
|---|---|---|---|
| 20 | Spread universe-bias repair | 922,652 (name, month) cells | **525,933 cells (57%) were being deleted** — the cheapest 57%. Fixed to a two-bound bracket; positive control 4/4, later 5/5 |
| 21 | Long-history panel build | 31 instruments, 98.6 yr headline | Built. Caveat carried: 98.6 years is **one instrument**; a genuine multi-asset test starts ~1965 with 8 |
| 22 | Impact-model recalibration | 28,992,477 SEP bars | Flat 100bps/side replaced by `Y·σ·√(Q/V)`. Old model rejected by its own control (§4, D-2) |
| 23 | Spread-anchor correction | 1,364,189 cells | AGK anchors were mapped to the wrong quintile definition; error +3.4% to **+583%**. Worth only 1–4bps (§4, D-3) |
| 24 | **Independent verification of low-vol B2** | 536-line report | Reproduced bit-for-bit (max Δ 7.1e-15) and from an independent re-implementation (Δ 0.0). **Three headline levels wrong**; verdict survives as MARGINAL, `PROMOTE` unavailable under any accounting |
| 25 | Long-history synthesis | re-measures banked studies + 5 new diagnostics | Took away three things: the 4-seed negative control (6.1σ → **2.35σ** at 200 seeds); leave-one-instrument-out (never run, and it breaks carry); the "ZIRP regime" diagnosis (contradicted by carry's own decade decomposition) |

**Two studies died to their own verifier: PEAD (iteration 3, killed by iteration 2b) and
multi-asset trend (iteration 3, killed inside its own study by the vol-matched test).** A
third, low-vol B2, survived its verifier but at materially lower numbers.

### 2.3 The 14 studies that preceded the night

| # | study | data | sample | headline | verdict |
|---|---|---|---|---|---|
| P1 | Fama-French factor loadings | 30 large caps (survivorship) | 84 mo, 2015–2025 | best (size) Sharpe 0.73, **DSR 0.83** | FAIL |
| P2 | Learned cross-sectional combination | 30 large caps + EDGAR | 2015–2025 | Sharpe 0.92, **DSR 0.467**, IC +0.113 | FAIL (default-deny) |
| P3 | Free richer fundamentals, 14 factors | 140 EDGAR names | 243 mo, 2006–2026 | Sharpe 0.67, **DSR 0.543** | FAIL |
| P4 | Insider (SEC Form 4) alpha | 140 names | 233 mo, 2007–2026 | Sharpe **−0.35**, DSR 0.007 | NOT-DEPLOYABLE |
| P5 | Sharadar survivorship-free fundamentals, full sample | 21,916 names | 329 rebals, 1999–2026 | Sharpe 0.52, DSR **0.905**, rank-IC **−0.0373** | FAIL; later downgraded as the same micro-cap artefact class |
| P6 | Sharadar DEV baseline | 14,591 names | 202 rebals | Sharpe **1.33**, DSR 1.000, rank-IC −0.0155 | FAIL on rank-IC — and the rank-IC criterion caught what DSR 1.000 and PBO 0.00 missed |
| P7–P12 | Sharadar liquidity ladder (dev1, dev1b, dev2a, dev2b, dev2c, dev3) | top-1000 / top-500 / top-1500 / $5M floor / 20bp | 6 runs | net Sharpe **−0.28 to −0.02**; rank-IC now +0.010…+0.018 | ALL DEAD. The divergence collapsed and the real edge (~+0.013 rank-IC) is unmonetisable |
| P13 | Track C: engine on Sharadar prices | 8 megacaps | — | 18.37%/yr, Sharpe 1.15, DD 17.09% — reproduces the yfinance headline to ~0 | Not an alpha study; robustness PASSED |
| P14 | Capacity curve | Sharadar, DEV | 6 bands, $32k → $99.9M | Spearman ρ **−0.943, p 0.0080** | H1 SUPPORTED and **NOT DEPLOYABLE** — every band loses to buy-and-hold, −2.6%/yr to −15.5%/yr |

**Cumulative trial ledger: 26 entering the night, 47 leaving it.** The ledger is not fully
auditable — see §4, D-8.

---

## 3. THE ONE SURVIVOR

**trend + passive, inverse-volatility weighted.**
Source: `research/sleeves/_portfolio/portfolio_attack_trend_passive.json`,
`portfolio_longhistory_books.json`.

| property | value |
|---|---|
| Sharpe | **0.9033** over **738 months (61.5 years)** |
| volatility, 1× | 8.99% |
| CAGR, 1×, total return | 13.07% (mean excess 8.12%/yr over a 4.63%/yr cash rate) |
| max drawdown, 1× | −15.66% |
| DSR bars cleared | n = 46 (0.4988), **n = 104 (0.5378), n = 304 (0.5840)** — all cleared |
| decades | **positive in all seven** |
| vs its own benchmark at matched volatility | **+2.11%/yr, t +2.34** |
| trend alone vs the same benchmark | **−1.31%/yr, t −0.31** |
| first half / second half | 0.9218 (1965–1995) / 0.8840 (1995–2026) |

**This is the best result the programme has ever produced and it is a genuine one.** The
comparison that makes it real is the last two rows: the blend beats a benchmark that trend
*alone* loses to. It is not a repackaged beta.

### Its limits, stated plainly

1. **It is 72.2% passive by risk weight** (`weights`: passive 0.7218, trend 0.2782). The
   survivor is mostly the thing everything else lost to, with a 28% trend overlay.
2. **Half-Kelly is unreachable.** 3S²/8 = 30.6% theoretical growth requires 45.2%
   volatility = **5.02× leverage**, giving a **−78.8% measured / −92.8% bootstrap**
   drawdown. Ruin at **6.50×**. The honest rung is 2.10× → **19.86%** engine, **17.42%**
   after the ×0.877 reconciliation.
3. **The bond bull carries part of it.** Excluding 1981-10 → 2021-12 takes it to **0.8245**
   (passive falls 0.6691 → 0.4387 over the same exclusion, so the blend degrades less than
   its dominant leg — but it does degrade).
4. **The confidence interval spans the target.** Analytic 95% CI **[0.65, 1.16]**;
   block-bootstrap [0.658, 1.157]. 0.894 sits inside it. The point estimate clears; the
   interval does not settle the question.
5. **Three of seven decades score below 0.894**, and the weakest is the most recent full
   one: the 2010s at Sharpe **0.458**, 3.43%/yr.
6. **Survivorship.** The 18 instruments are hindsight-selected survivors and the passive leg
   inherits that bias in full. Direction is toward optimism.
7. **It was found by a search over 234 configurations** and was not pre-registered. It is
   the correct next thing to pre-register — it is not a validated strategy today.

---

## 4. DEFECTS FOUND IN OUR OWN MEASUREMENT STACK

**This is the most valuable output of the run.** Twelve of the fifteen items below change a
measured number; four of them change a verdict or a headline. Every one was found by an
impossible number or a control, not by a passing test.

*Repair status, as of 2026-07-28 09:15 — this section records what the run FOUND; repairs
began landing after it ended. **D-7 (dating) is fixed; D-11 (delisting) too,
in a repair which found the same off-by-one in SIX live call sites including the verifier
that was supposed to catch it; D-9 (12-month leverage) is in flight.** Everything else below
is open.*

### 4.1 In the cost model

**D-1 — THE UNIVERSE BIAS. 525,933 of 922,652 eligible (name, month) cells were being
deleted, and they were the cheapest 57%.**
`spread_with_resolution` returns `upper_bound` for liquid names — meaning the true spread is
*below* the estimator's resolution floor, i.e. the name is CHEAP — and the instruction was to
exclude them. Measured (`scripts/measure_spread_universe_bias.py`):

| | kept by iteration 1 | deleted by iteration 1 |
|---|---:|---:|
| median dollar volume/day | $850,300 | **$5,408,200** (6.4×) |
| median share price | $15.99 | $21.27 |
| median spread, realistic bound | 153.1bps | **36.2bps** (0.24×) |

The rule admitted a name *precisely when it was expensive*. Universe 396,719 → 922,652 cells
(**+132.6%**). Effect on re-tested sleeves: PEAD cost/round-trip 219.1 → 115.2bps, excess
−2.97%/yr → +1.85%/yr; reversal cost 158.6 → 57.8bps, Sharpe −3.619 → −0.854. **Neither
survived anyway** — which is the point: the repair removed the cost excuse and the signals
still had nothing.

**D-2 — THE IMPACT OVERCHARGE. 17.9× a measured live-execution all-in cost.**
`capacity_study.IMPACT_COEFFICIENT = 0.1` with `impact = 0.1·√(participation)` charged
**100bps per side / 200bps round trip at the registered 1% participation cap, flat,
regardless of liquidity or volatility** — the formula carried no `σ` term at all. Against
Frazzini, Israel & Moskowitz (2018) Table II Panel A (median **5.54bps** all-in one-way on
$1.7tn of live US large-cap executions at 0.9% of ADV), at FIM's own conditions the old
model charges **94.87bps of impact**, and **99.37bps of total one-way cost including our own
half-spread — 17.94×**. (Impact alone against the all-in figure is 17.1×; the 17.9× quoted
throughout the log is the total-cost ratio. Both are recorded here so the comparison is not
overstated.) Check E of the new control exists precisely to reject the old coefficient, and
does.

Replaced by `Y·σ·√(Q/V)` (Tóth et al. 2011, PRX 1 021006 eq. 1), bracketed
conservative 0.358070 / realistic 0.041842, inputs measured on 28,992,477 SEP bars. The
charge is now 6.78–24.09bps/side depending on the name's volatility (a 3.56× spread) versus
a flat 100bps before (1.00×). Worth **60–70bps per round trip** on the two sleeves that used
the flat model and 6–33bps on three that used Y = 1.0. **It moved exactly one verdict**:
low-vol/quality B2's excess went −5.54%/yr → −0.13% (conservative) / +0.75% (realistic) —
still failing its registered 2%/yr excess gate.

**D-3 — THE SPREAD ANCHOR ERROR, and the framing error on top of it.**
`AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME` claimed to place Ardia-Guidotti-Kroencke Table 4 Panel
C's five quintile spreads at the dollar volume those quintiles trade at. It did not: AGK
quintile by **market capitalisation** over an explicitly **unscreened** universe, and the
anchors were the **dollar-volume** quintiles of a **liquidity-screened** universe. Quintile
*k* of a screened universe is strictly more liquid, so every spread level was pinned too far
right:

| AGK quintile | anchor as shipped | measured, AGK's own definition | error |
|---|---:|---:|---:|
| Q1 (3.14%) | $153,850 | $22,511 | **+583%** |
| Q2 (2.09%) | $732,550 | $207,734 | +253% |
| Q3 (1.08%) | $2,758,144 | $1,760,933 | +57% |
| Q4 (0.30%) | $9,557,127 | $9,246,530 | +3.4% |
| Q5 (0.09%) | $55,046,590 | $63,098,932 | −12.8% |

**And the "3.6–4.4× too dear" size originally put on it was wrong by more than an order of
magnitude.** Bound (b) is `min(estimate, schedule)`, and in $1M–$5M the EDGE estimate is
already the cheaper of the two **84% of the time**. Real effect: **−1.5 / −3.5 / −0.9 /
+0.2 bps** round trip across bands B2–B5. **No verdict moves.** The structural finding is
the useful part: *the spread schedule was never what made the illiquid bands expensive — the
EDGE measurement is.* Cutting the schedule to the minimum legal tick would relieve only a
further 14 / 16 / 8 / 3bps per side.

A residual is left standing and explicitly **not** closed: at the median Russell-2000
constituent ($3.31M/day) the corrected schedule still charges 33.1bps/side against FIM's
13.53bps all-in (2.4×). Three candidate causes are named and none is measured, so it is
disclosed rather than fudged. **The schedule stays dear.**

### 4.2 In the statistics

**D-4 — THE VARIANCE-DRAG ILLUSION. It killed PEAD.**
Geometric excess = arithmetic excess − (σ_s² − σ_b²)/2. The corrected PEAD book ran at
**12.42% volatility against its benchmark's 22.80%** because the sizing rule left 37.3% of
the book in cash. On compound returns it showed +1.85%/yr of excess. On the drag-immune
arithmetic measure over the same 212 months: strategy +5.898%/yr, benchmark +6.072%/yr,
**active −0.173%/yr, t −0.030, p 0.976, IR −0.007**. The two series earn the same average
return. This is exactly the shape that passes a careless review.

**D-5 — ITS HIGH-VOLATILITY TWIN. It killed multi-asset trend's headline.**
Requiring *arithmetic* active return to defeat D-4 created the mirror trap. Trend reported
**+8.07%/yr active at t(NW) 2.54** — and that is leverage, not alpha: the book runs at 22.8%
volatility against an 8.79% benchmark. The tell is measured:

| vol target | 10% | 20% | 40% | 60% | 120% |
|---|---:|---:|---:|---:|---:|
| arithmetic active | 1.21% | **8.07%** | 20.24% | 26.76% | 32.49% |
| active t(NW) | 0.63 | **2.54** | 3.64 | 3.85 | 3.99 |
| strategy's OWN t | 4.60 | 4.66 | 4.73 | 4.69 | 4.68 |

**As leverage rises the "active t" converges to the strategy's own t** — it stops testing
"does this beat the benchmark" and tests only "is the return positive". A scale-dependent
active-return test is not a comparison. Vol-matched, trend's active return is
**−0.51%/yr at t −0.31**. Jensen alpha is no defence either: 13.87%/yr at t 4.63 with beta
0.013, and 0.612 × √61.5 = 4.80 — with zero beta the alpha t-stat *is* the strategy's own
Sharpe t-stat.

> **PEAD faked GEOMETRIC excess by being LOWER vol than its benchmark.
> Trend faked ARITHMETIC active return by being HIGHER vol. Same trap, opposite sign.**

The settled statistic: `volmatched_active = bench_vol × Sharpe_gap`, verified exactly
(residuals 0.0, 2.8e-17, 1.1e-16 on three books). The matched-vol test **is** the Sharpe
comparison, and therefore cannot reverse a ranking — which is why it is the right one.

**D-6 — THE DSR GATE HAS NO BENCHMARK-RELATIVE CRITERION.**
Confirmed directly from source. `research/validation.py:401-467` applies seven checks, all
absolute: `mean_rank_ic > 0.01`, `sharpe_net > 0.75`, `stability_score > 0.60`,
`deflated_sharpe_proxy > 0.25`, `deflated_sharpe_ratio >= 0.95`, no leakage flags, no regime
Sharpe < −0.50. **Nothing compares the book to buy-and-hold** — no active return, no alpha,
no information ratio, no excess-over-benchmark term. `deflated_sharpe_ratio` *accepts* an
`sr_benchmark` argument (`validation.py:315`), but the gate never supplies one, so deflation
is always against the multiple-testing threshold and never against a passive book's Sharpe.

The gate has now failed in **both** directions on measured data:

| | strategy | passive benchmark |
|---|---:|---:|
| multi-asset trend, net DSR | **0.612 — passes** | **0.669 — passes by more** |
| defensive / BAB, DSR | **0.089 — rejected** | **0.994 — would have passed** |

*"It would have promoted a strategy that is WORSE THAN DOING NOTHING."* Every prior study
that "failed DSR" was failing a test that could not have distinguished skill from beta
anyway. **Never quote a DSR without the benchmark's DSR beside it.**

(Incidental: the docstring says "ALL six conditions" and lists six, while the body checks
seven — `deflated_sharpe_ratio >= 0.95` is enforced but undocumented.)

### 4.3 In the data plumbing

**D-7 — THE ONE-MONTH DATING DEFECT. Invisible to within-series statistics; it survived a
bit-for-bit independent verification.**
`research/sleeves/lowvol_retest.py::run_band` labelled each monthly slot by the **formation**
month and filled it with `forward_return`, the **following** month's return. Proved against
correctly-dated SPX: the low-vol book reads rho **+0.189 contemporaneous** and **+0.769 at
SPX(t+1)**.

Mean, volatility, Sharpe, drawdown, Newey-West t and the vol-matched active return are **all
invariant to a constant shift**. That is why iteration 10's independent re-implementation
reproduced the series to 0.0 and never saw it. It becomes visible only when the series is
joined to another **by date**:

| correlation | before | after |
|---|---:|---:|
| rho(lowvol, carry) | −0.018 | **−0.392** |
| rho(lowvol, passive) | +0.317 | **+0.571** |

Scope: it does **not** change low-vol's standalone verdict. It **invalidates every
cross-series join** involving that output, past or future — which is the entire content of
iterations 13 and 15.

**D-8 — THE STALE TRIAL LEDGER, and a ledger that is not auditable.**
`research/sleeves/_portfolio/portfolio_decision.py:32-33` hard-codes
`N_TRIALS_PROGRAMME = 38` with the comment *"the count the low-vol gate was evaluated at"*,
plus `N_COMBOS_SEARCHED = 31`, so the pair was deflated at **n = 69** when iteration 14 had
already moved the ledger to **47** (correct combined count: 78). Worse, the ledger's own
history does not reconcile: it runs 26 → 32 → 33 → 34 → 36 → 38 → **[unexplained +6]** → 44
→ 46 → 47, and the long-history synthesis admits *"the register is being double-counted
across parallel agents — trend and carry each independently wrote n_trials = 36 from a base
of 34"*. The bars are nearly flat in *n* (six extra trials move the 22.4-year bar by 0.015
against 0.33 for sample length), **so no verdict turns on this** — but a trial ledger that
cannot be reconstructed is not a control.

**D-9 — THE 12-MONTH FULL-LEVERAGE BUG.** In the scaler shared by every sleeve on
`multiasset_trend`'s code path: `k = min(vol_target/σ_book, GROSS_CAP/gross)`, and for a
book's first **12 months** `σ_book` does not exist, so `k` falls through to the cap and the
book runs at **full 10× at every volatility target identically**. Twelve months out of 629
move defensive's full-sample Sharpe by **0.050**. Changes no verdict (all dead) — must be
fixed before any survivor is trusted.

**D-10 — UNPRICED SELL LEGS.** A name leaving the tradable universe still has to be sold.
**777 of B2's 1,783 exits (43.6% of exits, 21.4% of all 3,624 legs traded) were counted in
turnover and charged nothing**; 560 / 392 / 97 in B3–B5. Charging them costs **1.02%/yr**.

**D-11 — THE DELISTING WINDOW OFF-BY-ONE.** The registered rule is `at < delisted_on <=
at + 62 days`, but the vendor dates a delisting **on the ticker's last traded bar** — median
gap **0 days** — so the strict `<` rejects the modal case. Measured across B2–B5: the window
fired **39 times out of 3,018**; of **7,580** last-observation cells, **6,322** carry a
delisting record whose median terminal return is **−1.00** that was never booked; the
strategy held names through their last bar **179 times** and booked a terminal return on
**zero** of them. `delisting_drag_annual = 0.0` was a dead code path, not a finding. The
repair is one character. Repairing it *raises* the low-vol headline (+7.37% → +7.54%),
because the bug had been paying the **benchmark** more than the strategy.

**D-12 — FUTURES ROLL CONTAMINATION: 5 of 9 new series are worse than the NATGAS_F the
original panel builder condemned.** Single-day variance ratio against a clean-futures noise
floor of 1.46–2.19 (GOLD_F 1.46, WTI_F 2.01, SILVER_F 2.08, COPPER_F 2.19):

| series | ratio | | series | ratio |
|---|---:|---|---|---:|
| **HOGS_F** | **10.11** | | SUGAR_F | 3.71 |
| **CATTLE_F** | **7.55** | | SOYBEAN_F | 3.52 |
| **CORN_F** | **4.03** | | *NATGAS_F (condemned)* | *2.91* |

Attribution measured, not asserted: for CORN_F the roll window (d14–15) carries 15.8% of
variance at ratio **2.38** while the USDA WASDE news window (d9–12) carries 14.6% at ratio
**1.08**. Corroborated against roll-managed funds a person could actually have held: **WEAT
earns 12.23%/yr less than ZW=F front-month; COW earns 19.36%/yr less than HE=F.** The
un-back-adjusted front-month series **manufacture return**. **Dropping the five contaminated
series takes N_eff from 8.38 to 6.76 — half the measured independence was splice noise** —
and the `expanded_roll_managed` run peaks at **13.13%**, *below* the starting point.

**D-13 — FIVE OF SIX ITERATION-1 SLEEVES PERSISTED NO RETURN SERIES.** Only PEAD wrote one.
Cross-sleeve correlation was therefore uncomputable at the time, which is the direct cause
of the correlation matrix having to wait until iteration 13 and then running on a 143-month
common window that flattered every sleeve in it.

**D-14 — TWO PANEL-LEVEL ARTEFACTS INHERITED BY EVERY MULTI-ASSET SLEEVE.** (a) The equity
indices are **price** indices, not total-return indices, and national dividend yields
differ, so any long-horizon return signal ranks high-dividend markets as permanently cheap —
measured symptom: **FTSE100 was long in 93% of months**. (b) Inverse-volatility sizing put
**65% of gross notional in three US Treasury points, US5Y alone 42.7%**: the sizing rule, not
the signal, was choosing the portfolio.

**D-15 — ELEVEN ACCOUNTING DEFECTS IN ITERATION 1 ALONE**, every one caught by an impossible
number rather than by a test. The dominant one recurred in **four** sleeves independently:
`build_monthly_panel` stamps each ticker's *own* last bar of the month, so iterating distinct
panel dates as a calendar produces phantom periods, singleton cross-sections and, when a
delisting lands mid-month, a −100% month for an entire universe. Others: costs read only
from currently-selected names charge buys and not sells (doubling the insider sleeve's true
turnover, 2.9× → 5.8×); a $1.00 notional equity makes every trade hit the 1%-of-value
commission cap and manufactures 27–72%/yr of fictitious cost; exits of names that left the
universe carry a NaN spread that silently zeroes their liquidation cost; and pandas parses
SF2's `securityadcode` value `'NA'` as missing, so reading SF2 without
`keep_default_na=False` matches zero open-market purchases and the whole study measures
nothing while appearing to run.

### 4.4 The errors in the run's own synthesis

These are errors in the run's own synthesis and projections, not in the measurement stack. They are listed here because the run's credibility
rests on them being visible.

**M-1 — THE CONSTANT-`s` ASSUMPTION, refuted by iteration 14.** Iteration 12 projected the
ceiling upward by raising effective breadth while holding per-bet Sharpe fixed, and named
N_eff ≈ 13 as the requirement. The test was run. **N_eff rose exactly as predicted, 5.13 →
8.38 (+3.25), and bought nothing, because per-bet Sharpe FELL 0.279 → 0.169.** So
`S = s·√N_eff` went **down**, 0.632 → 0.489. The synthesis predicted a 25.2% measured peak on that
window; the measured peak is **6.52%**. The ceiling *model* is sound — `0.71·S²/2` predicts
the expanded panel's peak to within **0.76pp** — the constant-`s` assumption was the error,
and it was mine.

**M-2 — THE EQUAL-SHARPE PORTFOLIO FORMULA OVERSTATES BY 45%.**
`S = s·√(N/(1 + (N−1)ρ))` assumes equal sleeve Sharpes. With the measured ones it predicts
**1.036 against a measured 0.714**. Every three-sleeve projection quoted during the night
was inflated by that formula. **Use the measured covariance; never the shortcut.** (Where
two sleeves have equal volatility the formula reproduces the measured number to 2.2e-16 —
that is algebra, not validation, and it was initially reported as agreement.)

**M-3 — HALF-KELLY GROWTH WAS QUOTED AS A REACHABLE RETURN.** `3S²/8` is correct arithmetic
and misleading as a deployable number, because it silently assumes running at σ = S/2. On
the best result of the night (low-vol B2, Sharpe 0.878, natural volatility 14.6%, measured
max drawdown 49.5%):

| target | implied vol | leverage | implied max DD | verdict |
|---|---|---|---|---|
| half-Kelly, 28.9%/yr | 43.9% | **3.0×** | ~100%+ | **RUIN** |
| 20% vol | 20% | 1.37× | ~68% | survivable, barely |
| natural 14.6% | 14.6% | 1.0× | 49.5% | measured |

At 1.37× the compound return is roughly 15–17%/yr, not 28.9%. **Kelly assumes continuously
rebalanced IID lognormal returns; real drawdowns are fat-tailed and path-dependent, and a
strategy that is dead cannot compound at any rate.**

**M-4 — THE SPREAD-VS-IMPACT COMPARISON WAS MISLABELLED, twice in one note.** (a) *"our SPREAD
schedule charging 48.9–58.9bps per side"* was not the spread schedule — those are the impact
control's **total** one-way figures, half-spread **plus** impact; the spread part is 46.9bps.
(b) FIM Table II Panel A is **market impact** (spread + impact, measured from arrival price),
**not implementation shortfall** (which adds delay — the 1.05bps gap between the two means).
Both `capacity_study.py` and `impact_positive_control.py` describe it as "spread + impact +
delay". The comparison *quantity* was right; the docstrings credit the benchmark with a cost
component it does not contain, which makes our costs look closer to theirs than they are.

**M-5 — THE 58-VS-65 FIGURE, and the correction was wrong too.** Iteration 13 reported
"58 of 234 combinations clear 0.894". Iteration 16 "corrected" this to 65. **Neither is a
correction of the other, because 58 was never a clearing count**: 58 is the number of sleeve
**subsets** (58 subsets × 4 weighting schemes = 234 configurations). No committed file
records 58 clearing anything. What the files record is: v1 gives 28 of 93 (max-overlap
window) and 84 of 189 (common window); `portfolio_decision.json` records 74 of 189; v2
records **65 of 234**. So a subset count was reported as a clearing count, then "corrected"
into a different, genuine clearing count from a different file, and the log records the
change as a numerical correction rather than a category error.

**M-6 — ITERATION 13's CORRELATION TABLE DOES NOT REPRODUCE.** The synthesis called the correlation
structure *"the genuinely useful finding here, not the Sharpe"*. **Six of its seven entries
match no committed result file** (§8, D-4). Only rho(carry, defensive) = +0.604 is correct.
The qualitative conclusions that survive — low-vol is negatively correlated to trend, carry
and defensive are largely one bet — happen to hold at the file's own numbers, but the table
as published is not checkable and should not be re-quoted.

**M-7 — ARITHMETIC SLIPS.** Iteration 12's per-bet-Sharpe table gives "0.291 → 9.38" where
its own preceding table and the result file both give **9.43**. Iteration 12 also quotes
per-bet s = 0.2913 where the file gives 0.29117 (a 0.6678 → 0.668 rounding). Iteration 15
quotes the 17.75-year DSR bar as "~0.91" where the files give 0.9234 (n = 38), 0.9443
(n = 47) and 0.9807 (n = 69); and low-vol's excess-basis Sharpe as 0.485 where the files give
0.4869.

**M-8 — THE WRONG BOOK'S UNCONSTRAINED PEAK WAS ATTRIBUTED.** Iteration 17 reads "Corrected
pair, bill+150bp: 18.05%/yr at DD≤50% … Unconstrained peak 60.82% at 7.45x on a −91.12%
drawdown". The 60.82% / 7.45× / −91.12% figures belong to the pair **as claimed**
(`attack4_leverage.claim_basis_made_coherent`), not to the corrected pair, whose
unconstrained peak is 40.44% at 6.25× on −80.61%.

**M-9 — THREE CONFIGURATIONS WERE MIXED INTO ONE TABLE ROW.** Iteration 1's tsmom row reads
"breadth 98 · net Sharpe −0.37 · excess −2.4%". No configuration has all three: SENSITIVITY-B
is 98.0 bets/yr at +0.0576 and −2.35% and carries `gate_eligible: false`; PRIMARY-STICKY is
106.6 bets/yr at −0.3682 and −8.79%. The gate-eligible pairing is **−0.37 with −8.79%**. The
underlying result document states this correctly; the log entry did not.

---

## 5. WHAT IS ACTUALLY ACHIEVABLE

**≈18%/yr at a ≤50% drawdown on institutional financing. ≈16% at retail financing.
≈13–14% at a ≤35% drawdown. Every one of those figures is an upper bound, and each carries a
caveat that is individually capable of removing it.**

1. **It is an UPPER BOUND on the engine's own admission.** The leverage engine that produced
   it reads rich against iteration 11 by a factor of **0.877** (a 12.3% haircut) — validated,
   not assumed, by reproducing iteration 11's 12.30% from this engine's 14.11%. The haircut
   is already applied to every figure quoted. What is *not* corrected for is everything the
   engine does not model: no impact on the futures legs, no borrow availability at 2×
   gross, no tax, no slippage beyond the modelled spread.
2. **The pair that produced the 18.05% figure is DEAD on deflation.** Corrected Sharpe
   0.9212 against bars of 0.9443 (n = 47), 1.0872 (n = 234) and 1.1022 (n = 281) — it fails
   **all three, including at n = 47 alone**, before any credit for the 234-configuration
   search. PBO over 109 configurations is 0.3231.
3. **It rests on hindsight-selected survivor instruments.** All 18 panel instruments were
   chosen knowing which survived; the breadth-expansion additions are worse still (four of
   six validation instruments — NIB, JO, BAL, COW — were delisted on 2023-07-21; VXX's
   history starts in 2018 only because the prior ETN matured). Direction of bias: upward.
4. **The window flatters it.** The pair's 213 months contain **two** bear markets in 17.75
   years. Its dominant leg is a **bear-market payoff**: excluding both bears, the pair's raw
   Sharpe *rises* to 1.0031, but its vol-matched active return against passive goes
   **+5.39%/yr (t 1.62) → −1.55%/yr (t −0.38)**, and passive scores **1.1299 on that window
   against the pair's 1.0031**. **Outside the bear markets it loses to buy-and-hold.**
   Correctly labelled, it is a crash hedge.
5. **The like-for-like comparison is much less impressive than the headline.** 18.05% is
   measured on the pair's own 213 months. On the *same engine, same solve, same window*,
   passive returns **5.80%**. On its own 61.5 years passive returns **10.80%**. The
   frequently-quoted "passive 12.30%" is iteration 11's **volatility-targeted, observed-path**
   figure and is **not on the same solve convention** as the 18.05% — comparing them directly
   overstates the gap.
6. **FINANCING MOVES THE OUTCOME MORE THAN EVERY STRATEGY DECISION COMBINED.** A UK retail
   account faces roughly **bill + 300bp**, not bill + 150bp. Iteration 11 measured the
   swing directly: moving financing from +50bp to +300bp changes the τ = 40% compound return
   by **16.5 percentage points** (17.99% → 1.46%), and at retail margin the volatility-targeted
   ladder **inverts** past τ ≈ 20%. In iteration 17's static-leverage construction the
   inversion does not appear because the drawdown cap binds at 1.3–1.9× where the financing
   drag is still small (retail costs 1.8pp at ≤50%, 0.6pp at ≤35%) — the two constructions
   are different and neither confirms nor refutes the other. **Either way, the borrowing rate
   is the largest single lever on the outcome, and it is not a research variable.**
7. **A large share of the headline is just the cash rate.** **4.64pp of iteration 11's 12.30%
   is the average T-bill rate** over the sample. The same book earned 5.86%/yr in the 2010s,
   when bills paid 0.56%.
8. **There is a capital floor.** The low-vol sleeve's deployable capacity is **$138k**. PEAD
   at a $10,000 account trades $50 tickets and pays **140bps round trip in commission alone**
   — more than its entire 114bps of net alpha per bet. That sleeve does not exist below
   roughly $500,000.

**The defensible statement is: a diversified, cheaply-traded, levered-to-a-survivable-drawdown
book plausibly compounds in the low-to-mid teens, and the single largest determinant of where
in that range it lands is the financing rate, not the strategy.**

---

## 6. WHAT WOULD ACTUALLY BE REQUIRED FOR 30%

The ceiling is `S²/2` (full Kelly) with `S = s·√N_eff`. Raising the *signal* on the existing
panel cannot move it; 23 studies tried. Only **effective breadth** moves it, and effective
breadth is not instrument count.

**The measured starting point:** portfolio Sharpe 0.6678 across a correlation-effective
**N_eff = 5.26** ⟹ per-bet Sharpe **s = 0.2912**. The panel holds 18 instruments and behaves
like 5.26 independent bets. The three Treasury maturities alone count **1.17** — three
maturities are one bet.

| requirement | N_eff needed | vs the measured 5.26 |
|---|---:|---:|
| 30%/yr, full Kelly, idealised `S²/2` | 7.08 | +1.8 |
| 30%/yr, half Kelly, idealised `3S²/8` | 9.44 | +4.2 |
| **30%/yr, half Kelly, at the MEASURED 0.71 efficiency** | **13.29** | **+8.0** |
| **30%/yr at the MEASURED post-expansion per-bet Sharpe of 0.169** | **≈ 39.5** | **≈ +34** |

The last row is the honest one, and it is why the answer is no. Iteration 14 measured the
marginal effective bets each new asset class actually supplies (2011+):

| class added | marginal N_eff |
|---|---:|
| **agriculture** | **+2.90** |
| livestock | +0.92 |
| real assets | +0.72 |
| foreign sovereigns (gilts + Bunds + JGBs) | **+0.08** — they are **1.40 effective bets between them**; global duration is ONE trade |
| volatility | −0.07 |
| credit | −0.27 |

Agriculture supplied +2.90 of the total +3.25; the other twelve additions were worth +0.35
jointly — **and agriculture is where the roll contamination lives** (D-12). At the observed
rate of **0.171 effective bets per instrument**, reaching N_eff ≈ 40 requires roughly **182
further instruments**.

**No free universe supplies that.** Probed and confirmed absent, not assumed: no free non-US
sovereign yield series (`^GDBR*`, `^GBGB*`, `^JP*` all return empty), no Bund/gilt/JGB
futures, no free freight or carbon index. And the length/breadth trade is strictly bad: the
full-sample panel clears its DSR bar (0.499) but does not move the ceiling, while the 2011+
window where breadth is real carries a bar of **1.011 that nothing clears** and a **lower**
peak (6.52% vs 10.35%).

**Conclusion: 30%/yr is closed on free data.** Not "not found" — the ceiling was measured,
the one remaining lever was tested, and it moved the answer by **0.22 percentage points**.

---

## 7. METHOD NOTES WORTH KEEPING

These are the standing rules the run produced. They are the durable asset.

1. **COMPARE AT MATCHED VOLATILITY.** Never raw geometric excess (D-4 fakes it low) and
   never raw arithmetic active return (D-5 fakes it high). Use
   `volmatched_active = bench_vol × Sharpe_gap` against the strategy's own universe levered
   to the strategy's own volatility. It is verified exact and cannot reverse a ranking.
2. **RUN THE BENCHMARK THROUGH THE IDENTICAL GATE and require the candidate to beat it.**
   The gate has no benchmark-relative criterion (D-6) and has failed in both directions.
   Never quote a DSR without the benchmark's DSR beside it.
3. **CHECK CORRELATIONS WITH THE CONSTRUCTION OVERLAP REMOVED.** Value's −0.164 correlation
   to trend became **−0.013** once the 12 months where the 5-year reversal window contains
   the 12-month momentum window were removed. Defensive passed the same test (+0.020 →
   −0.037) and was still useless — which is a cleaner failure. The diagnostic discriminates;
   run it before believing any diversification claim.
4. **REPORT KELLY FIGURES ONLY WITH (a) THE VOLATILITY THEY REQUIRE, (b) THE LEVERAGE THAT
   IMPLIES ON THE SLEEVE'S NATURAL VOLATILITY, AND (c) THE MEASURED MAX DRAWDOWN SCALED BY
   THAT LEVERAGE.** A growth rate requiring a leverage whose implied drawdown exceeds ~60% is
   not a reachable return; it is arithmetic (M-3).
5. **SOLVE LEVERAGE AGAINST A BOOTSTRAPPED DRAWDOWN, NOT THE OBSERVED PATH.** Solving against
   a single observed maximum systematically over-levers and flattered the pair by 7–9pp.
6. **USE MEASURED COVARIANCE, NEVER THE EQUAL-SHARPE SHORTCUT** (M-2, 45% overstatement).
   And weight by RISK, not dollars: equal-risk beat equal-dollar by **0.109 of Sharpe for
   free** on trend+carry, because the sleeves differ 5.7× in volatility.
7. **REPORT PER DECADE, ALWAYS.** It caught trend (post-2009 0.180 vs a benchmark's 0.777),
   carry (the 2010s carry it at 0.86; leave the decade out and it is 0.100) and defensive.
8. **CRISIS EXCLUSION MUST EXCLUDE *ALL* CRISES.** Iteration 8 excluded 2008–2011 and
   reported the sleeve "stronger". True — and 2008–2011 is where that sleeve did *least*
   well. Excluding **both** bear markets took it to +4.87%/yr at t 1.45, below its own gate.
   The dot-com bust alone carried +23.94% at t 3.63.
9. **BUILD THE POSITIVE CONTROL FIRST, AND GIVE IT A LEG THE OLD MODEL MUST FAIL.** Both cost
   repairs were shipped this way. Check E — "the old coefficient must FAIL check A" — is what
   proved the gate had teeth.
10. **BRACKET, DON'T POINT-ESTIMATE, WHERE THE DECOMPOSITION IS UNSTABLE.** Two bounds:
    conservative (a pass here is REAL) and realistic (a fail here is DEAD); in between is
    UNDETERMINED. `realistic <= conservative` by construction, verified elementwise over all
    922,652 cells, so a cheaper bound can only move a verdict from "dead" to "undetermined",
    never to "real".
11. **PERSIST EVERY SLEEVE'S DATED RETURN SERIES, AND DECLARE WHAT ITS INDEX MEANS** (D-7,
    D-13). Within-series statistics cannot detect a dating error; only a dated join can.
12. **NOMINAL BREADTH IS NOT BREADTH.** Trend's 135.6 nominal bets/yr are 57.5 effective
    (eff N 4.79); carry's 156 nominal are **1.34 measured sign flips/yr**; tsmom's 7,408
    nominal collapse to 98. Breadth bought from **events** costs ~200 bets per unit of
    turnover; breadth bought from **frequency** costs 12.7 — a 16× difference.
13. **FORECAST THE BENCHMARK IN THE PRE-REGISTRATION, NOT JUST THE STRATEGY.** Value's
    prediction was well calibrated on correlation (−0.25 predicted vs −0.164 realised) and
    badly wrong on active return (+0.5% vs −9.84%) purely because it never forecast what the
    benchmark would earn.
14. **NEVER PREDICT A SHARPE BY MOVING A RETURN AGAINST A FIXED VOLATILITY** when the same
    correction changes how much capital is deployed. PEAD's return doubled and its volatility
    doubled with it; the ratio barely moved.

---

## 8. DISCREPANCY REGISTER — where the log and the result files disagree

The result file wins in every row. None of these changes the run's conclusion; several make
it stronger.

| # | log says | result file says | source |
|---|---|---|---|
| D-1 | it-1: tsmom "breadth 98, net Sharpe −0.37, excess −2.4%" | no configuration has all three: SENSITIVITY-B = 98.0 / +0.0576 / −2.35% (`gate_eligible: false`); PRIMARY-STICKY = 106.6 / **−0.3682** / **−8.79%** | `_out/tsmom_multitimeframe_result.json` |
| D-2 | it-3: multi-asset trend "top (instrument, month) cell 1.93% of P&L" | stored as **−0.0193** — a *negative* 1.93% (USDX 1978-11). The positive-denominator figure is 0.32% | `_multiasset_trend/result.json` |
| D-3 | it-12: per-bet s table "0.291 → N_eff 9.38" | 0.8/0.29117² = **9.43** — the value the log's own preceding table gives. No file contains 9.38 | `_breadth/neff.json` |
| D-4 | it-13: correlation table (7 entries, 143-mo window) | **6 of 7 do not match.** File: lowvol~trend **−0.164** (log −0.198); lowvol~carry **−0.017** (log **+0.044 — sign flipped**); lowvol~seasonal **+0.126** (log +0.093); lowvol~defensive **−0.097** (log −0.040); lowvol~passive **+0.355** (log +0.335); seasonal~passive **+0.476** (log +0.469). Only carry~defensive +0.604 matches | `_portfolio/portfolio_correlation_result.json` |
| D-5 | it-13: "58 of 234 combinations clear 0.894", corrected by it-16 to 65 | **58 is the sleeve-SUBSET count** (58 × 4 schemes = 234), not a clearing count. Clearing counts on record: 28/93 and 84/189 (v1), 74/189 (`portfolio_decision.json`), **65/234** (v2) | v1 + v2 JSON, `portfolio_decision.json` |
| D-6 | it-13: "best = 1.160" | v1's best is **1.163** (lowvol+trend+carry, ERC); 1.1603 is trend+carry+passive inverse-vol. v2's maximum is **1.2411** | v1 write-up (private history), v2 JSON |
| D-7 | it-13: window-control "lowvol 0.465 vs 0.614" | file: **0.742** on the common window vs **0.878** full (registered) — the log mixes the corrected full-sample figure (0.614) with a common-window number that matches neither file | v1 `common_window.sleeve_sharpes` |
| D-8 | it-15: "the DSR bar at 17.75 years is ~0.91" | **0.9234** (n = 38), **0.9443** (n = 47), **0.9807** (n = 69) | `portfolio_decision.json`, `pair_deflation.json` |
| D-9 | it-15: low-vol "0.485 on an excess basis" | **0.4869** | `_pair_deflation/controls.json`, v2 JSON |
| D-10 | it-8: "6,322 last-observation cells carry a delisting record" | **7,580** last-observation cells, of which **6,322** carry such a record | `lowvol_retest_result.md:169` |
| D-11 | it-17: "Corrected pair … Unconstrained peak 60.82% at 7.45× on a −91.12% drawdown" | those belong to the pair **as claimed**; the **corrected** pair's peak is **40.44% at 6.25× on −80.61%** | `pair_deflation.json attack4_leverage` |
| D-12 | it-17: "All 18 controls reproduced first" | `controls.json` holds **19** checks (all passing) | `_pair_deflation/controls.json` |
| D-13 | it-6: seasonal perfect-foresight control "3.978" | `_seasonal/result.json` says **3.977**; the study's own write-up says 3.978 | `_seasonal/result.json` vs `multiasset_seasonal_result.md:311` |
| D-14 | it-4: "94.87bps of impact against an all-in cost of 5.54bps — **17.9×**" | 94.87/5.54 = **17.1×**. The 17.9× is **(94.87 + 4.50 half-spread)/5.54 = 17.94** — a total-cost ratio, not an impact ratio | `scripts/impact_positive_control.py` check E |
| D-15 | test-suite counts (1191 / 1207 / 1220 / 1282 / 1353 / 1379 / 1389 passed) | recorded **only in the internal session narrative**. Not independently checkable from a committed artefact | — |

**Two structural notes on the log itself.** (a) Iteration numbers are **duplicated and
out of order**: there are two entries each numbered 3, 4, 5 and 8; iteration 2c (01:25)
appears after iteration 3 (01:10); the "Iteration 5 — SYNTHESIS" entry appears between
iterations 10 and 11. This is a consequence of parallel agents appending concurrently and it
makes the log hard to audit chronologically — the commit timestamps are the reliable
ordering. (b) The count of "studies so far" quoted inside the log is internally inconsistent
(eleven, twelve, thirteen, fifteen and twenty-three all appear); §2 above is the reconciled
count.

---

## 9. WHAT THE RUN ACTUALLY BOUGHT

Stated without salesmanship, because the honest total is small and real:

1. **A definite, measured answer to the question that was asked.** 30%/yr is above the
   maximum of the leverage-return curve on this data. That is a stronger and more useful
   result than another marginal sleeve would have been.
2. **A measurement stack that is materially more honest than it was 13 hours ago** — two
   cost-model defects worth 60–70bps a round trip and 57% of the tradable universe, a
   dating defect that no within-series statistic could ever have caught, and eleven
   accounting defects in the first hour alone.
3. **One genuine result: trend + passive, Sharpe 0.9033 over 61.5 years**, positive in all
   seven decades, beating at matched volatility a benchmark that trend alone loses to. It is
   72% passive, its confidence interval spans the target, and it has not been
   pre-registered. It is the correct next thing to test properly.
4. **Fourteen standing method rules** (§7), each paid for with a study that died.

**What it did not buy: any route to 30%/yr, and any deployable strategy.** The 2016+
confirmation window remains **UNFIRED**, correctly — nothing came close to earning it.
