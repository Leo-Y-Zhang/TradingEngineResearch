# RESULT — The Breadth Sleeve Hunt: six candidate alpha sleeves, DEV window

**Run:** 2026-07-28. Six sleeves pre-registered, built, run once on the DEV window
(1997-12-31 → 2015-12-31 subsets), then adversarially verified.
**Governing documents:** `docs/project-control/specs/2026-07-28-the-breadth-lever.md`,
`docs/project-control/specs/2026-07-27-the-dsr-sample-length-finding.md`,
`research/medallion_style_alpha_search/capacity_curve_result.md`.

**Verdict: 0 of 6 survived. Nothing is deployable. The 2016+ confirmation window
remains UNFIRED and no sleeve came close to earning the right to fire it.**

Arithmetic in this document is reproduced by `scripts/synthesise_breadth_sleeve_hunt.py`,
which re-reads the four machine-readable sleeve artefacts and asserts the transcribed
scalars match before computing anything.

---

## 1. Results

| # | sleeve | breadth (bets/yr) | net Sharpe | excess over OWN benchmark | verdict | survived verification |
|---|---|---:|---:|---:|---|---|
| 1 | Short-horizon cross-sectional reversal (weekly, dollar-neutral) | 577.2 | **−3.6189** | **−52.81%** | DEAD | No |
| 2 | Post-earnings-announcement drift, SF1 ARQ SUE, 40d hold | 476.9 | **+0.3422** | **−2.97%** | DEAD | No |
| 3 | Time-series momentum, multi-timeframe, 200 names + sector baskets | 98.0 | **+0.0576** ⚠ | **−2.35%** ⚠ | DEAD | No |
| 4 | Institutional ownership flow (SF3 13F QoQ change) | 4.0 | **−0.4447** | **−6.54%** | DEAD | No |
| 5 | Insider transaction clustering (SF2 distinct 90d buyers) | 1162.2 | **−0.1200** | **−8.69%** | DEAD | No |
| 6 | Low-volatility / quality composite (best band, B2 $200k–$1M/day) | 93.5 | **+0.3244** | **−5.54%** | DEAD | No |

⚠ **Sleeve 3's headline is not gate-eligible.** The reported net Sharpe 0.0576 comes from
`SENSITIVITY-B`, which costs unresolved-spread names at a flat 20bps and therefore
violates rule 3. Its own result JSON carries `gate_eligible: false` (verified on disk).
The two rule-3-compliant configurations at the same 15% vol target are **net Sharpe −1.02**
(PRIMARY, excess −16.13%/yr) and **−0.37** (PRIMARY-STICKY, excess −8.79%/yr). Read the
row at −0.37 for gate purposes.

**"Survived verification: No" needs a precise reading.** Verification did not overturn six
live sleeves. Every one of the six returned DEAD from its own single registered run and
self-reported it. The verification step had nothing to refute — it confirmed that none of
the negative verdicts was an artefact of a broken harness, which matters, because between
them the six runs found and fixed **eleven** accounting defects (below). No sleeve was ever
alive.

Supporting numbers, all measured:

| sleeve | gross Sharpe | turnover/yr | cost drag/yr | sample |
|---|---:|---:|---:|---|
| reversal | 0.5193 | 45.29× | 71.81% | 17.7yr, 921 weekly cross-sections |
| PEAD (40d) | 1.0749 | 2.39× | 5.57% | 17.7yr, 212 months |
| tsmom (SENS-B) | 0.4455 | 21.19× | 6.65% | 17.0yr, 205 rebalances |
| instflow | 0.0808 | 6.61× | 7.71% | **2.2yr, 9 rebalances** |
| insider | 0.5300 | 5.83× | 13.73% | 7.7yr, 92 months |
| lowvol (B2) | 1.1160 | 9.06× | 10.82% | 17.8yr, 207 rebalances |

---

## 2. The honest headline

**Nothing survived. Six independent signals, three of them with genuinely strong and
statistically real gross edges, all lost to an equal-weight buy-and-hold of their own
universe after honest per-name costs. The excess column is negative six times out of six,
by −2.35% to −52.81% a year.**

This is the eleventh consecutive negative result in the programme, and it is the first one
that says something new, because this round was designed to test a *mechanism* rather than
to find a signal. Three findings are worth more than the verdicts:

**(a) The signals were not the problem.** Four of six produced a real gross edge: PEAD
gross Sharpe 1.07 with +256bps of alpha per bet and monotone decile ordering in both halves
of the sample; low-vol/quality gross Sharpe 1.12 and +5.3%/yr gross excess in B2, decaying
monotonically with liquidity exactly as its leverage-constraint mechanism predicts; insider
purchases +5.45%/yr gross excess at t=3.23; reversal IC +0.0216 at t=4.91 over 921 weekly
cross-sections. Two of these gross Sharpes clear the DSR bar their own sample length
demands (§5). The programme does not have a signal-discovery problem.

**(b) The binding constraint is cost per round-trip, and it is now measured.** Across the
six sleeves the cost of one round trip in the spread-measurable US equity universe is
**117–236 basis points**, and gross alpha per round trip is **15–256 basis points**:

| sleeve | gross alpha per round-trip | cost per round-trip | cover ratio |
|---|---:|---:|---:|
| **PEAD (40d)** | **256.0 bp** | **219.1 bp** | **1.17** |
| tsmom (SENS-B, not gate-eligible) | 29.3 bp | 31.4 bp | 0.93 |
| lowvol (B2) | 58.2 bp | 119.5 bp | 0.49 |
| insider | 93.5 bp | 235.5 bp | 0.40 |
| instflow | 17.8 bp | 116.6 bp | 0.15 |
| reversal | 15.3 bp | 158.6 bp | 0.10 |

Only **one sleeve at one of its three pre-declared horizons** earned more per trade than it
paid, and by 37bps on a 219bps bill — 1.17× cover. (PEAD's own directly-measured
round-trip cost of 219.1bps is used here; the cost-drag ÷ turnover derivation gives 233bps
and cover 1.10. Both tell the same story. Every other row uses cost-drag ÷ turnover.) At
20d and 60d PEAD's cover is 0.58 and 0.82 — negative net alpha per bet. **This ~120–240bps
figure, not any Sharpe, is the number this session actually established.**

**(c) Every one of these verdicts is conditional on a cost model that can only see the
expensive half of the tape, and five of six sleeves flagged it independently.** Rule 3
excludes names whose EDGE spread does not resolve — but EDGE only resolves a spread sitting
1.5× above its volatility-scaled noise floor, so a name is admitted *precisely when it is
expensive, or when its estimate took an upward noise draw*. The consequence, measured four
separate times: reversal's tradable universe has a median spread of 100bps against 52bps
for the names it excluded; PEAD's traded universe 126bps at $1.99M/day median volume, with
47% of qualifying filings dropped; insider's 126bps at $1.42M/day, its picks at 143bps and
$498k/day; institutional-flow 91.5bps with 68% of cells dropped. The programme has now
measured six signals on the illiquid, wide-spread minority of US equities and has said
**nothing** about how any of them behaves in large caps. That is a limitation of the
measurement apparatus, not a finding about liquid names, and it is the single largest
open question left by this session (§6).

The counter-evidence to (c), which must be stated because it is decisive for one sleeve:
the reversal study ran the test directly. With **spreads set to zero** it still nets
−2.96%/yr; with spread, impact *and* borrow all zero — IBKR commissions alone, a physically
impossible cost model — its net Sharpe caps at **0.41**, below the 0.75 promotion gate. No
cost-model improvement can rescue a 45×-turnover sleeve. The cost-measurement criticism is
live for the low-turnover sleeves (PEAD at 2.39×, insider at 5.83×) and refuted for the
high-turnover ones.

---

## 3. BREADTH ANALYSIS — did higher breadth deliver higher Sharpe?

This was the session's central question. The answer has two halves and they point in
opposite directions.

### 3.1 The cross-sleeve test: NO relationship, and it has no power

Rank-correlating breadth against Sharpe across the six sleeves (exact permutation p-values,
all 720 orderings enumerated):

| x | y | Spearman ρ | exact p |
|---|---|---:|---:|
| reported breadth | **gross** Sharpe | **+0.257** | 0.658 |
| reported breadth | **net** Sharpe | **−0.143** | 0.803 |
| structural breadth | gross Sharpe | +0.200 | 0.714 |
| structural breadth | net Sharpe | −0.257 | 0.658 |
| reported breadth | cost drag | +0.429 | 0.419 |
| turnover | cost drag | +0.486 | 0.356 |

**Breadth did not predict net Sharpe. The sign is negative and nothing here is
significant.** The highest-breadth sleeve (insider, 1,162 bets/yr) ranks 4th of 6 on net
Sharpe; the second-highest (reversal, 577) ranks **last, at −3.62**. The two best net
Sharpes came from the 5th- and 6th-ranked breadths (PEAD 477 → +0.342, low-vol 93.5 →
+0.324).

Two caveats that stop this being over-read in either direction. First, n=6 with p≥0.36
everywhere: this test could not have detected a real effect. Second — and more serious —
**the six sleeves did not use a common breadth estimator**, so the ranks are measured in
inconsistent units. Reported breadth mixes Grinold-implied counts, effective-N counts,
statutory event counts and entry-event counts. The `structural breadth` row above
re-ranks using only estimators computed from a correlation/residual matrix or a statutory
count (independent of realised returns) and the answer does not change.

### 3.2 The within-sleeve test: the Fundamental Law's GROSS arithmetic held

The cross-sleeve rank test is the weak evidence. The strong evidence is inside the sleeves,
and there it works.

**One clean, non-circular confirmation.** The insider sleeve measured breadth structurally
— effective independent names N_eff = 96.85 of 151 held, from the ratio of mean per-name
residual variance to the variance of the equal-weight residual portfolio return, times 12
rebalances = **1,162 bets/yr** — and measured its IC separately at **+0.0134**. Grinold
predicts IR = 0.0134 × √1162 = **0.457**. Realised gross excess IR was **0.530**. The law
under-predicted by 16%, on a breadth number derived from return *covariance* and an IC
derived from return *ranks*, neither of which was solved from the other. (The sleeve's own
note that its Spearman IC is deflated by 79% tied zeros makes 0.457 a lower bound, which is
the right direction.) **This is the session's real methodological result: the Fundamental
Law's gross arithmetic is confirmed in the one place it could be honestly tested.**

**One confirmation that must be thrown out as circular.** The reversal sleeve reports
IC 0.0216 × √577 = 0.519 against a realised gross Sharpe of 0.519 — apparently perfect. It
is perfect because its headline breadth was *back-solved* as BR = (IR/IC)², which makes
IR = IC·√BR an identity, not a test. It should be read as "the implied breadth of this
sleeve is 577", which is a useful statement, and not as evidence for the law.

**One structural refutation of the premise.** The tsmom sleeve measured the participation
ratio of its universe's daily return correlation matrix: **N_eff = 9.29 out of 206
instruments.** US single names are roughly nine independent instruments, not two hundred.
Independently, its signals changed sign only 10.55 times a year summed across all three
timeframes, not 36. Nominal breadth 7,408/yr collapsed to a measured **98/yr — 1.3% of
nominal**. Adding ~8.5 sector baskets bought instruments, not independence. At IC 0.045 and
BR 98, reaching IR 1.2 needs BR ≈ 700 — about 7× more independence than the US
cross-section can supply *at any rebalance frequency*.

### 3.3 The reconciliation: breadth is not free, and its price varies 300×

Gross Sharpe scales as √BR. Cost scales **linearly** with turnover. So the lever pays only
if breadth can be bought without buying turnover — and the measured price of breadth varies
by more than two orders of magnitude across these six sleeves:

| sleeve | bets per unit of annual turnover | how breadth was bought |
|---|---:|---|
| **PEAD** | **199.6** | discrete events, natural 40-day holding period |
| **insider** | **199.3** | discrete events, wide cross-section, monthly |
| reversal | 12.7 | **frequency** — 52 rebalances/yr |
| lowvol | 10.3 | cross-section, but 70–83% of exits forced by the universe filter |
| tsmom | 4.6 | frequency + leverage, on 9 independent instruments |
| instflow | 0.6 | statutory quarterly filing |

**This is the refinement the breadth-lever spec needs.** The spec says breadth requires
"trading cheaply and often". The measurement says the second half is wrong, or at least
subordinate: the two event-driven sleeves achieved ~200 bets per unit of turnover, 16× the
frequency-driven reversal sleeve, and they did it at 2.4× and 5.8× annual turnover
respectively. Reversal is the direct demonstration of the failure mode — it bought 577 bets
by trading 45.3× a year, paid 158.6bps per turn, and turned a genuine +0.52 gross Sharpe
into −3.62 net. **Breadth bought cross-sectionally or from events is nearly free; breadth
bought from frequency is priced at the round-trip cost and loses the race by construction.**

### 3.4 Answer

**Gross: yes, the law held where it could be tested (insider, predicted 0.457 vs realised
0.530).** **Net: no — ρ = −0.14, and the highest-breadth sleeve in the study produced the
worst net Sharpe in the study.** Breadth is a real multiplier on gross skill and it is not
the binding constraint. The binding constraint is that in the spread-measurable US equity
universe, one round trip costs 117–236bps and only one of six signals, at one of its three
horizons, generated more alpha per round trip than that.

---

## 4. PORTFOLIO ARITHMETIC

**Survivors: zero.** No sleeve has positive excess over its own benchmark; no sleeve has a
net Sharpe at or above the 0.75 promotion gate. `S = s·√(N / (1 + (N−1)ρ))` is undefined at
N = 0. **There is no portfolio to build and no reachable annual return to state. The honest
answer to "what annual return is actually reachable from this sleeve set" is: none above
passive ownership of the same names**, which returned 10.04%/yr at Sharpe 0.46 in the
low-vol study's B2 band and 4.47%/yr at Sharpe 0.31 in the reversal universe.

**Pairwise correlations are not computable either.** Only 1 of the 6 sleeves persisted a
return series to disk (`research/sleeves/_pead_output/equity_top_decile_{20,40,60}d.parquet`,
4,462 daily points). The other five wrote summary scalars only. Recording this as a process
defect: **every future sleeve must persist its daily net return series**, or cross-sleeve
combination can never be evaluated without re-running everything.

Two counterfactuals, computed because the shape of the answer is informative, both
explicitly **NOT achievable and NOT gate-eligible**:

**A — combine the two least-bad net Sharpes** (PEAD 0.3422, low-vol 0.3244; mean s = 0.3333).
ρ is unknown, so it is swept:

| assumed ρ | combined S | half-Kelly growth 3S²/8 | implied vol S/2 |
|---:|---:|---:|---:|
| 0.0 | 0.471 | **8.33%/yr** | 23.6% |
| 0.2 | 0.430 | 6.94%/yr | 21.5% |
| 0.5 | 0.385 | 5.55%/yr | 19.2% |

Even at the impossible ρ = 0, two-sleeve combination reaches **8.3%/yr at 23.6%
volatility** — and both inputs have *negative excess over their own benchmarks*, so this
book loses to buying the same stocks and holding them, at more than twice their volatility.
Half-Kelly on a negative edge prescribes a **zero** allocation. The correct portfolio weight
on everything in this study is 0.

**B — all six gross Sharpes at zero cost** (mean s = 0.6278, N = 6). This is the ceiling of
"combine many mediocre signals", and it is physically impossible:

| assumed ρ | combined S | half-Kelly growth | implied vol |
|---:|---:|---:|---:|
| 0.0 | 1.538 | 88.7%/yr | 76.9% |
| 0.2 | 1.087 | 44.3%/yr | 54.4% |
| 0.5 | 0.822 | 25.3%/yr | 41.1% |

The gap between B and reality is the entire content of this session: **six signals whose
gross combination would clear every target the programme has ever set, and a cost floor
that removes all of it.** Note also how fast B decays in ρ — at ρ = 0.5 (six US equity
long-only-ish books; plausible) even the zero-cost ceiling gives 25.3%/yr, and the
reversal sleeve's *measured* zero-cost ceiling was net Sharpe 0.41, not 0.52.

---

## 5. TRIAL ACCOUNTING

**Cumulative n_trials: 26 (prior programme) + 6 (this session, one per sleeve run) = 32.**

Each sleeve pre-registered one hypothesis and ran it once; the variants inside them
(PEAD's three horizons, low-vol's four bands, reversal's two universes, tsmom's three
cost/universe treatments × three vol targets, institutional-flow's void run 1) were all
declared in advance and all reported, which is the programme's convention for one trial.
**A stricter ledger that counted declared variants separately would put the count near 40**,
so the n=40 column is carried below and nothing in the conclusions depends on which count
is used.

Required standalone annual Sharpe for **DSR ≥ 0.95**, computed with the same algebra as
`research/validation.py::deflated_sharpe_ratio` under normal monthly returns:

| OOS years | n=26 | **n=32** | n=40 |
|---:|---:|---:|---:|
| **7** | 1.451 | **1.488** | 1.527 |
| 10 | 1.196 | 1.226 | 1.257 |
| 15 | 0.966 | 0.989 | 1.014 |
| 20 | 0.832 | 0.852 | 0.873 |
| 30 | 0.675 | 0.691 | 0.708 |
| **40** | 0.583 | **0.597** | 0.612 |
| 50 | 0.521 | 0.533 | 0.546 |

**Answer to the registered question: at n=32, DSR ≥ 0.95 demands standalone Sharpe 1.49 on
a 7-year window and 0.60 on a 40-year window.** Six additional trials cost **+0.037 Sharpe
at 7 years and +0.014 at 40 years** — the multiple-testing penalty is nearly free at the
long end and the sample-length term dominates completely, which is the DSR spec's finding
holding up under a 23% increase in the trial count.

*Reproduction note:* this closed form reproduces the recorded anchors at the short end
exactly (7yr/n=26 → 1.451 vs the recorded 1.45; 7yr/n=40 → 1.527 vs 1.52) and sits ~0.02
low at the long end (40yr/n=26 → 0.583 vs the recorded 0.60), consistent with the original
table being read off a coarse grid. Cross-checked by feeding a normal-scores series at the
solved Sharpe back into the repo's own `deflated_sharpe_ratio`, which returns DSR 0.9514
against a 0.9500 target.

**Every sleeve against the bar its own sample length demands, at n=32:**

| sleeve | years | bar | gross Sharpe | | net Sharpe | |
|---|---:|---:|---:|---|---:|---|
| reversal | 17.7 | 0.91 | 0.519 | fail | −3.619 | fail |
| **PEAD** | 17.7 | 0.91 | **1.075** | **PASS** | 0.342 | fail |
| tsmom | 17.0 | 0.93 | 0.446 | fail | 0.058 | fail |
| instflow | **2.2** | **3.06** | 0.081 | fail | −0.445 | fail |
| insider | 7.7 | 1.42 | 0.530 | fail | −0.120 | fail |
| **low-vol** | 17.8 | 0.91 | **1.116** | **PASS** | 0.324 | fail |

Two sleeves clear the deflated bar on the gross side. **None clears it net, and none is
close** — the best net Sharpe in the study is 0.342 against a bar of 0.91. This table also
kills one attractive-looking idea outright: the insider signal's gross IR of 0.530 sits far
below the **1.42** its 7.7-year SF2 history demands, so **no cost-model correction whatever
can make that sleeve gate-eligible on the data that exists** — its sample is too short.
Same for institutional flow, at a bar of 3.06 on 2.2 years.

---

## 6. The single most promising direction remaining

**Fix the spread measurement so the liquid half of the tape becomes admissible, validate
the new estimator against ground truth *before* pointing it at any strategy, and then
re-test PEAD under a fresh pre-registration.**

The reason is arithmetic, not preference, and it is the only direction where the numbers
already measured leave room:

1. **PEAD is the only sleeve whose alpha per round trip exceeds its cost per round trip**
   (256.0bps vs 219.1bps, cover 1.17). Everything else in the study needs its gross edge to
   *grow*; PEAD only needs its bill to shrink.
2. **PEAD has the sample length to clear the deflated bar.** 17.7 years → bar 0.91 at n=32,
   against a gross Sharpe of 1.075 that already passes. Insider (bar 1.42 on 7.7 years) and
   institutional flow (bar 3.06 on 2.2 years) are arithmetically unreachable regardless of
   costs, which removes them from contention permanently on this data.
3. **The correction has a known size and a known target.** Spread is ~80% of measured cost.
   If liquid-name spreads are 5–20bps rather than the 126bps median PEAD was charged, its
   round-trip bill falls to roughly 50–90bps against 256bps of alpha, taking net alpha per
   bet from +37bps to roughly +170–200bps and net Sharpe from 0.342 toward ~0.7–0.8 — into
   the 0.75 promotion gate and within sight of the 0.91 DSR bar. That is a specific,
   falsifiable prediction, and it is the right shape for a pre-registration.
4. **PEAD buys breadth the cheap way.** 199.6 bets per unit of turnover, at only 2.39×
   annual turnover — tied for best in the study with insider, and 16× better than the
   frequency-driven reversal sleeve. §3.3 says this is the property that decides whether
   the breadth lever pays.
5. **Five of six sleeves independently identified the same defect**, from different data and
   different signals: the traded universe is systematically the expensive minority because
   EDGE admits a name only on an upward noise draw. When five independent studies converge
   on one measurement flaw, fixing the measurement is worth more than a seventh signal.

**The three conditions that make this legitimate rather than tuning.** (a) The estimator
must be selected by calibration against known ground truth and *never* by its effect on a
strategy result — the precedent the capacity study set when it evaluated three spread
estimators at zero trial cost. (b) It is a **new pre-registration at n=33+**, written before
the re-run, with the 0.7–0.8 net-Sharpe prediction in (3) stated in advance so the result
can falsify it. (c) If the corrected spreads come back at 60–90bps rather than 5–20bps, the
prediction fails and PEAD is dead for good — that outcome must be reported, not re-cut.

**Why not the multi-market futures programme**, which the breadth-lever spec favours and
which this session's tsmom result supports strongly (N_eff = 9.29 of 206 US names; the US
cross-section simply cannot supply the independence). It remains the correct *strategic*
direction, and it is the only one that also solves the sample-length problem — a 40-year
proxy history drops the bar from 1.49 to 0.60 at n=32, and 0.60 is inside trend following's
documented 0.5–0.8 range. But it needs 30–50 years of futures or proxy history that is not
on disk, and the data-integrity work (roll assumptions, contract survivorship, splicing) is
the study rather than a preliminary. The cost-measurement route runs entirely on data
already present, tests a defect four sleeves independently identified, and can be finished
before any download is authorised. **Do the cheap decisive test first; the futures
programme is next, not instead.**

---

## 7. What else this session established, recorded so it is not re-learned

- **At least eleven accounting defects were found across the six sleeves, every one of them
  caught by an impossible number rather than by a test.** The dominant one recurred in **four** sleeves
  independently: `build_monthly_panel` stamps each ticker's *own* last bar of the month, so
  iterating distinct panel dates as a calendar produces phantom periods, singleton
  cross-sections and, when a delisting lands mid-month, a −100% month for an entire
  universe. **This should be fixed in the panel builder itself, not re-fixed by every future
  sleeve.** Others worth carrying: costs read only from currently-selected names charge buys
  and not sells (doubled the insider sleeve's true turnover, 2.9× → 5.8×); a $1.00 notional
  equity makes every trade hit the 1%-of-value commission cap and manufactures 27–72%/yr of
  fictitious cost; exits of names that left the universe carry a NaN spread that silently
  zeroes their liquidation cost; and pandas parses SF2's `securityadcode` value `'NA'` as
  missing, so reading SF2 without `keep_default_na=False` matches zero open-market purchases
  and the whole study measures nothing while appearing to run.
- **Negative controls passed everywhere they were run.** Reversal's fixed-seed per-date
  permutation collapsed gross Sharpe from +0.52 to −0.19; tsmom's 8-seed random-sign placebo
  returned 0.066 ± 0.291 against a live 0.4455. The harnesses are not manufacturing alpha —
  which is exactly what makes the negative verdicts trustworthy.
- **The reversal sleeve's gross edge is concentrated in one crisis.** Era split of gross
  long/short return: 1998–2002 −1.1%/yr, 2003–2007 +3.2%/yr, **2008–2011 +24.6%/yr**
  (Sharpe 0.99), 2012–2015 +5.1%/yr — and −1.5%/yr for 2012–2015 on the secondary
  universe. A DEV-window result that lives in one crisis is not a forward-looking edge, and
  it is consistent with reversal having been competed away post-decimalisation.
- **The insider *clustering* hypothesis is specifically refuted**, while a weaker one
  survives: clustered (≥2 buyers) minus single-buyer is +1.05%/yr at t=0.57, but
  single-buyer minus no-buyer is +5.57%/yr at t=3.49. The entire economically large step is
  0 → 1 buyer.
- **A convergence worth trusting:** the low-vol/quality sleeve landed within 0.1%/yr of the
  prior fundamental-composite study's B2 result (−5.5% vs −5.5% excess; Sharpe 0.32 vs
  0.36) from a completely unrelated signal. Two independent signals hitting the same number
  is evidence that the **cost floor in that capacity band, not signal quality, determines
  the answer** — the same conclusion §2(b) reaches from the per-round-trip table.
- **Do not re-run any of these six hypotheses with adjusted lookbacks, universes, bands or
  rebalance frequencies.** That is the selection bias the whole apparatus exists to refuse.
  The only legitimate follow-up is the one in §6, and only under the three conditions
  attached to it.

## 8. Data hygiene

DEV window only. The `load_prices` guard in `research/capacity_panel.py` was not bypassed by
any sleeve; the maximum bar date touched anywhere in the session is 2015-12-31. **The 2016+
confirmation window remains UNFIRED**, correctly — the prereg permits firing it only at a
model that passes the DEV-side gate, and no sleeve reached a positive excess, let alone the
0.75 net Sharpe gate or the 0.91–1.49 DSR bar. Nothing was downloaded. All artefact filters
(±100% return cap, $2 price floor, ≥90% non-zero-volume over trailing 63 days) were active,
and delisting terminal returns were booked once, gated on the event falling within 62 days
of the exit, with the name then removed from the book — the two prior bugs that produced
−60%/yr and −112%/yr are absent by direct count in every sleeve that checked.

**Reproduce §2–§5 with:** `.venv/Scripts/python.exe scripts/synthesise_breadth_sleeve_hunt.py`
