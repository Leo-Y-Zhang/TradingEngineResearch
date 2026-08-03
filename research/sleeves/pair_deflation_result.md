# RESULT — DEFLATING THE lowvol+trend PAIR

**Target:** `research/sleeves/_portfolio/portfolio_decision.json` headline —
*low-vol B2 + multi-asset trend, inverse-vol weighted, 213 months (1998-04..2015-12),
Sharpe 1.2166, vol 11.31%, CAGR 13.95%, max DD −19.08% at 1x* — recorded in
the internal research log iteration 15.

**Posture:** refute by default. Every number below was measured by running code in
`research/sleeves/_pair_deflation/`, not read back out of a result file. No new backtest,
no new signal, no new configuration was created; this study therefore does **not** raise
the programme's `n_trials`, which stays at 47.

Reproduce, in order:

```
.venv/Scripts/python.exe -m research.sleeves._pair_deflation.controls
.venv/Scripts/python.exe -m research.sleeves._pair_deflation.pair_deflation
```

Machine-readable output: `_pair_deflation/controls.json`, `_pair_deflation/pair_deflation.json`.

---

## VERDICT: **DEAD**

**The 1.2166 is real arithmetic on the wrong series.** It was computed from the
**registered** low-vol book (net Sharpe 0.8779) — the book that iteration 10's independent
adversarial verification had **already corrected to 0.614**. `portfolio_correlation_v2`
(commit `fcf9be4`, 08:14:44) and iteration 15's log entry (`1af10c0`, 08:15:18) were written
by concurrent agents 34 seconds apart; iteration 15 quotes the v1 file and does not carry
the v2 corrections. Repairing that one input, and nothing else, takes the pair to **0.9212**.

**Relationship to iteration 16 (`b722e51`).** That entry reached DEAD from the v2 study
while this one was running. **This is an independent corroboration on a different
construction, not an inheritance** — it reproduces the claim bit-for-bit first, isolates
which repair moves it, and runs the four tests iteration 15 demanded and v2 did not: the
deflation at n = 234 and n = 281 specifically, the bear-market exclusion, the frozen-weight
out-of-sample split, and the pair's own drawdown-capped leverage ladder. Two of those
produce findings that are **new**, and one of them **refutes a charge iteration 15 made**
(see §3).

At 0.9212 the pair **fails its deflated bar at every trial count that matters**: it misses
0.9443 at the programme ledger of 47 alone, 1.0872 at the 234-combination search, and
1.1022 at 234+47 = 281. Its DSR is **0.826 at 234 trials and 0.810 at 281** against a
required 0.95.

**The passive benchmark clears the same gate and the pair does not.** Monthly equal-weight
passive scores 0.6691 on 61.5 years against a bar of **0.5808 at n = 281** (DSR 0.9806).
The pair fails at 17.75 years because 17.75 years is not enough history to carry a 281-trial
deflation at Sharpe 0.92.

**Survivable-drawdown compound return (the number that answers "how close to 30%/yr"):**
**≈18%/yr at a ≤50% drawdown and ≈14%/yr at ≤35%**, primary financing (bill+150bp);
**≈16% and ≈13%** at retail (bill+300bp). Not 30%, and not from a validated book.

Two of the brief's specific charges do **not** survive, and are recorded as refuted:

- **The pair is NOT weight-overfit.** Frozen out-of-sample weights score 0.8290 against
  0.8626 refit in-sample — a gap of 0.034. Inverse-vol is effectively parameter-free.
- **The pair's raw Sharpe does NOT depend on the two bear markets.** It *rises* when they
  are excluded (0.9212 → 1.0031). The bear dependence is entirely in the
  **benchmark-relative** statistic — see §2, where it is severe.

---

## 0. CONTROLS — every recorded anchor reproduced BEFORE anything new was computed

`controls.py` aborts the run if any of these fails. All 18 passed.

| anchor | recorded | measured | \|diff\| |
|---|---:|---:|---:|
| `dsr_sharpe_bar(7yr, n=32)` | 1.488 | 1.4881138751 | 1.1e-04 |
| `dsr_sharpe_bar(40yr, n=32)` | 0.597 | 0.5970964146 | 9.6e-05 |
| `dsr_sharpe_bar(17.75yr, n=38)` — low-vol B2's bar | 0.9234 | 0.9233854511 | 1.5e-05 |
| **the claim: pair Sharpe** | 1.2165535517187802 | 1.2165535517 | **0.000e+00** |
| the claim: pair vol / CAGR / max DD | as filed | identical | **0.000e+00** |
| the claim: inverse-vol weight on low-vol | 0.6075611495217327 | identical | 0.000e+00 |
| low-vol registered, standalone | 0.877853588402183 | identical | 0.000e+00 |
| low-vol **corrected**, standalone (total basis) | 0.6138 | 0.6138478845 | 4.8e-05 |
| low-vol corrected, **excess** basis | 0.4869 | 0.4868894081 | 1.1e-05 |
| trend, own 738 months | 0.6116 | 0.6116253670 | 2.5e-05 |
| passive **monthly** | 0.6691 | 0.6691135560 | 1.4e-05 |
| passive **daily** | 0.7065 | 0.7064764801 | 2.4e-05 |
| v2's corrected pair [equal weight] | 0.9260 | 0.9259603726 | 4.0e-05 |
| v2's corrected pair [inverse vol] | 0.9212 | 0.9211519719 | 4.8e-05 |

The claim reproduces **bit-for-bit**. Nothing below rests on a harness disagreement.

The leverage engine was validated the same way, against `portfolio_correlation_v2`'s
recorded passive ladder (§4).

---

## THE DEFECT LADDER — which repair moves 1.2166, and by how much

One repair per rung, inverse-vol throughout, same 213 months.

| rung | pair Sharpe | low-vol standalone | bar @ n=234 | clears? |
|---|---:|---:|---:|---|
| 0. **as claimed (v1)** | **1.2166** | 0.8779 | 1.0872 | yes |
| 1. + one-month realignment | 1.2464 | 0.8779 | 1.0872 | yes |
| 2. + common excess-over-cash basis | 1.1351 | 0.7381 | 1.0872 | yes |
| 3. + **iteration-10 corrected low-vol book** (= v2) | **0.9212** | 0.4869 | 1.0872 | **no** |

**The dominant defect is rung 3, and it is not a judgement call.** The claim used the
*registered* low-vol book after an independent verification had established that the
registered levels were wrong (delisting window off by one, 777 exit legs charged nothing,
same-bar execution). Repairing the one-month dating error actually **helps** the claim
(+0.030), so the misalignment is not what inflated it; the convention mix is worth −0.111;
using the uncorrected book is worth **−0.214**.

---

## 1. DEFLATION — the pair against 234 and 281 trials

`bar` = the annual Sharpe needed for DSR ≥ 0.95 at that trial count and sample length
(`research.multiasset.panel.dsr_sharpe_bar`, Gaussian, therefore a **floor**).
`DSR` = `research.validation.deflated_sharpe_ratio` on the actual return series, which
prices the measured skew and kurtosis as well.

| book | yrs | S | bar@47 | **bar@234** | **bar@281** | DSR@47 | **DSR@234** | **DSR@281** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **PAIR as claimed (v1, inverse-vol)** | 17.8 | **1.2166** | 0.9443 | 1.0872 | 1.1022 | 0.9947 | **0.9769** | **0.9735** |
| **PAIR corrected (v2, inverse-vol)** | 17.8 | **0.9212** | 0.9443 | 1.0872 | 1.1022 | 0.9333 | **0.8256** | **0.8102** |
| PAIR corrected (v2, equal weight) | 17.8 | 0.9260 | 0.9443 | 1.0872 | 1.1022 | 0.9394 | 0.8379 | 0.8231 |
| low-vol B2 registered, alone | 17.8 | 0.8779 | 0.9443 | 1.0872 | 1.1022 | 0.8554 | 0.6901 | 0.6692 |
| low-vol B2 corrected, alone (excess) | 17.8 | 0.4869 | 0.9443 | 1.0872 | 1.1022 | 0.3744 | 0.1883 | 0.1730 |
| trend alone, on the pair's window | 17.8 | 0.6695 | 0.9443 | 1.0872 | 1.1022 | 0.7567 | 0.5524 | 0.5292 |
| trend alone, own 738 months | 61.5 | 0.6116 | 0.4999 | 0.5732 | 0.5808 | 0.9944 | **0.9758** | **0.9722** |
| passive monthly, on the pair's window | 17.8 | 0.4753 | 0.9443 | 1.0872 | 1.1022 | 0.3588 | 0.1773 | 0.1626 |
| **passive monthly, own 738 months** | 61.5 | **0.6691** | 0.4999 | 0.5732 | **0.5808** | 0.9964 | **0.9832** | **0.9806** |
| passive daily, own 736 months | 61.3 | 0.7065 | 0.5006 | 0.5739 | 0.5816 | 0.9987 | 0.9927 | 0.9914 |

### The decision file deflated at 69 trials, not 281

`portfolio_decision.json` records `dsr_bar_programme_trials = 0.9234` and
`dsr_bar_incl_combo_search = 0.9807`. Those invert to **n = 38 and n = 69**
(`portfolio_decision.py` hard-codes `N_TRIALS_PROGRAMME = 38`, `N_COMBOS_SEARCHED = 31`).
Measured: `dsr_sharpe_bar(17.75, n)` = 0.923385 at 38, **0.980742 at 69**, 1.087239 at 234,
1.102190 at 281.

So the file's "including the combination search" bar undercounts the search by a factor of
3–4. Its own `scan` array holds **189** configurations, and the study that produced the
correlation matrix searched **234** (58 subsets × 4 schemes). Neither is 31. The programme
ledger was also stale at 38 when iteration 14 had already moved it to 47.

**Read honestly, in both directions.**

- **On the series the claim was actually computed from, the pair SURVIVES deflation.**
  1.2166 clears 1.0872 at 234 and 1.1022 at 281, and its DSR stays above 0.95 at both.
  The brief's expectation that the pair would fail its deflated bar is **not** what the
  arithmetic says on that series, and saying otherwise would be dishonest.
- **The series is the problem, not the deflation.** On the corrected series the pair fails
  at 234, at 281, and — decisively — **at 47, the programme ledger alone**, before any
  credit is taken for the combination search. Deflation is not what kills it; the
  correction is. Deflation only removes the fallback argument that 0.92 is close enough.
- **Both constituents fail alone**, at every trial count, on their shared window.
- **Passive clears at 281 and the pair does not.** That comparison decides the question
  the study exists to answer.

### PBO — measuring the selection process directly

CSCV (Bailey/Borwein/López de Prado/Zhu, 16 splits) over **109 configurations**
(every sleeve subset × weighting scheme available on the 213-month window) —
the search the pair emerged from:

**PBO = 0.3231.** Best in-sample config = `lowvol+trend+defensive+passive [inverse_vol]`
at 1.2769 — *not the pair*, which corroborates iteration 13's finding that the window
flatters `defensive`.

PBO of 0.32 is real but is **not** the catastrophic ≥0.5 that would mean the search is
pure noise-mining. This is consistent with §3: the weighting is not what is overfit. It
is a 32% probability that the in-sample winner lands in the worse half out of sample —
enough to distrust "best of 234", not enough to call the whole exercise noise.

---

## 2. THE BEAR-MARKET DEPENDENCE — the charge is UPHELD, but not where the brief looked

Windows taken verbatim from `lowvol_retest_verification.md` §5: dot-com = 2000-01..2002-12,
GFC = 2008-01..2011-12. `vs pasv` = vol-matched active against monthly equal-weight passive
on the same surviving months, `vm t` its Newey-West(4) t-statistic.

### Corrected basis (the honest one)

| window | n | pair S | mean/yr | NW4 t | low-vol S | trend S | passive S | **vs pasv** | **vm t** | bar@234 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full window | 213 | 0.9212 | 10.92% | 3.77 | 0.4869 | 0.6706 | 0.4667 | **+5.39%** | **1.62** | 1.0872 |
| ex dot-com | 177 | 0.9143 | 11.34% | 3.37 | 0.4668 | 0.7390 | 0.7849 | **+1.60%** | **0.45** | 1.1993 |
| ex GFC | 165 | 0.9977 | 11.54% | 3.56 | 0.7628 | 0.6595 | 0.5592 | +5.07% | 1.31 | 1.2452 |
| **ex BOTH bears** | **129** | **1.0031** | 12.29% | 3.11 | 0.8278 | 0.7552 | 1.1299 | **−1.55%** | **−0.38** | 1.4224 |
| dot-com only | 36 | 1.0069 | 8.87% | 1.97 | 0.5901 | 0.3607 | −0.8225 | **+16.12%** | **2.36** | 3.0879 |
| GFC only | 48 | 0.6810 | 8.79% | 1.38 | 0.0972 | 0.7017 | 0.2570 | +5.47% | 0.89 | 2.5390 |

### As-claimed basis, for comparison

| window | n | pair S | low-vol S | passive S | vs pasv | vm t |
|---|---:|---:|---:|---:|---:|---:|
| full window | 213 | 1.2166 | 0.8779 | 0.4753 | +8.39% | 2.54 |
| ex dot-com | 177 | 1.1931 | 0.8071 | 0.7967 | +4.55% | 1.33 |
| ex GFC | 165 | 1.3547 | 1.2436 | 0.5718 | +8.39% | 2.23 |
| **ex BOTH bears** | 129 | **1.3572** | 1.2552 | 1.1517 | **+2.21%** | **0.58** |
| dot-com only | 36 | 1.3269 | 1.2295 | −0.8225 | +22.94% | 3.11 |

**What this establishes.**

1. **The naive form of the charge is refuted.** The pair's *absolute* Sharpe does not
   collapse without the bears — it goes **up**, 0.9212 → 1.0031 corrected, 1.2166 → 1.3572
   as claimed. Anyone reporting "the pair dies without the bears" on the raw Sharpe would
   be wrong.
2. **The real form of the charge is upheld, and it is worse than stated.** Passive's own
   Sharpe on the bear-free window is **1.1299** — *higher than the pair's 1.0031*. Measured
   against passive at matched volatility the corrected pair earns **−1.55%/yr at t −0.38**
   outside the two bears. **It loses to buy-and-hold.**
3. **The whole benchmark-relative edge is 36 months.** Dot-com alone carries **+16.12%/yr
   at t 2.36** (claim basis: +22.94% at t 3.11), exactly matching the constituent
   verification's finding that the dot-com bust carries low-vol's entire vol-matched
   active. The GFC contributes +5.47% at t 0.89 — nothing.
4. Even on its own terms, the bear-free window's deflated bar rises to **1.4224** at
   n = 234 (129 months is a shorter sample), which neither basis clears.

**Correct label: a crash hedge, not a return engine.** It must never be described as an
all-weather book. Its absolute return is fine in calm markets; it simply has no edge over
holding the panel in calm markets, which is the property that matters.

---

## 3. OUT-OF-SAMPLE SPLIT — the weighting is clean, the low-vol leg decays

213 months split 106 / 107. Weights fitted on half 1 and applied **unchanged** to half 2
("frozen"), against weights refitted inside half 2 ("refit", i.e. cheating).

### Corrected basis, split at 2007-03

| scheme | H1 in-sample | **H2 FROZEN** | H2 refit | full window | H2 bar @ n=234 (8.9yr) |
|---|---:|---:|---:|---:|---:|
| inverse_vol | 1.0084 | **0.8290** | 0.8626 | 0.9212 | **1.5769** |
| equal_weight | 0.9830 | **0.8776** | 0.8776 | 0.9260 | 1.5769 |
| erc | 1.0084 | 0.8290 | 0.8626 | 0.9212 | 1.5769 |

### As-claimed basis, split at 2007-02

| scheme | H1 in-sample | **H2 FROZEN** | H2 refit | full window | H2 bar @ n=234 |
|---|---:|---:|---:|---:|---:|
| inverse_vol | 1.6270 | **0.9311** | 0.9405 | 1.2166 | 1.5769 |
| equal_weight | 1.4242 | 0.9258 | 0.9258 | 1.1430 | 1.5769 |
| erc | 1.6270 | 0.9311 | 0.9405 | 1.2166 | 1.5769 |

### Standalone halves

| basis | low-vol H1 | low-vol H2 | trend H1 | trend H2 |
|---|---:|---:|---:|---:|
| as claimed | 1.3729 | **0.5606** | 0.7285 | 0.6241 |
| corrected | 0.6593 | **0.3886** | 0.7009 | 0.6483 |

**Findings, both directions.**

- **The pair is not weight-overfit, and the brief's third charge is refuted.** Frozen
  weights lose only **0.034** of Sharpe against refitting (0.8290 vs 0.8626) on the
  corrected basis, and **0.009** as claimed. Inverse-vol on two sleeves has essentially no
  free parameters; there is nothing there to overfit. A pair that "only works in-sample"
  is not what the split shows.
- **What the split does show is decay in the low-vol leg**: 0.6593 → 0.3886 corrected,
  1.3729 → 0.5606 as claimed. Trend is stable (0.70 → 0.65). The first half contains the
  dot-com bust — this is the same finding as §2 seen through a different cut.
- **Neither half clears its own deflated bar.** At 8.9 years the bar at n = 234 is
  **1.5769**; the best out-of-sample half reads 0.9311. Splitting the sample cannot rescue
  the result — it makes the bar worse faster than it makes the Sharpe better.

---

## 4. THE LEVERAGE QUESTION, DONE PROPERLY

**The half-Kelly figure in `portfolio_decision.json` is not used for any conclusion here.**
For the record, it asserts 55.50% growth requiring 60.83% volatility = **5.376× leverage**
on a book whose 1× drawdown is −19.08%. The whole ladder was rebuilt with financing charged
and the levered path recompounded: on a coherent excess basis the as-claimed book's
**unconstrained peak is 60.82%/yr at 7.45× — at a measured drawdown of −91.12%.**
Ruin. Forbidden by standing rule 7. It is quoted only to be discarded.

**Method.** An excess-return book levered `L` returns `L·x − max(L−1,0)·spread/12 + cash`,
`cash` = `US_CASH_13W`. The low-vol leg is converted to excess so the borrowing charge is
coherent (v1 mixed a total-return leg into a levered book, which implicitly borrows at 0%).
Leverage is solved by bisection against two drawdown definitions: the **observed path**,
and the **95th percentile of drawdown magnitude across a 12-month-block bootstrap** of the
same months (2,000 resamples). The second is the honest one — solving against a single
observed maximum systematically over-levers.

### Engine validation, before any pair number is quoted

| quantity | this engine | `portfolio_correlation_v2` recorded | iteration 11 recorded |
|---|---:|---:|---:|
| passive monthly Sharpe | 0.66911 | 0.6691 | 0.6678 |
| passive, DD≤50%, observed path | +14.11% @ 1.98× | +14.02% @ 1.95× (0.05 grid) | ~1.9× |
| passive, DD≤50%, bootstrap p95 | +12.32% @ 1.45× | +12.13% @ 1.40× (4000 draws) | — |
| passive, unconstrained peak | +19.34% | +19.34% | 15.83% (vol-targeted) |

The engine agrees with v2 to the grid resolution. It reads **rich by a known factor**:
iteration 11 levered to a *volatility target* and charged its own rebalancing, this engine
applies *static* leverage to an already-costed series; v2 measured the ratio at **0.877**.
`14.11% × 0.877 = 12.37%`, which reproduces iteration 11's **12.30%** — so the factor is
validated, not assumed, and it is applied to every figure below.

### THE ANSWER: highest compound return at a survivable drawdown

`×0.877` is the iteration-11-calibrated column and is the one to quote.

| book | financing | cap | solved against | leverage | engine | **×0.877** |
|---|---|---:|---|---:|---:|---:|
| **pair, corrected** | bill+150bp | 50% | observed path | 3.43× | 31.43% | 27.57% |
| **pair, corrected** | **bill+150bp** | **50%** | **bootstrap p95** | **1.89×** | 20.58% | **18.05%** |
| **pair, corrected** | bill+150bp | 35% | observed path | 2.36× | 24.23% | 21.25% |
| **pair, corrected** | **bill+150bp** | **35%** | **bootstrap p95** | **1.30×** | 15.61% | **13.69%** |
| pair, corrected | bill+300bp retail | 50% | observed path | 3.27× | 26.16% | 22.94% |
| **pair, corrected** | **bill+300bp retail** | **50%** | **bootstrap p95** | **1.82×** | 18.56% | **16.28%** |
| pair, corrected | bill+300bp retail | 35% | observed path | 2.28× | 21.30% | 18.68% |
| **pair, corrected** | **bill+300bp retail** | **35%** | **bootstrap p95** | **1.28×** | 14.93% | **13.09%** |
| pair, as claimed | bill+150bp | 50% | bootstrap p95 | 1.92× | 24.85% | 21.80% |
| pair, as claimed | bill+150bp | 35% | bootstrap p95 | 1.30× | 18.18% | 15.95% |
| pair, as claimed | bill+300bp retail | 50% | bootstrap p95 | 1.86× | 22.64% | 19.85% |
| pair, as claimed | bill+300bp retail | 35% | bootstrap p95 | 1.28× | 17.45% | 15.30% |
| passive, **own 61.5 years** | bill+150bp | 50% | bootstrap p95 | 1.45× | 12.32% | 10.80% |
| passive, own 61.5 years | bill+150bp | 35% | bootstrap p95 | 1.01× | 10.66% | 9.35% |
| passive, own 61.5 years | bill+300bp retail | 50% | bootstrap p95 | 1.41× | 11.49% | 10.08% |
| passive, **the pair's 213 months only** | bill+150bp | 50% | bootstrap p95 | 1.15× | 6.61% | 5.80% |
| passive, the pair's 213 months only | bill+300bp retail | 50% | bootstrap p95 | 1.14× | 6.37% | 5.59% |

**The number that answers "how close to 30%/yr is this": ≈18%/yr** at a ≤50% drawdown on
primary financing, **≈14%/yr** at ≤35%, from the corrected pair. At retail margin, **≈16%**
and **≈13%**. Solving against the observed path instead flatters this by 7–9 percentage
points and is the wrong solve.

**On the retail inversion.** Iteration 11 found retail financing *inverts* the leverage
ladder past τ≈20%. That is not reproduced here and the two are not comparable: iteration 11
levered to a **volatility target**, this engine applies **static** leverage inside a
drawdown cap, and the cap binds at 1.3–1.9× where the financing drag is still small.
Retail financing costs 1.8pp at ≤50% and 0.6pp at ≤35% here. The inversion finding is
neither confirmed nor refuted by this study — different construction.

**Note the last two rows.** Passive levered on *the pair's own 213 months* reaches only
5.80%/yr, against 10.80% on its own 61.5 years. **The 1998–2015 window is unusually bad for
passive as well as unusually good for low-vol.** Part of the pair's apparent superiority is
the denominator.

---

## 5. CONSTITUENT HONESTY — can two individually-failed sleeves be promoted?

**No, and the project's own rules already say so.**

| sleeve | its own registered verdict | Sharpe used in the claim | honest Sharpe | its own gate |
|---|---|---:|---:|---|
| low-vol B2 | **MARGINAL** (`lowvol_retest_verification.md`) | 0.8779 | **0.6138** total / **0.4869** excess | **FAILS** gate (iii): needs 0.9234 |
| trend | loses to its own universe at matched vol | 0.6695 (on this window) | 0.6116 (738 months) | no benchmark-relative pass |

- Low-vol **fails its own registered promotion gate (iii)** by 0.31 of Sharpe on the
  corrected series. Its DSR at the programme ledger of 47 is **0.3744**; at 234 it is
  **0.1883**. It is not a validated sleeve by any reading.
- Trend is the honest half of the pair — 0.6116 on **738 months**, DSR **0.9722 at 281
  trials**, and it clears its bar. But it does so at a level far below what 30%/yr needs,
  and it does not beat the passive book it is built from (0.6116 vs 0.6691).
- **Iteration 3d already recorded the rule: combining sleeves that individually failed does
  not produce a validated strategy.** It applies here. A portfolio Sharpe is not a
  promotion; the gate is applied to what is promoted, and neither leg passes it.

### `selection_rule` has no benchmark-relative criterion — confirmed by reading the code

`research/validation.py::selection_rule` (line 401) requires all of: `mean_rank_ic > 0.01`,
`sharpe_net > 0.75`, `stability_score > 0.60`, `deflated_sharpe_proxy > 0.25`,
`deflated_sharpe_ratio >= 0.95`, no leakage flags, and no regime Sharpe < −0.50.

**None of the seven is benchmark-relative.** A candidate that loses to buy-and-hold can
satisfy every one of them. That is exactly what happened when the gate passed trend at
0.612 while the passive book it is built from scored 0.669 — and it is what would happen
again to the corrected pair on criterion 2 alone (0.9212 > 0.75) if the DSR criterion were
not there to stop it. **The DSR criterion is currently the only thing standing between this
programme and promoting a book that loses to buy-and-hold.** That is a single point of
failure in the gate and it should be recorded as one.

*(This is an observation about the gate, not a proposal to change it. The 2026-07-27
amendment spec left `selection_rule` unchanged and this study does not reopen it.)*

---

## 6. THE BENCHMARK, AT MATCHED VOLATILITY AND THROUGH THE SAME GATE

**Label:** the benchmark used throughout is the **MONTHLY-rebalanced** equal-weight book of
the 18 panel instruments, Sharpe **0.6691**. The 0.7065 figure recorded elsewhere is the
**DAILY-rebalanced** variant (they correlate +0.9993 and differ in vol scaling, 8.79% vs
24.80%; Sharpe is scale-invariant so the correlation and Sharpe tables do not turn on it,
but the leverage ladder does).

### Matched volatility, on the pair's own 213 months

| comparison | n | k (bench scaled to pair risk) | **vol-matched active** | **NW4 t** | S pair | S bench |
|---|---:|---:|---:|---:|---:|---:|
| pair **as claimed** vs passive | 213 | 1.1853 | **+8.39%/yr** | **+2.54** | 1.2166 | 0.4753 |
| pair **corrected** vs passive | 213 | 1.2404 | **+5.39%/yr** | **+1.62** | 0.9212 | 0.4667 |
| pair corrected, **ex both bears** | 129 | — | **−1.55%/yr** | **−0.38** | 1.0031 | 1.1299 |

The corrected pair's edge over passive **fails a t > 2 test on the full window** and
**reverses sign outside the bears**.

### Through the deflated gate

| book | years | S | bar @ 234 | bar @ 281 | DSR @ 281 | verdict |
|---|---:|---:|---:|---:|---:|---|
| **passive monthly, own 738 months** | 61.5 | 0.6691 | 0.5732 | **0.5808** | **0.9806** | **CLEARS** |
| passive daily, own 736 months | 61.3 | 0.7065 | 0.5739 | 0.5816 | 0.9914 | CLEARS |
| trend alone, own 738 months | 61.5 | 0.6116 | 0.5732 | 0.5808 | 0.9722 | CLEARS |
| **pair corrected, 213 months** | 17.8 | 0.9212 | 1.0872 | **1.1022** | **0.8102** | **FAILS** |
| pair as claimed, 213 months | 17.8 | 1.2166 | 1.0872 | 1.1022 | 0.9735 | clears — on the wrong series |

**Passive clears the deflated gate at 281 trials. The corrected pair does not.** It is
worth being precise about *why*: passive's Sharpe is lower, but it has 61.5 years against
17.75, and sample length beats point-estimate Sharpe in this gate. That is iteration 12's
sample-length lever, and it is the only lever in this document that actually works.

---

## WHAT I VERIFIED vs WHAT I TOOK ON TRUST

**Verified by running code:** every anchor in §0 including bit-for-bit reproduction of the
claim; the defect ladder; DSR and the DSR bar at four trial counts on ten books; PBO over
109 configurations; six bear-market windows on two bases with vol-matched actives and NW(4)
t-statistics; the 106/107 split with frozen and refit weights under three schemes; the full
leverage ladder under two financing rates and two drawdown caps solved against both the
observed path and a block bootstrap, with the engine cross-validated against v2's and
iteration 11's recorded passive figures; the matched-volatility comparison against both
benchmark variants; `selection_rule`'s criteria read directly from source.

**Taken on trust (not re-derived here):** the correctness of
`extract_lowvol_corrected_monthly`'s regeneration of the corrected book (it self-asserts
against eight verification targets and those assertions pass, but I did not re-run the
underlying `_lowvol_verify` attack chain); `portfolio_correlation_v2`'s proof that the
low-vol series is dated one month early (I reproduced the corrected series, not the
alignment control); the 0.877 optimism factor as a *constant* rather than a book-specific
ratio — it was measured on passive and applied to the pair, which is an approximation and
is labelled as one; the claim that the combination search was 234 configurations and the
programme ledger 47.

---

## THE DIRECT ANSWER

- **Does the pair clear its deflated bar?** On the series it was computed from, yes
  (1.2166 vs 1.0872 at n=234 and 1.1022 at n=281). On a corrected series, **no** — it fails
  at 47, 234 and 281, with DSR 0.933 / 0.826 / 0.810.
- **Is it a crash hedge?** Yes, in the sense that matters. Its absolute return survives
  without the bears; its **edge over buy-and-hold does not** (−1.55%/yr, t −0.38). The
  entire benchmark-relative result is 36 months of the dot-com bust.
- **Does it survive out of sample?** Partly. The *weighting* survives cleanly (0.034 of
  Sharpe lost to freezing). The *low-vol leg* decays 0.66 → 0.39. Neither half clears the
  bar at its own length.
- **How close to 30%/yr?** **≈18%/yr at ≤50% drawdown, ≈14%/yr at ≤35%**, primary
  financing; ≈16% / ≈13% retail. From a book that fails its gate — so these are the
  properties of an unvalidated backtest, not a route.
- **Verdict: DEAD.** The headline number came from a superseded input, the pair fails
  deflation once that input is repaired, it loses to passive outside two bear markets, and
  passive clears the same gate that it fails. **The honest headline remains what iteration
  11, v2 and iteration 16 already recorded: ~12–17%/yr at a survivable drawdown, and 30%/yr
  is not reachable on this panel.**

### What this study adds that was not already banked

1. **Bit-for-bit reproduction of the claim, then a defect ladder** that separates the
   repairs: realignment **+0.030** (it *helps* the claim), convention **−0.111**, corrected
   low-vol book **−0.214**. The dominant defect is a single stale input, not the alignment
   bug that got the attention.
2. **The deflation the brief demanded**: bars of 1.0872 at n = 234 and 1.1022 at n = 281,
   with DSR 0.826 / 0.810 — and the discovery that `portfolio_decision.json` deflated at
   **n = 69**.
3. **PBO = 0.3231** over 109 configurations — the selection process measured directly.
4. **The bear-market exclusion, which had not been run on the pair.** Its raw Sharpe rises
   without the bears; its edge over passive goes to **−1.55%/yr at t −0.38**. This is the
   sharpest single number in the file.
5. **The frozen-weight out-of-sample split**, which **refutes** iteration 15's implicit
   weight-overfitting charge (0.034 of Sharpe lost to freezing).
6. **The pair's own drawdown-capped leverage ladder** at two financing rates, replacing the
   forbidden half-Kelly figure with ≈18% / ≈14%.
7. **`selection_rule` has no benchmark-relative criterion**, read from source — so the DSR
   criterion is the gate's only defence against promoting a book that loses to
   buy-and-hold.

**`portfolio_decision.json` must not be quoted. Its headline rests on a superseded input
and it deflates at 69 trials when the honest count is 281. Use
`portfolio_correlation_v2.json` and this file.**
