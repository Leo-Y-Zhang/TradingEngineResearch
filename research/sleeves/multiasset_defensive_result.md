# RESULT — DEFENSIVE / BETTING-AGAINST-BETA on the long-history multi-asset panel

**Pre-registered** in `multiasset_defensive_prereg.md` before any backtest was run.
**Run once**, six declared arms, no tuning, no arm dropped.
Code: `multiasset_defensive.py`, `multiasset_defensive_run.py`,
`multiasset_defensive_verify.py`. Receipts: `research/sleeves/_defensive/`.
Suite green at 1,389 passed / 1 skipped.

---

## VERDICT: **DEAD**

Sleeve #15. Net of 10bps, over **52.4 years (1974-02 → 2026-06, 629 months)**:

| | strategy | benchmark (equal-weight long-only, same universe) |
|---|---:|---:|
| annual Sharpe | **0.114** | **0.690** |
| Newey-West t | 0.695 | 4.722 |
| **DSR** (n_trials=38) | **0.089** | **0.994** |
| DSR bar at 52.4yr | 0.530 | 0.530 |
| arithmetic return | +2.50%/yr | +6.25%/yr |
| geometric return | **+0.05%/yr** | +6.00%/yr |
| volatility | 22.0% | 9.1% |
| max drawdown | **−85.9%** | −30.4% |

**Vol-matched active return: −5.22%/yr, Newey-West t = −2.63.**
(Same figure under the brief's convention — benchmark levered 2.43x to the strategy's own
volatility — is **−12.68%/yr at the identical t = −2.63**.)

This sleeve is not marginal and it is not ambiguous. It fails all four pre-registered
deployment criteria, it is beaten by its own passive universe by a *statistically
significant* margin, and its own benchmark clears the DSR gate while it scores 0.089.

**It does not go in the portfolio. Adding it to trend+carry lowers the combined Sharpe
from 0.655 to 0.542.**

---

## 1. What actually happened: panel-wide BAB is a levered bond-and-dollar book

This was **predicted in advance** (prereg P2) and it is the most useful thing the study
produced. It is confirmed harder than predicted.

Against an equal-weight proxy whose variance is dominated by equity and commodities, the
low-beta half of this panel is *always* the same names. Measured mean betas since 2006:

| | USDX | US30Y | US5Y | US10Y | JPYUSD | EURUSD | GBPUSD | ... | HSI | COPPER | SILVER | WTI |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| beta | −0.51 | −0.02 | 0.03 | 0.04 | 0.18 | 0.51 | 0.54 | | 1.79 | 1.83 | 2.42 | 2.59 |

So `betaL` averages **0.097** while `betaH` averages **1.797**, and the hedge ratio
`rho = betaL/betaH` averages **0.081** (median 0.028). Consequences, all measured:

- the short leg is **7.5% of mean gross exposure**. There is essentially no short book.
- `rho` is clipped at zero — `betaL` outright negative — in **41.7% of months**.
- mean **net exposure is +4.37x** long.
- gross leverage sits **53.6% in the three bond series, 24.3% in USDX**, and only
  **10.5% across all seven equity indices combined**.
- correlation to a simple equal-weight book of the three bonds: **0.631**, beta 1.609,
  R² 0.398, and the **alpha over levered bonds is −0.77%/yr (t = −0.27)**.

**Panel-wide "betting against beta" on a multi-asset panel is levered bonds and a long
dollar position, and it adds nothing over just holding levered bonds.** It ran over
1974-2020, which contains the largest bond bull market in recorded history, and it still
could not produce a Sharpe of 0.12.

This is the same family of failure as the value sleeve's: a construction that *looks* like
a distinct mechanism and is arithmetically something else. It is caught here by
measurement, in advance, rather than discovered afterwards.

---

## 2. The correlation — the deliverable, and it is genuine

The sleeve was chosen for its correlation, so this is the number that mattered.

| | correlation to trend | correlation to carry |
|---|---:|---:|
| PRIMARY, net 10bps | **+0.020** | **+0.478** |
| PRIMARY, gross | +0.020 | +0.478 |
| **S3, overlap removed** | **−0.037** | **+0.440** |
| **change** | **−0.057** | **−0.038** |
| Spearman (primary) | +0.045 | +0.425 |
| months | 629 | 269 |

**Prediction P5 confirmed: the correlations are ECONOMIC, not mechanical.** Arm S3
re-estimates *every* position input — the beta and the inverse-vol sizing — on months
`t-47 .. t-12`, so **nothing in that book has seen the 12 months trend's signal is
computed from**, and the correlations barely move. Contrast the value sleeve, where the
same excision moved trend correlation from −0.164 to −0.013 and destroyed the entire
thesis.

Two caveats, both measured rather than waved away:

- The two books share a volatility-targeting estimator, which can create co-movement in
  *magnitude* without any co-movement in direction. Measured: **correlation of |returns|
  to trend is 0.286** against a return correlation of 0.020. The shared machinery is
  visible, and it is not what the return correlation is made of.
- **Correlation to carry is +0.478, outside the pre-registered [0.00, +0.40] band.** Both
  books are long bonds. That is not a defect in either sleeve; it is the finding that
  these two are substantially the same bet.

And the correlation is worth nothing here anyway, because of §4.

---

## 3. Every arm, reported

At the 20% vol target, net of 10bps, 52.4 years except where stated:

| arm | Sharpe | DSR | vol-matched active | t | verdict |
|---|---:|---:|---:|---:|---|
| **PRIMARY** (panel-wide) | **0.114** | 0.089 | −5.22%/yr | **−2.63** | DEAD |
| S1 within-block | 0.150 | 0.160 | −4.86%/yr | −2.91 | DEAD |
| S2 hedged to zero beta | 0.153 | 0.143 | −4.86%/yr | −2.56 | DEAD |
| S3 overlap removed | 0.154 | 0.145 | −4.86%/yr | −2.55 | DEAD |
| S4 unscreened panel | 0.114 | 0.089 | −5.22%/yr | −2.63 | **vacuous, see below** |
| S5 placebo (263 mo) | −0.008 | 0.014 | −4.46%/yr | −1.81 | null as designed |

DSR bar at 52.4 years, n_trials = 38: **0.530**. Nothing is within 0.37 of it.

**S1, within-block, is the arm Frazzini-Pedersen would actually run**, and it is the more
honest construction: three of the four block books are exactly beta-neutral by
construction (equity, rates and commodity ex-ante book betas are 0 to machine precision,
with `rho` of 0.543, 0.377 and 0.435). It still only reaches 0.150 — and it is dominated
by the same instrument, with **45.4% of gross leverage in the 5-year bond**, because
inverse-vol sizing hands the lowest-volatility instrument the largest notional.

**The FX block inside S1 is broken and is reported as broken.** Its low-beta name is
always USDX, which is approximately minus the basket of the other three, so `betaL`
averages **−0.90** and `rho` is clipped to zero in **100% of the months the FX block is
on** (2007 onward). The FX sub-book is an unhedged long-dollar / short-currency position,
not a BAB book. Its block proxy is not degenerate — proxy vol 6.9% against 9.0% mean
member vol, mean pairwise member correlation −0.16 — so this is a ranking problem, not a
variance problem.

**S4 is a no-op and cannot be counted as a robustness pass.** Measured: the screened and
unscreened **month-end** panels are **identical over this universe (0 differing cells)**.
The quarantine drops a corrupt daily *level*, so the genuine move across it survives as
one valid two-day return and monthly compounding is unchanged. Any month-end sleeve that
reports "unchanged under the unscreened panel" has learned nothing. Recorded here so no
future study spends an arm on it.

---

## 4. Sharpe per decade — the sleeve fails this outright

Net of 10bps, primary arm, against the same-period benchmark:

| decade | months | strategy Sharpe | strategy return | benchmark Sharpe |
|---|---:|---:|---:|---:|
| 1970s | 71 | **−1.018** | −29.0%/yr | +0.283 |
| 1980s | 120 | +0.472 | +10.0%/yr | +0.947 |
| 1990s | 120 | +0.207 | +4.3%/yr | +0.969 |
| 2000s | 120 | +0.629 | +14.0%/yr | +0.393 |
| 2010s | 120 | +0.498 | +9.5%/yr | +0.611 |
| 2020s | 78 | **−0.580** | −11.6%/yr | +0.785 |

Two negative decades, and **the strategy loses to its own passive universe in five decades
out of six.** The one decade it wins is the 2000s, when the benchmark was hurt by two
equity bear markets and the strategy was long bonds.

P&L concentration: **2008 alone is 38.9% of the signed lifetime P&L** (5.07% on an
absolute basis), 31 of 53 calendar years positive. The signed shares are unstable here
and should be read with care — the denominator is a near-zero lifetime total, which is
also why the largest single instrument share reads 75% (N225) and the largest single cell
share reads *negative* 15.6%. The absolute-basis figures are the meaningful ones and they
are unremarkable: largest single cell 0.38% of gross absolute P&L. **This result was not
made or destroyed by one print.**

---

## 5. Costs are not what killed it

| | 2bps | 10bps |
|---|---:|---:|
| net Sharpe | 0.123 | 0.114 |
| cost drag | 0.049%/yr | 0.247%/yr |

Turnover is **4.94x/yr**. Gross Sharpe is 0.125. **The entire cost bracket is worth 0.011
of Sharpe.** Unlike every equity-cross-section study in this programme, costs are
irrelevant here — the gross edge simply does not exist.

---

## 6. Adversarial verification — nine checks, all run

`multiasset_defensive_verify.py` → `_defensive/verification.json`.

- **W3 — POINT-IN-TIME PROOF BY TRUNCATION. PASS.** The book was re-run on a panel that
  physically *ends* in 1999 and the pre-1999 weights are identical to the full-sample run
  to **1.8e-15** (betas identical to 0.0) across 853 months. Nothing reads forward.
- **W4 — perfect-foresight positive control. PASS.** Ranking by next month's return
  instead of by beta, with every other component untouched, gives **net Sharpe 3.687**.
  The pipeline can express an edge, so the negative result is interpretable.
- **W6 — the naive book, measured rather than asserted.** Long low-beta / short high-beta
  with no beta neutralisation: **Sharpe −0.370, realised beta to the proxy −2.124
  (t = −27.6), R² 0.548.** The pre-registration's claim that the un-neutralised version is
  just a levered short of the panel is confirmed as a measurement. By contrast the real
  BAB book's **realised beta is +0.021 (t = 0.21), R² 0.00007** — despite `rho` being
  clipped at zero in 41.7% of months, the book *is* beta-neutral ex post. **The
  construction works. The premium is not there.**
- **W2 — beta sanity on known answers. PASS.** NASDAQ beta > SPX beta; USDX beta negative
  (−0.158 full sample, −0.514 since 2006); since 2006 every equity beta exceeds every bond
  beta. *(One check written into the script came back false: NASDAQ is not the
  highest-beta equity full-sample — HSI is, at 1.94 vs 1.76. Hong Kong really is the more
  volatile market over 1974-2026, so this is the estimator working. The check was
  rewritten to the defensible form and both are recorded.)*
- **W8 — placebo.** A random ranking rarely clears the beta-spread guard, so the placebo
  book is on for only 263 months and churns at **95.8x turnover/yr against the real book's
  4.94x**. On its own months: placebo −0.008, real +0.096. The real signal beats noise;
  both round to nothing.
- **W9 — independent recomputation from the written CSV.** Every headline reproduces.
  The variance-drag identity `geo_excess = arith_active − (var_s − var_b)/2` closes to
  0.18%/yr (a second-order log term).

### W5 — a pre-registered prediction that was WRONG, and why

The prereg asserted the Sharpe would be invariant to the vol target, because gross return,
turnover and cost all scale with `k`. Measured, it is not:

| target | Sharpe, all months | Sharpe, excluding the 12 no-vol-estimate months | cap binding |
|---|---:|---:|---:|
| 10% | 0.064 | 0.163 | 1.0% |
| 20% | **0.114** | 0.164 | 1.0% |
| 40% | 0.164 | 0.192 | 25.2% |
| **spread** | **0.100** | **0.029** | |

The cause is a **latent defect in the machinery this sleeve inherited from the trend
sleeve**: `k = min(vol_target/sigma_book, GROSS_CAP/gross)`, and for the book's first
**12 months** `sigma_book` does not exist yet, so `k` silently falls through to the cap
and the book runs at **full 10x leverage at every target identically**. Twelve months out
of 629, at the start of the sample, move the full-sample Sharpe by 0.050. The residual
0.029 spread is the 40% target's cap binding a quarter of the time.

**Neither is edge, and the verdict is unchanged either way** — 0.164 is still 0.37 below
the DSR bar. But the defect is real, it is in the shared code path, and any sleeve built
on `multiasset_trend`'s scaler carries it.

### W10 — a methodological finding worth banking

Three arms reported a vol-matched active return of −0.0486 to four decimals. That is not a
coincidence. Algebraically, with `scale = sd_b/sd_s`:

```
mean(a*scale − b) * 12  ==  vol_bench_annual * (Sharpe_strat − Sharpe_bench)
```

verified to **0.0 / 2.8e-17 / 1.1e-16** on the primary, naive and foresight books.

**The mandated matched-volatility comparison IS the Sharpe comparison, rescaled by the
benchmark's volatility.** It cannot reverse a Sharpe ranking, which is exactly why it is
the right test and the raw arithmetic active return — which *can* reverse one, and did, on
trend — is the wrong one. Its t-stat is the honest significance of the Sharpe gap. Any
future sleeve can skip the ceremony and report the Sharpe gap with this t-stat.

---

## 7. Portfolio arithmetic — the sleeve makes the book worse

All three sleeves exist together only on carry's window: **269 months, 2004-02 → 2026-06**.
Measured correlations on that window:

| | defensive | trend | carry |
|---|---:|---:|---:|
| **defensive** | 1.000 | +0.066 | **+0.478** |
| **trend** | | 1.000 | −0.044 |
| **carry** | | | 1.000 |

Sharpes on that window: defensive **0.179**, trend 0.475, carry 0.430.
Effective N of the correlation matrix: **2.59**.

Equal-risk blends, measured by building the blend and computing its Sharpe (not by
formula), each rescaled to 20%/yr volatility:

| blend | Sharpe | geometric | max DD | NW t | DSR |
|---|---:|---:|---:|---:|---:|
| **trend + carry** | **0.655** | +11.7%/yr | −26.7% | 2.98 | 0.844 |
| trend + defensive | 0.448 | | | | |
| carry + defensive | 0.354 | | | | |
| **trend + carry + defensive** | **0.542** | +9.3%/yr | −40.3% | 2.41 | 0.705 |

The trend+carry figure of **0.6546 reproduces the previously banked 0.655 exactly**, which
is the cross-check that this measurement path is the same one.

The brief's formula `S = s*sqrt(N/(1+(N-1)*rho))` with mean pairwise `rho = 0.166` gives
0.5420; the exact equal-risk answer `S = sum(s_i)/sqrt(1'C1)` gives **0.5420**. They agree
to 16 decimal places here, so the approximation costs nothing at this correlation
structure.

**Half-Kelly growth, reported under the programme's standing rule of 2026-07-28** — never
the bare `g = 3S²/8`, always with the volatility it requires, the leverage that implies on
the series' own volatility, and the measured drawdown scaled by that leverage (linearly,
which is optimistic for a fat-tailed path):

| blend | Sharpe | half-Kelly growth | required vol | leverage | measured max DD | **implied max DD** |
|---|---:|---:|---:|---:|---:|---:|
| trend + carry | 0.655 | 16.1%/yr | 32.7% | 1.64x | −26.7% | **−43.8%** |
| trend + carry + defensive | 0.542 | 11.0%/yr | 27.1% | 1.36x | −40.3% | **−54.7%** |
| defensive alone | 0.114 | 0.5%/yr | 5.7% | 0.26x | −85.9% | −22.2% |
| **30%/yr target** | **0.894** | 30.0%/yr | **44.7%** | — | — | — |

**Adding this sleeve costs 5.1 points of half-Kelly growth AND deepens the implied
drawdown from −43.8% to −54.7%.** It is not merely useless, it is dilutive on both axes.

The defensive row is the clearest statement of how dead it is: at half-Kelly the sleeve
would be run at **0.26x** — you would have to *de*-lever it to a quarter of its natural
size — for a growth rate of 0.5%/yr, against a drawdown it has already realised of −85.9%
at 1.0x.

### What a third sleeve would actually have to be

Holding trend and carry at their measured Sharpes on this overlap (0.475 and 0.430) and
solving the equal-risk expression for the third sleeve's Sharpe:

- **at defensive's measured correlations** (+0.066 to trend, +0.478 to carry), the third
  sleeve needs Sharpe **0.883** — essentially the 30%/yr target *by itself*. A sleeve
  correlated 0.48 to one you already own buys almost nothing.
- **if it were uncorrelated to both**, it needs Sharpe **0.621**.
- defensive delivered **0.179**.

That pair of numbers is the useful output of this study for whatever comes next. The
sleeve-stacking route to 30%/yr requires a **third sleeve at Sharpe ≥ 0.62 that is
genuinely uncorrelated to both trend and carry** — and fifteen studies have not produced a
standalone Sharpe above 0.672 gross, ever.

And under the corrected Kelly rule the requirement is harsher still: 30%/yr at half-Kelly
needs Sharpe 0.894 **run at 44.7% volatility**. On the trend+carry blend that is **2.24x**
leverage on a book that already realised −26.7%, i.e. an implied **−59.8%** drawdown before
the third sleeve does anything. The target is a Sharpe the account has to survive, not a
Sharpe it has to print.

---

## 8. Scoring the pre-registered predictions honestly

| | prediction | measured | |
|---|---|---|---|
| P1 | net Sharpe 0.40, 80% CI [0.10, 0.75] | **0.114** | interval HIT, point far too optimistic |
| P2 | mean `rho` < 0.35, short leg < 25% of gross | **0.081 / 7.5%** | **HIT**, more extreme than predicted |
| P3 | S1 within-block 0.30–0.55, mean `rho` > 0.5 | **0.150**; `rho` 0.54/0.38/0.43/0.00 | **MISS** on both |
| P4 | corr to trend ∈ [−0.20, +0.20] | **+0.020** | HIT |
| P4 | corr to carry ∈ [0.00, +0.40] | **+0.478** | **MISS** — too high |
| P5 | `abs(delta corr)` < 0.10 on overlap removal | **0.057 / 0.038** | HIT |
| P6 | DSR bar near 0.53; 25% chance of clearing | bar **0.530**; DSR **0.089** | bar HIT, did not clear |
| P7 | 20% chance vol-matched t > 2 | **t = −2.63** | did not clear, and significantly wrong-signed |
| P8 | DEAD 50% / MARGINAL 35% / LIVE 15% | **DEAD** | |
| — | vol-target Sharpe invariance | **violated, 0.100 spread** | **WRONG**, cause diagnosed in W5 |

Four hits, three misses, one wrong methodological assertion. **Recorded rather than
quietly dropped**, because the value of pre-registration is entirely in scoring it.

---

## 9. Against the four pre-registered deployment criteria

| # | criterion | result |
|---|---|---|
| 1 | net Sharpe ≥ DSR bar (0.530) | **FAIL** — 0.114, DSR 0.089 |
| 2 | vol-matched active t > 2 | **FAIL** — t = −2.63, wrong sign and significant |
| 3 | no negative decade | **FAIL** — 1970s −1.02, 2020s −0.58; loses to passive in 5 of 6 |
| 4 | correlations < 0.3, surviving S3 | **HALF** — trend +0.020 ✓ and genuinely economic ✓; carry **+0.478** ✗ |

**DEAD.**

---

## 10. What this study is worth

The sleeve is worthless. Four things from it are not:

1. **The DSR gate has now been observed failing in both directions.** It passed trend at
   0.612 while the passive benchmark scored 0.669. Here it *rejects* the strategy at 0.089
   while **passing the passive benchmark at 0.994**. The gate is a test of "is this Sharpe
   real", never "is this Sharpe worth having". **It must never be quoted without the
   benchmark's own DSR beside it.**
2. **`volmatched_active = bench_vol x Sharpe_gap`**, exactly (W10). The matched-volatility
   ceremony is the Sharpe comparison. Report the Sharpe gap and this t-stat.
3. **A latent defect in the shared scaler** (W5): the first 12 months of any book built on
   `multiasset_trend`'s machinery run at the full gross cap because no volatility estimate
   exists yet. Worth 0.05 of Sharpe here. Every sleeve on that code path has it.
4. **The third-sleeve requirement is now a number**: Sharpe ≥ 0.62 *and* uncorrelated to
   both, or ≥ 0.88 at defensive's correlations. That is the bar any future diversifier has
   to clear, and it is high enough that the sleeve-stacking route to 30%/yr should
   probably be considered closed until something produces a standalone Sharpe well above
   anything this programme has yet measured.

Fifteen studies, zero deployable. The mechanism was genuinely distinct from trend and
carry — the overlap test proves the low trend correlation was economic, not a construction
artefact, which is more than value could say. **It simply does not pay on this panel.**

No raw panel rows are committed. Every figure here is a derived statistic from the run
recorded in `research/sleeves/_defensive/`.
