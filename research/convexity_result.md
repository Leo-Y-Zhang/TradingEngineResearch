# RESULT — CONVEXITY AND THE LEVERAGE CEILING: **the ceiling stands, and the quadratic model was already too generous**

**Code** `research/sleeves/_convexity/convexity.py` + `research/sleeves/_convexity/convexity_run.py` ·
**artefact** `research/sleeves/_convexity/result.json` · re-run is **byte-identical**
(payload md5 `bea994cae90ac987a5f8ad44c572ac10` on consecutive runs) ·
suite **1516 passed / 1 skipped** · **trial ledger unchanged at 47** (§10).

**Sample** 738 months, 61.5 years, 1965-01-31 → 2026-06-30, the same 18 instruments.
**No new backtest configuration was searched.** Every return series analysed here was
banked by a prior study and read off disk.

---

# THE VERDICT

> ## **No. Convexity does not move the ceiling. It moves it the other way.**
>
> At **every** leverage, for **every** leg, under **every** financing rate, the ACTUAL
> empirically compounded return is **BELOW** the second-order approximation — never above
> it — and the true optimum sits at **LOWER** leverage, never higher.
>
> | leg (bill+150bp) | 2nd-order peak | **true empirical peak** | **gap (c)−(a)** |
> |---|---:|---:|---:|
> | passive (equal weight) | 20.42 % @ L 5.70 | **19.34 % @ L 5.05** | **−1.07 pp, −0.65×** |
> | trend | 23.40 % @ L 2.40 | **22.74 % @ L 2.20** | **−0.66 pp, −0.20×** |
> | book (trend+passive) | 37.77 % @ L 6.45 | **33.65 % @ L 5.75** | **−4.12 pp, −0.70×** |
>
> **The quadratic model is not conservative. It is optimistic.** The ceiling the programme
> reported was not an artefact holding the answer down; if anything it was flattering.

**The mechanism the brief described is real and it does operate — it is just far too small
and it is outgunned.** At the trend leg's own optimum the third moment is worth
**+0.33 pp** of annual log growth. The fourth moment is worth **−0.94 pp**. **Net −0.61 pp.**
The alternating series does exactly what a fair reading predicts: skew helps, kurtosis takes
back nearly three times as much, and the truth ends up below where the quadratic left it.

**And the skew that was supposed to do the lifting cannot be distinguished from zero.**
Trend's skewness is **+0.262**. Under the brief's `SE ≈ sqrt(6/T) = 0.090` that is a
z of +2.9. That standard error is **wrong for these series** — it is the sampling error of
skewness *under normality*, and trend's excess kurtosis is **4.14**. Measured rather than
assumed, the standard error is **0.277** (§2), the 95 % interval is **[−0.316, +0.783]**,
and the estimate is **not distinguishable from zero**.

---

## 1. Controls — five, all run before any result was read

| control | required | measured | ✅ |
|---|---|---|:--:|
| **C1** iteration 11's τ-ladder, rebuilt here | 12.29549 % / −47.2874 % / 1.8769× | **0.12295487559393847 / −0.4728738560103498 / 1.8769448605144072** | ✅ exact |
| **C1b** the 15.83 % peak, τ swept in 0.01 steps | ≈ 15.83 % | **15.8281 % at τ = 0.39**, DD −87.77 %, mean leverage 4.85× | ✅ |
| **C2** the survivor book's Sharpe | 0.9033 | **0.9033140238851138** | ✅ |
| **C5** iteration 22's **constant-leverage** ladder | DD≤50 % 25.25708 % @ 3.10×; peak 33.65031 % @ 5.75× | **0.2525708355508056 @ 3.10 · 0.3365031299320007 @ 5.75** | ✅ exact |
| **C3** financing identity at L = 1 (borrow must be zero) | 0.0 | **0.0** | ✅ |
| **C4** the expansion on a Gaussian where the answer is known | order-2 ≈ empirical | 5.7907 % vs **5.7791 %**; measured skew 0.0003, excess kurtosis −0.006 | ✅ |

The whole study rests on C1/C1b/C5: **this module reproduces both the vol-targeted ladder
that produced 15.83 % / 12.30 % and the constant-leverage ladder that produced iteration
22's survivor path, to the last digit, before it changes anything.**

---

## 2. THE MOMENTS — measured, with the standard error that actually applies

Monthly excess returns, net of 10 bps round trip. `SE_norm` = the brief's normal-theory
standard error; `SE_block` = 10 000-replication circular block bootstrap, 12-month blocks,
seed 20260728 — **the one to believe**.

| series | mo | mean | vol | Sharpe | **skew** | SE_norm | **SE_block** | 95 % CI | **≠ 0 ?** | ex-kurt |
|---|---:|---:|---:|---:|---:|---:|---:|---|:--:|---:|
| **trend** | 738 | 13.95 % | 22.80 % | 0.612 | **+0.262** | 0.090 | **0.277** | [−0.32, +0.78] | **NO** | 4.14 |
| **passive** | 738 | 5.88 % | 8.79 % | 0.669 | **−0.482** | 0.090 | **0.165** | [−0.81, −0.17] | **YES (negative)** | 1.54 |
| **book (trend+passive)** | 738 | 8.12 % | 8.99 % | 0.903 | **−0.229** | 0.090 | **0.326** | [−0.90, +0.34] | **NO** | 3.34 |
| trend, gross | 738 | 15.32 % | 22.79 % | 0.672 | +0.263 | 0.090 | 0.276 | [−0.32, +0.78] | NO | 4.14 |
| carry | 269 | 1.72 % | 3.99 % | 0.430 | +0.137 | 0.149 | 0.288 | [−0.37, +0.71] | NO | 1.58 |
| defensive | 629 | 2.50 % | 22.00 % | 0.114 | −0.107 | 0.097 | 0.253 | [−0.58, +0.40] | NO | 2.06 |
| seasonal | 736 | 11.50 % | 24.56 % | 0.468 | **+4.053** | 0.090 | **1.336** | [+0.53, +5.51] | **YES** | **41.74** |
| low-vol B2 (corrected) | 213 | 9.89 % | 16.12 % | 0.614 | −0.747 | 0.167 | 0.383 | [−1.28, +0.19] | NO | 2.77 |
| value | 533 | −1.73 % | 20.96 % | −0.082 | +0.483 | 0.106 | 0.283 | [−0.08, +1.04] | NO | 2.64 |

**Stated plainly, as required: of the nine banked series, exactly two have a skew
distinguishable from zero — passive (negatively) and seasonal (positively). Neither of the
two the argument needs (trend, the book) does.**

### 2b. The brief's standard error understates the true one by 1.8× to 7.2×

| | trend | passive | book | trend, no overlay | seasonal |
|---|---:|---:|---:|---:|---:|
| `sqrt(6/T)` (normal theory) | 0.090 | 0.090 | 0.090 | 0.090 | 0.090 |
| **iid bootstrap** (shape only) | **0.299** | 0.186 | 0.316 | 0.671 | 1.336 |
| **block bootstrap** (shape + autocorrelation) | **0.277** | 0.165 | 0.326 | 0.647 | 1.336 |
| **inflation factor** | **3.07×** | 1.83× | 3.62× | **7.19×** | **14.83×** |

The iid and block figures are almost identical, which localises the cause: **it is
non-normality, not autocorrelation.** `sqrt(6/T)` is derived under a Gaussian; these series
carry excess kurtosis of 1.5 to 41.7, and skewness is estimated from exactly the tail
observations that the kurtosis says are unreliable. **Any skew argument built on a
`sqrt(6/T)` t-statistic on financial returns is built on a standard error that does not
apply.** That includes the framing this study was handed, and it includes the +2.9 that
trend's skew would otherwise have scored.

Seasonal is the caution in the round: **skew +4.05 with excess kurtosis +41.7** is not an
option-like payoff, it is one or two enormous months in 736.

---

## 3. THE FUNG-HSIEH CLAIM, TESTED ON OUR OWN DATA: **ABSENT**

Trend's monthly return regressed on the passive leg's return **and its square**
(square demeaned), Newey-West 6 lags. The straddle signature is a significantly positive
coefficient on the squared term.

| dependent series | β on passive² | **t** | R² | β up-leg | β down-leg | up − down | t(up−down) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **trend (net)** | +5.510 | **+0.93** | **0.010** | +0.286 | −0.285 | +0.571 | **+0.90** |
| trend (gross) | +5.432 | +0.92 | 0.009 | +0.278 | −0.277 | +0.555 | +0.88 |
| trend, no vol overlay | +18.504 | +1.07 | 0.022 | +0.638 | −1.042 | +1.680 | +0.97 |
| trend, no inverse-vol sizing | +24.428 | +0.77 | 0.012 | +0.872 | −1.062 | +1.933 | +0.62 |
| book (trend+passive) | +1.533 | +0.93 | 0.507 | +0.801 | +0.643 | +0.159 | +0.90 |
| **passive, vol-targeted** | **−1.582** | **−2.11** | 0.933 | +2.292 | +2.495 | −0.203 | −1.86 |
| *placebo: passive on itself* | +0.000 | +0.02 | 1.000 | +1.000 | +1.000 | 0.000 | −0.02 |

**It is not there.** t = +0.93 on the quadratic term and t = +0.90 on the piecewise
convexity. **The literature result does not survive in our construction, and this document
says so.**

Two honest qualifications, both of which cut against reading anything into it:

- **The sign pattern is right.** β up-leg **+0.286**, β down-leg **−0.285** — a textbook V.
  If the effect existed it would look like this. It is simply not distinguishable from
  noise at 738 months.
- **There is almost nothing to be convex about.** The quadratic model explains **1.0 % of
  trend's variance**. Trend is essentially orthogonal to the passive leg, which is the
  entire reason the two combine well (Sharpe 0.669 + 0.612 → 0.903) — but it also means
  the passive leg is the wrong axis on which to look for trend's payoff shape.

**The one significant coefficient in the table is negative and it belongs to the overlay.**
Passive run through the volatility-targeting overlay is **CONCAVE** in passive:
β = −1.582, **t = −2.11**. That is the first sign of §4's result.

---

## 4. THE LEVERAGE CURVES — (a) 2nd order, (b) 3rd order, (c) the truth

`R_t(L) = cash_t + L·r_t − max(L−1,0)·spread/12`, `r_t` the banked unlevered monthly excess
already net of 10 bps (so trading cost scales with L). All three curves are expressed
identically as `exp(12·g) − 1`, so **the only difference between them is where the Taylor
series was cut.** Nothing else moves.

### A coefficient this study refuses to guess at, and did not need to

The brief gives the cubic term as `γσ³L³/6`. The Taylor series of `log(1+R)` about its mean
gives `+M₃/3`, not `M₃/6`; the `1/6` is the cumulant-generating-function coefficient. The
two differ by a factor of two in how much the skew is allowed to help. **Both are reported
below and neither is mixed with the other** — and it does not matter, because curve (c)
requires no convention at all. Same at fourth order: `−M₄/4` (log expansion) versus
`−M₄/24` (cumulant).

### 4a. Trend, bill + 150 bp — the leg the whole argument rests on

| L | (a) order 2 | (b) order 3 `M₃/3` | order 3 `M₃/6` | order 4 `M₄/4` | **(c) EMPIRICAL** | **(c)−(a)** | max DD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 17.31 % | 17.35 % | 17.33 % | 17.30 % | **17.23 %** | −0.08 | −50.7 % |
| 1.50 | 20.85 % | 20.98 % | 20.92 % | 20.73 % | **20.72 %** | −0.14 | −69.7 % |
| 2.00 | 22.90 % | 23.20 % | 23.05 % | 22.41 % | **22.55 %** | −0.35 | −82.6 % |
| **2.20** | 23.27 % | 23.68 % | 23.47 % | 22.52 % | **22.74 %** ← peak | **−0.54** | −86.4 % |
| **2.40** | **23.40 %** ← peak (a) | 23.92 % | 23.66 % | 22.28 % | 22.56 % | −0.84 | −89.6 % |
| **2.50** | 23.36 % | **23.95 %** ← peak (b) | 23.66 % | 22.03 % | 22.31 % | −1.05 | −90.9 % |
| 3.00 | 22.23 % | 23.24 % | 22.73 % | 19.30 % | **18.94 %** | −3.29 | −95.9 % |
| 3.25 | 21.08 % | 22.35 % | 21.71 % | 17.00 % | **11.78 %** | −9.30 | −99.7 % |

**Read the peak row.** The third-order term *does* move the optimum higher and later —
23.40 % @ 2.40 becomes 23.95 % @ 2.50, exactly the direction the brief predicted. **And it
moves it AWAY from the truth, which is 22.74 % @ 2.20.** The fourth-order term brings it
back to 22.52 %, close to and now just under the measured figure. The series alternates and
converges from above, and every partial sum above second order is still wrong by more than
the skew was worth.

### 4b. Passive and the book, bill + 150 bp

| L | | passive (a) | passive (c) | gap | | book (a) | book (c) | gap | book DD |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1.00 | | 10.65 % | **10.61 %** | −0.05 | | 13.14 % | **13.07 %** | −0.07 | −15.7 % |
| 2.00 | | 14.28 % | **14.18 %** | −0.10 | | 19.43 % | **19.28 %** | −0.15 | −32.1 % |
| 3.00 | | 17.12 % | **16.91 %** | −0.21 | | 25.06 % | **24.76 %** | −0.30 | −48.1 % |
| 4.00 | | 19.11 % | **18.69 %** | −0.42 | | 29.90 % | **29.30 %** | −0.60 | −63.5 % |
| 5.00 | | 20.20 % | **19.34 %** | −0.86 | | 33.84 % | **32.57 %** | −1.27 | −78.5 % |
| 5.75 | | 20.42 % | 18.98 % | −1.44 | | 36.15 % | **33.65 %** ← peak (c) | −2.50 | −89.4 % |
| 6.45 | | — | — | — | | **37.77 %** ← peak (a) | 29.48 % | **−8.30** | −99.4 % |
| 6.50 | | 20.11 % | 17.67 % | −2.43 | | 37.87 %, still rising | **RUIN** | — | −100 % |

**The book's second-order curve is still climbing at L = 6.50, where it reads 37.87 %. The
real book is WIPED OUT there** — one monthly return of −100 % or worse. That is the single
most vivid demonstration in this document of what the quadratic model does not see. Its
last non-ruined maximum, 37.77 % at L = 6.45, already sits **8.30 pp** above the truth.

**The gap is negative at every non-ruined rung of every curve computed — 1 045 rows across
three legs and three financing rates, zero exceptions.** It is small at low leverage
(−0.05 pp at L = 1) and grows steeply with L, which is precisely the third and fourth
moments becoming material, both on net pulling **down**.

---

## 5. WHERE THE SKEW COMES FROM — and the deflating finding runs the OTHER way

Two sizing layers were switched on and off. **L1** = per-instrument inverse-volatility
sizing. **L2** = the book-level volatility-targeting overlay (`k_t = τ/σ_book,t`, 36-month
causal window). **Skewness is scale-invariant**, so removing L2 needs no replacement
constant and introduces no look-ahead: the un-overlaid book is just the raw
position-weighted series. Every comparison is window-matched (726 months) so the overlay's
12-month warm-up cannot masquerade as a layer effect.

| comparison | n | skew before | skew after | **Δ** | ex-kurt before → after | Sharpe before → after |
|---|---:|---:|---:|---:|---|---|
| **vol-target overlay ON TREND** | 726 | +0.683 | **+0.257** | **−0.426** | 9.16 → 4.14 | 0.577 → 0.666 |
| **vol-target overlay ON PASSIVE** | 726 | −0.486 | **−0.680** | **−0.193** | 1.49 → 2.35 | 0.674 → 0.593 |
| overlay on trend (no L1) | 726 | +0.233 | −0.151 | −0.384 | 8.18 → 5.20 | 0.473 → 0.568 |
| L1 inverse-vol sizing, no overlay | 738 | +0.241 | +0.687 | **+0.446** | 8.35 → 9.15 | 0.470 → 0.592 |
| L1 inverse-vol sizing, with overlay | 726 | −0.151 | +0.257 | **+0.408** | 5.20 → 4.14 | 0.568 → 0.666 |
| *the signal itself vs passive* | 738 | −0.482 (passive) | +0.687 (trend, raw) | +1.169 | 1.54 → 9.15 | 0.669 → 0.592 |

> ### **The overlay is not the source of the convexity. It is the destroyer of it.**
>
> Volatility targeting takes **−0.426** off trend's skewness and **−0.193** off passive's.
> It pushes skew DOWN on both legs. The hypothesis the brief flagged as "the most likely
> deflating finding" — that the convexity is the overlay, is available to anyone, and is
> therefore not a reason to prefer trend — **is refuted, and refuted in the direction that
> makes the overlay look bad rather than the signal look good.**

**Why.** Trend's largest gains arrive inside extended high-volatility episodes. The overlay
estimates volatility from a trailing 36-month window, so by the second and third month of a
crisis it has already cut the notional — it truncates the right tail it was supposed to be
riding. The same measurement in the Fung-Hsieh frame (§3) shows it as the one significant
*concavity* in the table (t = −2.11 on passive).

**What IS the source is the signal plus the per-instrument inverse-vol sizing.** Raw trend
carries skew **+0.687** against passive's **−0.482** — a spread of **+1.169**, the largest
single effect measured in this study — and L1 sizing contributes **+0.45** of it. That is
a genuine structural difference between the two legs and it is the part of the Fung-Hsieh
thesis that does survive here.

**And it is still worth nothing at the ceiling**, for three independent reasons, each
sufficient on its own:

1. **+0.687 is not distinguishable from zero either.** Its block-bootstrap SE is **0.647**
   (a 7.2× inflation over normal theory) — z ≈ 1.06.
2. **It is not stable across decades.** Raw trend's skew is **+1.14** in the 1960s,
   **−1.17** in the 1980s, **+1.43** in the 2000s, **−0.35** in the 2010s (§7). The
   full-sample figure is two decades carrying five, which is the same shape as iteration
   22's "ONE DECADE CARRIES IT".
3. **The overlay removes most of it before it reaches the book anyway**, and the overlay is
   what the banked headline runs on.

---

## 6. THE HONEST BOUND — with the fourth moment, as required

### 6a. The peak and the DD ≤ 50 % rung under the TRUE empirical curve

| leg · financing | **peak (true)** | at L | its DD | **DD ≤ 50 % (true)** | at L | DD ≤ 35 % |
|---|---:|---:|---:|---:|---:|---:|
| passive · bill+150bp | **19.34 %** | 5.05× | −90.0 % | **14.02 %** | 1.95× | 11.95 % |
| trend · bill+150bp | **22.74 %** | 2.20× | −86.4 % | **16.71 %** | 0.95× | 13.36 % |
| **book · bill+150bp** | **33.65 %** | 5.75× | −89.4 % | **25.26 %** | 3.10× | 20.15 % |
| passive · retail+300bp | 13.62 % | 3.50× | −78.9 % | **12.27 %** | 1.85× | 11.26 % |
| trend · retail+300bp | 20.74 % | 2.00× | −83.9 % | **16.71 %** | 0.95× | 13.36 % |
| book · retail+300bp | 24.96 % | 5.20× | −82.0 % | **21.27 %** | 3.05× | 18.12 % |

Trend's DD ≤ 50 % rung is identical under both financing rates because its optimal
leverage there is **0.95×** — below 1, so nothing is borrowed and the spread never applies.
A leg that has to be *de*-levered to survive a 50 % drawdown is not a route to 30 %.

**None of these figures is new and none of them is a convexity effect.** The book rows
reproduce iteration 22's `ladder_observed_path` to the last digit (control C5), and
**iteration 22 already corrected its 25.26 % down to ~15.4 %/yr** via a bootstrap-p95
haircut and the ninth defect's return-convention charge (×0.877). Those corrections stand
untouched; this study neither re-litigates nor benefits from them.

### 6b. The term-by-term account at each leg's own optimum, bill + 150 bp

Annual log-growth contributions. `M₃/3, M₄/4` = log expansion; `M₃/6, M₄/24` = the brief's.

| | passive @ 5.05× | trend @ 2.20× | book @ 5.75× |
|---|---:|---:|---:|
| μ (annual, levered) | 28.25 % | 33.51 % | 44.22 % |
| σ (annual, levered) | 44.33 % | 50.18 % | 51.69 % |
| skew γ | −0.484 | **+0.268** | −0.226 |
| excess kurtosis | 1.56 | **4.11** | 3.33 |
| order 1 (μL alone) | 32.64 % | 39.81 % | 55.61 % |
| **− variance term** | **−9.82 pp** | **−12.59 pp** | **−13.36 pp** |
| **= order 2** | **20.23 %** | **23.27 %** | **36.15 %** |
| **+ third moment** `M₃/3` | −0.41 pp | **+0.33 pp** | −0.30 pp |
| **− fourth moment** `M₄/4` | −0.37 pp | **−0.94 pp** | −0.94 pp |
| **NET of 3rd + 4th** | **−0.77 pp** | **−0.61 pp** | **−1.24 pp** |
| = order 4 | 19.31 % | 22.52 % | 34.47 % |
| *(brief's coefficients: 3rd / 4th)* | *−0.20 / −0.06* | *+0.16 / −0.16* | *−0.15 / −0.16* |
| *(brief's coefficients: **net**)* | *−0.26 pp* | ***+0.007 pp*** | *−0.31 pp* |
| **(c) TRUE EMPIRICAL** | **19.34 %** | **22.74 %** | **33.65 %** |
| **truncation error of the quadratic** | **−0.89 pp** | **−0.54 pp** | **−2.50 pp** |

> **The fair answer the brief asked for, both halves of it: the third moment is worth
> +0.33 pp on the only leg where it is positive at all, and the fourth moment costs
> −0.94 pp on the same leg. The net of the two is NEGATIVE on all three legs** under the
> log expansion, at −0.77, −0.61 and −1.24 pp.

Under the brief's own smaller coefficients the nets are −0.26 pp, **+0.007 pp** and
−0.31 pp. **That +0.007 pp is the only non-negative higher-order net anywhere in this
study** — seven thousandths of a percentage point of annual growth on the trend leg, against
a measured truncation error of −0.54 pp at the same leverage. It is reported because a fair
answer includes it; it does not move anything.

### 6c. Against the banked 15.83 % and 12.30 %

**One number in this study does move, and it has nothing to do with skew.** Iteration 11's
15.83 % came from a **volatility-targeted** ladder. Running the **identical equal-weight
book** over the **identical 726-month window** at **constant leverage** instead:

| bill + 150 bp, same book, same window | peak | at | DD ≤ 50 % | at | DD ≤ 35 % |
|---|---:|---|---:|---|---:|
| **vol-targeted** (iteration 11's convention) | **15.83 %** | τ 0.39, mean 4.85× | **12.59 %** | τ 0.16, mean 2.00× | 10.56 % |
| **constant leverage** | **19.60 %** | L 5.05 | **14.17 %** | L 1.95 | 11.87 % |
| **the overlay costs** | **−3.77 pp** | | **−1.58 pp** | | **−1.31 pp** |
| *retail +300 bp: vol-targeted* | 11.13 % | τ 0.21 | 10.83 % | τ 0.15 | 10.11 % |
| *retail +300 bp: constant leverage* | **13.84 %** | L 3.55 | **12.41 %** | L 1.85 | 11.37 % |
| *the overlay costs* | **−2.72 pp** | | **−1.58 pp** | | −1.26 pp |

(The banked 12.30 % is the τ = 0.15 rung; sweeping τ in 0.01 steps rather than the recorded
six finds 12.59 % at τ = 0.16, still inside −50 %. Same ladder, finer grid, no new
configuration.)

**So the ceiling does move — from 15.83 % to 19.60 % — and the cause is that volatility
targeting was costing 3.77 pp, which is the same finding as §5 seen from the other end.**
Constant leverage is also the *simpler* rule: it needs no volatility estimate at all.

**It changes nothing about the target.** 19.60 % comes with a **−90.0 %** drawdown. The
survivable figure moves from 12.30 % to **14.17 %** at −49.4 %. **30 %/yr is still above the
maximum of the curve for the passive book**, and every qualifier iteration 11 attached still
attaches: 4.64 pp of it is the average bill rate; the 18 instruments are hindsight-selected
survivors (bias **upward**, the largest present); the equity legs are price-only; and
iteration 22's ninth defect (the panel treats price returns as excess returns, +0.748 %/yr
and +0.0489 of Sharpe overstated on the one instrument where both conventions exist) is
**not** repaired in any number in this document.

---

## 7. Per-decade

### 7a. Skewness by decade — the instability that settles §5

| decade | mo | trend | trend, **no overlay** | passive | book | passive, **vol-targeted** |
|---|---:|---:|---:|---:|---:|---:|
| 1960s | 60 | **+1.164** | **+1.136** | +0.060 | +0.435 | −0.542 |
| 1970s | 120 | −0.289 | −0.470 | −0.435 | −0.610 | −0.746 |
| 1980s | 120 | −0.205 | **−1.171** | −0.355 | −0.699 | −0.406 |
| 1990s | 120 | +0.345 | +0.196 | −0.619 | −0.186 | −0.531 |
| 2000s | 120 | **+0.908** | **+1.434** | −0.958 | +0.245 | −1.106 |
| 2010s | 120 | −0.054 | −0.350 | −0.150 | −0.265 | −0.218 |
| 2020s | 78 | +0.148 | +0.021 | −0.208 | −0.613 | −0.883 |

**Trend's skew is negative in three of seven decades and the full-sample positive figure is
carried by the 1960s and the 2000s.** Passive's is negative in six of seven — that part of
the thesis (equities are negatively skewed) holds up cleanly and consistently.

### 7b. Compound return by decade at the DD ≤ 50 % leverage, bill + 150 bp

| decade | passive @ 1.95× | its DD | book @ 3.10× | its DD |
|---|---:|---:|---:|---:|
| 1960s (60 mo) | 6.35 % | −16.0 % | 16.75 % | −15.7 % |
| 1970s | 12.17 % | −37.9 % | 20.57 % | −27.5 % |
| 1980s | **25.40 %** | −25.3 % | **45.39 %** | −49.6 % |
| 1990s | 22.19 % | −24.4 % | 36.84 % | −42.3 % |
| 2000s | 7.31 % | −49.4 % | 27.05 % | −40.3 % |
| **2010s** | **6.92 %** | −21.8 % | **5.48 %** | −41.9 % |
| 2020s (78 mo) | 16.10 % | −29.2 % | 24.00 % | −27.8 % |

The 2010s row is the same warning iteration 11 and iteration 22 both raised, unchanged:
**the decade with the lowest bill rate is the worst decade for both books, and for the book
it is 5.48 %.**

---

## 8. What was refuted, including in this study's own framing

| claim | verdict |
|---|---|
| "Positive skew moves the optimum higher and later" | **Mechanically true, empirically worthless.** It moves the *approximation* higher and later, away from the truth, by +0.33 pp against a −0.94 pp fourth-order term. |
| "Trend has an option-like convex payoff (Fung & Hsieh 2001)" | **NOT PRESENT in our construction.** t = +0.93 on passive², t = +0.90 piecewise, R² = 0.010. Sign pattern correct, magnitude indistinguishable from noise. |
| "The ceiling is an artefact of the quadratic truncation" | **REFUTED, with the sign reversed.** The truncation was flattering by 0.54–2.50 pp at each leg's own optimum (and by 8.30 pp at the book's second-order optimum), not suppressive. |
| "The convexity is the vol-targeting overlay, available to anyone" | **REFUTED.** The overlay *removes* skew: −0.426 on trend, −0.193 on passive, and it is the only significant *concavity* in the Fung-Hsieh table (t −2.11). |
| "SE(γ) ≈ sqrt(6/T) ≈ 0.09 at 738 months" | **WRONG BY 1.8×–7.2× on these series.** It assumes normality; measured excess kurtosis is 1.5–41.7. Trend's real SE is 0.277. |
| *(mine, and it holds)* "The banked 15.83 % is the ceiling" | **Moves to 19.60 %** — not from convexity but because vol targeting cost 3.77 pp. Still far below 30 %, still at −90 % drawdown. |

---

## 9. Limits of this study, stated against itself

1. **Leverage is chosen ex post on the full sample**, in both conventions, and so is the
   DD ≤ 50 % constraint. Every figure in §6 is a hindsight-optimal rung. That was equally
   true of the banked 15.83 % / 12.30 % and the comparison is like-for-like, but neither is
   an achievable ex-ante number.
2. **Iteration 22's ninth defect is not repaired here.** The panel's return convention
   overstates the equity block; nothing in this document is adjusted for it, so every level
   is an upper bound. Only the *differences* between curves — which is what this study is
   about — are robust to it, because the convention cancels in a difference.
3. **Survivorship.** 18 hindsight-selected survivors, bias upward, unchanged and unrepaired.
4. **The moment estimates are weak and this document leads with that.** At 738 months a skew
   of ±0.28 is not measurable on trend. The honest statement is not "trend's skew is +0.26",
   it is "trend's skew is somewhere in [−0.32, +0.78] and we cannot rule out zero".
5. **Curve (c) is not an approximation but it is still one sample path.** It is the realised
   compound return of one 61.5-year history, not an expectation.
6. **The single-degree-of-freedom escape not taken.** A time-varying leverage rule
   conditioned on something other than trailing volatility might beat both conventions
   tested here. Testing one would be a new configuration and would cost a trial. It was not
   done.

---

## 10. The trial ledger — **stays at 47**, and the reasoning holds

`research.trial_ledger.cumulative_trials()` returns **47**; this module reads it and never
states it.

**No increment is warranted, for a reason that is checkable rather than convenient.** A
trial is a strategy CONFIGURATION whose selection could have been driven by its own result.
Nothing here is one:

- Every return series analysed was produced by a study that already paid for it — trend and
  its benchmark (iteration 3), carry (3c/3d), defensive (7), seasonal (6), value (3d),
  low-vol B2 (5/8/10), the survivor book (iteration 22).
- **Leverage is a position-sizing choice applied to an already-run series, not a search over
  strategies.** Iteration 11 spent its 2 trials on W1 and W2 and charged nothing for its own
  six-rung leverage ladder — this study sweeps the same axis on a finer grid and inherits
  that treatment. Charging for leverage rungs now would retroactively invalidate iteration
  11's own accounting.
- **The moment estimates and the Fung-Hsieh regression are descriptive statistics of banked
  series, not selections.** No series was chosen because of what its skew turned out to be;
  all nine on disk are reported, including the three that make the thesis look worst.
- The one thing that could be argued as a search is the constant-leverage-versus-vol-target
  comparison. It is a comparison of two *sizing conventions* applied to the same banked
  book, both pre-existing in this repo (iteration 11 used one, iteration 22 the other), and
  it is reported in full in both directions rather than as a best-of. **If a future study
  builds a strategy on constant leverage rather than vol targeting, that study should pay
  the trial, not this one.**

The DSR bar at 61.5 years and n = 47 is **0.4999** (measured, `dsr_sharpe_bar(61.5,
n_trials=cumulative_trials())`). Nothing in this document changes any banked Sharpe, so no
verdict moves.

---

## 11. What this buys the programme

1. **The 30 % question is not reopened.** Convexity was the last untested structural
   objection to the ceiling, and it fails in the direction that makes the ceiling firmer.
   The true curve peaks **lower and earlier** than the quadratic model at every leg, every
   leverage, every financing rate.
2. **A method correction the whole programme should carry: `sqrt(6/T)` is not the standard
   error of skewness on financial returns.** It understates by 1.8× to 7.2× here because it
   assumes normality. Any future higher-moment claim in this repo must use the bootstrap
   SE, and the module to do it is now in the tree.
3. **Volatility targeting costs 3.77 pp of peak compound return and 1.58 pp of the
   survivable rung on this panel, and it destroys skew rather than creating it.** That is a
   genuine, measured, previously unrecorded finding about a technique used in every levered
   number this programme has produced. **It raises the honest passive ceiling from 15.83 %
   to 19.60 % and the survivable figure from 12.30 % to 14.17 %** — and both are still an
   arithmetic universe away from 30 %.
4. **Do not fund a strategy on trend's positive skew.** It is +0.262 with a 95 % interval of
   [−0.32, +0.78], it is negative in three of seven decades, the overlay the sleeve actually
   runs removes 62 % of what the raw signal generates, and the Fung-Hsieh straddle signature
   that would justify it is absent at t = +0.93.
5. **The fourth moment is the one that matters, and it is never good news.** Excess kurtosis
   of 4.14 on trend and 3.33 on the book costs −0.94 pp of annual log growth at their own
   optima — three times what the skew was worth. Any future expansion truncated at third
   order in this repo is truncated at exactly the wrong place.

---

**Re-analysis only. No new configuration searched. Ledger unchanged at 47. Five controls
reproduced before any result was read, two of them to the last digit. Byte-identical on
re-run.**

## → **THE CEILING STANDS. Skew is a red herring — and the quadratic model it was supposed to
correct was already too kind by 0.54 to 2.50 percentage points.**
