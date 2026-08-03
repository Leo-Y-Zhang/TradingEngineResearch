# RESULT — THE PORTFOLIO CORRELATION MEASUREMENT (v2)

**This supersedes the v1 write-up committed at `7c589b5`. Every headline number in v1 was
wrong, for two reasons that are defects and not judgement calls, and the corrected answer
points the opposite way.**

> **The short answer.** `rho(low-vol, trend) = −0.211`, `rho(low-vol, carry) = −0.392`,
> `rho(low-vol, defensive) = −0.303`, `rho(low-vol, passive) = +0.571`, on 213 / 144 / 213 /
> 213 overlapping months. Several combinations clear Sharpe 0.894 on a point estimate — 65
> of 234 configurations do. **But every one of them that contains low-vol is a measurement of
> the 1998–2016 stock/bond regime rather than of a strategy**, which is demonstrated below
> with a long-history equity proxy, and the only combination clearing 0.894 on a long sample
> (`trend + passive`, Sharpe **0.9033** on 738 months) fails at the drawdown its own
> half-Kelly leverage requires. **The best compound return reachable at a survivable
> drawdown is ≈17–20%/yr, not 30%.** No combination of these five sleeves is a validated
> strategy, because combining sleeves that individually failed does not validate anything.

Reproduce, in order:

```
.venv/Scripts/python.exe -m research.sleeves._portfolio.extract_lowvol_corrected_monthly
.venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_correlation_v2
.venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_window_control
.venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_longhistory_books
.venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_attack_lowvol_regime
.venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_attack_trend_passive
```

Machine-readable output in `research/sleeves/_portfolio/*.json`. **No new data, no new
signal, no new strategy was built. No live path was touched. No Sharadar rows are committed.**

---

## 0. THE TWO DEFECTS THAT CHANGE THE ANSWER

### Defect 1 — the low-vol series is dated ONE MONTH EARLY

`lowvol_retest.run_band` labels each monthly slot by the **formation** month but fills it
with `forward_return`, which is the close-to-close return of the **following** month. Every
low-vol monthly observation is therefore dated one month early.

This is invisible to any statistic computed *within* the series. Mean, volatility, Sharpe,
drawdown and the vol-matched active are all unchanged by shifting every observation by one
month, which is why the iteration-10 adversarial verification — which reproduced the book
bit-for-bit and re-derived it on an independent code path — did not catch it. **It only
becomes visible when the series is joined to another series by date, and this study is the
first thing in the programme to do that.**

Measured proof, against correctly-dated SPX (`alignment_control` in
`portfolio_correlation_v2.json`). The low-vol book's own benchmark is a long-only US equity
universe, so it must correlate strongly with SPX **in the same month**:

| series probed | shift applied | rho at SPX(t−1) | **rho at SPX(t)** | rho at SPX(t+1) |
|---|---:|---:|---:|---:|
| low-vol benchmark, **as stored** | 0 | +0.0642 | **+0.1891** | — |
| low-vol benchmark, **shifted +1m** | +1 | +0.0642 | **+0.7687** | +0.1891 |
| trend | 0 | −0.0864 | −0.0279 | +0.0388 |
| carry | 0 | −0.0055 | −0.1657 | −0.0733 |
| seasonal | 0 | +0.0632 | **+0.4355** | −0.0214 |
| defensive | 0 | +0.0182 | −0.1367 | −0.1130 |
| passive (monthly) | 0 | +0.0485 | **+0.8269** | +0.0347 |
| passive (daily) | 0 | +0.0461 | **+0.8269** | +0.0284 |

The ordering reverses exactly as a one-month shift predicts. **All five multi-asset series
are correctly dated** — every one with market exposure peaks at lag 0. Trend and defensive
are long/short books with near-zero market beta, so the probe is uninformative for them
rather than failing; they are built by the same panel loader as passive and seasonal, which
both peak at lag 0.

**Both low-vol series (registered and corrected) are shifted +1 month before any
correlation in this document.** The effect on the answer:

| pair | v1 (misaligned) | **v2 (aligned)** |
|---|---:|---:|
| low-vol ~ trend | −0.191 | **−0.211** |
| **low-vol ~ carry** | **−0.018** | **−0.392** |
| low-vol ~ seasonal | +0.143 | +0.241 |
| **low-vol ~ defensive** | −0.087 | **−0.303** |
| **low-vol ~ passive** | +0.317 | **+0.571** |

v1's headline claim — that `rho(low-vol, carry) = −0.018`, i.e. "genuinely uncorrelated" —
was an artefact of the misalignment. The true figure is −0.392.

**One consequence worth recording:** after the repair, low-vol's returns are earned
1998-05 → **2016-01**, one month past the stated DEV cutoff of 2015-12. The signal inputs
respect the cutoff (`extract_lowvol_monthly` asserts it); the final `forward_return` uses
the 2016-01 close. That is a one-month price bleed into the confirmation window, small but
real, and it is disclosed rather than hidden.

### Defect 2 — v1 mixed return conventions and then levered the result

The five multi-asset sleeves are **excess returns over the 13-week bill**
(`multiasset_trend.load_excess_panel`, prereg §2). Low-vol is a **total return** — the
registered convention carries no risk-free deduction, which is why the iteration-10
verification reports "risk-free rate 2%/yr" as an *additional, unregistered* correction.
v1 put both into one covariance matrix and then levered it, which implicitly borrows at 0%.

Everything here is on a common **excess-over-cash** basis, using
`_data/multiasset/cash_monthly.parquet::US_CASH_13W` (mean 4.51%/yr full sample, **2.04%/yr
over 1998-05→2016-01**). The conversion is corroborated by the verification: it reported
that a 2% risk-free rate takes corrected low-vol to Sharpe 0.490, and the measured bill rate
of 2.04% takes it to **0.4869**.

### Also corrected

**v1 used the wrong low-vol book.** It headlined the *registered* net Sharpe 0.8779 and ran
its "corrected" sensitivity at 0.677 — the *builder's* self-correction — when iteration 10's
independent verification had already established **0.614**. This study regenerates the
verified corrected book (§1).

**v1 mislabelled the benchmark.** Per iteration 11, the recorded **0.7065 is a
DAILY-rebalanced** equal-weight book; the **monthly** equivalent is **0.668**. This study
uses the monthly one throughout and labels it.

**One thing v1 got right and it is confirmed here.** v1 corrected iteration 6's recorded
lesson that the equal-Sharpe shortcut `S = s·sqrt(N/(1+(N−1)ρ))` "overstates by 45%".
Evaluated with *measured* mean Sharpe and *measured* mean pairwise correlation the formula is
algebraically identical to the inverse-vol portfolio Sharpe: across all configurations the
largest discrepancy measured here is **3.33e-16**. Iteration 6's inputs were wrong, not the
formula. The practical instruction stands unchanged — **use the measured covariance** — and
this study does.

---

## 1. PROVENANCE — every series, and the one that had to be regenerated twice

All returns are monthly, net of the sleeve's own registered cost model, on an
excess-over-cash basis.

| sleeve | source | column | n | window (earned) | Sharpe as stored | **Sharpe, excess basis** | vol |
|---|---|---|---:|---|---:|---:|---:|
| **low-vol B2, corrected** | `_portfolio/lowvol_b2_corrected_monthly.parquet` † | `net_conservative` | 213 | 1998-05 → 2016-01 | 0.6138 | **0.4869** | 16.17% |
| low-vol B2, registered | `_portfolio/lowvol_b2_net_monthly.parquet` † | `net_conservative` | 213 | 1998-05 → 2016-01 | 0.8779 | 0.7381 | 14.68% |
| trend | `_multiasset_trend/primary_20pct_monthly.csv` | `net_10bps` | 738 | 1965-01 → 2026-06 | 0.6116 | 0.6116 | 22.80% |
| carry | `_carry_output/carry_primary_net_monthly.parquet` | `net` | 269 | 2004-02 → 2026-06 | 0.4301 | 0.4301 | 3.99% |
| seasonal | `_seasonal/seasonal_composite_20pct_monthly.parquet` | `seasonal_net_10bps` | 736 | 1965-03 → 2026-06 | 0.4680 | 0.4680 | 24.56% |
| defensive/BAB | `_defensive/defensive_primary_net_monthly.parquet` | `net` | 629 | 1974-02 → 2026-06 | 0.1136 | 0.1136 | 22.00% |
| **passive, MONTHLY-rebalanced** | `_multiasset_trend/primary_20pct_monthly.csv` | `bench_net_10bps` | 738 | 1965-01 → 2026-06 | **0.6691** | 0.6691 | 8.79% |
| passive, DAILY-rebalanced | `_seasonal/seasonal_composite_20pct_monthly.parquet` | `bench_net_10bps` | 736 | 1965-03 → 2026-06 | **0.7065** | 0.7065 | 24.80% |

**Every sleeve except low-vol persisted a monthly series.** Low-vol persisted band-level
scalars only; `lowvol_retest_run.py` discards the monthly arrays that live on the in-memory
`BandBooks` dataclass. **It was not reconstructed approximately.** Two regenerations exist:

† `extract_lowvol_monthly.py` (inherited from v1) re-executes the *registered* code path and
refuses to persist unless it reproduces seven registered statistics to 1e-9. It does.

† `extract_lowvol_corrected_monthly.py` (new) re-executes the *corrected* path — the same
`_lowvol_verify.attack5_structure.combined_repair` with `repair_delisting=True,
charge_free_exits=True` on the same next-trading-day execution frame as
`_lowvol_verify.attack7_final` — and refuses to persist unless it reproduces the
verification's corrected headline. **All eight targets reproduced:**

```
gross_annual                +0.145072  vs verification +0.1451   OK
net_annual                  +0.098937  vs               +0.0989   OK
net_sharpe                  +0.613848  vs               +0.614    OK
bench_annual                +0.057138  vs               +0.0571   OK
bench_sharpe                +0.230573  vs               +0.231    OK
vol_matched_active_annual   +0.061774  vs               +0.0618   OK
vol_matched_active_tstat    +2.124299  vs               +2.12     OK
dsr                         +0.585616  vs               +0.586    OK
```

**Pipeline validation, before anything new was computed.** Every previously-banked
correlation reproduces from the persisted files, exactly:

| banked | reproduced here |
|---|---|
| rho(trend, carry) = −0.0441 | **−0.0441** |
| rho(seasonal, trend) = +0.027 | **+0.0272** |
| rho(seasonal, carry) = −0.040 | **−0.0395** |
| rho(defensive, trend) = +0.020 | **+0.0198** |
| rho(defensive, carry) = +0.478 | **+0.4775** |
| passive (daily) = 0.7065 | **0.7065** |
| passive (monthly) = 0.668 | **0.6691** |

**The two benchmark variants.** They correlate **+0.9993** and differ in both rebalancing
frequency and vol scaling (8.79% vs 24.80%). Sharpe is scale-invariant, so nothing in the
correlation or Sharpe tables turns on which is used; the **leverage ladder does**, and it
uses the monthly, un-levered 8.79%-vol book, which is the one iteration 11 rebuilt
independently at 0.6678.

---

## 2. THE CORRELATION MATRIX — overlapping months only, with error bars

Every pair on its own maximal overlap. `SE` is the delta-method standard error on the
r-scale, `(1−r²)/sqrt(n−1)`. The Fisher interval is exact under bivariate normality; the
**block-bootstrap interval (12-month circular blocks, 4,000 resamples) is the one that
survives autocorrelation and is the one to quote.**

| pair | **n** | window | **rho** | SE | Fisher 95% | **block-boot 95%** |
|---|---:|---|---:|---:|---|---|
| **low-vol ~ trend** | **213** | 1998-05→2016-01 | **−0.2105** | 0.066 | [−0.335, −0.078] | [−0.414, **+0.112**] |
| **low-vol ~ carry** | **144** | 2004-02→2016-01 | **−0.3923** | 0.071 | [−0.522, −0.244] | [−0.544, −0.191] |
| low-vol ~ seasonal | 213 | 1998-05→2016-01 | +0.2411 | 0.065 | [+0.110, +0.364] | [+0.163, +0.326] |
| low-vol ~ defensive | 213 | 1998-05→2016-01 | −0.3027 | 0.062 | [−0.420, −0.175] | [−0.455, −0.152] |
| **low-vol ~ passive** | 213 | 1998-05→2016-01 | **+0.5710** | 0.046 | [+0.473, +0.655] | [+0.409, +0.687] |
| trend ~ carry | 269 | 2004-02→2026-06 | −0.0441 | 0.061 | [−0.163, +0.076] | [−0.262, +0.176] |
| trend ~ seasonal | 736 | 1965-03→2026-06 | +0.0272 | 0.037 | [−0.045, +0.099] | [−0.096, +0.159] |
| trend ~ defensive | 629 | 1974-02→2026-06 | +0.0198 | 0.040 | [−0.058, +0.098] | [−0.128, +0.163] |
| trend ~ passive | 738 | 1965-01→2026-06 | +0.0051 | 0.037 | [−0.067, +0.077] | [−0.180, +0.213] |
| carry ~ seasonal | 269 | 2004-02→2026-06 | −0.0395 | 0.061 | [−0.158, +0.080] | [−0.147, +0.056] |
| **carry ~ defensive** | 269 | 2004-02→2026-06 | **+0.4775** | 0.047 | [+0.380, +0.565] | [+0.348, +0.585] |
| carry ~ passive | 269 | 2004-02→2026-06 | −0.1508 | 0.060 | [−0.266, −0.032] | [−0.293, −0.024] |
| seasonal ~ defensive | 629 | 1974-02→2026-06 | +0.0838 | 0.040 | [+0.006, +0.161] | [−0.017, +0.199] |
| **seasonal ~ passive** | 736 | 1965-03→2026-06 | **+0.5131** | 0.027 | [+0.458, +0.564] | [+0.456, +0.593] |
| defensive ~ passive | 629 | 1974-02→2026-06 | +0.0087 | 0.040 | [−0.070, +0.087] | [−0.115, +0.134] |

**Read the n column before the rho column.** The two correlations this study exists to
measure rest on **213** and **144** months. At n = 144 the delta-method SE is 0.071 and the
95% interval is ±0.15 wide; at n = 738 it is ±0.07. A correlation on 144 months is a
different kind of object from one on 738.

Note that `rho(low-vol, trend) = −0.211` has a **Fisher interval excluding zero but a
block-bootstrap interval that includes it (+0.112)**. Under autocorrelation it is
**negative-to-zero**, not reliably negative.

Three correlations that are large and matter: **low-vol ~ passive = +0.571** (low-vol is a
long-only equity book and behaves like one — it also correlates **+0.61 with SPX** and
**+0.58** with an equal-weight SPX/NASDAQ/DJIA proxy); **carry ~ defensive = +0.478**
(defensive is levered bonds, carry is largely bond/FX carry — the same trade, as iteration 7
found); **seasonal ~ passive = +0.513** (seasonal is long-the-market with a calendar mask).

### Common-window matrix, all six series, 144 months (2004-02 → 2016-01)

```
                 lowvol   trend   carry  seasonal  defensive  passive
lowvol           1.0000 -0.1650 -0.3923    0.2506    -0.3818   0.6829
trend           -0.1650  1.0000  0.0889    0.1829     0.1948  -0.0830
carry           -0.3923  0.0889  1.0000   -0.0536     0.6103  -0.2203
seasonal         0.2506  0.1829 -0.0536    1.0000    -0.0075   0.4735
defensive       -0.3818  0.1948  0.6103   -0.0075     1.0000  -0.1172
passive          0.6829 -0.0830 -0.2203    0.4735    -0.1172   1.0000
```

Sleeve Sharpes on this window: low-vol 0.476, trend 0.726, carry 0.493, seasonal 0.194,
defensive 0.566, passive 0.600. **Note `rho(trend, carry)` reads +0.089 here against its
banked −0.044** — the multi-asset correlations are not stable across windows (§3c).

---

## 3. THE OVERLAP DIAGNOSTIC — the test that killed the value sleeve

The value sleeve died when its −0.164 became −0.013 after the shared signal window was
removed; defensive's +0.020 → −0.037 held. The brief asks for the same treatment wherever a
low correlation could be a construction artefact rather than economic independence.

**Scope statement first.** The value/defensive test was arm S3: *re-estimating the signal
inputs* on months t−47…t−12 so no lookback window is shared with trend. **That test cannot
be run for low-vol without running a new backtest**, which is a new configuration and raises
`n_trials` for every result in the programme. It was **not run.** What follows is a
different battery — and one arm of it, §3b, is a *stronger* test than S3 for this case,
because it uses 61 years of data where S3 would have used 17.75.

### (a) Lead–lag — the direct test for a window-misalignment artefact

If two sleeves act on the same information at different lags, the contemporaneous
correlation is near zero while a lagged one is large. **This is the test that found
Defect 1.** After the repair, every low-vol pair peaks at lag 0:

| pair | contemporaneous | largest \|rho\| over k = −12…+12 | at k |
|---|---:|---:|---:|
| low-vol ~ trend | −0.2105 | −0.2152 | +1 |
| low-vol ~ carry | −0.3923 | **−0.3923** | **0** |
| low-vol ~ seasonal | +0.2411 | **+0.2411** | **0** |
| low-vol ~ defensive | −0.3027 | **−0.3027** | **0** |
| low-vol ~ passive | +0.5710 | **+0.5710** | **0** |

No hidden lagged linkage remains. **The correlations are correctly aligned. They are still
not economic — see (b).**

### (b) THE PROXY TEST — and it kills the low-vol diversification case

The suspicion is specific and testable. Low-vol is a **long-only US equity book**
(rho +0.571 to passive, +0.61 to SPX). Carry is a bond/FX carry book; defensive/BAB is
levered bonds. Over 1998–2016 equities and bonds were strongly **negatively** correlated —
and that is a *regime*, not a constant:

| decade | rho(SPX excess, US10Y excess) |
|---|---:|
| 1960s | +0.1912 |
| 1970s | +0.3315 |
| 1980s | +0.2698 |
| 1990s | +0.3479 |
| **2000s** | **−0.2455** |
| **2010s** | **−0.4730** |
| 2020s | +0.2824 |
| **full sample (774 mo)** | **+0.1099** |
| **low-vol's window (213 mo)** | **−0.3136** |

The shift is **−0.424**. So: substitute a **long-history equity proxy** for low-vol — it has
61 years where low-vol has 17.75 — and ask whether the negative correlations appear on the
window and vanish off it. Proxy = equal weight of SPX, NASDAQ and DJIA excess returns:

| proxy vs | rho, full sample | **rho ON low-vol's window** | **rho OFF that window** | window − off |
|---|---:|---:|---:|---:|
| trend | −0.0837 | **−0.1890** | **+0.0540** | −0.2430 |
| carry | −0.1669 | **−0.2897** | **−0.0159** | −0.2738 |
| defensive | −0.2227 | **−0.3862** | **+0.0046** | −0.3909 |
| seasonal | +0.4457 | +0.3874 | +0.5214 | −0.1340 |
| passive | +0.8228 | +0.8391 | +0.8003 | +0.0388 |

Side by side with what low-vol actually measured:

| | low-vol measured | equity proxy, SAME window | equity proxy, OFF window |
|---|---:|---:|---:|
| ~ trend | **−0.2105** | −0.1890 | **+0.0540** |
| ~ carry | **−0.3923** | −0.2897 | **−0.0159** |
| ~ defensive | **−0.3027** | −0.3862 | **+0.0046** |

**A plain equity index reproduces low-vol's entire diversification benefit on that window,
and loses all of it off that window.** The result is identical for SPX, NASDAQ and DJIA
individually. **Low-vol's low correlation to trend, carry and defensive is the 1998–2016
stock/bond regime, not a property of the low-vol signal.** This is the same failure mode
that killed value: a correlation that measures the window rather than the economics.

### (c) Window sensitivity — and it cuts against every low-vol book twice

Every multi-asset sleeve, re-measured on low-vol's 213-month window:

| sleeve | n full | Sharpe full | Sharpe on window | **premium** | DD≤50% CAGR full | on window | premium |
|---|---:|---:|---:|---:|---:|---:|---:|
| trend | 738 | +0.6116 | +0.6706 | +0.059 | +16.71% | +20.93% | +4.22% |
| seasonal | 736 | +0.4680 | +0.2506 | **−0.217** | +11.52% | +5.17% | −6.35% |
| **defensive** | 629 | +0.1136 | **+0.5421** | **+0.429** | +5.26% | +13.78% | +8.51% |
| carry | 269 | +0.4301 | +0.4928 | +0.063 | +3.47% | +3.83% | +0.36% |
| **passive (monthly)** | 738 | +0.6691 | **+0.4667** | **−0.202** | +14.02% | +7.78% | −6.24% |
| passive (daily) | 736 | +0.7065 | +0.5267 | −0.180 | +17.27% | +11.06% | −6.21% |

Mean premium **−0.008** — so v1's claim that "roughly 0.15 of Sharpe in every low-vol-window
number is the window" is wrong as a general statement. The truth is worse and more specific:

1. **Defensive gains +0.429 of Sharpe on that window** — a sleeve correctly measured DEAD at
   0.114 over 52 years reads 0.542 there. Every book containing defensive on this window is
   a window artefact twice over: its Sharpe *and* its diversification.
2. **The passive benchmark LOSES 0.202 of Sharpe on that window.** So "the low-vol book
   beats passive on its window" is partly a statement about the benchmark having a bad
   window — the same two bear markets in which low-vol's excess is concentrated. Iteration 10
   already established low-vol's excess is a **bear-market payoff** (dot-com bust alone,
   34 of 213 months, +23.94%/yr at t 3.63; both bears excluded, +4.87%/yr at t 1.45, below
   the gate). The window and the payoff are the same fact seen twice.

Correlations among the multi-asset sleeves also drift by 0.12–0.23 between windows —
`rho(trend, defensive)` +0.020 → +0.247, `rho(defensive, passive)` +0.009 → −0.196,
`rho(trend, carry)` −0.044 → +0.089. **A correlation measured on one 12–18-year window is
not a structural constant.**

### (d) Partial correlation, removing the passive benchmark

| pair | raw | partial (ex-passive) | delta |
|---|---:|---:|---:|
| low-vol ~ trend | −0.2105 | −0.1778 | +0.033 |
| low-vol ~ carry | −0.3923 | −0.3395 | +0.053 |
| low-vol ~ seasonal | +0.2411 | **−0.0607** | −0.302 |
| low-vol ~ defensive | −0.3027 | −0.2336 | +0.069 |
| carry ~ defensive | +0.4775 | +0.4813 | +0.004 |

Removing the market factor does not explain low-vol's negative correlations — but the
proxy test in (b) already showed that the *market factor* is not the mechanism; the
*stock/bond regime* is, and stripping the multi-asset benchmark does not strip that.

### (e) Split-half stability

| pair | first half | second half | spread |
|---|---:|---:|---:|
| low-vol ~ carry | −0.4154 | −0.4034 | 0.012 |
| low-vol ~ seasonal | +0.2519 | +0.2409 | 0.011 |
| low-vol ~ trend | −0.0903 | −0.2776 | **0.187** |
| **low-vol ~ passive** | +0.4153 | +0.6928 | **0.278** |
| **low-vol ~ defensive** | −0.1269 | −0.4323 | **0.305** |

Both halves fall inside the same regime, so stability here is not evidence of structural
independence — and the two largest movers are exactly the two pairs the regime governs.

**Verdict on the diagnostic: FAILED, in the same way value failed.** Low-vol's measured
independence from trend, carry and defensive is a window property that a plain equity index
reproduces and loses. The correlations in §2 are correctly measured; they are not
transportable outside 1998–2016.

---

## 4. PORTFOLIO SHARPE FROM THE MEASURED COVARIANCE

The equal-Sharpe shortcut was **not** used to produce any number here. Weights come from the
measured covariance under four schemes: **equal weight**, **inverse-vol**,
**inverse-variance**, and **true equal-risk-contribution (ERC)** solved from the measured
covariance. 234 configurations (58 sleeve subsets × 4 schemes, subsets with n < 24 dropped).

**Control:** the shortcut `S = s̄·sqrt(N/(1+(N−1)ρ̄))`, evaluated with the *measured* mean
Sharpe and *measured* mean pairwise correlation, matches the inverse-vol portfolio Sharpe to
**3.33e-16** across all configurations. It is algebraically the same object; iteration 6's
45% error was an input error. Every number below is nonetheless from the covariance matrix.

### The books that matter

| combination | **n** | years | window | equal wt | inv-vol | inv-var | **ERC** |
|---|---:|---:|---|---:|---:|---:|---:|
| lowvol+trend+carry+defensive+passive | 144 | 12.0 | 2004-02→2016-01 | +1.2146 | +1.2270 | +0.9901 | **+1.2411** |
| lowvol+trend+carry+defensive | 144 | 12.0 | 2004-02→2016-01 | +1.1572 | +1.1429 | +0.7851 | **+1.2258** |
| lowvol+trend+carry | 144 | 12.0 | 2004-02→2016-01 | +1.0369 | +1.1795 | +0.7713 | **+1.1808** |
| lowvol+trend+defensive | 213 | 17.8 | 1998-05→2016-01 | +1.0436 | +1.0827 | +1.0915 | **+1.0928** |
| **lowvol+trend** | **213** | 17.8 | 1998-05→2016-01 | **+0.9260** | +0.9212 | +0.8781 | +0.9212 |
| **trend+passive** | **738** | **61.5** | **1965-01→2026-06** | +0.8099 | **+0.9033** | — | **+0.9033** |
| trend+seasonal+passive | 736 | 61.3 | 1965-03→2026-06 | — | +0.8654 | — | +0.8805 |
| trend+seasonal+defensive+passive | 629 | 52.4 | 1974-02→2026-06 | — | +0.8605 | +0.8642 | +0.8573 |
| **passive alone (monthly)** | 738 | 61.5 | 1965-01→2026-06 | **+0.6691** | — | — | — |
| trend+carry | 269 | 22.4 | 2004-02→2026-06 | +0.5461 | +0.6546 | +0.5092 | +0.6546 |
| trend alone | 738 | 61.5 | 1965-01→2026-06 | +0.6116 | — | — | — |
| low-vol alone (corrected) | 213 | 17.8 | 1998-05→2016-01 | **+0.4869** | — | — | — |
| seasonal alone | 736 | 61.3 | — | +0.4680 | — | — | — |
| carry alone | 269 | 22.4 | — | +0.4301 | — | — | — |
| defensive alone | 629 | 52.4 | — | +0.1136 | — | — | — |

**65 of 234 configurations clear 0.894.**

### The DSR gate, charging the combination search

`n_trials` stood at **46** after iteration 11. This study examined **58 sleeve subsets**, so
the search-inclusive count is **104**. Both bars are shown; DSR is evaluated at 46.

| book | years | **S** | bar @46 | **bar @104** | DSR @46 | verdict |
|---|---:|---:|---:|---:|---:|---|
| **trend+passive** [inv-vol] | **61.5** | **0.9033** | 0.4988 | **0.5378** | **1.000** | **clears both, by a wide margin** |
| trend+seasonal+passive [ERC] | 61.3 | 0.8805 | 0.4995 | 0.5385 | 1.000 | clears both |
| trend+seasonal+defensive+passive [ERC] | 52.4 | 0.8573 | 0.5409 | 0.5832 | 1.000 | clears both |
| passive alone | 61.5 | 0.6691 | 0.4988 | 0.5378 | 0.997 | clears both |
| trend alone | 61.5 | 0.6116 | 0.4988 | 0.5378 | 0.995 | clears both |
| lowvol+trend+defensive [ERC] | 17.8 | 1.0928 | 0.9422 | 1.0180 | 0.974 | clears both — *but see §3b/§3c* |
| **lowvol+trend** [equal wt] | 17.8 | **0.9260** | **0.9422** | **1.0180** | 0.940 | **FAILS both** |
| lowvol+trend+carry+defensive [eq wt] | 12.0 | 1.1572 | **1.1576** | **1.2526** | 0.906 | **FAILS both** (by 0.0004 on the first) |

**This reverses v1's central claim.** v1 reported that `low-vol + trend` "passes the DSR gate
on both readings" at Sharpe 1.217 against a 0.981 bar and called it "the first book in this
programme to clear DSR while also beating the passive benchmark". On the **corrected** low-vol
series it measures **0.9260 against a 0.9422 bar and fails**, exactly as low-vol itself does
against its own 0.9234 bar at 0.614. The correction removes the result.

The only book that clears the DSR gate comfortably **and** survives §3 is `trend + passive`,
and it does so because 61.5 years drops the bar to 0.4988 — iteration 12's sample-length
lever, not a stronger signal. `lowvol+trend+defensive` clears both bars arithmetically but
is disqualified on substance: defensive reads 0.542 on that window against 0.114 over its
own 52 years (§3c), and its diversification is reproduced and then lost by a plain equity
index (§3b).

**Inverse-variance is the wrong scheme and the table shows why.** Carry runs at 3.99% vol
against trend's 22.80%; inverse-variance puts ~90%+ of the book in carry and drags
`lowvol+trend+carry` from 1.18 to 0.771. Reported because it was asked for.

### Does low-vol actually help? Yes — and iteration 7's bar was computed at the wrong rho

Iteration 7 established that a third sleeve needs Sharpe **0.621 uncorrelated** to help.
Corrected low-vol is 0.614 on the total-return basis and **0.4869 on the excess basis**,
both below it. But that bar assumed **rho = 0**. The measured rho is **−0.211**, and the bar
falls with rho. For two sleeves at equal risk,
`S_port = (S₁+S₂)/sqrt(2(1+ρ))`; to beat trend alone (0.6706 on that window) at ρ = −0.211
the second sleeve needs only **S₂ ≥ 0.169**.

So the measured arithmetic is: **low-vol does raise the two-sleeve Sharpe, 0.671 → 0.926.**
That is real, it comes from the covariance matrix, and it is not what iteration 7's bar
predicted — because the bar was evaluated at ρ = 0.

**And it does not survive §3b.** The −0.211 is the regime, not the sleeve. Off that window
the same equity exposure correlates **+0.054** with trend, at which the two-sleeve Sharpe
becomes `(0.6116+0.4869)/sqrt(2×1.054) = 0.757` — barely above trend alone and far below
0.894.

---

## 5. THE KELLY BLOCK — with an explicit financing charge

Standing rule: full-Kelly leverage `L = S/σ` gives growth `S²/2`; half-Kelly `L = S/(2σ)`
gives growth `3S²/8` and requires portfolio volatility `S/2`.

**Leverage is charged.** Following iteration 11's method, an excess-return book levered `L`
times returns `L·x − max(L−1,0)·spread/12 + cash`, with `spread` = **bill + 150bp** (primary).
Iteration 11 measured that this charge is what makes the leverage-return curve concave; v1
omitted it entirely, which is why v1 reported a reachable +52.7%/yr. Drawdown is reported
three ways: the **linear scaling** the brief asks for, the **measured** drawdown of the
actually-levered compounded path, and the **block-bootstrap 95th percentile**, because the
observed maximum of a 213-month path is one draw.

| book | n | **S** | 1× vol | half-K growth (theory) | req. vol | **leverage** | DD linear | **DD measured** | **DD boot p95** | ruin at | reachable? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **trend+passive** [inv-vol/ERC] | **738** | 0.9033 | 8.99% | 30.6% | 45.2% | **5.02×** | −83.4% | **−78.8%** | **−92.8%** | 6.50× | **NO** |
| lowvol+trend [equal wt] | 213 | 0.9260 | 12.46% | 32.2% | 46.3% | 3.72× | −64.8% | **−56.7%** | **−81.9%** | 10.90× | **NO** |
| lowvol+trend+defensive [ERC] | 213 | 1.0928 | 9.83% | 44.8% | 54.6% | 5.56× | −103.8% | **−74.5%** | **−86.7%** | 9.85× | **NO** |
| lowvol+trend+carry+defensive [eq wt] | 144 | 1.1572 | 8.33% | 50.2% | 57.9% | 6.95× | −72.7% | **−65.4%** | **−87.0%** | 10.80× | **NO** |
| trend+carry [ERC] | 269 | 0.6546 | 4.70% | 16.1% | 32.7% | 6.96× | −44.6% | −58.6% | −81.2% | 23.60× | **NO** |
| passive alone | 738 | 0.6691 | 8.79% | 16.8% | 33.5% | 3.81× | −115.6% | −79.8% | −93.7% | 7.75× | **NO** |
| trend alone | 738 | 0.6116 | 22.80% | 14.0% | 30.6% | 1.34× | −71.0% | −64.4% | −74.7% | 3.30× | **NO** |

The **DD linear** column is the naive `maxDD(1×) × leverage` the brief asks for. It is
reported alongside the recompounded figure because it is unreliable **in both directions**:
for `lowvol+trend+defensive` it reads −103.8%, which is arithmetically impossible; for
`trend+passive` and `lowvol+trend` it overstates (−83.4% vs −78.8%, −64.8% vs −56.7%); for
`trend+carry` it *understates* badly (−44.6% vs −58.6%). **The recompounded and bootstrap
columns are the ones to use.**

**Every book fails the drawdown test at the leverage its own half-Kelly growth requires.**
Not one of them. The theoretical half-Kelly growth rates of 30–50%/yr all sit behind
measured drawdowns of 57–79% on the observed path and 82–93% on a bootstrap resample.

### What is actually reachable, solving leverage against a drawdown cap

Leverage solved so the drawdown stays inside the cap. Two solutions per book: against the
**observed path** (what the sample maximum permits) and against the **bootstrap 95th
percentile** (what a bad-but-not-extreme resampling of the *same months* permits). The
second is the honest one — solving against a sample maximum systematically over-levers.

| book | n | DD≤50%, observed path | **DD≤50%, bootstrap p95** | DD≤35%, bootstrap |
|---|---:|---:|---:|---:|
| **trend+passive** [inv-vol] | **738** | +25.26% @ 3.10× | **+19.86% @ 2.10×** | **+15.94% @ 1.45×** |
| trend+seasonal+passive [ERC] | 736 | +26.50% @ 2.90× | +19.64% @ 1.80× | — |
| lowvol+trend+carry+defensive [eq wt] | 144 | +40.63% @ 5.00× | +29.17% @ 3.30× | — |
| lowvol+trend+defensive [ERC] | 213 | +32.45% @ 3.25× | +26.35% @ 2.50× | — |
| lowvol+trend [equal wt] | 213 | +31.30% @ 3.15× | +21.31% @ 1.85× | — |
| trend alone | 738 | +16.71% @ 0.95× | +15.09% @ 0.80× | — |
| **passive alone** | 738 | +14.02% @ 1.95× | **+12.13% @ 1.40×** | — |
| low-vol alone | 213 | +8.05% @ 0.85× | +6.47% @ 0.60× | — |

**The bootstrap costs 8–11 percentage points of compound return on every book.** That gap is
the price of having solved leverage against a single observed path.

### Reconciliation with iteration 11 — the engine is validated, and it is optimistic

Iteration 11 measured, on the 61.5-year panel: peak compound **15.83%/yr**, and **12.30%/yr
at a ≤50% drawdown cap from plain equal weight, at ~1.9× average leverage.** The same book
through this engine:

| | iteration 11 | measured here | agree? |
|---|---:|---:|---|
| Sharpe of the monthly EW book | 0.6678 | **0.6691** | yes, to 0.001 |
| leverage at the ≤50% DD rung | ~1.9× | **1.95×** | yes |
| compound at that rung | **12.30%** | **14.02%** | **1.7pp rich** |
| peak compound at any leverage | **15.83%** | **19.34%** | **3.5pp rich** |

The Sharpe and the leverage match; the compound return does not, and the direction is known.
Iteration 11 levered to a **volatility target** (time-varying leverage, which is worse for
drawdown) and charged its own rebalancing costs; this engine applies **static** leverage to
an already-costed series. **This engine therefore reads ~12% rich (ratio 0.877) and every
compound return in this document should be read as an upper bound.** Applying that factor to
the best long-history rung: `19.86% × 0.877 ≈ **17.4%/yr**`.

### Financing is still the largest single lever

| book | bill+50bp | **bill+150bp** | bill+300bp (retail) |
|---|---:|---:|---:|
| trend+passive, DD≤50% observed | +27.87% | +25.26% | **+21.27%** |
| trend+passive, peak at any leverage | +40.32% | +33.65% | **+24.96%** |
| passive alone, DD≤50% observed | +15.32% | +14.02% | +12.27% |

Iteration 11's finding stands: moving the borrowing rate from +50bp to +300bp costs 4–15pp
of compound return, which is more than any strategy decision measured in this study.
**At retail margin the answer degrades further, and the borrowing rate is not a research
variable.**

---

## 6. THE BEST COMBINATION, AND WHY IT IS NOT THE ONE WITH THE HIGHEST SHARPE

The highest Sharpe is `lowvol+trend+carry+defensive+passive` [ERC] at **1.2411**. It is
disqualified by §3b and §3c: on 144 months, inside one regime, containing defensive
(measured 0.114 over 52 years, reading 0.566 there), and its diversification is reproduced
by a plain equity index that loses it off the window. **It is a measurement of 2004–2016.**

The best combination that survives is **`trend + passive` (inverse-vol / ERC)**, weights
trend **0.278** / passive **0.722**, **Sharpe 0.9033 over 738 months (61.5 years)**. It was
put through iteration 11's killers:

| killer | result |
|---|---|
| **K1. Bond bull** (exclude 1981-10→2021-12, 255 months left) | book **0.9033 → 0.8245**, a fall of only 0.079, against passive alone **0.6691 → 0.4387** and iteration 11's EW book 0.668 → 0.439. **The combination is far more robust than either leg — but 0.8245 does NOT clear 0.894.** |
| **K2. Decade stability** | positive in all 7 decades: 0.823 / 0.787 / 1.156 / 1.010 / 1.051 / **0.458** / 0.955. **3 of 7 below 0.894**, none negative. |
| **K3. Vol-matched active vs its own benchmark** | book vs passive **+2.11%/yr, NW t +2.34 — it PASSES the standing programme rule.** Trend *alone* vs passive is **−1.31%/yr, t −0.31**, confirming that trend loses to its own universe. **The gain is diversification, not signal.** |
| **K4. Split halves** | book 0.9218 (1965–1995) / 0.8840 (1995–2026); rho(trend, passive) +0.047 / −0.032. Stable. |
| **K5. DSR** | 61.5-year bars 0.4988 (n=46) / 0.5378 (n=104, charging this study's 58-subset search) / 0.5840 (n=304). **0.9033 clears all three; DSR = 1.000.** |

**Its Kelly block, in full:**

- Sharpe **0.9033**, 1× vol **8.99%**, unlevered CAGR **+13.07%** at max DD **−15.7%**
  (of which **4.63%/yr is the average T-bill rate**, not the strategy — iteration 11's point).
- Half-Kelly growth (theory) **30.6%/yr**; volatility it requires **45.2%**; implied
  leverage **5.02×**.
- Max drawdown at that leverage: **−74.0%** linear-scaled, **−78.8%** measured on the
  recompounded path, **−92.8%** at the bootstrap 95th percentile. Ruin (a month ≤ −100%)
  occurs at **6.50×**.
- **Half-Kelly is therefore NOT reachable** — it needs 5.02× against a ruin threshold of
  6.50×, a margin of 1.3×.
- At a ≤50% drawdown cap: **+25.26% @ 3.10×** (observed) / **+19.86% @ 2.10×** (bootstrap).
- At a ≤35% cap: **+15.94% @ 1.45×** (bootstrap).
- After the iteration-11 reconciliation factor of 0.877: **≈17.4%/yr at a 50% drawdown.**

**Two caveats that are not small.** The panel is **18 hindsight-selected survivors**
(iteration 11: "bias UPWARD and the largest present"), and its correlation-effective N is
**5.26**. Both legs of this book live on that panel, so the 0.9033 inherits the full
survivorship bias.

---

## 7. THE DIRECT ANSWER

> ### Does ANY combination reach Sharpe 0.894?
>
> **On a point estimate, yes — 65 of 234 configurations do. On any reading that survives
> scrutiny, no.**
>
> - Every combination containing **low-vol** clears it on 144–213 months inside a single
>   regime, and **§3b shows a plain equity index reproduces its entire diversification
>   benefit on that window and loses it off that window.** Those books measure 1998–2016.
> - The one combination clearing 0.894 on a long sample is **`trend + passive` = 0.9033 on
>   738 months**, which clears its DSR bar at every trial count and beats its own benchmark
>   at matched vol (+2.11%/yr, t +2.34). But its **95% interval is [0.65, 1.16]** — the lower
>   bound is far below 0.894 — and **outside the 1981–2021 bond bull it is 0.8245**, below
>   0.894.
> - **No book in the study clears 0.894 at the lower bound of its own 95% interval.**
>
> ### And the Sharpe was never the binding constraint anyway
>
> **Every book fails the drawdown test at the leverage its own half-Kelly growth requires.**
> `trend+passive` at 0.9033 needs **5.02× leverage** for its 30.6%/yr theoretical growth,
> producing a **−78.8% measured / −92.8% bootstrap** drawdown against a ruin point of 6.50×.
> A 30%/yr growth rate is arithmetically available at Sharpe 0.894 **only if leverage is
> free and drawdown is unbounded.** It is neither.
>
> ### The best achievable compound return at a survivable drawdown
>
> | | at ≤50% DD | at ≤35% DD |
> |---|---:|---:|
> | `trend + passive`, observed path | +25.26% @ 3.10× | +20.15% |
> | `trend + passive`, **bootstrap p95** | **+19.86% @ 2.10×** | **+15.94% @ 1.45×** |
> | `trend + passive`, bootstrap **× iteration-11 factor 0.877** | **≈17.4%** | **≈14.0%** |
> | `trend + passive`, at retail margin (bill+300bp), observed | +21.27% | +18.12% |
> | passive alone (iteration 11's measured rung) | **12.30%** | 10.56% |
>
> **≈17–20%/yr at a 50% drawdown is the honest number.** That is **5–8 percentage points
> above iteration 11's 12.30% from plain equal weight**, and the gain is bought by adding
> the trend sleeve to buy-and-hold at inverse-vol weights — diversification against a
> +0.005 correlation, not signal. It is **not 30%**, and 30% is not reachable by any
> combination measured here.

---

## 8. WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT

**Established by measurement:**

1. **`rho(low-vol, ·)` is measured for the first time**, on correctly-aligned, correctly-
   converted series: trend **−0.211** (n 213), carry **−0.392** (n 144), seasonal **+0.241**,
   defensive **−0.303**, passive **+0.571**. The gap in the programme is closed.
2. **A one-month dating defect exists in `lowvol_retest.run_band`** and is proved against
   correctly-dated SPX. It invalidates every low-vol correlation ever computed, including
   all of v1's. It does **not** affect low-vol's standalone statistics.
3. **Low-vol's diversification is the 1998–2016 stock/bond regime, not the signal.** A
   long-history equity proxy reproduces it on the window (trend −0.189, carry −0.290,
   defensive −0.386) and loses it off the window (+0.054, −0.016, +0.005).
4. **`trend + passive` measures Sharpe 0.9033 over 61.5 years**, clears its DSR bar at
   n_trials 46/104/304, survives the bond-bull exclusion at 0.8245, is positive in all
   seven decades, and beats its own benchmark at matched vol (+2.11%/yr, t +2.34) — which
   trend alone does not (−1.31%/yr, t −0.31).
5. **Every book fails its own half-Kelly drawdown.** The best survivable rung is
   ≈17–20%/yr at a 50% drawdown, against iteration 11's 12.30% from passive alone.
6. **Defensive gains +0.429 of Sharpe on the low-vol window** (0.114 → 0.542) and the
   passive benchmark **loses 0.202** there. Books built on that window are flattered twice.
7. The equal-Sharpe shortcut is not structurally broken (max gap 3.33e-16); iteration 6's
   inputs were. Confirms v1.
8. This engine reads **~12% rich** against iteration 11's fully-costed ladder, and the
   reconciliation is quantified rather than assumed.

**NOT established, and the write-up would be dishonest without saying so:**

1. **This is not a validated strategy, and it is not a route to 30%/yr.** It combines
   sleeves that individually failed their own gates: trend loses to its own universe, carry
   cannot clear DSR on 22 years, seasonal is negative vol-matched in every cell, defensive
   is DEAD at 0.114, and low-vol is MARGINAL and misses its own 0.9234 DSR bar at 0.614.
   **A portfolio of individually unvalidated sleeves is an unvalidated portfolio.** A high
   combined Sharpe from diversification does not retroactively validate its components.
2. **`trend + passive` is 72% buy-and-hold of a hindsight-selected 18-instrument panel**
   whose correlation-effective N is 5.26. The survivorship bias is real, upward, and
   unquantified. Its excess over plain buy-and-hold is +2.11%/yr.
3. **Low-vol's entire contribution is in-sample.** 1998-05 → 2016-01 is its DEV window; the
   confirmation window has never been read and was deliberately not read here.
4. **The samples are too short to distinguish 0.9 from 0.5** anywhere low-vol is involved:
   the 213-month interval is [0.45, 1.40], the 144-month one [0.58, 1.74].
5. **Capacity binds before anything else.** Band B2's deployable capital is **$138,110**,
   and B3/B4/B5 were all measured dead on `gate_tstat_pass`, so it does not extend by moving
   up a band. At the leverage above, any low-vol-containing book caps out in the tens of
   thousands of dollars.
6. **The S3 signal-window re-estimation test was not run for low-vol** (§3). The proxy test
   is a different and, on this question, stronger instrument — but it is not the same test.
7. **Inherited defects not repaired here.** Iteration 7's latent bug — a book's first 12
   months run at the full 10× gross cap because no vol estimate exists — is present in
   trend, seasonal and defensive and was worth 0.050 of Sharpe where measured. It is
   uncorrected in every multi-asset series used above.
8. **Drawdown estimates remain badly-conditioned.** Even the bootstrap resamples the same
   213 or 738 months; the true left tail is fatter than any resampling of an observed
   sample can show.

**The single highest-value action this points to is not another sleeve.** It is what
iteration 12 already concluded: the ceiling is set by **effective breadth**, and this study
adds a measurement in support of it — `trend + passive`, two streams at rho +0.005, buys
+0.234 of Sharpe over passive alone and 5–8pp of compound return, which is more than any
signal improvement measured in 23 studies. **Genuinely independent return streams are the
only lever that has moved the number.** Low-vol looked like one and, measured properly,
is not: it is an equity book whose apparent independence was a regime.

**No live trading, no broker path, no Sharadar rows committed, nothing public.**
