# PRE-REGISTRATION — DEFENSIVE / BETTING-AGAINST-BETA on the long-history multi-asset panel

**Written 2026-07-28, BEFORE any backtest of this sleeve was run.** Nothing below is
tuned. The sleeve is run ONCE with exactly these settings, all six declared arms are
reported whatever they say, and the primary arm is designated here in advance.

Sleeve #15 of the programme. Fourteen studies, zero deployable.

---

## 0. Why this sleeve, and what would make it worth having

The portfolio arithmetic is the whole reason:

```
S_combined = s * sqrt( N / (1 + (N-1) * rho) )
```

With trend at gross Sharpe 0.672 (61.5yr) and carry at net 0.430 (22.4yr), the measured
equal-risk pair reaches **0.655**. Half-Kelly compound growth is `g = 3*S^2/8`, so
**30%/yr requires S = 0.894**. The pair is 0.239 short. A third sleeve at s = 0.45 with
average correlation 0.0 to the other two would lift a three-sleeve book to roughly
`0.52 * sqrt(3) = 0.90`. So a *mediocre but genuinely uncorrelated* sleeve is worth more
here than a better correlated one. **The headline deliverable is the correlation, not the
standalone Sharpe** — and it must be an ECONOMIC correlation, not a construction artefact.

That last clause is not boilerplate. The cross-asset VALUE sleeve (#14) reported a
correlation of **-0.164** to trend and it was **fake**: its 5-year reversal window
*contains* trend's 12-month momentum window, so the two signals were partly the same
numbers with opposite signs. Removing 12 months of overlap moved the correlation to
**-0.013**. Any sleeve whose signal is a function of past returns inherits this hazard.
**A beta estimated on a trailing 36-month window of returns is exactly such a function**,
and its window contains all four of trend's lookbacks (1/3/6/12m). So the overlap test is
mandatory here and is pre-registered as arm S3 below.

### The mechanism

Frazzini & Pedersen, *Betting Against Beta*, JFE 111(1), 2014. Investors who want more
return than the market but cannot or will not use leverage bid up high-beta assets
instead. That pushes high-beta prices up and their expected returns down, so the security
market line is flatter than CAPM says. A book that is long low-beta assets and short
high-beta assets, with each leg scaled to unit beta so the book is beta-neutral, harvests
the difference.

This is a **leverage-constraint** story. It is not a momentum story and it is not a yield
story. That distinctness is the entire reason this sleeve was chosen over anything else,
and it is what value failed to be: value's "different mechanism" turned out to be
momentum's own window with a minus sign.

**The naive version is not this sleeve.** Long low-beta and short high-beta WITHOUT the
1/beta leg scaling is simply a short position in the panel proxy. It would earn the equity
risk premium with a minus sign, look terrible, and say nothing about BAB. The beta
neutralisation IS the construction.

---

## 1. Universe (fixed)

Identical to the trend sleeve's `PRIMARY_UNIVERSE` — 18 instruments in 4 blocks — so that
any correlation measured between the two sleeves is a correlation between two *strategies*
and not between two return conventions.

| block | instruments |
|---|---|
| equity (7) | SPX, NASDAQ, FTSE100, N225, DAX, HSI, ASX200 |
| rates (3) | US5Y_TR, US10Y_TR, US30Y_TR |
| commodity (4) | GOLD_F, WTI_F, SILVER_F, COPPER_F |
| fx (4) | USDX, EURUSD, GBPUSD, JPYUSD |

`NATGAS_F` is excluded — the integrity report proves its front-month series is
roll-contaminated (65.7% of its >15% bars fall in the roll window, 2.74x base rate).
`SPY`, `GLD`, `IEF`, `TLT`, `DBC`, `EFA`, `EEM` are excluded as duplicates of instruments
already present.

**Month-end panel only.** The daily panel has a ~1-hour futures/equity session overlap
that is a genuine lookahead at daily frequency (integrity report §6b); it is negligible
monthly.

**Excess returns.** Loaded through `research.sleeves.multiasset_trend.load_excess_panel`,
which subtracts the 13-week bill accrual from the three bond total-return series and
leaves price/futures/spot series alone (they are already excess). Interior nulls are
treated as a zero return with no position held; leading nulls stay NaN so eligibility can
see them.

---

## 2. Signal — the beta estimate

At each month-end `t`, for each eligible instrument `i`:

**Panel proxy.** `p_s` = the equal-weight mean return, in month `s`, across every
instrument that (a) has a non-null return in `s` and (b) has at least 36 months of
observed history as of `s`. Both conditions are computable at time `s` from data at or
before `s`, so the proxy series is point-in-time by construction and is never revised.

**Beta.** OLS slope of `r_i` on `p` over the 36 months ending at `t` inclusive:

```
beta_i(t) = Cov(r_i, p)_[t-35 .. t] / Var(p)_[t-35 .. t]
```

requiring at least 24 non-null pairs. Trailing, causal, no shrinkage in the primary arm.

**No shrinkage.** Frazzini-Pedersen shrink toward 1 (`0.6*beta + 0.4*1`) because their
betas come from noisy single stocks. This panel has 18 liquid aggregates and shrinkage
toward 1 is an equity-market convention that has no meaning for a currency pair. Declining
it is a decision, recorded here, not an oversight.

---

## 3. Construction — the book

**Rank weights (Frazzini-Pedersen).** Rank the `n` eligible instruments by `beta_i(t)`
ascending, `z_i` in `1..n`, `zbar = (n+1)/2`:

```
wL_i  proportional to  max(0, zbar - z_i) / sigma_i
wH_i  proportional to  max(0, z_i - zbar) / sigma_i
```

each normalised to sum to 1. `sigma_i` is the 36-month trailing annualised volatility
(min 24 obs) — this is the pre-registered "inverse-vol size", applied *inside* the rank
weighting so the extremes still carry the most weight but a 60%-vol commodity does not
dominate a 5%-vol bond leg.

**Leg betas.** `betaL = sum_i wL_i * beta_i`, `betaH = sum_i wH_i * beta_i`.

**Beta neutralisation.** The BAB book is `(1/betaL)*L - (1/betaH)*H`. Since the whole book
is subsequently scaled to a volatility target, the leading `1/betaL` is absorbed by the
scaler and the only load-bearing quantity is the hedge ratio:

```
rho(t) = clip( betaL / betaH , 0, 3 )
u(t)   = wL - rho(t) * wH
```

This is algebraically the FP book whenever `betaL > 0`, and it is stated in the form that
makes the degenerate cases visible.

**Guards (declared now, and their binding frequency is a reported result):**
- book OFF unless `n_eligible >= 6`
- book OFF unless `betaH - betaL >= 0.10` (no beta spread, no bet)
- book OFF unless `betaH > 0.05` (a "high-beta" leg with no beta is not a leg)
- `rho` clipped at 0 below and 3 above

**The clip at 0 is a known compromise and is reported, not hidden.** If `betaL < 0` the
exactly-beta-neutral solution requires going LONG the high-beta leg, which is no longer
betting against beta. We refuse that and accept a book with residual negative beta
instead. **The REALISED full-sample and per-decade beta of the book to the proxy is
therefore a mandatory reported number**, and if it is materially non-zero the sleeve does
not get to call itself beta-neutral.

**Sizing and execution.**
- positions decided at month-end `t` are held during `t+1` (`shift(1)`, no exceptions)
- book scaler `k(t) = min( target_vol / sigma_book(t) , GROSS_CAP / gross_unit(t) )`,
  where `sigma_book` is the trailing 36-month (min 12) annualised volatility of the
  *unscaled* book return, estimated from returns realised at or before `t`
- `GROSS_CAP = 10x` book equity
- monthly rebalance, no other trading

**Costs.** `cost = 0.5 * c * turnover`, `c` in {2bps, 10bps}, turnover = sum of absolute
weight changes. Both brackets reported for everything.

---

## 4. The arms — all six declared now, all six reported

| arm | what changes | why it is here |
|---|---|---|
| **PRIMARY** | panel-wide beta, as §2-§3 | the brief's construction |
| S1 WITHIN_BLOCK | BAB run separately inside each block against that block's own equal-weight proxy, block books combined at equal risk | see the prediction in §6 |
| S2 HEDGED | panel-wide, plus an explicit short of `-beta_book * proxy` to force ex-ante zero beta | tests whether the `rho >= 0` clip is load-bearing |
| S3 OVERLAP_REMOVED | beta estimated on months `t-47 .. t-12` — the same 36-month width, but with trend's entire 12-month lookback excised | **the mechanical-vs-economic correlation test.** This is the arm the value sleeve wishes it had run first |
| S4 UNSCREENED | identical, on `returns_monthly_unscreened.parquet` | the 8 quarantined 2008 FX closes are a cleaning *decision*; a result that depends on it is not a result |
| S5 PLACEBO | beta ranks replaced by a fixed random permutation per month (seed 20260728) | if a random book scores like the real one, the machinery and not the signal is producing the number |

Each arm is reported at vol targets **10% / 20% / 40%** and cost brackets **2bps / 10bps**.
The primary headline is **PRIMARY, 20% target, 10bps net**.

*Note stated in advance:* a vol-targeted book's Sharpe is invariant to the target because
gross return, turnover and therefore cost all scale linearly with `k`. The 10/20/40 sweep
is consequently a test of **one thing only — whether the gross cap binds** — and will be
reported as such rather than dressed up as three results.

---

## 5. Mandatory tests (the ones that killed prior sleeves)

1. **Matched-volatility benchmark.** Benchmark = equal-weight LONG-ONLY over the same
   eligible set, same convention, same cost model, then LEVERED to the strategy's own
   realised volatility. Report **raw geometric excess** (flatters low-vol books — killed
   PEAD), **raw arithmetic active return** (flatters high-vol books — killed trend's
   headline), **and** the vol-matched active return with a Newey-West t-stat.
   The variance-drag identity `geo_excess = arith_active - (var_s - var_b)/2` is printed
   as a residual so the reader can see the two agree.
2. **The benchmark goes through the DSR gate too.** DSR passed trend at 0.612 while the
   passive benchmark scored 0.669. The gate has no benchmark-relative criterion and is
   therefore not, on its own, evidence of an edge.
3. **Sharpe per decade.** A full-sample pass carried by one decade is a FAILURE and will
   be called one. Carry died after 2019 (2010s 0.86, 2020s 0.078) and that is why it is
   not deployable.
4. **Correlation to trend and to carry**, on the overlapping months, from the sleeves'
   own written return series on disk — then **re-measured under arm S3** with trend's
   12-month window excised from the beta estimate. `|delta|` is the headline number of
   this study.
5. **P&L concentration** — top cell share, top instrument share, top year share. A single
   name-month was once 13% of a study's total P&L.
6. **DSR bar at this sample's own length**, `n_trials = 38`.
7. **Realised book beta** to the proxy, full sample and per decade (see §3).
8. **Placebo** (arm S5) and a **perfect-foresight positive control** in the verification
   script — if the pipeline cannot express a known edge, a negative result from it is
   uninterpretable.

---

## 6. PREDICTIONS — recorded in advance, to be scored honestly

**P1 — Headline Sharpe.** Primary arm, net of 10bps, full sample:
**point prediction 0.40**, 80% interval **[0.10, 0.75]**.
Reasoning: FP report ~0.7-0.8 for US-equity BAB over 1926-2012, but that is a
several-hundred-name cross-section. Eighteen instruments rebalanced monthly is roughly
6-18 effective bets a year, and `IR = IC * sqrt(BR)` punishes that hard. Trend got 0.672
gross on this panel and carry 0.430 net; there is no reason to expect defensive to beat
either.

**P2 — Panel-wide BAB will partly degenerate into "long bonds and FX".** Against an
equal-weight proxy that is dominated by equity and commodity variance, the low-beta half
of this panel is the three bond series and the four currencies. Their leg beta `betaL`
will be near zero, so `rho = betaL/betaH` will be small and the short leg will be small.
**Predicted: mean `rho` below 0.35, and the short leg below 25% of mean gross exposure.**
If that happens the sleeve is not really betting against beta panel-wide — it is holding
levered bonds, over a sample (1974-2020) containing the largest bond bull market in
recorded history. **That would be a construction artefact of the same family as value's,
and it must be reported as one.** This is precisely why FP themselves run BAB *within*
asset class, and why arm S1 exists.

**P3 — Within-block (S1) will be the more honest arm** and is predicted to score
*similar or slightly better* than panel-wide: **0.30 to 0.55**, with a materially larger
`rho` (predicted mean above 0.5) and a genuinely two-sided book.

**P4 — Correlation to trend: predicted in [-0.20, +0.20]**, i.e. approximately
uncorrelated. Correlation to carry: predicted in **[0.00, +0.40]** and positive, because
both sleeves will tend to be long bonds when the curve is steep and bond vol is low.

**P5 — The overlap effect (S3) will be SMALL: predicted `|delta corr| < 0.10`.** A beta is
a second moment; trend's signal is a first moment. Sharing a window is not the same as
sharing a statistic, which is what made value's case so severe. **But the prediction is
worth little and the measurement is worth everything** — value's -0.164 also looked
economically reasonable until it was tested.

**P6 — DSR.** The bar at `n_trials = 38` is 0.489 at 61.5 years, 0.532 at 52 years, 0.608
at 40 years. The primary sample is expected to start in the early-to-mid 1970s (six
eligible instruments plus a 36-month beta window), i.e. roughly 50-55 years, so the bar
will be near **0.53**. At the P1 point prediction of 0.40 the sleeve **fails** the gate.
**Predicted probability the primary arm clears DSR >= 0.95: 25%.**

**P7 — Beating the vol-matched benchmark.** Predicted probability the vol-matched active
return is positive with Newey-West `t > 2.0`: **20%**. Trend could not do it. Carry
managed `t = 1.22`.

**P8 — Verdict distribution stated in advance.** DEAD 50% / MARGINAL 35% / LIVE 15%,
where LIVE means net Sharpe clears the DSR bar AND the vol-matched active t-stat exceeds
2 AND no decade is negative.

---

## 7. What would make this sleeve deployable, stated before seeing any number

All four, together:
1. net-of-10bps Sharpe **at or above the DSR bar at this sample's length** (~0.53);
2. **vol-matched** active return positive with Newey-West `t > 2`;
3. **no decade** with a negative Sharpe, and no decade carrying more than half the P&L;
4. correlation to trend and to carry **both under 0.3 in absolute value**, and that
   correlation **surviving arm S3** — i.e. economic, not mechanical.

Anything less is MARGINAL. A negative Sharpe, a failed placebo, or a correlation that
evaporates under S3 is DEAD.

**A clean DEAD stated honestly is worth more than a hedged MARGINAL.** Fourteen studies in,
the programme's asset is its willingness to say so.

---

## 8. Files

| path | contents |
|---|---|
| `research/sleeves/multiasset_defensive.py` | the sleeve, run once |
| `research/sleeves/multiasset_defensive_verify.py` | adversarial verification |
| `research/sleeves/_defensive/` | result JSON + monthly return series |
| `research/sleeves/multiasset_defensive_result.md` | the result and the verdict |

No raw panel rows are committed anywhere — derived statistics only.
