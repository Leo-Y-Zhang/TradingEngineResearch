# SURVIVOR VERIFICATION — the standalone attack on `trend + passive`

**Every DEAD candidate in this programme got a dedicated adversarial verifier, and several
died only because of it. The one survivor never did.** Its Sharpe 0.9033 came out of the
portfolio study that produced it, not out of an attack on it. This document is the missing
attack. The prior was DEAD; the finding is narrower than that and sharper than "survives".

> ## VERDICT: **SURVIVES WITH CORRECTIONS — but the claim that survives is not the claim
> that was made.**
>
> **DEAD: the headline.** `Sharpe 0.9033` is **not a measurement of excess returns.** Seven
> of the eighteen instruments are equity **PRICE indices** and the panel treats a price
> return as if it were already an excess return. Measured directly, assumption-free, on the
> one instrument where both conventions exist for 401 months: the panel's `SPX` reads
> **+9.631%/yr, Sharpe 0.6503**; the true excess return (`SPY` total return − 13-week bill)
> reads **+8.883%/yr, Sharpe 0.6015**. The panel overstates by **+0.748%/yr and +0.0489 of
> Sharpe on one instrument in the friendliest era of the sample.** Charge the equity block
> properly and the book lands at **0.8206** (best-supported), **0.7565 … 0.8650** across
> every defensible parameterisation. **Every one of them is below 0.894.** The breakeven is
> a charge of **16.3 bps/yr** — the claim had 16 basis points of headroom against a defect
> worth hundreds.
>
> **SURVIVES: the diversification premium, on the full sample.** The trend leg's marginal
> contribution is real and significant, and it survives every correction in this document
> because it is a *difference* between two legs on the same panel, so the convention error
> cancels: mean-variance spanning alpha **+13.87%/yr, NW t +4.63**; vol-matched active over
> passive alone **+2.11%/yr, NW t +2.35** (**+2.16%/yr, t +2.43** *after* the convention
> charge); Sharpe gain +0.234 with a block-bootstrap 95% CI of **[+0.032, +0.441]** and
> **P(gain ≤ 0) = 1.2%**. The book is **not** "passive with extra steps" — see below.
>
> **REFUTED: three of the charges in the brief.** (1) *"72.2% passive by risk weight"* is
> **wrong** — 72.2% is the CAPITAL weight; the measured **risk contribution is 50.00% /
> 50.00%** and the return contribution 47.8% / 52.2%. (2) *"the passive leg holds
> Treasuries and 1981–2021 was a 40-year rate decline"* — rates are **6.9%** of the passive
> leg's P&L (equity 74.2%, commodity 19.9%, FX −0.9%), and the trend leg earns **more** from
> rates OUTSIDE the bond bull (+6.22%/yr) than inside it (+4.59%/yr). (3) *"4 of 6
> validation instruments were delisted in 2023"* — **false for this panel**: there are ten
> validation-role tickers and **all 31 series in the file run to 2026-06-30.**
>
> **THE TWO FINDINGS THAT MATTER MOST FOR DEPLOYMENT IN 2026**
> - **One decade carries the edge.** Leave-one-decade-out: drop the **2000s** and the
>   vol-matched active falls from +2.11%/yr (t 2.35) to **+1.29%/yr (t 1.39) — not
>   significant.** Drop any *other* decade and it stays at +2.00…+2.75%/yr, t 2.11…2.81.
>   Only the 2000s has its own decade t above 2.
> - **Since 2010 it IS just passive.** 198 months: book 0.6685, **trend 0.1950**, passive
>   0.6850, vol-matched active **−0.13%/yr, t −0.09.** The most recent 16.5 years contain
>   none of the claimed edge.
>
> **THE HONEST SURVIVABLE-DRAWDOWN RETURN: ≈15.4%/yr at a 50% drawdown** (2.05× leverage,
> bill+150bp financing, bootstrap-p95 drawdown, × iteration 11's 0.877 reconciliation
> factor), against a corrected passive-alone comparator of **≈9.4%/yr**. Not the recorded
> ≈17.42%. At a 35% cap, **≈12.8%/yr**. **P(true Sharpe < 0.894) = 71.3%** on the corrected
> book; **46.8%** even on the uncorrected one.

Reproduce, in order:

```
.venv/Scripts/python.exe -m research.sleeves._survivor.survivor_verification
.venv/Scripts/python.exe -m research.sleeves._survivor.survivor_verification_supp
```

Machine-readable output in `research/sleeves/_survivor/*.json`. **No new data, no new
signal, no live path, no broker path, no vendor rows committed, nothing public.** Suite:
1,515 passed / 1 skipped; the single failure (`tests/test_trial_ledger.py`, an unregistered
count in `research/_gate_review/`) belongs to the concurrent gate review, not to this work.

---

## 0. THE BASELINE REPRODUCES BIT-FOR-BIT

Before attacking it, it has to be the same object.

| | measured here | recorded |
|---|---:|---:|
| Sharpe, `trend + passive` [inverse-vol] | **0.903314** | 0.9033 |
| n / years | 738 / 61.5 | 738 / 61.5 |
| trend alone / passive alone | 0.6116 / 0.6691 | 0.6116 / 0.6691 |
| vol: book / trend / passive | 8.99% / 22.80% / 8.79% | 8.99% / 22.80% / 8.79% |
| ρ(trend, passive) | +0.00511 | +0.005 |
| capital weights | 0.27818 / 0.72182 | 0.278 / 0.722 |

The trend sleeve also re-runs from source to `9.9e-17` against
`_multiasset_trend/primary_20pct_monthly.csv`, and the passive leg was independently
rebuilt from the raw panel to a max absolute error of **9.97e-17**. Everything below is
measured on that reproduction, not read off the recorded file.

---

## 1. ATTACK 1 — **IS IT JUST PASSIVE?** No, and the premise of the charge is wrong

### 1a. 72.2% is the CAPITAL weight. The RISK weight is 50/50, exactly.

Inverse-vol weighting at ρ ≈ 0 equalises risk contributions by construction, and the
measurement confirms it to fourteen decimal places:

| decomposition | trend | passive |
|---|---:|---:|
| **capital weight** | 0.2782 | **0.7218** |
| **risk contribution** (`w·(Σw)/σ_p`) | **0.500000** | **0.500000** |
| **return contribution** (%/yr) | +3.879% | +4.244% |
| return share | 47.76% | 52.24% |
| **variance share** | 49.75% | 49.75% |
| variance share, interaction term (both columns) | **0.51%** | **0.51%** |

The book's total excess return is **+8.124%/yr on 8.99% vol**. Trend supplies **47.8%** of
it and **half** the risk. **"A passive book wearing a trend overlay" is not what the
covariance matrix says.** The recorded write-up's phrase "72% buy-and-hold" is true of
capital and false of risk, and the brief inherited it as a risk statement. Correcting the
record here goes *in the survivor's favour*, which is why it is stated first.

### 1b. The trend leg's marginal contribution IS significant

Three independent framings, all measured:

| test | statistic | verdict |
|---|---:|---|
| **Mean-variance spanning** — regress trend on passive; a non-zero α means the candidate moves the efficient frontier | α = **+13.87%/yr**, β = +0.013, **NW t = +4.63** | **significant** |
| **Vol-matched active** of the book over passive alone (the programme's standing rule) | **+2.11%/yr, NW t = +2.35** | **significant** |
| **Sharpe difference**, joint 12-month circular block bootstrap, 10,000 draws | +0.2342, 95% CI **[+0.0318, +0.4409]**, **P(≤0) = 1.18%** | **significant** |

Contrast with trend *alone* against passive: **−1.31%/yr, t −0.31.** Both are true and they
are consistent — trend does not *beat* passive, it is *uncorrelated with* passive, and the
second is what pays. The recorded claim on this point is exactly right.

### 1c. The weights are not a hindsight artefact

The published weights come from the **full-sample** volatilities, which no operator in 1965
had. Recomputed causally, using only information dated before the month held:

| weighting | n | Sharpe | vol-matched active | mean passive weight |
|---|---:|---:|---:|---:|
| full-sample inverse-vol (published) | 738 | 0.9033 | +2.11%/yr, t 2.35 | 0.7218 |
| **rolling 60-month inverse-vol, lagged 1m** | 702 | **0.9236** | +2.13%/yr, t 2.30 | 0.7213 |
| **expanding-window inverse-vol, lagged 1m** | 702 | **0.9273** | +2.11%/yr, t 2.39 | 0.7338 |

**Causal weights make it slightly better, not worse.** The weighting is clean.

Fixed-weight sensitivity (capital share to passive): 0.50 → 0.8099 · 0.60 → 0.8625 ·
**0.7218 → 0.9033** · 0.80 → 0.8921 · 0.90 → 0.8113. The published point sits at the top of
that curve, which is what inverse-vol is supposed to do at equal Sharpes and ρ ≈ 0, but it
does mean **±10 points of weight costs 0.01–0.04 of Sharpe.**

**Attack 1 verdict: FAILS to kill. The trend leg is half the risk, half the return, and its
marginal contribution is significant on the full sample.** The kill comes in Attack 2.

---

## 2. ATTACK 2 — **THE DECADE PROBLEM.** This one lands.

### 2a. By decade, for the combination and for each leg

| decade | n | **book** | trend | passive | vol-matched active vs passive | its t |
|---|---:|---:|---:|---:|---:|---:|
| 1960s | 60 | 0.8229 | 0.6531 | 0.2711 | +3.29%/yr | 1.16 |
| 1970s | 120 | 0.7870 | 0.4823 | 0.4920 | +2.20%/yr | 0.89 |
| 1980s | 120 | **1.1560** | 0.8773 | 0.9474 | +2.18%/yr | 1.10 |
| 1990s | 120 | 1.0097 | 0.7717 | 0.9693 | +0.47%/yr | 0.35 |
| **2000s** | 120 | 1.0510 | **0.9080** | 0.3934 | **+5.65%/yr** | **2.43** |
| **2010s** | 120 | **0.4581** | **0.0495** | 0.6110 | **−1.15%/yr** | −0.56 |
| 2020s | 78 | 0.9545 | 0.3854 | 0.7853 | +1.46%/yr | 0.71 |

Three of seven below 0.894, confirmed. **But the decade table understates the problem, and
the leave-one-out table finds it.**

### 2b. Leave-one-decade-out — the result stands on the 2000s

| decade removed | n | book Sharpe | **vol-matched active** | **its t** |
|---|---:|---:|---:|---:|
| 1960s | 678 | 0.9127 | +2.00%/yr | 2.12 |
| 1970s | 618 | 0.9244 | +2.08%/yr | 2.18 |
| 1980s | 618 | 0.8479 | +2.03%/yr | 2.11 |
| 1990s | 618 | 0.8852 | +2.36%/yr | 2.33 |
| **2000s** | **618** | **0.8758** | **+1.29%/yr** | **1.39** |
| 2010s | 618 | 0.9779 | +2.75%/yr | 2.81 |
| 2020s | 660 | 0.8970 | +2.20%/yr | 2.26 |

**Remove the 2000s and the diversification premium stops being significant.** Every other
decade is removable at no cost. The dot-com bust and the GFC — the two events trend
following is famous for — are the sample. That is not a defect in the measurement; it is
the honest statement of what the +2.11%/yr is made of, and it is a statement about how
often the payoff arrives, which matters directly for whether an operator can hold it.

### 2c. Since 2010 the book is indistinguishable from buy-and-hold

| era | n | book | trend | passive | vol-matched active | t |
|---|---:|---:|---:|---:|---:|---:|
| 1965-01 → 2009-12 | 540 | **0.9806** | 0.7544 | 0.6633 | **+2.96%/yr** | **2.75** |
| **2010-01 → 2026-06** | **198** | **0.6685** | **0.1950** | 0.6850 | **−0.13%/yr** | **−0.09** |
| 1996 → 2026 | 366 | 0.8617 | 0.5383 | 0.6559 | +1.87%/yr | 1.48 |
| 2000 → 2026 | 318 | 0.8188 | 0.4835 | 0.5611 | +2.12%/yr | 1.56 |

Rolling 120-month windows: median 0.9315 (passive 0.6800), **42.5% of windows below
0.894**, minimum **0.3915** (ending 2020-03), and the book beats passive in 87.9% of them.

**Attack 2 verdict: LANDS.** A strategy whose entire measured edge is earned before 2010,
whose trend leg has run at Sharpe 0.195 for sixteen and a half years, and whose full-sample
significance evaporates when one decade is removed, is **not deployable in 2026 on this
evidence**, regardless of its 1970s record. This is the same objection the brief raised and
the measurement supports it more strongly than the decade table alone did.

---

## 3. ATTACK 3 — **THE BOND BULL.** The charge is aimed at the wrong leg.

Bond-bull exclusion reproduces exactly: **0.9033 → 0.8245** on 255 months outside
1981-10 → 2021-12 (trend 0.6116 → 0.5628, passive 0.6691 → 0.4387). Outside the bull the
vol-matched active is **+2.82%/yr at t 1.77** — larger, and no longer significant at 5%.

### 3a. P&L attribution, both legs, by asset block

| block | **share of passive P&L** | share of trend P&L | passive %/yr in bull | ex bull | trend %/yr in bull | ex bull |
|---|---:|---:|---:|---:|---:|---:|
| equity | **74.2%** | 43.6% | +4.53% | +4.04% | +7.22% | +5.65% |
| **rates** | **6.9%** | **33.6%** | +1.33% | **−1.35%** | +4.59% | **+6.22%** |
| commodity | 19.9% | 9.8% | +1.42% | +0.69% | +1.65% | +1.24% |
| fx | −0.9% | 13.0% | −0.07% | −0.02% | +2.29% | +1.42% |

**The premise of the charge is refuted.** The passive leg is an equity book: Treasuries are
**6.9%** of its P&L. And where duration *does* matter — 33.6% of the trend leg — the sign
is the opposite of the accusation: **trend earns +6.22%/yr from rates outside the bond bull
and +4.59%/yr inside it.** Forty years of falling yields is not what the trend leg's rates
P&L is.

### 3b. Rates excluded entirely — two constructions, one conclusion

| construction | n | book | trend | passive | vol-matched active | t |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 738 | 0.9033 | 0.6116 | 0.6691 | +2.11%/yr | **2.35** |
| **strip the rates P&L from both legs** (attribution; every other decision unchanged) | 738 | **0.7905** | 0.5026 | 0.6629 | **+1.06%/yr** | **1.29** |
| **re-run on a rates-free universe** (a new configuration; the panel cannot start until 1974-01) | 630 | **0.8564** | 0.5545 | 0.6719 | **+2.20%/yr** | **1.56** |
| the rates-free re-run, outside the bond bull | 147 | 0.9871 | — | — | — | — |

**Rates-free, the marginal edge is not significant on either construction.** That is a real
dependency and it should be recorded — but it is a dependency on a *third of the trend
book*, not on the bond bull. The rates-free book *outside* the bull reads 0.987.

**Attack 3 verdict: the bond-bull charge FAILS; a rates-dependency charge SUCCEEDS.**
0.8245 outside the bull is confirmed and is below 0.894.

---

## 4. ATTACK 4 — **SURVIVORSHIP.** Direction upward, size bounded, one sub-claim false.

### 4a. The delisting sub-claim is false for this panel

Every one of the **31** series in `returns_all_monthly.parquet` — 27 tradable plus the ten
validation-role tickers (`SPY TLT GLD DBC EFA EEM IEF BIL IEI SLV`) — has data through
**2026-06-30**. Zero end before 2026. The brief's "4 of 6 validation instruments were
delisted in 2023" is a fact about the Sharadar equity panel, not about this one, and it
does not transfer.

### 4b. Jackknife — drop one instrument at a time

Book Sharpe ranges **0.8408 … 0.9343** across all 18 single-instrument deletions. The
extremes: drop **N225** → 0.8408 (19.6% of passive P&L, +8.33%/yr own return); drop
**NASDAQ** → 0.8525; drop **FTSE100** → 0.9343.

**The ordering is the survivorship channel in plain sight**: the instruments that cost the
most to remove are the ones with the highest realised returns.

| top passive contributors removed | book Sharpe |
|---|---:|
| — | 0.9033 |
| N225 | 0.8408 |
| N225, NASDAQ | 0.7738 |
| N225, NASDAQ, SPX | **0.7114** |
| + HSI, DAX (five) | **0.6658** |

**Direction: upward. Size: replacing the two best-realised markets with average ones costs
~0.13 of Sharpe; the three best, ~0.19.** That is the honest bound this panel can give.

### 4c. The two instruments the prereg excluded — adding them back HELPS

`NATGAS_F` (excluded as roll-contaminated) and `DJIA` (excluded as redundant with SPX) were
both excluded in the prereg on data-receipt grounds. Adding them back:

| universe | n | book | trend | passive | vol-matched | t |
|---|---:|---:|---:|---:|---:|---:|
| 18 (as run) | 738 | 0.9033 | 0.6116 | 0.6691 | +2.11% | 2.35 |
| + NATGAS_F | 738 | **0.9037** | 0.6127 | 0.6732 | +2.10% | 2.32 |
| + DJIA | 738 | **0.9071** | 0.6131 | 0.6743 | +2.11% | 2.34 |
| + both | 738 | **0.9080** | 0.6138 | 0.6788 | +2.10% | 2.30 |

**The exclusions cost the book Sharpe rather than buying it.** The universe was not
selected on returns. This part of the survivorship charge is refuted; 4b's part stands.

### 4d. What cannot be measured here, stated as such

The panel cannot represent markets that ceased to exist, and no major exchange index went
to zero over 1965–2026, so index-level survivorship over *this* window is small. The
genuine bias is **researcher selection in 2026 of seven developed equity markets** — and
4b bounds what that is worth on the instruments present. It does not bound what a market
absent from Yahoo's history would have done. **Unquantified, upward, and disclosed.**

---

## 5. ATTACK 5 — **THE DATING AND DELISTING DEFECTS.** Clean, and verified rather than
trusted.

### 5a. The alignment probe

| series | ρ at k=−2 | −1 | **0** | +1 | +2 | power | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| passive | −0.0543 | +0.0347 | **+0.8269** | +0.0485 | +0.0107 | yes | **ALIGNED** |
| book | +0.0200 | +0.0518 | **+0.5635** | −0.0268 | +0.0145 | yes | **ALIGNED** |
| trend | +0.0827 | +0.0388 | −0.0279 | −0.0864 | +0.0098 | **no** | UNINFORMATIVE |

The passive leg and the book peak at lag 0 against correctly-dated SPX with real power. The
trend leg is a near-zero-beta long/short book, so the probe has **no power** and its verdict
is reported as *unproven*, not as a pass. **The audit's claim is therefore not sufficient
for the trend leg, and the brief was right to say so.**

### 5b. The position-lag ladder — the test that does not need the audit

Rebuild the entire book at three position lags, changing nothing else:

| position lag | trend | passive | **book** |
|---|---:|---:|---:|
| **0** (contemporaneous — the cheating case, and exactly what the low-vol dating defect looked like) | **4.3660** | 0.6680 | **3.6154** |
| **1** (the shipped code) | **0.6116** | 0.6691 | **0.9033** |
| **2** (one month late) | 0.3992 | 0.6654 | 0.7304 |

**Decisive.** Same-month information is worth Sharpe 4.37 in this book. Had the trend leg
carried the low-vol dating defect — a slot labelled `t` holding month `t+1`'s return — the
shipped number would be ~4.4, not 0.61. It sits *exactly* on the causal rung, and both
neighbours are unmissably far away. Combined with the structural fact that the passive leg
is built from the same `xz` frame and probes ALIGNED with ρ 0.827, **the trend leg's index
is the realisation month. Dating: CLEAN, verified independently of the audit.**

### 5c. Delisting

`research/sleeves/multiasset_trend.py` never imports `research.delisting`; the panel is 18
exchange indices, continuous futures and FX spot from one vendor; none of them delist; and
every series runs to 2026-06-30 (§4a). **The delisting defect has no surface here.**

---

## 6. ATTACK 6 — **THE 12-MONTH LEVERAGE BUG (P3).** Present, pre-fix, and worth +0.002.

| | |
|---|---|
| policy behind the headline | `cap` — **the registered (defective) behaviour** |
| **are the survivor's numbers post-fix?** | **NO. They are pre-fix.** |
| months with no volatility estimate | **12** (1965-01 … 1965-12) |
| gross leverage in those months | **10.0× every month** — the cap alone, with no vol estimate behind it |
| book Sharpe, registered | 0.90331 |
| **book Sharpe, repaired (`NO_ESTIMATE_FLAT`)** | **0.90547** (n 726) |
| **delta** | **+0.00216** |
| vol-matched active, repaired | +2.10%/yr, t 2.32 |

**Stated plainly, as required: the survivor's numbers are NOT post-fix.** Repairing it
*raises* the Sharpe by 0.0022 and costs 12 months of sample. Immaterial, and in the
survivor's favour.

**Bonus check — is the 20% vol target tuned?** Book Sharpe across the four pre-registered
targets: 0.10 → 0.9024 · **0.20 → 0.9033** · 0.40 → 0.8951 · 0.60 → 0.8825. Not tuned.

**Bonus check — P&L concentration.** Top single month **2.27%** of total P&L (1986-03), top
12 months 20.1%, top 5% of months 47.6%, 62.6% of months positive. No single-period
dependence. The programme's 13%-of-P&L failure mode is absent.

---

## 7. ATTACK 7 — **COSTS.** The headline rests on 10bps and has 1.8bps of headroom.

The 0.9033 rests on **10bps round-trip charged as half-spread × turnover, to both legs**.
The trend leg turns over **27.52 units of notional per year** and pays **1.376%/yr**; the
passive leg turns over 0.077 units/yr and pays 0.004%/yr.

| round-trip cost | **book** | trend | passive | vol-matched active | t |
|---|---:|---:|---:|---:|---:|
| 0 bps (gross) | 0.9465 | 0.6723 | 0.6696 | +2.49%/yr | 2.79 |
| **2 bps** (the prereg's optimistic bracket) | **0.9378** | 0.6602 | 0.6695 | +2.41%/yr | 2.70 |
| 5 bps | 0.9249 | 0.6420 | 0.6693 | +2.30%/yr | 2.57 |
| **10 bps** (the headline) | **0.9033** | 0.6116 | 0.6691 | +2.11%/yr | 2.35 |
| 15 bps | 0.8817 | 0.5813 | 0.6689 | +1.91%/yr | 2.14 |
| **20 bps** (realistic for HSI / ASX / copper / silver at retail) | **0.8601** | 0.5509 | 0.6687 | +1.72%/yr | 1.92 |
| 30 bps | 0.8169 | 0.4901 | 0.6682 | +1.34%/yr | 1.49 |
| 50 bps | 0.7306 | 0.3687 | 0.6673 | +0.57%/yr | 0.63 |
| 75 bps | 0.6230 | 0.2175 | 0.6662 | −0.39%/yr | −0.43 |

**Breakeven cost for Sharpe 0.894: 11.81 bps.** The headline is 1.8bps inside it. Cost
alone does not kill the book, but it removes the margin, and the vol-matched t drops below
2 at 20bps.

**One unpriced leg found and priced.** The equal-weight benchmark charges only changes in
the weight *vector*, so drifting back to 1/N every month is free. Measured true rebalancing
turnover is **0.385 units/yr against 0.077 charged** — five times as much — worth
**1.5 bps/yr** at 10bps and moving the book 0.9033 → **0.9021**. Real, unpriced, and
immaterial. Recorded so it is not found again.

---

## 8. ATTACK 8 — **THE CI.** The point estimate is not the story, and it is a coin flip.

12-month circular block bootstrap, 10,000 resamples, seed 20260728; cross-checked against
Lo (2002)'s autocorrelation-adjusted analytic standard error.

| | bootstrap | Lo (2002) |
|---|---:|---:|
| 95% interval | **[0.659, 1.155]** | [0.632, 1.175] (SE 0.1385) |
| **P(true Sharpe < 0.894)** | **46.8%** | **47.3%** |
| **P(true Sharpe < 0.75)** | **10.6%** | **13.4%** |
| P(true Sharpe < passive alone, 0.6691) | 3.0% | — |
| P(true Sharpe < 0.60) | 0.8% | — |

**On the uncorrected book the true Sharpe is below the 30%/yr target with probability
0.47.** The recorded [0.65, 1.16] is reproduced. On the **convention-corrected** book
(§9), the same bootstrap gives **P(< 0.894) = 71.3%** and **P(< 0.75) = 28.4%**.

DSR bars at 61.5 years, using `trial_ledger.cumulative_trials()` = **47**, not a hardcoded
number: 0.4999 at 47 · 0.5382 at 105 (charging the v2 58-subset search) · 0.5526 at 145
(also charging this verification's ~40 configurations) · 0.6304 at 1,000. **Even the
corrected 0.8206 clears every one of them** — DSR was never the binding constraint here,
which is iteration 12's finding restated.

Two caveats on the bars, both measured: book skew **−0.229**, excess kurtosis **+3.37**,
Jarque-Bera **356** — strongly non-Gaussian, and `dsr_sharpe_bar` assumes Gaussian, so the
true bar is **above** every figure quoted. And the bootstrap resamples the same 738 months,
so the real left tail is fatter than anything it can show.

---

## 9. ATTACK 9 — **THE RETURN CONVENTION.** This is what kills the headline.

Nobody in the programme has attacked this, and it is upstream of every number in it.

### 9a. What the panel does

`multiasset_trend.load_excess_panel` subtracts the 13-week bill from **three** series — the
constant-maturity par-bond total returns — and treats **everything else as already being an
excess return**. But:

- **Seven equity instruments are PRICE indices** (`SPX NASDAQ FTSE100 N225 HSI ASX200`
  price-only; **`DAX` is the DAX Performance-Index, a TOTAL-RETURN index**). A price return
  is `total − dividends`; an excess return is `total − risk-free`. Using one as the other
  **overstates by (risk-free − dividend yield)**, and it overstates **most** for DAX, whose
  dividend credit is zero because dividends are already inside the index.
- **Four FX series are spot only** — the interest differential that *is* the return to a
  currency position is absent. Three of the four are long-foreign / short-USD, so the
  omission runs the same way.
- **Four commodity series are front-month continuous futures spliced without
  back-adjustment**, so the roll shows up as a price move.

### 9b. The measurement — assumption-free, on the one instrument where both exist

`SPY` is `auto_adjust=True`, i.e. a genuine total return, from 1993-02 to 2026-06 (401
months). The true excess return is `SPY − 13-week bill`. Side by side with what the panel
actually used:

| | mean %/yr | **Sharpe** |
|---|---:|---:|
| panel's `SPX`, treated as an excess return | **+9.631%** | **0.6503** |
| **true excess (`SPY` − bill)** | **+8.883%** | **0.6015** |
| **panel overstates by** | **+0.748%/yr** | **+0.0489** |

And that is the **friendliest** part of the sample. The overstatement is
`(risk-free − dividend yield)`, and the risk-free half of it is measured here: the 13-week
bill averaged **2.534%/yr over 1993–2026** and **7.126%/yr over 1965–1993** (4.631%/yr full
sample). The dividend half over 1965–1993 is **not measurable from this panel** — `SPY`
begins in 1993 — so §9c brackets it rather than assuming it. What is *not* an assumption is
that the bill was **2.8× higher** in the era this panel cannot check.

Corroborating gaps, all measured (a physical-ETF total return minus the panel's series):

| pair | n | window | gap %/yr |
|---|---:|---|---:|
| GLD − GOLD_F | 260 | 2004-11 → 2026-06 | **−0.576** (≈0.4%/yr of it is GLD's fee) |
| SLV − SILVER_F | 242 | 2006-05 → 2026-06 | **−0.670** (≈0.5%/yr is SLV's fee) |
| DBC − WTI_F | 245 | 2006-02 → 2026-06 | **−8.346** (DBC is optimised-roll and diversified, so this is an indication, not a clean read) |
| IEF − US10Y_TR | 287 | 2002-08 → 2026-06 | +0.467 — **the rates conversion is sound** |
| TLT − US30Y_TR | 287 | 2002-08 → 2026-06 | −0.117 — **sound** |
| IEI − US5Y_TR | 234 | 2007-01 → 2026-06 | +0.478 — **sound** |

**Every convention error runs the same way: the panel overstates.** The rates block, the
one place the panel does the conversion properly, checks out against its ETFs.

### 9c. Charge it, and the headline dies under every parameterisation

Charge `(13-week bill − dividend yield q)` on the equity block, monthly, *before* the
signals are built, so the strategy sees the corrected series:

| scenario | **book** | trend | passive | vol-matched active | t |
|---|---:|---:|---:|---:|---:|
| none (as published) | **0.9033** | 0.6116 | 0.6691 | +2.11%/yr | 2.35 |
| q = 4.0%/yr uniform — *implausibly generous* (the measured modern yield is 1.785%) | **0.8650** | 0.5910 | 0.6259 | +2.14%/yr | 2.41 |
| **q = 3.0%/yr uniform** | **0.8159** | 0.5644 | 0.5728 | +2.15%/yr | 2.41 |
| q = 3.0%, **N225 exempt** (Japanese policy rates ≈ 0 for thirty years — the one instrument whose true charge may be negative, and the largest single passive contributor) | **0.8316** | 0.5711 | 0.5908 | +2.14%/yr | 2.41 |
| **q = 3.0%, N225 exempt, DAX charged the full bill** (it is a total-return index) — **the best-supported correction** | **0.8206** | 0.5660 | 0.5773 | **+2.16%/yr** | **2.43** |
| q = 1.785%/yr uniform — the **measured** SPY−SPX yield applied to all 61 years | **0.7565** | 0.5342 | 0.5083 | +2.18%/yr | 2.40 |

**Every row is below 0.894.** The breakeven is a charge of **16.3 bps/yr** on the equity
block — the claim had sixteen basis points of headroom against a defect measured at 75bps
on one instrument in the mildest era, and worth 272bps under the measured-yield reading.

**And the diversification premium is untouched: +2.14 … +2.18%/yr at t 2.40–2.43 in every
scenario.** That is exactly what should happen — the charge is common to both legs and
cancels in the difference. **The convention defect destroys the LEVEL claim and leaves the
RELATIVE claim standing.** Which is the whole finding of this document.

---

## 10. THE CORRECTED NUMBERS

| quantity | **published** | **corrected** | why |
|---|---:|---:|---|
| Sharpe, 738 months | 0.9033 | **0.8206** | §9c, best-supported convention charge |
| — under the harshest defensible charge | — | 0.7565 | §9c |
| — under an implausibly generous charge | — | 0.8650 | §9c |
| 95% CI | [0.65, 1.16] | **[0.577, 1.072]** | §8 on the corrected book |
| **P(true Sharpe < 0.894)** | (not stated) | **71.3%** (46.8% uncorrected) | §8 |
| **P(true Sharpe < 0.75)** | (not stated) | **28.4%** (10.6% uncorrected) | §8 |
| passive share **by risk** | "72.2%" | **50.00%** | §1a — the 72.2% is capital |
| trend's share of return | (implied small) | **47.8%** | §1a |
| vol-matched active vs passive | +2.11%/yr, t +2.34 | **+2.16%/yr, t +2.43** | §9c — *unchanged by every correction* |
| …with the **2000s** removed | — | **+1.29%/yr, t 1.39** | §2b — **not significant** |
| …**2010-2026 only** | — | **−0.13%/yr, t −0.09** | §2c |
| rates share of the **passive** leg | (implied large) | **6.9%** | §3a |
| ex-bond-bull Sharpe | 0.8245 | 0.8245 (confirmed) | §3 |
| breakeven cost | (not stated) | **11.81 bps** vs a 10bps headline | §7 |
| 12-month leverage bug | (uncorrected) | **pre-fix; repair worth +0.0022** | §6 |
| DSR bar at the ledger's count | 0.4988 @ 46 | 0.4999 @ **47** | §8 — still cleared |

### The honest survivable-drawdown return

Static leverage, bill+150bp financing, drawdown solved against the **bootstrap 95th
percentile** (solving against the observed path systematically over-levers), then multiplied
by iteration 11's measured **0.877** reconciliation factor because this engine reads ~12%
rich against a fully-costed vol-targeted ladder.

| book | leverage | CAGR | **× 0.877** |
|---|---:|---:|---:|
| published, DD ≤ 50% | 2.10× | +19.86% | **+17.42%** |
| **corrected, DD ≤ 50%** | **2.05×** | **+17.58%** | **+15.42%** |
| **corrected, DD ≤ 35%** | **1.45×** | **+14.56%** | **+12.77%** |
| corrected **passive alone**, DD ≤ 50% (the comparator) | 1.35× | +10.76% | **+9.44%** |
| published passive alone, DD ≤ 50% | 1.40× | +12.13% | +10.64% |

> **≈15.4%/yr at a 50% drawdown, ≈12.8%/yr at 35%.** Not 17.42%, and not 30%. The
> incremental gain over convention-corrected buy-and-hold is **≈6 percentage points**, and
> it is bought entirely by the ρ = +0.005 between the two legs. Half-Kelly at the corrected
> Sharpe remains out of reach for the same reason as before: it needs **4.63×** (vol 8.86%
> against a required 41.03%) against a ruin point of **6.25×**, and delivers a **−77.4%**
> measured drawdown on the recompounded path — +26.8%/yr of theoretical growth behind a
> drawdown nobody holds.

---

## 11. WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT

**Established by measurement:**

1. **The panel's excess-return convention is wrong for the seven equity instruments and
   approximate for eight more.** Measured on the one pair where both conventions exist for
   401 months: **+0.748%/yr and +0.0489 of Sharpe of overstatement**, in the mildest era of
   the sample. Every defensible correction puts the book **below 0.894**; the breakeven is
   **16.3 bps/yr**. **`Sharpe 0.9033` should not be quoted again without this correction.**
2. **The diversification premium is real, significant, and robust to every correction in
   this document**: spanning α **+13.87%/yr, t +4.63**; vol-matched active **+2.11%/yr,
   t +2.35** (**+2.16%, t +2.43** corrected); Sharpe-gain bootstrap CI **[+0.032, +0.441]**,
   P(≤0) = **1.2%**.
3. **It is not a passive book with an overlay.** Risk contribution **50.00 / 50.00**, return
   contribution **47.8 / 52.2**. The "72.2% passive" figure is a capital weight and has been
   mis-read as a risk weight, including in the brief for this verification.
4. **The dating is clean, verified without trusting the audit.** The position-lag ladder
   reads 3.6154 / **0.9033** / 0.7304 at lags 0 / 1 / 2. A one-month dating error in either
   direction would be unmissable.
5. **The delisting defect has no surface here** — no import, no delistings, all 31 series
   live to 2026-06-30. The brief's "4 of 6 validation instruments delisted in 2023" is
   **false for this panel**.
6. **The survivor's numbers are pre-fix on the P3 12-month leverage bug** (12 months at 10×
   gross in 1965). Repairing it is worth **+0.0022** of Sharpe.
7. **The full-sample significance rests on one decade.** Remove the 2000s: **+1.29%/yr,
   t 1.39**. Remove any other: +2.00 … +2.75%/yr, t 2.11 … 2.81.
8. **Since 2010 the book adds nothing over passive alone**: 198 months, vol-matched active
   **−0.13%/yr, t −0.09**, trend leg Sharpe **0.195**.
9. **Rates are 6.9% of the passive leg and 33.6% of the trend leg, and trend's rates P&L is
   LARGER outside the bond bull than inside it.** The bond-bull charge is aimed at the wrong
   leg; a rates-dependency charge is the correct one, and it removes significance
   (t 1.29–1.56).
10. **Costs leave 1.8bps of headroom** (breakeven 11.81bps at a 10bps headline), and the
    equal-weight leg has an unpriced rebalancing cost of **1.5 bps/yr** (immaterial: −0.0012
    of Sharpe).
11. **Neither the weights, the vol target, nor the universe exclusions are hindsight.**
    Causal weights give 0.9236–0.9273 (*better*); the four vol targets give 0.8825–0.9033;
    adding back the two prereg-excluded instruments gives 0.9037–0.9080 (*better*).
12. **P&L concentration is clean** (top month 2.27%, top 12 months 20.1%) and the return
    distribution is **not Gaussian** (skew −0.23, excess kurtosis +3.37, JB 356), so every
    DSR bar quoted is a floor.

**NOT established, and this document would be dishonest without saying so:**

1. **The equity dividend correction is measured for ONE instrument over ONE-THIRD of the
   sample.** Pre-1993 dividend yields, and every non-US dividend yield, are assumptions —
   which is why §9c brackets them rather than picking one. The *direction* is not an
   assumption; the *magnitude* is.
2. **The FX interest differential and the commodity roll are not corrected**, only
   signposted. Both run the same way as the equity correction, so **0.8206 is itself an
   upper bound.**
3. **A convention-corrected panel has never been built.** §9c charges a constant against the
   existing panel; it does not rebuild the instruments from total-return sources. That is
   the right next step and it is not done here.
4. **Survivorship is bounded, not quantified.** §4b prices removing the best-realised
   instruments *present*; nothing here prices the markets absent from the vendor's history.
5. **This verification ran ~40 new configurations.** They are sensitivities on an existing
   result rather than a search for a new one, but they are not free, and §8 quotes the DSR
   bar with them charged (0.5526 at 145). The ledger's cumulative count stands at **47** and
   was **not** modified by this work.
6. **Nothing here re-validates the sleeves.** Trend loses to its own universe (−1.31%/yr,
   t −0.31) and passive is buy-and-hold. A significant *diversification* premium between two
   individually unvalidated legs is a statement about their correlation, not about either
   one's edge.
7. **No live path, no broker path, no account action, no vendor rows, nothing public, and
   nothing here is advice.**

**The single instruction this points to.** The programme's ceiling has been measured
repeatedly at 17–20%/yr; this document lowers it to **≈15.4%/yr at a 50% drawdown** and
shows that the last sixteen years of the sample contain none of the edge. **Before any
further sleeve work, the panel's return conventions should be repaired at source** — every
number in twenty-five studies stands on them, and this is the first study to check them.
