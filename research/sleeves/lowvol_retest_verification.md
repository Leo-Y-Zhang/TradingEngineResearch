# ADVERSARIAL VERIFICATION — LOW-VOLATILITY / QUALITY RE-TEST

**Target:** `research/sleeves/lowvol_retest_result.json`, `lowvol_retest.py`,
`lowvol_retest_prereg.md` (prereg committed `0b12f93` before the code).
**Posture:** refute by default. Every number below was measured by re-running code, not
read back out of the result file. Verification code lives in
`research/sleeves/_lowvol_verify/`.

---

## VERDICT

**SURVIVES — as MARGINAL, at materially corrected numbers. It does NOT survive as
anything stronger, and three of the published headline levels are wrong.**

The signal is real. It is not a delisting artefact, not a lookahead artefact, not a
concentration artefact, and the vol-matched statistic is computed correctly and in the
right direction. It reproduces bit-for-bit, it survives a realistic execution lag, it
survives a 3× spread stress, and its point-in-time construction is clean under direct
re-computation from raw bars.

But:

1. **`delisting_drag_annual = 0.0` is a dead code path, not a measurement.** The 62-day
   window is off by one and rejects **every** bankruptcy the book ever held. Repairing it
   moves the published gross, net and benchmark returns by 1.7–2.9 percentage points each.
2. **21.4% of all legs traded are charged zero transaction cost.**
3. **61.1% of traded legs breach the registered 1% participation cap.**
4. **The excess is concentrated in bear markets.** Outside the dot-com bust and
   2008–2011 the vol-matched active t-statistic is **1.45** — below the registered gate.
5. The sleeve **fails its own registered promotion gate (iii)**, exactly as the prereg
   predicted. `MARGINAL` is the correct verdict and `PROMOTE` is not available.

**Independent corroboration.** Defects 1 and 2 were found independently by the building
agent (the internal research log, iteration 8) while this verification was running. Its corrected
figures — net Sharpe 0.677, vol-matched +6.52%/yr, t +2.30 — agree with mine
(0.683 / +6.60% / +2.34 once commission is charged on the repaired legs) to within 0.006 of
Sharpe and 0.08pp of the headline. Two separate implementations reaching the same
correction is the strongest available evidence that the correction itself is right.

**One claim in that log does not survive.** It reports "crisis exclusion makes it
STRONGER: +9.37%/yr, t +3.08 excluding 2008–2011". That is true and it is the wrong
crisis. Excluding **2000–2002 as well**, the vol-matched active falls to **+4.87%/yr at
t = 1.45**, below the registered gate. The dot-com bust alone — 34 of 213 months — carries
a vol-matched active of **+23.94%/yr at t 3.63**. See §5.

**Corrected headline (B2, conservative bound, registered raw-return convention):**

| | published | corrected |
|---|---:|---:|
| gross annual | 16.58% | **14.51%** |
| net annual | 12.85% | **9.89%** |
| net Sharpe | 0.878 | **0.614** |
| benchmark annual | 8.34% | **5.71%** |
| benchmark Sharpe | 0.374 | **0.231** |
| one-way cost | 54.8bp | **~67bp** |
| **vol-matched active** | **+7.37%/yr** | **+6.18%/yr** |
| **its NW t-stat** | **+2.64** | **+2.12** |
| DSR (n=38) | 0.874 | **0.586** |
| registered gate (iii) net Sharpe ≥ 0.9234 | FAIL | FAIL |

Corrections applied: delisting window repaired, previously-free exit legs charged,
execution moved to the next trading day's close. Under an additional (unregistered) 2%
risk-free rate the vol-matched active falls to **+5.48%/yr, t 1.88 — which fails the
registered t-gate.**

---

## 0. REPRODUCTION — passed

| band | max abs difference, fresh run vs published JSON |
|---|---:|
| B2 / B3 / B5 | 7.105e-15 |
| B4 | 8.882e-16 |

Checked across 30 headline fields per band (gross, net under both bounds, benchmark,
vol-matched, costs, turnover, forced-exit share, DSR bar, capital). An **independent
re-implementation** of the same registered rules (`_lowvol_verify/instrumented.py`, written
from the prereg rather than copied) reproduces the gross return series to **0.0** and the
cost series to 1e-17 in all four bands, with identical leg and rebalance counts. Nothing in
this report rests on a harness disagreement.

---

## 1. FORCED EXITS — the premise is REFUTED, but it exposed an unrelated cost hole

**Verified.** The published `forced_exit_share = 0.4649` for B2 is correct: 1,783 exits,
954 discretionary (53.5%), 829 forced (46.5%).

### Where the names actually go

| destination at the exit month | n | share of forced | post-1m | post-12m | vs panel, 12m |
|---|---:|---:|---:|---:|---:|
| B3 $1M–5M — **graduated up** | 328 | 39.6% | +1.07% | +10.00% | −8.63% |
| B1 $50k–200k — **volume dried up** | 438 | 52.8% | +0.89% | +12.95% | −4.99% |
| no band at all (<$50k/day) | 1 | 0.1% | +0.13% | −29.61% | −55.04% |
| still in B2, lost its rank or SF1 coverage | 62 | 7.5% | −3.74% | +10.54% | −14.59% |

**Zero** of the 829 forced exits are names that had vanished from the price panel. Every
one was still trading in the month it left the strategy's universe. Only 7 left because
the price fell below the $2 floor and 10 because trading stopped on >10% of days.

### Is the exit return fictitious?

No. `forward_return` is built on the **full ticker series**
(`panel.groupby("ticker")["closeadj"].shift(-1) / closeadj - 1`), over every month-end row
whether eligible or not. Verified directly: recomputing it over all 1,345,888 full-panel
rows gives **max |recomputed − stored| = 0.000e+00** on 1,331,590 comparable cells. Running
the *same* check inside the filtered universe disagrees on 11,653 cells by up to 1615% —
which is the positive proof that the column is not built on the filtered grid.

Consequence: a name that crashes on its way out of B2 hands that crash to the month
**before** it leaves, and the book eats it. The "systematically dodging the losers"
hypothesis does not hold.

Forced-exit names do go on to lag the panel by −0.50% / −2.02% / −6.84% at 1/3/12 months —
but so do discretionary exits (−3.11% at 12m), the exit trigger is trailing-only
information (63-day median dollar volume and the month-end close), and a one-trading-day
execution lag (§6) barely dents the result.

### What the attack DID find: 777 free exit legs

`run_band` skips the cost block entirely when an exiting ticker is absent from the band's
cross-section:

```python
for ticker in traded:
    if ticker not in priced.index:
        continue        # <-- no spread, no impact, no commission
```

That is **777 of 1,783 exits (43.6%), and 21.4% of all 3,624 legs traded, executed for
free** — and they are the *least* liquid exits, so they are the ones a real book would pay
most to get out of. Charging each a full one-way leg at the name's last observed spread and
impact inputs:

| book | net | Sharpe | vol-matched | t |
|---|---:|---:|---:|---:|
| published | 12.85% | 0.878 | +7.37% | +2.64 |
| **charge a real leg on every free exit** | **11.86%** | **0.810** | **+6.38%** | **+2.28** |
| forced to hold each forced exit +1 month | 11.93% | 0.850 | +6.68% | +2.56 |
| forced to hold each forced exit +3 months | 10.31% | 0.742 | +5.11% | +1.97 |
| +3 months **and** charged on the way out | 9.35% | 0.673 | +4.15% | +1.59 |

The +3-month retention is punitive rather than fair (a name that graduated *up* to B3 is
trivially sellable), but it bounds the damage. The free-exit charge is not optional — it is
a defect.

---

## 2. VARIANCE DRAG — the calculation is CORRECT and it is conservative

Recomputed by hand, without calling the function under test:

- `k = sd_strategy / sd_benchmark = 0.6575477571` (published 0.6575477571).
- vol-matched active `= mean(net − k·bench) × 12 = +7.371166%/yr` (published +7.371166%).
- Identity `σ_s × (Sharpe_s − Sharpe_b) = +7.371166%` — exact. **The statistic is nothing
  but a Sharpe gap in return units.**

**Direction.** `k < 1`, so the benchmark is levered **down** to the strategy's risk. That is
the correct construction for an equal-risk comparison and it is the direction that
**flatters** a low-volatility book: it discards 34.2% of the benchmark's mean. The prereg
declares this trap explicitly and reports the raw excess (+4.52%/yr, t 1.20) beside it.
This is handled honestly.

**Autocorrelation.** ρ₁ = +0.149, ρ₂ = +0.045, ρ₃ = +0.088 — mild.

| lag structure | t (vol-matched) | t (net) | t (raw active) |
|---|---:|---:|---:|
| iid | 3.060 | 3.707 | 1.299 |
| **NW 4 (registered)** | **2.637** | **2.889** | **1.204** |
| NW 5 (Newey–West rule for T=213) | 2.607 | 2.862 | 1.200 |
| NW 6 | 2.598 | 2.875 | 1.202 |
| NW 12 | 2.582 | 3.059 | 1.181 |
| NW 24 | 2.807 | 3.077 | 1.403 |

The Newey–West adjustment **lowers** the t-statistic relative to iid, and the published
NW-4 figure is essentially the minimum across sensible lag choices. No lag-selection
cherry-pick.

**k is estimated in-sample and the t-stat ignores that.** Stationary bootstrap (4,000
draws, k re-estimated inside every resample):

| mean block | vol-matched active | 95% CI | P(≤ +2%) |
|---:|---:|---|---:|
| 1 month | +7.37% | [+2.52%, +11.98%] | 0.016 |
| 3 months | +7.41% | [+2.10%, +12.68%] | 0.023 |
| 6 months | +7.41% | [+1.76%, +13.32%] | 0.033 |
| 12 months | +7.47% | [+2.24%, +12.87%] | 0.019 |

Gate (i) clears with ~97% confidence once k-estimation error is priced.

**Risk-free rate.** Sharpes and the vol-matched active are computed on **raw** returns, so
de-levering the benchmark to 0.658× parks 34.2% of capital at 0%. Over 1998–2015 3-month
T-bills averaged ~2%/yr.

| assumed rf | SR strategy | SR benchmark | vol-matched | t |
|---:|---:|---:|---:|---:|
| 0.0% | 0.878 | 0.374 | +7.37% | 2.64 |
| 2.0% | 0.741 | 0.285 | +6.69% | 2.39 |
| 4.0% | 0.605 | 0.195 | +6.00% | 2.15 |

The gate survives on its own; the DSR miss (§7) gets worse.

---

## 3. DELISTINGS — **REAL BUG, and the largest single error in the result**

`lowvol_retest.exit_return` and the vectorised `terminal_on_exit` both test

```python
at < delisted_on <= at + pd.Timedelta(days=DELISTING_WINDOW_DAYS)
```

Sharadar's ACTIONS row for a delisting is dated **on the name's last traded bar**. The gap
between the exit date and the delisting date is therefore exactly **zero**, and the strict
`<` rejects it.

Forensics on B2's 58 held name-months that sit at a name's last panel row (30 of which are
the 2015-12 sample truncation, leaving 28 real ones):

| ACTIONS type | n | terminal | median gap (days) | inside the 62-day window |
|---|---:|---:|---:|---:|
| acquisitionby | 17 | 0.00 | 0 | 0 |
| acquisitionof | 2 | 0.00 | −1362 | 0 |
| **delisted** | **9** | **−1.00** | **0** | **0** |

All nine bankruptcies had a gap of exactly 0 days and all nine were booked at **0.0%**.
Panel-wide, `terminal_on_exit` is non-zero on **1 of B2's 1,817 last-observation cells**.
`delisting_drag_annual = 0.0` measures nothing.

### Repairing it (`<` → `<=`), applied to the shared `realised_return` column

| band | book | gross | net | Sharpe | bench | bSh | drag | vol-matched | t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | published | 16.58% | 12.85% | 0.878 | 8.34% | 0.374 | +0.00% | +7.37% | +2.64 |
| B2 | **repaired** | **14.89%** | **11.16%** | **0.746** | **5.41%** | **0.242** | **−1.69%** | **+7.54%** | **+2.68** |
| B2 | maximal: every unexplained last obs = −100% | 12.83% | 9.10% | 0.521 | 3.68% | 0.156 | −3.76% | +6.38% | +2.12 |
| B3 | published → repaired | 10.96→9.83% | 8.71→7.58% | 0.680→0.584 | 8.10→5.60% | | 0→−1.13% | +4.12→+4.37% | 1.39→1.52 |
| B4 | published → repaired | 10.18→9.80% | 8.66→8.28% | 0.682→0.655 | 7.95→5.95% | | −0.38→−0.75% | +4.06→+4.86% | 1.44→1.73 |
| B5 | published → repaired | 8.71→8.15% | 7.63→7.06% | 0.712→0.656 | 7.71→6.08% | | 0→−0.56% | +3.76→+4.02% | 1.51→1.60 |

**The bug flatters the benchmark more than the strategy.** Repairing it costs the strategy
1.69%/yr and the benchmark 2.93%/yr, so the raw excess *rises* (4.52% → 5.76%) and the
vol-matched active is essentially unchanged (+7.37% → +7.54%). The published *levels* are
wrong; the published *conclusion* is not.

The two bug classes the prereg was guarding against are genuinely absent:

- **Terminal return applied by date:** yes — the window is applied, it is simply the wrong
  boundary.
- **Names removed after booking the exit:** yes, verified. `holdings.difference_update(closing_out)`
  fires on both paths and no name re-books a terminal return.
- Last-observation rate, B2: universe 0.91% vs holdings 0.93% — the strategy meets
  *slightly more* of them, so the bug was not a strategy-side gift in the deciding band.

---

## 4. CONCENTRATION — clean, verified

6,210 name-months, **974 distinct names ever held**, total attributed gross P&L 9.2351 on a
1.0 starting equity (a book that ends at 15.42×).

| unit | largest share of total P&L | top-3 | top-5 | top-10 |
|---|---:|---:|---:|---:|
| single name-month | 1.68% | 4.68% | 7.36% | 13.55% |
| single **name**, all months | 3.33% (TPL) | 9.02% | 14.22% | 24.41% |
| single **month** | 5.44% (2014-01) | 14.72% | 22.37% | 38.49% |

- Herfindahl over names 0.0215 → **47 effective names**.
- Zero names contribute >5% of total P&L. Worst name −1.88%.
- Gross notional: max weight 3.33%, top-3 10.00% — true, but trivial under equal weighting
  and therefore not evidence of anything.

Note on denominators: the result file's `largest_abs_share_of_gross_pnl = 0.32%` divides by
**Σ|P&L| = 52.77**, whereas the 1.68% above divides by **net total P&L = 9.24**. Both are
correct; the 0.32% figure is the more flattering framing and should not be quoted without
the base. Either way there is no concentration problem.

Nothing here resembles the prior sleeve with 13% of P&L in one name-month or 65% of gross
notional in three instruments.

---

## 5. PER-DECADE — the claim is not supportable, and the edge lives in bear markets

The sample is **1998-04 to 2015-12**. "Per decade" means 21 months of the 1990s, 120 of the
2000s and 72 of the 2010s. A 21-month stub is not a decade. There is **one full market
cycle and roughly two independent sub-samples here, not three**, and the per-decade table in
the result file should not be read as three independent confirmations.

| window | n | net ann | bench ann | net Sh | bench Sh | vol-matched | t |
|---|---:|---:|---:|---:|---:|---:|---:|
| FULL SAMPLE | 213 | 12.85% | 8.34% | 0.878 | 0.374 | +7.37% | +2.64 |
| 1990s (21 months) | 21 | 1.39% | 6.94% | 0.141 | 0.274 | **−1.32%** | −0.14 |
| 2000s | 120 | 12.17% | 8.05% | 0.744 | 0.329 | +6.80% | +1.84 |
| 2010s (72 months) | 72 | 17.34% | 9.23% | 1.372 | 0.538 | +10.54% | +4.57 |
| first half | 106 | 16.21% | 11.80% | 1.373 | 0.527 | +9.99% | +2.35 |
| second half | 107 | 9.53% | 4.90% | 0.561 | 0.221 | +5.78% | +2.18 |
| 2000–2002 dot-com | 34 | 19.04% | −9.11% | 1.304 | −0.336 | **+23.94%** | +3.63 |
| 2008–2011 crisis | 48 | 7.00% | 7.19% | 0.318 | 0.240 | +1.72% | +0.34 |
| excluding 2008–2011 | 165 | 14.56% | 8.67% | 1.244 | 0.443 | +9.37% | +3.08 |
| **excluding BOTH bear markets** | **131** | 13.39% | 13.29% | 1.232 | 0.784 | **+4.87%** | **+1.45** |
| 2003–2007 | 60 | 15.97% | 18.45% | 1.416 | 1.136 | +3.17% | +0.99 |
| 2008–2012 | 60 | 10.79% | 8.48% | 0.521 | 0.308 | +4.40% | +1.04 |
| 2013–2015 | 36 | 13.06% | 5.32% | 1.377 | 0.441 | +8.88% | +3.14 |

The 2000–2002 bust alone — 34 of 213 months — delivers a vol-matched active of **+23.94%/yr
at t 3.63**, while the benchmark loses 9.11%/yr. Strip both bear markets and the sleeve
clears +2%/yr but **fails the registered t > 2.0 gate at t = 1.45**.

**This directly qualifies the "crisis exclusion makes it STRONGER" result.** Excluding only
2008–2011 does make it stronger (+9.37%, t 3.08) — but 2008–2011 is the crisis in which this
sleeve did *least* well (vol-matched +1.72%, t 0.34). Removing a weak stretch necessarily
raises the average. The test that matters is removing the stretch that carries the result,
and that is 2000–2002. A crisis-exclusion test is only evidence if it excludes the crisis
the edge actually came from.

Calendar years: it underperforms its own benchmark in only 3 of 18 years, but violently —
**−24.7% (1999), −48.6% (2003), −37.9% (2009)**. That is textbook defensive-factor
behaviour: it wins in busts and gets destroyed in junk rallies. Max drawdown 49.5%.

This is the honest limit of the claim: **the sleeve is a bear-market payoff, not an
all-weather alpha.** That is a defensible economic property (the leverage-constraint story
predicts exactly it), and it is not the same thing as a robust +7.37%/yr.

---

## 6. LOOKAHEAD — clean, verified against raw bars

**252-day vol and beta.** Recomputed from the daily bars for 300 randomly sampled
(ticker, month-end) cells, using only bars with `date <= month-end`:

- max |vol recomputed − stored| = **3.76e-11**
- max |beta recomputed − stored| = **1.23e-09**
- windows that reached past the month-end: **0**

A lookahead produces a one-sided error, not a 1e-11 rounding gap. Trailing construction
confirmed.

**SF1 join.** 2,832 ART filings read from `SF1.csv` for 55 sampled tickers (median
`datekey − calendardate` = 45 days) and re-joined onto the cached quality values:

| join | cached values reproduced exactly |
|---|---:|
| **on `datekey` (filing date)** | **100.00%** of 3,713 |
| on `calendardate` (period end) | 39.00% of 3,692 |

Zero filings dated after the rebalance date. **Point-in-time confirmed.**

**Return alignment.** `realised_return` at month *m* is the return from month *m* to
month *m+1*, so a holding always earns strictly after the bar its signal was computed on.

**Same-bar execution.** The book does compute its signal on the month-end close and trade at
that same close. Rebuilt from the daily bars so the book trades at the **next trading day's
close** instead:

| book | gross | net | Sharpe | bench | vol-matched | t |
|---|---:|---:|---:|---:|---:|---:|
| published — trade at the signal close | 16.58% | 12.85% | 0.878 | 8.34% | +7.37% | +2.64 |
| **trade at the NEXT DAY's close** | **16.20%** | **12.47%** | **0.795** | 8.64% | **+6.98%** | **+2.44** |
| one-**month** stale signal (decay test) | 14.19% | 10.49% | 0.698 | 8.34% | +4.86% | +1.55 |

The realistic execution lag costs 0.39 percentage points of vol-matched active and the gate
still clears. The one-month figure is signal **decay**, not lookahead — but it does say the
edge has a real short-horizon component and would not survive quarterly implementation.

**Not just low beta.** OLS of the net book on its own benchmark: β = **0.4988**, annualised
α = **+8.69%/yr**, NW t = **+3.11**. A passive 0.50× benchmark book returned +4.16%/yr at
11.11% vol; the sleeve returned +12.85% at 14.64%. The excess is not explained by holding
less of the same thing.

---

## 7. THE DSR MISS — confirmed, and it is a plain FAIL

`dsr_sharpe_bar(years=17.75, n_trials=38, target=0.95)` = **0.923385**, identical to the
published `dsr_sharpe_bar`. Sample length verified at 213 months = 17.75 years, `n_trials`
verified at 38 throughout.

| n_trials | 1 | 10 | 20 | 32 | 37 | **38** | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sharpe bar | 0.3926 | 0.7755 | 0.8563 | 0.9060 | 0.9207 | **0.9234** | 0.9503 | 1.0145 |

Net Sharpe (conservative bound) **0.8779 < 0.9234 → FAIL**, by 0.0455 of Sharpe. The stream
would need +0.67%/yr more to clear it. It also fails at n_trials 32, and only passes if the
trial count is dropped below ~20 — which would be revisionist.

**Stated plainly: the sleeve does not pass its registered promotion gate.** Prereg §6
requires all four of (i) vol-matched > +2%, (ii) t > 2.0, (iii) net Sharpe ≥ 0.9234,
(iv) DSR > benchmark DSR. (iii) fails. `MARGINAL` is the correct pre-committed verdict, the
result file records it correctly, and the prereg predicted this outcome in advance
("I expect this to FAIL the DSR gate even if it passes the excess gate"). That forecast
deserves credit. `PROMOTE` is not available and no reading of this result reaches it.

Under the corrections of §1 and §3 the net Sharpe falls to **0.614** and the DSR to
**0.586**, so the gap to the bar roughly triples.

---

## 8. CAPACITY — executable, but the registered participation cap is breached

- Deployable capital **$138,110**, position value **$4,603.66**
  = 30 × 1% × the band's **median** dollar volume ($460,366).
- That position size is **one full-sample constant applied to every name**, so a name at
  the bottom of the band ($200k/day) is traded at **2.30%** of its volume, not 1%.

Participation per traded leg (n = 2,847 priced legs):

| min | 5% | 25% | **50%** | 75% | 95% | 99% | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.461% | 0.494% | 0.732% | **1.234%** | 1.799% | 2.214% | 2.287% | 2.301% |

**61.1% of legs exceed the registered `PARTICIPATION_LIMIT = 1%`**; 16.1% exceed 2%. The
square-root impact model does charge for the excess, so this is a **breach of a registered
constraint rather than an unpriced cost** — but the prereg says "position cap as a share of
the name's median dollar volume" and no such per-name cap exists in the code.

Executability itself is fine: 6.81 turnovers/yr on $138,110 is ~$470k/yr of one-way
notional, ~204 orders/yr of ~$4,600 in names doing $200k–$1M/day. Nobody notices that. **The
binding constraint is not liquidity — it is that the only band which clears the excess gate
holds $138k.** B3 ($685k), B4 ($3.1M) and B5 ($19.6M) all fail on t-statistic. This result
does not scale.

---

## 9. THE UNIVERSE CORRECTION ITSELF — holds up

The re-test's headline change is readmitting `upper_bound` cells. If they were charged too
cheaply the whole result would be an artefact of the correction.

| regime | cells (B2) | median $vol | median conservative spread | median vol |
|---|---:|---:|---:|---:|
| measured | 114,095 | $423k | 176bp | 50.1% |
| upper_bound | 85,993 | $514k | 78bp | 51.6% |

`bounds_from_estimate` sets the conservative bound to `max(estimate, tick_floor)` for
**both** regimes, so an `upper_bound` name is charged its full ceiling estimate — the most
its spread could possibly be. The cheaper median reflects genuinely higher dollar volume,
not a discount. And the book holds **37.2%** upper_bound cells against **43.0%** in the
universe, so the correction is not a device for loading up on the cheapest-priced names.

**Spread stress** (multiply every `upper_bound` spread):

| × | cost/yr | one-way | net | Sharpe | vol-matched | t | gate (i)+(ii) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 3.73% | 54.8bp | 12.85% | 0.878 | +7.37% | +2.64 | passes |
| 2.0 | 4.40% | 64.6bp | 12.18% | 0.833 | +6.71% | +2.40 | passes |
| 3.0 | 5.07% | 74.5bp | 11.51% | 0.788 | +6.04% | +2.16 | passes |
| 5.0 | 6.41% | 94.1bp | 10.18% | 0.698 | +4.72% | +1.68 | **FAILS** |

**Fragility to flat slippage** — the single most useful number for deciding whether to
believe this:

| extra bps per leg | net | Sharpe | vol-matched | t | gate |
|---:|---:|---:|---:|---:|---|
| 0 | 12.85% | 0.878 | +7.37% | +2.64 | passes |
| 20 | 11.49% | 0.785 | +6.01% | +2.14 | passes |
| **30** | 10.81% | 0.738 | +5.33% | +1.89 | **FAILS** |
| 50 | 9.45% | 0.645 | +3.97% | +1.40 | FAILS |
| 100 | 6.05% | 0.412 | +0.56% | +0.19 | FAILS |

**The gate breaks at roughly 25bp of unmodelled slippage per leg**, on top of the 54.8bp
one-way already charged. Total all-in budget before failure: ~80bp one-way in names doing
$200k–$1M/day. That is a real but not generous margin.

---

## 10. EVERY CORRECTION STACKED — B2, conservative bound

| book | gross | net | Sharpe | bench | bSh | raw exc | **vol-matched** | **t** | DSR | gate (i)+(ii) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PUBLISHED | 16.58% | 12.85% | 0.878 | 8.34% | 0.374 | +4.52% | **+7.37%** | **+2.64** | 0.874 | passes |
| + delisting window repaired | 14.89% | 11.16% | 0.746 | 5.41% | 0.242 | +5.76% | +7.54% | +2.68 | 0.754 | passes |
| + free exit legs charged | 14.89% | 10.28% | 0.686 | 5.41% | 0.242 | +4.87% | +6.66% | +2.36 | 0.685 | passes |
| **+ next-trading-day execution** | **14.51%** | **9.89%** | **0.614** | **5.71%** | **0.231** | **+4.18%** | **+6.18%** | **+2.12** | **0.586** | **passes** |
| + risk-free rate 2%/yr | 14.51% | 9.89% | 0.490 | 5.71% | 0.150 | +4.18% | +5.48% | **+1.88** | 0.586 | **FAILS** |

Every one of these books fails registered gate (iii), net Sharpe ≥ 0.9234.

---

## WHAT I VERIFIED vs WHAT I TOOK ON TRUST

**Verified by re-running / re-computing:**
reproduction of every published field; an independent re-implementation of the backtest;
`forward_return` construction over all 1.35M panel rows; 252-day vol and beta against raw
daily bars; the SF1 `datekey` join against `SF1.csv`; the vol-matched active identity and
its scale factor by hand; Newey–West at six lag structures; a stationary bootstrap with k
re-estimated; the DSR bar at eight trial counts; every exit event classified against the
unfiltered panel with 1/3/12-month follow-up returns; the delisting window against the
ACTIONS dates; P&L concentration by name, name-month and month; participation per traded
leg; four counterfactual books and two cost stresses.

**Taken on trust (not independently re-derived):**
the correctness of the EDGE spread estimator itself and its `measured` / `upper_bound`
regime labelling; the FIM-2018 calibration behind `IMPACT_COEFFICIENT_*`; Sharadar's
`closeadj` split/dividend adjustment; Sharadar's ACTIONS coverage of delistings (a missing
bankruptcy is invisible to every test here — the "maximal" scenario in §3 bounds it);
`deflated_sharpe_ratio`'s formula (the bar it produces was checked for internal
consistency, not against Bailey & López de Prado's paper); the claim that `n_trials = 38` is
the true cumulative count of configurations tried by this programme.

---

## REQUIRED FIXES, in priority order

1. **`DELISTING_WINDOW_DAYS` boundary.** Change `at < delisted_on` to `at <= delisted_on` in
   both `exit_return` and the vectorised `in_window` in `lowvol_retest.py` **and** in
   `low_vol_quality.py`, which carries the same defect. Re-run and re-publish every level.
   Until then `delisting_drag_annual` must not be reported as a measurement.
2. **Charge the free exit legs.** A held name absent from the cross-section still has to be
   sold. Price it at its last observed spread and impact inputs.
3. **Enforce the registered per-name 1% participation cap**, or amend the prereg to say the
   cap is applied to the band median rather than to each name.
4. **Move execution to the next trading day's close**, or state explicitly that the result
   assumes same-close execution.
5. **Retract the per-decade framing.** 21 months is not a decade. Report the bear-market
   dependence instead: outside 2000–2002 and 2008–2011 the vol-matched active t is 1.45.
6. Report the vol-matched active net of an assumed risk-free rate, or state that
   de-levering the benchmark is assumed to earn nothing.

## WHAT SURVIVED, AND IT MATTERS

Seventeen studies died. This one does not, and it does not die under a deliberate attempt to
kill it. Point-in-time construction is clean under direct re-computation. Concentration is
clean. The vol-matched statistic is computed correctly, its direction is right, its
autocorrelation adjustment is conservative, and it survives bootstrap resampling that prices
the estimation error the published t-stat ignores. The alpha is not disguised beta
(β = 0.50, α = +8.69%/yr, t = 3.11). The two genuine defects found here move the *levels*
substantially but move the *headline* by roughly 1.2 percentage points, because the larger
of them was paying the benchmark more than it was paying the strategy.

It is a real, bear-market-weighted, $138k-capacity, MARGINAL result that misses its own
promotion bar. That is worth more than the seventeen that died, and it is not worth more
than that.
