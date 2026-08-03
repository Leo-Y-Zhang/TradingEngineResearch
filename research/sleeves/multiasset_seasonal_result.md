# RESULT — CALENDAR SEASONALITY on the long-history multi-asset panel: **DEAD**

**Pre-registered** in `research/sleeves/multiasset_seasonal_prereg.md`, committed as `4b0ca5d`
**before** `research/sleeves/multiasset_seasonal.py` existed. Run **once**. No tuning, no
second look, no window moved after the fact.

**Code** `research/sleeves/multiasset_seasonal.py` · **artefacts** `research/sleeves/_seasonal/`
· re-run is **byte-identical** (`result.json` md5 `31417607beca0a0ff921bdc65f1de920` on two
consecutive runs).

**Sample** 736 months, **61.33 years**, 1965-03-31 → 2026-06-30, 18 instruments (the trend
sleeve's `PRIMARY_UNIVERSE`, reused unchanged). **Trials spent: 3** (E1, E2, E3).

---

## 1. Headline

| | |
|---|---|
| Composite (E4) net Sharpe, 2 bps | **0.6231** |
| Composite (E4) net Sharpe, 10 bps | **0.4680** |
| **Its own benchmark — equal-weight long-only, levered to the same vol** | **0.7065** |
| Vol-matched active return vs that benchmark | **−2.06 %/yr (t = −0.64)** at 2 bps · **−5.86 %/yr (t = −1.84)** at 10 bps |
| Geometric excess (the *flattering* statistic) | **−6.17 %/yr** — negative too |
| DSR ≥ 0.95 bar at 61.33 yr | 0.4808 (n=32) · **0.4973 (n=44)** · 0.5093 (n=56) · 0.5848 (n=304) |
| Half-Kelly reachable return at 0.4680 | **8.21 %/yr** |
| **30 %/yr requires** | **Sharpe 0.894** |

**VERDICT: DEAD**, by the pre-registered rule (§11 of the prereg), and not narrowly.
The failing conditions are listed in §9.

**The one-sentence version.** Every seasonal effect tested loses to a levered long-only
holding of its own universe, in **both** the pre-publication and the post-publication era,
and the only leg with genuine calendar content (turn-of-the-month) has a **breakeven cost of
14.5 bps round trip** and is therefore a cost story rather than a return story.

---

## 2. The three effects, measured (20 % vol target, net of the cost bracket)

| effect | gross S | net 2 bps | net 10 bps | benchmark S | vol-matched active @2bps | @10bps | breakeven | days in market |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **E1 turn-of-month** | 0.7451 | 0.6455 | 0.2455 | 0.6993 | −1.07 % (t −0.39) | **−8.98 % (t −3.29)** | **14.5 bps** | 18.3 % |
| **E2 Halloween** | 0.6138 | 0.6084 | 0.5868 | 0.7021 | −2.06 % (t −0.77) | −2.53 % (t −0.95) | 216.7 bps | 49.6 % |
| **E3 January (equity)** | 0.3661 | 0.3631 | 0.3511 | 0.7199 | **−12.70 % (t −2.19)** | **−13.04 % (t −2.26)** | 173.8 bps | 8.3 % |
| **E4 composite** | 0.6616 | 0.6231 | 0.4680 | 0.7065 | −2.06 % (t −0.64) | −5.86 % (t −1.84) | 31.6 bps | 58.8 % |

**The vol-matched active return is negative in every cell of that table.** So is the
geometric excess (E1 −9.66 %, E2 −2.79 %, E3 −13.21 %, E4 −6.17 % at 10 bps). The two
statistics that fail in opposite directions — geometric excess flatters low-vol strategies
and killed PEAD; raw arithmetic active flatters high-vol ones and killed trend's headline —
**agree here**, which is the cleanest form a negative result can take.

At **10 % / 20 % / 40 %** vol targets the composite nets **0.4142 / 0.4680 / 0.6094** at
10 bps. The rise with the target is not an edge: it is the **gross cap** (10× equity)
binding on 3 / 261 / 1,588 days respectively, which truncates leverage in high-volatility
periods and acts as an unintended volatility limiter. It is reported because it is real, and
labelled because it is not the signal.

---

## 3. The publication test — the most informative result in this study

This is the test the brief singled out, and it inverts under the correct statistic.

**Raw sleeve Sharpe, split at each effect's own publication year:**

| effect | split | pre | post | reads as |
|---|---:|---:|---:|---|
| E1 turn-of-month | 1987 | +0.043 (263 m) | **+0.355** (474 m) | *improved* |
| E2 Halloween | 2002 | +0.533 (443 m) | **+0.666** (294 m) | *improved* |
| E3 January | 1976 | +0.440 (131 m) | +0.338 (606 m) | decayed |

Two of three apparently got **better** after being written about, which would be a
remarkable finding. It is an artefact. **The benchmark improved by more:**

| effect | benchmark pre | benchmark post |
|---|---:|---:|
| E1 | +0.628 | +0.736 |
| E2 | +0.671 | +0.747 |
| E3 | +0.391 | +0.781 |

Splitting the **vol-matched active** series instead — the sleeve minus its own levered
universe, era by era — removes the market drift and gives the actual answer:

| effect | active pre | active post |
|---|---:|---:|
| **E1 turn-of-month** | **−0.545** (t −2.42) | **−0.372** (t −2.35) |
| **E2 Halloween** | **−0.157** (t −1.02) | **−0.090** (t −0.36) |
| **E3 January** | **+0.237** (t +0.82) | **−0.461** (t −3.20) |

> **Did the effects survive being written about? The question does not arise for E1 and E2,
> because they never beat their own universe in the first place — their active Sharpe is
> negative on both sides of the publication date. Only E3 shows the classic decay signature
> (a positive-but-insignificant active edge before Rozeff & Kinney 1976, a significantly
> negative one after), and it is the weakest effect of the three.**

**The confound, quantified rather than waved at.** The eligible universe is not constant
across the split: it averages **2.6 instruments before 1987 and 14.6 after**; 4.1 before 2002
and 17.3 after; 1.6 before 1976 and 12.9 after. Mean eligible instruments by decade: 1930s
0.9 · 1960s 2.2 · 1970s 5.2 · 1980s 7.3 · 1990s 10.3 · 2000s 15.2 · 2010s-2020s 18.0. The
early era is therefore close to "SPX plus two Treasury points", and **no era comparison on
this panel is ceteris paribus.** The active-return split is nonetheless the right test,
because both the sleeve and its benchmark are drawn from the *same* universe in each era.

---

## 4. Two controls, and what they say

**4a. The pre-registered placebo** (prereg §7.11) — the turn-of-month window moved to the
**10th–13th business days**, the mid-month interior no paper claims anything about:

| | gross S | net 10 bps S |
|---|---:|---:|
| live E1 (last + first three) | **0.7451** | **0.2455** |
| placebo (10th–13th) | 0.4522 | −0.0202 |

E1 stands clear of its placebo. **The turn-of-month window does contain real calendar
information gross.** That is the single positive finding in this study, and costs erase it.

**4b. The date-scramble drift baseline** (added as a machinery control; permutes returns
*within each instrument's own live dates*, so availability windows and NaN patterns are
preserved exactly while every calendar alignment is destroyed — positions are untouched
because they never read a return). 4 fixed seeds. What survives is the market drift the
sleeve's in-market fraction would have earned with **no calendar information at all**:

| | live gross S | scrambled baseline (mean ± sd) | live vs baseline |
|---|---:|---:|---:|
| E1 turn-of-month | 0.7451 | 0.4384 ± 0.1496 | **+2.05 sd** |
| E2 Halloween | 0.6138 | 0.7380 ± 0.0525 | **−2.37 sd** |
| E4 composite | 0.6616 | 0.6016 ± 0.1891 | +0.32 sd |

**Halloween's Sharpe of 0.61 is worse than being in the market on a random half of the days.**
The composite is statistically indistinguishable from its no-information baseline.

*Caveat that must travel with 4b:* scrambling also destroys volatility clustering, which
lowers the fat-tailedness of the compounded monthly series and therefore **biases the
baseline upward**. So "live above baseline" (E1) is strong evidence and "live below
baseline" (E2) is suggestive rather than conclusive. It is stated because it cuts against
the reading I would otherwise prefer.

---

## 5. Correlation — the brief's premise was right and **my own prediction was wrong**

Pre-registered prediction **P4** said rho(seasonal, trend) would be **positive, +0.15 to
+0.45**, on the argument that a long-flat seasonal book and a trend book are both net long
and therefore share market beta whatever their signals do. **That prediction is refuted:**

| pair | rho | months |
|---|---:|---:|
| **seasonal composite vs trend** | **+0.0272** | 736 |
| **seasonal composite vs carry** | **−0.0395** | 269 |
| trend vs carry *(cross-check)* | −0.0441 | 269 |
| E1 vs trend / carry | −0.0115 / −0.0026 | 737 / 269 |
| E2 vs trend / carry | +0.0246 / −0.0177 | 737 / 269 |
| E3 vs trend / carry | +0.0360 / −0.0759 | 737 / 269 |

At 736 months the standard error of a correlation is ~0.037, so **+0.027 is statistically
indistinguishable from zero.** The trend-vs-carry figure reproduces the carry study's
recorded −0.0441 exactly, which independently validates this pipeline's return series.

**The brief's structural argument therefore holds at the P&L level, not merely at the signal
level, and my objection to it was wrong.** The mechanism I missed: the trend sleeve is
volatility-targeted and long-short, so its net market exposure oscillates around zero and
its realised beta is small; the seasonal book's directional exposure is concentrated in
calendar windows that are uncorrelated with when trend happens to be long.

**A genuinely uncorrelated sleeve is exactly what the portfolio needs — and this one still
is not deployable, because it loses to passive.** Zero correlation to two things is worth
nothing if the thing itself has no edge over owning the market.

---

## 6. Portfolio arithmetic, measured

| portfolio | Sharpe | half-Kelly compound | months |
|---|---:|---:|---:|
| trend + carry (equal risk) — *incumbent* | **0.6546** | 16.07 %/yr | 269 |
| **+ seasonal (equal risk)** | **0.7140** | **19.12 %/yr** | 269 |
| + seasonal (point-in-time risk parity) | 0.7301 | 19.99 %/yr | 257 |
| trend + seasonal (full 61 yr overlap) | 0.7549 | 21.37 %/yr | 736 |
| *the levered long-only benchmark, on its own* | *0.7065* | *18.72 %/yr* | *736* |
| **required for 30 %/yr** | **0.894** | 30 % | — |

Adding seasonal moves the pair from **0.655 → 0.714**, i.e. **+0.059** — inside the
pre-registered P8 band of "< +0.10". Decades of the three-sleeve combination: 2000s 0.906 ·
2010s 0.657 · 2020s 0.588; max drawdown −14.8 % at a 10 % vol scaling.

**The formula check.** `S = s·√(N/(1+(N−1)ρ̄))` with N = 3, measured ρ̄ = −0.019 and mean
sleeve Sharpe 0.503 gives **1.036** — against a **measured 0.714**. The formula overstates by
45 % because it assumes the three sleeves have equal Sharpes, and they do not (0.612 / 0.430 /
0.468). **The measured number is the one that counts; the formula is not a route to 30 %.**

**Reaches Sharpe 0.894? NO** — not the sleeve (0.468 net), not the pair (0.655), not the
three-sleeve combination (0.714, or 0.730 under point-in-time risk parity).

---

## 7. Secondaries — declared in advance, reported unconditionally, none promotable

| # | secondary | result |
|---|---|---|
| **S1** equity-block only | E1 gross 0.650 / net10 0.442 vs bench 0.670, active −5.53 % (t −1.47); E2 gross 0.641 / net10 0.632 vs bench 0.680, active −1.52 % (t −0.38). **Both still lose to the benchmark** — the cross-asset extension is not what killed it. |
| **S2** long–short | Destroys the sleeve: E1 gross 0.009 / net10 **−0.415**; E2 0.284 / 0.247; E3 −0.351 / **−0.367**; composite −0.057 / **−0.271**, active −22.2 % (t −4.05). Shorting the unfavourable window shorts the market's drift. Its correlations *are* lower (−0.020 to trend, −0.018 to carry) — a more uncorrelated sleeve with a **negative** Sharpe. |
| **S3** observed-trading-day calendar for E1 | gross **0.7923** vs 0.7451 on the Mon-Fri grid — the conservative §2b choice did attenuate the effect, by ≈0.05 Sharpe, exactly as disclosed in advance. It changes nothing: net10 **0.2043**, active −9.88 % (t −3.73), active Sharpe −0.534 pre-1987 / −0.445 post. |
| **S4** unscreened panel | gross 0.6621 / net10 0.4685 / active −5.87 % (t −1.84) vs 0.6616 / 0.4680 / −5.86 % screened. **The verdict is completely insensitive to the quarantine decision.** |

---

## 8. Risk, concentration and attribution (composite, 20 %)

- **Max drawdown −73.0 %**, trough 1974-12; monthly skew **+4.06**; worst month −33.2 %;
  realised vol 24.6 % against a 20 % target.
- **P&L concentration breaches the 3 % alarm**: top (instrument, month) cell = **3.36 %** of
  net P&L (NASDAQ, 1975-01). Top instrument NASDAQ **20.5 %**. For E3 alone the top cell is
  **12.2 %** (SPX, 1967-01) — a single month is an eighth of that leg's entire P&L.
- **Block attribution: equity 84.8 %, commodity 10.5 %, rates 6.0 %, FX −1.2 %.** The sleeve
  is an equity-timing bet wearing a multi-asset coat.
- The drawdown and the skew have the same cause, and it is a **construction consequence of
  the pre-registered equal-risk rule**, disclosed rather than repaired: the January leg is
  flat 11 months in 12, so its own trailing volatility is small and `1/σ` hands it enormous
  notional in the month it is on. The six largest months of the composite are five Januaries
  and one April. This is what the pre-registration bought — the rule was fixed before the
  run, so the artefact is visible instead of tuned away.
- Turnover 92.8 ×/yr (E1 alone 191.3 ×/yr); mean gross leverage 2.27–4.06× depending on leg.

---

## 9. The verdict against the pre-registered rule

`S_net` (2 bps) = 0.6231 · `S_cons` (10 bps) = 0.4680 · `bar_44` = 0.4973 · `bar_304` = 0.5848.

| PROMISING condition | required | measured | pass |
|---|---|---|:--:|
| `S_net ≥ bar_44` | ≥ 0.4973 | 0.6231 | ✅ |
| `S_cons > 0` | > 0 | 0.4680 | ✅ |
| **vol-matched active t ≥ 2.0** | ≥ +2.0 | **−0.64** | ❌ |
| no decade with negative Sharpe | — | **1960s −0.072** | ❌ |
| post-pub ≥ ½ pre-pub (active) | — | **negative in both eras** | ❌ |
| top (instrument, month) share < 3 % | < 3 % | **3.36 %** | ❌ |
| E1 beats the placebo | — | 0.745 vs 0.452 | ✅ |

MARGINAL needs `S_net ≥ 0.35` (✅) **and** vol-matched active t ≥ 1.5 (**−0.64**, ❌).

## **→ DEAD.**

**The DSR gate applied to the benchmark, as mandated.** The equal-weight long-only book's
net Sharpe of **0.7065 clears the DSR bar at all four trial counts** (32 / 44 / 56 / **304**).
So the comparison is not "a marginal strategy against a marginal benchmark": the passive
alternative is itself statistically solid, and the sleeve loses to it.

---

## 10. Trial and hypothesis accounting

**Calendar hypotheses tested: 3.** E1, E2, E3, each named, dated and cited in the
pre-registration before the data was touched. E4 is a fixed equal-risk combination with no
free parameter and is not counted. S1–S4, the placebo and the scramble control are
non-promotable by pre-registration and are not counted. **No window was moved, no month was
added or removed, no instrument subset was searched.** The nine other documented calendar
anomalies that could have been reached for after these failed are listed in prereg §1a
precisely so that their absence is auditable.

**Deflation for the inherited search (prereg §8a).** Three hypotheses is my own count; the
literature that produced them searched a far larger space, and inheriting a published window
inherits its selection bias. Enumerated in advance: 264 contiguous month-of-year windows ×
direction, plus 42 month-boundary day windows × direction, giving **n_trials = 304**.

| n_trials | DSR bar @ 61.33 yr | E4 net 2 bps = 0.6231 | E4 net 10 bps = 0.4680 |
|---:|---:|:--:|:--:|
| 32 | 0.4808 | clears | fails |
| **44** (honest cumulative) | **0.4973** | clears | **fails** |
| 56 | 0.5093 | clears | fails |
| **304** (inherited search) | **0.5848** | clears | fails |

The composite clears the deflated bar at the realistic cost bound and fails at the
conservative one. **That is not what decides the verdict** — the vol-matched active return
is, and it is negative at both bounds. A strategy can clear DSR and still be worthless if a
levered index fund beats it, which is the third time this programme has measured exactly
that.

---

## 11. Predictions scorecard (prereg §10, scored honestly)

| # | prediction | outcome |
|---|---|---|
| P1 | E4 gross in [0.30, 0.65], point 0.42 | **missed** — 0.6616, just above the band |
| P2 | E1 [0.30,0.80] · E2 [0.25,0.60] · E3 [0.00,0.35] | **1 / 3** — 0.745 ✅, 0.614 ✗, 0.366 ✗ (both narrowly high) |
| P3 | every effect's post-pub Sharpe **lower** | **REFUTED** on raw Sharpe (E1, E2 rose) and on active Sharpe (E1, E2 rose from a negative base). Only E3 decayed. |
| P4 | rho(seasonal, trend) **positive, +0.15…+0.45** | **REFUTED** — +0.027, indistinguishable from zero. The brief was right and I was wrong. |
| P5 | vol-matched active not significantly positive for any effect | ✅ — none positive; E3 significantly **negative** |
| P6 | E1 breakeven < 25 bps; E2/E3 > 100 bps | ✅ **exactly** — 14.5 · 216.7 · 173.8 bps |
| P7 | sleeve alone does not reach 0.894 | ✅ — 0.468 |
| P8 | 3-sleeve Sharpe gain < +0.10 over 0.655 | ✅ — +0.059 |

Four clean hits, two refutations, two magnitude misses. **The two refutations are the
valuable part**: seasonality's returns really are uncorrelated to trend and carry, and the
"post-publication decay" story does not survive contact with a correct benchmark — the
effects did not decay, they were never there relative to owning the market.

---

## 12. Verification (all controls green, `_seasonal/result.json → verification`)

| check | result |
|---|---|
| DSR bar reproduces both recorded anchors (1.4881 @ 7 yr, 0.5971 @ 40 yr, n=32) | ✅ |
| Position matrix is **bit-identical** when rebuilt against a panel of pure noise — positions are a function of (calendar, lagged monthly vol) and nothing else | ✅ |
| **Perfect-foresight positive control on the identical pipeline** (signal = sign of the day's realised return): gross Sharpe **3.978**, net 3.824 | ✅ — the machinery can find an edge, so the negative is a finding, not a bug |
| Daily → monthly compounding exact to 1e-12 | ✅ |
| TOM window has exactly 4 flagged business days in **every** month, 1965–2026 | ✅ |
| Halloween flags exactly 6 months/yr | ✅ |
| Re-run byte-identical | ✅ |

**A defect found and fixed before the result was read.** The first implementation dropped the
months in which the sleeve was flat instead of scoring them as zeros, and masked the
benchmark to the sleeve's own trading days. That inflates the annualised Sharpe by
1/√(time-in-market) — a factor of **√12 for the January leg**, which showed a fictitious
1.063 against a fictitious benchmark of 1.552. Corrected, it is 0.351 against 0.720. The
corrected construction runs the sample from the first day the book carries risk to the last
complete calendar month, scores every day inside it including flat ones, and holds the
benchmark on all of them.

---

## 13. What this buys the programme

1. **Fourth study in a row to die against the same statistic.** Trend, value, seasonality and
   now every seasonal leg individually lose to a **vol-matched levered holding of their own
   universe**. Carry is the sole exception and it cannot clear DSR. The binding problem is
   not the DSR bar and not costs — it is that **none of these signals beats owning the panel.**
2. **The passive benchmark is the strongest single number in the study: Sharpe 0.7065 over
   61.3 years, clearing DSR at n_trials = 304, worth 18.7 %/yr at half Kelly** — better than
   trend (0.61), carry (0.43), seasonal (0.47), and within 0.06 of the whole three-sleeve
   combination (0.714). *Caveat that must travel with it:* the 18 instruments are a
   hindsight-selected set of surviving major markets, so 0.7065 is an **upper bound** on what
   was investable ex ante; the equity legs are price-only (dividends excluded, ≈2–4 %/yr) and
   the bond legs omit roll-down (≈0.5 %/yr), both of which push the other way.
3. **The multi-sleeve thesis survives its second real test.** Measured ρ ≈ 0 to both existing
   sleeves, from a signal that cannot share an estimation window by construction — this is
   the *opposite* of the value sleeve, whose diversification was overlap. The arithmetic is
   sound; what is missing is sleeves with a positive edge to put into it. **Three genuinely
   uncorrelated sleeves at 0.714 is still 0.18 of Sharpe short of 30 %/yr.**
4. **Costs are decisive for anything that trades the calendar.** Turn-of-the-month is the one
   leg with measurable, placebo-beating calendar content, and it breaks even at **14.5 bps
   round trip**. At the panel's realistic 1–5 bps it survives (net 0.646); at 10 bps it is
   0.246. Any TOM implementation is a bet on execution quality, not on the anomaly.
5. **Do not spend further trials on the calendar.** Nine remaining documented anomalies are
   listed in prereg §1a and were deliberately left untested. Given that the three
   best-evidenced ones all fail against a vol-matched benchmark in *both* eras, and that the
   composite is +0.32 sd from a no-information baseline, the prior on the rest is poor and
   each one raises the DSR bar for every other sleeve.

---

**One run. No tuning. No second look. Banked as DEAD.**
