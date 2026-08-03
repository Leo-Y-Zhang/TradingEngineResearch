# RESULT — `trend + passive` through its registered gates on the repaired panel

Pre-registered in `trend_passive_prereg.md`, including eleven numbered predictions and a
pre-committed promotion decision, **all written before any gate was run**. Reproduce:

```
.venv/Scripts/python.exe scripts/run_trend_passive_gate.py
```

Machine-readable output in `research/sleeves/_trend_passive/trend_passive_gate.json`.
No tuning, no search, no new candidate, no ledger entry, no live path, nothing public.

> ## VERDICT: **every gate passes — and the sleeve is still not promoted, exactly as
> registered in advance.**
>
> **DSR = 0.9982** at the full 281-trial deflation against a bar of **0.5808**; the
> harshest bracket bound clears at **0.9963**. `benchmark_relative_rule` returns
> **BEATS against all four registered benchmarks**, `benchmark_sensitive=False`,
> `promotable=True` — **at all three bracket bounds**.
>
> **And it changes nothing.** The pre-committed decision in prereg §6 stands: a
> full-sample gate cannot see that the significance rests on the 2000s, nor that
> **since 2010 the book adds −0.021%/yr at t −0.01 over passive alone.** A pass means
> the premium is not a multiple-testing artefact. That is a statement about the past.
>
> **The honest number: ≈14.3%/yr at a 50% drawdown at bill+150bp, ≈12.7%/yr at
> bill+300bp**, against a corrected passive comparator of ≈8.6% and ≈8.2%.

---

## 1. The gates

| bound | book Sharpe | DSR @281 | DSR @47 | bar @281 | nominated | promotable |
|---|---:|---:|---:|---:|---|---|
| conservative | 0.7499 | **0.9963** | 0.9995 | 0.5808 | BEATS | **True** |
| **central** | **0.7834** | **0.9982** | 0.9998 | 0.5808 | BEATS | **True** |
| realistic | 0.8464 | **0.9996** | 1.0000 | 0.5808 | BEATS | **True** |

**Deflating for 281 trials instead of 47 costs almost nothing** — 0.9998 → 0.9982 — and
that is the point of the long sample rather than a sign the deflation is toothless. 738
months is enough that a 281-trial haircut moves the required Sharpe to 0.5808 while the
harshest bound sits at 0.7499.

### The benchmark-shopping detector (C7), central bound

| benchmark | verdict | Sharpe gap | ρ |
|---|---|---:|---:|
| (i) own-universe equal weight *(nominated)* | **BEATS** | +0.2756 | +0.6884 |
| (ii) passive monthly EW of the 18 | **BEATS** | +0.2888 | +0.6889 |
| (iii) equal-**RISK** passive *(substituted for C7's daily member)* | **BEATS** | +0.2820 | +0.6598 |
| (iv) 60/40 equity/rates *(added)* | **BEATS** | +0.3221 | +0.6429 |

The two benchmarks that were **added to make the panel harder** — equal-risk and 60/40 —
are the two the book beats by the *most*. It is not benchmark-sensitive.

### The ladders, book against the benchmark it must beat

| financing | book | passive alone | incremental |
|---|---|---|---:|
| bill + 150bp | 1.95× → **14.32%/yr** | 1.30× → 8.61%/yr | **+5.71 pp** |
| bill + 300bp | 1.85× → **12.65%/yr** | 1.25× → 8.15%/yr | **+4.50 pp** |

All after the 0.877 reconciliation factor, leverage solved against the **bootstrap 95th
percentile** drawdown at a 50% cap, never the observed path (rule 5).

---

## 2. THE SCORECARD — 12 of 14 correct, and the two misses matter

A forecast nobody grades is not a forecast.

| # | prediction | observed | |
|---|---|---:|---|
| P1 | DSR @281 central in 0.97–1.000 | 0.9982 | ✅ |
| P2 | DSR @281 conservative ≥ 0.95 | 0.9963 | ✅ |
| P3 | DSR bar @281 in 0.55–0.65 | 0.5808 | ✅ |
| P4 | nominated verdict BEATS | BEATS | ✅ |
| P5 | all panel members BEAT → promotable | True | ✅ |
| **P6** | **ρ(book, nominated) > 0.90** | **0.6884** | ❌ |
| P7a | bill+300bp DD50 leverage 1.85–1.95× | 1.85× | ✅ |
| P7b | bill+300bp DD50 CAGR×0.877 12.5–13.5% | 12.65% | ✅ |
| P8a | benchmark DD50 leverage 1.30–1.45× | 1.30× | ✅ |
| P8b | benchmark DD50 CAGR×0.877 8.0–9.5% | 8.61% | ✅ |
| P9 | incremental +4.5 to +6.5 pp | +5.71 pp | ✅ |
| **P10a** | **post-2010 active in 0.0 to +1.5%/yr** | **−0.021%/yr** | ❌ |
| P10b | post-2010 \|t\| < 2 (not significant) | −0.014 | ✅ |
| P11 | same verdict at all three bounds | all True | ✅ |

### P6 — I repeated the exact error this programme already refuted

I predicted ρ > 0.90 on the reasoning *"the book is 72% passive by capital."* **That is
the capital weight, and I used it as if it were a risk weight** — which is precisely the
mistake the survivor verification identified and corrected: the risk contribution is
**50.00 / 50.00**, not 72/28, and the brief for that verification made the same slip.

A book that is half trend by risk correlates with passive at **0.69**, not 0.90. The
error was recorded in a document I had read, in this repository, and I made it anyway.
**Worth more than the twelve correct predictions:** it demonstrates that the paired
bootstrap is doing real work — a ρ of 0.69 rather than 0.90 means the paired comparison
tightens the confidence interval *less* than I assumed, so the BEATS verdicts were
obtained against a **weaker** correlation benefit than predicted, not a stronger one.

### P10a — wrong in direction, right in substance

I predicted the post-2010 active would turn mildly positive on the repaired panel,
because the correction hurts the long-only leg more. It moved from **−0.13%/yr (t −0.09)
on the old panel to −0.021%/yr (t −0.014)** — toward zero, but not across it. Over 198
months the corrected book adds **nothing** over passive alone.

The miss is directional and immaterial in size; **the substantive claim (P10b: not
significant) was right, and it is the one the promotion decision rests on.**

---

## 3. The decision, unchanged from what was registered before the run

**NOT PROMOTED. No live path, no paper path, no allocation.**

This was committed in prereg §6 *before* the gates ran, so a clean sweep could not buy
its way past it:

* the full-sample premium **rests on the 2000s** — leave that decade out and the active
  falls to +1.29%/yr at t 1.39 (survivor §2b);
* **since 2010 there is no premium at all** — now confirmed on the repaired panel at
  −0.021%/yr, t −0.014, over 198 months;
* a full-sample DSR and a full-sample benchmark comparison **cannot see either**.

The gate answers *"is this premium a multiple-testing artefact?"* — no, it is not. It
does not answer *"is this premium available now?"*, and the only evidence bearing on that
question says no.

**The only result that would have changed this** was a post-2010 finding that was itself
significant. P10 predicted in advance that there would not be one. There was not.

---

## 4. What still stands against all of it

1. **The panel is still an upper bound.** 21.2% of live cells — the commodity roll and
   USDX — remain uncorrected, and every uncorrected error runs the same way.
2. **The EUR/GBP FX residual (~0.9%/yr) is unexplained**, and its registered explanation
   was refuted by its own diagnostic.
3. **37% of the equity block is bracketed rather than measured**, and N225 — the largest
   single passive contributor — is bracketed for its entire life after its measurement
   was rejected.
4. **Neither leg is individually validated.** Trend loses to its own universe; passive is
   buy-and-hold. A significant *diversification* premium between two unvalidated legs is
   a statement about their correlation, and ρ = 0.69 is that statement.
5. **≈14.3%/yr at a 50% drawdown is not 30%/yr**, and a 50% drawdown is not a drawdown
   anybody holds through.

---

## 5. What this closes

The survivor verification's closing instruction — *"before any further sleeve work, the
panel's return conventions should be repaired at source"* — is **discharged**, and the
sleeve it was blocking has now been through its registered gates on the repaired panel.

The programme's ceiling, measured three times and lowered each time: 17–20%/yr claimed →
≈17.4% published → ≈15.4% under the constant-charge approximation → **≈14.3%/yr at a 50%
drawdown, ≈12.7% at realistic retail financing.** Every step downward came from checking
something nobody had checked.
