# RESULT — the panel's return conventions, repaired at source

Pre-registered in `convention_repair_prereg.md` **before any corrected series existed**.
Reproduce, in order:

```
.venv/Scripts/python.exe scripts/build_convention_inputs.py --use-cache
.venv/Scripts/python.exe scripts/run_convention_repair.py
.venv/Scripts/python.exe scripts/run_convention_repair_book.py
```

Machine-readable output in `research/multiasset/_convention/*.json`. No strategy search,
no ledger entry, no live path, no broker path, no vendor rows committed, nothing public.

> ## VERDICT: **the repair confirms the constant-charge finding and then goes past it.**
>
> **The whole bracket sits below 0.894, and the central case sits below the 0.8206 the
> approximation produced.** The survivor verification charged a constant against the old
> panel and said its own answer was an upper bound. It was.
>
> | panel | book Sharpe | trend | passive | vol-matched active | t | P(S<0.894) |
> |---|---:|---:|---:|---:|---:|---:|
> | **old (as published)** | **0.9033** | 0.6116 | 0.6691 | +2.11%/yr | 2.35 | 46.8% |
> | constant-charge approximation (survivor §9c) | 0.8206 | 0.5660 | 0.5773 | +2.16%/yr | 2.43 | 71.3% |
> | **corrected — conservative** | **0.7499** | 0.5469 | 0.4788 | +2.38%/yr | 2.62 | **87.2%** |
> | **corrected — CENTRAL** | **0.7834** | 0.5708 | 0.5078 | **+2.44%/yr** | **2.68** | **80.4%** |
> | **corrected — realistic** | **0.8464** | 0.6125 | 0.5616 | +2.54%/yr | 2.79 | 63.8% |
>
> **The honest survivable-drawdown return falls again: ≈14.3%/yr at a 50% drawdown**
> (1.95×, bill+150bp, bootstrap-p95 drawdown, × the 0.877 reconciliation factor), against
> the survivor document's ≈15.4% and the original ≈17.4%. At a 35% cap, **≈12.1%/yr**.
>
> **The diversification premium does not merely survive — it widens**, to +2.44%/yr at
> t 2.68 (from +2.11%/yr at t 2.35). §4 shows exactly why, and it is not a compliment to
> the strategy: **both legs got worse, and the long-only leg got worse faster.**
>
> **The baseline reproduces bit-for-bit before anything was changed:** the old panel
> re-runs to Sharpe **0.903314** against the recorded 0.9033.

---

## 1. The controls decided this, and two of them fired

Method rule 9 says build the positive control first and give it a leg the old model must
fail. Six were registered. Four passed outright, one was falsified and resolved by
throwing away a measurement, and one diagnostic refuted its own hypothesis.

| control | result |
|---|---|
| **A — US equity** | **PASS with teeth.** Corrected `SPX` matches `SPY − bill` at **0.244%/yr** against a 0.25% budget, corr **0.9975**, n=401. The **uncorrected** `SPX` fails the identical test at **0.748%/yr**. The control discriminates 3:1. |
| **B — the block that must not move** | **PASS.** The three rates series came through **byte-identical**, `max |Δ| = 0` over every cell. |
| **C — FX carry** | **PASS on the registered leg.** Corrected `JPYUSD` against `FXY − bill` goes from **1.962%/yr** (spot-only) to **0.496%/yr**, inside the 0.75% budget. |
| **D — index type** | **FALSIFIED for N225.** See §2. |
| **E — the ETF bias budget** | **PASS, but smaller than predicted.** The US residual is **−0.146%/yr** over 389 months; the registered prediction was −0.3% to −1.5%. Direction right, magnitude a third of the low end. SPY's 0.0945% fee plus a small CRSP-vs-S&P composition difference is a sufficient explanation. |
| **F — no invented corrections** | **PASS.** With nothing registered to correct, the pipeline returns the panel unchanged. |

### The FX diagnostic refuted its own hypothesis, and that is recorded rather than buried

EUR and GBP still sit **0.94%/yr** and **0.86%/yr** above their currency-ETF benchmarks
after correction — outside the 0.75% budget. The pre-registration named these as
non-discriminating legs, so this is not a gate failure, but it is a residual and it was
worth explaining. The candidate explanation was that a currency-deposit ETF earns nothing
in a zero-rate era while its fee keeps accruing, which would put the residual in low-rate
months. **Measured, the opposite is true:**

| pair | gap when the foreign 3m rate ≤ 0.5% | gap when it is above 0.5% |
|---|---:|---:|
| JPYUSD | +0.478%/yr (180 mo) | +0.557%/yr (52 mo) |
| GBPUSD | +0.552%/yr (34 mo) | +0.910%/yr (201 mo) |
| EURUSD | +0.477%/yr (121 mo) | +1.394%/yr (121 mo) |

The residual is **larger** in normal-rate months, not smaller. **The hypothesis is
refuted and the residual is unexplained.** The FX correction is therefore verified only
on the JPY leg, and carries an unexplained ~0.9%/yr residual on EUR and GBP that this
repair does not claim to have resolved.

---

## 2. Control D was falsified, and the response was to reject a measurement, not to
## weaken a threshold

The registered prediction: DAX shows a gap below +0.5%/yr (its dividends are already
inside the index) and every price index shows one above +0.8%/yr.

| instrument | reference | measured gap | predicted | outcome |
|---|---|---:|---|---|
| **DAX** | EWG | **−1.526%/yr** | < +0.5 | **as predicted** — it is a total-return index |
| FTSE100 | EWU | +3.220%/yr | > +0.8 | as predicted |
| HSI | EWH | +2.330%/yr | > +0.8 | as predicted |
| ASX200 | EWA | +3.771%/yr | > +0.8 | as predicted |
| NASDAQ | QQQ | +0.294%/yr | > +0.8 | **below the floor** |
| **N225** | EWJ | **−0.027%/yr** | > +0.8 | **below the floor** |

The two low readings are **not the same kind of event**, and the distinction is
structural rather than fitted:

* **QQQ against the Nasdaq-100 price index is an index-MATCHED pair.** Same basket, one
  total-return and one price. Composition risk is zero by construction, so the measured
  gap **is** that index's dividend yield — and the Nasdaq-100's yield genuinely is small
  (0.53%/yr measured). The registered +0.8% floor was mis-specified for a low-yielding
  index, not violated by a bad measurement.
* **EWJ against the Nikkei 225 is UNMATCHED.** MSCI Japan is a broad
  capitalisation-weighted index; the Nikkei 225 is a price-weighted 225-stock index. The
  drift between them is larger than the dividend being measured, and it comes out
  negative enough to cancel the dividend entirely. **A dividend yield of −0.03%/yr is not
  a measurement of anything.**

**Resolution: N225's measurement is REJECTED.** It is bracketed over the whole 61 years,
its central bound charges it the **full bill with no dividend credit**, and its realistic
bound shows what a US-like yield would do. The threshold was not touched. `index_matched`
is a declared property of each pair, verifiable from construction.

**This is disclosed as a post-hoc amendment.** The matched/unmatched distinction was
made after seeing the result. It is defensible because it is structural and because the
resolution it produced is the *harsher* one for the instrument concerned — N225 now takes
the largest correction in the panel (−4.64%/yr).

---

## 3. What the corrected book actually rests on

The point of the provenance frame is that a measured correction and an assumed one must
never look alike. Over 9,379 live instrument-months:

| | share |
|---|---:|
| **MEASURED** — a total-return source overlaps the month | **33.6%** |
| BRACKETED — no source; the registered bracket covers it | 17.5% |
| ALREADY_EXCESS — the rates block, untouched and correct | 22.8% |
| UNCORRECTED — the commodity roll and USDX, unfixable from free data | 21.2% |
| EXEMPT — DAX, corrected by definition | 4.9% |

Within the equity block alone: **52.6% measured, 37.0% bracketed, 10.4% exempt.**

**The measurement windows are shorter than the pre-registration assumed, and the reason
is the currency leg, not the ETF.** The country ETFs all start 1996-03, but a
USD-quoted ETF has to be de-dollarised with a panel FX series, and those start later:

| instrument | first measured month | limited by | mean measured yield |
|---|---|---|---:|
| SPX | **1928-12** | nothing — the French library covers the whole sample | 3.63%/yr |
| N225 | 1997-10 | *(rejected — see §2)* | (0.76%/yr) |
| HSI | 1997-02 | no FX leg needed; HKD is pegged | 2.32%/yr |
| NASDAQ | 2000-02 | QQQ's 1999 inception | 0.53%/yr |
| FTSE100 | 2004-11 | **GBPUSD**, not EWU | 3.06%/yr |
| ASX200 | 2007-04 | **AUDUSD**, not EWA | 3.75%/yr |

So the honest reading of the equity correction is: **the US leg is measured over sixty
years; everything else is measured over twenty to thirty and assumed before that.**

---

## 4. Why the diversification premium widened — the mechanism, measured

The premium rising from +2.11%/yr to +2.44%/yr looks too convenient, so it was tested.

The correction applies a mean **−2.03%/yr** drag to the equity block. Weighted by the
trend leg's *own position sign*, the trend leg pays only **−0.076%/yr** of it. The reason
is visible in the weights:

| | months short | mean weight |
|---|---:|---:|
| SPX | 14.9% | +0.074 |
| NASDAQ | 12.2% | +0.050 |
| N225 | 20.3% | +0.027 |
| FTSE100 | 9.4% | +0.047 |
| DAX | 9.5% | +0.023 |
| HSI | 10.0% | +0.021 |
| ASX200 | 5.7% | +0.042 |

A short position **earns** the correction instead of paying it, and the corrected series
also generate fewer long signals — the trend book's mean net exposure falls from **0.516
to 0.391**. The passive leg is long-only by construction and pays the drag in full.

**This is real economics, not an artefact: collateral on a short position earns the
risk-free rate, which is precisely what the old panel was failing to account for.** But
it must not be read as good news. **Both legs got worse** — trend 0.6116 → 0.5708 and
passive 0.6691 → 0.5078. The premium is a *difference* that widened because the
comparator fell faster. Nothing here says the trend leg improved.

---

## 5. The corrected numbers

| quantity | published | constant-charge (survivor §9c) | **repaired at source** |
|---|---:|---:|---:|
| book Sharpe, 738 months | 0.9033 | 0.8206 | **0.7834** (bracket 0.7499 … 0.8464) |
| 95% bootstrap CI | [0.659, 1.155] | [0.577, 1.072] | **[0.539, 1.035]** |
| P(true Sharpe < 0.894) | 46.8% | 71.3% | **80.4%** (87.2% conservative) |
| P(true Sharpe < 0.75) | 10.6% | 28.4% | **39.0%** (49.3% conservative) |
| vol-matched active vs passive | +2.11%/yr, t 2.35 | +2.16%/yr, t 2.43 | **+2.44%/yr, t 2.68** |
| CAGR at DD ≤ 50%, ×0.877 | 17.42% | 15.42% | **14.32%** (1.95×) |
| CAGR at DD ≤ 35%, ×0.877 | 13.98% | 12.77% | **12.07%** (1.40×) |
| capital weights trend/passive | 0.278 / 0.722 | — | 0.280 / 0.720 |

**Per decade** (book Sharpe), old → corrected central:

| | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | 0.823 | 0.787 | 1.156 | 1.010 | 1.051 | 0.458 | 0.955 |
| **corrected** | 0.699 | 0.623 | 0.819 | 0.854 | **1.002** | **0.550** | 0.924 |

The correction bites hardest in the **1980s** (−0.34), which is exactly where the bill
was highest — the era whose risk-free rate the old panel was silently pocketing. The
2000s, the decade the whole result was already known to rest on, barely moves.

---

## 6. What this does NOT establish

1. **The commodity roll is still uncorrected** — 21.2% of live cells. Free back-adjusted
   futures history does not exist. Direction of the error is known (the panel overstates)
   and its size is bracketed by the ETF gaps already recorded in the survivor document.
   **So 0.7834 is itself still an upper bound**, for the same reason 0.8206 was.
2. **USDX is uncorrected.** The basket's interest differential is not modelled.
3. **The EUR and GBP FX residual (~0.9%/yr) is unexplained** and its registered
   explanation was refuted. §1.
4. **37% of the equity block is bracketed, not measured**, and N225 — the largest single
   passive contributor — is bracketed for its entire life.
5. **Nothing here re-validates any sleeve.** Trend still loses to its own universe;
   passive is still buy-and-hold. A widened premium between two individually unvalidated
   legs remains a statement about their correlation.
6. **No new candidate was selected, so no trial was added to the ledger.** This is a
   data-construction repair, not a search.
7. The pre-registration's own honest prior — *"if it lands above 0.8206, the first
   hypothesis is a bug in the repair"* — did not have to be invoked. It landed below.

---

## 7. What this unblocks

The panel is now internally consistent: **every instrument that can be a USD-funded
excess return is one, and every one that cannot is labelled.** That was the stated
precondition for the trend+passive pre-registration, which is the next document and is
written on the repaired panel — forecasting the strategy **and** the benchmark (rule 13),
gating on DSR and `benchmark_relative_rule`, deflating at the true ledger count plus the
234-configuration search, and solving leverage against a bootstrapped drawdown at
bill+150bp and bill+300bp.

**The number that pre-registration now has to beat is 0.7834, not 0.9033** — and the
comparator it has to beat is corrected passive alone, not the old one.
