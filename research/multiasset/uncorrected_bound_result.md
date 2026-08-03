# RESULT — 0.7834 is an upper bound by between 0.009 and 0.063 Sharpe

**Pre-registration:** `uncorrected_bound_prereg.md`, written before any charged panel or
re-run book existed.

**The answer is a range, not a number, and the width is the finding.** How much 0.7834
overstates depends almost entirely on how far you trust one reference — `USO` for WTI —
and the honest reporting is to say so rather than to pick the reading you prefer.

Both integrity gates passed. **B1**: the uncharged central panel reproduces the committed
book Sharpe at **0.783398 vs 0.7834**. **B2**: every charge lowers the book, so no sign error.

**This promotes nothing, re-opens no gate and re-selects no sleeve.** The trend+passive
book is fixed throughout; only its inputs are charged.

---

## 1. The headline

| bound | book Sharpe | ΔSharpe | lev @ DD50 | CAGR% | after ×0.877 |
|---|---:|---:|---:|---:|---:|
| central, uncharged | **0.7834** | — | 1.95 | 16.33 | **14.32** |
| `roll_free_only` *(robust core)* | 0.7744 | **0.0090** | 1.95 | 16.12 | **14.14** |
| `overlap_only` | 0.7524 | 0.0310 | 1.95 | 15.67 | 13.74 |
| `full_sample` *(registered headline)* | 0.7441 | 0.0393 | 1.95 | 15.50 | 13.59 |
| `full_sample_upper` | 0.7204 | **0.0630** | 1.90 | 14.81 | **12.99** |

**The registered three-way bracket agrees**: `overlap_only` → `full_sample_upper` spans
**0.0320**, inside the 0.05 limit. Adding the post-hoc roll-free cut widens the total
spread to **0.0540**, which trips the registered disagreement condition — so the result is
reported as a range, exactly as §4 required.

**Survivable-drawdown return falls from 14.32%/yr to between 14.14% and 12.99%.**

---

## 2. Why the range is wide: one reference does most of the work

Charging each instrument alone, against the USDX-corrected base of 0.786372:

| instrument | reference | roll-free? | gap %/yr | corr | ΔSharpe |
|---|---|---|---:|---:|---:|
| `GOLD_F` | GLD | **yes** | +2.363 | 0.992 | 0.0085 |
| `SILVER_F` | SLV | **yes** | +2.340 | 0.994 | 0.0034 |
| **`WTI_F`** | **USO** | no | **+14.015** | **0.766** | **0.0271** |
| `COPPER_F` | CPER | no | +2.598 | 0.984 | 0.0028 |

**WTI alone is roughly 69% of the full charge**, and it is the least defensible component:

* `USO` is itself a rolled product, and its front-month roll into persistent contango is a
  well-known drag on the fund rather than a property of WTI. Charging `WTI_F` 14%/yr
  partly bills the panel for **USO's construction**, not the panel's splice artefact.
* Its correlation with the panel series is **0.766**, against 0.98–0.99 for the other
  three. The two series are not tracking the same thing closely enough for a mean gap to
  be a clean charge.
* Its 95% upper bound is **+26.3%/yr** — an interval that wide is not a measurement.

The prereg fixed this ordering in advance (§2: *"GLD and SLV are the load-bearing pair…
USO and CPER are the weaker two"*), which is why the roll-free cut is a legitimate reading
rather than a convenient one. It is nonetheless **post-hoc as a bound** and labelled so.

### The materiality verdict flips on that choice

B5 registered a null threshold at 0.01 Sharpe. The harshest bound (0.0630) is **material**;
the robust core (**0.0090**) is **below the threshold**. So:

> **Using only references that cannot be blamed on their own roll, the uncorrected 21.2%
> does not materially move the headline.** Using all four, it moves it by ~0.04, and at the
> harshest defensible reading by ~0.06.

Both statements are true and neither should be quoted alone.

---

## 3. A correction to the earlier document

`convention_repair_prereg.md` §5 quoted `GLD − GOLD_F` = **−0.576%/yr** and
`SLV − SILVER_F` = **−0.670%/yr**. Those are in **total-return** terms while the panel
treats its commodity series as **excess** returns, so they understate the real gap by the
risk-free rate:

    0.576 + 1.786 (mean cash over the 260-month overlap) = 2.362  vs  measured 2.363

The reconciliation is exact to a thousandth of a point. **The previously quoted commodity
gaps were understated by the risk-free rate** — not wrong arithmetic, a convention
mismatch, and the same one the whole convention-repair programme exists to fix. The figures
in §2 above are the like-for-like ones.

---

## 4. USDX: a registered step that had been skipped is now done

The original convention-repair prereg §1 registered
`corrected = spot − (i_basket − i_US)_{t−1}/12` at constant published DXY weights. The
implementation **short-circuits USDX** (`run_convention_repair.py` line 285) and stamps it
`UNCORRECTED` (line 413). The result doc disclosed the outcome but not that a registered
step had been left undone.

It is now applied at the published weights (EUR 57.6, JPY 13.6, GBP 11.9, CAD 9.1,
SEK 4.2, CHF 3.6): **285 of 666 live months corrected (42.8%)**, mean charge
**−0.610%/yr**, standalone effect on the book **−0.0030 Sharpe**. Months lacking any
basket rate are left untouched and still counted uncorrected, so this does not silently
extend coverage it has not earned.

---

## 5. Honest limits

1. **Every reference starts 2004–2011; the panel starts in the 1990s.** Applying a
   post-2004 gap to the whole sample is an assumption, which is precisely what
   `overlap_only` (charges nothing before the reference exists, Δ0.0310) versus
   `full_sample` (Δ0.0393) brackets. The truth is not necessarily inside that range.
2. **A charge is not a correction.** Nothing about the roll has been reconstructed. The
   charged panels must never be described as corrected, and the commodity block remains
   **100% uncorrected** in the provenance frame.
3. `CPER` starts only 2011-11 (n=176), so copper's gap rests on the shortest overlap.
4. The roll-free cut is post-hoc as a *bound*, though the roll-free/rolled ordering it
   relies on was registered in advance.
5. The three registered bounds agree at 0.0320; the 0.0540 spread that trips the
   disagreement rule only appears once the post-hoc cut is included. Both are reported so
   the distinction cannot be lost.

---

## 6. What to quote from now on

> The corrected book Sharpe is **0.7834**. It remains an **upper bound**: charging the
> still-uncorrected commodity roll against tradable references lowers it by **0.009**
> (roll-free references only) to **0.063** (harshest bound), i.e. to between **0.774** and
> **0.720**, with survivable-drawdown return falling from **14.32%/yr** to between
> **14.14%** and **12.99%**. Most of the wider figure comes from a single weak reference
> (USO for WTI) and should not be quoted without that caveat.

The open thread is now **quantified rather than closed**. Closing it properly still needs
a back-adjusted continuous futures history, which does not exist in free data — that has
not changed, and no amount of ETF-referencing substitutes for it.
