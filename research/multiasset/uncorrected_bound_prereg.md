# PRE-REGISTRATION — putting a number on the uncorrected 21.2%

**Written 2026-07-31, BEFORE any charged panel or re-run book existed.** The quantities
known at writing time are stated in §1 so this cannot later be read as having predicted
them.

Governing instruction: `convention_repair_result.md` §7 item 1 — *"The commodity roll is
still uncorrected — 21.2% of live cells"* — and §5 item 5 of its prereg, *"the corrected
book is itself still an upper bound."* **That statement is qualitative. This makes it a
number.**

The question, in one line: **by how much is 0.7834 an upper bound?**

---

## 0. What the 21.2% actually is

From the committed `convention_repair.json`:

| block | live cells | uncorrected |
|---|---:|---:|
| equity | 4,435 | 0.0% |
| rates | 2,141 | 0.0% (already excess) |
| **commodity** | **1,241** | **100.0%** |
| **fx** | 1,562 | **47.6%** — this is USDX |

1,241 + ~743 = ~1,984 of 9,379 live cells ⇒ the 21.2%. Two components, and they are **not
the same kind of problem**:

* **Commodity roll** — `GOLD_F`, `WTI_F`, `SILVER_F`, `COPPER_F` are front-month continuous
  series spliced without back-adjustment. Free back-adjusted history does not exist, so
  this can be **bounded, not corrected**.
* **USDX** — the original prereg §1 registered a correction for it
  (`spot − (i_basket − i_US)_{t−1}/12`, basket-weighted) and the implementation **skipped
  it**: `run_convention_repair.py` short-circuits `USDX` at line 285 and stamps it
  `UNCORRECTED` at line 413. The result doc disclosed the outcome but not that it was a
  registered step left undone. **This is correctable with data already on disk** — the
  repo holds EZ, JP, GB, CA, SE and CH short rates, which is the entire DXY basket.

---

## 1. What is already known — disclosed, not claimed as prediction

* Corrected book Sharpe **0.7834**, bracket **0.7499 / 0.8464**; survivable return
  ≈**14.3%/yr at DD≤50**.
* Uncorrected commodity per-instrument stats, identical across all three bracket bounds
  because nothing is applied: `GOLD_F` 11.755%/yr Sharpe 0.7079; `WTI_F` 13.149%/yr
  Sharpe 0.3227; `SILVER_F` 14.447%/yr Sharpe 0.4622; `COPPER_F` 10.780%/yr Sharpe 0.4257.
* Gaps already quoted in the convention-repair prereg §5, measured against whatever
  reference was to hand: `GLD − GOLD_F` −0.576%/yr, `SLV − SILVER_F` −0.670%/yr,
  `DBC − WTI_F` −8.346%/yr. **DBC is a broad multi-commodity basket and is a poor
  reference for WTI specifically**; replacing it with `USO` is registered below as a
  method improvement, decided now and not after seeing the answer.

---

## 2. References — probed 2026-07-31, chosen for construction not convenience

| panel series | reference | why | coverage probed |
|---|---|---|---|
| `GOLD_F` | **GLD** | physically backed — holds bullion, **no roll at all**, so the gap is the cleanest available | 5,458 rows, 2004-11-18 → 2026-07-31 |
| `SILVER_F` | **SLV** | physically backed, same argument | 5,096 rows, 2006-04-28 → |
| `WTI_F` | **USO** | a WTI-only futures ETF with a published roll, replacing the broad DBC | 5,109 rows, 2006-04-10 → |
| `COPPER_F` | **CPER** | copper-only futures ETF; the only single-metal option | 3,697 rows, 2011-11-15 → |

`GLD` and `SLV` are the load-bearing pair: with no roll to perform, the gap against them
cannot be blamed on a different roll convention. `USO` and `CPER` are themselves rolled,
so their gaps measure *splice artefact versus a defined roll* rather than versus none, and
are the weaker two. That ordering is fixed now.

**The limitation that matters, registered before any result:** every reference starts
2004–2011 while the panel starts in the 1990s. **A gap measured on the overlap and applied
to the whole sample is an assumption, not a measurement**, and it is bracketed in §4 rather
than asserted.

---

## 3. What is computed

**(a) USDX, corrected — not bounded.** `corrected = spot_return − (i_basket − i_US)_{t−1}/12`
with the published DXY weights held constant: **EUR 57.6%, JPY 13.6%, GBP 11.9%,
CAD 9.1%, SEK 4.2%, CHF 3.6%**. A month is corrected only where every basket rate is
present; otherwise it stays uncorrected and is counted as such. This executes the step the
original prereg registered.

**(b) Commodity, charged.** Each series is charged a constant monthly drag equal to its
measured annualised gap / 12. Nothing about the roll is reconstructed; the charge is a
**bound**, and the write-up must never describe the charged panel as "corrected".

**(c) The book, re-run.** `book_from` and the survivor-verification statistics, called
exactly as `run_convention_repair_book.py` calls them, on the charged panels. **No sleeve
is re-selected, re-tuned or re-gated.** The trend+passive book is FIXED; this measures its
sensitivity to a known error, which is not the same thing as re-opening a closed sleeve.

---

## 4. The registered bracket

Three bounds, fixed now, because applying a post-2004 gap to a 1990s sample is an
assumption whose severity is exactly what should be bracketed:

1. **`overlap_only`** — charge each series only in months where its reference exists;
   charge zero before. The most conservative reading, and it charges nothing pre-2004.
2. **`full_sample`** *(headline)* — charge the measured gap over every month the series is
   live.
3. **`full_sample_upper`** — charge the **upper end of the gap's 95% interval** over every
   live month. The harshest defensible reading.

If the three disagree by more than **0.05 Sharpe**, the disagreement is the finding and the
answer is reported as a range rather than a number.

---

## 5. Registered predictions

**B1 — reproduction anchor (GATE).** With zero charge and USDX left uncorrected, the
pipeline must reproduce the committed book Sharpe **0.7834** to within **0.0005**. If it
does not, nothing below may be believed and the run is void.

**B2 — direction (GATE).** Every charge must **lower** the book Sharpe. Every identified
error in this panel has run one way — the panel overstates — and a charge that *raises* the
book means a sign error, not a discovery. Void on failure.

**B3 — the deliverable.** Report `ΔSharpe = 0.7834 − charged` under all three bounds, and
restate the survivable-drawdown return. The output is the sentence *"0.7834 is an upper
bound by at most X"*, with X the harshest bound.

**B4 — USDX materiality.** Report the corrected-cell count and the standalone effect of the
USDX correction on the book, separately from the commodity charge, so the two are never
conflated.

**B5 — the honest negative, registered in advance.** If the total effect is **smaller than
0.01 Sharpe**, the correct conclusion is *"the uncorrected 21.2% does not materially move
the headline"* — a null result, and it will be reported as the headline rather than dressed
up. A large cell count does not entitle this to be important.

### Decision rule

There is no pass/fail here — this is a measurement, not a hypothesis test. B1 and B2 are
integrity gates that can void it. The deliverable is the number and its bracket, whatever
they are.

---

## 6. What this can and cannot change

It **can** change the honesty statement attached to 0.7834, and it can move the *reported*
survivable return. It **cannot** promote anything, re-open the trend+passive gate, or alter
which sleeve was selected — all of that stays closed. No live path, nothing public.

If the charge is material, the correct response is to quote the charged figure alongside
0.7834 as the panel's honest range — **not** to quietly replace one number with another,
and **not** to treat a bounded charge as if it were a measured correction.
