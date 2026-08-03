# Long-history multi-asset panel — data integrity report

**Built** 2026-07-28 (UTC) by `scripts/build_multiasset_panel.py` (`--use-cache` reruns are
byte-stable). **Source** yfinance, free tier, 31 tickers, 0 fetch failures.
**Raw cache** `_data/multiasset/` — gitignored. Yahoo's terms forbid redistributing its
data, so no rows are committed; everything in this file is a derived statistic.

**No strategy was built.** This document is the receipt for the panel, nothing more.

---

## 1. Headline — and the number that matters more

| | |
|---|---|
| Instruments fetched | 31 (27 tradable + 1 cash + 3 validation-only) |
| Span | **1927-12-30 → 2026-07-28 = 98.58 years** |
| Daily panel | 25,305 rows × 27 columns, 255,278 populated cells |
| Month-end panel | 1,183 rows × 27 columns (1927-12-31 → 2026-06-30), 12,160 populated cells |
| Returns above ±50% in the shipped panel | **0** |
| Returns above ±50% a naive build would have produced | **241**, worst print **+1,214%** |
| `inf` values / all-NaN columns | 0 / none |

**The 98.58-year headline is one instrument.** Only `^GSPC` reaches 1927. Breadth and
sample length trade off directly, and this is the single most important fact for anything
built on this panel:

| Instruments required | Available from | Years | DSR bar (annual Sharpe, n_trials=32) | Half-Kelly return at that bar |
|---:|---:|---:|---:|---:|
| 1 | 1928 | 98.6 | 0.378 | 5.4% |
| 2 | 1962 | 64.6 | 0.468 | 8.2% |
| 4 | 1965 | 61.6 | 0.480 | 8.6% |
| 8 | 1984 | 42.6 | 0.578 | 12.5% |
| 12 | 1993 | 33.6 | 0.653 | 16.0% |
| 16 | 2001 | 25.6 | 0.750 | 21.1% |
| 20 | 2002 | 24.6 | 0.766 | 22.0% |
| 27 | 2006 | 20.6 | 0.839 | 26.4% |

Instruments live by year: 1930:1, 1950:1, 1965:4, 1972:6, 1980:7, 1990:10, 2000:14,
2004:25, 2008:27, 2025:27.

The DSR bar is computed by `research.multiasset.panel.dsr_sharpe_bar`, which **reproduces
both recorded anchors exactly** — 1.488 at 7 years and 0.597 at 40 years at n_trials=32 —
and is independently cross-checked against the repo's own
`research.validation.deflated_sharpe_ratio` (0.950–0.954 at the solved Sharpe). That
reproduction is what pins the convention: the anchors are **monthly** returns.

**Consequence, stated plainly.** 30%/yr at half Kelly needs Sharpe **0.894**
(g = 3S²/8). That is *above the DSR bar at every row in the table*. So the motivating
finding holds — a longer history really does lower the bar — but it holds so well that
**the DSR bar has stopped being the binding constraint.** Anything that genuinely earns
30%/yr here will clear DSR at n_trials=32. The remaining question is only whether the
Sharpe exists. Buying more history no longer buys anything; buying more history at the
cost of breadth now *hurts*.

Caveat on the bar: `dsr_sharpe_bar` assumes Gaussian returns. Real returns are skewed and
fat-tailed, which **raises** the bar. Treat every figure above as a floor.

---

## 2. What was obtained, per instrument

Sorted by first date. `% missing` is against business days inside the instrument's own
span, so 3–6% is the normal exchange-holiday load and is not a defect.

| key | ticker | first | last | n obs | years | % missing | max gap (d) | gaps >5d |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SPX | ^GSPC | 1927-12-30 | 2026-07-27 | 24,757 | 98.57 | 3.73 | 12 | 2 |
| US_CASH_13W | ^IRX | 1960-01-04 | 2026-07-27 | 16,624 | 66.56 | 4.27 | 7 | 1 |
| US5Y_TR | ^FVX | 1962-01-02 | 2026-07-27 | 16,127 | 64.56 | 4.26 | 7 | 1 |
| US10Y_TR | ^TNX | 1962-01-02 | 2026-07-27 | 16,127 | 64.56 | 4.26 | 7 | 1 |
| N225 | ^N225 | 1965-01-05 | 2026-07-28 | 15,133 | 61.56 | 5.78 | 11 | 63 |
| USDX | DX-Y.NYB | 1971-01-04 | 2026-07-27 | 14,107 | 55.56 | 2.68 | 7 | 1 |
| NASDAQ | ^IXIC | 1971-02-05 | 2026-07-27 | 13,983 | 55.47 | 3.38 | 7 | 1 |
| US30Y_TR | ^TYX | 1977-02-15 | 2026-07-27 | 12,385 | 49.44 | 3.99 | 7 | 1 |
| FTSE100 | ^FTSE | 1984-01-03 | 2026-07-27 | 10,751 | 42.56 | 3.19 | 5 | 0 |
| HSI | ^HSI | 1986-12-31 | 2026-07-27 | 9,762 | 39.57 | 5.44 | 7 | 23 |
| DAX | ^GDAXI | 1987-12-30 | 2026-07-27 | 9,752 | 38.57 | 3.10 | 6 | 10 |
| DJIA | ^DJI | 1992-01-02 | 2026-07-27 | 8,701 | 34.57 | 3.52 | 7 | 1 |
| ASX200 | ^AXJO | 1992-11-23 | 2026-07-28 | 8,512 | 33.68 | 3.13 | 7 | 2 |
| SPY | SPY | 1993-01-29 | 2026-07-27 | 8,429 | 33.49 | 3.53 | 7 | 1 |
| JPYUSD | JPY=X | 1996-10-30 | 2026-07-28 | 7,709 | 29.74 | 0.66 | 18 | 2 |
| WTI_F | CL=F | 2000-08-23 | 2026-07-27 | 6,508 | 25.92 | 3.78 | 5 | 0 |
| COPPER_F | HG=F | 2000-08-30 | 2026-07-27 | 6,504 | 25.91 | 3.77 | 5 | 0 |
| SILVER_F | SI=F | 2000-08-30 | 2026-07-27 | 6,501 | 25.91 | 3.82 | 5 | 0 |
| GOLD_F | GC=F | 2000-08-30 | 2026-07-27 | 6,499 | 25.91 | 3.85 | 5 | 0 |
| NATGAS_F | NG=F | 2000-08-30 | 2026-07-27 | 6,505 | 25.91 | 3.76 | 5 | 0 |
| EFA | EFA | 2001-08-27 | 2026-07-27 | 6,263 | 24.91 | 3.66 | 7 | 1 |
| TLT | TLT | 2002-07-30 | 2026-07-27 | 6,036 | 23.99 | 3.58 | 5 | 0 |
| IEF | IEF | 2002-07-30 | 2026-07-27 | 6,035 | 23.99 | 3.59 | 5 | 0 |
| EEM | EEM | 2003-04-14 | 2026-07-27 | 5,858 | 23.29 | 3.59 | 5 | 0 |
| EURUSD | EURUSD=X | 2003-12-01 | 2026-07-28 | 5,873 | 22.66 | 0.66 | 18 | 2 |
| GBPUSD | GBPUSD=X | 2003-12-01 | 2026-07-28 | 5,890 | 22.66 | 0.37 | 6 | 1 |
| GLD | GLD | 2004-11-18 | 2026-07-27 | 5,454 | 21.69 | 3.61 | 5 | 0 |
| DBC | DBC | 2006-02-06 | 2026-07-27 | 5,148 | 20.47 | 3.61 | 5 | 0 |
| *SLV* | SLV | 2006-04-28 | 2026-07-27 | 5,092 | 20.25 | 3.60 | 5 | 0 |
| *IEI* | IEI | 2007-01-11 | 2026-07-27 | 4,914 | 19.54 | 3.61 | 5 | 0 |
| *BIL* | BIL | 2007-05-30 | 2026-07-27 | 4,820 | 19.16 | 3.58 | 5 | 0 |

*Italic* = validation-only, excluded from the tradable panel and asserted disjoint from it
by the test suite.

**Disappointments, stated honestly.** `^DJI` only reaches 1992 on Yahoo despite the index
existing since 1896, so it adds almost nothing over `^GSPC`. All five futures start
2000-08 — Yahoo's `=F` continuous series simply do not go back further, so the commodity
sleeve cannot be tested on more than 25.9 years. There are **no gaps over 30 days** and
only two over 15 (both FX, in 2008), so coverage inside each span is solid.

---

## 3. Rates are yields, and they are converted, not `pct_change`d

`^TNX`, `^TYX`, `^FVX` are constant-maturity **yields in percent**; `^IRX` is a 13-week
bill **discount rate**. A naive return on any of them is not merely inaccurate — it is
**sign-inverted** against the bond, and it is the largest single source of garbage in this
dataset (§5).

**Conversion (`par_bond_total_return`).** At `t-1` buy a par bond of the stated maturity,
so its coupon is `y_{t-1}` and its price is exactly 100. One bar later, `dt` calendar days
have passed (ACT/365, so a Friday→Monday bar accrues three days of carry, not one);
reprice the same cash flows at `y_t`:

```
v = 1/(1 + y_t/m);  f = 1 - dt*m;  n = maturity*m        (m = 2, semi-annual)
dirty = v^f * [ (100*y_{t-1}/m) * (1 - v^n)/(1 - v) + 100 * v^(n-1) ]
return = dirty/100 - 1
```

Exact given the constant-maturity par-bond convention. Set `y_t = y_{t-1}` and it collapses
to pure carry — asserted in the tests, along with the sign of the capital leg, a
modified-duration cross-check, and the three-day weekend accrual.

**`^IRX` → cash.** Bank-discount basis converted to bond-equivalent yield
(`BEY = 365d/(360 - 91d)`), accrued ACT/365 on the **previous** bar's rate (you earn the
rate you bought at, which is also what makes it point-in-time safe). Skipping the
discount→BEY step understates cash by a few bps at modern rates and tens of bps at 1980s
rates.

**These conversions are validated against independent instruments, not asserted:**

| constructed | benchmark | daily corr | CAGR gap (bench − constructed) | vol ratio | overlap |
|---|---|---:|---:|---:|---:|
| US10Y_TR | IEF (7-10y) | 0.947 | +0.55%/yr | 1.14 | 6,028 |
| US30Y_TR | TLT (20y+) | 0.940 | +0.09%/yr | 1.09 | 6,028 |
| US5Y_TR | IEI (3-7y) | 0.940 | +0.51%/yr | 1.07 | 4,911 |
| US_CASH_13W | BIL (1-3mo) | 0.159 | −0.13%/yr | 0.30 | 4,816 |

Correlations of 0.94–0.95 against real, tradable bond funds are strong evidence the
conversion is right. The vol ratios above 1 are expected and consistent: an exact 10y par
bond is longer-duration than a 7–10y ladder.

**Two disclosed biases fall straight out of this table.**
1. The par-bond proxy **omits roll-down**. It resets to par every bar, so it never captures
   a bond ageing down an upward-sloping curve. Measured cost: **+0.51%/yr at 5y and
   +0.55%/yr at 10y in the ETF's favour, but only +0.09%/yr at 30y** — exactly the pattern
   an upward-sloping-at-the-short-end curve predicts. **The bond series therefore
   understate bond total return by roughly half a percent a year at 5y and 10y.** Any carry
   sleeve must not read that as a signal.
2. **The cash validation looks like a failure and is not.** A daily correlation of 0.159
   between a smooth deterministic accrual and a noisy ETF NAV is meaningless — hence the
   vol ratio of 0.30, which is the NAV noise the accrual correctly does not have. The right
   test is cumulative: over 19.2 years the accrual compounds to **+32.92%** against BIL's
   **+29.81%**, a gap of **0.125%/yr**, which is the order of a T-bill ETF's expense ratio.
   The level is right.

---

## 4. Every other return convention, and what it silently excludes

| Group | Convention | What is NOT in the return |
|---|---|---|
| SPX, DJIA, NASDAQ, FTSE100, N225, HSI, ASX200 | price index | **dividends** — measured at **1.95%/yr** for SPX vs SPY (see below); UK/AU yields are higher |
| DAX | **total-return index** | nothing — DAX is the Performance-Index and **is not comparable to the other seven** |
| SPY, TLT, GLD, DBC, EFA, EEM, IEF | `auto_adjust=True` total return | nothing material |
| GOLD_F, WTI_F, SILVER_F, COPPER_F, NATGAS_F | front-month continuous | **roll is not back-adjusted** (§6) |
| EURUSD, GBPUSD, USDX | spot | **the interest differential** — i.e. FX carry itself |
| JPYUSD | spot, **inverted** (`JPY=X` is JPY per USD) | the interest differential |

The dividend exclusion is measured, not assumed: **SPX vs SPY correlate 0.985 with a CAGR
gap of 1.95%/yr over 8,427 overlapping days**, which is the S&P dividend yield. That is a
direct quantitative confirmation that the price-index convention is what the panel says it
is — and a warning that the seven price indices carry a ~2–4%/yr downward drift bias that
is *not* alpha and must never be benchmarked against a total-return series.

**FX spot returns exclude carry.** A "carry" sleeve cannot be built from these price series
alone; the rate differential has to come from the yield panel or the instrument is not
usable for that purpose.

---

## 5. Anomaly hunt: 0 in the shipped panel, 241 in the naive one

Every bar with |return| > 50% was flagged and inspected.

**Constructed panel: 0.** **Naive `pct_change`-on-everything build: 241, worst +1,214%.**

| key | count | why |
|---|---:|---|
| US_CASH_13W (`^IRX`) | 239 | **treating a yield as a price** — a bill rate moving 0.07 → 0.92 is a +1,214% "return" |
| WTI_F (`CL=F`) | 2 | **negative price**, 2020-04-20 (−306%) and 2020-04-21 (−127%) |

This is the same failure class as the +9,900% print that was worth 13% of a prior study's
P&L. It is worth being explicit that **239 of the 241 come from four rate tickers the task
brief flagged in advance**, and the remaining two from a ratio through a negative price.

**Dispositions in the shipped panel:**

- **Non-positive prices.** WTI front-month settled negative on 2020-04-20. A ratio needs
  two positive numbers, so both straddling bars are **nulled, not zeroed and not dropped**
  (`n_returns_nulled_nonpositive = 2`). `^IRX` prints ≤ 0 on 7 bars; the cash accrual
  handles them without a ratio, so nothing is lost.
- **Long gaps.** Bars spanning >15 calendar days are nulled (2 bars, both FX, 2008 and
  2020). A bar labelled "one day" that contains three weeks corrupts any daily volatility
  estimate.
- **Everything else was kept.** The largest surviving moves were checked individually and
  are real market history: SPX −20.47% (1987-10-19); SPX +16.61% (1933-03-15, after the
  12-day Bank Holiday closure — the gap is genuine, not missing data); HSI −33.33%
  (1987-10-26, on reopening after a 7-day closure); HSI −21.75% (1989-06-05); N225 −14.90%
  (1987-10-20); the October 2008 and March 2020 clusters; COPPER_F −22.25% (2025-07-31).

### 5b. Eight corrupt FX closes, found by a calendar test

`SILVER_F` −31.35% on 2026-01-30 looked like a data error. It is not: **SLV fell 28.54% the
same day.** Measurement overruled the suspicion, and that is the standard applied to
everything in this section.

The FX spikes did not survive it. **9 of EURUSD's 10 largest daily moves and 7 of JPYUSD's
10 land on the 8th or 9th of a month, against a 6.6% base rate, every one of them in 2008.**
A market event has no opinion about the calendar; a vendor defect does. Corroborated
independently: on 2008-12-08 the dollar allegedly fell **17.31%** against EUR while rising
**17.7%** against JPY, with GBPUSD **+1.16%** and the dollar index **−1.79%** — mutually
contradictory, so at least two of those prints are corrupt, and they are exactly the two
with the calendar signature. Each also round-trips the next bar to within 2%.

**A generic screen was built, measured, and rejected.** "Large move that reverses next day"
removes Black Tuesday 1929, the 2020-03-12 crash, the 2024-08-05 yen-carry unwind and the
2008-10-10 FTSE bottom, because **real V-shaped crashes reverse too**. It is not usable.

What ships instead is an explicit, auditable list of **8 individually evidenced corrupt
closes** (`research/multiasset/instruments.py::QUARANTINE`) admitted by a uniform criterion:
8th/9th of a month in 2008, |return| > 5%, and dropping the close leaves a two-day return
under 2.5%. JPYUSD 2008-05-08 (−3.2%) fails the second test and is **deliberately kept**.

The **level** is dropped rather than the two returns nulled, so the genuine move across the
bad print survives as one valid two-day bar. Effect, measured before and after:

| | EURUSD | JPYUSD |
|---|---|---|
| largest daily move before | +17.31% | +18.35% |
| largest daily move after | +3.46% | +7.14% |
| top-10 moves sharing a day-of-month, before | 5/10 (lift 15.1×) | 4/10 (lift 12.1×) |
| top-10 moves sharing a day-of-month, after | 2/10 | 1/10 |

2/10 is the noise floor of this scan — GBPUSD and USDX, which were never affected, also sit
at 2/10.

**The unscreened panel ships alongside** (`returns_daily_unscreened.parquet`,
`returns_monthly_unscreened.parquet`) so any result can be tested against the cleaning
decision rather than having to trust it.

**The scan's two remaining flags are reported, not acted on.** `US10Y_TR` (day 19, 3/10) and
`NATGAS_F` (day 29, 3/10) still flag. US10Y_TR's three day-19 bars are 1980-02-19,
1980-12-19 and 2008-09-19 — genuine Volcker-era and Lehman-week volatility, no round trip,
no cross-instrument contradiction. With ~31 days × 31 instruments scanned, roughly 3 false
positives at 3/10 are expected; two were found. **Verdict: multiple-testing artefact, left
alone.** NATGAS_F is a different story (§6).

---

## 6. Two structural defects that no cleaning can remove

**a) Natural gas front-month is roll-contaminated.** Yahoo's `=F` series splice contracts at
expiry *without* back-adjusting, so the roll spread appears as a price move. Tested by
asking whether extreme bars know what day of the month it is:

| | bars |r|>15% | in days 24–31 | base rate | lift |
|---|---:|---:|---:|---:|
| NATGAS_F | 35 | **65.7%** | 24.0% | **2.74×** |
| WTI_F | 14 | 28.6% | 24.1% | 1.19× |

NG contracts expire ~3 business days before the delivery month, which is precisely that
window. **`NATGAS_F`'s 16.54%/yr headline return and 61.30% volatility are substantially
manufactured by the splice and should not be traded as a price series.** WTI, gold, silver
and copper show no such clustering — their extremes are the 2020 COVID/price-war moves,
which are real. Gold and silver futures also track their physical ETFs to within
**−0.40%/yr and −0.31%/yr** of CAGR, so front-month roll drag is not material there.

**b) Futures settle about an hour after the US equity close.** Lead-lag correlations:

| pair | lag −1 | lag 0 | lag +1 |
|---|---:|---:|---:|
| GOLD_F ~ GLD | −0.024 | 0.888 | **+0.115** |
| SILVER_F ~ SLV | −0.017 | 0.895 | **+0.090** |

Today's futures bar predicts *tomorrow's* ETF bar, which is the signature of a futures
session closing **after** the 16:00 ET ETF close. Consequence: **a daily-frequency signal
computed on futures and executed in ETFs (or index series) would use roughly an hour of
information the ETF close does not yet contain.** That is a genuine lookahead at daily
frequency. It is negligible at monthly frequency. **Prefer the month-end panel for any
cross-asset sleeve mixing futures with indices or ETFs**, or lag the futures signal a full
bar.

---

## 7. Survivorship

**yfinance returns only tickers that still exist today, and no delisted instrument is
present.** Stated directly because it cannot be fixed from this source.

**Residual bias direction: upward, but small — and this is exactly why the asset class is
usable where the equity cross-section was not.**

- **Index series (8 instruments) are close to immune.** An index survives constituent
  death by construction; `^GSPC` continues through every bankruptcy in it. The residual
  bias is *index-level* survivorship — an index that was discontinued would be absent. None
  of these eight were discontinued, and all are majors, so the bias is near zero. Any
  remaining effect is **constituent-selection drift inside the index**, which is the
  index's methodology, not our sampling.
- **Futures (5) are effectively immune.** A contract expires by design and is replaced;
  there is no "failure" mode that removes a commodity from the series. The real defect here
  is roll, not survivorship (§6).
- **FX (4) are immune over this window.** All four pairs still trade. A currency *can*
  disappear — the pre-1999 legacy currencies are the obvious case — but the panel's FX
  history starts in 1996/2003, after that redenomination.
- **Rates (4, incl. cash) are immune.** Constant-maturity yield curves are published series,
  not instruments that can fail.
- **The 7 ETFs carry real, though modest, survivorship bias.** Failed ETFs are liquidated
  and delisted, and none appear here. All seven are large, long-lived funds, but a
  hypothetical strategy that selected from "ETFs available in 2003" would have had a wider,
  worse menu than the one this panel offers.

**Net:** the bias is confined to 7 of 27 instruments, is small for those, and is orders of
magnitude below the single-name equity case where dead companies dominate the tail. This
does **not** license ignoring it: the ETF-era results (2006+, all 27 instruments) are the
most survivorship-flattered slice of the panel and also the shortest.

---

## 8. Integrity checks that passed

1. **Chronological order.** Every series sorted and deduped before any consecutive-bar
   calculation (`clean_levels`); indices asserted strictly increasing and unique on both
   panels. All 31 raw series arrived already sorted; **0 duplicate dates, 0 non-finite
   values dropped** — so this was a guard rather than a repair, but it is now unconditional.
2. **Month-end panel reconciled independently.** All **12,160** monthly cells recomputed
   from daily returns by a separate code path; **max absolute discrepancy 0.0**. This guards
   the groupby/period boundary, where an off-by-one would shift every monthly return by a
   bar and look completely normal downstream.
3. **No `inf`, no all-NaN columns** in any shipped panel.
4. **Month-end alignment.** Months are stamped at calendar month-end so instruments on
   different exchange calendars line up. A month with fewer than 5 observations for an
   instrument is NaN, so a stub is never presented as a month; the trailing partial month is
   dropped (hence the panel ends 2026-06-30, not 2026-07).
5. **38 unit tests** (`tests/test_multiasset_panel.py`), offline, no network. The bond
   conversion is pinned against closed-form identities that hold independently of the
   implementation — unchanged-yield ⇒ pure carry, yield rise ⇒ negative return monotone in
   maturity, capital leg ≈ −D_mod·Δy, three-day weekend accrual, the zero-yield limit — plus
   the two DSR anchors and a cross-check against `research.validation`.

---

## 9. Files

All under `_data/multiasset/` (gitignored).

| file | contents |
|---|---|
| `returns_daily.parquet` | **primary** — 25,305 × 27 tradable daily returns, quarantine applied |
| `returns_monthly.parquet` | **primary** — 1,183 × 27 month-end compounded returns |
| `returns_daily_unscreened.parquet`, `returns_monthly_unscreened.parquet` | identical but retaining the 8 quarantined closes, for sensitivity |
| `returns_all_daily.parquet`, `returns_all_monthly.parquet` | as above plus cash and the 3 validation instruments |
| `cash_daily.parquet`, `cash_monthly.parquet` | `^IRX` risk-free accrual, for excess returns |
| `yields_daily.parquet`, `yields_monthly.parquet` | rate **levels in decimal** (`US5Y_YLD`, `US10Y_YLD`, `US30Y_YLD`, `US13W_YLD`) — the carry-signal input; term spread = `US10Y_YLD − US13W_YLD` |
| `levels_daily.parquet`, `levels_monthly.parquet` | raw levels, native units |
| `instruments.csv` | key → ticker, asset class, currency, return method, role, notes |
| `coverage.csv`, `summary_stats.csv` | §2 and per-instrument annualised statistics |
| `extreme_returns.csv`, `extreme_returns_naive.csv` | §5, shipped vs naive build |
| `quarantine_audit.csv` | §5b, with a `matched` flag per entry |
| `integrity.json` | every number in this document, machine-readable |

Code: `research/multiasset/{instruments,panel}.py`, `scripts/build_multiasset_panel.py`,
`tests/test_multiasset_panel.py`.

---

## 10. Read this before building anything on the panel

1. **Use the month-end panel for cross-asset work.** The daily panel has a ~1-hour
   futures/equity session overlap (§6b) that is a real lookahead at daily frequency.
2. **Never benchmark a price index against a total-return series.** SPX vs SPY is 1.95%/yr
   of pure convention. DAX is total-return while the other seven equity indices are not.
3. **`NATGAS_F` is not a clean price series** (§6a). Exclude it or accept that a large part
   of its return is a splice artefact.
4. **The bond series understate total return by ~0.5%/yr at 5y and 10y** (no roll-down, §3).
5. **FX spot excludes the interest differential**, so these four series cannot express FX
   carry on their own (§4).
6. **Equity index returns are in local currency** (GBP, JPY, EUR, HKD, AUD). A USD investor's
   return differs by the FX move; the panel does not convert.
7. **Decide breadth vs length deliberately** (§1). 42.6 years at 8 instruments or 20.6 years
   at 27 — not both. Since the DSR bar is no longer binding at any of these lengths, **the
   argument now favours breadth over length**, which reverses the prior on entering this
   iteration.
8. **Report Sharpe per decade.** The panel spans regimes from 1928; a full-sample number
   carried by one era is not deployable.
