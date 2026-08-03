# PRE-REGISTRATION — repairing the panel's return conventions AT SOURCE

**Written 2026-07-31, BEFORE any corrected series was built and before any corrected
statistic was computed.** Every source, every window, every tolerance and every
pass/fail rule below was fixed in advance. If a corrected number disagrees with a hope,
the number is what gets reported.

Governing instruction: `research/sleeves/survivor_verification.md` §11 — *"Before any
further sleeve work, the panel's return conventions should be repaired at source."*
That document charged a **constant** against the existing panel (§9c) and said so:
*"A convention-corrected panel has never been built … it does not rebuild the
instruments from total-return sources. That is the right next step and it is not done
here."* This is that step.

Standing method rules this document is written to obey: **rule 9** (build the positive
control first, and give it a leg the old model must fail), **rule 10** (bracket, don't
point-estimate, where the decomposition is unstable), **rule 11** (persist dated series
and declare what the index means).

---

## 0. What is wrong, in one paragraph

`multiasset_trend.load_excess_panel` subtracts the 13-week bill from **three** series
(the constant-maturity par-bond total returns) and treats **all fifteen others as
already being excess returns**. They are not. Seven equity instruments are index levels
— six price-only, one (DAX) total-return — so using them as excess returns overstates
by `(risk-free − dividend yield)`, and by the **full** risk-free rate for DAX. Four FX
series are spot only, so the interest differential that *is* the return to a currency
position is missing. Four commodity series are front-month continuous futures spliced
without back-adjustment, so the roll appears as a price move. **Every one of these
errors runs the same way: the panel overstates.**

---

## 1. Scope — what is corrected, what is not, decided now

| block | instruments | treatment | why |
|---|---|---|---|
| **equity** | SPX, NASDAQ, FTSE100, N225, HSI, ASX200 | `corrected = price_return − rf + q` with **q measured**, per instrument, per month | price indices; dividends are a real cash return the panel never credited |
| **equity** | DAX | `corrected = index_return − rf`, **q ≡ 0** | it is the DAX Performance-Index; dividends are already inside it, so it gets no dividend credit and pays the full bill |
| **rates** | US5Y_TR, US10Y_TR, US30Y_TR | **unchanged, byte-identical** | the one block the panel already converts correctly; validated against IEI/IEF/TLT at +0.478 / +0.467 / −0.117 %/yr |
| **fx** | EURUSD, GBPUSD, JPYUSD | `corrected = spot_return + (i_foreign − i_US)_{t−1}/12` | under CIP this is the return to a fully-collateralised long-foreign/short-USD position |
| **fx** | USDX | `corrected = spot_return − (i_basket − i_US)_{t−1}/12`, basket-weighted | DXY is long USD against six currencies; the differential runs the other way |
| **commodity** | GOLD_F, WTI_F, SILVER_F, COPPER_F | **NOT corrected. Bracketed and signposted.** | a back-adjusted continuous series does not exist in free data; §5 |

**Cash convention.** `rf` is the panel's own `US_CASH_13W` accrual, unchanged. Every
corrected series is a **USD-funded excess return**, so the whole panel finally shares
one convention. That is the deliverable.

---

## 2. Sources — confirmed reachable and dated before this document was written

Every source below was probed on 2026-07-31 and returned data; first/last dates are
measured, not assumed.

| source | what it gives | span confirmed |
|---|---|---|
| **Kenneth French data library**, `F-F_Research_Data_Factors` (monthly) | `Mkt-RF + RF` = the CRSP value-weighted US equity **total** return | **1926-07 → 2026-05**, n=1199 |
| `^GSPC` (already in the panel) | the S&P 500 **price** return | 1927-12 → |
| `^SP500TR` | S&P 500 **total-return** index — independent second read | 1988-01 → |
| `SPY` (already cached) | total return, `auto_adjust` | 1993-02 → |
| `QQQ` / `^NDX` | Nasdaq-100 total / price pair | 1999-03 / 1985-10 → |
| `EWU EWJ EWG EWH EWA` | MSCI country **total-return** ETFs (USD) | all **1996-03-18** → |
| `_data/carry/short_rates_monthly.parquet` (already built) | OECD 3-month interbank rates, 10 currencies | **1956-01 → 2026-06** (US 744, GB 829, EZ 385, JP 290 months) |
| `FXE FXB FXY` | currency-deposit ETFs — spot **plus** accrued foreign interest | 2005-12 / 2006-06 / 2007-02 → |
| `GLD SLV USO GSG` | physical / rolled commodity references | 2004-11 / 2006-05 / 2006-04 / 2006-07 → |

**Two of these already exist inside this repo and are reused rather than rewritten:**
`research/multiasset/carry.py::fx_excess_returns` (the exact CIP construction in §1,
already lag-correct and already reviewed) and
`research/multiasset/carry.py::realised_dividend_yield` (the trailing
total-vs-price gross-up). The FX correction is therefore not new machinery — it is
machinery this repo built, validated, and then never wired into the trend panel.

---

## 3. How `q` is measured, per instrument — fixed now

**US (SPX).** `q_US,t` = the trailing 12-month realised gap between the French US total
return and the `^GSPC` price return. This is measured over the **entire 1965–2026
sample**, not a sub-window, and it is the only equity instrument for which that is true.
Cross-checked against `^SP500TR` (1988+) and `SPY` (1993+); all three must agree inside
§4's Control A budget.

**NASDAQ.** `q_NDX,t` from the `QQQ` / `^NDX` pair, 1999+. Before 1999: §3b.

**FTSE100, N225, HSI, ASX200.** `q_c,t` from `ETF_c` (USD total return) minus the local
price index return plus the currency return, 1996+:
`q_c ≈ (ETF_c) − (index_c + fx_c)`. Before 1996: §3b.

> **This estimator is biased DOWN and that is disclosed, not hidden.** The country ETF
> pays a management fee (~0.5%/yr) and suffers dividend withholding (~0.3%/yr), and it
> tracks the MSCI country index rather than the local headline index. So the measured
> gap is `q − fee − withholding ± composition`, i.e. **less** than the true gross
> dividend yield. A smaller `q` means a **larger** charge, which is **conservative**:
> it can only push the corrected book down. §4 Control E puts a number on the bias
> instead of waving at it.

**DAX gets no measurement.** `q ≡ 0` by definition of the index. §4 Control D tests
that definition against data rather than accepting it.

### 3b. Before the measurement window — the bracket (rule 10)

For the five instruments whose ETF pair starts in 1996 (and NASDAQ's in 1999) the
pre-measurement era is an **assumption, not a measurement**, and is bracketed by three
constructions fixed here:

| bound | construction | role |
|---|---|---|
| **conservative** (harshest) | `q = 0` before the pair exists — the instrument pays the full bill | a pass here is REAL |
| **central** | `q_c,t = q̄_c,measured × (q_US,t / q̄_US,measured)` — hold the country's *relative* yield fixed and let the **measured US era-path** carry the time variation | the reported headline |
| **realistic** (kindest) | `q_c,t = (q̄_c,measured + bias) × (q_US,t / q̄_US,measured)`, and `q_measured,t + bias` inside the window | a fail here is DEAD |

`realistic ≥ central ≥ conservative` by construction and this is **asserted elementwise
in code**, not argued: any month where the ordering breaks is a bug and fails the build.

> **AMENDMENT, 2026-07-31, made BEFORE any corrected series was built and before any
> corrected statistic existed.** The realistic bound was first registered as
> "`q̄_c,measured` held flat, grossed up by the bias budget". That definition **can
> cross the central bound and therefore is not a bracket**: before 1996 the US dividend
> yield ran well above its modern mean, so the era ratio `q_US,t / q̄_US` exceeds 1, and
> a flat `q̄_c + bias` can sit *below* `q̄_c × ratio`. All three bounds now share the era
> path; they differ only in whether the pre-window yield is zero (conservative) and
> whether the ETF's fee/withholding/composition bias is added back (realistic). Nothing
> else changed, no result informed this, and the elementwise assertion registered in the
> same paragraph is what the flaw was caught against.

**The single largest unmeasured quantity is N225 before 1996 — 31 years of a 61-year
sample, on the instrument the survivor document names as the largest single passive
contributor.** It is called out here so that no reader has to discover it. Japanese
policy rates sat near zero for much of the *later* sample but were **not** near zero in
1965–1990, so the "N225 exempt" kindness used in the survivor document's §9c bracket is
**not** carried into the pre-1990 era.

---

## 4. THE POSITIVE CONTROLS — built first, each with a leg the old panel must fail

Every threshold below is a **budget derived from known frictions**, written before the
measurement. None was chosen by looking at a result.

**Control A — US equity, the assumption-free one.**
Over 1993-02 → 2026-06 (401 months), the **corrected** `SPX` must match `SPY − bill`:
`|annualised mean gap| ≤ 0.25%/yr` and monthly `corr ≥ 0.98`.
*Budget:* SPY expense ratio 0.0945%/yr + CRSP-VW-vs-S&P-500 dividend-yield difference
(~0.1%/yr, small caps yield less) + tracking ≈ 0.20%/yr, rounded to 0.25%.
**The leg the OLD panel must fail:** the same test on the uncorrected `SPX` — recorded
gap **+0.748%/yr**, which is 3× the budget. *If the old panel passes Control A, the
control is broken and nothing else in this document may be believed.*

**Control B — the block that must NOT move.**
The three rates series in the corrected panel must be **byte-identical** to the old
panel: `max |Δ| = 0` over every cell. *This is the anti-rigging control:* a repair that
"improves" everything is a repair that is measuring its own wishes. Both panels pass.

**Control C — FX, where the omission is one-signed.**
Corrected `JPYUSD` vs `FXY − US bill` over 2007-02 → 2026-06: `|gap| ≤ 0.75%/yr`.
*Budget:* FXY fee 0.40%/yr + post-2008 cross-currency basis 10–50bps.
**The leg the OLD panel must fail:** the same test on spot-only `JPYUSD`. JPY 3-month
rates sat far below USD for essentially the whole window, so the omission is large and
one-signed. Run identically on `GBPUSD`/`FXB` and `EURUSD`/`FXE` and reported, but
those two are **not** the discriminating legs — their differentials change sign inside
the window and may be small on average, which is informative, not a failure.

**Control D — DAX, testing the definition instead of asserting it.**
Measure `ETF_c − (index_c + fx_c)` for all six countries over 1996+. Pre-registered
prediction: **DAX < +0.5%/yr** (it pays no dividend credit; the gap is roughly minus the
fee) while **every price index > +0.8%/yr**. If DAX comes out with the price indices,
the registry's claim that it is a total-return index is wrong and §1 changes.

**Control E — the bias budget, measured not assumed.**
`EWU − (FTSE100 + GBPUSD)` and its four siblings are compared against the same
construction for the **US**, where the true answer is independently known from French.
The US residual `(EWUS-style measured q) − (French-measured q)` **is** the fee +
withholding + composition bias, measured once, and it is the number used to gross up
the "realistic" bound in §3b. Pre-registered expectation: the residual is negative and
between −0.3% and −1.5%/yr.

**Control F — the pipeline must not invent corrections.**
Run the entire correction pipeline over the three rates series. It must return a
correction of exactly zero. A pipeline that finds a dividend yield in a bond total
return is broken.

---

## 5. What is NOT corrected, stated plainly

1. **Commodity roll.** Front-month continuous splicing without back-adjustment remains
   in `GOLD_F`, `WTI_F`, `SILVER_F`, `COPPER_F`. Free back-adjusted history does not
   exist. The measured ETF gaps (`GLD − GOLD_F` −0.576%/yr, `SLV − SILVER_F`
   −0.670%/yr, `DBC − WTI_F` −8.346%/yr) are reported as a **bracket on the residual
   error**, not applied. Direction: the panel overstates.
2. **Pre-1996 non-US dividend yields** (and pre-1999 NASDAQ) are an assumption. §3b.
3. **Cross-currency basis** post-2008, 10–50bps/yr, is inside the CIP assumption.
4. **USDX basket weights** are the published DXY weights held constant; the index's own
   1999 euro re-composition is not modelled.
5. Therefore **the corrected book is itself still an upper bound**, exactly as the
   survivor document said of its own 0.8206.

---

## 6. Acceptance — what makes the repaired panel shippable

All six controls pass, with A and C failing on the old panel and B passing on both;
the three bracket bounds are ordered elementwise; the corrected panel carries a
**per-instrument, per-month provenance frame** recording `MEASURED` / `BRACKETED` /
`EXEMPT` / `UNCORRECTED` for every cell, so no reader has to guess which numbers are
measurements; and `research/multiasset/convention_repair_result.md` reports the
corrected panel's statistics **beside** the old panel's, with the bracket, and with the
fraction of the sample that is measured rather than assumed.

**The honest prior, written down before the run** (survivor §11): the corrected book is
expected to land **below 0.894**, and below the 0.8206 that document's constant-charge
approximation produced, because the FX and commodity errors it signposted run the same
way. If it lands **above** 0.8206, that is a surprise and the first hypothesis is a bug
in this repair, not a discovery.

---

## 7. What this does not do

It does not re-validate any sleeve, does not run a new strategy search, does not touch
`selection_rule` or `benchmark_relative_rule`, does not add a trial to the ledger (no
candidate is being selected — this is a data-construction repair), places no order,
touches no broker path, commits no vendor rows, and publishes nothing. The
trend+passive pre-registration that follows is a **separate** document written on the
repaired panel, not part of this one.
