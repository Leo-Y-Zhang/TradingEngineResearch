# DEFECT FIXES — what was repaired, what was audited clean, and what was left alone

**Scope.** The six defects the overnight run found in our own code (P1..P6). The research
questions were already answered; this pass exists so those defects cannot silently corrupt
future work.

**Governing rule, applied without exception.** Every repair is a SWITCH whose DEFAULT
reproduces the banked number bit-for-bit. Not one banked result was changed. Where a
repair moves a number, the before and after are printed below as measured figures, not as
a restated headline.

**Suite.** 1389 passed / 1 skipped → **1516 passed / 1 skipped** (+127 tests, six new test
modules). `ruff` clean on every touched file. `mypy` is at its exact pre-existing baseline
of **161 errors in 29 files** — measured against a clean `git worktree` of the pre-work
commit — so this work adds none.

**Commits.** `c86ead5` (P1), `1f95a45` (P2), `1ef4cc7` (P4/P5/P6 + all tests). The P3
book-scaler, the P4 low-vol source and the P5 ledger source were swept into `2b869f3` by a
concurrent agent while this work was in flight; their tests are in `1ef4cc7`.

---

## P1 — THE DATING DEFECT

`lowvol_retest.run_band` labelled each monthly slot with the **formation** month and filled
it with `forward_return`, the **following** month's return. Every slot was dated one month
early.

**Why nothing caught it.** Mean, volatility, Sharpe, drawdown, Newey-West t and the
vol-matched active return are all invariant to shifting every observation by a constant.
An independent bit-for-bit re-implementation reproduced the series exactly and could not
see it. It only bites on a cross-series join.

### (a) The labelling

`run_band` gains `date_convention`. `FORMATION` is the **default** and reproduces every
banked number bit-for-bit; `REALISATION` relabels `months` and `pnl_by_name_month` so the
index means the month the return was earned. The return arrays are byte-identical under
both — proven by a test that compares `gross`, `benchmark`, `benchmark_rankable` and all
three cost arrays element-wise.

Kept as a switch rather than flipped because `_decade_sharpes` groups on `books.months`, so
the banked `benchmark_decades` and per-bound `decades` blocks in
`lowvol_retest_result.json` would move for months that cross a decade boundary.

### (b) The probe, promoted

`research/alignment.py` — shared, importable, unit-tested. Lag `k` compares `series(t)`
against `reference(t + k)`, so `suggested_shift_months` is literally the repair. It
relabels by CALENDAR, not by positional `.shift`, so a gap in either series cannot corrupt
the measurement. It reports `UNINFORMATIVE` rather than a pass when the reference has no
power over a market-neutral book.

`portfolio_correlation_v2.alignment_control` now delegates to it and **reproduces its
banked JSON block exactly — max |delta| = 0.0 across all 8 series and all 3 lags.** Its
legacy key names are kept verbatim, with the sign inversion documented at the call site.

### (c) The audit — every other sleeve

| sleeve | verdict | evidence |
|---|---|---|
| `multiasset_trend` | **CORRECT** | `n.shift(1)` weights × unshifted returns |
| `multiasset/carry` | **CORRECT** | `pos.shift(1) * ret`, cost shifted onto the earning month |
| `multiasset_seasonal` | **CORRECT** | daily P&L compounded by the month it occurred |
| `multiasset_defensive` | **CORRECT** | `u.shift(1)` weights held during month t |
| `riskparity` | **CORRECT** | `w.shift(1)` held × unshifted excess |
| `multiasset_value` | **CORRECT** | same idiom |
| `reversal_retest`, `short_horizon_reversal` | **CORRECT** | labelled by the EXECUTION bar, not the signal bar |
| `tsmom_multitimeframe` | **CORRECT** | daily, booked on the day earned |
| `breadth_ladder` | **CORRECT** | inherits `riskparity.levered` |
| `capacity_study` | **N/A, latent** | books `forward_return` against the formation date but never attaches dates to the return array; nothing on disk is wrong, and zipping `dates` onto `net_series` would make it wrong |
| `low_vol_quality` | **N/A, latent** | identical construction to `lowvol_retest`, one step short of exposing it (`SleeveResult` is scalars only) |
| `pead`, `pead_retest` | **DEFECTIVE on the benchmark leg** | see below |

Measured, not asserted. `research/alignment.probe_alignment` over every dated artefact on
disk, against SPX:

```
lowvol_corrected.benchmark        MISALIGNED  k=+1  max|rho| 0.769
lowvol_registered.benchmark       MISALIGNED  k=+1  max|rho| 0.814
trend.bench_net_10bps             ALIGNED     k= 0  max|rho| 0.827
seasonal.composite                ALIGNED     k= 0  max|rho| 0.436
defensive.bench_net_10bps         ALIGNED     k= 0  max|rho| 0.832
value.bench_net_10bps             ALIGNED     k= 0  max|rho| 0.835
reversal.quintile_long_only       ALIGNED     k= 0  max|rho| 0.804
trend.net_10bps / carry.net / defensive.net   UNINFORMATIVE — market-neutral, no power
```

**Nothing but the four low-vol columns is misaligned.** Those four are the known,
compensated-for defect and are now DECLARED as `FORMATION` in the registry.

### (d) The guard

`research/sleeve_registry.py` declares, for **26 dated artefacts**, what the index MEANS
and whether an equity reference should have power over it.
`tests/test_dating_alignment.py` (28 tests):

- the probe on synthetic data where the answer is known by construction, including gap
  robustness, month-start/month-end joins, and duplicate refusal;
- **`run_band` driven end to end on a synthetic panel with a planted market factor** —
  the defect is REPRODUCED under the registered default (`MISALIGNED`, `k=+1`) and shown
  repaired under `REALISATION`. This is the test that would have caught it;
- every artefact on disk measured against a tracked reference, with the DECLARED
  convention asserted against the MEASURED one, in both directions;
- an assertion that `portfolio_correlation_v2.NEEDS_MONTH_SHIFT` is exactly the set of
  formation-dated sources, so a consumer that drops the compensating shift breaks;
- the reference itself re-anchored on SPX where the vendor panel is present.

Mutation-checked: declaring the low-vol artefacts `REALISATION` makes the guard fire on
all four columns.

### NEW FINDING, NOT IN THE BRIEF: `pead` / `pead_retest`

Their strategy leg is CORRECT (daily equity curve resampled month-end) but their benchmark
leg is `panel.groupby(date.to_period("M"))["forward_return"].mean()` — **formation-dated**
(`pead.py:641-644`, `pead_retest.py:580-583`). The two legs are one month out of phase
*inside one file*, so every "excess vs universe" and every decade figure in
`pead_retest_result.md` is a difference of misaligned series — not merely mislabelled, but
numerically wrong.

**Deliberately left alone.** Repairing it changes a banked verdict, and both sleeves were
already ruled DEAD on other grounds. It is recorded here and is the first thing to fix if
either is ever revived. Neither writes a monthly artefact, so it has not contaminated any
cross-sleeve join.

---

## P2 — THE DELISTING OFF-BY-ONE

`(exit, exit+62]` with a STRICT lower edge, against a Sharadar ACTIONS date that falls **on**
the ticker's last traded bar — median gap 0 days. The strict edge rejects the modal case:
it fired 39 times out of 3,018 while 6,322 last-observation cells carried a record whose
median terminal return is −1.00.

`research/delisting.py` holds ONE definition: `REGISTERED_WINDOW = (1, 62)` and
`CORRECTED_WINDOW = (0, 62)`, with scalar and vectorised predicates. The registered window
is proven equal to the original expression over 500 random offsets.

| site | verdict | action |
|---|---|---|
| `low_vol_quality.run_band` ×2 | **DEFECTIVE** | `delisting_window` parameter, default REGISTERED |
| `capacity_study.run_band` | **DEFECTIVE** (62 hard-coded) | same |
| `institutional_flow.run_portfolio` | **DEFECTIVE** | same |
| `institutional_flow.forward_horizon_return` | **DEFECTIVE** | lower edge parameterised; the upper edge stays horizon-specific |
| `insider_clustering.build_universe` | **DEFECTIVE** | same |
| `lowvol_retest.run_band` | already parameterised | rewired to the shared module |
| `scripts/verify_institutional_flow_sleeve.py` | **DEFECTIVE** | it hard-coded the SAME strict edge as the sleeve it verifies, so it reproduced the bug and **could never have detected it**. It now reads both windows and prints both counts. |
| `pead`, `pead_retest`, `tsmom_multitimeframe` | **CORRECT** (`0 <= gap <=`) | left alone |
| `reversal_retest`, `short_horizon_reversal`, `scripts/verify_reversal_retest.py` | **CORRECT** (inclusive, entry-anchored) | left alone |
| `research/sleeves/_lowvol_verify/*` | strict **on purpose** | left alone — it replicates iteration 1 bug-for-bug for forensics |

`DELISTING_WINDOW_DAYS` had three copies; it now has one.

`tests/test_delisting_window.py` (32 tests): a 320-day exhaustive sweep proving the repair
changes **exactly** the gap-0 day and nothing else; a synthetic panel whose top-ranked name
dies with the delisting dated on its last bar, where the registered window books
`delisting_drag_annual == 0.0` and the corrected one books the loss; a signature test that
all six call sites expose the switch and default to REGISTERED; and a grep guard that fails
if the strict idiom is reintroduced anywhere under `research/`.

---

## P3 — THE 12-MONTH LEVERAGE BUG

`k = pd.concat([k_raw, k_cap], axis=1).min(axis=1)`, and `DataFrame.min` defaults to
`skipna=True`. `k_raw` is NaN for a book's first `BOOK_VOL_MIN = 12` months, so `min`
silently drops it and the book runs at exactly `GROSS_CAP = 10x` with no volatility
estimate behind it. There is no `isfinite` branch anywhere.

**And the diagnostic hid it.** `cap_binding = (k_raw > k_cap) & k_raw.notna() & ...`
excludes precisely the months where the cap is the only thing setting leverage. Measured on
the trend sleeve: **12 months at exactly 10.0x, and `cap_binding` reports 0.**

`research/book_scaler.py` holds the shared scaler with an explicit `no_estimate` policy:
`NO_ESTIMATE_CAP` (registered, default) and `NO_ESTIMATE_FLAT` (the repair). Both masks are
returned under BOTH policies.

| module | verdict |
|---|---|
| `multiasset_trend` | **AFFECTED and HIDDEN** — the root; fixed behind the switch |
| `multiasset_value` | **AFFECTED and HIDDEN** — byte-identical clone |
| `multiasset_seasonal` | **AFFECTED, already disclosed** — rewired, count preserved |
| `multiasset_defensive` | **AFFECTED, already counted** — rewired; keeps its own convention of folding the hole into `cap_binding` |
| `riskparity` | **NOT AFFECTED** — `Series.clip` PROPAGATES NaN where `DataFrame.min` skips it, so the book simply goes flat |
| `tsmom_multitimeframe` | **NOT AFFECTED** — explicit `isfinite` branch that goes flat. This is the correct pattern and it already existed in the repo. |
| `breadth_ladder`, `multiasset/carry`, `breadth_neff`, `_portfolio/*` | **N/A** |

### MEASURED on the real panel — registered default vs the flat repair, net-10bps Sharpe

| book | registered | repaired | delta |
|---|---:|---:|---:|
| defensive @ 20% | 0.1136 | 0.1639 | **+0.0503** |
| defensive @ 40% | 0.1642 | 0.1922 | +0.0280 |
| value @ 20% | −0.0824 | −0.1177 | −0.0353 |
| trend @ 20% | 0.6116 | 0.6097 | −0.0019 |
| trend @ 10% | 0.6104 | 0.6097 | −0.0007 |

The brief's "worth 0.050 of Sharpe" is **reproduced exactly**: it is the defensive sleeve at
its 20% target. On the trend book the effect is ~0.002 — the defect is real everywhere and
material in only one place, and that distinction was not previously recorded.

**No banked number moved.** The default was verified bit-for-bit against the pre-change
modules — `gross`, `net`, `weights`, `cap_binding` — on trend, value and defensive across
four volatility targets. Exactly 12 months are affected in every book.

`tests/test_book_scaler.py` (28 tests) pins the pandas mechanism itself (`min` skips NaN,
`clip` propagates it — which is *why* `riskparity` was immune), the fall-through, the
concealment, the repair touching only those 12 months, each sleeve at exactly 10x for its
first 12 months, a signature audit, and a grep guard against the raw idiom.

---

## P4 — UNPRICED SELL LEGS

A name that leaves the tradable universe still has to be **sold**. The exit leg was counted
in TURNOVER and charged nothing.

| module | verdict | action |
|---|---|---|
| `lowvol_retest.run_band` | AFFECTED, switch present | the counter now increments **unconditionally**, so how many legs *would* be free no longer depends on whether the run charged them |
| `low_vol_quality.run_band` | **AFFECTED** — the direct ancestor | `charge_unpriced_exits` added, default `False`, with `unpriced_exit_legs` / `charged_unpriced_exit_legs` on `SleeveResult` |
| `institutional_flow.run_portfolio` | **AFFECTED (partial)** | the accrual fallback priced most exits, but two `continue`s still exited free with no counter. Last observed `(spread, price, mdv)` is now carried and both counters reported. |
| `capacity_study` | **NOT AFFECTED** | it intersects `traded` with the priced cross-section *before* computing turnover, so the two agree. (Related but distinct: the exit is still free, and here it is *masked* rather than counted.) |
| `insider_clustering` | **ALREADY FIXED** | union index over targets and current holdings; its comment describes this exact defect |
| `short_horizon_reversal`, `reversal_retest` | **ALREADY FIXED** | forward-filled cost basis + a hard `RuntimeError` rather than a skip |
| `pead`, `pead_retest` | **NOT AFFECTED** | event-driven; each position carries its own entry-time cost inputs |
| `multiasset_value`, `multiasset_defensive`, `riskparity`, `tsmom`, `breadth_ladder` | **N/A** | flat-bps turnover costing, no per-name skip possible |

`tests/test_unpriced_exit_legs.py` (13 tests) uses a synthetic panel in which a held name
**migrates out of the band while still trading** — not a delisting, exactly the case the
cost model skipped. It asserts `legs_traded` and `turnover_annual` are IDENTICAL under both
settings while cost rises and net falls; that gross return is unchanged (the signal cannot
see costs, so holdings must be identical); and that all three sleeves expose the switch and
report both counters.

---

## P5 — THE STALE TRIAL LEDGER

Six different "cumulative" counts were live in the code at once: 9, 34, 36, 38, 46, 47.
`portfolio_decision.json` deflated against **38** while the ledger stood at **47**, and a
sibling in the same directory hard-coded **46**.

`research/trial_ledger.py` is the single machine-readable source of truth:
`CUMULATIVE_TRIALS = 47`, with a cited checkpoint for every recorded movement
(`internal research log:179 / :643 / :1023 / :1742 / :1909`).

- `deflated_sharpe_ratio` and `dsr_sharpe_bar` accept `n_trials=None` and read it. **Their
  defaults are unchanged** — 32 is the anchor the frequency convention is pinned by and 1
  is baked into every banked result; a default that moved with the ledger would silently
  re-deflate all of them.
- Corroboration: the ledger bar at 17.75yr, n=47 computes to **0.9443077723019092**, which
  is exactly the value `_pair_deflation/controls.json` had already recorded independently.
- **Measured cost of the staleness:** deflating at 38 instead of 47 understates the
  17.75-year Sharpe bar by **0.0209** (0.92339 → 0.94431).
- `portfolio_decision.py` now emits `dsr_bar_ledger_trials` / `dsr_ledger_trials` beside
  its frozen 38, so the understatement is visible rather than recoverable only by
  inverting a bar. Its constants are kept verbatim and registered as **STALE** — the
  banked JSON was produced at them.
- `FROZEN_TRIAL_COUNTS` registers **every** hard-coded count in `research/` and `scripts/`
  with its kind: `PROGRAMME`, `ANCHOR` (the DSR formula's pinned n=32 reproduction), `LADDER`
  (declared sensitivity), `HISTORICAL` (the separate equity-alpha programme that ended at 23).

`tests/test_trial_ledger.py` (17 tests) fails if any file hard-codes a count not registered
there, with two mutation checks: a brand-new module with `N_TRIALS = 51` fails, and a NEW
count inside an already-registered file fails.

---

## P6 — THE REFUTED CONSTANT

`SMALL_CAP_DOLLAR_VOLUME_RANGE = (1e7, 5e7)` is refuted. FIM define "small cap" by
market-cap RANK; ranks 1001-3000 measure a median of **$3.31M/day**, IQR $1.16M-$8.14M.
$10M-$50M/day is the bottom half of the **Russell 1000** here (median $52.1M/day).

**Documented, not swapped**, and the measured replacement is imported from
`research.spread_estimation` rather than copied. A new DECLARED, **non-gated**
`check_b_measured_liquidity_disclosure` re-runs the containment at the correct liquidity,
where it FAILS, and decomposes the failure:

| term | measured |
|---|---:|
| modelled bracket at $3.31M/day | **[34.19, 42.05] bps** |
| FIM measured | 13.53 bps |
| **half-spread term** | **33.15 bps** — 97% of the floor |
| impact term (a) / (b) | 8.90 / 1.04 bps — **inside** FIM's 13.53 |

So **re-running impact check B at the correct liquidity would fail containment via the
half-spread term**, which is the E5 residual `spread_positive_control.py` already discloses
and deliberately leaves standing. The impact coefficients are not what breaks. Moving the
bucket without resolving E5 would turn a passing control into a failing one, and correctly
so — which is why the registered bucket was not simply replaced.

(The disclosure reads no tape so it is unit-testable; it defaults to the FIM anchor
volatility and prints [34.2, 42.0]. the internal research log iteration 9 quotes [36.0, 46.0] for
the same bucket because it used the bucket's own measured volatility. Both fail, and both
fail on the same term.)

`tests/test_impact_small_cap_bucket.py` (9 tests) pins the refutation, that the two buckets
do not even overlap, that containment fails at the measured liquidity, that the failure is
the half-spread and not the impact model, and that the disclosure can never flip the
control's verdict.

---

## DELIBERATELY LEFT ALONE

1. **Every banked number.** All six repairs are switches defaulting to the registered
   behaviour. No result file, parquet or CSV was rewritten.
2. **The `pead` / `pead_retest` benchmark dating defect** (found in this pass, not in the
   brief). Repairing it changes a banked verdict on two sleeves already ruled DEAD, and it
   has contaminated no cross-sleeve join because neither writes a monthly artefact.
   Recorded above as the first thing to fix if either is revived.
3. **The banked low-vol parquets stay formation-dated.** Rewriting them would change an
   artefact a concurrent report is written against. The registry DECLARES the convention
   and the test pins it in both directions, so the compensating shift cannot be dropped and
   the dating cannot change without the declaration changing in the same commit.
4. **`research/sleeves/_lowvol_verify/*`** keeps the strict delisting edge and the free
   exit leg. It exists to replicate iteration 1 bug-for-bug for forensics; repairing it
   would destroy the comparison. Excluded from the grep guard by name, with the reason.
5. **The E5 half-spread residual** (~33bps/side at the median Russell 2000 constituent
   against FIM's 13.53). Closing it needs a factor nobody here has measured — AGK's pooled
   era mix, the `era_multiplier` floor at 1.0, FIM's patient algorithmic execution.
   Charging less on the strength of any of them would be a guess in the direction that
   flatters every strategy.
6. **`capacity_study`'s latent dating shape.** It books `forward_return` against the
   formation date but never attaches dates to the return array, so nothing on disk is
   wrong. Flagged in the audit table above rather than restructured.
7. **Pre-existing `ruff` and `mypy` findings** in files this pass touched (unused imports
   in `multiasset_trend` / `multiasset_defensive`, an `E741`, 161 `mypy` errors tree-wide).
   Verified unchanged against a clean worktree of the pre-work commit. Fixing them is
   unrelated churn in files a concurrent agent is also editing. The one `mypy` error this
   work *did* introduce (a narrowing failure in `institutional_flow`) was found and fixed.
8. **`capacity_study`'s masked free exit** — the leg is dropped from turnover as well as
   from cost, so the two agree. Understated turnover is a separate finding and is recorded
   in the P4 table rather than silently changed.
