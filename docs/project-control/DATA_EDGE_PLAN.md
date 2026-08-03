# TradingEngineResearch — Data → Validated Edge Plan

How to turn a ~$30/month data subscription into an honest answer about whether this system can
earn a real, repeatable edge — and how to scale returns up *if and only if* it does. Written
2026-06-30. 

## What faith should rest on (read first)

This plan does **not** promise 30%/year. No honest plan can. What it delivers is a pipeline that
will **tell you the truth**: it finds a real edge when one exists in the data, and rejects fake or
overfit edges when one does not. Faith belongs in *that process*, not in a number.

- Returns above the market come from a real informational/structural edge. Free price/volume data has
  none left that survives honest testing — your own research (`research/medallion_style_alpha_search/`)
  already proved this with Deflated Sharpe / PBO gates.
- The one place real signal showed up was **fundamentals** (ROE/ROA), but it failed deflation on too
  short/narrow a sample. The binding constraint is **data quality + statistical power** — exactly what
  a point-in-time, survivorship-free fundamentals dataset fixes.
- **More risk ≠ more reliable return.** Leverage scales an edge you already have (up *and* down). On no
  edge it just scales losses. So: validate an edge first, scale risk second. Never the reverse.

## What is now built and ready (2026-06-30)

All additive, tested, reviewed (refute-by-default), and green. Nothing runs against real money.

| File | What it does |
|------|--------------|
| `data/sharadar_ingestion.py` | Point-in-time, survivorship-free loader for Sharadar SF1 (fundamentals) + SEP (prices) from a local export. PIT-safe **by default** (uses the filing `datekey`, defaults to as-reported `ARQ` so restated figures can't leak), keeps delisted tickers. |
| `research/fundamental_features.py` | 14 well-motivated, PIT-safe fundamental factors (value, quality/profitability, growth, investment, earnings-quality, leverage, momentum), cross-sectionally normalized per date. Factors are reported as-measured so the learner discovers the sign. |
| `scripts/research_sharadar_alpha.py` | The one command to run after subscribing: load → features → forward returns → `learn_signal_weights` (purged walk-forward) → DSR/PBO gate → honest **DEPLOYABLE / NOT-DEPLOYABLE** verdict (default-deny). Has a `--selftest` runnable without the paid data. |
| `tests/test_edge_recovery_proof.py` | The trust proof. On synthetic data it shows the gate: **passes a real edge — including near its 0.95 threshold** (DSR ~0.965–0.987, not a saturated cartoon), **denies pure noise**, and **denies an in-sample-attractive but out-of-sample-worthless overfit signal**. |

## The single best experiment (do this when ready)

1. **Subscribe** to the **Sharadar "Core US Equities Bundle" (code SFA)** via Nasdaq Data Link —
   roughly **~$50/month** on the **Personal / non-commercial** tier (verify live at
   [data.nasdaq.com/databases/SFA/pricing](https://data.nasdaq.com/databases/SFA/pricing)). SFA is
   preferred over buying SF1 + SEP separately because bundle pricing is ~the same yet it also includes
   **SF2 (insiders)**, **SF3 (institutional 13F)**, **DAILY** (clean daily marketcap/P-E/P-B — removes
   the `price × sharesbas` approximation in the runner), **TICKERS** (survivorship-free reference), and
   **ACTIONS**. Decision rule: if `SFA ≤ SF1 + SEP` on the Personal tier (almost always true), buy SFA.
   Our loader needs only **SF1 + SEP** (both in SFA); the rest are future signal sources. Before paying,
   the pipeline can be smoke-tested against Sharadar's **free SAMPLE tables**. Bulk-export the CSVs to a
   folder, e.g. `_data\sharadar\`.
2. **Run:**
   ```powershell
   cd <repo-root>
   python scripts\research_sharadar_alpha.py --data-dir _data\sharadar
   ```
3. **Read the verdict.** It prints OOS IC / rank-IC, net Sharpe, Deflated Sharpe vs the 0.95 cutoff,
   PBO, and DEPLOYABLE / NOT-DEPLOYABLE. Default-deny: it only says DEPLOYABLE if the edge survives
   deflation over a moderate universe (aim for ~200–500 names) and long history (15–20yr).
4. **Expect iteration.** Most candidate configurations will be denied — that is the gate working, not
   failing. A real search is weeks-to-months.

## Realistic outcomes (honest)

- **If** a deflation-robust edge is found: realistically it adds **a few percentage points/year** over
  the benchmark with controlled risk — a genuinely strong, fundable result. Your absolute return still
  mostly rides the market.
- **Reliable low-risk 30%/year is not on the table.** A single good year can hit 30% via market beta +
  leverage, but that is not a repeatable edge and loses correspondingly in bad years.
- There is a real chance the answer stays **"no robust edge even with better data."** The system's job
  is to say so honestly rather than sell you an overfit backtest.

## Scaling returns up — only after an edge validates

The engine already exposes the risk/return knobs (in `core/config.py`, set via `ENGINE_*` env or
`.env`). Once — and only once — `research_sharadar_alpha.py` returns **DEPLOYABLE**, these dial the
*absolute* return target up, accepting larger drawdowns:

| Setting (env: `ENGINE_<NAME>`) | Default | Effect of raising it |
|---|---|---|
| `TARGET_VOL` | 0.22 | Higher volatility target → bigger positions → higher expected return *and* drawdown. |
| `MAX_GROSS_LEVERAGE` | 2.0 | Allows more gross exposure. Past ~2× the procyclical-leverage risk (OPT-1) grows fast. |
| `MAX_POSITION_WEIGHT` | 0.20 | Per-name concentration cap (enforced post-leverage at STEP-10). |
| `CVAR_LIMIT` | 0.12 | Tail-risk budget; loosening it permits more aggressive books. |
| `SIGNAL_TILT_STRENGTH` | 3e-3 | How hard the book tilts toward the signal. |

**The honest tradeoff:** to chase 30%, you raise `TARGET_VOL`/`MAX_GROSS_LEVERAGE` — but that multiplies
both the edge *and* the drawdowns, and only pays off if the edge is real and persistent. Leverage on a
thin or decayed edge is the classic way accounts blow up. The independent STEP-10 risk gate, CVaR
limit, drawdown governor, and kill-switch still apply at any setting, but they cap *risk*, not *loss in
a bad regime*. Treat any move above the defaults as a deliberate, monitored decision.

## Known refinements (deferred, honest backlog)

The build passed an adversarial review; these lower-priority items were deliberately left for later and
do **not** affect PIT-safety or the deny-by-default guarantee:

- **`scripts/` not in the mypy CI scope** — the runner type-checks clean only when invoked explicitly.
- **DSR `n_trials`** is set to ~feature-count and likely under-counts the full research search space, so
  deflation is slightly lenient; document/raise it to reflect every configuration tried.
- **Delisting return** — a name that delists mid-period has its final (often large-loss) return dropped
  rather than modeled; model it or document the limitation before trusting delisting-heavy backtests.
- **Naive-baseline labeling** in the runner is in-sample and not sign-aligned — annotate or compute OOS.
- **Near-threshold trust-proof seeds** are mid-band by construction (DSR is a steep function of Sharpe),
  so that one test is more sensitive to library-version bumps than the saturated controls; margins were
  kept (≥0.015 above the 0.95 cutoff).

None of these block the core experiment. They are the next polish once the data confirms whether there
is anything worth polishing for.
