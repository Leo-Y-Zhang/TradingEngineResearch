# Backtest / Walk-Forward Harness — Design

**Date:** 2026-06-09 · **ROADMAP:** Phase 2, item 1 (the first Phase-2 sub-project) ·
**Status:** approved (design), pre-implementation

## Purpose

Make TradingEngineResearch's returns *measurable*: replay the real TradingEngineResearch engine over a price
history, net of cost, on purged walk-forward splits, and report standard
risk-adjusted performance metrics. "Can't improve what you can't measure" — this is
the highest-leverage Phase-2 item and the foundation for evaluating every later
alpha/learning change.

## Approach (chosen: A — PAPER-mode engine replay)

Step the real `TradingEngine` (PAPER mode) forward over rebalance dates. At each date
`t`: build PIT-safe `CycleInputs` from data strictly ≤ `t`, call `run_cycle`, take the
risk-approved target book, hold it to the next date, and book the net-of-cost return.
This exercises the entire 13-step pipeline (regime, crisis tightening, optimiser, the
Phase-1 CVaR enforcement, the fail-closed risk gate + drawdown governor, TCA).

Rejected: (B) calling `optimise_portfolio` directly — bypasses meta-labelling, the risk
gate, crisis tightening, TCA (low fidelity); (C) a vectorised parallel backtest — would
drift from the real engine.

## Components

New `backtesting/` package (currently an empty stub):

- **`backtesting/metrics.py`** — pure, side-effect-free performance functions on a
  returns series: `ann_return`, `ann_vol`, `sharpe`, `sortino`, `max_drawdown`,
  `calmar`, `hit_rate`. Pure by design so they are the natural target for the later
  property-based-tests Phase-2 item. `periods_per_year` is a parameter (default 252).
- **`backtesting/harness.py`** — `Backtester` + `BacktestResult`.

### Engine enhancement (small, backward-compatible)

Add `target_weights: dict = field(default_factory=dict)` to `CycleResult`, populated
with the **risk-approved book** (the gate's `scaled_weights`). This is the authoritative
intended allocation; today it is only implicit in `order_intents` (which omit exited
names and untraded holds, making book reconstruction error-prone). The new field also
benefits monitoring. Default-valued, so existing `CycleResult` construction and tests
are unaffected.

## `Backtester`

```
Backtester(mode="PAPER", capital_gbp=1_000_000.0, rebalance="W",
           cost_bps_floor=1.0, stale_threshold_seconds=300.0, seed=42)
run(prices: DataFrame, splitter: PurgedWalkForwardSplitter | None = None) -> BacktestResult
```

Per rebalance date `t` (resampled from `prices.index` at `rebalance`):
1. Build `CycleInputs` from data strictly ≤ `t`: `prices`/`returns_matrix` slice,
   `current_weights` = carried book, `drawdown_current` from the **running equity curve**
   (so the drawdown governor genuinely engages during drawdowns), microstructure stub.
2. `result = engine.run_cycle(inputs)`.
3. New book: `current_weights` (carry, no trade) if `result.blocked`, else
   `result.target_weights`. Names absent from the new book are exited (weight 0).
4. Realised period return over `[t, t+1]` = `book · forward_asset_returns`.
5. Cost = `turnover(new_book, prev_book) · max(cycle_cost_bps, cost_bps_floor) · 1e-4`,
   where `cycle_cost_bps` comes from the cycle (`optimizer_result.expected_cost_bps` /
   TCA). Net return = gross − cost.
6. Append to the equity curve; record per-rebalance weights/turnover/cost.

## Walk-forward / purged splits

With a `splitter` (`research.validation.PurgedWalkForwardSplitter`), evaluate over the
splitter's **test** segments only and report per-segment + aggregate metrics, honouring
purge/embargo gaps. This makes the harness leakage-safe by construction and ready for
per-window model fitting later. Without a splitter, it runs a single continuous pass.

## `BacktestResult`

`equity_curve` (Series), `returns` (Series), `weights_history` / `turnover_history` /
`cost_history`, a `metrics` dict (`cagr`, `ann_return`, `ann_vol`, `sharpe`, `sortino`,
`max_drawdown`, `calmar`, `hit_rate`, `avg_turnover`, `total_cost_bps`, `n_rebalances`),
`per_split` (list of per-segment metric dicts), and `summary() -> str`.

## Determinism

Timestamps come from the data (never wall-clock — the engine already stamps the audit
from `asof_time`), and RESEARCH/PAPER RNG is seeded. Two runs on the same inputs are
bit-identical. This directly sets up the separate "determinism test" Phase-2 item.

## Testing (TDD, red→green)

- `metrics.py`: unit tests with known series → known Sharpe / max-drawdown / etc.
- harness smoke: synthetic price history → sane equity curve, all metric keys present,
  returns series length == number of rebalances.
- determinism: two runs → identical equity curve.
- behavioural: an engineered drawdown shrinks exposure (governor) vs a benign run.
- walk-forward: with a splitter, `per_split` has one entry per test segment.

Runs on the existing synthetic price generator now; swaps to the recorded real-data
fixture once that sub-project lands. Property-based tests are their own Phase-2 item
(needs `hypothesis`).

## Scope guard (YAGNI)

No plotting, no multi-asset-class / futures roll, no parameter optimisation or
grid-search, no parallelism, no live/paper broker wiring. Just a correct, deterministic,
net-of-cost engine replay with metrics and purged walk-forward support.

## Out of scope (later Phase-2 sub-projects)

Real feature ingestion + recorded yfinance fixture; `evaluate_factor()` / `selection_rule()`
promotion gating; RESEARCH determinism test (item); property-based kernel tests (item).
