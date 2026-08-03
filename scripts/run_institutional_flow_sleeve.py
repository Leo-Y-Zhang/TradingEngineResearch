"""Run the institutional-ownership-flow sleeve ONCE and print the measured result.

Registered design: `research/sleeves/institutional_flow_prereg.md`. One configuration,
one run. Nothing in this script may be adjusted after seeing a number; if the sleeve
fails, the failure with its numbers is the result.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR  # noqa: E402
from research.sleeves.institutional_flow import (  # noqa: E402
    BOOK_SIZE,
    GATE_EXCESS,
    GATE_IC_T,
    GATE_SHARPE,
    HOLDING_MONTHS,
    MIN_DOLLAR_VOLUME,
    N_POSITIONS,
    N_POSITIONS_TERCILE,
    REBALANCES_PER_YEAR,
    SleeveResult,
    build_signal_panel,
    equal_weight_universe_selector,
    forward_horizon_return,
    information_coefficient,
    long_short_spread,
    market_month_ends,
    rebalance_schedule,
    run_portfolio,
    top_n_selector,
)

logger = logging.getLogger(__name__)

HORIZON_COLUMN = "forward_horizon_return"


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    ownership = pd.read_parquet(PANEL_DIR / "sf3_ownership_dev.parquet")
    marketcap = pd.read_parquet(PANEL_DIR / "quarter_end_marketcap_dev.parquet")

    # The DEV guard, asserted rather than assumed. Every one of these tables is built by
    # a loader that filters on the cutoff, but a single stale cache would silently make
    # the whole study a lookahead.
    for name, frame, column in (("panel", panel, "date"),
                                ("ownership", ownership, "calendardate"),
                                ("marketcap", marketcap, "marketcap_date")):
        latest = frame[column].max()
        if latest > DEV_CUTOFF:
            raise ValueError(f"{name} contains {column} {latest} past the DEV cutoff")
    return panel, delistings, ownership, marketcap


def _tercile_test(signals: pd.DataFrame, accrual: pd.DataFrame,
                  delistings: pd.DataFrame) -> dict[str, dict]:
    """H2: does the effect concentrate where institutional ownership is LOW?

    Terciles are formed on the ownership LEVEL inside each cross-section, so the split is
    point-in-time and does not use the full-sample distribution.
    """
    labelled = []
    for _, frame in signals.groupby("date", sort=True):
        frame = frame.dropna(subset=["own_q"]).copy()
        if len(frame) < 3 * N_POSITIONS_TERCILE:
            continue
        frame["ownership_tercile"] = pd.qcut(
            frame["own_q"].rank(method="first"), 3,
            labels=["low", "mid", "high"])
        labelled.append(frame)
    if not labelled:
        return {}
    labelled_frame = pd.concat(labelled, ignore_index=True)

    results: dict[str, dict] = {}
    for tercile in ("low", "mid", "high"):
        subset = labelled_frame[labelled_frame["ownership_tercile"] == tercile].copy()
        if subset.empty:
            continue
        strategy = run_portfolio(accrual, subset, top_n_selector(N_POSITIONS_TERCILE),
                                 delistings, charge_costs=True,
                                 n_positions=N_POSITIONS_TERCILE)
        benchmark = run_portfolio(accrual, subset, equal_weight_universe_selector(),
                                  delistings, charge_costs=False)
        _, ic_mean, _, ic_t, ic_n = information_coefficient(subset, HORIZON_COLUMN)
        results[tercile] = {
            "net_return": strategy.annual_return,
            "benchmark": benchmark.annual_return,
            "excess": strategy.annual_return - benchmark.annual_return,
            "sharpe": strategy.sharpe,
            "ic_mean": ic_mean,
            "ic_t": ic_t,
            "ic_n": ic_n,
            "mean_ownership": float(subset["own_q"].mean()),
            "names_per_date": float(subset.groupby("date").size().mean()),
        }
    return results


def run() -> SleeveResult:
    panel, delistings, ownership, marketcap = _load()

    # The market-wide month-end grid. Raw panel dates include each delisting name's last
    # mid-month bar, and using those as rebalance dates produces one-name cross-sections.
    month_ends = market_month_ends(panel)
    panel = panel[panel["date"].isin(month_ends)].copy()
    schedule = rebalance_schedule(
        pd.DatetimeIndex(ownership["calendardate"].unique()).sort_values(), month_ends)
    logger.info("schedule: %d quarterly rebalances, %s .. %s", len(schedule),
                schedule["rebalance_date"].min().date(),
                schedule["rebalance_date"].max().date())

    signals = build_signal_panel(panel, ownership, marketcap, schedule)
    if signals.empty:
        raise SystemExit("no eligible cross-sections; the sleeve cannot be run")

    horizon = forward_horizon_return(panel, delistings, months=HOLDING_MONTHS)
    signals = signals.merge(horizon, on=["ticker", "date"], how="left")

    accrual = panel[["ticker", "date", "close", "closeadj", "median_dollar_volume",
                     "forward_return"]].copy()

    strategy = run_portfolio(accrual, signals, top_n_selector(N_POSITIONS),
                             delistings, charge_costs=True)
    benchmark = run_portfolio(accrual, signals, equal_weight_universe_selector(),
                              delistings, charge_costs=False)

    ic_series, ic_mean, ic_se, ic_t, ic_n = information_coefficient(signals,
                                                                    HORIZON_COLUMN)
    signals_1m = signals.rename(columns={"forward_return": "_fr"})
    signals_1m["forward_1m"] = signals["forward_return"].clip(-1.0, 1.0)
    _, ic1_mean, _, ic1_t, _ = information_coefficient(signals_1m, "forward_1m")
    ls_annual, ls_sharpe = long_short_spread(signals, HORIZON_COLUMN)

    # Universe composition. Reported because the `measured`-spread requirement is not
    # neutral: EDGE only resolves a spread that sits 1.5x above its volatility-scaled
    # noise floor, so the tradable universe is systematically the WIDE-spread names and
    # the tight-spread ones are excluded as unmeasurable. That is the honest treatment of
    # the cost model, and it is also a real bias in what this sleeve is allowed to trade.
    at_rebalance = panel[panel["date"].isin(signals["date"].unique())]
    diagnostics = {
        "median_spread_bps": float(signals["spread"].median() * 1e4),
        "mean_spread_bps": float(signals["spread"].mean() * 1e4),
        "median_dollar_volume": float(signals["median_dollar_volume"].median()),
        "cells_at_rebalance_dates": float(len(at_rebalance)),
        "cells_measured": float((at_rebalance["spread_regime"] == "measured").sum()),
        "cells_upper_bound": float((at_rebalance["spread_regime"]
                                    == "upper_bound").sum()),
        "signal_autocorrelation": float(
            signals.sort_values(["ticker", "date"])
            .groupby("ticker")["signal"]
            .apply(lambda s: s.autocorr(1) if len(s) > 2 else np.nan)
            .mean()),
    }

    notes: list[str] = []
    n_rebalances = signals["date"].nunique()
    if n_rebalances < 20:
        notes.append(
            f"ONLY {n_rebalances} REBALANCES. Sharadar SF3 begins 2013-06-30, so the "
            "DEV window contains 11 quarters of 13F data in total. Every statistic "
            "below is estimated on a sample too short to distinguish from noise.")

    return SleeveResult(
        strategy=strategy,
        benchmark=benchmark,
        ic_mean=ic_mean,
        ic_std_error=ic_se,
        ic_t_stat=ic_t,
        ic_count=ic_n,
        ic_by_date=ic_series,
        ic_1m_mean=ic1_mean,
        ic_1m_t_stat=ic1_t,
        long_short_annual=ls_annual,
        long_short_sharpe=ls_sharpe,
        rebalance_dates=list(pd.DatetimeIndex(signals["date"].unique()).sort_values()),
        universe_size_mean=float(signals.groupby("date").size().mean()),
        diagnostics=diagnostics,
        tercile_results=_tercile_test(signals, accrual, delistings),
        notes=notes,
    )


def report(result: SleeveResult) -> str:
    strategy, benchmark = result.strategy, result.benchmark
    months = len(strategy.net_returns)
    years = months / 12.0
    components = strategy.cost_components
    total_components = sum(components.values()) or 1.0

    lines = [
        "=" * 78,
        "SLEEVE: INSTITUTIONAL OWNERSHIP FLOW (Sharadar SF3 13F holdings)",
        "Pre-registration: research/sleeves/institutional_flow_prereg.md",
        "=" * 78,
        "",
        "SAMPLE",
        f"  rebalances                {len(result.rebalance_dates):>10d}",
        f"  first / last rebalance    {result.rebalance_dates[0].date()!s:>10} .. "
        f"{result.rebalance_dates[-1].date()!s}",
        f"  monthly return periods    {months:>10d}   ({years:.2f} years)",
        f"  eligible names / date     {result.universe_size_mean:>10.0f}",
        f"  median spread             "
        f"{result.diagnostics['median_spread_bps']:>10.0f} bps",
        f"  median dollar volume      "
        f"{result.diagnostics['median_dollar_volume']:>10,.0f} USD/day",
        f"  cells at rebalance dates  "
        f"{result.diagnostics['cells_at_rebalance_dates']:>10,.0f}",
        f"    of which spread MEASURED"
        f"{result.diagnostics['cells_measured']:>10,.0f}",
        f"    excluded, upper_bound   "
        f"{result.diagnostics['cells_upper_bound']:>10,.0f}",
        f"  signal autocorr (q-on-q)  "
        f"{result.diagnostics['signal_autocorrelation']:>10.3f}",
        f"  book size                 {BOOK_SIZE:>10,.0f} USD "
        f"({N_POSITIONS} x {BOOK_SIZE / N_POSITIONS:,.0f})",
        f"  min median dollar volume  {MIN_DOLLAR_VOLUME:>10,.0f} USD/day",
        "",
        "BREADTH  (the whole point -- prereg §7)",
        f"  rebalances per year       {REBALANCES_PER_YEAR:>10.1f}",
        "  independent cross-sections         1",
        f"  INDEPENDENT BETS PER YEAR {REBALANCES_PER_YEAR:>10.1f}",
        f"  total independent bets in sample "
        f"{len(result.rebalance_dates):>3d}",
        "  Grinold: IR ~= IC * sqrt(BR). At BR = 4, IR = 1.0 needs IC = 0.50.",
        "",
        "HEADLINE  (net of per-name costs)",
        f"  gross return   {strategy.annual_gross:>9.2%}/yr",
        f"  NET return     {strategy.annual_return:>9.2%}/yr",
        f"  net volatility {strategy.annual_volatility:>9.2%}",
        f"  NET SHARPE     {strategy.sharpe:>9.2f}   "
        f"+/- {result.sharpe_standard_error:.2f} standard error",
        f"  max drawdown   {strategy.max_drawdown:>9.2%}",
        f"  turnover       {strategy.annual_turnover:>9.2f} x/yr",
        f"  cost drag      {strategy.annual_cost_drag:>9.2%}/yr",
        f"  mean positions {np.mean(strategy.position_counts):>9.1f}",
        "",
        "BENCHMARK  (equal-weight buy-and-hold of THIS universe, zero costs)",
        f"  benchmark return {benchmark.annual_return:>9.2%}/yr",
        f"  benchmark vol    {benchmark.annual_volatility:>9.2%}",
        f"  benchmark Sharpe {benchmark.sharpe:>9.2f}",
        f"  gross excess     {result.gross_excess_annual:>9.2%}/yr   "
        "(signal contribution BEFORE costs)",
        f"  EXCESS           {result.excess_annual:>9.2%}/yr   <-- the number that "
        "decides",
        "",
        "COST DECOMPOSITION  (share of total charged cost)",
        f"  half-spread    {components['spread'] / total_components:>9.1%}",
        f"  market impact  {components['impact'] / total_components:>9.1%}",
        f"  commission     {components['commission'] / total_components:>9.1%}",
        "",
        "SIGNAL QUALITY",
        f"  IC (3m horizon, Spearman)  mean {result.ic_mean:>7.4f}  "
        f"t {result.ic_t_stat:>6.2f}  n {result.ic_count}",
        f"  IC (1m horizon, Spearman)  mean {result.ic_1m_mean:>7.4f}  "
        f"t {result.ic_1m_t_stat:>6.2f}",
        f"  long/short decile spread   {result.long_short_annual:>7.2%}/yr  "
        f"Sharpe {result.long_short_sharpe:>5.2f}   NOT DEPLOYABLE (no borrow)",
        "",
        "MECHANISM TEST H2 -- does the effect live in the LOW-ownership tercile?",
        "  tercile   mean own   names/date    net      bench    excess    IC     IC t",
    ]
    for tercile in ("low", "mid", "high"):
        row = result.tercile_results.get(tercile)
        if row is None:
            continue
        lines.append(
            f"  {tercile:<8} {row['mean_ownership']:>8.1%} {row['names_per_date']:>12.0f}"
            f" {row['net_return']:>8.2%} {row['benchmark']:>9.2%}"
            f" {row['excess']:>9.2%} {row['ic_mean']:>7.4f} {row['ic_t']:>7.2f}")

    lines += [
        "",
        "VERDICT  (gates fixed in advance: excess > "
        f"{GATE_EXCESS:.0%}, Sharpe >= {GATE_SHARPE}, IC t >= {GATE_IC_T})",
        f"  {result.verdict}",
    ]
    if result.notes:
        lines.append("")
        lines.append("NOTES")
        for note in result.notes:
            lines.append(f"  ! {note}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    result = run()
    text = report(result)
    print(text)

    out = REPO / "research" / "sleeves" / "institutional_flow_result.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
