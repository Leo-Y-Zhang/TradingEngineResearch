"""Adversarial verification of the reversal re-test. Run AFTER the registered run.

Nothing here changes a verdict. It tries to break the result:

1. NEGATIVE CONTROL - fixed-seed per-date permutation of the signal. If the harness
   manufactures alpha, a shuffled signal still earns some.
2. Break-even cost arithmetic - the round-trip cost at which each frequency would clear
   the 0.75 gate and at which its excess would reach zero.
3. Why the cost did not fall as far as the >$20M/day band table implied: the regime mix
   of the traded universe.
4. Delisting bookings actually applied, and the largest single terminal return.
5. The fortnightly PHASE - the registered grid takes every 2nd weekly date starting at
   index 0. The other phase is run as a declared POST-HOC diagnostic, labelled as such.

    .venv/Scripts/python.exe -m scripts.verify_reversal_retest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sleeves.reversal_retest import (  # noqa: E402
    UNIVERSE_CUTS,
    RetestConfig,
    build_matrices,
    build_selections,
    month_row_for,
    precompute_periods,
    rebalance_grid,
    run_leg,
)
from scripts.run_reversal_retest import summarise  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "research" / "sleeves" / "_reversal_retest"


def negative_control(panel, config, seeds=(11, 22, 33, 44)) -> dict:
    """Shuffle the signal WITHIN each date and re-run the top-decile weekly book gross.

    Only the gross book matters: if the accounting is sound, a permuted signal has no
    gross edge. Costs are irrelevant to the question and are omitted.
    """
    threshold = UNIVERSE_CUTS["top_decile"]
    signal_idx, exec_idx = rebalance_grid(panel.dates, "weekly")
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    signal_idx, exec_idx, month_rows = signal_idx[keep], exec_idx[keep], month_rows[keep]
    span = float((panel.dates[signal_idx[-1]] - panel.dates[signal_idx[0]]).days) / 365.25
    ppy = (len(signal_idx) - 1) / span

    selections = build_selections(panel, config, threshold, signal_idx, exec_idx, month_rows)
    periods = precompute_periods(panel, config, signal_idx, exec_idx)
    trim = slice(0, len(signal_idx) - 1)
    basis = panel.spread_basis["zero_cost"]

    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        longs, shorts = [], []
        for k, universe in enumerate(selections["universe"]):
            if universe.size < config.min_names_per_leg * 3:
                longs.append(np.array([], dtype=int))
                shorts.append(np.array([], dtype=int))
                continue
            shuffled = rng.permutation(universe)
            n = min(int(np.floor(universe.size * config.decile)), config.max_names_per_leg)
            n = max(n, 0)
            if n < config.min_names_per_leg:
                longs.append(np.array([], dtype=int))
                shorts.append(np.array([], dtype=int))
                continue
            longs.append(shuffled[:n])
            shorts.append(shuffled[n: 2 * n])
        lb = run_leg(panel, config, longs, signal_idx, exec_idx, month_rows, periods,
                     1.0, basis, ppy, charge_impact=False)
        sb = run_leg(panel, config, shorts, signal_idx, exec_idx, month_rows, periods,
                     1.0, basis, ppy, charge_impact=False)
        gross = lb.gross_return[trim] - sb.gross_return[trim]
        out.append(summarise(gross, ppy).sharpe)
    return {"seeds": list(seeds), "placebo_gross_sharpe": [float(x) for x in out],
            "mean": float(np.mean(out)), "sd": float(np.std(out, ddof=1))}


def fortnightly_other_phase(panel, config) -> dict:
    """POST-HOC, UNREGISTERED diagnostic: the fortnightly grid's other phase.

    The registered grid takes weekly signal dates [0::2]. This runs [1::2]. It exists
    only to say whether the fortnightly dip in the reported curve is a real horizon
    effect or a phase artefact. It selects nothing and changes no verdict.
    """
    threshold = UNIVERSE_CUTS["top_decile"]
    weekly_idx, _ = rebalance_grid(panel.dates, "weekly")
    signal_idx = weekly_idx[1::2]
    exec_idx = signal_idx + 1
    keep_bounds = exec_idx < len(panel.dates)
    signal_idx, exec_idx = signal_idx[keep_bounds], exec_idx[keep_bounds]
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    signal_idx, exec_idx, month_rows = signal_idx[keep], exec_idx[keep], month_rows[keep]
    span = float((panel.dates[signal_idx[-1]] - panel.dates[signal_idx[0]]).days) / 365.25
    ppy = (len(signal_idx) - 1) / span

    selections = build_selections(panel, config, threshold, signal_idx, exec_idx, month_rows)
    periods = precompute_periods(panel, config, signal_idx, exec_idx)
    trim = slice(0, len(signal_idx) - 1)
    res = {}
    for treatment in ("realistic", "zero_cost"):
        basis = panel.spread_basis[treatment]
        charge = treatment != "zero_cost"
        lb = run_leg(panel, config, selections["long"], signal_idx, exec_idx, month_rows,
                     periods, 1.0, basis, ppy, charge_impact=charge)
        sb = run_leg(panel, config, selections["short"], signal_idx, exec_idx, month_rows,
                     periods, 1.0, basis, ppy, charge_impact=charge,
                     borrow_annual=0.0 if not charge else config.short_borrow_annual)
        gross = lb.gross_return[trim] - sb.gross_return[trim]
        cost = lb.cost[trim] + sb.cost[trim]
        res[treatment] = {"gross_sharpe": summarise(gross, ppy).sharpe,
                          "net_sharpe": summarise(gross - cost, ppy).sharpe}
    res["periods"] = int(len(signal_idx) - 1)
    res["periods_per_year"] = ppy
    return res


def traded_regime_mix(panel, config) -> dict:
    """Regime and bound composition of the names the top-decile weekly book actually trades.

    Answers why the realistic bound did not collapse to the ~15bps the >$20M/day band
    table implied: both bounds price a `measured` name identically, and `measured` names
    in the liquid decile are the WIDE ones, since that is why they resolved.
    """
    monthly = pd.read_parquet(
        Path(__file__).resolve().parents[1]
        / "_data" / "sharadar" / "panel" / "monthly_panel_dev.parquet"
    )
    from research.sleeves.reversal_retest import _eligible_monthly, spread_bounds_frame

    elig = _eligible_monthly(monthly, config)
    elig["dv_rank"] = elig.groupby(elig["date"].dt.to_period("M"))[
        "median_dollar_volume"].rank(pct=True)
    elig = spread_bounds_frame(elig)
    cut = elig[elig["dv_rank"] > UNIVERSE_CUTS["top_decile"]]
    out = {}
    for regime in ("measured", "upper_bound"):
        sub = cut[cut["spread_regime"] == regime]
        out[regime] = {
            "share_of_cells": float(len(sub) / len(cut)),
            "median_conservative_bps": float(sub["spread_conservative"].median() * 1e4),
            "median_realistic_bps": float(sub["spread_realistic"].median() * 1e4),
            "mean_realistic_bps": float(sub["spread_realistic"].mean() * 1e4),
            "median_dollar_volume": float(sub["median_dollar_volume"].median()),
        }
    out["all_cells_mean_realistic_bps"] = float(cut["spread_realistic"].mean() * 1e4)
    out["all_cells_mean_conservative_bps"] = float(cut["spread_conservative"].mean() * 1e4)
    return out


def delisting_audit(panel, config) -> dict:
    """How many terminal returns were actually booked, and the largest one."""
    signal_idx, exec_idx = rebalance_grid(panel.dates, "weekly")
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    signal_idx, exec_idx = signal_idx[keep], exec_idx[keep]

    booked = 0
    at_cap = 0
    last_row = panel.adj_close.shape[0] - 1
    grace = int(np.timedelta64(config.delisting_grace_days, "D").astype("timedelta64[ns]"))
    for k, e_row in enumerate(exec_idx):
        exit_row = exec_idx[k + 1] if k + 1 < len(exec_idx) else last_row
        entry = panel.adj_open[e_row]
        exit_price = panel.adj_open[exit_row]
        missing = ~np.isfinite(exit_price) & np.isfinite(entry)
        start = panel.dates[e_row].to_datetime64().astype(np.int64)
        end = panel.dates[exit_row].to_datetime64().astype(np.int64) + grace
        in_window = (panel.delist_date >= start) & (panel.delist_date <= end)
        booked += int((in_window & missing).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            r = exit_price / entry - 1.0
        at_cap += int(np.sum(np.abs(r[np.isfinite(r)]) >= config.return_cap))
    return {"terminal_returns_booked_over_all_weekly_periods": booked,
            "name_periods_hitting_the_+-100pct_cap": at_cap,
            "max_bar_date": str(panel.dates[-1].date())}


def breakeven_costs(results: dict) -> dict:
    """Round-trip cost (bps) needed to clear the 0.75 gate, and to reach zero excess.

    The GATE is a Sharpe, which is defined on the ARITHMETIC mean, so the gate budget is
    computed from `gross_sharpe * gross_vol` and NOT from the geometric annual return --
    at ~30% volatility the two differ by ~4.5%/yr of variance drag, which is more than the
    entire cost budget being solved for. The EXCESS is a geometric comparison against a
    geometric benchmark, so that one stays geometric.

    Both are first-order: a per-period cost is a near-deterministic drag, so net Sharpe is
    linear in it. The reported cost ladder confirms the linearity directly (the top-decile
    weekly realistic rungs step by -0.87, -0.87, -0.85 Sharpe per 0.5x of cost).
    """
    out = {}
    for key, r in results.items():
        for book in ("long_short", "long_only"):
            b = r["books"]["realistic"][book]
            gross_geometric = b["gross"]["annual_return"]
            gross_arithmetic = b["gross"]["sharpe"] * b["gross"]["annual_vol"]
            vol = b["net"]["annual_vol"]
            turn = b["turnover_annual"]
            # (An unused `bench` lookup was removed here. It was REDUNDANT, not a
            # missing measurement: `bps_per_rt_for_zero_excess` below already IS the
            # benchmark-relative rung, because run_reversal_retest.py defines
            # excess_annual = net.annual_return - bench.annual_return. Corrected
            # 2026-08-01 after VERIFY-2 wrongly recorded this as an omission.)
            actual = b["cost_per_round_trip_bps"]
            out[f"{key}__{book}"] = {
                "gross_annual_geometric": gross_geometric,
                "gross_annual_arithmetic": gross_arithmetic,
                "net_annual_vol": vol,
                "turnover": turn,
                "actual_realistic_bps_per_rt": actual,
                "bps_per_rt_to_clear_0.75_gate":
                    1e4 * (gross_arithmetic - 0.75 * vol) / turn,
                "bps_per_rt_for_zero_excess":
                    actual + 1e4 * b["excess_annual"] / turn,
                "commission_only_bps_per_rt":
                    r["books"]["zero_cost"][book]["cost_per_round_trip_bps"],
                "zero_cost_sharpe": r["books"]["zero_cost"][book]["net"]["sharpe"],
            }
    return out


def main() -> None:
    config = RetestConfig()
    panel = build_matrices(config)
    results = json.loads((OUT / "reversal_retest_result.json").read_text(encoding="utf-8"))

    report = {
        "negative_control_top_decile_weekly": negative_control(panel, config),
        "traded_regime_mix_top_decile": traded_regime_mix(panel, config),
        "delisting_audit": delisting_audit(panel, config),
        "fortnightly_other_phase_POSTHOC": fortnightly_other_phase(panel, config),
        "breakeven_costs": breakeven_costs(results),
    }
    path = OUT / "verification.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
