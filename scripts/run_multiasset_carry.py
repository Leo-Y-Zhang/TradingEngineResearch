"""The ONE registered run of the cross-asset carry sleeve.

    .venv/Scripts/python.exe scripts/run_multiasset_carry.py

Pre-registration: ``research/sleeves/multiasset_carry_prereg.md``, written first. This
script implements that document and nothing else. It is run once; the numbers it prints
are the result whether they are good or bad.

Inputs, all gitignored: ``_data/multiasset/`` (the integrity-proven panel) and
``_data/carry/`` (``scripts/build_carry_inputs.py``). Output: derived statistics only, to
``research/sleeves/_carry_output/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.carry import (  # noqa: E402
    FX_INSTRUMENTS,
    MONTHS_PER_YEAR,
    TREND_EXCLUDE,
    backtest,
    benchmark_positions,
    carry_positions,
    decompose_pnl,
    fx_excess_returns,
    newey_west_tstat,
    ols_alpha,
    performance,
    realised_dividend_yield,
    sharpe_by_decade,
    trend_positions,
)
from research.multiasset.panel import dsr_sharpe_bar  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "_data" / "multiasset"
CARRY = REPO / "_data" / "carry"
OUT = REPO / "research" / "sleeves" / "_carry_output"

N_TRIALS = 36
COST_REALISTIC_BPS = 3.0
COST_CONSERVATIVE_BPS = 10.0
BOND_KEYS = ("US5Y_TR", "US10Y_TR", "US30Y_TR")
BOND_YIELD = {"US5Y_TR": "US5Y_YLD", "US10Y_TR": "US10Y_YLD", "US30Y_TR": "US30Y_YLD"}
CONTROL_SEEDS = (11, 23, 47, 91)
HALF_KELLY = 3.0 / 8.0


# ── universe construction ─────────────────────────────────────────────────────

def build_universe(*, unscreened: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """The 13-instrument carry universe: excess returns and the carry that ranks them."""
    suffix = "_unscreened" if unscreened else ""
    panel = pd.read_parquet(PANEL / f"returns_monthly{suffix}.parquet")
    yields = pd.read_parquet(PANEL / "yields_monthly.parquet")
    cash = pd.read_parquet(PANEL / "cash_monthly.parquet")["US_CASH_13W"]
    fx_spot = pd.read_parquet(CARRY / f"fx_spot_returns_monthly{suffix}.parquet")
    rates = pd.read_parquet(CARRY / "short_rates_monthly.parquet")

    idx = panel.index.union(fx_spot.index).sort_values()
    panel = panel.reindex(idx)
    yields = yields.reindex(idx)
    cash = cash.reindex(idx)
    fx_spot = fx_spot.reindex(idx)

    excess = pd.DataFrame(index=idx, dtype=float)
    carry = pd.DataFrame(index=idx, dtype=float)
    asset_class: dict[str, str] = {}

    # rates — carry is the term spread over the 13-week bill
    for key in BOND_KEYS:
        excess[key] = panel[key] - cash
        carry[key] = yields[BOND_YIELD[key]] - yields["US13W_YLD"]
        asset_class[key] = "rates"

    # FX — carry IS the interest differential, and it is also part of the return
    fx_excess, fx_carry = fx_excess_returns(fx_spot, rates.reindex(idx), FX_INSTRUMENTS)
    for key in fx_excess.columns:
        excess[key] = fx_excess[key]
        carry[key] = fx_carry[key]
        asset_class[key] = "fx"

    # equity — carry is the trailing realised dividend yield over the bill
    excess["SPY_EQ"] = panel["SPY"] - cash
    carry["SPY_EQ"] = realised_dividend_yield(panel["SPY"], panel["SPX"]) - yields["US13W_YLD"]
    asset_class["SPY_EQ"] = "equity"

    return excess, carry, asset_class


def build_trend_universe(*, unscreened: bool) -> pd.DataFrame:
    """The trend reference universe: panel excess returns plus the nine CIP-consistent FX."""
    suffix = "_unscreened" if unscreened else ""
    panel = pd.read_parquet(PANEL / f"returns_monthly{suffix}.parquet")
    cash = pd.read_parquet(PANEL / "cash_monthly.parquet")["US_CASH_13W"]
    fx_spot = pd.read_parquet(CARRY / f"fx_spot_returns_monthly{suffix}.parquet")
    rates = pd.read_parquet(CARRY / "short_rates_monthly.parquet")

    idx = panel.index.union(fx_spot.index).sort_values()
    panel, cash, fx_spot = panel.reindex(idx), cash.reindex(idx), fx_spot.reindex(idx)

    drop = set(TREND_EXCLUDE) | {"USDX", "EURUSD", "GBPUSD", "JPYUSD"}
    keep = [c for c in panel.columns if c not in drop]
    out = panel[keep].sub(cash, axis=0)
    fx_excess, _ = fx_excess_returns(fx_spot, rates.reindex(idx), FX_INSTRUMENTS)
    for key in fx_excess.columns:
        out[key] = fx_excess[key]
    return out


# ── reporting helpers ─────────────────────────────────────────────────────────

def concentration(pnl: pd.DataFrame) -> dict[str, object]:
    """Largest single (instrument, month) share of net P&L. One name-month was once 13%."""
    values = pnl.fillna(0.0)
    total = float(values.to_numpy().sum())
    gross_abs = float(np.abs(values.to_numpy()).sum())
    flat = values.stack()
    if flat.empty:
        return {}
    worst = flat.reindex(flat.abs().sort_values(ascending=False).index).head(1)
    stamp, key = worst.index[0]
    return {
        "max_cell_pnl": float(worst.iloc[0]),
        "max_cell_instrument": str(key),
        "max_cell_date": str(pd.Timestamp(stamp).date()),
        "max_cell_share_of_total_pnl": float(worst.iloc[0]) / total if total else float("nan"),
        "max_cell_share_of_gross_abs_pnl": abs(float(worst.iloc[0])) / gross_abs if gross_abs else float("nan"),
        "total_pnl": total,
        "gross_abs_pnl": gross_abs,
    }


def attribution(pnl: pd.DataFrame, asset_class: dict[str, str]) -> dict[str, object]:
    per_instrument = pnl.fillna(0.0).sum().sort_values()
    total = float(per_instrument.sum())
    by_class: dict[str, float] = {}
    for key, value in per_instrument.items():
        by_class.setdefault(asset_class.get(str(key), "other"), 0.0)
        by_class[asset_class.get(str(key), "other")] += float(value)
    return {
        "per_instrument_pnl": {str(k): float(v) for k, v in per_instrument.items()},
        "per_instrument_share": {str(k): (float(v) / total if total else float("nan"))
                                 for k, v in per_instrument.items()},
        "by_asset_class_pnl": by_class,
        "by_asset_class_share": {k: (v / total if total else float("nan"))
                                 for k, v in by_class.items()},
    }


def breadth(positions: pd.DataFrame, live_index: pd.Index) -> dict[str, float]:
    pos = positions.reindex(live_index).fillna(0.0)
    signs = np.sign(pos)
    flips = ((signs != signs.shift(1)) & (signs != 0) & (signs.shift(1) != 0)).sum().sum()
    years = len(live_index) / MONTHS_PER_YEAR
    held = (pos != 0).sum(axis=1)
    return {
        "sign_flips_total": int(flips),
        "sign_flips_per_year": float(flips) / years if years else float("nan"),
        "mean_instruments_held": float(held.mean()),
        "mean_gross_notional": float(pos.abs().sum(axis=1).mean()),
        "mean_net_notional": float(pos.sum(axis=1).mean()),
        "years": round(years, 2),
    }


def run_sleeve(excess: pd.DataFrame, carry: pd.DataFrame, *, keys: list[str] | None = None,
               permute_seed: int | None = None, min_instruments: int = 6) -> dict[str, object]:
    sub_excess = excess[keys] if keys else excess
    sub_carry = carry[keys] if keys else carry
    positions, vol, n_elig = carry_positions(sub_carry, sub_excess, permute_seed=permute_seed,
                                             min_instruments=min_instruments)
    real = backtest(positions, sub_excess, round_trip_bps=COST_REALISTIC_BPS)
    cons = backtest(positions, sub_excess, round_trip_bps=COST_CONSERVATIVE_BPS)
    bench_pos = benchmark_positions(sub_excess, vol, n_elig, min_instruments=min_instruments)
    bench = backtest(bench_pos, sub_excess, round_trip_bps=COST_REALISTIC_BPS)
    return {"positions": positions, "vol": vol, "n_eligible": n_elig,
            "realistic": real, "conservative": cons, "benchmark": bench,
            "excess": sub_excess, "carry": sub_carry}


def statistics_block(res: dict[str, object], asset_class: dict[str, str]) -> dict[str, object]:
    real, cons, bench = res["realistic"], res["conservative"], res["benchmark"]
    net, gross, bench_net = real["net"], real["gross"], bench["net"]

    mean_a, se_a, t_a = newey_west_tstat(net)
    diff = (net - bench_net.reindex(net.index)).dropna()
    mean_c, _se_c, t_c = newey_west_tstat(diff)

    perf_net = performance(net)
    years = perf_net["years"]
    bar = dsr_sharpe_bar(years, n_trials=N_TRIALS)

    parts = decompose_pnl(res["positions"], res["excess"], res["carry"])
    live_index = net.index

    return {
        "gross": performance(gross),
        "net_realistic": perf_net,
        "net_conservative": performance(cons["net"]),
        "benchmark": performance(bench_net),
        "statistic_A_arithmetic_active_annual": mean_a * MONTHS_PER_YEAR,
        "statistic_A_tstat": t_a,
        "statistic_A_se_monthly": se_a,
        "statistic_B_alpha_vs_own_universe": ols_alpha(net, bench_net),
        "statistic_C_arithmetic_diff_annual": mean_c * MONTHS_PER_YEAR,
        "statistic_C_tstat": t_c,
        "sharpe_by_decade_net": sharpe_by_decade(net),
        "sharpe_by_decade_gross": sharpe_by_decade(gross),
        "concentration": concentration(real["pnl"]),
        "attribution": attribution(real["pnl"], asset_class),
        "decomposition": parts,
        "breadth": breadth(res["positions"], live_index),
        "turnover_per_year": float(real["turnover"].mean()) * MONTHS_PER_YEAR,
        "cost_drag_realistic_annual": float(real["cost"].mean()) * MONTHS_PER_YEAR,
        "cost_drag_conservative_annual": float(cons["cost"].mean()) * MONTHS_PER_YEAR,
        "n_missing_return_cells": real["n_missing_return_cells"],
        "mean_n_eligible": float(pd.Series(res["n_eligible"]).reindex(live_index).mean()),
        "dsr_bar": bar,
        "clears_dsr_bar": bool(perf_net["sharpe"] >= bar),
        "half_kelly_return_at_net_sharpe": HALF_KELLY * perf_net["sharpe"] ** 2,
        "first_month": str(live_index.min().date()),
        "last_month": str(live_index.max().date()),
    }


def vol_targeted(net: pd.Series, *, target: float = 0.10, window: int = 36,
                 lo: float = 0.25, hi: float = 4.0) -> pd.Series:
    """Trailing-vol overlay, declared in advance as a SECONDARY, never the primary."""
    trailing = net.rolling(window, min_periods=24).std() * np.sqrt(MONTHS_PER_YEAR)
    scale = (target / trailing).clip(lo, hi).shift(1)
    return (net * scale).dropna()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"n_trials": N_TRIALS,
                                 "cost_bps_round_trip": {"realistic": COST_REALISTIC_BPS,
                                                         "conservative": COST_CONSERVATIVE_BPS}}

    excess, carry, asset_class = build_universe(unscreened=False)
    report["universe"] = {
        "n_instruments": int(excess.shape[1]),
        "keys": list(excess.columns),
        "asset_class": asset_class,
        "coverage": {k: {"first": str(excess[k].first_valid_index().date()),
                         "last": str(excess[k].last_valid_index().date()),
                         "n_months": int(excess[k].notna().sum())}
                     for k in excess.columns},
    }

    # ── PRIMARY ───────────────────────────────────────────────────────────────
    primary = run_sleeve(excess, carry)
    report["primary"] = statistics_block(primary, asset_class)
    primary_net = primary["realistic"]["net"]
    primary_net.to_frame("net").to_parquet(OUT / "carry_primary_net_monthly.parquet")
    primary["realistic"]["gross"].to_frame("gross").to_parquet(
        OUT / "carry_primary_gross_monthly.parquet")

    # negative control
    controls = []
    for seed in CONTROL_SEEDS:
        res = run_sleeve(excess, carry, permute_seed=seed)
        controls.append(performance(res["realistic"]["net"])["sharpe"])
    live_sharpe = report["primary"]["net_realistic"]["sharpe"]
    report["negative_control"] = {
        "seeds": list(CONTROL_SEEDS),
        "sharpes": [float(s) for s in controls],
        "mean": float(np.mean(controls)),
        "sd": float(np.std(controls, ddof=1)),
        "live_sharpe": float(live_sharpe),
        "sd_above_control": (float(live_sharpe) - float(np.mean(controls))) / float(np.std(controls, ddof=1))
        if float(np.std(controls, ddof=1)) > 0 else float("nan"),
    }

    # ── SECONDARIES ───────────────────────────────────────────────────────────
    bonds = [k for k in excess.columns if asset_class[k] == "rates"]
    fx = [k for k in excess.columns if asset_class[k] == "fx"]
    # A 3-instrument class cannot satisfy the primary's N>=6 rule, so the bonds-only
    # secondary uses N>=3 — the whole class. Declared in the pre-registration; it changes
    # nothing about the primary.
    report["secondary_bonds_only"] = statistics_block(
        run_sleeve(excess, carry, keys=bonds, min_instruments=3), asset_class)
    report["secondary_fx_only"] = statistics_block(
        run_sleeve(excess, carry, keys=fx, min_instruments=6), asset_class)

    vt = vol_targeted(primary_net)
    report["secondary_vol_targeted"] = {
        "performance": performance(vt),
        "sharpe_by_decade": sharpe_by_decade(vt),
    }

    excess_u, carry_u, _ = build_universe(unscreened=True)
    unscreened = run_sleeve(excess_u, carry_u)
    report["secondary_unscreened"] = {
        "performance": performance(unscreened["realistic"]["net"]),
        "sharpe_by_decade": sharpe_by_decade(unscreened["realistic"]["net"]),
    }

    # ── TREND REFERENCE ───────────────────────────────────────────────────────
    trend_excess = build_trend_universe(unscreened=False)
    t_pos, t_vol, t_elig = trend_positions(trend_excess)
    t_real = backtest(t_pos, trend_excess, round_trip_bps=COST_REALISTIC_BPS)
    t_cons = backtest(t_pos, trend_excess, round_trip_bps=COST_CONSERVATIVE_BPS)
    t_bench = backtest(benchmark_positions(trend_excess, t_vol, t_elig), trend_excess,
                       round_trip_bps=COST_REALISTIC_BPS)
    trend_net = t_real["net"]
    trend_net.to_frame("net").to_parquet(OUT / "trend_reference_net_monthly.parquet")
    mean_t, _se_t, t_t = newey_west_tstat(trend_net)
    report["trend_reference"] = {
        "n_instruments": int(trend_excess.shape[1]),
        "keys": list(trend_excess.columns),
        "gross": performance(t_real["gross"]),
        "net_realistic": performance(trend_net),
        "net_conservative": performance(t_cons["net"]),
        "benchmark": performance(t_bench["net"]),
        "statistic_A_arithmetic_active_annual": mean_t * MONTHS_PER_YEAR,
        "statistic_A_tstat": t_t,
        "statistic_B_alpha_vs_own_universe": ols_alpha(trend_net, t_bench["net"]),
        "sharpe_by_decade_net": sharpe_by_decade(trend_net),
        "concentration": concentration(t_real["pnl"]),
        "breadth": breadth(t_pos, trend_net.index),
        "turnover_per_year": float(t_real["turnover"].mean()) * MONTHS_PER_YEAR,
        "dsr_bar": dsr_sharpe_bar(performance(trend_net)["years"], n_trials=N_TRIALS),
    }

    # ── THE TWO-SLEEVE TEST ───────────────────────────────────────────────────
    both = pd.concat([primary_net.rename("carry"), trend_net.rename("trend")], axis=1).dropna()
    rho = float(both["carry"].corr(both["trend"]))
    s_carry = performance(both["carry"])["sharpe"]
    s_trend = performance(both["trend"])["sharpe"]

    # point-in-time risk parity: weights from each sleeve's own trailing vol, lagged
    vol_c = both["carry"].rolling(36, min_periods=24).std().shift(1)
    vol_t = both["trend"].rolling(36, min_periods=24).std().shift(1)
    inv_c, inv_t = 1.0 / vol_c, 1.0 / vol_t
    w_c = inv_c / (inv_c + inv_t)
    combo_rp = (w_c * both["carry"] + (1.0 - w_c) * both["trend"]).dropna()
    combo_ew = (0.5 * both["carry"] + 0.5 * both["trend"]).dropna()

    s_mean = float(np.mean([s_carry, s_trend]))
    formula = s_mean * np.sqrt(2.0 / (1.0 + rho)) if rho > -1 else float("nan")
    perf_rp = performance(combo_rp)
    perf_ew = performance(combo_ew)
    report["two_sleeve"] = {
        "overlap_months": int(len(both)),
        "overlap_first": str(both.index.min().date()),
        "overlap_last": str(both.index.max().date()),
        "correlation": rho,
        "carry_sharpe_over_overlap": s_carry,
        "trend_sharpe_over_overlap": s_trend,
        "formula_sharpe_equal_sleeves": float(formula),
        "formula_inputs": {"s_mean": s_mean, "N": 2, "rho": rho},
        "measured_risk_parity": perf_rp,
        "measured_risk_parity_by_decade": sharpe_by_decade(combo_rp),
        "measured_equal_weight": perf_ew,
        "half_kelly_return_measured_rp": HALF_KELLY * perf_rp["sharpe"] ** 2,
        "half_kelly_return_measured_ew": HALF_KELLY * perf_ew["sharpe"] ** 2,
        "half_kelly_return_formula": HALF_KELLY * float(formula) ** 2,
        "dsr_bar_at_overlap": dsr_sharpe_bar(len(both) / MONTHS_PER_YEAR, n_trials=N_TRIALS),
        "sharpe_needed_for_30pct_half_kelly": float(np.sqrt(0.30 / HALF_KELLY)),
    }
    combo_rp.to_frame("net").to_parquet(OUT / "two_sleeve_risk_parity_monthly.parquet")

    (OUT / "multiasset_carry_result.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ── print ─────────────────────────────────────────────────────────────────
    p = report["primary"]
    print("=" * 78)
    print(f"CROSS-ASSET CARRY  {p['first_month']} -> {p['last_month']}  "
          f"({p['net_realistic']['years']} yr, {p['net_realistic']['n_months']} months)")
    print("=" * 78)
    print(f"instruments {report['universe']['n_instruments']}  mean eligible/month "
          f"{p['mean_n_eligible']:.1f}")
    print(f"gross  Sharpe {p['gross']['sharpe']:+.3f}  vol {p['gross']['annual_vol']:.2%}")
    print(f"net(3bp) Sharpe {p['net_realistic']['sharpe']:+.3f}  vol "
          f"{p['net_realistic']['annual_vol']:.2%}  arith {p['net_realistic']['arithmetic_annual']:+.2%}")
    print(f"net(10bp) Sharpe {p['net_conservative']['sharpe']:+.3f}")
    print(f"A: arithmetic active {p['statistic_A_arithmetic_active_annual']:+.3%}/yr  "
          f"t = {p['statistic_A_tstat']:+.3f}")
    print(f"B: alpha vs own universe {p['statistic_B_alpha_vs_own_universe']['alpha_annual']:+.3%}/yr  "
          f"t = {p['statistic_B_alpha_vs_own_universe']['t_alpha']:+.3f}  "
          f"beta {p['statistic_B_alpha_vs_own_universe']['beta']:+.3f}")
    print(f"C: vs own-universe benchmark {p['statistic_C_arithmetic_diff_annual']:+.3%}/yr  "
          f"t = {p['statistic_C_tstat']:+.3f}")
    print(f"benchmark Sharpe {p['benchmark']['sharpe']:+.3f}  "
          f"arith {p['benchmark']['arithmetic_annual']:+.2%}")
    print(f"maxDD {p['net_realistic']['max_drawdown']:.2%}  skew "
          f"{p['net_realistic']['skew']:+.2f}  worst month {p['net_realistic']['worst_month']:.2%}")
    print(f"DSR bar (n={N_TRIALS}, {p['net_realistic']['years']}yr) {p['dsr_bar']:.3f}  "
          f"clears: {p['clears_dsr_bar']}")
    print(f"half-Kelly return at net Sharpe: {p['half_kelly_return_at_net_sharpe']:.2%}/yr")
    print("decades net: " + "  ".join(
        f"{k} {v['sharpe']:+.2f}" for k, v in p["sharpe_by_decade_net"].items()))
    print(f"turnover {p['turnover_per_year']:.2f}/yr  cost drag "
          f"{p['cost_drag_realistic_annual']:.3%} (3bp) / {p['cost_drag_conservative_annual']:.3%} (10bp)")
    print(f"breadth {p['breadth']['sign_flips_per_year']:.1f} flips/yr  "
          f"held {p['breadth']['mean_instruments_held']:.1f}  gross notional "
          f"{p['breadth']['mean_gross_notional']:.2f}  net {p['breadth']['mean_net_notional']:+.2f}")
    print(f"decomposition: accrual {p['decomposition']['accrual_share']:+.1%}  "
          f"price {p['decomposition']['price_share']:+.1%}")
    print(f"concentration: max cell {p['concentration']['max_cell_share_of_total_pnl']:+.2%} of "
          f"net P&L ({p['concentration']['max_cell_instrument']} "
          f"{p['concentration']['max_cell_date']})")
    print("by class: " + "  ".join(
        f"{k} {v:+.1%}" for k, v in p["attribution"]["by_asset_class_share"].items()))
    nc = report["negative_control"]
    print(f"negative control: {nc['mean']:+.3f} +/- {nc['sd']:.3f}  live "
          f"{nc['live_sharpe']:+.3f}  ({nc['sd_above_control']:+.1f} sd)")
    print(f"missing return cells under a live position: {p['n_missing_return_cells']}")

    print("-" * 78)
    for name in ("secondary_bonds_only", "secondary_fx_only"):
        s = report[name]
        if not s:
            continue
        print(f"{name}: {s['first_month']}->{s['last_month']} net Sharpe "
              f"{s['net_realistic']['sharpe']:+.3f}  gross {s['gross']['sharpe']:+.3f}  "
              f"A {s['statistic_A_arithmetic_active_annual']:+.2%}/yr t={s['statistic_A_tstat']:+.2f}")
    print(f"secondary_vol_targeted: net Sharpe "
          f"{report['secondary_vol_targeted']['performance']['sharpe']:+.3f}")
    print(f"secondary_unscreened:   net Sharpe "
          f"{report['secondary_unscreened']['performance']['sharpe']:+.3f}")

    print("-" * 78)
    t = report["trend_reference"]
    print(f"TREND REFERENCE ({t['n_instruments']} instruments) "
          f"{t['net_realistic']['n_months']} months, {t['net_realistic']['years']}yr")
    print(f"  gross Sharpe {t['gross']['sharpe']:+.3f}  net(3bp) {t['net_realistic']['sharpe']:+.3f}"
          f"  net(10bp) {t['net_conservative']['sharpe']:+.3f}")
    print(f"  A arithmetic active {t['statistic_A_arithmetic_active_annual']:+.3%}/yr "
          f"t={t['statistic_A_tstat']:+.2f}  maxDD {t['net_realistic']['max_drawdown']:.2%}")
    print("  decades: " + "  ".join(
        f"{k} {v['sharpe']:+.2f}" for k, v in t["sharpe_by_decade_net"].items()))

    print("-" * 78)
    c = report["two_sleeve"]
    print(f"TWO-SLEEVE TEST over {c['overlap_months']} overlapping months "
          f"({c['overlap_first']} -> {c['overlap_last']})")
    print(f"  rho(carry, trend) = {c['correlation']:+.4f}")
    print(f"  carry {c['carry_sharpe_over_overlap']:+.3f}   trend {c['trend_sharpe_over_overlap']:+.3f}")
    print(f"  formula S = s*sqrt(N/(1+(N-1)rho)) = {c['formula_sharpe_equal_sleeves']:+.3f}"
          f"   -> half-Kelly {c['half_kelly_return_formula']:.2%}/yr")
    print(f"  MEASURED risk-parity combo Sharpe {c['measured_risk_parity']['sharpe']:+.3f}"
          f"  -> half-Kelly {c['half_kelly_return_measured_rp']:.2%}/yr")
    print(f"  MEASURED equal-weight combo Sharpe {c['measured_equal_weight']['sharpe']:+.3f}"
          f"  -> half-Kelly {c['half_kelly_return_measured_ew']:.2%}/yr")
    print(f"  combo maxDD {c['measured_risk_parity']['max_drawdown']:.2%}  "
          f"decades: " + "  ".join(
              f"{k} {v['sharpe']:+.2f}" for k, v in c["measured_risk_parity_by_decade"].items()))
    print(f"  DSR bar at overlap {c['dsr_bar_at_overlap']:.3f}   "
          f"Sharpe needed for 30%/yr at half Kelly {c['sharpe_needed_for_30pct_half_kelly']:.3f}")
    print("=" * 78)
    print(f"wrote {OUT / 'multiasset_carry_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
