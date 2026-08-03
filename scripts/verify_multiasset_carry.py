"""Adversarial verification of the cross-asset carry result. Tries to KILL it.

    .venv/Scripts/python.exe scripts/verify_multiasset_carry.py

The registered run is finished; this script does not change it. It attacks the result
along the six lines that have killed results in this programme before, plus two that are
specific to carry:

1. **Lookahead.** Re-run truncated and require the surviving positions to be identical.
2. **Disguised beta.** A cross-asset book that is persistently long duration in a
   40-year bond bull market is a beta bet wearing a carry hat. Regress on the actual
   factors, not just on the own-universe basket.
3. **Structural tilt.** Mean position per instrument and per class: is the "signal"
   just "the US curve is usually upward sloping and the yen usually yields least"?
4. **Era dependence.** Rolling 5-year Sharpe and leave-one-decade-out.
5. **Concentration.** Already flagged; here it is re-derived and stress-tested by
   deleting the worst cells.
6. **The accrual is 98% of P&L.** That makes the whole result an accounting spread that
   an unmodelled financing cost can erase. Price the two that are real and unmodelled:
   the CIP cross-currency basis, and a bid-ask on the differential itself.
7. **The variance-drag trap** — arithmetic vs geometric, restated explicitly.
8. **The composition puzzle:** rates contribute 46% of P&L while a bonds-only sleeve
   LOSES money. Decompose the cross-class bet that produces that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.carry import (  # noqa: E402
    MONTHS_PER_YEAR,
    backtest,
    carry_positions,
    newey_west_tstat,
    ols_alpha,
    performance,
)
from scripts.run_multiasset_carry import (  # noqa: E402
    COST_REALISTIC_BPS,
    OUT,
    PANEL,
    build_universe,
)

HALF_KELLY = 3.0 / 8.0


def main() -> int:
    excess, carry, asset_class = build_universe(unscreened=False)
    pos, vol, n_elig = carry_positions(carry, excess)
    res = backtest(pos, excess, round_trip_bps=COST_REALISTIC_BPS)
    net = res["net"]
    out: dict[str, object] = {}

    print("=" * 78)
    print("ADVERSARIAL VERIFICATION — cross-asset carry")
    print("=" * 78)

    # ── 1. lookahead ─────────────────────────────────────────────────────────
    cut = pd.Timestamp("2015-12-31")
    pos_t, _, _ = carry_positions(carry.loc[:cut], excess.loc[:cut])
    same = np.allclose(pos.loc[:cut].to_numpy(), pos_t.to_numpy(), equal_nan=True)
    out["lookahead_identical_on_truncation"] = bool(same)
    print(f"1. lookahead: positions through {cut.date()} identical when the future is "
          f"deleted: {same}")

    # ── 2. disguised beta ────────────────────────────────────────────────────
    panel = pd.read_parquet(PANEL / "returns_monthly.parquet")
    cash = pd.read_parquet(PANEL / "cash_monthly.parquet")["US_CASH_13W"]
    factors = pd.DataFrame({
        "duration_US10Y": (panel["US10Y_TR"] - cash),
        "equity_SPY": (panel["SPY"] - cash),
        "dollar_USDX": panel["USDX"],
        "commodity_DBC": (panel["DBC"] - cash),
    }).reindex(net.index)
    betas = {}
    for name in factors.columns:
        betas[name] = ols_alpha(net, factors[name])
        print(f"2. vs {name:15s} beta {betas[name]['beta']:+.3f}  "
              f"alpha {betas[name]['alpha_annual']:+.3%}/yr  t {betas[name]['t_alpha']:+.2f}")
    # joint regression
    joint = pd.concat([net.rename("y"), factors], axis=1).dropna()
    X = np.column_stack([np.ones(len(joint)), joint[factors.columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, joint["y"].to_numpy(), rcond=None)
    resid = pd.Series(joint["y"].to_numpy() - X @ coef, index=joint.index)
    _m, _se, t_joint = newey_west_tstat(resid + coef[0])
    out["single_factor_betas"] = betas
    out["joint_alpha_annual"] = float(coef[0] * MONTHS_PER_YEAR)
    out["joint_alpha_tstat"] = float(t_joint)
    out["joint_betas"] = {n: float(c) for n, c in zip(factors.columns, coef[1:])}
    print(f"   JOINT 4-factor alpha {coef[0]*MONTHS_PER_YEAR:+.3%}/yr  t {t_joint:+.2f}  "
          f"betas " + " ".join(f"{n.split('_')[0]} {c:+.2f}"
                               for n, c in zip(factors.columns, coef[1:])))

    # ── 3. structural tilt ───────────────────────────────────────────────────
    live_pos = pos.reindex(net.index)
    mean_pos = live_pos.mean()
    frac_long = (live_pos > 0).mean()
    by_class_pos: dict[str, float] = {}
    for key in live_pos.columns:
        by_class_pos.setdefault(asset_class[key], 0.0)
        by_class_pos[asset_class[key]] += float(mean_pos[key])
    out["mean_position"] = {k: float(v) for k, v in mean_pos.items()}
    out["fraction_of_months_long"] = {k: float(v) for k, v in frac_long.items()}
    out["mean_position_by_class"] = by_class_pos
    print("3. structural tilt — mean position (fraction of months long):")
    for key in live_pos.columns:
        print(f"     {key:10s} {mean_pos[key]:+.3f}  ({frac_long[key]:5.1%} long)")
    print("     by class: " + "  ".join(f"{k} {v:+.3f}" for k, v in by_class_pos.items()))

    # a constant book fixed at the average position — how much of the P&L is just the tilt?
    static = pd.DataFrame(np.tile(mean_pos.to_numpy(), (len(pos), 1)),
                          index=pos.index, columns=pos.columns)
    static_res = backtest(static, excess, round_trip_bps=COST_REALISTIC_BPS)
    static_perf = performance(static_res["net"].reindex(net.index).dropna())
    dyn = (net - static_res["net"].reindex(net.index)).dropna()
    dyn_perf = performance(dyn)
    out["static_tilt_performance"] = static_perf
    out["dynamic_residual_performance"] = dyn_perf
    print(f"   STATIC average book: Sharpe {static_perf['sharpe']:+.3f}  "
          f"arith {static_perf['arithmetic_annual']:+.3%}/yr")
    print(f"   DYNAMIC residual   : Sharpe {dyn_perf['sharpe']:+.3f}  "
          f"arith {dyn_perf['arithmetic_annual']:+.3%}/yr  "
          f"(t {newey_west_tstat(dyn)[2]:+.2f})")

    # The static book above uses the FULL-SAMPLE mean position, which is hindsight. The
    # honest version uses an EXPANDING mean — the average book you could actually have
    # held, knowing only the past. If THAT captures the P&L, the carry signal's content is
    # its cross-sectional level, not its time variation.
    expanding = pos.reindex(net.index).expanding().mean().shift(1)
    exp_res = backtest(expanding.reindex(pos.index).fillna(0.0), excess,
                       round_trip_bps=COST_REALISTIC_BPS)
    exp_net = exp_res["net"].reindex(net.index).dropna()
    exp_perf = performance(exp_net)
    exp_resid = (net - exp_res["net"].reindex(net.index)).dropna()
    out["expanding_static_performance"] = exp_perf
    out["expanding_dynamic_residual"] = performance(exp_resid)
    out["expanding_dynamic_residual_tstat"] = float(newey_west_tstat(exp_resid)[2])
    print(f"   EXPANDING static book (point-in-time): Sharpe {exp_perf['sharpe']:+.3f}  "
          f"arith {exp_perf['arithmetic_annual']:+.3%}/yr")
    print(f"   residual over it     : Sharpe {performance(exp_resid)['sharpe']:+.3f}  "
          f"arith {performance(exp_resid)['arithmetic_annual']:+.3%}/yr  "
          f"t {newey_west_tstat(exp_resid)[2]:+.2f}")

    # ── 4. era dependence ────────────────────────────────────────────────────
    roll = net.rolling(60).mean() / net.rolling(60).std() * np.sqrt(MONTHS_PER_YEAR)
    out["rolling_5y_sharpe"] = {"min": float(roll.min()), "max": float(roll.max()),
                                "median": float(roll.median()),
                                "pct_negative": float((roll < 0).mean())}
    print(f"4. rolling 5y Sharpe: min {roll.min():+.2f}  median {roll.median():+.2f}  "
          f"max {roll.max():+.2f}  negative {(roll < 0).mean():.1%} of windows")
    loo = {}
    for dec in sorted({(d.year // 10) * 10 for d in net.index}):
        keep = net[(pd.DatetimeIndex(net.index).year // 10) * 10 != dec]
        loo[f"drop_{dec}s"] = performance(keep)["sharpe"] if len(keep) > 24 else float("nan")
        print(f"   leave-out {dec}s -> Sharpe {loo[f'drop_{dec}s']:+.3f}  (n={len(keep)})")
    out["leave_one_decade_out_sharpe"] = {k: float(v) for k, v in loo.items()}

    # ── 5. concentration stress ──────────────────────────────────────────────
    pnl = res["pnl"].fillna(0.0)
    flat = pnl.stack()
    ordered = flat.reindex(flat.abs().sort_values(ascending=False).index)
    total = float(flat.sum())
    trimmed = {}
    for k in (1, 5, 10):
        drop_idx = ordered.head(k).index
        trimmed_pnl = pnl.copy()
        for stamp, key in drop_idx:
            trimmed_pnl.loc[stamp, key] = 0.0
        series = trimmed_pnl.sum(axis=1) - res["cost"].reindex(pnl.index).fillna(0.0)
        trimmed[f"drop_worst_{k}"] = performance(series.dropna())["sharpe"]
        print(f"5. drop the {k:2d} largest |cells|: Sharpe "
              f"{trimmed[f'drop_worst_{k}']:+.3f}  (top cell = "
              f"{float(ordered.iloc[0])/total:+.1%} of net P&L)")
    out["concentration_stress"] = {k: float(v) for k, v in trimmed.items()}
    out["top10_cells"] = [
        {"date": str(pd.Timestamp(s).date()), "instrument": str(k), "pnl": float(v),
         "share_of_total": float(v) / total}
        for (s, k), v in ordered.head(10).items()
    ]

    # ── 6. the accrual is the whole result — price what is unmodelled ────────
    fx_keys = [k for k, v in asset_class.items() if v == "fx"]
    fx_gross = live_pos[fx_keys].abs().sum(axis=1)
    print(f"6. FX gross notional mean {fx_gross.mean():.3f} of a {live_pos.abs().sum(axis=1).mean():.3f} book")
    basis_tests = {}
    for bps_per_year in (10.0, 25.0, 50.0, 100.0):
        drag = fx_gross * (bps_per_year / 10000.0) / MONTHS_PER_YEAR
        stressed = (net - drag.reindex(net.index).fillna(0.0)).dropna()
        p = performance(stressed)
        basis_tests[f"{bps_per_year:.0f}bps_per_year_on_fx_notional"] = {
            "sharpe": p["sharpe"], "arithmetic_annual": p["arithmetic_annual"],
            "tstat": newey_west_tstat(stressed)[2]}
        print(f"   CIP basis / forward drag {bps_per_year:5.0f} bps/yr on FX notional -> "
              f"Sharpe {p['sharpe']:+.3f}  arith {p['arithmetic_annual']:+.3%}/yr  "
              f"t {newey_west_tstat(stressed)[2]:+.2f}")
    out["cip_basis_stress"] = basis_tests

    # what the price leg alone did
    car = carry.reindex(index=excess.index, columns=pos.columns).fillna(0.0)
    accrual = (pos * car / MONTHS_PER_YEAR).shift(1).sum(axis=1).reindex(net.index)
    price = (net + res["cost"].reindex(net.index).fillna(0.0)) - accrual
    out["accrual_leg"] = performance(accrual.dropna())
    out["price_leg"] = performance(price.dropna())
    print(f"   accrual leg: {accrual.mean()*12:+.3%}/yr, vol {accrual.std()*np.sqrt(12):.3%}")
    print(f"   price   leg: {price.mean()*12:+.3%}/yr, vol {price.std()*np.sqrt(12):.3%}, "
          f"t {newey_west_tstat(price.dropna())[2]:+.2f}")

    # ── 7. variance drag, restated ───────────────────────────────────────────
    p_net = performance(net)
    out["arithmetic_vs_geometric"] = {
        "arithmetic_annual": p_net["arithmetic_annual"],
        "geometric_annual": p_net["geometric_annual"],
        "variance_drag": p_net["arithmetic_annual"] - p_net["geometric_annual"],
    }
    print(f"7. arithmetic {p_net['arithmetic_annual']:+.3%}/yr vs geometric "
          f"{p_net['geometric_annual']:+.3%}/yr (drag "
          f"{p_net['arithmetic_annual']-p_net['geometric_annual']:+.3%})")

    # ── 8. the composition puzzle ────────────────────────────────────────────
    bonds = [k for k, v in asset_class.items() if v == "rates"]
    fx = [k for k, v in asset_class.items() if v == "fx"]
    class_pnl = {}
    for name, keys in (("rates", bonds), ("fx", fx), ("equity", ["SPY_EQ"])):
        leg = res["pnl"][keys].fillna(0.0).sum(axis=1).reindex(net.index)
        p = performance(leg)
        class_pnl[name] = {"sharpe": p["sharpe"], "arithmetic_annual": p["arithmetic_annual"],
                           "tstat": newey_west_tstat(leg)[2],
                           "mean_net_position": float(live_pos[keys].sum(axis=1).mean())}
        print(f"8. {name:7s} leg alone: arith {p['arithmetic_annual']:+.3%}/yr  "
              f"Sharpe {p['sharpe']:+.3f}  t {newey_west_tstat(leg)[2]:+.2f}  "
              f"mean net position {live_pos[keys].sum(axis=1).mean():+.3f}")
    out["per_class_leg"] = class_pnl

    # ── 9. how many such sleeves would the target actually need? ─────────────
    # S = s * sqrt(N / (1 + (N-1) rho)) solved for N at S = 0.894 (30%/yr, half Kelly).
    two = json.loads((OUT / "multiasset_carry_result.json").read_text(encoding="utf-8"))["two_sleeve"]
    rho = float(two["correlation"])
    target = float(np.sqrt(0.30 / HALF_KELLY))
    needed = {}
    for label, s in (("carry_as_reported", float(p_net["sharpe"])),
                     ("carry_after_50bps_basis", basis_tests["50bps_per_year_on_fx_notional"]["sharpe"]),
                     ("carry_ex_2010s", float(loo["drop_2010s"])),
                     ("trend_reference_over_overlap", float(two["trend_sharpe_over_overlap"]))):
        k = (target / s) ** 2 if s > 0 else float("inf")
        n = k * (1.0 - rho) / (1.0 - k * rho) if (1.0 - k * rho) > 0 else float("inf")
        ceiling = s / np.sqrt(rho) if rho > 0 else float("inf")
        needed[label] = {"sleeve_sharpe": s, "n_sleeves_needed": float(n),
                         "ceiling_as_N_to_infinity": float(ceiling)}
        print(f"9. at sleeve Sharpe {s:+.3f} and rho {rho:+.3f}: "
              f"{'IMPOSSIBLE' if not np.isfinite(n) or n < 0 else f'{n:.1f} sleeves'} "
              f"needed for S=0.894; ceiling at N->inf = {ceiling:.2f}")
    out["sleeves_needed_for_30pct"] = needed
    out["rho_used"] = rho

    # ── verdict inputs ───────────────────────────────────────────────────────
    print("-" * 78)
    print(f"net Sharpe {p_net['sharpe']:+.4f} -> half-Kelly "
          f"{HALF_KELLY*p_net['sharpe']**2:.2%}/yr; 30%/yr needs 0.894")
    (OUT / "multiasset_carry_verification.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT / 'multiasset_carry_verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
