"""Run the pre-registered multi-asset trend sleeve ONCE and print everything.

    .venv/Scripts/python.exe -m scripts.run_multiasset_trend

Pre-registration: research/sleeves/multiasset_trend_prereg.md
Writes research/sleeves/_multiasset_trend/result.json (derived statistics only -- no
vendor rows, per the programme's data-licence limits).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.multiasset_trend import (
    COST_BRACKETS,
    LOOKBACKS,
    MONTHS,
    PRIMARY_UNIVERSE,
    SPLIT_DATE,
    VOL_TARGETS,
    TrendConfig,
    active_report,
    annual_sharpe,
    concentration,
    decade_sharpe,
    effective_n,
    kelly_report,
    load_excess_panel,
    max_drawdown,
    newey_west_tstat,
    run_trend,
)

OUT = Path("research/sleeves/_multiasset_trend")
N_TRIALS_ANCHOR = 32
N_TRIALS_HONEST = 36


def _fmt(v, nd=3):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def summarise(series: pd.Series) -> dict:
    return {
        "months": int(len(series)),
        "sharpe": annual_sharpe(series),
        "mean_annual": float(series.mean() * MONTHS),
        "vol_annual": float(series.std(ddof=1) * math.sqrt(MONTHS)),
        "max_drawdown": max_drawdown(series),
        "worst_month": float(series.min()),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "tstat": newey_west_tstat(series),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"universe": list(PRIMARY_UNIVERSE), "lookbacks": list(LOOKBACKS)}

    x, interior = load_excess_panel()
    report["panel"] = {
        "instruments": len(PRIMARY_UNIVERSE),
        "first_month": str(x.index.min().date()),
        "last_month": str(x.index.max().date()),
        "interior_nulls": int(interior.to_numpy().sum()),
    }

    print("=" * 78)
    print("MULTI-ASSET TIME-SERIES MOMENTUM -- pre-registered single run")
    print("=" * 78)

    # ── PRIMARY at every vol target ───────────────────────────────────────────
    primary_runs: dict[float, object] = {}
    for tgt in VOL_TARGETS:
        primary_runs[tgt] = run_trend(TrendConfig(name="PRIMARY"), vol_target=tgt,
                                      x=x, interior=interior)

    base = primary_runs[0.10]
    span_years = len(base.gross) / MONTHS
    report["sample"] = {
        "first": str(base.gross.index.min().date()),
        "last": str(base.gross.index.max().date()),
        "months": int(len(base.gross)),
        "years": span_years,
    }
    print(f"\nSample: {report['sample']['first']} -> {report['sample']['last']}  "
          f"({len(base.gross)} months, {span_years:.1f} years)")

    counts = base.eligible_count.reindex(base.gross.index)
    report["breadth"] = {
        "mean_eligible_instruments": float(counts.mean()),
        "min_eligible": int(counts.min()),
        "max_eligible": int(counts.max()),
        "bets_per_year_nominal": float(counts.mean() * MONTHS),
        "rebalances_per_year": MONTHS,
    }

    full_period = x.loc["2007-01-01":].dropna(axis=1, how="all")
    corr_full = full_period.corr()
    report["breadth"]["effective_n_2007plus"] = effective_n(corr_full)
    report["breadth"]["effective_n_full_sample"] = effective_n(x.corr())
    report["breadth"]["bets_per_year_effective"] = float(
        report["breadth"]["effective_n_2007plus"] * MONTHS)
    print(f"Breadth: mean {counts.mean():.1f} live instruments "
          f"({int(counts.min())}-{int(counts.max())}), "
          f"nominal {counts.mean() * MONTHS:.0f} bets/yr, "
          f"effective N {report['breadth']['effective_n_2007plus']:.2f} "
          f"=> {report['breadth']['bets_per_year_effective']:.0f} effective bets/yr")

    # DSR bars at the realised sample length
    bars = {
        f"n_trials_{n}": dsr_sharpe_bar(span_years, n_trials=n)
        for n in (N_TRIALS_ANCHOR, N_TRIALS_HONEST)
    }
    report["dsr_bar"] = bars
    print(f"DSR>=0.95 bar at {span_years:.1f}y: "
          f"n=32 -> {bars['n_trials_32']:.3f}, n=36 -> {bars['n_trials_36']:.3f}")

    # ── The headline table ────────────────────────────────────────────────────
    print("\n--- PRIMARY: Sharpe by vol target and cost bracket ---")
    print(f"{'target':>7} {'gross':>8} {'2bps':>8} {'10bps':>8} {'cost2%':>8} "
          f"{'cost10%':>8} {'lev':>7} {'cap%':>6} {'turn/yr':>8}")
    vt_table = {}
    for tgt in VOL_TARGETS:
        r = primary_runs[tgt]
        row = {
            "gross_sharpe": annual_sharpe(r.gross),
            "net_sharpe_2bps": annual_sharpe(r.net["2bps"]),
            "net_sharpe_10bps": annual_sharpe(r.net["10bps"]),
            "gross_return_annual": float(r.gross.mean() * MONTHS),
            "net_return_annual_2bps": float(r.net["2bps"].mean() * MONTHS),
            "net_return_annual_10bps": float(r.net["10bps"].mean() * MONTHS),
            "realised_vol": float(r.gross.std(ddof=1) * math.sqrt(MONTHS)),
            "cost_annual_2bps": float((r.gross - r.net["2bps"]).mean() * MONTHS),
            "cost_annual_10bps": float((r.gross - r.net["10bps"]).mean() * MONTHS),
            "mean_gross_leverage": float(r.gross_leverage.mean()),
            "max_gross_leverage": float(r.gross_leverage.max()),
            "cap_binding_frac": float(r.cap_binding.mean()),
            "turnover_per_year": float(r.turnover.mean() * MONTHS),
            "mean_net_exposure": float(r.net_exposure.mean()),
            "max_drawdown_net_10bps": max_drawdown(r.net["10bps"]),
        }
        vt_table[f"{tgt:.0%}"] = row
        print(f"{tgt:>7.0%} {row['gross_sharpe']:>8.3f} {row['net_sharpe_2bps']:>8.3f} "
              f"{row['net_sharpe_10bps']:>8.3f} {row['cost_annual_2bps']:>8.2%} "
              f"{row['cost_annual_10bps']:>8.2%} {row['mean_gross_leverage']:>7.2f} "
              f"{row['cap_binding_frac']:>6.1%} {row['turnover_per_year']:>8.2f}")
    report["vol_targets"] = vt_table

    # ── Distribution + benchmark + active return ──────────────────────────────
    print("\n--- PRIMARY @ 20% vol target: distribution ---")
    ref = primary_runs[0.20]
    report["primary_20pct"] = {
        "gross": summarise(ref.gross),
        "net_2bps": summarise(ref.net["2bps"]),
        "net_10bps": summarise(ref.net["10bps"]),
        "bench_gross": summarise(ref.bench_gross),
        "bench_net_10bps": summarise(ref.bench_net["10bps"]),
    }
    for label in ("gross", "net_2bps", "net_10bps", "bench_gross", "bench_net_10bps"):
        s = report["primary_20pct"][label]
        print(f"  {label:>16}: Sharpe {s['sharpe']:>7.3f}  ret {s['mean_annual']:>7.2%}  "
              f"vol {s['vol_annual']:>6.2%}  maxDD {s['max_drawdown']:>7.2%}  "
              f"skew {s['skew']:>6.2f}  worst {s['worst_month']:>7.2%}")

    print("\n--- ACTIVE RETURN vs equal-weight long-only of the SAME instruments ---")
    active = {}
    for tgt in VOL_TARGETS:
        r = primary_runs[tgt]
        for cost in COST_BRACKETS:
            key = f"{tgt:.0%}_{cost}"
            active[key] = active_report(r.net[cost], r.bench_net[cost])
    report["active"] = active
    print(f"{'config':>12} {'arith act':>10} {'t(NW)':>7} {'jensen a':>9} {'t':>7} "
          f"{'beta':>6} {'geo exc':>9} {'var drag':>9}")
    for key, a in active.items():
        print(f"{key:>12} {a['arith_active_annual']:>10.2%} {a['arith_active_tstat']:>7.2f} "
              f"{a['jensen_alpha_annual']:>9.2%} {a['jensen_alpha_tstat']:>7.2f} "
              f"{a['jensen_beta']:>6.2f} {a['geometric_excess_annual']:>9.2%} "
              f"{a['variance_drag_annual']:>9.2%}")

    # ── Per-decade ────────────────────────────────────────────────────────────
    print("\n--- SHARPE PER DECADE (PRIMARY @20% vol) ---")
    dec = {
        "gross": decade_sharpe(ref.gross),
        "net_10bps": decade_sharpe(ref.net["10bps"]),
        "bench_net_10bps": decade_sharpe(ref.bench_net["10bps"]),
        "active_10bps": decade_sharpe(
            (ref.net["10bps"] - ref.bench_net["10bps"]).dropna()),
    }
    report["decades"] = dec
    print(f"{'decade':>8} {'mo':>4} {'gross':>8} {'net10':>8} {'bench':>8} {'active ret':>11}")
    for d in sorted(dec["net_10bps"]):
        g = dec["gross"][d]["sharpe"]
        n10 = dec["net_10bps"][d]["sharpe"]
        bn = dec["bench_net_10bps"][d]["sharpe"]
        ar = dec["active_10bps"][d]["mean_annual"]
        print(f"{d:>8} {dec['net_10bps'][d]['months']:>4} {g:>8.3f} {n10:>8.3f} "
              f"{bn:>8.3f} {ar:>11.2%}")

    # ── Pre/post 2009 ─────────────────────────────────────────────────────────
    print("\n--- PRE-2009 vs 2009+ (PRIMARY @20% vol, 10bps) ---")
    split = {}
    for label, sl in (("pre2009", slice(None, "2008-12-31")), ("post2009", slice(SPLIT_DATE, None))):
        s = ref.net["10bps"].loc[sl]
        b = ref.bench_net["10bps"].loc[sl]
        split[label] = {**summarise(s), **{f"active_{k}": v for k, v in
                                           active_report(s, b).items()}}
        print(f"  {label:>9}: n={len(s):>4}  net Sharpe {annual_sharpe(s):>7.3f}  "
              f"arith active {split[label]['active_arith_active_annual']:>7.2%} "
              f"(t {split[label]['active_arith_active_tstat']:>5.2f})")
    report["split_2009"] = split

    # ── Concentration ─────────────────────────────────────────────────────────
    conc = concentration(ref.pnl.loc[ref.gross.index])
    report["concentration"] = conc
    print(f"\nP&L concentration: top cell {conc['top_cell']} = "
          f"{conc['top_cell_share']:.2%} of total P&L; "
          f"top instrument {conc['top_instrument']} = {conc['top_instrument_share']:.2%}")

    # ── Single-lookback books ─────────────────────────────────────────────────
    print("\n--- SINGLE-LOOKBACK SUB-BOOKS (20% vol, 10bps) ---")
    sub = {}
    sub_series = {}
    for lb in LOOKBACKS:
        r = run_trend(TrendConfig(name=f"LB{lb}", lookbacks=(lb,)), vol_target=0.20,
                      x=x, interior=interior)
        sub[f"lb{lb}"] = {
            "gross_sharpe": annual_sharpe(r.gross),
            "net_sharpe_10bps": annual_sharpe(r.net["10bps"]),
            "turnover_per_year": float(r.turnover.mean() * MONTHS),
            "arith_active": active_report(r.net["10bps"], r.bench_net["10bps"])[
                "arith_active_annual"],
        }
        sub_series[f"lb{lb}"] = r.net["10bps"]
        print(f"  lookback {lb:>2}m: gross {sub[f'lb{lb}']['gross_sharpe']:>7.3f}  "
              f"net10 {sub[f'lb{lb}']['net_sharpe_10bps']:>7.3f}  "
              f"turn/yr {sub[f'lb{lb}']['turnover_per_year']:>5.2f}  "
              f"active {sub[f'lb{lb}']['arith_active']:>7.2%}")
    sub_corr = pd.DataFrame(sub_series).corr()
    report["single_lookback"] = sub
    report["lookback_correlation"] = sub_corr.round(4).to_dict()
    print("  sub-book correlation:")
    print(sub_corr.round(3).to_string().replace("\n", "\n    "))

    # ── SENSITIVITY-B: block risk parity ──────────────────────────────────────
    print("\n--- SENSITIVITY-B: block risk parity (equity/rates/commodity/fx) ---")
    rb = run_trend(TrendConfig(name="SENSITIVITY-B", block_risk_parity=True),
                   vol_target=0.20, x=x, interior=interior)
    ab = active_report(rb.net["10bps"], rb.bench_net["10bps"])
    report["sensitivity_b"] = {
        "gross_sharpe": annual_sharpe(rb.gross),
        "net_sharpe_2bps": annual_sharpe(rb.net["2bps"]),
        "net_sharpe_10bps": annual_sharpe(rb.net["10bps"]),
        "active": ab,
        "decades": decade_sharpe(rb.net["10bps"]),
        "max_drawdown_net_10bps": max_drawdown(rb.net["10bps"]),
    }
    print(f"  gross {annual_sharpe(rb.gross):.3f}  net2 {annual_sharpe(rb.net['2bps']):.3f}  "
          f"net10 {annual_sharpe(rb.net['10bps']):.3f}  "
          f"arith active {ab['arith_active_annual']:.2%} (t {ab['arith_active_tstat']:.2f})")

    # ── Unscreened panel ──────────────────────────────────────────────────────
    xu, iu = load_excess_panel(unscreened=True)
    ru = run_trend(TrendConfig(name="UNSCREENED", unscreened=True), vol_target=0.20,
                   x=xu, interior=iu)
    au = active_report(ru.net["10bps"], ru.bench_net["10bps"])
    report["unscreened"] = {
        "gross_sharpe": annual_sharpe(ru.gross),
        "net_sharpe_10bps": annual_sharpe(ru.net["10bps"]),
        "arith_active_annual": au["arith_active_annual"],
        "arith_active_tstat": au["arith_active_tstat"],
    }
    print(f"\nUnscreened panel (8 quarantined 2008 FX closes retained): "
          f"gross {annual_sharpe(ru.gross):.3f}  net10 {annual_sharpe(ru.net['10bps']):.3f}  "
          f"active {au['arith_active_annual']:.2%}")

    # ── Negative control ──────────────────────────────────────────────────────
    print("\n--- NEGATIVE CONTROL: sign-randomised signal, 8 fixed seeds ---")
    ctrl_gross, ctrl_net, ctrl_act = [], [], []
    for seed in range(8):
        rc = run_trend(TrendConfig(name=f"CTRL{seed}", randomise_seed=seed),
                       vol_target=0.20, x=x, interior=interior)
        ctrl_gross.append(annual_sharpe(rc.gross))
        ctrl_net.append(annual_sharpe(rc.net["10bps"]))
        ctrl_act.append(active_report(rc.net["10bps"], rc.bench_net["10bps"])[
            "arith_active_annual"])
    live_gross = annual_sharpe(ref.gross)
    live_act = active["20%_10bps"]["arith_active_annual"]
    cg, ca = np.array(ctrl_gross), np.array(ctrl_act)
    report["negative_control"] = {
        "gross_sharpe_mean": float(cg.mean()),
        "gross_sharpe_sd": float(cg.std(ddof=1)),
        "net10_sharpe_mean": float(np.mean(ctrl_net)),
        "active_mean": float(ca.mean()),
        "active_sd": float(ca.std(ddof=1)),
        "live_gross_sharpe": live_gross,
        "live_active": live_act,
        "gross_sd_above_control": float((live_gross - cg.mean()) / cg.std(ddof=1))
        if cg.std(ddof=1) > 0 else float("nan"),
        "active_sd_above_control": float((live_act - ca.mean()) / ca.std(ddof=1))
        if ca.std(ddof=1) > 0 else float("nan"),
    }
    nc = report["negative_control"]
    print(f"  control gross Sharpe {cg.mean():+.3f} +/- {cg.std(ddof=1):.3f} "
          f"vs live {live_gross:+.3f}  =>  {nc['gross_sd_above_control']:+.2f} sd")
    print(f"  control arith active {ca.mean():+.2%} +/- {ca.std(ddof=1):.2%} "
          f"vs live {live_act:+.2%}  =>  {nc['active_sd_above_control']:+.2f} sd")

    # ── Kelly + verdict inputs ────────────────────────────────────────────────
    best_net = max(vt_table[k]["net_sharpe_10bps"] for k in vt_table)
    kel_10 = kelly_report(best_net)
    kel_2 = kelly_report(max(vt_table[k]["net_sharpe_2bps"] for k in vt_table))
    report["kelly"] = {"net_10bps": kel_10, "net_2bps": kel_2,
                       "gross": kelly_report(live_gross)}
    print("\n--- KELLY ---")
    for label, s, kk in (("gross", live_gross, report["kelly"]["gross"]),
                         ("net 2bps", max(vt_table[k]['net_sharpe_2bps'] for k in vt_table), kel_2),
                         ("net 10bps", best_net, kel_10)):
        print(f"  {label:>10}: Sharpe {s:>6.3f} -> half-Kelly {kk['half_kelly_growth']:>6.2%}/yr "
              f"at {kk['implied_vol']:>5.1%} vol   (full-Kelly {kk['full_kelly_growth']:>6.2%})")
    print(f"  30%/yr at half Kelly needs Sharpe 0.894. "
          f"Shortfall on net 10bps: {0.894 - best_net:+.3f}")

    report["dsr_pass"] = {
        "net_10bps_vs_n32": bool(best_net >= bars["n_trials_32"]),
        "net_10bps_vs_n36": bool(best_net >= bars["n_trials_36"]),
        "gross_vs_n36": bool(live_gross >= bars["n_trials_36"]),
    }

    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=float))
    ref.net["10bps"].to_frame("net_10bps").assign(
        gross=ref.gross, bench_net_10bps=ref.bench_net["10bps"],
        # VERIFY-2(a): persist the book-vol scaler k(t) so the defensive study's
        # mechanical-co-movement question can be answered without recomputing this
        # book in-process. DECISION-dated: the value on row t was computed at
        # month-end t and scales the weights held during t+1 (the same convention
        # as DefensiveResult.scaler). Appended LAST; existing columns unchanged.
        scaler=ref.scaler,
    ).to_csv(OUT / "primary_20pct_monthly.csv")
    print(f"\nWrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
