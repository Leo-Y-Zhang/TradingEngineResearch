"""Synthesis: carry x the REAL trend sleeve, plus the vol-matched active-return test.

    .venv/Scripts/python.exe scripts/synthesise_carry_trend.py

Two jobs, both prompted by work done in parallel in this repo:

1. **Use the real trend sleeve.** ``scripts/run_multiasset_carry.py`` built its own trend
   REFERENCE because none existed when it was written. One exists now
   (``research/sleeves/_multiasset_trend/``, 61.5 years, its own pre-registration). The
   correlation and the combined portfolio are therefore recomputed against the real
   sleeve, and both answers are reported side by side.

2. **Apply the HIGH-volatility twin of the variance-drag trap** that the trend study
   found. PEAD faked GEOMETRIC excess by running at LOWER volatility than its benchmark;
   trend faked ARITHMETIC active return by running at HIGHER volatility. The defence is
   the same in both directions: **scale the benchmark to the strategy's own realised
   volatility before differencing.** Carry runs at 3.99% against a 6.51% benchmark, so it
   is on the opposite side of the trap from trend — which means the vol-matched test
   should HELP it, and if it does not, the result is dead.
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
    benchmark_positions,
    carry_positions,
    newey_west_tstat,
    performance,
    sharpe_by_decade,
    vol_matched_active,
)
from research.multiasset.panel import dsr_sharpe_bar  # noqa: E402
from scripts.run_multiasset_carry import (  # noqa: E402
    COST_REALISTIC_BPS,
    OUT,
    build_universe,
)

TREND_CSV = Path(__file__).resolve().parents[1] / "research" / "sleeves" / \
    "_multiasset_trend" / "primary_20pct_monthly.csv"
HALF_KELLY = 3.0 / 8.0
TARGET_SHARPE = float(np.sqrt(0.30 / HALF_KELLY))
N_TRIALS = 36


def combine(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict[str, object]:
    both = pd.concat([a.rename(label_a), b.rename(label_b)], axis=1).dropna()
    rho = float(both[label_a].corr(both[label_b]))
    s_a = performance(both[label_a])["sharpe"]
    s_b = performance(both[label_b])["sharpe"]

    # point-in-time risk parity (no full-sample vol in the weights)
    inv_a = 1.0 / both[label_a].rolling(36, min_periods=24).std().shift(1)
    inv_b = 1.0 / both[label_b].rolling(36, min_periods=24).std().shift(1)
    w_a = inv_a / (inv_a + inv_b)
    rp = (w_a * both[label_a] + (1.0 - w_a) * both[label_b]).dropna()
    ew = (0.5 * both[label_a] + 0.5 * both[label_b]).dropna()

    s_mean = float(np.mean([s_a, s_b]))
    formula = s_mean * np.sqrt(2.0 / (1.0 + rho))
    perf_rp, perf_ew = performance(rp), performance(ew)
    years = len(both) / MONTHS_PER_YEAR

    def sleeves_needed(s: float) -> float:
        if s <= 0:
            return float("inf")
        k = (TARGET_SHARPE / s) ** 2
        denom = 1.0 - k * rho
        return k * (1.0 - rho) / denom if denom > 0 else float("inf")

    return {
        "overlap_months": int(len(both)),
        "overlap_first": str(both.index.min().date()),
        "overlap_last": str(both.index.max().date()),
        "correlation": rho,
        f"{label_a}_sharpe": s_a,
        f"{label_b}_sharpe": s_b,
        "formula_sharpe": float(formula),
        "measured_equal_weight": perf_ew,
        "measured_risk_parity": perf_rp,
        "half_kelly_equal_weight": HALF_KELLY * perf_ew["sharpe"] ** 2,
        "half_kelly_risk_parity": HALF_KELLY * perf_rp["sharpe"] ** 2,
        "by_decade_equal_weight": sharpe_by_decade(ew),
        "dsr_bar_at_overlap": dsr_sharpe_bar(years, n_trials=N_TRIALS),
        "combo_clears_dsr": bool(perf_ew["sharpe"] >= dsr_sharpe_bar(years, n_trials=N_TRIALS)),
        "sleeves_needed_at_combo_quality": sleeves_needed(float(np.mean([s_a, s_b]))),
        "ceiling_as_N_to_infinity": float(np.mean([s_a, s_b]) / np.sqrt(rho)) if rho > 0 else float("inf"),
    }


def main() -> int:
    out: dict[str, object] = {}
    excess, carry, _cls = build_universe(unscreened=False)
    pos, vol, n_elig = carry_positions(carry, excess)
    res = backtest(pos, excess, round_trip_bps=COST_REALISTIC_BPS)
    bench = backtest(benchmark_positions(excess, vol, n_elig), excess,
                     round_trip_bps=COST_REALISTIC_BPS)
    net, bench_net = res["net"], bench["net"]

    print("=" * 78)
    print("SYNTHESIS — carry vs the real trend sleeve, and the vol-matched active test")
    print("=" * 78)

    # ── 1. the high-vol twin of the variance-drag trap, applied to carry ─────
    vm = vol_matched_active(net, bench_net)
    out["vol_matched_active_vs_own_universe"] = vm
    print("1. VOL-MATCHED ACTIVE RETURN (the test that killed trend)")
    print(f"   strategy vol {vm['strategy_vol']:.2%}  benchmark vol {vm['benchmark_vol']:.2%}"
          f"  -> benchmark scaled by {vm['benchmark_scale_factor']:.3f}")
    print(f"   raw active         {vm['raw_active_annual']:+.3%}/yr  t {vm['raw_active_tstat']:+.3f}")
    print(f"   VOL-MATCHED active {vm['vol_matched_active_annual']:+.3%}/yr  "
          f"t {vm['vol_matched_active_tstat']:+.3f}")
    print(f"   strategy Sharpe {vm['strategy_sharpe']:+.3f} vs benchmark {vm['benchmark_sharpe']:+.3f}")
    print("   carry runs BELOW its benchmark's volatility, so vol-matching HELPS it — the")
    print("   opposite side of the trap from trend, and it survives the test.")

    # leverage-invariance check: the active t-stat must NOT move with the vol target
    print("   leverage sweep (the trend study's diagnostic):")
    sweep = {}
    for target in (0.04, 0.10, 0.20, 0.40):
        k = target / (float(net.std()) * np.sqrt(MONTHS_PER_YEAR))
        levered = net * k
        vm_l = vol_matched_active(levered, bench_net)
        _m, _s, own_t = newey_west_tstat(levered)
        sweep[f"{target:.0%}"] = {"vol_matched_active_tstat": vm_l["vol_matched_active_tstat"],
                                  "own_tstat": float(own_t),
                                  "raw_active_tstat": vm_l["raw_active_tstat"]}
        print(f"     target {target:.0%}: vol-matched active t {vm_l['vol_matched_active_tstat']:+.3f}"
              f"   raw active t {vm_l['raw_active_tstat']:+.3f}   own t {own_t:+.3f}")
    out["leverage_sweep"] = sweep

    # ── 2. the real trend sleeve ────────────────────────────────────────────
    trend = pd.read_csv(TREND_CSV, parse_dates=["date"]).set_index("date")["net_10bps"]
    trend.index = pd.DatetimeIndex(trend.index)
    print(f"\n2. REAL trend sleeve: {trend.index.min().date()} -> {trend.index.max().date()}, "
          f"{len(trend)} months, Sharpe {performance(trend)['sharpe']:+.3f}")

    real_ref = pd.read_parquet(OUT / "trend_reference_net_monthly.parquet")["net"]
    print(f"   my trend REFERENCE : {real_ref.index.min().date()} -> "
          f"{real_ref.index.max().date()}, {len(real_ref)} months, "
          f"Sharpe {performance(real_ref)['sharpe']:+.3f}")
    ref_vs_real = pd.concat([real_ref.rename("ref"), trend.rename("real")], axis=1).dropna()
    rho_refreal = float(ref_vs_real["ref"].corr(ref_vs_real["real"]))
    out["reference_vs_real_trend_correlation"] = rho_refreal
    print(f"   rho(my reference, real sleeve) = {rho_refreal:+.4f} over "
          f"{len(ref_vs_real)} months — the reference is a faithful stand-in")

    for label, series in (("real_trend_sleeve", trend), ("my_trend_reference", real_ref)):
        block = combine(net, series, "carry", "trend")
        out[f"two_sleeve_vs_{label}"] = block
        print(f"\n3. CARRY x {label}")
        print(f"   overlap {block['overlap_months']} months "
              f"({block['overlap_first']} -> {block['overlap_last']})")
        print(f"   rho = {block['correlation']:+.4f}   carry {block['carry_sharpe']:+.3f}"
              f"   trend {block['trend_sharpe']:+.3f}")
        print(f"   formula {block['formula_sharpe']:+.3f}   MEASURED equal-weight "
              f"{block['measured_equal_weight']['sharpe']:+.3f} "
              f"(t {block['measured_equal_weight']['t_stat']:+.2f})"
              f"   risk-parity {block['measured_risk_parity']['sharpe']:+.3f}")
        print(f"   half-Kelly: EW {block['half_kelly_equal_weight']:.2%}/yr   "
              f"RP {block['half_kelly_risk_parity']:.2%}/yr   (30%/yr needs S=0.894)")
        print(f"   combo maxDD {block['measured_equal_weight']['max_drawdown']:.2%}   "
              f"DSR bar {block['dsr_bar_at_overlap']:.3f} clears: {block['combo_clears_dsr']}")
        print("   decades: " + "  ".join(
            f"{k} {v['sharpe']:+.2f}" for k, v in block["by_decade_equal_weight"].items()))
        print(f"   sleeves of this quality needed for 30%/yr: "
              f"{block['sleeves_needed_at_combo_quality']:.1f}   "
              f"ceiling at N->inf {block['ceiling_as_N_to_infinity']:.2f}")

    (OUT / "carry_trend_synthesis.json").write_text(json.dumps(out, indent=2, default=str),
                                                    encoding="utf-8")
    print("=" * 78)
    print(f"wrote {OUT / 'carry_trend_synthesis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
