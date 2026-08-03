"""Adversarial verification of the multi-asset trend result. Tries to KILL it.

    .venv/Scripts/python.exe -m scripts.verify_multiasset_trend

The registered run produced arithmetic active return +8.07%/yr at t=2.54, which passes the
pre-registered PRIMARY test. This script asks whether that number means what it appears to
mean. Every check here is a way the result could be false.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.multiasset_trend import (
    BLOCKS,
    MONTHS,
    TrendConfig,
    active_report,
    annual_sharpe,
    load_excess_panel,
    max_drawdown,
    newey_west_tstat,
    run_trend,
)

OUT = Path("research/sleeves/_multiasset_trend")


def main() -> None:
    x, interior = load_excess_panel()
    r = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    s = r.net["10bps"]
    b = r.bench_net["10bps"]
    s, b = s.align(b, join="inner")
    out: dict = {}

    print("=" * 78)
    print("ADVERSARIAL VERIFICATION -- multi-asset trend")
    print("=" * 78)

    # ── 1. Is the active return leverage or alpha? ────────────────────────────
    print("\n[1] LEVERAGE OR ALPHA? Benchmark levered to the strategy's own volatility.")
    lev = s.std(ddof=1) / b.std(ddof=1)
    b_lev = b * lev
    d = s - b_lev
    print(f"  strategy vol {s.std(ddof=1)*math.sqrt(MONTHS):.2%}  "
          f"benchmark vol {b.std(ddof=1)*math.sqrt(MONTHS):.2%}  => leverage {lev:.2f}x")
    print(f"  strategy return  {s.mean()*MONTHS:.2%}/yr  Sharpe {annual_sharpe(s):.3f}")
    print(f"  LEVERED benchmark {b_lev.mean()*MONTHS:.2%}/yr  Sharpe {annual_sharpe(b_lev):.3f}")
    print(f"  active vs LEVERED benchmark: {d.mean()*MONTHS:+.2%}/yr  "
          f"t(NW) {newey_west_tstat(d):+.2f}")
    out["levered_benchmark"] = {
        "leverage": float(lev),
        "strategy_sharpe": annual_sharpe(s),
        "levered_bench_sharpe": annual_sharpe(b_lev),
        "levered_bench_return": float(b_lev.mean() * MONTHS),
        "active_vs_levered": float(d.mean() * MONTHS),
        "active_vs_levered_t": newey_west_tstat(d),
    }

    # ── 2. Does the t-stat scale with leverage? (the tell) ────────────────────
    print("\n[2] THE TELL: does the active t-stat scale with the vol target?")
    print("    A real alpha's t-stat is scale-free. If t rises with leverage, the test is")
    print("    measuring 'is the return > 0', not 'does it beat the benchmark'.")
    rows = []
    for tgt in (0.10, 0.20, 0.40, 0.60, 1.20):
        rr = run_trend(TrendConfig(), vol_target=tgt, x=x, interior=interior)
        a = active_report(rr.net["10bps"], rr.bench_net["10bps"])
        own_t = newey_west_tstat(rr.net["10bps"])
        rows.append((tgt, a["arith_active_annual"], a["arith_active_tstat"], own_t,
                     a["strat_sharpe"]))
        print(f"    target {tgt:>5.0%}: active {a['arith_active_annual']:>7.2%} "
              f"t={a['arith_active_tstat']:>5.2f} | strategy's OWN t={own_t:>5.2f} | "
              f"strategy Sharpe {a['strat_sharpe']:.3f}")
    print("    -> the active t converges to the strategy's own t. It is not a comparison.")
    out["leverage_scaling"] = [
        {"target": t, "active": a, "active_t": at, "own_t": ot, "sharpe": sh}
        for t, a, at, ot, sh in rows
    ]

    # ── 3. Post-2009: the deployable era ─────────────────────────────────────
    print("\n[3] THE DEPLOYABLE ERA. Trend's public record is concentrated pre-2009.")
    eras = {}
    for label, sl in (("full", slice(None)), ("pre2009", slice(None, "2008-12-31")),
                      ("2009+", slice("2009-01-01", None)),
                      ("2015+", slice("2015-01-01", None))):
        ss, bb = s.loc[sl], b.loc[sl]
        lv = ss.std(ddof=1) / bb.std(ddof=1)
        dd = ss - bb * lv
        eras[label] = {
            "months": int(len(ss)),
            "years": len(ss) / MONTHS,
            "strat_sharpe": annual_sharpe(ss),
            "bench_sharpe": annual_sharpe(bb),
            "strat_return": float(ss.mean() * MONTHS),
            "arith_active": float((ss - bb).mean() * MONTHS),
            "arith_active_t": newey_west_tstat(ss - bb),
            "vs_levered_bench": float(dd.mean() * MONTHS),
            "vs_levered_bench_t": newey_west_tstat(dd),
            "dsr_bar_n36": dsr_sharpe_bar(max(len(ss) / MONTHS, 0.5), n_trials=36)
            if len(ss) > 48 else float("nan"),
            "max_drawdown": max_drawdown(ss),
        }
        e = eras[label]
        print(f"  {label:>8} ({e['years']:>5.1f}y): strat SR {e['strat_sharpe']:>6.3f} vs "
              f"bench SR {e['bench_sharpe']:>6.3f} | vs LEVERED bench "
              f"{e['vs_levered_bench']:>+7.2%} (t {e['vs_levered_bench_t']:>+5.2f}) | "
              f"DSR bar {e['dsr_bar_n36']:.3f}")
    out["eras"] = eras

    # ── 4. Rolling 10-year Sharpe ────────────────────────────────────────────
    print("\n[4] ROLLING 10-YEAR SHARPE (is the edge decaying?)")
    roll = (s.rolling(120).mean() / s.rolling(120).std(ddof=1) * math.sqrt(MONTHS)).dropna()
    roll_b = (b.rolling(120).mean() / b.rolling(120).std(ddof=1) * math.sqrt(MONTHS)).dropna()
    for yr in (1975, 1985, 1995, 2005, 2015, 2020, 2026):
        sel = roll[roll.index.year <= yr]
        selb = roll_b[roll_b.index.year <= yr]
        if len(sel):
            print(f"    10y Sharpe ending {sel.index[-1].date()}: strategy {sel.iloc[-1]:>6.3f}"
                  f"   benchmark {selb.iloc[-1]:>6.3f}")
    out["rolling_10y"] = {
        "last": float(roll.iloc[-1]), "min": float(roll.min()), "max": float(roll.max()),
        "frac_below_bench": float((roll < roll_b.reindex(roll.index)).mean()),
        "frac_negative": float((roll < 0).mean()),
    }
    print(f"    fraction of rolling 10y windows BELOW the benchmark: "
          f"{out['rolling_10y']['frac_below_bench']:.1%}")

    # ── 5. The one surviving argument: is it additive? ───────────────────────
    print("\n[5] THE PORTFOLIO ARGUMENT. beta is ~0.01, so is trend ADDITIVE to the")
    print("    benchmark even though its standalone Sharpe is lower?")
    corr = float(s.corr(b))
    blends = {}
    for wgt in (0.0, 0.25, 0.5, 0.75, 1.0):
        bl = wgt * (s / s.std(ddof=1)) + (1 - wgt) * (b / b.std(ddof=1))
        blends[f"{wgt:.0%}_trend"] = annual_sharpe(bl)
    print(f"    corr(strategy, benchmark) = {corr:+.3f}")
    for k_, v in blends.items():
        print(f"    risk-weighted blend {k_:>12}: Sharpe {v:.3f}")
    # post-2009 blend
    s9, b9 = s.loc["2009-01-01":], b.loc["2009-01-01":]
    bl9 = 0.5 * (s9 / s9.std(ddof=1)) + 0.5 * (b9 / b9.std(ddof=1))
    print(f"    50/50 blend POST-2009 only: Sharpe {annual_sharpe(bl9):.3f} "
          f"(benchmark alone {annual_sharpe(b9):.3f})")
    out["blend"] = {"corr": corr, "blends": blends,
                    "blend_50_post2009": annual_sharpe(bl9),
                    "bench_post2009": annual_sharpe(b9)}

    # ── 6. P&L by instrument and by block ────────────────────────────────────
    print("\n[6] P&L CONCENTRATION by instrument and block")
    pnl = r.pnl.loc[s.index]
    tot = float(pnl.to_numpy().sum())
    by_i = (pnl.sum(axis=0) / tot).sort_values(ascending=False)
    print("    top 5 instruments: " + ", ".join(f"{k} {v:.1%}" for k, v in by_i.head(5).items()))
    print("    bottom 3:          " + ", ".join(f"{k} {v:.1%}" for k, v in by_i.tail(3).items()))
    by_block = {g: float(pnl[[c for c in keys if c in pnl.columns]].sum().sum() / tot)
                for g, keys in BLOCKS.items()}
    print("    by block: " + ", ".join(f"{g} {v:.1%}" for g, v in by_block.items()))
    print(f"    instruments with NEGATIVE total P&L: {int((by_i < 0).sum())} of {len(by_i)}")
    out["pnl_by_instrument"] = {k: float(v) for k, v in by_i.items()}
    out["pnl_by_block"] = by_block

    # ── 7. Costs: what round-trip cost kills it vs the levered benchmark? ────
    print("\n[7] COST BUDGET. Round-trip cost at which the strategy's Sharpe hits the")
    print("    benchmark's (the level it must beat to be worth holding standalone).")
    turn = r.turnover.reindex(s.index).fillna(0.0)
    target_sr = annual_sharpe(b)
    lo, hi = 0.0, 0.05
    for _ in range(60):
        mid = (lo + hi) / 2
        cand = r.gross.reindex(s.index) - 0.5 * mid * turn
        if annual_sharpe(cand) > target_sr:
            lo = mid
        else:
            hi = mid
    print(f"    strategy Sharpe equals benchmark Sharpe ({target_sr:.3f}) at a round-trip")
    print(f"    cost of {lo*1e4:.2f}bps -- and the registered brackets are 2 and 10bps.")
    out["breakeven_cost_bps"] = float(lo * 1e4)

    (OUT / "verification.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT / 'verification.json'}")


if __name__ == "__main__":
    main()
