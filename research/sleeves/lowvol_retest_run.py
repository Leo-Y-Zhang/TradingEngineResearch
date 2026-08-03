"""Run the LOW-VOL / QUALITY RE-TEST on the corrected cost model. ONE run, no tuning.

Pre-registration: `research/sleeves/lowvol_retest_prereg.md`, committed at `0b12f93`
before any of this code existed.

Reads nothing after 2015-12-31.

    .venv/Scripts/python.exe -m research.sleeves.lowvol_retest_run
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR
from research.sleeves.low_vol_quality import build_signal
from research.sleeves.lowvol_retest import (
    BAND_ORDER,
    GATE_EXCESS,
    GATE_TSTAT,
    N_TRIALS,
    attach_spread_bounds,
    evaluate_band,
    overall_verdict,
    run_band,
    verdict_for,
)
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, load_universe

REPO = Path(__file__).resolve().parents[2]
RESULT_JSON = REPO / "research" / "sleeves" / "lowvol_retest_result.json"
RISK_CACHE = PANEL_DIR / "risk_features_dev.parquet"

log = logging.getLogger("lowvol_retest")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    started = time.monotonic()

    universe = load_universe()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    if universe["date"].max() > DEV_CUTOFF:
        raise ValueError("confirmation window leaked into the universe")

    log.info("corrected universe: %s cells (%s measured, %s upper_bound), %s tickers, "
             "%s -> %s", f"{len(universe):,}",
             f"{int((universe['spread_regime'] == 'measured').sum()):,}",
             f"{int((universe['spread_regime'] == 'upper_bound').sum()):,}",
             f"{universe['ticker'].nunique():,}",
             universe["date"].min().date(), universe["date"].max().date())

    if not RISK_CACHE.exists():
        raise FileNotFoundError(f"{RISK_CACHE} missing; run scripts/run_low_vol_quality_sleeve.py")
    risk = pd.read_parquet(RISK_CACHE)
    quality = pd.read_parquet(QUALITY_CACHE)

    merged = (
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    )
    log.info("coverage: vol %.1f%%, beta %.1f%%, gp %.1f%%, lev %.1f%%, accr %.1f%%",
             *[100 * merged[c].notna().mean() for c in
               ("realised_vol", "beta", "gross_profitability", "debt_to_equity",
                "accruals")])

    log.info("bracketing every cell's spread under both bounds")
    merged = attach_spread_bounds(merged)
    for regime in ("measured", "upper_bound"):
        block = merged[merged["spread_regime"] == regime]
        log.info("  %-12s %s cells, median spread %.1f / %.1f bps (cons / real)",
                 regime, f"{len(block):,}",
                 float(block["spread_conservative"].median()) * 1e4,
                 float(block["spread_realistic"].median()) * 1e4)

    merged = build_signal(merged)
    log.info("signal defined for %s of %s cells (%.1f%%)",
             f"{merged['signal'].notna().sum():,}", f"{len(merged):,}",
             100 * merged["signal"].notna().mean())

    evaluations: list[dict] = []
    for band in BAND_ORDER:
        books = run_band(merged, band, delistings)
        if books is None:
            log.warning("%s: insufficient data", band)
            continue
        evaluations.append(evaluate_band(books, n_trials=N_TRIALS))
        log.info("%s done", band)

    _report(evaluations)

    payload = {
        "run_utc": pd.Timestamp.utcnow().isoformat(),
        "dev_cutoff": str(DEV_CUTOFF.date()),
        "prereg": "research/sleeves/lowvol_retest_prereg.md",
        "n_trials": N_TRIALS,
        "gate": {"vol_matched_excess": GATE_EXCESS, "tstat": GATE_TSTAT},
        "verdict": overall_verdict(evaluations),
        "bands": evaluations,
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("results written to %s", RESULT_JSON)
    log.info("done in %.1f min", (time.monotonic() - started) / 60)
    return 0


def _report(evaluations: list[dict]) -> None:
    line = "=" * 120
    print("\n" + line)
    print("LOW-VOL / QUALITY RE-TEST - CORRECTED UNIVERSE + CORRECTED IMPACT - DEV, ONE RUN")
    print("prereg research/sleeves/lowvol_retest_prereg.md (committed 0b12f93, before the code)")
    print(line)

    for label in ("conservative", "realistic"):
        print(f"\n--- {label.upper()} BOUND " + "-" * (104 - len(label)))
        print(f"{'band':>13} {'capital':>11} {'gross':>7} {'cost':>7} {'net':>7} "
              f"{'CAGR':>7} {'vol':>6} {'SHARPE':>7} {'maxDD':>7} {'DSR':>6} "
              f"{'bench':>7} {'bvol':>6} {'bSh':>6} {'bDSR':>6}")
        for e in evaluations:
            b = e["bounds"][label]
            n, bm = b["net"], e["benchmark"]
            print(f"{e['band']:>13} ${e['deployable_capital']/1e3:>9.0f}k "
                  f"{e['gross']['annual_arithmetic']:>6.1%} {b['cost_annual_total']:>6.1%} "
                  f"{n['annual_arithmetic']:>6.1%} {n['cagr']:>6.1%} "
                  f"{n['volatility']:>5.1%} {n['sharpe']:>7.3f} {n['max_drawdown']:>6.1%} "
                  f"{n['dsr']:>6.3f} {bm['annual_arithmetic']:>6.1%} "
                  f"{bm['volatility']:>5.1%} {bm['sharpe']:>6.3f} {bm['dsr']:>6.3f}")

    print("\n" + "-" * 120)
    print("THE DECIDING NUMBER - MATCHED-VOLATILITY ACTIVE RETURN (benchmark levered to the")
    print("strategy's own vol). Raw geometric excess flatters a low-vol book; this does not.")
    print(f"{'band':>13} {'bound':>13} {'geometric':>10} {'arith active':>13} "
          f"{'t':>6} {'VOL-MATCHED':>12} {'t':>6} {'k':>6} {'vs rankable':>12} {'t':>6}")
    for e in evaluations:
        for label in ("conservative", "realistic"):
            b = e["bounds"][label]
            v, vr = b["vol_matched"], b["vol_matched_vs_rankable"]
            print(f"{e['band']:>13} {label:>13} {b['excess_geometric']:>+9.2%} "
                  f"{v['raw_active_annual']:>+12.2%} {v['raw_active_tstat']:>+6.2f} "
                  f"{v['vol_matched_active_annual']:>+11.2%} "
                  f"{v['vol_matched_active_tstat']:>+6.2f} "
                  f"{v['benchmark_scale_factor']:>6.3f} "
                  f"{vr['vol_matched_active_annual']:>+11.2%} "
                  f"{vr['vol_matched_active_tstat']:>+6.2f}")

    print("\n" + "-" * 120)
    print("COST DECOMPOSITION (annualised; one-way bps averaged over legs actually traded)")
    print(f"{'band':>13} {'bound':>13} {'spread':>8} {'impact':>8} {'comm+fx':>9} "
          f"{'total':>8} {'one-way':>9} {'turnover':>9} {'forced exits':>13}")
    for e in evaluations:
        for label in ("conservative", "realistic"):
            b = e["bounds"][label]
            print(f"{e['band']:>13} {label:>13} {b['cost_annual_spread']:>7.2%} "
                  f"{b['cost_annual_impact']:>7.2%} {b['cost_annual_commission']:>8.2%} "
                  f"{b['cost_annual_total']:>7.2%} {b['cost_one_way_bps']:>7.1f}bp "
                  f"{e['turnover_annual']:>8.1f}x {e['forced_exit_share']:>12.0%}")

    print("\n" + "-" * 120)
    print("SHARPE PER DECADE (net, conservative bound / benchmark)")
    decades = sorted({d for e in evaluations for d in e["bounds"]["conservative"]["decades"]})
    print(f"{'band':>13} " + " ".join(f"{d:>19}" for d in decades))
    for e in evaluations:
        cells = []
        for d in decades:
            s = e["bounds"]["conservative"]["decades"].get(d)
            bm = e["benchmark_decades"].get(d)
            cells.append(f"{s['sharpe']:>+7.2f} /{bm['sharpe']:>+6.2f} ({s['n_months']:>3})"
                         if s and bm else " " * 19)
        print(f"{e['band']:>13} " + " ".join(cells))

    print("\n" + "-" * 120)
    print("CONCENTRATION AND UNIVERSE DIAGNOSTICS")
    print(f"{'band':>13} {'|P&L| max':>10} {'top1/net':>9} {'top10/net':>10} "
          f"{'max wt':>7} {'top3 wt':>8} {'UB univ':>8} {'UB held':>8} {'hold vol':>9} "
          f"{'univ vol':>9} {'DSR bar':>8}")
    for e in evaluations:
        c = e["concentration"]
        print(f"{e['band']:>13} {c['largest_abs_share_of_gross_pnl']:>9.2%} "
              f"{c['largest_share_of_net_pnl']:>8.1%} {c['top10_share_of_net_pnl']:>9.1%} "
              f"{c['max_gross_notional_weight']:>6.1%} {c['top3_gross_notional_weight']:>7.1%} "
              f"{e['upper_bound_share_universe']:>7.0%} {e['upper_bound_share_held']:>7.0%} "
              f"{e['mean_holding_vol']:>8.1%} {e['mean_universe_vol']:>8.1%} "
              f"{e['dsr_sharpe_bar']:>8.4f}")

    print("\n" + "=" * 120)
    print("VERDICT (pre-committed rule, prereg section 6)")
    print("=" * 120)
    for e in evaluations:
        c = e["bounds"]["conservative"]
        flags = (f"excess {'PASS' if c['gate_excess_pass'] else 'fail'} | "
                 f"t {'PASS' if c['gate_tstat_pass'] else 'fail'} | "
                 f"DSR-bar {'PASS' if c['gate_dsr_bar_pass'] else 'fail'} | "
                 f"beats-bench-DSR {'PASS' if c['gate_beats_benchmark_dsr'] else 'fail'}")
        print(f"  {e['band']:>13}  {verdict_for(e):>13}  [{e['bracket_verdict']}]  {flags}")
    print(f"\n  OVERALL: {overall_verdict(evaluations)}")
    for e in evaluations:
        for note in e["notes"]:
            print(f"  ! {e['band']}: {note}")


if __name__ == "__main__":
    sys.exit(main())
