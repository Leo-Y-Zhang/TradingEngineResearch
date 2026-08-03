"""The corrected panel's BOOK: what `trend + passive` is worth once the conventions
are repaired at source.

    .venv/Scripts/python.exe scripts/run_convention_repair_book.py

Runs the pre-registered trend sleeve and its passive benchmark on each of the three
corrected panels written by ``scripts/run_convention_repair.py``, and on the OLD panel
as the reproduction anchor. Every statistic is computed by the survivor verification's
own functions so the comparison is like for like -- this script chooses no method.

The honest prior, recorded in the pre-registration BEFORE any of this ran: the corrected
book is expected below 0.894 and below the 0.8206 the constant-charge approximation
produced. If it lands above, the first hypothesis is a bug in the repair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.convention import BRACKET_BOUNDS  # noqa: E402
from research.sleeves._survivor.survivor_verification import (  # noqa: E402
    BLOCK,
    RNG_SEED,
    VOL_TARGET,
    book_from,
    boot_sharpe,
    cagr,
    circular_blocks,
    levered_total,
    sharpe,
    vm,
)
from research.sleeves.multiasset_trend import (  # noqa: E402
    TrendConfig,
    load_excess_panel,
    run_trend,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_data" / "multiasset"
OUT_DIR = ROOT / "research" / "multiasset" / "_convention"
FINANCING = 0.0150               # bill + 150bp, the survivor's primary assumption
RECONCILIATION = 0.877           # iteration 11's measured engine-vs-ladder factor
COST = "10bps"

#: What the survivor verification recorded, for a like-for-like line-up.
RECORDED = {"published_book": 0.9033, "constant_charge_best_supported": 0.8206,
            "half_kelly_30pct_needs": 0.894}


def decade_table(series: pd.Series) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for decade in sorted({y // 10 * 10 for y in series.index.year}):
        chunk = series[(series.index.year // 10 * 10) == decade]
        if len(chunk) >= 24:
            out[str(int(decade))] = {"n": int(len(chunk)), "sharpe": round(sharpe(chunk), 4)}
    return out


def leverage_ladder(book: pd.Series, cash: pd.Series) -> dict[str, dict[str, float]]:
    """Solve leverage against the BOOTSTRAP 95th-percentile drawdown, never the observed
    path -- method rule 5, which is worth 7-9pp of overstatement if ignored."""
    excess = book.to_numpy()
    csn = cash.reindex(book.index).to_numpy()
    blocks = circular_blocks(len(excess), BLOCK, np.random.default_rng(RNG_SEED + 1), 2000)
    rows: dict[str, dict[str, float]] = {}
    for cap in (0.35, 0.50):
        chosen = 0.0
        for lev in np.arange(0.05, 5.0001, 0.05):
            total = levered_total(excess, csn, float(lev), FINANCING)
            paths = total[blocks]
            curve = np.cumprod(1.0 + paths, axis=1)
            drawdown = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
            if abs(float(np.percentile(drawdown, 5))) <= cap:
                chosen = float(lev)
            else:
                break
        total = levered_total(excess, csn, chosen, FINANCING)
        rows[f"dd{int(cap * 100)}"] = {
            "leverage": round(chosen, 2),
            "cagr_pct": round(cagr(total) * 100.0, 3),
            "cagr_after_reconciliation_pct": round(cagr(total) * RECONCILIATION * 100.0, 3),
        }
    return rows


def evaluate(name: str, panel: pd.DataFrame, interior: pd.DataFrame,
             cash: pd.Series) -> dict:
    result = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=panel, interior=interior)
    trend, passive = result.net[COST], result.bench_net[COST]
    book, weights = book_from(trend, passive)
    # The programme's own vol-matched active and ITS t-statistic: the benchmark is
    # scaled to the strategy's volatility by a single full-sample constant, which is the
    # only form that cannot be inflated by levering the strategy (method rule 1).
    active = vm(book, passive)
    boots = boot_sharpe(book)
    return {
        "name": name,
        "n_months": int(len(book)),
        "sharpe_book": round(sharpe(book), 6),
        "sharpe_trend": round(sharpe(trend), 4),
        "sharpe_passive": round(sharpe(passive), 4),
        "weight_trend": round(float(weights[0]), 5),
        "weight_passive": round(float(weights[1]), 5),
        "vol_matched_active_pct_yr": round(active["annual"] * 100.0, 4),
        "vol_matched_active_tstat": round(active["tstat"], 3),
        "boot_ci95": [round(float(np.nanpercentile(boots, 2.5)), 4),
                      round(float(np.nanpercentile(boots, 97.5)), 4)],
        "p_below_0894": round(float(np.nanmean(boots < 0.894)), 4),
        "p_below_075": round(float(np.nanmean(boots < 0.75)), 4),
        "sharpe_by_decade": decade_table(book),
        "sharpe_since_2010": round(sharpe(book[book.index >= "2010-01-01"]), 4),
        "leverage_ladder": leverage_ladder(book, cash),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old, interior = load_excess_panel()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"].reindex(old.index)

    runs: dict[str, dict] = {"old_panel": evaluate("old_panel", old, interior, cash)}
    for bound in BRACKET_BOUNDS:
        path = DATA / f"returns_monthly_corrected_{bound}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - run scripts/run_convention_repair.py first")
        runs[bound] = evaluate(bound, pd.read_parquet(path), interior, cash)

    anchor = runs["old_panel"]["sharpe_book"]
    out = {
        "recorded": RECORDED,
        "reproduction_anchor": {
            "measured_old_panel_sharpe": anchor,
            "recorded_published": RECORDED["published_book"],
            "reproduces": bool(abs(anchor - RECORDED["published_book"]) < 5e-4),
        },
        "runs": runs,
        "verdict": {
            "central_below_0894": bool(runs["central"]["sharpe_book"] < 0.894),
            "central_below_constant_charge_0.8206": bool(
                runs["central"]["sharpe_book"] < RECORDED["constant_charge_best_supported"]),
            "bracket": [runs["conservative"]["sharpe_book"], runs["central"]["sharpe_book"],
                        runs["realistic"]["sharpe_book"]],
        },
    }
    (OUT_DIR / "convention_repair_book.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print("THE CORRECTED BOOK")
    print("=" * 78)
    print(f"\nreproduction anchor: old panel Sharpe {anchor:.6f} vs recorded "
          f"{RECORDED['published_book']} -> "
          f"{'REPRODUCES' if out['reproduction_anchor']['reproduces'] else 'DOES NOT REPRODUCE'}")
    print(f"\n  {'panel':<14}{'book':>9}{'trend':>8}{'passive':>9}{'vm active':>11}"
          f"{'vm t':>7}{'P(S<.894)':>11}{'since2010':>11}")
    for name, r in runs.items():
        print(f"  {name:<14}{r['sharpe_book']:>9.4f}{r['sharpe_trend']:>8.4f}"
              f"{r['sharpe_passive']:>9.4f}{r['vol_matched_active_pct_yr']:>10.2f}%"
              f"{r['vol_matched_active_tstat']:>7.2f}{r['p_below_0894']:>11.3f}"
              f"{r['sharpe_since_2010']:>11.4f}")

    print(f"\n  {'panel':<14}{'lev@DD50':>10}{'CAGR%':>9}{'x0.877':>9}"
          f"{'lev@DD35':>10}{'CAGR%':>9}{'x0.877':>9}")
    for name, r in runs.items():
        a, b = r["leverage_ladder"]["dd50"], r["leverage_ladder"]["dd35"]
        print(f"  {name:<14}{a['leverage']:>10.2f}{a['cagr_pct']:>9.2f}"
              f"{a['cagr_after_reconciliation_pct']:>9.2f}"
              f"{b['leverage']:>10.2f}{b['cagr_pct']:>9.2f}"
              f"{b['cagr_after_reconciliation_pct']:>9.2f}")

    lo, mid, hi = out["verdict"]["bracket"]
    print(f"\n  BRACKET  conservative {lo:.4f}  <=  central {mid:.4f}  <=  "
          f"realistic {hi:.4f}")
    print(f"  central below 0.894 (half-Kelly 30%/yr): "
          f"{out['verdict']['central_below_0894']}")
    print(f"  central below the constant-charge 0.8206: "
          f"{out['verdict']['central_below_constant_charge_0.8206']}")
    print(f"\nwrote {OUT_DIR / 'convention_repair_book.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
