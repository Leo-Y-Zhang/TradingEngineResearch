"""THE REGISTERED GATE RUN for `trend + passive` on the repaired panel. Executes once.

    .venv/Scripts/python.exe scripts/run_trend_passive_gate.py

Pre-registered in ``research/sleeves/trend_passive_prereg.md``. Every gate, benchmark,
trial count and tolerance was fixed there before this script existed, together with
eleven numbered predictions -- and this script SCORES ITSELF against those predictions
and prints the ones it got wrong, because a forecast nobody grades is not a forecast.

No tuning, no search, no new candidate, so no ledger entry. No live path, no broker path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.benchmark_gate import benchmark_relative_rule  # noqa: E402
from research.multiasset.carry import trailing_vol  # noqa: E402
from research.multiasset.convention import BRACKET_BOUNDS  # noqa: E402
from research.multiasset.panel import dsr_sharpe_bar  # noqa: E402
from research.sleeves._survivor.survivor_verification import (  # noqa: E402
    BLOCK,
    RNG_SEED,
    VOL_TARGET,
    book_from,
    cagr,
    circular_blocks,
    levered_total,
    sharpe,
    vm,
)
from research.sleeves.multiasset_trend import (  # noqa: E402
    BLOCKS,
    TrendConfig,
    load_excess_panel,
    run_trend,
)
from research.validation import deflated_sharpe_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_data" / "multiasset"
OUT_DIR = ROOT / "research" / "sleeves" / "_trend_passive"
COST = "10bps"
MPY = 12
RECONCILIATION = 0.877

# Prereg 2: the ledger's 47 PLUS the 234 configurations of the unregistered
# portfolio-combination search that produced this candidate.
TRIAL_COUNTS = {"registered_281": 281, "ledger_only_47": 47, "panel_convention_32": 32}
PRIMARY_COUNT = 281
DSR_TARGET = 0.95
FINANCINGS = {"bill_plus_150bp": 0.0150, "bill_plus_300bp": 0.0300}
#: The equal-weight leg's unpriced rebalancing cost, measured in the survivor
#: verification at 1.5 bps/yr (immaterial: -0.0012 of Sharpe). Charged so that every
#: benchmark carries a cost, which C3 requires before any comparison is legitimate.
EW_REBALANCE_COST_ANNUAL = 0.00015


def ladder(series: pd.Series, cash: pd.Series, financing: float) -> dict[str, dict]:
    excess = series.to_numpy()
    csn = cash.reindex(series.index).to_numpy()
    blocks = circular_blocks(len(excess), BLOCK, np.random.default_rng(RNG_SEED + 1), 2000)
    rows: dict[str, dict] = {}
    for cap in (0.35, 0.50):
        chosen = 0.0
        for lev in np.arange(0.05, 5.0001, 0.05):
            total = levered_total(excess, csn, float(lev), financing)
            curve = np.cumprod(1.0 + total[blocks], axis=1)
            dd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
            if abs(float(np.percentile(dd, 5))) <= cap:
                chosen = float(lev)
            else:
                break
        total = levered_total(excess, csn, chosen, financing)
        rows[f"dd{int(cap * 100)}"] = {
            "leverage": round(chosen, 2),
            "cagr_pct": round(cagr(total) * 100.0, 3),
            "cagr_after_reconciliation_pct": round(cagr(total) * RECONCILIATION * 100.0, 3),
        }
    return rows


def build_panel_members(x: pd.DataFrame, index: pd.Index) -> dict[str, pd.Series]:
    """The registered benchmark panel (prereg 3a), all monthly, all cost-charged.

    Member (iii) replaces C7's daily passive -- the repaired panel is monthly by
    construction -- and (iv) is added so the shopping detector is HARDER to pass, not
    easier. Both are genuinely different opportunity costs, not re-frequencied copies.
    """
    live = x.where(x.notna())
    charge = EW_REBALANCE_COST_ANNUAL / MPY

    equal_weight = live.mean(axis=1) - charge

    # (iii) equal RISK: inverse trailing vol, causal, renormalised over what is live.
    vol = trailing_vol(x)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    inv = inv.where(live.notna())
    weights = inv.div(inv.sum(axis=1), axis=0)
    equal_risk = (live * weights).sum(axis=1, min_count=1) - charge

    # (iv) 60/40 equity/rates, monthly rebalanced.
    eq = live[list(BLOCKS["equity"])].mean(axis=1)
    rt = live[list(BLOCKS["rates"])].mean(axis=1)
    sixty_forty = (0.60 * eq + 0.40 * rt) - charge

    return {
        "ii_passive_monthly_EW_18": equal_weight.reindex(index),
        "iii_equal_RISK_passive": equal_risk.reindex(index),
        "iv_60_40_equity_rates": sixty_forty.reindex(index),
    }


def evaluate(name: str, panel: pd.DataFrame, interior: pd.DataFrame,
             cash: pd.Series) -> dict:
    result = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=panel, interior=interior)
    trend, passive = result.net[COST], result.bench_net[COST]
    book, _ = book_from(trend, passive)

    members = {"i_own_universe_EW": passive, **build_panel_members(panel, book.index)}
    aligned = pd.concat({"book": book, **members}, axis=1).dropna()
    candidate = aligned["book"]
    nominated = aligned["i_own_universe_EW"]
    registered_panel = {k: aligned[k] for k in members if k != "i_own_universe_EW"}
    registered_panel["i_own_universe_EW"] = nominated

    verdict = benchmark_relative_rule(candidate, nominated, registered_panel, MPY)

    years = len(book) / MPY
    dsr = {label: round(float(deflated_sharpe_ratio(book, n_trials=n)), 6)
           for label, n in TRIAL_COUNTS.items()}
    bar = {label: round(float(dsr_sharpe_bar(years, n_trials=n)), 4)
           for label, n in TRIAL_COUNTS.items()}

    post2010 = book.index >= "2010-01-01"
    active_post = vm(book[post2010], passive[post2010])

    return {
        "name": name,
        "n_months": int(len(book)),
        "sharpe_book": round(sharpe(book), 6),
        "sharpe_passive": round(sharpe(passive), 6),
        "dsr": dsr,
        "dsr_bar_annual_sharpe": bar,
        "dsr_passes_at_281": bool(dsr["registered_281"] >= DSR_TARGET),
        "benchmark_gate": {
            "nominated_verdict": verdict.nominated.verdict,
            "nominated_gap": round(float(verdict.nominated.sharpe_gap), 5),
            "nominated_rho": round(float(verdict.nominated.rho), 5),
            "panel": {k: {"verdict": v.verdict,
                          "gap": round(float(v.sharpe_gap), 5),
                          "rho": round(float(v.rho), 5)}
                      for k, v in verdict.panel.items()},
            "benchmark_sensitive": bool(verdict.benchmark_sensitive),
            "promotable": bool(verdict.promotable),
        },
        "ladder_book": {k: ladder(book, cash, f) for k, f in FINANCINGS.items()},
        "ladder_benchmark_passive_alone": {
            k: ladder(passive, cash, f) for k, f in FINANCINGS.items()},
        "post2010_vol_matched_active_pct_yr": round(active_post["annual"] * 100.0, 4),
        "post2010_tstat": round(active_post["tstat"], 3),
        "post2010_n": int(post2010.sum()),
    }


def score(runs: dict[str, dict]) -> list[dict]:
    """Grade every registered prediction. A forecast nobody grades is not a forecast."""
    c = runs["central"]
    cons = runs["conservative"]
    gate = c["benchmark_gate"]
    book300 = c["ladder_book"]["bill_plus_300bp"]["dd50"]
    bench150 = c["ladder_benchmark_passive_alone"]["bill_plus_150bp"]["dd50"]
    book150 = c["ladder_book"]["bill_plus_150bp"]["dd50"]
    incremental = book150["cagr_after_reconciliation_pct"] - bench150["cagr_after_reconciliation_pct"]
    verdicts = {r["benchmark_gate"]["promotable"] for r in runs.values()}

    checks = [
        ("P1", "DSR at n=281, central in 0.97-1.000",
         c["dsr"]["registered_281"], 0.97 <= c["dsr"]["registered_281"] <= 1.0),
        ("P2", "DSR at n=281, conservative >= 0.95",
         cons["dsr"]["registered_281"], cons["dsr"]["registered_281"] >= DSR_TARGET),
        ("P3", "DSR bar at n=281 in 0.55-0.65",
         c["dsr_bar_annual_sharpe"]["registered_281"],
         0.55 <= c["dsr_bar_annual_sharpe"]["registered_281"] <= 0.65),
        ("P4", "nominated verdict is BEATS",
         gate["nominated_verdict"], gate["nominated_verdict"] == "BEATS"),
        ("P5", "all panel members BEAT -> promotable",
         gate["promotable"], bool(gate["promotable"])),
        ("P6", "rho(book, nominated) > 0.90",
         gate["nominated_rho"], gate["nominated_rho"] > 0.90),
        ("P7a", "bill+300bp DD50 leverage in 1.85-1.95",
         book300["leverage"], 1.85 <= book300["leverage"] <= 1.95),
        ("P7b", "bill+300bp DD50 CAGRx0.877 in 12.5-13.5",
         book300["cagr_after_reconciliation_pct"],
         12.5 <= book300["cagr_after_reconciliation_pct"] <= 13.5),
        ("P8a", "benchmark DD50 leverage in 1.30-1.45",
         bench150["leverage"], 1.30 <= bench150["leverage"] <= 1.45),
        ("P8b", "benchmark DD50 CAGRx0.877 in 8.0-9.5",
         bench150["cagr_after_reconciliation_pct"],
         8.0 <= bench150["cagr_after_reconciliation_pct"] <= 9.5),
        ("P9", "incremental over benchmark in +4.5 to +6.5pp",
         round(incremental, 3), 4.5 <= incremental <= 6.5),
        ("P10a", "post-2010 active in 0.0 to +1.5%/yr",
         c["post2010_vol_matched_active_pct_yr"],
         0.0 <= c["post2010_vol_matched_active_pct_yr"] <= 1.5),
        ("P10b", "post-2010 |t| < 2 (not significant)",
         c["post2010_tstat"], abs(c["post2010_tstat"]) < 2.0),
        ("P11", "same promotable verdict at all three bounds",
         sorted(str(v) for v in verdicts), len(verdicts) == 1),
    ]
    return [{"id": i, "claim": claim, "observed": obs, "correct": bool(ok)}
            for i, claim, obs, ok in checks]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old, interior = load_excess_panel()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"].reindex(old.index)

    runs: dict[str, dict] = {}
    for bound in BRACKET_BOUNDS:
        path = DATA / f"returns_monthly_corrected_{bound}.parquet"
        runs[bound] = evaluate(bound, pd.read_parquet(path), interior, cash)

    scored = score(runs)
    n_right = sum(1 for s in scored if s["correct"])
    central = runs["central"]
    gate_pass = bool(
        runs["conservative"]["dsr_passes_at_281"]
        and central["benchmark_gate"]["promotable"]
        and not central["benchmark_gate"]["benchmark_sensitive"]
        and len({r["benchmark_gate"]["promotable"] for r in runs.values()}) == 1)

    out = {
        "prereg": "research/sleeves/trend_passive_prereg.md",
        "trial_counts": TRIAL_COUNTS,
        "runs": runs,
        "predictions_scored": scored,
        "predictions_correct": f"{n_right}/{len(scored)}",
        "gate_passes": gate_pass,
        "promotion_decision": (
            "NOT PROMOTED. Pre-committed in prereg section 6 BEFORE these gates ran: a "
            "full-sample gate cannot see that the significance rests on the 2000s or "
            "that the book adds nothing over passive since 2010. A pass means the "
            "premium is not a multiple-testing artefact - a statement about the past, "
            "not a reason to trade it."),
    }
    (OUT_DIR / "trend_passive_gate.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 78)
    print("TREND + PASSIVE -- REGISTERED GATE RUN ON THE REPAIRED PANEL")
    print("=" * 78)
    print(f"\n  {'bound':<14}{'book':>9}{'DSR@281':>10}{'DSR@47':>9}{'bar@281':>9}"
          f"{'nominated':>12}{'promotable':>12}")
    for name, r in runs.items():
        g = r["benchmark_gate"]
        print(f"  {name:<14}{r['sharpe_book']:>9.4f}{r['dsr']['registered_281']:>10.4f}"
              f"{r['dsr']['ledger_only_47']:>9.4f}"
              f"{r['dsr_bar_annual_sharpe']['registered_281']:>9.4f}"
              f"{g['nominated_verdict']:>12}{str(g['promotable']):>12}")

    print("\n--- BENCHMARK PANEL (central), the shopping detector ---")
    for key, v in central["benchmark_gate"]["panel"].items():
        print(f"  {key:<26} {v['verdict']:<14} gap {v['gap']:+.4f}  rho {v['rho']:+.4f}")
    print(f"  benchmark_sensitive: {central['benchmark_gate']['benchmark_sensitive']}")

    print("\n--- LADDERS (central), book vs the benchmark it must beat ---")
    for fin in FINANCINGS:
        b = central["ladder_book"][fin]["dd50"]
        p = central["ladder_benchmark_passive_alone"][fin]["dd50"]
        print(f"  {fin:<18} book {b['leverage']:.2f}x -> "
              f"{b['cagr_after_reconciliation_pct']:.2f}%   "
              f"passive alone {p['leverage']:.2f}x -> "
              f"{p['cagr_after_reconciliation_pct']:.2f}%   "
              f"incremental {b['cagr_after_reconciliation_pct'] - p['cagr_after_reconciliation_pct']:+.2f}pp")

    print(f"\n--- POST-2010 (central), n={central['post2010_n']} ---")
    print(f"  vol-matched active {central['post2010_vol_matched_active_pct_yr']:+.3f}%/yr  "
          f"t {central['post2010_tstat']:+.2f}")

    print(f"\n--- PREDICTIONS SCORED: {n_right}/{len(scored)} correct ---")
    for s in scored:
        mark = "ok  " if s["correct"] else "WRONG"
        print(f"  {mark} {s['id']:<5} {s['claim']:<48} observed {s['observed']}")

    print(f"\n  GATE PASSES: {gate_pass}")
    print(f"  {out['promotion_decision']}")
    print(f"\nwrote {OUT_DIR / 'trend_passive_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
