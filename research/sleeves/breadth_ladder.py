"""Re-run iteration 11's LEVERAGE LADDER on the expanded panel.

    .venv/Scripts/python.exe -m research.sleeves.breadth_ladder

Every mechanic — weights, eligibility, the 36-month volatility estimate, the leverage
scaler, the explicit financing charge on borrowed notional, the drawdown/recovery report
— is IMPORTED from ``research/sleeves/riskparity.py`` and ``multiasset_trend.py``
unmodified. Nothing is re-implemented, so the only thing that differs between iteration
11's numbers and these is the universe.

THE CONTROL comes first. The runner rebuilds iteration 11's headline on the original 18
(12.2955% compound at -47.2874% drawdown, and a 15.83% peak) and refuses to report the
expanded numbers if it cannot reproduce them. Without that, a difference in the headline
could be a difference in the harness rather than in the breadth.

Judged on MEASURED COMPOUND RETURN, never on ``S^2/2``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.breadth_universe import (
    ALL_BLOCKS,
    ORIGINAL_18,
    UNIVERSES,
    load_combined_panel,
)
from research.sleeves.multiasset_trend import (
    MONTHS,
    annual_sharpe,
    decade_sharpe,
    effective_n,
)
from research.sleeves.riskparity import (
    COSTS,
    FINANCING,
    VOL_TARGETS,
    build_book,
    drawdown_report,
    ladder,
    levered,
    weight_concentration,
)

_OUT = Path("research/sleeves/_breadth")
_DATA = Path("_data/multiasset")

SCHEMES = ("ew", "rp_naive", "rp_bucket")
HEAD_COST = "10bps"                        # the conservative bound is the headline
HEAD_FIN = "primary_bill_plus_150bp"
RETAIL_FIN = "retail_bill_plus_300bp"

# Iteration 11's recorded headline on the original 18, reproduced as a control.
CONTROL = {"compound_dd50": 0.1229548756, "mdd_dd50": -0.472873856, "peak_compound": 0.1583}
CONTROL_TOL = 1e-6
PEAK_TOL = 0.0005

# The universes reported. Each answers a stated question.
RUNS: dict[str, str] = {
    "original_18": "CONTROL — iteration 11's universe, must reproduce its headline",
    "expanded_37": "HEADLINE — every free-data addition, front-month futures as fetched",
    "expanded_no_livestock": "drops the two structurally roll-contaminated livestock series",
    "expanded_no_rollcontam": "drops all five series whose splice signature beats NATGAS_F",
    "expanded_no_vol": "drops VIXY, which cannot be held long by anyone",
    "expanded_long_history_only": "only additions with 15+ years of history",
    "expanded_roll_managed": "front-month ag/livestock REPLACED by the roll-managed, "
                             "actually-holdable ETFs — what a person could really own",
}

# The one run that changes the DATA rather than the key list.
SUBSTITUTE_RUNS: frozenset[str] = frozenset({"expanded_roll_managed"})
UNIVERSE_ALIAS: dict[str, str] = {"expanded_roll_managed": "expanded_37"}


def _cash() -> pd.Series:
    return pd.read_parquet(_DATA / "cash_monthly.parquet")["US_CASH_13W"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 10)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def fine_grid(bk, cash: pd.Series, *, cost: float, spread: float,
              since: str | None = None) -> dict:
    """Iteration 11's descriptive fine grid: tau from 5% to 60% in 1% steps.

    Leverage is a pure scaling, not a searched parameter, so this is DESCRIPTIVE — it
    locates the maximum of the leverage-return curve and the point at which the drawdown
    constraint binds. Both are properties of the curve, not choices.
    """
    rows = []
    for tau in np.arange(0.05, 0.605, 0.01):
        L = levered(bk, cash, tau=float(tau), cost=cost, spread=spread)
        # ``since`` slices the RESULT, never the input: eligibility, the 36-month
        # volatility estimate and the leverage scaler are still computed on the full
        # history, exactly as a real book would have been run into that date.
        tot = L["total"] if since is None else L["total"].loc[since:]
        d = drawdown_report(tot)
        rows.append({"tau": round(float(tau), 4), "compound": d["compound_annual"],
                     "mdd": d["max_drawdown"], "ruin": d["ruin"],
                     "mean_leverage": float(L["k"].mean())})
    live = [r for r in rows if not r["ruin"] and np.isfinite(r["compound"])]
    peak = max(live, key=lambda r: r["compound"]) if live else None
    out: dict = {"curve": rows, "peak": peak}
    for lim in (0.35, 0.50, 0.60):
        ok = [r for r in live if r["mdd"] >= -lim]
        out[f"best_at_mdd_le_{int(lim * 100)}"] = (
            max(ok, key=lambda r: r["compound"]) if ok else None)
    return out


def run_universe(name: str, keys: tuple[str, ...], cash: pd.Series,
                 *, substitute_roll_managed: bool = False) -> dict:
    """Full ladder for one universe: every scheme, both cost brackets, both financings."""
    x, interior = load_combined_panel(tuple(keys),
                                      substitute_roll_managed=substitute_roll_managed)
    blocks = {b: tuple(k for k in ks if k in x.columns) for b, ks in ALL_BLOCKS.items()}
    blocks = {b: ks for b, ks in blocks.items() if ks}
    books = {s: build_book(s, x, interior, s, blocks=blocks) for s in SCHEMES}

    span = books["ew"].excess.dropna()
    years = len(span) / MONTHS
    res: dict = {
        "universe": list(keys),
        "sample": {"first": str(span.index[0].date()), "last": str(span.index[-1].date()),
                   "months": int(len(span)), "years": years,
                   "n_instruments": len(keys),
                   "mean_eligible": float(books["ew"].elig_count.reindex(span.index).mean()),
                   "mean_eligible_last_15y":
                       float(books["ew"].elig_count.reindex(span.index).loc["2011":].mean())},
    }

    one = pd.Series(1.0, index=x.index)
    res["unlevered"] = {}
    for s, bk in books.items():
        row = {}
        for cl, c in COSTS.items():
            L = levered(bk, cash, tau=0.0, cost=c, spread=0.0, k_override=one)
            d = drawdown_report(L["total"])
            row[cl] = {"sharpe": annual_sharpe(L["net_excess"]),
                       "mean_excess_annual": float(L["net_excess"].mean() * MONTHS),
                       "vol_annual": float(L["net_excess"].std(ddof=1) * math.sqrt(MONTHS)),
                       "compound_annual_total": d["compound_annual"],
                       "max_drawdown_total": d["max_drawdown"],
                       "turnover_per_year": float(L["turnover"].mean() * MONTHS)}
        res["unlevered"][s] = row

    res["ladder"] = {
        s: {fl: {cl: ladder(books[s], cash, cost_label=cl, financing_label=fl)
                 for cl in COSTS}
            for fl in (HEAD_FIN, RETAIL_FIN)}
        for s in SCHEMES
    }

    # headline: highest compound at a survivable drawdown, over every scheme and rung
    surv: dict = {}
    for lim in (0.35, 0.50, 0.60):
        best = None
        for s in SCHEMES:
            for tau, row in res["ladder"][s][HEAD_FIN][HEAD_COST].items():
                if row["max_drawdown"] >= -lim and not row["ruin"]:
                    if best is None or row["compound_annual"] > best["compound_annual"]:
                        best = {"book": s, "target_vol": float(tau), **row}
        surv[f"mdd_le_{int(lim * 100)}pct"] = best
    res["survivable"] = surv

    res["fine_grid"] = {
        s: fine_grid(books[s], cash, cost=COSTS[HEAD_COST], spread=FINANCING[HEAD_FIN])
        for s in SCHEMES
    }
    res["fine_grid_retail"] = {
        "ew": fine_grid(books["ew"], cash, cost=COSTS[HEAD_COST],
                        spread=FINANCING[RETAIL_FIN])
    }
    peaks = {s: res["fine_grid"][s]["peak"] for s in SCHEMES}
    res["peak_any_leverage"] = max(
        (dict(p, book=s) for s, p in peaks.items() if p), key=lambda p: p["compound"])
    best50 = {s: res["fine_grid"][s]["best_at_mdd_le_50"] for s in SCHEMES}
    res["best_at_mdd_le_50_fine"] = max(
        (dict(p, book=s) for s, p in best50.items() if p), key=lambda p: p["compound"])

    res["dsr"] = {
        "years": years,
        "bars": {str(n): dsr_sharpe_bar(years, n_trials=n) for n in (32, 46, 56, 304)},
        "sharpe_needed_for_30pct_half_kelly": math.sqrt(0.30 * 8.0 / 3.0),
    }
    res["decades"] = {
        s: decade_sharpe(levered(books[s], cash, tau=0.0, cost=COSTS[HEAD_COST],
                                 spread=0.0, k_override=one)["net_excess"])
        for s in SCHEMES
    }
    res["concentration_ew"] = weight_concentration(books["ew"].w, books["ew"].live)

    # ── THE MOST FAVOURABLE TEST ─────────────────────────────────────────────
    # On 61.5 years the additions are only live for the last stretch, so the full-sample
    # ladder understates whatever breadth is worth. This evaluates the SAME book only
    # over windows in which the additions are actually carrying weight. It is the
    # friendliest test that can honestly be run, and the DSR bar rises accordingly.
    ev: dict = {}
    for since in ("2004-01-01", "2011-01-01"):
        fg = {s: fine_grid(books[s], cash, cost=COSTS[HEAD_COST],
                           spread=FINANCING[HEAD_FIN], since=since)
              for s in SCHEMES}
        one_e = pd.Series(1.0, index=x.index)
        unlev = {s: annual_sharpe(levered(books[s], cash, tau=0.0, cost=COSTS[HEAD_COST],
                                          spread=0.0, k_override=one_e)["net_excess"].loc[since:])
                 for s in SCHEMES}
        peak_rows = [dict(fg[s]["peak"], book=s) for s in SCHEMES if fg[s]["peak"]]
        b50 = [dict(fg[s]["best_at_mdd_le_50"], book=s)
               for s in SCHEMES if fg[s]["best_at_mdd_le_50"]]
        yrs = float(len(books["ew"].excess.dropna().loc[since:]) / MONTHS)
        # The decomposition that explains the headline. S = s * sqrt(N_eff) is only a
        # gain if the ADDED bets carry the same per-bet Sharpe s as the existing ones.
        # This measures s directly instead of assuming it.
        sl = x.loc[since:]
        sl = sl.loc[:, [c for c in sl.columns if sl[c].notna().sum() >= 12]]
        neff = effective_n(sl.corr())
        per_inst = {c: annual_sharpe(sl[c]) for c in sl.columns}
        ev[since[:4]] = {
            "years": yrs,
            "mean_eligible": float(books["ew"].elig_count.loc[since:].mean()),
            "unlevered_sharpe": unlev,
            "peak_any_leverage": max(peak_rows, key=lambda p: p["compound"]) if peak_rows else None,
            "best_at_mdd_le_50": max(b50, key=lambda p: p["compound"]) if b50 else None,
            "dsr_bar_n46": dsr_sharpe_bar(yrs, n_trials=46),
            "n_eff": neff,
            "per_bet_sharpe_implied": unlev["ew"] / math.sqrt(neff) if neff > 0 else float("nan"),
            "mean_instrument_sharpe": float(np.nanmean(list(per_inst.values()))),
            "instrument_sharpe": per_inst,
        }
    res["evaluated_since"] = ev
    return res


def main() -> dict:
    cash = _cash()
    out: dict = {"runs": {}, "rationale": RUNS}

    # ── CONTROL FIRST ─────────────────────────────────────────────────────────
    ctrl = run_universe("original_18", ORIGINAL_18, cash)
    got50 = ctrl["survivable"]["mdd_le_50pct"]
    got_peak = ctrl["peak_any_leverage"]["compound"]
    checks = {
        "compound_dd50": (got50["compound_annual"], CONTROL["compound_dd50"],
                          abs(got50["compound_annual"] - CONTROL["compound_dd50"]) <= CONTROL_TOL),
        "mdd_dd50": (got50["max_drawdown"], CONTROL["mdd_dd50"],
                     abs(got50["max_drawdown"] - CONTROL["mdd_dd50"]) <= CONTROL_TOL),
        "peak_compound": (got_peak, CONTROL["peak_compound"],
                          abs(got_peak - CONTROL["peak_compound"]) <= PEAK_TOL),
    }
    out["control"] = {k: {"measured": a, "recorded": b, "ok": bool(c)}
                      for k, (a, b, c) in checks.items()}
    out["control"]["all_ok"] = all(c for _, _, c in checks.values())
    for k, (a, b, c) in checks.items():
        print(f"CONTROL {'OK ' if c else 'FAIL'} {k:16s} measured {a:+.6f}  recorded {b:+.6f}")
    if not out["control"]["all_ok"]:
        raise AssertionError("CONTROL FAILED — the harness does not reproduce iteration 11; "
                             "expanded numbers would not be comparable.")
    out["runs"]["original_18"] = ctrl

    # ── the expanded runs ─────────────────────────────────────────────────────
    for name in RUNS:
        if name in out["runs"]:
            continue
        keys = UNIVERSES[UNIVERSE_ALIAS.get(name, name)]
        out["runs"][name] = run_universe(
            name, keys, cash, substitute_roll_managed=name in SUBSTITUTE_RUNS)

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "ladder.json").write_text(json.dumps(_jsonable(out), indent=2, sort_keys=True),
                                      encoding="utf-8")

    # ── print ────────────────────────────────────────────────────────────────
    print("\n=== HEADLINE: the two numbers that are the entire result ===")
    print(f"{'universe':30s}{'n':>4s}{'yrs':>7s}{'DD<=50% compound':>19s}{'its DD':>9s}"
          f"{'peak compound':>15s}{'its DD':>9s}{'Sharpe':>8s}")
    for name, r in out["runs"].items():
        b = r["best_at_mdd_le_50_fine"]
        p = r["peak_any_leverage"]
        print(f"{name:30s}{r['sample']['n_instruments']:>4d}{r['sample']['years']:>7.1f}"
              f"{100 * b['compound']:>18.2f}%{100 * b['mdd']:>8.1f}%"
              f"{100 * p['compound']:>14.2f}%{100 * p['mdd']:>8.1f}%"
              f"{r['unlevered']['ew'][HEAD_COST]['sharpe']:>8.4f}")

    print("\n=== EQUAL-WEIGHT LADDER, 10bps, bill+150bp (compound / max DD) ===")
    print(f"{'universe':30s}" + "".join(f"{f'tau={t:.0%}':>18s}" for t in VOL_TARGETS))
    for name, r in out["runs"].items():
        line = f"{name:30s}"
        for t in VOL_TARGETS:
            row = r["ladder"]["ew"][HEAD_FIN][HEAD_COST][f"{t:.2f}"]
            line += f"{100 * row['compound_annual']:>9.2f}%/{100 * row['max_drawdown']:>6.0f}%"
        print(line)

    print("\n=== SAME, at RETAIL financing (bill+300bp) ===")
    print(f"{'universe':30s}" + "".join(f"{f'tau={t:.0%}':>18s}" for t in VOL_TARGETS))
    for name, r in out["runs"].items():
        line = f"{name:30s}"
        for t in VOL_TARGETS:
            row = r["ladder"]["ew"][RETAIL_FIN][HEAD_COST][f"{t:.2f}"]
            line += f"{100 * row['compound_annual']:>9.2f}%/{100 * row['max_drawdown']:>6.0f}%"
        print(line)

    print("\n=== 2bps cost bracket, equal weight, bill+150bp ===")
    for name, r in out["runs"].items():
        rungs = [r["ladder"]["ew"][HEAD_FIN]["2bps"][f"{t:.2f}"] for t in VOL_TARGETS]
        ok = [q for q in rungs if q["max_drawdown"] >= -0.50 and not q["ruin"]]
        best = max(ok, key=lambda q: q["compound_annual"]) if ok else None
        print(f"  {name:30s} best rung at DD<=50%: "
              f"{100 * best['compound_annual']:6.2f}% at {100 * best['max_drawdown']:5.1f}% "
              f"(tau {best['target_vol']:.0%})" if best else f"  {name:30s} none survivable")

    print("\n=== THE FRIENDLIEST TEST: evaluated only while the additions are live ===")
    for since in ("2004", "2011"):
        print(f"-- from {since} --")
        print(f"{'universe':30s}{'yrs':>6s}{'elig':>6s}{'Sharpe':>8s}{'DSRbar':>8s}"
              f"{'DD<=50%':>10s}{'peak':>9s}{'itsDD':>8s}")
        for name, r in out["runs"].items():
            e = r["evaluated_since"][since]
            b, p = e["best_at_mdd_le_50"], e["peak_any_leverage"]
            print(f"{name:30s}{e['years']:>6.1f}{e['mean_eligible']:>6.1f}"
                  f"{e['unlevered_sharpe']['ew']:>8.3f}{e['dsr_bar_n46']:>8.3f}"
                  f"{100 * b['compound']:>9.2f}%{100 * p['compound']:>8.2f}%"
                  f"{100 * p['mdd']:>7.1f}%")
    return out


if __name__ == "__main__":
    main()
