"""ADVERSARIAL ATTACK ON THE ONE BOOK THAT CLEARS 0.894 ON A LONG SAMPLE.

`trend + passive` (inverse-vol / ERC) measures Sharpe 0.9033 over 738 months. That is the
only combination in the study that clears the 30%/yr Kelly bar on a sample long enough for
the DSR bar to be low. Before it is reported as a result it gets the same killers that
killed risk parity in iteration 11.

  K1. THE BOND BULL. Iteration 11: excluding 1981-10 -> 2021-12 the equal-weight panel
      Sharpe falls 0.668 -> 0.439. Does the combined book survive?
  K2. DECADE STABILITY. A book that only works in two decades is not a 61-year book.
  K3. IS IT JUST THE BENCHMARK? Vol-matched active of the book against passive alone.
      Positive raw return with negative vol-matched active is not an edge -- the standing
      programme rule.
  K4. SUB-PERIOD SHARPE and the correlation that produces the diversification, both halves.
  K5. THE UNLEVERED TRUTH. What it compounds at 1x, before any leverage decision.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_attack_trend_passive
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.carry import vol_matched_active
from research.multiasset.panel import dsr_sharpe_bar

from research.sleeves._portfolio.portfolio_correlation_v2 import (
    CASH_PATH, FINANCING, MPY, SCHEMES, SOURCES, _load, load_source, ann_vol, cagr,
    leverage_curve, levered_total, max_dd, sharpe,
)
from research.sleeves._portfolio.portfolio_window_control import solve_leverage_for_dd

OUT_DIR = Path(__file__).resolve().parent
SPREAD = FINANCING["primary_bill_plus_150bp"]
BULL_FIRST, BULL_LAST = "1981-10-31", "2021-12-31"


def main() -> int:
    cash = _load(CASH_PATH, "US_CASH_13W")
    raw = {k: load_source(k) for k in SOURCES}
    series = {k: ((s - cash.reindex(s.index)).dropna() if SOURCES[k][2] == "total" else s)
              for k, s in raw.items()}

    f = pd.concat({"trend": series["trend"], "passive": series["passive_monthly"]},
                  axis=1).dropna()
    w = SCHEMES["inverse_vol"](f)
    port = pd.Series(f.to_numpy() @ w, index=f.index)
    cs = cash.reindex(f.index)
    out: dict = {"weights": {c: float(x) for c, x in zip(f.columns, w)},
                 "n": int(len(f)), "sharpe": sharpe(port), "vol": ann_vol(port)}

    # ---- K1. THE BOND BULL ---------------------------------------------------
    mask = ~((f.index >= BULL_FIRST) & (f.index <= BULL_LAST))
    ex = port[mask]
    out["K1_bond_bull"] = {
        "excluded_window": [BULL_FIRST, BULL_LAST],
        "n_outside": int(mask.sum()),
        "sharpe_full": sharpe(port), "sharpe_outside_bull": sharpe(ex),
        "fall": sharpe(port) - sharpe(ex),
        "trend_full": sharpe(f["trend"]), "trend_outside": sharpe(f["trend"][mask]),
        "passive_full": sharpe(f["passive"]), "passive_outside": sharpe(f["passive"][mask]),
        "iteration11_ew_full": 0.6678, "iteration11_ew_outside": 0.439,
        "clears_0894_outside": bool(sharpe(ex) >= 0.894),
    }

    # ---- K2. DECADE STABILITY -----------------------------------------------
    dec = {}
    for d, block in port.groupby(port.index.year // 10 * 10):
        if len(block) >= 24:
            dec[str(int(d))] = {"n": int(len(block)), "sharpe": sharpe(block),
                                "annual": float(block.mean() * MPY)}
    out["K2_decades"] = dec
    vals = [v["sharpe"] for v in dec.values()]
    out["K2_summary"] = {"min": min(vals), "max": max(vals),
                         "n_negative": sum(1 for v in vals if v < 0),
                         "n_below_0894": sum(1 for v in vals if v < 0.894),
                         "n_decades": len(vals)}

    # ---- K3. IS IT JUST THE BENCHMARK? --------------------------------------
    vm = vol_matched_active(port, f["passive"])
    out["K3_vs_passive"] = {
        "vol_matched_active_annual": vm["vol_matched_active_annual"],
        "vol_matched_active_tstat": vm["vol_matched_active_tstat"],
        "book_annual": float(port.mean() * MPY),
        "passive_annual": float(f["passive"].mean() * MPY),
        "raw_excess": float((port - f["passive"]).mean() * MPY),
    }
    vm_t = vol_matched_active(f["trend"], f["passive"])
    out["K3_trend_vs_passive"] = {
        "vol_matched_active_annual": vm_t["vol_matched_active_annual"],
        "vol_matched_active_tstat": vm_t["vol_matched_active_tstat"],
    }

    # ---- K4. SPLIT HALVES AND THE CORRELATION THAT DOES THE WORK ------------
    h = len(f) // 2
    out["K4_halves"] = {
        "first": {"window": [str(f.index[0].date()), str(f.index[h - 1].date())],
                  "sharpe_book": sharpe(port[:h]), "sharpe_trend": sharpe(f["trend"][:h]),
                  "sharpe_passive": sharpe(f["passive"][:h]),
                  "rho": float(np.corrcoef(f["trend"][:h], f["passive"][:h])[0, 1])},
        "second": {"window": [str(f.index[h].date()), str(f.index[-1].date())],
                   "sharpe_book": sharpe(port[h:]), "sharpe_trend": sharpe(f["trend"][h:]),
                   "sharpe_passive": sharpe(f["passive"][h:]),
                   "rho": float(np.corrcoef(f["trend"][h:], f["passive"][h:])[0, 1])},
    }

    # ---- K5. THE UNLEVERED TRUTH AND THE LADDER -----------------------------
    tot1 = levered_total(port.to_numpy(), cs.to_numpy(), 1.0, SPREAD)
    out["K5_unlevered"] = {"cagr_total_1x": cagr(tot1), "max_dd_1x": max_dd(tot1),
                           "mean_excess_annual": float(port.mean() * MPY),
                           "cash_mean_annual": float(cs.mean() * MPY)}
    out["K5_ladder"] = {lab: leverage_curve(port.to_numpy(), cs.to_numpy(), sp)
                        for lab, sp in FINANCING.items()}
    out["K5_dd50_bootstrap"] = solve_leverage_for_dd(
        port.to_numpy(), cs.to_numpy(), 0.50, use_bootstrap=True)
    out["K5_dd35_bootstrap"] = solve_leverage_for_dd(
        port.to_numpy(), cs.to_numpy(), 0.35, use_bootstrap=True)

    # ---- DSR at several trial counts ----------------------------------------
    out["dsr_bars"] = {str(n): dsr_sharpe_bar(len(f) / MPY, n_trials=n)
                       for n in (46, 104, 304)}

    (OUT_DIR / "portfolio_attack_trend_passive.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("ATTACK ON trend+passive [inverse_vol]  --  n=%d, Sharpe %.4f, vol %.2f%%"
          % (out["n"], out["sharpe"], out["vol"] * 100))
    print("  weights", {k: round(v, 4) for k, v in out["weights"].items()})
    print("=" * 100)
    k = out["K1_bond_bull"]
    print(f"K1 BOND BULL  excluding {BULL_FIRST[:7]}..{BULL_LAST[:7]}  ({k['n_outside']} months left)")
    print(f"    book    {k['sharpe_full']:+.4f} -> {k['sharpe_outside_bull']:+.4f}   "
          f"(fall {k['fall']:+.4f})   clears 0.894 outside: {k['clears_0894_outside']}")
    print(f"    trend   {k['trend_full']:+.4f} -> {k['trend_outside']:+.4f}")
    print(f"    passive {k['passive_full']:+.4f} -> {k['passive_outside']:+.4f}   "
          f"(iteration 11 measured its EW book 0.6678 -> 0.439)")
    print()
    print("K2 DECADES")
    for d, v in out["K2_decades"].items():
        print(f"    {d}s  n={v['n']:>3}  Sharpe {v['sharpe']:+.4f}  annual {v['annual']:+.2%}")
    s = out["K2_summary"]
    print(f"    -> {s['n_below_0894']} of {s['n_decades']} decades below 0.894; "
          f"{s['n_negative']} negative; range {s['min']:+.3f} .. {s['max']:+.3f}")
    print()
    v = out["K3_vs_passive"]
    print("K3 VOL-MATCHED ACTIVE vs PASSIVE ALONE (the standing programme rule)")
    print(f"    book vs passive     {v['vol_matched_active_annual']:+.2%}/yr  "
          f"NW t {v['vol_matched_active_tstat']:+.2f}")
    print(f"    trend alone vs passive {out['K3_trend_vs_passive']['vol_matched_active_annual']:+.2%}"
          f"/yr  NW t {out['K3_trend_vs_passive']['vol_matched_active_tstat']:+.2f}")
    print()
    print("K4 HALVES")
    for half, v in out["K4_halves"].items():
        print(f"    {half:>6} {v['window'][0]}..{v['window'][1]}  book {v['sharpe_book']:+.4f}  "
              f"trend {v['sharpe_trend']:+.4f}  passive {v['sharpe_passive']:+.4f}  "
              f"rho {v['rho']:+.4f}")
    print()
    u = out["K5_unlevered"]
    print(f"K5 UNLEVERED  CAGR {u['cagr_total_1x']:+.2%}  maxDD {u['max_dd_1x']:+.1%}  "
          f"(of which cash {u['cash_mean_annual']:+.2%})")
    for lab, c in out["K5_ladder"].items():
        print(f"    {lab:>28}  peak {c['peak_cagr']:+.2%} @{c['peak_leverage']:.2f}x "
              f"(DD {c['peak_max_dd']:+.1%})   DD<=50% {c['dd_cap_50']['cagr']:+.2%} "
              f"@{c['dd_cap_50']['leverage']:.2f}x   DD<=35% {c['dd_cap_35']['cagr']:+.2%}")
    b = out["K5_dd50_bootstrap"]
    print(f"    bootstrap-p95 DD<=50%: {b['cagr']:+.2%} @{b['leverage']:.2f}x")
    b = out["K5_dd35_bootstrap"]
    print(f"    bootstrap-p95 DD<=35%: {b['cagr']:+.2%} @{b['leverage']:.2f}x")
    print()
    print("DSR Sharpe bars at 61.5 years:", {k: round(v, 4) for k, v in out["dsr_bars"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
