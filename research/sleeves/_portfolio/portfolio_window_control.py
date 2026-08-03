"""WINDOW CONTROL, DRAWDOWN UNCERTAINTY, AND THE ITERATION-11 RECONCILIATION.

Three things `portfolio_correlation_v2` does not answer on its own:

1. **How much of the low-vol-window result is the WINDOW?** Low-vol exists only on
   1998-04 -> 2015-12. Every multi-asset sleeve is re-measured on that window and on its
   own full history, and the difference is the window premium that must be deducted.
2. **How badly does solving leverage against a SAMPLE MAXIMUM drawdown over-lever?**
   The observed max DD of a 213-month path is one draw. A stationary block bootstrap gives
   the distribution it came from.
3. **Does this machinery reproduce iteration 11's measured ladder?** That iteration
   measured a 12.30%/yr survivable rung and a 15.83%/yr ceiling on the 61.5-year panel.
   If the leverage engine here disagrees with that, the engine is wrong.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_window_control
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.sleeves._portfolio.portfolio_correlation_v2 import (
    CASH_PATH, FINANCING, SCHEMES, SOURCES, _load, load_source, ann_vol, cagr,
    is_ruined, leverage_curve, levered_total, max_dd, sharpe,
)

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260728)
BOOT_N = 5000
BLOCK = 12
LV_FIRST, LV_LAST = "1998-05-31", "2016-01-31"
SPREAD = FINANCING["primary_bill_plus_150bp"]

log = logging.getLogger("window_control")


def dd_bootstrap(total: np.ndarray) -> dict:
    """Circular moving-block bootstrap of the maximum drawdown of a levered path."""
    n = len(total)
    nb = int(math.ceil(n / BLOCK))
    starts = RNG.integers(0, n, size=(BOOT_N, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(BOOT_N, -1)[:, :n] % n
    paths = total[idx]
    ruined = (paths <= -1.0).any(axis=1)
    curve = np.cumprod(1.0 + np.where(paths <= -1.0, -0.999999, paths), axis=1)
    dd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
    dd = np.where(ruined, -1.0, dd)
    return {"observed": max_dd(total),
            "boot_median": float(np.percentile(dd, 50)),
            "boot_p75": float(np.percentile(dd, 75)),
            "boot_p95": float(np.percentile(dd, 5)),   # 5th pct of a negative = worst tail
            "boot_p99": float(np.percentile(dd, 1)),
            "p_ruin": float(ruined.mean()),
            "p_worse_than_observed": float((dd < max_dd(total)).mean())}


def solve_leverage_for_dd(x: np.ndarray, cash: np.ndarray, cap: float,
                          *, use_bootstrap: bool) -> dict:
    """Largest leverage whose drawdown stays inside `cap`.

    `use_bootstrap=False` uses the single observed path (what the sample maximum permits).
    `use_bootstrap=True` uses the 95th-percentile bootstrap drawdown, i.e. the leverage
    that survives a bad-but-not-extreme resampling of the SAME months.
    """
    best = None
    for L in np.arange(0.25, 30.001, 0.05):
        tot = levered_total(x, cash, float(L), SPREAD)
        if is_ruined(tot):
            break
        d = dd_bootstrap(tot)["boot_p95"] if use_bootstrap else max_dd(tot)
        if abs(d) > cap:
            break
        best = {"leverage": float(L), "cagr": cagr(tot), "max_dd_observed": max_dd(tot),
                "dd_used": d, "vol": ann_vol(tot)}
    return best or {"leverage": 0.0, "cagr": 0.0, "max_dd_observed": 0.0,
                    "dd_used": 0.0, "vol": 0.0}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    cash = _load(CASH_PATH, "US_CASH_13W")
    raw = {k: load_source(k) for k in SOURCES}
    series = {}
    for k, s in raw.items():
        series[k] = (s - cash.reindex(s.index)).dropna() if SOURCES[k][2] == "total" else s

    out: dict = {}

    # ---- 1. WINDOW CONTROL ---------------------------------------------------
    win: dict[str, dict[str, Any]] = {}
    for k in ["trend", "seasonal", "defensive", "carry", "passive_monthly", "passive_daily"]:
        s = series[k]
        sub = s.loc[LV_FIRST:LV_LAST]
        row: dict[str, Any] = {
               "n_full": int(len(s)), "sharpe_full": sharpe(s), "vol_full": ann_vol(s),
               "n_lowvol_window": int(len(sub)),
               "sharpe_lowvol_window": sharpe(sub) if len(sub) > 24 else None,
               "vol_lowvol_window": ann_vol(sub) if len(sub) > 24 else None}
        if row["sharpe_lowvol_window"] is not None:
            row["sharpe_premium"] = row["sharpe_lowvol_window"] - row["sharpe_full"]
            c_full = cash.reindex(s.index).to_numpy()
            c_sub = cash.reindex(sub.index).to_numpy()
            row["dd50_cagr_full"] = leverage_curve(
                s.to_numpy(), c_full, SPREAD)["dd_cap_50"]["cagr"]
            row["dd50_cagr_lowvol_window"] = leverage_curve(
                sub.to_numpy(), c_sub, SPREAD)["dd_cap_50"]["cagr"]
            row["dd50_cagr_premium"] = (row["dd50_cagr_lowvol_window"]
                                        - row["dd50_cagr_full"])
        win[k] = row
    out["window_control"] = win
    prem = [v["sharpe_premium"] for v in win.values() if v.get("sharpe_premium") is not None]
    out["mean_sharpe_premium_on_lowvol_window"] = float(np.mean(prem))

    # ---- 2. DRAWDOWN UNCERTAINTY + BOOTSTRAP-SOLVED LEVERAGE -----------------
    books = {
        "lowvol+trend [equal_weight]": (["lowvol", "trend"], "equal_weight"),
        "lowvol+trend [erc]": (["lowvol", "trend"], "erc"),
        "lowvol+trend+carry [erc]": (["lowvol", "trend", "carry"], "erc"),
        "lowvol+trend+defensive [erc]": (["lowvol", "trend", "defensive"], "erc"),
        "lowvol+trend+carry+defensive [erc]": (
            ["lowvol", "trend", "carry", "defensive"], "erc"),
        "trend+carry [erc]": (["trend", "carry"], "erc"),
        "trend [1x]": (["trend"], "equal_weight"),
        "passive_monthly [1x]": (["passive_monthly"], "equal_weight"),
        "lowvol [1x]": (["lowvol"], "equal_weight"),
    }
    dd: dict[str, dict[str, Any]] = {}
    for label, (combo, scheme) in books.items():
        f = pd.concat({c: series[c] for c in combo}, axis=1).dropna()
        w = SCHEMES[scheme](f) if len(combo) > 1 else np.array([1.0])
        port = f.to_numpy() @ w
        cs = cash.reindex(f.index).to_numpy()
        s = sharpe(port)
        lev_half = s / (2.0 * ann_vol(port))
        entry: dict[str, Any] = {
            "combo": combo, "scheme": scheme, "n": int(len(f)),
            "first": str(f.index.min().date()), "last": str(f.index.max().date()),
            "sharpe": s, "vol": ann_vol(port),
            "half_kelly_leverage": lev_half,
            "half_kelly_dd_bootstrap": dd_bootstrap(
                levered_total(port, cs, lev_half, SPREAD)),
            "dd50_observed_path": solve_leverage_for_dd(port, cs, 0.50, use_bootstrap=False),
            "dd50_bootstrap_p95": solve_leverage_for_dd(port, cs, 0.50, use_bootstrap=True),
            "dd60_observed_path": solve_leverage_for_dd(port, cs, 0.60, use_bootstrap=False),
            "dd60_bootstrap_p95": solve_leverage_for_dd(port, cs, 0.60, use_bootstrap=True),
        }
        dd[label] = entry
        log.info("%-40s S %.3f  dd50 obs %.2f%%  dd50 boot %.2f%%",
                 label, s, entry["dd50_observed_path"]["cagr"] * 100,
                 entry["dd50_bootstrap_p95"]["cagr"] * 100)
    out["drawdown_uncertainty"] = dd

    # ---- 3. ITERATION-11 RECONCILIATION -------------------------------------
    s = series["passive_monthly"]
    cs = cash.reindex(s.index).to_numpy()
    curve = leverage_curve(s.to_numpy(), cs, SPREAD)
    out["iteration11_reconciliation"] = {
        "note": ("Iteration 11 measured, on the 61.5-year 18-instrument panel: peak "
                 "compound 15.83%/yr at -87.8% DD, and 12.30%/yr at a <=50% DD cap, from "
                 "plain equal weight. The book here is the monthly equal-weight benchmark "
                 "series persisted by the trend sleeve, levered STATICALLY, whereas "
                 "iteration 11 levered to a VOLATILITY TARGET (time-varying leverage) and "
                 "charged its own rebalancing costs. The two should agree in magnitude, "
                 "not to the digit."),
        "measured_here": curve,
        "iteration11_peak_cagr": 0.1583,
        "iteration11_dd50_cagr": 0.1230,
        "sharpe_here": sharpe(s),
        "iteration11_sharpe": 0.6678,
    }

    # financing sensitivity on the same book -- iteration 11's largest single lever
    out["financing_sensitivity_passive"] = {
        lab: leverage_curve(s.to_numpy(), cs, sp) for lab, sp in FINANCING.items()}

    # and on the best portfolio book
    f = pd.concat({c: series[c] for c in ["lowvol", "trend"]}, axis=1).dropna()
    port = f.to_numpy() @ SCHEMES["equal_weight"](f)
    cs2 = cash.reindex(f.index).to_numpy()
    out["financing_sensitivity_lowvol_trend"] = {
        lab: leverage_curve(port, cs2, sp) for lab, sp in FINANCING.items()}

    (OUT_DIR / "portfolio_window_control.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 104)
    print("WINDOW CONTROL -- every multi-asset sleeve on its OWN history vs on low-vol's 1998-2015")
    print("=" * 104)
    print(f"{'sleeve':>18} {'n full':>7} {'S full':>8} {'n win':>6} {'S window':>9} "
          f"{'premium':>9} {'DD50 full':>10} {'DD50 win':>10} {'premium':>9}")
    for k, v in win.items():
        if v.get("sharpe_premium") is None:
            continue
        print(f"{k:>18} {v['n_full']:>7} {v['sharpe_full']:>+8.4f} "
              f"{v['n_lowvol_window']:>6} {v['sharpe_lowvol_window']:>+9.4f} "
              f"{v['sharpe_premium']:>+9.4f} {v['dd50_cagr_full']:>+10.2%} "
              f"{v['dd50_cagr_lowvol_window']:>+10.2%} {v['dd50_cagr_premium']:>+9.2%}")
    print(f"\n  mean Sharpe premium on the low-vol window: "
          f"{out['mean_sharpe_premium_on_lowvol_window']:+.4f}")

    print("\n" + "=" * 104)
    print("DRAWDOWN UNCERTAINTY -- observed path vs block-bootstrap, and the leverage each permits")
    print("=" * 104)
    print(f"{'book':>38} {'n':>5} {'S':>7} {'hK DD obs':>10} {'hK DD p95':>10} {'P(ruin)':>8} "
          f"{'DD50 obs':>16} {'DD50 boot p95':>18}")
    for k, v in dd.items():
        b = v["half_kelly_dd_bootstrap"]
        o, p = v["dd50_observed_path"], v["dd50_bootstrap_p95"]
        print(f"{k:>38} {v['n']:>5} {v['sharpe']:>+7.3f} {b['observed']:>+10.1%} "
              f"{b['boot_p95']:>+10.1%} {b['p_ruin']:>8.1%} "
              f"{o['cagr']:>+9.2%}@{o['leverage']:>5.2f}x {p['cagr']:>+11.2%}@{p['leverage']:>5.2f}x")

    print("\n" + "=" * 104)
    print("ITERATION-11 RECONCILIATION (monthly equal-weight benchmark, 61.5yr, bill+150bp)")
    print("=" * 104)
    r = out["iteration11_reconciliation"]
    print(f"  Sharpe here {r['sharpe_here']:.4f}   iteration 11 {r['iteration11_sharpe']:.4f}")
    print(f"  peak compound here {curve['peak_cagr']:+.2%} at {curve['peak_leverage']:.2f}x "
          f"(DD {curve['peak_max_dd']:+.1%})   iteration 11 {r['iteration11_peak_cagr']:+.2%}")
    print(f"  DD<=50% here {curve['dd_cap_50']['cagr']:+.2%} at "
          f"{curve['dd_cap_50']['leverage']:.2f}x   iteration 11 {r['iteration11_dd50_cagr']:+.2%}")
    print(f"  ruin leverage {curve['ruin_leverage']}")

    print("\n" + "=" * 104)
    print("FINANCING SENSITIVITY -- the largest single lever iteration 11 found")
    print("=" * 104)
    for name, blk in (("passive EW 61.5yr", out["financing_sensitivity_passive"]),
                      ("lowvol+trend EW 17.75yr", out["financing_sensitivity_lowvol_trend"])):
        print(f"  {name}")
        for lab, c in blk.items():
            print(f"      {lab:>28}  peak {c['peak_cagr']:>+7.2%}  "
                  f"DD<=50% {c['dd_cap_50']['cagr']:>+7.2%} at {c['dd_cap_50']['leverage']:>5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
