"""IS LOW-VOL'S DIVERSIFICATION ECONOMIC, OR IS IT THE 1998-2016 STOCK/BOND REGIME?

Once the one-month dating defect is repaired, low-vol measures rho -0.392 to carry and
-0.303 to defensive, and those two negative numbers are what lift the multi-sleeve books
above Sharpe 1.2. They have to be attacked before they are believed.

The suspicion is specific and testable. Low-vol is a LONG-ONLY US EQUITY book: after the
dating repair it correlates **+0.571** with the passive multi-asset benchmark. Carry is a
bond/FX carry book and defensive/BAB is levered bonds. Over 1998-2016 equities and bonds
were strongly NEGATIVELY correlated -- the post-1998 "risk-off" regime -- and that regime
is not a constant: it was positive before 1998 and has been positive again since 2022.

If that is the explanation, then a LONG-HISTORY EQUITY PROXY standing in for low-vol will
show the same negative correlations on 1998-2016 and NOT show them on the full sample.
The proxy has 61 years of history where low-vol has 17.75, so it can answer the question
low-vol's own sample cannot.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_attack_lowvol_regime
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.sleeves._portfolio.portfolio_correlation_v2 import (
    CASH_PATH, REPO, SOURCES, _load, load_source,
)

OUT_DIR = Path(__file__).resolve().parent
LV_FIRST, LV_LAST = "1998-05-31", "2016-01-31"


def rho(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    f = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(f) < 24:
        return float("nan"), int(len(f))
    return float(np.corrcoef(f["a"], f["b"])[0, 1]), int(len(f))


def main() -> int:
    cash = _load(CASH_PATH, "US_CASH_13W")
    series = {}
    for k in SOURCES:
        s = load_source(k)
        series[k] = (s - cash.reindex(s.index)).dropna() if SOURCES[k][2] == "total" else s

    panel = pd.read_parquet(REPO / "_data" / "multiasset" / "returns_monthly.parquet")
    panel.index = pd.DatetimeIndex(panel.index).to_period("M").to_timestamp(
        how="end").normalize()
    # Equity proxies for low-vol, as EXCESS returns over the same 13-week bill.
    proxies = {}
    for key in ("SPX", "NASDAQ", "DJIA"):
        proxies[key] = (panel[key] - cash.reindex(panel.index)).dropna()
    proxies["EQ_EW3"] = pd.concat(list(proxies.values()), axis=1).dropna().mean(axis=1)
    # Bond excess return, for the raw stock/bond regime measurement.
    bond = (panel["US10Y_TR"] - cash.reindex(panel.index)).dropna()

    out: dict = {}

    # ---- A. THE RAW STOCK/BOND REGIME, BY DECADE ----------------------------
    f = pd.concat({"eq": proxies["SPX"], "bd": bond}, axis=1).dropna()
    dec = {}
    for d, blk in f.groupby(f.index.year // 10 * 10):
        if len(blk) >= 24:
            dec[str(int(d))] = {"n": int(len(blk)),
                                "rho_spx_bond": float(np.corrcoef(blk["eq"], blk["bd"])[0, 1])}
    out["A_stock_bond_regime_by_decade"] = dec
    r_full, n_full = rho(proxies["SPX"], bond)
    r_win, n_win = rho(proxies["SPX"].loc[LV_FIRST:LV_LAST], bond.loc[LV_FIRST:LV_LAST])
    out["A_summary"] = {"rho_full": r_full, "n_full": n_full,
                        "rho_lowvol_window": r_win, "n_lowvol_window": n_win,
                        "shift": r_win - r_full}

    # ---- B. THE PROXY TEST: low-vol replaced by a long-history equity book ---
    tests = {}
    for pname, p in proxies.items():
        row = {}
        for other in ("trend", "carry", "seasonal", "defensive", "passive_monthly"):
            o = series[other]
            r_f, n_f = rho(p, o)
            r_w, n_w = rho(p.loc[LV_FIRST:LV_LAST], o.loc[LV_FIRST:LV_LAST])
            r_out, n_out = rho(p.drop(p.loc[LV_FIRST:LV_LAST].index, errors="ignore"),
                               o.drop(o.loc[LV_FIRST:LV_LAST].index, errors="ignore"))
            row[other] = {"rho_full": r_f, "n_full": n_f,
                          "rho_lowvol_window": r_w, "n_lowvol_window": n_w,
                          "rho_outside_window": r_out, "n_outside_window": n_out,
                          "window_minus_outside": r_w - r_out}
        tests[pname] = row
    out["B_equity_proxy"] = tests

    # ---- C. LOW-VOL'S OWN MEASURED CORRELATIONS, for the side-by-side --------
    lv = series["lowvol"]
    out["C_lowvol_measured"] = {
        other: {"rho": rho(lv, series[other])[0], "n": rho(lv, series[other])[1]}
        for other in ("trend", "carry", "seasonal", "defensive", "passive_monthly")}
    out["C_lowvol_vs_equity_proxy"] = {
        p: {"rho": rho(lv, proxies[p])[0], "n": rho(lv, proxies[p])[1]} for p in proxies}

    (OUT_DIR / "portfolio_attack_lowvol_regime.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    print("=" * 104)
    print("A. THE STOCK/BOND REGIME IS NOT A CONSTANT  (SPX excess vs US10Y excess)")
    print("=" * 104)
    for d, v in dec.items():
        print(f"    {d}s  n={v['n']:>3}  rho(SPX, 10y) {v['rho_spx_bond']:+.4f}")
    a = out["A_summary"]
    print(f"    full sample   rho {a['rho_full']:+.4f} (n={a['n_full']})")
    print(f"    low-vol window rho {a['rho_lowvol_window']:+.4f} (n={a['n_lowvol_window']})"
          f"   shift {a['shift']:+.4f}")

    print("\n" + "=" * 104)
    print("B. THE PROXY TEST -- a LONG-HISTORY equity book standing in for low-vol")
    print("   If low-vol's negative rhos are the regime, the proxy reproduces them ON the")
    print("   window and loses them OFF it.")
    print("=" * 104)
    print(f"{'proxy':>10} {'vs':>17} {'full':>9} {'on window':>11} {'off window':>11} "
          f"{'win - off':>11}")
    for pname, row in tests.items():
        for other, v in row.items():
            print(f"{pname:>10} {other:>17} {v['rho_full']:>+9.4f} "
                  f"{v['rho_lowvol_window']:>+11.4f} {v['rho_outside_window']:>+11.4f} "
                  f"{v['window_minus_outside']:>+11.4f}")
        print()

    print("=" * 104)
    print("C. LOW-VOL'S OWN NUMBERS, SIDE BY SIDE  (n = 213 / 144)")
    print("=" * 104)
    for k, v in out["C_lowvol_measured"].items():
        pr = tests["EQ_EW3"][k]
        print(f"    lowvol ~ {k:<17} {v['rho']:>+8.4f} (n={v['n']:>3})   "
              f"equity proxy on the SAME window {pr['rho_lowvol_window']:>+8.4f}   "
              f"proxy OFF window {pr['rho_outside_window']:>+8.4f}")
    print()
    print("    low-vol vs the equity proxies themselves:",
          {k: round(v["rho"], 4) for k, v in out["C_lowvol_vs_equity_proxy"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
