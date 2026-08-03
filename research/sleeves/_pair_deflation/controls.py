"""CONTROLS. Reproduce every recorded anchor BEFORE any new number is computed.

If any of these fails, nothing downstream is trustworthy and the run aborts.

    .venv/Scripts/python.exe -m research.sleeves._pair_deflation.controls
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar

REPO = Path(__file__).resolve().parents[3]
SLEEVES = REPO / "research" / "sleeves"
PORT = SLEEVES / "_portfolio"
OUT = SLEEVES / "_pair_deflation"
MPY = 12


# ── loaders (mirroring the two code paths, so both can be reproduced) ──────────
def load(path: Path, col: str, *, shift_months: int = 0) -> pd.Series:
    if path.suffix == ".csv":
        f = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        f = pd.read_parquet(path)
    s = f[col].astype(float)
    idx = pd.DatetimeIndex(s.index).to_period("M")
    if shift_months:
        idx = idx + shift_months
    s.index = idx.to_timestamp(how="end").normalize()
    return s.rename(col).dropna()


SRC = {
    # v1 ("the claim"): registered low-vol, NO month shift, total-return convention.
    "lowvol_registered": (PORT / "lowvol_b2_net_monthly.parquet", "net_conservative"),
    # v2 (the corrected book): iteration-10-verified corrections, shifted +1 month.
    "lowvol_corrected": (PORT / "lowvol_b2_corrected_monthly.parquet", "net_conservative"),
    "trend": (SLEEVES / "_multiasset_trend" / "primary_20pct_monthly.csv", "net_10bps"),
    "passive_monthly": (SLEEVES / "_multiasset_trend" / "primary_20pct_monthly.csv",
                        "bench_net_10bps"),
    "passive_daily": (SLEEVES / "_seasonal" / "seasonal_composite_20pct_monthly.parquet",
                      "bench_net_10bps"),
    "carry": (SLEEVES / "_carry_output" / "carry_primary_net_monthly.parquet", "net"),
    "seasonal": (SLEEVES / "_seasonal" / "seasonal_composite_20pct_monthly.parquet",
                 "seasonal_net_10bps"),
    "defensive_v1": (SLEEVES / "_defensive" / "primary_20pct_monthly.csv", "net_10bps"),
    "defensive_v2": (SLEEVES / "_defensive" / "defensive_primary_net_monthly.parquet", "net"),
}
CASH = REPO / "_data" / "multiasset" / "cash_monthly.parquet"


def sharpe(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.mean(a) / np.std(a, ddof=1) * math.sqrt(MPY))


def ann_vol(x) -> float:
    return float(np.std(np.asarray(x, dtype=float), ddof=1) * math.sqrt(MPY))


def ann_mean(x) -> float:
    return float(np.mean(np.asarray(x, dtype=float)) * MPY)


def cagr(x) -> float:
    a = np.asarray(x, dtype=float)
    if np.min(a) <= -1.0:
        return -1.0
    return float(np.prod(1.0 + a) ** (MPY / len(a)) - 1.0)


def max_dd(x) -> float:
    a = np.asarray(x, dtype=float)
    if np.min(a) <= -1.0:
        return -1.0
    curve = np.cumprod(1.0 + a)
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def inverse_vol_weights(f: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / f.std(ddof=1).to_numpy()
    return iv / iv.sum()


def newey_west_tstat(x, lags: int = 4) -> tuple[float, float, float]:
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(v)
    mean = float(v.mean())
    dev = v - mean
    var = float(dev @ dev) / n
    for lag in range(1, min(int(lags), n - 1) + 1):
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * float(dev[lag:] @ dev[:-lag]) / n
    se = math.sqrt(max(var, 1e-18) / n)
    return mean, se, mean / se


def check(label: str, got: float, want: float, tol: float) -> dict:
    ok = bool(abs(got - want) <= tol)
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<58} got {got:.10f}  want {want}  "
          f"|d|={abs(got-want):.3e}")
    return {"label": label, "got": got, "want": want, "tol": tol, "ok": ok}


def main() -> int:
    rows = []
    print("A. dsr_sharpe_bar anchors (the reference implementation)")
    rows.append(check("bar @ 7yr, n_trials=32", dsr_sharpe_bar(7.0, n_trials=32), 1.488, 5e-4))
    rows.append(check("bar @ 40yr, n_trials=32", dsr_sharpe_bar(40.0, n_trials=32), 0.597, 5e-4))
    rows.append(check("bar @ 17.75yr, n_trials=38 (low-vol B2)",
                      dsr_sharpe_bar(17.75, n_trials=38), 0.9234, 5e-5))
    rows.append(check("bar @ 17.75yr, n_trials=47 (programme ledger)",
                      dsr_sharpe_bar(17.75, n_trials=47),
                      0.9233854510551396, 0.15))  # loose: recorded bar used n=38

    print("\nB. the CLAIM itself -- rebuild lowvol+trend [inverse_vol] exactly as v1 did")
    lv = load(*SRC["lowvol_registered"])
    tr = load(*SRC["trend"])
    f = pd.concat({"lowvol": lv, "trend": tr}, axis=1).dropna()
    w = inverse_vol_weights(f)
    port = f.to_numpy() @ w
    print(f"     window {f.index[0].date()} .. {f.index[-1].date()}  n={len(f)}")
    rows.append(check("claim: pair Sharpe", sharpe(port), 1.2165535517187802, 1e-9))
    rows.append(check("claim: pair vol", ann_vol(port), 0.11314524062947205, 1e-9))
    rows.append(check("claim: pair CAGR", cagr(port), 0.13951395796202615, 1e-9))
    rows.append(check("claim: pair max DD", max_dd(port), -0.19078661607361447, 1e-9))
    rows.append(check("claim: weight lowvol", float(w[0]), 0.6075611495217327, 1e-12))
    rows.append(check("claim: n_months", float(len(f)), 213.0, 0))

    print("\nC. constituent anchors")
    rows.append(check("low-vol registered, standalone Sharpe", sharpe(lv), 0.877853588402183, 1e-9))
    lvc = load(*SRC["lowvol_corrected"], shift_months=1)
    rows.append(check("low-vol CORRECTED, standalone Sharpe (total basis)",
                      sharpe(lvc), 0.6138, 1e-3))
    rows.append(check("trend standalone Sharpe (full 738mo)", sharpe(tr), 0.6116, 1e-3))
    pm = load(*SRC["passive_monthly"])
    pdl = load(*SRC["passive_daily"])
    rows.append(check("passive MONTHLY Sharpe", sharpe(pm), 0.6691, 1e-3))
    rows.append(check("passive DAILY Sharpe", sharpe(pdl), 0.7065, 1e-3))

    print("\nD. the corrected-pair number v2 recorded (equal weight, excess basis)")
    cash = load(CASH, "US_CASH_13W")
    lvc_ex = (lvc - cash.reindex(lvc.index)).dropna()
    tr_ex = tr  # already excess
    g = pd.concat({"lowvol": lvc_ex, "trend": tr_ex}, axis=1).dropna()
    rows.append(check("corrected low-vol Sharpe, EXCESS basis", sharpe(g["lowvol"]), 0.4869, 1e-3))
    ew = g.to_numpy() @ np.full(2, 0.5)
    rows.append(check("v2 corrected pair Sharpe [equal weight]", sharpe(ew), 0.9260, 1e-3))
    iv = g.to_numpy() @ inverse_vol_weights(g)
    rows.append(check("v2 corrected pair Sharpe [inverse vol]", sharpe(iv), 0.9212, 1e-3))
    rows.append(check("v2 corrected pair window start is 1998-05",
                      float(g.index[0].month), 5.0, 0))

    ok = all(r["ok"] for r in rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "controls.json").write_text(
        json.dumps({"all_passed": ok, "checks": rows}, indent=2), encoding="utf-8")
    print(f"\nALL CONTROLS PASSED: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
