"""VERIFY-2(a): correlate the trend and defensive books' vol-targeting scalers k(t).

    .venv/Scripts/python.exe -m scripts.verify2a_scaler_correlation

The defensive study asked whether the SHARED vol-targeting machinery alone creates
co-movement between the two books, and could only report the |return| correlation as a
PROXY because the trend book's scaler was never persisted (register item VERIFY-2(a),
`docs/project-control/RISK_AND_DEFECT_REGISTER.md`). The trend runner now writes k(t)
into `research/sleeves/_multiasset_trend/primary_20pct_monthly.csv` as the `scaler`
column; this DIAGNOSTIC reads it, recomputes the defensive PRIMARY book at the same 20%
volatility target in-process (a deterministic recomputation -- it selects nothing and
rewrites no published receipt), and reports the direct correlation the register asked
for. Both series are DECISION-dated: k at month-end t scales the weights held during
t+1, the same convention in both sleeves.

Writes `research/sleeves/_defensive/verify2a_scaler_correlation.json` (a NEW receipt;
nothing existing is modified) and prints the numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from research.sleeves.multiasset_defensive import DefensiveConfig, run_defensive
from research.sleeves.multiasset_trend import PRIMARY_UNIVERSE, load_excess_panel

TREND_CSV = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")
OUT_JSON = Path("research/sleeves/_defensive/verify2a_scaler_correlation.json")
VOL_TARGET = 0.20


def main() -> int:
    trend = pd.read_csv(TREND_CSV, parse_dates=["date"]).set_index("date")
    if "scaler" not in trend.columns:
        print(
            "BLOCKED: no `scaler` column in "
            f"{TREND_CSV} (columns: {list(trend.columns)}).\n"
            "Re-run the trend book first: "
            ".venv/Scripts/python.exe -m scripts.run_multiasset_trend"
        )
        return 2

    # The defensive PRIMARY book, exactly as `multiasset_defensive_run.main()` builds
    # it: the trend module's panel loader over the same registered universe, 20% target.
    x, interior = load_excess_panel(universe=PRIMARY_UNIVERSE)
    prim = run_defensive(DefensiveConfig(name="PRIMARY"), vol_target=VOL_TARGET,
                         x=x, interior=interior)

    kt_trend = trend["scaler"].dropna()
    kt_def = prim.scaler.dropna()
    a, b = kt_def.align(kt_trend, join="inner")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]

    if len(a) < 24:
        print(f"NOT COMPUTED: only {len(a)} overlapping decision months.")
        return 2

    # The proxy the study reported instead, recomputed on the SAME overlap so the two
    # numbers are like-for-like.
    ra, rb = prim.net["10bps"].align(trend["net_10bps"], join="inner")
    rmask = ra.notna() & rb.notna()

    report = {
        "question": (
            "Does the SHARED vol-targeting machinery alone create co-movement? "
            "Direct test: correlate the two books' k(t) scalers (VERIFY-2(a))."
        ),
        "convention": (
            "Both scalers are DECISION-dated: k at month-end t is computed from book "
            "returns realised at or before t and scales the weights held during t+1."
        ),
        "months": int(len(a)),
        "first": str(a.index.min().date()),
        "last": str(a.index.max().date()),
        "pearson": float(a.corr(b)),
        "spearman": float(stats.spearmanr(a, b).statistic),
        "abs_return_proxy_pearson_same_overlap": float(
            ra[rmask].abs().corr(rb[rmask].abs())),
        "abs_return_proxy_months": int(rmask.sum()),
        "sources": {
            "trend_scaler": str(TREND_CSV),
            "defensive_scaler": (
                "run_defensive(DefensiveConfig('PRIMARY'), vol_target=0.20) recomputed "
                "in-process (deterministic; not persisted by the defensive study)"
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VERIFY-2(a) -- k(t) scaler correlation, trend vs defensive")
    print(f"  overlap: {report['months']} months  {report['first']} -> {report['last']}")
    print(f"  pearson  {report['pearson']:+.4f}")
    print(f"  spearman {report['spearman']:+.4f}")
    print(f"  (|return| proxy on the same overlap: "
          f"{report['abs_return_proxy_pearson_same_overlap']:+.4f} "
          f"over {report['abs_return_proxy_months']} months)")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
