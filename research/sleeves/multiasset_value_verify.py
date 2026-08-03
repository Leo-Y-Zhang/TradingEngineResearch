"""Adversarial verification of the cross-asset VALUE sleeve result.

The sleeve came back DEAD. A dead result is as capable of being a bug as a live one -- a
sign error in the value score would produce exactly this -- so every load-bearing claim is
re-derived here by a path that does not reuse the sleeve's own arithmetic.

What is checked
===============
V1  The value score is economically sane on cases whose answer is known in advance
    (NASDAQ after 2000, SPX after 1929) -- a sign error would show here first.
V2  Per-block Spearman IC of value score vs NEXT month's return, computed directly from
    the panel with no book, no sizing and no cost model. If the equity block's IC is
    negative, the negative equity sub-book is the signal and not the machinery.
V3  A PERFECT-FORESIGHT positive control: replace the value score with next month's actual
    return and re-run the identical book. If that does not produce a large Sharpe, the
    pipeline cannot express an edge and every negative result from it is uninterpretable.
V4  Active return and the trend correlation recomputed from the written CSV, independently
    of the in-memory objects.
V5  Gross-leverage concentration by instrument -- the inverse-vol sizing hands the low-vol
    5y bond a very large notional and that needs to be stated as a number.
V6  The DSR bar reproduces the programme's two recorded anchors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.multiasset_trend import annual_sharpe, load_excess_panel, newey_west_tstat
from research.sleeves.multiasset_value import (
    BLOCKS,
    COST_BRACKETS,
    VALUE_UNIVERSE,
    ValueConfig,
    load_yield_spreads,
    run_value,
    value_scores,
)

OUT = Path("research/sleeves/_value")
TREND = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")


def main() -> dict:
    x, _ = load_excess_panel(universe=VALUE_UNIVERSE)
    spreads = load_yield_spreads(x.index)
    v = value_scores(x, spreads)
    ref = run_value(ValueConfig(), vol_target=0.20, x=x, spreads=spreads)
    report: dict = {}

    # ── V1: does the score say the obvious things? ────────────────────────────
    def peak(key: str, lo: str, hi: str) -> dict:
        w = v[key].loc[lo:hi].dropna()
        return {"argmax_date": str(w.idxmax().date()), "max": float(w.max()),
                "argmin_date": str(w.idxmin().date()), "min": float(w.min())}

    report["v1_signal_sanity"] = {
        "NASDAQ_1998_2006": peak("NASDAQ", "1998-01-01", "2006-12-31"),
        "SPX_1928_1940": peak("SPX", "1928-01-01", "1940-12-31"),
        "GOLD_F_2006_2020": peak("GOLD_F", "2006-01-01", "2020-12-31"),
        "note": (
            "A 5-year reversal score must PEAK (cheapest) right after a crash and TROUGH "
            "(dearest) at a top. NASDAQ's score must peak ~2002-2005 and trough ~2000."
        ),
    }

    # ── V2: raw predictive content, no book at all ────────────────────────────
    fwd = x.shift(-1)
    ic_rows: dict[str, dict[str, Any] | None] = {}
    for block, keys in BLOCKS.items():
        cols = [k for k in keys if k in v.columns]
        ics = []
        for t in v.index:
            s = v.loc[t, cols].dropna()
            f = fwd.loc[t, cols].dropna()
            common = s.index.intersection(f.index)
            if len(common) >= 3:
                ic = stats.spearmanr(s[common], f[common]).statistic
                if np.isfinite(ic):
                    ics.append((t, float(ic)))
        if not ics:
            ic_rows[block] = None
            continue
        ser = pd.Series([c for _, c in ics], index=[t for t, _ in ics])
        ic_rows[block] = {
            "months": int(len(ser)),
            "mean_ic": float(ser.mean()),
            "ic_tstat_newey_west": newey_west_tstat(ser),
            "pct_positive": float(100.0 * (ser > 0).mean()),
        }
    report["v2_rank_ic_vs_next_month"] = ic_rows
    report["v2_note"] = (
        "Computed straight from the panel: no positions, no vol scaling, no costs. A "
        "negative mean IC means the signal itself is anti-predictive there."
    )

    # ── V3: perfect-foresight positive control on the REAL panel ──────────────
    # Feed the book next month's realised return as the value score by swapping the
    # scoring function's output in place: same sizing, same costs, same benchmark.
    import research.sleeves.multiasset_value as mv

    real_scores = mv.value_scores

    def cheat_scores(xx, ss, *, skip=0, uniform_rates=False):  # noqa: ANN001, ARG001
        base = real_scores(xx, ss)
        return xx.shift(-1).where(base.notna())

    try:
        mv.value_scores = cheat_scores
        cheat = run_value(ValueConfig(name="V3_PERFECT_FORESIGHT"), vol_target=0.20,
                          x=x, spreads=spreads)
    finally:
        mv.value_scores = real_scores

    report["v3_perfect_foresight"] = {
        "months": int(len(cheat.net["10bps"])),
        "gross_sharpe": annual_sharpe(cheat.gross),
        "net_sharpe_10bps": annual_sharpe(cheat.net["10bps"]),
        "verdict": (
            "PIPELINE CAN EXPRESS AN EDGE"
            if annual_sharpe(cheat.net["10bps"]) > 2.0
            else "PIPELINE SUSPECT -- perfect foresight did not produce a large Sharpe"
        ),
    }

    # ── V4: independent recomputation from the written receipts ───────────────
    csv = pd.read_csv(OUT / "primary_20pct_monthly.csv", parse_dates=["date"]).set_index("date")
    a = csv["net_10bps"] - csv["bench_net_10bps"]
    trend = pd.read_csv(TREND, parse_dates=["date"]).set_index("date")["net_10bps"]
    m, tr = csv["net_10bps"].align(trend, join="inner")
    report["v4_from_csv"] = {
        "months": int(len(csv)),
        "net_sharpe_10bps": annual_sharpe(csv["net_10bps"]),
        "bench_sharpe_10bps": annual_sharpe(csv["bench_net_10bps"]),
        "arith_active_annual": float(a.mean() * 12.0),
        "arith_active_tstat_newey_west": newey_west_tstat(a),
        "arith_active_tstat_iid": float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))),
        "corr_to_trend": float(m.corr(tr)),
        "corr_to_trend_spearman": float(stats.spearmanr(m, tr).statistic),
        "identity_geometric_equals_arith_minus_drag": float(
            (np.expm1(np.log1p(csv["net_10bps"]).mean() * 12)
             - np.expm1(np.log1p(csv["bench_net_10bps"]).mean() * 12))
            - (a.mean() * 12.0
               - (csv["net_10bps"].var(ddof=1) - csv["bench_net_10bps"].var(ddof=1)) / 2.0 * 12.0)
        ),
    }

    # ── V5: where the gross leverage actually sits ────────────────────────────
    live_w = ref.weights.loc[ref.gross.index]
    gross_by = live_w.abs().mean()
    report["v5_gross_leverage_share"] = {
        c: float(gross_by[c] / gross_by.sum()) for c in gross_by.sort_values(ascending=False).index
    }
    report["v5_mean_gross_leverage"] = float(live_w.abs().sum(axis=1).mean())

    # ── V6: the DSR bar reproduces the recorded anchors ───────────────────────
    report["v6_dsr_anchors"] = {
        "7yr_n32": round(dsr_sharpe_bar(7.0, n_trials=32), 4),
        "40yr_n32": round(dsr_sharpe_bar(40.0, n_trials=32), 4),
        "expected": {"7yr_n32": 1.488, "40yr_n32": 0.597},
    }

    # ── Cost bracket bookkeeping ──────────────────────────────────────────────
    report["v7_cost_arithmetic"] = {
        "turnover_per_year": float(ref.turnover.mean() * 12.0),
        "cost_drag_2bps_annual": float((ref.gross - ref.net["2bps"]).mean() * 12.0),
        "cost_drag_10bps_annual": float((ref.gross - ref.net["10bps"]).mean() * 12.0),
        "implied_ratio": float(COST_BRACKETS["10bps"] / COST_BRACKETS["2bps"]),
    }

    (OUT / "verification.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(main(), indent=2, default=str))
