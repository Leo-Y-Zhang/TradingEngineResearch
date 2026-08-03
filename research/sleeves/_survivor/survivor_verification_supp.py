"""SUPPLEMENT to `survivor_verification` — the four questions the first pass raised.

  S1  LEAVE-ONE-DECADE-OUT. The full-sample vol-matched active is +2.11%/yr at t +2.35,
      but only ONE decade's own t-stat clears 2. Drop each decade in turn and see which
      one the result is standing on.
  S2  THE CONVENTION CHARGE, BRACKETED. `A9` charged (US cash - the measured 1.785%
      SPY-vs-SPX dividend yield) on all seven equity price indices. That single uniform
      charge is the harshest defensible version. Bracket it: kinder dividend yields, a
      Japan exemption (JGB yields were ~0 for thirty years, so N225 is the one instrument
      whose true charge may be negative), and DAX charged correctly as the TOTAL-RETURN
      index it is.
  S3  P&L CONCENTRATION. A single name-month was once 13% of total P&L in this
      programme. Check the book.
  S4  MOMENTS. `dsr_sharpe_bar` assumes Gaussian returns; skew and fat tails RAISE the
      real bar. Measure them.

    .venv/Scripts/python.exe -m research.sleeves._survivor.survivor_verification_supp
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.sleeves.multiasset_trend import (
    BLOCKS,
    TrendConfig,
    load_excess_panel,
    run_trend,
)
from research.sleeves._survivor.survivor_verification import (
    BLOCK,
    DATA,
    MPY,
    RNG_SEED,
    TREND_CSV,
    VOL_TARGET,
    ann_mean,
    book_from,
    boot_sharpe,
    cagr,
    circular_blocks,
    dd_ladder,
    levered_total,
    sharpe,
    vm,
)

OUT_DIR = Path(__file__).resolve().parent
EQ = list(BLOCKS["equity"])


def main() -> int:
    saved = pd.read_csv(TREND_CSV, index_col=0, parse_dates=True)
    trend, passive = saved["net_10bps"].dropna(), saved["bench_net_10bps"].dropna()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"]
    book, _ = book_from(trend, passive)
    f = pd.concat({"trend": trend, "passive": passive}, axis=1).dropna()
    out: dict = {}

    # ── S1. LEAVE-ONE-DECADE-OUT ─────────────────────────────────────────────
    loo: dict = {}
    for d in sorted({y // 10 * 10 for y in book.index.year}):
        keep = (book.index.year // 10 * 10) != d
        b, p = book[keep], f["passive"][keep]
        loo[str(int(d))] = {"n": int(keep.sum()), "sharpe_book": sharpe(b),
                            "sharpe_passive": sharpe(p), "vm": vm(b, p)}
    out["S1_leave_one_decade_out"] = loo

    # ── S2. THE CONVENTION CHARGE, BRACKETED ─────────────────────────────────
    x, interior = load_excess_panel()
    c_m = cash.reindex(x.index)

    def charged(qs: dict[str, float]) -> dict:
        xa = x.copy()
        for k, q in qs.items():
            xa[k] = xa[k] - (c_m - q / MPY)
        r = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=xa, interior=interior)
        b, w = book_from(r.net["10bps"], r.bench_net["10bps"])
        return {"sharpe_book": sharpe(b), "sharpe_trend": sharpe(r.net["10bps"]),
                "sharpe_passive": sharpe(r.bench_net["10bps"]),
                "weight_passive": float(w[1]), "vm": vm(b, r.bench_net["10bps"]),
                "mean_charge_annual_uniform": float(
                    ann_mean(c_m) - float(np.mean(list(qs.values()))))}

    scen: dict = {}
    scen["none"] = {"sharpe_book": sharpe(book), "sharpe_trend": sharpe(trend),
                    "sharpe_passive": sharpe(passive), "weight_passive": 0.7218,
                    "vm": vm(book, passive), "mean_charge_annual_uniform": 0.0}
    scen["q=1.785pct_measured_all_equity"] = charged({k: 0.01785 for k in EQ})
    scen["q=3.0pct_all_equity"] = charged({k: 0.030 for k in EQ})
    scen["q=4.0pct_all_equity"] = charged({k: 0.040 for k in EQ})
    # N225 exempt: Japanese policy rates sat near zero for three decades, so the
    # (r_local - q_local) charge on the Nikkei is plausibly ZERO or negative. This is the
    # kindest treatment of the one instrument that most rewards kindness.
    scen["q=3.0pct_N225_exempt"] = charged(
        {k: 0.030 for k in EQ if k != "N225"} | {"N225": float(ann_mean(c_m))})
    # DAX is the DAX Performance-Index: dividends ARE in it, so its dividend credit is 0
    # and its charge is the full risk-free rate.
    scen["q=3.0pct_N225_exempt_DAX_full_rf"] = charged(
        {k: 0.030 for k in EQ if k not in ("N225", "DAX")}
        | {"N225": float(ann_mean(c_m)), "DAX": 0.0})
    out["S2_convention_charge_bracket"] = scen

    # ── S3. P&L CONCENTRATION ────────────────────────────────────────────────
    tot = float(book.sum())
    srt = book.sort_values(ascending=False)
    out["S3_concentration"] = {
        "top_month_share": float(srt.iloc[0] / tot),
        "top_month": str(srt.index[0].date()),
        "top12_months_share": float(srt.iloc[:12].sum() / tot),
        "top5pct_months_share": float(srt.iloc[:int(0.05 * len(srt))].sum() / tot),
        "positive_month_frac": float((book > 0).mean()),
        "passive_top12_share": float(
            passive.sort_values(ascending=False).iloc[:12].sum() / passive.sum()),
        "trend_top12_share": float(
            trend.sort_values(ascending=False).iloc[:12].sum() / trend.sum()),
    }

    # ── S4. MOMENTS ──────────────────────────────────────────────────────────
    def moments(s: pd.Series) -> dict:
        a = s.dropna()
        return {"skew": float(a.skew()), "excess_kurtosis": float(a.kurt()),
                "min_month": float(a.min()), "max_month": float(a.max()),
                "ac1": float(a.autocorr(1)), "ac2": float(a.autocorr(2)),
                "ac12": float(a.autocorr(12)),
                "worst_12m_rolling": float(a.rolling(12).sum().min()),
                "jb_stat": float(len(a) / 6 * (a.skew() ** 2
                                               + a.kurt() ** 2 / 4))}
    out["S4_moments"] = {"book": moments(book), "trend": moments(trend),
                         "passive": moments(passive)}

    # ── S5. THE CORRECTED BOOK'S OWN LADDER ──────────────────────────────────
    # the most defensible convention charge: q = 3.0%/yr on the equity block, Japan
    # exempt (its local risk-free rate was ~0 for thirty years), DAX charged the full
    # bill because it is a TOTAL-RETURN index.
    xc = x.copy()
    for k in EQ:
        q = float(ann_mean(c_m)) if k == "N225" else (0.0 if k == "DAX" else 0.030)
        xc[k] = xc[k] - (c_m - q / MPY)
    rc_ = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=xc, interior=interior)
    bc, wc = book_from(rc_.net["10bps"], rc_.bench_net["10bps"])
    ex, csn = bc.to_numpy(), cash.reindex(bc.index).to_numpy()
    bidx = circular_blocks(len(ex), BLOCK, np.random.default_rng(RNG_SEED + 1), 2000)
    rows: dict = {}
    for cap in (0.35, 0.50):
        chosen = 0.0
        for lev in np.arange(0.05, 5.0001, 0.05):
            levered = levered_total(ex, csn, float(lev), 0.0150)
            paths = levered[bidx]
            curve = np.cumprod(1.0 + paths, axis=1)
            dd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
            if abs(float(np.percentile(dd, 5))) <= cap:
                chosen = float(lev)
            else:
                break
        chosen_path = levered_total(ex, csn, chosen, 0.0150)
        rows[f"dd{int(cap * 100)}_bootstrap_p95"] = {
            "leverage": round(chosen, 2), "cagr": cagr(chosen_path),
            "cagr_after_iteration11_factor": cagr(chosen_path) * 0.877}
    bs = boot_sharpe(bc)
    rows["sharpe"] = sharpe(bc)
    rows["boot_ci95"] = [float(np.nanpercentile(bs, 2.5)),
                         float(np.nanpercentile(bs, 97.5))]
    rows["p_below_0894"] = float(np.nanmean(bs < 0.894))
    rows["p_below_075"] = float(np.nanmean(bs < 0.75))
    rows["observed_ladder"] = dd_ladder(ex, csn, 0.0150)
    rows["weight_passive"] = float(wc[1])
    out["S5_corrected_book_ladder"] = rows

    (OUT_DIR / "survivor_verification_supp.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
