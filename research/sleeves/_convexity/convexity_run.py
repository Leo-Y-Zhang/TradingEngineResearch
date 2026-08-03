"""Run the convexity re-analysis ONCE and write ``_convexity/result.json``.

Controls run BEFORE any result is read, in this order:
  C1  the banked tau-targeted ladder must reproduce 12.2955% / -47.2874% / 1.8769x
  C2  the survivor book must reproduce Sharpe 0.9033 from the banked CSV
  C3  at L = 1 with a zero spread the levered total must equal cash + excess exactly
  C4  the growth expansion must reproduce the empirical figure when carried to enough
      orders on a synthetic series where the answer is known
"""

from __future__ import annotations

from typing import Any

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.sleeves.multiasset_trend import (
    MONTHS,
    load_excess_panel,
)
from research.sleeves.riskparity import build_book, ladder
from research.trial_ledger import cumulative_trials

from research.sleeves._convexity.convexity import (
    C3,
    C4,
    DATA,
    FINANCING,
    N_BOOT,
    OUT_DIR,
    PRIMARY_FIN,
    RNG_SEED,
    TREND_CSV,
    VOL_TARGET,
    cagr,
    curve_peaks,
    decade_moments,
    growth_orders,
    levered_total,
    leverage_curve,
    matched_window_skew,
    max_dd,
    moment_report,
    nw_ols,
    payload_md5,
    sharpe,
    skewness,
    trend_variants,
    vol_target_overlay,
)

BANKED = {
    "riskparity_ew_tau15_compound": 0.12295487559393847,
    "riskparity_ew_tau15_maxdd": -0.4728738560103498,
    "riskparity_ew_tau15_meanlev": 1.8769448605144072,
    "riskparity_ew_peak_compound": 0.15828,      # iteration 11, tau = 39%
    "survivor_book_sharpe": 0.9033,
    # iteration 22, `_survivor/survivor_verification.json::A10_honest_number
    # .ladder_observed_path.primary_bill_plus_150bp` -- the CONSTANT-leverage ladder.
    "survivor_constL_dd50_compound": 0.25257083555080584,
    "survivor_constL_dd50_leverage": 3.10,
    "survivor_constL_peak_compound": 0.3365031299320007,
    "survivor_constL_peak_leverage": 5.75,
}


def inv_vol_weights(frame: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / frame.std(ddof=1).to_numpy()
    return iv / iv.sum()


def load_series() -> tuple[dict[str, pd.Series], pd.Series, pd.DataFrame, pd.DataFrame]:
    """Every banked monthly series on disk, all EXCESS returns net of 10bps."""
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"]
    x, interior = load_excess_panel()

    saved = pd.read_csv(TREND_CSV, index_col=0, parse_dates=True)
    trend = saved["net_10bps"].dropna()
    passive = saved["bench_net_10bps"].dropna()
    f = pd.concat({"trend": trend, "passive": passive}, axis=1).dropna()
    w = inv_vol_weights(f)
    book = pd.Series(f.to_numpy() @ w, index=f.index)

    s: dict[str, pd.Series] = {
        "trend": trend,
        "passive": passive,
        "book_trend_plus_passive": book,
        "trend_gross": saved["gross"].dropna(),
    }

    def _add(key: str, path: str, col: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        d = pd.read_parquet(p)
        if col in d.columns:
            s[key] = d[col].dropna()

    _add("carry", "research/sleeves/_carry_output/carry_primary_net_monthly.parquet", "net")
    _add("defensive", "research/sleeves/_defensive/defensive_primary_net_monthly.parquet",
         "net")
    _add("seasonal", "research/sleeves/_seasonal/seasonal_composite_20pct_monthly.parquet",
         "seasonal_net_10bps")
    _add("lowvol_b2_corrected",
         "research/sleeves/_portfolio/lowvol_b2_corrected_monthly.parquet",
         "net_conservative")
    vp = Path("research/sleeves/_value/primary_20pct_monthly.csv")
    if vp.exists():
        s["value"] = pd.read_csv(vp, index_col=0, parse_dates=True)["net_10bps"].dropna()

    return s, cash, x, interior


def controls(cash: pd.Series, x: pd.DataFrame, interior: pd.DataFrame,
             series: dict[str, pd.Series]) -> dict:
    out: dict = {}

    # C1 -- the banked tau-targeted ladder, rebuilt here.
    ew = build_book("ew", x, interior, "ew")
    lad = ladder(ew, cash, cost_label="10bps", financing_label=PRIMARY_FIN)
    r15 = lad["0.15"]
    out["C1_tau_ladder"] = {
        "compound": r15["compound_annual"], "max_dd": r15["max_drawdown"],
        "mean_leverage": r15["mean_leverage"], "months": r15["months"],
        "reproduces": bool(
            abs(r15["compound_annual"] - BANKED["riskparity_ew_tau15_compound"]) < 1e-12
            and abs(r15["max_drawdown"] - BANKED["riskparity_ew_tau15_maxdd"]) < 1e-12),
    }
    # The tau-peak, swept finely, so the 15.83% is measured here and not quoted.
    taus = np.round(np.arange(0.05, 1.0001, 0.01), 4)
    tau_rows = []
    for tau in taus:
        from research.sleeves.riskparity import levered as _lev
        L = _lev(ew, cash, tau=float(tau), cost=0.0010,
                 spread=FINANCING[PRIMARY_FIN])
        tot = L["total"].to_numpy(dtype=float)
        tau_rows.append({"tau": float(tau), "compound": cagr(tot), "max_dd": max_dd(tot),
                         "mean_leverage": float(L["k"].mean())})
    best = max(tau_rows, key=lambda r: r["compound"])
    out["C1b_tau_peak"] = {
        "tau": best["tau"], "compound": best["compound"], "max_dd": best["max_dd"],
        "mean_leverage": best["mean_leverage"],
        "reproduces_15.83": bool(abs(best["compound"] - BANKED["riskparity_ew_peak_compound"])
                                 < 0.002),
    }
    out["C1c_tau_rows"] = tau_rows

    # C2 -- the survivor book.
    bk = series["book_trend_plus_passive"]
    out["C2_survivor_sharpe"] = {
        "sharpe": sharpe(bk),
        "reproduces_0.9033": bool(abs(sharpe(bk) - BANKED["survivor_book_sharpe"]) < 5e-4),
    }

    # C5 -- the CONSTANT-leverage ladder must reproduce iteration 22's observed path.
    bkk = series["book_trend_plus_passive"]
    cur = leverage_curve(bkk, cash, FINANCING[PRIMARY_FIN])
    pk = curve_peaks(cur)
    out["C5_survivor_constL_ladder"] = {
        "dd50": pk["dd50"], "peak_empirical": pk["peak_empirical"],
        "reproduces_dd50": bool(
            abs(pk["dd50"]["compound"] - BANKED["survivor_constL_dd50_compound"]) < 1e-12
            and abs(pk["dd50"]["leverage"] - BANKED["survivor_constL_dd50_leverage"]) < 1e-9),
        "reproduces_peak": bool(
            abs(pk["peak_empirical"]["compound"] - BANKED["survivor_constL_peak_compound"])
            < 1e-12
            and abs(pk["peak_empirical"]["leverage"] - BANKED["survivor_constL_peak_leverage"])
            < 1e-9),
    }

    # C3 -- financing arithmetic at L = 1.
    p = series["passive"]
    c = cash.reindex(p.index)
    t1 = levered_total(p.to_numpy(float), c.to_numpy(float), 1.0,
                       FINANCING[PRIMARY_FIN])
    out["C3_L1_identity"] = {
        "max_abs_error": float(np.max(np.abs(t1 - (p.to_numpy(float) + c.to_numpy(float))))),
    }

    # C4 -- the expansion is arithmetically correct on a series whose growth is known.
    rng = np.random.default_rng(RNG_SEED)
    synth = rng.normal(0.005, 0.03, 100_000)
    g = growth_orders(synth)
    out["C4_expansion_on_gaussian"] = {
        "empirical": g["empirical"], "order2": g["order2"],
        "order3_log_expansion": g["order3_log_expansion"],
        "order4_log_expansion": g["order4_log_expansion"],
        "skew_should_be_0": g["skew"], "exkurt_should_be_0": g["exkurt"],
        "order2_within_20bp_of_empirical": bool(
            abs(g["order2"] - g["empirical"]) < 0.002),
    }
    return out


def fung_hsieh(series: dict[str, pd.Series], variants: dict[str, pd.Series]) -> dict:
    """Is the trend leg's payoff CONVEX in the passive leg? The straddle signature."""
    out: dict = {}
    passive = series["passive"]

    def one(name: str, y_s: pd.Series) -> dict:
        f = pd.concat({"y": y_s, "p": passive}, axis=1).dropna()
        y = f["y"].to_numpy(float)
        p = f["p"].to_numpy(float)
        # Model Q: quadratic. Centre the square so the linear beta stays interpretable.
        q = p ** 2 - (p ** 2).mean()
        mq = nw_ols(y, np.column_stack([np.ones(len(p)), p, q]))
        # Model S: piecewise -- the actual straddle. up-beta > 0 > down-beta.
        mp = nw_ols(y, np.column_stack(
            [np.ones(len(p)), np.maximum(p, 0.0), np.minimum(p, 0.0)]))
        beta_up, beta_dn = mp["beta"][1], mp["beta"][2]
        # t-stat on (up - down) via a reparameterisation: y ~ 1 + p + max(p,0)
        mr = nw_ols(y, np.column_stack([np.ones(len(p)), p, np.maximum(p, 0.0)]))
        return {
            "n": int(len(p)),
            "quadratic": {"alpha": mq["beta"][0] * MONTHS, "beta_linear": mq["beta"][1],
                          "beta_square": mq["beta"][2], "t_square": mq["t"][2],
                          "r2": mq["r2"]},
            "piecewise": {"beta_up": beta_up, "beta_down": beta_dn,
                          "t_up": mp["t"][1], "t_down": mp["t"][2],
                          "convexity_up_minus_down": beta_up - beta_dn,
                          "t_up_minus_down": mr["t"][2],
                          "r2": mp["r2"]},
        }

    for name in ("trend", "trend_gross", "book_trend_plus_passive"):
        out[name] = one(name, series[name])
    for name, s in variants.items():
        out[name] = one(name, s)
    out["passive_vol_targeted"] = one("passive_vol_targeted",
                                      vol_target_overlay(passive))
    # Placebo: passive on itself must show ZERO convexity by construction.
    out["_placebo_passive_on_itself"] = one("passive", passive)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series, cash, x, interior = load_series()
    variants = trend_variants(x, interior)
    passive_vt = vol_target_overlay(series["passive"])

    out: dict = {
        "_meta": {
            "question": "does positive skew move the leverage ceiling above 15.83%/yr",
            "n_trials_ledger": cumulative_trials(),
            "new_backtest_configurations_searched": 0,
            "bootstrap": {"seed": RNG_SEED, "reps": N_BOOT, "block_months": 12},
            "financing": FINANCING,
            "taylor_coefficients": {"c3": C3, "c4": C4},
            "vol_target_overlay": VOL_TARGET,
        }
    }

    out["controls"] = controls(cash, x, interior, series)

    # ── 1. THE MOMENTS ────────────────────────────────────────────────────────
    all_series = dict(series)
    all_series.update(variants)
    all_series["passive_vol_targeted"] = passive_vt
    out["moments"] = {k: moment_report(k, v) for k, v in all_series.items()}
    out["decade_moments"] = {
        k: decade_moments(all_series[k])
        for k in ("trend", "passive", "book_trend_plus_passive",
                  "trend_raw_L1_noL2", "passive_vol_targeted")
    }

    # ── 2. FUNG-HSIEH ─────────────────────────────────────────────────────────
    out["fung_hsieh"] = fung_hsieh(series, variants)

    # ── 3. THE LEVERAGE CURVES ────────────────────────────────────────────────
    curves: dict = {}
    for fin_label, spread in FINANCING.items():
        for leg in ("passive", "trend", "book_trend_plus_passive"):
            cur = leverage_curve(series[leg], cash, spread)
            curves[f"{leg}|{fin_label}"] = {
                "months": cur["months"], "start": cur["start"], "end": cur["end"],
                "peaks": curve_peaks(cur),
                "rows": [r for r in cur["rows"]
                         if abs(round(r["leverage"] / 0.25) * 0.25 - r["leverage"]) < 1e-9],
            }
    out["leverage_curves"] = curves

    # ── 3b. THE SAME BOOK, VOL-TARGETED vs CONSTANT LEVERAGE ──────────────────
    # Iteration 11's 15.83% / 12.30% came from a TAU-TARGETED ladder. Run the identical
    # book at CONSTANT leverage: any difference is the overlay, not convexity, and it
    # must not be allowed to contaminate the (c)-minus-(a) measurement.
    from research.sleeves.riskparity import levered as _lev
    ew = build_book("ew", x, interior, "ew")
    ones = pd.Series(1.0, index=ew.x.index)
    ew_1x = _lev(ew, cash, tau=0.0, cost=0.0010, spread=0.0,
                 k_override=ones)["net_excess"]
    # WINDOW MATCH: the tau ladder cannot start until 12 months of book volatility
    # exist, so it runs on 726 months where the constant-leverage book has 738. Compare
    # them on the tau ladder's own window or the extra year is scored as an overlay
    # effect. It is not one.
    tau_index = _lev(ew, cash, tau=0.15, cost=0.0010,
                     spread=FINANCING[PRIMARY_FIN])["total"].index
    ew_1x_matched = ew_1x.reindex(tau_index).dropna()
    overlay_cmp: dict = {}
    for fin_label in ("primary_bill_plus_150bp", "retail_bill_plus_300bp"):
        spread = FINANCING[fin_label]
        cur = leverage_curve(ew_1x_matched, cash, spread)
        pk = curve_peaks(cur)
        taus = np.round(np.arange(0.05, 1.0001, 0.01), 4)
        rows = []
        for tau in taus:
            L = _lev(ew, cash, tau=float(tau), cost=0.0010, spread=spread)
            tot = L["total"].reindex(tau_index).dropna().to_numpy(dtype=float)
            rows.append({"tau": float(tau), "compound": cagr(tot), "max_dd": max_dd(tot),
                         "mean_leverage": float(L["k"].mean())})
        tau_peak = max(rows, key=lambda r: r["compound"])
        tau_dd50_rows = [r for r in rows if abs(r["max_dd"]) <= 0.50]
        tau_dd50 = (max(tau_dd50_rows, key=lambda r: r["compound"])
                    if tau_dd50_rows else None)
        overlay_cmp[fin_label] = {
            "months_constL": cur["months"],
            "months_tau": int(len(tau_index)),
            "window": f"{cur['start']} -> {cur['end']}",
            "constant_leverage": {"peak": pk["peak_empirical"], "dd50": pk["dd50"],
                                  "dd35": pk["dd35"]},
            "vol_targeted": {"peak": tau_peak, "dd50": tau_dd50},
            "overlay_cost_at_peak_pp": (tau_peak["compound"]
                                        - pk["peak_empirical"]["compound"]) * 100.0,
            "overlay_cost_at_dd50_pp": ((tau_dd50["compound"] - pk["dd50"]["compound"])
                                        * 100.0) if tau_dd50 else None,
        }
    out["overlay_vs_constant_leverage"] = overlay_cmp

    # ── 4. WHERE THE SKEW COMES FROM ──────────────────────────────────────────
    out["skew_decomposition"] = {
        "layers": {k: {"skew": out["moments"][k]["skew"],
                       "z": out["moments"][k]["z_skew_iid"],
                       "sharpe": out["moments"][k]["sharpe"],
                       "vol_annual": out["moments"][k]["vol_annual"],
                       "months": out["moments"][k]["months"]}
                   for k in ("trend_raw_noL1_noL2", "trend_raw_L1_noL2",
                             "trend_noL1_L2", "trend_L1_L2", "trend",
                             "passive", "passive_vol_targeted")},
        "overlay_delta_on_trend": (out["moments"]["trend_L1_L2"]["skew"]
                                   - out["moments"]["trend_raw_L1_noL2"]["skew"]),
        "overlay_delta_on_passive": (out["moments"]["passive_vol_targeted"]["skew"]
                                     - out["moments"]["passive"]["skew"]),
        # Window-matched, so the overlay's 12-month warm-up cannot be mistaken for it.
        "matched_window": {
            "overlay_on_trend": matched_window_skew(variants["trend_raw_L1_noL2"],
                                                    variants["trend_L1_L2"]),
            "overlay_on_trend_unit": matched_window_skew(variants["trend_raw_noL1_noL2"],
                                                         variants["trend_noL1_L2"]),
            "overlay_on_passive": matched_window_skew(series["passive"], passive_vt),
            "L1_inverse_vol_sizing_no_overlay": matched_window_skew(
                variants["trend_raw_noL1_noL2"], variants["trend_raw_L1_noL2"]),
            "L1_inverse_vol_sizing_with_overlay": matched_window_skew(
                variants["trend_noL1_L2"], variants["trend_L1_L2"]),
            "signal_vs_passive": matched_window_skew(series["passive"],
                                                     variants["trend_raw_L1_noL2"]),
        },
    }

    # ── 5. THE HONEST BOUND ───────────────────────────────────────────────────
    bound: dict = {}
    for fin_label in ("primary_bill_plus_150bp", "retail_bill_plus_300bp"):
        for leg in ("passive", "book_trend_plus_passive", "trend"):
            k = f"{leg}|{fin_label}"
            pk = curves[k]["peaks"]
            bound[k] = {
                "peak_empirical": pk["peak_empirical"],
                "peak_order2": pk["peak_order2"],
                "gap_peak_compound_pp": pk["gap_peak_compound_pp"],
                "gap_peak_leverage": pk["gap_peak_leverage"],
                "dd50": pk["dd50"], "dd35": pk["dd35"],
            }
    # The term-by-term account at the empirical peak of the book, primary financing.
    for leg in ("passive", "book_trend_plus_passive", "trend"):
        cur = curves[f"{leg}|{PRIMARY_FIN}"]
        Lstar = cur["peaks"]["peak_empirical"]["leverage"]
        e, c = series[leg].align(cash.reindex(series[leg].index), join="inner")
        tot = levered_total(e.to_numpy(float), c.to_numpy(float), Lstar,
                            FINANCING[PRIMARY_FIN])
        bound[f"terms_at_empirical_peak|{leg}"] = {"leverage": Lstar,
                                                   **growth_orders(tot)}
    out["honest_bound"] = bound

    # Per-decade compound of the book at its DD<=50% leverage.
    dec: dict = {}
    for leg in ("passive", "book_trend_plus_passive"):
        pk = curves[f"{leg}|{PRIMARY_FIN}"]["peaks"]["dd50"]
        if pk is None:
            continue
        Lstar = pk["leverage"]
        e, c = series[leg].align(cash.reindex(series[leg].index), join="inner")
        tot = pd.Series(levered_total(e.to_numpy(float), c.to_numpy(float), Lstar,
                                      FINANCING[PRIMARY_FIN]), index=e.index)
        decade_rows: dict[str, dict[str, Any]] = {}
        for d, grp in tot.groupby((tot.index.year // 10) * 10):
            v = grp.to_numpy(float)
            decade_rows[f"{int(d)}s"] = {"months": int(v.size), "compound": cagr(v),
                                         "max_dd": max_dd(v), "skew": skewness(v)}
        dec[leg] = {"leverage": Lstar, "decades": decade_rows}
    out["per_decade_at_dd50"] = dec

    out["_meta"]["payload_md5"] = payload_md5(
        {k: v for k, v in out.items() if k != "_meta"})
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=1, default=str),
                                         encoding="utf-8")
    print(json.dumps(out["controls"]["C1_tau_ladder"], indent=1))
    print(json.dumps(out["controls"]["C1b_tau_peak"], indent=1))
    print(json.dumps(out["controls"]["C2_survivor_sharpe"], indent=1))
    print("C3 max abs error", out["controls"]["C3_L1_identity"]["max_abs_error"])
    print(json.dumps(out["controls"]["C4_expansion_on_gaussian"], indent=1))
    print("payload md5", out["_meta"]["payload_md5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
