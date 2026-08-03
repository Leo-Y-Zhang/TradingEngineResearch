"""ADVERSARIAL DEFLATION OF THE lowvol+trend PAIR.

Six attacks, in the order the brief sets them:
  1. deflate for the 234-combination search and the 47-trial programme ledger
  2. bear-market dependence (dot-com out, GFC out, both out)
  3. out-of-sample split (fit the weighting on half 1, apply unchanged to half 2)
  4. leverage against a SURVIVABLE drawdown cap, financing charged, not half-Kelly
  5. constituent honesty
  6. benchmark at matched volatility, through the same deflated gate

Every number is measured here. Run `controls.py` first -- it aborts if any recorded
anchor fails to reproduce.

    .venv/Scripts/python.exe -m research.sleeves._pair_deflation.pair_deflation
"""

from __future__ import annotations

from typing import Any

import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy import optimize

from research.multiasset.panel import dsr_sharpe_bar
from research.validation import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

from research.sleeves._pair_deflation.controls import (
    CASH, MPY, OUT, SRC, ann_mean, ann_vol, cagr, inverse_vol_weights, load,
    max_dd, newey_west_tstat, sharpe,
)

RNG = np.random.default_rng(20260728)
BOOT_BLOCK = 12
BOOT_N = 2000

# The trial counts the brief demands.
N_COMBOS = 234                 # combinations searched by the study that produced the pair
N_LEDGER = 47                  # cumulative programme trials after iteration 14
N_BOTH = N_COMBOS + N_LEDGER   # 281

FINANCING = {"primary_bill_plus_150bp": 0.0150, "retail_bill_plus_300bp": 0.0300}
DD_CAPS = (0.50, 0.35)

# Bear-market definitions taken VERBATIM from lowvol_retest_verification.md section 5,
# so the pair is measured on the same windows the constituent was.
DOTCOM = ("2000-01-01", "2002-12-31")
GFC = ("2008-01-01", "2011-12-31")


# ── weighting schemes ─────────────────────────────────────────────────────────
def equal_weights(f):
    return np.full(f.shape[1], 1.0 / f.shape[1])


def inverse_variance_weights(f):
    iv = 1.0 / f.var(ddof=1).to_numpy()
    return iv / iv.sum()


def erc_weights(f):
    cov = f.cov(ddof=1).to_numpy()
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    def obj(y):
        w = np.exp(y)
        w = w / w.sum()
        rc = w * (cov @ w)
        return float(np.sum((rc - rc.mean()) ** 2) * 1e8)

    best, bestv = None, np.inf
    for seed in (np.zeros(n), np.log(inverse_vol_weights(f))):
        res = optimize.minimize(obj, seed, method="Nelder-Mead",
                                options={"maxiter": 40000, "xatol": 1e-12, "fatol": 1e-16})
        if res.fun < bestv:
            best, bestv = res.x, res.fun
    w = np.exp(best)
    return w / w.sum()


SCHEMES = {"equal_weight": equal_weights, "inverse_vol": inverse_vol_weights,
           "inverse_variance": inverse_variance_weights, "erc": erc_weights}


# ── series construction ───────────────────────────────────────────────────────
def build_series() -> dict:
    """Both bases. `claim_*` is v1 exactly as published; `corr_*` is the corrected book
    on a common excess-over-cash basis (v2's repairs)."""
    cash = load(CASH, "US_CASH_13W")
    s = {
        "lowvol_reg": load(*SRC["lowvol_registered"]),
        "lowvol_corr": load(*SRC["lowvol_corrected"], shift_months=1),
        "trend": load(*SRC["trend"]),
        "seasonal": load(*SRC["seasonal"]),
        "defensive_v1": load(*SRC["defensive_v1"]),
        "defensive_v2": load(*SRC["defensive_v2"]),
        "carry": load(*SRC["carry"]),
        "passive_monthly": load(*SRC["passive_monthly"]),
        "passive_daily": load(*SRC["passive_daily"]),
        "cash": cash,
    }
    # excess-basis versions of the two low-vol books (the multi-asset ones already are)
    s["lowvol_reg_ex"] = (s["lowvol_reg"] - cash.reindex(s["lowvol_reg"].index)).dropna()
    s["lowvol_corr_ex"] = (s["lowvol_corr"] - cash.reindex(s["lowvol_corr"].index)).dropna()
    return s


def frame(series: dict, names: dict[str, str]) -> pd.DataFrame:
    return pd.concat({k: series[v] for k, v in names.items()}, axis=1).dropna()


def port_of(f: pd.DataFrame, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    w = SCHEMES[scheme](f)
    return f.to_numpy() @ w, w


# ── attack 1: deflation ───────────────────────────────────────────────────────
def deflation_row(label: str, r: np.ndarray, years: float | None = None) -> dict:
    years = len(r) / MPY if years is None else years
    s = sharpe(r)
    row: dict[str, Any] = {
        "label": label, "n_months": int(len(r)), "years": years, "sharpe": s,
        "skew": float(pd.Series(r).skew()),
        "kurtosis_excess": float(pd.Series(r).kurt()),
    }
    for n in (1, N_LEDGER, N_COMBOS, N_BOTH):
        row[f"dsr_n{n}"] = float(deflated_sharpe_ratio(r, n_trials=n))
        row[f"bar_n{n}"] = float(dsr_sharpe_bar(years, n_trials=n))
        row[f"clears_n{n}"] = bool(s >= row[f"bar_n{n}"])
    return row


# ── attack 4: leverage against a drawdown cap, financing charged ──────────────
def levered_total(x_excess, cash, lev: float, spread: float):
    return lev * x_excess - max(lev - 1.0, 0.0) * spread / MPY + cash


def _boot_idx(n: int, n_boot: int, block: int, rng) -> np.ndarray:
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    offs = np.arange(block)
    return (starts[:, :, None] + offs[None, None, :]).reshape(n_boot, -1)[:, :n] % n


def _dd_paths(mat: np.ndarray) -> np.ndarray:
    """Max drawdown of each row of a (B x T) return matrix. -1.0 if the path is ruined."""
    ruined = mat.min(axis=1) <= -1.0
    curve = np.cumprod(1.0 + np.where(ruined[:, None], 0.0, mat), axis=1)
    dd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
    return np.where(ruined, -1.0, dd)


def _dd_boot_p95(mat: np.ndarray) -> float:
    """The 95th percentile of drawdown MAGNITUDE across bootstrap paths -- i.e. the
    bad-but-not-extreme resample. `_dd_paths` returns negative numbers, so the worst
    tail is the 5th percentile of the signed series; taking the 95th of the signed
    series would return the BEST path and silently over-lever."""
    return float(np.percentile(np.abs(_dd_paths(mat)), 95))


def leverage_at_dd_cap(x_excess: np.ndarray, cash: np.ndarray, spread: float,
                       cap: float, *, idx: np.ndarray | None = None) -> dict:
    """Largest leverage whose max drawdown stays inside `cap`, and the compound return
    there. Solved two ways: against the OBSERVED path, and against the 95th percentile
    of a block-bootstrap resampling of the same months (the honest one -- solving
    against a single observed maximum systematically over-levers)."""
    def dd_obs(L):
        return abs(max_dd(levered_total(x_excess, cash, L, spread)))

    def dd_boot(L):
        return _dd_boot_p95(levered_total(x_excess, cash, L, spread)[idx])

    out: dict[str, dict[str, Any] | None] = {}
    for key, fn in (("observed_path", dd_obs), ("bootstrap_p95", dd_boot)):
        if key == "bootstrap_p95" and idx is None:
            continue
        lo, hi = 0.0, 40.0
        if fn(0.05) > cap:                       # cannot even hold 0.05x
            out[key] = None
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fn(mid) > cap:
                hi = mid
            else:
                lo = mid
        tot = levered_total(x_excess, cash, lo, spread)
        record: dict[str, Any] = {"leverage": lo, "cagr": cagr(tot),
                                  "max_dd_observed": max_dd(tot), "vol": ann_vol(tot)}
        if idx is not None:
            record["max_dd_boot_p95"] = -_dd_boot_p95(tot[idx])
        out[key] = record
    return out


def leverage_block(label: str, x_excess: np.ndarray, cash: np.ndarray) -> dict:
    idx = _boot_idx(len(x_excess), BOOT_N, BOOT_BLOCK,
                    np.random.default_rng(20260728))
    blk: dict[str, Any] = {"label": label, "n_months": int(len(x_excess)),
           "sharpe_excess": sharpe(x_excess), "vol_excess_1x": ann_vol(x_excess),
           "cash_mean_annual": ann_mean(cash)}
    for fname, spread in FINANCING.items():
        blk[fname] = {}
        for cap in DD_CAPS:
            blk[fname][f"dd_cap_{int(cap*100)}"] = leverage_at_dd_cap(
                x_excess, cash, spread, cap, idx=idx)
        # peak of the whole ladder, for reference
        grid = np.arange(0.05, 20.001, 0.05)
        rows = []
        for L in grid:
            tot = levered_total(x_excess, cash, float(L), spread)
            if tot.min() <= -1.0:
                break
            rows.append((float(L), cagr(tot), max_dd(tot)))
        arr = np.array(rows)
        j = int(np.argmax(arr[:, 1]))
        blk[fname]["unconstrained_peak"] = {"leverage": arr[j, 0], "cagr": arr[j, 1],
                                            "max_dd": arr[j, 2]}
    return blk


# ── attack 6: matched-volatility benchmark comparison ─────────────────────────
def vol_matched(active: np.ndarray, bench: np.ndarray) -> dict:
    """The repo's convention: scale the benchmark to the strategy's risk, then the
    difference in return units. Identity: sigma_s * (SR_s - SR_b)."""
    k = float(np.std(active, ddof=1) / np.std(bench, ddof=1))
    diff = active - k * bench
    mean, se, t = newey_west_tstat(diff, lags=4)
    return {"k_scale": k, "vol_matched_active_annual": mean * MPY,
            "nw4_tstat": t,
            "identity_check": ann_vol(active) / math.sqrt(MPY) * math.sqrt(MPY)
                              * (sharpe(active) - sharpe(bench)) / MPY * MPY,
            "sharpe_strategy": sharpe(active), "sharpe_benchmark": sharpe(bench)}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    s = build_series()
    res: dict = {"trial_counts": {"combos_searched": N_COMBOS,
                                  "programme_ledger": N_LEDGER, "both": N_BOTH}}

    # THE TWO BASES ------------------------------------------------------------
    claim_f = frame(s, {"lowvol": "lowvol_reg", "trend": "trend"})
    claim_port, claim_w = port_of(claim_f, "inverse_vol")

    corr_f = frame(s, {"lowvol": "lowvol_corr_ex", "trend": "trend"})
    corr_port_iv, corr_w_iv = port_of(corr_f, "inverse_vol")
    corr_port_ew, _ = port_of(corr_f, "equal_weight")

    res["bases"] = {
        "claim_v1": {"window": [str(claim_f.index[0].date()), str(claim_f.index[-1].date())],
                     "n": len(claim_f), "weights": dict(zip(claim_f.columns, claim_w.tolist())),
                     "sharpe": sharpe(claim_port), "vol": ann_vol(claim_port),
                     "cagr": cagr(claim_port), "max_dd": max_dd(claim_port),
                     "note": "registered low-vol book, NO one-month realignment, "
                             "total-return low-vol mixed with excess-return trend"},
        "corrected_v2": {"window": [str(corr_f.index[0].date()), str(corr_f.index[-1].date())],
                         "n": len(corr_f),
                         "weights": dict(zip(corr_f.columns, corr_w_iv.tolist())),
                         "sharpe_inverse_vol": sharpe(corr_port_iv),
                         "sharpe_equal_weight": sharpe(corr_port_ew),
                         "vol": ann_vol(corr_port_iv), "max_dd": max_dd(corr_port_iv),
                         "note": "iteration-10-corrected low-vol, +1 month realignment, "
                                 "common excess-over-cash basis"},
    }

    # DEFECT LADDER — which correction moves 1.2166, and by how much ------------
    # Each rung adds ONE repair to the rung above, so the contributions are separable.
    lv_reg_shift = load(*SRC["lowvol_registered"], shift_months=1)
    lv_reg_shift_ex = (lv_reg_shift - s["cash"].reindex(lv_reg_shift.index)).dropna()
    ladder_defs = [
        ("0. as claimed (v1)", {"lowvol": s["lowvol_reg"], "trend": s["trend"]}),
        ("1. + one-month realignment", {"lowvol": lv_reg_shift, "trend": s["trend"]}),
        ("2. + common excess-over-cash basis",
         {"lowvol": lv_reg_shift_ex, "trend": s["trend"]}),
        ("3. + iteration-10 corrected low-vol book (= v2)",
         {"lowvol": s["lowvol_corr_ex"], "trend": s["trend"]}),
    ]
    ladder = []
    for label, cols in ladder_defs:
        g = pd.concat(cols, axis=1).dropna()
        p, w = port_of(g, "inverse_vol")
        ladder.append({"rung": label, "n": len(g), "sharpe": sharpe(p),
                       "vol": ann_vol(p), "cagr": cagr(p), "max_dd": max_dd(p),
                       "lowvol_standalone_sharpe": sharpe(g["lowvol"].to_numpy()),
                       "weights": dict(zip(g.columns, w.tolist())),
                       "bar_n234": float(dsr_sharpe_bar(len(g) / MPY, n_trials=N_COMBOS)),
                       "clears_bar_n234": bool(sharpe(p) >= dsr_sharpe_bar(
                           len(g) / MPY, n_trials=N_COMBOS))})
    res["defect_ladder"] = ladder

    # ATTACK 1 — DEFLATION ------------------------------------------------------
    passive_m = s["passive_monthly"]
    passive_win = passive_m.reindex(claim_f.index).dropna()
    res["attack1_deflation"] = [
        deflation_row("PAIR as claimed (v1, inverse-vol)", claim_port),
        deflation_row("PAIR corrected (v2, inverse-vol, excess)", corr_port_iv),
        deflation_row("PAIR corrected (v2, equal weight, excess)", corr_port_ew),
        deflation_row("low-vol B2 registered, alone", claim_f["lowvol"].to_numpy()),
        deflation_row("low-vol B2 corrected, alone (excess)", corr_f["lowvol"].to_numpy()),
        deflation_row("trend alone, on the pair's 213-month window",
                      claim_f["trend"].to_numpy()),
        deflation_row("trend alone, its own 738 months", s["trend"].to_numpy()),
        deflation_row("passive monthly EW, on the pair's window", passive_win.to_numpy()),
        deflation_row("passive monthly EW, its own 738 months", passive_m.to_numpy()),
        deflation_row("passive daily EW, its own 736 months", s["passive_daily"].to_numpy()),
    ]

    # PBO over the configurations actually searchable on this window -------------
    pool = {"lowvol": claim_f["lowvol"], "trend": claim_f["trend"],
            "seasonal": s["seasonal"], "defensive": s["defensive_v1"],
            "passive": s["passive_monthly"]}
    grid = pd.concat(pool, axis=1).dropna()
    cfg_names, cfg_cols = [], []
    for k in range(1, len(pool) + 1):
        for combo in itertools.combinations(pool.keys(), k):
            sub = grid[list(combo)]
            for scheme in SCHEMES:
                if len(combo) == 1 and scheme != "equal_weight":
                    continue
                w = SCHEMES[scheme](sub)
                cfg_names.append(f"{'+'.join(combo)} [{scheme}]")
                cfg_cols.append(sub.to_numpy() @ w)
    perf = np.column_stack(cfg_cols)
    pbo = probability_of_backtest_overfitting(perf, n_splits=16)
    best_j = int(np.argmax([sharpe(c) for c in cfg_cols]))
    res["attack1_pbo"] = {
        "n_configs": len(cfg_names), "n_months": int(perf.shape[0]),
        "window": [str(grid.index[0].date()), str(grid.index[-1].date())],
        "pbo_cscv_16_splits": float(pbo),
        "best_in_sample_config": cfg_names[best_j],
        "best_in_sample_sharpe": sharpe(cfg_cols[best_j]),
        "note": "CSCV over every subset x scheme available on this window; this is the "
                "selection process the pair came out of, measured directly",
    }

    # ATTACK 2 — BEAR-MARKET DEPENDENCE ----------------------------------------
    def mask_out(idx, *windows):
        m = pd.Series(True, index=idx)
        for a, b in windows:
            m &= ~((idx >= a) & (idx <= b))
        return m.to_numpy()

    def mask_in(idx, a, b):
        return np.asarray((idx >= a) & (idx <= b))

    bears = {}
    for basis, f, port in (("claim_v1", claim_f, claim_port),
                           ("corrected_v2", corr_f, corr_port_iv)):
        idx = f.index
        bench = passive_m.reindex(idx).to_numpy()
        rows: dict[str, dict[str, Any]] = {}
        variants = {
            "full_window": np.ones(len(idx), dtype=bool),
            "ex_dotcom": mask_out(idx, DOTCOM),
            "ex_gfc": mask_out(idx, GFC),
            "ex_both_bears": mask_out(idx, DOTCOM, GFC),
            "dotcom_only": mask_in(idx, *DOTCOM),
            "gfc_only": mask_in(idx, *GFC),
        }
        for name, m in variants.items():
            p = port[m]
            lv = f["lowvol"].to_numpy()[m]
            tr = f["trend"].to_numpy()[m]
            _, _, t = newey_west_tstat(p, lags=4)
            vm = vol_matched(p, bench[m])
            rows[name] = {"n_months": int(m.sum()), "pair_sharpe": sharpe(p),
                          "pair_mean_annual": ann_mean(p), "pair_vol": ann_vol(p),
                          "pair_nw4_t": t,
                          "lowvol_sharpe": sharpe(lv), "trend_sharpe": sharpe(tr),
                          "passive_sharpe": sharpe(bench[m]),
                          "vol_matched_active_vs_passive": vm["vol_matched_active_annual"],
                          "vol_matched_nw4_t": vm["nw4_tstat"],
                          "bar_at_this_length_n234": float(
                              dsr_sharpe_bar(max(len(p) / MPY, 0.5), n_trials=N_COMBOS)),
                          "dsr_n234": float(deflated_sharpe_ratio(p, n_trials=N_COMBOS))}
        bears[basis] = rows
    res["attack2_bear_dependence"] = bears

    # ATTACK 3 — OUT-OF-SAMPLE SPLIT -------------------------------------------
    oos = {}
    for basis, f in (("claim_v1", claim_f), ("corrected_v2", corr_f)):
        n = len(f)
        h = n // 2                              # 106 / 107
        f1, f2 = f.iloc[:h], f.iloc[h:]
        split_rows = {"split_point": str(f2.index[0].date()),
                "n_first": len(f1), "n_second": len(f2)}
        for scheme in ("inverse_vol", "equal_weight", "erc"):
            w1 = SCHEMES[scheme](f1)
            w2 = SCHEMES[scheme](f2)
            wall = SCHEMES[scheme](f)
            split_rows[scheme] = {
                "weights_fit_first_half": dict(zip(f.columns, w1.tolist())),
                "weights_fit_second_half": dict(zip(f.columns, w2.tolist())),
                "weights_fit_full": dict(zip(f.columns, wall.tolist())),
                "first_half_sharpe_insample": sharpe(f1.to_numpy() @ w1),
                "second_half_sharpe_FROZEN_weights": sharpe(f2.to_numpy() @ w1),
                "second_half_sharpe_refit_insample": sharpe(f2.to_numpy() @ w2),
                "full_window_sharpe": sharpe(f.to_numpy() @ wall),
                "second_half_cagr_frozen": cagr(f2.to_numpy() @ w1),
                "second_half_maxdd_frozen": max_dd(f2.to_numpy() @ w1),
                "second_half_bar_n234": float(
                    dsr_sharpe_bar(len(f2) / MPY, n_trials=N_COMBOS)),
                "second_half_dsr_n234": float(
                    deflated_sharpe_ratio(f2.to_numpy() @ w1, n_trials=N_COMBOS)),
            }
        split_rows["standalone_first_half"] = {c: sharpe(f1[c].to_numpy()) for c in f.columns}
        split_rows["standalone_second_half"] = {c: sharpe(f2[c].to_numpy()) for c in f.columns}
        oos[basis] = split_rows
    res["attack3_out_of_sample"] = oos

    # ATTACK 4 — LEVERAGE AT A SURVIVABLE DRAWDOWN ------------------------------
    # No cash leg is built for the raw claimed basis: attack 4 deliberately levers the
    # MADE-COHERENT basis instead, and the raw claim is characterised separately below
    # (claim_sharpe / half-Kelly / max_dd). Removed rather than left dangling.
    claim_ex_f = frame(s, {"lowvol": "lowvol_reg_ex", "trend": "trend"})
    claim_ex_port, claim_ex_w = port_of(claim_ex_f, "inverse_vol")
    cash_claim_ex = s["cash"].reindex(claim_ex_f.index).to_numpy()
    cash_corr = s["cash"].reindex(corr_f.index).to_numpy()
    cash_pass = s["cash"].reindex(passive_m.index).to_numpy()

    res["attack4_leverage"] = {
        "method": "excess book levered L times, borrowing charged on max(L-1,0) at "
                  "bill+spread; the bill leg is inside `cash`. Drawdown solved on the "
                  "recompounded levered path, and on the 95th percentile of a "
                  "12-month-block bootstrap of the same months.",
        "claim_basis_made_coherent": leverage_block(
            "PAIR as claimed, low-vol converted to excess (inverse-vol)",
            claim_ex_port, cash_claim_ex),
        "corrected_basis": leverage_block(
            "PAIR corrected (inverse-vol, excess)", corr_port_iv, cash_corr),
        "passive_monthly_own_history": leverage_block(
            "passive monthly EW, 738 months", passive_m.to_numpy(), cash_pass),
        "passive_on_pair_window": leverage_block(
            "passive monthly EW, the pair's 213 months only",
            passive_win.to_numpy(), s["cash"].reindex(passive_win.index).to_numpy()),
        "half_kelly_for_contrast": {
            "claim_sharpe": sharpe(claim_port),
            "half_kelly_growth_theoretical": 3.0 * sharpe(claim_port) ** 2 / 8.0,
            "half_kelly_required_vol": sharpe(claim_port) / 2.0,
            "half_kelly_leverage_on_1x_vol": sharpe(claim_port) / (2.0 * ann_vol(claim_port)),
            "max_dd_1x": max_dd(claim_port),
            "note": "reported only to show what the decision file quoted; forbidden "
                    "by standing rule 7 and not used for any conclusion",
        },
        "claim_ex_weights": dict(zip(claim_ex_f.columns, claim_ex_w.tolist())),
    }

    # ENGINE VALIDATION — this leverage engine against v2's recorded passive anchors,
    # and against iteration 11's independently-rebuilt figure. Nothing above is
    # trustworthy unless these agree.
    pv = res["attack4_leverage"]["passive_monthly_own_history"]["primary_bill_plus_150bp"]
    res["attack4_engine_validation"] = {
        "passive_sharpe": {"measured": sharpe(passive_m.to_numpy()),
                           "v2_recorded": 0.6691, "it11_recorded": 0.6678},
        "passive_dd50_observed": {
            "measured_cagr": pv["dd_cap_50"]["observed_path"]["cagr"],
            "measured_leverage": pv["dd_cap_50"]["observed_path"]["leverage"],
            "v2_recorded": "+14.02% @ 1.95x (0.05-step grid; this run bisects)"},
        "passive_dd50_bootstrap": {
            "measured_cagr": pv["dd_cap_50"]["bootstrap_p95"]["cagr"],
            "measured_leverage": pv["dd_cap_50"]["bootstrap_p95"]["leverage"],
            "v2_recorded": "+12.13% @ 1.40x (4000 resamples; this run uses 2000)"},
        "passive_unconstrained_peak": {
            "measured_cagr": pv["unconstrained_peak"]["cagr"],
            "v2_recorded": 0.1934, "it11_recorded_vol_targeted": 0.1583},
        "known_optimism_factor": {
            "value": 0.877,
            "why": "iteration 11 levered to a VOLATILITY TARGET and charged its own "
                   "rebalancing; this engine applies STATIC leverage to an "
                   "already-costed series. v2 measured the ratio at 0.877 on the same "
                   "passive book. Every compound return here is an upper bound and "
                   "should be multiplied by ~0.877 to compare with iteration 11.",
        },
    }

    # The headline table the brief asks for, with the known optimism factor applied.
    OPT = 0.877
    survivable = []
    for key in ("claim_basis_made_coherent", "corrected_basis",
                "passive_on_pair_window", "passive_monthly_own_history"):
        b = res["attack4_leverage"][key]
        for fname in FINANCING:
            for cap in DD_CAPS:
                for how in ("observed_path", "bootstrap_p95"):
                    v = b[fname][f"dd_cap_{int(cap*100)}"].get(how)
                    if v is None:
                        continue
                    survivable.append({
                        "book": key, "financing": fname, "dd_cap": cap, "solved_against": how,
                        "leverage": v["leverage"], "cagr_engine": v["cagr"],
                        "cagr_x0877": v["cagr"] * OPT})
    res["attack4_survivable_table"] = {"optimism_factor": OPT, "rows": survivable}

    # ATTACK 6 — BENCHMARK, MATCHED VOLATILITY ---------------------------------
    def bench_pair(port: np.ndarray, index) -> dict:
        g = pd.concat({"pair": pd.Series(port, index=index),
                       "passive": passive_m}, axis=1).dropna()
        out = vol_matched(g["pair"].to_numpy(), g["passive"].to_numpy())
        out["n"] = int(len(g))
        out["window"] = [str(g.index[0].date()), str(g.index[-1].date())]
        return out

    res["attack6_benchmark"] = {
        "benchmark_label": "passive, MONTHLY-rebalanced equal weight of the 18 panel "
                           "instruments (Sharpe 0.6691). The 0.7065 figure is the "
                           "DAILY-rebalanced variant and is NOT used here.",
        "claim_vs_passive": bench_pair(claim_port, claim_f.index),
        "corrected_vs_passive": bench_pair(corr_port_iv, corr_f.index),
        "passive_full_history_sharpe": sharpe(passive_m.to_numpy()),
        "passive_on_claim_window_sharpe": sharpe(passive_win.to_numpy()),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pair_deflation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # ── console summary ────────────────────────────────────────────────────────
    print("=" * 94)
    print("DEFECT LADDER — one repair per rung, inverse-vol throughout")
    print("=" * 94)
    print(f"{'rung':<48}{'n':>5}{'S':>9}{'lowvol S':>10}{'bar@234':>9}{'clears':>8}")
    for r in ladder:
        print(f"{r['rung']:<48}{r['n']:>5}{r['sharpe']:>9.4f}"
              f"{r['lowvol_standalone_sharpe']:>10.4f}{r['bar_n234']:>9.4f}"
              f"{str(r['clears_bar_n234']):>8}")

    print("\n" + "=" * 94)
    print("ATTACK 1 — DEFLATION.  bar = Sharpe needed for DSR>=0.95 at that trial count")
    print("=" * 94)
    print(f"{'book':<44}{'yrs':>6}{'S':>8}{'bar@47':>9}{'bar@234':>9}{'bar@281':>9}"
          f"{'DSR@234':>9}{'DSR@281':>9}")
    for r in res["attack1_deflation"]:
        print(f"{r['label']:<44}{r['years']:>6.1f}{r['sharpe']:>8.4f}"
              f"{r['bar_n47']:>9.4f}{r['bar_n234']:>9.4f}{r['bar_n281']:>9.4f}"
              f"{r['dsr_n234']:>9.4f}{r['dsr_n281']:>9.4f}")
    p = res["attack1_pbo"]
    print(f"\nPBO (CSCV, {p['n_configs']} configs x {p['n_months']} months) = "
          f"{p['pbo_cscv_16_splits']:.4f}   best IS = {p['best_in_sample_config']} "
          f"@ {p['best_in_sample_sharpe']:.4f}")

    print("\n" + "=" * 94)
    print("ATTACK 2 — BEAR-MARKET DEPENDENCE")
    print("=" * 94)
    for basis, rows in res["attack2_bear_dependence"].items():
        print(f"\n[{basis}]")
        print(f"{'window':<18}{'n':>5}{'pair S':>9}{'mean/yr':>10}{'NW4 t':>8}"
              f"{'lowvol S':>10}{'trend S':>9}{'pasv S':>9}{'vs pasv':>10}{'vm t':>7}"
              f"{'bar@234':>9}")
        for name, rec in rows.items():
            print(f"{name:<18}{rec['n_months']:>5}{rec['pair_sharpe']:>9.4f}"
                  f"{rec['pair_mean_annual']*100:>9.2f}%{rec['pair_nw4_t']:>8.2f}"
                  f"{rec['lowvol_sharpe']:>10.4f}{rec['trend_sharpe']:>9.4f}"
                  f"{rec['passive_sharpe']:>9.4f}"
                  f"{rec['vol_matched_active_vs_passive']*100:>9.2f}%"
                  f"{rec['vol_matched_nw4_t']:>7.2f}"
                  f"{rec['bar_at_this_length_n234']:>9.4f}")

    print("\n" + "=" * 94)
    print("ATTACK 3 — OUT-OF-SAMPLE SPLIT")
    print("=" * 94)
    for basis, rows in res["attack3_out_of_sample"].items():
        print(f"\n[{basis}] split at {rows['split_point']}  "
              f"({rows['n_first']} / {rows['n_second']} months)")
        for scheme in ("inverse_vol", "equal_weight", "erc"):
            r = rows[scheme]
            print(f"  {scheme:<16} H1 IS {r['first_half_sharpe_insample']:>7.4f} | "
                  f"H2 FROZEN {r['second_half_sharpe_FROZEN_weights']:>7.4f} | "
                  f"H2 refit {r['second_half_sharpe_refit_insample']:>7.4f} | "
                  f"full {r['full_window_sharpe']:>7.4f} | "
                  f"H2 bar@234 {r['second_half_bar_n234']:>6.4f}")
        print(f"  standalone H1 {rows['standalone_first_half']}")
        print(f"  standalone H2 {rows['standalone_second_half']}")

    print("\n" + "=" * 94)
    print("ATTACK 4 — HIGHEST COMPOUND RETURN AT A SURVIVABLE DRAWDOWN")
    print("=" * 94)
    for key in ("claim_basis_made_coherent", "corrected_basis",
                "passive_on_pair_window", "passive_monthly_own_history"):
        b = res["attack4_leverage"][key]
        print(f"\n[{b['label']}]  n={b['n_months']}  S(excess)={b['sharpe_excess']:.4f}")
        for fname in FINANCING:
            print(f"  {fname}")
            for cap in DD_CAPS:
                c = b[fname][f"dd_cap_{int(cap*100)}"]
                for how in ("observed_path", "bootstrap_p95"):
                    v = c.get(how)
                    if v is None:
                        print(f"    DD<={int(cap*100)}%  {how:<15} UNREACHABLE at any leverage")
                    else:
                        print(f"    DD<={int(cap*100)}%  {how:<15} "
                              f"{v['cagr']*100:>7.2f}%/yr @ {v['leverage']:>5.2f}x "
                              f"(observed DD {v['max_dd_observed']*100:>6.2f}%)")
            pk = b[fname]["unconstrained_peak"]
            print(f"    peak (unconstrained) {pk['cagr']*100:>7.2f}%/yr @ {pk['leverage']:.2f}x "
                  f"at DD {pk['max_dd']*100:.2f}%")

    print("\n" + "=" * 94)
    print("ATTACK 6 — BENCHMARK AT MATCHED VOLATILITY")
    print("=" * 94)
    b6 = res["attack6_benchmark"]
    for pair_key in ("claim_vs_passive", "corrected_vs_passive"):
        bv = b6[pair_key]
        print(f"  {pair_key:<24} n={bv['n']} k={bv['k_scale']:.4f}  vol-matched active "
              f"{bv['vol_matched_active_annual']*100:+.2f}%/yr  NW4 t {bv['nw4_tstat']:+.2f}  "
              f"(S_pair {bv['sharpe_strategy']:.4f} vs S_bench {bv['sharpe_benchmark']:.4f})")
    print(f"  passive on the pair's 213-month window: "
          f"Sharpe {b6['passive_on_claim_window_sharpe']:.4f}; "
          f"own 738 months: {b6['passive_full_history_sharpe']:.4f}")

    print(f"\nwrote {OUT / 'pair_deflation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
