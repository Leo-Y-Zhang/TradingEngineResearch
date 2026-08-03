"""Measurement support for the 2026-07-28 benchmark-relative gate REVIEW.

This script measures, it does not decide. It answers four questions that the review
document is not allowed to answer by assertion:

  Q1. On the programme's REAL candidate/benchmark series, what is the standard error of
      the Sharpe DIFFERENCE -- analytically (Jobson-Korkie/Memmel, which uses the
      candidate-benchmark correlation) and by stationary bootstrap?
  Q2. Is a "candidate DSR > benchmark DSR" criterion equivalent to a Sharpe comparison?
      Specifically: does the multiple-testing deflation term cancel in the comparison?
  Q3. Can a candidate with a LOWER Sharpe score a HIGHER DSR (i.e. can the DSR-comparison
      form be gamed by return-shape rather than by return)?
  Q4. What is the false-pass rate of a bare "gap > 0" relative criterion under the null,
      and what Sharpe gap is actually detectable at the sample lengths this programme has?

Run:  .venv/Scripts/python.exe -m research._gate_review.sharpe_difference_power

Writes summary STATISTICS only to research/_gate_review/sharpe_difference_power.json.
No return series, no row-level data of any kind, is written or printed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.trial_ledger import cumulative_trials
from research.validation import deflated_sharpe_ratio

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sharpe_difference_power.json"
MPY = 12
RNG = np.random.default_rng(20260728)


# ── helpers ───────────────────────────────────────────────────────────────────

# Read the registered cumulative trial count rather than hardcoding it. The
# trial-ledger guard test exists precisely because a stale hardcode (38 vs 47)
# understated a deflation bar by 0.0209 earlier in this run.
_LEDGER_N = cumulative_trials()


def ann_sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(MPY))


def ann_vol(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) * math.sqrt(MPY))


def memmel_se(sa: float, sb: float, rho: float, T: int) -> float:
    """SE of the ANNUALISED Sharpe difference, Jobson-Korkie with Memmel's correction.

    Var(sr_a - sr_b) = (1/T) [ 2(1-rho) + 0.5(sr_a^2 + sr_b^2 - 2 rho^2 sr_a sr_b) ]
    with sr in PER-PERIOD units; annualise by multiplying the variance by MPY.
    """
    a, b = sa / math.sqrt(MPY), sb / math.sqrt(MPY)      # per-period Sharpes
    v = (2.0 * (1.0 - rho) + 0.5 * (a * a + b * b - 2.0 * rho * rho * a * b)) / T
    return float(math.sqrt(max(v, 0.0) * MPY))


def unpaired_se(sa: float, sb: float, T: int) -> float:
    """SE of the difference if the two series were treated as INDEPENDENT (rho=0)."""
    return memmel_se(sa, sb, 0.0, T)


def stationary_bootstrap_gap(a: np.ndarray, b: np.ndarray, B: int = 20000,
                             mean_block: float = 6.0) -> dict:
    """Paired stationary bootstrap (Politis-Romano) of the annualised Sharpe gap.

    Resamples the two series JOINTLY (same index draws) so the candidate-benchmark
    dependence is preserved, and so is serial dependence within each.
    """
    T = len(a)
    p = 1.0 / mean_block
    gaps = np.empty(B)
    for i in range(B):
        idx = np.empty(T, dtype=np.int64)
        j = RNG.integers(0, T)
        for t in range(T):
            idx[t] = j
            if RNG.random() < p:
                j = RNG.integers(0, T)
            else:
                j = (j + 1) % T
        ra, rb = a[idx], b[idx]
        sda, sdb = ra.std(ddof=1), rb.std(ddof=1)
        if sda <= 0 or sdb <= 0:
            gaps[i] = np.nan
            continue
        gaps[i] = (ra.mean() / sda - rb.mean() / sdb) * math.sqrt(MPY)
    g = gaps[np.isfinite(gaps)]
    obs = ann_sharpe(a) - ann_sharpe(b)
    return {
        "B": int(g.size),
        "mean_block_months": mean_block,
        "observed_gap": obs,
        "bootstrap_se": float(g.std(ddof=1)),
        "ci95_lo": float(np.percentile(g, 2.5)),
        "ci95_hi": float(np.percentile(g, 97.5)),
        "ci90_lo": float(np.percentile(g, 5.0)),
        # one-sided p for H0: true gap <= 0, via the centred bootstrap distribution
        "p_one_sided": float(np.mean((g - g.mean()) >= obs)),
    }


# ── Q1: real pairs ────────────────────────────────────────────────────────────

def real_pairs() -> dict:
    S = REPO / "research" / "sleeves"
    out: dict = {}

    tr = pd.read_csv(S / "_multiasset_trend" / "primary_20pct_monthly.csv",
                     parse_dates=["date"]).set_index("date")
    lv = pd.read_parquet(REPO / "research" / "sleeves" / "_portfolio"
                         / "lowvol_b2_corrected_monthly.parquet")
    # CASH CONVENTION CONTROL. `portfolio_correlation_v2.SOURCES` marks the low-vol
    # series "total", i.e. the programme subtracts the 13-week bill before comparing.
    # Trend and its benchmark are already "excess". Report the low-vol pair BOTH ways
    # so the convention cannot silently change the verdict.
    cash = pd.read_parquet(REPO / "_data" / "multiasset" / "cash_monthly.parquet")
    cser = cash["US_CASH_13W"] if "US_CASH_13W" in cash.columns else cash.iloc[:, 0]
    c = cser.reindex(lv.index)
    lv_ex = lv.copy()
    lv_ex["net_conservative"] = lv["net_conservative"] - c
    lv_ex["benchmark"] = lv["benchmark"] - c

    # the survivor: inverse-volatility blend of trend and the passive benchmark
    f = tr[["net_10bps", "bench_net_10bps"]].dropna()
    iv = 1.0 / f.std(ddof=1).to_numpy()
    w = iv / iv.sum()
    surv = (f.to_numpy() @ w)

    pairs = {
        "trend_vs_passive_monthly": (f["net_10bps"].to_numpy(),
                                     f["bench_net_10bps"].to_numpy(),
                                     "multi-asset trend net 10bps",
                                     "equal-weight passive, same 18 instruments, net 10bps"),
        "trend_plus_passive_vs_passive": (surv, f["bench_net_10bps"].to_numpy(),
                                          "trend+passive inverse-vol book",
                                          "equal-weight passive, same 18 instruments"),
        "lowvol_b2_cons_vs_own_universe_TOTAL": (lv["net_conservative"].to_numpy(),
                                                 lv["benchmark"].to_numpy(),
                                                 "low-vol B2 net conservative (total return)",
                                                 "own-universe equal weight (B2), total return"),
        "lowvol_b2_cons_vs_own_universe_EXCESS": (lv_ex["net_conservative"].to_numpy(),
                                                  lv_ex["benchmark"].to_numpy(),
                                                  "low-vol B2 net conservative (excess of 13w bill)",
                                                  "own-universe equal weight (B2), excess of bill"),
    }
    for k, (a, b, la, lb) in pairs.items():
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        T = len(a)
        sa, sb = ann_sharpe(a), ann_sharpe(b)
        rho = float(np.corrcoef(a, b)[0, 1])
        se_p = memmel_se(sa, sb, rho, T)
        se_u = unpaired_se(sa, sb, T)
        gap = sa - sb
        out[k] = {
            "candidate": la, "benchmark": lb,
            "T_months": T, "T_years": T / MPY,
            "sharpe_candidate": sa, "sharpe_benchmark": sb,
            "vol_candidate_ann": ann_vol(a), "vol_benchmark_ann": ann_vol(b),
            "rho": rho,
            "sharpe_gap": gap,
            "volmatched_active_pct_yr": gap * ann_vol(b) * 100.0,
            "se_paired_memmel": se_p,
            "se_unpaired_rho0": se_u,
            "se_ratio_unpaired_over_paired": se_u / se_p,
            "t_paired": gap / se_p,
            "p_one_sided_paired_normal": float(1.0 - norm.cdf(gap / se_p)),
            "bootstrap": stationary_bootstrap_gap(a, b),
            # what the gate's own DSR criterion says about each leg, at the ledger count
            "dsr_candidate_at_ledger": float(deflated_sharpe_ratio(a, n_trials=_LEDGER_N)),
            "dsr_benchmark_at_ledger": float(deflated_sharpe_ratio(b, n_trials=_LEDGER_N)),
        }
    return out


# ── Q2: does the deflation term cancel in a DSR comparison? ───────────────────

def dsr_cancellation() -> dict:
    """DSR = Phi((SR - sigma*k(n))/sigma) = Phi(SR/sigma - k(n)).

    If both legs are deflated at the same n, k(n) is COMMON and cancels:
        DSR_a > DSR_b  <=>  SR_a/sigma_a > SR_b/sigma_b
    i.e. the comparison carries NO multiple-testing protection at all. Verified here
    against the repo's own implementation, not asserted.
    """
    S = REPO / "research" / "sleeves"
    tr = pd.read_csv(S / "_multiasset_trend" / "primary_20pct_monthly.csv",
                     parse_dates=["date"]).set_index("date")
    f = tr[["net_10bps", "bench_net_10bps"]].dropna()
    a = f["net_10bps"].to_numpy()
    b = f["bench_net_10bps"].to_numpy()

    rows = []
    for n in (1, 2, 26, 34, 46, _LEDGER_N, 100, 281, 1000, 10000):
        da = float(deflated_sharpe_ratio(a, n_trials=n))
        db = float(deflated_sharpe_ratio(b, n_trials=n))
        rows.append({"n_trials": n, "dsr_candidate": da, "dsr_benchmark": db,
                     "candidate_wins": bool(da > db), "margin": da - db})
    return {
        "explanation": ("DSR_a > DSR_b is invariant to n_trials when both legs are "
                        "deflated at the same n -- the deflation term cancels."),
        "rows": rows,
        "sign_invariant_across_all_n": len({r["candidate_wins"] for r in rows}) == 1,
        "margin_range": [min(r["margin"] for r in rows), max(r["margin"] for r in rows)],
    }


# ── Q3: can a LOWER Sharpe produce a HIGHER DSR? ──────────────────────────────

def dsr_shape_reversal() -> dict:
    """Construct two return streams on the SAME T where the candidate has the LOWER
    Sharpe but the HIGHER DSR, using the repo's own deflated_sharpe_ratio.

    Mechanism: sigma_SR^2 = (1 - g3*SR + (g4-1)/4*SR^2)/(T-1). Positive skew SHRINKS
    sigma_SR and therefore RAISES DSR; negative skew does the reverse. A positively
    skewed weak candidate can out-DSR a negatively skewed strong benchmark.
    """
    T = 738
    found = None
    # candidate: positively skewed (lottery-like). benchmark: negatively skewed.
    for target_gap in (0.02, 0.05, 0.08, 0.12, 0.20):
        # benchmark: negative skew via a mixture with rare large losses
        base = RNG.standard_normal(T)
        shock = (RNG.random(T) < 0.03) * -RNG.gamma(3.0, 1.4, T)
        b = base + shock
        b = (b - b.mean()) / b.std(ddof=1)
        # candidate: positive skew, mirror construction
        base2 = RNG.standard_normal(T)
        shock2 = (RNG.random(T) < 0.03) * RNG.gamma(3.0, 1.4, T)
        a = base2 + shock2
        a = (a - a.mean()) / a.std(ddof=1)
        sr_b_m = 0.20                                   # per-period (monthly) Sharpe
        sr_a_m = sr_b_m - target_gap / math.sqrt(MPY)   # strictly WORSE by target_gap ann.
        aa = a + sr_a_m
        bb = b + sr_b_m
        da = float(deflated_sharpe_ratio(aa, n_trials=_LEDGER_N))
        db = float(deflated_sharpe_ratio(bb, n_trials=_LEDGER_N))
        rec = {
            "T_months": T,
            "ann_sharpe_candidate": ann_sharpe(aa),
            "ann_sharpe_benchmark": ann_sharpe(bb),
            "ann_sharpe_gap": ann_sharpe(aa) - ann_sharpe(bb),
            "dsr_candidate_at_ledger": da, "dsr_benchmark_at_ledger": db,
            "candidate_worse_on_sharpe": bool(ann_sharpe(aa) < ann_sharpe(bb)),
            "candidate_better_on_dsr": bool(da > db),
        }
        if rec["candidate_worse_on_sharpe"] and rec["candidate_better_on_dsr"]:
            found = rec
            break
    return {"reversal_found": found is not None, "example": found}


# ── Q4: false-pass rate and minimum detectable gap ────────────────────────────

def null_false_pass(n_sims: int = 20000) -> dict:
    """Under H0 (true Sharpe gap = 0), how often does a bare 'measured gap > 0'
    criterion pass? And how often does a t>1.645 requirement pass?"""
    res = {}
    for T in (213, 738):
        for rho in (0.0, 0.5, 0.9):
            sr_m = 0.669 / math.sqrt(MPY)
            cov = np.array([[1.0, rho], [rho, 1.0]])
            L = np.linalg.cholesky(cov)
            z = RNG.standard_normal((n_sims, T, 2)) @ L.T + sr_m
            sa = z[:, :, 0].mean(1) / z[:, :, 0].std(1, ddof=1) * math.sqrt(MPY)
            sb = z[:, :, 1].mean(1) / z[:, :, 1].std(1, ddof=1) * math.sqrt(MPY)
            gap = sa - sb
            se = np.array([memmel_se(x, y, rho, T) for x, y in zip(sa[:2000], sb[:2000])])
            t = gap[:2000] / se
            res[f"T{T}_rho{rho}"] = {
                "T_months": T, "rho": rho,
                "false_pass_bare_gap_gt_0": float(np.mean(gap > 0)),
                "false_pass_t_gt_1_645": float(np.mean(t > 1.645)),
                "empirical_sd_of_gap": float(gap.std(ddof=1)),
                "memmel_se_at_truth": memmel_se(0.669, 0.669, rho, T),
            }
    return res


def minimum_detectable_gap() -> dict:
    """Annualised Sharpe gap detectable at alpha=0.05 one-sided with 80% power,
    i.e. gap = (z_0.95 + z_0.80) * SE, for the programme's actual sample lengths."""
    z = norm.ppf(0.95) + norm.ppf(0.80)
    sb = 0.669
    rows = []
    for T in (213, 269, 533, 629, 738):
        for rho in (0.0, 0.005, 0.25, 0.5, 0.709, 0.802, 0.9):
            se = memmel_se(sb, sb, rho, T)
            rows.append({"T_months": T, "T_years": round(T / MPY, 2), "rho": rho,
                         "se_gap": se, "mdg_80pct_power": z * se,
                         "significance_only_gap_1_645se": norm.ppf(0.95) * se})
    return {"z_sum_alpha05_power80": z, "benchmark_sharpe_assumed": sb, "rows": rows}


def main() -> int:
    out = {
        "provenance": "research/_gate_review/sharpe_difference_power.py, seed 20260728",
        "Q1_real_pairs": real_pairs(),
        "Q2_dsr_deflation_cancels": dsr_cancellation(),
        "Q3_dsr_shape_reversal": dsr_shape_reversal(),
        "Q4_null_false_pass": null_false_pass(),
        "Q4_minimum_detectable_gap": minimum_detectable_gap(),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
