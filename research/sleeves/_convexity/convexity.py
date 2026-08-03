"""THE CONVEXITY RE-ANALYSIS — does positive skew move the leverage ceiling?

THE CLAIM UNDER TEST
====================
Every ceiling number in this programme (iteration 11: peak compound **15.83%/yr**,
survivable **12.30%/yr** at DD<=50%) was read off a curve whose SHAPE comes from the
SECOND-ORDER growth expansion

    g ~= mu*L - sigma^2 * L^2 / 2

The quadratic term is what makes the curve concave, and it silently truncates the third
moment to zero. If the trend leg has the option-like positive skew Fung & Hsieh (2001)
document for trend followers, the cubic term ``+gamma*sigma^3*L^3/6`` is positive and
pushes the optimum both HIGHER and LATER — and the ceiling would be an artefact.

WHAT THIS MODULE DOES ABOUT IT
==============================
It does not argue. It measures three curves at every leverage:

    (a) the second-order approximation      exp(12*(m1 - M2/2)) - 1
    (b) the third-order approximation       + the M3 term
    (c) the ACTUAL empirically compounded return, prod(1+R_t)^(12/T) - 1

(c) is not an approximation of anything. **The gap between (c) and (a) is the entire
question.** All three are expressed identically as ``exp(12 * g_monthly) - 1`` so the only
difference between them is where the Taylor series was cut. Nothing else moves.

THE EXPANSION, AND A COEFFICIENT THIS MODULE REFUSES TO GUESS AT
================================================================
The task statement gives the third-order term as ``gamma*sigma^3*L^3/6``. The Taylor
series of ``log(1+R)`` about the mean gives ``+M3/3``, not ``M3/6`` --

    E[log(1+R)] = log(1+m) - M2/(2(1+m)^2) + M3/(3(1+m)^3) - M4/(4(1+m)^4) + ...

-- while the cumulant generating function ``log E[e^tX] = k1*t + k2*t^2/2 + k3*t^3/6``
gives the 1/6. The two conventions differ by a factor of two in how much the skew is
allowed to help. **This module reports BOTH** (`c3 = 1/3` log-expansion, `c3 = 1/6` as
stated in the brief) and lets curve (c) settle it, because (c) needs no convention at all.
Same for fourth order: ``-M4/4`` (log expansion) and ``-M4/24`` (cumulant).

LEVERAGE CONVENTION
===================
``R_t(L) = cash_t + L * r_t - max(L-1, 0) * spread / 12`` where ``r_t`` is the banked
UNLEVERED monthly excess return already net of 10bps round-trip. Trading cost therefore
scales with L automatically. This is the identical convention as
``research/sleeves/_survivor/survivor_verification.py::levered_total``, which is what
produced iteration 22's constant-leverage ladder. Financing spreads are iteration 11's:
bill+150bp primary, bill+300bp retail, bill+50bp optimistic.

CONSTANT L vs THE TAU-TARGETED LADDER
=====================================
Iteration 11's 15.83% came from a **volatility-targeted** ladder (k_t = tau/sigma_book,t),
not a constant leverage. A moment expansion in L is only meaningful at constant L, so the
expansion comparison runs at constant L and the tau-ladder is re-run separately as a
CONTROL that must reproduce 12.2955% / -47.2874% before any of this is believed.

NOT A NEW TRIAL
===============
No backtest configuration is searched here. Every return series analysed was banked by a
prior study; this module re-reads them, computes moments, and compounds them at different
leverages. Leverage is a position-sizing choice applied after the fact to an already-run
series, not a strategy parameter being selected -- iteration 11 spent its 2 trials on W1/W2
and charged nothing for its own six-rung leverage ladder, for exactly this reason. The
ledger therefore stays where it is; this module reads it, it never states it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.sleeves.multiasset_trend import (
    BOOK_VOL_MIN,
    BOOK_VOL_WINDOW,
    GROSS_CAP,
    MONTHS,
    TrendConfig,
    _positions,
)

# ── Fixed constants, all inherited ────────────────────────────────────────────

DATA = Path("_data/multiasset")
OUT_DIR = Path(__file__).resolve().parent
TREND_CSV = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")
VOL_TARGET = 0.20                       # the trend sleeve's banked headline target
NW_LAG = 6
RNG_SEED = 20260728                     # repo convention
N_BOOT = 10_000
BLOCK = 12                              # circular block bootstrap, months

FINANCING: dict[str, float] = {         # iteration 11's spreads over the 13-week bill
    "optimistic_bill_plus_50bp": 0.0050,
    "primary_bill_plus_150bp": 0.0150,
    "retail_bill_plus_300bp": 0.0300,
}
PRIMARY_FIN = "primary_bill_plus_150bp"

#: Taylor coefficients. ``log`` = expansion of log(1+R) about its mean; ``brief`` = the
#: cumulant-style coefficients quoted in the task statement. Both reported, never mixed.
C3 = {"log_expansion": 1.0 / 3.0, "brief_cumulant": 1.0 / 6.0}
C4 = {"log_expansion": 1.0 / 4.0, "brief_cumulant": 1.0 / 24.0}

LEV_GRID = np.round(np.arange(0.05, 8.0001, 0.05), 4)


# ── Primitives ────────────────────────────────────────────────────────────────

def ann_mean(x) -> float:
    return float(pd.Series(x).dropna().mean() * MONTHS)


def ann_vol(x) -> float:
    return float(pd.Series(x).dropna().std(ddof=1) * math.sqrt(MONTHS))


def sharpe(x) -> float:
    a = pd.Series(x).dropna()
    if len(a) < 8 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * math.sqrt(MONTHS))


def is_ruined(a: np.ndarray) -> bool:
    return bool(np.min(np.asarray(a, dtype=float)) <= -1.0)


def cagr(total) -> float:
    a = np.asarray(total, dtype=float)
    if is_ruined(a):
        return -1.0
    return float(np.exp(np.log1p(a).mean() * MONTHS) - 1.0)


def max_dd(total) -> float:
    a = np.asarray(total, dtype=float)
    if is_ruined(a):
        return -1.0
    curve = np.cumprod(1.0 + a)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def skewness(a: np.ndarray) -> float:
    """Fisher-Pearson g1 (the population estimator, ddof=0 in the denominator)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    d = a - a.mean()
    m2 = (d ** 2).sum() / n
    m3 = (d ** 3).sum() / n
    return float(m3 / m2 ** 1.5) if m2 > 0 else float("nan")


def excess_kurtosis(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    n = a.size
    d = a - a.mean()
    m2 = (d ** 2).sum() / n
    m4 = (d ** 4).sum() / n
    return float(m4 / m2 ** 2 - 3.0) if m2 > 0 else float("nan")


def se_skew_simple(n: int) -> float:
    """The brief's SE(gamma) ~= sqrt(6/T)."""
    return math.sqrt(6.0 / n)


def se_skew_exact(n: int) -> float:
    """Exact iid SE of g1 (Cramer): sqrt(6n(n-1) / ((n-2)(n+1)(n+3)))."""
    return math.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))


def se_kurt_exact(n: int) -> float:
    return math.sqrt(4.0 * (n ** 2 - 1) * se_skew_exact(n) ** 2 / ((n - 3) * (n + 5)))


def circular_blocks(n: int, block: int, rng: np.random.Generator, reps: int) -> np.ndarray:
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nb))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(reps, nb * block) % n
    return idx[:, :n]


def boot_moment(a: np.ndarray, fn, reps: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """Circular-block bootstrap CI of a moment estimator. Blocks keep autocorrelation."""
    a = np.asarray(a, dtype=float)
    idx = circular_blocks(a.size, BLOCK, np.random.default_rng(seed), reps)
    vals = np.array([fn(a[row]) for row in idx])
    vals = vals[np.isfinite(vals)]
    return {
        "boot_mean": float(vals.mean()),
        "boot_se": float(vals.std(ddof=1)),
        "ci2.5": float(np.percentile(vals, 2.5)),
        "ci97.5": float(np.percentile(vals, 97.5)),
        "p_le_0": float((vals <= 0).mean()),
    }


def nw_ols(y: np.ndarray, X: np.ndarray, lag: int = NW_LAG) -> dict:
    """OLS with Newey-West (Bartlett) HAC standard errors. X must include a constant."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None])
    for lag_i in range(1, min(lag, n - 1) + 1):
        w = 1.0 - lag_i / (lag + 1.0)
        G = (X[lag_i:] * e[lag_i:, None]).T @ (X[:-lag_i] * e[:-lag_i, None])
        S += w * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se],
        "t": [float(b / s) if s > 0 else float("nan") for b, s in zip(beta, se)],
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(n),
    }


# ── The leverage curves ───────────────────────────────────────────────────────

def levered_total(excess: np.ndarray, cash: np.ndarray, lev: float,
                  spread: float) -> np.ndarray:
    """L units of notional funded by 1 of equity and (L-1) borrowed at bill + spread.

    ``excess`` is already net of round-trip trading cost at 1x, so ``lev * excess``
    scales the trading cost with the notional, which is correct.
    """
    return lev * excess - max(lev - 1.0, 0.0) * spread / MONTHS + cash


def central_moments(a: np.ndarray) -> tuple[float, float, float, float]:
    """(m1, M2, M3, M4) -- the mean and the 2nd/3rd/4th central moments, per month."""
    a = np.asarray(a, dtype=float)
    n = a.size
    m1 = a.mean()
    d = a - m1
    return (float(m1), float((d ** 2).sum() / n), float((d ** 3).sum() / n),
            float((d ** 4).sum() / n))


def growth_orders(a: np.ndarray) -> dict:
    """Truncated growth estimates for one already-levered total-return series.

    All returned as ANNUAL COMPOUND, ``exp(12*g_monthly) - 1``, exactly as the empirical
    figure is computed, so the ONLY difference between them is the truncation.
    """
    m1, M2, M3, M4 = central_moments(a)
    g1 = m1
    g2 = m1 - M2 / 2.0
    out = {
        "m1_monthly": m1, "M2_monthly": M2, "M3_monthly": M3, "M4_monthly": M4,
        "mu_annual": m1 * MONTHS,
        "sigma_annual": math.sqrt(M2 * MONTHS),
        "skew": skewness(a),
        "exkurt": excess_kurtosis(a),
        "order1": float(np.exp(g1 * MONTHS) - 1.0),
        "order2": float(np.exp(g2 * MONTHS) - 1.0),
        "empirical": cagr(a),
        "max_dd": max_dd(a),
        "ruin": is_ruined(a),
    }
    for label, c3 in C3.items():
        g3 = g2 + c3 * M3
        out[f"order3_{label}"] = float(np.exp(g3 * MONTHS) - 1.0)
        out[f"order4_{label}"] = float(np.exp((g3 - C4[label] * M4) * MONTHS) - 1.0)
        out[f"term3_{label}_annual"] = float(c3 * M3 * MONTHS)
        out[f"term4_{label}_annual"] = float(-C4[label] * M4 * MONTHS)
    out["term2_annual"] = float(-M2 / 2.0 * MONTHS)
    return out


def leverage_curve(excess: pd.Series, cash: pd.Series, spread: float,
                   grid: np.ndarray = LEV_GRID) -> dict:
    """Curves (a) 2nd order, (b) 3rd order, (c) empirical, at every leverage."""
    e, c = excess.align(cash.reindex(excess.index), join="inner")
    ev = e.to_numpy(dtype=float)
    cv = c.to_numpy(dtype=float)
    rows: list[dict] = []
    for lev in grid:
        tot = levered_total(ev, cv, float(lev), spread)
        r = growth_orders(tot)
        r["leverage"] = float(lev)
        rows.append(r)
        if r["ruin"]:
            break
    return {"rows": rows, "months": int(len(ev)),
            "start": str(e.index[0].date()), "end": str(e.index[-1].date())}


def curve_peaks(curve: dict, dd_caps=(0.35, 0.50, 0.60)) -> dict:
    rows = [r for r in curve["rows"] if not r["ruin"]]
    if not rows:
        return {}

    def peak_of(key: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        return {"leverage": best["leverage"], "compound": best[key],
                "max_dd": best["max_dd"]}

    out: dict[str, Any] = {"peak_empirical": peak_of("empirical"),
           "peak_order2": peak_of("order2"),
           "peak_order1": peak_of("order1")}
    for label in C3:
        out[f"peak_order3_{label}"] = peak_of(f"order3_{label}")
        out[f"peak_order4_{label}"] = peak_of(f"order4_{label}")
    for cap in dd_caps:
        ok = [r for r in rows if abs(r["max_dd"]) <= cap]
        if ok:
            b = max(ok, key=lambda r: r["empirical"])
            out[f"dd{int(cap * 100)}"] = {"leverage": b["leverage"],
                                          "compound": b["empirical"],
                                          "max_dd": b["max_dd"],
                                          "order2_here": b["order2"]}
        else:
            out[f"dd{int(cap * 100)}"] = None
    # The single number the whole study is about.
    pe, p2 = out["peak_empirical"], out["peak_order2"]
    out["gap_peak_compound_pp"] = (pe["compound"] - p2["compound"]) * 100.0
    out["gap_peak_leverage"] = pe["leverage"] - p2["leverage"]
    return out


# ── Vol-targeting overlay, isolated so it can be applied to ANY series ─────────

def vol_target_overlay(excess: pd.Series, *, target: float = VOL_TARGET,
                       window: int = BOOK_VOL_WINDOW, min_obs: int = BOOK_VOL_MIN,
                       cap: float = GROSS_CAP) -> pd.Series:
    """Apply the trend sleeve's OWN book-level vol-targeting rule to any return series.

    Causal by construction: sigma is estimated through t and the scale is held during
    t+1. This is the overlay that mechanically cuts exposure after a volatile (usually
    losing) stretch, and it is the thing most likely to be manufacturing convexity.
    Available to any strategy, which is the point of testing it on passive too.
    """
    s = excess.dropna()
    sig = s.rolling(window, min_periods=min_obs).std(ddof=1) * math.sqrt(MONTHS)
    k = (target / sig.replace(0.0, np.nan)).clip(upper=cap)
    return (s * k.shift(1)).dropna()


# ── Trend variants for the source-of-skew decomposition ───────────────────────

def trend_variants(x: pd.DataFrame, interior: pd.DataFrame) -> dict[str, pd.Series]:
    """Rebuild the trend leg with the two sizing layers switched on and off.

    Layer 1 = per-instrument inverse-volatility sizing (``UNIT_VOL / sigma_i``).
    Layer 2 = the book-level volatility-targeting overlay (``k_t = tau / sigma_book,t``).

    **Skewness is scale-invariant**, so "remove the book overlay" needs no replacement
    constant and introduces no look-ahead whatsoever: the un-overlaid book is simply the
    raw position-weighted return series. That makes this counterfactual exact.

    Layer 2 is reproduced here rather than taken from ``run_trend`` so the SAME rule can
    be applied to the passive leg. The one deliberate departure from the registered
    sleeve: this overlay starts only once a 12-month volatility estimate EXISTS, where
    the registered sleeve lets the gross cap alone set the leverage for 12 months (the
    known P3 defect, worth +0.0022 of book Sharpe when repaired). Every layer comparison
    below is therefore also reported on the MATCHED window so the 12 months cannot be
    mistaken for a layer effect.
    """
    cfg = TrendConfig()
    n, _eligible, _count = _positions(x, cfg)
    held = interior.reindex_like(n).fillna(False)
    n = n.mask(held, 0.0)
    xz = x.fillna(0.0)

    # L1 on, L2 off -- the raw trend book, no book-level vol target.
    pos = n.shift(1).fillna(0.0)
    raw = (pos * xz).sum(axis=1).where(pos.abs().sum(axis=1) > 0).dropna()

    # L1 off (equal notional per live signal, sign preserved), L2 off.
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = n.div(n.abs().replace(0.0, np.nan)).fillna(0.0)
    pos_u = unit.shift(1).fillna(0.0)
    raw_unit = (pos_u * xz).sum(axis=1).where(pos_u.abs().sum(axis=1) > 0).dropna()

    return {
        "trend_raw_noL1_noL2": raw_unit,
        "trend_raw_L1_noL2": raw,
        "trend_L1_L2": vol_target_overlay(raw),
        "trend_noL1_L2": vol_target_overlay(raw_unit),
    }


def matched_window_skew(before: pd.Series, after: pd.Series) -> dict:
    """Skew before and after an overlay, measured on the overlay's OWN window.

    The overlay costs 12 months at the front. Without this, a 12-month window change
    could masquerade as a layer effect. It cannot masquerade past this function.
    """
    b = before.reindex(after.index).dropna()
    a = after.reindex(b.index).dropna()
    ab = np.asarray(b, dtype=float)
    aa = np.asarray(a, dtype=float)
    return {
        "months": int(len(aa)),
        "skew_before": skewness(ab), "skew_after": skewness(aa),
        "delta": skewness(aa) - skewness(ab),
        "exkurt_before": excess_kurtosis(ab), "exkurt_after": excess_kurtosis(aa),
        "sharpe_before": sharpe(ab), "sharpe_after": sharpe(aa),
        "z_before": skewness(ab) / se_skew_exact(len(ab)),
        "z_after": skewness(aa) / se_skew_exact(len(aa)),
    }


def boot_skew_iid(a: np.ndarray, reps: int = N_BOOT, seed: int = RNG_SEED) -> dict:
    """IID (non-block) bootstrap. Separates NON-NORMALITY from AUTOCORRELATION.

    ``sqrt(6/T)`` is the sampling SE of skewness **under normality**. These series carry
    excess kurtosis of 1.5 to 42, so that formula does not apply to any of them. This
    bootstrap says by how much it is wrong for reasons of shape alone; the block
    bootstrap adds the autocorrelation on top.
    """
    a = np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(reps, a.size))
    vals = np.array([skewness(a[row]) for row in idx])
    vals = vals[np.isfinite(vals)]
    return {"boot_se": float(vals.std(ddof=1)),
            "ci2.5": float(np.percentile(vals, 2.5)),
            "ci97.5": float(np.percentile(vals, 97.5))}


# ── Reporting helpers ─────────────────────────────────────────────────────────

def moment_report(name: str, s: pd.Series) -> dict:
    a = s.dropna().to_numpy(dtype=float)
    n = a.size
    g1, g2 = skewness(a), excess_kurtosis(a)
    se_s, se_k = se_skew_exact(n), se_kurt_exact(n)
    bs = boot_moment(a, skewness)
    bk = boot_moment(a, excess_kurtosis)
    bi = boot_skew_iid(a)
    return {
        "name": name, "months": n, "years": round(n / 12.0, 1),
        "start": str(s.dropna().index[0].date()), "end": str(s.dropna().index[-1].date()),
        "mean_annual": ann_mean(a), "vol_annual": ann_vol(a), "sharpe": sharpe(a),
        "skew": g1,
        "se_skew_brief_sqrt6overT": se_skew_simple(n),
        "se_skew_exact_iid": se_s,
        "z_skew_iid": g1 / se_s,
        "skew_boot": bs,
        "skew_boot_iid": bi,
        "se_inflation_vs_normal_theory": bs["boot_se"] / se_s,
        "skew_distinguishable_from_zero": bool(
            abs(g1 / se_s) >= 1.96 and (bs["ci2.5"] > 0 or bs["ci97.5"] < 0)),
        "exkurt": g2,
        "se_exkurt_exact_iid": se_k,
        "z_exkurt_iid": g2 / se_k,
        "exkurt_boot": bk,
        "worst_month": float(a.min()), "best_month": float(a.max()),
    }


def decade_moments(s: pd.Series) -> dict:
    out: dict[str, Any] = {}
    a = s.dropna()
    for dec, grp in a.groupby((a.index.year // 10) * 10):
        v = grp.to_numpy(dtype=float)
        if v.size < 24:
            out[f"{int(dec)}s"] = {"months": int(v.size), "note": "too short for a moment"}
            continue
        out[f"{int(dec)}s"] = {
            "months": int(v.size), "mean_annual": ann_mean(v), "vol_annual": ann_vol(v),
            "sharpe": sharpe(v), "skew": skewness(v), "exkurt": excess_kurtosis(v),
            "se_skew_exact_iid": se_skew_exact(v.size),
            "z_skew_iid": skewness(v) / se_skew_exact(v.size),
        }
    return out


def payload_md5(obj) -> str:
    return hashlib.md5(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()
