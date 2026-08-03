"""THE PORTFOLIO CORRELATION MEASUREMENT -- v2, on the CORRECTED low-vol series.

v1 (`portfolio_correlation.py` + `portfolio_decision.py`) is superseded. It had four
defects, all of which move the answer:

  1. It used the REGISTERED low-vol book (net Sharpe 0.8779). Iteration 10's independent
     verification corrected that to **0.614**. v1's "corrected" sensitivity used 0.677,
     which was the BUILDER's self-correction, not the verified one, and it rescaled the
     mean only rather than using the corrected return series.
  2. It mixed return conventions. The five multi-asset series are EXCESS returns over the
     13-week bill; low-vol is a TOTAL return. v1 put them in one covariance matrix and
     then levered the result, which implicitly borrows at 0%.
  3. It levered with NO FINANCING CHARGE. Iteration 11 measured that financing is what
     makes the leverage-return curve concave, and that moving from bill+50bp to bill+300bp
     swings the outcome by more than every strategy decision in the study combined.
  4. It labelled the 0.7065 passive benchmark as "the registered one" without recording
     that it is DAILY-rebalanced; the monthly-rebalanced equivalent is 0.668.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_correlation_v2
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from research.alignment import lag_correlations
from research.multiasset.panel import dsr_sharpe_bar
from research.validation import deflated_sharpe_ratio

REPO = Path(__file__).resolve().parents[3]
SLEEVES = REPO / "research" / "sleeves"
OUT_DIR = SLEEVES / "_portfolio"
CASH_PATH = REPO / "_data" / "multiasset" / "cash_monthly.parquet"

MPY = 12
RNG = np.random.default_rng(20260728)
BOOT_BLOCK = 12
BOOT_N = 4000

TARGET_SHARPE_30PCT = 0.894      # 30%/yr at half Kelly
THIRD_SLEEVE_BAR = 0.621         # iteration 7's uncorrelated-third-sleeve bar
N_TRIALS_PROGRAMME = 46          # iteration 11 left the cumulative count at 46
FINANCING = {"optimistic_bill_plus_50bp": 0.0050,
             "primary_bill_plus_150bp": 0.0150,
             "retail_bill_plus_300bp": 0.0300}

log = logging.getLogger("portfolio_v2")

# ── Sources ───────────────────────────────────────────────────────────────────
# (path, column, convention)  convention: "excess" over the 13-week bill, or "total"
SOURCES: dict[str, tuple[Path, str, str]] = {
    "lowvol":    (OUT_DIR / "lowvol_b2_corrected_monthly.parquet", "net_conservative", "total"),
    "lowvol_reg": (OUT_DIR / "lowvol_b2_net_monthly.parquet", "net_conservative", "total"),
    "trend":     (SLEEVES / "_multiasset_trend" / "primary_20pct_monthly.csv", "net_10bps", "excess"),
    "carry":     (SLEEVES / "_carry_output" / "carry_primary_net_monthly.parquet", "net", "excess"),
    "seasonal":  (SLEEVES / "_seasonal" / "seasonal_composite_20pct_monthly.parquet",
                  "seasonal_net_10bps", "excess"),
    "defensive": (SLEEVES / "_defensive" / "defensive_primary_net_monthly.parquet", "net", "excess"),
    # Two benchmark variants. LABEL WHICHEVER IS USED.
    "passive_daily":   (SLEEVES / "_seasonal" / "seasonal_composite_20pct_monthly.parquet",
                        "bench_net_10bps", "excess"),
    "passive_monthly": (SLEEVES / "_multiasset_trend" / "primary_20pct_monthly.csv",
                        "bench_net_10bps", "excess"),
}

# The five sleeves plus ONE benchmark, for the headline matrix.
MATRIX_NAMES = ["lowvol", "trend", "carry", "seasonal", "defensive", "passive_monthly"]
SLEEVE_NAMES = ["lowvol", "trend", "carry", "seasonal", "defensive"]


# ── THE ONE-MONTH DATING DEFECT ───────────────────────────────────────────────
# `lowvol_retest.run_band` labels each month by the FORMATION month but fills it with
# `forward_return`, which is the close-to-close return of the FOLLOWING month. Every
# low-vol monthly slot is therefore dated ONE MONTH EARLY.
#
# This is invisible to any statistic computed WITHIN the series -- mean, volatility,
# Sharpe, drawdown and the vol-matched active are all unchanged by shifting every
# observation by one month -- which is why the iteration-10 verification did not catch it.
# It is only visible when the series is JOINED TO ANOTHER SERIES BY DATE, which this
# study is the first thing in the programme to do.
#
# Measured proof (`alignment_control` below): the low-vol book's own US-equity benchmark
# correlates with SPX at +0.189 contemporaneously and +0.769 at SPX(t+1). After the shift
# the ordering reverses to +0.769 contemporaneous / +0.189 at the lag, which is what a
# US-equity book must look like. Every multi-asset series already peaks at k=0.
NEEDS_MONTH_SHIFT = {"lowvol", "lowvol_reg"}


def _load(path: Path, col: str, *, shift_months: int = 0) -> pd.Series:
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


def load_source(key: str, col: str | None = None) -> pd.Series:
    path, default_col, _ = SOURCES[key]
    return _load(path, col or default_col,
                 shift_months=1 if key in NEEDS_MONTH_SHIFT else 0)


def reference_series() -> pd.Series:
    """Correctly-dated SPX — the external anchor the whole probe rests on."""
    spx = pd.read_parquet(REPO / "_data" / "multiasset" / "returns_monthly.parquet")["SPX"]
    spx.index = pd.DatetimeIndex(spx.index).to_period("M").to_timestamp(how="end").normalize()
    return spx


def alignment_control() -> dict:
    """Every series against correctly-dated SPX at k=-1/0/+1. The lag of largest |rho|
    must be 0 for a series whose dates mean 'the month this return was earned'.

    The measurement is now `research.alignment.lag_correlations`, which is shared,
    importable and unit-tested. The KEY NAMES below are the ones this study banked and
    are kept verbatim: `rho_lag_minus1` is the correlation of the series against the
    reference ONE MONTH AHEAD (a pandas `.shift(-1)` on the reference), i.e. the shared
    module's lag k=+1. The sign convention is therefore inverted relative to
    `research.alignment`, which is exactly why the shared module spells its own out.
    """
    spx = reference_series()
    out = {}
    for key, (path, col, _) in SOURCES.items():
        probe = "benchmark" if key in NEEDS_MONTH_SHIFT else col
        rows = {}
        for label, sh in (("uncorrected", 0), ("corrected", 1 if key in NEEDS_MONTH_SHIFT else 0)):
            s = _load(path, probe, shift_months=sh)
            # shared-module lag k compares s(t) with spx(t+k); the banked key
            # `rho_lag_<n>` is the pandas shift, so the two differ by a sign.
            rho, _n = lag_correlations(s, spx, lags=(-1, 0, 1))
            vals = {-k: rho[k] for k in (-1, 0, 1)}
            rows[label] = {"rho_lag_minus1": vals[-1], "rho_lag_0": vals[0],
                           "rho_lag_plus1": vals[1],
                           "argmax_abs_lag": int(max(vals, key=lambda k: abs(vals[k])))}
        out[key] = {"probe_column": probe, "shift_applied_months":
                    1 if key in NEEDS_MONTH_SHIFT else 0, **rows}
    return out


# ── Statistics ────────────────────────────────────────────────────────────────
def sharpe(x) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.mean(a) / np.std(a, ddof=1) * math.sqrt(MPY))


def ann_vol(x) -> float:
    return float(np.std(np.asarray(x, dtype=float), ddof=1) * math.sqrt(MPY))


def ann_mean(x) -> float:
    return float(np.mean(np.asarray(x, dtype=float)) * MPY)


def is_ruined(x) -> bool:
    """A month at or below -100% wipes the account out. There is no path after it."""
    return bool(np.min(np.asarray(x, dtype=float)) <= -1.0)


def cagr(x) -> float:
    a = np.asarray(x, dtype=float)
    if is_ruined(a):
        return -1.0
    return float(np.prod(1.0 + a) ** (MPY / len(a)) - 1.0)


def max_dd(x) -> float:
    a = np.asarray(x, dtype=float)
    if is_ruined(a):
        return -1.0
    curve = np.cumprod(1.0 + a)
    return float(np.min(curve / np.maximum.accumulate(curve) - 1.0))


def corr_with_error(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r plus BOTH standard errors and BOTH intervals.

    `se_delta` is the large-sample delta-method SE on the r scale, (1-r^2)/sqrt(n-1).
    The Fisher interval is exact under bivariate normality; the block bootstrap is the
    one that survives autocorrelation and is the one to quote.
    """
    n = len(a)
    r = float(np.corrcoef(a, b)[0, 1])
    se_delta = (1.0 - r * r) / math.sqrt(max(n - 1, 1))
    z = math.atanh(max(min(r, 1 - 1e-12), -1 + 1e-12))
    se_z = 1.0 / math.sqrt(max(n - 3, 1))
    lo_f, hi_f = math.tanh(z - 1.96 * se_z), math.tanh(z + 1.96 * se_z)
    t = r * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - r * r, 1e-12))
    p = float(2 * stats.t.sf(abs(t), df=max(n - 2, 1)))

    # circular moving-block bootstrap on the PAIR, preserving joint dependence
    nb = int(math.ceil(n / BOOT_BLOCK))
    starts = RNG.integers(0, n, size=(BOOT_N, nb))
    offs = np.arange(BOOT_BLOCK)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(BOOT_N, -1)[:, :n] % n
    aa, bb = a[idx], b[idx]
    aa = aa - aa.mean(axis=1, keepdims=True)
    bb = bb - bb.mean(axis=1, keepdims=True)
    num = (aa * bb).sum(axis=1)
    den = np.sqrt((aa * aa).sum(axis=1) * (bb * bb).sum(axis=1))
    boot = num / np.where(den == 0, np.nan, den)
    lo_b, hi_b = np.nanpercentile(boot, [2.5, 97.5])
    return {"n": n, "r": r, "se_delta": se_delta, "se_fisher_z": se_z,
            "ci95_fisher": [lo_f, hi_f], "ci95_block_boot": [float(lo_b), float(hi_b)],
            "t": t, "p_two_sided": p,
            "boot_se": float(np.nanstd(boot, ddof=1))}


def block_bootstrap_sharpe(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    nb = int(math.ceil(n / BOOT_BLOCK))
    starts = RNG.integers(0, n, size=(BOOT_N, nb))
    offs = np.arange(BOOT_BLOCK)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(BOOT_N, -1)[:, :n] % n
    s = x[idx]
    sh = s.mean(axis=1) / s.std(axis=1, ddof=1) * math.sqrt(MPY)
    lo, hi = np.nanpercentile(sh, [2.5, 97.5])
    return float(lo), float(hi)


# ── Weighting schemes, all from the MEASURED covariance ───────────────────────
def equal_weights(f: pd.DataFrame) -> np.ndarray:
    return np.full(f.shape[1], 1.0 / f.shape[1])


def inverse_vol_weights(f: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / f.std(ddof=1).to_numpy()
    return iv / iv.sum()


def inverse_variance_weights(f: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / f.var(ddof=1).to_numpy()
    return iv / iv.sum()


def erc_weights(f: pd.DataFrame) -> np.ndarray:
    """True equal-risk-contribution, solved from the measured covariance."""
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
    if best is None:
        raise RuntimeError("the ERC optimiser returned no solution for this book")
    w = np.exp(best)
    return w / w.sum()


SCHEMES = {"equal_weight": equal_weights,
           "inverse_vol": inverse_vol_weights,
           "inverse_variance": inverse_variance_weights,
           "erc": erc_weights}


# ── Leverage with an EXPLICIT financing charge (iteration 11's method) ────────
def levered_total(x_excess: np.ndarray, cash: np.ndarray, lev: float, spread: float) -> np.ndarray:
    """Total return of an excess-return book levered `lev` times.

    Borrowing is charged on max(lev-1, 0) notional at bill + `spread`; the bill leg is
    already inside `cash`, so only the spread is an extra deduction.
    """
    return lev * x_excess - max(lev - 1.0, 0.0) * spread / MPY + cash


def leverage_curve(x_excess: np.ndarray, cash: np.ndarray, spread: float,
                   grid: np.ndarray | None = None) -> dict:
    grid = np.arange(0.25, 30.001, 0.05) if grid is None else grid
    rows, ruin_lev = [], None
    for L in grid:
        tot = levered_total(x_excess, cash, float(L), spread)
        if is_ruined(tot):
            ruin_lev = float(L)
            break
        rows.append((float(L), cagr(tot), max_dd(tot), ann_vol(tot)))
    arr = np.array(rows)
    peak = int(np.argmax(arr[:, 1]))
    out = {"ruin_leverage": ruin_lev,
           "peak_leverage": arr[peak, 0], "peak_cagr": arr[peak, 1],
           "peak_max_dd": arr[peak, 2], "peak_vol": arr[peak, 3],
           "peak_is_at_ruin_boundary": bool(peak == len(arr) - 1)}
    for cap in (0.35, 0.50, 0.60):
        ok = arr[np.abs(arr[:, 2]) <= cap]
        if len(ok):
            j = int(np.argmax(ok[:, 1]))
            out[f"dd_cap_{int(cap*100)}"] = {"leverage": ok[j, 0], "cagr": ok[j, 1],
                                             "max_dd": ok[j, 2], "vol": ok[j, 3]}
        else:
            out[f"dd_cap_{int(cap*100)}"] = None
    return out


def kelly_block(x_excess: np.ndarray, cash: np.ndarray, label: str,
                spread: float = FINANCING["primary_bill_plus_150bp"]) -> dict:
    s = sharpe(x_excess)
    vol = ann_vol(x_excess)
    lev_half = s / (2.0 * vol)
    tot_half = levered_total(x_excess, cash, lev_half, spread)
    tot_1x = levered_total(x_excess, cash, 1.0, spread)
    curve = leverage_curve(x_excess, cash, spread)
    return {
        "label": label,
        "n_months": int(len(x_excess)),
        "years": len(x_excess) / MPY,
        "sharpe_excess": s,
        "vol_excess_1x": vol,
        "mean_excess_annual": ann_mean(x_excess),
        "cagr_total_1x": cagr(tot_1x),
        "max_dd_1x": max_dd(tot_1x),
        "cash_rate_mean_annual": ann_mean(cash),
        # theoretical half-Kelly
        "half_kelly_growth_theoretical": 3.0 * s * s / 8.0,
        "half_kelly_required_vol": s / 2.0,
        "half_kelly_leverage": lev_half,
        # measured at that leverage
        "half_kelly_cagr_measured": cagr(tot_half),
        "half_kelly_dd_linear_scaled": max_dd(x_excess) * lev_half,
        "half_kelly_dd_measured": max_dd(tot_half),
        "half_kelly_vol_measured": ann_vol(tot_half),
        "half_kelly_ruined": is_ruined(tot_half),
        "half_kelly_reachable_at_60pct_dd": bool(
            (not is_ruined(tot_half)) and abs(max_dd(tot_half)) <= 0.60),
        "full_kelly_growth_theoretical": s * s / 2.0,
        "leverage_curve": curve,
        "worst_month_1x": float(np.min(x_excess)),
        "clears_0894": bool(s >= TARGET_SHARPE_30PCT),
        "financing_spread": spread,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────
def partial_corr(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ra: np.ndarray = a - np.polyval(np.polyfit(c, a, 1), c)
    rb: np.ndarray = b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])


def lead_lag(a: pd.Series, b: pd.Series, kmax: int = 12) -> dict:
    out = {}
    for k in range(-kmax, kmax + 1):
        f = pd.concat({"a": a, "b": b.shift(k)}, axis=1).dropna()
        if len(f) >= 24:
            out[str(k)] = {"n": int(len(f)),
                           "r": float(np.corrcoef(f["a"], f["b"])[0, 1])}
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    out: dict = {}

    raw = {k: load_source(k) for k in SOURCES}
    conv = {k: v[2] for k, v in SOURCES.items()}
    cash = _load(CASH_PATH, "US_CASH_13W")

    # ---- 0. DATING ALIGNMENT CONTROL, run BEFORE anything is correlated ------
    align = alignment_control()
    out["alignment_control"] = align
    bad = [k for k, v in align.items() if v["corrected"]["argmax_abs_lag"] != 0
           and abs(v["corrected"]["rho_lag_0"]) < 0.05]
    out["alignment_control_note"] = (
        "Series whose largest |rho| against SPX is not at lag 0 AND whose lag-0 rho is "
        "below 0.05 are long/short books with no market exposure, for which the probe is "
        "uninformative rather than failing: " + ", ".join(bad))
    print("=" * 100)
    print("DATING ALIGNMENT CONTROL vs correctly-dated SPX")
    print("=" * 100)
    print(f"{'series':>18} {'shift':>6} {'k=-1':>9} {'k=0':>9} {'k=+1':>9} {'argmax':>7}")
    for k, v in align.items():
        c = v["corrected"]
        print(f"{k:>18} {v['shift_applied_months']:>6} {c['rho_lag_minus1']:>+9.4f} "
              f"{c['rho_lag_0']:>+9.4f} {c['rho_lag_plus1']:>+9.4f} "
              f"{c['argmax_abs_lag']:>+7d}")

    # ---- 1. PROVENANCE, and the excess/total reconciliation ------------------
    prov: dict[str, dict[str, Any]] = {}
    series: dict[str, pd.Series] = {}
    for k, s in raw.items():
        if conv[k] == "total":
            ex = (s - cash.reindex(s.index)).dropna()
        else:
            ex = s
        series[k] = ex
        prov[k] = {
            "path": str(SOURCES[k][0].relative_to(REPO)).replace("\\", "/"),
            "column": SOURCES[k][1],
            "convention_on_disk": conv[k],
            "n": int(len(s)),
            "first": str(s.index.min().date()), "last": str(s.index.max().date()),
            "sharpe_as_stored": sharpe(s), "vol_as_stored": ann_vol(s),
            "annual_mean_as_stored": ann_mean(s),
            "max_dd_as_stored": max_dd(s),
            "sharpe_excess_basis": sharpe(ex), "vol_excess_basis": ann_vol(ex),
            "annual_mean_excess_basis": ann_mean(ex),
            "n_excess_basis": int(len(ex)),
        }
    out["provenance"] = prov
    out["cash"] = {"path": "_data/multiasset/cash_monthly.parquet",
                   "column": "US_CASH_13W",
                   "mean_annual_full": ann_mean(cash),
                   "mean_annual_1998_2015": ann_mean(
                       cash.loc["1998-04-30":"2015-12-31"])}

    # ---- 2. PAIRWISE CORRELATION MATRIX, each pair on its own overlap --------
    pairs = {}
    for a, b in itertools.combinations(MATRIX_NAMES, 2):
        f = pd.concat({a: series[a], b: series[b]}, axis=1).dropna()
        if len(f) < 24:
            continue
        d = corr_with_error(f[a].to_numpy(), f[b].to_numpy())
        d["window"] = [str(f.index.min().date()), str(f.index.max().date())]
        pairs[f"{a}~{b}"] = d
    out["pairwise"] = pairs

    # the same pairs on the REGISTERED low-vol, so the effect of the correction is visible
    reg_pairs = {}
    for b in ["trend", "carry", "seasonal", "defensive", "passive_monthly"]:
        f = pd.concat({"lowvol_reg": series["lowvol_reg"], b: series[b]}, axis=1).dropna()
        reg_pairs[f"lowvol_reg~{b}"] = corr_with_error(
            f["lowvol_reg"].to_numpy(), f[b].to_numpy())
    out["pairwise_registered_lowvol"] = reg_pairs

    # the same pairs against the DAILY-rebalanced benchmark, so the label matters visibly
    daily_pairs = {}
    for a in SLEEVE_NAMES:
        f = pd.concat({a: series[a], "pd_": series["passive_daily"]}, axis=1).dropna()
        daily_pairs[f"{a}~passive_daily"] = corr_with_error(
            f[a].to_numpy(), f["pd_"].to_numpy())
    f = pd.concat({"m": series["passive_monthly"], "d": series["passive_daily"]},
                  axis=1).dropna()
    daily_pairs["passive_monthly~passive_daily"] = corr_with_error(
        f["m"].to_numpy(), f["d"].to_numpy())
    out["pairwise_passive_daily"] = daily_pairs
    out["benchmark_variants"] = {
        "passive_daily": {"sharpe": sharpe(series["passive_daily"]),
                          "n": int(len(series["passive_daily"])),
                          "label": "DAILY-rebalanced equal weight (the recorded 0.7065)"},
        "passive_monthly": {"sharpe": sharpe(series["passive_monthly"]),
                            "n": int(len(series["passive_monthly"])),
                            "label": ("MONTHLY-rebalanced equal weight; iteration 11 "
                                      "rebuilt this independently at 0.6678")},
    }

    # ---- 3. COMMON-WINDOW MATRIX --------------------------------------------
    common = pd.concat({k: series[k] for k in MATRIX_NAMES}, axis=1).dropna()
    out["common_window"] = {
        "n": int(len(common)),
        "first": str(common.index.min().date()), "last": str(common.index.max().date()),
        "corr": common.corr().round(6).to_dict(),
        "sharpe": {k: sharpe(common[k]) for k in common.columns},
        "vol": {k: ann_vol(common[k]) for k in common.columns},
    }
    # and on low-vol's own 213-month window
    lv_win = pd.concat({k: series[k] for k in MATRIX_NAMES if k != "carry"},
                       axis=1).dropna()
    out["lowvol_window"] = {
        "n": int(len(lv_win)),
        "first": str(lv_win.index.min().date()), "last": str(lv_win.index.max().date()),
        "corr": lv_win.corr().round(6).to_dict(),
        "sharpe": {k: sharpe(lv_win[k]) for k in lv_win.columns},
    }

    # ---- 4. THE OVERLAP DIAGNOSTIC ------------------------------------------
    diag: dict = {}

    # (a) partial correlation removing the passive benchmark (the market factor)
    pc = {}
    for a, b in itertools.combinations(SLEEVE_NAMES, 2):
        f = pd.concat({a: series[a], b: series[b], "p": series["passive_monthly"]},
                      axis=1).dropna()
        if len(f) < 24:
            continue
        r_raw = float(np.corrcoef(f[a], f[b])[0, 1])
        r_par = partial_corr(f[a].to_numpy(), f[b].to_numpy(), f["p"].to_numpy())
        pc[f"{a}~{b}"] = {"n": int(len(f)), "raw": r_raw, "partial_ex_passive": r_par,
                          "delta": r_par - r_raw}
    diag["partial_ex_passive"] = pc

    # (b) LEAD-LAG. This is the direct test for the value-sleeve failure mode: if two
    #     sleeves act on the SAME information at DIFFERENT lags, the contemporaneous
    #     correlation is near zero while a lagged one is large. A flat lead-lag profile
    #     is evidence that a low contemporaneous rho is economic, not a window artefact.
    ll = {}
    for b in ["trend", "carry", "seasonal", "defensive", "passive_monthly"]:
        prof = lead_lag(series["lowvol"], series[b])
        vals = {k: v["r"] for k, v in prof.items()}
        contemp = vals["0"]
        worst_k = max(vals, key=lambda k: abs(vals[k]))
        ll[f"lowvol~{b}"] = {"profile": prof, "contemporaneous": contemp,
                             "max_abs_lag": worst_k, "max_abs_r": vals[worst_k],
                             "exceeds_contemporaneous": bool(
                                 abs(vals[worst_k]) > abs(contemp) + 2 * (
                                     1.0 / math.sqrt(prof[worst_k]["n"])))}
    diag["lead_lag"] = ll

    # (c) window sensitivity -- every multi-asset pair re-measured on low-vol's window
    ws = {}
    lv_index = series["lowvol"].index
    for a, b in itertools.combinations(["trend", "carry", "seasonal", "defensive",
                                        "passive_monthly"], 2):
        full = pd.concat({a: series[a], b: series[b]}, axis=1).dropna()
        win = full.reindex(lv_index).dropna()
        if len(win) < 24:
            continue
        ws[f"{a}~{b}"] = {"n_full": int(len(full)),
                          "r_full": float(np.corrcoef(full[a], full[b])[0, 1]),
                          "n_window": int(len(win)),
                          "r_window": float(np.corrcoef(win[a], win[b])[0, 1])}
        ws[f"{a}~{b}"]["delta"] = ws[f"{a}~{b}"]["r_window"] - ws[f"{a}~{b}"]["r_full"]
    diag["window_sensitivity"] = ws

    # (d) split-half stability of every low-vol pair
    sh = {}
    for b in ["trend", "carry", "seasonal", "defensive", "passive_monthly"]:
        f = pd.concat({"lowvol": series["lowvol"], b: series[b]}, axis=1).dropna()
        h = len(f) // 2
        r1 = float(np.corrcoef(f["lowvol"][:h], f[b][:h])[0, 1])
        r2 = float(np.corrcoef(f["lowvol"][h:], f[b][h:])[0, 1])
        sh[f"lowvol~{b}"] = {"n": int(len(f)), "first_half": r1, "second_half": r2,
                             "spread": abs(r1 - r2)}
    diag["split_half"] = sh
    out["diagnostics"] = diag

    # ---- 5. PORTFOLIO SHARPE FROM THE MEASURED COVARIANCE --------------------
    combos: list[dict[str, Any]] = []
    universe = SLEEVE_NAMES + ["passive_monthly"]
    for size in range(1, len(universe) + 1):
        for combo in itertools.combinations(universe, size):
            f = pd.concat({c: series[c] for c in combo}, axis=1).dropna()
            if len(f) < 24:
                continue
            for scheme, wfun in SCHEMES.items():
                w = wfun(f) if size > 1 else np.array([1.0])
                port = f.to_numpy() @ w
                combos.append({
                    "combo": list(combo), "scheme": scheme,
                    "n_months": int(len(f)), "years": len(f) / MPY,
                    "first": str(f.index.min().date()), "last": str(f.index.max().date()),
                    "weights": {c: float(x) for c, x in zip(f.columns, w)},
                    "sharpe": sharpe(port), "vol": ann_vol(port),
                    "mean_annual": ann_mean(port),
                    "clears_0894": bool(sharpe(port) >= TARGET_SHARPE_30PCT),
                })
                if size == 1:
                    break
    out["combinations"] = sorted(combos, key=lambda e: -e["sharpe"])
    out["n_combinations"] = len(combos)
    out["n_clearing_0894"] = sum(1 for e in combos if e["clears_0894"])

    # SANITY CONTROL: the equal-Sharpe shortcut, evaluated with MEASURED inputs, must
    # equal the equal-weight-of-vol-normalised portfolio. Records the size of the gap.
    shortcut = []
    for e in combos:
        if len(e["combo"]) < 2 or e["scheme"] != "inverse_vol":
            continue
        f = pd.concat({c: series[c] for c in e["combo"]}, axis=1).dropna()
        n = f.shape[1]
        s_bar = float(np.mean([sharpe(f[c]) for c in f.columns]))
        cm = f.corr().to_numpy()
        rho_bar = float((cm.sum() - n) / (n * (n - 1)))
        approx = s_bar * math.sqrt(n / (1.0 + (n - 1) * rho_bar))
        shortcut.append({"combo": e["combo"], "measured": e["sharpe"],
                         "shortcut_with_measured_inputs": approx,
                         "abs_gap": abs(approx - e["sharpe"])})
    out["equal_sharpe_shortcut_control"] = {
        "max_abs_gap": max(s["abs_gap"] for s in shortcut) if shortcut else None,
        "rows": shortcut,
    }

    # ---- 6. KELLY / LEVERAGE, WITH FINANCING ---------------------------------
    headline = [
        ["lowvol"], ["trend"], ["carry"], ["seasonal"], ["defensive"],
        ["passive_monthly"],
        ["lowvol", "trend"], ["lowvol", "trend", "carry"],
        ["lowvol", "trend", "defensive"], ["lowvol", "trend", "carry", "defensive"],
        ["trend", "carry"], SLEEVE_NAMES,
        ["lowvol", "trend", "carry", "passive_monthly"],
    ]
    kelly = []
    for book_combo in headline:
        f = pd.concat({c: series[c] for c in book_combo}, axis=1).dropna()
        if len(f) < 24:
            continue
        cs = cash.reindex(f.index).to_numpy()
        for scheme in ("equal_weight", "inverse_vol", "inverse_variance", "erc"):
            if len(book_combo) == 1 and scheme != "equal_weight":
                continue
            w = SCHEMES[scheme](f) if len(book_combo) > 1 else np.array([1.0])
            port = f.to_numpy() @ w
            blk = kelly_block(port, cs, f"{'+'.join(book_combo)} [{scheme}]")
            blk["combo"], blk["scheme"] = book_combo, scheme
            blk["weights"] = {c: float(x) for c, x in zip(f.columns, w)}
            blk["first"], blk["last"] = str(f.index.min().date()), str(f.index.max().date())
            lo, hi = block_bootstrap_sharpe(port)
            blk["sharpe_ci95_block_boot"] = [lo, hi]
            n = len(port)
            se = float(math.sqrt((1 + 0.5 * (blk["sharpe_excess"] / math.sqrt(MPY)) ** 2) / n)
                       * math.sqrt(MPY))
            blk["sharpe_se_analytic"] = se
            blk["sharpe_ci95_analytic"] = [blk["sharpe_excess"] - 1.96 * se,
                                           blk["sharpe_excess"] + 1.96 * se]
            blk["dsr_bar_n46"] = dsr_sharpe_bar(blk["years"], n_trials=N_TRIALS_PROGRAMME)
            blk["dsr_bar_incl_search"] = dsr_sharpe_bar(
                blk["years"], n_trials=N_TRIALS_PROGRAMME + out["n_combinations"] // 4)
            blk["dsr_n46"] = float(deflated_sharpe_ratio(port, n_trials=N_TRIALS_PROGRAMME))
            # financing sensitivity on the leverage curve
            blk["financing_sensitivity"] = {
                lab: leverage_curve(port, cs, sp) for lab, sp in FINANCING.items()}
            kelly.append(blk)
    out["kelly"] = kelly

    best = max((b for b in kelly if len(b["combo"]) > 1), key=lambda b: b["sharpe_excess"])
    out["best_by_sharpe"] = best["label"]

    (OUT_DIR / "portfolio_correlation_v2.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    # ---- console summary -----------------------------------------------------
    print("\n" + "=" * 100)
    print("PROVENANCE (as stored / on a common EXCESS-over-cash basis)")
    print("=" * 100)
    print(f"{'sleeve':>17} {'n':>5} {'window':>20} {'conv':>7} {'Sh(stored)':>11} "
          f"{'Sh(excess)':>11} {'vol':>7}")
    for k in ["lowvol", "lowvol_reg", "trend", "carry", "seasonal", "defensive",
              "passive_monthly", "passive_daily"]:
        p = prov[k]
        print(f"{k:>17} {p['n']:>5} {p['first'][:7]}->{p['last'][:7]:>8} "
              f"{p['convention_on_disk']:>7} {p['sharpe_as_stored']:>11.4f} "
              f"{p['sharpe_excess_basis']:>11.4f} {p['vol_excess_basis']:>7.2%}")

    print("\n" + "=" * 100)
    print("PAIRWISE CORRELATIONS (corrected low-vol, excess basis)")
    print("=" * 100)
    print(f"{'pair':>34} {'n':>5} {'rho':>8} {'SE':>7} {'Fisher 95%':>20} {'boot 95%':>20}")
    for k, v in pairs.items():
        print(f"{k:>34} {v['n']:>5} {v['r']:>+8.4f} {v['se_delta']:>7.3f} "
              f"[{v['ci95_fisher'][0]:>+7.3f},{v['ci95_fisher'][1]:>+7.3f}] "
              f"[{v['ci95_block_boot'][0]:>+7.3f},{v['ci95_block_boot'][1]:>+7.3f}]")

    print("\n" + "=" * 100)
    print("TOP COMBINATIONS BY MEASURED SHARPE (excess basis)")
    print("=" * 100)
    print(f"{'combo':>52} {'scheme':>17} {'n':>5} {'Sharpe':>8} {'>=0.894':>8}")
    for e in out["combinations"][:18]:
        print(f"{'+'.join(e['combo']):>52} {e['scheme']:>17} {e['n_months']:>5} "
              f"{e['sharpe']:>+8.4f} {'YES' if e['clears_0894'] else 'no':>8}")
    print(f"\n  {out['n_clearing_0894']} of {out['n_combinations']} clear 0.894")
    print(f"  equal-Sharpe shortcut control: max |gap| vs measured = "
          f"{out['equal_sharpe_shortcut_control']['max_abs_gap']:.3e}")

    print("\n" + "=" * 100)
    print("KELLY / LEVERAGE WITH FINANCING AT BILL+150bp")
    print("=" * 100)
    print(f"{'book':>44} {'n':>5} {'S':>7} {'vol':>7} {'hK lev':>7} {'hK CAGR':>8} "
          f"{'hK DD':>8} {'peak':>8} {'peak DD':>8} {'DD<=50%':>9} {'DD<=60%':>9}")
    for bk in kelly:
        c50 = bk["leverage_curve"]["dd_cap_50"]
        c60 = bk["leverage_curve"]["dd_cap_60"]
        print(f"{bk['label']:>44} {bk['n_months']:>5} {bk['sharpe_excess']:>+7.3f} "
              f"{bk['vol_excess_1x']:>7.2%} {bk['half_kelly_leverage']:>7.2f} "
              f"{bk['half_kelly_cagr_measured']:>+8.2%} {bk['half_kelly_dd_measured']:>+8.1%} "
              f"{bk['leverage_curve']['peak_cagr']:>+8.2%} "
              f"{bk['leverage_curve']['peak_max_dd']:>+8.1%} "
              f"{(c50['cagr'] if c50 else float('nan')):>+9.2%} "
              f"{(c60['cagr'] if c60 else float('nan')):>+9.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
