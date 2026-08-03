"""
TradingEngineResearch — Volatility Model
============================
The single source of truth for volatility and covariance estimation.

Both volatility paths live here:
  - GJR-GARCH(1,1,1) with Student-t errors, via the ``arch`` library, capturing
    the volatility-clustering and leverage (asymmetry) effects.
  - HAR-RV (Corsi 2009), via ``statsmodels`` OLS, capturing long-memory realised
    volatility across daily / weekly / monthly horizons.

The ensemble switching logic (by history length) and the shared helpers
``vol_ratio_current()`` and ``rmt_denoise_cov()`` — both consumed by
``core.crisis_manager`` and ``core.engine.optimizer`` — also live here.

Design notes
------------
The master prompt specifies ``fit_gjr_garch`` / ``fit_har_rv`` as returning
parameter dicts, and ``forecast_vol`` as consuming those dicts. A parameter set
alone is not enough to produce a *conditioned* forecast, so each fitter also
stores the minimal latest state it computed (last conditional variance / last
residual for GARCH; latest daily/weekly/monthly RV components plus the next-day
RV forecast for HAR). These are additive keys — every interface key the spec
names is still present and unchanged — so downstream callers that read the
documented keys keep working.

All volatilities are returned **annualised** (multiply daily σ by √252) and as
decimal fractions (0.20 == 20% annualised).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Accepted return-series input: any float sequence or a 1-D numpy array.
Returns = Sequence[float] | np.ndarray

__all__ = [
    "fit_gjr_garch",
    "fit_har_rv",
    "forecast_vol",
    "fit",
    "vol_ratio_current",
    "rmt_denoise_cov",
]

# ── Constants ──────────────────────────────────────────────────────────────────

TRADING_DAYS: int = 252
ANNUALISATION: float = math.sqrt(TRADING_DAYS)

MIN_GJR_SAMPLES: int = 30           # minimum returns to attempt a GARCH-family fit
MIN_HAR_SAMPLES: int = 60           # minimum returns to fit HAR-RV (Corsi 2009)
MIN_ENSEMBLE_SAMPLES: int = 60      # history at/above which the ensemble is used

# VREG-1 fail-safe. With fewer than two observations there is no variance to
# measure, and the previous answer — 0.0 — is the most dangerous one a risk path
# can give: zero variance reads as a RISK-FREE asset, so a vol-targeting scaler
# sees unlimited risk-adjusted room and levers up to its cap on an instrument it
# knows nothing about. Fail SAFE instead, substituting a deliberately high
# unknown-volatility prior — the same posture `optimizer._gaussian_cvar` already
# takes for a short-history book ("treated as risky, not risk-free"). 30% annual
# matches the conservative imputation DATA-1 established for a missing idio_vol,
# and is deliberately above that function's 20% single-name prior: here we have
# essentially NO observations, so the wider prior is the honest one.
UNKNOWN_ANNUAL_VOL: float = 0.30
UNKNOWN_DAILY_VAR: float = (UNKNOWN_ANNUAL_VOL ** 2) / TRADING_DAYS

# ``arch`` is numerically happiest when returns are expressed in percent. We fit
# in percent space (returns * 100) and convert the variance-scale parameter
# (omega) and the stored variance state back to decimal² space so every dict this
# module emits is in consistent decimal units. alpha/beta/gamma/nu are scale-free.
_ARCH_SCALE: float = 100.0


class _NonConvergenceError(RuntimeError):
    """Raised internally when a GARCH-family optimisation fails to converge."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_returns(returns) -> np.ndarray:
    """Coerce an iterable of returns to a finite 1-D float array."""
    arr = np.asarray(returns, dtype=float).ravel()
    return arr[np.isfinite(arr)]


# ── 7.1 / 7.2 — Fitters ────────────────────────────────────────────────────────

def fit_gjr_garch(returns: Returns) -> dict:
    """
    Fit GJR-GARCH(1,1,1) with Student-t errors.

        σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·1[ε_{t-1}<0] + β·σ²_{t-1}

    Returns ``{omega, alpha, gamma, beta, nu}`` (plus latest-state keys used by
    ``forecast_vol``). Minimum 30 samples. On non-convergence, falls back to a
    symmetric GARCH(1,1) with ``gamma=0.0`` and logs a WARNING.
    """
    r = _clean_returns(returns)
    if r.size < MIN_GJR_SAMPLES:
        raise ValueError(
            f"fit_gjr_garch requires >= {MIN_GJR_SAMPLES} samples, got {r.size}."
        )

    try:
        from arch import arch_model

        am = arch_model(
            r * _ARCH_SCALE, mean="Constant", vol="GARCH",
            p=1, o=1, q=1, dist="t", rescale=False,
        )
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
        if int(getattr(res, "convergence_flag", 0)) != 0:
            raise _NonConvergenceError(
                f"GJR-GARCH convergence_flag={res.convergence_flag}"
            )

        p = res.params
        omega = float(p["omega"]) / (_ARCH_SCALE ** 2)
        alpha = float(p["alpha[1]"])
        gamma = float(p["gamma[1]"])
        beta = float(p["beta[1]"])
        nu = float(p["nu"])
        return {
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "nu": nu,
            "cond_var_last": float(res.conditional_volatility[-1] ** 2) / (_ARCH_SCALE ** 2),
            "resid_last": float(res.resid[-1]) / _ARCH_SCALE,
            "method": "gjr",
            "converged": True,
            "persistence": alpha + beta + gamma / 2.0,
        }

    except Exception as exc:  # noqa: BLE001 — intentional: any failure → safe fallback
        logger.warning(
            "GJR-GARCH fit failed or did not converge (%s); "
            "falling back to symmetric GARCH(1,1).", exc,
        )
        return _fit_garch_fallback(r)


def _fit_garch_fallback(r: np.ndarray) -> dict:
    """Symmetric GARCH(1,1) fallback (``gamma`` forced to 0.0)."""
    try:
        from arch import arch_model

        am = arch_model(
            r * _ARCH_SCALE, mean="Constant", vol="GARCH",
            p=1, o=0, q=1, dist="t", rescale=False,
        )
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
        if int(getattr(res, "convergence_flag", 0)) != 0:
            raise _NonConvergenceError(
                f"GARCH(1,1) convergence_flag={res.convergence_flag}"
            )

        p = res.params
        alpha = float(p["alpha[1]"])
        beta = float(p["beta[1]"])
        return {
            "omega": float(p["omega"]) / (_ARCH_SCALE ** 2),
            "alpha": alpha,
            "gamma": 0.0,
            "beta": beta,
            "nu": float(p.get("nu", float("inf"))),
            "cond_var_last": float(res.conditional_volatility[-1] ** 2) / (_ARCH_SCALE ** 2),
            "resid_last": float(res.resid[-1]) / _ARCH_SCALE,
            "method": "garch_fallback",
            "converged": True,
            "persistence": alpha + beta,
        }

    except Exception as exc:  # noqa: BLE001 — last-resort: never crash the vol path
        logger.warning(
            "GARCH(1,1) fallback also failed (%s); using rolling-variance params.",
            exc,
        )
        var_daily = float(np.var(r, ddof=1)) if r.size >= 2 else UNKNOWN_DAILY_VAR
        return _rolling_std_params(var_daily)


def _rolling_std_params(var_daily: float) -> dict:
    """A GARCH-shaped params dict that simply carries a constant daily variance."""
    var_daily = max(float(var_daily), 0.0)
    return {
        "omega": var_daily,
        "alpha": 0.0,
        "gamma": 0.0,
        "beta": 0.0,
        "nu": float("inf"),
        "cond_var_last": var_daily,
        "resid_last": 0.0,
        "method": "rolling_std",
        "converged": False,
        "persistence": 0.0,
    }


def fit_har_rv(returns: Returns) -> dict:
    """
    Fit the HAR-RV model (Corsi 2009) by OLS:

        RV_{t+1} = c + β_d·RV^(d)_t + β_w·RV^(w)_t + β_m·RV^(m)_t + ε_{t+1}

    The realised-variance proxy is the squared daily return (no intraday or
    high/low data is available from a plain return series). Returns
    ``{c, beta_d, beta_w, beta_m, r_squared}`` plus the latest RV components and
    the next-day RV forecast used by ``forecast_vol``. Minimum 60 samples.
    """
    r = _clean_returns(returns)
    if r.size < MIN_HAR_SAMPLES:
        raise ValueError(
            f"fit_har_rv requires >= {MIN_HAR_SAMPLES} samples, got {r.size}."
        )

    import statsmodels.api as sm

    rv = pd.Series(r ** 2)
    rv_d = rv
    rv_w = rv.rolling(5).mean()
    rv_m = rv.rolling(22).mean()

    design = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m})
    design["target"] = rv.shift(-1)          # next-day realised variance
    design = design.dropna()

    if design.shape[0] < 5:
        raise ValueError(
            "fit_har_rv: insufficient non-NaN rows after constructing HAR lags."
        )

    # has_constant="add" forces the intercept column even when an RV regressor is
    # itself constant (constant-magnitude returns make the RV columns collinear).
    # Without it, add_constant skips the intercept, OLS returns 3 params, and the
    # 4-way unpack below raised "not enough values to unpack (expected 4, got 3)".
    x = sm.add_constant(design[["rv_d", "rv_w", "rv_m"]].to_numpy(), has_constant="add")
    y = design["target"].to_numpy()
    # A zero-variance target (constant-magnitude returns) makes statsmodels' R²
    # (1 − ssr/tss with tss=0) divide by zero. Extract params + R² inside the guard
    # and sanitise R² to 0.0 for the degenerate case; vol_ratio_current handles the
    # rest. (model.rsquared is lazy, so it must be read inside the errstate.)
    with np.errstate(divide="ignore", invalid="ignore"):
        model = sm.OLS(y, x).fit()
        params = [float(v) for v in model.params]
        r_squared = float(model.rsquared)
    if not math.isfinite(r_squared):
        r_squared = 0.0
    c, beta_d, beta_w, beta_m = params

    rv_d_last = float(rv_d.iloc[-1])
    rv_w_last = float(rv_w.iloc[-1])
    rv_m_last = float(rv_m.iloc[-1])
    rv_forecast = max(
        c + beta_d * rv_d_last + beta_w * rv_w_last + beta_m * rv_m_last, 0.0
    )

    return {
        "c": c,
        "beta_d": beta_d,
        "beta_w": beta_w,
        "beta_m": beta_m,
        "r_squared": r_squared,
        "rv_d": rv_d_last,
        "rv_w": rv_w_last,
        "rv_m": rv_m_last,
        "rv_forecast": rv_forecast,
    }


# ── 7.3 — Ensemble forecast ─────────────────────────────────────────────────────

def _garch_variance_forecast(gjr_params: dict, horizon: int = 1) -> float:
    """Daily-variance forecast (decimal²) from GJR-GARCH params, horizon-aware.

    Uses the stored last conditional variance and residual to seed the one-step
    forecast, then mean-reverts toward the unconditional variance over the
    horizon (the closed-form average of the GARCH recursion).
    """
    if gjr_params.get("method") == "rolling_std":
        return max(float(gjr_params.get("omega", 0.0)), 0.0)

    omega = float(gjr_params["omega"])
    alpha = float(gjr_params["alpha"])
    beta = float(gjr_params["beta"])
    gamma = float(gjr_params["gamma"])

    persistence = min(max(alpha + beta + gamma / 2.0, 0.0), 0.999999)
    long_run = omega / (1.0 - persistence) if persistence < 1.0 else None

    var_last = float(gjr_params.get("cond_var_last", long_run if long_run else omega))
    resid_last = float(gjr_params.get("resid_last", 0.0))
    indicator = 1.0 if resid_last < 0.0 else 0.0

    sigma2_1 = omega + (alpha + gamma * indicator) * resid_last ** 2 + beta * var_last
    sigma2_1 = max(sigma2_1, 0.0)
    if long_run is None:
        long_run = sigma2_1

    h = max(int(horizon), 1)
    if h == 1:
        return sigma2_1

    # Average of E[σ²_{t+k}] = LR + p^(k-1)·(σ²_1 − LR) over k = 1..h.
    if abs(1.0 - persistence) < 1e-12:
        avg = sigma2_1
    else:
        geom = (1.0 - persistence ** h) / (h * (1.0 - persistence))
        avg = long_run + (sigma2_1 - long_run) * geom
    return max(avg, 0.0)


def forecast_vol(
    gjr_params: dict,
    har_params: dict | None = None,
    horizon: int = 1,
) -> float:
    """
    Annualised volatility forecast (decimal fraction).

    Switching:
      - ``har_params`` provided and ``horizon <= 5`` → ensemble
        ``σ = sqrt(0.60·RV_HAR + 0.40·σ²_GARCH)``.
      - ``horizon > 5`` → GJR-GARCH recursive forecast only.
      - rolling-std params → annualised rolling standard deviation.
    """
    if gjr_params is None:
        raise ValueError("forecast_vol requires gjr_params.")
    horizon = max(int(horizon), 1)

    if gjr_params.get("method") == "rolling_std":
        daily_var = max(float(gjr_params.get("omega", 0.0)), 0.0)
    elif har_params is not None and horizon <= 5:
        rv_har = max(float(har_params.get("rv_forecast", 0.0)), 0.0)
        garch_var = _garch_variance_forecast(gjr_params, horizon)
        daily_var = 0.60 * rv_har + 0.40 * garch_var
    else:
        daily_var = _garch_variance_forecast(gjr_params, horizon)

    daily_var = max(daily_var, 0.0)
    return float(math.sqrt(daily_var * TRADING_DAYS))


def fit(returns: Returns) -> dict:
    """
    Fit the volatility ensemble, selecting the method by history length.

    PRESERVE EXISTING SIGNATURE. Returns ``{gjr_params, har_params, method}``
    where ``method`` ∈ {"ensemble", "gjr_only", "rolling_std"}:
      - ``>= 60d`` history → ensemble (GJR-GARCH + HAR-RV)
      - ``30–59d``        → GJR-GARCH only
      - ``< 30d``         → rolling standard-deviation fallback
    """
    r = _clean_returns(returns)
    n = r.size

    if n >= MIN_ENSEMBLE_SAMPLES:
        return {
            "gjr_params": fit_gjr_garch(r),
            "har_params": fit_har_rv(r),
            "method": "ensemble",
        }
    if n >= MIN_GJR_SAMPLES:
        return {
            "gjr_params": fit_gjr_garch(r),
            "har_params": None,
            "method": "gjr_only",
        }

    var_daily = float(np.var(r, ddof=1)) if n >= 2 else UNKNOWN_DAILY_VAR
    return {
        "gjr_params": _rolling_std_params(var_daily),
        "har_params": None,
        "method": "rolling_std",
    }


# ── 7.4 — Vol ratio helper ──────────────────────────────────────────────────────

def vol_ratio_current(
    portfolio_returns: Returns,
    gjr_params: dict | None = None,
    har_params: dict | None = None,
) -> float:
    """
    Ratio of the 5-day-ahead volatility forecast to the 60-day realised
    baseline. Used by the crisis manager (vol-explosion detector) and the
    optimizer (covariance scaling).

    Returns 1.0 when there is insufficient history (< 5 observations) or when the
    baseline is degenerate. Never returns a negative value.
    """
    r = _clean_returns(portfolio_returns)
    if r.size < 5:
        return 1.0

    baseline = r[-60:]
    if baseline.size < 2:
        return 1.0
    baseline_vol = float(np.std(baseline, ddof=1)) * ANNUALISATION
    if baseline_vol <= 1e-12:
        return 1.0

    try:
        if gjr_params is None:
            fitted = fit(r)
            gjr_params = fitted["gjr_params"]
            har_params = fitted["har_params"]
        vol_5d = forecast_vol(gjr_params, har_params, horizon=5)
    except Exception as exc:  # noqa: BLE001 — a degenerate fit must degrade, not crash
        logger.warning("vol_ratio_current: vol fit/forecast failed (%s); returning 1.0.", exc)
        return 1.0

    ratio = vol_5d / baseline_vol
    if not math.isfinite(ratio) or ratio < 0.0:
        return 1.0
    return float(ratio)


# ── 7.5 — RMT covariance denoising ──────────────────────────────────────────────

def _project_psd(cov: np.ndarray, rel_floor: float = 1e-10) -> np.ndarray:
    """Nearest-PSD projection (Item 3): clip eigenvalues to a small positive floor
    so the result is positive *definite* — keeping the mean-variance QP strictly
    convex and the CVaR LP well-posed. The floor is relative to the largest
    eigenvalue; symmetry is enforced first. A no-op when the input is already PD,
    so a valid PSD covariance is returned bit-for-bit unchanged.
    """
    sym = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(sym)
    lam_max = float(vals.max()) if vals.size else 0.0
    floor = rel_floor * lam_max if lam_max > 0.0 else 0.0
    if float(vals.min()) >= floor:
        return sym
    vals = np.clip(vals, floor, None)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def rmt_denoise_cov(cov: np.ndarray, T: int) -> np.ndarray:
    """
    Marchenko-Pastur covariance denoising.

        q = n / T,   σ² = trace(cov) / n,   λ_max = σ²·(1 + √q)²

    Eigenvalues at or below ``λ_max`` are treated as noise and replaced with
    their mean (preserving their summed contribution, hence the trace).

    Fallbacks: if ``T < 2n`` return the original covariance and log a WARNING;
    if ``n == 1`` return it unchanged.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("rmt_denoise_cov: cov must be a square 2-D matrix.")

    n = c.shape[0]
    if n == 1:
        return c.copy()
    if T < 2 * n:
        logger.warning(
            "rmt_denoise_cov: T=%d < 2n=%d — too few observations to denoise; "
            "returning original covariance.", T, 2 * n,
        )
        return _project_psd(c.copy())

    c_sym = 0.5 * (c + c.T)
    eigvals, eigvecs = np.linalg.eigh(c_sym)

    q = n / float(T)
    sigma2 = float(np.trace(c_sym)) / n
    lambda_max = sigma2 * (1.0 + math.sqrt(q)) ** 2

    new_eigvals = eigvals.copy()
    noise_mask = eigvals <= lambda_max
    if noise_mask.any():
        new_eigvals[noise_mask] = float(eigvals[noise_mask].mean())

    denoised = (eigvecs * new_eigvals) @ eigvecs.T
    denoised = 0.5 * (denoised + denoised.T)

    # Rescale to preserve the original trace (guards against float drift).
    trace_orig = float(np.trace(c_sym))
    trace_new = float(np.trace(denoised))
    if trace_new > 1e-18:
        denoised *= trace_orig / trace_new

    # Guarantee a valid (PSD) covariance for the optimiser (Item 3). Applied AFTER
    # the trace-rescale — flooring then rescaling could re-introduce negativity.
    return _project_psd(denoised)
