"""
TradingEngineResearch — Portfolio Optimizer
===============================
Black-Litterman portfolio construction with exact CVaR, a regime-aware
covariance blend, and source-gated views.

Pipeline (STEP 9):
  1. Covariance — three-way blend of Ledoit-Wolf, a stress (worst-5%) sample
     covariance, and an EWMA covariance, then Marchenko-Pastur denoised.
  2. Prior — CAPM reverse-optimised equilibrium returns π = λ·Σ·w_mkt.
  3. Views — ML / TradingEngineResearch / insider expected returns, each gated by a
     `ViewSourceTracker` (during warm-up, or rolling Sharpe >= its floor —
     both configurable, defaults 20 obs / -0.30; see OPT-4 on the class) and,
     for ML, decayed by realised-vol stress. Combined into the Black-Litterman
     posterior with a regime-aware tau (crisis override min(tau, 0.02)).
  4. Allocation — long-only mean-variance under per-asset, sector, budget, ADV,
     CVaR, and vol-target constraints; sub-threshold weights are zeroed and the
     book is vol-scaled. NOTE: the scaler levers UP as well as down, bounded by
     `max_gross_leverage` (which the shipped config sets to 2.0, NOT 1.0) — this
     header used to claim "never levered up", which was false for every default
     run. `max_lever_up_step` optionally ramps that rise (OPT-1).

CVaR is the Rockafellar-Uryasev (2000) exact LP (`scipy.optimize.linprog` HiGHS),
with a Gaussian fallback when history is short, plus a Cornish-Fisher modified-ES
diagnostic for fat-tailed regimes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import kurtosis, norm, skew
from sklearn.covariance import LedoitWolf

from strategies.volatility_model import rmt_denoise_cov, vol_ratio_current

logger = logging.getLogger(__name__)

__all__ = [
    "capm_equilibrium_returns",
    "black_litterman_posterior",
    "portfolio_cvar",
    "portfolio_cvar_exact",
    "compute_portfolio_cvar_cf",
    "ledoit_wolf_cov",
    "estimate_covariance_nonlinear",
    "impact_cost",
    "capacity_penalty",
    "exposure_penalty",
    "ViewSourceTracker",
    "get_view_tracker",
    "reset_view_tracker",
    "optimise",
    "optimise_portfolio",
]

# ── Constants (master prompt Part 14.6) ─────────────────────────────────────────

_MAX_WEIGHT = {"normal": 0.03, "crisis": 0.015}
_MIN_WEIGHT = 0.0025                 # weights below this are zeroed
_SECTOR_CAP = 0.20
_CVAR_LIMIT = {"normal": 0.05, "crisis": 0.025}
_TARGET_VOL = {"normal": 0.10, "crisis": 0.05}
_ADV_PARTICIPATION = {"normal": 0.05, "stressed": 0.02}
_TAU_BY_REGIME = {"trending": 0.05, "mean_reverting": 0.10, "high_vol": 0.02}
_RISK_AVERSION = 2.5
_LAMBDA_TURN = 0.10
_TRADING_DAYS = 252
_ML_CONFIDENCE_FLOOR = 0.30


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


# ── 14.7 CAPM equilibrium prior ─────────────────────────────────────────────────

def capm_equilibrium_returns(
    sigma: np.ndarray,
    market_weights: Optional[np.ndarray] = None,
    lambda_: float = _RISK_AVERSION,
) -> np.ndarray:
    """Reverse-optimised equilibrium excess returns ``π = λ·Σ·w_mkt``."""
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)  # empty universe → no equilibrium (avoid 1/0)
    if market_weights is None:
        market_weights = np.full(n, 1.0 / n)
    market_weights = np.asarray(market_weights, dtype=float)
    return lambda_ * sigma @ market_weights


# ── 14.7 Black-Litterman posterior ──────────────────────────────────────────────

def black_litterman_posterior(
    sigma: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: Optional[np.ndarray] = None,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Black-Litterman posterior mean:

        μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]

    When ``omega`` is None it defaults to ``diag(P·τΣ·Pᵀ)`` (Idzorek).
    """
    sigma = np.asarray(sigma, dtype=float)
    pi = np.asarray(pi, dtype=float).ravel()
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).ravel()

    if P.size == 0 or Q.size == 0:
        return pi

    tau_sigma = tau * sigma
    if omega is None:
        omega = np.diag(np.diag(P @ tau_sigma @ P.T))
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    omega = omega + np.eye(omega.shape[0]) * 1e-10   # guard singularity

    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)

    a = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    return np.linalg.solve(a, b) if np.linalg.cond(a) < 1e12 else np.linalg.pinv(a) @ b


# ── 14.1 Exact CVaR (Rockafellar-Uryasev 2000) ──────────────────────────────────

def _gaussian_cvar(
    weights: np.ndarray, returns_matrix: np.ndarray, cov: Optional[np.ndarray] = None
) -> float:
    """Gaussian CVaR approximation: ``1.65 · daily portfolio σ``.

    Never returns a silent ``0.0`` when a covariance is available. With
    insufficient history (``T < 2``) it falls back to ``cov`` when supplied, else
    a conservative 20%-annual-vol single-name prior — a freshly-started book is
    treated as risky, not risk-free (a zero would be maximally optimistic exactly
    when the system knows least; cf. docs/ARCHITECTURE.md golden rule 2, fail-closed).
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns_matrix, dtype=float)
    if cov is None:
        if r.ndim == 2 and r.shape[0] >= 2:
            cov = np.cov(r, rowvar=False)
        else:
            cov = np.eye(w.size) * (0.20 ** 2 / _TRADING_DAYS)
    cov_2d = np.atleast_2d(np.asarray(cov, dtype=float))
    daily_vol = float(np.sqrt(max(w @ cov_2d @ w, 0.0)))
    return 1.65 * daily_vol


def portfolio_cvar_exact(
    weights: np.ndarray, returns_matrix: np.ndarray, confidence: float = 0.95
) -> float:
    """
    Exact CVaR via the Rockafellar-Uryasev LP for *fixed* weights.

    Falls back to the Gaussian approximation when ``T < 30`` or the LP fails.
    Returned as a positive loss magnitude.
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2:
        return 0.0
    T = r.shape[0]
    if T < 30:
        return _gaussian_cvar(w, r)

    port_ret = r @ w                       # per-period portfolio return
    coef = 1.0 / ((1.0 - confidence) * T)

    # Variables: x = [zeta, u_1..u_T].  minimise zeta + coef·Σ u_t
    c = np.concatenate([[1.0], np.full(T, coef)])
    # u_t >= -port_ret_t - zeta  ⇒  -zeta - u_t <= port_ret_t
    a_ub = np.column_stack([np.full(T, -1.0), -np.eye(T)])
    b_ub = port_ret
    bounds = [(None, None)] + [(0.0, None)] * T

    try:
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success:
            return float(res.fun)
        logger.warning("CVaR LP did not converge (%s); using Gaussian fallback.", res.message)
    except Exception as exc:  # noqa: BLE001 — numerical safety
        logger.warning("CVaR LP raised (%s); using Gaussian fallback.", exc)
    return _gaussian_cvar(w, r)


def portfolio_cvar(
    weights: np.ndarray, returns_matrix: np.ndarray, confidence: float = 0.95
) -> float:
    """CVaR entry point — exact LP when ``T >= 30``, Gaussian fallback otherwise."""
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2 or r.shape[0] < 30:
        return _gaussian_cvar(np.asarray(weights, dtype=float).ravel(), r)
    return portfolio_cvar_exact(weights, returns_matrix, confidence)


def _enforce_cvar_limit(
    weights: np.ndarray,
    returns_matrix: Optional[np.ndarray],
    sigma: Optional[np.ndarray],
    limit: float,
    max_iter: int = 6,
) -> tuple[np.ndarray, float, bool]:
    """Enforce ``CVaR(weights) <= limit`` by de-levering toward cash (Item 1).

    CVaR is positively homogeneous of degree 1 in gross exposure, so a single
    rescale by ``limit / cvar`` is exact; the short loop only guards LP / numeric
    noise. The book is only ever scaled DOWN (never levered up) — the residual
    goes to cash, mirroring the vol-target's "never lever up" semantics. When
    ``returns_matrix`` is None the conservative prior covariance ``sigma`` is used
    (so the no-history path is enforced too). Returns ``(weights, cvar, binding)``.
    """
    w: np.ndarray = np.asarray(weights, dtype=float).ravel()

    def _cvar(ww: np.ndarray) -> float:
        if returns_matrix is not None:
            return portfolio_cvar(ww, returns_matrix)
        return _gaussian_cvar(ww, np.empty((0, w.size)), cov=sigma)

    cvar = _cvar(w)
    was_binding = False
    for _ in range(max_iter):
        if cvar <= limit or cvar <= 0.0 or w.sum() <= 1e-12:
            break
        was_binding = True
        w = w * (limit / cvar)
        cvar = _cvar(w)
    return w, cvar, was_binding


# ── P5 Cornish-Fisher modified CVaR ──────────────────────────────────────────────

def compute_portfolio_cvar_cf(
    weights: np.ndarray,
    returns_history: pd.DataFrame | np.ndarray,
    confidence: float = 0.95,
    window: int = 252,
    method: str = "cornish_fisher",
) -> dict:
    """
    Cornish-Fisher (Boudt et al. 2008) modified CVaR for fat-tailed returns.

    Falls back to the historical empirical CVaR when moment estimates are extreme
    (``|γ₁| > 3`` or ``γ₂ > 20``). Returns a diagnostics dict; ``cvar`` is a
    positive loss magnitude.
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns_history, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns_history must be 2-D (T × n).")

    port = (r @ w)[-window:]
    mu_p = float(np.mean(port))
    sigma_p = float(np.std(port, ddof=1)) if port.size > 1 else 0.0
    g1 = float(skew(port)) if port.size > 2 else 0.0
    g2 = float(kurtosis(port, fisher=True)) if port.size > 3 else 0.0

    z_a = float(norm.ppf(confidence))
    one_minus = 1.0 - confidence

    if method == "cornish_fisher" and (abs(g1) > 3.0 or g2 > 20.0):
        logger.warning(
            "compute_portfolio_cvar_cf: extreme moments (skew=%.2f, kurt=%.2f); "
            "falling back to historical CVaR.", g1, g2,
        )
        method = "historical"

    if method == "historical":
        k = max(int(round(one_minus * port.size)), 1)
        worst = np.sort(port)[:k]
        cvar = float(-np.mean(worst))
        var = float(-np.sort(port)[k - 1])
        return {
            "cvar": cvar, "var": var, "portfolio_skew": g1,
            "portfolio_kurtosis": g2, "cf_quantile_adjustment": 0.0, "method": "historical",
        }

    if method == "gaussian":
        cvar = -mu_p + sigma_p * float(norm.pdf(z_a)) / one_minus
        var = -(mu_p + sigma_p * (-z_a))
        return {
            "cvar": float(cvar), "var": float(var), "portfolio_skew": g1,
            "portfolio_kurtosis": g2, "cf_quantile_adjustment": 0.0, "method": "gaussian",
        }

    # Cornish-Fisher adjusted quantile.
    z_cf = (
        z_a
        + (1.0 / 6.0) * (z_a ** 2 - 1.0) * g1
        + (1.0 / 24.0) * (z_a ** 3 - 3.0 * z_a) * g2
        - (1.0 / 36.0) * (2.0 * z_a ** 3 - 5.0 * z_a) * g1 ** 2
    )
    var_cf = mu_p + sigma_p * (-z_cf)
    cvar_cf = -mu_p + sigma_p * (float(norm.pdf(z_cf)) / one_minus) * (
        1.0
        + (1.0 / 6.0) * g1 * (2.0 * z_cf ** 2 - 1.0 - z_cf / z_a) * z_a
        + (1.0 / 24.0) * g2 * (3.0 * z_cf * z_a - z_cf ** 3 / z_a) * z_a
    )
    # P5 sanity floor (Item 8): the Cornish-Fisher expansion can under-estimate the
    # Gaussian tail for negative skew with low kurtosis. ES must never fall below the
    # Gaussian baseline — fat tails should make sizing MORE conservative, never less
    # (upgrade-spec lines 74/512). max() also subsumes the old abs() sign-guard.
    cvar_gaussian = -mu_p + sigma_p * float(norm.pdf(z_a)) / one_minus
    cvar_cf = max(float(cvar_cf), float(cvar_gaussian))
    return {
        "cvar": float(cvar_cf),
        "var": float(var_cf),
        "portfolio_skew": g1,
        "portfolio_kurtosis": g2,
        "cf_quantile_adjustment": float(z_cf - z_a),
        "method": "cornish_fisher",
    }


# ── 14.2 / P3 Covariance ─────────────────────────────────────────────────────────

def _ewma_cov(r: np.ndarray, span: int = 60) -> np.ndarray:
    df = pd.DataFrame(r)
    ewm_cov = df.ewm(span=span, min_periods=min(10, r.shape[0])).cov()
    try:
        return ewm_cov.loc[df.index[-1]].to_numpy()
    except Exception:  # noqa: BLE001
        return np.cov(r, rowvar=False)


def ledoit_wolf_cov(returns_matrix: np.ndarray, crisis_mode: bool = False) -> np.ndarray:
    """Three-way covariance blend (LW + stress + EWMA), then RMT-denoised."""
    r = np.asarray(returns_matrix, dtype=float)
    T, n = r.shape
    if n == 1:
        return np.array([[float(np.var(r, ddof=1)) if T > 1 else 0.0]])

    sigma_lw = LedoitWolf().fit(r).covariance_

    row_sums = r.sum(axis=1)
    threshold = np.percentile(row_sums, 5)
    worst = r[row_sums <= threshold]
    sigma_stress = np.cov(worst, rowvar=False) if worst.shape[0] >= 2 else sigma_lw

    sigma_ewma = _ewma_cov(r)

    if crisis_mode:
        blend = 0.30 * sigma_lw + 0.40 * sigma_stress + 0.30 * sigma_ewma
    else:
        blend = 0.50 * sigma_lw + 0.20 * sigma_stress + 0.30 * sigma_ewma

    return rmt_denoise_cov(blend, T)


def estimate_covariance_nonlinear(
    returns: pd.DataFrame | np.ndarray,
    frequency: str = "daily",
    min_periods: int = 252,
    fallback_to_linear: bool = True,
) -> dict:
    """
    Analytical nonlinear Ledoit-Wolf shrinkage (P3) when ``nlshrink`` is
    available, else linear Ledoit-Wolf. Returns the P3 diagnostics dict.
    """
    r = np.asarray(returns, dtype=float)
    T, p = r.shape
    concentration = p / max(T, 1)
    method_used = "linear_LW_fallback"
    cov = LedoitWolf().fit(r).covariance_

    if not (fallback_to_linear and concentration > 0.9):
        try:
            from nlshrink import nonlinear_shrinkage

            cov = np.asarray(nonlinear_shrinkage(r), dtype=float)
            method_used = "nonlinear_LW"
        except Exception as exc:  # noqa: BLE001 — optional dependency
            logger.info("nlshrink unavailable (%s); using linear Ledoit-Wolf.", exc)

    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(d, d)
    sample_eig = np.linalg.eigvalsh(np.cov(r, rowvar=False))
    shrunk_eig = np.linalg.eigvalsh(cov)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensities = np.where(sample_eig > 1e-18, shrunk_eig / sample_eig, 1.0)

    return {
        "cov_matrix": cov,
        "corr_matrix": corr,
        "shrinkage_intensities": intensities,
        "concentration_ratio": float(concentration),
        "method_used": method_used,
    }


# ── 14.5 Penalty terms ───────────────────────────────────────────────────────────

def impact_cost(weight_delta: np.ndarray, adv: np.ndarray, sigma: np.ndarray) -> float:
    """Square-root market-impact cost of a weight change (dimensionless)."""
    dw = np.abs(np.asarray(weight_delta, dtype=float))
    adv = np.asarray(adv, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    participation = dw / (adv + 1e-9)
    return float(np.sum(sigma * np.sqrt(np.clip(participation, 0.0, None))))


def capacity_penalty(weights: np.ndarray, adv: np.ndarray, capital: float) -> float:
    """Penalise notional that exceeds the per-name ADV participation cap."""
    w = np.asarray(weights, dtype=float)
    adv = np.asarray(adv, dtype=float)
    notional = np.abs(w) * capital
    cap_notional = _ADV_PARTICIPATION["normal"] * adv
    overage = np.clip(notional - cap_notional, 0.0, None) / max(capital, 1e-9)
    return float(np.sum(overage ** 2))


def exposure_penalty(weights: np.ndarray, sector_map: dict) -> float:
    """Penalise sector weight beyond the cap. ``sector_map`` maps index → sector."""
    w = np.asarray(weights, dtype=float)
    sectors: dict[str, float] = {}
    for i, weight in enumerate(w):
        sector = sector_map.get(i, "_unmapped")
        sectors[sector] = sectors.get(sector, 0.0) + weight
    return float(sum(max(sw - _SECTOR_CAP, 0.0) ** 2 for sw in sectors.values()))


# ── 14.4 View-source tracker ─────────────────────────────────────────────────────

class ViewSourceTracker:
    """Tracks per-source predictive Sharpe and gates inactive view sources.

    OPT-4 note — the gate is deliberately lenient, and both knobs are now
    settable so the strictness is a recorded decision rather than a constant
    buried here. **Defaults reproduce the shipped behaviour exactly**, so
    changing nothing changes nothing:

    * ``sharpe_floor`` (default ``-0.30``) — a source keeps full influence until
      its rolling raw Sharpe falls BELOW this. At the default, a source that
      loses money persistently but mildly (say a steady −0.29) is never gated.
      Raising it to ``0.0`` gates any source that is not at least break-even.
    * ``gate_during_warmup`` (default ``False``) — with the default, a brand-new
      source has FULL influence for its first ``warmup`` observations, i.e. an
      unproven source is trusted exactly while nothing is known about it. Set
      ``True`` to withhold a source until it has earned a measurable record.

    Both are risk-appetite choices, not correctness fixes: tightening either
    changes portfolio weights, so it is the operator's call. See
    ``docs/project-control/RISK_AND_DEFECT_REGISTER.md`` (OPT-4).
    """

    _WARMUP = 20
    _SHARPE_FLOOR = -0.30

    def __init__(
        self,
        sharpe_floor: Optional[float] = None,
        warmup: Optional[int] = None,
        gate_during_warmup: bool = False,
    ) -> None:
        self._prediction_log: dict[str, list[tuple[float, float]]] = {}
        self.sharpe_floor = self._SHARPE_FLOOR if sharpe_floor is None else float(sharpe_floor)
        self.warmup = self._WARMUP if warmup is None else int(warmup)
        self.gate_during_warmup = bool(gate_during_warmup)

    def record(self, source: str, predicted: float, actual: float) -> None:
        self._prediction_log.setdefault(source, []).append((float(predicted), float(actual)))

    def rolling_sharpe(self, source: str, lookback: int = 60) -> float:
        log = self._prediction_log.get(source, [])
        if len(log) < 2:
            return 0.0
        # PnL proxy: sign(prediction) · actual realised return.
        pnl = np.array([np.sign(p) * a for p, a in log[-lookback:]])
        if pnl.size < 2 or np.std(pnl) < 1e-12:
            return 0.0
        # Un-annualised Sharpe over the lookback (the -0.30 gate is a raw ratio).
        return float(np.mean(pnl) / np.std(pnl))

    def is_active(self, source: str) -> bool:
        log = self._prediction_log.get(source, [])
        if len(log) < self.warmup:
            # Default False = trust an unproven source. See the class docstring.
            return not self.gate_during_warmup
        return self.rolling_sharpe(source) >= self.sharpe_floor


_VIEW_TRACKER: Optional[ViewSourceTracker] = None


def get_view_tracker() -> ViewSourceTracker:
    """Process-wide ViewSourceTracker singleton."""
    global _VIEW_TRACKER
    if _VIEW_TRACKER is None:
        _VIEW_TRACKER = ViewSourceTracker()
    return _VIEW_TRACKER


def reset_view_tracker() -> None:
    global _VIEW_TRACKER
    _VIEW_TRACKER = None


# ── Allocation ───────────────────────────────────────────────────────────────────

def _as_dict(values: Optional[dict]) -> dict:
    """Normalise an optional per-symbol mapping to a dict.

    OPT-5: this was named ``_align(values, symbols)`` and took a ``symbols``
    argument it never used, so it read as if it restricted the mapping to the
    current universe when it does nothing of the kind. Renamed rather than
    "fixed" deliberately: actually filtering to ``symbols`` would silently drop
    view inputs and change portfolio weights, which is a financially material
    change for the operator to decide, not a tidy-up. Downstream lookups are
    keyed by symbol anyway, so extra keys are inert.
    """
    return values or {}


def _ml_view(raw) -> tuple[float, float]:
    """Extract (expected_return, confidence) from an ML prediction entry."""
    if isinstance(raw, (tuple, list)) and len(raw) >= 5:
        return float(raw[0]), float(raw[4])
    return float(raw), 1.0


def _solve_mean_variance(
    mu: np.ndarray, sigma: np.ndarray, max_weight: float,
    w_prev: np.ndarray, lambda_risk: float = _RISK_AVERSION,
    lambda_turn: float = _LAMBDA_TURN,
) -> np.ndarray:
    """Long-only mean-variance with budget and box constraints (SLSQP)."""
    n = len(mu)

    def objective(w: np.ndarray) -> float:
        return float(-(w @ mu) + 0.5 * lambda_risk * (w @ sigma @ w) + lambda_turn * np.sum((w - w_prev) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, max_weight)] * n
    w0 = np.full(n, 1.0 / n)

    try:
        res = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        w = res.x if res.success else w0
    except Exception as exc:  # noqa: BLE001
        logger.warning("mean-variance solve failed (%s); using equal weight.", exc)
        w = w0

    w = np.clip(w, 0.0, max_weight)
    total = w.sum()
    return w / total if total > 0 else w0


def optimise(mu: np.ndarray, cov: np.ndarray, capital_gbp: float) -> dict[str, float]:
    """Legacy mean-variance entry point — returns ``{asset_i: weight}``."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    max_w = max(_MAX_WEIGHT["normal"], 1.0 / n)
    w = _solve_mean_variance(mu, cov, max_w, np.zeros(n))
    return {f"asset_{i}": float(wi) for i, wi in enumerate(w)}


def optimise_portfolio(
    symbols: list[str],
    ml_predictions: Optional[dict] = None,
    signal_scores: Optional[dict] = None,
    engine_returns: Optional[dict] = None,
    insider_flows: Optional[dict] = None,
    capital_gbp: float = 1_000_000.0,
    returns_matrix: Optional[np.ndarray] = None,
    crisis_mode: bool = False,
    regime: str = "mean_reverting",
    w_prev: Optional[dict] = None,
    adv: Optional[dict] = None,
    sector_map: Optional[dict] = None,
    crisis_severity: Optional[float] = None,
    target_vol: Optional[float] = None,
    max_gross_leverage: float = 1.0,
    max_position_weight: Optional[float] = None,
    cvar_limit_override: Optional[float] = None,
    signal_tilt_strength: float = 5e-4,
    max_lever_up_step: Optional[float] = None,
) -> dict:
    """Full Black-Litterman + constrained optimisation; returns the 14.8 diagnostics.

    Risk-budget overrides (default to the conservative module constants, so existing
    callers are unchanged): ``target_vol`` (annualised vol target), ``max_gross_leverage``
    (the vol scaler may lever UP to this multiple to reach the target — 1.0 = never lever),
    ``max_position_weight`` (per-name cap, normal mode), ``cvar_limit_override``. These let
    the run-loop/backtester run the engine more aggressively while every constraint
    (CVaR, caps, crisis tightening, the STEP-10 risk gate) still applies.

    ``max_lever_up_step`` (OPT-1, **default None = today's behaviour, unchanged**)
    bounds how fast the book may lever UP. The vol-target scaler is procyclical: it
    reads TRAILING realised vol, so it reaches ``max_gross_leverage`` exactly when
    vol has been LOWEST — historically the run-up to a vol spike — and at a monthly
    rebalance cadence it cannot pre-empt the gap that follows. Setting e.g. ``0.25``
    caps each rebalance's gross exposure at 1.25x the previous book's gross, so
    leverage ramps in over several cycles instead of arriving in one step, while
    DE-levering stays instant and unbounded. This is a risk-appetite choice, not a
    correctness fix — it trades some upside capture for less gap exposure — so it is
    off unless the operator turns it on. See RISK_AND_DEFECT_REGISTER.md (OPT-1)."""
    n = len(symbols)
    if n == 0:
        # Empty universe → empty book, explicitly (fail-closed, never a 1/0 crash).
        return {"weights": {}, "binding_constraints": [], "turnover_estimate": 0.0,
                "expected_return": 0.0, "expected_risk": 0.0, "expected_cost_bps": 0.0,
                "cvar_95": 0.0, "capacity_flags": [], "view_sources_active": {}}
    mode = "crisis" if crisis_mode else "normal"

    # Graduated crisis tightening (Item 7): the CONTINUOUS crisis severity scales
    # the vol target and CVaR limit (upgrade-spec P4). With no severity supplied
    # this is a no-op (1.0, 1.0). The legacy binary ``crisis_mode`` still applies a
    # floor so the tightening is never looser than the historical crisis behaviour
    # — protection is monotone (a defensive cycle can only get tighter, not looser).
    from core.crisis_manager import crisis_scalars  # local import: avoids any cycle
    if crisis_severity is not None:
        vol_scalar, cvar_scalar = crisis_scalars(crisis_severity)
    else:
        vol_scalar, cvar_scalar = 1.0, 1.0
    if crisis_mode:
        vol_scalar = min(vol_scalar, _TARGET_VOL["crisis"] / _TARGET_VOL["normal"])
        cvar_scalar = min(cvar_scalar, _CVAR_LIMIT["crisis"] / _CVAR_LIMIT["normal"])
    cvar_base = cvar_limit_override if cvar_limit_override is not None else _CVAR_LIMIT["normal"]
    cvar_limit = cvar_base * cvar_scalar
    ml_predictions = _as_dict(ml_predictions)
    engine_returns = _as_dict(engine_returns)
    insider_flows = _as_dict(insider_flows)
    signal_scores = _as_dict(signal_scores)

    prev = np.array([(w_prev or {}).get(s, 0.0) for s in symbols], dtype=float)

    # 1. Covariance.
    if returns_matrix is not None and np.asarray(returns_matrix).ndim == 2 \
            and np.asarray(returns_matrix).shape[1] == n and np.asarray(returns_matrix).shape[0] >= 2:
        r = np.asarray(returns_matrix, dtype=float)
        sigma = ledoit_wolf_cov(r, crisis_mode)
        port_returns = r @ np.full(n, 1.0 / n)
    else:
        r = None
        sigma = np.eye(n) * (0.20 ** 2 / _TRADING_DAYS)
        port_returns = np.array([])

    # 2. Prior.
    pi = capm_equilibrium_returns(sigma)

    # 3. Views (gated by source health + ML vol-decay).
    tracker = get_view_tracker()
    active = {s: tracker.is_active(s) for s in ("ml", "engine", "insider")}
    for source, is_on in active.items():
        if not is_on:
            logger.warning(
                "RISK_EVENT VIEW_GATED: source '%s' inactive (rolling Sharpe < %s).",
                source, tracker.sharpe_floor,
            )

    vol_ratio = vol_ratio_current(port_returns.tolist()) if port_returns.size >= 5 else 1.0
    ml_decay = _clip(1.0 - 0.20 * vol_ratio, 0.30, 1.0)

    view_rows: list[np.ndarray] = []
    view_q: list[float] = []
    for i, symbol in enumerate(symbols):
        contribs: list[float] = []
        if active["ml"] and symbol in ml_predictions:
            mu_i, conf_i = _ml_view(ml_predictions[symbol])
            if conf_i * ml_decay > _ML_CONFIDENCE_FLOOR:
                contribs.append(mu_i)
        if active["engine"] and symbol in engine_returns:
            contribs.append(float(engine_returns[symbol]))
        if active["insider"] and symbol in insider_flows:
            contribs.append(float(np.tanh(float(insider_flows[symbol])) * 0.002))
        if contribs:
            row = np.zeros(n)
            row[i] = 1.0
            view_rows.append(row)
            view_q.append(float(np.mean(contribs)))

    tau = _TAU_BY_REGIME.get(regime, 0.10)
    if crisis_mode:
        tau = min(tau, 0.02)

    if view_rows:
        mu_bl = black_litterman_posterior(sigma, pi, np.vstack(view_rows), np.array(view_q), tau=tau)
    else:
        mu_bl = pi

    # Mild signal-score tilt.
    tilt = np.array([float(signal_scores.get(s, 0.0)) for s in symbols], dtype=float)
    mu_opt = mu_bl + float(signal_tilt_strength) * np.clip(tilt, -1.0, 1.0)

    # 4. Allocation under constraints.
    binding: list[str] = []
    base_cap = _MAX_WEIGHT[mode]
    if mode == "normal" and max_position_weight is not None:
        base_cap = float(max_position_weight)   # aggressive concentration (normal only; crisis stays tight)
    eff_cap = base_cap
    if n * base_cap < 1.0:                       # tiny universe — relax for feasibility
        eff_cap = max(base_cap, 1.0 / n + 1e-9)
        binding.append("per_asset_cap_relaxed_small_universe")

    w = _solve_mean_variance(mu_opt, sigma, eff_cap, prev)

    # Zero sub-threshold weights and renormalise.
    w[w < _MIN_WEIGHT] = 0.0
    if w.sum() > 0:
        w = w / w.sum()

    if np.any(np.isclose(w, eff_cap, atol=1e-4)):
        binding.append("per_asset_cap")

    # Vol target. The scaler de-levers to the target and (when max_gross_leverage > 1)
    # may lever UP to the target, bounded by max_gross_leverage. The CVaR limit + caps
    # below still constrain the resulting book.
    port_var = float(w @ sigma @ w)
    port_vol_annual = float(np.sqrt(max(port_var, 0.0) * _TRADING_DAYS))
    target_vol_base = target_vol if target_vol is not None else _TARGET_VOL["normal"]
    target_vol_eff = target_vol_base * vol_scalar
    max_lev = max(1.0, float(max_gross_leverage))
    vol_scale = min(target_vol_eff / port_vol_annual, max_lev) if port_vol_annual > 1e-9 else 1.0

    # OPT-1 leverage ramp (opt-in; None = unchanged). Bound only the UPWARD move,
    # relative to the gross exposure we already held: de-levering must stay
    # instant and unbounded, or the "fix" would slow the book down exactly when
    # it needs to shrink. With no previous book there is nothing to ramp from, so
    # the first levered step is allowed up to 1.0 (unlevered) and grows thereafter.
    if max_lever_up_step is not None and vol_scale > 1.0:
        step = max(0.0, float(max_lever_up_step))
        prev_gross = float(np.sum(np.abs(prev)))
        base_gross = float(np.sum(np.abs(w)))
        if base_gross > 1e-12:
            allowed_gross = (prev_gross if prev_gross > 1e-12 else 1.0) * (1.0 + step)
            ramp_cap = allowed_gross / base_gross
            if ramp_cap < vol_scale:
                vol_scale = max(1.0, ramp_cap)
                binding.append("leverage_ramp")

    if vol_scale < 1.0:
        binding.append("vol_target")
    elif vol_scale > 1.0:
        binding.append("vol_target_levered")
    w = w * vol_scale

    # CVaR limit — ENFORCED, not flag-only (Item 1): de-lever toward cash until
    # CVaR <= limit. Usually a no-op because the vol target already bounds CVaR
    # well below the limit; this is the hard-constraint safety net (spec 14.6,
    # "iteratively enforced before final allocation"). The no-history path uses the
    # real prior covariance (sigma), never a fabricated 1-row matrix (Item 2).
    w, cvar_95, cvar_binding = _enforce_cvar_limit(w, r, sigma, cvar_limit)
    if cvar_binding:
        binding.append("cvar_limit")

    # Capacity flags.
    capacity_flags: list[str] = []
    if adv is not None:
        for i, symbol in enumerate(symbols):
            adv_i = float(adv.get(symbol, np.inf))
            cap_part = _ADV_PARTICIPATION["stressed" if crisis_mode else "normal"]
            if w[i] * capital_gbp > cap_part * adv_i:
                capacity_flags.append(symbol)

    # Sector binding.
    if sector_map is not None:
        idx_sector_map = {i: sector_map.get(s, "_unmapped") for i, s in enumerate(symbols)}
        if exposure_penalty(w, idx_sector_map) > 0.0:
            binding.append("sector_cap")

    weights = {symbol: float(w[i]) for i, symbol in enumerate(symbols)}
    turnover = float(np.sum(np.abs(w - prev)))
    expected_return = float(w @ mu_bl)
    expected_risk = float(np.sqrt(max(w @ sigma @ w, 0.0) * _TRADING_DAYS))
    expected_cost_bps = float(turnover * 10.0)

    return {
        "weights": weights,
        "expected_return": expected_return,
        "expected_risk": expected_risk,
        "expected_cost_bps": expected_cost_bps,
        "cvar_95": float(cvar_95),
        "binding_constraints": binding,
        "turnover_estimate": turnover,
        "capacity_flags": capacity_flags,
        "view_sources_active": active,
    }
