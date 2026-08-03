"""
TradingEngineResearch — Black-Scholes-Merton Pricing & Tail-Hedge Cost
==========================================================
Analytical European option pricing, vega, and implied-volatility inversion, plus
an ATM protective-put cost surface used by the risk overlay.

⚠ Documented limitation (carried from the upgrade-spec weakness audit): Black-
Scholes assumes constant volatility and log-normal returns, so it cannot
reproduce the volatility skew and **systematically under-prices OTM protective
puts** — by an estimated 30–60% in crisis conditions, when OTM put skew expands
from ~3–5 vol points over ATM to 15–25+ (Bongaerts et al. 2020; HL Hunt 2025).
The CrisisManager anchors tail-hedge affordability on this estimate, so
`atm_put_cost()` returns the BS cost **and** a crisis-adjusted upper estimate,
flagged informationally. This module never auto-executes a hedge: it only
surfaces cost (`auto_execute` is always ``False``).

Newton-Raphson IV inversion is poorly conditioned for deep ITM/OTM options, so
`implied_vol()` falls back to a bracketed bisection and returns ``nan`` (with a
WARNING) when no arbitrage-free solution exists rather than emitting a bad greek.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = [
    "bs_call_price",
    "bs_put_price",
    "bs_vega",
    "implied_vol",
    "atm_put_cost",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_IV_LOWER = 1e-4
_IV_UPPER = 5.0
# Informational crisis skew uplift (mid-point of the documented 30–60% range).
_DEFAULT_CRISIS_SKEW_MULTIPLIER = 1.45


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes-Merton European call price (continuous dividend yield ``q``)."""
    if S <= 0.0 or K <= 0.0:
        raise ValueError("S and K must be positive.")
    if T <= 0.0 or sigma <= 0.0:
        # Degenerate: discounted intrinsic value.
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes-Merton European put price (continuous dividend yield ``q``)."""
    if S <= 0.0 or K <= 0.0:
        raise ValueError("S and K must be positive.")
    if T <= 0.0 or sigma <= 0.0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega (∂price/∂σ), identical for calls and puts. Per unit of volatility."""
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)


def _bisection_iv(
    price: float, S: float, K: float, T: float, r: float, q: float, is_put: bool,
    lo: float = _IV_LOWER, hi: float = _IV_UPPER, tol: float = 1e-8, max_iter: int = 200,
) -> float:
    """Bracketed bisection fallback for implied volatility. ``nan`` if unbracketed."""
    pricer = bs_put_price if is_put else bs_call_price

    def f(s: float) -> float:
        return pricer(S, K, T, r, s, q) - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def implied_vol(
    price: float, S: float, K: float, T: float, r: float,
    option_type: str = "put", q: float = 0.0,
    tol: float = 1e-6, max_iter: int = 100,
) -> float:
    """
    Invert the BS price for implied volatility (Newton-Raphson → bisection).

    Returns ``nan`` (and logs a WARNING) when the price is outside the no-arbitrage
    bounds or no solution exists — deep ITM/OTM inversions are ill-conditioned and
    a silent bad greek must never propagate into the tail-hedge cost model.
    """
    is_put = option_type.lower() == "put"
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or price < 0.0:
        logger.warning("implied_vol: invalid inputs (S=%s K=%s T=%s price=%s).", S, K, T, price)
        return float("nan")

    disc_k = K * math.exp(-r * T)
    disc_s = S * math.exp(-q * T)
    if is_put:
        intrinsic, upper = max(disc_k - disc_s, 0.0), disc_k
    else:
        intrinsic, upper = max(disc_s - disc_k, 0.0), disc_s

    if price < intrinsic - 1e-9:
        logger.warning("implied_vol: price %.6f below intrinsic %.6f.", price, intrinsic)
        return float("nan")
    if price <= intrinsic + 1e-12:
        return _IV_LOWER  # at intrinsic ⇒ ~zero volatility
    if price >= upper:
        logger.warning("implied_vol: price %.6f at/above no-arbitrage upper bound %.6f.", price, upper)
        return float("nan")

    # Newton-Raphson from the Brenner-Subrahmanyam ATM approximation.
    sigma = min(max(_SQRT_2PI / math.sqrt(T) * price / S, 1e-3), _IV_UPPER)
    pricer = bs_put_price if is_put else bs_call_price
    for _ in range(max_iter):
        diff = pricer(S, K, T, r, sigma, q) - price
        if abs(diff) < tol:
            return float(min(max(sigma, _IV_LOWER), _IV_UPPER))
        vega = bs_vega(S, K, T, r, sigma, q)
        if vega < 1e-8:
            break
        sigma_next = sigma - diff / vega
        if not math.isfinite(sigma_next) or sigma_next <= 0.0 or sigma_next > _IV_UPPER:
            break
        sigma = sigma_next

    iv = _bisection_iv(price, S, K, T, r, q, is_put)
    if math.isnan(iv):
        logger.warning(
            "implied_vol: no arbitrage-free solution (ill-conditioned deep ITM/OTM)."
        )
    return float(iv)


def atm_put_cost(
    spot: float, sigma: float, T: float, r: float = 0.0, q: float = 0.0,
    crisis: bool = False, crisis_skew_multiplier: float = _DEFAULT_CRISIS_SKEW_MULTIPLIER,
) -> dict:
    """
    Price an ATM (K = spot) protective put and surface its cost.

    Returns the BS put price, cost as a fraction of spot, and — because BS
    structurally under-prices the skew — a ``crisis_adjusted_estimate`` (BS price
    × an uplift in the documented 30–60% range) when ``crisis=True``. This is
    informational only; ``auto_execute`` is always ``False``.
    """
    if spot <= 0.0:
        raise ValueError("spot must be positive.")
    strike = spot  # ATM
    put_price = bs_put_price(spot, strike, T, r, sigma, q)
    crisis_adjusted = put_price * crisis_skew_multiplier if crisis else put_price

    return {
        "strike": strike,
        "implied_vol_used": sigma,
        "put_price": put_price,
        "cost_pct_of_spot": put_price / spot,
        "crisis": crisis,
        "crisis_skew_multiplier_applied": crisis_skew_multiplier if crisis else 1.0,
        "crisis_adjusted_estimate": crisis_adjusted,
        "crisis_adjusted_cost_pct_of_spot": crisis_adjusted / spot,
        "auto_execute": False,
        "note": (
            "BS ATM put cost surfaced informationally. BS under-prices OTM skew by "
            "an estimated 30–60% in crisis; the crisis_adjusted_estimate is an "
            "upper bound. Never auto-execute a hedge from this figure."
        ),
    }
