"""
TradingEngineResearch — Backtest performance metrics
=========================================
Pure, side-effect-free performance functions over a series of *period* returns
(decimal, e.g. 0.01 = +1%). Every function accepts a pandas Series, numpy array,
or sequence of floats, and returns a plain ``float``. Degenerate inputs (empty,
single-point, or zero-variance) return ``0.0`` rather than NaN/inf so downstream
summaries stay finite.

``periods_per_year`` annualises (252 trading days by default; pass 12 for monthly,
52 for weekly, etc.).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "hit_rate",
    "summarize",
]


def _arr(returns: object) -> np.ndarray:
    return np.asarray(returns, dtype=float).ravel()


def ann_return(returns: object, periods_per_year: int = 252) -> float:
    """Geometric annualised return: ``prod(1 + r) ** (ppy / n) - 1``."""
    r = _arr(returns)
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    if growth <= 0.0:
        return -1.0                       # total wipe-out (or worse) → -100%
    years = r.size / float(periods_per_year)
    if years <= 0.0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def ann_vol(returns: object, periods_per_year: int = 252) -> float:
    """Annualised volatility: sample std (ddof=1) scaled by ``sqrt(ppy)``."""
    r = _arr(returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: object, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio. Zero-variance → 0.0."""
    r = _arr(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    sd = float(np.std(excess, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino(returns: object, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio using target-downside deviation against ``risk_free``.

    Downside deviation = ``sqrt(mean(min(r - target, 0) ** 2))`` over all periods.
    """
    r = _arr(returns)
    if r.size < 2:
        return 0.0
    target = risk_free / periods_per_year
    excess = r - target
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= 0.0:
        return 0.0
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: object) -> float:
    """Maximum peak-to-trough drawdown as a positive magnitude (0.25 = -25%)."""
    r = _arr(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(-np.min(drawdowns))


def calmar(returns: object, periods_per_year: int = 252) -> float:
    """Calmar ratio: annualised return / max drawdown. No drawdown → 0.0."""
    mdd = max_drawdown(returns)
    if mdd <= 0.0:
        return 0.0
    return float(ann_return(returns, periods_per_year) / mdd)


def hit_rate(returns: object) -> float:
    """Fraction of strictly-positive periods."""
    r = _arr(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0.0))


def summarize(returns: object, periods_per_year: int = 252) -> dict[str, float]:
    """All metrics in one dict (each value a plain float)."""
    return {
        "ann_return": ann_return(returns, periods_per_year),
        "ann_vol": ann_vol(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
    }
