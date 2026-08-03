"""
TradingEngineResearch — Statistical-Arbitrage Sleeve
========================================
Residual mean-reversion in the spirit of Avellaneda & Lee (2010),
"Statistical Arbitrage in the US Equities Market".

Each symbol's return is regressed on an equal-weight market factor; the residual
(idiosyncratic) return is cumulated into a process ``X_t`` modelled as a
mean-reverting Ornstein-Uhlenbeck / AR(1) process. The dimensionless **s-score**
measures how far ``X_t`` sits from its equilibrium in units of its stationary
standard deviation. A large negative s-score (idiosyncratically cheap relative to
peers) is a BUY; a large positive s-score is a SELL. The resulting book is
market-neutral by construction, and confidence scales with the estimated
reversion speed.

Requires ≥ 2 symbols. Point-in-time safe; deterministic.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

from research.alpha_factory import SignalOutput

logger = logging.getLogger(__name__)

__all__ = ["SLEEVE_NAME", "generate_signals"]

SLEEVE_NAME = "stat_arb"

_WINDOW = 60
_ENTRY_S = 1.25
_EXPECTED_HORIZON = 5
_DECAY_HALF_LIFE = 3


def _asof(prices: pd.DataFrame, asof_timestamp: datetime | None) -> datetime:
    if asof_timestamp is not None:
        return asof_timestamp
    last = prices.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    if isinstance(last, datetime):
        return last
    logger.debug("stat_arb: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def _flat(symbol: str, asof: datetime, horizon: int, half_life: int) -> SignalOutput:
    return SignalOutput(
        symbol=symbol, direction="FLAT", raw_score=0.0,
        expected_horizon=horizon, decay_half_life=half_life,
        confidence_proxy=0.0, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
    )


def _ou_s_score(ri: np.ndarray, factor: np.ndarray) -> tuple[float | None, float]:
    """Avellaneda-Lee s-score of a symbol's residual against a market factor.

    Returns ``(s_score, b)`` where ``b`` is the AR(1) coefficient of the residual
    process. ``s_score`` is ``None`` when the residual is not mean-reverting
    (``b`` outside ``(0, 1)``) or degenerate.
    """
    factor_c = factor - factor.mean()
    denom = float(np.dot(factor_c, factor_c))
    if denom > 1e-12:
        beta = float(np.dot(ri - ri.mean(), factor_c) / denom)
    else:
        beta = 0.0
    residual = ri - beta * factor
    x = np.cumsum(residual)

    x0, x1 = x[:-1], x[1:]
    if x0.size < 3:
        return None, 1.0

    # OLS fit of X_t = a + b·X_{t-1}
    design = np.vstack([np.ones_like(x0), x0]).T
    coef, *_ = np.linalg.lstsq(design, x1, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if not (0.0 < b < 1.0):
        return None, b

    ou_resid = x1 - (a + b * x0)
    var_eq = float(np.var(ou_resid, ddof=1)) / (1.0 - b * b)
    sigma_eq = math.sqrt(var_eq) if var_eq > 0.0 else 0.0
    if sigma_eq <= 1e-12:
        return None, b

    equilibrium = a / (1.0 - b)
    s_score = (float(x[-1]) - equilibrium) / sigma_eq
    return s_score, b


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    window: int = _WINDOW,
    entry_s: float = _ENTRY_S,
    expected_horizon: int = _EXPECTED_HORIZON,
    decay_half_life: int = _DECAY_HALF_LIFE,
) -> list[SignalOutput]:
    """Generate one stat-arb `SignalOutput` per symbol column in `prices`."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)

    if prices.shape[1] < 2:
        return [
            _flat(str(c), asof, expected_horizon, decay_half_life) for c in prices.columns
        ]

    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if returns.shape[0] < window + 2:
        return [
            _flat(str(c), asof, expected_horizon, decay_half_life) for c in prices.columns
        ]

    window_returns = returns.iloc[-window:]
    n_assets = window_returns.shape[1]
    total = window_returns.sum(axis=1).to_numpy(dtype=float)

    signals: list[SignalOutput] = []
    for symbol in prices.columns:
        sym = str(symbol)
        ri = window_returns[symbol].to_numpy(dtype=float)
        # Leave-one-out market factor: the symbol is excluded from its own
        # benchmark so a large idiosyncratic move cannot contaminate the factor.
        factor = (total - ri) / (n_assets - 1)
        s_score, b = _ou_s_score(ri, factor)
        if s_score is None:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        raw = float(np.clip(-math.tanh(s_score / 2.0), -1.0, 1.0))
        mr_quality = float(np.clip(1.0 - b, 0.0, 1.0))
        confidence = float(np.clip(min(abs(s_score), 3.0) / 3.0 * mr_quality, 0.0, 1.0))

        if s_score < -entry_s:
            direction = "BUY"
        elif s_score > entry_s:
            direction = "SELL"
        else:
            direction = "FLAT"
            confidence = min(confidence, 0.2)

        signals.append(
            SignalOutput(
                symbol=sym, direction=direction, raw_score=raw,
                expected_horizon=expected_horizon, decay_half_life=decay_half_life,
                confidence_proxy=confidence, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
            )
        )
    return signals
