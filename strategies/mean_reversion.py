"""
TradingEngineResearch — Mean-Reversion Sleeve
=================================
Bollinger-style z-score reversion, *gated by an AR(1) reversion-speed estimate*.

For each symbol the standardised deviation of the latest price from its rolling
mean, ``z = (P − MA) / SD``, drives the signal: deeply oversold (``z ≪ 0``) is a
BUY, overbought (``z ≫ 0``) is a SELL. Confidence is scaled by how strongly the
series actually mean-reverts — the AR(1) coefficient ``φ`` of the de-meaned
level. A fast reverter (low ``φ``) is trusted; a near-random-walk or trending
series (``φ → 1``) is distrusted, so the sleeve does not fight a trend.

Point-in-time safe; deterministic.
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

SLEEVE_NAME = "mean_reversion"

_WINDOW = 20
_ENTRY_Z = 1.0
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
    logger.debug("mean_reversion: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def _flat(symbol: str, asof: datetime, horizon: int, half_life: int) -> SignalOutput:
    return SignalOutput(
        symbol=symbol, direction="FLAT", raw_score=0.0,
        expected_horizon=horizon, decay_half_life=half_life,
        confidence_proxy=0.0, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
    )


def _ar1_coef(x: np.ndarray) -> float:
    """AR(1) coefficient φ of a de-meaned series (1.0 ⇒ random walk / no reversion)."""
    if x.size < 3:
        return 1.0
    xc = x - x.mean()
    x0, x1 = xc[:-1], xc[1:]
    denom = float(np.dot(x0, x0))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(x0, x1) / denom)


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    window: int = _WINDOW,
    entry_z: float = _ENTRY_Z,
    expected_horizon: int = _EXPECTED_HORIZON,
    decay_half_life: int = _DECAY_HALF_LIFE,
) -> list[SignalOutput]:
    """Generate one mean-reversion `SignalOutput` per symbol column in `prices`."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)
    min_len = 2 * window

    signals: list[SignalOutput] = []
    for symbol in prices.columns:
        sym = str(symbol)
        vals = prices[symbol].dropna().to_numpy(dtype=float)
        if vals.size < min_len:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        recent = vals[-window:]
        ma = float(np.mean(recent))
        sd = float(np.std(recent, ddof=1))
        if sd <= 1e-12:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        z = (vals[-1] - ma) / sd
        raw = float(np.clip(-math.tanh(z / 2.0), -1.0, 1.0))

        phi = _ar1_coef(vals[-2 * window:])
        mr_quality = float(np.clip(1.0 - phi, 0.0, 1.0))
        confidence = float(np.clip(min(abs(z), 3.0) / 3.0 * mr_quality, 0.0, 1.0))

        if z < -entry_z:
            direction = "BUY"
        elif z > entry_z:
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
