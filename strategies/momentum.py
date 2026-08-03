"""
TradingEngineResearch — Momentum / Trend Sleeve
===================================
Risk-adjusted time-series ("absolute") momentum with a 12-1 lookback.

Each symbol's signal is its trailing total return over a long lookback window,
*skipping the most recent ~1 month* to avoid the well-documented short-term
reversal (Jegadeesh & Titman 1993; the canonical "12-1" momentum), divided by the
realised volatility of that window so signals are comparable across assets. The
output is the standardised `SignalOutput` dataclass consumed by the pipeline
(STEP 4 → STEP 5).

Point-in-time safe: only prices up to the last row are used; `asof_timestamp`
defaults to the last index value. Deterministic — no RNG.
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

SLEEVE_NAME = "momentum"

_LOOKBACK = 252           # ~12 months of trading days
_SKIP = 21                # skip most recent ~1 month (short-term reversal)
_EXPECTED_HORIZON = 20    # momentum is a multi-week effect
_DECAY_HALF_LIFE = 10
_DEADBAND = 0.10          # |raw_score| below this → FLAT


def _asof(prices: pd.DataFrame, asof_timestamp: datetime | None) -> datetime:
    if asof_timestamp is not None:
        return asof_timestamp
    last = prices.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    if isinstance(last, datetime):
        return last
    logger.debug("momentum: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def _flat(symbol: str, asof: datetime, horizon: int, half_life: int) -> SignalOutput:
    return SignalOutput(
        symbol=symbol, direction="FLAT", raw_score=0.0,
        expected_horizon=horizon, decay_half_life=half_life,
        confidence_proxy=0.0, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
    )


def _trend_consistency(window: np.ndarray, chunk: int = 21) -> float:
    """Fraction of non-overlapping `chunk`-day sub-returns sharing the net sign."""
    if window.size < chunk + 1:
        return 0.0
    chunk_rets: list[float] = []
    for i in range(chunk, window.size, chunk):
        a, b = window[i - chunk], window[i]
        if a > 0:
            chunk_rets.append(b / a - 1.0)
    if not chunk_rets:
        return 0.0
    arr = np.asarray(chunk_rets)
    overall = np.sign(arr.sum())
    if overall == 0:
        return 0.0
    return float(np.mean(np.sign(arr) == overall))


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    lookback: int = _LOOKBACK,
    skip: int = _SKIP,
    expected_horizon: int = _EXPECTED_HORIZON,
    decay_half_life: int = _DECAY_HALF_LIFE,
) -> list[SignalOutput]:
    """Generate one momentum `SignalOutput` per symbol column in `prices`."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)
    min_len = lookback + skip + 2

    signals: list[SignalOutput] = []
    for symbol in prices.columns:
        sym = str(symbol)
        vals = prices[symbol].dropna().to_numpy(dtype=float)
        if vals.size < min_len:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        p_recent = vals[-1 - skip]                 # price `skip` days ago
        p_past = vals[-1 - skip - lookback]        # price lookback+skip days ago
        if p_past <= 0.0 or p_recent <= 0.0:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        momentum = p_recent / p_past - 1.0

        end = vals.size - skip
        start = max(end - lookback - 1, 0)
        window = vals[start:end]
        rets = np.diff(window) / window[:-1] if window.size >= 2 else np.array([])
        period_vol = float(np.std(rets, ddof=1)) * math.sqrt(lookback) if rets.size >= 2 else 0.0

        risk_adj = momentum / period_vol if period_vol > 1e-9 else momentum
        raw = float(np.clip(math.tanh(risk_adj), -1.0, 1.0))
        consistency = _trend_consistency(window, chunk=21)
        confidence = float(np.clip(0.5 * min(abs(risk_adj), 1.0) + 0.5 * consistency, 0.0, 1.0))

        if raw > _DEADBAND:
            direction = "BUY"
        elif raw < -_DEADBAND:
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
