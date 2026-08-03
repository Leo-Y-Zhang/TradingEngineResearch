"""
TradingEngineResearch — Carry Sleeve (equity dividend-yield carry)
======================================================
Cross-sectional equity carry: the "carry" of holding a stock if its price is
unchanged is its dividend yield, so this sleeve tilts toward high-trailing-yield
names and away from low/zero-yield names. The signal is the standardised
(z-scored) trailing dividend yield across the current universe:

    z_i = (yield_i - mean_yield) / std_yield ,   raw_i = tanh(sensitivity * z_i)

High relative yield → BUY (positive carry); below-average yield → SELL. Trailing
dividend yields are supplied by the caller (`data.price_ingestion`), point-in-time
safe; this sleeve only ranks them cross-sectionally. Deterministic — no RNG.

Returns the standardised `SignalOutput` (STEP 4 → STEP 5). With no yields supplied,
fewer than two names, or no cross-sectional dispersion, every name is FLAT.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from research.alpha_factory import SignalOutput

logger = logging.getLogger(__name__)

__all__ = ["SLEEVE_NAME", "generate_signals"]

SLEEVE_NAME = "carry"

_SENSITIVITY = 0.7         # tanh sensitivity to the yield z-score
_DEADBAND = 0.10           # |raw_score| below this → FLAT
_EXPECTED_HORIZON = 20     # dividend carry is a slow, multi-week tilt
_DECAY_HALF_LIFE = 20


def _asof(prices: pd.DataFrame, asof_timestamp: datetime | None) -> datetime:
    if asof_timestamp is not None:
        return asof_timestamp
    last = prices.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    if isinstance(last, datetime):
        return last
    logger.debug("carry: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def _flat(symbol: str, asof: datetime) -> SignalOutput:
    return SignalOutput(
        symbol=symbol, direction="FLAT", raw_score=0.0,
        expected_horizon=_EXPECTED_HORIZON, decay_half_life=_DECAY_HALF_LIFE,
        confidence_proxy=0.0, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
    )


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    dividend_yields: Optional[dict] = None,
    sensitivity: float = _SENSITIVITY,
) -> list[SignalOutput]:
    """One carry `SignalOutput` per symbol column in ``prices`` (cross-sectional
    dividend-yield z-score). ``dividend_yields`` maps symbol → trailing yield."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)
    symbols = [str(c) for c in prices.columns]
    yields = dividend_yields or {}

    # Need a cross-section with real dispersion to rank carry.
    if len(symbols) < 2 or not yields:
        return [_flat(s, asof) for s in symbols]

    vals = np.array([float(yields.get(s, 0.0)) for s in symbols], dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std <= 1e-12:
        return [_flat(s, asof) for s in symbols]

    signals: list[SignalOutput] = []
    for i, sym in enumerate(symbols):
        z = (vals[i] - mean) / std
        raw = float(np.clip(math.tanh(sensitivity * z), -1.0, 1.0))
        confidence = float(np.clip(min(abs(z), 2.0) / 2.0, 0.0, 1.0))

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
                expected_horizon=_EXPECTED_HORIZON, decay_half_life=_DECAY_HALF_LIFE,
                confidence_proxy=confidence, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
            )
        )
    return signals
