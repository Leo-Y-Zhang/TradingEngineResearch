"""
TradingEngineResearch — Volatility-Overlay Sleeve
=====================================
A risk-off / low-volatility overlay: trim into volatility *expansions* and lean in
when volatility *compresses*. For each symbol it compares recent realised volatility
to a longer baseline and emits a defensive (negative) score when vol is elevated and
a constructive (positive) score when vol is compressed:

    raw = -tanh( sensitivity · (recent_vol / baseline_vol - 1) )

This harvests the volatility-timing / low-vol premium and de-risks names whose vol is
spiking, improving the book's risk-adjusted return. Output is the standardised
`SignalOutput` (STEP 4 → STEP 5).

Point-in-time safe: only trailing prices are used; `asof_timestamp` defaults to the
last index value. Deterministic — no RNG.
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

SLEEVE_NAME = "volatility_overlay"

_SHORT_WINDOW = 20         # recent realised-vol window (~1 month)
_BASELINE_WINDOW = 60      # baseline realised-vol window (~1 quarter)
_SENSITIVITY = 2.0         # tanh sensitivity to the vol ratio's deviation from 1
_DEADBAND = 0.10           # |raw_score| below this → FLAT
_EXPECTED_HORIZON = 10
_DECAY_HALF_LIFE = 5


def _asof(prices: pd.DataFrame, asof_timestamp: datetime | None) -> datetime:
    if asof_timestamp is not None:
        return asof_timestamp
    last = prices.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    if isinstance(last, datetime):
        return last
    logger.debug("volatility_overlay: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def _flat(symbol: str, asof: datetime, horizon: int, half_life: int) -> SignalOutput:
    return SignalOutput(
        symbol=symbol, direction="FLAT", raw_score=0.0,
        expected_horizon=horizon, decay_half_life=half_life,
        confidence_proxy=0.0, sleeve_name=SLEEVE_NAME, asof_timestamp=asof,
    )


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    short_window: int = _SHORT_WINDOW,
    baseline_window: int = _BASELINE_WINDOW,
    sensitivity: float = _SENSITIVITY,
    expected_horizon: int = _EXPECTED_HORIZON,
    decay_half_life: int = _DECAY_HALF_LIFE,
) -> list[SignalOutput]:
    """One volatility-overlay `SignalOutput` per symbol column in ``prices``."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)
    min_len = baseline_window + 2

    signals: list[SignalOutput] = []
    for symbol in prices.columns:
        sym = str(symbol)
        vals = prices[symbol].dropna().to_numpy(dtype=float)
        if vals.size < min_len:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        rets = np.diff(vals) / vals[:-1]
        if rets.size < baseline_window:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        recent_vol = float(np.std(rets[-short_window:], ddof=1))
        baseline_vol = float(np.std(rets[-baseline_window:], ddof=1))
        if baseline_vol <= 1e-12:
            signals.append(_flat(sym, asof, expected_horizon, decay_half_life))
            continue

        vol_ratio = recent_vol / baseline_vol
        signal = sensitivity * (vol_ratio - 1.0)
        raw = float(np.clip(-math.tanh(signal), -1.0, 1.0))    # vol expansion → defensive (negative)
        confidence = float(np.clip(min(abs(signal), 1.0), 0.0, 1.0))

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
