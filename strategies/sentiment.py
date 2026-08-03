"""
TradingEngineResearch — Sentiment Sleeve (news/event-driven)
================================================
Event sleeve over per-symbol news sentiment: the caller supplies a score in
``[-1, 1]`` per symbol (produced by ``nlp.sentiment_pipeline`` — FinBERT or its
deterministic lexicon fallback, aggregated with a 60-minute time decay), and
this sleeve converts it into the standardised ``SignalOutput`` (STEP 4 → STEP 5):

    raw_i = sentiment_i ,   BUY above the deadband, SELL below, else FLAT

News alpha decays fast, so the horizon and half-life are short (days, not
weeks). Sentiment is point-in-time by construction — the aggregator only sees
headlines whose ``age_minutes`` is non-negative at the cycle's asof time.
Deterministic — no RNG; with no scores supplied every name is FLAT.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from research.alpha_factory import SignalOutput

logger = logging.getLogger(__name__)

__all__ = ["SLEEVE_NAME", "generate_signals"]

SLEEVE_NAME = "sentiment"

_DEADBAND = 0.15           # |sentiment| below this → FLAT (headline noise floor)
_EXPECTED_HORIZON = 2      # news alpha is fast: a couple of days at most
_DECAY_HALF_LIFE = 1


def _asof(prices: pd.DataFrame, asof_timestamp: datetime | None) -> datetime:
    if asof_timestamp is not None:
        return asof_timestamp
    last = prices.index[-1]
    if isinstance(last, pd.Timestamp):
        return last.to_pydatetime()
    if isinstance(last, datetime):
        return last
    logger.debug("sentiment: non-datetime index and no asof_timestamp; using epoch.")
    return datetime(1970, 1, 1)


def generate_signals(
    prices: pd.DataFrame,
    asof_timestamp: datetime | None = None,
    sentiment_scores: Optional[dict] = None,
) -> list[SignalOutput]:
    """One sentiment ``SignalOutput`` per symbol column in ``prices``.
    ``sentiment_scores`` maps symbol → aggregated sentiment in ``[-1, 1]``."""
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    asof = _asof(prices, asof_timestamp)
    scores = sentiment_scores or {}

    signals: list[SignalOutput] = []
    for sym in (str(c) for c in prices.columns):
        raw = float(np.clip(float(scores.get(sym, 0.0)), -1.0, 1.0))
        confidence = float(np.clip(abs(raw), 0.0, 1.0))

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
