"""
TradingEngineResearch — Sentiment Aggregator
================================
Collapses a set of scored headlines into a single per-symbol sentiment in
``[-1, 1]``, weighting each headline by an exponential time decay so that fresh
news dominates and only the last 60 minutes count.

  sentiment(headline) = positive − negative                      ∈ [−1, 1]
  weight(headline)     = exp(−0.05 · age_minutes)
  symbol_sentiment     = Σ(sentiment · weight) / Σ(weight)

Headlines older than the 60-minute window are ignored; if none remain the
result is ``0.0`` (neutral). When a headline carries symbol metadata
(``symbol`` or ``symbols``) it is filtered to the requested symbol; otherwise it
is assumed pre-filtered and included.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = ["aggregate", "WINDOW_MINUTES", "DECAY_RATE"]

WINDOW_MINUTES = 60.0
DECAY_RATE = 0.05


def _relevant_to_symbol(headline: dict, symbol: str) -> bool:
    symbols = headline.get("symbols")
    if symbols is not None:
        return symbol in symbols
    if "symbol" in headline:
        return headline["symbol"] == symbol
    return True   # no symbol metadata ⇒ assume pre-filtered


def aggregate(
    symbol: str,
    scored_headlines: list[dict],
    window_minutes: float = WINDOW_MINUTES,
    decay_rate: float = DECAY_RATE,
) -> float:
    """
    Time-decay-weighted mean sentiment for ``symbol`` in ``[-1, 1]``.

    Each entry in ``scored_headlines`` is expected to carry ``positive`` and
    ``negative`` probabilities plus ``age_minutes`` (and optionally ``symbol`` /
    ``symbols``). Returns ``0.0`` when there are no relevant, in-window headlines.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for headline in scored_headlines:
        age = float(headline.get("age_minutes", 0.0))
        if age < 0.0 or age > window_minutes:
            continue
        if not _relevant_to_symbol(headline, symbol):
            continue

        sentiment = float(headline.get("positive", 0.0)) - float(headline.get("negative", 0.0))
        sentiment = max(min(sentiment, 1.0), -1.0)
        weight = math.exp(-decay_rate * age)

        weighted_sum += sentiment * weight
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return float(max(min(weighted_sum / total_weight, 1.0), -1.0))
