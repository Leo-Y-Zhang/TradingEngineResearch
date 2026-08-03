"""
TradingEngineResearch — Sentiment pipeline (news → FinBERT → aggregated per-symbol sentiment)
=================================================================================
End-to-end glue that turns raw news headlines into a per-symbol sentiment score in
``[-1, 1]``: each headline is scored by the FinBERT scorer (transformer if available,
deterministic lexicon fallback otherwise) and the per-symbol scores are combined by
the time-decay aggregator. This is the producer of the model's ``sentiment_score``
feature and of the sentiment event sleeve's input.

Each news item is a dict carrying the headline text (``headline`` or ``text``),
``age_minutes``, and ``symbol`` (or ``symbols``). News data is supplied by the caller;
note that free feeds (e.g. yfinance ``Ticker.news``) provide only *recent* headlines,
not dated history, so this path is meaningful live but the sentiment feature cannot be
backfilled for the historical backtest.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from nlp.finbert_scorer import get_scorer
from nlp.sentiment_aggregator import aggregate

logger = logging.getLogger(__name__)

__all__ = ["compute_sentiment_scores"]


def compute_sentiment_scores(
    news_items: list[dict],
    symbols: list[str],
    scorer: Optional[Any] = None,
    window_minutes: Optional[float] = None,
) -> dict[str, float]:
    """Score each headline with FinBERT and aggregate to a per-symbol sentiment in
    ``[-1, 1]``. Symbols with no relevant in-window news get 0.0 (neutral)."""
    symbols = [str(s) for s in symbols]
    if not news_items:
        return {s: 0.0 for s in symbols}

    engine = scorer if scorer is not None else get_scorer()
    scored: list[dict] = []
    for item in news_items:
        text = str(item.get("headline") or item.get("text") or "")
        probs = engine.score(text)
        scored.append({**item, **probs})

    kwargs = {"window_minutes": window_minutes} if window_minutes is not None else {}
    return {s: float(aggregate(s, scored, **kwargs)) for s in symbols}
