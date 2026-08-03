"""
TradingEngineResearch — FinBERT Sentiment Scorer
====================================
Headline sentiment with **temperature-scaled** FinBERT probabilities.

FinBERT is over-confident out of the box, so its logits are divided by a
temperature ``T = 1.5`` before the softmax — this widens the distribution and
makes the probabilities usable as soft features. Robustness is paramount: this
module **never crashes and never sends text to an external service**.

Degradation ladder (each step logs a WARNING):
  1. Model loads → temperature-scaled probabilities.
  2. Logit extraction fails → raw softmax of whatever logits were obtained.
  3. Model unavailable (no transformers/torch, or download fails) → a small,
     deterministic lexicon fallback so the pipeline still gets a valid signal.

Every path returns a dict with keys ``positive``, ``negative``, ``neutral``
summing to 1.0. ``batch_score`` is non-blocking: if CPU load exceeds the
threshold it defers (returns ``[]``) rather than contending for the box.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "LABELS",
    "TEMPERATURE",
    "temperature_scaled_probs",
    "raw_softmax_probs",
    "FinBERTScorer",
    "get_scorer",
    "reset_scorer",
]

TEMPERATURE = 1.5
LABELS: tuple[str, str, str] = ("positive", "negative", "neutral")
_MODEL_NAME = "ProsusAI/finbert"
_CPU_THRESHOLD = 80.0
_NEUTRAL = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

# Minimal deterministic lexicon for the offline fallback path.
_POS_WORDS = frozenset({
    "beat", "beats", "surge", "surges", "gain", "gains", "rise", "rises", "profit",
    "profits", "growth", "strong", "record", "upgrade", "upgraded", "bullish",
    "outperform", "raised", "jumps", "soars", "rally", "rallies", "wins", "approval",
})
_NEG_WORDS = frozenset({
    "miss", "misses", "missed", "fall", "falls", "drop", "drops", "loss", "losses",
    "weak", "cut", "cuts", "downgrade", "downgraded", "bearish", "plunge", "plunges",
    "slump", "warns", "warning", "lawsuit", "probe", "decline", "declines", "fraud",
})


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def temperature_scaled_probs(
    logits, temperature: float = TEMPERATURE, labels: tuple[str, ...] = LABELS
) -> dict:
    """Softmax of ``logits / temperature`` mapped to the label dict (sums to 1.0)."""
    probs = _softmax(np.asarray(logits, dtype=float) / temperature)
    return {lab: float(probs[i]) for i, lab in enumerate(labels)}


def raw_softmax_probs(logits, labels: tuple[str, ...] = LABELS) -> dict:
    """Plain softmax of ``logits`` (the raw-softmax fallback path)."""
    probs = _softmax(np.asarray(logits, dtype=float))
    return {lab: float(probs[i]) for i, lab in enumerate(labels)}


def _cpu_percent_default() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.0))
    except Exception:  # noqa: BLE001 — psutil optional; absence must not block scoring
        return 0.0


class FinBERTScorer:
    """Temperature-scaled FinBERT scorer with graceful offline degradation."""

    def __init__(
        self,
        temperature: float = TEMPERATURE,
        model_name: str = _MODEL_NAME,
        device: str = "cpu",
        cpu_threshold: float = _CPU_THRESHOLD,
        cpu_percent_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.temperature = temperature
        self.model_name = model_name
        self.device = device
        self.cpu_threshold = cpu_threshold
        self._cpu_percent_fn = cpu_percent_fn or _cpu_percent_default

        self._model: Any = None
        self._tokenizer: Any = None
        self._available: Optional[bool] = None
        self._label_order: list[int] = [0, 1, 2]

    # ── model loading ────────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch  # noqa: F401 — required backend
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._label_order = self._resolve_label_order(self._model.config)
            self._available = True
        except Exception as exc:  # noqa: BLE001 — offline / missing dep ⇒ lexicon fallback
            logger.warning(
                "FinBERT unavailable (%s); using deterministic lexicon fallback.", exc
            )
            self._available = False
        return self._available

    @staticmethod
    def _resolve_label_order(config) -> list[int]:
        """Indices into the model's logit vector for (positive, negative, neutral)."""
        id2label = getattr(config, "id2label", None) or {}
        label_to_idx = {str(v).lower(): int(k) for k, v in id2label.items()}
        return [label_to_idx.get(lab, i) for i, lab in enumerate(LABELS)]

    def _order_logits(self, logits: np.ndarray) -> np.ndarray:
        if logits.size != 3:
            return logits
        try:
            return np.array([logits[i] for i in self._label_order], dtype=float)
        except (IndexError, TypeError):
            return logits

    def _extract_logits(self, headline: str) -> np.ndarray:
        import torch

        encoded = self._tokenizer(  # type: ignore[misc]
            headline, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            output = self._model(**encoded)  # type: ignore[misc]
        return output.logits.detach().cpu().numpy().ravel()

    # ── scoring ──────────────────────────────────────────────────────────────

    def score(self, headline: str) -> dict:
        """Score a single headline; always returns a valid prob dict (sums to 1.0)."""
        if not headline or not headline.strip():
            return dict(_NEUTRAL)

        if not self._ensure_model():
            return self._lexicon_score(headline)

        try:
            ordered = self._order_logits(self._extract_logits(headline))
            return temperature_scaled_probs(ordered, self.temperature)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FinBERT temperature scaling unavailable — using raw softmax (%s)", exc
            )
            try:
                ordered = self._order_logits(self._extract_logits(headline))
                return raw_softmax_probs(ordered)
            except Exception:  # noqa: BLE001 — last resort
                return self._lexicon_score(headline)

    def batch_score(self, headlines: list[str]) -> list[dict]:
        """Score many headlines; defers (returns ``[]``) when CPU load is high."""
        load = self._cpu_percent_fn()
        if load > self.cpu_threshold:
            logger.info(
                "FinBERT batch_score deferred: CPU %.0f%% > %.0f%% threshold.",
                load, self.cpu_threshold,
            )
            return []
        return [self.score(h) for h in headlines]

    @staticmethod
    def _lexicon_score(headline: str) -> dict:
        words = set(re.findall(r"[a-z]+", headline.lower()))
        pos = len(words & _POS_WORDS)
        neg = len(words & _NEG_WORDS)
        if pos == 0 and neg == 0:
            return dict(_NEUTRAL)
        probs = _softmax(np.array([pos, neg, 0.5], dtype=float))
        return {"positive": float(probs[0]), "negative": float(probs[1]), "neutral": float(probs[2])}


# ── Module-level singleton ──────────────────────────────────────────────────────

_SCORER: Optional[FinBERTScorer] = None


def get_scorer() -> FinBERTScorer:
    """Return the process-wide FinBERTScorer singleton (created on first use)."""
    global _SCORER
    if _SCORER is None:
        _SCORER = FinBERTScorer()
    return _SCORER


def reset_scorer() -> None:
    """Drop the singleton (tests / restarts)."""
    global _SCORER
    _SCORER = None
