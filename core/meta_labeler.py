"""
TradingEngineResearch — Meta-Label Trade Admission
======================================
The final gate between a raw signal and a trade candidate. It converts model
outputs into a net-of-cost edge, admits a trade only when *all* admission
conditions hold simultaneously, and sizes the survivor by conviction, liquidity,
and crowding.

A trade is admitted only if, together:
  • ``p_positive   >= 0.55``
  • ``p_tail_loss  <= 0.25``
  • ``confidence   >= 0.35``
  • ``expected_net_edge_bps >= min_edge_bps`` (regime-dependent)

In ``stressed_exec`` the bar is raised further: any trade whose edge is below
``1.5 × min_edge`` is blocked as marginal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["TradeDecision", "compute", "MIN_EDGE_BPS"]

# Minimum net edge (bps) required to admit a trade, by execution regime.
MIN_EDGE_BPS: dict[str, float] = {
    "normal_exec": 10.0,
    "cautious_exec": 15.0,
    "stressed_exec": 20.0,
}

_TAIL_PENALTY_PER_UNIT = 50.0   # bps of penalty per unit of tail-loss probability
_P_POSITIVE_MIN = 0.55
_P_TAIL_LOSS_MAX = 0.25
_CONFIDENCE_MIN = 0.35


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


@dataclass
class TradeDecision:
    """The admission verdict and sizing for a single trade candidate."""

    take_trade: bool
    size_multiplier: float                  # [0.0, 1.0]
    hold_horizon_override: int | None
    rejection_reason: str | None
    expected_net_edge_bps: float


def compute(
    mu: float,
    sigma: float,
    p_positive: float,
    p_tail_loss: float,
    confidence: float,
    expected_cost_bps: float,
    execution_regime: str,
    crowding_score: float,
    liquidity_score: float,
    regime: str,
) -> TradeDecision:
    """
    Admit or reject a trade candidate and compute its size multiplier.

    ``mu`` is the expected fractional return; ``expected_cost_bps`` the modelled
    round-trip cost. All probabilities are in ``[0, 1]``.
    """
    tail_penalty_bps = p_tail_loss * _TAIL_PENALTY_PER_UNIT
    expected_net_edge_bps = 10_000.0 * mu - expected_cost_bps - tail_penalty_bps

    min_edge_bps = MIN_EDGE_BPS.get(execution_regime, MIN_EDGE_BPS["normal_exec"])

    # Size by conviction, then damp for thin liquidity and crowded books.
    base = (p_positive - p_tail_loss) * confidence
    liquidity_adj = _clip(liquidity_score, 0.5, 1.0)
    crowding_adj = _clip(1.0 - crowding_score, 0.5, 1.0)
    size_multiplier = _clip(base * liquidity_adj * crowding_adj, 0.0, 1.0)

    # Admission: every condition must hold.
    reasons: list[str] = []
    if p_positive < _P_POSITIVE_MIN:
        reasons.append(f"p_positive {p_positive:.3f} < {_P_POSITIVE_MIN}")
    if p_tail_loss > _P_TAIL_LOSS_MAX:
        reasons.append(f"p_tail_loss {p_tail_loss:.3f} > {_P_TAIL_LOSS_MAX}")
    if confidence < _CONFIDENCE_MIN:
        reasons.append(f"confidence {confidence:.3f} < {_CONFIDENCE_MIN}")
    if expected_net_edge_bps < min_edge_bps:
        reasons.append(f"edge {expected_net_edge_bps:.1f}bps < {min_edge_bps}bps")
    if execution_regime == "stressed_exec" and expected_net_edge_bps < 1.5 * min_edge_bps:
        reasons.append(
            f"stressed_exec marginal: edge {expected_net_edge_bps:.1f}bps < "
            f"{1.5 * min_edge_bps:.1f}bps"
        )

    take_trade = not reasons
    if not take_trade:
        logger.debug("meta_labeler reject (regime=%s): %s", execution_regime, "; ".join(reasons))

    return TradeDecision(
        take_trade=take_trade,
        size_multiplier=size_multiplier if take_trade else 0.0,
        hold_horizon_override=None,
        rejection_reason="; ".join(reasons) if reasons else None,
        expected_net_edge_bps=expected_net_edge_bps,
    )
