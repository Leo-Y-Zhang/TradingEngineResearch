"""
TradingEngineResearch — Slippage Model
==========================
A square-root market-impact estimate of execution slippage, in basis points.

    slippage_bps = spread_bps / 2  +  impact_coef · volatility_bps · √participation

The half-spread is the unavoidable cost of crossing; the impact term grows with
the square root of participation (the standard concave impact law, Almgren et
al. 2005). Slippage is monotonically non-decreasing in participation and never
negative.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = ["estimate_slippage", "IMPACT_COEF"]

IMPACT_COEF = 0.10                 # impact coefficient on volatility·√participation
_MIN_LIQUIDITY_ADV = 5_000_000.0   # below this ADV, impact is penalised


def estimate_slippage(
    qty: float,
    adv: float,
    volatility: float,
    spread_bps: float,
    participation: float | None = None,
    side: str = "BUY",
) -> float:
    """
    Estimate execution slippage in basis points.

    ``volatility`` is expected in bps (e.g. daily realised vol expressed in bps).
    ``participation`` defaults to ``|qty| / adv`` when not supplied. The result is
    always ≥ 0 and increases with participation.
    """
    adv = float(adv)
    if participation is None:
        participation = abs(float(qty)) / adv if adv > 0 else 1.0
    participation = float(min(max(participation, 0.0), 1.0))

    half_spread = max(float(spread_bps), 0.0) / 2.0
    impact = IMPACT_COEF * max(float(volatility), 0.0) * math.sqrt(participation)

    # Thin-liquidity penalty: scale impact up as ADV falls below the reference.
    if 0.0 < adv < _MIN_LIQUIDITY_ADV:
        impact *= _MIN_LIQUIDITY_ADV / adv

    return float(half_spread + impact)
