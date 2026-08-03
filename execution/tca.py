"""
TradingEngineResearch — Transaction-Cost Analysis (TCA)
===========================================
Ex-ante cost prediction, ex-post cost attribution, and an adaptive feedback loop
that nudges the cost-model coefficients toward realised experience.

The ex-ante model (Part 17.1):

    cost_bps = spread_bps/2 + fee_bps
             + k1 · volatility · √participation          (temporary impact)
             + k2 · participation / liquidity_score        (queue/displacement)

``k1`` and ``k2`` are stateful (the `TCAModel` singleton): every ``ex_post`` cycle
calls ``update_cost_priors`` which exponentially smooths them toward the observed
coefficients, so subsequent ``ex_ante_cost_model`` calls reflect live conditions.

TCA outputs feed the meta-labeller's `expected_cost_bps`, the optimizer's impact
penalty, and the backtest cost model.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "TCAModel",
    "ex_ante_cost_model",
    "ex_post_cost_analysis",
    "update_cost_priors",
    "get_tca_model",
    "reset_tca_model",
    "DEFAULT_K1",
    "DEFAULT_K2",
]

DEFAULT_K1 = 0.10
DEFAULT_K2 = 0.05
_EMA_ALPHA = 0.10
_LIQUIDITY_REF_ADV = 5_000_000.0


def _liquidity_score(adv: float) -> float:
    """Market-liquidity scalar in (0, 1] from ADV (independent of participation)."""
    adv = max(float(adv), 0.0)
    return float(min(max(adv / (adv + _LIQUIDITY_REF_ADV), 0.05), 1.0))


class TCAModel:
    """Holds the adaptive impact coefficients ``k1``/``k2``."""

    def __init__(self, k1: float = DEFAULT_K1, k2: float = DEFAULT_K2, ema_alpha: float = _EMA_ALPHA) -> None:
        self.k1 = k1
        self.k2 = k2
        self.ema_alpha = ema_alpha

    def ex_ante_cost_model(
        self, symbol: str, qty: float, side: str, spread_bps: float,
        volatility: float, adv: float, participation: float, fee_bps: float = 0.5,
    ) -> float:
        """Expected round-trip cost in bps (monotone non-decreasing in participation)."""
        part = max(float(participation), 0.0)
        liq = _liquidity_score(adv)
        cost = (
            max(float(spread_bps), 0.0) / 2.0
            + float(fee_bps)
            + self.k1 * max(float(volatility), 0.0) * math.sqrt(part)
            + self.k2 * part / liq
        )
        return float(cost)

    def update_cost_priors(self, ex_post_results: dict) -> None:
        """EMA-update ``k1``/``k2`` toward the observed coefficients."""
        a = self.ema_alpha
        if ex_post_results.get("observed_k1") is not None:
            self.k1 = (1.0 - a) * self.k1 + a * float(ex_post_results["observed_k1"])
        if ex_post_results.get("observed_k2") is not None:
            self.k2 = (1.0 - a) * self.k2 + a * float(ex_post_results["observed_k2"])
        logger.debug("TCA priors updated: k1=%.4f, k2=%.4f", self.k1, self.k2)


# ── Module-level singleton + functional API ─────────────────────────────────────

_TCA_MODEL: Optional[TCAModel] = None


def get_tca_model() -> TCAModel:
    """Process-wide TCAModel singleton."""
    global _TCA_MODEL
    if _TCA_MODEL is None:
        _TCA_MODEL = TCAModel()
    return _TCA_MODEL


def reset_tca_model() -> None:
    global _TCA_MODEL
    _TCA_MODEL = None


def ex_ante_cost_model(
    symbol: str, qty: float, side: str, spread_bps: float,
    volatility: float, adv: float, participation: float, fee_bps: float = 0.5,
) -> float:
    """Expected cost in bps using the live (adaptive) coefficients."""
    return get_tca_model().ex_ante_cost_model(
        symbol, qty, side, spread_bps, volatility, adv, participation, fee_bps
    )


def update_cost_priors(ex_post_results: dict) -> None:
    """Feed realised costs back into the live coefficients."""
    get_tca_model().update_cost_priors(ex_post_results)


def _side_sign(direction: str) -> float:
    return 1.0 if str(direction).upper() == "BUY" else -1.0


def ex_post_cost_analysis(fills: list, decisions: list) -> dict:
    """
    Attribute realised execution cost from fills and the originating decisions.

    Decomposes implementation shortfall into a decision→arrival (spread/delay) and
    an arrival→fill (impact) leg, reports the realised vs expected delta, the
    passive-fill ratio, and the back-implied ``observed_k1``/``observed_k2`` used
    to update the cost priors.
    """
    if not fills:
        return {
            "realized_spread_cost_bps": 0.0, "realized_impact_bps": 0.0,
            "realized_fee_bps": 0.0, "total_realized_cost_bps": 0.0,
            "vs_expected_delta_bps": 0.0, "passive_fill_ratio": 0.0,
            "observed_k1": None, "observed_k2": None,
        }

    # OrderIntent carries no order_id (only FillEvent does), so decisions and
    # fills are matched by symbol. This is what lets the side sign (BUY/SELL) be
    # applied correctly — a missed join would silently default every fill to BUY.
    decision_by_symbol = {getattr(d, "symbol", None): d for d in decisions}
    fee_bps = 0.5

    spread_costs: list[float] = []
    impact_costs: list[float] = []
    slippages: list[float] = []
    passive_flags: list[float] = []

    for fill in fills:
        decision = decision_by_symbol.get(getattr(fill, "symbol", None))
        sign = _side_sign(getattr(decision, "direction", "BUY")) if decision else 1.0

        decision_price = float(getattr(fill, "decision_price", 0.0)) or 1.0
        arrival_price = float(getattr(fill, "arrival_price", decision_price)) or decision_price
        fill_price = float(getattr(fill, "fill_price", arrival_price))

        spread_leg = sign * (arrival_price - decision_price) / decision_price * 10_000.0
        impact_leg = sign * (fill_price - arrival_price) / arrival_price * 10_000.0
        spread_costs.append(spread_leg)
        impact_costs.append(impact_leg)
        slippages.append(float(getattr(fill, "slippage_bps", spread_leg + impact_leg)))
        # A fill at/under the arrival reference is treated as a passive (improved) fill.
        passive_flags.append(1.0 if impact_leg <= 0.0 else 0.0)

    n = len(fills)
    realized_spread = sum(spread_costs) / n
    realized_impact = sum(impact_costs) / n
    realized_fee = fee_bps
    total_realized = realized_spread + realized_impact + realized_fee

    expected = [float(getattr(d, "expected_cost_bps", 0.0)) for d in decisions]
    expected_avg = sum(expected) / len(expected) if expected else total_realized
    vs_expected = total_realized - expected_avg

    passive_ratio = sum(passive_flags) / n

    # Back-implied coefficients: scale current priors by realised/expected impact.
    model = get_tca_model()
    expected_impact = max(expected_avg - (realized_spread + realized_fee), 1e-6)
    ratio = float(min(max(realized_impact / expected_impact, 0.2), 5.0)) if expected_impact > 0 else 1.0
    observed_k1 = model.k1 * ratio
    observed_k2 = model.k2 * ratio

    return {
        "realized_spread_cost_bps": float(realized_spread),
        "realized_impact_bps": float(realized_impact),
        "realized_fee_bps": float(realized_fee),
        "total_realized_cost_bps": float(total_realized),
        "vs_expected_delta_bps": float(vs_expected),
        "passive_fill_ratio": float(passive_ratio),
        "observed_k1": float(observed_k1),
        "observed_k2": float(observed_k2),
    }
