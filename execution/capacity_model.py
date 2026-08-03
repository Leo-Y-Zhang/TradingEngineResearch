"""
TradingEngineResearch — Capacity Model
==========================
How much capital a name can absorb before alpha is eroded by impact, and whether
a target position is realistically tradeable.

Hard rules (Part 17.2):
  • Never trade a name below the minimum dollar-volume threshold (default $5M ADV).
  • Never exceed the configured ADV participation (5% normal, 2% stressed).
  • Poor-capacity names are downweighted or rejected automatically.

``capacity_score ∈ [0, 1]``: 0 ⇒ do not trade, 1 ⇒ full capacity available.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "estimate_capacity",
    "capacity_score",
    "portfolio_capacity_report",
    "MIN_ADV_GBP",
    "PARTICIPATION_CAP",
]

MIN_ADV_GBP = 5_000_000.0
PARTICIPATION_CAP = {"normal": 0.05, "stressed": 0.02}


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def estimate_capacity(
    symbol: str,
    alpha_horizon_days: int,
    adv: float,
    spread_bps: float,
    volatility: float,
    participation_cap: float = 0.05,
) -> float:
    """
    Estimated deployable capital (GBP): the notional that can be accumulated at
    the participation cap over the alpha horizon, damped by a cost drag that
    grows with spread and volatility.
    """
    adv = float(adv)
    if adv <= 0.0 or alpha_horizon_days <= 0:
        return 0.0
    base = participation_cap * adv * float(alpha_horizon_days)
    drag = 1.0 / (1.0 + max(float(spread_bps), 0.0) / 50.0 + max(float(volatility), 0.0) / 500.0)
    return float(max(base * drag, 0.0))


def capacity_score(
    symbol: str,
    target_weight: float,
    capital_base: float,
    market_state: dict,
) -> float:
    """
    Tradeability of a target position in ``[0, 1]``.

    Returns 0.0 when ADV is below the minimum threshold. Otherwise blends the
    capacity headroom (capacity vs required notional) with a single-day
    participation check against the regime cap.
    """
    adv = float(market_state.get("adv", 0.0))
    if adv < MIN_ADV_GBP:
        return 0.0

    required = abs(float(target_weight)) * float(capital_base)
    if required <= 0.0:
        return 1.0

    cap = estimate_capacity(
        symbol,
        int(market_state.get("alpha_horizon_days", 5)),
        adv,
        float(market_state.get("spread_bps", 5.0)),
        float(market_state.get("volatility", 100.0)),
        float(market_state.get("participation_cap", PARTICIPATION_CAP["normal"])),
    )
    score = _clip01(cap / required)

    part_cap = PARTICIPATION_CAP["stressed" if market_state.get("stressed") else "normal"]
    one_day_participation = required / adv
    if one_day_participation > part_cap:
        score = min(score, part_cap / one_day_participation)

    return _clip01(score)


def portfolio_capacity_report(
    weights: dict,
    capital_base: float,
    market_data: dict,
) -> dict:
    """Per-symbol and aggregate capacity, flagging names that breach the caps."""
    per_symbol: dict[str, float] = {}
    flags: list[str] = []

    for symbol, weight in weights.items():
        state = market_data.get(symbol, {}) or {}
        per_symbol[symbol] = capacity_score(symbol, weight, capital_base, state)

        adv = float(state.get("adv", 0.0))
        required = abs(float(weight)) * float(capital_base)
        part_cap = PARTICIPATION_CAP["stressed" if state.get("stressed") else "normal"]
        if adv < MIN_ADV_GBP:
            flags.append(f"{symbol}:below_min_adv")
        elif adv > 0 and required / adv > part_cap:
            flags.append(f"{symbol}:exceeds_participation_cap")

    scores = list(per_symbol.values())
    return {
        "per_symbol": per_symbol,
        "aggregate_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "n_tradeable": sum(1 for s in scores if s > 0.0),
        "flags": flags,
    }
