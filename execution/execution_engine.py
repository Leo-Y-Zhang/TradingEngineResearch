"""
TradingEngineResearch — Execution Engine
============================
Order lifecycle, regime-aware child-order scheduling, and post-trade reporting.

Core principle (Part 16.1): never default to market orders. Passive, limit-first
execution is the rule; aggressive crossing is reserved for URGENT_DERISK orders.
The child-order scheduler adapts to the execution regime:

  • normal_exec   — passive limit orders, slower pace
  • cautious_exec — half the participation (slower), tighter cancellation
  • stressed_exec — reduce size by 50% and avoid aggressive crossing, unless the
                    order is a de-risking order

The order state machine permits only the transitions in Part 16.2; any other
transition raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from data.data_contracts import OrderIntent, normalize_mode

logger = logging.getLogger(__name__)

__all__ = [
    "OrderState",
    "OrderStateMachine",
    "ChildOrderPlan",
    "ExecutionReport",
    "schedule_order",
    "compute_execution_report",
    "TERMINAL_STATES",
    "REGIME_PARTICIPATION_CAP",
]

# ADV participation caps by execution regime (Part 17.2): never exceed 5% ADV in
# normal/cautious operation, 2% when the market is stressed.
REGIME_PARTICIPATION_CAP: dict[str, float] = {
    "normal_exec": 0.05,
    "cautious_exec": 0.05,
    "stressed_exec": 0.02,
}


# ── 16.2 Order state machine ─────────────────────────────────────────────────────

class OrderState(str, Enum):
    NEW = "NEW"
    STAGED = "STAGED"
    WORKING = "WORKING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.NEW: {OrderState.STAGED, OrderState.REJECTED},
    OrderState.STAGED: {OrderState.WORKING, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED},
    OrderState.WORKING: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED},
    OrderState.PARTIAL: {OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}

TERMINAL_STATES: set[OrderState] = {
    OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED,
}


class OrderStateMachine:
    """Enforces the Part 16.2 order-state transition graph."""

    def __init__(self, state: OrderState = OrderState.NEW) -> None:
        self.state = state
        self.history: list[OrderState] = [state]

    def can_transition(self, to_state: OrderState) -> bool:
        return to_state in _TRANSITIONS[self.state]

    def transition(self, to_state: OrderState) -> OrderState:
        if not self.can_transition(to_state):
            raise ValueError(f"Invalid order-state transition: {self.state.value} → {to_state.value}")
        self.state = to_state
        self.history.append(to_state)
        return self.state

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


# ── 16.3 Child-order scheduling ──────────────────────────────────────────────────

@dataclass
class ChildOrderPlan:
    """A single scheduled child slice of a parent order."""

    symbol: str
    side: str                         # "BUY" | "SELL"
    qty: float
    order_type: str                   # "LIMIT" | "MARKET"
    limit_offset_bps: Optional[float] # passive limit offset from arrival (None for MARKET)
    scheduled_offset_minutes: float
    participation: float
    slice_index: int
    tag: str                          # "passive" | "derisk"


def _n_slices(time_to_close: float, half_life: float) -> int:
    """Number of slices: spread over the alpha half-life, bounded by the session."""
    horizon = min(max(time_to_close, 1.0), 2.0 * max(half_life, 1.0))
    return int(min(max(round(horizon / 15.0), 1), 12))


def schedule_order(
    parent_order: OrderIntent,
    market_state: dict,
    mode: str = "RESEARCH",
) -> list[ChildOrderPlan]:
    """
    Slice a parent `OrderIntent` into regime-aware child orders.

    ``market_state`` may carry: ``target_qty`` (or ``capital_gbp`` + ``price`` to
    derive it), ``max_participation``, ``spread_bps``, ``time_to_close``
    (minutes), and ``execution_regime``.

    ``mode`` is the explicit TRADING_MODE (Rule 7): it is never inferred. An
    unknown mode raises, and a LIVE order that has not passed the pre-trade risk
    gate (``risk_approved``) is refused before any child order is produced.
    """
    mode = normalize_mode(mode)
    parent_order.validate_for_mode(mode)

    side = parent_order.direction
    urgency = parent_order.urgency
    half_life = float(parent_order.alpha_half_life_minutes)
    regime = market_state.get("execution_regime", "normal_exec")
    is_derisk = urgency == "URGENT_DERISK"

    target_qty = market_state.get("target_qty")
    if target_qty is None:
        price = float(market_state.get("price", market_state.get("arrival_price", 0.0)) or 0.0)
        capital = float(market_state.get("capital_gbp", 0.0))
        target_qty = abs(parent_order.target_weight) * capital / price if price > 0 else 0.0
    qty = abs(float(target_qty))

    max_participation = float(market_state.get("max_participation", 0.05))
    spread_bps = float(market_state.get("spread_bps", 5.0))
    time_to_close = float(market_state.get("time_to_close", 390.0))

    # Size: stressed regime halves non-de-risking orders.
    if regime == "stressed_exec" and not is_derisk:
        qty *= 0.50

    # Order type: passive limits everywhere except an URGENT_DERISK order.
    if is_derisk:
        order_type, tag, limit_offset = "MARKET", "derisk", None
    else:
        order_type, tag = "LIMIT", "passive"
        # Post just inside the spread (passive), tighter when cautious.
        limit_offset = -(spread_bps / 2.0) if regime != "cautious_exec" else -(spread_bps / 4.0)

    # Participation: clamp to the regime ADV cap (5% normal/cautious, 2% stressed),
    # then cautious halves the pace. An URGENT_DERISK exit is exempt from the cap.
    if is_derisk:
        participation = max_participation
    else:
        participation = min(max_participation, REGIME_PARTICIPATION_CAP.get(regime, 0.05))
        if regime == "cautious_exec":
            participation *= 0.50

    n = 1 if is_derisk else _n_slices(time_to_close, half_life)
    slice_qty = qty / n if n > 0 else qty
    interval = time_to_close / n if n > 0 else 0.0

    return [
        ChildOrderPlan(
            symbol=parent_order.symbol, side=side, qty=slice_qty, order_type=order_type,
            limit_offset_bps=limit_offset, scheduled_offset_minutes=i * interval,
            participation=participation, slice_index=i, tag=tag,
        )
        for i in range(n)
    ]


# ── 16.5 Execution report ────────────────────────────────────────────────────────

@dataclass
class ExecutionReport:
    """Post-trade execution quality summary."""

    symbol: str
    order_id: str
    expected_cost_bps: float
    realized_cost_bps: float
    fill_rate: float
    avg_slippage_bps: float
    passive_fill_ratio: float
    implementation_shortfall_bps: float
    warnings: list[str] = field(default_factory=list)
    execution_regime_used: str = "normal_exec"


def _side_sign(direction: str) -> float:
    return 1.0 if str(direction).upper() == "BUY" else -1.0


def compute_execution_report(
    parent_order: OrderIntent,
    fills: list,
    target_qty: float,
    execution_regime: str = "normal_exec",
    expected_cost_bps: Optional[float] = None,
) -> ExecutionReport:
    """Build an `ExecutionReport` from the fills of a parent order."""
    sign = _side_sign(parent_order.direction)
    expected = float(expected_cost_bps if expected_cost_bps is not None else parent_order.expected_cost_bps)

    warnings: list[str] = []
    if not fills:
        warnings.append("NO_FILLS")
        return ExecutionReport(
            symbol=parent_order.symbol, order_id=getattr(parent_order, "order_id", parent_order.symbol),
            expected_cost_bps=expected, realized_cost_bps=0.0, fill_rate=0.0,
            avg_slippage_bps=0.0, passive_fill_ratio=0.0, implementation_shortfall_bps=0.0,
            warnings=warnings, execution_regime_used=execution_regime,
        )

    filled_qty = sum(abs(float(getattr(f, "qty", 0.0))) for f in fills)
    fill_rate = filled_qty / abs(target_qty) if target_qty else 0.0

    slippages = [float(getattr(f, "slippage_bps", 0.0)) for f in fills]
    avg_slippage = sum(slippages) / len(slippages)

    shortfalls: list[float] = []
    passive_flags: list[float] = []
    for f in fills:
        decision_price = float(getattr(f, "decision_price", 0.0)) or 1.0
        arrival_price = float(getattr(f, "arrival_price", decision_price)) or decision_price
        fill_price = float(getattr(f, "fill_price", decision_price))
        shortfalls.append(sign * (fill_price - decision_price) / decision_price * 10_000.0)
        passive_flags.append(1.0 if sign * (fill_price - arrival_price) <= 0.0 else 0.0)

    implementation_shortfall = sum(shortfalls) / len(shortfalls)
    passive_ratio = sum(passive_flags) / len(passive_flags)
    realized_cost = avg_slippage

    if fill_rate < 0.999:
        warnings.append("INCOMPLETE_FILL")
    if expected > 0 and realized_cost > 2.0 * expected:
        warnings.append("COST_OVERRUN")

    return ExecutionReport(
        symbol=parent_order.symbol,
        order_id=getattr(parent_order, "order_id", parent_order.symbol),
        expected_cost_bps=expected,
        realized_cost_bps=float(realized_cost),
        fill_rate=float(min(fill_rate, 1.0)),
        avg_slippage_bps=float(avg_slippage),
        passive_fill_ratio=float(passive_ratio),
        implementation_shortfall_bps=float(implementation_shortfall),
        warnings=warnings,
        execution_regime_used=execution_regime,
    )
