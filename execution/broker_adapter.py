"""
TradingEngineResearch — OrderManager ⇄ broker adapter (Phase 6 wiring)
=============================================================
Bridges a real :class:`~broker.protocol.BrokerProtocol` broker
(:class:`~broker.paper.PaperBroker` / :class:`~broker.ibkr.IBKRBroker`) onto the
contract the :class:`~execution.order_manager.OrderManager` expects of a broker.

The mismatch this closes:
  • OrderManager calls ``broker.submit(order: dict, mode)`` with **one** order dict
    (``{order_id, symbol, side, qty, …}``) and reads ``broker.connected``.
  • A real broker exposes ``submit(child_plans: list, mode) -> list[FillEvent]``
    (a *list* of child-order plans) and a ``connected`` property.

The adapter is a **pure shape translation**: it wraps the single order dict in a
one-element ``[plan]`` list whose element carries the attributes both real brokers
read (``symbol``/``side``/``qty``/``order_type``/``limit_offset_bps``), forwards
``mode`` **unchanged**, and returns the broker's ``FillEvent``s (which the
OrderManager reads via ``order_id``/``qty``/``fill_price``). It adds **no** market
access and **no** mode logic of its own — the LIVE-only gate (golden rule 1) stays
entirely inside the wrapped broker (``IBKRBroker.submit`` still refuses any non-LIVE
mode and any disconnected session). A submit that raises (timeout/network) is left
to propagate so the OrderManager records ``SUBMISSION_UNCERTAIN`` (never a blind
rejection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["OrderManagerBrokerAdapter", "plan_from_order"]


@dataclass(frozen=True)
class _OrderPlan:
    """The minimal child-plan view both real brokers consume via ``getattr``."""

    symbol: str
    side: str
    qty: float
    order_type: str = "LIMIT"
    limit_offset_bps: Optional[float] = None
    order_ref: Optional[str] = None


def plan_from_order(order: dict) -> _OrderPlan:
    """Translate an OrderManager order dict into a single broker child plan.

    Only ``symbol``/``side``/``qty`` are supplied by the OrderManager; ``order_type``
    and ``limit_offset_bps`` accept optional overrides and otherwise default to a
    passive ``LIMIT`` at the reference price.
    """
    return _OrderPlan(
        symbol=str(order.get("symbol", "")),
        side=str(order.get("side", "BUY")).upper(),
        qty=float(order.get("qty", 0.0) or 0.0),
        order_type=str(order.get("order_type", "LIMIT")).upper(),
        limit_offset_bps=order.get("limit_offset_bps"),
        order_ref=order.get("order_ref") or order.get("order_id"),
    )


class OrderManagerBrokerAdapter:
    """Adapt a ``BrokerProtocol`` broker to the OrderManager broker contract."""

    def __init__(self, broker: Any) -> None:
        self._broker = broker

    @property
    def connected(self) -> bool:
        return bool(self._broker.connected)

    @property
    def last_broker_order_ids(self) -> dict:
        """The wrapped broker's order_ref -> broker-order-id map from its last submit
        (LIVE6B-3), so the OrderManager can record each order's broker-assigned id."""
        return getattr(self._broker, "last_broker_order_ids", {})

    def submit(self, order: dict, mode: str) -> list:
        """Wrap the single ``order`` dict in a one-element plan list and delegate to
        the wrapped broker's ``submit``; return its fills unchanged. Any exception
        (timeout/network) propagates by design."""
        return list(self._broker.submit([plan_from_order(order)], mode))
