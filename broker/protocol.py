"""
TradingEngineResearch — Broker Protocol
===========================
The structural interface every broker adapter implements. The engine depends
on this protocol, never on a concrete broker: LIVE is the only mode that may
reach one (golden rule 1), and STEP 12 calls exactly ``submit``.

Implementations: ``broker.paper.PaperBroker`` (deterministic local fills, zero
market access — the LIVE-code-path stand-in) and ``broker.ibkr.IBKRBroker``
(Interactive Brokers via ib-insync).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from data.data_contracts import BrokerOpenOrder, BrokerState, FillEvent

__all__ = ["BrokerProtocol"]


@runtime_checkable
class BrokerProtocol(Protocol):
    """What the engine (STEP 12) and the run-loop require of a broker."""

    @property
    def connected(self) -> bool:
        """True only when the adapter has a live session it can trade through."""
        ...

    def submit(self, child_plans: list, mode: str) -> list[FillEvent]:
        """Submit child-order plans; return the resulting fills."""
        ...

    def account_state(self, asof_time: datetime) -> BrokerState:
        """The broker's current view of the account (validated per mode by the caller)."""
        ...

    def open_orders(self, asof_time: datetime) -> list[BrokerOpenOrder]:
        """Read-only snapshot of resting (open) orders for reconnect resync (LIVE6B-3): a
        list of ``{order_ref, broker_order_id, status, symbol, filled_qty}`` entries, each
        optionally carrying ``avg_fill_price`` (the broker's TRUE avg fill price for the filled
        portion, preferred over ref_price when a disconnect-fill is booked; omit/None if unknown).
        NEVER places an order (golden rule 1)."""
        ...
