"""
Phase 6 — OrderManagerBrokerAdapter tests.

The adapter bridges a real ``BrokerProtocol`` broker (``PaperBroker`` / ``IBKRBroker``,
whose ``submit(child_plans: list, mode) -> list[FillEvent]``) onto the OrderManager
broker contract (``submit(order: dict, mode)`` for a SINGLE order, plus ``connected``).
It must be a pure shape translation that never weakens the wrapped broker's
LIVE-only safety.
"""

from __future__ import annotations

from broker.ibkr import IBKRBroker
from broker.paper import PaperBroker
from execution.broker_adapter import OrderManagerBrokerAdapter
from execution.order_lifecycle import OrderLifecycle, OrderStatus
from execution.order_manager import OrderManager
from ops.ledger import ImmutableLedger

TS = "2026-01-01T00:00:00"
ORDER = {"order_id": "o1", "symbol": "AAPL", "side": "BUY", "qty": 100}


class _SpyBroker:
    """Records the (child_plans, mode) it was asked to submit; returns preset fills."""

    def __init__(self, fills=None, connected=True, raises=False):
        self._fills = fills if fills is not None else []
        self.connected = connected
        self._raises = raises
        self.calls: list = []

    def submit(self, child_plans, mode):
        self.calls.append((list(child_plans), mode))
        if self._raises:
            raise TimeoutError("no acknowledgement from broker")
        return self._fills


def test_adapter_wraps_single_order_as_one_element_plan_list():
    spy = _SpyBroker()
    OrderManagerBrokerAdapter(spy).submit(ORDER, "PAPER")
    assert len(spy.calls) == 1
    child_plans, mode = spy.calls[0]
    assert mode == "PAPER"                       # mode forwarded verbatim
    assert len(child_plans) == 1                 # one dict order -> one child plan
    plan = child_plans[0]
    assert plan.symbol == "AAPL" and plan.side == "BUY" and plan.qty == 100.0


def test_paperbroker_through_adapter_returns_real_fills():
    adapter = OrderManagerBrokerAdapter(PaperBroker(prices={"AAPL": 190.0}))
    fills = adapter.submit({"order_id": "o1", "symbol": "AAPL", "side": "BUY", "qty": 10}, "PAPER")
    assert len(fills) == 1
    assert fills[0].symbol == "AAPL" and fills[0].qty == 10.0 and fills[0].fill_price > 0.0


def test_paperbroker_via_adapter_and_manager_reaches_filled():
    broker = OrderManagerBrokerAdapter(PaperBroker(prices={"AAPL": 190.0}))
    m = OrderManager(OrderLifecycle(), broker, mode="PAPER", ledger=ImmutableLedger())
    m.place("o1", "AAPL", "BUY", 100, TS)
    rec = m.lifecycle.get("o1")
    assert rec.status == OrderStatus.FILLED and rec.filled_qty == 100.0
    assert any(e.event_type == "FILL" for e in m.ledger.events())
    assert m.ledger.verify_chain()


def test_connected_delegates_to_wrapped_broker():
    assert OrderManagerBrokerAdapter(PaperBroker()).connected is True
    assert OrderManagerBrokerAdapter(_SpyBroker(connected=False)).connected is False


def test_ibkr_live_only_gate_preserved_through_adapter():
    adapter = OrderManagerBrokerAdapter(IBKRBroker())   # never connected in a unit test
    assert adapter.connected is False
    assert adapter.submit(ORDER, "PAPER") == []          # IBKR refuses non-LIVE
    assert adapter.submit(ORDER, "LIVE") == []           # disconnected -> nothing sent


def test_submit_exception_propagates_to_uncertain_through_manager():
    broker = OrderManagerBrokerAdapter(_SpyBroker(raises=True))
    m = OrderManager(OrderLifecycle(), broker, mode="PAPER", ledger=ImmutableLedger())
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.SUBMISSION_UNCERTAIN
    assert m.can_resubmit("o1") is False
