"""
Phase 6 — OrderManager tests (lifecycle + ledger + broker, §15 safety flows).
"""

from __future__ import annotations

from execution.order_lifecycle import OrderLifecycle, OrderStatus
from execution.order_manager import OrderManager
from ops.ledger import ImmutableLedger

TS = "2026-01-01T00:00:00"


class _FakeBroker:
    def __init__(self, fills=None, connected=True, raises=False):
        self._fills = fills or []
        self.connected = connected
        self._raises = raises

    def submit(self, order, mode):
        if self._raises:
            raise TimeoutError("no acknowledgement from broker")
        return self._fills


def _mgr(fills=None, connected=True, raises=False, with_ledger=True) -> OrderManager:
    ledger = ImmutableLedger() if with_ledger else None
    return OrderManager(OrderLifecycle(), _FakeBroker(fills, connected, raises),
                        mode="PAPER", ledger=ledger)


def test_full_fill_to_filled_with_trail():
    m = _mgr(fills=[{"fill_id": "f1", "qty": 100, "fill_price": 190.0}])
    m.place("o1", "AAPL", "BUY", 100, TS)
    rec = m.lifecycle.get("o1")
    assert rec.status == OrderStatus.FILLED and rec.filled_qty == 100
    types = [e.event_type for e in m.ledger.events()]
    assert "ORDER_INTENT" in types and "SUBMISSION" in types and "FILL" in types
    assert m.ledger.verify_chain()


def test_partial_fill():
    m = _mgr(fills=[{"fill_id": "f1", "qty": 40}])
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.PARTIALLY_FILLED


def test_no_fills_is_working_acked():
    m = _mgr(fills=[])
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.WORKING


def test_submit_timeout_is_uncertain_not_rejected():
    m = _mgr(raises=True)
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.SUBMISSION_UNCERTAIN
    assert m.can_resubmit("o1") is False
    assert any(e.event_type == "SUBMISSION_UNCERTAIN" for e in m.ledger.events())


def test_disconnected_broker_is_unknown_no_resubmit():
    m = _mgr(connected=False)
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.BROKER_UNKNOWN
    assert m.can_resubmit("o1") is False


def test_duplicate_fill_callback_is_idempotent():
    m = _mgr(fills=[{"fill_id": "f1", "qty": 60}, {"fill_id": "f1", "qty": 60}])
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").filled_qty == 60          # applied once


def test_reconcile_open_orders_resyncs():
    m = _mgr(raises=True)
    m.place("o1", "AAPL", "BUY", 100, TS)                  # -> uncertain
    changed = m.reconcile_open_orders({"o1": "WORKING"}, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.WORKING
    assert "o1" in changed


def test_no_ledger_does_not_crash():
    m = _mgr(fills=[{"fill_id": "f1", "qty": 100}], with_ledger=False)
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").status == OrderStatus.FILLED


def test_place_returns_record_and_broker_fills():
    # additive contract the engine STEP-12 relies on: (record, broker_fills)
    m = _mgr(fills=[{"fill_id": "f1", "qty": 100, "fill_price": 190.0}])
    rec, fills = m.place("o1", "AAPL", "BUY", 100, TS)
    assert rec is m.lifecycle.get("o1")
    assert [f["fill_id"] for f in fills] == ["f1"]            # the broker's real fills, threaded back


def test_place_returns_empty_fills_on_disconnect_timeout_and_ack():
    rec, fills = _mgr(connected=False).place("o1", "AAPL", "BUY", 100, TS)
    assert rec.status == OrderStatus.BROKER_UNKNOWN and fills == []        # disconnect
    rec, fills = _mgr(raises=True).place("o2", "AAPL", "BUY", 100, TS)
    assert rec.status == OrderStatus.SUBMISSION_UNCERTAIN and fills == []  # timeout
    rec, fills = _mgr(fills=[]).place("o3", "AAPL", "BUY", 100, TS)
    assert rec.status == OrderStatus.WORKING and fills == []              # acked, no fills


def test_place_attaches_broker_reported_commission_to_lifecycle():
    # §17 cash leg (a): a broker fill carrying a commission attaches it to the order record
    # (§15: even a terminal order accepts it) and audits it on the manager's own FILL event.
    m = _mgr(fills=[{"fill_id": "f1", "qty": 100, "fill_price": 190.0, "commission": 1.3}])
    m.place("o1", "AAPL", "BUY", 100, TS)
    rec = m.lifecycle.get("o1")
    assert rec.status == OrderStatus.FILLED
    assert [c["amount"] for c in rec.commissions] == [1.3]
    assert m.ledger.events("FILL")[0].payload.get("commission") == 1.3


def test_place_without_commission_attaches_nothing():
    m = _mgr(fills=[{"fill_id": "f1", "qty": 100, "fill_price": 190.0}])
    m.place("o1", "AAPL", "BUY", 100, TS)
    assert m.lifecycle.get("o1").commissions == []          # nothing reported -> nothing invented
