"""
Phase 6(c) LIVE6B-3 — reconnect-resync machinery.
  • Slice 2: id plumbing (order_ref carried to the broker; broker_order_id captured back;
    symbol/side/ref_price recorded on the OrderRecord). Observably inert.
  • Slice 3: key-correct broker.open_orders() -> reconcile -> resync (added below later).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from broker.ibkr import order_params_from_plan
from broker.paper import PaperBroker
from execution.broker_adapter import OrderManagerBrokerAdapter, plan_from_order
from execution.order_lifecycle import OrderLifecycle, OrderStatus
from execution.order_manager import OrderManager

TS = "2024-01-01T00:00:00"


class _IdBroker:
    """OrderManager-contract broker that reports a broker-assigned id per submit."""
    connected = True

    def __init__(self, fills=None):
        self._fills = fills if fills is not None else []
        self.last_broker_order_ids: dict = {}

    def submit(self, order, mode):
        self.last_broker_order_ids = {order["order_id"]: "BRK-123"}
        return self._fills


# ── slice 2: id plumbing ──────────────────────────────────────────────────────────────

def test_plan_from_order_carries_order_ref_from_id():
    assert plan_from_order({"order_id": "OID", "symbol": "AAPL", "side": "BUY", "qty": 10}).order_ref == "OID"


def test_order_params_from_plan_carries_order_ref_limit_and_market():
    lmt = SimpleNamespace(symbol="AAPL", side="BUY", qty=10.0, order_type="LIMIT",
                          limit_offset_bps=5.0, order_ref="OID")
    assert order_params_from_plan(lmt, 100.0)["order_ref"] == "OID"
    mkt = SimpleNamespace(symbol="AAPL", side="BUY", qty=10.0, order_type="MARKET", order_ref="OID2")
    assert order_params_from_plan(mkt, 100.0)["order_ref"] == "OID2"


def test_place_records_symbol_side_ref_price():
    m = OrderManager(OrderLifecycle(), _IdBroker(fills=[{"fill_id": "f1", "qty": 10}]), mode="LIVE")
    m.place("o1", "AAPL", "BUY", 10, TS, ref_price=99.5)
    rec = m.lifecycle.get("o1")
    assert rec.symbol == "AAPL" and rec.side == "BUY" and rec.ref_price == pytest.approx(99.5)


def test_place_captures_broker_order_id_on_fill():
    m = OrderManager(OrderLifecycle(), _IdBroker(fills=[{"fill_id": "f1", "qty": 10}]), mode="LIVE")
    m.place("o1", "AAPL", "BUY", 10, TS)
    assert m.lifecycle.get("o1").broker_order_id == "BRK-123"


def test_place_captures_broker_order_id_on_ack_no_fill():
    m = OrderManager(OrderLifecycle(), _IdBroker(fills=[]), mode="LIVE")   # acked, no fills yet
    m.place("o1", "AAPL", "BUY", 10, TS)
    rec = m.lifecycle.get("o1")
    assert rec.status == OrderStatus.WORKING and rec.broker_order_id == "BRK-123"


def test_paper_broker_populates_last_broker_order_ids_and_fills_unchanged():
    pb = PaperBroker(prices={"AAPL": 100.0})
    plan = plan_from_order({"order_id": "OID", "symbol": "AAPL", "side": "BUY", "qty": 10})
    fills = pb.submit([plan], "PAPER")
    assert len(fills) == 1 and fills[0].symbol == "AAPL"          # fills byte-identical
    assert pb.last_broker_order_ids.get("OID") == fills[0].order_id


def test_adapter_delegates_last_broker_order_ids():
    pb = PaperBroker(prices={"AAPL": 100.0})
    adapter = OrderManagerBrokerAdapter(pb)
    adapter.submit({"order_id": "OID", "symbol": "AAPL", "side": "BUY", "qty": 10}, "PAPER")
    assert adapter.last_broker_order_ids.get("OID") is not None


# ── slice 3: key-correct open_orders -> reconcile -> resync ────────────────────────────

def _uncertain(lc: OrderLifecycle, oid="o1", symbol="AAPL", side="BUY"):
    lc.create(oid, 100.0, TS, symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING):
        lc.transition(oid, st, TS)
    lc.mark_submission_uncertain(oid, TS)
    return lc.get(oid)


def _working(lc: OrderLifecycle, oid="o1", symbol="AAPL", side="BUY"):
    lc.create(oid, 100.0, TS, symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition(oid, st, TS)
    return lc.get(oid)


def _entry(order_ref, status, broker_order_id="B1", symbol="AAPL", filled_qty=0.0):
    return {"order_ref": order_ref, "broker_order_id": broker_order_id,
            "status": status, "symbol": symbol, "filled_qty": filled_qty}


def test_reconcile_list_by_order_ref_resolves_and_enriches():
    lc = OrderLifecycle()
    _uncertain(lc, "o1")
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    changed = m.reconcile_open_orders([_entry("o1", "WORKING", broker_order_id="B7")], TS)
    assert "o1" in changed and lc.get("o1").status == OrderStatus.WORKING
    assert lc.get("o1").broker_order_id == "B7"                # enriched from broker truth


def test_reconcile_resolves_working_order_to_cancelled():
    lc = OrderLifecycle()
    _working(lc, "o1")
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    changed = m.reconcile_open_orders([_entry("o1", "CANCELLED")], TS)
    assert "o1" in changed and lc.get("o1").status == OrderStatus.CANCELLED


def test_reconcile_matches_by_broker_order_id_fallback():
    lc = OrderLifecycle()
    _uncertain(lc, "o1")
    lc.set_broker_order_id("o1", "BROKERX", TS)
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    # broker entry whose order_ref did NOT round-trip — match on broker_order_id
    changed = m.reconcile_open_orders([_entry(None, "WORKING", broker_order_id="BROKERX", filled_qty=0.0)], TS)
    assert "o1" in changed and lc.get("o1").status == OrderStatus.WORKING


def test_reconcile_broker_fill_discrepancy_parks_hold():
    # review fix: the broker filled MORE than we locally booked -> record the true fill and PARK
    # for operator reconciliation (never silently resolve to FILLED + prune, losing the position).
    lc = OrderLifecycle()
    _working(lc, "o1")
    lc.record_fill("o1", "f", 40.0, TS)                       # locally 40/100 (PARTIALLY_FILLED)
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders([_entry("o1", "FILLED", filled_qty=70.0)], TS)   # broker says 70 filled
    rec = lc.get("o1")
    assert rec.status == OrderStatus.RECONCILIATION_HOLD     # parked, not silently FILLED
    assert rec.filled_qty == pytest.approx(70.0)             # true fill recorded -> overlay residual correct


def test_reconcile_unmatched_nonterminal_parks_hold():
    lc = OrderLifecycle()
    _uncertain(lc, "o1")
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders([], TS)                            # broker has no record
    assert lc.get("o1").status == OrderStatus.RECONCILIATION_HOLD


def test_reconcile_legacy_dict_passthrough_still_works():
    lc = OrderLifecycle()
    _uncertain(lc, "o1")
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders({"o1": "WORKING"}, TS)             # legacy already-collapsed map
    assert lc.get("o1").status == OrderStatus.WORKING


def test_reconcile_never_submits():
    class _NoSubmit:
        connected = True
        def __init__(self): self.calls = 0
        def submit(self, *a):
            self.calls += 1
            return []
    lc = OrderLifecycle()
    _uncertain(lc, "o1")
    b = _NoSubmit()
    OrderManager(lc, b, mode="LIVE").reconcile_open_orders([_entry("o1", "WORKING")], TS)
    assert b.calls == 0                                        # resync is read-only — never submits


def test_paper_broker_open_orders_is_empty():
    pb = PaperBroker(prices={"AAPL": 100.0})
    assert pb.open_orders(datetime(2024, 1, 1, tzinfo=timezone.utc)) == []


def test_ibkr_open_orders_disconnected_is_empty():
    from broker.ibkr import IBKRBroker
    assert IBKRBroker().open_orders(datetime(2024, 1, 1, tzinfo=timezone.utc)) == []  # fail-closed read


# ── review fixes (verification round 2) ───────────────────────────────────────────────

def test_map_ib_status_pendingcancel_is_not_terminal():
    from broker.ibkr import _map_ib_status
    assert _map_ib_status("PendingCancel") == "WORKING"   # cancel UNconfirmed -> the order can still fill
    assert _map_ib_status("Cancelled") == "CANCELLED"
    assert _map_ib_status("ApiCancelled") == "CANCELLED"


def test_resync_does_not_park_resting_order_on_transient_omission():
    lc = OrderLifecycle()
    _working(lc, "w1")                                     # a resting WORKING order
    _uncertain(lc, "u1")                                   # a genuinely-uncertain order
    OrderManager(lc, _IdBroker(), mode="LIVE").reconcile_open_orders([], TS)   # broker lists NOTHING
    assert lc.get("w1").status == OrderStatus.WORKING      # resting order left recoverable, NOT bricked
    assert lc.get("u1").status == OrderStatus.RECONCILIATION_HOLD  # uncertain still parks (fail-closed)


# ── slice 4 (held-book reconciliation): surface discovered disconnect-fills ────────────
# A reconnect resync that finds the broker filled MORE than we locally booked records the
# unbooked delta on the manager's outbox so the run-loop can raise an OPEN reconciliation
# item; detection still parks RECONCILIATION_HOLD (the symbol stays frozen until resolved).

def test_reconcile_records_discovered_disconnect_fill():
    lc = OrderLifecycle()
    _working(lc, "o1", "AAPL", "BUY")
    lc.record_fill("o1", "f", 40.0, TS)                     # locally booked 40/100
    lc.get("o1").ref_price = 99.0
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders([_entry("o1", "FILLED", filled_qty=70.0)], TS)   # broker says 70 filled
    assert lc.get("o1").status == OrderStatus.RECONCILIATION_HOLD
    assert len(m.discovered_fills) == 1
    d = m.discovered_fills[0]
    assert d["order_id"] == "o1" and d["symbol"] == "AAPL" and d["side"] == "BUY"
    assert d["delta_qty"] == pytest.approx(30.0)            # 70 broker - 40 locally booked
    assert d["broker_filled_qty"] == pytest.approx(70.0)
    assert d["ref_price"] == pytest.approx(99.0)


def test_no_discovered_fill_when_broker_matches_local():
    lc = OrderLifecycle()
    _working(lc, "o1")
    lc.record_fill("o1", "f", 70.0, TS)                     # locally 70/100
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders([_entry("o1", "WORKING", filled_qty=70.0)], TS)  # broker also 70
    assert m.discovered_fills == []                         # nothing new discovered


def test_discovered_fill_carries_broker_avg_fill_price():
    # the booked FILL should use the broker's TRUE avg fill price when reported, not the
    # placement-time ref_price estimate (cash/NAV honesty).
    lc = OrderLifecycle()
    _working(lc, "o1", "AAPL", "BUY")
    lc.record_fill("o1", "f", 40.0, TS)
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    entry = {"order_ref": "o1", "broker_order_id": "B", "status": "FILLED",
             "symbol": "AAPL", "filled_qty": 70.0, "avg_fill_price": 101.5}
    m.reconcile_open_orders([entry], TS)
    assert m.discovered_fills[0]["avg_fill_price"] == pytest.approx(101.5)


def test_discovered_fill_avg_price_absent_is_none():
    lc = OrderLifecycle()
    _working(lc, "o1")
    lc.record_fill("o1", "f", 40.0, TS)
    m = OrderManager(lc, _IdBroker(), mode="LIVE")
    m.reconcile_open_orders([_entry("o1", "FILLED", filled_qty=70.0)], TS)   # entry has no avg_fill_price
    assert m.discovered_fills[0].get("avg_fill_price") is None
