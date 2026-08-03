"""
Phase 6(c) LIVE6B-2 — order-lifecycle persistence: OrderRecord / OrderLifecycle JSON
round-trips so an uncertain/working LIVE order is REMEMBERED across a restart, not forgotten.
(Engine snapshot/restore + the run-loop restart wiring are covered in test_step12_lifecycle
and test_run_loop.)
"""

from __future__ import annotations

from execution.order_lifecycle import OrderLifecycle, OrderRecord, OrderStatus, TERMINAL_STATES

TS = "2024-01-01T00:00:00"


def _uncertain(lc: OrderLifecycle, oid="o1", symbol="AAPL", side="BUY", qty=100.0):
    lc.create(oid, qty, TS, symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING):
        lc.transition(oid, st, TS)
    lc.mark_submission_uncertain(oid, TS)
    return lc.get(oid)


def _filled(lc: OrderLifecycle, oid="t1", symbol="MSFT", side="BUY", qty=10.0):
    lc.create(oid, qty, TS, symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition(oid, st, TS)
    lc.record_fill(oid, "f", qty, TS)
    return lc.get(oid)


def test_order_record_json_roundtrip_preserves_state():
    lc = OrderLifecycle()
    rec = _uncertain(lc, "o1")
    lc.set_broker_order_id("o1", "B1", TS)
    rec._seen_fills.add("f1")                                  # an applied fill id
    back = OrderRecord.from_json(rec.to_json())
    assert back.order_id == "o1" and back.status == OrderStatus.SUBMISSION_UNCERTAIN
    assert back.symbol == "AAPL" and back.side == "BUY" and back.broker_order_id == "B1"
    assert back.approved_qty == rec.approved_qty and back.filled_qty == rec.filled_qty
    assert back._seen_fills == {"f1"}                          # idempotency survives the restart
    assert back.history == rec.history


def test_from_json_unknown_status_parks_hold():
    rec = OrderRecord.from_json({"order_id": "x", "approved_qty": 10.0, "status": "BOGUS"})
    assert rec.status == OrderStatus.RECONCILIATION_HOLD       # fail-closed, non-terminal


def test_snapshot_nonterminal_excludes_terminal():
    lc = OrderLifecycle()
    _uncertain(lc, "u1")
    _filled(lc, "t1")                                          # terminal -> not persisted
    snap = lc.snapshot_nonterminal()
    assert [s["order_id"] for s in snap] == ["u1"]


def test_lifecycle_restore_rebuilds_nonterminal():
    lc = OrderLifecycle()
    _uncertain(lc, "u1")
    lc2 = OrderLifecycle()
    lc2.restore(lc.snapshot_nonterminal())
    assert lc2.get("u1").status == OrderStatus.SUBMISSION_UNCERTAIN
    assert lc2.get("u1").status not in TERMINAL_STATES


def test_lifecycle_restore_skips_malformed_record():
    lc = OrderLifecycle()
    lc.restore([{"approved_qty": 10.0, "status": "WORKING"}])  # missing order_id -> skipped
    assert lc.all() == []
