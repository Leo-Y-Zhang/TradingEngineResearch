"""
Phase 6 — order lifecycle state machine tests (directive §15 invariants).
"""

from __future__ import annotations

import pytest

from execution.order_lifecycle import InvalidTransition, OrderLifecycle, OrderStatus

TS = "2026-01-01T00:00:00"


def _to_working(lc: OrderLifecycle, oid: str = "o1", qty: float = 100.0):
    lc.create(oid, qty, TS)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition(oid, st, TS)
    return lc.get(oid)


def _to_submit_pending(lc: OrderLifecycle, oid: str, qty: float = 100.0):
    lc.create(oid, qty, TS)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING):
        lc.transition(oid, st, TS)
    return lc.get(oid)


class TestHappyPathAndTransitions:
    def test_happy_path_to_filled(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.record_fill("o1", "f1", 40, TS)
        assert lc.get("o1").status == OrderStatus.PARTIALLY_FILLED
        lc.record_fill("o1", "f2", 60, TS)
        rec = lc.get("o1")
        assert rec.status == OrderStatus.FILLED and rec.filled_qty == 100

    def test_invalid_transition_raises(self):
        lc = OrderLifecycle()
        lc.create("o1", 100, TS)
        with pytest.raises(InvalidTransition):
            lc.transition("o1", OrderStatus.FILLED, TS)


class TestFillInvariants:
    def test_idempotent_fill(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.record_fill("o1", "f1", 50, TS)
        lc.record_fill("o1", "f1", 50, TS)              # duplicate broker callback
        assert lc.get("o1").filled_qty == 50            # applied once

    def test_overfill_clamped(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.record_fill("o1", "f1", 150, TS)             # > approved 100
        rec = lc.get("o1")
        assert rec.filled_qty == 100 and rec.status == OrderStatus.FILLED
        assert "OVERFILL_CLAMPED" in rec.flags


class TestUncertaintyInvariants:
    def test_timeout_is_not_rejection(self):
        lc = OrderLifecycle()
        _to_submit_pending(lc, "o1")
        lc.mark_submission_uncertain("o1", TS)
        assert lc.get("o1").status == OrderStatus.SUBMISSION_UNCERTAIN
        assert lc.can_resubmit("o1") is False           # uncertain blocks resubmit

    def test_cancel_request_is_not_cancellation_and_fill_can_race(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.request_cancel("o1", TS)
        assert lc.get("o1").status == OrderStatus.CANCEL_PENDING   # NOT cancelled
        lc.record_fill("o1", "f1", 100, TS)             # a fill races the cancel
        assert lc.get("o1").status == OrderStatus.FILLED

    def test_broker_unknown_blocks_resubmit(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.mark_broker_unknown("o1", TS)
        assert lc.can_resubmit("o1") is False

    def test_can_resubmit_only_for_pre_broker_reject(self):
        lc = OrderLifecycle()
        lc.create("o1", 100, TS)
        lc.transition("o1", OrderStatus.VALIDATED, TS)
        lc.transition("o1", OrderStatus.REJECTED, TS)   # rejected before reaching broker
        assert lc.can_resubmit("o1") is True


class TestTerminalAndResync:
    def test_commission_attaches_in_terminal_state(self):
        lc = OrderLifecycle()
        _to_working(lc)
        lc.record_fill("o1", "f1", 100, TS)
        assert lc.get("o1").status == OrderStatus.FILLED
        lc.record_commission("o1", -1.25, TS)           # late commission on a terminal order
        assert lc.get("o1").commissions[0]["amount"] == -1.25

    def test_resync_resolves_uncertain_orders(self):
        lc = OrderLifecycle()
        _to_submit_pending(lc, "o1")
        lc.mark_submission_uncertain("o1", TS)
        _to_submit_pending(lc, "o2", 50)
        lc.mark_submission_uncertain("o2", TS)
        changed = lc.resync({"o1": "WORKING"}, TS)      # broker knows o1, not o2
        assert lc.get("o1").status == OrderStatus.WORKING
        assert lc.get("o2").status == OrderStatus.RECONCILIATION_HOLD
        assert set(changed) == {"o1", "o2"}
