"""
TradingEngineResearch — Order lifecycle state machine (Phase 6 / directive §15)
=====================================================================
A persistent, idempotent order-state manager that safely handles the *uncertain*
realities of live broker interaction — the gaps the simpler `execution_engine`
state machine could not express (audit EXEC-1/2/3).

Hard invariants (directive §15 / §7.3), each enforced + tested:
  • **Only approved transitions** occur (others raise ``InvalidTransition``).
  • **Idempotent fills** — a fill with an already-seen ``fill_id`` is a no-op, so a
    duplicate broker callback cannot duplicate financial effect.
  • **filled_qty never exceeds approved_qty** — an over-fill is clamped + flagged.
  • **A timeout is NOT a rejection** → ``SUBMISSION_UNCERTAIN`` (resubmission blocked).
  • **A cancel request is NOT a cancellation** → ``CANCEL_PENDING`` (only a broker
    confirmation reaches ``CANCELLED``); a fill can still race in.
  • **Unknown broker state blocks resubmission** (``can_resubmit`` fails closed).
  • **A terminal order still accepts late commissions/corrections** (never discarded).
  • **Reconnect → resynchronisation** reconciles uncertain/unknown orders against the
    broker's open-order truth.

This module records lifecycle *facts*; persisting them belongs in the immutable
ledger (`ops.ledger`) and submitting belongs in the broker adapter — kept separate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "OrderStatus", "OrderRecord", "OrderLifecycle", "InvalidTransition",
    "TERMINAL_STATES", "RESUBMIT_BLOCKING_STATES", "MAX_RETAINED_TERMINAL_ORDERS",
]

_EPS = 1e-9

# LIVE6B-4: cap the in-memory retention of TERMINAL orders over a long LIVE run. Only
# fully-resolved (FILLED/CANCELLED/REJECTED/EXPIRED) records are ever evicted — their
# economic effect is already in the held book + the immutable ledger; non-terminal /
# uncertain orders are NEVER pruned.
MAX_RETAINED_TERMINAL_ORDERS = 5000


class InvalidTransition(RuntimeError):
    """Raised on an unapproved order-state transition."""


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMIT_PENDING = "SUBMIT_PENDING"
    SUBMISSION_UNCERTAIN = "SUBMISSION_UNCERTAIN"   # timeout / no ack — NOT a rejection
    ACKED = "ACKED"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"               # cancel requested — NOT yet cancelled
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    BROKER_UNKNOWN = "BROKER_UNKNOWN"               # cannot determine broker-side state
    RECONCILIATION_HOLD = "RECONCILIATION_HOLD"     # parked pending manual/three-way resolution


TERMINAL_STATES = frozenset({
    OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
})

# States from which RE-SUBMITTING the order is unsafe (it may be live at the broker).
RESUBMIT_BLOCKING_STATES = frozenset({
    OrderStatus.SUBMIT_PENDING, OrderStatus.SUBMISSION_UNCERTAIN, OrderStatus.ACKED,
    OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCEL_PENDING,
    OrderStatus.BROKER_UNKNOWN, OrderStatus.RECONCILIATION_HOLD,
})

# Non-terminal states a reconnect resync may resolve against broker truth: uncertain/unknown
# AND resting (acked/working/partially-filled), so a broker-confirmed cancel/fill of a resting
# order is reflected and never leaves it stuck non-terminal blocking its symbol (LIVE6B-3).
_RESYNC_RESOLVABLE = frozenset({
    OrderStatus.SUBMISSION_UNCERTAIN, OrderStatus.BROKER_UNKNOWN,
    OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED,
})

_RESYNC_STATUS_MAP = {
    "WORKING": OrderStatus.WORKING, "ACKED": OrderStatus.ACKED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED, "FILLED": OrderStatus.FILLED,
    "CANCELLED": OrderStatus.CANCELLED, "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED, "RECONCILIATION_HOLD": OrderStatus.RECONCILIATION_HOLD,
}

_ALLOWED: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset({OrderStatus.VALIDATED, OrderStatus.REJECTED}),
    OrderStatus.VALIDATED: frozenset({OrderStatus.RISK_APPROVED, OrderStatus.REJECTED}),
    OrderStatus.RISK_APPROVED: frozenset({OrderStatus.SUBMIT_PENDING, OrderStatus.REJECTED}),
    OrderStatus.SUBMIT_PENDING: frozenset({
        OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.SUBMISSION_UNCERTAIN,
        OrderStatus.REJECTED, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED,
        OrderStatus.BROKER_UNKNOWN}),
    OrderStatus.SUBMISSION_UNCERTAIN: frozenset({
        OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.BROKER_UNKNOWN, OrderStatus.RECONCILIATION_HOLD, OrderStatus.REJECTED,
        OrderStatus.CANCELLED, OrderStatus.EXPIRED}),
    OrderStatus.ACKED: frozenset({
        OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCEL_PENDING, OrderStatus.CANCELLED, OrderStatus.REJECTED,
        OrderStatus.EXPIRED, OrderStatus.BROKER_UNKNOWN, OrderStatus.RECONCILIATION_HOLD}),
    OrderStatus.WORKING: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING,
        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
        OrderStatus.BROKER_UNKNOWN, OrderStatus.RECONCILIATION_HOLD}),
    OrderStatus.PARTIALLY_FILLED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING,
        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
        OrderStatus.BROKER_UNKNOWN, OrderStatus.RECONCILIATION_HOLD}),
    OrderStatus.CANCEL_PENDING: frozenset({
        OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.EXPIRED, OrderStatus.BROKER_UNKNOWN}),
    OrderStatus.BROKER_UNKNOWN: frozenset({
        OrderStatus.ACKED, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED,
        OrderStatus.RECONCILIATION_HOLD}),
    OrderStatus.RECONCILIATION_HOLD: frozenset({
        OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


@dataclass
class OrderRecord:
    order_id: str
    approved_qty: float
    status: OrderStatus = OrderStatus.CREATED
    filled_qty: float = 0.0
    broker_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    ref_price: float = 0.0
    flags: list[str] = field(default_factory=list)
    commissions: list[dict] = field(default_factory=list)
    _seen_fills: set[str] = field(default_factory=set)
    history: list[tuple[str, str, str]] = field(default_factory=list)  # (timestamp, status, note)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATES

    @property
    def remaining_qty(self) -> float:
        return max(self.approved_qty - self.filled_qty, 0.0)

    def to_json(self) -> dict:
        """Serialise this order so an uncertain/working LIVE order survives a restart (LIVE6B-2)."""
        return {
            "order_id": self.order_id, "approved_qty": self.approved_qty,
            "status": self.status.value, "filled_qty": self.filled_qty,
            "broker_order_id": self.broker_order_id, "symbol": self.symbol,
            "side": self.side, "ref_price": self.ref_price,
            "flags": list(self.flags), "commissions": list(self.commissions),
            "seen_fills": sorted(self._seen_fills), "history": [list(t) for t in self.history],
        }

    @classmethod
    def from_json(cls, data: dict) -> "OrderRecord":
        try:
            status = OrderStatus(str(data.get("status", "")))
        except ValueError:
            status = OrderStatus.RECONCILIATION_HOLD          # unknown status -> fail-closed (non-terminal)
        rec = cls(
            order_id=str(data["order_id"]), approved_qty=float(data["approved_qty"]),
            status=status, filled_qty=float(data.get("filled_qty", 0.0)),
            broker_order_id=data.get("broker_order_id"),
            symbol=str(data.get("symbol", "")), side=str(data.get("side", "")),
            ref_price=float(data.get("ref_price", 0.0)),
            flags=list(data.get("flags") or []),
            commissions=list(data.get("commissions") or []),
        )
        rec._seen_fills = set(data.get("seen_fills") or [])    # idempotent-fill dedup survives restart
        rec.history = [tuple(t) for t in (data.get("history") or [])]
        return rec


class OrderLifecycle:
    """Owns a set of orders and processes lifecycle events idempotently + safely."""

    def __init__(self) -> None:
        self._orders: dict[str, OrderRecord] = {}

    # ── access ──────────────────────────────────────────────────────────────────
    def get(self, order_id: str) -> OrderRecord:
        return self._orders[order_id]

    def all(self) -> list[OrderRecord]:
        return list(self._orders.values())

    def create(self, order_id: str, approved_qty: float, timestamp: str, *,
               symbol: str = "", side: str = "", ref_price: float = 0.0) -> OrderRecord:
        if order_id in self._orders:
            raise ValueError(f"order {order_id!r} already exists")
        if not (approved_qty > 0):
            raise ValueError("approved_qty must be > 0")
        rec = OrderRecord(order_id=order_id, approved_qty=float(approved_qty),
                          symbol=str(symbol), side=str(side), ref_price=float(ref_price))
        rec.history.append((timestamp, rec.status.value, "created"))
        self._orders[order_id] = rec
        return rec

    def set_broker_order_id(self, order_id: str, broker_order_id: str, timestamp: str) -> OrderRecord:
        """Record the broker's OWN order id for this order (captured at ack/fill) so a
        later reconnect resync can map broker open-order truth back to our record."""
        rec = self._orders[order_id]
        rec.broker_order_id = str(broker_order_id)
        rec.history.append((timestamp, rec.status.value, f"broker_order_id={broker_order_id}"))
        return rec

    def reconcile_broker_fill(self, order_id: str, broker_filled_qty: float, timestamp: str) -> OrderRecord:
        """Record a broker-reported filled quantity discovered at RESYNC (e.g. a fill that
        landed during a disconnect) WITHOUT a status transition: clamp to approved_qty and log
        it, so the pending-overlay residual (approved - filled) stays correct. The caller parks
        the order RECONCILIATION_HOLD so the not-locally-booked quantity is surfaced for operator
        reconciliation rather than silently resolved/pruned — never lose a real fill."""
        rec = self._orders[order_id]
        new_filled = min(float(broker_filled_qty), rec.approved_qty)
        if new_filled > rec.filled_qty + _EPS:
            rec.history.append((timestamp, rec.status.value,
                                f"resync: broker filled {new_filled} (was {rec.filled_qty})"))
            rec.filled_qty = new_filled
        return rec

    # ── core transition ─────────────────────────────────────────────────────────
    def _transition(self, rec: OrderRecord, to: OrderStatus, timestamp: str, note: str = "") -> None:
        if to == rec.status:
            return
        if to not in _ALLOWED.get(rec.status, frozenset()):
            raise InvalidTransition(f"{rec.order_id}: {rec.status.value} → {to.value} not allowed")
        rec.status = to
        rec.history.append((timestamp, to.value, note))

    def transition(self, order_id: str, to: OrderStatus, timestamp: str, note: str = "") -> OrderRecord:
        rec = self._orders[order_id]
        self._transition(rec, to, timestamp, note)
        return rec

    # ── event handlers (the safety-critical ones) ─────────────────────────────────
    def mark_submission_uncertain(self, order_id: str, timestamp: str) -> OrderRecord:
        """A submit that timed out / got no ack. NOT a rejection — the order may be live."""
        rec = self._orders[order_id]
        self._transition(rec, OrderStatus.SUBMISSION_UNCERTAIN, timestamp, "submit timeout — uncertain")
        return rec

    def mark_broker_unknown(self, order_id: str, timestamp: str) -> OrderRecord:
        rec = self._orders[order_id]
        self._transition(rec, OrderStatus.BROKER_UNKNOWN, timestamp, "broker state unknown")
        return rec

    def request_cancel(self, order_id: str, timestamp: str) -> OrderRecord:
        """Record a cancel REQUEST — does not cancel; only a broker confirm does."""
        rec = self._orders[order_id]
        self._transition(rec, OrderStatus.CANCEL_PENDING, timestamp, "cancel requested")
        return rec

    def record_fill(self, order_id: str, fill_id: str, qty: float, timestamp: str) -> OrderRecord:
        """Apply a fill IDEMPOTENTLY (dedup by fill_id) and never exceed approved_qty."""
        rec = self._orders[order_id]
        if fill_id in rec._seen_fills:
            return rec                                   # duplicate callback — no double effect
        rec._seen_fills.add(fill_id)
        applied = float(qty)
        if rec.filled_qty + applied > rec.approved_qty + _EPS:
            applied = rec.approved_qty - rec.filled_qty
            if "OVERFILL_CLAMPED" not in rec.flags:
                rec.flags.append("OVERFILL_CLAMPED")
        rec.filled_qty = min(rec.filled_qty + max(applied, 0.0), rec.approved_qty)
        target = OrderStatus.FILLED if rec.filled_qty >= rec.approved_qty - _EPS else OrderStatus.PARTIALLY_FILLED
        # A fill is valid even from CANCEL_PENDING (a fill can race a cancel).
        if target not in _ALLOWED.get(rec.status, frozenset()) and target != rec.status:
            # already terminal or not normally fillable: record the fact, keep status
            rec.history.append((timestamp, rec.status.value, f"late/spurious fill {fill_id} qty={qty}"))
            return rec
        self._transition(rec, target, timestamp, f"fill {fill_id} qty={qty} (filled {rec.filled_qty}/{rec.approved_qty})")
        return rec

    def record_commission(self, order_id: str, amount: float, timestamp: str) -> OrderRecord:
        """Commissions/corrections attach even to a TERMINAL order (never discarded)."""
        rec = self._orders[order_id]
        rec.commissions.append({"amount": float(amount), "timestamp": timestamp})
        rec.history.append((timestamp, rec.status.value, f"commission {amount}"))
        return rec

    def can_resubmit(self, order_id: str) -> bool:
        """Fail-closed: only an order that was REJECTED before reaching the broker
        (no broker id, no fills) is safe to resubmit. Any uncertain/unknown/live state
        is NOT resubmittable."""
        rec = self._orders[order_id]
        if rec.status in RESUBMIT_BLOCKING_STATES:
            return False
        return (rec.status == OrderStatus.REJECTED
                and rec.broker_order_id is None and rec.filled_qty <= _EPS)

    def resync(self, broker_open_orders: dict[str, str], timestamp: str) -> list[str]:
        """Reconnect resynchronisation. ``broker_open_orders`` maps order_id → broker
        status ('WORKING'|'FILLED'|'CANCELLED'|...). Orders in an uncertain/unknown
        state are resolved from broker truth; if the broker has no record, they are
        parked in RECONCILIATION_HOLD (never silently assumed dead). Returns the list
        of order_ids whose status changed."""
        changed: list[str] = []
        for rec in self._orders.values():
            if rec.status not in _RESYNC_RESOLVABLE:
                continue
            before = rec.status
            broker_status = broker_open_orders.get(rec.order_id)
            if broker_status is None:
                # For a genuinely-uncertain order (we never confirmed it reached the broker), a
                # missing record means "park for resolution". For a RESTING order (acked/working/
                # partially-filled) a single missing entry is most likely a transient/partial
                # open-orders query, NOT a death — leave it resting so it stays resync-recoverable
                # rather than permanently stuck in RECONCILIATION_HOLD (which is not itself
                # resync-resolvable) and bricking its symbol.
                if rec.status not in (OrderStatus.SUBMISSION_UNCERTAIN, OrderStatus.BROKER_UNKNOWN):
                    continue
                target, note = (OrderStatus.RECONCILIATION_HOLD,
                                "resync: broker has no record — held for resolution")
            else:
                target = _RESYNC_STATUS_MAP.get(str(broker_status).upper())  # type: ignore[assignment]
                if target is None:
                    continue
                note = f"resync: broker says {broker_status}"
            try:
                self._transition(rec, target, timestamp, note)
            except InvalidTransition:
                # Fail-closed: a status the state machine cannot reach from here is logged and
                # the order is left non-terminal (it keeps blocking its symbol) for resolution.
                logger.warning("resync: %s %s -> %s not an allowed transition; left unchanged.",
                               rec.order_id, before.value, target.value)
                continue
            if rec.status != before:
                changed.append(rec.order_id)
        return changed

    def prune_terminal(self, max_terminal: int) -> int:
        """Bound in-memory growth (LIVE6B-4): keep at most ``max_terminal`` TERMINAL
        orders (evict the OLDEST by creation time); NEVER touch a non-terminal/uncertain
        record. Evicted orders are FILLED/CANCELLED/REJECTED/EXPIRED — already booked into
        the held book and the immutable ledger — so dropping the in-memory record loses no
        safety or audit state. Returns the number pruned."""
        terminal = [r for r in self._orders.values() if r.status in TERMINAL_STATES]
        if len(terminal) <= max_terminal:
            return 0
        # Oldest-first by creation timestamp (first history entry, derived from asof_time
        # — deterministic/PIT, never wall-clock); order_id breaks ties.
        terminal.sort(key=lambda r: (r.history[0][0] if r.history else "", r.order_id))
        for rec in terminal[: len(terminal) - max_terminal]:
            del self._orders[rec.order_id]
        return len(terminal) - max_terminal

    def snapshot_nonterminal(self) -> list[dict]:
        """Serialise the NON-terminal orders (LIVE6B-2) — the ones that must survive a restart
        so an uncertain/working LIVE order is never forgotten. Terminal orders are already
        booked into the held book and the immutable ledger, so they are not persisted."""
        return [r.to_json() for r in self._orders.values() if r.status not in TERMINAL_STATES]

    def restore(self, records: list) -> None:
        """Rebuild orders from a snapshot (restart), setting status DIRECTLY (a restored
        uncertain/working state is not reachable via the normal transition graph). A malformed
        record is logged and skipped (it cannot be reconstructed without its id)."""
        for data in records:
            try:
                rec = OrderRecord.from_json(data)
            except Exception:  # noqa: BLE001 - one bad record must not break restore
                logger.warning("lifecycle restore: skipping malformed order record %r", data)
                continue
            self._orders[rec.order_id] = rec
