"""
TradingEngineResearch — Safe order manager (Phase 6 wiring)
==================================================
Ties the §15 :class:`~execution.order_lifecycle.OrderLifecycle` to a broker and the
immutable :class:`~ops.ledger.ImmutableLedger` into ONE safe submit → track →
reconcile flow. This is the layer that turns the lifecycle's invariants into actual
behaviour against a (paper/live) broker, while recording an auditable trail.

Broker contract (a thin adapter maps the real ``IBKRBroker`` onto it):
  • ``broker.connected`` → bool
  • ``broker.submit(order: dict, mode: str)`` → list of fills (dicts/objects with
    ``fill_id``/``qty``), and **may raise** (timeout / network).

Safety behaviour (all from the lifecycle, enforced here end-to-end):
  • disconnected broker at submit → ``BROKER_UNKNOWN`` (never a blind resubmit);
  • ``submit`` raises (timeout) → ``SUBMISSION_UNCERTAIN`` (NOT a rejection);
  • returned fills are applied **idempotently** and never exceed approved qty;
  • a connected submit with no fills → ``WORKING`` (acked, resting);
  • everything is appended to the ledger (fail-soft — audit never breaks execution).

This module does NOT decide WHAT to trade and does NOT itself reach a broker in
RESEARCH; it is the execution-safety wrapper the run-loop/engine can adopt.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from data.data_contracts import DiscoveredFill
from execution.order_lifecycle import OrderLifecycle, OrderStatus, TERMINAL_STATES

logger = logging.getLogger(__name__)

__all__ = ["OrderManager"]


def _fill_field(f: Any, name: str, default: Any = None) -> Any:
    if isinstance(f, dict):
        return f.get(name, default)
    return getattr(f, name, default)


class OrderManager:
    """Safe submit→track→reconcile over a synchronous broker."""

    def __init__(self, lifecycle: OrderLifecycle, broker: Any, *, mode: str,
                 ledger: Optional[Any] = None) -> None:
        self.lifecycle = lifecycle
        self.broker = broker
        self.mode = mode
        self.ledger = ledger
        # Held-book flow: an outbox of reconnect-resync-discovered disconnect-fills (a fill
        # that landed during a disconnect — broker filled more than we locally booked). The
        # engine drains it each LIVE cycle so the run-loop can raise a durable OPEN
        # reconciliation item. NEVER auto-applied to the book (directive Section 2/17).
        self.discovered_fills: list[DiscoveredFill] = []

    # ── ledger helper (fail-soft) ────────────────────────────────────────────────
    def _ledger(self, event_type: str, payload: dict, timestamp: str) -> None:
        if self.ledger is None:
            return
        try:
            self.ledger.append(event_type, payload, timestamp)
        except Exception:  # noqa: BLE001 - audit write must never break execution
            logger.exception("order_manager ledger append failed")

    # ── place an order safely ─────────────────────────────────────────────────────
    def place(self, order_id: str, symbol: str, side: str, qty: float,
              timestamp: str, ref_price: float = 0.0) -> tuple[Any, list]:
        """Create → risk-approve → submit one order, driving the lifecycle + ledger
        safely through whatever the broker does (fill / ack / timeout / disconnect).

        Returns ``(record, broker_fills)`` where ``broker_fills`` is the list of fills
        the broker actually returned for THIS submit (empty for the disconnected,
        timeout, and acked-no-fill paths). The engine's STEP-12 needs these genuine
        broker fills (real ``fill_price``/``slippage_bps``) to keep TCA and the
        achieved-book reconciliation faithful."""
        rec = self.lifecycle.create(order_id, qty, timestamp, symbol=symbol, side=side, ref_price=ref_price)
        self.lifecycle.transition(order_id, OrderStatus.VALIDATED, timestamp)
        self.lifecycle.transition(order_id, OrderStatus.RISK_APPROVED, timestamp)
        self.lifecycle.transition(order_id, OrderStatus.SUBMIT_PENDING, timestamp)
        self._ledger("ORDER_INTENT", {"order_id": order_id, "symbol": symbol,
                                      "side": side, "qty": qty, "mode": self.mode}, timestamp)

        # Disconnected broker → unknown state, NEVER a blind (re)submit.
        try:
            connected = bool(self.broker.connected)
        except Exception:  # noqa: BLE001 - an erroring connectivity probe is "not connected"
            connected = False
        if not connected:
            self.lifecycle.mark_broker_unknown(order_id, timestamp)
            self._ledger("NOTE", {"order_id": order_id, "event": "broker_unknown_pre_submit"}, timestamp)
            return rec, []

        self._ledger("SUBMISSION", {"order_id": order_id, "symbol": symbol,
                                    "side": side, "qty": qty}, timestamp)
        try:
            fills = self.broker.submit({"order_id": order_id, "symbol": symbol,
                                        "side": side, "qty": qty}, self.mode) or []
        except Exception as exc:  # noqa: BLE001 - timeout/network ≠ rejection
            self.lifecycle.mark_submission_uncertain(order_id, timestamp)
            self._ledger("SUBMISSION_UNCERTAIN", {"order_id": order_id, "error": str(exc)}, timestamp)
            return rec, []

        # LIVE6B-3: capture the broker's OWN order id (from the ack — available even before
        # any fill) so a later reconnect resync can key broker open-order truth back to us.
        bid = getattr(self.broker, "last_broker_order_ids", {}).get(order_id)
        if bid:
            self.lifecycle.set_broker_order_id(order_id, str(bid), timestamp)

        if not fills:
            # Connected, accepted, resting at the broker.
            self.lifecycle.transition(order_id, OrderStatus.WORKING, timestamp, "acked, no fills yet")
            self._ledger("ACK", {"order_id": order_id}, timestamp)
            return rec, []

        for f in fills:
            fid = str(_fill_field(f, "fill_id") or _fill_field(f, "order_id") or f"{order_id}-{len(rec.history)}")
            fqty = float(_fill_field(f, "qty", 0.0) or 0.0)
            self.lifecycle.record_fill(order_id, fid, fqty, timestamp)
            # §17 cash leg: a broker-REPORTED commission attaches to the order record (§15: even
            # a terminal order accepts it) and rides on the audit payload. On the engine path the
            # ledger is None by design — the run-loop's record_cycle (the sole ledger writer)
            # appends the COMMISSION event from the FillEvent itself.
            commission = _fill_field(f, "commission")
            if commission is not None:
                self.lifecycle.record_commission(order_id, float(commission), timestamp)
            self._ledger("FILL", {"order_id": order_id, "fill_id": fid, "symbol": symbol,
                                  "qty": fqty, "fill_price": _fill_field(f, "fill_price"),
                                  "commission": commission}, timestamp)
        return rec, fills

    # ── reconnect / resync ────────────────────────────────────────────────────────
    def reconcile_open_orders(self, broker_open_orders, timestamp: str) -> list[str]:
        """Reconcile the broker's open-order truth onto our lifecycle (reconnect handling).

        Accepts either a legacy ``dict[order_id -> status]`` (passed straight through), or a
        list of broker open-order entries ``{order_ref, broker_order_id, status, symbol,
        filled_qty}`` which are mapped back to OUR order_ids (by order_ref first,
        broker_order_id second) and enrich the matched record's broker_order_id before the
        lifecycle resync. Records the outcome. READ-ONLY — never submits."""
        if isinstance(broker_open_orders, dict):
            status_map = dict(broker_open_orders)                  # legacy / already-collapsed map
        else:
            status_map = self._map_broker_open_orders(list(broker_open_orders), timestamp)
        changed = self.lifecycle.resync(status_map, timestamp)
        self._ledger("RECONCILIATION", {"resynced": changed,
                                        "broker_open_orders": list(status_map)}, timestamp)
        return changed

    def _map_broker_open_orders(self, entries: list, timestamp: str) -> dict:
        """Collapse a list of broker open-order entries into the {order_id -> status} map the
        lifecycle resync consumes: match each non-terminal record by order_ref (primary) then
        broker_order_id (fallback), and enrich its broker_order_id from broker truth. Unmatched
        non-terminal records are omitted so resync parks them in RECONCILIATION_HOLD."""
        by_ref: dict = {}
        by_bid: dict = {}
        for e in entries:
            ref, bid = e.get("order_ref"), e.get("broker_order_id")
            if ref is not None:
                by_ref[str(ref)] = e
            if bid is not None:
                by_bid[str(bid)] = e
        status_map: dict = {}
        for rec in self.lifecycle.all():
            if rec.status in TERMINAL_STATES:
                continue
            e = by_ref.get(rec.order_id)
            if e is None and rec.broker_order_id is not None:
                e = by_bid.get(str(rec.broker_order_id))
            if e is None:
                continue
            mbid = e.get("broker_order_id")
            if mbid and rec.broker_order_id is None:
                self.lifecycle.set_broker_order_id(rec.order_id, str(mbid), timestamp)
            broker_filled = float(e.get("filled_qty", 0.0) or 0.0)
            if broker_filled > rec.filled_qty + 1e-9:
                # The broker filled MORE than we locally booked (e.g. a fill that landed during a
                # disconnect). We cannot book that quantity into the held book from here, so record
                # the true filled qty (keeps the pending-overlay residual correct), surface the
                # unbooked delta on the outbox (so the run-loop raises an OPEN reconciliation item
                # for operator-gated booking), and PARK the order for reconciliation instead of
                # silently resolving/pruning it (fail-closed: never lose a real fill).
                old_filled = rec.filled_qty
                self.lifecycle.reconcile_broker_fill(rec.order_id, broker_filled, timestamp)
                new_filled = rec.filled_qty                 # clamped to approved_qty by the lifecycle
                self.discovered_fills.append({
                    "order_id": rec.order_id, "symbol": rec.symbol, "side": rec.side,
                    "delta_qty": new_filled - old_filled, "broker_filled_qty": new_filled,
                    "ref_price": rec.ref_price,
                    # The broker's TRUE avg fill price when reported (preferred over ref_price at
                    # booking for an exact cash leg); None when the broker open-order entry omits it.
                    "avg_fill_price": e.get("avg_fill_price"),
                })
                status_map[rec.order_id] = OrderStatus.RECONCILIATION_HOLD.value
            else:
                status_map[rec.order_id] = str(e.get("status", ""))
        return status_map

    def can_resubmit(self, order_id: str) -> bool:
        return self.lifecycle.can_resubmit(order_id)
