"""
TradingEngineResearch — Immutable financial event ledger (Phase 3 foundation)
====================================================================
The directive's #1 paper/shadow finish-line requirement is an **immutable order,
execution, cash, position and P&L trail**. This module is that trail: an append-only,
**hash-chained** (tamper-evident) event log.

Properties (directive §7.5 accounting + §17 ledger):
  • **Append-only** — events are never modified or deleted. A mistake is corrected by
    appending an explicit reversing/adjustment event, never by editing history.
  • **Tamper-evident** — every event carries ``hash = sha256(canonical(record))`` where
    the record includes the previous event's hash, forming a chain. Altering any past
    event breaks the chain, which ``verify_chain()`` detects.
  • **Durable** — optionally persisted as append-only JSONL (flush + fsync per event)
    so a restart reconstructs the exact trail.
  • **Deterministic** — the caller supplies the timestamp (ISO-8601, from ``asof``,
    never wall-clock), so replays are bit-reproducible (engine convention).

This is the substrate the reconciliation layer (internal ↔ IBKR/Flex ↔ bank) will
check against; it does NOT itself place orders or compute P&L — it records facts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["LedgerEvent", "ImmutableLedger", "record_cycle", "replay_ledger_to_positions",
           "replay_ledger_to_balances", "EVENT_TYPES", "GENESIS_HASH"]

GENESIS_HASH = "0" * 64

# The auditable financial-event vocabulary (extend as the order/accounting paths grow).
EVENT_TYPES = frozenset({
    "ORDER_INTENT", "RISK_APPROVED", "SUBMISSION", "SUBMISSION_UNCERTAIN", "ACK",
    "PARTIAL_FILL", "FILL", "CANCEL_REQUEST", "CANCELLED", "REJECTED", "EXPIRED",
    "COMMISSION", "FEE", "CASH", "POSITION", "PNL", "CORPORATE_ACTION",
    "CORRECTION", "RECONCILIATION", "KILL_SWITCH", "NOTE",
})


def _canonical(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) — the hashing input."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _event_hash(seq: int, timestamp: str, event_type: str, payload: dict, prev_hash: str) -> str:
    body = _canonical({
        "seq": seq, "timestamp": timestamp, "event_type": event_type,
        "payload": payload, "prev_hash": prev_hash,
    })
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LedgerEvent:
    """One immutable, hash-chained financial event."""
    seq: int
    timestamp: str          # ISO-8601, supplied by the caller (never wall-clock)
    event_type: str
    payload: dict
    prev_hash: str
    hash: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ImmutableLedger:
    """Append-only, hash-chained event ledger; optionally durable as JSONL."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._events: list[LedgerEvent] = []
        self._path = Path(path) if path is not None else None
        if self._path is not None and self._path.exists():
            self._load()

    # ── append (the only mutation) ────────────────────────────────────────────────
    def append(self, event_type: str, payload: dict, timestamp: str) -> LedgerEvent:
        """Append a new event. ``event_type`` must be known; ``timestamp`` is an
        ISO-8601 string from the caller's clock-of-record. Returns the chained event.
        History is never modified — corrections are new appends."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type!r}; must be one of {sorted(EVENT_TYPES)}")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        seq = len(self._events)
        prev = self._events[-1].hash if self._events else GENESIS_HASH
        clean = json.loads(_canonical(payload))           # normalise to JSON-safe primitives
        h = _event_hash(seq, timestamp, event_type, clean, prev)
        ev = LedgerEvent(seq=seq, timestamp=timestamp, event_type=event_type,
                         payload=clean, prev_hash=prev, hash=h)
        self._events.append(ev)
        if self._path is not None:
            self._persist(ev)
        return ev

    def correct(self, original_seq: int, reason: str, payload: dict, timestamp: str) -> LedgerEvent:
        """Record a CORRECTION that references an earlier event WITHOUT modifying it
        (explicit reversing/adjustment entry — directive §7.5)."""
        body = {"corrects_seq": int(original_seq), "reason": reason, **payload}
        return self.append("CORRECTION", body, timestamp)

    # ── read ──────────────────────────────────────────────────────────────────────
    def events(self, event_type: Optional[str] = None) -> list[LedgerEvent]:
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.event_type == event_type]

    def __len__(self) -> int:
        return len(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_HASH

    # ── integrity ─────────────────────────────────────────────────────────────────
    def verify_chain(self) -> bool:
        """True iff the chain is intact: sequential, prev-hash-linked, and every
        event's stored hash matches a recomputation. Any tamper returns False."""
        prev = GENESIS_HASH
        for i, ev in enumerate(self._events):
            if ev.seq != i or ev.prev_hash != prev:
                return False
            if _event_hash(ev.seq, ev.timestamp, ev.event_type, ev.payload, ev.prev_hash) != ev.hash:
                return False
            prev = ev.hash
        return True

    # ── durability ────────────────────────────────────────────────────────────────
    def _persist(self, ev: LedgerEvent) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Append-only line, flushed + fsync'd so a crash can't lose a committed event.
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_json(), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _load(self) -> None:
        assert self._path is not None
        events: list[LedgerEvent] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            events.append(LedgerEvent(
                seq=int(d["seq"]), timestamp=str(d["timestamp"]), event_type=str(d["event_type"]),
                payload=dict(d["payload"]), prev_hash=str(d["prev_hash"]), hash=str(d["hash"]),
            ))
        self._events = events
        if not self.verify_chain():
            raise ValueError(f"ledger at {self._path} failed integrity check on load (tampered or truncated)")


def record_cycle(
    ledger: ImmutableLedger, result: Any, timestamp: str, recon_alert: Optional[dict] = None
) -> int:
    """Append a decision cycle's financial events to the immutable ledger; return the
    count appended. Records one FILL per fill (plus one COMMISSION per fill that carries a
    broker-REPORTED commission — §17 cash leg), one POSITION snapshot (the book after the
    cycle), and an optional RECONCILIATION event. Defensive (``getattr``) so it works with
    real ``CycleResult`` objects and test stubs alike. The CALLER must wrap this fail-soft:
    an audit-trail write must never break the trading cycle."""
    n = 0
    # A FillEvent carries only an unsigned qty; its BUY/SELL side lives on the matching
    # OrderIntent (same symbol-keyed reconciliation the engine uses for the achieved
    # book). Record the side + a signed qty so the trail can be replayed into positions
    # (replay_ledger_to_positions) — the authoritative 'internal' side of reconcile().
    side_by_symbol: dict[Any, str] = {}
    for intent in (getattr(result, "order_intents", None) or []):
        sym = getattr(intent, "symbol", None)
        direction = getattr(intent, "direction", None)
        if sym is not None and direction in ("BUY", "SELL"):
            side_by_symbol[sym] = direction
    for f in (getattr(result, "fills", None) or []):
        sym = getattr(f, "symbol", None)
        qty = getattr(f, "qty", None)
        side = side_by_symbol.get(sym)
        signed_qty = float(qty) * (1.0 if side == "BUY" else -1.0) if (qty is not None and side is not None) else None
        ledger.append("FILL", {
            "order_id": getattr(f, "order_id", None),
            "symbol": sym,
            "qty": qty,
            "side": side,
            "signed_qty": signed_qty,
            "fill_price": getattr(f, "fill_price", None),
            "slippage_bps": getattr(f, "slippage_bps", None),
        }, timestamp)
        n += 1
        # §17 cash leg: a broker-REPORTED commission (never inferred — None means the broker
        # reported none, e.g. Alpaca equities paper) is recorded as its own COMMISSION event so
        # replay_ledger_to_balances charges it against cash. Idempotent across a replay/re-record
        # of the same cycle: keyed on the broker's OWN fill id (unique per execution) and anchored
        # on the DURABLE ledger — the resolve_reconciliation dedup convention — so a duplicate
        # record can never double-count the cost.
        commission = getattr(f, "commission", None)
        if commission is not None:
            oid = getattr(f, "order_id", None)
            already = oid is not None and any(
                e.payload.get("order_id") == oid for e in ledger.events("COMMISSION"))
            if not already:
                ledger.append("COMMISSION", {
                    "order_id": oid, "symbol": sym,
                    "amount": float(commission), "source": "broker_fill",
                }, timestamp)
                n += 1
    book = getattr(result, "achieved_weights", None) or getattr(result, "target_weights", None) or {}
    ledger.append("POSITION", {
        "book": {str(k): float(v) for k, v in dict(book).items()},
        "blocked": bool(getattr(result, "blocked", False)),
        "live_orders_submitted": int(getattr(result, "live_orders_submitted", 0)),
    }, timestamp)
    n += 1
    if recon_alert:
        ledger.append("RECONCILIATION", dict(recon_alert), timestamp)
        n += 1
    return n


def replay_ledger_to_positions(ledger: ImmutableLedger, dust: float = 1e-9) -> dict[str, float]:
    """Reconstruct the internal share-position book from the immutable ledger's FILL
    events — the authoritative *internal* side for :func:`ops.reconciliation.reconcile`
    (directive §17 three-way reconciliation). Sums each symbol's signed fill quantities
    (BUY +, SELL −); positions within ``dust`` of zero (a fully-closed name) are dropped.

    Fail-safe: a FILL with no ``signed_qty`` (an older event, or a fill that had no
    matching intent at record time) cannot be signed, so it is SKIPPED with a warning —
    under-reporting a position is safer than mis-signing a sell as a buy when the output
    feeds a reconciliation check. Replays are exact only on a ledger written by the
    current ``record_cycle`` (which always signs fills)."""
    positions: dict[str, float] = {}
    for ev in ledger.events("FILL"):
        sym = ev.payload.get("symbol")
        signed = ev.payload.get("signed_qty")
        if sym is None:
            continue
        if signed is None:
            logger.warning("ledger replay: FILL seq=%s symbol=%s has no signed_qty; skipped "
                           "(cannot reconstruct its sign).", ev.seq, sym)
            continue
        positions[str(sym)] = positions.get(str(sym), 0.0) + float(signed)
    return {s: q for s, q in positions.items() if abs(q) > dust}


def replay_ledger_to_balances(ledger: ImmutableLedger, base_ccy: str = "GBP") -> dict[str, Any]:
    """Reconstruct the internal ``{positions, cash}`` book from the immutable ledger — the
    authoritative *internal* side for :func:`ops.reconciliation.reconcile` (directive §17,
    extending :func:`replay_ledger_to_positions` with the cash leg).

    Cash = Σ ``CASH`` events (deposits/withdrawals, signed ``amount``) − Σ trade cash flows
    (a BUY pays out ``signed_qty × fill_price``, a SELL takes in) − Σ ``COMMISSION``/``FEE``
    (``amount``, always a cost). The fill price already embeds slippage (the trade executed
    at the achieved price), so no extra slippage term is needed; explicit broker commissions
    appear only once recorded as COMMISSION/FEE events (``record_cycle`` appends one per
    broker-REPORTED fill commission — the ledger reflects exactly the cash facts recorded,
    nothing inferred). Single base-currency bucket (the engine is single-currency,
    ``capital_gbp``). A FILL missing the sign/price is skipped with a warning rather than
    mis-counted, and the returned ``cash_complete`` flag turns False so the LIVE reconciliation
    can SKIP (never false-break on) an unreliable cash figure."""
    cash = 0.0
    cash_complete = True
    for ev in ledger.events("CASH"):
        amt = ev.payload.get("amount")
        if amt is not None:
            cash += float(amt)
    for ev in ledger.events("FILL"):
        signed = ev.payload.get("signed_qty")
        price = ev.payload.get("fill_price")
        if signed is None or price is None:
            logger.warning("ledger balances: FILL seq=%s missing signed_qty/fill_price; "
                           "cash leg incomplete.", ev.seq)
            cash_complete = False
            continue
        cash -= float(signed) * float(price)
    for etype in ("COMMISSION", "FEE"):
        for ev in ledger.events(etype):
            amt = ev.payload.get("amount")
            if amt is not None:
                cash -= abs(float(amt))   # a commission/fee is always a cost
    return {"positions": replay_ledger_to_positions(ledger), "cash": {base_ccy: cash},
            "cash_complete": cash_complete}
