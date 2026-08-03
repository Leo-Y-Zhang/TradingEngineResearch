"""
TradingEngineResearch — Alpaca broker adapter (broker/alpaca.py)
=======================================================
A :class:`~broker.protocol.BrokerProtocol` adapter for **Alpaca** (US equities) — the convenient,
FREE paper-validation broker. Unlike IBKR (a desktop TWS/Gateway), Alpaca is a plain REST API with
instant free paper signup, so an operator can run a supervised paper session in minutes.

This is a HYBRID addition (directive §15 "compare the incumbent integration against alternatives" /
§2.5 hybrids): IBKR remains the production target; Alpaca gives a frictionless way to validate the
whole system (order lifecycle, ledger, reconciliation, the orderRef round-trip via Alpaca's
``client_order_id``) against a real broker in paper mode. Same safety contract as IBKR:

  • ``submit`` is **LIVE-only** and refuses when disconnected (golden rule 1 / fail-closed);
  • the paper endpoint is the default (``paper=True``) — a real-money Alpaca account is a separate,
    explicitly-configured, later gate;
  • ``is_paper`` is NOT set, so the run-loop's ``_reconcile`` reconciles against Alpaca's REAL
    reported positions (the internal sim ``PaperBroker`` sets ``is_paper=True`` and is skipped).

Notes / residuals (paper-validation simplifications, documented for the session):
  • Orders are submitted as MARKET orders for reliable paper fills; the engine's passive-limit
    execution nuance is an IBKR/production concern, not what this validates.
  • No pre-trade quote is fetched, so the FillEvent decision/arrival price = the achieved fill price
    (slippage 0); TCA slippage is not meaningful on this path. NB: zero-impact fills make STEP-12's
    ``tca.update_cost_priors`` pull the learned cost coefficients DOWN each cycle, so the in-run cost
    model degrades. It is per-process (not persisted), so it cannot leak into a later IBKR run — but
    run an Alpaca validation against a DISPOSABLE state dir, and do not read its TCA priors as real.
  • Alpaca reports USD; the BrokerState ``*_gbp`` fields therefore carry USD here. The positions leg
    (shares) that ``_reconcile`` uses is currency-agnostic; cash/NAV reconciliation is not yet wired.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from data.data_contracts import BrokerOpenOrder, BrokerState, FillEvent, to_aware_utc

logger = logging.getLogger(__name__)

__all__ = ["AlpacaBroker", "_map_alpaca_status"]


# Map an Alpaca order status to the resync status vocabulary the lifecycle consumes. Conservative:
# anything not clearly terminal maps to WORKING so a resting order stays resync-recoverable (never
# wrongly parked/terminal). Mirrors the IBKR ``_map_ib_status`` intent (PendingCancel can still fill).
_ALPACA_STATUS: dict[str, str] = {
    "filled": "FILLED",
    "partially_filled": "PARTIALLY_FILLED",
    "canceled": "CANCELLED",
    "cancelled": "CANCELLED",
    "expired": "EXPIRED",
    "rejected": "REJECTED",
    "done_for_day": "WORKING",
    "new": "WORKING", "accepted": "WORKING", "pending_new": "WORKING",
    "accepted_for_bidding": "WORKING", "pending_cancel": "WORKING",
    "pending_replace": "WORKING", "replaced": "WORKING", "held": "WORKING",
    "suspended": "WORKING", "calculated": "WORKING", "stopped": "WORKING",
    "pending_review": "WORKING",
}


def _map_alpaca_status(status: str) -> str:
    return _ALPACA_STATUS.get(str(status).strip().lower(), "WORKING")


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class AlpacaBroker:
    """BrokerProtocol adapter for Alpaca paper/live trading via ``alpaca-py``."""

    def __init__(self, api_key: Optional[str], secret_key: Optional[str], *, paper: bool = True,
                 account_id: Optional[str] = None,
                 client_factory: Optional[Callable[[], Any]] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 fill_wait_seconds: float = 1.5) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = bool(paper)
        self.account_id = account_id
        self._client_factory = client_factory     # inject a fake TradingClient in tests
        self._sleep = sleep
        self._fill_wait = float(fill_wait_seconds)
        self._client: Any = None
        self._connected = False
        self.last_broker_order_ids: dict = {}      # order_ref -> alpaca order id (last submit)

    # ── connectivity ─────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return bool(self._connected and self._client is not None)

    def _build_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from alpaca.trading.client import TradingClient   # lazy: optional 'brokers' dependency

        return TradingClient(self.api_key, self.secret_key, paper=self.paper)

    def connect(self, timeout: float = 10.0) -> bool:
        """Open a session: build the REST client and verify reachability + creds with a
        ``get_account`` probe. Fail-closed — any error leaves the broker disconnected."""
        try:
            client = self._build_client()
            client.get_account()                   # verifies creds + reachability
            self._client = client
            self._connected = True
        except Exception as exc:  # noqa: BLE001 — fail closed, report disconnected
            logger.warning("AlpacaBroker connect failed (%s); staying disconnected.", exc)
            self._client = None
            self._connected = False
        return self.connected

    def disconnect(self) -> None:
        self._client = None
        self._connected = False

    # ── trading ──────────────────────────────────────────────────────────────────
    def submit(self, child_plans: list, mode: str) -> list[FillEvent]:
        """Submit child orders. LIVE-only; disconnected ⇒ submits nothing (fail-closed)."""
        if str(mode).upper() != "LIVE":
            logger.warning("AlpacaBroker.submit called in mode=%s; refusing (LIVE only).", mode)
            return []
        if not self.connected:
            logger.warning("RISK_EVENT RED: AlpacaBroker.submit while disconnected; nothing sent.")
            return []
        return self._submit(child_plans)

    def _submit(self, child_plans: list) -> list[FillEvent]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        fills: list[FillEvent] = []
        self.last_broker_order_ids = {}
        for plan in child_plans:
            qty = abs(_f(getattr(plan, "qty", 0.0)))
            if qty <= 0:
                continue
            symbol = str(getattr(plan, "symbol", ""))
            side = (OrderSide.BUY if str(getattr(plan, "side", "BUY")).upper() == "BUY"
                    else OrderSide.SELL)
            order_ref = getattr(plan, "order_ref", None)
            req = MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                client_order_id=str(order_ref) if order_ref is not None else None)
            # A submit ERROR (403 insufficient buying power, a bad symbol, a network blip) is NOT a
            # silent 'acked, resting' — it PROPAGATES so OrderManager.place records the order as
            # SUBMISSION_UNCERTAIN (resync-resolvable + operator-visible), never a false WORKING that
            # would brick the symbol with no broker counterpart to resolve it against.
            placed = self._client.submit_order(req)
            if order_ref is not None:                           # capture the broker's id at submit
                self.last_broker_order_ids[str(order_ref)] = str(getattr(placed, "id", ""))
            try:
                fe = self._fill_event(self._await_fill(placed, order_ref), symbol)
                if fe is not None:
                    fills.append(fe)
            except Exception as exc:  # noqa: BLE001 — the order WAS submitted; a refetch/mapping
                # failure just means no CONFIRMED fill yet, so it rests and the resync recovers it.
                logger.warning("AlpacaBroker: fill processing for %s failed (%s); order rests.",
                               symbol, exc)
        return fills

    def _await_fill(self, placed: Any, order_ref: Any) -> Any:
        """Re-fetch the order once after a short wait so a fast market fill is captured
        synchronously (mirrors the IBKR adapter's sleep+read). An unfilled order rests → no
        FillEvent → the reconnect resync / reconciliation handles it later."""
        if _f(getattr(placed, "filled_qty", 0.0)) > 0:
            return placed
        self._sleep(self._fill_wait)
        try:
            if order_ref is not None:
                return self._client.get_order_by_client_id(str(order_ref))
            return self._client.get_order_by_id(getattr(placed, "id", ""))
        except Exception as exc:  # noqa: BLE001 — a refetch failure just means no confirmed fill yet
            logger.warning("AlpacaBroker: fill refetch failed (%s); treating as unfilled.", exc)
            return placed

    def _fill_event(self, order: Any, symbol: str) -> Optional[FillEvent]:
        fq = _f(getattr(order, "filled_qty", 0.0))
        price = _f(getattr(order, "filled_avg_price", 0.0))
        if fq <= 0 or price <= 0:
            return None
        # Normalise the fill time to tz-aware UTC (the canonical convention) so it never mixes with
        # the aware asof/price stamps the tracker/persistence aggregate it with.
        ts = to_aware_utc(getattr(order, "filled_at", None) or datetime.now(timezone.utc))
        # §17 cash leg: record ONLY a commission Alpaca actually reports (the trading-API Order
        # carries none for equities paper today — read defensively for a future surface). A junk/
        # negative/non-finite value is dropped, never recorded, and never fails the fill itself.
        raw = getattr(order, "commission", None)
        commission: Optional[float] = None
        if raw is not None:
            c = _f(raw, default=-1.0)
            if math.isfinite(c) and c >= 0.0:
                commission = c
        return FillEvent(order_id=str(getattr(order, "id", "")), symbol=symbol, qty=fq,
                         fill_price=price, decision_price=price, arrival_price=price,
                         slippage_bps=0.0, fill_timestamp=ts, commission=commission)

    # ── account ──────────────────────────────────────────────────────────────────
    def account_state(self, asof_time: datetime) -> BrokerState:
        """Alpaca's account view. Disconnected (or any query failure) ⇒ ``connected=False`` so the
        BrokerState LIVE gate refuses to trade."""
        if not self.connected:
            return BrokerState(broker="ALPACA", connected=False,
                               account_id=self.account_id, asof_timestamp=asof_time)
        try:
            acct = self._client.get_account()
            positions: dict[str, float] = {}
            for p in self._client.get_all_positions():
                q = abs(_f(getattr(p, "qty", 0.0)))
                side = str(getattr(getattr(p, "side", ""), "value", getattr(p, "side", ""))).lower()
                positions[str(getattr(p, "symbol", ""))] = -q if "short" in side else q
            return BrokerState(
                broker="ALPACA", connected=True,
                account_id=str(getattr(acct, "account_number", None) or self.account_id or ""),
                asof_timestamp=asof_time,
                nav_gbp=_f(getattr(acct, "portfolio_value", 0.0)) or None,   # NB: USD (see module note)
                cash_gbp=_f(getattr(acct, "cash", 0.0)) or None,
                buying_power_gbp=_f(getattr(acct, "buying_power", getattr(acct, "cash", 0.0))) or None,
                positions=positions)
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.warning("AlpacaBroker account query failed (%s); reporting disconnected.", exc)
            return BrokerState(broker="ALPACA", connected=False,
                               account_id=self.account_id, asof_timestamp=asof_time)

    # ── open orders (read-only; reconnect resync) ──────────────────────────────────
    @staticmethod
    def _map_order(o: Any) -> BrokerOpenOrder:
        fap = getattr(o, "filled_avg_price", None)
        status = str(getattr(getattr(o, "status", ""), "value", getattr(o, "status", "")))
        return {
            "order_ref": getattr(o, "client_order_id", None) or None,
            "broker_order_id": str(getattr(o, "id", "")),
            "status": _map_alpaca_status(status),
            "symbol": str(getattr(o, "symbol", "")),
            "filled_qty": _f(getattr(o, "filled_qty", 0.0)),
            "avg_fill_price": (_f(fap) if fap not in (None, "") else None),
        }

    def open_orders(self, asof_time: datetime) -> list[BrokerOpenOrder]:
        """Read-only snapshot for resync (LIVE6B-3): OPEN (resting) orders PLUS recently-CLOSED
        ones. The CLOSED leg is essential for Alpaca because a MARKET order goes terminal
        (filled/rejected/canceled) immediately and never appears under OPEN — without it the resync
        could never resolve a stuck WORKING record (a rejection would silently brick the symbol; a
        missed fill would be lost). Idempotent: the resync only acts on NON-terminal lifecycle
        records that match these entries. Returns [] when disconnected (fail-closed); never submits."""
        if not self.connected:
            return []
        try:
            from datetime import timedelta

            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            out: list[BrokerOpenOrder] = [
                self._map_order(o)
                for o in self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))]
            try:
                closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                                          after=asof_time - timedelta(hours=24), limit=100)
                out.extend(self._map_order(o) for o in self._client.get_orders(closed))
            except Exception as exc:  # noqa: BLE001 — the CLOSED leg is best-effort; OPEN still stands
                logger.warning("AlpacaBroker.open_orders CLOSED query failed (%s); OPEN only.", exc)
            return out
        except Exception as exc:  # noqa: BLE001 — a query failure yields NO truth (fail-closed)
            logger.warning("AlpacaBroker.open_orders query failed (%s); returning none.", exc)
            return []
