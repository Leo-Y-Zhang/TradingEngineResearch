"""
TradingEngineResearch — Interactive Brokers Adapter (ib-insync)
===================================================
The real-money broker. Fail-closed posture throughout:

  • ``connected`` is False unless an ib-insync session is provably up — the
    adapter never reports a connection it cannot demonstrate.
  • ``submit`` refuses any mode other than LIVE (golden rule 1: only LIVE may
    reach a broker) and submits nothing when disconnected.
  • ``account_state`` returns ``connected=False`` whenever the session or the
    account query fails, which the BrokerState LIVE gate then rejects.

ib-insync is imported lazily so the package (and the whole platform) works
without it; network paths are excluded from coverage — they require a running
IB Gateway/TWS and are validated in supervised paper-account sessions, never
in unit tests. The pure plan→order translation is fully unit-tested.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

from data.data_contracts import BrokerOpenOrder, BrokerState, FillEvent, to_aware_utc

logger = logging.getLogger(__name__)

__all__ = ["IBKRBroker", "order_params_from_plan"]


def order_params_from_plan(plan: Any, reference_price: float) -> dict:
    """Translate a ``ChildOrderPlan`` into IB order parameters (pure function).

    LIMIT plans price passively off the reference: BUY below it, SELL above it,
    by ``limit_offset_bps``. MARKET plans carry no limit price (the scheduler
    only emits them for URGENT_DERISK).
    """
    if not math.isfinite(reference_price) or reference_price <= 0.0:
        raise ValueError(f"reference_price must be positive and finite, got {reference_price}")
    side = str(getattr(plan, "side", "BUY")).upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"unknown side: {side!r}")
    qty = float(getattr(plan, "qty", 0.0))
    if not math.isfinite(qty) or qty <= 0.0:
        raise ValueError(f"qty must be positive and finite, got {qty}")

    params: dict[str, Any] = {
        "symbol": str(getattr(plan, "symbol", "")),
        "action": side,
        "total_quantity": qty,
        "order_ref": getattr(plan, "order_ref", None),   # LIVE6B-3: carry OUR id to IB orderRef
    }
    order_type = str(getattr(plan, "order_type", "LIMIT")).upper()
    if order_type == "MARKET":
        params["order_type"] = "MKT"
        return params

    params["order_type"] = "LMT"
    offset_bps = float(getattr(plan, "limit_offset_bps", None) or 0.0)
    signed = -1.0 if side == "BUY" else 1.0          # passive: improve vs the touch
    params["limit_price"] = round(
        reference_price * (1.0 + signed * offset_bps / 10_000.0), 4
    )
    return params


def _map_ib_status(ib_status: str) -> str:
    """Map an ib-insync ``OrderStatus.status`` onto our resync vocabulary (LIVE6B-3)."""
    s = (ib_status or "").strip().lower()
    if s in {"submitted", "presubmitted", "pendingsubmit", "apipending", "pendingcancel"}:
        return "WORKING"   # pendingcancel: the cancel is UNconfirmed and the order can still fill
    if s == "filled":
        return "FILLED"
    if s in {"cancelled", "apicancelled"}:
        return "CANCELLED"
    if s == "inactive":
        return "REJECTED"
    return "WORKING"   # unknown -> treat as still resting (conservative; never assumed dead)


def _fill_commission(fill: Any) -> Optional[float]:
    """The broker-REPORTED commission for one ib-insync fill, or ``None`` when IB has not
    (yet) delivered a CommissionReport — ib-insync attaches a default-constructed report
    (``execId == ''``) until the real one arrives, and reading ``0.0`` off that would INVENT
    a zero-commission fact (§17: record only facts). Non-finite/negative values are refused
    (``None``), never recorded. Pure translation — unit-tested without a gateway."""
    report = getattr(fill, "commissionReport", None)
    if report is None or not str(getattr(report, "execId", "") or ""):
        return None
    try:
        c = float(getattr(report, "commission", None))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c) or c < 0.0:
        return None
    return c


class IBKRBroker:
    """Interactive Brokers adapter over ib-insync (lazy-loaded)."""

    is_paper = False

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,                # 7497 = TWS paper; 7496 = TWS live; 4001/4002 = Gateway
        client_id: int = 1,
        account_id: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.account_id = account_id
        self._ib: Any = None
        self.last_broker_order_ids: dict = {}   # LIVE6B-3: order_ref -> broker order id (last submit)

    # ── connectivity ─────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        ib = self._ib
        try:
            return bool(ib is not None and ib.isConnected())
        except Exception:  # noqa: BLE001 — an erroring session is NOT connected
            return False

    def connect(self, timeout: float = 10.0) -> bool:  # pragma: no cover — needs a gateway
        """Open the ib-insync session. Returns the resulting connectivity."""
        try:
            from ib_insync import IB

            self._ib = IB()
            self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — fail closed, report disconnected
            logger.warning("IBKRBroker connect failed (%s); staying disconnected.", exc)
            self._ib = None
        return self.connected

    def disconnect(self) -> None:  # pragma: no cover — needs a gateway
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("IBKRBroker disconnect error (%s).", exc)
            self._ib = None

    # ── trading ──────────────────────────────────────────────────────────────

    def submit(self, child_plans: list, mode: str) -> list[FillEvent]:
        """Submit child orders. LIVE-only; disconnected ⇒ submits nothing."""
        if str(mode).upper() != "LIVE":
            logger.warning("IBKRBroker.submit called in mode=%s; refusing (LIVE only).", mode)
            return []
        if not self.connected:
            logger.warning("RISK_EVENT RED: IBKRBroker.submit while disconnected; nothing sent.")
            return []
        return self._submit_live(child_plans)

    def _submit_live(self, child_plans: list) -> list[FillEvent]:  # pragma: no cover — needs a gateway
        from ib_insync import LimitOrder, MarketOrder, Order, Stock

        fills: list[FillEvent] = []
        self.last_broker_order_ids = {}
        for plan in child_plans:
            try:
                ticker = self._ib.reqMktData(Stock(plan.symbol, "SMART", "USD"), snapshot=True)
                self._ib.sleep(1.0)
                reference = float(ticker.marketPrice())
                params = order_params_from_plan(plan, reference_price=reference)
                contract = Stock(params["symbol"], "SMART", "USD")
                order: Order
                if params["order_type"] == "MKT":
                    order = MarketOrder(params["action"], params["total_quantity"])
                else:
                    order = LimitOrder(params["action"], params["total_quantity"],
                                       params["limit_price"])
                ref = params.get("order_ref")
                if ref is not None:
                    order.orderRef = str(ref)                  # LIVE6B-3: tag with OUR id
                trade = self._ib.placeOrder(contract, order)
                if ref is not None:                            # capture the broker's own id at ack
                    self.last_broker_order_ids[str(ref)] = str(trade.order.orderId)
                self._ib.sleep(1.0)
                for f in trade.fills:
                    fills.append(FillEvent(
                        order_id=str(f.execution.execId),
                        symbol=params["symbol"],
                        qty=float(f.execution.shares),
                        fill_price=float(f.execution.price),
                        decision_price=reference,
                        arrival_price=reference,
                        slippage_bps=abs(float(f.execution.price) - reference)
                        / reference * 10_000.0,
                        fill_timestamp=to_aware_utc(f.time),
                        # §17 cash leg: IB's per-execution commission when its report has
                        # actually been delivered; None otherwise (never an invented 0.0).
                        commission=_fill_commission(f),
                    ))
            except Exception as exc:  # noqa: BLE001 — one bad plan must not kill the batch
                logger.warning("IBKRBroker: plan for %s failed (%s); continuing.",
                               getattr(plan, "symbol", "?"), exc)
        return fills

    # ── account ──────────────────────────────────────────────────────────────

    def account_state(self, asof_time: datetime) -> BrokerState:
        """The broker's account view. Disconnected (or any query failure) yields
        ``connected=False`` — the BrokerState LIVE gate then refuses to trade."""
        if not self.connected:
            return BrokerState(broker="IBKR", connected=False,
                               account_id=self.account_id, asof_timestamp=asof_time)
        return self._account_state_live(asof_time)

    def _account_state_live(self, asof_time: datetime) -> BrokerState:  # pragma: no cover — needs a gateway
        try:
            values = {v.tag: v.value for v in self._ib.accountSummary()
                      if not self.account_id or v.account == self.account_id}
            positions = {
                p.contract.symbol: float(p.position) for p in self._ib.positions()
                if not self.account_id or p.account == self.account_id
            }
            return BrokerState(
                broker="IBKR", connected=True, account_id=self.account_id,
                asof_timestamp=asof_time,
                nav_gbp=float(values.get("NetLiquidation", 0.0)) or None,
                cash_gbp=float(values.get("TotalCashValue", 0.0)) or None,
                buying_power_gbp=float(values.get("BuyingPower", 0.0)) or None,
                positions=positions,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed
            logger.warning("IBKRBroker account query failed (%s); reporting disconnected.", exc)
            return BrokerState(broker="IBKR", connected=False,
                               account_id=self.account_id, asof_timestamp=asof_time)

    # ── open orders (read-only; reconnect resync) ──────────────────────────────

    def open_orders(self, asof_time: datetime) -> list[BrokerOpenOrder]:
        """Read-only snapshot of resting orders for resync (LIVE6B-3). Returns [] when
        disconnected (fail-closed) and NEVER places an order."""
        if not self.connected:
            return []
        return self._open_orders_live()

    def _open_orders_live(self) -> list[BrokerOpenOrder]:  # pragma: no cover — needs a gateway
        out: list[BrokerOpenOrder] = []
        try:
            for t in self._ib.openTrades():
                out.append({
                    "order_ref": getattr(t.order, "orderRef", None) or None,
                    "broker_order_id": str(getattr(t.order, "orderId", "")),
                    "status": _map_ib_status(str(getattr(t.orderStatus, "status", ""))),
                    "symbol": str(getattr(t.contract, "symbol", "")),
                    "filled_qty": float(getattr(t.orderStatus, "filled", 0.0) or 0.0),
                    # TRUE avg fill price for the filled portion (preferred over ref_price when a
                    # disconnect-fill is booked); None if IB has not reported one yet.
                    "avg_fill_price": (float(getattr(t.orderStatus, "avgFillPrice", 0.0) or 0.0)
                                       or None),
                })
        except Exception as exc:  # noqa: BLE001 — a query failure yields NO truth (fail-closed)
            logger.warning("IBKRBroker.open_orders query failed (%s); returning none.", exc)
            return []
        return out
