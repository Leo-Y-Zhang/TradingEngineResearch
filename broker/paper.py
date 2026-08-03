"""
TradingEngineResearch — Paper Broker
========================
A deterministic, zero-market-access broker: fills every valid child order
locally at ``reference price ± half spread``. Its purpose is to exercise the
LIVE code path (engine STEP 12 → broker.submit) without any possibility of a
real order, so the broker wiring is tested before IBKR ever sees a request.

Determinism: fills are a pure function of the supplied reference prices and
the child plans; order ids derive from a submission counter, never randomness
or wall-clock.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from data.data_contracts import BrokerOpenOrder, BrokerState, FillEvent

logger = logging.getLogger(__name__)

__all__ = ["PaperBroker"]


class PaperBroker:
    """Deterministic local-fill broker (see module docstring)."""

    is_paper = True

    def __init__(
        self,
        prices: Optional[dict] = None,
        spread_bps: float = 6.0,
        nav_gbp: float = 1_000_000.0,
        cash_gbp: Optional[float] = None,
        asof_time: Optional[datetime] = None,
    ) -> None:
        self.prices: dict[str, float] = {str(k): float(v) for k, v in (prices or {}).items()}
        self.spread_bps = float(spread_bps)
        self.nav_gbp = float(nav_gbp)
        self.cash_gbp = float(cash_gbp) if cash_gbp is not None else float(nav_gbp)
        # Deterministic clock: fills are stamped from the caller-set asof time
        # (the run-loop updates it each cycle), never the wall clock.
        self.asof_time = asof_time or datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.positions: dict[str, float] = {}
        self.submitted: list = []
        self._fill_counter = 0
        self.last_broker_order_ids: dict = {}   # LIVE6B-3: order_ref -> broker order id (last submit)

    @property
    def connected(self) -> bool:
        return True                       # local simulation is always "up"

    def update_prices(self, prices: dict, asof_time: Optional[datetime] = None) -> None:
        """Refresh the reference prices (and the deterministic fill clock)."""
        self.prices.update({str(k): float(v) for k, v in prices.items()})
        if asof_time is not None:
            self.asof_time = asof_time

    def submit(self, child_plans: list, mode: str) -> list[FillEvent]:
        """Fill each valid plan at reference ± half spread. Plans for symbols
        without a reference price (or with non-positive qty) are skipped loudly."""
        fills: list[FillEvent] = []
        self.last_broker_order_ids = {}                      # LIVE6B-3: rebuilt per submit
        now = self.asof_time
        for plan in child_plans:
            self.submitted.append(plan)
            qty = float(getattr(plan, "qty", 0.0))
            symbol = str(getattr(plan, "symbol", ""))
            if qty <= 0.0:
                continue
            price = self.prices.get(symbol)
            if price is None or price <= 0.0:
                logger.warning("PaperBroker: no reference price for %s; plan skipped.", symbol)
                continue
            signed = 1.0 if getattr(plan, "side", "BUY") == "BUY" else -1.0
            half_spread = price * self.spread_bps / 2.0 / 10_000.0
            fill_price = max(price + signed * half_spread, 1e-6)
            self._fill_counter += 1
            self.positions[symbol] = self.positions.get(symbol, 0.0) + signed * qty
            self.cash_gbp -= signed * qty * fill_price
            fid = f"{symbol}-paperbroker-{self._fill_counter}"
            ref = getattr(plan, "order_ref", None)
            if ref is not None:
                self.last_broker_order_ids[ref] = fid        # LIVE6B-3: ref -> broker order id
            fills.append(FillEvent(
                order_id=fid,
                symbol=symbol, qty=qty, fill_price=fill_price,
                decision_price=price, arrival_price=price,
                slippage_bps=self.spread_bps / 2.0,
                fill_timestamp=now,
            ))
        return fills

    def account_state(self, asof_time: datetime) -> BrokerState:
        return BrokerState(
            broker="PAPER",
            connected=True,
            account_id="paper-local",
            asof_timestamp=asof_time,
            nav_gbp=self.nav_gbp,
            cash_gbp=self.cash_gbp,
            buying_power_gbp=max(self.cash_gbp, 0.0),
            positions=dict(self.positions),
        )

    def open_orders(self, asof_time: datetime) -> list[BrokerOpenOrder]:
        """Paper fills are synchronous and terminal — nothing ever rests — so there are no
        open orders to reconcile (read-only; never submits)."""
        return []
