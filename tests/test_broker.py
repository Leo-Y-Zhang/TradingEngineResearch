"""
Broker layer tests — Protocol, PaperBroker, IBKR translation (ROADMAP Phase 6).

The PaperBroker is the LIVE-code-path stand-in: deterministic local fills,
zero market access. The IBKR adapter's pure translation logic is tested
offline; its network paths require ib-insync + a gateway and are excluded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from broker.paper import PaperBroker
from broker.protocol import BrokerProtocol
from data.data_contracts import BrokerState
from execution.execution_engine import ChildOrderPlan

_T = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)


def _plan(symbol="AAPL", side="BUY", qty=100.0, order_type="LIMIT",
          offset=2.0, tag="passive") -> ChildOrderPlan:
    return ChildOrderPlan(symbol=symbol, side=side, qty=qty, order_type=order_type,
                          limit_offset_bps=offset if order_type == "LIMIT" else None,
                          scheduled_offset_minutes=0.0, participation=0.02,
                          slice_index=0, tag=tag)


class TestPaperBroker:

    def test_satisfies_the_protocol(self):
        broker = PaperBroker(prices={"AAPL": 100.0})
        assert isinstance(broker, BrokerProtocol)

    def test_fills_are_deterministic_and_priced_from_reference(self):
        broker = PaperBroker(prices={"AAPL": 100.0}, spread_bps=10.0)
        fills = broker.submit([_plan(side="BUY", qty=100.0)], mode="LIVE")
        assert len(fills) == 1
        f = fills[0]
        assert f.symbol == "AAPL" and f.qty == 100.0
        assert f.fill_price == pytest.approx(100.05)      # ref + half spread (BUY)
        again = broker.submit([_plan(side="BUY", qty=100.0)], mode="LIVE")
        assert again[0].fill_price == f.fill_price        # deterministic

    def test_sell_fills_below_reference(self):
        broker = PaperBroker(prices={"AAPL": 100.0}, spread_bps=10.0)
        f = broker.submit([_plan(side="SELL")], mode="LIVE")[0]
        assert f.fill_price == pytest.approx(99.95)

    def test_unknown_symbol_and_zero_qty_are_skipped(self):
        broker = PaperBroker(prices={"AAPL": 100.0})
        fills = broker.submit([_plan(symbol="ZZZZ"), _plan(qty=0.0)], mode="LIVE")
        assert fills == []

    def test_account_state_is_a_valid_contract(self):
        broker = PaperBroker(prices={"AAPL": 100.0}, nav_gbp=1_000_000.0)
        state = broker.account_state(asof_time=_T)
        assert isinstance(state, BrokerState)
        state.validate_for_mode("LIVE")                   # connected, positive NAV
        assert state.broker == "PAPER"

    def test_records_submissions_for_audit(self):
        broker = PaperBroker(prices={"AAPL": 100.0})
        broker.submit([_plan()], mode="LIVE")
        assert len(broker.submitted) == 1


class TestIBKRTranslation:

    def test_limit_plan_translates_to_limit_order_params(self):
        from broker.ibkr import order_params_from_plan
        p = order_params_from_plan(_plan(side="BUY", qty=100.0, order_type="LIMIT",
                                         offset=2.0), reference_price=100.0)
        assert p["action"] == "BUY"
        assert p["order_type"] == "LMT"
        assert p["total_quantity"] == 100.0
        assert p["limit_price"] == pytest.approx(99.98)   # passive: BUY below ref

    def test_market_plan_translates_to_market_order(self):
        from broker.ibkr import order_params_from_plan
        p = order_params_from_plan(_plan(side="SELL", order_type="MARKET",
                                         offset=None, tag="derisk"),
                                   reference_price=100.0)
        assert p["order_type"] == "MKT"
        assert p["action"] == "SELL"
        assert "limit_price" not in p

    def test_invalid_reference_price_raises(self):
        from broker.ibkr import order_params_from_plan
        with pytest.raises(ValueError, match="reference_price"):
            order_params_from_plan(_plan(), reference_price=0.0)

    def test_ibkr_broker_reports_disconnected_without_gateway(self):
        from broker.ibkr import IBKRBroker
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        assert broker.connected is False                  # never silently "up"
        state = broker.account_state(asof_time=_T)
        with pytest.raises(ValueError, match="connected"):
            state.validate_for_mode("LIVE")               # fail-closed by contract


class TestIBKRFillCommission:
    """§17 cash leg (a): the pure ib-insync commission extraction (offline-testable)."""

    def test_reads_a_delivered_commission_report(self):
        from types import SimpleNamespace

        from broker.ibkr import _fill_commission
        f = SimpleNamespace(commissionReport=SimpleNamespace(execId="e1", commission=1.32))
        assert _fill_commission(f) == pytest.approx(1.32)

    def test_none_until_the_report_is_actually_delivered(self):
        # ib-insync attaches a default-constructed CommissionReport (execId='') until the real
        # one arrives — reading 0.0 off that would INVENT a zero-commission fact.
        from types import SimpleNamespace

        from broker.ibkr import _fill_commission
        f = SimpleNamespace(commissionReport=SimpleNamespace(execId="", commission=0.0))
        assert _fill_commission(f) is None

    def test_none_on_missing_or_invalid_values(self):
        from types import SimpleNamespace

        from broker.ibkr import _fill_commission
        assert _fill_commission(SimpleNamespace()) is None                      # no report at all
        bad = SimpleNamespace(commissionReport=SimpleNamespace(execId="e1", commission=float("nan")))
        assert _fill_commission(bad) is None                                    # non-finite refused
        neg = SimpleNamespace(commissionReport=SimpleNamespace(execId="e1", commission=-2.0))
        assert _fill_commission(neg) is None                                    # negative refused
