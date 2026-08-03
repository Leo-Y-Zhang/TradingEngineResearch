"""
Tests for the Alpaca broker adapter (broker/alpaca.py) — the convenient free PAPER-validation
broker. Exercises the BrokerProtocol contract against a FAKE alpaca-py TradingClient (the real
paper API is hit only by the operator during the supervised session, like IBKR's gateway path).
The broker still builds the REAL alpaca request objects, so request construction is validated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca")  # optional 'brokers' extra; skip cleanly when not installed

from broker.alpaca import AlpacaBroker, _map_alpaca_status  # noqa: E402

_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _order(oid="aid", client_order_id="oref", symbol="AAPL", filled_qty="0",
           filled_avg_price=None, status="accepted"):
    return SimpleNamespace(id=oid, client_order_id=client_order_id, symbol=symbol,
                           filled_qty=filled_qty, filled_avg_price=filled_avg_price,
                           status=SimpleNamespace(value=status), filled_at=_TS)


def _plan(symbol="AAPL", side="BUY", qty=10, order_ref="oref", order_type="MARKET", slice_index=0):
    return SimpleNamespace(symbol=symbol, side=side, qty=qty, order_ref=order_ref,
                           order_type=order_type, slice_index=slice_index)


class _FakeClient:
    def __init__(self, account=None, positions=None, fill=None, open_orders=None,
                 closed_orders=None, raise_on_submit=False):
        self._acct = account or SimpleNamespace(account_number="DU1", cash="50000",
                                                portfolio_value="100000", status="ACTIVE")
        self._positions = positions if positions is not None else []
        self._fill = fill
        self._open = open_orders if open_orders is not None else []
        self._closed = closed_orders if closed_orders is not None else []
        self._raise_on_submit = raise_on_submit
        self.submitted: list = []
        self.raise_on_account = False

    def get_account(self):
        if self.raise_on_account:
            raise RuntimeError("invalid credentials")
        return self._acct

    def get_all_positions(self):
        return self._positions

    def submit_order(self, order_data):
        if self._raise_on_submit:
            raise RuntimeError("403 insufficient buying power")
        self.submitted.append(order_data)
        return _order(oid="aid-" + str(getattr(order_data, "client_order_id", "x")),
                      client_order_id=getattr(order_data, "client_order_id", None),
                      symbol=order_data.symbol, status="accepted")

    def get_order_by_client_id(self, client_id):
        return self._fill if self._fill is not None else _order(client_order_id=client_id)

    def get_orders(self, filter=None):  # noqa: A002 - matches the alpaca-py signature
        status = str(getattr(getattr(filter, "status", None), "value", "")).lower()
        return self._closed if status == "closed" else self._open


def _connected(client, **kw) -> AlpacaBroker:
    b = AlpacaBroker("key", "secret", client_factory=lambda: client, sleep=lambda _s: None, **kw)
    b.connect()
    return b


def test_connect_verifies_creds_and_sets_connected():
    assert _connected(_FakeClient()).connected is True


def test_connect_failure_stays_disconnected():
    c = _FakeClient()
    c.raise_on_account = True
    b = AlpacaBroker("k", "s", client_factory=lambda: c, sleep=lambda _s: None)
    assert b.connect() is False and b.connected is False


def test_disconnect_clears_session():
    b = _connected(_FakeClient())
    b.disconnect()
    assert b.connected is False


def test_submit_refused_off_live():
    assert _connected(_FakeClient()).submit([_plan()], "PAPER") == []


def test_submit_refused_when_disconnected():
    b = AlpacaBroker("k", "s", client_factory=lambda: _FakeClient(), sleep=lambda _s: None)  # not connected
    assert b.submit([_plan()], "LIVE") == []


def test_submit_roundtrips_client_order_id_and_returns_fill():
    fill = _order(oid="AID9", client_order_id="oref1", symbol="AAPL",
                  filled_qty="10", filled_avg_price="101.5", status="filled")
    c = _FakeClient(fill=fill)
    b = _connected(c)
    fills = b.submit([_plan(symbol="AAPL", side="BUY", qty=10, order_ref="oref1")], "LIVE")
    assert c.submitted[0].client_order_id == "oref1"          # orderRef -> client_order_id round-trip
    assert b.last_broker_order_ids["oref1"] == "aid-oref1"    # broker order id captured at submit
    assert len(fills) == 1
    assert fills[0].symbol == "AAPL" and fills[0].qty == 10.0 and fills[0].fill_price == 101.5


def test_submit_unfilled_order_returns_no_fill():
    c = _FakeClient(fill=_order(client_order_id="oref", filled_qty="0", status="accepted"))
    fills = _connected(c).submit([_plan(order_ref="oref")], "LIVE")
    assert fills == []                                        # rests at the broker -> resync handles it


def test_fill_commission_none_when_alpaca_reports_none():
    # §17 cash leg (a): Alpaca's trading API reports no commission on equity paper fills —
    # the FillEvent carries None (record only what the broker reports; never invent).
    fill = _order(oid="AID9", client_order_id="oref1", symbol="AAPL",
                  filled_qty="10", filled_avg_price="101.5", status="filled")
    fills = _connected(_FakeClient(fill=fill)).submit([_plan(order_ref="oref1")], "LIVE")
    assert fills[0].commission is None


def test_fill_commission_passed_through_when_reported():
    # defensively read: if the order object DOES carry a finite non-negative commission
    # (e.g. a future API surface), it is passed through onto the FillEvent.
    fill = _order(oid="AID9", client_order_id="oref1", symbol="AAPL",
                  filled_qty="10", filled_avg_price="101.5", status="filled")
    fill.commission = "0.25"
    fills = _connected(_FakeClient(fill=fill)).submit([_plan(order_ref="oref1")], "LIVE")
    assert fills[0].commission == pytest.approx(0.25)


def test_fill_commission_invalid_value_is_dropped_not_fatal():
    # a junk/negative reported value must neither crash the fill path nor be recorded.
    fill = _order(oid="AID9", client_order_id="oref1", symbol="AAPL",
                  filled_qty="10", filled_avg_price="101.5", status="filled")
    fill.commission = "-1.0"
    fills = _connected(_FakeClient(fill=fill)).submit([_plan(order_ref="oref1")], "LIVE")
    assert len(fills) == 1 and fills[0].commission is None    # fill kept, bad commission refused


def test_account_state_maps_positions_signed_and_nav():
    pos = [SimpleNamespace(symbol="AAPL", qty="100", side=SimpleNamespace(value="long")),
           SimpleNamespace(symbol="MSFT", qty="50", side=SimpleNamespace(value="short"))]
    bs = _connected(_FakeClient(positions=pos)).account_state(_TS)
    assert bs.connected is True
    assert bs.positions == {"AAPL": 100.0, "MSFT": -50.0}     # short -> negative
    assert bs.nav_gbp == 100000.0 and bs.cash_gbp == 50000.0


def test_account_state_disconnected_fails_closed():
    b = AlpacaBroker("k", "s", client_factory=lambda: _FakeClient(), sleep=lambda _s: None)  # not connected
    assert b.account_state(_TS).connected is False


def test_open_orders_maps_to_broker_open_order_shape():
    o = _order(oid="AID1", client_order_id="oref2", symbol="AAPL",
               filled_qty="3", filled_avg_price="100.0", status="partially_filled")
    e = _connected(_FakeClient(open_orders=[o])).open_orders(_TS)[0]
    assert e["order_ref"] == "oref2" and e["broker_order_id"] == "AID1" and e["symbol"] == "AAPL"
    assert e["filled_qty"] == 3.0 and e["avg_fill_price"] == 100.0
    assert e["status"] == "PARTIALLY_FILLED"


def test_open_orders_disconnected_is_empty():
    b = AlpacaBroker("k", "s", client_factory=lambda: _FakeClient(), sleep=lambda _s: None)
    assert b.open_orders(_TS) == []                           # fail-closed read


def test_open_orders_reports_recently_closed_terminal_orders():
    # P2 fix: an Alpaca market order goes TERMINAL immediately (e.g. rejected). It is NOT under
    # OPEN, so open_orders must ALSO surface recently-CLOSED orders -> the resync resolves a stuck
    # WORKING record to REJECTED (unblocking the symbol) instead of bricking it forever.
    rej = _order(oid="AIDR", client_order_id="orefR", symbol="AAPL", filled_qty="0", status="rejected")
    entries = _connected(_FakeClient(open_orders=[], closed_orders=[rej])).open_orders(_TS)
    rejected = [e for e in entries if e["order_ref"] == "orefR"]
    assert rejected and rejected[0]["status"] == "REJECTED" and rejected[0]["broker_order_id"] == "AIDR"


def test_open_orders_reports_closed_fill_for_resync_recovery():
    # a market order that FILLED but was missed by the single submit-poll is recovered: the CLOSED
    # query surfaces it with filled_qty/avg_fill_price so the resync's discovered-fill path books it.
    done = _order(oid="AIDF", client_order_id="orefF", symbol="AAPL",
                  filled_qty="10", filled_avg_price="100.0", status="filled")
    entries = _connected(_FakeClient(open_orders=[], closed_orders=[done])).open_orders(_TS)
    e = [x for x in entries if x["order_ref"] == "orefF"][0]
    assert e["status"] == "FILLED" and e["filled_qty"] == 10.0 and e["avg_fill_price"] == 100.0


def test_submit_failure_propagates_so_ordermanager_marks_uncertain():
    # P2 fix: a submit that ERRORS (e.g. 403) must NOT be swallowed into a false 'acked WORKING' --
    # it propagates so OrderManager.place records SUBMISSION_UNCERTAIN (resync-resolvable, visible).
    b = _connected(_FakeClient(raise_on_submit=True))
    with pytest.raises(Exception):
        b.submit([_plan(order_ref="oref")], "LIVE")


def test_status_mapping_is_conservative():
    assert _map_alpaca_status("filled") == "FILLED"
    assert _map_alpaca_status("partially_filled") == "PARTIALLY_FILLED"
    assert _map_alpaca_status("canceled") == "CANCELLED"
    assert _map_alpaca_status("rejected") == "REJECTED"
    assert _map_alpaca_status("expired") == "EXPIRED"
    assert _map_alpaca_status("pending_cancel") == "WORKING"  # a pending-cancel order can still fill
    assert _map_alpaca_status("accepted") == "WORKING"


def test_conforms_to_broker_protocol():
    from broker.protocol import BrokerProtocol
    assert isinstance(_connected(_FakeClient()), BrokerProtocol)


def test_is_not_internal_paper_so_reconcile_runs():
    # unlike the internal sim PaperBroker, an Alpaca paper account reports REAL positions, so
    # _reconcile must reconcile against it -> is_paper must be falsy.
    assert getattr(_connected(_FakeClient()), "is_paper", False) is False
