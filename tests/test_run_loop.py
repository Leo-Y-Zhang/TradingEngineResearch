"""Tests for the scheduled run-loop (ROADMAP Phase 6 item 3) — ops/run_loop.py.

The run-loop is the online analog of the backtest harness: build PIT inputs →
run cycle → carry book → persist. These tests pin orchestration semantics (book
carry, blocked-carry, achieved-vs-target, restart durability, persistence,
broker reconciliation, fail-closed universe) using a fake engine where the
real 13-step pipeline is not what's under test, plus one real-engine smoke test.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtesting.harness import _reset_engine_state
from core.config import EngineSettings
from core.engine.engine import CycleResult
from data.data_contracts import BrokerState
from ops.run_loop import (
    CycleBusyError,
    EngineService,
    LoopState,
    build_cycle_inputs,
    run_forever,
)

SYMBOLS = ["AAA", "BBB", "CCC"]


# ── live data-shape contract (the bug that left cycle_count=0 in a real LIVE run) ──────

def _tidy_prices(symbols=("AAPL",), n: int = 40) -> pd.DataFrame:
    """A synthetic tidy long OHLCV frame in the exact shape data.price_ingestion.fetch_prices
    returns (columns date,symbol,open,high,low,close,volume; datetime 'date'; RangeIndex)."""
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = []
    for s in symbols:
        for i, d in enumerate(dates):
            px = 100.0 + i
            rows.append({"date": d, "symbol": s, "open": px, "high": px, "low": px,
                         "close": px, "volume": 1.0e6})
    return pd.DataFrame(rows)[["date", "symbol", "open", "high", "low", "close", "volume"]]


def test_to_wide_produces_datetimeindex_close_matrix():
    from data.price_ingestion import to_wide

    wide = to_wide(_tidy_prices(symbols=("AAPL", "MSFT"), n=30))
    assert isinstance(wide.index, pd.DatetimeIndex)          # the shape build_cycle_inputs requires
    assert set(wide.columns) == {"AAPL", "MSFT"}
    assert wide["AAPL"].iloc[-1] == pytest.approx(100.0 + 29)


def test_live_price_shape_feeds_build_cycle_inputs():
    # the regression: the live provider's tidy frame MUST shape into the wide DatetimeIndex
    # matrix build_cycle_inputs requires, or every cycle dies at input assembly (cycle_count=0).
    from data.price_ingestion import to_wide

    wide = to_wide(_tidy_prices(symbols=("AAPL",), n=40))
    asof = wide.index[-1].to_pydatetime()
    inputs = build_cycle_inputs(wide, asof, ["AAPL"], {}, 100000.0)
    assert inputs.symbols == ["AAPL"] and inputs.asof_time == asof


def test_default_price_provider_shapes_to_wide(monkeypatch):
    # the live provider (network) must return the wide shape; inject a fake fetch so we cover the
    # SHAPING (the part that was broken) without hitting yfinance.
    from data import price_ingestion

    monkeypatch.setattr(price_ingestion, "fetch_prices",
                        lambda syms, start, end: _tidy_prices(tuple(syms), 40))
    from ops.run_loop import _default_price_provider

    wide = _default_price_provider(datetime(2024, 3, 1), ["AAPL"])
    assert isinstance(wide.index, pd.DatetimeIndex) and "AAPL" in wide.columns


def _prices(n: int = 120, symbols: list[str] = SYMBOLS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {}
    for i, s in enumerate(symbols):
        steps = rng.normal(0.0004, 0.012, size=n)
        data[s] = 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=idx)


def _provider(prices: pd.DataFrame):
    def provider(asof, symbols):
        sliced = prices.loc[: pd.Timestamp(asof)]
        return sliced if len(sliced) >= 2 else prices
    return provider


def _settings(tmp_path, mode: str = "RESEARCH", **kw) -> EngineSettings:
    return EngineSettings(
        mode=mode, universe=SYMBOLS, persistence={"state_dir": str(tmp_path)}, **kw
    )


def _result(mode="RESEARCH", blocked=False, target=None, achieved=None,
            live=0, alerts=None, snapshot=None, risk=None, fills=None, order_intents=None) -> CycleResult:
    return CycleResult(
        mode=mode, asof_time=datetime(2023, 1, 1), blocked=blocked,
        regime_label="NORMAL", regime_probs={}, crisis={"level": "NONE"},
        execution_regime="NORMAL", vol_forecasts={}, signal_scores={},
        predictions={}, decisions={}, optimizer_result={}, risk_snapshot=risk or {},
        target_weights=target or {}, achieved_weights=achieved,
        live_orders_submitted=live, monitoring_snapshot=snapshot or {},
        alerts=alerts or [], fills=fills or [], order_intents=order_intents or [],
    )


class _FakeEngine:
    def __init__(self, results: list[CycleResult]):
        self._results = list(results)
        self.i = 0
        self.seen: list = []

    def run_cycle(self, inputs):
        self.seen.append(inputs)
        r = self._results[min(self.i, len(self._results) - 1)]
        self.i += 1
        return r


class _FakeBroker:
    def __init__(self, positions=None, nav=1_000_000.0, connected=True, raises=False, cash="NAV"):
        self._positions = positions or {}
        self._nav = nav
        self._connected = connected
        self._raises = raises
        self._cash = cash          # "NAV" (legacy default) | explicit float | None = not reported

    @property
    def connected(self) -> bool:
        return self._connected

    def account_state(self, asof) -> BrokerState:
        if self._raises:
            raise RuntimeError("broker query failed")
        return BrokerState(
            broker="FAKE", connected=self._connected, nav_gbp=self._nav,
            cash_gbp=self._nav if self._cash == "NAV" else self._cash,
            buying_power_gbp=self._nav, positions=self._positions, asof_timestamp=asof,
        )

    def submit(self, child_plans, mode):
        return []


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(123)
    yield


# ── build_cycle_inputs ─────────────────────────────────────────────────────────────

def test_build_cycle_inputs_is_pit_safe():
    prices = _prices()
    asof = prices.index[50].to_pydatetime()
    inputs = build_cycle_inputs(prices.loc[: prices.index[50]], asof, SYMBOLS, {}, 1e6)
    assert inputs.symbols == SYMBOLS
    assert inputs.prices.index[-1] <= pd.Timestamp(asof)
    assert inputs.returns_matrix is not None
    assert set(inputs.market_microstructure) == set(SYMBOLS)


def test_build_cycle_inputs_validates():
    with pytest.raises(TypeError):
        build_cycle_inputs(pd.DataFrame({"AAA": [1, 2, 3]}), datetime(2023, 1, 1), ["AAA"], {}, 1e6)
    prices = _prices()
    with pytest.raises(ValueError):
        build_cycle_inputs(prices, prices.index[-1].to_pydatetime(), ["NOPE"], {}, 1e6)


# ── LoopState ───────────────────────────────────────────────────────────────────────

def test_loop_state_roundtrip():
    ls = LoopState(current_book={"AAA": 0.5}, cycle_count=3, live_orders_total=2,
                   peak_nav=1.2e6, last_asof="2023-01-01T00:00:00")
    assert LoopState.from_json(ls.to_json()) == ls


# ── run_once orchestration (fake engine) ─────────────────────────────────────────────

def test_run_once_persists_and_counts(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    result = svc.run_once(prices.index[-1].to_pydatetime())

    assert result.live_orders_submitted == 0
    assert svc.cycle_count == 1
    assert svc.live_orders_total == 0
    assert svc.current_book == {"AAA": 0.5}
    # both the learning state and the durable loop state were written
    assert (tmp_path / "state.json").exists()
    ls = json.loads((tmp_path / "loop_state.json").read_text())
    assert ls["cycle_count"] == 1
    assert ls["current_book"] == {"AAA": 0.5}


def test_book_carries_research(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5}), _result(target={"BBB": 0.3})])
    svc.run_once(prices.index[-2].to_pydatetime())
    assert svc.current_book == {"AAA": 0.5}
    svc.run_once(prices.index[-1].to_pydatetime())
    assert svc.current_book == {"BBB": 0.3}
    assert svc.cycle_count == 2


def test_blocked_cycle_carries_book_unchanged(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([
        _result(target={"AAA": 0.5}),
        _result(blocked=True, target={"ZZZ": 9.0}),  # would be wrong to adopt
    ])
    svc.run_once(prices.index[-2].to_pydatetime())
    svc.run_once(prices.index[-1].to_pydatetime())
    assert svc.current_book == {"AAA": 0.5}


def test_paper_carries_achieved_not_target(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(mode="PAPER", target={"AAA": 0.5}, achieved={"AAA": 0.4})])
    svc.broker = None  # isolate book selection from reconciliation
    svc.run_once(prices.index[-1].to_pydatetime())
    assert svc.current_book == {"AAA": 0.4}


def test_state_saved_each_cycle(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(), _result()])
    calls = {"n": 0}
    orig = svc.state_store.save

    def spy(**kw):
        calls["n"] += 1
        return orig(**kw)

    svc.state_store.save = spy  # type: ignore[method-assign]
    svc.run_once(prices.index[-2].to_pydatetime())
    svc.run_once(prices.index[-1].to_pydatetime())
    assert calls["n"] == 2


def test_restart_restores_book_and_counters(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    svc.run_once(prices.index[-1].to_pydatetime())

    svc2 = EngineService(_settings(tmp_path), price_provider=_provider(prices)).start()
    assert svc2.current_book == {"AAA": 0.5}
    assert svc2.cycle_count == 1


def test_empty_universe_fails_closed(tmp_path):
    s = EngineSettings(mode="RESEARCH", universe=[], persistence={"state_dir": str(tmp_path)})
    with pytest.raises(ValueError):
        EngineService(s, price_provider=_provider(_prices()))


def test_start_is_idempotent(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.start()
    svc.start()  # no error, no double-restore side effects
    assert svc.cycle_count == 0


def test_stop_persists(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    svc.run_once(prices.index[-1].to_pydatetime())
    svc.stop()
    assert (tmp_path / "loop_state.json").exists()


# ── broker reconciliation (surfacing only) ───────────────────────────────────────────

def test_reconciliation_surfaces_divergence_alert(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(mode="PAPER", target={"AAA": 0.5}, achieved={"AAA": 0.5})])
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0}, "seed")  # §17 internal book
    svc.broker = _FakeBroker(positions={})  # broker holds nothing → divergence vs the ledger
    svc.run_once(prices.index[-1].to_pydatetime())
    kinds = [a.get("kind") for a in svc.last_alerts if isinstance(a, dict)]
    assert "reconciliation" in kinds


def test_reconciliation_broker_failure_is_amber_not_fatal(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(mode="PAPER", target={"AAA": 0.5}, achieved={"AAA": 0.5})])
    svc.broker = _FakeBroker(raises=True)
    result = svc.run_once(prices.index[-1].to_pydatetime())  # must not raise
    assert result is not None
    recon = [a for a in svc.last_alerts if isinstance(a, dict) and a.get("kind") == "reconciliation"]
    assert recon and recon[0]["severity"] == "AMBER"


def test_no_reconciliation_when_aligned(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    nav = 1_000_000.0
    last = prices.iloc[-1]
    aligned = {"AAA": float(round(0.5 * nav / float(last["AAA"])))}
    svc.engine = _FakeEngine([_result(mode="PAPER", target={"AAA": 0.5}, achieved={"AAA": 0.5})])
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": aligned["AAA"]}, "seed")  # ledger == broker
    svc.broker = _FakeBroker(positions=aligned, nav=nav)
    svc.run_once(prices.index[-1].to_pydatetime())
    recon = [a for a in svc.last_alerts if isinstance(a, dict) and a.get("kind") == "reconciliation"]
    assert not recon


# ── §17: _reconcile uses the immutable ledger replay as the authoritative internal side ─

def _real_broker_svc(tmp_path, broker_positions, nav=1_000_000.0, cash="NAV"):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.broker = _FakeBroker(positions=broker_positions, nav=nav, cash=cash)  # is_paper absent -> real
    return svc


def test_reconcile_clean_when_ledger_replay_matches_broker(tmp_path):
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0}, "t")   # ledger book == broker
    assert svc._reconcile(datetime(2024, 1, 1)) is None          # clean -> no alert


def test_reconcile_surfaces_break_when_ledger_replay_differs(tmp_path):
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 60.0}, "t")    # ledger 60 vs broker 100
    alert = svc._reconcile(datetime(2024, 1, 1))
    assert alert is not None and alert["kind"] == "reconciliation"


def test_reconcile_missing_ledger_fill_surfaces_break(tmp_path):
    # the audit gap §17 exists to catch: the broker holds a position the ledger never recorded
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})              # no FILL seeded -> ledger empty
    alert = svc._reconcile(datetime(2024, 1, 1))
    assert alert is not None and alert["kind"] == "reconciliation"


def test_reconcile_internal_side_is_ledger_not_current_book(tmp_path):
    # current_book says one thing, the ledger says another -> the LEDGER governs (directive §17).
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})
    svc.current_book = {"AAA": 0.5}                               # weight book would imply ~broker match
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 40.0}, "t")    # but the ledger says 40
    alert = svc._reconcile(datetime(2024, 1, 1))
    assert alert is not None                                      # ledger 40 vs broker 100 -> break


def test_reconcile_no_op_for_no_broker_and_paper(tmp_path):
    from broker.paper import PaperBroker
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.broker = None
    assert svc._reconcile(datetime(2024, 1, 1)) is None           # no broker (RESEARCH)
    svc.broker = PaperBroker(nav_gbp=1_000_000.0)                 # is_paper True -> no-op
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0}, "t")
    assert svc._reconcile(datetime(2024, 1, 1)) is None           # paper no-op


def _trading_result(qty=100.0):
    """A CycleResult that traded this cycle: one signed fill + its intent + achieved book."""
    from types import SimpleNamespace
    fill = SimpleNamespace(order_id="o1", symbol="AAA", qty=qty, fill_price=100.0, slippage_bps=2.0)
    intent = SimpleNamespace(symbol="AAA", direction="BUY")
    return _result(mode="PAPER", achieved={"AAA": 0.5}, fills=[fill], order_intents=[intent])


def test_reconcile_records_this_cycles_fills_before_reconciling_no_spurious_break(tmp_path):
    # P1 regression fix: the ledger must reflect THIS cycle's fills BEFORE _reconcile reads it, so a
    # cycle that trades does not false-break against a broker that already holds the fill.
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_trading_result(qty=100.0)])
    # broker already reflects the fill on BOTH legs (positions AND the cash it paid out)
    svc.broker = _FakeBroker(positions={"AAA": 100.0}, cash=990_000.0)
    svc.run_once(prices.index[-1].to_pydatetime())
    recon = [a for a in svc.last_alerts if isinstance(a, dict) and a.get("kind") == "reconciliation"]
    assert not recon                                              # ledger (this cycle) == broker -> clean


def test_reconcile_real_divergence_still_breaks_after_ordering_fix(tmp_path):
    # the fix must not mask a GENUINE divergence: broker holds more than this cycle's recorded fill.
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_trading_result(qty=100.0)])
    # cash consistent -> the surfaced divergence is the genuine POSITIONS one
    svc.broker = _FakeBroker(positions={"AAA": 175.0}, cash=990_000.0)  # ledger 100 vs broker 175
    svc.run_once(prices.index[-1].to_pydatetime())
    recon = [a for a in svc.last_alerts if isinstance(a, dict) and a.get("kind") == "reconciliation"]
    assert recon                                                 # genuine divergence still surfaces


def test_run_once_records_broker_commission_to_ledger(tmp_path):
    # §17 cash leg (a): a cycle whose fills carry a broker-REPORTED commission records it as a
    # COMMISSION event via record_cycle — the run-loop stays the SOLE ledger writer (the engine
    # never appends; the fact rides back on the FillEvent).
    from types import SimpleNamespace
    prices = _prices()
    fill = SimpleNamespace(order_id="e1", symbol="AAA", qty=100.0, fill_price=100.0,
                           slippage_bps=2.0, commission=1.5)
    intent = SimpleNamespace(symbol="AAA", direction="BUY")
    res = _result(mode="PAPER", achieved={"AAA": 0.5}, fills=[fill], order_intents=[intent])
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([res])
    svc.broker = None
    svc.run_once(prices.index[-1].to_pydatetime())
    comms = svc.ledger.events("COMMISSION")
    assert len(comms) == 1
    assert comms[0].payload["amount"] == pytest.approx(1.5)
    assert comms[0].payload["order_id"] == "e1"


# ── §17 golden masters: pin _reconcile behavior around the cash-leg extension ──────────

def test_reconcile_cash_divergence_now_breaks(tmp_path):
    # CONSCIOUS golden-master flip (was ..._positions_only_ignores_broker_cash, pinned pre-slice):
    # with the opening deposit + priced signed fills recorded, the §17 cash leg is live — internal
    # replayed cash 990,000 vs broker 1,000,000 must now surface a structured cash break
    # (surfacing only; positions clean, so the break set is exactly {cash}).
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})              # _FakeBroker cash == nav == 1e6
    svc.ledger.append("CASH", {"action": "deposit", "amount": 1_000_000.0, "ccy": "GBP"}, "t")
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0, "fill_price": 100.0}, "t")
    alert = svc._reconcile(datetime(2024, 1, 1))
    assert alert is not None and alert["kind"] == "reconciliation"
    assert {b["dimension"] for b in alert["detail"]} == {"cash"}


def test_reconcile_golden_master_divergence_is_audited_to_ledger(tmp_path):
    # GOLDEN MASTER: a divergence surfaced by _reconcile is appended to the immutable ledger as
    # a RECONCILIATION event (carrying the structured break detail) as well as raised as an alert.
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_trading_result(qty=100.0)])
    svc.broker = _FakeBroker(positions={"AAA": 175.0})            # ledger 100 vs broker 175 -> break
    svc.run_once(prices.index[-1].to_pydatetime())
    events = svc.ledger.events("RECONCILIATION")
    assert events, "a surfaced break must be audited as a RECONCILIATION ledger event"
    detail = events[-1].payload.get("detail") or []
    assert any(b.get("dimension") == "position" for b in detail)  # structured break, never auto-applied


# ── §17 cash leg: _reconcile reconciles ledger-replayed cash against broker cash ────────

def _seed_cash_book(svc, deposit=1_000_000.0, qty=100.0, price=100.0):
    """Seed a self-contained ledger: opening deposit + one priced signed BUY fill."""
    svc.ledger.append("CASH", {"action": "deposit", "amount": deposit, "ccy": "GBP"}, "t")
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": qty, "fill_price": price}, "t")


def test_reconcile_cash_clean_when_ledger_matches_broker(tmp_path):
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0}, cash=990_000.0)  # 1e6 - 100x100
    _seed_cash_book(svc)
    assert svc._reconcile(datetime(2024, 1, 1)) is None


def test_reconcile_commissions_flow_into_cash_leg_no_false_break(tmp_path):
    # THE original blocker for wiring cash: unrecorded commissions would false-break the leg.
    # With §17(a) COMMISSION events recorded, the replayed cash charges them and reconciles.
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0}, cash=989_998.5)  # broker also charged 1.5
    _seed_cash_book(svc)
    svc.ledger.append("COMMISSION", {"order_id": "e1", "symbol": "AAA", "amount": 1.5,
                                     "source": "broker_fill"}, "t")
    assert svc._reconcile(datetime(2024, 1, 1)) is None


def test_reconcile_cash_leg_skipped_when_broker_reports_no_cash(tmp_path, caplog):
    # a broker with no cash truth (BrokerState.cash_gbp None) must SKIP the leg — never
    # false-break — and say so with a structured log line.
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0}, cash=None)
    _seed_cash_book(svc)
    with caplog.at_level(logging.INFO, logger="ops.run_loop"):
        assert svc._reconcile(datetime(2024, 1, 1)) is None
    assert any("cash leg skipped" in r.getMessage() for r in caplog.records)


def test_reconcile_cash_leg_skipped_without_opening_deposit(tmp_path):
    # no CASH baseline in the trail (a pre-cash-leg ledger): internal cash would read 0 and
    # false-break every cycle — the leg is skipped until the opening deposit event exists.
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0, "fill_price": 100.0}, "t")
    assert svc._reconcile(datetime(2024, 1, 1)) is None


def test_reconcile_cash_leg_skipped_when_replay_incomplete(tmp_path):
    # an unpriced FILL makes the replayed cash unreliable (understated by exactly that fill):
    # skip the cash leg rather than surface a break that trains the operator to ignore alerts.
    # The positions leg still reconciles (and here it matches).
    svc = _real_broker_svc(tmp_path, {"AAA": 100.0})
    svc.ledger.append("CASH", {"action": "deposit", "amount": 1_000_000.0, "ccy": "GBP"}, "t")
    svc.ledger.append("FILL", {"symbol": "AAA", "signed_qty": 100.0}, "t")    # no fill_price
    assert svc._reconcile(datetime(2024, 1, 1)) is None


def test_reconcile_cash_break_audited_via_run_once(tmp_path):
    # end-to-end LIVE-path convention: run_once seeds the deposit + records the fill, the broker
    # reports inconsistent cash -> the RECONCILIATION ledger event carries the cash break.
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_trading_result(qty=100.0)])
    svc.broker = _FakeBroker(positions={"AAA": 100.0})            # cash=nav=1e6 vs internal 990,000
    svc.run_once(prices.index[-1].to_pydatetime())
    events = svc.ledger.events("RECONCILIATION")
    assert events
    dims = {b.get("dimension") for b in (events[-1].payload.get("detail") or [])}
    assert "cash" in dims and "position" not in dims              # positions clean, cash breaks


# ── RUN-3: drawdown read fails SAFE, not OPEN ─────────────────────────────────────────

def test_first_cycle_seeds_initial_cash_deposit(tmp_path):
    # §17 cash leg: the ledger seeds the opening balance once so balances are self-contained.
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    svc.run_once(prices.index[-2].to_pydatetime())
    cash_events = svc.ledger.events("CASH")
    assert cash_events and cash_events[0].payload["action"] == "deposit"
    assert cash_events[0].payload["amount"] == pytest.approx(svc.capital_gbp)
    # idempotent — a second cycle does NOT re-seed
    svc.run_once(prices.index[-1].to_pydatetime())
    assert len(svc.ledger.events("CASH")) == 1


def test_current_drawdown_computes_from_peak(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.peak_nav = 1_000_000.0
    svc.broker = _FakeBroker(nav=900_000.0)               # NAV 10% below peak
    assert svc._current_drawdown() == pytest.approx(0.10)
    assert svc.last_drawdown == pytest.approx(0.10)


def test_current_drawdown_carries_last_known_on_broker_error(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.peak_nav = 1_000_000.0
    svc.broker = _FakeBroker(nav=900_000.0)
    assert svc._current_drawdown() == pytest.approx(0.10)  # establishes last_drawdown
    svc.broker = _FakeBroker(raises=True)                  # broker hiccup
    # RUN-3: must CARRY 0.10, NOT reset to a permissive 0.0 (which would disengage the governor)
    assert svc._current_drawdown() == pytest.approx(0.10)


def test_current_drawdown_carries_on_nonpositive_nav(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.peak_nav = 1_000_000.0
    svc.broker = _FakeBroker(nav=800_000.0)
    assert svc._current_drawdown() == pytest.approx(0.20)
    svc.broker = _FakeBroker(nav=0.0)                      # suspect non-positive NAV read
    assert svc._current_drawdown() == pytest.approx(0.20)  # carried, not treated as 'no drawdown'


# ── run_forever scheduling shell ─────────────────────────────────────────────────────

def test_run_forever_runs_max_cycles(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result()])
    times = iter([prices.index[i].to_pydatetime() for i in (-3, -2, -1)])
    n = run_forever(svc, interval_seconds=0.0, clock=lambda: next(times),
                    max_cycles=3, sleep=lambda s: None)
    assert n == 3
    assert svc.cycle_count == 3


def test_run_forever_swallows_cycle_errors(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))

    class Boom:
        def run_cycle(self, inputs):
            raise RuntimeError("boom")

    svc.engine = Boom()
    n = run_forever(svc, interval_seconds=0.0,
                    clock=lambda: prices.index[-1].to_pydatetime(),
                    max_cycles=2, sleep=lambda s: None)
    assert n == 2          # the loop continued past each failure
    assert svc.cycle_count == 0  # but no cycle completed


def test_run_forever_honours_should_stop(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result()])
    n = run_forever(svc, interval_seconds=0.0,
                    clock=lambda: prices.index[-1].to_pydatetime(),
                    max_cycles=5, should_stop=lambda: True, sleep=lambda s: None)
    assert n == 0


# ── real-engine smoke test (end-to-end, zero live orders) ────────────────────────────

def test_real_engine_research_cycle_places_no_orders(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    result = svc.run_once(prices.index[-1].to_pydatetime())
    assert result.mode == "RESEARCH"
    assert result.live_orders_submitted == 0
    assert svc.live_orders_total == 0
    assert svc.cycle_count == 1
    assert isinstance(svc.last_snapshot, dict)


# ── review fixes: LIVE broker connect (fail-closed) ──────────────────────────────────

class _LiveBroker:
    def __init__(self, connects=True):
        self._connects = connects
        self._connected = False
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        self._connected = self._connects

    @property
    def connected(self) -> bool:
        return self._connected

    def submit(self, child_plans, mode):
        return []

    def account_state(self, asof):
        return BrokerState(broker="FAKE", connected=self._connected, nav_gbp=1e6,
                           cash_gbp=1e6, buying_power_gbp=1e6, asof_timestamp=asof)


def test_start_connects_live_broker(tmp_path):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"  # exercise the start() LIVE branch without a real ib-insync stack
    broker = _LiveBroker(connects=True)
    svc.broker = broker
    svc.start()
    assert broker.connect_calls == 1


def test_start_live_fails_closed_when_broker_cannot_connect(tmp_path):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"
    svc.broker = _LiveBroker(connects=False)
    with pytest.raises(RuntimeError):
        svc.start()


# ── review fix: PAPER reconciliation no longer false-diverges ─────────────────────────

def test_paper_broker_is_not_falsely_reconciled(tmp_path):
    from broker.paper import PaperBroker

    prices = _prices()
    svc = EngineService(_settings(tmp_path, mode="PAPER"), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(mode="PAPER", target={"AAA": 0.5}, achieved={"AAA": 0.5})])
    svc.broker = PaperBroker(nav_gbp=1_000_000.0)  # is_paper=True → reconciliation skipped
    svc.run_once(prices.index[-1].to_pydatetime())
    recon = [a for a in svc.last_alerts if isinstance(a, dict) and a.get("kind") == "reconciliation"]
    assert not recon


# ── review fix: service-level lock serialises cycles ─────────────────────────────────

def test_try_run_once_is_busy_when_lock_held_elsewhere(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result()])
    held = threading.Event()
    release = threading.Event()

    def holder():
        with svc._lock:
            held.set()
            release.wait(2.0)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert held.wait(2.0)
        with pytest.raises(CycleBusyError):
            svc.try_run_once(prices.index[-1].to_pydatetime())
    finally:
        release.set()
        t.join()


def test_try_run_once_runs_when_free(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    result = svc.try_run_once(prices.index[-1].to_pydatetime())
    assert result is not None
    assert svc.cycle_count == 1


# ── RISK-6: durable kill-switch latch (directive §7.4 & §16) ──────────────────────────

def _hardstop(flags=("DRAWDOWN_KILL",)) -> CycleResult:
    """A blocked cycle reporting a latch-worthy hard stop (as risk_manager does when
    a kill switch is active or drawdown hits KILL)."""
    return _result(blocked=True, risk={"hard_stop": True, "active_flags": list(flags)})


def test_loop_state_roundtrip_with_latch():
    latch = {"latched": True, "reason": "KILL", "engaged_at": "2023-01-01T00:00:00",
             "reset_by": None, "reset_at": None}
    ls = LoopState(current_book={"AAA": 0.5}, cycle_count=1, kill_latch=latch)
    assert LoopState.from_json(ls.to_json()) == ls


def test_kill_latch_engages_on_hard_stop(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_hardstop()])
    svc.run_once(prices.index[-1].to_pydatetime())

    assert svc.kill_latch.is_latched
    kinds = [a.get("kind") for a in svc.last_alerts if isinstance(a, dict)]
    assert "kill_switch_latched" in kinds
    ks = svc.ledger.events("KILL_SWITCH")
    assert ks and ks[-1].payload["action"] == "engage"
    # the latch was persisted durably (so a restart re-enters HALTED)
    ls = json.loads((tmp_path / "loop_state.json").read_text())
    assert ls["kill_latch"]["latched"] is True


def test_latched_service_halts_subsequent_cycles(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    fake = _FakeEngine([_hardstop(), _result(target={"AAA": 0.9})])  # 2nd must never be adopted
    svc.engine = fake
    svc.run_once(prices.index[-2].to_pydatetime())   # engages the latch
    assert svc.kill_latch.is_latched
    assert svc.cycle_count == 1

    result = svc.run_once(prices.index[-1].to_pydatetime())  # halted — no engine run
    assert result.blocked is True
    assert result.regime_label == "HALTED"
    assert result.live_orders_submitted == 0
    assert fake.i == 1            # engine NOT called on the halted cycle
    assert svc.cycle_count == 1   # a halted cycle is not a decision cycle
    assert svc.current_book == {}  # book unchanged (target {AAA:0.9} never adopted)


def test_reset_kill_switch_clears_and_resumes(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    fake = _FakeEngine([_hardstop(), _result(target={"AAA": 0.5})])
    svc.engine = fake
    svc.run_once(prices.index[-2].to_pydatetime())  # engages
    assert svc.kill_latch.is_latched

    assert svc.reset_kill_switch("operator-alice", reason="all clear") is True
    assert not svc.kill_latch.is_latched
    assert any(e.payload.get("action") == "reset" for e in svc.ledger.events("KILL_SWITCH"))
    # idempotent: resetting an unlatched switch is a no-op returning False
    assert svc.reset_kill_switch("operator-alice") is False

    result = svc.run_once(prices.index[-1].to_pydatetime())  # trading resumes
    assert result.regime_label == "NORMAL"
    assert fake.i == 2
    assert svc.current_book == {"AAA": 0.5}


def test_kill_latch_survives_restart(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_hardstop()])
    svc.run_once(prices.index[-1].to_pydatetime())  # engages + persists
    assert svc.kill_latch.is_latched

    svc2 = EngineService(_settings(tmp_path), price_provider=_provider(prices)).start()
    assert svc2.kill_latch.is_latched               # restored from durable loop state
    svc2.engine = _FakeEngine([_result(target={"AAA": 0.5})])
    result = svc2.run_once(prices.index[-1].to_pydatetime())
    assert result.regime_label == "HALTED"          # a restarted service stays halted
    assert svc2.engine.i == 0                        # engine never ran


def test_status_reports_latched(tmp_path):
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.engine = _FakeEngine([_hardstop(flags=["KILL"])])
    assert svc.status()["kill_switch_latched"] is False
    svc.run_once(prices.index[-1].to_pydatetime())
    st = svc.status()
    assert st["kill_switch_latched"] is True
    assert "KILL" in (st["kill_switch_reason"] or "")


# ── LIVE6B-1/3: reconnect resync wiring ───────────────────────────────────────────────

class _ResyncSpyEngine:
    def __init__(self, pending=True):
        self._pending = pending
        self.resynced: list = []

    def has_pending_orders(self):
        return self._pending

    def resync_open_orders(self, open_orders, ts):
        self.resynced.append((open_orders, ts))
        return []

    def run_cycle(self, inputs):
        return _result()


class _OpenOrdersBroker:
    connected = True

    def __init__(self, open_orders):
        self._oo = open_orders

    def open_orders(self, asof):
        return self._oo

    def account_state(self, asof):
        return BrokerState(broker="FAKE", connected=True, nav_gbp=1e6, cash_gbp=1e6,
                           buying_power_gbp=1e6, positions={}, asof_timestamp=asof)

    def submit(self, child_plans, mode):
        return []


def test_maybe_resync_resyncs_live_when_pending(tmp_path):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"
    svc.engine = _ResyncSpyEngine(pending=True)
    oo = [{"order_ref": "o1", "broker_order_id": "B", "status": "WORKING", "symbol": "AAA", "filled_qty": 0.0}]
    svc.broker = _OpenOrdersBroker(oo)
    svc._maybe_resync(datetime(2024, 1, 1))
    assert len(svc.engine.resynced) == 1 and svc.engine.resynced[0][0] == oo


def test_maybe_resync_noop_off_live(tmp_path):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))   # RESEARCH
    svc.engine = _ResyncSpyEngine(pending=True)
    svc.broker = _OpenOrdersBroker([])
    svc._maybe_resync(datetime(2024, 1, 1))
    assert svc.engine.resynced == []                                               # never resync off-LIVE


def test_maybe_resync_failure_sets_needs_resync(tmp_path):
    class _BoomBroker(_OpenOrdersBroker):
        def open_orders(self, asof):
            raise RuntimeError("broker read failed")
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"
    svc.engine = _ResyncSpyEngine(pending=True)
    svc.broker = _BoomBroker([])
    svc._maybe_resync(datetime(2024, 1, 1))                                         # must not raise
    assert svc._needs_resync is True                                               # fail-closed gate set


def test_run_forever_stops_driving_once_latched(tmp_path):
    """The scheduled driver must not keep running decision cycles after a hard stop:
    once latched, every further tick is a cheap halted no-op (engine untouched)."""
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    fake = _FakeEngine([_hardstop(), _result(target={"AAA": 0.9})])
    svc.engine = fake
    n = run_forever(svc, interval_seconds=0.0,
                    clock=lambda: prices.index[-1].to_pydatetime(),
                    max_cycles=4, sleep=lambda s: None)
    assert n == 4                 # the driver still terminates on max_cycles
    assert fake.i == 1            # engine ran exactly once (the cycle that latched)
    assert svc.kill_latch.is_latched
    assert svc.cycle_count == 1   # only the one real decision cycle


# ── LIVE6B-2: order-lifecycle persistence across restart ──────────────────────────────

def test_loop_state_roundtrip_with_open_orders():
    oo = [{"order_id": "o1", "approved_qty": 100.0, "status": "SUBMISSION_UNCERTAIN",
           "filled_qty": 0.0, "broker_order_id": None, "symbol": "AAA", "side": "BUY",
           "ref_price": 0.0, "flags": [], "commissions": [], "seen_fills": [], "history": []}]
    ls = LoopState(current_book={"AAA": 0.5}, cycle_count=1, open_orders=oo)
    assert LoopState.from_json(ls.to_json()) == ls


def test_loop_state_loads_without_open_orders_key():
    data = {"current_book": {"AAA": 0.5}, "cycle_count": 1}     # old file: no open_orders key
    assert LoopState.from_json(data).open_orders == []


def test_restart_restores_open_orders_and_sets_needs_resync(tmp_path):
    from execution.order_lifecycle import OrderLifecycle, OrderStatus

    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"
    lc = OrderLifecycle()
    lc.create("o1", 100.0, "t", symbol="AAA", side="BUY")
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING):
        lc.transition("o1", st, "t")
    lc.mark_submission_uncertain("o1", "t")
    svc.engine._order_lifecycle = lc
    svc._persist()                                             # writes loop_state.json incl. open_orders

    svc2 = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc2.mode = "LIVE"
    svc2._restore_loop_state()
    assert svc2._needs_resync is True                          # force a resync before trusting it
    assert svc2.engine.has_pending_orders()                    # the uncertain order is REMEMBERED
    assert svc2.engine._order_lifecycle.get("o1").status == OrderStatus.SUBMISSION_UNCERTAIN


class _GateSpyEngine:
    """Captures the live_submits_blocked flag the run-loop set BEFORE calling run_cycle."""
    def __init__(self):
        self.live_submits_blocked = False
        self.seen_blocked = None

    def has_pending_orders(self):
        return False

    def resync_open_orders(self, open_orders, ts):
        return []

    def run_cycle(self, inputs):
        self.seen_blocked = self.live_submits_blocked
        return _result()


def test_run_loop_gates_live_submits_when_resync_fails(tmp_path):
    # review fix: end-to-end through the run-loop with a real-shaped engine — when the post-restart
    # resync cannot run (broker read fails), _needs_resync stays set and the engine STEP-12 gate
    # (live_submits_blocked) is engaged for the cycle.
    class _BoomBroker(_OpenOrdersBroker):
        def open_orders(self, asof):
            raise RuntimeError("broker read failed")
    prices = _prices()
    svc = EngineService(_settings(tmp_path), price_provider=_provider(prices))
    svc.mode = "LIVE"
    svc.engine = _GateSpyEngine()
    svc.broker = _BoomBroker([])
    svc._needs_resync = True
    svc.run_once(prices.index[-1].to_pydatetime())
    assert svc._needs_resync is True                           # resync failed -> stays gated
    assert svc.engine.seen_blocked is True                     # engine STEP-12 was gated this cycle


# ── slice 2 (held-book reconciliation): surface discovered disconnect-fills ────────────

class _DiscoverEngine:
    """Fake engine that, on resync, surfaces one discovered disconnect-fill via the outbox
    drain (the run-loop turns it into an OPEN reconciliation item)."""
    def __init__(self, discovered):
        self._discovered = list(discovered)
        self._drained = False
        self.booked: list = []
        self.cancelled: list = []

    def has_pending_orders(self):
        return True

    def resync_open_orders(self, open_orders, ts):
        return []

    def drain_discovered_fills(self):
        if self._drained:
            return []
        self._drained = True
        return list(self._discovered)

    def book_reconciled_fill(self, order_id, ts):
        self.booked.append((order_id, ts))

    def cancel_reconciled_order(self, order_id, ts):
        self.cancelled.append((order_id, ts))

    def run_cycle(self, inputs):
        return _result()


def _disc(order_id="o1", symbol="AAA", side="BUY", delta=30.0, broker_filled=70.0, ref=100.0):
    return {"order_id": order_id, "symbol": symbol, "side": side,
            "delta_qty": delta, "broker_filled_qty": broker_filled, "ref_price": ref}


def _live_discover_service(tmp_path, discovered):
    svc = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc.mode = "LIVE"
    svc.engine = _DiscoverEngine(discovered)
    svc.broker = _OpenOrdersBroker([])
    return svc


def test_maybe_resync_surfaces_discovered_fill_as_open_item(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    open_items = [i for i in svc.open_reconciliations if i["status"] == "OPEN"]
    assert len(open_items) == 1
    it = open_items[0]
    assert it["order_id"] == "o1" and it["symbol"] == "AAA" and it["delta_qty"] == pytest.approx(30.0)
    assert it["asof"] == datetime(2024, 1, 1).isoformat()      # aged (timestamped)
    # audited to the immutable ledger
    kinds = [e.payload.get("kind") for e in svc.ledger.events("RECONCILIATION")]
    assert "disconnect_fill_discovered" in kinds


def test_status_surfaces_open_reconciliations(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    assert len(svc.status()["open_reconciliations"]) == 1


def test_discovered_fill_is_not_duplicated_across_resyncs(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    svc.engine._drained = False                                # the SAME discovery re-surfaces
    svc._maybe_resync(datetime(2024, 1, 2))
    assert len([i for i in svc.open_reconciliations if i["status"] == "OPEN"]) == 1   # no duplicate


def test_loop_state_roundtrip_with_open_reconciliations():
    items = [{"id": "o1|70.0", "order_id": "o1", "symbol": "AAA", "side": "BUY",
              "delta_qty": 30.0, "broker_filled_qty": 70.0, "ref_price": 100.0,
              "asof": "2024-01-01T00:00:00", "status": "OPEN"}]
    ls = LoopState(current_book={"AAA": 0.5}, open_reconciliations=items)
    assert LoopState.from_json(ls.to_json()) == ls


def test_loop_state_loads_without_open_reconciliations_key():
    data = {"current_book": {"AAA": 0.5}, "cycle_count": 1}     # old file: no open_reconciliations key
    assert LoopState.from_json(data).open_reconciliations == []


def test_open_reconciliations_persist_across_restart(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    svc._persist()
    svc2 = EngineService(_settings(tmp_path), price_provider=_provider(_prices()))
    svc2.mode = "LIVE"
    svc2._restore_loop_state()
    assert len([i for i in svc2.open_reconciliations if i["status"] == "OPEN"]) == 1


# ── slice 3 (held-book reconciliation): operator-gated resolution ──────────────────────

def test_resolve_reconciliation_books_fill_and_closes_item(tmp_path):
    from ops.ledger import replay_ledger_to_positions
    svc = _live_discover_service(tmp_path, [_disc(delta=30.0, broker_filled=70.0, ref=100.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    resolved = svc.resolve_reconciliation(item_id, operator="alice", reason="confirmed vs Flex",
                                          timestamp=datetime(2024, 1, 2))
    assert resolved is True
    item = svc.open_reconciliations[0]
    assert item["status"] == "CLOSED" and item["operator"] == "alice"
    # booked as an explicit audited FILL the replay picks up
    fills = [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"]
    assert len(fills) == 1 and fills[0].payload["signed_qty"] == pytest.approx(30.0)
    assert replay_ledger_to_positions(svc.ledger).get("AAA") == pytest.approx(30.0)
    # held book updated by the signed weight (30 * 100 / 1e6)
    assert svc.current_book.get("AAA") == pytest.approx(0.003)
    # the lifecycle order was advanced out of HOLD (engine hook called -> symbol unfreezes)
    assert ("o1", datetime(2024, 1, 2).isoformat()) in svc.engine.booked
    # the resolution is itself audited
    kinds = [e.payload.get("kind") for e in svc.ledger.events("RECONCILIATION")]
    assert "disconnect_fill_resolved" in kinds


def test_resolve_reconciliation_sell_signs_negative(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc(side="SELL", delta=20.0, broker_filled=20.0, ref=50.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    svc.resolve_reconciliation(svc.open_reconciliations[0]["id"], operator="bob", reason="x",
                               timestamp=datetime(2024, 1, 2))
    fills = [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"]
    assert fills[0].payload["signed_qty"] == pytest.approx(-20.0)        # SELL -> negative
    assert svc.current_book.get("AAA") == pytest.approx(-0.001)          # -20 * 50 / 1e6


def test_resolve_unknown_or_closed_item_is_rejected(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    assert svc.resolve_reconciliation("nope", operator="a", reason="r",
                                      timestamp=datetime(2024, 1, 2)) is False
    item_id = svc.open_reconciliations[0]["id"]
    assert svc.resolve_reconciliation(item_id, operator="a", reason="r",
                                      timestamp=datetime(2024, 1, 2)) is True
    assert svc.resolve_reconciliation(item_id, operator="a", reason="r",
                                      timestamp=datetime(2024, 1, 3)) is False   # already closed


def test_resolve_aborts_and_keeps_item_open_if_audit_fails(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    book_before = dict(svc.current_book)

    real_append = svc.ledger.append
    def _boom(event_type, payload, ts):
        if event_type == "FILL":
            raise RuntimeError("ledger down")
        return real_append(event_type, payload, ts)
    svc.ledger.append = _boom  # type: ignore[method-assign]

    resolved = svc.resolve_reconciliation(svc.open_reconciliations[0]["id"], operator="a",
                                          reason="r", timestamp=datetime(2024, 1, 2))
    assert resolved is False
    assert svc.open_reconciliations[0]["status"] == "OPEN"   # left OPEN (no unaudited mutation)
    assert svc.current_book == book_before                   # held book unchanged
    assert svc.engine.booked == []                           # lifecycle not advanced


# ── slice 4 (held-book reconciliation): adversarial-review fixes ───────────────────────

def test_resolve_is_crash_idempotent_no_double_book(tmp_path):
    # P1 fix: the booking FILL is fsync'd before the item is closed/persisted. A crash in between
    # restores the item OPEN with the FILL already in the ledger; a re-resolve must NOT append a
    # second FILL (which would double-count the authoritative replay) — it re-applies the lost
    # book delta + closes.
    from ops.ledger import replay_ledger_to_positions
    svc = _live_discover_service(tmp_path, [_disc(delta=30.0, broker_filled=70.0, ref=100.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    # simulate the crash: the FILL was durably appended, but the item-close + book persist were lost
    svc.ledger.append("FILL", {"source": "RESYNC_RECONCILED", "reconciliation_id": item_id,
                               "order_id": "o1", "symbol": "AAA", "side": "BUY", "qty": 30.0,
                               "signed_qty": 30.0, "fill_price": 100.0, "price_estimated": True},
                      "2024-01-01T00:00:00")
    resolved = svc.resolve_reconciliation(item_id, operator="a", reason="retry after crash",
                                          timestamp=datetime(2024, 1, 2))
    assert resolved is True
    fills = [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"]
    assert len(fills) == 1                                   # NOT double-appended
    assert replay_ledger_to_positions(svc.ledger).get("AAA") == pytest.approx(30.0)  # counted once
    assert svc.current_book.get("AAA") == pytest.approx(0.003)   # lost book delta re-applied once
    assert svc.open_reconciliations[0]["status"] == "CLOSED"


def test_resolve_refuses_nonpositive_ref_price_and_keeps_item_open(tmp_path):
    # P2 fix: a 0/NaN ref_price would book the share qty for ZERO cost (free shares -> permanent
    # book/NAV divergence). Refuse, leave the item OPEN, book nothing.
    svc = _live_discover_service(tmp_path, [_disc(delta=30.0, broker_filled=70.0, ref=0.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    book_before = dict(svc.current_book)
    resolved = svc.resolve_reconciliation(item_id, operator="a", reason="r", timestamp=datetime(2024, 1, 2))
    assert resolved is False
    assert svc.open_reconciliations[0]["status"] == "OPEN"   # left OPEN (cannot book at 0 price)
    assert svc.current_book == book_before                   # no zero-cost shares booked
    assert [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"] == []


def test_resolve_reject_closes_item_without_booking(tmp_path):
    # P2 fix: a spurious/duplicate broker over-report is REJECTED — book nothing, audit it, and
    # cancel the parked order out so the symbol unfreezes (no real fill landed).
    from ops.ledger import replay_ledger_to_positions
    svc = _live_discover_service(tmp_path, [_disc(delta=30.0, broker_filled=70.0, ref=100.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    resolved = svc.resolve_reconciliation(item_id, operator="a", reason="spurious broker report",
                                          decision="REJECT", timestamp=datetime(2024, 1, 2))
    assert resolved is True
    item = svc.open_reconciliations[0]
    assert item["status"] == "CLOSED" and item["decision"] == "REJECT"
    assert [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"] == []
    assert replay_ledger_to_positions(svc.ledger).get("AAA") is None     # no phantom position
    kinds = [e.payload.get("kind") for e in svc.ledger.events("RECONCILIATION")]
    assert "disconnect_fill_rejected" in kinds
    assert ("o1", datetime(2024, 1, 2).isoformat()) in svc.engine.cancelled   # order cancelled -> unfreezes
    assert svc.engine.booked == []                           # never booked as a fill


def test_resolve_refuses_unknown_decision(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc()])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    resolved = svc.resolve_reconciliation(item_id, operator="a", reason="r",
                                          decision="MAYBE", timestamp=datetime(2024, 1, 2))
    assert resolved is False
    assert svc.open_reconciliations[0]["status"] == "OPEN"   # unknown decision -> no-op


def test_resolve_books_at_broker_avg_price_when_present(tmp_path):
    # P3 fix: book the disconnect-fill at the broker's TRUE avg fill price (exact cash leg),
    # not the placement-time ref_price estimate.
    svc = _live_discover_service(tmp_path, [{"order_id": "o1", "symbol": "AAA", "side": "BUY",
        "delta_qty": 30.0, "broker_filled_qty": 70.0, "ref_price": 100.0, "avg_fill_price": 105.0}])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    svc.resolve_reconciliation(item_id, operator="a", reason="r", timestamp=datetime(2024, 1, 2))
    fill = [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"][0]
    assert fill.payload["fill_price"] == pytest.approx(105.0)        # broker avg, not ref 100
    assert fill.payload["price_estimated"] is False
    assert svc.current_book.get("AAA") == pytest.approx(30.0 * 105.0 / 1e6)


def test_resolve_falls_back_to_ref_price_estimated_when_no_broker_price(tmp_path):
    svc = _live_discover_service(tmp_path, [_disc(delta=30.0, broker_filled=70.0, ref=100.0)])
    svc._maybe_resync(datetime(2024, 1, 1))
    item_id = svc.open_reconciliations[0]["id"]
    svc.resolve_reconciliation(item_id, operator="a", reason="r", timestamp=datetime(2024, 1, 2))
    fill = [e for e in svc.ledger.events("FILL") if e.payload.get("source") == "RESYNC_RECONCILED"][0]
    assert fill.payload["fill_price"] == pytest.approx(100.0)        # ref_price fallback
    assert fill.payload["price_estimated"] is True
