"""Tests for ops/observability.py + its wiring (ROADMAP Phase 6 item 5).

Covers the alert sinks, the metrics registry, the config factory, and that the
run-loop routes computed alerts to the sink and updates metrics each cycle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtesting.harness import _reset_engine_state
from core.config import EngineSettings, make_alert_sink
from core.engine.engine import CycleResult
from ops.observability import (
    CompositeAlertSink,
    JsonlAlertSink,
    LoggingAlertSink,
    MetricsRegistry,
    NullAlertSink,
    alert_severity,
    normalize_alert,
)
from ops.run_loop import EngineService

SYMBOLS = ["AAA", "BBB"]


# ── normalize / severity ─────────────────────────────────────────────────────────────

def test_normalize_alert_from_dict():
    a = normalize_alert({"severity": "RED", "category": "drawdown", "message": "x"})
    assert a["severity"] == "RED" and a["category"] == "drawdown"


def test_normalize_alert_from_object():
    class Ev:
        severity = "AMBER"
        event_type = "kill_switch"
        description = "tripped"
    a = normalize_alert(Ev())
    assert a["severity"] == "AMBER"
    assert a["category"] == "kill_switch"
    assert a["message"] == "tripped"


def test_normalize_alert_defaults_and_kind_alias():
    a = normalize_alert({"kind": "reconciliation"})
    assert a["severity"] == "INFO"
    assert a["category"] == "reconciliation"  # falls back to 'kind'


def test_alert_severity_defaults_info():
    assert alert_severity({}) == "INFO"
    assert alert_severity({"severity": "red"}) == "RED"


# ── sinks ────────────────────────────────────────────────────────────────────────────

def test_logging_alert_sink_maps_severity_to_level(caplog):
    sink = LoggingAlertSink("tradingengineresearch.alerts.test")
    with caplog.at_level(logging.WARNING, logger="tradingengineresearch.alerts.test"):
        sink.emit({"severity": "RED", "message": "boom"})
        sink.emit({"severity": "AMBER", "message": "warn"})
    levels = {r.levelno for r in caplog.records}
    assert logging.ERROR in levels  # RED → ERROR
    assert logging.WARNING in levels  # AMBER → WARNING


def test_logging_alert_sink_info_below_warning(caplog):
    sink = LoggingAlertSink("tradingengineresearch.alerts.test2")
    with caplog.at_level(logging.INFO, logger="tradingengineresearch.alerts.test2"):
        sink.emit({"severity": "INFO", "message": "fyi"})
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_jsonl_alert_sink_appends(tmp_path):
    path = tmp_path / "sub" / "alerts.jsonl"
    sink = JsonlAlertSink(path)
    sink.emit({"severity": "RED", "message": "a"})
    sink.emit({"severity": "INFO", "message": "b"})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["severity"] == "RED"
    assert json.loads(lines[1])["message"] == "b"


def test_composite_sink_fans_out_and_isolates_failures():
    received: list[dict] = []

    class Good:
        def emit(self, alert):
            received.append(alert)

    class Bad:
        def emit(self, alert):
            raise RuntimeError("sink down")

    sink = CompositeAlertSink([Bad(), Good()])  # Bad must not block Good
    sink.emit({"severity": "RED", "message": "x"})
    assert len(received) == 1


def test_null_sink_is_noop():
    NullAlertSink().emit({"severity": "RED"})  # must not raise


# ── metrics ──────────────────────────────────────────────────────────────────────────

def test_metrics_counters_and_gauges():
    m = MetricsRegistry()
    m.inc("cycles_total")
    m.inc("cycles_total", 2.0)
    m.set_gauge("book_size", 5)
    assert m.get_counter("cycles_total") == 3.0
    assert m.get_gauge("book_size") == 5.0
    assert m.get_gauge("missing") is None


def test_metrics_labels_are_distinct():
    m = MetricsRegistry()
    m.inc("alerts_total", severity="RED")
    m.inc("alerts_total", severity="RED")
    m.inc("alerts_total", severity="INFO")
    assert m.get_counter("alerts_total", severity="RED") == 2.0
    assert m.get_counter("alerts_total", severity="INFO") == 1.0


def test_metrics_prometheus_format():
    m = MetricsRegistry()
    m.inc("engine_cycles_total", 3.0)
    m.set_gauge("engine_book_size", 2.0)
    m.inc("engine_alerts_total", severity="RED")
    text = m.render_prometheus()
    assert "# TYPE engine_cycles_total counter" in text
    assert "engine_cycles_total 3.0" in text
    assert "# TYPE engine_book_size gauge" in text
    assert 'engine_alerts_total{severity="RED"} 1.0' in text


def test_metrics_snapshot_shape():
    m = MetricsRegistry()
    m.inc("c", 1.0)
    m.set_gauge("g", 2.0)
    snap = m.snapshot()
    assert snap["counters"][0]["name"] == "c"
    assert snap["gauges"][0]["value"] == 2.0


# ── config factory ───────────────────────────────────────────────────────────────────

def test_make_alert_sink_variants(tmp_path):
    base = {"persistence": {"state_dir": str(tmp_path)}, "universe": SYMBOLS}
    assert isinstance(make_alert_sink(EngineSettings(alerting={"sink": "logging"}, **base)), LoggingAlertSink)
    assert isinstance(make_alert_sink(EngineSettings(alerting={"sink": "null"}, **base)), NullAlertSink)
    assert isinstance(make_alert_sink(EngineSettings(alerting={"sink": "jsonl"}, **base)), JsonlAlertSink)
    assert isinstance(make_alert_sink(EngineSettings(alerting={"sink": "both"}, **base)), CompositeAlertSink)


def test_make_alert_sink_jsonl_default_path(tmp_path):
    settings = EngineSettings(alerting={"sink": "jsonl"}, persistence={"state_dir": str(tmp_path)})
    sink = make_alert_sink(settings)
    sink.emit({"severity": "INFO", "message": "x"})
    assert (tmp_path / "alerts.jsonl").exists()


# ── run-loop wiring ──────────────────────────────────────────────────────────────────

def _prices(n: int = 90, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {s: 100.0 * (1 + i * 0.1) * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
            for i, s in enumerate(SYMBOLS)}
    return pd.DataFrame(data, index=idx)


def _provider(prices):
    def provider(asof, symbols):
        sliced = prices.loc[: pd.Timestamp(asof)]
        return sliced if len(sliced) >= 2 else prices
    return provider


def _result(alerts=None, blocked=False) -> CycleResult:
    return CycleResult(
        mode="RESEARCH", asof_time=datetime(2023, 1, 1), blocked=blocked,
        regime_label="NORMAL", regime_probs={}, crisis={"level": "NONE"},
        execution_regime="NORMAL", vol_forecasts={}, signal_scores={},
        predictions={}, decisions={}, optimizer_result={}, risk_snapshot={},
        target_weights={"AAA": 0.5}, achieved_weights=None, live_orders_submitted=0,
        monitoring_snapshot={}, alerts=alerts or [],
    )


class _FakeEngine:
    def __init__(self, result):
        self.result = result

    def run_cycle(self, inputs):
        return self.result


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(99)
    yield


class _RecordingSink:
    def __init__(self):
        self.alerts: list[dict] = []

    def emit(self, alert):
        self.alerts.append(alert)


def test_run_once_routes_alerts_to_sink_and_updates_metrics(tmp_path):
    prices = _prices()
    sink = _RecordingSink()
    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                 persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices), alert_sink=sink)
    svc.engine = _FakeEngine(_result(alerts=[
        {"severity": "RED", "category": "drawdown", "message": "deep"},
        {"severity": "INFO", "category": "model", "message": "ok"},
    ]))
    svc.run_once(prices.index[-1].to_pydatetime())

    assert len(sink.alerts) == 2
    assert svc.metrics.get_counter("engine_cycles_total") == 1.0
    assert svc.metrics.get_counter("engine_alerts_total", severity="RED") == 1.0
    assert svc.metrics.get_gauge("engine_cycle_count") == 1.0


def test_run_once_metrics_count_blocked(tmp_path):
    prices = _prices()
    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                 persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices), alert_sink=NullAlertSink())
    svc.engine = _FakeEngine(_result(blocked=True))
    svc.run_once(prices.index[-1].to_pydatetime())
    assert svc.metrics.get_counter("engine_blocked_cycles_total") == 1.0


def test_alerting_failure_does_not_break_cycle(tmp_path):
    prices = _prices()

    class BoomSink:
        def emit(self, alert):
            raise RuntimeError("sink exploded")

    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                 persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices), alert_sink=BoomSink())
    svc.engine = _FakeEngine(_result(alerts=[{"severity": "RED", "message": "x"}]))
    result = svc.run_once(prices.index[-1].to_pydatetime())  # must not raise
    assert result is not None
    assert svc.cycle_count == 1
