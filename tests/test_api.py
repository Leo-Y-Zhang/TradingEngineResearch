"""Tests for the observation/control API (ROADMAP Phase 6 item 3) — ops/api.py.

Skipped entirely if FastAPI / its test client is unavailable (the ``app`` extra);
both are installed in the validated env.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backtesting.harness import _reset_engine_state  # noqa: E402
from core.config import EngineSettings  # noqa: E402
from core.engine.engine import CycleResult  # noqa: E402
from ops.api import create_app, summarize_result  # noqa: E402
from ops.run_loop import EngineService  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]


def _prices(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {s: 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=n)))
            for i, s in enumerate(SYMBOLS)}
    return pd.DataFrame(data, index=idx)


def _provider(prices: pd.DataFrame):
    def provider(asof, symbols):
        sliced = prices.loc[: pd.Timestamp(asof)]
        return sliced if len(sliced) >= 2 else prices
    return provider


def _result(target=None) -> CycleResult:
    return CycleResult(
        mode="RESEARCH", asof_time=datetime(2023, 1, 1), blocked=False,
        regime_label="NORMAL", regime_probs={}, crisis={"level": "NONE"},
        execution_regime="NORMAL", vol_forecasts={"AAA": 0.2}, signal_scores={},
        predictions={"AAA": (0.01, 0.0, 0.0, 0.0, 0.0)}, decisions={},
        optimizer_result={}, risk_snapshot={}, target_weights=target or {"AAA": 0.5},
        achieved_weights=None, live_orders_submitted=0,
        monitoring_snapshot={"RISK": {"drawdown": 0.0}}, alerts=[],
    )


class _FakeEngine:
    def __init__(self, result: CycleResult):
        self.result = result

    def run_cycle(self, inputs):
        return self.result


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(123)
    yield


def _client(tmp_path) -> tuple[TestClient, EngineService, pd.DataFrame]:
    prices = _prices()
    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                 persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices))
    svc.engine = _FakeEngine(_result())
    app = create_app(svc, clock=lambda: prices.index[-1].to_pydatetime())
    return TestClient(app), svc, prices


def test_summarize_result_is_jsonable():
    summary = summarize_result(_result())
    assert summary["mode"] == "RESEARCH"
    assert summary["live_orders_submitted"] == 0
    assert summary["target_weights"] == {"AAA": 0.5}
    # the summary must be JSON-serialisable (no live objects / numpy scalars)
    import json
    json.dumps(summary)


def test_summarize_result_none():
    assert summarize_result(None) is None


def test_health(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "RESEARCH"
    assert body["broker_connected"] is None  # RESEARCH has no broker


def test_root_serves_html_dashboard(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "TRADING ENGINE" in body
    # The dashboard reads the JSON endpoints client-side, so it must reference them.
    assert "/status" in body and "/book" in body


def test_cycle_latest_404_before_any_cycle(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    assert client.get("/cycle/latest").status_code == 404


def test_cycle_run_then_observe(tmp_path):
    client, svc, _prices = _client(tmp_path)
    r = client.post("/cycle/run")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "RESEARCH"
    assert body["live_orders_submitted"] == 0
    assert svc.cycle_count == 1

    assert client.get("/cycle/latest").status_code == 200
    status = client.get("/status").json()
    assert status["cycle_count"] == 1
    assert status["universe"] == SYMBOLS

    book = client.get("/book").json()
    assert book["current_book"] == {"AAA": 0.5}

    mon = client.get("/monitoring").json()
    assert "snapshot" in mon and "alerts" in mon


def test_cycle_run_disabled_in_live(tmp_path):
    client, svc, _prices = _client(tmp_path)
    svc.mode = "LIVE"  # the on-demand trigger must be refused in LIVE
    r = client.post("/cycle/run")
    assert r.status_code == 403
    assert svc.cycle_count == 0


def _engage_latch(svc, client):
    """Drive one hard-stop cycle so the durable kill-switch latch engages."""
    r = _result()
    r.blocked = True
    r.risk_snapshot = {"hard_stop": True, "active_flags": ["KILL"]}
    svc.engine = _FakeEngine(r)
    client.post("/cycle/run")


def test_status_exposes_kill_switch(tmp_path):
    client, svc, _prices = _client(tmp_path)
    assert client.get("/status").json()["kill_switch_latched"] is False
    _engage_latch(svc, client)
    st = client.get("/status").json()
    assert st["kill_switch_latched"] is True
    assert "KILL" in (st["kill_switch_reason"] or "")


def test_kill_switch_reset_clears_latch(tmp_path):
    client, svc, _prices = _client(tmp_path)
    _engage_latch(svc, client)
    assert client.get("/status").json()["kill_switch_latched"] is True

    resp = client.post("/kill-switch/reset", params={"operator": "alice", "reason": "all clear"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cleared"] is True
    assert body["status"]["kill_switch_latched"] is False
    assert client.get("/status").json()["kill_switch_latched"] is False


def test_kill_switch_reset_disabled_in_live(tmp_path):
    client, svc, _prices = _client(tmp_path)
    svc.mode = "LIVE"  # re-enabling LIVE over the unauthenticated API must be refused
    resp = client.post("/kill-switch/reset", params={"operator": "alice"})
    assert resp.status_code == 403


def test_kill_switch_reset_requires_operator(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    resp = client.post("/kill-switch/reset")  # missing required 'operator'
    assert resp.status_code == 422


def test_metrics_endpoints(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    client.post("/cycle/run")  # generate at least one cycle's metrics
    j = client.get("/metrics")
    assert j.status_code == 200
    body = j.json()
    assert "counters" in body and "gauges" in body

    p = client.get("/metrics/prometheus")
    assert p.status_code == 200
    assert "text/plain" in p.headers["content-type"]
    assert "engine_cycles_total" in p.text


# ── held-book reconciliation: operator-gated resolve endpoint ──────────────────────────

def _seed_open_item(svc):
    svc.open_reconciliations = [{"id": "rec1", "order_id": "o1", "symbol": "AAA", "side": "BUY",
                                 "delta_qty": 30.0, "broker_filled_qty": 70.0, "ref_price": 100.0,
                                 "asof": "2024-01-01T00:00:00", "status": "OPEN"}]


def test_status_exposes_open_reconciliations(tmp_path):
    client, svc, _prices = _client(tmp_path)
    _seed_open_item(svc)
    assert len(client.get("/status").json()["open_reconciliations"]) == 1


def test_reconciliation_resolve_books_open_item(tmp_path):
    client, svc, _prices = _client(tmp_path)
    _seed_open_item(svc)
    resp = client.post("/reconciliation/resolve",
                       params={"item_id": "rec1", "operator": "alice", "reason": "confirmed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved"] is True
    assert body["status"]["open_reconciliations"] == []          # closed -> no longer OPEN
    assert svc.open_reconciliations[0]["status"] == "CLOSED"


def test_reconciliation_resolve_disabled_in_live(tmp_path):
    client, svc, _prices = _client(tmp_path)
    svc.mode = "LIVE"   # booking a disconnect-fill over the unauthenticated API must be refused
    resp = client.post("/reconciliation/resolve", params={"item_id": "rec1", "operator": "alice"})
    assert resp.status_code == 403


def test_reconciliation_resolve_requires_operator(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    resp = client.post("/reconciliation/resolve", params={"item_id": "rec1"})  # missing 'operator'
    assert resp.status_code == 422


def test_reconciliation_resolve_unknown_item_returns_false(tmp_path):
    client, _svc, _prices = _client(tmp_path)
    resp = client.post("/reconciliation/resolve", params={"item_id": "nope", "operator": "alice"})
    assert resp.status_code == 200 and resp.json()["resolved"] is False


def test_reconciliation_resolve_reject_decision(tmp_path):
    client, svc, _prices = _client(tmp_path)
    _seed_open_item(svc)
    resp = client.post("/reconciliation/resolve",
                       params={"item_id": "rec1", "operator": "alice", "reason": "spurious",
                               "decision": "REJECT"})
    assert resp.status_code == 200 and resp.json()["resolved"] is True
    assert svc.open_reconciliations[0]["decision"] == "REJECT"


# ── Phase 8: control-API authentication ───────────────────────────────────────────────

def _auth_pair(tmp_path, token=None):
    """Same construction as _client above, plus an api_token. Returns the service
    too, so a test can flip mode to LIVE the way only a deployment could."""
    prices = _prices()
    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                   persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices))
    svc.engine = _FakeEngine(_result())
    client = TestClient(create_app(svc, clock=lambda: prices.index[-1].to_pydatetime(),
                                   api_token=token))
    return client, svc


def _auth_app(tmp_path, token=None):
    return _auth_pair(tmp_path, token)[0]


DATA_ROUTES = ("/status", "/book", "/monitoring", "/metrics", "/cycle/latest",
               "/metrics/prometheus")


def test_no_token_configured_keeps_todays_open_behaviour(tmp_path):
    c = _auth_app(tmp_path)
    # /cycle/latest legitimately 404s before any cycle has run; what matters here
    # is that nothing is challenged for credentials when no token is configured.
    for path in DATA_ROUTES:
        assert c.get(path).status_code != 401, path


def test_configured_token_is_required_on_data_and_control_routes(tmp_path):
    c = _auth_app(tmp_path, token="s3cret")
    for path in DATA_ROUTES:
        assert c.get(path).status_code == 401, path
    assert c.post("/cycle/run").status_code == 401
    assert c.post("/kill-switch/reset", params={"operator": "me"}).status_code == 401
    assert c.post("/reconciliation/resolve",
                  params={"item_id": "x", "operator": "me", "decision": "REJECT"}
                  ).status_code == 401


def test_correct_token_accepted_and_near_misses_rejected(tmp_path):
    c = _auth_app(tmp_path, token="s3cret")
    assert c.get("/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    for bad in ("Bearer wrong", "s3cret", "Basic s3cret", "Bearer  s3cret", "bearer s3cret", ""):
        assert c.get("/status", headers={"Authorization": bad}).status_code == 401, bad


def test_health_and_dashboard_stay_open(tmp_path):
    """A liveness probe should not need a secret, and / is static HTML, not data."""
    c = _auth_app(tmp_path, token="s3cret")
    assert c.get("/health").status_code == 200
    assert c.get("/").status_code == 200


def test_off_loopback_without_a_token_is_refused_before_binding():
    from ops.api import assert_bind_is_safe
    assert_bind_is_safe("127.0.0.1", None)        # loopback, no token: fine
    assert_bind_is_safe("::1", None)
    assert_bind_is_safe("0.0.0.0", "s3cret")      # exposed but authenticated: fine
    for host in ("0.0.0.0", "10.0.0.5", "192.168.1.20"):
        with pytest.raises(RuntimeError, match="without authentication"):
            assert_bind_is_safe(host, None)


def test_a_token_does_not_unlock_live_mutations(tmp_path):
    """Auth is not authorisation to trade. Adding a token must not silently
    re-open the actions the LIVE mode gates refuse - otherwise "we added auth"
    would quietly become "we enabled remote LIVE control"."""
    c, svc = _auth_pair(tmp_path, token="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    assert c.get("/status", headers=auth).status_code == 200   # token works
    svc.mode = "LIVE"
    assert c.post("/cycle/run", headers=auth).status_code == 403
    assert c.post("/kill-switch/reset", params={"operator": "me"},
                  headers=auth).status_code == 403
