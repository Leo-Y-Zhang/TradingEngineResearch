"""The ops API as it is actually SERVED — a real uvicorn socket, not a TestClient.

Round 2 shipped an application-level trusted-proxy control (``create_app(
trusted_proxy_header=...)``, OFF by default) and tested it entirely through
``fastapi.testclient``. That proved nothing about the deployed system:
``uvicorn.run()`` defaults ``proxy_headers=True`` with
``forwarded_allow_ips="127.0.0.1"``, and the shipped bind is loopback, so uvicorn's
own ``ProxyHeadersMiddleware`` trusted every caller and rewrote ``scope["client"]``
from an untrusted ``X-Forwarded-For`` **before the application ran**.

Measured against the pre-fix ``serve_combined`` on a real port
(``127.0.0.1:8731``), with ``X-Forwarded-For: 203.0.113.99, 198.51.100.7``::

    auth_failed client_ip= '198.51.100.7' peer_ip= '198.51.100.7' len(client_ip)= 12
    request     client_ip= '198.51.100.7' peer_ip= '198.51.100.7' len(client_ip)= 12

and with a 4,000-character header::

    auth_failed client_ip= 'AAAA...' len(client_ip)= 4000

So the trail recorded a forgery as fact, the rate-limiter key was attacker-chosen
(a fresh budget per request), and an unbounded attacker string reached the log.

Every test here drives :func:`ops.run_loop.build_api_server` — the same function
the container entrypoint runs — so the property under test is a property of the
shipped server, not of a test harness.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
import httpx  # noqa: E402

from backtesting.harness import _reset_engine_state  # noqa: E402
from core.config import EngineSettings  # noqa: E402
from core.engine.engine import CycleResult  # noqa: E402
from ops.api import api_uvicorn_kwargs, install_security_logging  # noqa: E402
from ops.run_loop import EngineService, build_api_server  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]
TOKEN = "s3cret-ops-token"
FORGED = "203.0.113.99, 198.51.100.7"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(123)
    yield


def _prices(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {s: 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=n)))
         for i, s in enumerate(SYMBOLS)}, index=idx)


def _result() -> CycleResult:
    return CycleResult(
        mode="RESEARCH", asof_time=datetime(2023, 1, 1), blocked=False,
        regime_label="NORMAL", regime_probs={}, crisis={"level": "NONE"},
        execution_regime="NORMAL", vol_forecasts={"AAA": 0.2}, signal_scores={},
        predictions={"AAA": (0.01, 0.0, 0.0, 0.0, 0.0)}, decisions={},
        optimizer_result={}, risk_snapshot={}, target_weights={"AAA": 0.5},
        achieved_weights=None, live_orders_submitted=0,
        monitoring_snapshot={"RISK": {"drawdown": 0.0}}, alerts=[],
    )


class _FakeEngine:
    def __init__(self, result: CycleResult) -> None:
        self.result = result

    def run_cycle(self, inputs):  # noqa: ANN001 - test double
        return self.result


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _service(tmp_path: Path) -> EngineService:
    prices = _prices()
    settings = EngineSettings(
        mode="RESEARCH", universe=SYMBOLS, api_token=TOKEN,
        persistence={"state_dir": str(tmp_path)},
    )
    svc = EngineService(settings, price_provider=lambda asof, symbols: prices)
    svc.engine = _FakeEngine(_result())
    return svc


@contextlib.contextmanager
def _served(tmp_path: Path):
    """Run the shipped server on a real loopback port with its durable trails on.

    Yields ``(base_url, trail_path, alert_path)``. The trails are what the test
    reads: the point is what the SERVER recorded about the caller, and there is no
    way to see that from the response body.
    """
    svc = _service(tmp_path)
    handles = install_security_logging(svc.settings)
    port = _free_port()
    server = build_api_server(svc, host="127.0.0.1", port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        if not server.started:
            raise RuntimeError("the API server did not start within 30s")
        yield (f"http://127.0.0.1:{port}", tmp_path / "security.jsonl",
               tmp_path / "security-alerts.jsonl")
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)
        for handler in (handles.trail, handles.alerts):
            if handler is not None:
                handler.flush()
        handles.detach()


def _events(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── the blocker: the server must not honour an untrusted forwarded header ─────────────

def test_the_shipped_server_ignores_a_forged_forwarded_header(tmp_path):
    """The recorded client IP must be the real socket peer, whatever the caller claims."""
    with _served(tmp_path) as (base, trail, _alerts):
        response = httpx.get(f"{base}/status", headers={"X-Forwarded-For": FORGED},
                             timeout=10.0)
        assert response.status_code == 401, response.text

    events = _events(trail)
    denied = [e for e in events if e.get("event") == "auth_failed"]
    assert denied, f"no auth_failed line in the trail: {events}"
    for event in denied:
        assert event["client_ip"] == "127.0.0.1", (
            f"the server recorded {event['client_ip']!r} as the client: an untrusted "
            "X-Forwarded-For was honoured before the application ran"
        )
        assert event["peer_ip"] == "127.0.0.1"
    assert "203.0.113.99" not in trail.read_text(encoding="utf-8")
    assert "198.51.100.7" not in trail.read_text(encoding="utf-8")


def test_the_shipped_server_keeps_a_forged_header_out_of_the_trail_entirely(tmp_path):
    """SEC-10: the round-2 code capped the forwarded value at 64 chars on the
    application path only, and the uvicorn path had no cap at all — 4,000 chosen
    characters per request went into the trail. With the middleware off the header
    is not read at all, which is the stronger property."""
    with _served(tmp_path) as (base, trail, _alerts):
        response = httpx.get(f"{base}/status", headers={"X-Forwarded-For": "A" * 4000},
                             timeout=10.0)
        assert response.status_code == 401

    text = trail.read_text(encoding="utf-8")
    assert "A" * 100 not in text, "an attacker-chosen string reached the security trail"
    longest = max(len(line) for line in text.splitlines())
    assert longest < 1000, f"a {longest}-character trail line is attacker-grown"


def test_a_rotating_forged_header_cannot_mint_a_fresh_rate_limit_budget(tmp_path):
    """The budget has to belong to the caller, not to a string the caller chooses.
    Under the pre-fix server each forged address was a different limiter key, so
    rotating the header bought an unlimited read budget."""
    with _served(tmp_path) as (base, trail, _alerts):
        seen = set()
        with httpx.Client(base_url=base, timeout=10.0) as client:
            for i in range(12):
                client.get("/status", headers={"X-Forwarded-For": f"9.9.9.{i}"})
    for event in _events(trail):
        if event.get("event") in {"auth_failed", "request"}:
            seen.add(event.get("client_ip"))
    assert seen == {"127.0.0.1"}, (
        f"the server distinguished callers by a caller-supplied header: {sorted(seen)}"
    )


# ── the server really serves: read a real response, not a status code ────────────────

def test_the_real_server_answers_the_operator_and_refuses_everyone_else(tmp_path):
    """A green suite is not a working application (standards §2). This starts the
    shipped server, makes real HTTP requests over a socket and reads the bodies."""
    with _served(tmp_path) as (base, trail, alerts):
        health = httpx.get(f"{base}/health", timeout=10.0)
        assert health.status_code == 200, health.text
        assert health.json()["status"] in {"ok", "degraded"}, health.json()

        anonymous = httpx.get(f"{base}/status", timeout=10.0)
        assert anonymous.status_code == 401
        assert anonymous.json()["detail"] == "missing or invalid bearer token"

        authorised = httpx.get(f"{base}/status", timeout=10.0,
                               headers={"Authorization": f"Bearer {TOKEN}"})
        assert authorised.status_code == 200, authorised.text
        body = authorised.json()
        assert body["mode"] == "RESEARCH"
        assert authorised.headers["X-Request-ID"]

    # ... and the security trail of that session is on disk, both files.
    assert any(e.get("event") == "request" for e in _events(trail))
    assert [e for e in _events(alerts) if e.get("event") == "auth_failed"]


# ── the other deployment entry point (uvicorn --factory) ─────────────────────────────

def test_api_uvicorn_kwargs_turns_the_proxy_middleware_off():
    assert api_uvicorn_kwargs()["proxy_headers"] is False


def test_the_container_entrypoint_serves_uvicorn_with_proxy_headers_off():
    """``docker run ... api`` does not go through build_api_server — it execs
    uvicorn directly, where the default is proxy_headers=ON. The flag is part of
    the control and is pinned here, because the shell script is the only place it
    can be stated."""
    script = (REPO_ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
    invocation = script.split("exec uvicorn", 1)
    assert len(invocation) == 2, "the api mode no longer execs uvicorn - re-check this test"
    command = invocation[1].split(";;", 1)[0]
    assert "--no-proxy-headers" in command, (
        "the container api entrypoint runs uvicorn with its proxy-header default ON, "
        "so an untrusted X-Forwarded-For rewrites the client address again"
    )


def test_uvicorn_really_does_default_the_middleware_on(tmp_path):
    """Negative control on the finding itself. If a future uvicorn stops enabling
    ProxyHeadersMiddleware by default over loopback, the tests above would pass for
    a reason that has nothing to do with this code, and we should know."""
    import uvicorn
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    default = uvicorn.Config(app=lambda *_: None, host="127.0.0.1", port=_free_port())
    assert default.proxy_headers is True
    assert "127.0.0.1" in default.forwarded_allow_ips
    middleware = ProxyHeadersMiddleware(app=lambda *_: None)
    assert "127.0.0.1" in middleware.trusted_hosts


# ── defence in depth for an operator who runs uvicorn by hand ────────────────────────

def test_a_hand_rolled_uvicorn_cannot_launder_a_forged_address_into_the_trail(tmp_path, caplog):
    """``api_uvicorn_kwargs`` covers both shipped entry points, but nothing stops an
    operator running ``uvicorn ops.api:...`` themselves and leaving the default on.
    ProxyHeadersMiddleware marks its rewrite by setting the client PORT to 0 — a
    real TCP peer never has source port 0 — so the app can still refuse to treat
    that address as a fact."""
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    from ops.api import UNTRUSTED_FORWARDED_IP, create_app
    from ops.api_security import SECURITY_LOGGER_NAME

    caplog.set_level(logging.DEBUG, logger=SECURITY_LOGGER_NAME)
    app = create_app(_service(tmp_path), api_token=TOKEN)
    # The REAL uvicorn middleware, wrapped around the app exactly as uvicorn wraps
    # it. "testclient" stands in for the loopback peer that its default
    # forwarded_allow_ips=127.0.0.1 trusts.
    served = ProxyHeadersMiddleware(app, trusted_hosts=["testclient"])
    with TestClient(served) as client:
        assert client.get("/status",
                          headers={"X-Forwarded-For": "203.0.113.99"}).status_code == 401
    events = [r.security for r in caplog.records
              if getattr(r, "security", None) and r.security["event"] == "auth_failed"]
    assert events
    assert events[0]["client_ip"] == UNTRUSTED_FORWARDED_IP
    assert events[0]["peer_ip"] == UNTRUSTED_FORWARDED_IP
    assert events[0]["forwarded_claim"] == "203.0.113.99", (
        "the claim must still be kept for forensics - just never used as an identity"
    )
