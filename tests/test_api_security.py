"""Security telemetry + rate limiting on the ops API — ops/api_security.py, ops/api.py.

The financial audit trail (``ops/ledger.py``) is strong; this covers the *security*
trail that was missing: who called the ops API, whether their credentials failed,
and whether anyone is hammering a control endpoint. Skipped entirely without
FastAPI (the ``app`` extra).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backtesting.harness import _reset_engine_state  # noqa: E402
from core.config import EngineSettings  # noqa: E402
from core.engine.engine import CycleResult  # noqa: E402
from ops.api import UNTRUSTED_FORWARDED_IP, create_app  # noqa: E402
from ops.api_security import (  # noqa: E402
    FIELD_MAX_CHARS,
    SECURITY_ALERT_RETENTION_DAYS,
    SECURITY_LOGGER_NAME,
    RateLimiter,
    RateLimitPolicy,
    attach_security_alert_file,
    attach_security_log_file,
    cap_field,
    detach_security_log_file,
)
from ops.run_loop import EngineService  # noqa: E402

SYMBOLS = ["AAA", "BBB", "CCC"]
TOKEN = "s3cret-ops-token"


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
    def __init__(self, result: CycleResult):
        self.result = result

    def run_cycle(self, inputs):
        return self.result


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(123)
    yield


@pytest.fixture()
def seclog(caplog):
    caplog.set_level(logging.DEBUG, logger=SECURITY_LOGGER_NAME)
    return caplog


def _build_app(tmp_path, *, token=None, policy=None, time_fn=None, trusted_proxy_header=None):
    prices = _prices()
    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                   persistence={"state_dir": str(tmp_path)})
    svc = EngineService(settings, price_provider=_provider(prices))
    svc.engine = _FakeEngine(_result())
    kwargs = {"clock": lambda: prices.index[-1].to_pydatetime(), "api_token": token}
    if policy is not None:
        kwargs["rate_limits"] = policy
    if time_fn is not None:
        kwargs["time_fn"] = time_fn
    if trusted_proxy_header is not None:
        kwargs["trusted_proxy_header"] = trusted_proxy_header
    return create_app(svc, **kwargs), svc


def _app(tmp_path, **kwargs):
    app, svc = _build_app(tmp_path, **kwargs)
    return TestClient(app), svc


def _events(caplog, name: str) -> list[dict]:
    return [r.security for r in caplog.records
            if getattr(r, "security", None) and r.security.get("event") == name]


# ── 1. security telemetry ─────────────────────────────────────────────────────────────

def test_failed_auth_emits_a_distinct_security_event(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    assert client.get("/status").status_code == 401

    events = _events(seclog, "auth_failed")
    assert len(events) == 1, "a failed _require_token check must emit exactly one auth_failed event"
    e = events[0]
    assert e["route"] == "/status"
    assert e["method"] == "GET"
    assert e["reason"] == "missing_credentials"
    assert e["client_ip"]
    assert e["request_id"] and e["request_id"] != "-"


def test_wrong_token_is_reported_as_invalid_not_missing(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    assert client.get("/book", headers={"Authorization": "Bearer nope"}).status_code == 401
    events = _events(seclog, "auth_failed")
    assert len(events) == 1 and events[0]["reason"] == "invalid_credentials"


def test_failed_auth_never_logs_the_token(tmp_path, seclog):
    """The whole point of the event is that it is safe to ship to a log sink."""
    client, _svc = _app(tmp_path, token=TOKEN)
    client.get("/status", headers={"Authorization": "Bearer " + TOKEN + "-almost"})
    client.get("/status")
    blob = "\n".join(r.getMessage() for r in seclog.records) + str(
        [getattr(r, "security", None) for r in seclog.records])
    assert TOKEN not in blob
    assert "almost" not in blob


def test_every_request_is_logged_with_id_route_and_outcome(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    r = client.get("/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200

    reqs = _events(seclog, "request")
    assert len(reqs) == 1
    e = reqs[0]
    assert e["route"] == "/status" and e["method"] == "GET"
    assert e["status_code"] == 200 and e["outcome"] == "ok"
    assert e["client_ip"] and e["request_id"]
    assert isinstance(e["duration_ms"], float)


def test_request_id_is_returned_so_a_client_report_can_be_correlated(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    r = client.get("/health")
    rid = r.headers.get("X-Request-ID")
    assert rid
    assert any(e["request_id"] == rid for e in _events(seclog, "request"))


def test_denied_requests_are_logged_at_warning(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    client.get("/status")
    denied = [r for r in seclog.records if getattr(r, "security", None)
              and r.security.get("event") == "request" and r.security.get("status_code") == 401]
    assert denied and denied[0].levelno == logging.WARNING
    assert denied[0].security["outcome"] == "denied"


# ── 2. rate limiting ──────────────────────────────────────────────────────────────────

def test_read_endpoints_are_rate_limited(tmp_path, seclog):
    policy = RateLimitPolicy(read_per_minute=3, write_per_minute=99)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    for _ in range(3):
        assert client.get("/status", headers=auth).status_code == 200
    r = client.get("/status", headers=auth)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert _events(seclog, "rate_limited")


def test_control_endpoints_have_their_own_tighter_budget(tmp_path):
    """A repeated kill-switch reset is a financial-safety event, so the mutating
    routes share a budget far tighter than the read budget - and exhausting it
    must not lock the operator out of the read surface they need to diagnose."""
    policy = RateLimitPolicy(read_per_minute=50, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    params = {"operator": "alice", "reason": "all clear"}
    assert client.post("/kill-switch/reset", params=params, headers=auth).status_code == 200
    assert client.post("/kill-switch/reset", params=params, headers=auth).status_code == 200
    assert client.post("/kill-switch/reset", params=params, headers=auth).status_code == 429
    # the other control routes share the same exhausted budget ...
    assert client.post("/cycle/run", headers=auth).status_code == 429
    assert client.post("/reconciliation/resolve",
                       params={"item_id": "x", "operator": "alice"},
                       headers=auth).status_code == 429
    # ... but observation still works.
    assert client.get("/status", headers=auth).status_code == 200


def test_rate_limiting_runs_before_auth_so_token_guessing_is_capped(tmp_path):
    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    assert client.get("/status", headers={"Authorization": "Bearer a"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer b"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer c"}).status_code == 429


def test_health_probe_is_never_rate_limited(tmp_path):
    policy = RateLimitPolicy(read_per_minute=1, write_per_minute=1)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_zero_disables_the_limiter(tmp_path):
    policy = RateLimitPolicy(read_per_minute=0, write_per_minute=0)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    for _ in range(30):
        assert client.get("/status", headers=auth).status_code == 200


def test_budget_refills_once_the_window_has_passed(tmp_path):
    now = {"t": 1_000.0}
    policy = RateLimitPolicy(read_per_minute=1, write_per_minute=1, window_seconds=60.0)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy, time_fn=lambda: now["t"])
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/status", headers=auth).status_code == 200
    assert client.get("/status", headers=auth).status_code == 429
    now["t"] += 61.0
    assert client.get("/status", headers=auth).status_code == 200


# ── 3. the limiter itself ─────────────────────────────────────────────────────────────

def test_limiter_is_per_key():
    now = {"t": 0.0}
    limiter = RateLimiter(2, window_seconds=10.0, time_fn=lambda: now["t"])
    assert limiter.check("read:1.1.1.1").allowed
    assert limiter.check("read:1.1.1.1").allowed
    assert not limiter.check("read:1.1.1.1").allowed
    assert limiter.check("read:2.2.2.2").allowed, "one client must not exhaust another's budget"


def test_limiter_reports_retry_after_and_slides():
    now = {"t": 100.0}
    limiter = RateLimiter(1, window_seconds=10.0, time_fn=lambda: now["t"])
    assert limiter.check("k").allowed
    denied = limiter.check("k")
    assert not denied.allowed and denied.retry_after == pytest.approx(10.0)
    now["t"] += 5.0
    assert limiter.check("k").retry_after == pytest.approx(5.0)
    now["t"] += 5.1
    assert limiter.check("k").allowed


def test_limiter_memory_is_bounded_when_every_key_is_still_active():
    """The DoS case the previous test missed. Re-verified in round 3.

    The old ``test_limiter_prunes_stale_keys`` used ``window_seconds=1.0`` and
    advanced the clock 0.01s per key, so only ~100 keys were ever *inside* the
    window; the stale-key prune could always free the map and the
    ``<= _MAX_KEYS + 1`` assertion was satisfied without the cap ever being
    exercised (measured peak: exactly 4096 against a 4097 bound).

    Here the clock never moves, so **every** key is active and unprunable. That is
    the attacker's shape — one request each from many sources inside one window.
    Measured against the old prune-based implementation: 60,000 keys retained
    against ``_MAX_KEYS=4096``, and 115.9s of CPU for those 60,000 calls because
    the O(n) prune ran under the global lock on every request past the cap.
    """
    limiter = RateLimiter(240, window_seconds=3600.0, time_fn=lambda: 1_000.0)
    keys = [f"read:10.0.{(i // 256) % 256}.{i % 256}" for i in range(RateLimiter._MAX_KEYS * 3)]
    # Assert the test's own PREMISE before the property. A generator that quietly
    # produced duplicates would leave this asserting a cap it never approached -
    # exactly the failure mode of the prune test this one replaced.
    assert len(set(keys)) == RateLimiter._MAX_KEYS * 3 > RateLimiter._MAX_KEYS
    for key in keys:
        limiter.check(key)
    assert limiter.tracked_keys <= RateLimiter._MAX_KEYS, (
        f"limiter retained {limiter.tracked_keys} keys against a cap of "
        f"{RateLimiter._MAX_KEYS}: memory is attacker-controlled"
    )
    # ... and it is genuinely FULL, not empty because everything was pruned.
    assert limiter.tracked_keys == RateLimiter._MAX_KEYS


def test_eviction_drops_the_coldest_key_not_the_active_caller():
    """A bounded map must evict *something*; it must not be the caller who is
    currently being throttled, or the cap itself becomes the bypass."""
    limiter = RateLimiter(1, window_seconds=3600.0, time_fn=lambda: 1_000.0)
    for i in range(RateLimiter._MAX_KEYS - 1):
        limiter.check(f"cold{i}")
    assert limiter.check("hot").allowed          # 'hot' is now the most recent key
    assert not limiter.check("hot").allowed      # ... and is at its limit
    for i in range(50):                          # push the map past the cap
        limiter.check(f"later{i}")
    assert limiter.tracked_keys <= RateLimiter._MAX_KEYS
    assert not limiter.check("hot").allowed, "an active throttle was evicted by unrelated keys"


def test_limiter_stays_fast_under_key_pressure():
    """Regression guard on the pathological prune (an O(n) scan under the global
    lock on every request past the cap). 20,000 active keys measured at 8.2s on
    the prune implementation; the 3s budget sits well below that and far above
    the bounded implementation's fixed cost."""
    limiter = RateLimiter(240, window_seconds=3600.0, time_fn=lambda: 1_000.0)
    started = time.perf_counter()
    for i in range(20_000):
        limiter.check(f"read:{i}")
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"20k limiter checks took {elapsed:.2f}s - the eviction path is not O(1)"


def test_a_drained_key_does_not_keep_its_timestamps():
    """Bounded-by-eviction must not mean 'never forgets': a key whose window has
    fully passed starts from a clean budget."""
    now = {"t": 0.0}
    limiter = RateLimiter(2, window_seconds=10.0, time_fn=lambda: now["t"])
    assert limiter.check("k").allowed
    assert limiter.check("k").allowed
    assert not limiter.check("k").allowed
    now["t"] += 11.0
    decision = limiter.check("k")
    assert decision.allowed and decision.remaining == 1


# ── 4. admin endpoints: every privileged route is server-side authenticated ───────────

MUTATING = {"/cycle/run", "/kill-switch/reset", "/reconciliation/resolve"}


def _route_dependency_names(route) -> set[str]:
    return {d.call.__name__ for d in route.dependant.dependencies if d.call is not None}


def test_the_route_check_actually_detects_an_unguarded_route():
    """Negative control. Without this, the assertion below could pass because the
    introspection is blind rather than because the routes are guarded."""
    from fastapi import FastAPI

    unguarded = FastAPI()

    @unguarded.post("/danger")
    def danger() -> dict:
        return {}

    route = next(r for r in unguarded.routes if getattr(r, "path", None) == "/danger")
    assert "_require_token" not in _route_dependency_names(route)


def test_every_mutating_route_carries_the_auth_dependency(tmp_path):
    """Finding 3 was a confirmation, not a defect: auth IS server-side on all three
    privileged routes. This pins it, so a fourth control route cannot be added
    without one."""
    _client, svc = _app(tmp_path, token=TOKEN)
    app = create_app(svc, api_token=TOKEN)
    mutating = [r for r in app.routes
                if getattr(r, "methods", None) and {"POST", "PUT", "PATCH", "DELETE"} & r.methods]
    assert {r.path for r in mutating} == MUTATING
    for route in mutating:
        names = _route_dependency_names(route)
        assert "_require_token" in names, f"{route.path} is reachable without authentication"
        assert any(n.startswith("_rate_limit") for n in names), f"{route.path} is unlimited"


# ── 5. whose budget is it? (SEC-4) ────────────────────────────────────────────────────

def test_an_unauthenticated_flood_cannot_exhaust_the_operators_read_budget(tmp_path):
    """In every recommended topology - loopback bind, or a compose port map - each
    caller presents the SAME client IP, so an IP-keyed budget is one global budget
    and any unauthenticated third party can lock the operator out of the surface
    they need to diagnose with. Measured before the fix: two anonymous 401s
    exhausted read_per_minute=2 and the token holder got a 429."""
    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    for _ in range(6):
        assert client.get("/status").status_code in (401, 429)
    r = client.get("/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200, "an anonymous flood denied the authenticated operator"


def test_the_authenticated_budget_is_still_a_budget(tmp_path):
    """Keying on identity must not hand the token holder an unlimited surface."""
    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/status", headers=auth).status_code == 200
    assert client.get("/status", headers=auth).status_code == 200
    assert client.get("/status", headers=auth).status_code == 429


def test_a_rotating_forwarded_header_cannot_mint_a_fresh_rate_limit_budget(tmp_path):
    """Replaces a test that proved nothing.

    The old ``test_a_forwarded_header_is_ignored_unless_it_is_explicitly_trusted``
    sent ``X-Forwarded-For`` to a bare ``TestClient``, where nothing reads that
    header at all when ``trusted_proxy_header`` is unset — so it passed whatever
    ``_client_ip`` did with the header, and it asserted a property the SHIPPED
    server did not have: ``uvicorn.run()`` defaults ``proxy_headers=True`` with
    ``forwarded_allow_ips="127.0.0.1"``, so on the loopback bind uvicorn's own
    middleware rewrote the client address from that header first.

    This version puts the REAL ``ProxyHeadersMiddleware`` in front of the app, which
    is the situation the old test only pretended to cover, and asserts the budget
    still belongs to one caller. (The server-level fix — turning that middleware off
    in both entry points — is proved against a real socket in
    ``tests/test_api_proxy_headers.py``; this is the application's own last line.)
    """
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    app, _svc = _build_app(tmp_path, token=TOKEN, policy=policy)
    client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts=["testclient"]))
    for i in range(2):
        assert client.get("/status", headers={"X-Forwarded-For": f"9.9.9.{i}"}).status_code == 401
    r = client.get("/status", headers={"X-Forwarded-For": "9.9.9.77"})
    assert r.status_code == 429, "spoofing X-Forwarded-For bought a fresh budget"


def test_an_explicitly_trusted_proxy_header_separates_real_clients(tmp_path):
    """Opt-in only: when the operator states that a proxy in front of the API sets
    the header, distinct forwarded clients get distinct budgets."""
    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy,
                        trusted_proxy_header="X-Forwarded-For")
    for _ in range(2):
        client.get("/status", headers={"X-Forwarded-For": "203.0.113.5"})
    assert client.get("/status", headers={"X-Forwarded-For": "203.0.113.5"}).status_code == 429
    assert client.get("/status", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 401


def test_the_security_log_never_carries_the_raw_token_as_an_identity(tmp_path, seclog):
    """Identity keying must not put the credential in the trail.

    The previous version of this test was VACUOUS and was proved so by mutation:
    replacing ``token_identity`` with the raw token left it passing. The reason is
    that ``identity`` is only ever a FIELD on the ``rate_limited`` event, and the
    old test made one successful request — so the mutated value was never logged
    and the "TOKEN not in blob" assertion had nothing to catch.

    So the test now drives the limiter over its budget with the valid credential,
    which is the only path that writes an identity, and pins the SHAPE of what it
    writes. ``TOKEN not in identity`` alone would still be weak — a truncated or
    reordered credential would slip through — so the fingerprint is checked against
    the value it is supposed to be.
    """
    policy = RateLimitPolicy(read_per_minute=1, write_per_minute=1)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/status", headers=auth).status_code == 200
    assert client.get("/status", headers=auth).status_code == 429

    throttled = _events(seclog, "rate_limited")
    assert throttled, "no rate_limited event, so no identity was ever written to the trail"
    identity = throttled[0]["identity"]
    assert TOKEN not in identity, f"the credential itself is the rate-limiter identity: {identity!r}"
    assert re.fullmatch(r"token:[0-9a-f]{12}", identity), (
        f"the identity is not the expected sha256 fingerprint: {identity!r}"
    )
    import hashlib

    assert identity == "token:" + hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()[:12]

    blob = "\n".join(r.getMessage() for r in seclog.records) + str(
        [getattr(r, "security", None) for r in seclog.records])
    assert TOKEN not in blob
    # Any prefix of the credential is a gift to an attacker with a wordlist.
    assert TOKEN[:8] not in blob


# ── 6. the generated docs are part of the admin surface (SEC-5) ───────────────────────

DOC_ROUTES = ("/openapi.json", "/docs", "/redoc")


def test_the_schema_and_docs_require_the_same_auth_as_the_rest(tmp_path):
    """/openapi.json enumerates every route AND every parameter name - including
    the kill-switch operator/reason fields. Measured before the fix: 200, six
    times, with the read budget already exhausted."""
    client, _svc = _app(tmp_path, token=TOKEN)
    for path in DOC_ROUTES:
        assert client.get(path).status_code == 401, f"{path} enumerates the admin surface anonymously"


def test_the_schema_is_served_to_an_authenticated_operator(tmp_path):
    """Gating is not deletion - the operator still gets their API reference."""
    client, _svc = _app(tmp_path, token=TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    spec = client.get("/openapi.json", headers=auth)
    assert spec.status_code == 200 and "/kill-switch/reset" in spec.json()["paths"]
    for path in ("/docs", "/redoc"):
        assert client.get(path, headers=auth).status_code == 200


def test_the_docs_share_the_read_budget(tmp_path):
    policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
    client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/openapi.json", headers=auth).status_code == 200
    assert client.get("/openapi.json", headers=auth).status_code == 200
    assert client.get("/docs", headers=auth).status_code == 429


# ── 7. detection: the trail has to outlive the console (SEC-6) ────────────────────────

def test_security_events_reach_a_durable_greppable_file(tmp_path):
    """Standards 4.10: an event that only ever reached a console is not detection.
    Under the uvicorn entrypoint the security logger has no handler at all and the
    root logger's lastResort is WARNING, so INFO-level request events were dropped
    outright and the WARNING ones died with the process."""
    path = tmp_path / "security.jsonl"
    handler = attach_security_log_file(path)
    try:
        client, _svc = _app(tmp_path, token=TOKEN)
        assert client.get("/status").status_code == 401
        handler.flush()
        lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    finally:
        detach_security_log_file(handler)

    events = [ln for ln in lines if ln.get("event") == "auth_failed"]
    assert events, f"no auth_failed line in the durable trail: {lines}"
    assert events[0]["route"] == "/status" and events[0]["reason"] == "missing_credentials"
    assert events[0]["level"] == "WARNING" and events[0]["ts"]
    assert any(ln.get("event") == "request" for ln in lines), "INFO request events are still dropped"
    assert TOKEN not in path.read_text(encoding="utf-8")


def test_the_durable_trail_is_size_capped(tmp_path):
    """A log an attacker can grow without bound is the next denial of service."""
    path = tmp_path / "security.jsonl"
    handler = attach_security_log_file(path)
    try:
        assert handler.maxBytes > 0 and handler.backupCount > 0
    finally:
        detach_security_log_file(handler)


def test_attaching_the_file_twice_does_not_duplicate_lines(tmp_path):
    path = tmp_path / "security.jsonl"
    first = attach_security_log_file(path)
    second = attach_security_log_file(path)
    try:
        assert first is second
        logging.getLogger(SECURITY_LOGGER_NAME).warning("probe", extra={"security": {"event": "probe"}})
        first.flush()
        probes = [ln for ln in path.read_text(encoding="utf-8").splitlines() if "probe" in ln]
        assert len(probes) == 1
    finally:
        detach_security_log_file(first)


def test_settings_wire_both_trails_to_the_state_dir_by_default(tmp_path):
    """The finding was that the events reached a console only, so the DEFAULT has
    to be durable - an opt-in trail nobody opted into is the same defect. Since
    SEC-9 that means BOTH files by default, not just the size-capped one."""
    from ops.api import install_security_logging

    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                   persistence={"state_dir": str(tmp_path)})
    handles = install_security_logging(settings)
    try:
        assert handles.trail is not None and handles.alerts is not None
        assert handles.trail.baseFilename == str((tmp_path / "security.jsonl").absolute())
        assert handles.alerts.baseFilename == str(
            (tmp_path / "security-alerts.jsonl").absolute())
    finally:
        handles.detach()


def test_the_durable_trail_can_be_turned_off_deliberately(tmp_path):
    from ops.api import install_security_logging

    settings = EngineSettings(mode="RESEARCH", universe=SYMBOLS,
                                   persistence={"state_dir": str(tmp_path)},
                                   api_security_log_enabled=False)
    handles = install_security_logging(settings)
    assert not handles.enabled and handles.trail is None and handles.alerts is None


def test_an_unwritable_path_never_takes_the_api_down(tmp_path):
    """Fail soft: losing the security trail is bad, refusing to serve is worse."""
    blocked = tmp_path / "security.jsonl"
    blocked.mkdir()  # a directory where the log file should be - open() cannot win
    assert attach_security_log_file(blocked) is None
    client, _svc = _app(tmp_path, token=TOKEN)
    assert client.get("/health").status_code == 200


# ── 8. every attacker-controlled field is capped before it is logged (SEC-10) ──────────

def test_cap_field_marks_what_it_truncated():
    assert cap_field("1.2.3.4") == "1.2.3.4"
    capped = cap_field("A" * 500)
    assert len(capped) == FIELD_MAX_CHARS
    assert capped.endswith("~"), "a silently cut value reads like a real one"


def test_an_oversized_forwarded_value_is_capped_before_it_is_logged(tmp_path, seclog):
    """``ops/api.py`` capped the TRUSTED-header path at 64 chars and left every other
    path uncapped. Measured against a real pre-fix server: a 4,000-character
    ``X-Forwarded-For`` put 4,000 attacker-chosen characters into the trail, once
    per request."""
    client, _svc = _app(tmp_path, token=TOKEN, trusted_proxy_header="X-Forwarded-For")
    assert client.get("/status", headers={"X-Forwarded-For": "B" * 4000}).status_code == 401
    event = _events(seclog, "auth_failed")[0]
    assert len(event["client_ip"]) <= FIELD_MAX_CHARS, (
        f"a {len(event['client_ip'])}-character address reached the trail"
    )
    assert event["client_ip"].endswith("~")


def test_an_oversized_peer_address_is_capped_before_it_is_logged(tmp_path, seclog):
    """The peer address is normally a real socket address, but uvicorn's
    ProxyHeadersMiddleware will put an arbitrary header value there. Capped at the
    same boundary as everything else, and kept as a claim rather than a fact."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    app, _svc = _build_app(tmp_path, token=TOKEN)
    client = TestClient(ProxyHeadersMiddleware(app, trusted_hosts=["testclient"]))
    assert client.get("/status", headers={"X-Forwarded-For": "C" * 4000}).status_code == 401
    event = _events(seclog, "auth_failed")[0]
    assert event["client_ip"] == UNTRUSTED_FORWARDED_IP
    assert len(event["forwarded_claim"]) <= FIELD_MAX_CHARS


def test_an_oversized_method_is_capped_before_it_is_logged(tmp_path, seclog):
    """h11 treats the method as a token of unbounded length, so it is
    attacker-controlled text on its way to a log like everything else."""
    client, _svc = _app(tmp_path, token=TOKEN)
    client.request("X" * 500, "/status")
    events = _events(seclog, "request")
    assert events and len(events[0]["method"]) <= 16


def test_an_unmatched_route_is_capped_before_it_is_logged(tmp_path, seclog):
    client, _svc = _app(tmp_path, token=TOKEN)
    client.get("/" + "z" * 4000)
    events = _events(seclog, "request")
    assert events and len(events[0]["route"]) <= 200


# ── 9. the alert trail survives the attack that generates it (SEC-9) ──────────────────

def _alert_lines(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_the_alert_trail_is_a_separate_file_from_the_request_trail(tmp_path):
    """The finding: ONE RotatingFileHandler carried both the INFO request lines and
    the WARNING auth_failed lines, capped at 5 MB x 5 (~30 MB). At ~580 bytes a
    request that is ~52,000 requests to roll the whole trail away - so an attacker
    erased the evidence of their own attack with the very traffic that made it, and
    the noisy INFO lines they generate did the erasing."""
    trail_path = tmp_path / "security.jsonl"
    alert_path = tmp_path / "security-alerts.jsonl"
    trail = attach_security_log_file(trail_path)
    alerts = attach_security_alert_file(alert_path)
    try:
        client, _svc = _app(tmp_path, token=TOKEN)
        auth = {"Authorization": f"Bearer {TOKEN}"}
        for _ in range(5):
            assert client.get("/status", headers=auth).status_code == 200
        assert client.get("/status").status_code == 401
        trail.flush()
        alerts.flush()
        trail_events = _alert_lines(trail_path)
        alert_events = _alert_lines(alert_path)
    finally:
        detach_security_log_file(trail)
        detach_security_log_file(alerts)

    assert [e for e in trail_events if e["event"] == "request" and e["level"] == "INFO"], (
        "the full trail lost the ordinary request lines"
    )
    assert [e for e in alert_events if e["event"] == "auth_failed"], (
        "the alert trail did not receive the auth_failed event"
    )
    assert not [e for e in alert_events if e["level"] == "INFO"], (
        "INFO request lines are in the alert trail, which is what made it floodable"
    )


def test_the_alert_trail_retention_is_time_based_so_volume_cannot_shorten_it(tmp_path):
    """A byte cap lets the ATTACKER choose when the oldest evidence disappears; a
    time cap does not, provided the write rate is bounded (which the aggregator
    below is what guarantees)."""
    path = tmp_path / "security-alerts.jsonl"
    handler = attach_security_alert_file(path)
    try:
        assert isinstance(handler, logging.handlers.TimedRotatingFileHandler)
        assert getattr(handler, "maxBytes", 0) == 0, "a size cap is an attacker-reachable cap"
        assert handler.backupCount == SECURITY_ALERT_RETENTION_DAYS == 30
        assert handler.when == "D"
        assert handler.level == logging.WARNING
    finally:
        detach_security_log_file(handler)


def test_a_flood_cannot_roll_the_alert_trail_over(tmp_path):
    """500 rejected requests must not be 500 alert lines. Aggregation is what keeps
    the write rate bounded, and a bounded write rate is what makes the time-based
    retention hold under attack."""
    now = {"t": 1_000.0}
    path = tmp_path / "security-alerts.jsonl"
    handler = attach_security_alert_file(path, time_fn=lambda: now["t"])
    try:
        policy = RateLimitPolicy(read_per_minute=0, write_per_minute=0)  # limiter off
        client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
        for _ in range(500):
            assert client.get("/status").status_code == 401
        handler.flush()
        during = _alert_lines(path)
        now["t"] += 61.0                       # the aggregation window rolls
        assert client.get("/status").status_code == 401
        handler.flush()
        after = _alert_lines(path)
    finally:
        detach_security_log_file(handler)

    failures = [e for e in during if e["event"] == "auth_failed"]
    assert len(failures) == 1, (
        f"500 requests wrote {len(failures)} auth_failed lines; a flood can still roll "
        "the alert trail over"
    )
    assert failures[0]["reason"] == "missing_credentials"

    carried = [e for e in after if e["event"] == "auth_failed"][-1]
    assert carried["repeat_suppressed"] == 499, (
        f"the suppressed count was lost, not carried: {carried}"
    )
    assert carried["repeat_clients"], "no sample of who was doing it"
    assert path.stat().st_size < 4096, (
        f"the alert trail grew to {path.stat().st_size} bytes on 501 requests"
    )


def test_aggregation_never_hides_a_DIFFERENT_kind_of_event(tmp_path):
    """Collapsing repeats must not collapse distinct signals: an invalid credential
    is a different event from a missing one, and a throttle is different again."""
    now = {"t": 1_000.0}
    path = tmp_path / "security-alerts.jsonl"
    handler = attach_security_alert_file(path, time_fn=lambda: now["t"])
    try:
        policy = RateLimitPolicy(read_per_minute=2, write_per_minute=2)
        client, _svc = _app(tmp_path, token=TOKEN, policy=policy)
        client.get("/status")                                        # missing_credentials
        client.get("/status", headers={"Authorization": "Bearer x"})  # invalid_credentials
        client.get("/status")                                        # 429: rate_limited
        handler.flush()
        events = _alert_lines(path)
    finally:
        detach_security_log_file(handler)

    reasons = {(e["event"], e.get("reason")) for e in events}
    assert ("auth_failed", "missing_credentials") in reasons
    assert ("auth_failed", "invalid_credentials") in reasons
    assert ("rate_limited", None) in reasons


def test_the_alert_trail_fails_soft_like_the_full_one(tmp_path):
    """Losing telemetry is bad; refusing to serve the control API is worse."""
    blocked = tmp_path / "security-alerts.jsonl"
    blocked.mkdir()
    assert attach_security_alert_file(blocked) is None
    client, _svc = _app(tmp_path, token=TOKEN)
    assert client.get("/health").status_code == 200


def test_attaching_the_alert_file_twice_does_not_duplicate_lines(tmp_path):
    path = tmp_path / "security-alerts.jsonl"
    first = attach_security_alert_file(path)
    second = attach_security_alert_file(path)
    try:
        assert first is second
    finally:
        detach_security_log_file(first)


def test_a_non_ascii_credential_is_a_401_not_a_500(tmp_path, seclog):
    """secrets.compare_digest raises TypeError on non-ASCII strings, and Starlette
    decodes headers as latin-1 - so a single high byte in the Authorization header
    turned an unauthenticated request into an unhandled 500 (a stack trace path
    reachable by anyone, standards 4.8). It is a rejected credential: 401."""
    client, _svc = _app(tmp_path, token=TOKEN)
    # Sent as raw bytes: a real client can put any byte on the wire, and
    # Starlette decodes header values as latin-1.
    r = client.get("/status", headers={b"Authorization": bytes([0xFF, 0xFE])})
    assert r.status_code == 401
    assert _events(seclog, "auth_failed")[0]["reason"] == "invalid_credentials"
    assert not _events(seclog, "request_failed")
