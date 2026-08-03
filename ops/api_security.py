"""
TradingEngineResearch — Security telemetry + rate limiting for the ops API
===============================================================
The *financial* audit trail is already strong: ``ops/ledger.py`` is an
append-only SHA-256 hash-chained ledger with per-event fsync and ``verify_chain``
tamper detection. What it does not record is anything about the **operator
surface** — who called the control API, whether their credentials failed, or
whether someone is hammering a control endpoint. This module supplies that
second trail.

Two small, dependency-free pieces (no FastAPI import here, so the quant core
stays importable without the ``app`` extra):

  • :class:`SecurityLog` — one structured event per API request, plus a distinct
    ``auth_failed`` event for every rejected credential check and a
    ``rate_limited`` event for every throttled call. Each record carries the
    payload as a ``security`` attribute (for a JSON log handler) *and* renders it
    into the message (so a plain text log stays greppable). **The token is never
    a field**: neither the configured one nor the one the caller supplied.

  • :class:`RateLimiter` — an in-process sliding-window limiter keyed by
    ``bucket:caller-identity``. Deliberately not a distributed limiter: the platform
    is a single process behind a loopback bind (see ``ops.api.assert_bind_is_safe``),
    and a per-process cap on ``POST /kill-switch/reset`` is a financial-safety
    control, not a quota-billing feature.

  • :func:`attach_security_log_file` — a durable, size-capped JSON-lines sink for
    the events above. Standards §4.10: an event that only ever reached a console
    is not detection. Under ``uvicorn --factory ops.api:create_app_from_settings``
    the security logger has no handler at all and the root logger falls back to
    ``logging.lastResort`` (WARNING), so every INFO-level ``request`` event was
    dropped outright and the WARNING ones died with the process.

  • :func:`attach_security_alert_file` — the SECOND trail, and the one that has to
    survive. **The detection could previously be erased by the attacker**: every
    event, INFO ``request`` included, shared one ``RotatingFileHandler`` capped at
    5 MB x 5. A request costs ~580 bytes, so ~52,000 requests rotated the whole
    ~30 MB away — including the ``auth_failed`` lines proving the attack that did
    it. The high-value events (WARNING and above: ``auth_failed``,
    ``rate_limited``, ``request_failed``, and any denied/5xx request) therefore now
    go to their OWN file with its OWN retention, and repeated lines are collapsed
    by :class:`_RepeatAggregator` so no volume of requests can roll it over.

Both fail *open* for the request they cannot serve rather than crashing a cycle —
an observability or throttling failure must never take the platform down — but the
limiter itself fails *closed* on its own budget: over the limit is a 429, always.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Optional, Union

__all__ = [
    "SECURITY_ALERT_LEVEL",
    "SECURITY_ALERT_RETENTION_DAYS",
    "SECURITY_ALERT_WINDOW_SECONDS",
    "SECURITY_LOGGER_NAME",
    "RateLimitDecision",
    "RateLimitPolicy",
    "RateLimiter",
    "SecurityLog",
    "SecurityLogHandles",
    "attach_security_alert_file",
    "attach_security_log_file",
    "cap_field",
    "detach_security_log_file",
]

SECURITY_LOGGER_NAME = "tradingengineresearch.api.security"

# Size cap for the high-volume trail (every event, INFO included). An attacker who
# can make us log can make us write; ~5 MB x 5 generations bounds that at ~30 MB,
# which is a bounded cost rather than a full disk on a machine that also has to
# keep trading. THIS FILE IS ALLOWED TO ROLL — that is what the alert trail below
# exists to survive.
SECURITY_LOG_MAX_BYTES = 5 * 1024 * 1024
SECURITY_LOG_BACKUPS = 5

# The alert trail: WARNING and above only, rotated by TIME rather than by size, so
# that no volume of requests can shorten its retention. 30 days is chosen because
# this is an unattended single-operator platform — the trail has to still be there
# when somebody next looks, and a month comfortably covers "I was away".
#
# Time-based retention is only safe if the write RATE is bounded, otherwise a flood
# just fills the current day's file instead. _RepeatAggregator bounds it: at most
# one line per (event, reason) per SECURITY_ALERT_WINDOW_SECONDS. With the ~6
# distinct WARNING+ shapes this API can produce that is <=6 lines/min.
#
# MEASURED (2026-08-03, 200 anonymous requests against the real app): the full trail
# took 590 bytes per request -- 53,327 requests to roll its whole 30 MB away, which
# is the finding -- while the SAME 200 requests produced 2 alert lines totalling 638
# bytes, the widest 330 bytes. So the alert trail's worst case is 6 x 60 x 24 x 330
# = ~2.9 MB/day and ~85 MB across the 30 days, against a few kilobytes in normal
# operation. Nothing an attacker sends changes those bounds.
SECURITY_ALERT_LEVEL = logging.WARNING
SECURITY_ALERT_RETENTION_DAYS = 30
SECURITY_ALERT_WINDOW_SECONDS = 60.0

# Bounds on the aggregator's own memory. The key is (event, reason), both of which
# are chosen by THIS codebase and not by the caller, so the map is small by
# construction; the cap is belt-and-braces in case a future event carries a
# caller-influenced reason.
_ALERT_MAX_KEYS = 64
_ALERT_MAX_SAMPLES = 5

# Every attacker-controlled string that reaches a log line is truncated to this
# many characters. See ops.api._who: uvicorn's ProxyHeadersMiddleware will put an
# arbitrary-length X-Forwarded-For value into ``scope["client"]``, and a 4,000-byte
# "address" in the trail is both a storage attack and a way to push real lines out.
FIELD_MAX_CHARS = 64


def cap_field(value: Any, limit: int = FIELD_MAX_CHARS) -> str:
    """Truncate an attacker-controlled value for logging, marking the truncation.

    The trailing ``~`` matters: a silently cut string reads like a real value, and
    an operator triaging a trail needs to know they are looking at a fragment.
    """
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "~"


class SecurityLog:
    """Emits structured security events for the ops API.

    ``emit`` attaches the event dict to the record as ``security`` so a JSON
    handler can ship it verbatim, and also formats it into the message so a plain
    text log remains readable and greppable.
    """

    def __init__(self, logger_name: str = SECURITY_LOGGER_NAME) -> None:
        self._log = logging.getLogger(logger_name)

    def emit(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        payload: dict[str, Any] = {"event": event, **fields}
        self._log.log(
            level,
            "API %s | %s",
            event,
            json.dumps(payload, default=str, sort_keys=True),
            extra={"security": payload},
        )


class _SecurityJsonFormatter(logging.Formatter):
    """One JSON object per line: the event payload plus when and how loud.

    Reads the ``security`` attribute the :class:`SecurityLog` attaches, so the
    durable line carries the same fields as the console one — and no others, which
    is what keeps the credential out of it.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "security", None)
        line: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {
            "event": "log", "message": record.getMessage()
        }
        line["ts"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        line["level"] = record.levelname
        line["logger"] = record.name
        return json.dumps(line, default=str, sort_keys=True)


def attach_security_log_file(
    path: Union[str, Path],
    *,
    level: int = logging.INFO,
    logger_name: str = SECURITY_LOGGER_NAME,
) -> Optional[logging.handlers.RotatingFileHandler]:
    """Route the security events to a durable, size-capped JSON-lines file.

    Returns the handler, or ``None`` if the file could not be opened — losing the
    security trail is bad, refusing to serve the API is worse, so this fails soft
    and says so on the module logger. Idempotent: calling it again with the same
    path returns the handler already installed rather than duplicating every line.

    This is *durable logging*, not alerting. Nobody is paged by it; it makes an
    ``auth_failed`` burst greppable after the fact. Detection is therefore partial
    and is recorded that way in ``docs/project-control/RISK_AND_DEFECT_REGISTER.md`` (SEC-6).
    """
    target = Path(path).expanduser()
    log = logging.getLogger(logger_name)
    resolved = str(target.absolute())
    for existing in log.handlers:
        if getattr(existing, "_engine_security_path", None) == resolved:
            assert isinstance(existing, logging.handlers.RotatingFileHandler)
            return existing
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            resolved, maxBytes=SECURITY_LOG_MAX_BYTES,
            backupCount=SECURITY_LOG_BACKUPS, encoding="utf-8", delay=False,
        )
    except OSError:
        logging.getLogger(__name__).warning(
            "could not open the security log at %s; security events stay on the "
            "console only and after-the-fact detection is unavailable", resolved,
            exc_info=True,
        )
        return None
    handler.setLevel(level)
    handler.setFormatter(_SecurityJsonFormatter())
    handler._engine_security_path = resolved  # type: ignore[attr-defined]
    log.addHandler(handler)
    if log.level == logging.NOTSET or log.level > level:
        log.setLevel(level)
    return handler


@dataclass
class _RepeatState:
    """One aggregation window for one (event, reason) key."""

    window_start: float
    suppressed: int = 0
    samples: set = dataclass_field(default_factory=set)


class _RepeatAggregator(logging.Filter):
    """Collapse repeated alert lines so a flood cannot roll the alert trail over.

    Keyed on ``(event, reason)`` — deliberately COARSE. Keying on the route or the
    client address would hand the attacker the key space: 52,000 distinct paths
    would be 52,000 "first occurrences" and the aggregation would buy nothing.

    The first record for a key in each window is passed through and carries
    ``repeat_suppressed`` (how many of its kind were dropped since the previous
    passed record) plus up to five distinct ``repeat_clients`` seen while
    suppressed. Everything else for that key inside the window is dropped **from
    this handler only** — the full ``security.jsonl`` trail still has every line,
    for as long as it has not rolled.

    Known limit, stated rather than hidden: the count for the final window of a
    burst is attached to the NEXT passed record, so if the burst stops and never
    resumes, up to one window's worth of counts is only in the full trail. That is
    the price of not writing a second record from inside a filter, and one window
    is 60 seconds.
    """

    def __init__(
        self,
        window_seconds: float = SECURITY_ALERT_WINDOW_SECONDS,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._window = float(window_seconds)
        self._time = time_fn
        self._lock = threading.Lock()
        self._state: "OrderedDict[str, _RepeatState]" = OrderedDict()

    @staticmethod
    def _key(payload: dict[str, Any]) -> str:
        return f"{payload.get('event', 'log')}|{payload.get('reason', '')}"

    def filter(self, record: logging.LogRecord) -> bool:
        payload = getattr(record, "security", None)
        if not isinstance(payload, dict):
            return True
        key = self._key(payload)
        now = self._time()
        with self._lock:
            state = self._state.get(key)
            if state is not None:
                self._state.move_to_end(key)
                if now - state.window_start < self._window:
                    state.suppressed += 1
                    if len(state.samples) < _ALERT_MAX_SAMPLES:
                        state.samples.add(cap_field(payload.get("client_ip", "unknown")))
                    return False
            carried = state.suppressed if state is not None else 0
            samples = sorted(state.samples) if state is not None else []
            self._state[key] = _RepeatState(window_start=now)
            while len(self._state) > _ALERT_MAX_KEYS:
                self._state.popitem(last=False)
        summary: dict[str, Any] = {"repeat_suppressed": carried}
        if carried and samples:
            summary["repeat_clients"] = samples
        record.security_repeat = summary  # type: ignore[attr-defined]
        return True


class _SecurityAlertFormatter(_SecurityJsonFormatter):
    """The alert-file line: the event payload plus the aggregator's repeat counts.

    A separate formatter rather than a mutated payload, so the aggregator attached
    to THIS handler can never add fields to the full trail's copy of the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = json.loads(super().format(record))
        repeat = getattr(record, "security_repeat", None)
        if isinstance(repeat, dict):
            line.update(repeat)
        return json.dumps(line, default=str, sort_keys=True)


def attach_security_alert_file(
    path: Union[str, Path],
    *,
    level: int = SECURITY_ALERT_LEVEL,
    retention_days: int = SECURITY_ALERT_RETENTION_DAYS,
    window_seconds: float = SECURITY_ALERT_WINDOW_SECONDS,
    logger_name: str = SECURITY_LOGGER_NAME,
    time_fn: Callable[[], float] = time.monotonic,
) -> Optional[logging.handlers.TimedRotatingFileHandler]:
    """Route the WARNING-and-above security events to their own durable trail.

    Separate from :func:`attach_security_log_file` on purpose. That file is capped
    by SIZE and is dominated by INFO ``request`` lines, so an attacker who can send
    ~52,000 requests rotates the evidence of their own attack out of it. This one
    takes only the events that matter, rotates by TIME (``retention_days`` daily
    generations), and is fed through :class:`_RepeatAggregator`, which bounds the
    write rate to one line per event shape per ``window_seconds``. Volume therefore
    cannot shorten the retention.

    Fails soft and idempotent for the same reasons as the full trail.
    """
    target = Path(path).expanduser()
    log = logging.getLogger(logger_name)
    resolved = str(target.absolute())
    for existing in log.handlers:
        if getattr(existing, "_engine_security_alert_path", None) == resolved:
            assert isinstance(existing, logging.handlers.TimedRotatingFileHandler)
            return existing
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.TimedRotatingFileHandler(
            resolved, when="D", interval=1, backupCount=int(retention_days),
            encoding="utf-8", delay=False, utc=True,
        )
    except OSError:
        logging.getLogger(__name__).warning(
            "could not open the security ALERT log at %s; high-value security "
            "events stay in the size-capped trail only, where a request flood can "
            "rotate them away", resolved, exc_info=True,
        )
        return None
    handler.setLevel(level)
    handler.setFormatter(_SecurityAlertFormatter())
    handler.addFilter(_RepeatAggregator(window_seconds=window_seconds, time_fn=time_fn))
    handler._engine_security_alert_path = resolved  # type: ignore[attr-defined]
    log.addHandler(handler)
    if log.level == logging.NOTSET or log.level > level:
        log.setLevel(level)
    return handler


@dataclass(frozen=True)
class SecurityLogHandles:
    """The two security sinks, so a caller can detach exactly what it attached.

    ``trail`` is every event, size-capped, allowed to roll. ``alerts`` is
    WARNING-and-above with its own time-based retention and repeat aggregation, and
    is the one that has to outlive an attack. Either may be ``None`` — attaching a
    trail fails soft, because losing telemetry must never stop the API serving.
    """

    trail: Optional[logging.Handler] = None
    alerts: Optional[logging.Handler] = None

    @property
    def enabled(self) -> bool:
        return self.trail is not None or self.alerts is not None

    def detach(self, *, logger_name: str = SECURITY_LOGGER_NAME) -> None:
        detach_security_log_file(self.trail, logger_name=logger_name)
        detach_security_log_file(self.alerts, logger_name=logger_name)


def detach_security_log_file(
    handler: Optional[logging.Handler], *, logger_name: str = SECURITY_LOGGER_NAME
) -> None:
    """Remove and close a handler installed by :func:`attach_security_log_file` or
    :func:`attach_security_alert_file`."""
    if handler is None:
        return
    logging.getLogger(logger_name).removeHandler(handler)
    handler.close()


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of one limiter check. ``retry_after`` is seconds until the
    oldest hit leaves the window (0 when the call was allowed)."""

    allowed: bool
    retry_after: float
    remaining: int


@dataclass(frozen=True)
class RateLimitPolicy:
    """Per-client request budgets for the ops API.

    ``read_per_minute`` covers the observation endpoints; the browser dashboard
    polls five of them every 3s (~100/min), so the default leaves headroom for a
    couple of open tabs while still stopping an enumeration flood.
    ``write_per_minute`` is a separate, far tighter budget shared by the three
    control endpoints (``/cycle/run``, ``/kill-switch/reset``,
    ``/reconciliation/resolve``). Either value at 0 disables that budget.
    """

    read_per_minute: int = 240
    write_per_minute: int = 10
    window_seconds: float = 60.0

    @classmethod
    def from_settings(cls, settings: Any) -> "RateLimitPolicy":
        """Read the budgets off a ``EngineSettings`` (or anything with the
        same attribute names), falling back to the defaults above."""
        return cls(
            read_per_minute=int(
                getattr(settings, "api_read_rate_limit_per_minute", cls.read_per_minute)
            ),
            write_per_minute=int(
                getattr(settings, "api_write_rate_limit_per_minute", cls.write_per_minute)
            ),
        )


class RateLimiter:
    """A thread-safe in-process sliding-window limiter keyed by an opaque string.

    Sliding rather than fixed-window because a fixed window lets a caller spend
    two full budgets across a boundary — acceptable for a quota, not for a
    kill-switch reset.

    **Memory is bounded by construction (SEC-4).** The map is an LRU: touching a
    key moves it to the end, and once the map exceeds ``_MAX_KEYS`` the coldest
    key is dropped — both O(1), so the worst case is ``_MAX_KEYS x limit``
    timestamps and a fixed per-request cost no matter how many distinct callers
    appear.

    The previous implementation pruned by scanning for keys whose window had
    fully drained. That is correct only when the attacker cooperates by going
    quiet: 60,000 keys inside one window prune to nothing, so the map grew to
    60,000 entries *and* every request past the cap paid an O(n) scan under the
    global lock — measured at 115.9s of CPU for those 60,000 calls. A rate
    limiter that becomes the denial of service is worse than none.

    Evicting the coldest key does mean a *dormant* caller's throttle can be
    forgotten under key pressure. That is the deliberate trade: a caller who is
    actively being throttled is by definition the most recently seen, so they
    survive eviction (pinned by a test), while the alternative — unbounded
    memory — is exploitable by any unauthenticated client.
    """

    _MAX_KEYS = 4096

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = int(limit)
        self._window = float(window_seconds)
        self._time = time_fn
        self._hits: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def tracked_keys(self) -> int:
        with self._lock:
            return len(self._hits)

    def check(self, key: str) -> RateLimitDecision:
        """Record one hit against ``key`` and say whether it is within budget."""
        if not self.enabled:
            return RateLimitDecision(allowed=True, retry_after=0.0, remaining=-1)
        now = self._time()
        cutoff = now - self._window
        with self._lock:
            hits: Optional[Deque[float]] = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            else:
                # Seen again: this key is now the most recent, so it is the last
                # thing eviction will reach. O(1).
                self._hits.move_to_end(key)
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                return RateLimitDecision(
                    allowed=False,
                    retry_after=max(0.0, hits[0] + self._window - now),
                    remaining=0,
                )
            hits.append(now)
            # Bounded by construction: at most one eviction per admitted request,
            # from the cold end. No scan, no lock held over a walk of the map.
            while len(self._hits) > self._MAX_KEYS:
                self._hits.popitem(last=False)
            return RateLimitDecision(
                allowed=True, retry_after=0.0, remaining=self._limit - len(hits)
            )
