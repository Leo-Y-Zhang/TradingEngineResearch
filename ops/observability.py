"""
TradingEngineResearch — Observability: alert sinks + metrics (ROADMAP Phase 6, item 5)
==========================================================================
The platform *computes* severity-graded alerts every cycle (``ops.monitoring.
alert_list`` → ``CycleResult.alerts`` → the run-loop's ``last_alerts``) but until
now they went nowhere. This module gives them somewhere to go, and adds a small
metrics registry so a running platform is observable beyond the in-memory snapshot.

Two pluggable surfaces, both fail-soft (an observability failure must never break
a trading cycle — golden rule 4 is about *risk/data/execution*; here we degrade
loudly but never raise into the cycle):

  • **Alert sinks** — ``AlertSink.emit(alert)``. ``LoggingAlertSink`` (severity →
    log level), ``JsonlAlertSink`` (durable append-only trail), ``CompositeAlertSink``
    (fan-out; one failing sink never blocks the others), ``NullAlertSink``.
  • **Metrics** — ``MetricsRegistry`` (counters + gauges, optional labels) with a
    JSON ``snapshot()`` and a Prometheus text ``render_prometheus()``.

Built from config via ``core.config.make_alert_sink`` and wired into
``ops.run_loop.EngineService.run_once``; the API exposes ``/metrics``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "AlertSink",
    "NullAlertSink",
    "LoggingAlertSink",
    "JsonlAlertSink",
    "CompositeAlertSink",
    "MetricsRegistry",
    "normalize_alert",
    "alert_severity",
    "SEVERITY_TO_LEVEL",
]

# Alert severities (mirrors ops.monitoring.ALERT_SEVERITIES) → stdlib log levels.
SEVERITY_TO_LEVEL: dict[str, int] = {
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "AMBER": logging.WARNING,
    "RED": logging.ERROR,
}


def normalize_alert(alert: Any) -> dict[str, Any]:
    """Coerce any alert (a monitoring dict, a RiskEvent-like object, or a string)
    to a plain dict carrying at least ``severity``/``category``/``message``."""
    if isinstance(alert, dict):
        d = dict(alert)
    else:
        d = {
            "severity": str(getattr(alert, "severity", "INFO")),
            "category": str(getattr(alert, "category", getattr(alert, "event_type", "alert"))),
            "message": str(getattr(alert, "message", getattr(alert, "description", str(alert)))),
        }
    d.setdefault("severity", "INFO")
    d.setdefault("category", d.get("kind", "alert"))
    d.setdefault("message", "")
    return d


def alert_severity(alert: Any) -> str:
    """The upper-cased severity of an alert (defaults to ``INFO``)."""
    return str(normalize_alert(alert).get("severity", "INFO")).upper()


# ── alert sinks ────────────────────────────────────────────────────────────────────

@runtime_checkable
class AlertSink(Protocol):
    """Anything that can receive an alert. Implementations must not raise into the
    caller — the cycle must survive an observability failure."""

    def emit(self, alert: dict[str, Any]) -> None: ...


class NullAlertSink:
    """Drops alerts (the safe default for tests / silent RESEARCH runs)."""

    def emit(self, alert: dict[str, Any]) -> None:
        return None


class LoggingAlertSink:
    """Routes alerts to the stdlib logger at a severity-mapped level."""

    def __init__(self, logger_name: str = "tradingengineresearch.alerts") -> None:
        self._log = logging.getLogger(logger_name)

    def emit(self, alert: dict[str, Any]) -> None:
        a = normalize_alert(alert)
        level = SEVERITY_TO_LEVEL.get(alert_severity(a), logging.INFO)
        self._log.log(level, "ALERT %s | %s", a["severity"],
                      json.dumps(a, default=str, sort_keys=True))


class JsonlAlertSink:
    """Appends each alert as one JSON line — a durable, greppable alert trail."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def emit(self, alert: dict[str, Any]) -> None:
        a = normalize_alert(alert)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(a, default=str, sort_keys=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class CompositeAlertSink:
    """Fan-out to several sinks; one sink failing never blocks the others."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, alert: dict[str, Any]) -> None:
        for sink in self._sinks:
            try:
                sink.emit(alert)
            except Exception:  # noqa: BLE001 - an observability failure must not break the cycle
                logger.exception("alert sink %r failed", sink)


# ── metrics ──────────────────────────────────────────────────────────────────────

_LabelKey = tuple[str, tuple[tuple[str, str], ...]]


class MetricsRegistry:
    """A tiny counters + gauges registry with optional labels. ``snapshot()``
    gives a JSON view; ``render_prometheus()`` gives Prometheus text exposition."""

    def __init__(self) -> None:
        self._counters: dict[_LabelKey, float] = {}
        self._gauges: dict[_LabelKey, float] = {}

    @staticmethod
    def _key(name: str, labels: Optional[dict[str, Any]]) -> _LabelKey:
        items = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        return (name, items)

    def inc(self, name: str, amount: float = 1.0, **labels: Any) -> None:
        k = self._key(name, labels)
        self._counters[k] = self._counters.get(k, 0.0) + float(amount)

    def set_gauge(self, name: str, value: float, **labels: Any) -> None:
        self._gauges[self._key(name, labels)] = float(value)

    def get_counter(self, name: str, **labels: Any) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def get_gauge(self, name: str, **labels: Any) -> Optional[float]:
        return self._gauges.get(self._key(name, labels))

    def snapshot(self) -> dict[str, Any]:
        def fmt(d: dict[_LabelKey, float]) -> list[dict[str, Any]]:
            return [{"name": n, "labels": dict(lbls), "value": v} for (n, lbls), v in sorted(d.items())]

        return {"counters": fmt(self._counters), "gauges": fmt(self._gauges)}

    def render_prometheus(self) -> str:
        lines: list[str] = []
        typed: set[str] = set()

        def render(d: dict[_LabelKey, float], kind: str) -> None:
            for (name, lbls), value in sorted(d.items()):
                if name not in typed:
                    lines.append(f"# TYPE {name} {kind}")
                    typed.add(name)
                if lbls:
                    rendered = ",".join(f'{k}="{v}"' for k, v in lbls)
                    lines.append(f"{name}{{{rendered}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        render(self._counters, "counter")
        render(self._gauges, "gauge")
        return "\n".join(lines) + "\n"
