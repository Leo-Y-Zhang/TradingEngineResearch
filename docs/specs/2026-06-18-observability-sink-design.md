# Design — Logging/Metrics/Alerting Sink (ROADMAP Phase 6, item 5)

**Status:** accepted · **Date:** 2026-06-18

## Problem

Every cycle computes severity-graded alerts (`ops.monitoring.alert_list` →
`CycleResult.alerts` → the run-loop's `last_alerts`) and a 4-section monitoring
snapshot, but the alerts **go nowhere** — they live only in memory on the last
result. There is also no metrics surface for a running platform. ROADMAP Phase 6
item 5: route alerts to real sinks and expose metrics.

## Design

New module `ops/observability.py` (no heavy deps; importable everywhere):

### Alert sinks
- `AlertSink` Protocol: `emit(alert: dict) -> None`.
- `LoggingAlertSink` — severity → stdlib log level (INFO→INFO, WARNING/AMBER→
  WARNING, RED→ERROR); structured JSON payload.
- `JsonlAlertSink(path)` — append-only durable JSONL trail (one alert per line).
- `CompositeAlertSink(sinks)` — fan-out; **one failing sink never blocks the
  others** (logged loudly). Fail-soft is the rule: an observability failure must
  never break a trading cycle.
- `NullAlertSink` — drops alerts (test/silent default).
- `normalize_alert(alert)` — coerces a monitoring dict, a RiskEvent-like object,
  or a string to a `{severity, category, message}` dict (handles the run-loop's
  `kind` key as a `category` alias).

### Metrics
- `MetricsRegistry` — counters + gauges with optional labels; `snapshot()`
  (JSON) and `render_prometheus()` (text exposition with `# TYPE` headers and
  `name{label="v"} value`).

### Config (`core/config.py`)
- `AlertingSettings{sink: logging|jsonl|both|null, alert_log_path}`.
- `make_alert_sink(settings)` factory (lazy import, like `make_state_store`);
  `jsonl`/`both` default the path to `{state_dir}/alerts.jsonl`.

### Wiring
- `EngineService` gains `alert_sink` (default `make_alert_sink(settings)`) and a
  `metrics: MetricsRegistry`. In `run_once`, `_record_observability(result,
  alerts)` emits every alert to the sink and updates metrics
  (`engine_cycles_total`, `engine_blocked_cycles_total`,
  `engine_alerts_total{severity}`, gauges `engine_cycle_count`,
  `engine_live_orders_total`, `engine_book_size`). Fully fail-soft.
- API: `GET /metrics` (JSON snapshot) and `GET /metrics/prometheus` (text).

## Testing (TDD)
`tests/test_observability.py` — normalize/severity, each sink (incl. composite
failure isolation), metrics counters/gauges/labels/prometheus/snapshot, the
config factory, and run-loop wiring (alerts reach an injected sink; metrics
update; a sink that raises does not break the cycle). API tests cover both
metrics endpoints. Gates: pytest + ruff + mypy.

## Non-goals
- A metrics backend/exporter daemon (Prometheus scrapes `/metrics/prometheus`).
- Paging/webhook sinks (PagerDuty/Slack) — a follow-up; the `AlertSink` Protocol
  makes them drop-in.
