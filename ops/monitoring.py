"""
TradingEngineResearch — Monitoring
======================
The single-pane health view (Part 19.2): a four-section `snapshot()` and a
severity-graded `alert_list()`.

`snapshot()` assembles HEALTH / TRADING / MODEL / RISK from whatever live state
the caller supplies (the TradingEngineResearch engine passes the cycle's risk snapshot, crisis
status, execution stats, etc.), filling every documented sub-key with a safe,
float-coerced default when an input is absent so the dashboard never sees a hole.
Model versions are pulled directly from the model registry.

`alert_list()` turns the snapshot's danger signals into discrete, operator-facing
alerts, each tagged with a severity in {INFO, WARNING, AMBER, RED}, and surfaces
any `RiskEvent`s carried on the state.
"""

from __future__ import annotations

import logging
from typing import Optional

from ops.model_registry import get_model_registry

logger = logging.getLogger(__name__)

__all__ = [
    "ALERT_SEVERITIES",
    "snapshot",
    "alert_list",
]

ALERT_SEVERITIES: tuple[str, ...] = ("INFO", "WARNING", "AMBER", "RED")

# Drawdown alert thresholds, mirroring the risk drawdown governor levels.
_DD_WARNING, _DD_AMBER, _DD_RED = 0.05, 0.08, 0.12
# Crisis severity alert thresholds.
_SEV_AMBER, _SEV_RED = 0.33, 0.66


def _f(state: dict, key: str, default: float = 0.0) -> float:
    return float(state.get(key, default))


def _attr(obj: object, name: str, default: float = 0.0) -> float:
    return float(getattr(obj, name, default))


def snapshot(state: Optional[dict] = None) -> dict:
    """Return the current 4-section system-health snapshot."""
    state = state or {}
    risk = state.get("risk_snapshot")
    crisis = state.get("crisis_status")
    registry = get_model_registry()
    live = registry.latest_live()
    shadow = registry.latest_shadow()

    # TRADING.active_kill_switches: prefer the risk snapshot's flags, else state.
    if risk is not None:
        active_kill_switches = list(getattr(risk, "active_flags", []) or [])
    else:
        active_kill_switches = list(state.get("active_kill_switches", []) or [])

    health = {
        "heartbeat_age_seconds": _f(state, "heartbeat_age_seconds"),
        "market_data_latency_ms": _f(state, "market_data_latency_ms"),
        "stale_feature_count": int(state.get("stale_feature_count", 0)),
        "broker_rejection_rate": _f(state, "broker_rejection_rate"),
        "failed_prediction_count": int(state.get("failed_prediction_count", 0)),
        "ibkr_connected": bool(state.get("ibkr_connected", False)),
    }

    trading = {
        "gross_exposure": _f(state, "gross_exposure"),
        "net_exposure": _f(state, "net_exposure"),
        "turnover_today": _f(state, "turnover_today"),
        "fill_rate": _f(state, "fill_rate"),
        "avg_slippage_bps": _f(state, "avg_slippage_bps"),
        "expected_vs_realized_cost_delta": _f(state, "expected_vs_realized_cost_delta"),
        "active_kill_switches": active_kill_switches,
    }

    model = {
        "rolling_ic_20d": _f(state, "rolling_ic_20d"),
        "calibration_error": _f(state, "calibration_error"),
        "shadow_vs_live_divergence": _f(state, "shadow_vs_live_divergence"),
        "drift_flags_active": list(state.get("drift_flags_active", []) or []),
        "last_refit_timestamp": state.get("last_refit_timestamp"),
        "model_version_live": (live.model_id if live is not None
                               else state.get("model_version_live")),
        "model_version_shadow": (shadow.model_id if shadow is not None
                                 else state.get("model_version_shadow")),
    }

    if risk is not None:
        risk_section = {
            "drawdown_current": _attr(risk, "drawdown_current"),
            "vol_utilization": _attr(risk, "target_vol_utilization"),
            "cvar_utilization": _attr(risk, "cvar_utilization"),
        }
    else:
        risk_section = {
            "drawdown_current": _f(state, "drawdown_current"),
            "vol_utilization": _f(state, "vol_utilization"),
            "cvar_utilization": _f(state, "cvar_utilization"),
        }
    if crisis is not None:
        risk_section["severity_score"] = _attr(crisis, "severity_score")
        risk_section["liquidity_stress_score"] = _attr(crisis, "liquidity_stress_score")
    else:
        risk_section["severity_score"] = _f(state, "severity_score")
        risk_section["liquidity_stress_score"] = _f(state, "liquidity_stress_score")

    return {"HEALTH": health, "TRADING": trading, "MODEL": model, "RISK": risk_section}


def _alert(severity: str, category: str, message: str) -> dict:
    return {"severity": severity, "category": category, "message": message}


def alert_list(state: Optional[dict] = None) -> list[dict]:
    """Derive active, severity-graded alerts from the current state."""
    snap = snapshot(state)
    state = state or {}
    alerts: list[dict] = []

    health, trading, model, risk = snap["HEALTH"], snap["TRADING"], snap["MODEL"], snap["RISK"]

    if trading["active_kill_switches"]:
        alerts.append(_alert("RED", "kill_switch",
                             f"Active kill switches: {', '.join(map(str, trading['active_kill_switches']))}"))

    dd = risk["drawdown_current"]
    if dd >= _DD_RED:
        alerts.append(_alert("RED", "drawdown", f"Drawdown {dd:.1%} breaches hard limit"))
    elif dd >= _DD_AMBER:
        alerts.append(_alert("AMBER", "drawdown", f"Drawdown {dd:.1%} elevated"))
    elif dd >= _DD_WARNING:
        alerts.append(_alert("WARNING", "drawdown", f"Drawdown {dd:.1%} above soft threshold"))

    sev = risk["severity_score"]
    if sev >= _SEV_RED:
        alerts.append(_alert("RED", "crisis", f"Crisis severity {sev:.2f}"))
    elif sev >= _SEV_AMBER:
        alerts.append(_alert("AMBER", "crisis", f"Crisis severity {sev:.2f}"))

    if risk["vol_utilization"] > 1.0:
        alerts.append(_alert("AMBER", "risk", f"Vol utilization {risk['vol_utilization']:.2f} over budget"))
    if risk["cvar_utilization"] > 1.0:
        alerts.append(_alert("AMBER", "risk", f"CVaR utilization {risk['cvar_utilization']:.2f} over budget"))
    if risk["liquidity_stress_score"] >= _SEV_RED:
        alerts.append(_alert("AMBER", "liquidity", f"Liquidity stress {risk['liquidity_stress_score']:.2f}"))

    if not health["ibkr_connected"]:
        alerts.append(_alert("AMBER", "connectivity", "IBKR broker not connected"))
    if health["heartbeat_age_seconds"] > 120.0:
        alerts.append(_alert("AMBER", "heartbeat", f"Heartbeat stale ({health['heartbeat_age_seconds']:.0f}s)"))
    if health["stale_feature_count"] > 0:
        alerts.append(_alert("WARNING", "data", f"{health['stale_feature_count']} stale feature(s)"))
    if health["broker_rejection_rate"] > 0.10:
        alerts.append(_alert("AMBER", "broker", f"Broker rejection rate {health['broker_rejection_rate']:.1%}"))
    if health["failed_prediction_count"] > 0:
        alerts.append(_alert("WARNING", "model", f"{health['failed_prediction_count']} failed prediction(s)"))

    if model["drift_flags_active"]:
        alerts.append(_alert("WARNING", "drift",
                             f"Active drift flags: {', '.join(map(str, model['drift_flags_active']))}"))

    # Shadow/challenger lifecycle: a validated candidate is surfaced for a HUMAN
    # decision — promotion is never automatic (spec STEP 13).
    candidate = state.get("shadow_promotion_candidate")
    if candidate:
        alerts.append(_alert("INFO", "model_promotion",
                             f"Shadow model {candidate} passed validation; "
                             "eligible for manual promotion"))

    # Surface any explicit RiskEvents carried on the state (already severity-graded).
    for event in state.get("risk_events", []) or []:
        severity = str(getattr(event, "severity", "INFO"))
        if severity not in ALERT_SEVERITIES:
            severity = "INFO"
        alerts.append(_alert(
            severity,
            str(getattr(event, "event_type", "risk_event")),
            str(getattr(event, "description", "")),
        ))

    return alerts
