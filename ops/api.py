"""
TradingEngineResearch — Observation/Control API (ROADMAP Phase 6, item 3)
=============================================================
A read-mostly FastAPI surface over a running :class:`ops.run_loop.EngineService`,
so a live platform can be observed: health, status, the latest monitoring
snapshot + alerts, the current book, and the latest cycle summary — plus a
guarded on-demand single-cycle trigger.

FastAPI is an optional dependency (the ``app`` extra); it is imported lazily
inside :func:`create_app` so the quant core stays importable without it.

The API never reaches a broker itself: ``POST /cycle/run`` delegates to
``EngineService.run_once``, which routes through the engine's mode gate
(RESEARCH/PAPER place zero live orders; only LIVE may submit). Responses are
plain JSON-able primitives — no live objects, numpy scalars, or raw timestamps
leak out.

Security telemetry and per-client rate limiting live in :mod:`ops.api_security`
and are wired in by :func:`create_app`: one structured event per request, a
distinct ``auth_failed`` event for every rejected credential check, and separate
request budgets for the observation and the control endpoints.
"""

from __future__ import annotations

import hashlib
import logging
import math
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.engine.engine import CycleResult
from ops.api_security import (
    RateLimiter,
    RateLimitPolicy,
    SecurityLog,
    SecurityLogHandles,
    cap_field,
)
from ops.run_loop import CycleBusyError, EngineService

__all__ = [
    "UNTRUSTED_FORWARDED_IP",
    "api_uvicorn_kwargs",
    "create_app",
    "create_app_from_settings",
    "summarize_result",
]

# What the trail records instead of a client address when the address in front of
# us did not come from a socket. See ``_peer_ip``.
UNTRUSTED_FORWARDED_IP = "forwarded-untrusted"

# HTTP methods are a token of unbounded length as far as h11 is concerned, so the
# method is attacker-controlled text on its way to a log like everything else.
_METHOD_MAX_CHARS = 16


def api_uvicorn_kwargs() -> dict[str, Any]:
    """The server-level settings every TradingEngineResearch deployment must serve uvicorn with.

    **``proxy_headers=False`` is a security control, not a preference.**
    ``uvicorn.run()`` defaults it to ``True`` with
    ``forwarded_allow_ips="127.0.0.1"``, and the recommended topology here is a
    LOOPBACK bind — so uvicorn's own ``ProxyHeadersMiddleware`` trusted every
    caller, rewrote ``scope["client"]`` from a caller-supplied ``X-Forwarded-For``,
    and did it OUTSIDE the application. Measured on a real server before this fix:
    a request from 127.0.0.1 carrying ``X-Forwarded-For: 203.0.113.99,
    198.51.100.7`` was recorded in the security trail as ``client_ip
    198.51.100.7``, and a 4,000-character header put 4,000 attacker-chosen
    characters in it. Every application-level check of that header was therefore
    moot: ``create_app(trusted_proxy_header=...)`` had already been bypassed.

    Turning uvicorn's middleware off leaves exactly one place a forwarded header
    may be honoured — the ``trusted_proxy_header`` the operator names explicitly —
    which also keeps ``peer_ip`` in the trail equal to the true socket peer. That
    is what lets an operator tell a request that came through their proxy from one
    that did not, a distinction uvicorn's rewrite destroys even when the proxy is
    real.
    """
    return {"proxy_headers": False}


def _clean(obj: Any) -> Any:
    """Recursively coerce a value to JSON-safe primitives (numpy scalars → py,
    datetimes → ISO-8601), so no engine internals leak through the API."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    # numpy scalar / other number-like
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _clean(item())
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def summarize_result(result: Optional[CycleResult]) -> Optional[dict[str, Any]]:
    """Coerce a ``CycleResult`` dataclass to a JSON-safe summary of primitives."""
    if result is None:
        return None
    decisions = result.decisions or {}
    admitted = [s for s, d in decisions.items() if getattr(d, "take_trade", False)]
    summary = {
        "mode": result.mode,
        "asof_time": result.asof_time,
        "blocked": bool(result.blocked),
        "regime_label": result.regime_label,
        "execution_regime": result.execution_regime,
        "crisis_level": (result.crisis or {}).get("level"),
        "vol_forecasts": result.vol_forecasts,
        "n_predictions": len(result.predictions or {}),
        "n_decisions": len(decisions),
        "admitted": admitted,
        "target_weights": result.target_weights,
        "achieved_weights": result.achieved_weights,
        "n_orders": len(result.order_intents or []),
        "n_fills": len(result.fills or []),
        "live_orders_submitted": int(result.live_orders_submitted),
        "n_alerts": len(result.alerts or []),
        "audit": result.audit,
    }
    return _clean(summary)


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TRADING ENGINE — Trading Terminal</title>
<style>
  :root{
    --bg:#0a0b0e; --panel:#101319; --panel2:#0d1015; --line:#1e2430; --line2:#161b23;
    --amber:#ffb000; --green:#33ff77; --green2:#2ecc71; --red:#ff4d4d; --cyan:#46c6ff;
    --muted:#7d8aa0; --txt:#d6deea; --txt-dim:#9aa7bd;
    color-scheme:dark;
  }
  *{box-sizing:border-box;}
  html,body{height:100%;}
  body{
    margin:0; color:var(--txt);
    background:
      radial-gradient(1200px 620px at 82% -12%, #11151d 0%, rgba(17,21,29,0) 60%),
      var(--bg);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    font-size:13px; -webkit-font-smoothing:antialiased;
  }
  /* ── top command bar ─────────────────────────────────────────── */
  .topbar{
    display:flex; align-items:center; gap:18px; flex-wrap:wrap;
    padding:12px 18px; border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,#0c0f15,#0a0b0e);
    position:sticky; top:0; z-index:10;
  }
  .brand{display:flex; flex-direction:column; line-height:1;}
  .wordmark{
    font-family:ui-monospace,"Cascadia Mono",Consolas,Menlo,monospace;
    font-weight:800; font-size:21px; letter-spacing:6px; color:var(--amber);
    text-shadow:0 0 20px rgba(255,176,0,.28);
  }
  .subtitle{margin-top:6px; font-size:9.5px; letter-spacing:3px; color:var(--muted); text-transform:uppercase;}
  .chips{display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin-left:auto;}
  .chip{
    display:inline-flex; align-items:center; gap:7px;
    padding:5px 10px; border:1px solid var(--line); border-radius:4px;
    background:#0c1016; font-size:10.5px; letter-spacing:1px;
  }
  .chip-k{color:var(--muted); text-transform:uppercase;}
  .chip-v{font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; font-weight:700; letter-spacing:1px;}
  .chip.live{color:var(--green); border-color:#1c3b28; background:#0b1610; font-weight:700; letter-spacing:1.5px;}
  .led{width:8px; height:8px; border-radius:50%; background:var(--muted); box-shadow:0 0 7px currentColor; display:inline-block;}
  .led.green{background:var(--green); color:var(--green);}
  .led.red{background:var(--red); color:var(--red);}
  .led.grey{background:var(--muted); color:var(--muted);}
  .blink{animation:blink 1.4s steps(1,end) infinite;}
  @keyframes blink{50%{opacity:.25;}}
  /* ── health strip ────────────────────────────────────────────── */
  .healthstrip{
    display:flex; align-items:center; gap:10px;
    padding:9px 20px; font-weight:700; letter-spacing:1.6px; font-size:11.5px; text-transform:uppercase;
    border-bottom:1px solid var(--line); color:var(--muted);
  }
  .healthstrip .hdot{width:9px; height:9px; border-radius:50%; background:currentColor; box-shadow:0 0 9px currentColor;}
  .healthstrip.ok{color:var(--green); background:linear-gradient(180deg,#0b1711,#0a120d);}
  .healthstrip.warn{color:var(--amber); background:linear-gradient(180deg,#171206,#120e05);}
  .healthstrip.bad{color:var(--red); background:linear-gradient(180deg,#190b0b,#120808);}
  .healthstrip.info{color:var(--cyan); background:linear-gradient(180deg,#08141c,#0a0f15);}
  /* ── panel grid ──────────────────────────────────────────────── */
  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:13px; padding:16px;}
  .panel{
    background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line); border-radius:6px; overflow:hidden;
    display:flex; flex-direction:column;
  }
  .panel-h{
    display:flex; align-items:center; justify-content:space-between;
    padding:8px 12px; border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,#12161d,#0e1117);
  }
  .ph-title{color:var(--amber); font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase;}
  .ph-dot{width:6px; height:6px; border-radius:50%; background:var(--amber); opacity:.6; animation:blink 2.2s steps(1,end) infinite;}
  .panel-b{padding:13px 14px; display:flex; flex-direction:column; gap:12px; flex:1;}
  /* ── big number ──────────────────────────────────────────────── */
  .bignum{display:flex; align-items:baseline; gap:11px;}
  .bn-v{
    font-family:ui-monospace,Consolas,monospace; font-size:38px; font-weight:800;
    color:var(--txt); letter-spacing:1px; line-height:1; font-variant-numeric:tabular-nums;
  }
  .bn-l{font-size:9.5px; letter-spacing:2px; color:var(--muted); text-transform:uppercase;}
  /* ── key/value rows ──────────────────────────────────────────── */
  .rows{display:flex; flex-direction:column;}
  .row{display:flex; justify-content:space-between; align-items:center; gap:14px; padding:5px 0; border-bottom:1px dotted var(--line2);}
  .row:last-child{border-bottom:none;}
  .row .k{color:var(--muted); font-size:10px; letter-spacing:1.2px; text-transform:uppercase; white-space:nowrap;}
  .row .v{
    font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;
    font-weight:600; text-align:right; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  }
  .green{color:var(--green);} .red{color:var(--red);} .cyan{color:var(--cyan);} .amber{color:var(--amber);} .muted{color:var(--muted);}
  /* ── tables ──────────────────────────────────────────────────── */
  .tbl{width:100%; border-collapse:collapse; font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums;}
  .tbl th{
    font-family:system-ui,sans-serif; text-align:left; color:var(--muted); font-size:9.5px;
    letter-spacing:1.5px; text-transform:uppercase; font-weight:600; padding:4px 8px; border-bottom:1px solid var(--line);
  }
  .tbl td{padding:6px 8px; border-bottom:1px solid var(--line2); font-size:12.5px;}
  .tbl td.sym{font-weight:700; color:var(--txt);}
  .tbl .num{text-align:right;}
  .tbl tfoot td{border-top:1px solid var(--line); border-bottom:none; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:1px;}
  /* ── sparkline ───────────────────────────────────────────────── */
  .spark{margin-top:auto;}
  .spark-l{font-size:9px; letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin-bottom:5px;}
  .sparksvg{width:100%; height:44px; display:block;}
  /* ── alerts ──────────────────────────────────────────────────── */
  .alert{display:flex; align-items:center; gap:9px; padding:6px 0; border-bottom:1px dotted var(--line2);}
  .alert:last-child{border-bottom:none;}
  .alert-msg{font-size:12px; color:var(--txt-dim);}
  .pill{font-size:9.5px; font-weight:700; letter-spacing:1px; padding:2px 7px; border-radius:3px; text-transform:uppercase;}
  .pill.sev-red{background:#2a0e0e; color:var(--red); border:1px solid #5a1d1d;}
  .pill.sev-amber{background:#241b06; color:var(--amber); border:1px solid #5a4413;}
  .pill.sev-cyan{background:#08202c; color:var(--cyan); border:1px solid #154a5e;}
  /* ── metrics grid ────────────────────────────────────────────── */
  .mgrid{display:grid; grid-template-columns:1fr 1fr; gap:1px 18px;}
  .mcell{display:flex; justify-content:space-between; gap:10px; padding:4px 0; border-bottom:1px dotted var(--line2); min-width:0;}
  .mk{color:var(--muted); font-size:9.5px; letter-spacing:.8px; text-transform:uppercase; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
  .mv{font-family:ui-monospace,Consolas,monospace; font-variant-numeric:tabular-nums; font-weight:600; color:var(--cyan); white-space:nowrap;}
  /* ── misc ────────────────────────────────────────────────────── */
  .empty{color:var(--muted); font-size:11px; letter-spacing:1px; text-transform:uppercase; padding:16px 4px; text-align:center;}
  .footer{
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:11px 18px; border-top:1px solid var(--line); margin-top:6px;
    font-size:10px; letter-spacing:1.2px; color:var(--muted); text-transform:uppercase;
  }
  .footer .blink{color:var(--green);}
  .foot-links{margin-left:auto; font-family:ui-monospace,Consolas,monospace; color:#5a6678; letter-spacing:.5px; text-transform:none;}
  @media (max-width:560px){ .subtitle{display:none;} .chips{gap:6px;} }
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">
    <div class="wordmark">TRADING ENGINE</div>
    <div class="subtitle">Systematic Trading Terminal</div>
  </div>
  <div class="chips">
    <div class="chip"><span class="chip-k">Mode</span><span class="chip-v amber" id="c_mode">--</span></div>
    <div class="chip"><span class="led grey" id="c_broker_led"></span><span class="chip-k">Broker</span><span class="chip-v muted" id="c_broker">--</span></div>
    <div class="chip"><span class="led grey" id="c_kill_led"></span><span class="chip-k">Kill</span><span class="chip-v muted" id="c_kill">--</span></div>
    <div class="chip"><span class="chip-k">UTC</span><span class="chip-v cyan" id="c_clock">--:--:--</span></div>
    <div class="chip live"><span class="blink">&#9679;</span>&nbsp;LIVE</div>
    <div class="chip"><span class="chip-v muted" id="updated">connecting&hellip;</span></div>
  </div>
</div>

<div class="healthstrip info" id="health"><span class="hdot"></span><span id="health_txt">INITIALISING TERMINAL&hellip;</span></div>

<div class="grid">
  <!-- SYSTEM -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">System</span><span class="ph-dot"></span></div>
    <div class="panel-b">
      <div class="bignum"><span class="bn-v" id="sys_cycle">--</span><span class="bn-l">Cycles<br/>Completed</span></div>
      <div class="rows">
        <div class="row"><span class="k">Mode</span><span class="v" id="sys_mode">--</span></div>
        <div class="row"><span class="k">Last Cycle (UTC)</span><span class="v" id="sys_last">--</span></div>
        <div class="row"><span class="k">Cycle State</span><span class="v" id="sys_blocked">--</span></div>
        <div class="row"><span class="k">Capital</span><span class="v" id="sys_capital">--</span></div>
        <div class="row"><span class="k">Universe</span><span class="v" id="sys_universe">--</span></div>
      </div>
      <div class="spark"><div class="spark-l">Cycle Throughput</div><div id="sys_spark"></div></div>
    </div>
  </section>

  <!-- EXECUTION -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">Execution</span><span class="ph-dot"></span></div>
    <div class="panel-b">
      <div class="bignum"><span class="bn-v" id="exec_live">--</span><span class="bn-l">Live Orders<br/>Cumulative</span></div>
      <div class="rows">
        <div class="row"><span class="k">Last Orders / Fills</span><span class="v" id="exec_of">--</span></div>
        <div class="row"><span class="k">Admitted</span><span class="v" id="exec_admit">--</span></div>
        <div class="row"><span class="k">Predictions / Decisions</span><span class="v" id="exec_pd">--</span></div>
        <div class="row"><span class="k">Regime</span><span class="v" id="exec_regime">--</span></div>
        <div class="row"><span class="k">Execution Regime</span><span class="v" id="exec_exreg">--</span></div>
        <div class="row"><span class="k">Crisis Level</span><span class="v" id="exec_crisis">--</span></div>
      </div>
    </div>
  </section>

  <!-- POSITIONS -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">Positions</span><span class="ph-dot"></span></div>
    <div class="panel-b"><div id="pos_body"><div class="empty">Awaiting book&hellip;</div></div></div>
  </section>

  <!-- RISK & EXPOSURE -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">Risk &amp; Exposure</span><span class="ph-dot"></span></div>
    <div class="panel-b">
      <div id="risk_body"><div class="empty">Awaiting snapshot&hellip;</div></div>
      <div class="spark"><div class="spark-l" id="risk_spark_l">Exposure</div><div id="risk_spark"></div></div>
    </div>
  </section>

  <!-- ALERTS -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">Alerts</span><span class="ph-dot"></span></div>
    <div class="panel-b"><div id="alerts_body"><div class="empty">No active alerts</div></div></div>
  </section>

  <!-- METRICS -->
  <section class="panel">
    <div class="panel-h"><span class="ph-title">Metrics</span><span class="ph-dot"></span></div>
    <div class="panel-b"><div id="metrics_body"><div class="empty">Awaiting metrics&hellip;</div></div></div>
  </section>
</div>

<div class="footer">
  <span class="blink">&#9679;</span> Live feed &middot; auto-refresh 3s &middot; read-only observability &middot; control endpoints are disabled on this surface in LIVE
  <span class="foot-links">/status &middot; /health &middot; /book &middot; /monitoring &middot; /metrics &middot; /cycle/latest</span>
</div>

<script>
"use strict";
const $ = function(id){ return document.getElementById(id); };
const HIST = { cycle: [], nav: [] };
const HMAX = 60;

function esc(s){
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function pretty(k){ return String(k).replace(/[._]+/g," ").trim().toUpperCase(); }
function pushHist(arr, v){
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return;
  arr.push(Number(v));
  while (arr.length > HMAX) arr.shift();
}
function fmtVal(v){
  if (v === null || v === undefined) return "--";
  if (typeof v === "boolean") return v ? "YES" : "NO";
  if (typeof v === "number"){
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    if (a !== 0 && a < 0.001) return v.toExponential(2);
    return v.toLocaleString("en-US", {minimumFractionDigits:2, maximumFractionDigits:4});
  }
  return String(v);
}
function fmtTs(iso){ return iso ? String(iso).replace("T"," ").slice(0,19) : "--"; }
function fmtMoney(v){
  const n = Number(v);
  if (isNaN(n)) return "--";
  const dp = Math.abs(n) >= 1000 ? 0 : 2;
  return "£" + n.toLocaleString("en-GB", {minimumFractionDigits:dp, maximumFractionDigits:dp});
}
function flatten(obj, prefix, out){
  out = out || {};
  if (!obj || typeof obj !== "object") return out;
  for (const k in obj){
    if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
    const v = obj[k];
    const key = prefix ? prefix + "." + k : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, key, out);
    else out[key] = v;
  }
  return out;
}
function sevClass(sev){
  if (/CRIT|FATAL|ERROR|SEVERE|HIGH|RED|DANGER/.test(sev)) return "sev-red";
  if (/WARN|MED|AMBER|ELEVAT|CAUTION/.test(sev)) return "sev-amber";
  return "sev-cyan";
}
function drawSpark(id, values, color){
  const el = $(id);
  if (!el) return;
  const W = 300, H = 44;
  if (!values || values.length < 2){
    el.innerHTML = '<svg class="sparksvg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none"></svg>';
    return;
  }
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++){ if (values[i] < min) min = values[i]; if (values[i] > max) max = values[i]; }
  const span = (max - min) || 1, n = values.length;
  let pts = "";
  let lx = 0, ly = 0;
  for (let i = 0; i < n; i++){
    const x = (i / (n - 1)) * (W - 4) + 2;
    const y = (H - 3) - ((values[i] - min) / span) * (H - 6);
    pts += x.toFixed(1) + "," + y.toFixed(1) + " ";
    lx = x; ly = y;
  }
  el.innerHTML =
    '<svg class="sparksvg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
    '<polyline fill="none" stroke="' + color + '" stroke-width="1.5" vector-effect="non-scaling-stroke" points="' + pts.trim() + '"/>' +
    '<circle cx="' + lx.toFixed(1) + '" cy="' + ly.toFixed(1) + '" r="2.4" fill="' + color + '"/>' +
    '</svg>';
}
function setHealth(cls, txt){
  const el = $("health");
  el.className = "healthstrip " + cls;
  $("health_txt").textContent = txt;
}

async function getJSON(path){
  const r = await fetch(path, {cache:"no-store"});
  if (!r.ok){ const e = new Error(path + " " + r.status); e.status = r.status; throw e; }
  return r.json();
}

function renderStatus(s){
  $("c_mode").textContent = s.mode || "--";
  $("sys_mode").textContent = s.mode || "--";
  $("sys_cycle").textContent = (s.cycle_count != null ? s.cycle_count : 0);
  $("exec_live").textContent = (s.live_orders_total != null ? s.live_orders_total : 0);
  $("sys_last").textContent = s.last_asof ? fmtTs(s.last_asof) : "NO CYCLE YET";
  $("sys_capital").textContent = (s.capital_gbp != null) ? fmtMoney(s.capital_gbp) : "--";
  const uni = s.universe || [];
  $("sys_universe").textContent = uni.length ? (uni.length + " · " + uni.join(" ")) : "--";
  $("sys_universe").title = uni.join(", ");

  const blk = $("sys_blocked");
  if (s.blocked === true){ blk.textContent = "BLOCKED"; blk.className = "v red"; }
  else if (s.blocked === false){ blk.textContent = "OK"; blk.className = "v green"; }
  else { blk.textContent = "--"; blk.className = "v muted"; }

  const bc = s.broker_connected;
  const led = $("c_broker_led"), bv = $("c_broker");
  if (bc === true){ led.className = "led green"; bv.textContent = "CONNECTED"; bv.className = "chip-v green"; }
  else if (bc === false){ led.className = "led red"; bv.textContent = "DISCONNECTED"; bv.className = "chip-v red"; }
  else { led.className = "led grey"; bv.textContent = "N/A"; bv.className = "chip-v muted"; }

  const latched = s.kill_switch_latched === true;
  $("c_kill_led").className = latched ? "led red" : "led green";
  const kv = $("c_kill");
  kv.textContent = latched ? "LATCHED" : "CLEAR";
  kv.className = latched ? "chip-v red" : "chip-v green";

  const recon = (s.open_reconciliations || []).length;
  if (latched) setHealth("bad", "Kill-switch latched — " + (s.kill_switch_reason || "trading halted"));
  else if (bc === false) setHealth("bad", "Broker disconnected — execution path is down");
  else if (recon > 0) setHealth("warn", recon + " open reconciliation" + (recon > 1 ? "s" : "") + " — operator action required");
  else if (bc === true) setHealth("ok", "System healthy — connected, armed, reconciled");
  else setHealth("info", "System nominal — no broker attached (" + (s.mode || "?") + ")");

  pushHist(HIST.cycle, Number(s.cycle_count) || 0);
  drawSpark("sys_spark", HIST.cycle, "#46c6ff");
}

function renderCycle(c){
  $("exec_of").textContent = (c.n_orders != null ? c.n_orders : 0) + " / " + (c.n_fills != null ? c.n_fills : 0);
  const adm = c.admitted || [];
  $("exec_admit").textContent = adm.length ? (adm.length + " · " + adm.join(" ")) : "0";
  $("exec_pd").textContent = (c.n_predictions != null ? c.n_predictions : 0) + " / " + (c.n_decisions != null ? c.n_decisions : 0);
  $("exec_regime").textContent = c.regime_label || "--";
  $("exec_exreg").textContent = c.execution_regime || "--";
  const lvl = String(c.crisis_level || "NONE").toUpperCase();
  const cel = $("exec_crisis");
  cel.textContent = lvl;
  cel.className = (lvl === "NONE" || lvl === "") ? "v green" : (/HIGH|CRIT|SEVERE|RED/.test(lvl) ? "v red" : "v amber");
}
function renderCycleMissing(){
  $("exec_of").textContent = "—";
  $("exec_admit").textContent = "—";
  $("exec_pd").textContent = "—";
  $("exec_regime").textContent = "AWAITING FIRST CYCLE";
  $("exec_exreg").textContent = "—";
  $("exec_crisis").textContent = "—";
  $("exec_crisis").className = "v muted";
}

function renderBook(b){
  const cur = (b && b.current_book) || {};
  const keys = Object.keys(cur);
  if (!keys.length){ $("pos_body").innerHTML = '<div class="empty">Flat — no positions</div>'; return; }
  keys.sort(function(a, b){ return Math.abs(cur[b]) - Math.abs(cur[a]); });
  let gross = 0, net = 0;
  let h = '<table class="tbl"><thead><tr><th>Symbol</th><th class="num">Weight</th><th class="num">Side</th></tr></thead><tbody>';
  for (const k of keys){
    const w = Number(cur[k]) || 0;
    gross += Math.abs(w); net += w;
    const cls = w > 0 ? "green" : (w < 0 ? "red" : "muted");
    const side = w > 0 ? "LONG" : (w < 0 ? "SHORT" : "FLAT");
    h += '<tr><td class="sym">' + esc(k) + '</td><td class="num ' + cls + '">' + (w * 100).toFixed(2) + '%</td><td class="num ' + cls + '">' + side + '</td></tr>';
  }
  h += '</tbody><tfoot><tr><td>Gross / Net</td><td class="num">' + (gross * 100).toFixed(1) + '%</td><td class="num">' + (net * 100).toFixed(1) + '%</td></tr></tfoot></table>';
  $("pos_body").innerHTML = h;
}

function renderMonitoring(m){
  const snap = (m && m.snapshot) || {};
  const flat = flatten(snap, "", {});
  const keys = Object.keys(flat);
  if (!keys.length){
    $("risk_body").innerHTML = '<div class="empty">Awaiting snapshot&hellip;</div>';
  } else {
    keys.sort();
    let h = '<div class="rows">';
    for (const k of keys){
      const v = flat[k];
      let disp, cls = "v";
      if (typeof v === "number"){
        if (/draw.?down|fill.?rate|turnover|exposure|vol|ratio|weight|pct|gross|net|leverage/i.test(k) && Math.abs(v) <= 25){
          disp = (v * 100).toFixed(2) + "%";
        } else {
          disp = fmtVal(v);
        }
        if (/draw.?down|loss/i.test(k) && v < 0) cls = "v red";
        else if (/draw.?down|loss/i.test(k)) cls = "v amber";
      } else {
        disp = fmtVal(v);
      }
      h += '<div class="row"><span class="k">' + esc(pretty(k)) + '</span><span class="' + cls + '" title="' + esc(disp) + '">' + esc(disp) + '</span></div>';
    }
    h += '</div>';
    $("risk_body").innerHTML = h;
  }

  let navKey = null;
  for (const k of keys){ if (/\bnav\b|equity|gross.?exposure|net.?exposure|exposure/i.test(k) && typeof flat[k] === "number"){ navKey = k; break; } }
  if (navKey){
    pushHist(HIST.nav, Number(flat[navKey]) || 0);
    $("risk_spark_l").textContent = pretty(navKey);
    drawSpark("risk_spark", HIST.nav, "#33ff77");
  }

  const alerts = (m && m.alerts) || [];
  if (!alerts.length){
    $("alerts_body").innerHTML = '<div class="empty">No active alerts</div>';
  } else {
    $("alerts_body").innerHTML = alerts.map(function(a){
      const sev = String((a && (a.level || a.severity)) || "INFO").toUpperCase();
      const msg = (a && (a.message || a.name)) || (typeof a === "string" ? a : JSON.stringify(a));
      return '<div class="alert"><span class="pill ' + sevClass(sev) + '">' + esc(sev) + '</span><span class="alert-msg">' + esc(msg) + '</span></div>';
    }).join("");
  }
}

function renderMetrics(m){
  const flat = flatten(m || {}, "", {});
  const keys = Object.keys(flat).sort();
  if (!keys.length){ $("metrics_body").innerHTML = '<div class="empty">Awaiting metrics&hellip;</div>'; return; }
  let h = '<div class="mgrid">';
  for (const k of keys){
    h += '<div class="mcell"><span class="mk" title="' + esc(pretty(k)) + '">' + esc(pretty(k)) + '</span><span class="mv">' + esc(fmtVal(flat[k])) + '</span></div>';
  }
  h += '</div>';
  $("metrics_body").innerHTML = h;
}

function showDisconnected(){
  setHealth("bad", "DISCONNECTED — is the service running?");
  $("updated").textContent = "DISCONNECTED";
  $("c_broker_led").className = "led red";
}

async function refresh(){
  let status;
  try {
    status = await getJSON("/status");
  } catch (e){
    showDisconnected();
    return;
  }
  try { renderStatus(status); } catch (e){}

  getJSON("/cycle/latest").then(renderCycle).catch(function(){ renderCycleMissing(); });
  getJSON("/book").then(renderBook).catch(function(){});
  getJSON("/monitoring").then(renderMonitoring).catch(function(){});
  getJSON("/metrics").then(renderMetrics).catch(function(){});

  const now = new Date();
  $("updated").textContent = "UPDATED " + now.toTimeString().slice(0, 8);
}

function tickClock(){
  $("c_clock").textContent = new Date().toISOString().slice(11, 19);
}

tickClock();
setInterval(tickClock, 1000);
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def assert_bind_is_safe(host: str, api_token: str | None) -> None:
    """Refuse to expose an UNAUTHENTICATED control API off loopback (Phase 8).

    RUN-5 made a direct run bind loopback by default, which stops accidental
    exposure but does nothing once someone deliberately sets
    ``ENGINE_API_HOST=0.0.0.0`` — and this API can trigger cycles, reset the
    kill switch and resolve reconciliation items. So the combination "reachable
    off-host" + "no token" is refused at STARTUP rather than served and regretted:
    fail-closed, in the same spirit as the mode gates.
    """
    if host in LOOPBACK_HOSTS or api_token:
        return
    raise RuntimeError(
        f"refusing to serve the control API on {host!r} without authentication: "
        "set ENGINE_API_TOKEN, or bind 127.0.0.1. This API can trigger "
        "cycles, reset the kill switch and resolve reconciliation items."
    )


def create_app(
    service: EngineService,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    api_token: str | None = None,
    rate_limits: RateLimitPolicy | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    trusted_proxy_header: str | None = None,
) -> Any:
    """Build the FastAPI app bound to ``service``. Lazy-imports FastAPI.

    ``api_token`` (Phase 8): when set, every endpoint that exposes book/NAV data
    or changes state requires ``Authorization: Bearer <token>``. ``/health`` stays
    open so a liveness probe needs no secret, and ``/`` still serves the dashboard
    HTML — but its client-side fetches will 401, so the dashboard is a
    loopback/no-token convenience, not an authenticated UI. The mode gates on the
    mutating endpoints are UNCHANGED: a token does not unlock LIVE mutations,
    because relaxing those is a risk decision for the operator, not a side effect
    of adding auth.

    ``rate_limits`` (SEC-2): a generous budget for the observation endpoints and a
    tight one shared by the three control endpoints, because a repeated
    ``/kill-switch/reset`` is a financial-safety event rather than a nuisance. The
    limiter is checked BEFORE the token, so guessing the token is capped too.
    ``/health`` and ``/`` are exempt (a liveness probe must never be throttled).
    ``time_fn`` is the monotonic clock behind both the limiter windows and request
    timing; injectable for tests.

    ``SEC-4`` — **whose budget is it?** The budget was keyed on the client IP, and
    in every recommended topology (loopback bind, or a compose port map where the
    peer is the docker gateway) *every* caller presents the same IP. That made one
    global budget which any unauthenticated third party could exhaust, denying the
    operator the surface they need to diagnose with — measured: two anonymous 401s
    locked the token holder out. So the budget is now keyed on the caller's
    **identity**: a request bearing the valid token is keyed on a fingerprint of
    that token (never the token itself), and everything else is keyed on the client
    address. An anonymous flood can therefore only exhaust the anonymous budget,
    while token guessing stays capped exactly as before.

    ``trusted_proxy_header`` (SEC-4): OFF by default, because ``X-Forwarded-For``
    is caller-supplied and a spoofable identity is worse than a coarse one — it
    would let an attacker mint a fresh budget per request. Set it only when the API
    is unreachable except through a proxy that *sets* the header itself; the value
    used is the RIGHTMOST entry, which is the address that immediately-adjacent
    proxy observed (the leftmost is whatever the client claimed).

    ``SEC-1``: every request emits one structured event on the
    ``tradingengineresearch.api.security`` logger (request id, client IP, route, method,
    status, outcome, duration), and each rejected credential check emits its own
    ``auth_failed`` event. The token is never logged. Route that logger to a file
    with ``ops.api_security.attach_security_log_file`` — the console alone is not
    detection.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import HTMLResponse, PlainTextResponse

    # PEP 563 (`from __future__ import annotations` above) turns every annotation
    # into a string, and FastAPI resolves a dependency's annotations against THIS
    # MODULE's globals — where a name imported lazily inside this function does not
    # exist. Without this, `request: Request` stays an unresolved string and FastAPI
    # treats it as a required query parameter (every guarded route 422s). Publishing
    # the one name used in a dependency signature is what lets the lazy import and
    # PEP 563 coexist; it is idempotent across repeated create_app calls.
    globals().setdefault("Request", Request)

    policy = rate_limits or RateLimitPolicy()
    seclog = SecurityLog()
    read_limiter = RateLimiter(policy.read_per_minute, policy.window_seconds, time_fn)
    write_limiter = RateLimiter(policy.write_per_minute, policy.window_seconds, time_fn)

    # Fingerprint, not the credential: this ends up in a rate-limiter key, and a
    # key is one careless log line away from being an exfiltrated token.
    token_identity = (
        "token:" + hashlib.sha256(api_token.encode("utf-8")).hexdigest()[:12]
        if api_token
        else None
    )
    expected_authorization = (
        f"Bearer {api_token}".encode("utf-8") if api_token else None
    )

    def _credential_matches(authorization: str | None) -> bool:
        """Constant-time bearer check — a timing oracle on an ops token is still an
        oracle. Compared as BYTES because ``secrets.compare_digest`` raises
        ``TypeError`` on a non-ASCII str, and Starlette decodes header values as
        latin-1: one high byte in the header turned an unauthenticated request into
        an unhandled 500 (a stack-trace path open to anyone, standards §4.8).
        ``surrogatepass`` never drops a character — an ``errors="ignore"`` encode
        would let ``Bearer ab\\xffc`` match the token ``abc``.
        """
        if not authorization or expected_authorization is None:
            return False
        return secrets.compare_digest(
            authorization.encode("utf-8", "surrogatepass"), expected_authorization
        )

    def _rewritten_by_a_proxy_middleware(request: Request) -> bool:
        """Did something in front of us replace the socket peer with a header value?

        A real TCP peer always has a nonzero source port. uvicorn's
        ``ProxyHeadersMiddleware`` sets ``scope["client"] = (forwarded_host, 0)``
        — port zero, explicitly, because it has lost the port. So port 0 is a
        reliable signal that the "address" in front of us is caller-supplied text.

        ``api_uvicorn_kwargs`` turns that middleware off in both shipped entry
        points, so this should never fire there. It is defence in depth for the
        operator who runs ``uvicorn`` by hand and leaves its default on: without it
        the app would launder a forged address into the trail and into the
        rate-limiter key as though it were a fact.
        """
        client = request.client
        return client is not None and getattr(client, "port", None) == 0

    def _peer_ip(request: Request) -> str:
        client = request.client
        if client is None:
            return "unknown"
        return cap_field(client.host)

    def _client_ip(request: Request) -> str:
        # The socket peer unless the operator has explicitly named a proxy header
        # they control. Untrusted by default: the header is caller-supplied, and a
        # spoofable identity in a security log — or in a rate-limiter key — is
        # worse than a coarse one.
        if trusted_proxy_header:
            forwarded = request.headers.get(trusted_proxy_header)
            if forwarded:
                # Rightmost = what the adjacent proxy saw. Length-capped because
                # it is attacker-controlled text on its way to a log.
                candidate = cap_field(forwarded.split(",")[-1].strip())
                if candidate:
                    return candidate
            return _peer_ip(request)
        if _rewritten_by_a_proxy_middleware(request):
            # Collapse to ONE constant rather than the forged value. Returning the
            # value (even capped) would still let a caller mint a fresh rate-limit
            # budget per request by rotating the header. The raw claim is kept, in
            # its own field, by ``_who``.
            return UNTRUSTED_FORWARDED_IP
        return _peer_ip(request)

    def _who(request: Request) -> dict[str, str]:
        """The caller-identifying fields for one security event, all length-capped.

        ``client_ip`` is what the platform is willing to treat as the caller,
        ``peer_ip`` is the socket peer, and ``forwarded_claim`` appears only when
        the two disagree because something rewrote the peer from a header we were
        never told to trust — the value is kept for forensics but is never used as
        an identity.
        """
        fields = {"client_ip": _client_ip(request), "peer_ip": _peer_ip(request)}
        if not trusted_proxy_header and _rewritten_by_a_proxy_middleware(request):
            fields["forwarded_claim"] = fields["peer_ip"]
            fields["peer_ip"] = UNTRUSTED_FORWARDED_IP
        return fields

    def _method(request: Request) -> str:
        return cap_field(request.method, _METHOD_MAX_CHARS)

    def _rate_limit_identity(request: Request) -> str:
        """Who the budget belongs to. A valid credential identifies its holder;
        everyone else is identified by address and shares the anonymous budget."""
        if token_identity and _credential_matches(request.headers.get("authorization")):
            return token_identity
        return f"ip:{_client_ip(request)}"

    def _request_id(request: Request) -> str:
        return str(getattr(request.state, "request_id", "-"))

    def _route(request: Request) -> str:
        # An unmatched path is attacker-controlled, so it is length-capped before
        # it reaches the log. (Control characters need no special handling: the
        # payload is JSON-encoded, which escapes them, so a path cannot forge a
        # second log line.)
        return request.url.path[:200]

    def _rate_limit(limiter: RateLimiter, bucket: str) -> Callable[[Request], None]:
        def _rate_limit_dependency(request: Request) -> None:
            if not limiter.enabled:
                return
            identity = _rate_limit_identity(request)
            decision = limiter.check(f"{bucket}:{identity}")
            if decision.allowed:
                return
            retry_after = max(1, int(math.ceil(decision.retry_after)))
            seclog.emit(
                "rate_limited",
                logging.WARNING,
                request_id=_request_id(request),
                **_who(request),
                identity=identity,
                method=_method(request),
                route=_route(request),
                bucket=bucket,
                limit=limiter.limit,
                window_seconds=policy.window_seconds,
                retry_after=retry_after,
            )
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded for {bucket} endpoints",
                headers={"Retry-After": str(retry_after)},
            )

        return _rate_limit_dependency

    def _require_token(
        request: Request, authorization: str | None = Header(default=None)
    ) -> None:
        if not api_token:
            return
        if not _credential_matches(authorization):
            # The highest-value security signal on this surface. The credential
            # itself is NEVER a field — neither the expected token nor the one
            # the caller sent, since logging an attacker's guess next to a
            # near-miss is how a token leaks into a log aggregator.
            seclog.emit(
                "auth_failed",
                logging.WARNING,
                request_id=_request_id(request),
                **_who(request),
                method=_method(request),
                route=_route(request),
                reason="missing_credentials" if not authorization else "invalid_credentials",
            )
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Order matters: throttle first, authenticate second.
    guard = [Depends(_rate_limit(read_limiter, "read")), Depends(_require_token)]
    write_guard = [Depends(_rate_limit(write_limiter, "control")), Depends(_require_token)]
    # SEC-5: FastAPI's generated documentation is part of the admin surface. Left
    # to itself it mounts /openapi.json, /docs and /redoc with NO dependencies, so
    # they answered 200 to anyone — including with the read budget exhausted — and
    # /openapi.json enumerates every route AND every parameter name, kill-switch
    # `operator`/`reason` included. That is a map of the control plane handed to an
    # unauthenticated caller.
    #
    # The choice made here is to GATE rather than DISABLE: the schema is genuinely
    # useful to the operator, disabling it in "non-development mode" would need a
    # notion of environment this platform deliberately does not have (mode is
    # RESEARCH/PAPER/LIVE, which is a trading concept, not a deployment one), and a
    # disabled-by-flag surface tends to come back on. So the auto-mounts are turned
    # off and re-registered behind exactly the same read budget + bearer token as
    # /status. When no token is configured the API is loopback-only anyway
    # (assert_bind_is_safe), so the docs are then exactly as reachable as the data.
    app = FastAPI(
        title="Trading Engine", version="6.0.0",
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    @app.middleware("http")
    async def _security_telemetry(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        request_id = secrets.token_hex(8)
        request.state.request_id = request_id
        started = time_fn()
        path = _route(request)
        try:
            response: Response = await call_next(request)
        except Exception:
            seclog.emit(
                "request_failed",
                logging.ERROR,
                request_id=request_id,
                **_who(request),
                method=_method(request),
                route=path,
                duration_ms=round((time_fn() - started) * 1000.0, 3),
            )
            raise
        status = int(response.status_code)
        if status >= 500:
            level, outcome = logging.ERROR, "server_error"
        elif status in (401, 403, 429):
            level, outcome = logging.WARNING, "denied"
        elif status >= 400:
            level, outcome = logging.INFO, "client_error"
        else:
            # A liveness probe every few seconds must not drown the trail.
            level = logging.DEBUG if path == "/health" else logging.INFO
            outcome = "ok"
        seclog.emit(
            "request",
            level,
            request_id=request_id,
            **_who(request),
            method=_method(request),
            route=path,
            status_code=status,
            outcome=outcome,
            duration_ms=round((time_fn() - started) * 1000.0, 3),
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        # A self-contained human dashboard at the root, so visiting the base URL shows a
        # readable status page instead of a 404. It reads the JSON endpoints client-side
        # (same-origin fetch) and auto-refreshes — no server state or extra deps.
        return _DASHBOARD_HTML

    @app.get("/openapi.json", include_in_schema=False, dependencies=guard)
    def openapi_schema() -> dict[str, Any]:
        schema: dict[str, Any] = app.openapi()
        return schema

    @app.get("/docs", include_in_schema=False, dependencies=guard,
             response_class=HTMLResponse)
    def swagger_ui() -> Any:
        # The browser then fetches /openapi.json, which needs the same token — so
        # the interactive docs are a loopback/no-token convenience exactly like the
        # dashboard, not an authenticated UI.
        from fastapi.openapi.docs import get_swagger_ui_html

        return get_swagger_ui_html(openapi_url="/openapi.json", title="Trading Engine — API")

    @app.get("/redoc", include_in_schema=False, dependencies=guard,
             response_class=HTMLResponse)
    def redoc_ui() -> Any:
        from fastapi.openapi.docs import get_redoc_html

        return get_redoc_html(openapi_url="/openapi.json", title="Trading Engine — API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return _clean({
            "status": "ok",
            "mode": service.mode,
            "cycle_count": service.cycle_count,
            "last_asof": service.last_asof,
            "broker_connected": service._broker_connected(),
        })

    @app.get("/status", dependencies=guard)
    def status() -> dict[str, Any]:
        return _clean(service.status())

    @app.get("/monitoring", dependencies=guard)
    def monitoring() -> dict[str, Any]:
        return _clean({"snapshot": service.last_snapshot, "alerts": service.last_alerts})

    @app.get("/book", dependencies=guard)
    def book() -> dict[str, Any]:
        last = service.last_result
        return _clean({
            "current_book": service.current_book,
            "target_weights": last.target_weights if last else {},
            "achieved_weights": last.achieved_weights if last else None,
        })

    @app.get("/metrics", dependencies=guard)
    def metrics() -> dict[str, Any]:
        return _clean(service.metrics.snapshot())

    @app.get("/metrics/prometheus", response_class=PlainTextResponse, dependencies=guard)
    def metrics_prometheus() -> str:
        return service.metrics.render_prometheus()

    @app.get("/cycle/latest", dependencies=guard)
    def cycle_latest() -> dict[str, Any]:
        summary = summarize_result(service.last_result)
        if summary is None:
            raise HTTPException(status_code=404, detail="no cycle has run yet")
        return summary

    @app.post("/cycle/run", dependencies=write_guard)
    def cycle_run() -> dict[str, Any]:
        # LIVE submits real orders in STEP 12; the on-demand trigger is an
        # unauthenticated surface, so it is disabled in LIVE — LIVE runs only on
        # the scheduled loop. (Deployment auth/loopback hardening is item 6.)
        if service.mode == "LIVE":
            raise HTTPException(
                status_code=403,
                detail="on-demand cycle trigger is disabled in LIVE; LIVE runs only on the scheduled loop",
            )
        # Re-entrancy guard via the SERVICE lock, so an API trigger cannot overlap
        # a scheduled cycle running in another thread (the engine is stateful).
        try:
            result = service.try_run_once(clock())
        except CycleBusyError:
            raise HTTPException(status_code=409, detail="a cycle is already running")
        summary = summarize_result(result)
        assert summary is not None  # run_once always returns a result
        return summary

    @app.post("/kill-switch/reset", dependencies=write_guard)
    def kill_switch_reset(operator: str, reason: str = "") -> dict[str, Any]:
        # RISK-6 (directive §7.4/§16): re-enabling trading after a hard stop is the
        # most sensitive control action. This API is unauthenticated (auth hardening is
        # item 6), so re-enabling LIVE over it is REFUSED — a LIVE reset must be a
        # deliberate, authenticated console action. PAPER/SHADOW (no live money) may
        # reset a running service here. ``operator`` is required (recorded to the ledger).
        if service.mode == "LIVE":
            raise HTTPException(
                status_code=403,
                detail="kill-switch reset is disabled on the unauthenticated API in LIVE; "
                       "re-enable LIVE via a deliberate authenticated console action (item 6)",
            )
        cleared = service.reset_kill_switch(operator, reason=reason or None, timestamp=clock())
        return _clean({"cleared": cleared, "status": service.status()})

    @app.post("/reconciliation/resolve", dependencies=write_guard)
    def reconciliation_resolve(item_id: str, operator: str, reason: str = "",
                               decision: str = "ACCEPT") -> dict[str, Any]:
        # Resolving a resync-discovered disconnect-fill is a financial-state change (held-book flow):
        # decision=ACCEPT books it into the held book; decision=REJECT discards a spurious/duplicate
        # broker report. This API is unauthenticated (auth hardening is item 6), so resolving in LIVE
        # is REFUSED — a LIVE resolution must be a deliberate, authenticated console action.
        # PAPER/SHADOW (no live money) may resolve here. ``operator`` is required (recorded to the
        # ledger). Returns resolved=False for an unknown/already-closed item or an invalid decision.
        if service.mode == "LIVE":
            raise HTTPException(
                status_code=403,
                detail="reconciliation resolve is disabled on the unauthenticated API in LIVE; "
                       "resolve a LIVE disconnect-fill via a deliberate authenticated console action (item 6)",
            )
        resolved = service.resolve_reconciliation(item_id, operator, reason=reason or None,
                                                  decision=decision, timestamp=clock())
        return _clean({"resolved": resolved, "status": service.status()})

    return app


def security_alert_path(trail_path: "Path") -> "Path":
    """``.../security.jsonl`` -> ``.../security-alerts.jsonl``."""
    return trail_path.with_name(f"{trail_path.stem}-alerts{trail_path.suffix}")


def install_security_logging(settings: Any) -> "SecurityLogHandles":
    """Point the ``tradingengineresearch.api.security`` logger at its TWO durable JSON-lines
    files (SEC-6), defaulting to ``{state_dir}/security.jsonl`` and
    ``{state_dir}/security-alerts.jsonl``.

    Both deployment entry points call this, because the finding was that the
    events only ever reached a console: under ``uvicorn --factory`` the security
    logger has no handler and the root logger's ``lastResort`` is WARNING, so the
    INFO-level ``request`` events were dropped and the ``auth_failed`` /
    ``rate_limited`` ones died with the process.

    **Why two files (SEC-9).** The first version wrote everything to one
    size-capped file, 5 MB x 5. At ~580 bytes per request that is ~52,000 requests
    to rotate the whole trail away — so an attacker could erase the record of their
    own attack simply by continuing it, and the noisy INFO ``request`` lines they
    generate are what does the erasing. The WARNING-and-above events now have their
    own file with time-based retention and repeat aggregation, which no volume of
    requests can roll over. The full trail keeps its size cap and is still allowed
    to roll; that is now a deliberate division of labour rather than a single trail
    that fails at both jobs.

    Returns both handlers (either may be ``None`` when disabled or unopenable —
    losing telemetry must never stop the API serving).

    **It is logging, not alerting.** Nothing pages the operator; it makes an
    ``auth_failed`` burst greppable after the fact. Detection is therefore
    PARTIAL and is recorded as partial in docs/project-control/RISK_AND_DEFECT_REGISTER.md (SEC-6) rather than scored
    as a pass.
    """
    from pathlib import Path as _Path

    from ops.api_security import (
        SecurityLogHandles,
        attach_security_alert_file,
        attach_security_log_file,
    )

    if not getattr(settings, "api_security_log_enabled", True):
        return SecurityLogHandles()
    path = getattr(settings, "api_security_log_path", None)
    if path is None:
        state_dir = _Path(getattr(settings.persistence, "state_dir", "state"))
        path = state_dir / "security.jsonl"
    path = _Path(path)
    alert_path = getattr(settings, "api_security_alert_log_path", None)
    return SecurityLogHandles(
        trail=attach_security_log_file(path),
        alerts=attach_security_alert_file(
            _Path(alert_path) if alert_path is not None else security_alert_path(path)
        ),
    )


def create_app_from_settings() -> Any:  # pragma: no cover - deployment entry point
    """ASGI factory for ``uvicorn --factory ops.api:create_app_from_settings``.

    Builds the service from the process settings (env / ``.env``) and starts it
    (restoring persisted state). Serve the *same* service object with the
    scheduled loop only inside one process — the service-level lock serialises
    them — or run the loop in a separate container (the recommended deployment)."""
    from core.config import get_settings

    settings = get_settings()
    install_security_logging(settings)
    service = EngineService(settings).start()
    return create_app(
        service,
        api_token=settings.api_token,
        rate_limits=RateLimitPolicy.from_settings(settings),
        trusted_proxy_header=settings.api_trusted_proxy_header,
    )
