"""
TradingEngineResearch — Scheduled Run-Loop (ROADMAP Phase 6, item 3)
========================================================
The *composition root* and *driver* that nothing else supplied: it builds the
engine, broker, and state store from :class:`core.config.EngineSettings`,
restores persisted learning state, and runs the 13-step pipeline on a schedule —
persisting after every cycle and carrying the book across ticks *and restarts*.

This is the **online** analog of ``backtesting.harness.Backtester`` (the offline
replay). They share one spine — *build PIT inputs → run cycle → carry the book* —
but the loop pulls fresh data, persists durably, reconciles against the broker,
and never weakens mode discipline (golden rule 1): RESEARCH plans no orders,
PAPER submits zero live orders, only LIVE may reach a broker, and only the engine
ever calls ``broker.submit``.

Design: ``docs/specs/2026-06-18-api-runloop-design.md``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from core.config import (
    EngineSettings,
    engine_kwargs,
    get_settings,
    load_vault,
    make_alert_sink,
    make_broker,
    make_state_store,
)
from core.engine.engine import CycleInputs, CycleResult, TradingEngine
from core.risk_manager import KillSwitchLatch
from data.data_contracts import DiscoveredFill, ReconciliationItem, normalize_mode
from ops.ledger import ImmutableLedger, record_cycle, replay_ledger_to_balances
from ops.reconciliation import reconcile
from ops.observability import MetricsRegistry, alert_severity, normalize_alert

logger = logging.getLogger(__name__)

__all__ = [
    "LoopState",
    "EngineService",
    "CycleBusyError",
    "build_cycle_inputs",
    "run_forever",
]


class CycleBusyError(RuntimeError):
    """Raised by :meth:`EngineService.try_run_once` when a cycle is already in
    flight (the non-blocking path the API maps to HTTP 409)."""

# Reconciliation: a whole-share rounding band on the internal-vs-broker position
# check. Divergence beyond this is *surfaced* as an alert, never auto-applied —
# correcting the book from the broker is a deliberate risk decision (non-goal here).
_RECON_SHARE_TOLERANCE = 1.0
# §17 cash leg: absolute per-currency band (one currency unit) on the ledger-replayed
# internal cash vs the broker's reported cash. Same surfacing-only discipline.
_RECON_CASH_TOLERANCE = 1.0

PriceProvider = Callable[[datetime, list], pd.DataFrame]


# ── durable loop state ───────────────────────────────────────────────────────────

@dataclass
class LoopState:
    """The run-loop's own durable state (the book + counters), persisted as a
    small atomic JSON sidecar so a restarted PAPER deployment does not lose its
    book to an empty in-memory broker. The SQL/JSON state store stays focused on
    the registry + performance tracker — this avoids schema churn there."""

    current_book: dict[str, float] = field(default_factory=dict)
    cycle_count: int = 0
    live_orders_total: int = 0
    peak_nav: float = 0.0
    last_asof: Optional[str] = None  # ISO-8601
    # RISK-6 / directive §7.4 & §16: the durable kill-switch latch (serialised
    # KillSwitchLatch). A hard stop must survive restarts and clear only on an
    # explicit operator reset — so it lives in the durable loop state, not memory.
    kill_latch: dict[str, Any] = field(default_factory=dict)
    # LIVE6B-2: the engine's non-terminal LIVE orders, persisted so a restart REMEMBERS an
    # uncertain/working order (never forgets it). Empty off-LIVE / when nothing is pending.
    open_orders: list[dict] = field(default_factory=list)
    # Held-book flow: durable OPEN reconciliation items for resync-discovered disconnect-fills,
    # each timestamped (asof) so they can be aged. An item is booked + closed ONLY by an explicit
    # operator action (resolve_reconciliation) — never auto-applied (directive Section 2/16/17).
    open_reconciliations: list[ReconciliationItem] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LoopState":
        return cls(
            current_book={str(k): float(v) for k, v in (data.get("current_book") or {}).items()},
            cycle_count=int(data.get("cycle_count", 0)),
            live_orders_total=int(data.get("live_orders_total", 0)),
            peak_nav=float(data.get("peak_nav", 0.0)),
            last_asof=data.get("last_asof"),
            kill_latch=dict(data.get("kill_latch") or {}),
            open_orders=list(data.get("open_orders") or []),   # back-compat: old files default []
            open_reconciliations=list(data.get("open_reconciliations") or []),  # back-compat: default []
        )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to ``path`` atomically and durably: a *unique* temp file in the
    same directory (so concurrent writers never collide on a shared name),
    flushed + ``fsync``-ed before the atomic ``os.replace`` (so a power loss
    cannot leave the rename committed over unflushed data), with cleanup on
    failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)  # atomic on POSIX and Windows
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ── input assembly (mirrors backtesting.harness._build_inputs) ────────────────────

def build_cycle_inputs(
    prices: pd.DataFrame,
    asof: datetime,
    symbols: list[str],
    current_weights: dict[str, float],
    capital_gbp: float,
    drawdown: float = 0.0,
) -> CycleInputs:
    """Assemble PIT-safe ``CycleInputs`` from a price history (index ≤ ``asof``,
    columns = ``symbols``). Identical in shape to the backtest harness so the
    online and offline paths exercise the engine the same way."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex.")
    cols = [s for s in symbols if s in prices.columns]
    if not cols:
        raise ValueError("none of the requested symbols are present in the price frame.")
    hist = prices[cols].sort_index()
    rets = hist.pct_change().dropna().to_numpy()
    last = hist.iloc[-1]
    micro = {
        s: {"spread_bps": 6.0, "adv": 2.0e7, "price": float(last[s]), "participation": 0.02}
        for s in cols
    }
    return CycleInputs(
        asof_time=asof,
        symbols=cols,
        prices=hist,
        returns_matrix=rets if rets.size else None,
        portfolio_returns=rets.mean(axis=1) if rets.size else None,
        portfolio_values=(1.0 + rets.mean(axis=1)).cumprod() if rets.size else None,
        current_weights=dict(current_weights),
        capital_gbp=float(capital_gbp),
        drawdown_current=float(drawdown),
        market_microstructure=micro,
    )


def _default_price_provider(asof: datetime, symbols: list[str]) -> pd.DataFrame:
    """Live data feed (network) — a trailing ~400-day daily window up to ``asof``, shaped into
    the WIDE close-price matrix (``DatetimeIndex`` × symbol) that ``build_cycle_inputs`` requires.
    The network ``fetch_prices`` call is not unit-tested (injected in tests); the ``to_wide``
    SHAPING that every cycle depends on IS covered (it was the missing step that left a real LIVE
    run stuck at cycle_count=0)."""
    from datetime import timedelta

    from data.price_ingestion import fetch_prices, ingest_prices, to_wide

    start = (asof - timedelta(days=400)).strftime("%Y-%m-%d")
    end = asof.strftime("%Y-%m-%d")
    tidy = fetch_prices(symbols, start, end)
    # Populate the (in-memory) feature store from the trailing window — STEP-6 fail-closes on a
    # symbol with no feature row in LIVE, so without this the engine can never trade. ingest_prices
    # registers PIT FeatureRows with tz-aware UTC asof stamps (the price features; STEP-6 imputes the
    # non-price ones). NB: re-ingesting each cycle accumulates rows in-memory (fine for a session;
    # get_features picks the latest ≤ asof) — a long-running deployment should dedup/persist them.
    ingest_prices(tidy)
    return to_wide(tidy)


# ── the service (composition root) ────────────────────────────────────────────────

class EngineService:
    """Builds the engine + broker + state store from settings and runs cycles.

    The engine itself is a pure function of ``CycleInputs``; this class supplies
    the clock-free orchestration around it: data in, persistence out, book carried.
    """

    def __init__(
        self,
        settings: Optional[EngineSettings] = None,
        *,
        symbols: Optional[list[str]] = None,
        price_provider: PriceProvider = _default_price_provider,
        state_dir: Optional[Path] = None,
        alert_sink: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.mode = normalize_mode(self.settings.mode)
        self.symbols = list(symbols if symbols is not None else self.settings.universe)
        if not self.symbols:
            # Fail closed: a trading platform must never run a blind universe.
            raise ValueError(
                "EngineService requires a non-empty universe — set ENGINE_UNIVERSE "
                "or pass symbols=[...]. Refusing to run blind."
            )
        self.price_provider = price_provider
        self.capital_gbp = float(self.settings.capital_gbp)

        # LIVE is the only mode that needs the vault (account id / broker secrets).
        vault = load_vault(self.settings) if self.mode == "LIVE" else None
        self.broker = make_broker(self.settings, vault)
        self.engine = TradingEngine(broker=self.broker, **engine_kwargs(self.settings))
        self.state_store = make_state_store(self.settings)

        self.state_dir = Path(state_dir) if state_dir is not None else self.settings.persistence.state_dir
        self._loop_state_path = self.state_dir / "loop_state.json"

        # Observability (item 5): route computed alerts to a sink and track metrics.
        self.alert_sink = alert_sink if alert_sink is not None else make_alert_sink(self.settings)
        self.metrics = MetricsRegistry()
        # Immutable, hash-chained audit trail (Phase 3): every cycle's fills + book +
        # reconciliation are appended here durably (the directive's #1 finish-line item).
        self.ledger = ImmutableLedger(self.state_dir / "ledger.jsonl")
        # RISK-6: durable hard-stop latch (restored from loop state in start()). Once
        # engaged it halts all new cycles until reset_kill_switch() — directive §7.4/§16.
        self.kill_latch = KillSwitchLatch()

        # in-memory snapshot of the latest cycle (served by the API)
        self.last_result: Optional[CycleResult] = None
        self.last_snapshot: dict[str, Any] = {}
        self.last_alerts: list[Any] = []
        self.current_book: dict[str, float] = {}
        self.cycle_count = 0
        self.live_orders_total = 0
        self.peak_nav = 0.0
        # RUN-3: last successfully-measured drawdown, carried forward on a broker NAV-read
        # failure so a transient hiccup never silently disengages the drawdown governor.
        self.last_drawdown = 0.0
        # LIVE6B-1/2: reconnect-resync state. _needs_resync forces a resync (e.g. after a
        # restart that restored uncertain orders) before trusting the lifecycle;
        # _broker_was_connected tracks the disconnect->reconnect edge that forces a resync.
        self._needs_resync = False
        self._broker_was_connected = False
        # Held-book flow: durable OPEN reconciliation items (resync-discovered disconnect-fills),
        # surfaced in status() and resolved ONLY by an explicit operator action.
        self.open_reconciliations: list[ReconciliationItem] = []
        self.last_asof: Optional[datetime] = None
        self._started = False
        # One reentrant lock guards every cycle entry point (run_once,
        # try_run_once, start) so the scheduled driver and an API-triggered
        # cycle can never run the stateful engine concurrently.
        self._lock = threading.RLock()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "EngineService":
        """Idempotent: restore learning state + the durable loop state, and (in
        LIVE) establish the broker session. Safe to call more than once."""
        with self._lock:
            if self._started:
                return self
            try:
                self.state_store.restore()
            except FileNotFoundError:
                logger.info("no persisted learning state found — starting fresh.")
            self._restore_loop_state()
            # LIVE must enter the loop with a live broker session. IBKRBroker does
            # NOT connect lazily — connect() must be called explicitly, else submit()
            # silently returns no fills and the platform is inertly disconnected.
            # Fail closed: refuse to start LIVE if the session cannot be established.
            if self.mode == "LIVE" and self.broker is not None:
                connect = getattr(self.broker, "connect", None)
                if callable(connect):
                    connect()
                if not self._broker_connected():
                    raise RuntimeError(
                        "LIVE run-loop requires a connected broker; broker.connect() did "
                        "not establish a session. Refusing to enter the LIVE loop "
                        "disconnected (fail-closed)."
                    )
            self._started = True
            return self

    def stop(self) -> None:
        """Persist final state and release the broker."""
        self._persist()
        disconnect = getattr(self.broker, "disconnect", None)
        if callable(disconnect):  # pragma: no cover - exercised only with a live broker
            try:
                disconnect()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("broker disconnect failed during stop()")

    # ── one cycle ────────────────────────────────────────────────────────────────

    def run_once(self, asof: Optional[datetime] = None) -> CycleResult:
        """Run exactly one decision cycle and persist. Serialised by the service
        lock so the scheduled driver and an API-triggered cycle never overlap
        (the engine and its process-wide singletons are stateful)."""
        with self._lock:
            return self._run_once_impl(asof)

    def try_run_once(self, asof: Optional[datetime] = None) -> CycleResult:
        """Non-blocking :meth:`run_once`: raise :class:`CycleBusyError` instead of
        waiting if a cycle is already in flight (the API maps this to HTTP 409)."""
        if not self._lock.acquire(blocking=False):
            raise CycleBusyError("a cycle is already running")
        try:
            return self._run_once_impl(asof)
        finally:
            self._lock.release()

    def _run_once_impl(self, asof: Optional[datetime] = None) -> CycleResult:
        if not self._started:
            self.start()
        # In production the clock is tz-aware UTC and the live price index (to_wide) + broker fills
        # are aware too, so the LIVE pipeline is consistently aware (the feature store requires it).
        # Tests pass a naive asof with naive price fixtures — also internally consistent. So do NOT
        # force the asof's awareness here: it must match its price data, which the caller supplies.
        asof = asof or datetime.now(timezone.utc)

        # RISK-6 (fail-closed): if the kill-switch latch is engaged, run NO decision
        # cycle — no data pull, no engine, no orders — until an explicit operator
        # reset_kill_switch(). This is the single choke point, so it also blocks an
        # API-triggered cycle, and it survives restarts (the latch is durable).
        if self.kill_latch.is_latched:
            return self._halted_cycle(asof)

        # LIVE6B-1/3: reconcile pending orders against broker truth BEFORE sizing the cycle.
        self._maybe_resync(asof)
        # LIVE6B-2: until a clean post-restart/reconnect resync clears _needs_resync, the engine
        # STEP-12 gate places NO new LIVE order this cycle (fail-closed).
        self.engine.live_submits_blocked = bool(self._needs_resync)

        prices = self.price_provider(asof, self.symbols)
        drawdown = self._current_drawdown()
        inputs = build_cycle_inputs(
            prices, asof, self.symbols, self.current_book, self.capital_gbp, drawdown
        )

        result = self.engine.run_cycle(inputs)

        # Carry the book exactly as the backtester does: unchanged when blocked;
        # else what was ACHIEVED (fills) in PAPER/LIVE, or the target in RESEARCH.
        if result.blocked:
            new_book = dict(self.current_book)
        else:
            achieved = result.achieved_weights
            src = achieved if (achieved is not None and self.mode != "RESEARCH") else result.target_weights
            new_book = {str(s): float(w) for s, w in (src or {}).items()}
        self.current_book = new_book

        # Persist learning state (registry + tracker) and the durable loop state.
        self.state_store.save(retention_days=self.settings.persistence.retention_days)

        # Refresh served snapshot.
        self.last_result = result
        self.last_snapshot = dict(result.monitoring_snapshot)
        alerts = list(result.alerts)
        self.cycle_count += 1
        self.live_orders_total += int(result.live_orders_submitted)
        self.last_asof = asof

        # Immutable audit trail (Phase 3). Record THIS cycle's fills + book to the ledger BEFORE
        # reconciling, so the §17 internal side (replay_ledger_to_positions) reflects this cycle and
        # is time-aligned with the broker (which already holds this cycle's executions) — otherwise a
        # cycle that trades false-breaks every time. Fail-soft: an audit write must never break a cycle.
        try:
            # Seed the opening cash balance ONCE (first ever event) so the ledger's cash
            # leg is self-contained — replay_ledger_to_balances starts from this deposit.
            if len(self.ledger) == 0:
                self.ledger.append("CASH", {"action": "deposit",
                                            "amount": float(self.capital_gbp), "ccy": "GBP"},
                                   asof.isoformat())
            record_cycle(self.ledger, result, asof.isoformat())   # FILLs + POSITION (RECONCILIATION below)
        except Exception:  # noqa: BLE001 - ledger write must not break the trading cycle
            logger.exception("ledger record_cycle (fills/position) failed")

        # Broker reconciliation (surfacing only) against the NOW-current ledger. Append any divergence
        # as an alert AND an audited RECONCILIATION ledger event (fail-soft). Never auto-corrects (§7.5).
        recon_alert = self._reconcile(asof)
        if recon_alert is not None:
            alerts.append(recon_alert)
            try:
                self.ledger.append("RECONCILIATION", dict(recon_alert), asof.isoformat())
            except Exception:  # noqa: BLE001 - audit write must not break the trading cycle
                logger.exception("ledger RECONCILIATION append failed")

        # RISK-6: engage the durable latch on a genuine hard stop (kill switch active
        # or KILL-level drawdown — set by risk_manager.check_pretrade). The cycle that
        # trips it is already blocked by STEP 10 (no orders); the latch then halts every
        # SUBSEQUENT cycle until an operator reset (no automatic re-enable — §7.4/§16).
        if result.risk_snapshot.get("hard_stop"):
            halt_alert = self._engage_kill_latch(result, asof)
            if halt_alert is not None:
                alerts.append(halt_alert)
        self.last_alerts = alerts

        self._record_observability(result, alerts)
        self._persist()
        return result

    # ── RISK-6: durable kill-switch latch (directive §7.4 & §16) ───────────────────

    def _engage_kill_latch(self, result: CycleResult, asof: datetime) -> Optional[dict[str, Any]]:
        """Engage the durable latch on a hard stop and record it to the immutable
        ledger. Returns the RED alert to surface, or ``None`` if it was already
        latched (no-op). The reason is the cycle's active risk flags."""
        flags = result.risk_snapshot.get("active_flags") or []
        reason = "; ".join(str(f) for f in flags) if flags else "kill switch / KILL drawdown"
        if not self.kill_latch.engage(reason, asof.isoformat()):
            return None
        try:
            self.ledger.append("KILL_SWITCH",
                               {"action": "engage", "reason": reason, "asof": asof.isoformat()},
                               asof.isoformat())
        except Exception:  # noqa: BLE001 - audit write must not break the cycle
            logger.exception("ledger KILL_SWITCH engage append failed")
        logger.error("KILL SWITCH LATCHED at %s — trading halted until operator reset (%s)",
                     asof.isoformat(), reason)
        return {"severity": "RED", "kind": "kill_switch_latched", "asof": asof.isoformat(),
                "message": f"Trading halted — hard stop ({reason}). Operator reset required."}

    def _make_halted_result(self, asof: datetime, reason: str) -> CycleResult:
        """A truthful 'halted' :class:`CycleResult` for a cycle skipped while latched:
        blocked, no orders, book unchanged. Lets the API/monitoring render the halt."""
        return CycleResult(
            mode=self.mode, asof_time=asof, blocked=True,
            regime_label="HALTED", regime_probs={}, crisis={"level": "NONE"},
            execution_regime="HALTED", vol_forecasts={}, signal_scores={},
            predictions={}, decisions={}, optimizer_result={},
            risk_snapshot={"hard_stop": True, "kill_switch_active": True,
                           "active_flags": ["KILL_SWITCH_LATCHED"]},
            target_weights=dict(self.current_book),
            monitoring_snapshot={"halted": True, "reason": reason},
            alerts=[],
        )

    def _halted_cycle(self, asof: datetime) -> CycleResult:
        """Skip a decision cycle because the latch is engaged: emit one RED alert and
        return a halted result. Deliberately lightweight — no engine run, no order
        submission, no book change, no cycle-count/metric inflation, and no per-tick
        ledger append (the engage/reset events already capture the latch transitions)."""
        reason = self.kill_latch.reason or "kill switch latched"
        alert = {"severity": "RED", "kind": "kill_switch_latched", "asof": asof.isoformat(),
                 "message": f"Cycle skipped — trading halted ({reason}). Operator reset required."}
        result = self._make_halted_result(asof, reason)
        self.last_result = result
        self.last_snapshot = dict(result.monitoring_snapshot)
        self.last_alerts = [alert]
        try:
            self.alert_sink.emit(normalize_alert(alert))
        except Exception:  # noqa: BLE001 - observability must not break the halt path
            logger.exception("alert sink emit failed (halted cycle)")
        return result

    def reset_kill_switch(self, operator: str, reason: Optional[str] = None,
                          timestamp: Optional[datetime] = None) -> bool:
        """Operator-only re-enable after a hard stop — the ONLY way to clear the latch
        (directive §7.4/§16: no automatic re-enable). Records the reset to the immutable
        ledger and persists. Serialised by the service lock so it cannot race a cycle.
        Returns True iff the latch was engaged (and is now cleared)."""
        with self._lock:
            ts = (timestamp or datetime.now(timezone.utc)).isoformat()
            was_latched = self.kill_latch.reset(operator, ts)
            if was_latched:
                try:
                    self.ledger.append("KILL_SWITCH",
                                       {"action": "reset", "operator": operator, "reason": reason or ""},
                                       ts)
                except Exception:  # noqa: BLE001 - audit write must not break the reset
                    logger.exception("ledger KILL_SWITCH reset append failed")
                self._persist()
                logger.warning("kill switch RESET by operator=%s reason=%s — trading re-enabled",
                               operator, reason)
            return was_latched

    def resolve_reconciliation(self, item_id: str, operator: str, reason: Optional[str] = None,
                               decision: str = "ACCEPT", timestamp: Optional[datetime] = None) -> bool:
        """Operator-gated resolution of a resync-discovered disconnect-fill (held-book flow).

        ``decision="ACCEPT"`` books the OPEN item ``item_id`` into the held book via an EXPLICIT,
        audited correction (never auto-applied — directive Section 2/16/17): append a signed ``FILL``
        flagged ``source=RESYNC_RECONCILED`` (so the ledger replay picks it up), move ``current_book``
        by the signed weight, advance the lifecycle order out of RECONCILIATION_HOLD (unfreezes its
        symbol), and CLOSE the item. ``decision="REJECT"`` declares the discovered fill spurious/
        duplicate: book NOTHING, audit the rejection, and cancel the parked order out (also unfreezes).

        Hardening (adversarial review): (a) **crash-idempotent** — the FILL is fsync'd before the item
        is closed/persisted, so a crash in between restores the item OPEN with the FILL already in the
        ledger; idempotency is anchored on the DURABLE ledger (``reconciliation_id``), never on the
        volatile item status, so a re-resolve never appends a second FILL (no double-count). (b) **fails
        closed on an invalid price** — a non-positive/non-finite ``ref_price`` would book the share qty
        for zero cost (free shares -> permanent book/NAV divergence), so it is refused and the item left
        OPEN. The audit write comes FIRST: an audit failure leaves the item OPEN (no unaudited change).
        Serialised by the service lock. Returns True iff an OPEN item was found and resolved."""
        with self._lock:
            ts = (timestamp or datetime.now(timezone.utc)).isoformat()
            decision = str(decision or "ACCEPT").upper()
            if decision not in ("ACCEPT", "REJECT"):
                logger.warning("resolve_reconciliation: unknown decision %r — refused.", decision)
                return False
            item = next((i for i in self.open_reconciliations
                         if i.get("id") == item_id and i.get("status") == "OPEN"), None)
            if item is None:
                return False
            order_id = item.get("order_id")
            sym = str(item.get("symbol"))

            if decision == "REJECT":
                # The discovered fill is spurious/duplicate: book NOTHING, audit it, and cancel the
                # parked order out so the symbol unfreezes (no real fill landed).
                cancel_hook = getattr(self.engine, "cancel_reconciled_order", None)
                if callable(cancel_hook):
                    try:
                        cancel_hook(order_id, ts)
                    except Exception:  # noqa: BLE001 - audit-only resolution still stands
                        logger.exception("resolve_reconciliation: cancel_reconciled_order failed")
                item.update({"status": "CLOSED", "decision": "REJECT", "operator": operator,
                             "reason": reason or "", "resolved_asof": ts})
                try:
                    self.ledger.append("RECONCILIATION", {
                        "kind": "disconnect_fill_rejected", "id": item_id, "operator": operator,
                        "reason": reason or "", "symbol": sym}, ts)
                except Exception:  # noqa: BLE001 - audit write must not break the resolution
                    logger.exception("resolve_reconciliation: reject audit append failed")
                self._persist()
                logger.warning("reconciliation REJECTED by operator=%s item=%s symbol=%s",
                               operator, item_id, sym)
                return True

            # decision == ACCEPT
            sign = 1.0 if str(item.get("side", "")).upper() == "BUY" else -1.0
            delta = float(item.get("delta_qty", 0.0) or 0.0)
            # Prefer the broker's TRUE avg fill price (exact cash leg); fall back to the placement
            # ref_price (flagged price_estimated) only when the broker did not report one.
            broker_price = item.get("avg_fill_price")
            if broker_price is not None and math.isfinite(float(broker_price)) and float(broker_price) > 0.0:
                price, price_estimated = float(broker_price), False
            else:
                price, price_estimated = float(item.get("ref_price", 0.0) or 0.0), True
            # Fail closed: refuse to book at a non-positive/non-finite price — it would record the
            # share quantity for ZERO cost (free shares -> permanent book/NAV divergence). Leave OPEN.
            if not math.isfinite(price) or price <= 0.0:
                logger.warning("RISK_EVENT AMBER: resolve_reconciliation refused — item %s has a "
                               "non-positive/non-finite ref_price (%r); left OPEN.", item_id, price)
                try:
                    self.alert_sink.emit(normalize_alert({
                        "severity": "AMBER", "kind": "reconciliation", "asof": ts,
                        "message": (f"Reconciliation {item_id} cannot be booked: invalid price "
                                    f"{price}; item left OPEN for a priced resolution.")}))
                except Exception:  # noqa: BLE001 - observability must not break the cycle
                    logger.exception("alert sink emit failed (resolve invalid price)")
                return False
            signed_qty = sign * delta
            # Crash-idempotency: anchor on the DURABLE ledger (not the volatile item status). If a
            # RESYNC_RECONCILED FILL for THIS item already exists (a crash after the FILL append but
            # before the close/persist), do NOT append a second one — replay would double-count.
            already_booked = any(e.payload.get("reconciliation_id") == item_id
                                 for e in self.ledger.events("FILL"))
            if not already_booked:
                # Audited correction FIRST — if the audit write fails, do NOT mutate the held book.
                try:
                    self.ledger.append("FILL", {
                        "source": "RESYNC_RECONCILED", "reconciliation_id": item_id,
                        "order_id": order_id, "symbol": sym, "side": item.get("side"),
                        "qty": delta, "signed_qty": signed_qty, "fill_price": price,
                        "price_estimated": price_estimated}, ts)
                except Exception:  # noqa: BLE001 - no unaudited book mutation
                    logger.exception("resolve_reconciliation: FILL audit append failed; item stays OPEN")
                    return False
            # Apply the held-book delta. In the only reachable already_booked path (crash before
            # persist) the delta was lost with the persist, so it must still be (re)applied here;
            # a normal double-call cannot reach this (a CLOSED item fails the OPEN guard above).
            capital = self.capital_gbp if self.capital_gbp > 0 else 1.0
            self.current_book[sym] = self.current_book.get(sym, 0.0) + signed_qty * price / capital
            # Advance the lifecycle order out of RECONCILIATION_HOLD (unfreezes the symbol).
            book_hook = getattr(self.engine, "book_reconciled_fill", None)
            if callable(book_hook):
                try:
                    book_hook(order_id, ts)
                except Exception:  # noqa: BLE001 - the audited book change already stands
                    logger.exception("resolve_reconciliation: book_reconciled_fill failed")
            item.update({"status": "CLOSED", "decision": "ACCEPT", "operator": operator,
                         "reason": reason or "", "resolved_asof": ts})
            try:
                self.ledger.append("RECONCILIATION", {
                    "kind": "disconnect_fill_resolved", "id": item_id, "operator": operator,
                    "reason": reason or "", "symbol": sym, "signed_qty": signed_qty}, ts)
            except Exception:  # noqa: BLE001 - audit write must not break the resolution
                logger.exception("resolve_reconciliation: resolve audit append failed")
            self._persist()
            logger.warning("reconciliation RESOLVED by operator=%s item=%s symbol=%s signed_qty=%s",
                           operator, item_id, sym, signed_qty)
            return True

    def _record_observability(self, result: CycleResult, alerts: list) -> None:
        """Emit each alert to the configured sink and update cycle metrics. Fully
        fail-soft: an observability failure never breaks the trading cycle."""
        for alert in alerts:
            try:
                self.alert_sink.emit(normalize_alert(alert))
            except Exception:  # noqa: BLE001 - observability must not break the cycle
                logger.exception("alert sink emit failed")
        try:
            m = self.metrics
            m.inc("engine_cycles_total")
            if result.blocked:
                m.inc("engine_blocked_cycles_total")
            m.set_gauge("engine_cycle_count", self.cycle_count)
            m.set_gauge("engine_live_orders_total", self.live_orders_total)
            m.set_gauge("engine_book_size", float(len(self.current_book)))
            for alert in alerts:
                m.inc("engine_alerts_total", severity=alert_severity(alert))
        except Exception:  # noqa: BLE001 - metrics are best-effort
            logger.exception("metrics update failed")

    # ── reconciliation, drawdown, persistence helpers ─────────────────────────────

    def _maybe_resync(self, asof: datetime) -> None:
        """LIVE-only reconnect resync (LIVE6B-1/3): resolve uncertain/unknown/resting orders
        against the broker's open-order truth BEFORE the engine sizes the next delta. Triggers
        on a disconnect->reconnect edge, while any order is pending, or while a restart flagged
        ``_needs_resync``. READ-ONLY (``open_orders`` never submits). Fail-soft: a broker-read
        failure leaves ``_needs_resync`` set and surfaces an AMBER alert; it never crashes the
        cycle."""
        if self.mode != "LIVE" or self.broker is None:
            return
        connected = bool(self._broker_connected())
        edge = connected and not self._broker_was_connected
        self._broker_was_connected = connected
        pending = bool(getattr(self.engine, "has_pending_orders", lambda: False)())
        if not (edge or pending or self._needs_resync) or not connected:
            return
        try:
            self.engine.resync_open_orders(self.broker.open_orders(asof), asof.isoformat())
            self._needs_resync = False
            # Held-book flow: surface any disconnect-fill the resync discovered (broker filled
            # more than we locally booked) as a durable OPEN reconciliation item. NEVER booked
            # here — an explicit operator resolve_reconciliation() books it (directive Section 2/17).
            drain = getattr(self.engine, "drain_discovered_fills", None)
            if callable(drain):
                for item in drain():
                    self._surface_discovered_fill(item, asof)
        except Exception:  # noqa: BLE001 - a resync failure must not crash the cycle
            self._needs_resync = True
            logger.exception("reconnect resync failed; LIVE submits remain gated until it succeeds")
            try:
                self.alert_sink.emit(normalize_alert({
                    "severity": "AMBER", "kind": "resync_failed", "asof": asof.isoformat(),
                    "message": "Reconnect resync failed; pending orders unresolved."}))
            except Exception:  # noqa: BLE001 - observability must not break the cycle
                logger.exception("alert sink emit failed (resync)")

    def _surface_discovered_fill(self, item: DiscoveredFill, asof: datetime) -> None:
        """Record a resync-discovered disconnect-fill as a durable OPEN reconciliation item
        (held-book flow): append to ``open_reconciliations`` (idempotent by id), audit a
        ``RECONCILIATION`` ledger event, and emit an AMBER alert. The symbol stays frozen
        (the order is parked RECONCILIATION_HOLD); booking is operator-gated — nothing is
        applied to the held book here (directive Section 2/16/17). Fail-soft."""
        item_id = f"{item.get('order_id')}|{item.get('broker_filled_qty')}"
        if any(i.get("id") == item_id for i in self.open_reconciliations):
            return                                            # already surfaced — no duplicate
        record: ReconciliationItem = {
            "id": item_id, "order_id": item["order_id"], "symbol": item["symbol"],
            "side": item["side"], "delta_qty": float(item.get("delta_qty", 0.0) or 0.0),
            "broker_filled_qty": float(item.get("broker_filled_qty", 0.0) or 0.0),
            "ref_price": float(item.get("ref_price", 0.0) or 0.0),
            # The broker's TRUE avg fill price when reported (preferred over ref_price at booking);
            # None -> resolve falls back to ref_price (flagged price_estimated).
            "avg_fill_price": item.get("avg_fill_price"),
            "asof": asof.isoformat(), "status": "OPEN",
        }
        self.open_reconciliations.append(record)
        try:
            self.ledger.append("RECONCILIATION", {"kind": "disconnect_fill_discovered", **record},
                               asof.isoformat())
        except Exception:  # noqa: BLE001 - audit write must not break the cycle
            logger.exception("ledger RECONCILIATION (disconnect_fill_discovered) append failed")
        try:
            self.alert_sink.emit(normalize_alert({
                "severity": "AMBER", "kind": "reconciliation", "asof": asof.isoformat(),
                "message": (f"Disconnect-fill discovered for {record['symbol']} "
                            f"(order {record['order_id']}, delta {record['delta_qty']}); "
                            f"symbol frozen pending operator reconciliation.")}))
        except Exception:  # noqa: BLE001 - observability must not break the cycle
            logger.exception("alert sink emit failed (disconnect_fill_discovered)")

    def _reconcile(self, asof: datetime) -> Optional[dict[str, Any]]:
        """Reconcile the directive §17 AUTHORITATIVE internal side — the immutable
        event-ledger book (``replay_ledger_to_balances``: the signed sum of every recorded
        FILL, plus cash = deposit − trade flows − recorded COMMISSION/FEE) — against the
        broker's positions AND cash, and surface (never correct) any divergence. No-op
        without a broker (RESEARCH) and for a paper simulation.

        Flat-start assumption: the ledger replay is the true internal book ONLY because the
        deployment starts FLAT (cash only) and every position change is a recorded signed
        FILL; likewise the cash leg is true ONLY if the seeded opening deposit
        (``capital_gbp``) equals the account's actual starting cash — a mismatch there is a
        REAL config-vs-account divergence and is meant to surface. A future non-flat
        deployment MUST first seed audited one-time opening-position/cash events, or the
        replay will under-report. Positions compare signed shares directly; cash compares
        single-bucket account-currency amounts (the broker's ``cash_gbp`` carries the account
        currency — USD on Alpaca — exactly as ``capital_gbp`` seeded it). The cash leg is
        SKIPPED (with a structured log line, never a false break) when the broker reports no
        cash, when no opening CASH deposit exists yet, or when the replay is incomplete.
        NAV leg: deliberately DEFERRED — an internal NAV needs current position marks, which
        this recon context does not cheaply have (comparing the broker's NAV against our
        cash+marks belongs with a pricing source; the positions+cash legs already triangulate
        it). Surfacing-only: a break NEVER auto-corrects the book (directive §7.5)."""
        if self.broker is None:
            return None
        if getattr(self.broker, "is_paper", False):
            # PAPER fills are simulated inside the engine (STEP 12 → _simulate_fills),
            # so the PaperBroker never sees them and its positions stay empty — it is
            # not an authoritative book to reconcile against. Reconciliation is
            # meaningful only against a broker the engine actually submits through.
            return None
        try:
            bs = self.broker.account_state(asof)
        except Exception:  # noqa: BLE001 - a broker query failure must not break the cycle
            logger.exception("broker.account_state failed during reconciliation")
            return {"severity": "AMBER", "kind": "reconciliation",
                    "message": "broker account_state query failed"}
        nav = float(bs.nav_gbp) if bs.nav_gbp else self.capital_gbp
        if nav > self.peak_nav:
            self.peak_nav = nav
        balances = replay_ledger_to_balances(self.ledger)           # §17 legs (signed shares + cash)
        broker_shares = {str(k): float(v) for k, v in dict(bs.positions).items()}
        internal: dict[str, Any] = {"positions": balances["positions"]}
        broker: dict[str, Any] = {"positions": broker_shares}
        # §17 cash leg — included only when BOTH sides are reliable: a false cash break every
        # cycle would train the operator to ignore reconciliation alerts (worse than no check).
        if bs.cash_gbp is None:
            logger.info("reconciliation cash leg skipped: broker %s reports no cash (asof=%s).",
                        getattr(bs, "broker", "?"), asof.isoformat())
        elif not self.ledger.events("CASH"):
            logger.warning("reconciliation cash leg skipped: no opening CASH deposit in the "
                           "ledger yet — internal cash has no baseline (asof=%s).", asof.isoformat())
        elif not balances.get("cash_complete", False):
            logger.warning("RISK_EVENT AMBER: reconciliation cash leg skipped: the trail holds "
                           "FILLs missing sign/price, so the replayed cash is unreliable "
                           "(asof=%s).", asof.isoformat())
        else:
            internal["cash"] = balances["cash"]
            broker["cash"] = {"GBP": float(bs.cash_gbp)}            # same single-bucket key as replay
        # Structured reconciliation; surfaces breaks as an alert (and, via the caller, an audited
        # RECONCILIATION ledger event); never auto-corrects the book from the broker.
        report = reconcile(internal, broker, asof=asof.isoformat(),
                           share_tol=_RECON_SHARE_TOLERANCE, cash_tol=_RECON_CASH_TOLERANCE)
        return report.to_alert(mode=self.mode)

    def _current_drawdown(self) -> float:
        """Drawdown from peak NAV when a broker reports NAV; 0.0 with no broker
        (RESEARCH has no real book). Peak is durable across restarts.

        RUN-3 (fail-safe, not fail-OPEN): a broker NAV-read failure must NOT reset the
        drawdown to a permissive 0.0 — that would silently disengage the drawdown
        governor exactly when the broker is misbehaving. Instead we carry the
        last-known drawdown and log loudly, so the governor stays engaged and the
        failure is surfaced (it is also raised as an alert in ``_reconcile``)."""
        if self.broker is None:
            return 0.0
        try:
            nav = self.broker.account_state(datetime.now(timezone.utc)).nav_gbp
        except Exception:  # noqa: BLE001 - a hiccup must not silently disable the governor
            logger.warning("RISK_EVENT AMBER: broker NAV read failed in _current_drawdown; "
                           "carrying last-known drawdown=%.4f (governor stays engaged).",
                           self.last_drawdown)
            return self.last_drawdown
        if not nav or nav <= 0:
            # A zero/negative NAV read is itself suspect — do not treat it as 'no drawdown'.
            logger.warning("RISK_EVENT AMBER: broker reported non-positive NAV (%r); "
                           "carrying last-known drawdown=%.4f.", nav, self.last_drawdown)
            return self.last_drawdown
        if nav > self.peak_nav:
            self.peak_nav = float(nav)
        dd = 0.0 if self.peak_nav <= 0 else max(0.0, 1.0 - float(nav) / self.peak_nav)
        self.last_drawdown = dd
        return dd

    def _restore_loop_state(self) -> None:
        if not self._loop_state_path.exists():
            return
        try:
            data = json.loads(self._loop_state_path.read_text(encoding="utf-8"))
            ls = LoopState.from_json(data)
        except (OSError, ValueError, TypeError):
            logger.exception("could not read loop state at %s — starting fresh", self._loop_state_path)
            return
        self.current_book = ls.current_book
        self.cycle_count = ls.cycle_count
        self.live_orders_total = ls.live_orders_total
        self.peak_nav = ls.peak_nav
        # Held-book flow: durable OPEN reconciliation items survive a restart (an unresolved
        # disconnect-fill must not be forgotten — the operator still needs to resolve it).
        self.open_reconciliations = list(ls.open_reconciliations)
        # RISK-6: a hard stop must survive restarts — restore the latch so a crashed/
        # restarted deployment re-enters HALTED, not trading.
        self.kill_latch = KillSwitchLatch.from_json(ls.kill_latch)
        # LIVE6B-2: REMEMBER uncertain/working LIVE orders across a restart and force a resync
        # before trusting/extending them (fail-closed: the submit gate stays on until resync).
        if self.mode == "LIVE" and ls.open_orders:
            restore = getattr(self.engine, "restore_open_orders", None)
            if callable(restore):
                restore(ls.open_orders)
            self._needs_resync = True
        if ls.last_asof:
            try:
                self.last_asof = datetime.fromisoformat(ls.last_asof)
            except ValueError:
                self.last_asof = None

    def _snapshot_open_orders(self) -> list:
        """The engine's non-terminal LIVE orders to persist (LIVE6B-2). Guarded so a fake
        engine (or one without the hook) yields [] and never breaks persistence."""
        snap = getattr(self.engine, "snapshot_open_orders", None)
        if not callable(snap):
            return []
        try:
            return list(snap())
        except Exception:  # noqa: BLE001 - persistence must not break the cycle
            logger.exception("snapshot_open_orders failed")
            return []

    def _persist(self) -> None:
        ls = LoopState(
            current_book=self.current_book,
            cycle_count=self.cycle_count,
            live_orders_total=self.live_orders_total,
            peak_nav=self.peak_nav,
            last_asof=self.last_asof.isoformat() if self.last_asof else None,
            kill_latch=self.kill_latch.to_json(),
            open_orders=self._snapshot_open_orders(),
            open_reconciliations=list(self.open_reconciliations),
        )
        _atomic_write_json(self._loop_state_path, ls.to_json())

    # ── served views (consumed by ops.api) ────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "capital_gbp": self.capital_gbp,
            "universe": list(self.symbols),
            "cycle_count": self.cycle_count,
            "last_asof": self.last_asof.isoformat() if self.last_asof else None,
            "blocked": bool(self.last_result.blocked) if self.last_result else None,
            "live_orders_total": self.live_orders_total,
            "broker_connected": self._broker_connected(),
            # RISK-6: operators must be able to see (and act on) a hard stop.
            "kill_switch_latched": self.kill_latch.is_latched,
            "kill_switch_reason": self.kill_latch.reason,
            # Held-book flow: OPEN reconciliation items operators must see + resolve (aged by asof).
            "open_reconciliations": [i for i in self.open_reconciliations if i.get("status") == "OPEN"],
        }

    def _broker_connected(self) -> Optional[bool]:
        if self.broker is None:
            return None
        try:
            return bool(self.broker.connected)
        except Exception:  # noqa: BLE001
            return False


# ── the scheduling shell ──────────────────────────────────────────────────────────

def run_forever(
    service: EngineService,
    *,
    interval_seconds: Optional[float] = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    max_cycles: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Drive ``service.run_once`` on a fixed cadence. The loop body is fully
    injectable (``clock``/``max_cycles``/``should_stop``/``sleep``) so it is unit
    tested; the operator entry point below uses the wall-clock defaults.

    A per-cycle exception is logged and the loop continues — a transient data or
    broker error must not kill a long-running platform. ``KeyboardInterrupt`` and
    ``SystemExit`` propagate (clean shutdown). Returns the number of cycles run.
    """
    interval = float(interval_seconds if interval_seconds is not None else service.settings.cycle_interval_seconds)
    service.start()
    n = 0
    while True:
        if should_stop is not None and should_stop():
            break
        if max_cycles is not None and n >= max_cycles:
            break
        try:
            service.run_once(clock())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - a cycle failure must not stop the platform
            logger.exception("run-loop cycle %d failed; continuing", n)
        n += 1
        if max_cycles is not None and n >= max_cycles:
            break
        sleep(interval)
    return n


def build_api_server(service: EngineService, *, host: str, port: int) -> Any:
    """Construct the uvicorn server for ``service`` exactly as a deployment serves it.

    Extracted from :func:`serve_combined` so the *shipped* server construction is
    reachable from a test. It was not, and that is how the trusted-proxy control
    came to be bypassed at the server layer while every application-level test
    passed: the tests built the app with ``TestClient`` and so never saw
    ``uvicorn.run``'s ``proxy_headers=True`` default rewriting the client address
    in front of it. ``tests/test_api_proxy_headers.py`` drives THIS function
    against a real socket.

    The proxy-header settings come from :func:`ops.api.api_uvicorn_kwargs`, which
    is where the reasoning lives; they are not spelled out again here, so there is
    one place to change and no second copy to drift.
    """
    import uvicorn

    from ops.api import api_uvicorn_kwargs, assert_bind_is_safe, create_app
    from ops.api_security import RateLimitPolicy

    # Phase 8: if it IS bound off-loopback, it must be authenticated. Checked
    # before the socket opens, so the unsafe combination never serves a request.
    token = getattr(service.settings, "api_token", None)
    assert_bind_is_safe(host, token)
    app = create_app(
        service, api_token=token,
        rate_limits=RateLimitPolicy.from_settings(service.settings),
        trusted_proxy_header=getattr(service.settings, "api_trusted_proxy_header", None),
    )
    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, **api_uvicorn_kwargs()))


def serve_combined() -> None:  # pragma: no cover - deploy entry (loop thread + API, one process)
    """Run the scheduled loop AND serve the API over ONE shared service in a single
    process — the recommended single-container deployment. Sharing one service (and
    its lock) means there is exactly one writer to the state dir, and POST
    /cycle/run can never overlap a scheduled cycle. Use this rather than two
    containers pointed at the same state volume."""
    from ops.api import install_security_logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = EngineService().start()
    # SEC-6/SEC-9: the durable security trail AND the separately-retained alert
    # trail, both before the socket opens, so the very first auth_failed is on disk
    # and not only on a console nobody kept.
    install_security_logging(service.settings)
    logger.info("TradingEngineResearch combined (loop+API) starting in %s mode over %d symbols",
                service.mode, len(service.symbols))
    loop_thread = threading.Thread(target=run_forever, args=(service,), daemon=True)
    loop_thread.start()
    # RUN-5 (secure by default): a DIRECT run binds LOOPBACK only, so the control
    # API is never exposed on all interfaces by accident. A container deploy sets
    # ENGINE_API_HOST=0.0.0.0 explicitly (compose publishes the port to host loopback only).
    host = os.environ.get("ENGINE_API_HOST", "127.0.0.1")
    port = int(os.environ.get("ENGINE_API_PORT", "8000"))
    build_api_server(service, host=host, port=port).run()


def main() -> None:  # pragma: no cover - operator entry point (wall-clock loop)
    """``python -m ops.run_loop`` — start the scheduled loop from settings.
    ``python -m ops.run_loop --serve`` — combined loop + API in one process."""
    import sys

    if "--serve" in sys.argv[1:]:
        serve_combined()
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = EngineService().start()
    logger.info("TradingEngineResearch run-loop starting in %s mode over %d symbols", service.mode, len(service.symbols))
    try:
        run_forever(service)
    except (KeyboardInterrupt, SystemExit):
        logger.info("run-loop stopping")
    finally:
        service.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
