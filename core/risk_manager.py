"""
TradingEngineResearch — Risk Manager
========================
Pre- and post-trade risk enforcement: graduated drawdown governors, ten
hard kill-switches, a six-scenario stress battery, and PnL attribution.

Safety is Priority 1: a kill-switch trigger halts trading immediately and
requires a manual reset; drawdown governors scale exposure down in graduated
steps. Every check is pure and side-effect-free except for logging, so the same
logic runs identically in RESEARCH, PAPER, and LIVE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RiskSnapshot",
    "KillSwitchStatus",
    "check_drawdown",
    "check_kill_switches",
    "run_stress_tests",
    "decompose_pnl",
    "RiskManager",
    "get_risk_manager",
    "reset_risk_manager",
    "KILL_SWITCH_DEFAULTS",
]

_NAV_RECON_LIMIT = 0.005
_INTRADAY_LOSS_LIMIT = -0.03
_BROKER_REJECTION_RATE = 0.20
_SLIPPAGE_SPIKE_MULT = 3.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── 15.1 / 15.2 Dataclasses ─────────────────────────────────────────────────────

@dataclass
class RiskSnapshot:
    """A point-in-time picture of portfolio risk."""

    gross_exposure: float
    net_exposure: float
    max_single_name_pct: float
    max_sector_pct: float
    beta_exposure: float
    target_vol_utilization: float
    cvar_utilization: float
    drawdown_current: float
    illiquidity_score: float
    kill_switch_active: bool
    hard_stop: bool = False                  # latch-worthy hard stop (kill switch / KILL drawdown)
    active_flags: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_now)


@dataclass
class KillSwitchStatus:
    """The verdict of the kill-switch battery."""

    active: bool
    reason: Optional[str]
    triggered_at: Optional[datetime]
    requires_manual_reset: bool


# ── 15.3 Drawdown governors ──────────────────────────────────────────────────────

def check_drawdown(current_dd: float) -> str:
    """
    Map the current drawdown magnitude to a graduated action level.

      < 5%        → NORMAL   (no action)
      5% – 8%     → SOFT     (reduce new sizing 20%)
      8% – 12%    → MEDIUM   (reduce all exposure 30%, AMBER)
      12% – 15%   → HARD     (reduce all exposure 60%, suspend entries)
      > 15%       → KILL     (halt all trading, RED, manual restart)
    """
    dd = abs(float(current_dd))
    if dd < 0.05:
        return "NORMAL"
    if dd < 0.08:
        return "SOFT"
    if dd < 0.12:
        return "MEDIUM"
    if dd <= 0.15:
        return "HARD"
    return "KILL"


# ── 15.4 Kill switches ───────────────────────────────────────────────────────────

KILL_SWITCH_DEFAULTS: dict = {
    "market_data_age_s": 0.0,
    "market_data_threshold_s": 5.0,
    "feature_age_s": 0.0,
    "feature_threshold_s": 5.0,
    "recent_order_rejections": [],         # list of bool (last <= 10 orders)
    "realized_slippage_bps": 0.0,
    "expected_slippage_bps": 10.0,
    "fill_qty": 0.0,
    "expected_qty": 0.0,
    "fill_qty_tolerance": 0.0,
    "heartbeat_age_s": 0.0,
    "heartbeat_threshold_s": 30.0,
    "nav_reconciliation_error": 0.0,
    "intraday_pnl_pct": 0.0,
    "market_halted": False,
    "internal_exception": False,
}


def check_kill_switches(context: Optional[dict] = None, mode: str = "LIVE") -> KillSwitchStatus:
    """
    Evaluate all ten kill-switch triggers (master prompt Part 15.4).

    Any trigger sets ``active=True`` and ``requires_manual_reset=True``. The
    stale-data triggers apply only in LIVE mode.
    """
    ctx = {**KILL_SWITCH_DEFAULTS, **(context or {})}
    reasons: list[str] = []

    if mode == "LIVE" and ctx["market_data_age_s"] > ctx["market_data_threshold_s"]:
        reasons.append("STALE_MARKET_DATA")
    if mode == "LIVE" and ctx["feature_age_s"] > ctx["feature_threshold_s"]:
        reasons.append("STALE_FEATURES")

    rejections = ctx["recent_order_rejections"]
    if rejections and (sum(bool(x) for x in rejections) / len(rejections)) > _BROKER_REJECTION_RATE:
        reasons.append("BROKER_REJECTION_RATE")

    if ctx["expected_slippage_bps"] > 0 and \
            ctx["realized_slippage_bps"] > _SLIPPAGE_SPIKE_MULT * ctx["expected_slippage_bps"]:
        reasons.append("SLIPPAGE_SPIKE")

    if abs(ctx["fill_qty"] - ctx["expected_qty"]) > ctx["fill_qty_tolerance"]:
        reasons.append("FILL_MISMATCH")

    if ctx["heartbeat_age_s"] > ctx["heartbeat_threshold_s"]:
        reasons.append("MISSING_HEARTBEAT")

    if abs(ctx["nav_reconciliation_error"]) > _NAV_RECON_LIMIT:
        reasons.append("NAV_RECONCILIATION")

    if ctx["intraday_pnl_pct"] < _INTRADAY_LOSS_LIMIT:
        reasons.append("ABNORMAL_LOSS")

    if ctx["market_halted"]:
        reasons.append("MARKET_HALT")

    if ctx["internal_exception"]:
        reasons.append("INTERNAL_EXCEPTION")

    active = bool(reasons)
    if active:
        logger.warning("RISK_EVENT KILL_SWITCH active: %s", "; ".join(reasons))

    return KillSwitchStatus(
        active=active,
        reason="; ".join(reasons) if reasons else None,
        triggered_at=_now() if active else None,
        requires_manual_reset=active,
    )


# ── 15.5 Stress tests ────────────────────────────────────────────────────────────

def run_stress_tests(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    sector_map: Optional[dict] = None,
) -> dict:
    """
    Six stress scenarios; each value is an estimated portfolio loss (negative).
    """
    w = np.asarray(weights, dtype=float).ravel()
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2 or r.shape[1] != w.size:
        raise ValueError("returns_matrix must be (T × n) aligned with weights.")
    n = w.size
    port = r @ w

    # worst_5_days: mean of the five worst realised portfolio days.
    worst_k = np.sort(port)[: min(5, port.size)]
    worst_5 = float(np.mean(worst_k)) if worst_k.size else 0.0

    vols = np.std(r, axis=0, ddof=1) if r.shape[0] > 1 else np.zeros(n)

    # corr_to_1: variance with all pairwise correlations forced to 0.95.
    corr = np.full((n, n), 0.95)
    np.fill_diagonal(corr, 1.0)
    cov_corr = corr * np.outer(vols, vols)
    corr_to_1 = -1.65 * float(np.sqrt(max(w @ cov_corr @ w, 0.0)))

    # vol_x2: double volatilities (4× variance).
    cov_base = np.cov(r, rowvar=False) if r.shape[0] > 1 else np.zeros((n, n))
    vol_x2 = -1.65 * float(np.sqrt(max(w @ (4.0 * np.atleast_2d(cov_base)) @ w, 0.0)))

    # overnight_gap: a -3% gap applied to every position.
    overnight_gap = float(-0.03 * np.sum(w))

    # liquidity_cut: a 50% liquidity haircut → doubled liquidation cost proxy.
    liquidity_cut = float(-0.5 * np.sum(np.abs(w)) * 0.005)

    # sector_shock: -15% to the largest sector (largest single name if no map).
    if sector_map:
        sector_weights: dict[str, float] = {}
        for i, weight in enumerate(w):
            sector_weights.setdefault(sector_map.get(i, "_unmapped"), 0.0)
            sector_weights[sector_map.get(i, "_unmapped")] += weight
        largest = max(sector_weights.values()) if sector_weights else 0.0
    else:
        largest = float(np.max(np.abs(w))) if w.size else 0.0
    sector_shock = float(-0.15 * largest)

    return {
        "worst_5_days": worst_5,
        "corr_to_1": corr_to_1,
        "vol_x2": vol_x2,
        "overnight_gap": overnight_gap,
        "liquidity_cut": liquidity_cut,
        "sector_shock": sector_shock,
    }


# ── 15.6 PnL explain ─────────────────────────────────────────────────────────────

def decompose_pnl(
    pnl: float,
    weights: np.ndarray,
    factor_returns: np.ndarray,
    benchmark_return: float,
) -> dict:
    """
    Attribute realised PnL into interpretable components that sum to ``pnl``.

    ``factor_returns`` is the per-asset realised return vector. Market beta is
    proxied by net exposure; ``signal_alpha`` is the book's return in excess of
    that market exposure; ``residual`` reconciles to realised PnL (costs, timing,
    unexplained). ``factor_contribution`` and ``carry`` require additional inputs
    and are reported as 0.0 here.
    """
    w = np.asarray(weights, dtype=float).ravel()
    fr = np.asarray(factor_returns, dtype=float).ravel()
    net_exposure = float(np.sum(w))
    position_return = float(w @ fr) if fr.size == w.size else 0.0

    beta_contribution = net_exposure * float(benchmark_return)
    signal_alpha = position_return - beta_contribution
    factor_contribution = 0.0
    carry = 0.0
    costs_slippage = 0.0
    residual = float(pnl) - (
        beta_contribution + signal_alpha + factor_contribution + carry + costs_slippage
    )

    return {
        "signal_alpha": signal_alpha,
        "beta_contribution": beta_contribution,
        "factor_contribution": factor_contribution,
        "carry": carry,
        "costs_slippage": costs_slippage,
        "residual": residual,
    }


# ── Risk manager (pre-trade gate + singleton) ────────────────────────────────────

class RiskManager:
    """Builds risk snapshots and runs the pre-trade gate."""

    def check_pretrade(self, weights, market_state: Optional[dict] = None) -> RiskSnapshot:
        """Assemble a RiskSnapshot and run the drawdown + kill-switch gates."""
        state = market_state or {}
        if isinstance(weights, dict):
            w = np.array(list(weights.values()), dtype=float)
        else:
            w = np.asarray(weights, dtype=float).ravel()

        gross = float(np.sum(np.abs(w)))
        net = float(np.sum(w))
        max_single = float(np.max(np.abs(w))) if w.size else 0.0

        betas = state.get("betas")
        beta_exposure = float(w @ np.asarray(betas, dtype=float)) if betas is not None and len(betas) == w.size else net

        max_sector = 0.0
        sector_map = state.get("sector_map")
        if sector_map:
            sectors: dict[str, float] = {}
            for i, weight in enumerate(w):
                key = sector_map.get(i, "_unmapped")
                sectors[key] = sectors.get(key, 0.0) + weight
            max_sector = float(max(sectors.values())) if sectors else 0.0

        target_vol = float(state.get("target_vol", 0.10))
        port_vol = float(state.get("portfolio_vol", 0.0))
        target_vol_util = port_vol / target_vol if target_vol > 0 else 0.0

        cvar = float(state.get("cvar_95", 0.0))
        cvar_limit = float(state.get("cvar_limit", 0.05))
        cvar_util = cvar / cvar_limit if cvar_limit > 0 else 0.0

        drawdown = float(state.get("drawdown_current", 0.0))
        illiquidity = float(state.get("illiquidity_score", 0.0))

        kill = check_kill_switches(state.get("kill_context"), mode=state.get("mode", "LIVE"))
        dd_level = check_drawdown(drawdown)

        flags: list[str] = []
        if dd_level != "NORMAL":
            flags.append(f"DRAWDOWN_{dd_level}")
        if kill.active and kill.reason:
            flags.extend(kill.reason.split("; "))

        # Independent hard-limit enforcement (RISK-1 / directive §16): a SECOND,
        # deterministic line of defence beyond the optimizer's STEP-9 caps — if the
        # optimizer is ever bypassed or buggy, an over-limit book is still blocked here,
        # fail-closed. Back-compatible: only limits EXPLICITLY supplied in market_state
        # are enforced (callers that pass no limit keep the prior behaviour).
        eps = 1e-9
        hard: list[str] = []
        max_pos = state.get("max_position_weight")
        if max_pos is not None and max_single > float(max_pos) + eps:
            hard.append(f"CONCENTRATION_BREACH({max_single:.3f}>{float(max_pos):.3f})")
        max_lev = state.get("max_gross_leverage")
        if max_lev is not None and gross > float(max_lev) + eps:
            hard.append(f"LEVERAGE_BREACH({gross:.3f}>{float(max_lev):.3f})")
        if "cvar_limit" in state and cvar > cvar_limit + eps:
            hard.append(f"CVAR_BREACH({cvar:.4f}>{cvar_limit:.4f})")
        max_sec = state.get("max_sector_weight")
        if max_sec is not None and max_sector > float(max_sec) + eps:
            hard.append(f"SECTOR_BREACH({max_sector:.3f}>{float(max_sec):.3f})")
        flags.extend(hard)

        return RiskSnapshot(
            gross_exposure=gross,
            net_exposure=net,
            max_single_name_pct=max_single,
            max_sector_pct=max_sector,
            beta_exposure=beta_exposure,
            target_vol_utilization=target_vol_util,
            cvar_utilization=cvar_util,
            drawdown_current=drawdown,
            illiquidity_score=illiquidity,
            kill_switch_active=kill.active or dd_level == "KILL" or bool(hard),
            hard_stop=bool(kill.active or dd_level == "KILL"),
            active_flags=flags,
        )


@dataclass
class KillSwitchLatch:
    """Durable, operator-reset latch for hard stops (RISK-6 / directive §7.4 & §16:
    a hard stop requires explicit human re-enable — NO automatic re-enable). Once
    engaged it stays engaged across cycles AND restarts (serialise into durable state)
    until ``reset()`` is called. ``engage()`` with a clean signal is a no-op; only an
    explicit ``reset()`` clears it."""

    latched: bool = False
    reason: Optional[str] = None
    engaged_at: Optional[str] = None
    reset_by: Optional[str] = None
    reset_at: Optional[str] = None

    @property
    def is_latched(self) -> bool:
        return self.latched

    def engage(self, reason: str, timestamp: str) -> bool:
        """Engage on a hard stop. No-op if already latched. Returns True iff newly engaged."""
        if self.latched:
            return False
        self.latched = True
        self.reason = reason
        self.engaged_at = timestamp
        self.reset_by = None
        self.reset_at = None
        return True

    def reset(self, operator: str, timestamp: str) -> bool:
        """Explicit operator re-enable — the ONLY way to clear the latch. Returns True
        iff it was latched."""
        was = self.latched
        self.latched = False
        if was:
            self.reset_by = operator
            self.reset_at = timestamp
        return was

    def to_json(self) -> dict:
        return {"latched": self.latched, "reason": self.reason, "engaged_at": self.engaged_at,
                "reset_by": self.reset_by, "reset_at": self.reset_at}

    @classmethod
    def from_json(cls, data: Optional[dict]) -> "KillSwitchLatch":
        d = data or {}
        return cls(latched=bool(d.get("latched", False)), reason=d.get("reason"),
                   engaged_at=d.get("engaged_at"), reset_by=d.get("reset_by"), reset_at=d.get("reset_at"))


_RISK_MANAGER: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Process-wide RiskManager singleton."""
    global _RISK_MANAGER
    if _RISK_MANAGER is None:
        _RISK_MANAGER = RiskManager()
    return _RISK_MANAGER


def reset_risk_manager() -> None:
    global _RISK_MANAGER
    _RISK_MANAGER = None
