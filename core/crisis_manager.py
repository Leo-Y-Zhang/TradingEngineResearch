"""
TradingEngineResearch — Crisis Manager
==========================
The composite, probabilistic crisis scorer.

Seven detectors each return ``tuple[bool, float]`` — a hard-threshold fire flag
and a continuous severity score in ``[0, 1]``. A single weighted composite
score ``S`` (the 7-weight formula from the master prompt) maps to a graduated
crisis level:

    S < 0.20            → NORMAL
    0.20 <= S < 0.50    → ELEVATED
    0.50 <= S < 0.75    → CRISIS
    S >= 0.75           → CRITICAL

``defensive_mode`` is ``True`` only at CRISIS or CRITICAL. This continuous,
graduated model supersedes any earlier binary "≥2 detectors fired" voting
scheme. Results are cached for five minutes; ``get_crisis_manager()`` is the only
entry point and returns a process-wide singleton.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from strategies.volatility_model import vol_ratio_current

logger = logging.getLogger(__name__)

__all__ = [
    "CrisisLevel",
    "CrisisStatus",
    "CrisisManager",
    "level_from_severity",
    "crisis_scalars",
    "get_crisis_manager",
    "reset_crisis_manager",
]


class CrisisLevel(str, Enum):
    """Graduated crisis levels (ordered NORMAL → CRITICAL)."""

    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"
    CRITICAL = "CRITICAL"


# ── Composite weights (master prompt Part 9.2 — sum to 1.0) ─────────────────────

_W_CORR: float = 0.22
_W_VOL: float = 0.22
_W_DD: float = 0.16
_W_BREADTH: float = 0.15
_W_LIQ: float = 0.10
_W_GAP: float = 0.10
_W_EVENT: float = 0.05

# Regime-aware drawdown thresholds.
_DRAWDOWN_THRESHOLDS: dict[str, float] = {
    "mean_reverting": 0.03,
    "trending": 0.05,
    "high_vol": 0.075,
}

_DEFAULT_CACHE_TTL_SECONDS: float = 300.0  # 5-minute cache on assess()

# Ordered detector names — index-aligned with the assess() detector calls.
_DETECTOR_NAMES: tuple[str, ...] = (
    "correlation_spike",
    "vol_explosion",
    "drawdown_acceleration",
    "breadth_collapse",
    "liquidity_stress",
    "gap_risk",
    "event_risk",
)


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def level_from_severity(score: float) -> CrisisLevel:
    """Map a composite severity score in [0, 1] to a CrisisLevel."""
    if score < 0.20:
        return CrisisLevel.NORMAL
    if score < 0.50:
        return CrisisLevel.ELEVATED
    if score < 0.75:
        return CrisisLevel.CRISIS
    return CrisisLevel.CRITICAL


# P4 graduated risk-overlay scalars (upgrade-spec SPEC P4, lines 403-412).
# Map the continuous CrisisComposite to (vol_target_scalar, cvar_limit_scalar):
#   S < 0.35   Normal     (1.00, 1.00)   — no modification
#   S < 0.60   Elevated   (0.80, 0.85)
#   S < 0.80   Defensive  (0.60, 0.65)
#   S >= 0.80  Crisis     (0.50, 0.50)
# NOTE: these P4 bands (0.35/0.60/0.80) are deliberately distinct from the
# CrisisLevel thresholds (0.20/0.50/0.75); the scalars key off the raw severity.
_CRISIS_SCALAR_BANDS: tuple[tuple[float, float, float], ...] = (
    (0.35, 1.00, 1.00),
    (0.60, 0.80, 0.85),
    (0.80, 0.60, 0.65),
)
_CRISIS_SCALAR_FLOOR: tuple[float, float] = (0.50, 0.50)


def crisis_scalars(severity: float) -> tuple[float, float]:
    """Continuous severity → ``(vol_target_scalar, cvar_limit_scalar)`` (P4).

    Both multipliers are non-increasing in ``severity`` — higher severity tightens
    the base vol target and CVaR limit. Severity is clipped to ``[0, 1]``.
    """
    s = _clip01(severity)
    for upper, vol_scalar, cvar_scalar in _CRISIS_SCALAR_BANDS:
        if s < upper:
            return (vol_scalar, cvar_scalar)
    return _CRISIS_SCALAR_FLOOR


# ── CrisisStatus ────────────────────────────────────────────────────────────────

@dataclass
class CrisisStatus:
    """The result of a crisis assessment for a single cycle."""

    level: CrisisLevel
    defensive_mode: bool
    signals_fired: list[str]
    signal_values: dict[str, float]
    severity_score: float
    liquidity_stress_score: float
    gap_risk_score: float
    event_risk_score: float
    timestamp: float

    def as_dict(self) -> dict:
        """Serialise to a plain dict (level rendered as its string value)."""
        return {
            "level": self.level.value,
            "defensive_mode": self.defensive_mode,
            "signals_fired": list(self.signals_fired),
            "signal_values": dict(self.signal_values),
            "severity_score": self.severity_score,
            "liquidity_stress_score": self.liquidity_stress_score,
            "gap_risk_score": self.gap_risk_score,
            "event_risk_score": self.event_risk_score,
            "timestamp": self.timestamp,
        }


# ── Crisis manager ──────────────────────────────────────────────────────────────

class CrisisManager:
    """Runs the seven detectors, composites them, and caches the result."""

    def __init__(self, cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS) -> None:
        self._cache_ttl = float(cache_ttl_seconds)
        self._cache: CrisisStatus | None = None
        self._cache_time: float = 0.0
        self._last_status: CrisisStatus | None = None
        # Volatility params used by the vol-explosion detector (set per assess()).
        self._gjr_params: dict | None = None
        self._har_params: dict | None = None

    # -- 9.1 Detectors (all return tuple[bool, float]) ---------------------------

    def _detect_correlation_spike(self, returns_matrix) -> tuple[bool, float]:
        """EWMA cross-asset correlation spike (span=30, min_periods=10)."""
        if returns_matrix is None:
            return (False, 0.0)
        arr = np.asarray(returns_matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 10:
            return (False, 0.0)

        df = pd.DataFrame(arr)
        try:
            ewm_cov = df.ewm(span=30, min_periods=10).cov()
            last_cov = ewm_cov.loc[df.index[-1]].to_numpy()
        except Exception as exc:  # noqa: BLE001 — malformed input → no detection
            logger.warning("Correlation-spike detector skipped (%s).", exc)
            return (False, 0.0)

        diag = np.sqrt(np.clip(np.diag(last_cov), 0.0, None))
        denom = np.outer(diag, diag)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0.0, last_cov / denom, np.nan)

        iu = np.triu_indices(corr.shape[0], k=1)
        vals = corr[iu]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (False, 0.0)

        rho_mean = float(np.mean(vals))
        return (rho_mean >= 0.70, _clip01(rho_mean / 0.70))

    def _detect_vol_explosion(self, portfolio_returns) -> tuple[bool, float]:
        """Realised-vol explosion via the shared vol_ratio_current() helper."""
        if portfolio_returns is None:
            return (False, 0.0)
        vol_ratio = vol_ratio_current(
            portfolio_returns, self._gjr_params, self._har_params
        )
        return (vol_ratio >= 2.0, _clip01((vol_ratio - 1.0) / 1.0))

    def _detect_drawdown_acceleration(
        self, portfolio_values, current_regime: str = "mean_reverting"
    ) -> tuple[bool, float]:
        """Regime-aware drawdown breach (threshold scales with regime)."""
        threshold = _DRAWDOWN_THRESHOLDS.get(
            current_regime, _DRAWDOWN_THRESHOLDS["mean_reverting"]
        )
        if portfolio_values is None:
            return (False, 0.0)
        v = np.asarray(portfolio_values, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size < 2:
            return (False, 0.0)

        peak = float(np.max(v))
        current = float(v[-1])
        if peak <= 0.0:
            return (False, 0.0)

        dd_pct = max((peak - current) / peak, 0.0)
        s_dd = _clip01(dd_pct / threshold) if threshold > 0.0 else 0.0
        return (dd_pct >= threshold, s_dd)

    def _detect_breadth_collapse(self, position_pnls) -> tuple[bool, float]:
        """Fraction of positions losing money."""
        if position_pnls is None:
            return (False, 0.0)
        p = np.asarray(position_pnls, dtype=float).ravel()
        p = p[np.isfinite(p)]
        if p.size == 0:
            return (False, 0.0)

        pct_losing = float(np.mean(p < 0.0))
        return (pct_losing >= 0.70, _clip01((pct_losing - 0.50) / 0.20))

    def _detect_liquidity_stress(
        self, spread_bps: float, adv_ratio: float
    ) -> tuple[bool, float]:
        """Bid/ask spread blow-out or excessive ADV participation."""
        spread_bps = float(spread_bps)
        adv_ratio = float(adv_ratio)
        s_liq = _clip01(max(spread_bps / 25.0, adv_ratio / 0.05) - 1.0)
        return (spread_bps > 25.0 or adv_ratio > 0.05, s_liq)

    def _detect_gap_risk(self, overnight_gaps) -> tuple[bool, float]:
        """Largest absolute overnight gap."""
        if overnight_gaps is None:
            return (False, 0.0)
        g = np.asarray(overnight_gaps, dtype=float).ravel()
        g = g[np.isfinite(g)]
        if g.size == 0:
            return (False, 0.0)

        max_abs = float(np.max(np.abs(g)))
        return (max_abs > 0.03, _clip01(max_abs / 0.03))

    def _detect_event_risk(self, hours_to_event: float | None) -> tuple[bool, float]:
        """Proximity to a known high-impact event."""
        if hours_to_event is None:
            return (False, 0.0)
        h = float(hours_to_event)
        s_event = 1.0 if h < 1.0 else _clip01(4.0 / h)
        return (h < 4.0, s_event)

    # -- 9.2 / 9.4 Assessment ----------------------------------------------------

    def assess(
        self,
        returns_matrix=None,
        portfolio_returns=None,
        portfolio_values=None,
        position_pnls=None,
        spread_bps: float = 0.0,
        adv_ratio: float = 0.0,
        overnight_gaps=None,
        hours_to_event: float | None = None,
        current_regime: str = "mean_reverting",
        gjr_params: dict | None = None,
        har_params: dict | None = None,
        use_cache: bool = True,
    ) -> CrisisStatus:
        """Run all seven detectors and return the composite CrisisStatus.

        Results are cached for ``cache_ttl_seconds`` (default 5 minutes). Pass
        ``use_cache=False`` to force a fresh evaluation.
        """
        now = time.time()
        if (
            use_cache
            and self._cache is not None
            and (now - self._cache_time) < self._cache_ttl
        ):
            return self._cache

        self._gjr_params = gjr_params
        self._har_params = har_params

        results = [
            self._detect_correlation_spike(returns_matrix),
            self._detect_vol_explosion(portfolio_returns),
            self._detect_drawdown_acceleration(portfolio_values, current_regime),
            self._detect_breadth_collapse(position_pnls),
            self._detect_liquidity_stress(spread_bps, adv_ratio),
            self._detect_gap_risk(overnight_gaps),
            self._detect_event_risk(hours_to_event),
        ]
        fired_flags = [bool(flag) for flag, _ in results]
        scores = [float(score) for _, score in results]
        (
            s_corr, s_vol, s_dd, s_breadth, s_liq, s_gap, s_event,
        ) = scores

        severity = _clip01(
            _W_CORR * s_corr
            + _W_VOL * s_vol
            + _W_DD * s_dd
            + _W_BREADTH * s_breadth
            + _W_LIQ * s_liq
            + _W_GAP * s_gap
            + _W_EVENT * s_event
        )
        level = level_from_severity(severity)
        defensive = level in (CrisisLevel.CRISIS, CrisisLevel.CRITICAL)

        signal_values = dict(zip(_DETECTOR_NAMES, scores))
        signals_fired = [
            name for name, fired in zip(_DETECTOR_NAMES, fired_flags) if fired
        ]

        status = CrisisStatus(
            level=level,
            defensive_mode=defensive,
            signals_fired=signals_fired,
            signal_values=signal_values,
            severity_score=severity,
            liquidity_stress_score=s_liq,
            gap_risk_score=s_gap,
            event_risk_score=s_event,
            timestamp=now,
        )

        if defensive:
            logger.warning(
                "Crisis level %s (S=%.3f); defensive mode engaged. Fired: %s",
                level.value, severity, signals_fired,
            )

        self._cache = status
        self._cache_time = now
        self._last_status = status
        return status

    def is_defensive(self) -> bool:
        """Whether the most recent assessment requested defensive mode."""
        return self._last_status.defensive_mode if self._last_status else False

    def current_level(self) -> CrisisLevel:
        """The level from the most recent assessment (NORMAL if none yet)."""
        return self._last_status.level if self._last_status else CrisisLevel.NORMAL

    def reset_cache(self) -> None:
        """Clear the 5-minute cache so the next assess() recomputes."""
        self._cache = None
        self._cache_time = 0.0


# ── Module-level singleton ──────────────────────────────────────────────────────

_MANAGER: CrisisManager | None = None


def get_crisis_manager() -> CrisisManager:
    """Return the process-wide CrisisManager singleton (created on first use)."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = CrisisManager()
    return _MANAGER


def reset_crisis_manager() -> None:
    """Drop the singleton (tests / restarts)."""
    global _MANAGER
    _MANAGER = None
