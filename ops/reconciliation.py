"""
TradingEngineResearch — Reconciliation engine (Phase 3 / directive §17)
==============================================================
Structured reconciliation across **positions, cash, and NAV** between two states —
typically the internal record (engine book, or balances replayed from the immutable
ledger) and the broker (IBKR/Flex). Built to extend to a **three-way** check
(internal ↔ broker ↔ bank/accounting): just reconcile pairwise.

Philosophy (directive §7.5 / §17): differences are **surfaced, never auto-applied** —
correcting the book from the broker is a deliberate, audited risk decision, not
something a reconciliation pass does silently. Severe uncertainty (a non-finite value
on either side) **fails closed** to a BREAK. Breaks carry the dimension, key, both
values and the diff, so they can be aged and assigned downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["ReconBreak", "ReconReport", "reconcile"]


@dataclass(frozen=True)
class ReconBreak:
    dimension: str        # "position" | "cash" | "nav"
    key: str              # symbol / currency / "NAV"
    internal: float
    broker: float
    diff: float           # internal - broker (nan-safe sentinel if non-finite input)
    severity: str         # "WARNING" | "BREAK"

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "key": self.key, "internal": self.internal,
                "broker": self.broker, "diff": self.diff, "severity": self.severity}


@dataclass(frozen=True)
class ReconReport:
    asof: str
    breaks: tuple[ReconBreak, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return len(self.breaks) == 0

    def to_payload(self) -> dict[str, Any]:
        """Compact dict for the immutable ledger (a RECONCILIATION event)."""
        return {"asof": self.asof, "clean": self.clean,
                "n_breaks": len(self.breaks), "breaks": [b.to_dict() for b in self.breaks]}

    def to_alert(self, mode: str = "PAPER") -> Optional[dict[str, Any]]:
        """Surfacing alert (compatible with the run-loop shape), or None if clean.
        LIVE breaks are RED; otherwise WARNING."""
        if self.clean:
            return None
        dims = sorted({b.dimension for b in self.breaks})
        return {
            "severity": "RED" if mode == "LIVE" else "WARNING",
            "kind": "reconciliation",
            "message": f"internal vs broker diverges: {len(self.breaks)} break(s) across {dims}",
            "detail": [b.to_dict() for b in self.breaks],
        }


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _scalar_break(dim: str, key: str, internal: float, broker: float, tol: float) -> Optional[ReconBreak]:
    # Fail closed: a non-finite value on either side is always a BREAK.
    if not _finite(internal) or not _finite(broker):
        return ReconBreak(dim, key, float(internal) if _finite(internal) else math.nan,
                          float(broker) if _finite(broker) else math.nan, math.nan, "BREAK")
    diff = float(internal) - float(broker)
    if abs(diff) > tol:
        return ReconBreak(dim, key, float(internal), float(broker), diff, "BREAK")
    return None


def _map_breaks(dim: str, internal: dict, broker: dict, tol: float) -> list[ReconBreak]:
    out: list[ReconBreak] = []
    for key in sorted(set(internal) | set(broker)):
        b = _scalar_break(dim, str(key), float(internal.get(key, 0.0) or 0.0),
                          float(broker.get(key, 0.0) or 0.0), tol)
        if b is not None:
            out.append(b)
    return out


def reconcile(
    internal: dict[str, Any],
    broker: dict[str, Any],
    *,
    asof: str,
    share_tol: float = 1.0,
    cash_tol: float = 1.0,
    nav_tol_pct: float = 0.005,
) -> ReconReport:
    """Reconcile two states. Each is a dict with any of:
      ``positions`` {symbol: shares}, ``cash`` {currency: amount}, ``nav`` float.
    Returns a :class:`ReconReport` of all breaks. Tolerances: ``share_tol`` (whole-share
    band), ``cash_tol`` (per-currency absolute), ``nav_tol_pct`` (relative NAV band)."""
    breaks: list[ReconBreak] = []
    breaks += _map_breaks("position", dict(internal.get("positions") or {}),
                          dict(broker.get("positions") or {}), share_tol)
    breaks += _map_breaks("cash", dict(internal.get("cash") or {}),
                          dict(broker.get("cash") or {}), cash_tol)
    i_nav, b_nav = internal.get("nav"), broker.get("nav")
    if i_nav is not None and b_nav is not None:
        if not _finite(i_nav) or not _finite(b_nav):
            breaks.append(ReconBreak("nav", "NAV",
                                     float(i_nav) if _finite(i_nav) else math.nan,
                                     float(b_nav) if _finite(b_nav) else math.nan, math.nan, "BREAK"))
        else:
            tol = abs(float(b_nav)) * nav_tol_pct
            b = _scalar_break("nav", "NAV", float(i_nav), float(b_nav), tol)
            if b is not None:
                breaks.append(b)
    return ReconReport(asof=asof, breaks=tuple(breaks))
