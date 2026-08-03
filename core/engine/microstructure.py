"""
TradingEngineResearch — Microstructure / Order-Flow Imbalance
=================================================
Deep order-flow-imbalance (OFI) signal and its rejection gate.

OFI measures whether resting size is being added to the bid (buying pressure) or
the ask (selling pressure) faster than it is consumed. The deep (L2) variant
weights each of the top 5 levels by ``1/l`` so the touch dominates, normalises by
recent traded volume, and clips to ``[-1, 1]``. A simpler L1 variant is used when
only top-of-book is available. Unavailable / malformed data yields ``0.0`` (a
neutral signal) — never an exception.

Expected ``ibkr_data`` keys
---------------------------
L2 (preferred): ``bid_sizes`` & ``ask_sizes`` (lists, deepest-first up to 5),
with either ``prev_bid_sizes`` / ``prev_ask_sizes`` or precomputed
``delta_bid_sizes`` / ``delta_ask_sizes``, plus ``total_volume_5min``.
L1 (fallback): ``bid_size`` & ``ask_size`` with either ``prev_bid_size`` /
``prev_ask_size`` or ``bid_size_change`` / ``ask_size_change``.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["compute_ofi", "ofi_filter_gate"]

_MAX_LEVELS = 5
_OFI_REJECT_THRESHOLD = 0.30


def _as_list(value) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _level_deltas(data: dict, side: str, levels: int) -> list[float] | None:
    """Return per-level size deltas for ``side`` ('bid'/'ask'), or None."""
    precomputed = _as_list(data.get(f"delta_{side}_sizes"))
    if precomputed is not None:
        return precomputed[:levels]
    current = _as_list(data.get(f"{side}_sizes"))
    previous = _as_list(data.get(f"prev_{side}_sizes"))
    if current is None or previous is None:
        return None
    n = min(levels, len(current), len(previous))
    return [current[i] - previous[i] for i in range(n)]


def _bounded(value: float) -> float:
    """Clip to ``[-1, 1]`` and map a non-finite result (NaN from a bad tick) to the
    neutral 0.0 — the module contract is a bounded signal, never NaN/exception."""
    clipped = float(np.clip(value, -1.0, 1.0))
    return clipped if np.isfinite(clipped) else 0.0


def _l1_change(data: dict, side: str) -> float | None:
    direct = data.get(f"{side}_size_change")
    if direct is not None:
        return float(direct)
    current = data.get(f"{side}_size")
    previous = data.get(f"prev_{side}_size")
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def compute_ofi(ibkr_data: dict) -> float:
    """
    Normalised order-flow-imbalance signal in ``[-1, 1]`` (0.0 if unavailable).

    Positive ⇒ net bid-side pressure (buying); negative ⇒ ask-side pressure.
    """
    if not isinstance(ibkr_data, dict):
        return 0.0

    # ── L2 deep OFI ─────────────────────────────────────────────────────────
    bid_deltas = _level_deltas(ibkr_data, "bid", _MAX_LEVELS)
    ask_deltas = _level_deltas(ibkr_data, "ask", _MAX_LEVELS)
    total_volume = ibkr_data.get("total_volume_5min")

    if bid_deltas and ask_deltas and total_volume:
        # Pair levels by depth (a level-count mismatch uses the shorter side) and skip
        # any partial/bad level (NaN/inf) so one missing level cannot zero the whole
        # signal — accumulate only over valid level pairs. With no usable level, fall
        # through to the L1 path rather than returning a degenerate 0 here.
        levels = min(len(bid_deltas), len(ask_deltas))
        ofi_int = 0.0
        used = 0
        for level in range(1, levels + 1):
            bid_d, ask_d = bid_deltas[level - 1], ask_deltas[level - 1]
            if not (math.isfinite(bid_d) and math.isfinite(ask_d)):
                continue
            ofi_int += (1.0 / level) * (bid_d - ask_d)
            used += 1
        denom = float(total_volume) * 0.5
        if used > 0 and denom > 0.0:
            return _bounded(ofi_int / denom)

    # ── L1 fallback ─────────────────────────────────────────────────────────
    bid_change = _l1_change(ibkr_data, "bid")
    ask_change = _l1_change(ibkr_data, "ask")
    bid_size = ibkr_data.get("bid_size")
    ask_size = ibkr_data.get("ask_size")
    if bid_change is not None and ask_change is not None and bid_size is not None and ask_size is not None:
        denom = float(bid_size) + float(ask_size) + 1e-9
        return _bounded((bid_change - ask_change) / denom)

    return 0.0


def ofi_filter_gate(direction: str, ofi_norm: float) -> bool:
    """
    Microstructure veto: reject a trade that fights strong opposing order flow.

    Returns ``False`` (reject) when ``direction == "BUY"`` and ``ofi_norm < -0.30``,
    or ``direction == "SELL"`` and ``ofi_norm > 0.30``; ``True`` otherwise.
    """
    if direction == "BUY" and ofi_norm < -_OFI_REJECT_THRESHOLD:
        return False
    if direction == "SELL" and ofi_norm > _OFI_REJECT_THRESHOLD:
        return False
    return True
