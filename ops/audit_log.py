"""
TradingEngineResearch — Cycle Audit Log
===========================
Durable, append-only audit trail of decision cycles (spec STEP 13: "append cycle
summary to the audit trail"). One markdown table row per cycle, one file for the
whole deployment — grep-able, diff-able, and human-readable.

The engine writes here only when constructed with an explicit ``audit_log_path``
(the PAPER/LIVE run-loop opts in): research replays and the test suite must stay
free of disk I/O. Timestamps come from the cycle's ``asof_time``, never the wall
clock, so a replayed cycle audits identically.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["COLUMNS", "append_cycle_summary"]

COLUMNS: tuple[str, ...] = (
    "asof", "mode", "regime", "crisis_level", "blocked", "admitted",
    "order_intents", "fills", "live_orders_submitted", "alerts",
)

_HEADER = (
    "# TradingEngineResearch — Cycle Audit Trail\n\n"
    "Machine-appended by the TradingEngineResearch engine (STEP 13); one row per decision cycle.\n\n"
    "| " + " | ".join(COLUMNS) + " |\n"
    "|" + "|".join("---" for _ in COLUMNS) + "|\n"
)

_KEY_BY_COLUMN = {
    "asof": "asof_time",
    "crisis_level": "crisis_level",
    "live_orders_submitted": "live_orders_submitted",
    "order_intents": "order_intents",
}


def append_cycle_summary(path: str | Path, summary: dict) -> None:
    """Append one cycle-summary row to the audit file (creating it, with its
    header, on first use). Raises on I/O errors — the engine wraps this in its
    fail-soft integration layer and logs the degradation loudly."""
    path = Path(path)
    row = "| " + " | ".join(
        str(summary.get(_KEY_BY_COLUMN.get(col, col), "")) for col in COLUMNS
    ) + " |\n"
    if not path.exists():
        path.write_text(_HEADER + row, encoding="utf-8")
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row)
