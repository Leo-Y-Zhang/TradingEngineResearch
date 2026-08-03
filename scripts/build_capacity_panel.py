"""Build and cache the capacity-curve study panel (DEV window only).

Reads the raw Sharadar export, produces one row per (ticker, month-end) with price,
trailing liquidity, EDGE spread + resolution regime, liquidity band and forward return,
and caches it to ``_data/sharadar/panel/``.

Reads nothing after 2015-12-31: the DEV/CONFIRM split is enforced inside
``research.capacity_panel.load_prices``, not here, so it cannot be bypassed by a caller.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.capacity_panel import (  # noqa: E402
    BANDS,
    DEV_CUTOFF,
    PANEL_DIR,
    build_monthly_panel,
    delisting_returns,
    load_actions,
    load_prices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("panel")


def main() -> int:
    started = time.monotonic()
    log.info("DEV cutoff is %s; the 2016+ confirmation window is NOT read",
             DEV_CUTOFF.date())

    prices = load_prices()
    log.info("prices: %s rows, %s tickers, %s -> %s",
             f"{len(prices):,}", f"{prices['ticker'].nunique():,}",
             prices["date"].min().date(), prices["date"].max().date())
    assert prices["date"].max() <= DEV_CUTOFF, "confirmation window leaked into DEV"

    actions = load_actions()
    delistings = delisting_returns(actions)
    log.info("delistings resolved: %s names (%s terminal-loss, %s acquisition)",
             f"{len(delistings):,}",
             f"{(delistings['terminal_return'] < 0).sum():,}",
             f"{(delistings['terminal_return'] == 0).sum():,}")

    log.info("building monthly panel (the slow step; EDGE runs per eligible cell)")
    panel = build_monthly_panel(prices)
    log.info("panel: %s rows, %s tickers", f"{len(panel):,}",
             f"{panel['ticker'].nunique():,}")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = PANEL_DIR / "monthly_panel_dev.parquet"
    panel.to_parquet(panel_path, index=False)
    delistings.to_parquet(PANEL_DIR / "delistings.parquet", index=False)

    eligible = panel[panel["spread_regime"] != "ineligible"]
    log.info("eligible cells: %s of %s (%.1f%%)", f"{len(eligible):,}",
             f"{len(panel):,}", 100 * len(eligible) / max(len(panel), 1))

    print("\nBand coverage (eligible cells only):")
    print(f"{'band':>16} {'cells':>10} {'names':>8} {'measured':>10} "
          f"{'median spread':>15}")
    for label, _, _ in BANDS:
        rows = eligible[eligible["band"] == label]
        if rows.empty:
            print(f"{label:>16} {0:>10} {0:>8} {'-':>10} {'-':>15}")
            continue
        measured = rows[rows["spread_regime"] == "measured"]
        share = len(measured) / len(rows)
        median_bps = (measured["spread"].median() * 1e4
                      if not measured.empty else float("nan"))
        print(f"{label:>16} {len(rows):>10,} {rows['ticker'].nunique():>8,} "
              f"{share:>9.0%} {median_bps:>14.0f}bps")

    log.info("done in %.1f min -> %s", (time.monotonic() - started) / 60,
             panel_path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
