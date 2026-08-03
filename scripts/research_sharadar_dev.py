"""DEV-WINDOW exploration runner for the pre-registered Sharadar dev/confirm program
(``research/medallion_style_alpha_search/sharadar_confirmatory_prereg.md`` §2-3).

Runs the registered pipeline on the DEV extract ONLY (SF1 datekey and SEP dates both
hard-cut at 2015-12-31 when the extract was built). Everything this script prints is
**EXPLORATION** — dev-window results carry NO deployability weight; they exist to guide
iteration before the single pre-registered confirmation shot.

GUARDRAILS:
  * Reads ONLY ``_data/sharadar/dev/{sf1_dev,sep_dev}.parquet`` — never the confirm
    extract, never the raw export. Refuses to run if the frames contain any post-cut
    dates (belt-and-braces against a mis-built extract).
  * Banner + label mark every output line of this tool as exploration.

Usage:
  python scripts/research_sharadar_dev.py                       (unfiltered baseline)
  python scripts/research_sharadar_dev.py --top-n 1000          (DEV-1 liquidity mask)
  python scripts/research_sharadar_dev.py --top-n 1000 --cap 1.0 --cost-bps 20
  python scripts/research_sharadar_dev.py --min-dollar-vol 5e6  (absolute floor)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.research_sharadar_alpha import (  # noqa: E402
    DEFAULT_COST_BPS,
    DEFAULT_WARMUP_DAYS,
    _price_matrix,
    _rebalance_dates,
    build_liquidity_universe,
    print_report,
    run_research,
)

DEV_DIR = Path("_data/sharadar/dev")
DEV_CUT = pd.Timestamp("2015-12-31")

EXPLORATION_BANNER = (
    "=== EXPLORATION (DEV WINDOW <= 2015-12-31) ===\n"
    "Dev-window results guide iteration ONLY; they confer NO deployability\n"
    "(sharadar_confirmatory_prereg.md - one confirmation shot, frozen model, 2016+)."
)


def load_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    sf1 = pd.read_parquet(DEV_DIR / "sf1_dev.parquet")
    sep = pd.read_parquet(DEV_DIR / "sep_dev.parquet")
    if sf1["datekey"].max() > DEV_CUT or sep["date"].max() > DEV_CUT:
        raise RuntimeError(
            "DEV extract contains post-cut rows - rebuild it before any exploration "
            f"(sf1 max {sf1['datekey'].max()}, sep max {sep['date'].max()})"
        )
    return sf1, sep


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DEV-window exploration runner")
    p.add_argument("--top-n", type=int, default=None,
                   help="Liquidity mask: keep the N most liquid names per rebalance.")
    p.add_argument("--min-dollar-vol", type=float, default=None,
                   help="Liquidity mask: absolute trailing-median dollar-volume floor.")
    p.add_argument("--cap", type=float, default=None,
                   help="Clip forward returns to +/-cap (QA guard; e.g. 1.0 = +/-100%%).")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(EXPLORATION_BANNER)
    sf1, sep = load_dev()
    print(f"dev extract: sf1 {len(sf1):,} ARQ rows | sep {len(sep):,} price rows | "
          f"hard cut {DEV_CUT.date()} verified")

    mask = None
    tag = "unfiltered"
    if args.top_n is not None or args.min_dollar_vol is not None:
        px = _price_matrix(sep)
        panel_dates = _rebalance_dates(px, DEFAULT_WARMUP_DAYS)[:-1]
        mask = build_liquidity_universe(
            sep, panel_dates, top_n=args.top_n, min_dollar_volume=args.min_dollar_vol
        )
        per_date = mask.sum(axis=1)
        tag = (f"top_n={args.top_n} min_dv={args.min_dollar_vol}"
               f" (names/date min={int(per_date.min())} med={int(per_date.median())})")
        print(f"liquidity mask: {tag}")
    if args.cap is not None:
        tag += f" cap=+/-{args.cap:g}"
    if args.cost_bps != DEFAULT_COST_BPS:
        tag += f" cost={args.cost_bps:g}bps"

    report = run_research(
        sf1, sep, label=f"DEV 1999-2015 EXPLORATION [{tag}]",
        cost_bps=args.cost_bps, universe_mask=mask, fwd_return_cap=args.cap,
    )
    print_report(report)
    print(EXPLORATION_BANNER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
