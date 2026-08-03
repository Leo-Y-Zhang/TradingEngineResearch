"""Diagnostics for the reversal sleeve. Bounds and controls, not variants.

None of these is an alternative configuration that could be selected as "the result".
The registered answer is whatever `run_reversal_sleeve.py` printed. These exist to
establish three things that answer follows-on questions the raw numbers provoke:

1. SPREAD-FREE UPPER BOUND. The registered run dies on spread cost, and the spread
   estimate is the least certain input in the whole study. So: set the spread to zero
   and keep only commission and impact. If the sleeve still loses to its own universe
   with FREE spreads, then no improvement to the spread estimator can rescue it and
   the verdict does not depend on the cost model being right.

2. NEGATIVE CONTROL. Replace the signal with a per-date random permutation of itself.
   Gross return should collapse toward zero while costs stay put. If a shuffled signal
   also earns 7%/yr gross, the machinery is manufacturing alpha out of the decile
   construction (a low-price, high-volatility tilt) rather than measuring reversal.

3. ERA SPLIT. Gross Sharpe by period. Short-horizon reversal is the canonical example
   of an effect competed away by decimalisation and electronic market making; if the
   gross edge is concentrated pre-2003 that is decisive for whether it is deployable
   now, quite apart from costs.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sleeves.short_horizon_reversal import (  # noqa: E402
    SECONDARY_CONFIG,
    ReversalConfig,
    _run_leg,
    build_matrices,
    build_selections,
    month_row_for,
    weekly_grid,
)
from scripts.run_reversal_sleeve import summarise  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("diagnostics")

SEED = 20260728  # fixed before the run


def _grid(panel, config):
    signal_idx, exec_idx = weekly_grid(panel.dates)
    month_rows = month_row_for(panel, signal_idx)
    keep = (signal_idx >= config.lookback_days) & (month_rows >= 0)
    return signal_idx[keep], exec_idx[keep], month_rows[keep]


def _books(panel, config, selections, signal_idx, exec_idx, month_rows):
    empty = np.array([], dtype=int)
    ls_long = [ln if sh.size else empty
               for ln, sh in zip(selections["long"], selections["short"])]
    ls_long_book = _run_leg(panel, config, ls_long, signal_idx, exec_idx, month_rows,
                            config.long_leg_exposure)
    short_book = _run_leg(panel, config, selections["short"], signal_idx, exec_idx,
                          month_rows, config.short_leg_exposure,
                          borrow_annual=config.short_borrow_annual)
    long_book = _run_leg(panel, config, selections["long"], signal_idx, exec_idx,
                         month_rows, config.long_leg_exposure)
    universe_book = _run_leg(panel, config, selections["universe"], signal_idx,
                             exec_idx, month_rows, 1.0)
    trim = slice(0, len(signal_idx) - 1)
    return {
        "ls_gross": ls_long_book.gross_return[trim] - short_book.gross_return[trim],
        "ls_cost": ls_long_book.cost[trim] + short_book.cost[trim],
        "lo_gross": long_book.gross_return[trim],
        "lo_cost": long_book.cost[trim],
        "bench": universe_book.gross_return[trim],
        "dates": ls_long_book.dates[trim],
    }


def main() -> None:
    configs = (ReversalConfig(), SECONDARY_CONFIG)
    panels = build_matrices(configs)

    for config in configs:
        panel = panels[config.label]
        signal_idx, exec_idx, month_rows = _grid(panel, config)
        selections = build_selections(panel, config, signal_idx, exec_idx, month_rows)
        base = _books(panel, config, selections, signal_idx, exec_idx, month_rows)
        ppy = config.periods_per_year

        print(f"\n{'=' * 78}\n{config.label} -- DIAGNOSTICS\n{'=' * 78}")

        # --- 1. spread-free upper bound ---------------------------------------
        free = replace(config, label=config.label + "_zero_spread")
        zero_panel = panels[config.label]
        saved_spread = zero_panel.spread_cost_basis
        zero_panel.spread_cost_basis = np.zeros_like(saved_spread)
        bound = _books(zero_panel, free, selections, signal_idx, exec_idx, month_rows)
        zero_panel.spread_cost_basis = saved_spread

        bench = summarise(base["bench"], ppy)
        for name, key in (("long/short", "ls"), ("long-only", "lo")):
            gross = bound[f"{key}_gross"]
            net = gross - bound[f"{key}_cost"]
            stats = summarise(net, ppy)
            print(f"[1] spread-FREE bound {name:<11} net {stats.annual_return:+.2%}/yr  "
                  f"Sharpe {stats.sharpe:+.2f}  "
                  f"excess {stats.annual_return - bench.annual_return:+.2%}/yr  "
                  f"(residual cost {np.mean(bound[f'{key}_cost']) * ppy:.2%}/yr)")

        # --- 2. negative control ----------------------------------------------
        rng = np.random.default_rng(SEED)
        shuffled = {"universe": selections["universe"], "signal": selections["signal"],
                    "long": [], "short": []}
        for k, universe in enumerate(selections["universe"]):
            n_long = len(selections["long"][k])
            n_short = len(selections["short"][k])
            picks = rng.permutation(universe)
            shuffled["long"].append(picks[:n_long])
            # Disjoint draw, mirroring the real construction's non-overlapping legs.
            shuffled["short"].append(picks[n_long:n_long + n_short])
        control = _books(panel, config, shuffled, signal_idx, exec_idx, month_rows)
        for name, key in (("long/short", "ls"), ("long-only", "lo")):
            real = summarise(base[f"{key}_gross"], ppy)
            fake = summarise(control[f"{key}_gross"], ppy)
            print(f"[2] negative control  {name:<11} real gross Sharpe {real.sharpe:+.2f} "
                  f"({real.annual_return:+.2%}/yr)   random-signal gross Sharpe "
                  f"{fake.sharpe:+.2f} ({fake.annual_return:+.2%}/yr)")

        # --- 3. era split ------------------------------------------------------
        years = base["dates"].year.to_numpy()
        eras = ((1998, 2002), (2003, 2007), (2008, 2011), (2012, 2015))
        print("[3] era split (long/short):")
        for lo_year, hi_year in eras:
            mask = (years >= lo_year) & (years <= hi_year)
            if mask.sum() < 30:
                continue
            g = summarise(base["ls_gross"][mask], ppy)
            n = summarise(base["ls_gross"][mask] - base["ls_cost"][mask], ppy)
            b = summarise(base["bench"][mask], ppy)
            print(f"    {lo_year}-{hi_year}  n={mask.sum():3d}  "
                  f"gross {g.annual_return:+7.2%}/yr Sharpe {g.sharpe:+5.2f}   "
                  f"net {n.annual_return:+8.2%}/yr   "
                  f"universe EW {b.annual_return:+7.2%}/yr")


if __name__ == "__main__":
    main()
