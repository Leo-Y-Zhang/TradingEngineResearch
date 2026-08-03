"""Falsification check for the TSMOM sleeve: does the plumbing manufacture return?

The sleeve's gross Sharpe is ~0.45. Before believing it, replace the trend signal with a
seeded RANDOM sign per (instrument, rebalance) while keeping everything else identical --
the same universe, the same inverse-volatility magnitudes, the same vol targeting, the
same delisting accounting, the same daily marking. A leak in the pipeline (a
forward-looking price, a survivorship filter, an accounting error that pays the book)
would show up as a positive placebo Sharpe.

Eight seeds are run so the placebo produces a DISTRIBUTION rather than a single draw. The
spread of that distribution is also the honest standard error of the sleeve's own Sharpe
estimate over this sample, which is the number that decides whether a 0.45 means anything.

    .venv/Scripts/python.exe -m scripts.tsmom_placebo_check
"""

from __future__ import annotations

import logging

import numpy as np

from research.sleeves.tsmom_multitimeframe import (
    BOOK_VOL_WINDOW,
    TRADING_DAYS,
    annualised,
    build_daily_matrix,
    build_universe,
    load_inputs,
    plan_book,
    simulate,
)
from scripts.run_tsmom_sleeve import load_prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("placebo")

SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
TARGET_VOL = 0.15


def main() -> None:
    panel, delistings, sector_of = load_inputs()
    # Run against the most favourable declared universe: if anything leaks, it leaks
    # hardest where the strategy looks best.
    universe = build_universe(panel, sector_of, mode="liquid_schedule")
    tickers = {t for members in universe.members.values() for t in members}
    matrix = build_daily_matrix(load_prices(tickers), tickers, delistings)

    plan = plan_book(matrix, universe)
    original = [rebalance.base_weights.copy() for rebalance in plan.rebalances]

    real = simulate(matrix, plan, TARGET_VOL, charge_costs=False)
    real_cagr, real_vol, real_sharpe = annualised(real.returns)
    logger.info(
        "REAL    gross %.2f%%/yr, vol %.1f%%, Sharpe %.3f, net exposure %.2f, gross %.2f",
        real_cagr * 100, real_vol * 100, real_sharpe, real.mean_net_exposure,
        real.mean_gross_exposure,
    )

    sharpes = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for rebalance, weights in zip(plan.rebalances, original, strict=True):
            # Keep every magnitude, destroy only the direction: same book size, same
            # shape, same names, random view.
            held = weights != 0.0
            randomised = weights.copy()
            randomised[held] = np.abs(weights[held]) * rng.choice(
                [-1.0, 1.0], size=int(held.sum())
            )
            rebalance.base_weights = randomised
            start = max(0, rebalance.day - BOOK_VOL_WINDOW + 1)
            history = matrix.returns[start : rebalance.day + 1]
            history = np.where(np.isfinite(history), history, 0.0)
            rebalance.book_vol = float(
                np.std(history @ randomised, ddof=1)
            ) * np.sqrt(TRADING_DAYS)

        placebo = simulate(matrix, plan, TARGET_VOL, charge_costs=False)
        cagr, vol, sharpe = annualised(placebo.returns)
        sharpes.append(sharpe)
        logger.info(
            "placebo seed %d: gross %.2f%%/yr, vol %.1f%%, Sharpe %.3f, net exposure %.2f",
            seed, cagr * 100, vol * 100, sharpe, placebo.mean_net_exposure,
        )

    mean = float(np.mean(sharpes))
    spread = float(np.std(sharpes, ddof=1))
    logger.info("placebo Sharpe: mean %.3f, sd %.3f over %d seeds", mean, spread,
                len(SEEDS))
    logger.info(
        "sleeve gross Sharpe %.3f is %.2f placebo standard deviations from chance; "
        "its own t-statistic over the sample is %.2f",
        real_sharpe, (real_sharpe - mean) / spread if spread > 0 else float("nan"),
        real_sharpe * np.sqrt(np.isfinite(real.returns).sum() / TRADING_DAYS),
    )


if __name__ == "__main__":
    main()
