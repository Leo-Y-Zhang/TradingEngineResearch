"""What it actually takes to run the incumbent engine at a 30%/yr return target.

The operator asked for the 30% number by the only route that needs no new research:
turn up ``target_vol`` on the proven engine and accept the risk. This measures that
honestly rather than asserting it.

WHAT THIS IS NOT. It is **not alpha**. The engine's edge over equal-weight buy-and-hold
is thin (Sharpe ~1.15 vs ~1.12); essentially all of the return is market beta with
cost-honest drawdown control. Raising ``target_vol`` scales beta, so it multiplies the
good years AND the bad ones. Nothing here is a validated edge and SIGNALS-5 still holds:
the live engine trades nothing without one.

THREE OPTIMISM SOURCES, stated before any number is read:
  1. **Survivorship.** The universe is 8 hand-picked mega-caps that were already winners
     in 2016. Names that failed over the window are absent by construction.
  2. **Window.** 2016-2024 contains one of the strongest equity bull runs on record.
     It has a COVID crash and a 2022 bear, but no 2000-2002 and no 2008.
  3. **Compounding drag.** Higher volatility lowers geometric return relative to
     arithmetic. At 30% vol the drag is roughly 4.5%/yr, and it is why leverage does not
     scale terminal wealth linearly.
The forward-looking honest expectation is therefore MATERIALLY WORSE than any row below.

The bootstrap is a stationary block bootstrap on realised monthly returns, which
preserves short-horizon autocorrelation but CANNOT invent a crisis worse than the worst
in the sample. Treat its tail as a floor on how bad things get, not a ceiling.
"""

from __future__ import annotations

import numpy as np

from backtesting.harness import Backtester
from data.price_ingestion import fetch_prices, ingest_prices
from scripts.backtest_real import DEFAULT_SYMBOLS, to_wide

TARGET_VOLS = (0.22, 0.28, 0.34, 0.40, 0.46)
MAX_LEVERAGE = 4.0
BOOTSTRAP_PATHS = 20_000
HORIZON_YEARS = 10
BLOCK_MONTHS = 6
SEED = 42


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def block_bootstrap(returns: np.ndarray, n_paths: int, horizon: int,
                    block: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stationary block bootstrap -> (annualised return, max drawdown) per path."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    annual, drawdowns = np.empty(n_paths), np.empty(n_paths)
    n_blocks = int(np.ceil(horizon / block))
    for i in range(n_paths):
        starts = rng.integers(0, n, size=n_blocks)
        path = np.concatenate([
            np.take(returns, np.arange(s, s + block) % n) for s in starts
        ])[:horizon]
        equity = np.cumprod(1.0 + path)
        annual[i] = equity[-1] ** (12.0 / horizon) - 1.0
        drawdowns[i] = max_drawdown(np.concatenate([[1.0], equity]))
    return annual, drawdowns


def main() -> int:
    tidy = fetch_prices(DEFAULT_SYMBOLS, "2016-01-01", "2024-12-31")
    prices = to_wide(tidy)
    ingest_prices(tidy, mode="PAPER")
    print(f"panel: {prices.shape[0]} days x {prices.shape[1]} symbols "
          f"(8 hand-picked survivors, 2016-2024)\n")

    print("=" * 96)
    print("REALISED (in-sample, on the optimistic panel described in the header)")
    print("=" * 96)
    print(f"{'target_vol':>11}{'ann ret':>10}{'ann vol':>10}{'Sharpe':>8}"
          f"{'maxDD':>9}{'Calmar':>8}{'worst 12m':>11}")

    series = {}
    for target in TARGET_VOLS:
        backtest = Backtester(
            mode="PAPER", rebalance="M", warmup=90, cost_bps_per_turnover=10.0,
            seed=SEED, target_vol=target, max_gross_leverage=MAX_LEVERAGE,
            max_position_weight=0.20, cvar_limit=0.12, signal_tilt_strength=3e-3,
        )
        result = backtest.run(prices)
        metrics = result.metrics
        returns = np.asarray(result.returns, dtype=float)
        series[target] = returns

        rolling = np.array([
            np.prod(1.0 + returns[i:i + 12]) - 1.0
            for i in range(max(len(returns) - 12, 1))
        ])
        worst_12m = float(rolling.min()) if len(rolling) else float("nan")

        print(f"{target:>11.0%}{metrics.get('ann_return', 0):>9.1%}"
              f"{metrics.get('ann_vol', 0):>10.1%}{metrics.get('sharpe', 0):>8.2f}"
              f"{metrics.get('max_drawdown', 0):>9.1%}"
              f"{metrics.get('calmar', 0):>8.2f}{worst_12m:>11.1%}")

    print("\n" + "=" * 96)
    print(f"FORWARD RISK - {BOOTSTRAP_PATHS:,} block-bootstrap paths, "
          f"{HORIZON_YEARS}y horizon, {BLOCK_MONTHS}-month blocks")
    print("=" * 96)
    print(f"{'target_vol':>11}{'median':>9}{'p5':>9}{'p95':>9}"
          f"{'med maxDD':>11}{'p95 maxDD':>11}{'P(lose $)':>11}{'P(DD>50%)':>11}")

    for target in TARGET_VOLS:
        returns = series[target]
        if len(returns) < BLOCK_MONTHS * 2:
            continue
        annual, drawdowns = block_bootstrap(
            returns, BOOTSTRAP_PATHS, HORIZON_YEARS * 12, BLOCK_MONTHS, SEED
        )
        print(f"{target:>11.0%}{np.median(annual):>9.1%}"
              f"{np.percentile(annual, 5):>9.1%}{np.percentile(annual, 95):>9.1%}"
              f"{np.median(drawdowns):>11.1%}{np.percentile(drawdowns, 95):>11.1%}"
              f"{float(np.mean(annual < 0)):>11.1%}"
              f"{float(np.mean(drawdowns > 0.50)):>11.1%}")

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  - The 22% row is the shipped default and reproduces the banked headline.")
    print("  - Any row reaching ~30% does so by SCALING BETA, not by earning alpha.")
    print("  - 'P(lose $)' is the share of 10-year paths ending below where they")
    print("    started. A strategy can have a fine median and still ruin a real")
    print("    account, because you live one path, not the median of twenty thousand.")
    print("  - The bootstrap resamples 2016-2024 and therefore CANNOT produce a crisis")
    print("    worse than that window's worst. There is no 2008 in it. Real forward")
    print("    tails are fatter than every number above.")
    print("  - Survivorship, window and compounding drag all push the honest forward")
    print("    expectation BELOW these rows. See the module header.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
