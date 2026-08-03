"""Sweep the signal-tilt strength to find whether expressing the validated sleeves
harder improves RISK-ADJUSTED returns (the safe way to beat the market).

Fetches once, then runs the engine backtest at the robust-aggressive (2x) risk
budget across tilt strengths. A higher Sharpe at higher tilt = the sleeves add
real alpha; a flat/declining Sharpe = the tilt is just adding noise/cost.
"""

from __future__ import annotations

from backtesting.harness import Backtester
from data.price_ingestion import fetch_prices, ingest_prices

SYMBOLS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"]
TILTS = [5e-4, 3e-3, 1e-2, 3e-2]


def main() -> int:
    tidy = fetch_prices(SYMBOLS, "2016-01-01", "2024-12-31")
    prices = tidy.pivot(index="date", columns="symbol", values="close").sort_index().dropna(how="any")
    ingest_prices(tidy, mode="PAPER")
    print(f"\n{'tilt':>8} {'ret':>8} {'vol':>8} {'sharpe':>8} {'maxdd':>8} {'calmar':>8} {'turn':>6}")
    for tilt in TILTS:
        bt = Backtester(mode="PAPER", rebalance="M", warmup=90, cost_bps_per_turnover=10.0, seed=42,
                        target_vol=0.22, max_gross_leverage=2.0, max_position_weight=0.20,
                        cvar_limit=0.12, signal_tilt_strength=tilt)
        m = bt.run(prices).metrics
        print(f"{tilt:>8.4f} {m['ann_return']:>7.2%} {m['ann_vol']:>7.2%} {m['sharpe']:>8.2f} "
              f"{m['max_drawdown']:>7.2%} {m['calmar']:>8.2f} {m['avg_turnover']:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
