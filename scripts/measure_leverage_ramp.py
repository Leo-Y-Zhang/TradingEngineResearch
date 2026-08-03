"""OPT-1 evidence: the leverage ramp, measured on real data, OFF vs ON.

The vol-target scaler is procyclical — it reads TRAILING realised vol, so it
reaches ``max_gross_leverage`` exactly when vol has been lowest. ``max_lever_up_step``
bounds how fast the book may lever UP. This script measures what that costs on the
banked real-data panel so the choice is made on numbers, not intuition.

    python scripts/measure_leverage_ramp.py

Pulls from the network (yfinance); not part of the unit suite. The OFF row should
reproduce the banked 18.37% / Sharpe 1.15 / 17.09% maxDD headline — if it does not,
distrust the run before drawing any conclusion from the comparison.
"""

from backtesting.harness import Backtester
from data.price_ingestion import fetch_prices, ingest_prices
from scripts.backtest_real import to_wide, DEFAULT_SYMBOLS

tidy = fetch_prices(DEFAULT_SYMBOLS, "2016-01-01", "2024-12-31")
prices = to_wide(tidy)
ingest_prices(tidy, mode="PAPER")
print(f"panel: {prices.shape[0]} days x {prices.shape[1]} symbols")

rows = []
for label, step in (("OFF (shipped)", None), ("ramp 0.25", 0.25), ("ramp 0.10", 0.10)):
    bt = Backtester(mode="PAPER", rebalance="M", warmup=90, cost_bps_per_turnover=10.0,
                    seed=42, target_vol=0.22, max_gross_leverage=2.0,
                    max_position_weight=0.20, cvar_limit=0.12,
                    signal_tilt_strength=3e-3, max_lever_up_step=step)
    m = bt.run(prices).metrics
    rows.append((label, m.get("ann_return", 0), m.get("ann_vol", 0), m.get("sharpe", 0),
                 m.get("max_drawdown", 0), m.get("calmar", 0)))

print()
print(f"{'setting':<16}{'ann ret':>9}{'ann vol':>9}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}")
for label, r, v, s, dd, c in rows:
    print(f"{label:<16}{r:>8.2%}{v:>9.2%}{s:>8.2f}{dd:>9.2%}{c:>8.2f}")
