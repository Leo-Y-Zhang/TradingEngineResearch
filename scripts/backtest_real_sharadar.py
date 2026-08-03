"""Real-data backtest of the TradingEngineResearch engine on the local Sharadar SEP export.

Track C of the pre-registered Sharadar program (research/medallion_style_alpha_search/
sharadar_confirmatory_prereg.md §6): engine-calibration / data-source sensitivity of
the banked yfinance headline (scripts/backtest_real.py), independent of the signal
study. Identical universe, window, engine config, and cost model — only the price
source changes (Sharadar SEP export instead of the yfinance network fetch), so any
metric delta measures data-source sensitivity, not configuration drift.

Adjustment convention: yfinance ``auto_adjust=True`` returns split+dividend-adjusted
OHLC. Sharadar SEP ``open/high/low/close`` are split-adjusted only, while ``closeadj``
is split+dividend adjusted; each row's OHLC is therefore scaled by ``closeadj/close``
(making close == closeadj exactly) to reproduce the yfinance semantics. Volume is
split-adjusted in both sources and is passed through unchanged. The end date is
treated as EXCLUSIVE, matching ``yf.download``.

Usage (offline; needs the raw export under _data/sharadar/):
    python scripts/backtest_real_sharadar.py                       # defaults below
    python scripts/backtest_real_sharadar.py 2018-01-01 2024-12-31 M  AAPL MSFT ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.harness import Backtester
from data.price_ingestion import ingest_prices

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"]
DEFAULT_START = "2016-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_REBALANCE = "M"  # monthly (period alias); "W" for weekly

REPO_ROOT = Path(__file__).resolve().parents[1]
SEP_GLOB = "_data/sharadar/SHARADAR_SEP_*.csv"
_CHUNK_ROWS = 2_000_000  # ~3.2 GB raw export; never load it whole
_PRICE_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def load_sharadar_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Chunk-read the raw SEP export → tidy long OHLCV matching ``fetch_prices``.

    Filters to ``symbols`` and ``start <= date < end`` (end EXCLUSIVE, the yfinance
    convention), then scales each row's OHLC by ``closeadj/close`` so the frame
    carries split+dividend-adjusted prices exactly like ``auto_adjust=True``.
    """
    candidates = sorted(REPO_ROOT.glob(SEP_GLOB))
    if not candidates:
        raise FileNotFoundError(f"No Sharadar SEP export under {REPO_ROOT / SEP_GLOB}")
    sep_path = max(candidates, key=lambda p: p.stat().st_size)  # the full export

    wanted = set(symbols)
    usecols = ["ticker", "date", "open", "high", "low", "close", "volume", "closeadj"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(sep_path, usecols=usecols, chunksize=_CHUNK_ROWS):
        # ISO dates compare correctly as strings — filter before the datetime parse.
        sub = chunk[chunk["ticker"].isin(wanted)
                    & (chunk["date"] >= start) & (chunk["date"] < end)]
        if not sub.empty:
            parts.append(sub.copy())
    if not parts:
        raise ValueError(f"SEP export has no rows for {symbols} in {start}..{end}")

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = (df.rename(columns={"ticker": "symbol"})
            .sort_values(["symbol", "date"])
            .drop_duplicates(subset=["symbol", "date"], keep="last"))

    ratio = df["closeadj"] / df["close"]
    for col in ("open", "high", "low"):
        df[col] = df[col] * ratio
    df["close"] = df["closeadj"]
    return df[_PRICE_COLUMNS].reset_index(drop=True)


def to_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Long (date,symbol,close,…) → wide close-price matrix (index=date, cols=symbols)."""
    wide = tidy.pivot(index="date", columns="symbol", values="close").sort_index()
    return wide.dropna(how="any")  # keep only dates where every symbol trades


def main(argv: list[str]) -> int:
    start = argv[0] if len(argv) > 0 else DEFAULT_START
    end = argv[1] if len(argv) > 1 else DEFAULT_END
    rebalance = argv[2] if len(argv) > 2 else DEFAULT_REBALANCE
    symbols = argv[3:] if len(argv) > 3 else DEFAULT_SYMBOLS

    print(f"Loading {len(symbols)} symbols {start}..{end} from the Sharadar SEP export: "
          f"{', '.join(symbols)}")
    tidy = load_sharadar_prices(symbols, start, end)
    prices = to_wide(tidy)
    print(f"Price matrix: {prices.shape[0]} trading days x {prices.shape[1]} symbols "
          f"({prices.index[0].date()}..{prices.index[-1].date()})")

    # The engine's STEP 6 reads the PIT feature store; replaying without populating
    # it starves the ML/admission path (engine never trades). Ingest the full
    # OHLCV history's price-derived features ONCE — get_features(asof) returns only
    # the PIT-safe subset per cycle, so this stays leak-free.
    n_rows = ingest_prices(tidy, mode="PAPER")
    print(f"Ingested {n_rows} PIT feature rows into the store.")

    # EXACT config of the banked yfinance headline (scripts/backtest_real.py) — any
    # delta below is attributable to the price source alone.
    bt = Backtester(mode="PAPER", rebalance=rebalance, warmup=90,
                    cost_bps_per_turnover=10.0, seed=42,
                    target_vol=0.22, max_gross_leverage=2.0,
                    max_position_weight=0.20, cvar_limit=0.12,
                    signal_tilt_strength=3e-3)
    result = bt.run(prices)

    m = result.metrics
    print("\n============  TRADING ENGINE — SHARADAR REAL-DATA BACKTEST (net of cost)  ============")
    print(result.summary())
    print("-" * 84)
    print(f"  rebalances        : {result.n_rebalances}")
    print(f"  annualised return : {m.get('ann_return', 0.0):>8.2%}")
    print(f"  annualised vol    : {m.get('ann_vol', 0.0):>8.2%}")
    print(f"  Sharpe            : {m.get('sharpe', 0.0):>8.2f}")
    print(f"  Sortino           : {m.get('sortino', 0.0):>8.2f}")
    print(f"  max drawdown      : {m.get('max_drawdown', 0.0):>8.2%}")
    print(f"  Calmar            : {m.get('calmar', 0.0):>8.2f}")
    print(f"  hit rate          : {m.get('hit_rate', 0.0):>8.2%}")
    print(f"  avg turnover/rebal: {m.get('avg_turnover', 0.0):>8.2f}")
    print(f"  total cost (bps)  : {m.get('total_cost_bps', 0.0):>8.1f}")
    print(f"  final equity (x)  : {float(result.equity_curve.iloc[-1]):>8.3f}")

    # Equal-weight buy-and-hold benchmark over the same window (net of nothing).
    rets = prices.pct_change().dropna()
    bench = rets.mean(axis=1)
    bench_ann = (1.0 + bench).prod() ** (252.0 / len(bench)) - 1.0
    bench_vol = float(bench.std() * np.sqrt(252.0))
    bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0.0
    print("-" * 84)
    print(f"  benchmark (EW B&H): ann_return={bench_ann:>7.2%}  ann_vol={bench_vol:>7.2%}  "
          f"sharpe={bench_sharpe:>5.2f}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
