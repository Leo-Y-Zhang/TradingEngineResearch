"""Bootstrap-fit + out-of-sample backtest (returns validation).

The plain replay never trades because the ML model cold-starts (predicts the
SAFE_FALLBACK 0.0 expected return → the meta-labeller admits nothing). In
production you would TRAIN the model on history before going live; this does
exactly that, with a clean split:

  • train the return model on features→forward-return pairs from [start, FIT_END)
  • replay the engine out-of-sample over [FIT_END, end], net of cost
  • compare to an equal-weight buy-and-hold benchmark over the SAME OOS window.

This isolates "does the engine, with a fitted model, actually trade and produce
net-of-cost returns?" — the core question for the returns goal. Network (yfinance).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core.ml_return_model as mlm
from backtesting import metrics as _metrics
from backtesting.harness import Backtester, _reset_engine_state
from core.engine.engine import TradingEngine
from data.feature_store import get_features
from data.price_ingestion import fetch_prices, ingest_prices

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "ABT", "UNH", "XOM", "CVX", "PG", "KO", "PEP",
    "WMT", "HD", "MCD", "NKE", "DIS", "CSCO", "INTC", "TXN", "VZ", "T",
]
START, FIT_END, END = "2016-01-01", "2019-01-01", "2024-12-31"
HORIZON = 21  # trading days ahead for the training label (≈1 month, the rebalance cadence)


def to_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    return tidy.pivot(index="date", columns="symbol", values="close").sort_index().dropna(how="any")


def _numeric_row(df: pd.DataFrame, symbol: str) -> dict:
    if symbol not in df.index:
        return {}
    return {k: float(v) for k, v in df.loc[symbol].to_dict().items()
            if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)}


def bootstrap_fit(prices: pd.DataFrame, fit_end: pd.Timestamp) -> int:
    """Train the return model on features→forward-return pairs strictly before fit_end."""
    model = mlm.get_model()
    train_idx = prices.loc[:fit_end].index
    rets = prices.pct_change()
    X, y_ret, y_vol = [], [], []
    # weekly sampling over the training window, leaving room for the forward label
    for i in range(60, len(train_idx) - HORIZON, 5):
        d = train_idx[i]
        feats = get_features(SYMBOLS, asof_time=d.to_pydatetime(), mode="PAPER")
        fwd = prices.iloc[i + HORIZON] / prices.iloc[i] - 1.0
        trail_vol = rets.iloc[max(0, i - 21):i].std()
        for sym in SYMBOLS:
            row = _numeric_row(feats, sym)
            if not row or sym not in fwd.index or not np.isfinite(fwd[sym]):
                continue
            X.append(model._vectorize(row))
            y_ret.append(float(fwd[sym]))
            y_vol.append(float(trail_vol.get(sym, 0.02)) if np.isfinite(trail_vol.get(sym, np.nan)) else 0.02)
    X = np.vstack(X)
    model.fit(X, np.array(y_ret), np.array(y_vol))
    return X.shape[0]


def main(argv: list[str]) -> int:
    print(f"Fetching {len(SYMBOLS)} symbols {START}..{END}")
    tidy = fetch_prices(SYMBOLS, START, END)
    prices = to_wide(tidy)
    ingest_prices(tidy, mode="PAPER")

    _reset_engine_state(42)                       # cold singletons (incl. model)
    n_train = bootstrap_fit(prices, pd.Timestamp(FIT_END))   # ... then pre-fit the model
    model = mlm.get_model()
    print(f"Bootstrap-fit on {n_train} samples; fitted={model._fitted}")

    # Replay OOS over [FIT_END, END], monthly, reusing the harness input builder +
    # book/cost accounting (but NOT bt.run(), which would reset our fitted model).
    bt = Backtester(mode="PAPER", rebalance="M", warmup=0, cost_bps_per_turnover=10.0)
    oos = prices.loc[FIT_END:]
    reb = [d for d in bt._rebalance_dates(prices.index) if d >= pd.Timestamp(FIT_END, tz=prices.index.tz)]
    engine = TradingEngine(mode="PAPER", capital_gbp=1_000_000.0)

    book: dict = {}
    equity, peak = 1.0, 1.0
    ret_dates, ret_vals, traded = [], [], 0
    gross_exposures: list[float] = []
    for t_now, t_next in zip(reb[:-1], reb[1:]):
        dd = 0.0 if peak <= 0 else max(0.0, 1.0 - equity / peak)
        inp = bt._build_inputs(prices.loc[:t_now], t_now, book, dd)
        r = engine.run_cycle(inp)
        new_book = dict(book) if r.blocked else {s: float(w) for s, w in (
            (r.achieved_weights if r.achieved_weights is not None else r.target_weights) or {}).items()}
        gross_exposures.append(float(sum(abs(v) for v in new_book.values())))
        if any(abs(v) > 1e-9 for v in new_book.values()):
            traded += 1
        syms = set(new_book) | set(book)
        turnover = float(sum(abs(new_book.get(s, 0.0) - book.get(s, 0.0)) for s in syms))
        fwd = prices.loc[t_next] / prices.loc[t_now] - 1.0
        gross = float(sum(new_book.get(s, 0.0) * float(fwd[s]) for s in new_book))
        net = gross - turnover * 10.0 * 1e-4
        equity *= (1.0 + net)
        peak = max(peak, equity)
        book = new_book
        ret_dates.append(t_next)
        ret_vals.append(net)

    returns = pd.Series(ret_vals, index=pd.DatetimeIndex(ret_dates))
    m = _metrics.summarize(returns, periods_per_year=12)
    bench = oos.pct_change().dropna().mean(axis=1)
    bench_ann = (1.0 + bench).prod() ** (252.0 / len(bench)) - 1.0
    bench_vol = float(bench.std() * np.sqrt(252.0))

    print("\n========  BOOTSTRAP OOS BACKTEST 2019..2024 (net of cost)  ========")
    print(f"  cycles traded     : {traded}/{len(ret_vals)}")
    print(f"  avg gross exposure: {float(np.mean(gross_exposures)) if gross_exposures else 0.0:>8.2%}")
    print(f"  ann return        : {m.get('ann_return',0.0):>8.2%}")
    print(f"  ann vol           : {m.get('ann_vol',0.0):>8.2%}")
    print(f"  Sharpe            : {m.get('sharpe',0.0):>8.2f}")
    print(f"  max drawdown      : {m.get('max_drawdown',0.0):>8.2%}")
    print(f"  final equity (x)  : {equity:>8.3f}")
    print(f"  benchmark EW B&H  : ann={bench_ann:>7.2%} vol={bench_vol:>7.2%} "
          f"sharpe={bench_ann/bench_vol if bench_vol>0 else 0:>5.2f}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
