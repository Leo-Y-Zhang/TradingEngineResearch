"""Diagnose WHERE the engine's book collapses to zero on real data.

Replays the engine cycle-by-cycle (like the harness) but prints the intermediate
state each cycle: regime/crisis, model-fitted, sample predictions, admitted count,
optimiser book size, risk-gate approval, blocked. Pinpoints whether the flat book
is an ML cold-start, an admission veto, an optimiser collapse, or a risk block.
"""

from __future__ import annotations

import sys

import pandas as pd

import core.ml_return_model as mlm
from backtesting.harness import Backtester, _reset_engine_state
from core.engine.engine import TradingEngine
from data.price_ingestion import fetch_prices, ingest_prices

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"]


def to_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    return tidy.pivot(index="date", columns="symbol", values="close").sort_index().dropna(how="any")


def main() -> int:
    tidy = fetch_prices(DEFAULT_SYMBOLS, "2016-01-01", "2024-12-31")
    prices = to_wide(tidy)
    ingest_prices(tidy, mode="PAPER")

    bt = Backtester(mode="PAPER", rebalance="M", warmup=90)
    dates = bt._rebalance_dates(prices.index)
    _reset_engine_state(42)
    engine = TradingEngine(mode="PAPER", capital_gbp=1_000_000.0)

    book: dict = {}
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"{'date':10} {'regime':7} {'crisis':7} blk adm  sum|w| {'fitted':6} {'approved':8} ER_sample")
    for t in dates[:n]:
        inp = bt._build_inputs(prices.loc[:t], t, book, 0.0)
        r = engine.run_cycle(inp)
        adm = [s for s, d in r.decisions.items() if getattr(d, "take_trade", False)]
        tw = r.target_weights or {}
        model = mlm.get_model()
        fitted = getattr(model, "is_fitted", getattr(model, "_is_fitted", "?"))
        ers = {s: round(float(v[0]), 4) for s, v in list((r.predictions or {}).items())[:2]}
        approved = None
        try:
            approved = not r.risk_snapshot.get("kill_switch_active", False)
        except Exception:
            pass
        print(f"{str(t.date()):10} {r.regime_label[:7]:7} {str(r.crisis.get('level',''))[:7]:7} "
              f"{int(r.blocked)}  {len(adm):3} {sum(abs(x) for x in tw.values()):6.3f} "
              f"{str(fitted):6} {str(approved):8} {ers}")
        if not r.blocked:
            ach = r.achieved_weights
            src = ach if ach is not None else r.target_weights
            book = {s: float(w) for s, w in (src or {}).items()}
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
