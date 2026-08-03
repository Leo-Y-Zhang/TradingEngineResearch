"""
Stage B research — do cross-sectional Fama-French factor LOADINGS carry a real,
deflation-surviving edge?  (honest test, not a knob-tune)

Pipeline:
  universe prices (yfinance) -> daily returns -> at each month-end compute trailing
  FF loadings per stock (PIT-safe) -> form cross-sectional candidate factors -> rank
  -> long top tercile / short bottom tercile -> next-month return net of cost ->
  strategy monthly return series -> Deflated Sharpe Ratio (deflated for the number of
  factors tried) -> verdict.

A factor only counts as a *real* edge if its DSR survives (>= 0.95) AFTER deflation
for multiple testing. Run:  python scripts/research_ff_factors.py
NOTE: hits the network (yfinance). Survivorship caveat: the universe is current large
caps (documented limitation, METH-1) — a first read, not a deployable result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.factor_ingestion import factor_loadings, load_fama_french  # noqa: E402
from research.validation import deflated_sharpe_ratio  # noqa: E402

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH", "XOM", "CVX", "COP", "PG", "KO", "PEP",
    "WMT", "HD", "MCD", "DIS", "VZ", "T", "INTC", "CSCO", "ORCL", "IBM",
]
START, END = "2015-01-01", "2026-01-01"
WINDOW = 126          # trailing days for loadings
COST_BPS = 10.0       # per-leg turnover cost
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fama_french_daily_sample.csv"


def _prices() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(UNIVERSE, start=START, end=END, progress=False, auto_adjust=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return close.dropna(axis=1, thresh=int(0.8 * len(close)))  # keep mostly-complete names


def _terc_ls(factor: pd.Series, fwd: pd.Series, cost: float) -> float | None:
    s = pd.concat([factor, fwd], axis=1).dropna()
    s.columns = ["f", "r"]
    if len(s) < 6:
        return None
    q1, q2 = s["f"].quantile([1 / 3, 2 / 3])
    longs, shorts = s[s["f"] >= q2]["r"], s[s["f"] <= q1]["r"]
    if longs.empty or shorts.empty:
        return None
    return float(longs.mean() - shorts.mean()) - cost  # equal-weight L/S, minus 1 leg cost proxy


def main() -> None:
    ff = load_fama_french(FIX)
    px = _prices()
    rets = px.pct_change()
    print(f"universe kept: {px.shape[1]} names | {px.index[0].date()}..{px.index[-1].date()}")

    rebal = px.resample("ME").last().index
    rebal = [d for d in rebal if d in px.index and d >= px.index[0] + pd.Timedelta(days=400)]
    cost = COST_BPS / 10_000.0

    factors = ["value(beta_hml)", "size(beta_smb)", "low_beta(-beta_mkt)", "momentum_12_1"]
    series: dict[str, list[float]] = {f: [] for f in factors}

    for i in range(len(rebal) - 1):
        t, t1 = rebal[i], rebal[i + 1]
        fwd = (px.loc[t1] / px.loc[t] - 1.0)
        load = factor_loadings(rets, ff, t, window=WINDOW)
        if len(load) < 9:
            continue
        bh = pd.Series({s: v["beta_hml"] for s, v in load.items()})
        bs = pd.Series({s: v["beta_smb"] for s, v in load.items()})
        bm = pd.Series({s: -v["beta_mkt"] for s, v in load.items()})
        hist = px[px.index <= t]
        mom = (hist.iloc[-21] / hist.iloc[-252] - 1.0) if len(hist) > 252 else pd.Series(dtype=float)
        for name, fac in zip(factors, [bh, bs, bm, mom]):
            r = _terc_ls(fac, fwd, cost) if len(fac) else None
            if r is not None:
                series[name].append(r)

    print(f"\nrebalances used: {len(rebal)-1}  | factors tried: {len(factors)} (DSR deflated for this)\n")
    print(f"{'factor':<22}{'n':>4}{'ann_ret':>9}{'ann_vol':>9}{'Sharpe':>8}{'DSR':>7}  verdict")
    n_trials = len(factors)
    for name in factors:
        a = np.asarray(series[name], dtype=float)
        if a.size < 12:
            print(f"{name:<22}{a.size:>4}  (insufficient)")
            continue
        ann_ret = float(a.mean() * 12)
        ann_vol = float(a.std(ddof=1) * np.sqrt(12))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        dsr = deflated_sharpe_ratio(a, n_trials=n_trials)
        verdict = "REAL edge (survives deflation)" if dsr >= 0.95 else "no robust edge"
        print(f"{name:<22}{a.size:>4}{ann_ret:>9.2%}{ann_vol:>9.2%}{sharpe:>8.2f}{dsr:>7.2f}  {verdict}")

    print("\nHonest note: long-short L/S, equal weight, ~10bps cost, current-constituent")
    print("universe (survivorship). A DSR < 0.95 means the apparent edge does NOT survive")
    print("multiple-testing/non-normality deflation — i.e. not trustworthy as alpha.")


if __name__ == "__main__":
    main()
