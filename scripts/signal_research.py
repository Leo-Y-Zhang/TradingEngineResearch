"""Alpha signal research — measure which signals actually predict cross-sectional
forward returns OUT-OF-SAMPLE on a broad universe (the foundation of real alpha).

For a battery of parameter-light, PIT-safe candidate signals (price/volume only —
what yfinance provides), at each monthly rebalance we cross-sectionally rank the
signal and correlate it with the NEXT month's return (the Information Coefficient).
A signal with a robust, statistically significant mean IC has genuine predictive
edge; one whose IC is ~0 / insignificant does not — no amount of optimiser tuning
turns the latter into alpha. Also reports the net-of-cost top-minus-bottom quintile
long-short spread. Honest, OOS, cost-aware — report the real numbers.

Usage: python scripts/signal_research.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data.price_ingestion import fetch_prices

EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ADBE", "CRM", "ORCL", "CSCO",
    "INTC", "IBM", "TXN", "QCOM", "AVGO", "JPM", "BAC", "WFC", "GS", "MS",
    "C", "AXP", "BLK", "SCHW", "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO",
    "LLY", "BMY", "AMGN", "GILD", "PG", "KO", "PEP", "WMT", "HD", "MCD",
    "NKE", "COST", "LOW", "SBUX", "TGT", "XOM", "CVX", "CAT", "BA", "HON",
    "UNP", "GE", "MMM", "DIS", "VZ", "T", "CMCSA", "NEE",
]
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "LTC-USD",
    "BCH-USD", "LINK-USD", "XLM-USD", "TRX-USD", "ETC-USD", "XMR-USD", "EOS-USD",
    "AAVE-USD", "ATOM-USD", "ALGO-USD", "XTZ-USD",
]

_PRESET = (sys.argv[1].lower() if len(sys.argv) > 1 else "equity")
if _PRESET == "crypto":
    UNIVERSE = CRYPTO_UNIVERSE
    START, END = "2019-01-01", "2024-12-31"
else:
    UNIVERSE = EQUITY_UNIVERSE
    START, END = "2015-01-01", "2024-12-31"
FWD = 21          # forward-return horizon (trading days ≈ 1 month)
STEP = 21         # rebalance cadence
COST_BPS = 10.0   # round-trip cost for the long-short spread


def _wide(tidy: pd.DataFrame, field: str) -> pd.DataFrame:
    return tidy.pivot(index="date", columns="symbol", values=field).sort_index()


def signals_at(px: pd.DataFrame, vol: pd.DataFrame, i: int) -> dict[str, pd.Series]:
    """PIT cross-sectional signals using only data up to row i."""
    p = px.iloc[: i + 1]
    last = p.iloc[-1]
    ret1m = p.iloc[-1] / p.iloc[-1 - 21] - 1.0 if i >= 21 else last * np.nan
    daily = p.pct_change()
    out: dict[str, pd.Series] = {}
    if i >= 252:
        out["mom_12_1"] = p.iloc[-1 - 21] / p.iloc[-1 - 252] - 1.0          # 12m skip-1m momentum
        out["prox_52w_high"] = last / p.iloc[-252:].max()                    # nearness to 52w high
        out["trend_200"] = last / p.iloc[-200:].mean() - 1.0                 # long trend
    if i >= 126:
        out["mom_6_1"] = p.iloc[-1 - 21] / p.iloc[-1 - 126] - 1.0           # 6m skip-1m momentum
        out["low_vol"] = -daily.iloc[-126:].std()                           # low-volatility anomaly
        out["skew_126"] = -daily.iloc[-126:].skew()                         # low-skew preference
    if i >= 21:
        out["rev_1m"] = -ret1m                                              # short-term reversal
    if i >= 63 and vol is not None:
        dollar = (px * vol).iloc[-63:].mean()
        out["illiq"] = -dollar                                              # liquidity (prefer liquid)
        out["vol_trend"] = vol.iloc[-21:].mean() / vol.iloc[-63:].mean()    # rising volume
    return out


def main() -> int:
    print(f"Fetching {len(UNIVERSE)} symbols {START}..{END} ...")
    tidy = fetch_prices(UNIVERSE, START, END)
    px = _wide(tidy, "close").dropna(how="all")
    vol = _wide(tidy, "volume").reindex_like(px)
    # keep symbols with full history (clean cross-section)
    px = px.dropna(axis=1)
    vol = vol[px.columns]
    n_days, n_sym = px.shape
    print(f"Clean panel: {n_days} days x {n_sym} symbols ({px.index[0].date()}..{px.index[-1].date()})\n")

    ic_lists: dict[str, list[float]] = {}
    ls_lists: dict[str, list[float]] = {}
    for i in range(252, n_days - FWD, STEP):
        fwd = px.iloc[i + FWD] / px.iloc[i] - 1.0
        sigs = signals_at(px, vol, i)
        for name, s in sigs.items():
            common = s.dropna().index.intersection(fwd.dropna().index)
            if len(common) < 15:
                continue
            sv, fv = s[common], fwd[common]
            ic = sv.rank().corr(fv.rank())                                  # Spearman IC
            if np.isfinite(ic):
                ic_lists.setdefault(name, []).append(float(ic))
            # top-minus-bottom quintile spread, net of round-trip cost
            q = sv.rank(pct=True)
            top, bot = fv[q >= 0.8], fv[q <= 0.2]
            if len(top) and len(bot):
                ls_lists.setdefault(name, []).append(float(top.mean() - bot.mean() - 2 * COST_BPS * 1e-4))

    periods_per_year = 252.0 / STEP
    rows = []
    for name, ics in ic_lists.items():
        arr = np.array(ics)
        mean_ic = arr.mean()
        t_stat = mean_ic / (arr.std(ddof=1) + 1e-12) * np.sqrt(len(arr))
        ls = np.array(ls_lists.get(name, [0.0]))
        ls_ann = ls.mean() * periods_per_year
        ls_sharpe = (ls.mean() / (ls.std(ddof=1) + 1e-12)) * np.sqrt(periods_per_year)
        rows.append((name, mean_ic, t_stat, len(arr), ls_ann, ls_sharpe))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    print(f"{'signal':>14} {'mean_IC':>9} {'t_stat':>8} {'n':>4} {'LS_ann':>8} {'LS_Sharpe':>10}")
    print("-" * 60)
    for name, ic, t, n, lsa, lss in rows:
        flag = "  <-- significant" if abs(t) >= 2.0 else ""
        print(f"{name:>14} {ic:>9.4f} {t:>8.2f} {n:>4} {lsa:>7.2%} {lss:>10.2f}{flag}")
    print("\nIC t-stat >= 2 ~= statistically real edge on this universe/period (OOS, parameter-light).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
