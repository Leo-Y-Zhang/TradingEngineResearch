"""
Stage B / #2 — combine many weak signals into one composite and test whether the
COMBINATION carries a deflation-surviving edge (the Medallion principle: alpha is in
the combination, not any single factor).

Signals (all PIT-safe, cross-sectional):
  • value      = FF HML loading (beta_hml)
  • size       = FF SMB loading (beta_smb)
  • low_beta   = -FF MKT loading
  • quality    = EDGAR ROE, ROA (filing-date PIT-safe)
  • momentum   = price 12-1

Composite = equal-weight average of cross-sectional z-scores. Long top tercile /
short bottom tercile, ~10bps cost, monthly. Judged by the real Deflated Sharpe Ratio,
deflated for the number of things tried (the 5 singles + the composite).

Run: python scripts/research_combined_alpha.py   (network: yfinance + SEC EDGAR)
Survivorship caveat (METH-1): current large caps. A first read, not deployable.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.edgar_ingestion import pit_fundamental  # noqa: E402
from data.factor_ingestion import factor_loadings, load_fama_french  # noqa: E402
from research.validation import deflated_sharpe_ratio  # noqa: E402

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH", "XOM", "CVX", "COP", "PG", "KO", "PEP",
    "WMT", "HD", "MCD", "DIS", "VZ", "T", "INTC", "CSCO", "ORCL", "IBM",
]
START, END = "2015-01-01", "2026-01-01"
TAGS = ["StockholdersEquity", "NetIncomeLoss", "Assets"]
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fama_french_daily_sample.csv"
UA = {"User-Agent": "TradingEngineResearch research 268190724+Leo-Y-Zhang@users.noreply.github.com"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read()


def _prices() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(UNIVERSE, start=START, end=END, progress=False, auto_adjust=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return close.dropna(axis=1, thresh=int(0.8 * len(close)))


def _edgar(tickers) -> pd.DataFrame:
    import json as _j
    cikmap = {str(v["ticker"]).upper(): int(v["cik_str"])
              for v in _j.loads(_get("https://www.sec.gov/files/company_tickers.json")).values()}
    rows = []
    for tk in tickers:
        cik = cikmap.get(tk)
        if not cik:
            continue
        for tag in TAGS:
            try:
                cc = _j.loads(_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"))
            except Exception:
                continue
            for _u, facts in cc.get("units", {}).items():
                for u in facts:
                    if u.get("filed") and u.get("end") and u.get("val") is not None:
                        rows.append((tk, tag, pd.Timestamp(u["filed"]), pd.Timestamp(u["end"]), float(u["val"])))
            time.sleep(0.12)   # respect SEC 10 req/s
    return pd.DataFrame(rows, columns=["ticker", "tag", "filed", "period_end", "value"])


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if len(s) < 6 or s.std(ddof=0) == 0:
        return pd.Series(dtype=float)
    return (s - s.mean()) / s.std(ddof=0)


def _terc_ls(factor: pd.Series, fwd: pd.Series, cost: float):
    d = pd.concat([factor, fwd], axis=1).dropna()
    d.columns = ["f", "r"]
    if len(d) < 9:
        return None
    q1, q2 = d["f"].quantile([1 / 3, 2 / 3])
    lo, sh = d[d["f"] >= q2]["r"], d[d["f"] <= q1]["r"]
    if lo.empty or sh.empty:
        return None
    return float(lo.mean() - sh.mean()) - cost


def main() -> None:
    ff = load_fama_french(FIX)
    px = _prices()
    rets = px.pct_change()
    print(f"universe: {px.shape[1]} names {px.index[0].date()}..{px.index[-1].date()} | fetching EDGAR…")
    funds = _edgar(list(px.columns))
    print(f"EDGAR rows: {len(funds)} for {funds['ticker'].nunique()} names")

    rebal = [d for d in px.resample("ME").last().index
             if d in px.index and d >= px.index[0] + pd.Timedelta(days=400)]
    cost = 10.0 / 10_000.0
    names = ["value", "size", "low_beta", "roe", "roa", "momentum", "COMPOSITE"]
    series: dict[str, list[float]] = {n: [] for n in names}

    for i in range(len(rebal) - 1):
        t, t1 = rebal[i], rebal[i + 1]
        fwd = px.loc[t1] / px.loc[t] - 1.0
        load = factor_loadings(rets, ff, t, window=126)
        if len(load) < 9:
            continue
        raw = {
            "value": pd.Series({s: v["beta_hml"] for s, v in load.items()}),
            "size": pd.Series({s: v["beta_smb"] for s, v in load.items()}),
            "low_beta": pd.Series({s: -v["beta_mkt"] for s, v in load.items()}),
            "roe": pd.Series({s: (lambda e, n: n / e if e else np.nan)(
                pit_fundamental(funds, s, "StockholdersEquity", t) or 0.0,
                pit_fundamental(funds, s, "NetIncomeLoss", t) or np.nan) for s in px.columns}),
            "roa": pd.Series({s: (lambda a, n: n / a if a else np.nan)(
                pit_fundamental(funds, s, "Assets", t) or 0.0,
                pit_fundamental(funds, s, "NetIncomeLoss", t) or np.nan) for s in px.columns}),
            "momentum": (px.loc[:t].iloc[-21] / px.loc[:t].iloc[-252] - 1.0) if len(px.loc[:t]) > 252 else pd.Series(dtype=float),
        }
        zs = {k: _zscore(v) for k, v in raw.items()}
        comp = pd.concat([z for z in zs.values() if len(z)], axis=1).mean(axis=1, skipna=True)
        for k in names[:-1]:
            r = _terc_ls(raw[k], fwd, cost) if len(raw[k]) else None
            if r is not None:
                series[k].append(r)
        rc = _terc_ls(comp, fwd, cost) if len(comp) else None
        if rc is not None:
            series["COMPOSITE"].append(rc)

    print(f"\nrebalances: {len(rebal)-1} | trials for DSR deflation: {len(names)}\n")
    print(f"{'signal':<12}{'n':>4}{'ann_ret':>9}{'Sharpe':>8}{'DSR':>7}  verdict")
    for n in names:
        a = np.asarray(series[n], dtype=float)
        if a.size < 12:
            print(f"{n:<12}{a.size:>4}  (insufficient)")
            continue
        ann = float(a.mean() * 12)
        vol = float(a.std(ddof=1) * np.sqrt(12))
        sh = ann / vol if vol > 0 else 0.0
        dsr = deflated_sharpe_ratio(a, n_trials=len(names))
        v = "REAL edge (survives deflation)" if dsr >= 0.95 else "no robust edge"
        tag = "  <<< COMPOSITE" if n == "COMPOSITE" else ""
        print(f"{n:<12}{a.size:>4}{ann:>9.2%}{sh:>8.2f}{dsr:>7.2f}  {v}{tag}")


if __name__ == "__main__":
    main()
