"""Alpha-slice step 2 — does a LEARNED combination of richer features carry a
deflation-surviving edge?

Builds a PIT-safe cross-sectional feature panel (Fama-French factor loadings + price/volume
+ optional EDGAR fundamentals), then runs ``research.alpha_factory.learn_signal_weights``
(ridge, purged walk-forward) and judges it by the real Deflated Sharpe Ratio. Compares
against the NAIVE equal-weight composite baseline. Honest by construction: a combination
that does not clear DSR >= 0.95 + selection_rule is reported as no robust edge.

Run: python scripts/research_learned_alpha.py   (network: yfinance + Ken French + SEC EDGAR)
Survivorship caveat (METH-1): current large caps — a first read, not deployable.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.edgar_ingestion import pit_fundamental  # noqa: E402
from data.factor_ingestion import (  # noqa: E402
    factor_loadings,
    fetch_fama_french_factors,
    load_fama_french,
)
from research.alpha_factory import learn_signal_weights  # noqa: E402
from research.validation import PurgedWalkForwardSplitter, deflated_sharpe_ratio, selection_rule  # noqa: E402

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH", "XOM", "CVX", "COP", "PG", "KO", "PEP",
    "WMT", "HD", "MCD", "DIS", "VZ", "T", "INTC", "CSCO", "ORCL", "IBM",
]
# --broad: ~500-name S&P 500 cross-section (survivorship-biased — caveat METH-1) to test
# whether the LEARNED combination survives deflation with far more cross-sectional power.
# EDGAR fundamentals are skipped in broad mode (per-name fetch is too slow at 500 names);
# features are factor loadings + price/volume only.
BROAD = "--broad" in sys.argv[1:]
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
START, END = "2015-01-01", "2026-01-01"
TAGS = ["StockholdersEquity", "NetIncomeLoss", "Assets"]
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fama_french_daily_sample.csv"
UA = {"User-Agent": "TradingEngineResearch research 268190724+Leo-Y-Zhang@users.noreply.github.com"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read()


def _universe() -> list[str]:
    if not BROAD:
        return UNIVERSE
    try:
        df = pd.read_csv(SP500_CSV)
        return [str(s).replace(".", "-") for s in df["Symbol"].tolist()]
    except Exception as exc:  # noqa: BLE001
        print(f"  S&P 500 constituents fetch failed ({exc}); falling back to the 30-name set")
        return UNIVERSE


def _prices(univ: list[str]) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(univ, start=START, end=END, progress=False, auto_adjust=True)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    return close.dropna(axis=1, thresh=int(0.8 * len(close)))


def _edgar(tickers) -> pd.DataFrame:
    try:
        cikmap = {str(v["ticker"]).upper(): int(v["cik_str"])
                  for v in json.loads(_get("https://www.sec.gov/files/company_tickers.json")).values()}
    except Exception as exc:  # noqa: BLE001
        print(f"  EDGAR cik map failed ({exc}); proceeding without fundamentals")
        return pd.DataFrame(columns=["ticker", "tag", "filed", "period_end", "value"])
    rows = []
    for tk in tickers:
        cik = cikmap.get(tk)
        if not cik:
            continue
        for tag in TAGS:
            try:
                cc = json.loads(_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"))
            except Exception:  # noqa: BLE001
                continue
            for _u, facts in cc.get("units", {}).items():
                for u in facts:
                    if u.get("filed") and u.get("end") and u.get("val") is not None:
                        rows.append((tk, tag, pd.Timestamp(u["filed"]), pd.Timestamp(u["end"]), float(u["val"])))
            time.sleep(0.12)
    return pd.DataFrame(rows, columns=["ticker", "tag", "filed", "period_end", "value"])


def _zscore_xs(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (one rebalance date), NaN-robust."""
    s = s.astype(float)
    good = s.dropna()
    if len(good) < 6 or good.std(ddof=0) == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - good.mean()) / good.std(ddof=0)


def _ff() -> pd.DataFrame:
    try:
        ff = fetch_fama_french_factors()
        print(f"  Fama-French: live {ff.index[0].date()}..{ff.index[-1].date()} ({len(ff)} days)")
        return ff
    except Exception as exc:  # noqa: BLE001
        print(f"  Fama-French live fetch failed ({exc}); using fixture sample")
        return load_fama_french(FIX)


def main() -> None:
    print(f"Fetching prices, Fama-French{'' if BROAD else ', EDGAR'}...  [mode={'BROAD S&P500' if BROAD else '30-name'}]")
    px = _prices(_universe())
    rets = px.pct_change()
    ff = _ff()
    funds = pd.DataFrame(columns=["ticker", "tag", "filed", "period_end", "value"]) if BROAD else _edgar(list(px.columns))
    have_funds = not funds.empty
    print(f"universe: {px.shape[1]} names {px.index[0].date()}..{px.index[-1].date()} | "
          f"EDGAR rows: {len(funds)} ({funds['ticker'].nunique() if have_funds else 0} names)")

    rebal = [d for d in px.resample("ME").last().index
             if d in px.index and d >= px.index[0] + pd.Timedelta(days=400)]
    syms = list(px.columns)
    feat_names = ["value", "size", "low_beta", "momentum", "reversal", "low_vol"]
    if have_funds:
        feat_names += ["roe", "roa"]

    panel: dict[str, pd.DataFrame] = {n: pd.DataFrame(index=rebal[:-1], columns=syms, dtype=float)
                                      for n in feat_names}
    fwd = pd.DataFrame(index=rebal[:-1], columns=syms, dtype=float)
    comp = pd.DataFrame(index=rebal[:-1], columns=syms, dtype=float)   # naive equal-weight composite

    for i in range(len(rebal) - 1):
        t, t1 = rebal[i], rebal[i + 1]
        fwd.loc[t] = (px.loc[t1] / px.loc[t] - 1.0).reindex(syms)
        load = factor_loadings(rets, ff, t, window=126)
        hist = px.loc[:t]
        raw = {
            "value": pd.Series({s: load.get(s, {}).get("beta_hml", np.nan) for s in syms}),
            "size": pd.Series({s: load.get(s, {}).get("beta_smb", np.nan) for s in syms}),
            "low_beta": pd.Series({s: -load.get(s, {}).get("beta_mkt", np.nan) for s in syms}),
            "momentum": (hist.iloc[-21] / hist.iloc[-252] - 1.0).reindex(syms) if len(hist) > 252 else pd.Series(np.nan, index=syms),
            "reversal": (-(hist.iloc[-1] / hist.iloc[-21] + 1e-12) + 1.0).reindex(syms) if len(hist) > 21 else pd.Series(np.nan, index=syms),
            "low_vol": (-rets.loc[:t].tail(63).std()).reindex(syms) if len(rets.loc[:t]) > 63 else pd.Series(np.nan, index=syms),
        }
        if have_funds:
            raw["roe"] = pd.Series({s: (lambda e, n: n / e if e else np.nan)(
                pit_fundamental(funds, s, "StockholdersEquity", t) or 0.0,
                pit_fundamental(funds, s, "NetIncomeLoss", t) or np.nan) for s in syms})
            raw["roa"] = pd.Series({s: (lambda a, n: n / a if a else np.nan)(
                pit_fundamental(funds, s, "Assets", t) or 0.0,
                pit_fundamental(funds, s, "NetIncomeLoss", t) or np.nan) for s in syms})
        zs = {k: _zscore_xs(v) for k, v in raw.items()}
        for k in feat_names:
            panel[k].loc[t] = zs[k].reindex(syms).to_numpy()
        comp.loc[t] = pd.concat([zs[k] for k in feat_names], axis=1).mean(axis=1, skipna=True).reindex(syms).to_numpy()

    n_dates = len(rebal) - 1
    test = max(3, n_dates // 5)
    splitter = PurgedWalkForwardSplitter(train_size=max(6, n_dates - 3 * test),
                                         valid_size=test, test_size=test, embargo_size=1, label_horizon=1)
    n_trials = len(feat_names) + 2     # features tried + naive composite + learned config

    weights, res = learn_signal_weights(panel, fwd, splitter=splitter, n_trials=n_trials, cost_drag_bps=10.0,
                                        periods_per_year=12)

    # Naive equal-weight composite baseline: per-date dollar-neutral LS return.
    comp_ret = []
    for t in comp.index:
        c = comp.loc[t].astype(float).to_numpy()
        y = fwd.loc[t].astype(float).to_numpy()
        m = np.isfinite(c) & np.isfinite(y)
        if int(m.sum()) < 3:
            continue
        w = c[m] - c[m].mean()
        d = np.abs(w).sum()
        if d > 0:
            comp_ret.append(float((w / d) @ y[m]) - 10.0 / 1e4)
    comp_ret_a = np.asarray(comp_ret, dtype=float)
    comp_dsr = deflated_sharpe_ratio(comp_ret_a, n_trials=n_trials) if comp_ret_a.size >= 4 else 0.0
    comp_sh = (float(comp_ret_a.mean() / comp_ret_a.std(ddof=1) * np.sqrt(12))
               if comp_ret_a.size > 1 and comp_ret_a.std(ddof=1) > 0 else 0.0)

    print("\n================  LEARNED vs NAIVE alpha (real-data, net of 10bps, monthly)  ================")
    print(f"features ({len(feat_names)}): {', '.join(feat_names)}")
    print(f"rebalances: {n_dates} | walk-forward test={test} | DSR trials={n_trials}")
    print("-" * 80)
    print("LEARNED ridge combination:")
    print(f"  mean_ic={res.mean_ic:+.4f}  rank_ic={res.mean_rank_ic:+.4f}  net_Sharpe={res.sharpe_net:.2f}  "
          f"DSR={res.deflated_sharpe_ratio:.3f}  stability={res.stability_score:.2f}")
    print("  weights: " + ", ".join(f"{k}={weights.get(k, 0.0):+.3f}" for k in feat_names))
    print(f"  selection_rule -> {'PASS (deployable)' if selection_rule(res) else 'FAIL (default-deny)'}")
    print("-" * 80)
    print(f"NAIVE equal-weight composite:  net_Sharpe={comp_sh:.2f}  DSR={comp_dsr:.3f}")
    print("-" * 80)
    verdict = ("LEARNED combination carries a deflation-surviving edge"
               if selection_rule(res) else
               "no robust edge survives deflation (honest: consistent with the no-easy-alpha prior)")
    print(f"VERDICT: {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
