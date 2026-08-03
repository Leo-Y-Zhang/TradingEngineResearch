"""Broad-universe alpha research (S&P 500 breadth).

Breadth is the lever that turns a weak signal into a tradeable edge
(IR = IC x sqrt(breadth)). This fetches the full S&P 500, measures each signal's
OOS information coefficient, then backtests a COMPOSITE-factor long-only top-quintile
and a long-short, net of cost, vs an equal-weight benchmark.

CAVEAT: uses CURRENT constituents (survivorship bias inflates results) — so weak
results here are a strong negative; strong results must be discounted. Honest, OOS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FALLBACK = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "PG", "KO", "WMT", "HD", "MRK"]
START, END = "2015-01-01", "2024-12-31"
FWD = 21
STEP = 21
COST_BPS = 10.0


def get_tickers() -> list[str]:
    for url in ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",):
        try:
            df = pd.read_csv(url)
            return [str(s).replace(".", "-") for s in df["Symbol"].tolist()]
        except Exception:  # noqa: BLE001
            pass
    return FALLBACK


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


def main() -> int:
    import yfinance as yf

    tickers = get_tickers()
    print(f"Fetching {len(tickers)} tickers {START}..{END} (large download)...")
    raw = yf.download(tickers, start=START, end=END, interval="1d", progress=False, auto_adjust=True)
    px = raw["Close"].sort_index().dropna(axis=1)          # keep full-history names
    n_days, n_sym = px.shape
    print(f"Clean panel: {n_days} days x {n_sym} symbols ({px.index[0].date()}..{px.index[-1].date()})\n")

    def signals(i: int) -> dict[str, pd.Series]:
        p = px.iloc[: i + 1]
        last = p.iloc[-1]
        daily = p.pct_change()
        return {
            "mom_12_1": p.iloc[-1 - 21] / p.iloc[-1 - 252] - 1.0,
            "mom_6_1": p.iloc[-1 - 21] / p.iloc[-1 - 126] - 1.0,
            "rev_1m": -(last / p.iloc[-1 - 21] - 1.0),
            "low_vol": -daily.iloc[-126:].std(),
            "trend_200": last / p.iloc[-200:].mean() - 1.0,
        }

    ic_lists: dict[str, list[float]] = {}
    comp_long, comp_ls, bench = [], [], []
    for i in range(252, n_days - FWD, STEP):
        fwd = px.iloc[i + FWD] / px.iloc[i] - 1.0
        sig = signals(i)
        for name, s in sig.items():
            c = s.dropna().index.intersection(fwd.dropna().index)
            if len(c) >= 50:
                ic = s[c].rank().corr(fwd[c].rank())
                if np.isfinite(ic):
                    ic_lists.setdefault(name, []).append(float(ic))
        # composite = mean of z-scored momentum + low-vol + reversal (the diversified factor)
        comp = (zscore(sig["mom_12_1"]) + zscore(sig["mom_6_1"]) + zscore(sig["low_vol"])
                + 0.5 * zscore(sig["rev_1m"]))
        c = comp.dropna().index.intersection(fwd.dropna().index)
        if len(c) < 50:
            continue
        cv, fv = comp[c], fwd[c]
        q = cv.rank(pct=True)
        top, bot = fv[q >= 0.8], fv[q <= 0.2]
        comp_long.append(float(top.mean() - COST_BPS * 1e-4))
        comp_ls.append(float(top.mean() - bot.mean() - 2 * COST_BPS * 1e-4))
        bench.append(float(fv.mean()))

    ppy = 252.0 / STEP

    def stats(x: list[float]) -> tuple[float, float, float]:
        a = np.array(x)
        ann = (1 + a).prod() ** (ppy / len(a)) - 1.0
        sharpe = a.mean() / (a.std(ddof=1) + 1e-12) * np.sqrt(ppy)
        eq = (1 + a).cumprod()
        mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        return ann, sharpe, mdd

    print(f"{'signal':>12} {'mean_IC':>9} {'t_stat':>8} {'n':>4}")
    print("-" * 38)
    for name, ics in sorted(ic_lists.items(), key=lambda kv: -abs(np.mean(kv[1]) / (np.std(kv[1]) + 1e-12))):
        a = np.array(ics)
        t = a.mean() / (a.std(ddof=1) + 1e-12) * np.sqrt(len(a))
        flag = "  <-- significant" if abs(t) >= 2 else ""
        print(f"{name:>12} {a.mean():>9.4f} {t:>8.2f} {len(a):>4}{flag}")

    cl, bl = stats(comp_long), stats(bench)
    ls = stats(comp_ls)
    print("\n=========  COMPOSITE FACTOR STRATEGY (net of cost, OOS, survivorship-biased)  =========")
    print(f"  composite top-quintile (long): ann={cl[0]:>7.2%}  sharpe={cl[1]:>5.2f}  maxdd={cl[2]:>7.2%}")
    print(f"  composite long-short         : ann={ls[0]:>7.2%}  sharpe={ls[1]:>5.2f}  maxdd={ls[2]:>7.2%}")
    print(f"  equal-weight benchmark       : ann={bl[0]:>7.2%}  sharpe={bl[1]:>5.2f}  maxdd={bl[2]:>7.2%}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
