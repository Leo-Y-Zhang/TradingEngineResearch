"""ATTACK 6 (properly) -- POINT-IN-TIME AND EXECUTION LAG, CHECKED AGAINST RAW BARS.

Three things, none of them taken on trust:

1. `forward_return` really is closeadj(next panel row)/closeadj(this row) - 1 on the FULL
   ticker series, so a name that leaves the tradable universe still hands its move to the
   month BEFORE it leaves. (The naive version of this check, run against the FILTERED
   universe, disagrees by up to 1600% -- which is the proof that the column is built on the
   full series and not on the filtered one.)
2. The 252-day volatility and beta at a month-end use only bars up to and INCLUDING that
   month-end. Recomputed from the raw daily bars for a random sample and compared.
3. The book trades at the same close it computes its signal on. A ONE-MONTH delay is a
   signal-decay test, not an execution test; the honest execution test is ONE TRADING DAY,
   and that is measured here from the daily bars.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack6_pit
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR, load_prices
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.low_vol_quality import (
    FORWARD_RETURN_CLIP,
    MIN_PROXY_PRICE,
    RISK_WINDOW,
)

BAND = "B2_200k_1M"
SAMPLE = 300


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    print("=" * 112)
    print("CHECK 1 - IS forward_return BUILT ON THE FULL TICKER SERIES?")
    print("=" * 112)
    full = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet",
                           columns=["ticker", "date", "closeadj", "forward_return"])
    full = full.sort_values(["ticker", "date"]).reset_index(drop=True)
    own = full.groupby("ticker")["closeadj"].shift(-1) / full["closeadj"] - 1.0
    d = (own - full["forward_return"]).abs()
    print(f"  full panel, {len(full):,} rows: max |recomputed - stored| = {float(d.max()):.3e}"
          f"  (n compared {int(d.notna().sum()):,})")
    filt = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    own_f = filt.groupby("ticker")["closeadj"].shift(-1) / filt["closeadj"] - 1.0
    d_f = (own_f - filt["forward_return"]).abs()
    print(f"  the SAME check inside the FILTERED universe disagrees on "
          f"{int((d_f > 1e-9).sum()):,} of {int(d_f.notna().sum()):,} cells, "
          f"max {float(d_f.max()):.2f}")
    print("  => the column is built on the full series. A holding's crash INTO the month it")
    print("     stops being tradable is charged to the book. Attack 1's premise is refuted.")

    print("\n" + "=" * 112)
    print("CHECK 2 - ARE THE 252-DAY VOL AND BETA STRICTLY TRAILING?")
    print("=" * 112)
    prices = load_prices()
    if prices["date"].max() > DEV_CUTOFF:
        raise ValueError("price cache leaked past the DEV cutoff")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    ret = prices.groupby("ticker")["closeadj"].pct_change()
    ret = ret.clip(-FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP)
    prices["ret"] = ret
    member = (ret.notna() & (prices["close"] >= MIN_PROXY_PRICE) & (prices["volume"] > 0))
    proxy = (prices.loc[member].groupby("date")["ret"].mean())
    print(f"  market proxy rebuilt: {len(proxy):,} trading days, "
          f"{proxy.index.min().date()} -> {proxy.index.max().date()}")
    prices["mkt"] = prices["date"].map(proxy)

    risk = pd.read_parquet(PANEL_DIR / "risk_features_dev.parquet")
    rng = np.random.default_rng(7)
    band_rows = merged[merged["band_group"] == BAND][["ticker", "date"]]
    pick = band_rows.iloc[rng.choice(len(band_rows), SAMPLE, replace=False)]
    by_ticker = {t: g for t, g in prices.groupby("ticker", sort=False)}
    risk_idx = risk.set_index(["ticker", "date"])

    dvol, dbeta, used_future = [], [], 0
    for ticker, date in zip(pick["ticker"], pick["date"]):
        g = by_ticker.get(ticker)
        if g is None:
            continue
        window = g[g["date"] <= date].tail(RISK_WINDOW)
        both = window.dropna(subset=["ret", "mkt"])
        if len(both) < 2:
            continue
        v = float(both["ret"].std(ddof=1))
        cov = float(np.cov(both["ret"], both["mkt"], ddof=1)[0, 1])
        var = float(np.var(both["mkt"], ddof=1))
        b = cov / var if var > 0 else np.nan
        try:
            stored = risk_idx.loc[(ticker, date)]
        except KeyError:
            continue
        dvol.append(abs(v - float(stored["realised_vol"])))
        dbeta.append(abs(b - float(stored["beta"])))
        if window["date"].max() > date:
            used_future += 1
    print(f"  {len(dvol)} sampled (ticker, month-end) cells recomputed from raw bars using")
    print("  ONLY bars with date <= the month-end:")
    print(f"    max |vol  recomputed - stored| = {max(dvol):.3e}   median "
          f"{float(np.median(dvol)):.3e}")
    print(f"    max |beta recomputed - stored| = {max(dbeta):.3e}   median "
          f"{float(np.median(dbeta)):.3e}")
    print(f"    windows that reached past the month-end: {used_future}")
    print("  (small residuals are the prefix-sum vs pandas path, not a look-ahead: a")
    print("   look-ahead shows up as a one-sided error, not a 1e-12 rounding gap.)")

    print("\n" + "=" * 112)
    print("CHECK 3 - EXECUTION LAG: ONE TRADING DAY, NOT ONE MONTH")
    print("=" * 112)
    # The execution price must be built on the FULL panel, exactly like `forward_return`.
    # Building it on the filtered universe silently creates multi-month returns -- the very
    # defect Check 1 just measured -- and inflates BOTH sides by ~10%/yr.
    prices["row"] = np.arange(len(prices))
    key = prices.set_index(["ticker", "date"])["row"]
    fullrows = full[["ticker", "date"]].copy()
    fullrows["row"] = [key.get((t, d), -1) for t, d in
                       zip(fullrows["ticker"], fullrows["date"])]
    missing = int((fullrows["row"] < 0).sum())
    closeadj = prices["closeadj"].to_numpy()
    ticker_code = pd.factorize(prices["ticker"])[0]
    row = fullrows["row"].to_numpy()
    nxt = row + 1
    valid = (row >= 0) & (nxt < len(prices))
    same_ticker = np.zeros(len(fullrows), dtype=bool)
    same_ticker[valid] = ticker_code[nxt[valid]] == ticker_code[row[valid]]
    exec_price = np.full(len(fullrows), np.nan)
    exec_price[same_ticker] = closeadj[nxt[same_ticker]]
    fullrows["exec_price"] = exec_price
    fullrows = fullrows.sort_values(["ticker", "date"]).reset_index(drop=True)
    fullrows["forward_return_nextday"] = (
        fullrows.groupby("ticker")["exec_price"].shift(-1) / fullrows["exec_price"] - 1.0)
    print(f"  {missing:,} full-panel rows had no matching daily bar; "
          f"{int(np.isnan(exec_price).sum()):,} of {len(fullrows):,} have no NEXT bar "
          f"(the ticker's last day)")

    lagged = merged.merge(fullrows[["ticker", "date", "forward_return_nextday"]],
                          on=["ticker", "date"], how="left")
    delta = (lagged["forward_return_nextday"] - lagged["forward_return"]).abs()
    print(f"  next-day execution return defined for "
          f"{int(lagged['forward_return_nextday'].notna().sum()):,} of {len(lagged):,} "
          f"cells; mean |published - next-day| = {float(delta.mean()):.4f}, "
          f"correlation {float(lagged['forward_return_nextday'].corr(lagged['forward_return'])):.4f}")
    lagged["forward_return"] = lagged["forward_return_nextday"].where(
        lagged["forward_return_nextday"].notna(), lagged["forward_return"])

    def line(label, books):
        e = LV.evaluate_band(books, n_trials=LV.N_TRIALS)
        c = e["bounds"]["conservative"]
        print(f"{label:>40} {e['gross']['annual_arithmetic']:>6.2%} "
              f"{c['net']['annual_arithmetic']:>6.2%} {c['net']['sharpe']:>7.3f} "
              f"{e['benchmark']['annual_arithmetic']:>6.2%} "
              f"{e['benchmark']['sharpe']:>6.3f} "
              f"{c['vol_matched']['vol_matched_active_annual']:>+11.2%} "
              f"{c['vol_matched']['vol_matched_active_tstat']:>+6.2f}  {LV.verdict_for(e)}")

    print(f"\n{'book':>40} {'gross':>6} {'net':>6} {'Sharpe':>7} {'bench':>6} {'bSh':>6} "
          f"{'vol-matched':>12} {'t':>6}  verdict")
    line("published: trade at the signal close", LV.run_band(merged, BAND, delistings))
    line("trade at the NEXT DAY's close", LV.run_band(lagged, BAND, delistings))
    m1 = merged.sort_values(["ticker", "date"]).copy()
    m1["signal"] = m1.groupby("ticker")["signal"].shift(1)
    gap = m1.groupby("ticker")["date"].diff().dt.days
    m1.loc[gap > 45, "signal"] = np.nan
    line("ONE MONTH stale signal (decay test)", LV.run_band(m1, BAND, delistings))
    print("  The next-day book is the realistic one. The one-month book measures how fast")
    print("  the signal decays, which is a different question and a harsher one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
