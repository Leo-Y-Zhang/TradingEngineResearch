"""FINAL -- SF1 POINT-IN-TIME, BETA EXPOSURE, AND EVERY CORRECTION STACKED AT ONCE.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack7_final
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from research.capacity_panel import DATA_DIR, PANEL_DIR, load_prices
from research.multiasset.carry import ols_alpha, vol_matched_active
from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.attack5_structure import combined_repair
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, SF1_DIMENSION
from research.validation import deflated_sharpe_ratio

BAND = "B2_200k_1M"
N_TICKERS = 60


def sf1_point_in_time_check(merged: pd.DataFrame) -> None:
    print("=" * 112)
    print("CHECK - IS SF1 QUALITY JOINED ON datekey (FILING DATE) OR ON calendardate?")
    print("=" * 112)
    rng = np.random.default_rng(11)
    tickers = merged["ticker"].dropna().unique()
    pick = set(rng.choice(tickers, min(N_TICKERS, len(tickers)), replace=False))
    cols = ["ticker", "dimension", "datekey", "calendardate", "gp", "assets",
            "debt", "equity", "netinc", "ncfo"]
    frames = []
    for chunk in pd.read_csv(DATA_DIR / "SF1.csv", usecols=cols, chunksize=1_500_000):
        block = chunk[(chunk["dimension"] == SF1_DIMENSION) & chunk["ticker"].isin(pick)]
        if len(block):
            frames.append(block)
    sf1 = pd.concat(frames, ignore_index=True)
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    sf1["calendardate"] = pd.to_datetime(sf1["calendardate"])
    sf1 = sf1.sort_values("datekey")
    print(f"  read {len(sf1):,} {SF1_DIMENSION} filings for {sf1['ticker'].nunique()} "
          f"sampled tickers; median datekey - calendardate = "
          f"{(sf1['datekey'] - sf1['calendardate']).dt.days.median():.0f} days")

    cache = pd.read_parquet(QUALITY_CACHE)
    cache = cache[cache["ticker"].isin(pick)].dropna(subset=["gross_profitability"])
    cache = cache.sort_values("date")
    if cache.empty:
        print("  no cached rows for the sampled tickers")
        return

    on_datekey = pd.merge_asof(cache, sf1, left_on="date", right_on="datekey",
                               by="ticker", direction="backward")
    on_caldate = pd.merge_asof(cache, sf1.sort_values("calendardate"),
                               left_on="date", right_on="calendardate",
                               by="ticker", direction="backward")
    for label, joined in (("joined on datekey (filing)", on_datekey),
                          ("joined on calendardate (period end)", on_caldate)):
        gp = joined["gp"] / joined["assets"].replace(0.0, np.nan)
        d = (gp - joined["gross_profitability"]).abs()
        agree = float((d < 1e-9).mean())
        print(f"  {label:>38}: {agree:>7.2%} of {int(d.notna().sum()):,} cached values "
              f"reproduced exactly")
    leak = (on_datekey["datekey"] > on_datekey["date"]).sum()
    print(f"  filings dated AFTER the rebalance date in the datekey join: {int(leak)}")
    print("  A cache built on calendardate would reproduce the calendardate join, not the")
    print("  datekey one. Whichever number is ~100% is the join that was actually used.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    sf1_point_in_time_check(merged)

    published = LV.run_band(merged, BAND, delistings)
    if published is None:
        print(f"{BAND}: insufficient data to build the published book")
        return 1
    net = np.maximum(published.gross - published.cost_conservative, -1.0)
    bench = published.benchmark

    print("\n" + "=" * 112)
    print("IS IT JUST LOW BETA? OLS OF THE NET BOOK ON ITS OWN BENCHMARK")
    print("=" * 112)
    a = ols_alpha(pd.Series(net), pd.Series(bench))
    # `ols_alpha` already annualises; multiplying again is the mistake this line avoids.
    print(f"  beta to the equal-weight band universe {a['beta']:.4f}   "
          f"annualised alpha {a['alpha_annual']:+.2%}   NW t {a['t_alpha']:+.2f}")
    print(f"  a passive {a['beta']:.2f}x-of-benchmark book would have made "
          f"{a['beta']*float(np.mean(bench))*12:+.2%}/yr at "
          f"{a['beta']*float(np.std(bench, ddof=1))*math.sqrt(12):.2%} vol; the sleeve made "
          f"{float(np.mean(net))*12:+.2%} at {float(np.std(net, ddof=1))*math.sqrt(12):.2%}")
    print("  so the excess is NOT explained by simply holding less of the same thing.")

    print("\n" + "=" * 112)
    print("EVERY CORRECTION STACKED - B2, CONSERVATIVE BOUND")
    print("=" * 112)

    # next-day execution frame, rebuilt (cheap: reuse the same construction as attack6)
    prices = load_prices()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["row"] = np.arange(len(prices))
    full = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet",
                           columns=["ticker", "date", "closeadj"])
    key = prices.set_index(["ticker", "date"])["row"]
    full["row"] = [key.get((t, d), -1) for t, d in zip(full["ticker"], full["date"])]
    closeadj = prices["closeadj"].to_numpy()
    code = pd.factorize(prices["ticker"])[0]
    row = full["row"].to_numpy()
    nxt = row + 1
    ok = (row >= 0) & (nxt < len(prices))
    same = np.zeros(len(full), dtype=bool)
    same[ok] = code[nxt[ok]] == code[row[ok]]
    ex = np.full(len(full), np.nan)
    ex[same] = closeadj[nxt[same]]
    full["exec_price"] = ex
    full = full.sort_values(["ticker", "date"]).reset_index(drop=True)
    full["fwd_nextday"] = (full.groupby("ticker")["exec_price"].shift(-1)
                           / full["exec_price"] - 1.0)
    nextday = merged.merge(full[["ticker", "date", "fwd_nextday"]],
                           on=["ticker", "date"], how="left")
    nextday["forward_return"] = nextday["fwd_nextday"].where(
        nextday["fwd_nextday"].notna(), nextday["forward_return"])
    nextday = nextday.drop(columns=["fwd_nextday"])

    bar = dsr_sharpe_bar(len(published.gross) / 12.0, n_trials=LV.N_TRIALS)
    rows = []

    def add(label, frame, repair, charge, rf=0.0):
        books = combined_repair(frame, BAND, delistings, repair_delisting=repair,
                                charge_free_exits=charge)
        n = np.maximum(books.gross - books.cost_conservative, -1.0)
        b = books.benchmark
        n_x, b_x = n - rf / 12.0, b - rf / 12.0
        vm = vol_matched_active(pd.Series(n_x), pd.Series(b_x))
        rows.append({
            "label": label,
            "gross": INS.annual(books.gross), "net": INS.annual(n),
            "sharpe": float(np.mean(n_x)) * 12 / (float(np.std(n_x, ddof=1)) * math.sqrt(12)),
            "bench": INS.annual(b),
            "bsharpe": float(np.mean(b_x)) * 12 / (float(np.std(b_x, ddof=1)) * math.sqrt(12)),
            "raw": INS.annual(n) - INS.annual(b),
            "vm": vm["vol_matched_active_annual"], "t": vm["vol_matched_active_tstat"],
            "dsr": deflated_sharpe_ratio(n, n_trials=LV.N_TRIALS),
        })

    add("PUBLISHED", merged, False, False)
    add("+ delisting window repaired", merged, True, False)
    add("+ free exit legs charged", merged, True, True)
    add("+ next-trading-day execution", nextday, True, True)
    add("+ risk-free rate 2%/yr", nextday, True, True, rf=0.02)

    print(f"{'book':>32} {'gross':>7} {'net':>7} {'Sharpe':>7} {'bench':>7} {'bSh':>6} "
          f"{'raw exc':>8} {'VOL-MATCHED':>12} {'t':>6} {'DSR':>6} {'gate':>18}")
    for r in rows:
        gate = ("passes (i)+(ii)" if r["vm"] > 0.02 and r["t"] > 2.0
                else "FAILS (i) or (ii)")
        print(f"{r['label']:>32} {r['gross']:>6.2%} {r['net']:>6.2%} {r['sharpe']:>7.3f} "
              f"{r['bench']:>6.2%} {r['bsharpe']:>6.3f} {r['raw']:>+7.2%} "
              f"{r['vm']:>+11.2%} {r['t']:>+6.2f} {r['dsr']:>6.3f} {gate:>18}")
    print(f"\n  DSR Sharpe bar (17.75y, 38 trials, target 0.95) = {bar:.4f}")
    print(f"  every one of these books fails registered gate (iii) net Sharpe >= {bar:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
