"""Adversarial verification of the LOW-VOL / QUALITY RE-TEST result.

Not a second run of the strategy: the configuration is identical and every number here is
either an input diagnostic or a declared post-hoc robustness read on the SAME book. It
exists because four things in the headline are exactly the shape of a defect:

1. `delisting_drag_annual` is **0.000** in three of four bands. If the delisting path is
   dead, the strategy is getting a free pass on its failures -- the -112%/yr bug class.
2. The benchmark FELL from iteration 1's 10.04%/yr to 8.34%/yr when the universe was
   corrected. A benchmark that drops when it gets bigger needs explaining.
3. The vol-matched active return credits the strategy for running quiet, and it does so at
   a **zero risk-free rate**. De-levering a benchmark to 0.658x in the real world parks
   34.2% in T-bills, which earn something.
4. Every prior sleeve that looked alive turned out to live in the 2008-2011 crisis.

    .venv/Scripts/python.exe -m research.sleeves.lowvol_retest_verify
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.multiasset.carry import newey_west_tstat, vol_matched_active
from research.sleeves.low_vol_quality import build_signal
from research.sleeves.lowvol_retest import (
    BAND_ORDER,
    MONTHS_PER_YEAR,
    attach_spread_bounds,
    run_band,
)
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, load_universe

log = logging.getLogger("verify")

CRISIS = (pd.Period("2008-01", "M"), pd.Period("2011-12", "M"))
# Average 3-month US T-bill yield 1998-2015 is ~2.0%/yr. Used ONLY to show how much the
# vol-matched active return moves once de-levering earns interest; not used in any headline.
ASSUMED_RISK_FREE = 0.02


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = load_universe()
    risk = pd.read_parquet(PANEL_DIR / "risk_features_dev.parquet")
    quality = pd.read_parquet(QUALITY_CACHE)
    merged = (
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    )
    merged = build_signal(attach_spread_bounds(merged))
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    return merged, delistings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged, delistings = _load()

    print("=" * 108)
    print("CHECK 1 - IS THE DELISTING PATH ALIVE?")
    print("=" * 108)
    print(f"delistings table: {len(delistings):,} names, terminal return "
          f"min {delistings['terminal_return'].min():+.3f} "
          f"median {delistings['terminal_return'].median():+.3f} "
          f"max {delistings['terminal_return'].max():+.3f}, "
          f"{int((delistings['terminal_return'] <= -0.99).sum()):,} total losses")
    print(f"{'band':>13} {'last-obs cells':>15} {'share of universe':>18} "
          f"{'with delist rec':>16} {'in 62d window':>15} {'mean terminal':>14}")
    for band in BAND_ORDER:
        rows = merged[merged["band_group"] == band]
        last = rows[rows["forward_return"].isna()]
        has_record = last["ticker"].isin(set(delistings["ticker"]))
        dates = pd.to_datetime(
            last["ticker"].map({r.ticker: r.date for r in delistings.itertuples()}),
            errors="coerce")
        values = pd.to_numeric(
            last["ticker"].map({r.ticker: float(r.terminal_return)
                                for r in delistings.itertuples()}), errors="coerce")
        in_window = (dates.notna() & (dates > last["date"])
                     & (dates <= last["date"] + pd.Timedelta(days=62)))
        print(f"{band:>13} {len(last):>15,} {len(last)/len(rows):>17.2%} "
              f"{int(has_record.sum()):>16,} {int(in_window.sum()):>15,} "
              f"{(values[in_window].mean() if in_window.any() else float('nan')):>+14.3f}")
    print("  Both the strategy and the benchmark read the SAME `realised_return` column,")
    print("  so a last-observation cell that resolves to 0.0 costs both sides nothing.")

    print("\n" + "=" * 108)
    print("CHECK 2 - DID THE STRATEGY EVER ACTUALLY HOLD A NAME THROUGH ITS LAST BAR?")
    print("=" * 108)
    print(f"{'band':>13} {'held name-months':>17} {'held last-obs':>14} "
          f"{'held w/ terminal':>17} {'worst held terminal':>20}")
    for band in BAND_ORDER:
        books = run_band(merged, band, delistings)
        if books is None:
            log.warning("%s: insufficient data", band)
            continue
        rows = merged[merged["band_group"] == band].copy()
        rows["month"] = rows["date"].dt.to_period("M")
        key = rows.set_index(["ticker", "month"])
        held_last, held_terminal, worst = 0, 0, 0.0
        for ticker, month, _pnl in books.pnl_by_name_month:
            if (ticker, month) not in key.index:
                continue
            row = key.loc[(ticker, month)]
            if pd.isna(row["forward_return"]):
                held_last += 1
                terminal = float(row["terminal_on_exit"]) if "terminal_on_exit" in row \
                    else 0.0
                if terminal != 0.0:
                    held_terminal += 1
                    worst = min(worst, terminal)
        print(f"{band:>13} {len(books.pnl_by_name_month):>17,} {held_last:>14,} "
              f"{held_terminal:>17,} {worst:>+20.3f}")

    print("\n" + "=" * 108)
    print("CHECK 3 - WHY THE BENCHMARK FELL WHEN THE UNIVERSE GREW")
    print("=" * 108)
    print(f"{'band':>13} {'regime':>13} {'cells':>9} {'mean fwd ret':>13} "
          f"{'median $vol':>13} {'median vol':>11} {'median spread':>14}")
    for band in BAND_ORDER:
        rows = merged[merged["band_group"] == band]
        for regime in ("measured", "upper_bound"):
            block = rows[rows["spread_regime"] == regime]
            forward = block["forward_return"].clip(-1.0, 1.0)
            print(f"{band:>13} {regime:>13} {len(block):>9,} "
                  f"{float(forward.mean()) * 12:>12.2%} "
                  f"${block['median_dollar_volume'].median()/1e3:>11.0f}k "
                  f"{block['realised_vol'].median() * np.sqrt(252):>10.1%} "
                  f"{block['spread_conservative'].median() * 1e4:>12.0f}bp")

    print("\n" + "=" * 108)
    print("CHECK 4 - RISK-FREE RATE: DE-LEVERING THE BENCHMARK PARKS CASH THAT EARNS")
    print("=" * 108)
    print(f"  assumed risk-free {ASSUMED_RISK_FREE:.1%}/yr (avg 3M T-bill 1998-2015)")
    print(f"{'band':>13} {'k':>7} {'vol-matched (rf=0)':>19} {'correction':>12} "
          f"{'vol-matched (rf=2%)':>20} {'still > +2%?':>13}")
    for band in BAND_ORDER:
        books = run_band(merged, band, delistings)
        if books is None:
            log.warning("%s: insufficient data", band)
            continue
        net = np.maximum(books.gross - books.cost_conservative, -1.0)
        matched = vol_matched_active(pd.Series(net), pd.Series(books.benchmark))
        k = matched["benchmark_scale_factor"]
        correction = -(1.0 - k) * ASSUMED_RISK_FREE
        adjusted = matched["vol_matched_active_annual"] + correction
        print(f"{band:>13} {k:>7.3f} {matched['vol_matched_active_annual']:>+18.2%} "
              f"{correction:>+11.2%} {adjusted:>+19.2%} "
              f"{'YES' if adjusted > 0.02 else 'no':>13}")

    print("\n" + "=" * 108)
    print("CHECK 5 - DOES THE EDGE LIVE IN THE CRISIS? (declared diagnostic, not the verdict)")
    print("=" * 108)
    print(f"  crisis window excluded: {CRISIS[0]} .. {CRISIS[1]}")
    print(f"{'band':>13} {'n ex-crisis':>12} {'net Sharpe':>11} {'bench Sharpe':>13} "
          f"{'vol-matched':>12} {'t':>7} {'full-sample vm':>15}")
    for band in BAND_ORDER:
        books = run_band(merged, band, delistings)
        if books is None:
            log.warning("%s: insufficient data", band)
            continue
        net = np.maximum(books.gross - books.cost_conservative, -1.0)
        keep = np.array([not (CRISIS[0] <= m <= CRISIS[1]) for m in books.months])
        full = vol_matched_active(pd.Series(net), pd.Series(books.benchmark))
        sub = vol_matched_active(pd.Series(net[keep]), pd.Series(books.benchmark[keep]))
        print(f"{band:>13} {int(keep.sum()):>12} {sub['strategy_sharpe']:>11.3f} "
              f"{sub['benchmark_sharpe']:>13.3f} "
              f"{sub['vol_matched_active_annual']:>+11.2%} "
              f"{sub['vol_matched_active_tstat']:>+7.2f} "
              f"{full['vol_matched_active_annual']:>+14.2%}")

    print("\n" + "=" * 108)
    print("CHECK 6 - HOW LONG A FORWARD RETURN IS THE PANEL ACTUALLY MEASURING?")
    print("=" * 108)
    gaps = merged.sort_values(["ticker", "date"]).groupby("ticker")["date"].diff()
    gap_months = (gaps.dt.days / 30.44).round()
    print(f"  {int((gap_months == 1).sum()):,} of {int(gap_months.notna().sum()):,} "
          f"consecutive panel rows are 1 month apart "
          f"({(gap_months == 1).mean():.3%}); "
          f"{int((gap_months > 1).sum()):,} span more than one month.")
    print("  A multi-month gap makes that cell's `forward_return` span the gap. It is read")
    print("  IDENTICALLY by the strategy and the benchmark, and it is inherited from the")
    print("  panel, not introduced here.")

    print("\n" + "=" * 108)
    print("CHECK 7 - MONTHLY WIN RATE AND WORST STRETCHES (conservative bound)")
    print("=" * 108)
    print(f"{'band':>13} {'win rate':>9} {'vs bench':>9} {'worst 12m':>10} "
          f"{'best 12m':>9} {'longest underperf':>18}")
    for band in BAND_ORDER:
        books = run_band(merged, band, delistings)
        if books is None:
            log.warning("%s: insufficient data", band)
            continue
        net = np.maximum(books.gross - books.cost_conservative, -1.0)
        active = net - books.benchmark
        roll = pd.Series(net).rolling(12).apply(lambda w: (1 + w).prod() - 1.0)
        cumulative = np.cumsum(active)
        peak, longest, current = -np.inf, 0, 0
        for value in cumulative:
            if value >= peak:
                peak, current = value, 0
            else:
                current += 1
                longest = max(longest, current)
        print(f"{band:>13} {(net > 0).mean():>8.1%} {(active > 0).mean():>8.1%} "
              f"{roll.min():>9.1%} {roll.max():>8.1%} {longest:>17} mo")
        mean, _se, t = newey_west_tstat(pd.Series(active))
        log.info("  %s raw active mean %.3f%%/yr t=%.2f", band,
                 mean * MONTHS_PER_YEAR * 100, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
