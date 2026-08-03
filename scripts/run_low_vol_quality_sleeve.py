"""Run the LOW-VOLATILITY / QUALITY sleeve on the DEV window and print the verdict.

Pre-registration: the module docstring of ``research/sleeves/low_vol_quality.py``, written
before this was run. One configuration, run once.

Reads nothing after 2015-12-31.

    .venv/Scripts/python.exe scripts/run_low_vol_quality_sleeve.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.sharadar_ingestion import build_panel, load_sf1  # noqa: E402
from research.capacity_panel import DATA_DIR, DEV_CUTOFF, PANEL_DIR  # noqa: E402
from research.sleeves.low_vol_quality import (  # noqa: E402
    BAND_GROUPS,
    N_POSITIONS,
    QUALITY_LEGS,
    RISK_WINDOW,
    band_group,
    build_signal,
    risk_features,
    run_band,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lowvol")

RISK_CACHE = PANEL_DIR / "risk_features_dev.parquet"
QUALITY_CACHE = PANEL_DIR / "quality_art_dev.parquet"
RESULT_JSON = REPO / "research" / "sleeves" / "low_vol_quality_result.json"

# Sharadar SF1 dimension: As-Reported, Trailing Twelve Months. AR* is the ORIGINAL filing
# and is never restated, so it cannot leak a later correction backwards; TTM rather than a
# single quarter because gross profit is a flow and quarterly gp/assets is dominated by
# fiscal seasonality across firms with different year ends.
SF1_DIMENSION = "ART"


def load_dev_prices() -> pd.DataFrame:
    """Memory-lean read of the DEV price cache, with the DEV guarantee re-asserted.

    ``research.capacity_panel.load_prices`` is the canonical guarded loader, but it
    materialises the 29M-row ticker column as Python strings, which does not fit
    alongside the rolling-window arrays on this machine. This reads the SAME cache file
    (whose name encodes the cutoff), dictionary-encodes the ticker, and then re-checks the
    max date rather than trusting the filename -- the guard is duplicated, never weakened.
    """
    path = PANEL_DIR / f"prices_to_{DEV_CUTOFF.date()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_capacity_panel.py")

    columns = ["ticker", "date", "close", "closeadj", "volume"]
    table = pq.read_table(path, columns=columns)
    index = table.schema.get_field_index("ticker")
    table = table.set_column(
        index, "ticker", table["ticker"].cast(pa.dictionary(pa.int32(), pa.string()))
    )
    frame = table.to_pandas()
    del table
    gc.collect()

    if frame["date"].max() > DEV_CUTOFF:
        raise ValueError(
            f"price cache contains bars after the DEV cutoff {DEV_CUTOFF.date()}; the "
            "2016+ confirmation window must stay unfired"
        )
    return frame


def build_risk_cache(force: bool = False) -> pd.DataFrame:
    if RISK_CACHE.exists() and not force:
        log.info("loading cached trailing risk features")
        return pd.read_parquet(RISK_CACHE)

    log.info("loading DEV price cache")
    prices = load_dev_prices()
    log.info("prices: %s rows, %s tickers, %s -> %s", f"{len(prices):,}",
             f"{prices['ticker'].nunique():,}", prices["date"].min().date(),
             prices["date"].max().date())

    log.info("computing trailing %s-day realised vol and equal-weight-market beta",
             RISK_WINDOW)
    features = risk_features(prices)
    del prices
    gc.collect()

    features.to_parquet(RISK_CACHE, index=False)
    log.info("risk features: %s (ticker, month-end) cells, %s with a usable estimate",
             f"{len(features):,}", f"{features['realised_vol'].notna().sum():,}")
    return features


def build_quality_cache(grid: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Point-in-time gp/assets, debt/equity and accruals on the rebalance grid."""
    if QUALITY_CACHE.exists() and not force:
        log.info("loading cached quality features")
        return pd.read_parquet(QUALITY_CACHE)

    log.info("loading SF1 dimension %s (as-reported TTM, PIT on filing datekey)",
             SF1_DIMENSION)
    sf1 = load_sf1(DATA_DIR / "SF1.csv", dimension=SF1_DIMENSION)
    sf1 = sf1[sf1["datekey"] <= DEV_CUTOFF]
    log.info("SF1: %s filings, %s tickers, datekey %s -> %s", f"{len(sf1):,}",
             f"{sf1['ticker'].nunique():,}", sf1["datekey"].min().date(),
             sf1["datekey"].max().date())

    merged = build_panel(sf1, grid, dimension=SF1_DIMENSION)
    del sf1
    gc.collect()

    # datekey <= date is what makes this point-in-time. Assert it rather than assume it.
    filed = merged["filed_datekey"]
    leaked = int((filed.notna() & (filed > merged["date"])).sum())
    if leaked:
        raise ValueError(f"{leaked} rows carry a filing published after the rebalance date")

    assets = merged["assets"].replace(0.0, np.nan)
    equity = merged["equity"].replace(0.0, np.nan)
    out = pd.DataFrame({
        "ticker": merged["ticker"],
        "date": merged["date"],
        "gross_profitability": merged["gp"] / assets,
        "debt_to_equity": merged["debt"] / equity,
        "accruals": (merged["netinc"] - merged["ncfo"]) / assets,
    }).replace([np.inf, -np.inf], np.nan)

    out.to_parquet(QUALITY_CACHE, index=False)
    for column in QUALITY_LEGS:
        log.info("  %s: %s of %s cells populated", column,
                 f"{out[column].notna().sum():,}", f"{len(out):,}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="rebuild the cached risk and quality features")
    arguments = parser.parse_args()
    started = time.monotonic()

    panel_path = PANEL_DIR / "monthly_panel_dev.parquet"
    panel = pd.read_parquet(panel_path)
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    if panel["date"].max() > DEV_CUTOFF:
        raise ValueError("confirmation window leaked into the monthly panel")

    # Only genuinely MEASURED spreads. A name at the resolution floor is untradeable, not
    # cheap, and costing it at the floor is the bias the positive control exists to catch.
    panel["band_group"] = panel["band"].map(band_group)
    universe = panel[
        (panel["spread_regime"] == "measured") & panel["band_group"].notna()
    ].copy()
    log.info("universe: %s measured cells, %s tickers, %s -> %s",
             f"{len(universe):,}", f"{universe['ticker'].nunique():,}",
             universe["date"].min().date(), universe["date"].max().date())

    risk = build_risk_cache(force=arguments.force)
    grid = universe[["ticker", "date", "close"]].copy()
    grid["volume"] = 1.0  # build_panel wants a price frame; only ticker/date are used here
    quality = build_quality_cache(grid, force=arguments.force)

    merged = (
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    )
    log.info("coverage on the measured universe: vol %.1f%%, beta %.1f%%, "
             "gp %.1f%%, lev %.1f%%, accr %.1f%%",
             *[100 * merged[c].notna().mean() for c in
               ("realised_vol", "beta", "gross_profitability", "debt_to_equity",
                "accruals")])

    merged = build_signal(merged)
    log.info("signal defined for %s of %s cells (%.1f%%)",
             f"{merged['signal'].notna().sum():,}", f"{len(merged):,}",
             100 * merged["signal"].notna().mean())

    print("\n" + "=" * 118)
    print("LOW-VOLATILITY / QUALITY SLEEVE - DEV WINDOW (<= 2015-12-31), ONE RUN")
    print("pre-registered in research/sleeves/low_vol_quality.py (module docstring)")
    print("=" * 118)
    header = (f"{'band':>13} {'capital':>10} {'gross':>7} {'grSh':>6} {'net':>7} "
              f"{'CAGR':>7} {'vol':>7} {'SHARPE':>7} {'maxDD':>7} {'bench':>7} "
              f"{'EXCESS':>8} {'cost':>7} {'turn':>6} {'BR/yr':>7} {'n':>5}")
    print(header)
    print("-" * 118)

    results = []
    for band in BAND_GROUPS:
        result = run_band(merged, band, delistings)
        if result is None:
            print(f"{band:>13} {'insufficient measurable data':>60}")
            continue
        results.append(result)
        print(f"{band:>13} ${result.deployable_capital / 1e3:>8.0f}k "
              f"{result.gross_return_annual:>6.1%} {result.gross_sharpe:>6.2f} "
              f"{result.net_return_annual:>6.1%} {result.net_cagr:>6.1%} "
              f"{result.net_volatility:>6.1%} {result.net_sharpe:>7.2f} "
              f"{result.max_drawdown:>6.1%} {result.benchmark_return_annual:>6.1%} "
              f"{result.excess_annual:>+7.1%} {result.cost_drag_annual:>6.1%} "
              f"{result.turnover_annual:>5.1f}x {result.breadth_per_year:>6.1f} "
              f"{result.n_months:>5}")

    print("\n" + "-" * 118)
    print("COST DECOMPOSITION (annualised, one-way legs charged on entry and exit)")
    print(f"{'band':>13} {'spread':>9} {'impact':>9} {'commission+fx':>15} "
          f"{'total':>9} {'median spread':>15}")
    for result in results:
        band_rows = merged[merged["band_group"] == result.band]
        median_spread = float(band_rows["spread"].median()) * 1e4
        print(f"{result.band:>13} {result.cost_spread_annual:>8.2%} "
              f"{result.cost_impact_annual:>8.2%} "
              f"{result.cost_commission_annual:>14.2%} "
              f"{result.cost_drag_annual:>8.2%} {median_spread:>13.0f}bps")

    print("\n" + "-" * 118)
    print("BENCHMARKS (registered = whole measurable universe; diagnostic = rankable only)")
    print(f"{'band':>13} {'net':>8} {'bench (all)':>13} {'excess':>9} "
          f"{'bench (rankable)':>18} {'excess':>9}")
    for result in results:
        print(f"{result.band:>13} {result.net_return_annual:>7.1%} "
              f"{result.benchmark_return_annual:>12.1%} {result.excess_annual:>+8.1%} "
              f"{result.benchmark_rankable_annual:>17.1%} "
              f"{result.excess_vs_rankable:>+8.1%}")
    print("  Both sides book delisting terminal returns from the SAME column. The book's")
    print("  own delisting drag, annualised: "
          + ", ".join(f"{r.band} {r.delisting_drag_annual:+.2%}" for r in results))

    print("\n" + "-" * 118)
    print("DID THE SIGNAL DO WHAT IT CLAIMS? (realised risk of the book vs its universe)")
    print(f"{'band':>13} {'holding vol':>13} {'universe vol':>14} {'bench vol':>11} "
          f"{'net vol':>9} {'maxDD':>8} {'bench maxDD':>13} {'median XS':>11}")
    for result in results:
        print(f"{result.band:>13} {result.mean_holding_vol:>12.1%} "
              f"{result.mean_universe_vol:>13.1%} {result.benchmark_volatility:>10.1%} "
              f"{result.net_volatility:>8.1%} {result.max_drawdown:>7.1%} "
              f"{result.benchmark_max_drawdown:>12.1%} "
              f"{result.median_cross_section:>10.0f}")

    print("\n" + "-" * 118)
    print("WHERE THE TURNOVER COMES FROM, AND WHAT ZERO-SIZE COSTS WOULD LOOK LIKE")
    print("(both DECLARED DIAGNOSTICS - neither is the verdict)")
    print(f"{'band':>13} {'turnover':>10} {'forced exits':>14} "
          f"{'net ex-impact':>15} {'Sharpe':>8} {'excess ex-impact':>18}")
    for result in results:
        print(f"{result.band:>13} {result.turnover_annual:>9.1f}x "
              f"{result.forced_exit_share:>13.0%} "
              f"{result.net_ex_impact_annual:>14.1%} "
              f"{result.net_ex_impact_sharpe:>8.2f} "
              f"{result.excess_ex_impact:>+17.1%}")
    print("  'forced exits' = share of sales caused by the name ceasing to be RANKABLE")
    print("  (band flip or spread no longer resolving), not by its signal rank falling.")
    print("  'ex-impact' sets market impact to zero, i.e. trades at infinitesimal size.")
    print("  It is an unattainable upper bound: a position IS 1% of the name's daily")
    print("  volume here by construction, so impact cannot be zero at this capital.")

    print("\n" + "-" * 118)
    print("BREADTH (the lever this programme has never pulled)")
    for result in results:
        print(f"  {result.band:>13}: {result.effective_bets_per_rebalance:>5.1f} "
              f"effective independent bets per rebalance from {N_POSITIONS} positions "
              f"-> {result.breadth_per_year:>5.1f} per year")
    print("  Upper bound only. All 30 slots are filled by ONE monthly signal, so the")
    print("  number of independent FORECASTS is 12/yr; the figures above measure how")
    print("  much the realised position residuals diversified, not signal independence.")

    print("\n" + "=" * 118)
    print("VERDICT (pre-committed decision rule)")
    print("=" * 118)
    beat = [r for r in results if r.excess_annual > 0]
    promising = [r for r in beat if r.excess_annual > 0.02 and r.net_sharpe >= 0.75]
    if promising:
        verdict = "PROMISING"
    elif beat:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"
    print(f"  {verdict}")
    for result in results:
        relation = "BEATS" if result.excess_annual > 0 else "LOSES TO"
        print(f"    {result.band:>13} {result.excess_annual:>+8.2%}/yr  {relation} its "
              f"own equal-weight buy-and-hold   (net Sharpe {result.net_sharpe:.2f})")

    ordering = [r.excess_annual for r in results]
    print(f"\n  Excess ordering least->most liquid: "
          f"{['%.1f%%' % (100 * x) for x in ordering]}")
    print("  H1 also predicted the excess DECLINES with liquidity. Ordering above is the")
    print("  direct read; no significance test is claimed on four points.")

    for result in results:
        for note in result.notes:
            print(f"  ! {result.band}: {note}")

    # Machine-readable dump so the printed table never has to be re-derived by hand.
    payload = {
        "run_utc": pd.Timestamp.utcnow().isoformat(),
        "dev_cutoff": str(DEV_CUTOFF.date()),
        "verdict": verdict,
        "bands": [asdict(r) for r in results],
    }
    RESULT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("results written to %s", RESULT_JSON)

    log.info("done in %.1f min", (time.monotonic() - started) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
