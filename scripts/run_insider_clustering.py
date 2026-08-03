"""Run the registered insider-clustering sleeve ONCE and print measured numbers.

Design: `research/sleeves/insider_clustering_prereg.md`. Nothing in this script chooses
a parameter; every threshold comes from the registered module. It prints the headline,
the declared diagnostics, the filter accounting and the breadth measurement, and it does
not decide anything.

    .venv/Scripts/python.exe -m scripts.run_insider_clustering
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from research.sleeves.insider_clustering import (
    DECILE,
    PANEL_DIR,
    annualise,
    build_universe,
    cluster_signal,
    daily_volatility_panel,
    dedupe_purchase_legs,
    load_purchase_legs,
    run_backtest,
)


def _select_top_decile(frame: pd.DataFrame) -> pd.DataFrame:
    """Top decile by (n_buyers, value_ratio) descending, never padded [S8].

    Padding the decile with zero-signal names when too few carry a signal would dilute
    the thing being measured into the universe average and quietly turn the test into a
    test of nothing.
    """
    size = math.ceil(DECILE * len(frame))
    with_signal = frame[frame["n_buyers"] > 0]
    if with_signal.empty:
        return with_signal
    ranked = with_signal.sort_values(
        ["n_buyers", "value_ratio"], ascending=[False, False], kind="stable"
    )
    return ranked.head(min(size, len(ranked)))


def _select_bucket(low: int, high: float):
    """Diagnostic selector: every name with n_buyers in [low, high]."""

    def selector(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["n_buyers"] >= low) & (frame["n_buyers"] <= high)]

    return selector


def main() -> None:
    pd.set_option("display.width", 200)

    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    cells, rebalance_dates, universe_report = build_universe(panel, delistings)
    print("=" * 78)
    print("UNIVERSE (prereg S6/S7)")
    print(universe_report.render())
    print(f"  rebalances               {len(rebalance_dates):>12,}")
    print(f"  first / last             {rebalance_dates[0].date()} .. "
          f"{rebalance_dates[-1].date()}")

    vol = daily_volatility_panel()
    cells = cells.merge(vol, on=["ticker", "date"], how="left")
    # A name with no volatility estimate cannot be charged impact honestly, so it is not
    # investable. Reported rather than silently defaulted to zero (which would be free).
    missing_vol = int(cells["daily_vol"].isna().sum())
    cells = cells[cells["daily_vol"].notna()].reset_index(drop=True)
    print(f"  - no volatility estimate {missing_vol:>12,}")
    print(f"  FINAL investable cells   {len(cells):>12,}")

    legs = load_purchase_legs()
    legs, dedupe_report = dedupe_purchase_legs(legs)
    print()
    print("INSIDER PURCHASE LEGS (prereg S3/S4)")
    print(dedupe_report.render())

    signal = cluster_signal(legs, rebalance_dates)
    cells = cells.merge(signal, on=["ticker", "date"], how="left")
    cells["n_buyers"] = cells["n_buyers"].fillna(0.0)
    cells["buy_value"] = cells["buy_value"].fillna(0.0)
    # Cluster purchases as a fraction of one month of the name's own dollar volume.
    cells["value_ratio"] = cells["buy_value"] / (
        cells["median_dollar_volume"] * 21.0
    )

    covered = cells.groupby("date")["n_buyers"].apply(lambda s: (s > 0).sum())
    print()
    print("SIGNAL COVERAGE")
    print(f"  cells with n_buyers>=1   {int((cells['n_buyers'] > 0).sum()):>12,} "
          f"({(cells['n_buyers'] > 0).mean():.1%} of investable cells)")
    print(f"  names with a signal per month: median {covered.median():.0f}, "
          f"min {covered.min():.0f}, max {covered.max():.0f}")
    print("  n_buyers distribution (cells with a signal):")
    print(cells.loc[cells["n_buyers"] > 0, "n_buyers"].value_counts()
          .sort_index().head(10).to_string())

    # ---------------- benchmark: equal weight, own universe, zero cost ----------
    bench = cells.groupby("date")["realised_return"].mean()
    bench_stats = annualise(bench)

    # ---------------- headline: registered top-decile book ---------------------
    result = run_backtest(cells, rebalance_dates, _select_top_decile, "top_decile")
    monthly = result.monthly.set_index("date")
    net_stats = annualise(monthly["net"])
    gross_stats = annualise(monthly["gross"])

    aligned = monthly["net"] - bench.reindex(monthly.index)

    print()
    print("=" * 78)
    print("HEADLINE (registered, one trial): long-only top decile by (n_buyers, "
          "value_ratio)")
    print(f"  months                   {len(monthly):>12,}")
    print(f"  median names held        {int(np.median(result.holdings_count)):>12,}")
    print(f"  gross return  {gross_stats['cagr']:>10.2%}   gross vol "
          f"{gross_stats['vol']:>8.2%}   gross Sharpe {gross_stats['sharpe']:>6.2f}")
    print(f"  NET   return  {net_stats['cagr']:>10.2%}   net   vol "
          f"{net_stats['vol']:>8.2%}   NET   Sharpe {net_stats['sharpe']:>6.2f}")
    print(f"  net maxDD     {net_stats['maxdd']:>10.2%}")
    print(f"  benchmark (EW own universe, zero cost) {bench_stats['cagr']:>8.2%}  "
          f"vol {bench_stats['vol']:.2%}  Sharpe {bench_stats['sharpe']:.2f}")
    print(f"  EXCESS (net CAGR - benchmark CAGR)     "
          f"{net_stats['cagr'] - bench_stats['cagr']:>8.2%}")
    print(f"  mean monthly excess, annualised        "
          f"{aligned.mean() * 12:>8.2%}   t-stat "
          f"{aligned.mean() / (aligned.std(ddof=1) / np.sqrt(len(aligned))):.2f}")

    turnover_annual = float(monthly["turnover"].mean() * 12.0)
    cost_drag = float(monthly["cost"].mean() * 12.0)
    total_cost = sum(result.cost_components.values())
    print()
    print("COSTS (prereg S9, per name, both sides)")
    print(f"  annual one-way turnover  {turnover_annual:>12.2f}x")
    print(f"  annual cost drag         {cost_drag:>12.2%}")
    for name, value in result.cost_components.items():
        print(f"    {name:<20} {value / total_cost:>10.1%} of cost")

    # ---------------- declared diagnostics (S12) -------------------------------
    print()
    print("DIAGNOSTICS (declared in advance as non-gate-eligible)")
    buckets = {
        "n_buyers == 0": (0, 0),
        "n_buyers == 1": (1, 1),
        "n_buyers == 2": (2, 2),
        "n_buyers >= 3": (3, np.inf),
        "n_buyers >= 2 (clustered)": (2, np.inf),
    }
    for label, (low, high) in buckets.items():
        subset = cells[(cells["n_buyers"] >= low) & (cells["n_buyers"] <= high)]
        by_month = subset.groupby("date")["realised_return"].mean()
        stats_ = annualise(by_month)
        print(f"  {label:<26} cells {len(subset):>8,}  gross EW CAGR "
              f"{stats_['cagr']:>8.2%}  vs bench {bench_stats['cagr']:.2%}")

    for label, (low, high) in [("clustered >=2, NET", (2, np.inf)),
                               ("single  ==1, NET", (1, 1))]:
        diag = run_backtest(cells, rebalance_dates, _select_bucket(low, high), label)
        dm = diag.monthly.set_index("date")
        st = annualise(dm["net"])
        print(f"  {label:<26} net CAGR {st['cagr']:>8.2%}  net Sharpe "
              f"{st['sharpe']:>6.2f}  median held "
              f"{int(np.median(diag.holdings_count)):>5,}")

    # ---------------- breadth (rule 7 / prereg S11) ----------------------------
    print()
    print("BREADTH (rule 7)")
    naive = 12 * float(np.median(result.holdings_count))

    # Residual of each held name against its own universe's equal-weight return that
    # month. N_eff = per-name residual variance / equal-weight portfolio residual
    # variance -- the exact identity for an equal-weight book of correlated names.
    cells = cells.merge(bench.rename("bench"), on="date", how="left")
    cells["resid"] = cells["realised_return"] - cells["bench"]
    held_keys = set()
    for date in rebalance_dates:
        frame = cells[cells["date"] == date]
        if frame.empty:
            continue
        picks = _select_top_decile(frame)
        held_keys.update(zip(picks["ticker"], picks["date"], strict=True))
    held_mask = pd.Series(
        list(zip(cells["ticker"], cells["date"], strict=True)), index=cells.index
    ).isin(held_keys)
    held = cells[held_mask]
    name_resid_var = float(np.var(held["resid"].to_numpy(dtype=float), ddof=1))
    port_resid = held.groupby("date")["resid"].mean()
    port_resid_var = float(np.var(port_resid.to_numpy(dtype=float), ddof=1))
    n_eff = name_resid_var / port_resid_var if port_resid_var > 0 else np.nan

    ics = []
    for date, frame in cells.groupby("date"):
        if len(frame) < 30 or frame["n_buyers"].nunique() < 2:
            continue
        rho = stats.spearmanr(frame["n_buyers"], frame["realised_return"]).statistic
        if np.isfinite(rho):
            ics.append(rho)
    ic_mean = float(np.mean(ics)) if ics else np.nan
    ic_t = (float(np.mean(ics)) / (float(np.std(ics, ddof=1)) / math.sqrt(len(ics)))
            if len(ics) > 1 else np.nan)

    print(f"  naive bets/yr (12 x median names held)      {naive:>10,.0f}")
    print(f"  effective independent names N_eff           {n_eff:>10.2f}")
    print(f"  EFFECTIVE INDEPENDENT BETS PER YEAR         {12 * n_eff:>10.1f}")
    print(f"  cross-sectional IC (mean monthly Spearman)  {ic_mean:>10.4f}  "
          f"t={ic_t:.2f} over {len(ics)} months")
    if np.isfinite(ic_mean) and ic_mean != 0:
        print(f"  Grinold implied IR = IC*sqrt(BR)            "
              f"{ic_mean * math.sqrt(12 * n_eff):>10.2f}")

    out = PANEL_DIR / "insider_clustering_monthly.parquet"
    monthly.reset_index().assign(
        bench=bench.reindex(monthly.index).to_numpy()
    ).to_parquet(out, index=False)
    print()
    print(f"monthly series written to {out}")


if __name__ == "__main__":
    main()
