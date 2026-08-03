"""ATTACK 1 -- FORCED EXITS. 46.5% of exits in B2 are forced. Where do those names GO?

Three separate questions, because they have different answers:

(a) WHAT triggered the exit. Reconstructed against the UNFILTERED monthly panel, so the
    name is followed out of the strategy's universe rather than lost at its edge.
(b) WAS THE LAST RETURN REAL. A forced exit is only honest if the book already ate the
    move that caused it. `forward_return` in the panel is built on the FULL ticker series
    (shift(-1) within ticker over every month-end row, eligible or not), so a name that
    crashes below the $2 floor still hands its crash to the month BEFORE the exit. That
    claim is tested here rather than asserted.
(c) WHAT THE BOOK WOULD HAVE EARNED had it been unable to get out. Two counterfactual
    books: forced retention for 1 and 3 extra months at the name's real full-panel return,
    and a book that charges a full trading leg on the exits that are currently free.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack1_forced_exits
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.multiasset.carry import vol_matched_active
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.low_vol_quality import (
    FORWARD_RETURN_CLIP,
    MIN_CROSS_SECTION,
    N_POSITIONS,
    band_group,
)

BAND = "B2_200k_1M"
HORIZONS = (1, 3, 12)


def full_panel() -> pd.DataFrame:
    """The UNFILTERED monthly panel: every name-month, eligible or not."""
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    panel["month"] = panel["date"].dt.to_period("M")
    panel["band_group"] = panel["band"].map(band_group)
    return panel


def horizon_returns(panel: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Forward h-month returns on closeadj, on the ticker's OWN row sequence.

    Deliberately NOT re-indexed onto a dense calendar: the panel has at most one row per
    (ticker, month), so shifting by h rows is shifting by h months except across a gap, and
    the gap frequency is reported so the approximation is visible.
    """
    out = panel[["ticker", "month", "date", "closeadj", "close", "band", "spread_regime",
                 "median_dollar_volume", "trading_fraction"]].copy()
    out = out.sort_values(["ticker", "month"]).reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False)["closeadj"]
    for h in horizons:
        out[f"fwd_{h}m"] = (grouped.shift(-h) / out["closeadj"] - 1.0)
        out[f"alive_{h}m"] = grouped.shift(-h).notna()
    return out


def _delist_map(delistings: pd.DataFrame) -> tuple[dict, dict]:
    return ({r.ticker: r.date for r in delistings.itertuples()},
            {r.ticker: float(r.terminal_return) for r in delistings.itertuples()})


def counterfactual_book(panel: pd.DataFrame, band: str, delistings: pd.DataFrame,
                        sticky_months: int = 0, charge_free_exits: bool = False,
                        full: pd.DataFrame | None = None) -> dict:
    """Re-run the book with forced exits held ``sticky_months`` extra months.

    A stranded name keeps its equal weight and earns its REAL return from the unfiltered
    panel (the return the strategy claims it never had to take). After ``sticky_months`` it
    is dropped at whatever price the panel last showed, which is still generous.

    ``charge_free_exits`` additionally charges a full one-way leg on exits the published
    book pays nothing for, priced at the name's LAST OBSERVED spread/impact inputs.
    """
    rows = INS.prepare(panel[panel["band_group"] == band], delistings)
    full = full if full is not None else full_panel()
    real = full.set_index(["ticker", "month"])["forward_return"].clip(
        -FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP)
    dd, dv = _delist_map(delistings)

    deployable = N_POSITIONS * 0.01 * float(rows["median_dollar_volume"].median())
    position_value = deployable / N_POSITIONS
    by_month = {m: f for m, f in rows.groupby("month", sort=True)}
    months = sorted(by_month)

    holdings: set[str] = set()
    stranded: dict[str, int] = {}                 # ticker -> months left to hold
    last_inputs: dict[str, tuple] = {}            # ticker -> (spread_cons, mdv, vol, px)
    gross: list[float] = []
    cost: list[float] = []
    benchmark: list[float] = []
    extra_legs = 0

    for month in months:
        cross_section = by_month[month]
        rankable = cross_section[cross_section["signal"].notna()]
        period_cost = 0.0

        if len(rankable) >= MIN_CROSS_SECTION:
            target = set(rankable.nlargest(N_POSITIONS, "signal")["ticker"])
            traded = target ^ (holdings - set(stranded))
            present = set(cross_section["ticker"])
            weight = 1.0 / max(len(target), 1)
            priced = cross_section.set_index("ticker")

            newly_stranded = {}
            for ticker in (holdings - set(stranded)) - target:
                if ticker not in rankable["ticker"].values:
                    if sticky_months > 0:
                        newly_stranded[ticker] = sticky_months
                    if charge_free_exits and ticker not in present:
                        inputs = last_inputs.get(ticker)
                        if inputs is not None:
                            spread_c, mdv, vol, _px = inputs
                            from research.capacity_study import (
                                IMPACT_COEFFICIENT_CONSERVATIVE, impact_fraction)
                            impact = impact_fraction(position_value, mdv, vol,
                                                     IMPACT_COEFFICIENT_CONSERVATIVE)
                            impact = impact if np.isfinite(impact) else 0.0
                            period_cost += weight * (spread_c / 2.0 + impact)
                            extra_legs += 1

            for ticker in traded:
                if ticker not in priced.index:
                    continue
                row = priced.loc[ticker]
                mdv = float(row["median_dollar_volume"])
                price = float(row["close"])
                vol = float(row["realised_vol"])
                vol = vol if np.isfinite(vol) and vol > 0 else None
                from research.capacity_study import (
                    IMPACT_COEFFICIENT_CONSERVATIVE, impact_fraction)
                impact = impact_fraction(position_value, mdv, vol,
                                         IMPACT_COEFFICIENT_CONSERVATIVE)
                impact = impact if np.isfinite(impact) else 0.0
                period_cost += weight * (float(row["spread_conservative"]) / 2.0 + impact
                                         + INS.commission_fraction(position_value, price))
                last_inputs[ticker] = (float(row["spread_conservative"]), mdv, vol, price)

            holdings = target | set(stranded) | set(newly_stranded)
            stranded.update(newly_stranded)

        cost.append(period_cost)
        universe = cross_section["realised_return"].dropna()
        benchmark.append(float(universe.mean()) if len(universe) else 0.0)

        if not holdings:
            gross.append(0.0)
            continue

        indexed = cross_section.set_index("ticker")
        realised: list[float] = []
        closing: list[str] = []
        for ticker in sorted(holdings):
            if ticker in stranded:
                value = real.get((ticker, month), np.nan)
                if not np.isfinite(value):
                    on = dd.get(ticker)
                    value = (float(dv.get(ticker, 0.0))
                             if on is not None and month.to_timestamp(how="end") < on
                             <= month.to_timestamp(how="end") + pd.Timedelta(days=62)
                             else 0.0)
                    closing.append(ticker)
                realised.append(float(value))
                stranded[ticker] -= 1
                if stranded[ticker] <= 0 and ticker not in closing:
                    closing.append(ticker)
                continue
            if ticker not in indexed.index:
                closing.append(ticker)
                realised.append(0.0)
                continue
            row = indexed.loc[ticker]
            realised.append(float(row["realised_return"]))
            if pd.isna(row["forward_clipped"]):
                closing.append(ticker)
        for ticker in closing:
            holdings.discard(ticker)
            stranded.pop(ticker, None)

        gross.append(max(float(np.mean(realised)), -1.0))

    gross_a = np.asarray(gross, dtype=float)
    cost_a = np.asarray(cost, dtype=float)
    bench_a = np.asarray(benchmark, dtype=float)
    net = np.maximum(gross_a - cost_a, -1.0)
    matched = vol_matched_active(pd.Series(net), pd.Series(bench_a))
    return {
        "sticky_months": sticky_months, "charge_free_exits": charge_free_exits,
        "n_months": len(net), "extra_legs": extra_legs,
        "gross_annual": INS.annual(gross_a), "cost_annual": float(cost_a.sum())
                                                            / (len(cost_a) / 12.0),
        "net_annual": INS.annual(net), "net_sharpe": INS.sharpe(net),
        "net_vol": float(np.std(net, ddof=1)) * np.sqrt(12.0),
        "bench_sharpe": INS.sharpe(bench_a),
        "vm_annual": matched.get("vol_matched_active_annual", np.nan),
        "vm_tstat": matched.get("vol_matched_active_tstat", np.nan),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    full = full_panel()
    horizons = horizon_returns(full)
    hz = horizons.set_index(["ticker", "month"])

    print("=" * 112)
    print("ATTACK 1 - FORCED EXITS: WHERE DO THE NAMES GO?")
    print("=" * 112)

    book = INS.run(merged, BAND, delistings)
    if book is None:
        print(f"{BAND}: insufficient data to instrument the book")
        return 1
    exits = book.exits.copy()
    print(f"band {BAND}: {len(exits):,} exits over {len(book.months)} months, "
          f"{book.n_rebalances} rebalances, {book.legs_traded:,} legs")
    counts = exits["kind"].value_counts()
    for kind, n in counts.items():
        print(f"  {kind:>28} {n:>6,}  {n/len(exits):>7.1%}")
    forced = exits[exits["kind"] != "discretionary"]
    print(f"  {'FORCED (published statistic)':>28} {len(forced):>6,}  "
          f"{len(forced)/len(exits):>7.1%}")
    free = exits[~exits["charged"]]
    print(f"  {'exits charged NO cost at all':>28} {len(free):>6,}  "
          f"{len(free)/len(exits):>7.1%} of exits, {len(free)/book.legs_traded:>6.1%} of legs")

    # --- (a) why did the forced ones leave? -------------------------------------------
    print("\n" + "-" * 112)
    print("(a) WHY THE NAME LEFT - state of the name in the UNFILTERED panel at the exit month")
    print("-" * 112)
    idx = pd.MultiIndex.from_frame(exits[["ticker", "month"]])
    present = idx.isin(hz.index)
    reason = pd.Series("absent from the price panel entirely", index=exits.index)
    sub = hz.reindex(idx)
    sub.index = exits.index
    reason[present & (sub["close"] < 2.0)] = "price fell below the $2 floor"
    reason[present & (sub["close"] >= 2.0) & (sub["trading_fraction"] < 0.90)] = \
        "stopped trading on >10% of days"
    ok = present & (sub["close"] >= 2.0) & (sub["trading_fraction"] >= 0.90)
    reason[ok & (sub["spread_regime"] == "unmeasurable")] = "spread stopped resolving"
    reason[ok & sub["band"].isna()] = "dollar volume left every band"
    reason[ok & sub["band"].notna() & (sub["band"] != "B2_200k_1M")] = \
        "moved to another liquidity band"
    reason[ok & (sub["band"] == "B2_200k_1M")
           & sub["spread_regime"].isin(["measured", "upper_bound"])] = \
        "still tradable - lost its quality/vol rank or its SF1 coverage"
    exits["reason"] = reason
    exits["is_forced"] = exits["kind"] != "discretionary"
    table = exits.groupby(["reason", "is_forced"]).size().unstack(fill_value=0)
    print(f"{'reason':>52} {'discretionary':>14} {'FORCED':>8} {'total':>8}")
    for r, row in table.iterrows():
        d = int(row.get(False, 0))
        f = int(row.get(True, 0))
        print(f"{r:>52} {d:>14,} {f:>8,} {d+f:>8,}")

    # --- (b) post-exit returns ---------------------------------------------------------
    print("\n" + "-" * 112)
    print("(b) WHAT THE NAME DID AFTER THE BOOK LET GO OF IT (closeadj, unfiltered panel)")
    print("    'dodged' = the strategy avoided this return. Negative => it dodged a loser.")
    print("-" * 112)
    for h in HORIZONS:
        exits[f"post_{h}m"] = sub[f"fwd_{h}m"].to_numpy()
        exits[f"alive_{h}m"] = sub[f"alive_{h}m"].to_numpy()
    universe_by_month = {}
    for h in HORIZONS:
        col = horizons.groupby("month")[f"fwd_{h}m"].mean()
        universe_by_month[h] = col
    print(f"{'group':>44} {'n':>6} " + " ".join(
        f"{'mean ' + str(h) + 'm':>10} {'med ' + str(h) + 'm':>9} {'vs univ':>9}"
        for h in HORIZONS))
    for label, block in (("discretionary exits", exits[~exits["is_forced"]]),
                         ("FORCED exits (all)", exits[exits["is_forced"]]),
                         ("  ...forced & still in the panel",
                          exits[exits["is_forced"] & present]),
                         ("  ...forced & gone from the panel",
                          exits[exits["is_forced"] & ~present]),
                         ("all exits", exits)):
        cells = []
        for h in HORIZONS:
            v = block[f"post_{h}m"].dropna()
            u = block["month"].map(universe_by_month[h])
            rel = (block[f"post_{h}m"] - u).dropna()
            cells.append(f"{v.mean():>10.2%} {v.median():>9.2%} {rel.mean():>+9.2%}"
                         if len(v) else f"{'-':>10} {'-':>9} {'-':>9}")
        print(f"{label:>44} {len(block):>6,} " + " ".join(cells))

    kept = book.holdings_log
    print(f"\n  for scale: {len(kept):,} held name-months, mean realised return booked "
          f"{kept['realised_return'].mean():.3%}/mo; "
          f"universe mean {np.mean(book.benchmark):.3%}/mo")

    # --- (c) counterfactual books -----------------------------------------------------
    print("\n" + "-" * 112)
    print("(c) COUNTERFACTUALS - what the book earns if the exits are not free")
    print("-" * 112)
    print(f"{'book':>52} {'gross':>8} {'cost':>7} {'net':>8} {'vol':>7} {'Sharpe':>8} "
          f"{'vol-matched':>12} {'t':>7}")
    scenarios = [
        ("published (reproduced)", 0, False),
        ("charge a real leg on every free exit", 0, True),
        ("forced to hold each forced exit +1 month", 1, False),
        ("forced to hold each forced exit +3 months", 3, False),
        ("+3 months AND charged on the way out", 3, True),
    ]
    for label, sticky, charge in scenarios:
        r = counterfactual_book(merged, BAND, delistings, sticky_months=sticky,
                                charge_free_exits=charge, full=full)
        print(f"{label:>52} {r['gross_annual']:>7.2%} {r['cost_annual']:>6.2%} "
              f"{r['net_annual']:>7.2%} {r['net_vol']:>6.2%} {r['net_sharpe']:>8.3f} "
              f"{r['vm_annual']:>+11.2%} {r['vm_tstat']:>+7.2f}"
              + (f"   (+{r['extra_legs']} legs charged)" if charge else ""))

    # --- delisting path ----------------------------------------------------------------
    print("\n" + "-" * 112)
    print("(d) THE DELISTING PATH - is delisting_drag_annual = 0.0 real or dead?")
    print("-" * 112)
    log = book.holdings_log
    last = log[log["is_last_obs"]]
    print(f"  held name-months at the name's LAST panel row: {len(last):,} of {len(log):,}")
    print(f"  of those, non-zero terminal return booked: "
          f"{int((last['realised_return'] != 0).sum()):,}")
    print(f"  months in which any holding hit its last row: {last['month'].nunique()}")
    if len(last):
        tail = last["month"].value_counts().sort_index()
        print(f"  first {tail.index.min()}  last {tail.index.max()}   "
              f"({int((last['month'] == last['month'].max()).sum())} of them in the very "
              f"last month, which is sample truncation, not delisting)")
    known = set(delistings["ticker"])
    print(f"  of {last['ticker'].nunique()} distinct such names, "
          f"{len(set(last['ticker']) & known)} appear in delistings.parquet at all")
    # what really happened to them
    lt = last[last["month"] < last["month"].max()]
    if len(lt):
        li = pd.MultiIndex.from_frame(lt[["ticker", "month"]])
        ls = hz.reindex(li)
        print(f"  excluding the truncation month: {len(lt)} exits at a last panel row; "
              f"mean 12m forward return where measurable "
              f"{ls['fwd_12m'].mean() if ls['fwd_12m'].notna().any() else float('nan'):.2%} "
              f"(n={int(ls['fwd_12m'].notna().sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
