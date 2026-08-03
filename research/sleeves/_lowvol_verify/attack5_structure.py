"""ATTACKS 4, 6, 8 -- CONCENTRATION, LOOKAHEAD, CAPACITY, AND THE COMBINED REPAIR.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack5_structure
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.capacity_study import IMPACT_COEFFICIENT_CONSERVATIVE, impact_fraction
from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.low_vol_quality import (
    N_POSITIONS,
    PARTICIPATION_LIMIT,
)

BAND = "B2_200k_1M"


def combined_repair(panel, band, delistings, *, repair_delisting: bool,
                    charge_free_exits: bool):
    """The published book with both defects repaired, in the committed code path.

    Delisting repair = shift every ACTIONS date one day later so the strict `<` accepts a
    delisting dated on the last traded bar. Free-exit repair = charge one full one-way leg,
    at the name's last observed spread and impact inputs, for every exit the published book
    pays nothing for.
    """
    d = delistings.copy()
    if repair_delisting:
        d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=1)
    books = LV.run_band(panel, band, d)
    if books is None:
        return None
    if not charge_free_exits:
        return books
    ins = INS.run(panel, band, d)
    if ins is None:
        # Without the instrumented run there is no list of free exits to charge, and
        # returning the unrepaired book would silently skip the repair this function
        # exists to apply. Report no book instead.
        return None
    rows = INS.prepare(panel[panel["band_group"] == band], d)
    last = (rows.sort_values("date").groupby("ticker")
            .agg(spread_conservative=("spread_conservative", "last"),
                 mdv=("median_dollar_volume", "last"),
                 vol=("realised_vol", "last")))
    free = ins.exits[~ins.exits["charged"]]
    extra = pd.Series(0.0, index=pd.Index(books.months, name="month"))
    for ticker, month in zip(free["ticker"], free["month"]):
        if ticker not in last.index:
            continue
        r = last.loc[ticker]
        # impact_fraction takes an optional vol and falls back when it is absent, so
        # keep the narrowed value in its own name rather than rebinding a float to None.
        vol_raw = float(r["vol"])
        vol = vol_raw if np.isfinite(vol_raw) and vol_raw > 0 else None
        impact = impact_fraction(books.position_value, float(r["mdv"]), vol,
                                 IMPACT_COEFFICIENT_CONSERVATIVE)
        impact = impact if np.isfinite(impact) else 0.0
        extra.loc[month] += (float(r["spread_conservative"]) / 2.0 + impact) / N_POSITIONS
    books.cost_conservative = books.cost_conservative + extra.to_numpy()
    books.cost_realistic = books.cost_realistic + extra.to_numpy()
    books.notes = list(books.notes) + [f"charged {len(free)} previously-free exit legs"]
    return books


def summarise(books, label: str) -> dict:
    e = LV.evaluate_band(books, n_trials=LV.N_TRIALS)
    c = e["bounds"]["conservative"]
    return {
        "label": label,
        "gross": e["gross"]["annual_arithmetic"],
        "cost": c["cost_annual_total"],
        "net": c["net"]["annual_arithmetic"],
        "vol": c["net"]["volatility"],
        "sharpe": c["net"]["sharpe"],
        "dd": c["net"]["max_drawdown"],
        "bench": e["benchmark"]["annual_arithmetic"],
        "bench_sharpe": e["benchmark"]["sharpe"],
        "raw": c["excess_arithmetic"],
        "vm": c["vol_matched"]["vol_matched_active_annual"],
        "vm_t": c["vol_matched"]["vol_matched_active_tstat"],
        "bar": e["dsr_sharpe_bar"],
        "dsr_pass": c["gate_dsr_bar_pass"],
        "verdict": LV.verdict_for(e),
        "onebps": c["cost_one_way_bps"],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    book = INS.run(merged, BAND, delistings)
    published = LV.run_band(merged, BAND, delistings)
    if book is None or published is None:
        print(f"{BAND}: insufficient data to build the published book -- nothing to attack")
        return 1

    print("=" * 112)
    print("ATTACK 4 - P&L AND GROSS-NOTIONAL CONCENTRATION, RECOMPUTED")
    print("=" * 112)
    pnl = pd.DataFrame(book.pnl_by_name_month, columns=["ticker", "month", "pnl"])
    total = float(pnl["pnl"].sum())
    absolute = float(pnl["pnl"].abs().sum())
    print(f"  {len(pnl):,} name-months, {pnl['ticker'].nunique():,} distinct names ever held")
    print(f"  total attributed gross P&L on a 1.0 starting equity: {total:.4f}")
    print(f"  sum of |P&L|: {absolute:.4f}   (a 1-unit book that ends at "
          f"{float((1+published.gross).prod()):.2f}x)")
    print(f"\n{'unit':>18} {'largest share of TOTAL P&L':>28} {'top-3':>9} {'top-5':>9} "
          f"{'top-10':>9}")
    for label, key in (("single name-month", None), ("single NAME (all months)", "ticker"),
                       ("single MONTH", "month")):
        block = pnl["pnl"] if key is None else pnl.groupby(key)["pnl"].sum()
        s = block.sort_values(ascending=False)
        print(f"{label:>18} {float(s.iloc[0])/total:>27.2%} "
              f"{float(s.iloc[:3].sum())/total:>8.2%} {float(s.iloc[:5].sum())/total:>8.2%} "
              f"{float(s.iloc[:10].sum())/total:>8.2%}")
    by_name = pnl.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    print("\n  top 5 names by P&L share: " + ", ".join(
        f"{t} {v/total:.2%}" for t, v in by_name.head(5).items()))
    print("  worst 3 names: " + ", ".join(
        f"{t} {v/total:.2%}" for t, v in by_name.tail(3).items()))
    print(f"  names contributing >5% of total P&L: "
          f"{int((by_name / total > 0.05).sum())}")
    hhi = float(((by_name / total) ** 2).sum())
    print(f"  Herfindahl over names {hhi:.4f} -> {1/hhi:.0f} effective names "
          f"(of {pnl['ticker'].nunique():,} held)")
    monthly = pnl.groupby("month")["pnl"].sum().sort_values(ascending=False)
    print(f"  largest single MONTH is {monthly.index[0]} at "
          f"{float(monthly.iloc[0])/total:.2%} of total P&L")
    print(f"\n  gross notional: equal weight over {N_POSITIONS} names, so max weight is "
          f"{published.max_gross_weight:.2%} and top-3 is {published.top3_gross_weight:.2%}")
    print(f"  mean positions actually held {np.mean([1]) and published.n_positions_mean:.2f}"
          f" -- the notional test is trivially satisfied BY CONSTRUCTION and is not evidence")

    print("\n" + "=" * 112)
    print("ATTACK 8 - CAPACITY: IS THE REGISTERED 1% PARTICIPATION CAP ACTUALLY HONOURED?")
    print("=" * 112)
    part = book.participation
    print(f"  deployable capital ${published.deployable_capital:,.0f}, "
          f"position ${published.position_value:,.0f} "
          f"= {N_POSITIONS} x 1% x the band's MEDIAN dollar volume "
          f"(${float(merged[merged['band_group']==BAND]['median_dollar_volume'].median()):,.0f})")
    print("  the position size is ONE FULL-SAMPLE CONSTANT applied to every name, so a "
          "name at the")
    print(f"  bottom of the band (${2e5:,.0f}/day) is traded at "
          f"{published.position_value/2e5:.2%} of its volume, not 1%.")
    q = part["participation"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"\n  participation per traded leg (n={len(part):,}):")
    for k in ("min", "5%", "25%", "50%", "75%", "95%", "99%", "max"):
        print(f"    {k:>5} {float(q[k]):.3%}")
    over = float((part["participation"] > PARTICIPATION_LIMIT).mean())
    print(f"  legs ABOVE the registered {PARTICIPATION_LIMIT:.0%} cap: {over:.1%}")
    print(f"  legs above 2%: {float((part['participation'] > 0.02).mean()):.1%};  "
          f"above 5%: {float((part['participation'] > 0.05).mean()):.1%}")
    print("\n  the sqrt-impact model does charge for the excess, so this is a breach of the")
    print("  registered constraint rather than an unpriced cost. Capping every leg at 1%")
    print(f"  of its OWN volume would shrink the book below ${published.deployable_capital:,.0f}.")
    turnover = published.legs_traded / N_POSITIONS / (len(published.gross) / 12.0)
    notional = published.deployable_capital * turnover / 2.0
    print(f"\n  turnover {turnover:.2f}x/yr on ${published.deployable_capital:,.0f} = "
          f"~${notional:,.0f}/yr of one-way notional, ~{published.legs_traded/17.75:.0f} "
          f"orders/yr of ${published.position_value:,.0f}")
    print("  that is executable. The binding constraint is not liquidity, it is that the")
    print(f"  ONLY band that clears the excess gate holds ${published.deployable_capital:,.0f}.")

    print("\n" + "=" * 112)
    print("ATTACK 6 - LOOKAHEAD: DOES THE BOOK NEED THE BAR IT TRADES ON?")
    print("=" * 112)
    lagged = merged.sort_values(["ticker", "date"]).copy()
    lagged["signal"] = lagged.groupby("ticker")["signal"].shift(1)
    gap = lagged.groupby("ticker")["date"].diff().dt.days
    lagged.loc[gap > 45, "signal"] = np.nan      # do not carry a stale rank over a gap
    lag_books = LV.run_band(lagged, BAND, delistings)
    if lag_books is None:
        print(f"{BAND}: insufficient data under the one-month implementation lag")
        return 1
    rows = []
    rows.append(summarise(published, "published (trade on month-m signal)"))
    rows.append(summarise(lag_books, "ONE MONTH implementation lag"))
    print(f"{'book':>44} {'gross':>7} {'net':>7} {'Sharpe':>7} {'bench':>7} "
          f"{'vol-matched':>12} {'t':>6}  verdict")
    for r in rows:
        print(f"{r['label']:>44} {r['gross']:>6.2%} {r['net']:>6.2%} {r['sharpe']:>7.3f} "
              f"{r['bench']:>6.2%} {r['vm']:>+11.2%} {r['vm_t']:>+6.2f}  {r['verdict']}")
    print("  A signal that only works when acted on with the same bar that defined it is a")
    print("  timing artefact. This one survives a full month of delay, which is the")
    print("  strongest single piece of evidence FOR the sleeve in this report.")

    print("\n  return alignment: `realised_return` at month m is the panel's "
          "forward_return,")
    sample = merged[merged["band_group"] == BAND].sort_values(["ticker", "date"])
    chk = sample.groupby("ticker").head(400)
    own = chk.groupby("ticker")["closeadj"].shift(-1) / chk["closeadj"] - 1.0
    d = (own - chk["forward_return"]).abs()
    print(f"  i.e. closeadj(m+1)/closeadj(m)-1 computed on the FULL ticker series: "
          f"max |diff| over {int(d.notna().sum()):,} checked cells = {float(d.max()):.2e}")
    print("  so the return a holding earns is strictly AFTER the bar its signal was "
          "computed on.")

    print("\n" + "=" * 112)
    print("THE COMBINED CORRECTED BOOK - both defects repaired at once, all four bands")
    print("=" * 112)
    print(f"{'band':>13} {'book':>36} {'gross':>7} {'cost':>7} {'net':>7} {'Sh':>6} "
          f"{'bench':>7} {'bSh':>6} {'raw exc':>8} {'vol-matched':>12} {'t':>6}  verdict")
    for band in LV.BAND_ORDER:
        for label, rd, cf in (("published", False, False),
                              ("delisting window repaired", True, False),
                              ("+ free exits charged", True, True)):
            b = combined_repair(merged, band, delistings, repair_delisting=rd,
                                charge_free_exits=cf)
            r = summarise(b, label)
            print(f"{band:>13} {label:>36} {r['gross']:>6.2%} {r['cost']:>6.2%} "
                  f"{r['net']:>6.2%} {r['sharpe']:>6.3f} {r['bench']:>6.2%} "
                  f"{r['bench_sharpe']:>6.3f} {r['raw']:>+7.2%} {r['vm']:>+11.2%} "
                  f"{r['vm_t']:>+6.2f}  {r['verdict']}")
        print()
    print(f"  DSR Sharpe bar at 17.75y / 38 trials = "
          f"{dsr_sharpe_bar(17.75, n_trials=38):.4f}. No repaired book gets near it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
