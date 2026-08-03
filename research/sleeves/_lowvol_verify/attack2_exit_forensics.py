"""ATTACK 1b / 3 -- EXIT FORENSICS AND THE DELISTING PATH.

Attack 1 showed 92% of forced exits are "moved to another liquidity band". That answer is
worthless until the direction is known: a name that leaves B2 because it got MORE liquid is
a healthy graduation; a name that leaves because its volume DRIED UP is a distressed name
and the book is walking away from it at an unmarked price.

It also showed only 58 of 6,210 held name-months sit at the name's last panel row, and that
NONE of them booked a non-zero terminal return -- while 47 of the 58 names DO appear in
`delistings.parquet`. Either the 62-day window is doing its job or it is silently letting
bankruptcies through at 0%. That is the -60%/yr bug class and it is settled here.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack2_exit_forensics
"""

from __future__ import annotations

import logging

import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.attack1_forced_exits import full_panel, horizon_returns
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.lowvol_retest import BAND_ORDER

BAND = "B2_200k_1M"
BAND_RANK = {"B1_under_200k": 1, "B2_200k_1M": 2, "B3_1M_5M": 3, "B4_5M_25M": 4,
             "B5_25M_200M": 5, "B6_200M_plus": 6}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    full = full_panel()
    hz = horizon_returns(full).set_index(["ticker", "month"])

    book = INS.run(merged, BAND, delistings)
    if book is None:
        print(f"{BAND}: insufficient data to instrument the book")
        return 1
    exits = book.exits.copy()
    idx = pd.MultiIndex.from_frame(exits[["ticker", "month"]])
    sub = hz.reindex(idx)
    sub.index = exits.index
    exits["band_at_exit"] = sub["band"].to_numpy()
    exits["rank_at_exit"] = exits["band_at_exit"].map(BAND_RANK)
    exits["is_forced"] = exits["kind"] != "discretionary"
    for h in (1, 3, 12):
        exits[f"post_{h}m"] = sub[f"fwd_{h}m"].to_numpy()

    print("=" * 112)
    print("ATTACK 1b - DID THE FORCED-EXIT NAMES GRADUATE OR DIE?")
    print("=" * 112)
    moved = exits[exits["is_forced"] & exits["rank_at_exit"].notna()
                  & (exits["rank_at_exit"] != 2)]
    print(f"  {len(moved):,} forced exits landed in a DIFFERENT band that month")
    print(f"{'destination band':>22} {'n':>6} {'share':>8} {'post 1m':>9} {'post 3m':>9} "
          f"{'post 12m':>10}")
    for band, block in moved.groupby("band_at_exit"):
        direction = "MORE liquid" if BAND_RANK[band] > 2 else "LESS liquid (dried up)"
        print(f"{band:>22} {len(block):>6,} {len(block)/len(moved):>7.1%} "
              f"{block['post_1m'].mean():>+8.2%} {block['post_3m'].mean():>+8.2%} "
              f"{block['post_12m'].mean():>+9.2%}   {direction}")
    up = moved[moved["rank_at_exit"] > 2]
    down = moved[moved["rank_at_exit"] < 2]
    print(f"\n  graduated UP  : {len(up):,} ({len(up)/len(moved):.1%})   "
          f"post-12m {up['post_12m'].mean():+.2%}")
    print(f"  fell DOWN     : {len(down):,} ({len(down)/len(moved):.1%})   "
          f"post-12m {down['post_12m'].mean():+.2%}")
    print("  A name that falls out of B2 into B1 is the distressed case. Its price move")
    print("  INTO the exit month is already booked (panel forward_return is built on the")
    print("  full ticker series); what the book skips is everything after the exit.")

    # what the universe did over the same months, as the fair comparator
    univ12 = horizon_returns(full).groupby("month")["fwd_12m"].mean()
    for label, block in (("graduated up", up), ("fell down", down)):
        rel = (block["post_12m"] - block["month"].map(univ12)).dropna()
        print(f"  {label:>14}: 12m vs the whole panel's contemporaneous mean "
              f"{rel.mean():+.2%}  (n={len(rel)})")

    print("\n" + "=" * 112)
    print("ATTACK 3 - THE DELISTING PATH: is delisting_drag_annual = 0.0 GENUINE?")
    print("=" * 112)
    log = book.holdings_log
    last = log[log["is_last_obs"]].copy()
    truncation = last["month"].max()
    real = last[last["month"] < truncation].copy()
    print(f"  {len(last)} held name-months at a last panel row; {len(real)} excluding the "
          f"{truncation} truncation month")

    dd = {r.ticker: r.date for r in delistings.itertuples()}
    da = {r.ticker: r.action for r in delistings.itertuples()}
    dv = {r.ticker: float(r.terminal_return) for r in delistings.itertuples()}
    exit_date = merged.set_index(["ticker", merged["date"].dt.to_period("M")])["date"]
    real["exit_date"] = [exit_date.get((t, m), pd.NaT)
                         for t, m in zip(real["ticker"], real["month"])]
    real["delist_date"] = real["ticker"].map(dd)
    real["action"] = real["ticker"].map(da)
    real["terminal"] = real["ticker"].map(dv)
    real["gap_days"] = (pd.to_datetime(real["delist_date"])
                        - pd.to_datetime(real["exit_date"])).dt.days
    print(f"{'action':>26} {'n':>5} {'terminal':>9} {'median gap (days)':>19} "
          f"{'inside 62d window':>19}")
    for action, block in real.groupby(real["action"].fillna("NO RECORD AT ALL")):
        inside = int(((block["gap_days"] > 0) & (block["gap_days"] <= 62)).sum())
        print(f"{str(action):>26} {len(block):>5} "
              f"{block['terminal'].mean() if block['terminal'].notna().any() else float('nan'):>9.2f} "
              f"{block['gap_days'].median() if block['gap_days'].notna().any() else float('nan'):>19.0f} "
              f"{inside:>19}")
    loss = real[real["terminal"] == -1.0]
    print(f"\n  BANKRUPT / INVOLUNTARY DELISTINGS held to the last bar: {len(loss)}")
    if len(loss):
        print(f"    gap from exit to delisting, days: "
              f"{sorted(loss['gap_days'].dropna().astype(int).tolist())}")
        drag = -1.0 * len(loss) / 30.0 / (len(book.months) / 12.0)
        print(f"    if every one of them were booked at -100%: {drag:+.2%}/yr of drag")
    missed = real[(real["terminal"] == -1.0) & ~((real["gap_days"] > 0)
                                                & (real["gap_days"] <= 62))]
    print(f"    of which MISSED by the 62-day window and booked at 0.0%: {len(missed)}")

    print("\n  the same test on the BENCHMARK's own universe (both sides read one column):")
    print(f"{'band':>13} {'universe cells':>15} {'last-obs cells':>15} {'rate':>8} "
          f"{'held rate':>10} {'strategy advantage':>19}")
    for band in BAND_ORDER:
        rows = merged[merged["band_group"] == band]
        u_rate = float(rows["forward_return"].isna().mean())
        b = INS.run(merged, band, delistings)
        if b is None:
            print(f"{band:>13}  insufficient data")
            continue
        h_rate = float(b.holdings_log["is_last_obs"].mean())
        print(f"{band:>13} {len(rows):>15,} {int(rows['forward_return'].isna().sum()):>15,} "
              f"{u_rate:>7.2%} {h_rate:>9.2%} "
              f"{'strategy hits FEWER' if h_rate < u_rate else 'strategy hits MORE':>19}")
    print("  A last-observation cell books terminal_on_exit (0.0 unless a delisting lands")
    print("  in the 62-day window) for the STRATEGY and for the BENCHMARK alike. Whichever")
    print("  side meets more of them gets more of the free pass.")

    print("\n" + "=" * 112)
    print("ATTACK 3b - IS THE 62-DAY WINDOW EVER SATISFIED ANYWHERE IN THE PANEL?")
    print("=" * 112)
    for band in BAND_ORDER:
        rows = merged[merged["band_group"] == band]
        nz = INS.prepare(rows, delistings)
        hit = nz["terminal_on_exit"] != 0.0
        print(f"{band:>13} {int(hit.sum()):>7,} cells of {len(nz):,} carry a non-zero "
              f"terminal_on_exit ({hit.mean():.4%}); "
              f"of the {int(nz['forward_return'].isna().sum()):,} last-obs cells, "
              f"{int((hit & nz['forward_return'].isna()).sum()):,} do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
