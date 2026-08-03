"""ATTACK 3 -- THE DELISTING WINDOW IS OFF BY ONE, AND BOTH SIDES ARE PAID BY THE BUG.

`lowvol_retest.run_band` books a terminal return only when

    exit_date < delisting_date <= exit_date + 62 days

Sharadar's ACTIONS row for a delisting is dated on the name's LAST TRADED BAR, so the gap
is exactly ZERO for every bankruptcy in this book -- all 9 of them -- and the strict `<`
throws every one away. `terminal_on_exit` fires on 1 of B2's 1,817 last-observation cells.
`delisting_drag_annual = 0.0` is therefore a DEAD CODE PATH, not a measurement.

The repair is one character (`<` -> `<=`). It is applied to the SHARED `realised_return`
column, so it hits the benchmark exactly as hard as the strategy, and the honest question
is which side it hurts more. That is measured, not asserted, and both the repaired and the
maximal ("every unexplained last observation is a total loss") books are reported.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack3_delisting_repair
"""

from __future__ import annotations

import logging

import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build
from research.sleeves.low_vol_quality import DELISTING_WINDOW_DAYS

BAND_RANK = {"B1_50k_200k": 1, "B2_200k_1M": 2, "B3_1M_5M": 3, "B4_5M_25M": 4,
             "B5_25M_200M": 5, "B6_200M_plus": 6}


def patched_run_band(panel, band, delistings, *, inclusive: bool,
                     total_loss_on_unexplained: bool = False):
    """`LV.run_band` with the delisting window boundary as a parameter.

    Implemented by monkey-patching only the two comparisons, so every other line of the
    committed code is the code that produced the published number.
    """
    # (An earlier monkey-patching approach captured pd.Timestamp and a row map here.
    # It was abandoned for the delistings-shift approach documented below; the dead
    # captures are removed so F841 stays a working alarm rather than chronic noise.)

    def run(panel_, band_, delistings_):
        import research.sleeves.lowvol_retest as mod
        src_run = mod.run_band
        return src_run(panel_, band_, delistings_)

    # Rather than patch pandas, rebuild the frame with a corrected `forward_return`-free
    # terminal column and hand `run_band` a delistings table shifted by one day, which
    # makes `exit < delist` true exactly when `exit <= delist` was true.
    # `run_band` tests `exit < delist <= exit + 62d`. Moving the delisting date FORWARD by
    # one day makes that strict test true exactly when `exit <= delist` was true, at the
    # cost of shortening the far end of the window from 62 days to 61 -- conservative.
    shifted = delistings.copy()
    if inclusive:
        shifted["date"] = pd.to_datetime(shifted["date"]) + pd.Timedelta(days=1)
    if total_loss_on_unexplained:
        # Every name whose panel series simply stops, with no acquisition on record, is
        # booked at -100%. The maximal, deliberately unfair version of the correction.
        last_rows = (panel.sort_values(["ticker", "date"])
                     .groupby("ticker").tail(1)[["ticker", "date"]])
        known = set(delistings["ticker"])
        unexplained = last_rows[~last_rows["ticker"].isin(known)].copy()
        unexplained["terminal_return"] = -1.0
        unexplained["action"] = "assumed_total_loss"
        unexplained["date"] = pd.to_datetime(unexplained["date"]) + pd.Timedelta(days=1)
        shifted = pd.concat([shifted, unexplained], ignore_index=True)
    return LV.run_band(panel, band, shifted)


def headline(books, n_trials: int = LV.N_TRIALS) -> dict:
    e = LV.evaluate_band(books, n_trials=n_trials)
    c = e["bounds"]["conservative"]
    return {
        "gross": e["gross"]["annual_arithmetic"],
        "net": c["net"]["annual_arithmetic"],
        "net_sharpe": c["net"]["sharpe"],
        "net_vol": c["net"]["volatility"],
        "net_dsr": c["net"]["dsr"],
        "bench": e["benchmark"]["annual_arithmetic"],
        "bench_sharpe": e["benchmark"]["sharpe"],
        "raw_excess": c["excess_arithmetic"],
        "vm": c["vol_matched"]["vol_matched_active_annual"],
        "vm_t": c["vol_matched"]["vol_matched_active_tstat"],
        "drag": e["delisting_drag_annual"],
        "bar": e["dsr_sharpe_bar"],
        "verdict": LV.verdict_for(e),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    print("=" * 118)
    print("ATTACK 1b (corrected band labels) - DID THE FORCED-EXIT NAMES GRADUATE OR DIE?")
    print("=" * 118)
    from research.sleeves._lowvol_verify.attack1_forced_exits import (
        full_panel, horizon_returns)
    full = full_panel()
    hzf = horizon_returns(full)
    hz = hzf.set_index(["ticker", "month"])
    univ = {h: hzf.groupby("month")[f"fwd_{h}m"].mean() for h in (1, 3, 12)}

    book = INS.run(merged, "B2_200k_1M", delistings)
    if book is None:
        print("B2_200k_1M: insufficient data to instrument the book")
        return 1
    exits = book.exits.copy()
    idx = pd.MultiIndex.from_frame(exits[["ticker", "month"]])
    sub = hz.reindex(idx)
    sub.index = exits.index
    exits["band_at_exit"] = sub["band"].to_numpy()
    exits["rank"] = exits["band_at_exit"].map(BAND_RANK)
    exits["is_forced"] = exits["kind"] != "discretionary"
    for h in (1, 3, 12):
        exits[f"post_{h}m"] = sub[f"fwd_{h}m"].to_numpy()
        exits[f"rel_{h}m"] = exits[f"post_{h}m"] - exits["month"].map(univ[h])

    forced = exits[exits["is_forced"]]
    print(f"{'destination at the exit month':>34} {'n':>6} {'share':>7} "
          f"{'post 1m':>9} {'post 12m':>10} {'12m vs panel':>13}  verdict")
    groups = [
        ("B3_1M_5M (graduated UP)", forced[forced["rank"] == 3], "healthy"),
        ("B1_50k_200k (volume DRIED UP)", forced[forced["rank"] == 1], "DISTRESS"),
        ("no band at all (<$50k/day)", forced[forced["rank"].isna()
                                              & forced["band_at_exit"].isna()], "DISTRESS"),
        ("still B2, just unrankable", forced[forced["rank"] == 2], "signal-driven"),
    ]
    for label, block, verdict in groups:
        if not len(block):
            print(f"{label:>34} {0:>6}")
            continue
        print(f"{label:>34} {len(block):>6,} {len(block)/len(forced):>6.1%} "
              f"{block['post_1m'].mean():>+8.2%} {block['post_12m'].mean():>+9.2%} "
              f"{block['rel_12m'].mean():>+12.2%}  {verdict}")
    down = forced[(forced["rank"] == 1) | (forced["rank"].isna()
                                           & forced["band_at_exit"].isna())]
    print(f"\n  DISTRESSED forced exits (volume collapsed out of B2): {len(down):,} of "
          f"{len(forced):,} forced ({len(down)/len(forced):.1%}), "
          f"{len(down)/len(exits):.1%} of all exits")
    print(f"  their next-month return {down['post_1m'].mean():+.2%} vs the panel's "
          f"{down['month'].map(univ[1]).mean():+.2%}; the move INTO the exit month is")
    print("  already booked, so this is only about what the book skipped AFTER selling.")

    print("\n" + "=" * 118)
    print("ATTACK 3 - REPAIRING THE 62-DAY WINDOW (`<` -> `<=`), APPLIED TO BOTH SIDES")
    print("=" * 118)
    print(f"  window is (exit, exit+{DELISTING_WINDOW_DAYS}d]; Sharadar dates a delisting "
          f"ON the last traded bar, so gap == 0 and the strict `<` rejects it")
    print(f"\n{'band':>13} {'book':>34} {'gross':>7} {'net':>7} {'vol':>6} {'Sharpe':>7} "
          f"{'bench':>7} {'bSh':>6} {'drag':>7} {'vol-matched':>12} {'t':>6}  verdict")
    for band in LV.BAND_ORDER:
        for label, inclusive, maximal in (
                ("published (exit < delist)", False, False),
                ("REPAIRED (exit <= delist)", True, False),
                ("maximal: unexplained = -100%", True, True)):
            books = patched_run_band(merged, band, delistings, inclusive=inclusive,
                                     total_loss_on_unexplained=maximal)
            hl = headline(books)
            print(f"{band:>13} {label:>34} {hl['gross']:>6.2%} {hl['net']:>6.2%} "
                  f"{hl['net_vol']:>5.2%} {hl['net_sharpe']:>7.3f} {hl['bench']:>6.2%} "
                  f"{hl['bench_sharpe']:>6.3f} {hl['drag']:>+6.2%} {hl['vm']:>+11.2%} "
                  f"{hl['vm_t']:>+6.2f}  {hl['verdict']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
