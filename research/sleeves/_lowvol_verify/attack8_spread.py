"""ATTACK -- THE UNIVERSE CORRECTION ITSELF. 37.2% of held cells are `upper_bound`.

That 37.2% is MEASURED (6,210 held cells, 2026-07-31), against a 43.0% universe
share. It was previously a hardcoded constant printed as though computed; see
VERIFY-1 in the defect register.

The re-test's headline change is that `upper_bound` names were let back in. If their spread
is charged too cheaply, the whole result is an artefact of the correction rather than of the
signal. `bounds_from_estimate` charges the CONSERVATIVE bound at `max(estimate, tick)` for
every regime -- so an `upper_bound` name pays its full ceiling estimate -- but that is a
claim about code, and the claim that matters is what the book actually paid.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack8_spread
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.memguard import start as memguard_start
from research.multiasset.carry import vol_matched_active
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build

BAND = "B2_200k_1M"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    # This script loads a full capacity panel unattended. Guard the machine:
    # on a low-memory breach it terminates itself rather than paging the box.
    memguard_start(floor_gb=0.9, label="attack8 spread verification")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    band = merged[merged["band_group"] == BAND]

    print("=" * 112)
    print("WHAT THE `upper_bound` CELLS ARE ACTUALLY CHARGED")
    print("=" * 112)
    print(f"{'regime':>14} {'cells':>9} {'median $vol':>13} {'median cons':>13} "
          f"{'median real':>13} {'median vol':>11} {'mean fwd ret':>13}")
    for regime in ("measured", "upper_bound"):
        b = band[band["spread_regime"] == regime]
        print(f"{regime:>14} {len(b):>9,} ${b['median_dollar_volume'].median()/1e3:>11.0f}k "
              f"{b['spread_conservative'].median()*1e4:>11.0f}bp "
              f"{b['spread_realistic'].median()*1e4:>11.0f}bp "
              f"{b['realised_vol'].median()*np.sqrt(252):>10.1%} "
              f"{b['forward_return'].mean()*12:>12.2%}")
    print("  conservative == the EDGE estimate itself for BOTH regimes; for `upper_bound`")
    print("  that estimate is the resolution CEILING, so the conservative book charges the")
    print("  most the spread could possibly be. That is the right direction.")

    book = INS.run(merged, BAND, delistings)
    if book is None:
        print(f"{BAND}: insufficient data to instrument the book")
        return 1
    held = book.holdings_log

    # DEFECT FIXED 2026-07-31. This line previously read
    #     float((merged.set_index(['ticker','date']) is not None) and 0.372)
    # which evaluates to the literal 0.372 for ANY input, empty frames included:
    # `X is not None` is True, and `True and 0.372` is 0.372. It printed the
    # published number while looking like a measurement, and `held` -- the frame
    # that should have produced it -- was assigned and never used, which is how
    # ruff's F841 surfaced it. The claim two prints below ("the book holds
    # PROPORTIONALLY FEWER upper_bound names than the universe") was therefore
    # comparing a hardcoded constant against a genuinely computed universe share.
    # Computed from the book now; if the join cannot be made the script SAYS so
    # rather than printing a number.
    # holdings_log["month"] is a pandas PERIOD (instrumented.py builds it with
    # .dt.to_period("M")), so joining it against a Timestamp `date` silently
    # matches nothing. The first version of this fix did exactly that and printed
    # NOT COMPUTED, which is precisely why the guard clause exists.
    regime = merged[["ticker", "date", "spread_regime"]].copy()
    regime["month"] = regime["date"].dt.to_period("M")
    joined = held.merge(regime[["ticker", "month", "spread_regime"]],
                        on=["ticker", "month"], how="inner")
    universe_share = float((band["spread_regime"] == "upper_bound").mean())

    if len(joined):
        share = float((joined["spread_regime"] == "upper_bound").mean())
        print(f"\n  held cells that are upper_bound: {share:.1%} "
              f"(measured over {len(joined):,} held cells; "
              f"the published figure was 37.2%)")
        print(f"  universe cells that are upper_bound: {universe_share:.1%}")
        # Stated only when the comparison it rests on actually exists. This used
        # to print unconditionally, i.e. even when the held share was NOT COMPUTED.
        if share < universe_share:
            print("  the book holds PROPORTIONALLY FEWER upper_bound names than the")
            print("  universe, so the correction is not a device for loading up on")
            print("  the cheapest-priced cells.")
        else:
            print("  WARNING: the book holds proportionally MORE upper_bound names")
            print("  than the universe. The original claim fails on this measurement.")
    else:
        print("\n  held cells that are upper_bound: NOT COMPUTED - the holdings/regime "
              "join produced no rows. Do NOT quote 37.2% as measured.")
        print(f"  universe cells that are upper_bound: {universe_share:.1%}")
        print("  NO CONCLUSION is drawn about the book's composition: the held share")
        print("  is unknown, so there is nothing to compare the universe share with.")

    print("\n" + "=" * 112)
    print("STRESS - MULTIPLY THE SPREAD ON EVERY `upper_bound` CELL")
    print("=" * 112)
    print(f"{'upper_bound spread x':>22} {'cost/yr':>9} {'one-way':>9} {'net':>8} "
          f"{'Sharpe':>8} {'vol-matched':>12} {'t':>7}  gate (i)+(ii)")
    for mult in (1.0, 1.5, 2.0, 3.0, 5.0):
        stressed = merged.copy()
        mask = stressed["spread_regime"] == "upper_bound"
        stressed.loc[mask, "spread_conservative"] *= mult
        stressed.loc[mask, "spread_realistic"] *= mult
        books = LV.run_band(stressed, BAND, delistings)
        if books is None:
            print(f"  x{mult}: insufficient data, stress level skipped")
            continue
        e = LV.evaluate_band(books, n_trials=LV.N_TRIALS)
        c = e["bounds"]["conservative"]
        vm = c["vol_matched"]
        ok = (vm["vol_matched_active_annual"] > 0.02
              and vm["vol_matched_active_tstat"] > 2.0)
        print(f"{mult:>21.1f}x {c['cost_annual_total']:>8.2%} "
              f"{c['cost_one_way_bps']:>7.1f}bp {c['net']['annual_arithmetic']:>7.2%} "
              f"{c['net']['sharpe']:>8.3f} {vm['vol_matched_active_annual']:>+11.2%} "
              f"{vm['vol_matched_active_tstat']:>+7.2f}  {'passes' if ok else 'FAILS'}")

    print("\n" + "=" * 112)
    print("STRESS - A FLAT EXTRA COST ON EVERY LEG (how much slippage kills the gate?)")
    print("=" * 112)
    books = LV.run_band(merged, BAND, delistings)
    if books is None:
        print(f"{BAND}: insufficient data to price the flat-cost stress")
        return 1
    legs_per_month = books.legs_traded / len(books.gross)
    print(f"  {books.legs_traded:,} legs over {len(books.gross)} months = "
          f"{legs_per_month:.2f} legs/month at 1/{30} weight each")
    print(f"{'extra bps per leg':>20} {'extra cost/yr':>14} {'net':>8} {'Sharpe':>8} "
          f"{'vol-matched':>12} {'t':>7}  gate (i)+(ii)")
    for bps in (0, 10, 20, 30, 40, 50, 75, 100):
        extra = np.zeros(len(books.gross))
        # rebuild per-month leg counts from the instrumented run
        counts = pd.Series(0.0, index=pd.Index(books.months, name="month"))
        for m in book.exits["month"]:
            counts.loc[m] += 1
        for m in book.entries["month"]:
            counts.loc[m] += 1
        extra = counts.to_numpy() * (bps / 1e4) / 30.0
        net = np.maximum(books.gross - books.cost_conservative - extra, -1.0)
        vm = vol_matched_active(pd.Series(net), pd.Series(books.benchmark))
        ok = (vm["vol_matched_active_annual"] > 0.02
              and vm["vol_matched_active_tstat"] > 2.0)
        print(f"{bps:>19}bp {float(extra.sum())/(len(extra)/12):>13.2%} "
              f"{INS.annual(net):>7.2%} {INS.sharpe(net):>8.3f} "
              f"{vm['vol_matched_active_annual']:>+11.2%} "
              f"{vm['vol_matched_active_tstat']:>+7.2f}  {'passes' if ok else 'FAILS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
