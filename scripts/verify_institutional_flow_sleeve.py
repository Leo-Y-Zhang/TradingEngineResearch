"""Independent checks on the institutional-flow sleeve's accounting and PIT discipline.

These are the checks the prior programme learned to run only after two accounting defects
had already produced impossible numbers (`capacity_curve_result.md` §4). They assert
rather than print, so a regression fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR  # noqa: E402
from research.delisting import CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW  # noqa: E402
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW  # noqa: E402
from research.delisting import in_window_mask  # noqa: E402
from research.sleeves.institutional_flow import (  # noqa: E402
    FILING_LAG_DAYS,
    build_signal_panel,
    market_month_ends,
    rebalance_schedule,
)


def main() -> int:
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    ownership = pd.read_parquet(PANEL_DIR / "sf3_ownership_dev.parquet")
    marketcap = pd.read_parquet(PANEL_DIR / "quarter_end_marketcap_dev.parquet")
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    month_ends = market_month_ends(panel)
    panel = panel[panel["date"].isin(month_ends)].copy()
    schedule = rebalance_schedule(
        pd.DatetimeIndex(ownership["calendardate"].unique()).sort_values(), month_ends)
    signals = build_signal_panel(panel, ownership, marketcap, schedule)

    failures: list[str] = []

    # 1. DEV window. Nothing may be read past the cutoff.
    for name, series in (("panel", panel["date"]),
                         ("signals", signals["date"]),
                         ("ownership", ownership["calendardate"]),
                         ("marketcap", marketcap["marketcap_date"])):
        if series.max() > DEV_CUTOFF:
            failures.append(f"{name} reads past DEV cutoff: {series.max()}")

    # 2. Point in time. Every rebalance must sit at or after the 13F deadline of the
    #    quarter it trades on. This is the check that the whole sleeve rests on.
    for row in schedule.itertuples():
        deadline = row.quarter + pd.Timedelta(days=FILING_LAG_DAYS)
        if row.rebalance_date < deadline:
            failures.append(
                f"LOOKAHEAD: quarter {row.quarter.date()} traded on "
                f"{row.rebalance_date.date()}, before its {deadline.date()} deadline")
        lag = (row.rebalance_date - row.quarter).days
        if lag > 110:
            failures.append(f"quarter {row.quarter.date()} traded {lag} days late")

    # 3. The rebalance grid must be real month-ends, not a delisting name's last bar.
    for date in signals["date"].unique():
        names = int((signals["date"] == date).sum())
        if names < 100:
            failures.append(f"cross-section {pd.Timestamp(date).date()} has {names} "
                            "names - phantom month-end?")

    # 4. Ownership must be a plausible fraction. The SF1 share route produced 0.01%.
    own = signals["own_q"].dropna()
    if not (0.15 <= own.mean() <= 0.95):
        failures.append(f"mean ownership {own.mean():.4f} is not a plausible fraction")

    # 5. Delisting exposure. The two prior defects both showed up as an implausible
    #    number of terminal returns being booked, so count them explicitly.
    terminal = delistings.set_index("ticker")
    held_window = signals[["ticker", "date"]].drop_duplicates()
    delist_date = held_window["ticker"].map(terminal["date"])
    # Read the window from `research.delisting` rather than restating it. This verifier
    # previously hard-coded the same strict lower edge as the sleeve it checks, so it
    # reproduced the off-by-one bug-for-bug and could never have detected it.
    in_window = in_window_mask(held_window["date"], delist_date, REGISTERED_DELISTING_WINDOW)
    corrected = in_window_mask(held_window["date"], delist_date, CORRECTED_DELISTING_WINDOW)
    print(f"signal cells                       {len(signals):>10,}")
    print(f"distinct tickers                   "
          f"{signals['ticker'].nunique():>10,}")
    print(f"cells with a delisting in 62 days  {int(in_window.sum()):>10,} "
          f"({in_window.mean():.3%})")
    print(f"  same, CORRECTED lower edge       {int(corrected.sum()):>10,} "
          f"({corrected.mean():.3%})")
    print(f"mean institutional ownership       {own.mean():>10.1%}")
    print(f"rebalances                         {len(schedule):>10,}")
    print(f"first / last rebalance             "
          f"{schedule['rebalance_date'].min().date()} .. "
          f"{schedule['rebalance_date'].max().date()}")
    print(f"mean filing lag actually used      "
          f"{(schedule['rebalance_date'] - schedule['quarter']).dt.days.mean():>10.0f}"
          " days")

    # 6. The signal must be a proper cross-sectional z-score.
    stats = signals.groupby("date")["signal"].agg(["mean", "std"])
    if not np.allclose(stats["mean"], 0.0, atol=1e-8):
        failures.append("signal is not cross-sectionally demeaned")
    if not np.allclose(stats["std"], 1.0, atol=1e-6):
        failures.append("signal is not cross-sectionally unit-variance")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
