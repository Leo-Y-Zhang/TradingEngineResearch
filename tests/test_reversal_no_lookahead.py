"""The two anti-look-ahead guarantees in the short-horizon reversal sleeve.

Look-ahead is the failure mode that most reliably manufactures a backtest edge, and
it does it quietly: the returns simply come out better and nothing raises. Both
functions tested here exist specifically to prevent it, and neither had a test.

  weekly_grid    the signal is the last bar of a calendar week and execution is the
                 NEXT bar. Same-bar execution would let the book trade on a close it
                 could not have known.
  month_row_for  the monthly liquidity/spread panel is read one month BEHIND the
                 signal, because a month's row is stamped with each ticker's own last
                 trading day and is not knowable until the month is over.

Expected values are derived from those stated rules, not from running the functions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from research.sleeves.short_horizon_reversal import month_row_for, weekly_grid


# ── weekly_grid ───────────────────────────────────────────────────────────────

def test_execution_is_always_the_bar_after_the_signal():
    """The whole point of the function: no same-bar execution, ever."""
    dates = pd.bdate_range("2020-01-01", periods=60)

    signal_idx, exec_idx = weekly_grid(dates)

    assert len(signal_idx) > 0
    assert np.array_equal(exec_idx, signal_idx + 1)
    assert (exec_idx > signal_idx).all()


def test_the_signal_bar_is_the_last_trading_day_of_its_week():
    dates = pd.bdate_range("2020-01-06", periods=15)   # three full Mon-Fri weeks

    signal_idx, _ = weekly_grid(dates)

    chosen = dates[signal_idx]
    for day in chosen:
        # a Friday in a full business week, i.e. weekday 4
        assert day.weekday() == 4, f"{day.date()} is not the last trading day of its week"
    # one signal per calendar week present in the index, minus any dropped tail
    assert len(set(dates.to_period("W"))) - len(chosen) <= 1


def test_a_final_signal_with_no_following_bar_is_dropped():
    """The last week cannot be executed, so it must not appear at all."""
    dates = pd.bdate_range("2020-01-06", periods=15)   # ends on a Friday

    signal_idx, exec_idx = weekly_grid(dates)

    assert exec_idx.max() < len(dates)
    # the final bar is itself a week-end signal, but nothing follows it
    assert (len(dates) - 1) not in signal_idx.tolist()


def test_every_returned_index_is_inside_the_date_range():
    dates = pd.bdate_range("2019-06-03", periods=200)

    signal_idx, exec_idx = weekly_grid(dates)

    assert signal_idx.min() >= 0
    assert exec_idx.max() <= len(dates) - 1
    assert len(signal_idx) == len(exec_idx)


def test_a_short_week_still_signals_on_its_last_available_bar():
    """A holiday-shortened week ends on Thursday; the signal must follow the data."""
    dates = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09",
                              "2020-01-13", "2020-01-14"])   # Fri 10th missing

    signal_idx, exec_idx = weekly_grid(dates)

    # the first week's last available bar is Thursday the 9th, at row 3
    assert signal_idx[0] == 3
    assert exec_idx[0] == 4
    assert dates[signal_idx[0]].weekday() == 3


# ── month_row_for ─────────────────────────────────────────────────────────────

def _panel(dates: pd.DatetimeIndex, months: pd.PeriodIndex):
    """month_row_for reads only `dates` and `months`, so this is a faithful stand-in."""
    return SimpleNamespace(dates=dates, months=months)


def test_the_monthly_row_used_is_the_previous_month_never_the_current_one():
    dates = pd.DatetimeIndex(["2020-03-05", "2020-04-07", "2020-05-11"])
    months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03", "2020-04"], freq="M")

    rows = month_row_for(_panel(dates, months), np.arange(len(dates)))

    # a March signal reads February (row 1), April reads March (row 2), May reads April (3)
    assert rows.tolist() == [1, 2, 3]
    for i, d in enumerate(dates):
        assert months[rows[i]] == d.to_period("M") - 1


def test_a_signal_whose_previous_month_is_absent_reports_minus_one():
    """Missing is flagged, not silently mapped onto some other month's liquidity."""
    dates = pd.DatetimeIndex(["2020-03-05"])
    months = pd.PeriodIndex(["2020-03", "2020-04"], freq="M")   # no February

    rows = month_row_for(_panel(dates, months), np.array([0]))

    assert rows.tolist() == [-1]


def test_the_current_month_is_never_selected_even_when_it_is_available():
    """The current month exists in the panel here; it still must not be chosen."""
    dates = pd.DatetimeIndex(["2020-04-20"])
    months = pd.PeriodIndex(["2020-03", "2020-04"], freq="M")

    rows = month_row_for(_panel(dates, months), np.array([0]))

    assert rows.tolist() == [0]
    assert months[rows[0]] == pd.Period("2020-03", freq="M")
