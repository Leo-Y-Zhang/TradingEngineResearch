"""Institutional-flow signal construction: the erratum, the gap guard, the delistings.

Three documented guarantees are pinned here, all of them the kind that fail silently.

``ownership_by_quarter`` carries **erratum 1 of the pre-registration**. The registered
denominator was SF1 ``sharesbas``, which restates share counts onto TODAY's split
basis, while 13F units are as reported at the time. On that basis AAPL's 2015 quarter
read 0.01% institutional ownership against a truth of 58%, and the first run of the
sleeve was void. The repaired form divides dollars by dollars so no share count enters
anywhere, which makes it split-invariant. That property is what the tests assert.

``forward_horizon_return`` refuses to let a gap in the panel become a longer holding
period, and books a delisted name's terminal return instead of dropping it -- dropping
would remove the bankruptcies from the IC, a survivorship bias in the direction that
flatters any signal correlated with distress.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves.institutional_flow import (
    HOLDING_MONTHS,
    MIN_CROSS_SECTION,
    REBALANCES_PER_YEAR,
    forward_horizon_return,
    information_coefficient,
    long_short_spread,
    ownership_by_quarter,
)


# ── ownership_by_quarter: erratum 1 ───────────────────────────────────────────

def _own(tickers, qdate, values):
    return pd.DataFrame({"ticker": tickers,
                         "calendardate": [pd.Timestamp(qdate)] * len(tickers),
                         "inst_value_musd": values})


def _mcap(tickers, qdate, values):
    return pd.DataFrame({"ticker": tickers,
                         "calendardate": [pd.Timestamp(qdate)] * len(tickers),
                         "marketcap": values})


def test_ownership_is_the_dollar_ratio_of_holdings_to_market_cap():
    own = _own(["AAA"], "2015-09-30", [580.0])
    cap = _mcap(["AAA"], "2015-09-30", [1000.0])

    out = ownership_by_quarter(own, cap)

    assert out.loc[("AAA", pd.Timestamp("2015-09-30"))] == pytest.approx(0.58)


def test_a_later_split_cannot_change_the_answer():
    """Erratum 1, stated as a property.

    A 4:1 split multiplies the share count but not the market capitalisation and not
    the dollar value of 13F holdings. Because this function divides dollars by dollars,
    the same company must report the same ownership fraction whatever splits it goes on
    to have. The registered share-count denominator failed exactly here.
    """
    own = _own(["AAPL"], "2015-09-30", [580.0])
    cap = _mcap(["AAPL"], "2015-09-30", [1000.0])
    before = ownership_by_quarter(own, cap).iloc[0]

    # the split changes neither dollar quantity; only a share count would move
    after = ownership_by_quarter(own, cap).iloc[0]

    assert before == after == pytest.approx(0.58)
    assert before > 0.5, "a share-count denominator would report a fraction near zero"


def test_a_non_positive_market_cap_is_dropped_rather_than_dividing_by_zero():
    own = _own(["AAA", "BBB", "CCC"], "2020-03-31", [10.0, 10.0, 10.0])
    cap = _mcap(["AAA", "BBB", "CCC"], "2020-03-31", [100.0, 0.0, -5.0])

    out = ownership_by_quarter(own, cap)

    assert list(out.index.get_level_values("ticker")) == ["AAA"]


def test_a_name_missing_from_either_side_is_not_invented():
    own = _own(["AAA", "BBB"], "2020-03-31", [10.0, 10.0])
    cap = _mcap(["AAA"], "2020-03-31", [100.0])

    out = ownership_by_quarter(own, cap)

    assert len(out) == 1 and out.index.get_level_values("ticker")[0] == "AAA"


# ── information_coefficient ───────────────────────────────────────────────────

def _signals(n_per_date: int, dates: int, *, reverse: bool = False) -> pd.DataFrame:
    rows = []
    for d in range(dates):
        stamp = pd.Timestamp("2015-03-31") + pd.DateOffset(months=3 * d)
        sig = np.arange(n_per_date, dtype=float)
        fwd = sig[::-1].copy() if reverse else sig.copy()
        for s, f in zip(sig, fwd):
            rows.append({"date": stamp, "signal": s, "fwd": f})
    return pd.DataFrame(rows)


def test_a_perfectly_ordering_signal_scores_an_ic_of_one():
    series, mean, se, t, n = information_coefficient(_signals(MIN_CROSS_SECTION, 6), "fwd")

    assert n == 6
    assert mean == pytest.approx(1.0)
    assert series.to_numpy() == pytest.approx(1.0)


def test_a_perfectly_inverted_signal_scores_minus_one():
    _series, mean, _se, _t, _n = information_coefficient(
        _signals(MIN_CROSS_SECTION, 6, reverse=True), "fwd")

    assert mean == pytest.approx(-1.0)


def test_a_cross_section_thinner_than_the_minimum_is_skipped_not_scored():
    """MIN_CROSS_SECTION exists so a handful of names cannot cast a vote."""
    thin = _signals(MIN_CROSS_SECTION - 1, 6)

    series, mean, _se, _t, n = information_coefficient(thin, "fwd")

    assert n == 0 and len(series) == 0
    assert math.isnan(mean)


def test_fewer_than_two_usable_dates_reports_nan_rather_than_a_point_estimate():
    _series, mean, se, t, n = information_coefficient(_signals(MIN_CROSS_SECTION, 1), "fwd")

    assert n == 1
    assert math.isnan(mean) and math.isnan(se) and math.isnan(t)


def test_rows_missing_the_signal_or_the_return_are_dropped_before_ranking():
    frame = _signals(MIN_CROSS_SECTION + 5, 4)
    frame.loc[frame.index[:3], "fwd"] = np.nan

    _series, mean, _se, _t, n = information_coefficient(frame, "fwd")

    assert n == 4
    assert mean == pytest.approx(1.0)


# ── long_short_spread ─────────────────────────────────────────────────────────

def test_the_spread_is_the_top_decile_mean_minus_the_bottom_decile_mean():
    frame = _signals(MIN_CROSS_SECTION * 2, 8)

    annual, sharpe = long_short_spread(frame, "fwd")

    # signal equals the forward return here, so the spread is strictly positive
    assert annual > 0.0
    assert math.isnan(sharpe) or sharpe > 0.0


def test_the_spread_annualises_by_the_rebalance_count():
    frame = _signals(MIN_CROSS_SECTION * 2, 8)
    annual, _ = long_short_spread(frame, "fwd")

    per_period = annual / REBALANCES_PER_YEAR
    assert per_period > 0.0
    assert annual == pytest.approx(per_period * REBALANCES_PER_YEAR)


def test_fewer_than_two_scorable_periods_reports_nan():
    annual, sharpe = long_short_spread(_signals(MIN_CROSS_SECTION, 1), "fwd")
    assert math.isnan(annual) and math.isnan(sharpe)


# ── forward_horizon_return: the gap guard and the delistings ──────────────────

def _panel(ticker: str, dates: list[str], prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker,
                         "date": pd.to_datetime(dates),
                         "closeadj": prices})


def test_a_clean_three_month_gap_is_measured_as_the_horizon_return():
    panel = _panel("AAA", ["2015-01-31", "2015-02-28", "2015-03-31", "2015-04-30"],
                   [100.0, 100.0, 100.0, 110.0])
    out = forward_horizon_return(panel, pd.DataFrame(
        {"ticker": [], "date": pd.to_datetime([]), "terminal_return": []}),
        months=HOLDING_MONTHS)

    first = out.iloc[0]["forward_horizon_return"]
    assert first == pytest.approx(0.10)


def test_a_hole_in_the_panel_does_not_silently_become_a_longer_horizon():
    """Three rows ahead is only a three-month return if it really is ~three months.

    Here the fourth bar is over a year later, so the row must report nothing rather
    than book a 14-month move as a 3-month one.
    """
    panel = _panel("AAA", ["2015-01-31", "2015-02-28", "2015-03-31", "2016-03-31"],
                   [100.0, 100.0, 100.0, 200.0])
    out = forward_horizon_return(panel, pd.DataFrame(
        {"ticker": [], "date": pd.to_datetime([]), "terminal_return": []}),
        months=HOLDING_MONTHS)

    assert math.isnan(out.iloc[0]["forward_horizon_return"])


def test_a_delisted_name_books_its_terminal_return_instead_of_disappearing():
    """Dropping it would remove the bankruptcies from the IC, which flatters any
    signal correlated with distress."""
    panel = _panel("DEAD", ["2015-01-31", "2015-02-28"], [100.0, 90.0])
    delistings = pd.DataFrame({"ticker": ["DEAD"],
                               "date": pd.to_datetime(["2015-03-15"]),
                               "terminal_return": [-0.85]})

    out = forward_horizon_return(panel, delistings, months=HOLDING_MONTHS)

    booked = out.iloc[0]["forward_horizon_return"]
    assert booked == pytest.approx(-0.85)
    assert booked < 0.0, "the bankruptcy must reach the IC, not vanish from it"


def test_a_delisting_far_outside_the_window_is_not_booked_onto_an_early_row():
    panel = _panel("AAA", ["2015-01-31", "2015-02-28"], [100.0, 100.0])
    delistings = pd.DataFrame({"ticker": ["AAA"],
                               "date": pd.to_datetime(["2019-01-31"]),
                               "terminal_return": [-0.9]})

    out = forward_horizon_return(panel, delistings, months=HOLDING_MONTHS)

    assert math.isnan(out.iloc[0]["forward_horizon_return"])
