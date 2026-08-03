"""Contract tests for the risk-parity primitives.

``research/sleeves/riskparity.py`` is imported by ``breadth_ladder`` and by the
convexity study, so its numbers feed research conclusions -- and it had no direct
test at all. Every expected value below is derived by hand from the docstring or the
prereg, never by calling the function under test and recording what came back. A test
that mirrors the implementation cannot catch the implementation being wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.sleeves.multiasset_trend import ELIGIBLE_MIN_OBS
from research.sleeves.riskparity import (
    Book,
    drawdown_report,
    eligibility,
    levered,
    weights_ew,
    weights_rp_naive,
)


def _months(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-31", periods=n, freq="ME")


# ── Eligibility (prereg: >=36 observations AND a positive trailing vol) ────────

def test_eligibility_requires_both_enough_history_and_a_positive_vol():
    n = ELIGIBLE_MIN_OBS + 4
    idx = _months(n)
    x = pd.DataFrame({"A": 0.01, "B": 0.01}, index=idx)
    sigma = pd.DataFrame({"A": 0.10, "B": 0.10}, index=idx)

    elig = eligibility(x, sigma)

    # the count is cumulative and inclusive, so the row that reaches the minimum
    # is the first eligible one
    assert not elig["A"].iloc[ELIGIBLE_MIN_OBS - 2]
    assert elig["A"].iloc[ELIGIBLE_MIN_OBS - 1]
    assert elig["A"].iloc[-1]


def test_eligibility_is_false_where_vol_is_zero_or_missing_however_long_the_history():
    n = ELIGIBLE_MIN_OBS + 4
    idx = _months(n)
    x = pd.DataFrame({"A": 0.01, "B": 0.01}, index=idx)
    sigma = pd.DataFrame({"A": 0.10, "B": 0.10}, index=idx)
    sigma.loc[idx[-1], "A"] = 0.0          # a zero vol would divide by zero downstream
    sigma.loc[idx[-2], "B"] = np.nan

    elig = eligibility(x, sigma)

    assert not elig.loc[idx[-1], "A"]
    assert not elig.loc[idx[-2], "B"]
    assert elig.loc[idx[-1], "B"]


# ── Weights ───────────────────────────────────────────────────────────────────

def test_equal_weights_split_one_evenly_across_the_eligible_set_only():
    idx = _months(2)
    elig = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [False, False]},
                        index=idx)
    sigma = pd.DataFrame(0.1, index=idx, columns=["A", "B", "C"])

    w = weights_ew(elig, sigma)

    assert w.loc[idx[0]].to_dict() == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})
    assert w.loc[idx[1]].to_dict() == pytest.approx({"A": 1.0, "B": 0.0, "C": 0.0})


def test_weights_are_all_zero_when_nothing_is_eligible_rather_than_nan():
    """A row with no eligible instrument must renormalise to zeros, not divide by zero."""
    idx = _months(1)
    elig = pd.DataFrame({"A": [False], "B": [False]}, index=idx)
    sigma = pd.DataFrame(0.1, index=idx, columns=["A", "B"])

    for w in (weights_ew(elig, sigma), weights_rp_naive(elig, sigma)):
        assert w.loc[idx[0]].to_dict() == pytest.approx({"A": 0.0, "B": 0.0})
        assert not w.isna().any().any()


def test_inverse_vol_weights_are_proportional_to_one_over_sigma():
    """W1: w_i proportional to 1/sigma_i, long only, summing to one.

    With sigma = (0.10, 0.20) the halved-risk name gets exactly twice the weight, so
    the split is 2/3 and 1/3 -- derived from the definition, not from the function.
    """
    idx = _months(1)
    elig = pd.DataFrame({"A": [True], "B": [True]}, index=idx)
    sigma = pd.DataFrame({"A": [0.10], "B": [0.20]}, index=idx)

    w = weights_rp_naive(elig, sigma)

    assert w.loc[idx[0], "A"] == pytest.approx(2.0 / 3.0)
    assert w.loc[idx[0], "B"] == pytest.approx(1.0 / 3.0)
    assert w.loc[idx[0]].sum() == pytest.approx(1.0)
    # the defining ratio: w_A / w_B == sigma_B / sigma_A
    assert w.loc[idx[0], "A"] / w.loc[idx[0], "B"] == pytest.approx(0.20 / 0.10)


# ── Drawdown and the ruin guarantee ───────────────────────────────────────────

def test_drawdown_report_matches_a_hand_compounded_path():
    """Path 1.10, 0.80, 1.05, 1.10 compounds to 1.0164 with a trough at 0.88."""
    idx = _months(4)
    total = pd.Series([0.10, -0.20, 0.05, 0.10], index=idx)

    rep = drawdown_report(total)

    assert rep["months"] == 4
    assert rep["max_drawdown"] == pytest.approx(0.88 / 1.10 - 1.0)   # -0.20 exactly
    assert rep["worst_month"] == pytest.approx(-0.20)
    assert rep["best_month"] == pytest.approx(0.10)
    assert rep["compound_annual"] == pytest.approx(1.0164 ** 3 - 1.0)
    assert rep["dd_peak"] == str(idx[0].date())
    assert rep["dd_trough"] == str(idx[1].date())
    # 1.0164 never regains the 1.10 peak, so it is still underwater at the end
    assert rep["recovered"] is False
    assert rep["dd_recovery"] is None
    assert rep["ruin"] is False


def test_drawdown_report_reports_recovery_when_the_peak_is_regained():
    idx = _months(3)
    total = pd.Series([0.10, -0.50, 1.00], index=idx)   # 1.10 -> 0.55 -> 1.10

    rep = drawdown_report(total)

    assert rep["max_drawdown"] == pytest.approx(-0.50)
    assert rep["dd_peak"] == str(idx[0].date())
    assert rep["dd_trough"] == str(idx[1].date())
    assert rep["recovered"] is True
    assert rep["dd_recovery"] == str(idx[2].date())
    assert rep["months_peak_to_trough"] == 1
    assert rep["months_trough_to_recovery"] == 1


def test_drawdown_is_measured_from_the_running_peak_not_from_starting_capital():
    """Pinning a definitional nuance, because it is easy to misread the number.

    ``curve = (1 + r).cumprod()`` and the peak is that curve's running maximum, so the
    curve STARTS at the first month's value rather than at 1.0. An opening loss,
    before any higher peak has been set, therefore reports no drawdown at all.

    For a real book this is immaterial -- the running peak exceeds the opening value
    within months, and no 700-month series has its deepest drawdown in month one -- but
    it means `max_drawdown` is peak-to-trough, NOT loss-against-initial-capital, and
    the two differ on short or front-loaded series.
    """
    idx = _months(2)
    rep = drawdown_report(pd.Series([-0.50, 0.00], index=idx))

    assert rep["max_drawdown"] == pytest.approx(0.0)
    assert rep["worst_month"] == pytest.approx(-0.50)   # the loss is still reported


def test_ruin_is_terminal_and_no_later_gain_undoes_it():
    """The documented guarantee: -100% ends the account and is never compounded through.

    Without the truncation a +1000% month afterwards would still multiply into a
    curve of zero, but the series would keep reporting months that cannot exist. This
    is the property most worth pinning, because a book that silently trades on past
    ruin produces a plausible-looking number from an impossible path.
    """
    idx = _months(3)
    total = pd.Series([0.05, -1.00, 10.0], index=idx)

    rep = drawdown_report(total)

    assert rep["ruin"] is True
    assert rep["ruin_date"] == str(idx[1].date())
    assert rep["compound_annual"] == -1.0
    assert rep["months"] == 2, "the series must stop at ruin, not run on into recovery"
    assert rep["max_drawdown"] == pytest.approx(-1.0)


def test_a_return_worse_than_minus_one_also_counts_as_ruin():
    idx = _months(2)
    rep = drawdown_report(pd.Series([0.01, -1.50], index=idx))
    assert rep["ruin"] is True
    assert rep["compound_annual"] == -1.0


# ── Leverage, cost and financing ──────────────────────────────────────────────

def _flat_book(n: int = 6, ret: float = 0.01) -> Book:
    """Two instruments, constant 50/50 weights, constant excess return."""
    idx = _months(n)
    cols = ["A", "B"]
    x = pd.DataFrame(ret, index=idx, columns=cols)
    w = pd.DataFrame(0.5, index=idx, columns=cols)
    return Book(
        name="flat",
        w=w,
        x=x,
        excess=pd.Series(ret, index=idx),
        live=pd.Series(True, index=idx),
        sigma_book=pd.Series(0.10, index=idx),
        elig_count=pd.Series(2, index=idx),
        extra={},
    )


def test_at_unit_leverage_with_no_costs_the_book_earns_its_own_return():
    bk = _flat_book(ret=0.01)
    cash = pd.Series(0.0, index=bk.x.index)

    out = levered(bk, cash, tau=0.10, cost=0.0, spread=0.0,
                  k_override=pd.Series(1.0, index=bk.x.index))

    # weights decided at t-1 earn t's return: 0.5 x 0.01 x 2 instruments
    assert out["net_excess"].to_numpy() == pytest.approx(0.01)
    assert out["financing"].to_numpy() == pytest.approx(0.0)


def test_financing_is_charged_only_on_the_borrowed_part_of_the_notional():
    """spread is annual on max(k-1, 0), so at k=2 exactly one unit is borrowed."""
    bk = _flat_book(ret=0.01)
    cash = pd.Series(0.0, index=bk.x.index)
    spread = 0.015

    out = levered(bk, cash, tau=0.10, cost=0.0, spread=spread,
                  k_override=pd.Series(2.0, index=bk.x.index))

    assert out["financing"].to_numpy() == pytest.approx(spread / 12.0)
    # gross doubles with the leverage, and financing comes off it
    assert out["gross_excess"].dropna().to_numpy() == pytest.approx(0.02)
    assert out["net_excess"].to_numpy() == pytest.approx(0.02 - spread / 12.0)


def test_no_financing_is_charged_below_unit_leverage():
    bk = _flat_book(ret=0.01)
    cash = pd.Series(0.0, index=bk.x.index)

    out = levered(bk, cash, tau=0.10, cost=0.0, spread=0.03,
                  k_override=pd.Series(0.5, index=bk.x.index))

    assert out["financing"].to_numpy() == pytest.approx(0.0)


def test_total_return_adds_the_cash_rate_back_to_the_excess_return():
    """The panel holds EXCESS returns, so total = net excess + the bill."""
    bk = _flat_book(ret=0.01)
    monthly_bill = 0.002
    cash = pd.Series(monthly_bill, index=bk.x.index)

    out = levered(bk, cash, tau=0.10, cost=0.0, spread=0.0,
                  k_override=pd.Series(1.0, index=bk.x.index))

    assert out["total"].to_numpy() == pytest.approx(0.01 + monthly_bill)


def test_a_flat_borrow_rate_is_a_subsidy_when_the_bill_rate_is_higher():
    """Documented behaviour of flat_rate: the charge is (flat/12 - cash) per unit.

    When the bill exceeded the flat rate -- the 1970s and 1980s -- that quantity is
    negative, so the book is PAID to borrow. The docstring says so explicitly; this
    pins it so nobody 'fixes' it into an absolute charge later.
    """
    bk = _flat_book(ret=0.01)
    flat = 0.03
    bill_monthly = 0.01                       # 12%/yr, far above the flat rate
    cash = pd.Series(bill_monthly, index=bk.x.index)

    out = levered(bk, cash, tau=0.10, cost=0.0, spread=None, flat_rate=flat,
                  k_override=pd.Series(2.0, index=bk.x.index))

    expected = (flat / 12.0 - bill_monthly) * 1.0
    assert expected < 0.0
    assert out["financing"].to_numpy() == pytest.approx(expected)


def test_trading_cost_is_half_the_round_trip_applied_to_turnover():
    """cost_s = 0.5 * cost * turnover, i.e. a round-trip rate charged one way."""
    bk = _flat_book(n=4, ret=0.0)
    cash = pd.Series(0.0, index=bk.x.index)
    cost = 0.0020

    out = levered(bk, cash, tau=0.10, cost=cost, spread=0.0,
                  k_override=pd.Series(1.0, index=bk.x.index))

    assert out["cost"].to_numpy() == pytest.approx(0.5 * cost * out["turnover"].to_numpy())
    # weights never move after the book is established, so later turnover is zero
    assert out["turnover"].iloc[-1] == pytest.approx(0.0)
