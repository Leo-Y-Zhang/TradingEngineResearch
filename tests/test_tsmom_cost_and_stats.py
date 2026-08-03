"""The TSMOM cost model and return statistics.

``tsmom_multitimeframe`` is the largest untested module in the sleeve set (412
statements, 0% covered). Its data-loading half needs a real panel, but the cost
model and the return statistics are pure and are exactly where a quiet error would
flatter the book, so they are what is pinned here.

The commission schedule and the spread fallback are stated in the module's own
docstrings; every expected number below is worked out from those rules by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.sleeves.tsmom_multitimeframe import (
    COMMISSION_CAP_FRACTION,
    COMMISSION_MIN_ORDER,
    COMMISSION_PER_SHARE,
    LIQUID_SPREAD_SUBSTITUTE,
    TRADING_DAYS,
    annualised,
    execution_cost,
    max_drawdown,
)


# ── execution_cost: the free-lunch guards ─────────────────────────────────────

def test_no_trade_costs_nothing():
    assert execution_cost(np.zeros(3), np.full(3, 0.001), np.full(3, 50.0)) == 0.0


def test_cost_is_half_the_spread_on_notional_plus_commission():
    """One-way execution pays half the quoted spread, then the per-share schedule.

    10,000 dollars at a 10bp spread and a 50 dollar price: 0.5 x 0.001 x 10000 = 5.00
    of spread, and 200 shares at 0.0035 = 0.70 of commission, which clears both the
    0.35 order minimum and the 1% cap.
    """
    cost = execution_cost(np.array([10_000.0]), np.array([0.001]), np.array([50.0]))
    assert cost == pytest.approx(5.0 + 0.70)


def test_an_unknown_spread_is_not_free():
    """The documented guard: a name that left the measurable universe still has to be
    sold, and pricing that exit at zero would be a free lunch.

    This is the same class of defect the low-vol work found in its own book -- exits
    that cost nothing -- so it is worth a test that fails loudly if the fallback is
    ever removed.
    """
    notional, price = 10_000.0, 50.0
    cost = execution_cost(np.array([notional]), np.array([np.nan]), np.array([price]))

    expected_spread = 0.5 * LIQUID_SPREAD_SUBSTITUTE * notional
    assert cost == pytest.approx(expected_spread + 0.70)
    assert cost > 0.0

    # and it must be no cheaper than a name whose spread is known and tighter
    known = execution_cost(np.array([notional]), np.array([0.0005]), np.array([price]))
    assert cost > known


def test_the_order_minimum_applies_to_small_trades():
    """500 dollars at 50 dollars a share is 10 shares = 3.5 cents, below the 0.35 floor."""
    cost = execution_cost(np.array([500.0]), np.array([0.0]), np.array([50.0]))
    assert cost == pytest.approx(COMMISSION_MIN_ORDER)


def test_commission_is_capped_at_one_percent_of_trade_value():
    """A penny stock would otherwise pay unbounded per-share commission.

    20 dollars at 0.10 a share is 200 shares = 0.70 of per-share charge, which is well
    above 1% of the 20 dollar trade, so the cap binds at 0.20.
    """
    cost = execution_cost(np.array([20.0]), np.array([0.0]), np.array([0.10]))
    assert cost == pytest.approx(COMMISSION_CAP_FRACTION * 20.0)


def test_an_unknown_price_falls_back_to_the_cap_rather_than_charging_nothing():
    """Share count is unknowable without a price, so the 1% cap is charged."""
    cost = execution_cost(np.array([10_000.0]), np.array([0.0]), np.array([np.nan]))
    assert cost == pytest.approx(COMMISSION_CAP_FRACTION * 10_000.0)


def test_costs_add_across_names():
    trade = np.array([10_000.0, 500.0])
    spreads = np.array([0.001, 0.0])
    prices = np.array([50.0, 50.0])

    both = execution_cost(trade, spreads, prices)
    first = execution_cost(trade[:1], spreads[:1], prices[:1])
    second = execution_cost(trade[1:], spreads[1:], prices[1:])

    assert both == pytest.approx(first + second)


def test_a_sale_costs_the_same_as_a_purchase_of_the_same_size():
    """Cost is charged on absolute notional; direction must not matter."""
    buy = execution_cost(np.array([10_000.0]), np.array([0.001]), np.array([50.0]))
    sell = execution_cost(np.array([-10_000.0]), np.array([0.001]), np.array([50.0]))
    assert buy == pytest.approx(sell)


# ── annualised ────────────────────────────────────────────────────────────────

def test_annualised_compounds_a_year_of_daily_returns():
    r = np.full(TRADING_DAYS, 0.001)

    cagr, vol, _sharpe = annualised(r)

    assert cagr == pytest.approx(1.001 ** TRADING_DAYS - 1.0)
    assert vol == pytest.approx(0.0, abs=1e-12)


def test_a_perfectly_constant_series_yields_a_meaningless_sharpe_not_nan():
    """Recording actual behaviour, and a small robustness note with it.

    The zero-volatility guard is ``if vol > 0.0``, an exact comparison. numpy's
    ddof=1 standard deviation of a constant array is not exactly zero -- it is
    floating-point noise around 1e-17 -- so the guard does not fire and the Sharpe
    comes back around 7e16 rather than NaN.

    Practically harmless: it needs a constant NON-ZERO return every single day, which
    no real book produces, and an all-zero series gives 0/tiny = 0 rather than a large
    number. Left as-is rather than changed to a tolerance, because this is research
    code and the fix would alter a published code path to no measured benefit. Pinned
    so the behaviour is known rather than surprising.
    """
    _cagr, vol, sharpe = annualised(np.full(TRADING_DAYS, 0.001))

    assert vol == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(sharpe)
    assert sharpe > 1e10


def test_annualised_needs_at_least_two_points():
    for r in (np.array([]), np.array([0.01])):
        assert all(np.isnan(v) for v in annualised(r))


def test_a_wiped_out_account_reports_minus_one_hundred_percent_not_a_complex_root():
    """Equity of zero has no real n-th root, so CAGR is reported as -1 explicitly."""
    r = np.array([0.01, -1.0, 0.5])

    cagr, _vol, _sharpe = annualised(r)

    assert cagr == -1.0


def test_annualised_ignores_non_finite_observations():
    clean = np.array([0.01, -0.02, 0.005, 0.004])
    dirty = np.array([0.01, np.nan, -0.02, np.inf, 0.005, 0.004])

    assert annualised(dirty) == pytest.approx(annualised(clean), nan_ok=True)


# ── max_drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_is_reported_as_a_positive_fraction_here():
    """1.10 -> 0.55 -> 0.66 against a running peak of 1.10 is a 50% drawdown.

    NOTE THE SIGN. This module returns a POSITIVE fraction, while
    ``riskparity.drawdown_report`` returns a NEGATIVE one for the same concept. Both
    conventions are in the repo; mixing them up flips a comparison silently, so the
    difference is pinned on both sides rather than left to be rediscovered.
    """
    dd = max_drawdown(np.array([0.10, -0.50, 0.20]))
    assert dd == pytest.approx(0.5)
    assert dd > 0.0


def test_a_monotonically_rising_path_has_no_drawdown():
    assert max_drawdown(np.array([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_non_finite_returns_are_treated_as_flat_days_not_dropped():
    """A missing day must not shorten the path and quietly rebase the peak."""
    with_gap = max_drawdown(np.array([0.10, np.nan, -0.50]))
    without = max_drawdown(np.array([0.10, 0.0, -0.50]))
    assert with_gap == pytest.approx(without)


def test_commission_constants_are_the_documented_ibkr_schedule():
    """A guard on the numbers the cost tests above are derived from."""
    assert COMMISSION_PER_SHARE == 0.0035
    assert COMMISSION_MIN_ORDER == 0.35
    assert COMMISSION_CAP_FRACTION == 0.01
    assert LIQUID_SPREAD_SUBSTITUTE == 0.0020
