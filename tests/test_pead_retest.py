"""Guards for the PEAD RE-TEST on the corrected universe.

The defect these exist to stop recurring is the ITERATION-1 UNIVERSE BIAS: excluding every
name whose EDGE spread regime is ``upper_bound`` deleted 525,933 of 922,652 eligible
(name, month) cells -- the cheap, liquid half of the tape -- and forced six strategies into
the expensive tail. The first two tests below are that regression, in both directions:
cheap names must be ADMITTED, absent names must still be REFUSED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.sleeves import pead, pead_retest


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2005-01-03", periods=400)


def _bars(calendar: pd.DatetimeIndex, length: int, price: float = 20.0,
          drift: float = 0.0, spread: float = 0.02, seed: int = 7,
          volume: float = 500_000.0) -> pead.TickerBars:
    """A synthetic name with a GENUINE embedded bid-ask spread.

    The efficient price, the intraday range and the bid-ask bounce are simulated
    separately, because bars with a deterministic band around the close contain no spread
    at all and the estimator correctly refuses to quote one.
    """
    rng = np.random.default_rng(seed)
    efficient = price * np.exp(np.cumsum(rng.normal(drift, 0.02, length)))
    half = spread / 2.0
    intraday = np.abs(rng.normal(0.0, 0.015, length))
    true_high = efficient * np.exp(intraday)
    true_low = efficient * np.exp(-intraday)
    side_close = rng.choice([-1.0, 1.0], length)
    side_open = rng.choice([-1.0, 1.0], length)
    close = efficient * (1.0 + side_close * half)
    open_ = efficient * (1.0 + side_open * half)
    return pead.TickerBars(
        day_index=np.arange(length, dtype=np.int32),
        open_=open_,
        high=true_high * (1.0 + half),
        low=true_low * (1.0 - half),
        close=close,
        closeadj=close,
        volume=np.full(length, volume),
        dollar_volume=close * volume,
    )


def _signals(day: int, ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame([{"ticker": ticker, "filing_day": day, "sue": 3.0}])


def _screen_map(signals: pd.DataFrame, bars: dict[str, pead.TickerBars],
                calendar: pd.DatetimeIndex):
    screens, _ = pead_retest.screen_all(signals, bars, calendar)
    return screens


# ---------------------------------------------------------------------------
# THE REGRESSION: the iteration-1 universe bias, in both directions
# ---------------------------------------------------------------------------

def test_a_cheap_upper_bound_name_is_now_admitted(calendar):
    """The whole point of the re-test. A tight-spread liquid name must TRADE.

    Iteration 1 rejected this exact case with reason ``spread_upper_bound`` and thereby
    deleted the cheap half of the market.
    """
    bars = _bars(calendar, 400, spread=0.0005, volume=2_000_000.0)
    screen = pead_retest.screen_at_filing(bars, 200, calendar)
    assert screen.regime == "upper_bound", "fixture must exercise the disputed regime"
    assert screen.passed, "the iteration-1 universe bias has come back"


def test_an_unmeasurable_name_is_still_refused(calendar):
    """`upper_bound` is cheap; `unmeasurable` is ABSENT. The schedule must not price it."""
    bars = _bars(calendar, 400)
    bars.high[:] = bars.close        # no genuine range anywhere -> nothing to measure
    bars.low[:] = bars.close
    screen = pead_retest.screen_at_filing(bars, 200, calendar)
    assert not screen.passed
    assert screen.reason in {"thin_trading", "spread_unmeasurable"}


def test_the_realistic_bound_is_never_dearer_than_the_conservative_one(calendar):
    """`realistic <= conservative` for every admitted name, by construction."""
    for seed, spread, volume in ((1, 0.0005, 5e6), (2, 0.02, 5e5), (3, 0.004, 2e6)):
        bars = _bars(calendar, 400, spread=spread, seed=seed, volume=volume)
        screen = pead_retest.screen_at_filing(bars, 200, calendar)
        if not screen.passed:
            continue
        assert screen.spread_realistic <= screen.spread_conservative + 1e-12


def test_a_measured_name_prices_identically_under_both_bounds(calendar):
    """Strictly additive: the schedule has no business overriding a real measurement."""
    bars = _bars(calendar, 400, spread=0.02, volume=5e5)
    screen = pead_retest.screen_at_filing(bars, 200, calendar)
    assert screen.passed and screen.regime == "measured"
    assert screen.spread_realistic == pytest.approx(screen.spread_conservative)


def test_the_cheaper_bound_can_only_help_the_book(calendar):
    """(b) charges no more spread than (a), so its equity cannot end lower."""
    bars = {"AAA": _bars(calendar, 400, spread=0.0005, volume=2_000_000.0)}
    signals = _signals(100)
    screens = _screen_map(signals, bars, calendar)
    positions, _ = pead_retest.build_positions(signals, bars, calendar, {}, 40, screens)
    assert len(positions) == 1
    cons = pead_retest.simulate_book(
        positions, calendar, np.asarray(positions.spread_conservative))
    real = pead_retest.simulate_book(
        positions, calendar, np.asarray(positions.spread_realistic))
    gross = pead_retest.simulate_book(positions, calendar, None)
    assert real.total_cost <= cons.total_cost
    assert real.equity.iloc[-1] >= cons.equity.iloc[-1]
    assert gross.equity.iloc[-1] >= real.equity.iloc[-1]
    assert gross.total_cost == 0.0


# ---------------------------------------------------------------------------
# The point-in-time and delisting defects, re-guarded on the new code path
# ---------------------------------------------------------------------------

def test_entry_is_strictly_after_the_filing_date(calendar):
    """A filing can be accepted after the close; datekey itself is a look-ahead."""
    bars = {"AAA": _bars(calendar, 400)}
    signals = _signals(100)
    positions, _ = pead_retest.build_positions(
        signals, bars, calendar, {}, 20, _screen_map(signals, bars, calendar))
    assert positions.entry_day[0] == 101
    assert positions.exit_day[0] == 121


def test_screen_uses_only_bars_up_to_the_filing_day(calendar):
    """Corrupting post-filing bars must not move the screen."""
    bars = _bars(calendar, 400)
    clean = pead_retest.screen_at_filing(bars, 200, calendar)
    bars.high[201:] = 1e6
    bars.low[201:] = 1e-6
    bars.close[201:] = 1e6
    dirty = pead_retest.screen_at_filing(bars, 200, calendar)
    assert clean.spread_conservative == pytest.approx(dirty.spread_conservative)
    assert clean.spread_realistic == pytest.approx(dirty.spread_realistic)
    assert clean.daily_vol == pytest.approx(dirty.daily_vol)


def test_an_unrelated_later_delisting_is_not_booked(calendar):
    """The 2012-bankruptcy-charged-to-a-2005-exit defect."""
    bars = {"AAA": _bars(calendar, 400)}
    signals = _signals(100)
    far_future = (calendar[200] + pd.Timedelta(days=3000), -1.0)
    positions, _ = pead_retest.build_positions(
        signals, bars, calendar, {"AAA": far_future}, 20,
        _screen_map(signals, bars, calendar))
    assert positions.delisted == [False]
    assert positions.gross_return[0] > -0.5


def test_a_timely_delisting_is_booked_once_and_the_name_removed(calendar):
    """The -112%/yr defect: booked once, then out of the book forever."""
    bars = {"AAA": _bars(calendar, 130)}
    signals = _signals(100)
    terminal = {"AAA": (calendar[129] + pd.Timedelta(days=10), -1.0)}
    positions, _ = pead_retest.build_positions(
        signals, bars, calendar, terminal, 60, _screen_map(signals, bars, calendar))
    assert positions.delisted == [True]
    assert positions.gross_return[0] == pytest.approx(-1.0)
    book = pead_retest.simulate_book(positions, calendar, None)
    assert book.open_at_end == 0
    total = book.equity.iloc[-1] / pead_retest.START_CAPITAL - 1.0
    assert total >= -pead_retest.MAX_POSITION_FRACTION - 1e-9


def test_return_cap_binds_and_the_daily_path_compounds_to_it(calendar):
    bars = {"AAA": _bars(calendar, 400, drift=0.20)}
    signals = _signals(100)
    positions, _ = pead_retest.build_positions(
        signals, bars, calendar, {}, 20, _screen_map(signals, bars, calendar))
    assert positions.gross_return[0] == pytest.approx(pead_retest.RETURN_CAP)
    compounded = float(np.prod(1.0 + positions.mark_return[0]) - 1.0)
    assert compounded == pytest.approx(pead_retest.RETURN_CAP, rel=1e-9)


# ---------------------------------------------------------------------------
# Costs, the $0.35 minimum, and the reported diagnostics
# ---------------------------------------------------------------------------

def test_the_035_order_minimum_is_detected_and_priced():
    """The term that makes a small account a different strategy (prereg s5)."""
    cost, at_minimum = pead_retest.trade_cost_fraction(
        5_000.0, 100.0, spread=0.0, daily_vol=0.0, median_dollar_volume=1e12)
    assert at_minimum, "50 shares at $0.0035 is $0.175, so the $0.35 floor must bind"
    assert cost == pytest.approx(0.35 / 5_000.0)          # 0.70bps

    cost, at_minimum = pead_retest.trade_cost_fraction(
        1_000_000.0, 20.0, spread=0.0, daily_vol=0.0, median_dollar_volume=1e12)
    assert not at_minimum
    assert cost == pytest.approx(0.0035 * 50_000 / 1_000_000)

    cost, at_minimum = pead_retest.trade_cost_fraction(
        20.0, 2.0, spread=0.0, daily_vol=0.0, median_dollar_volume=1e12)
    assert cost == pytest.approx(0.01)                    # 1%-of-value cap binds
    assert not at_minimum, "the cap, not the minimum, is what binds on a $20 ticket"


def test_pnl_concentration_shares_sum_to_one_and_the_alarm_fires(calendar):
    """A dominant single name-month must be flagged, not averaged away."""
    tickers = {f"T{i}": _bars(calendar, 400, seed=100 + i) for i in range(6)}
    signals = pd.concat([_signals(100, name) for name in tickers], ignore_index=True)
    screens = _screen_map(signals, tickers, calendar)
    positions, _ = pead_retest.build_positions(signals, tickers, calendar, {}, 40,
                                               screens)
    assert len(positions) >= 3, "fixture must open several positions"
    book = pead_retest.simulate_book(positions, calendar, None)
    report = pead_retest.pnl_concentration(positions, book, calendar)

    shares = np.array([item["share_of_total"] for item in report["top"]])
    assert shares.sum() == pytest.approx(1.0, abs=1e-9)   # 6 positions, top-10 covers all
    # A handful of equally-weighted positions is concentrated by construction, which is
    # exactly the condition the alarm exists to announce.
    assert report["exceeds_alarm"]
    assert report["total_net_pnl"] == pytest.approx(
        book.equity.iloc[-1] - pead_retest.START_CAPITAL, rel=1e-9)


def test_sharpe_by_decade_flags_a_thin_decade():
    index = pd.date_range("2009-01-31", periods=30, freq="ME")
    returns = pd.Series(np.linspace(0.01, 0.02, 30), index=index)
    decades = pead_retest.sharpe_by_decade(returns)
    assert decades["2000s"]["months"] == 12 and decades["2000s"]["thin"] is True
    assert decades["2010s"]["months"] == 18 and decades["2010s"]["thin"] is True


def test_registered_verdict_is_the_table_in_the_prereg():
    assert pead_retest.registered_verdict(0.05, 0.80, 0.06, 0.85)[0] == "PROMISING"
    assert pead_retest.registered_verdict(-0.01, 0.30, 0.02, 0.90)[0] == "UNDETERMINED"
    assert pead_retest.registered_verdict(-0.01, 0.30, 0.005, 0.35)[0] == "MARGINAL"
    assert pead_retest.registered_verdict(-0.01, 0.30, -0.005, 0.35)[0] == "DEAD"
    # Positive excess but below the 0.75 promotion gate is MARGINAL, never a pass.
    assert pead_retest.registered_verdict(0.01, 0.74, 0.01, 0.74)[0] == "MARGINAL"
    # A Sharpe inversion is reported, not hidden and not crashed on.
    verdict, inverted = pead_retest.registered_verdict(0.05, 0.80, 0.05, 0.70)
    assert verdict == "PROMISING" and inverted is True


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_benchmark_includes_upper_bound_cells(tmp_path, monkeypatch):
    """The benchmark must move with the universe, or the excess is measured against
    a different set of names than the strategy trades."""
    dates = pd.date_range("2005-01-31", periods=24, freq="ME")
    rows = []
    for stamp in dates:
        rows.append({"ticker": "AAA", "date": stamp, "spread_regime": "measured",
                     "forward_return": 0.00})
        rows.append({"ticker": "BBB", "date": stamp, "spread_regime": "upper_bound",
                     "forward_return": 0.02})
        rows.append({"ticker": "CCC", "date": stamp, "spread_regime": "ineligible",
                     "forward_return": 0.50})
    pd.DataFrame(rows).to_parquet(tmp_path / "monthly_panel_dev.parquet", index=False)
    pd.DataFrame([{"ticker": "ZZZ", "date": pd.Timestamp("2005-06-20"),
                   "action": "bankruptcyliquidation", "terminal_return": -1.0}]
                 ).to_parquet(tmp_path / "delistings.parquet", index=False)
    monkeypatch.setattr(pead_retest, "PANEL_DIR", tmp_path)

    monthly = pead_retest.universe_benchmark({"AAA", "BBB", "CCC"}, dates[0], dates[-1])
    assert len(monthly) == 24
    # AAA at 0% and BBB at 2%; CCC is ineligible and must not appear at any weight.
    np.testing.assert_allclose(monthly.to_numpy(), 0.01)


def test_benchmark_does_not_turn_mid_month_delistings_into_minus_100_months(
        tmp_path, monkeypatch):
    """The panel's date is each ticker's OWN last bar of the month."""
    dates = pd.date_range("2005-01-31", periods=24, freq="ME")
    rows = [{"ticker": name, "date": stamp, "spread_regime": "upper_bound",
             "forward_return": 0.01}
            for stamp in dates for name in ("AAA", "BBB", "CCC")]
    rows.append({"ticker": "DDD", "date": pd.Timestamp("2005-06-12"),
                 "spread_regime": "upper_bound", "forward_return": np.nan})
    pd.DataFrame(rows).to_parquet(tmp_path / "monthly_panel_dev.parquet", index=False)
    pd.DataFrame([{"ticker": "DDD", "date": pd.Timestamp("2005-06-20"),
                   "action": "bankruptcyliquidation", "terminal_return": -1.0}]
                 ).to_parquet(tmp_path / "delistings.parquet", index=False)
    monkeypatch.setattr(pead_retest, "PANEL_DIR", tmp_path)

    monthly = pead_retest.universe_benchmark({"AAA", "BBB", "CCC", "DDD"},
                                             dates[0], dates[-1])
    assert len(monthly) == 24
    assert monthly.loc[monthly.index.month == 6].iloc[0] == pytest.approx(-0.2425)
    assert monthly.min() > -0.5
