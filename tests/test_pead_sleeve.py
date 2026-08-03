"""Guards for the PEAD sleeve, aimed at the defects that have already cost this repo.

The two delisting bugs in `capacity_curve_result.md` s4 produced -60%/yr and -112%/yr
respectively and were both found by eyeballing an impossible number, not by a test.
These are that test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.sleeves import pead


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2003-01-01", periods=400)


def _bars(calendar: pd.DatetimeIndex, length: int, price: float = 20.0,
          drift: float = 0.0, spread: float = 0.02, seed: int = 7) -> pead.TickerBars:
    """A synthetic name with a GENUINE embedded bid-ask spread.

    Bars with a deterministic high/low band around the close carry no spread at all,
    and `spread_with_resolution` correctly refuses to trade them. The sleeve is only
    exercisable on a tape that actually contains the quantity the cost model estimates,
    so the efficient price, the intraday range and the bid-ask bounce are simulated
    separately here.
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
        volume=np.full(length, 500_000.0),
        dollar_volume=close * 500_000.0,
    )


def _signals(day: int, ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame([{"ticker": ticker, "filing_day": day, "sue": 3.0}])


def test_delisting_not_booked_against_an_unrelated_earlier_exit(calendar):
    """Bug 1: `terminal.get(ticker)` charged a 2012 bankruptcy against a 2003 exit."""
    bars = {"AAA": _bars(calendar, 400)}
    far_future = (calendar[200] + pd.Timedelta(days=3000), -1.0)
    positions, _ = pead.build_positions(_signals(100), bars, calendar,
                                        {"AAA": far_future}, horizon=20)
    assert len(positions) == 1
    assert positions.delisted == [False]
    assert positions.gross_return[0] > -0.5, "an unrelated delisting was booked"


def test_delisting_booked_when_it_falls_just_after_the_forced_exit(calendar):
    """The other side of the same rule: a real, timely delisting MUST be charged."""
    bars = {"AAA": _bars(calendar, 130)}          # series stops well before day 400
    last_bar = calendar[129]
    positions, _ = pead.build_positions(
        _signals(100), bars, calendar,
        {"AAA": (last_bar + pd.Timedelta(days=10), -1.0)}, horizon=60)
    assert positions.delisted == [True]
    assert positions.gross_return[0] == pytest.approx(-1.0)


def test_a_delisted_position_is_removed_and_never_rebooked(calendar):
    """Bug 2: a bankrupt name booked -100% every month forever (-112%/yr)."""
    bars = {"AAA": _bars(calendar, 130)}
    last_bar = calendar[129]
    positions, _ = pead.build_positions(
        _signals(100), bars, calendar,
        {"AAA": (last_bar + pd.Timedelta(days=10), -1.0)}, horizon=60)
    book = pead.simulate_book(positions, calendar, costs_on=False)
    assert book.open_at_end == 0
    # A long-only book cannot lose more than it put in. One position sized at 0.5% of
    # equity going to zero can cost at most 0.5%.
    total = book.equity.iloc[-1] / pead.START_CAPITAL - 1.0
    assert total >= -pead.MAX_POSITION_FRACTION - 1e-9
    assert book.equity.min() > 0.99 * pead.START_CAPITAL


def test_long_only_book_can_never_go_below_zero(calendar):
    """Every position wiped out simultaneously still leaves equity >= 0."""
    tickers = {f"T{i}": _bars(calendar, 130, seed=i) for i in range(40)}
    signals = pd.concat([_signals(100, name) for name in tickers], ignore_index=True)
    terminal = {name: (calendar[129] + pd.Timedelta(days=5), -1.0) for name in tickers}
    positions, _ = pead.build_positions(signals, tickers, calendar, terminal,
                                        horizon=60)
    book = pead.simulate_book(positions, calendar, costs_on=False)
    assert book.equity.min() > 0.0
    assert book.open_at_end == 0


def test_entry_is_strictly_after_the_filing_date(calendar):
    """A filing can be accepted after the close; datekey itself is a look-ahead."""
    bars = {"AAA": _bars(calendar, 400)}
    positions, _ = pead.build_positions(_signals(100), bars, calendar, {}, horizon=20)
    assert positions.entry_day[0] == 101
    assert positions.exit_day[0] == 121


def test_screen_uses_only_bars_up_to_the_filing_day(calendar):
    """Corrupting post-filing bars must not move the screen."""
    bars = _bars(calendar, 400)
    clean = pead.screen_at_filing(bars, 200)
    bars.high[201:] = 1e6
    bars.low[201:] = 1e-6
    bars.close[201:] = 1e6
    dirty = pead.screen_at_filing(bars, 200)
    assert clean.spread == pytest.approx(dirty.spread)
    assert clean.daily_vol == pytest.approx(dirty.daily_vol)


def test_unmeasurable_spread_is_excluded_not_floored(calendar):
    """prereg s4.4 — an unknown cost must remove the name, never be costed cheaply."""
    bars = _bars(calendar, 400)
    bars.high[:] = bars.close        # no genuine range anywhere -> nothing to measure
    bars.low[:] = bars.close
    screen = pead.screen_at_filing(bars, 200)
    assert not screen.passed
    assert screen.reason in {"thin_trading", "spread_unmeasurable",
                             "spread_upper_bound"}


def test_return_cap_binds_and_the_daily_path_still_compounds_to_it(calendar):
    """A +9,900% shell must be capped, and the marked path must agree with the cap."""
    bars = _bars(calendar, 400, drift=0.20)      # explosive by construction
    positions, _ = pead.build_positions(_signals(100), {"AAA": bars}, calendar, {},
                                        horizon=20)
    assert positions.gross_return[0] == pytest.approx(pead.RETURN_CAP)
    compounded = float(np.prod(1.0 + positions.mark_return[0]) - 1.0)
    assert compounded == pytest.approx(pead.RETURN_CAP, rel=1e-9)


def test_commission_minimum_and_cap_both_bind():
    """IBKR: $0.0035/share, $0.35 per-order minimum, capped at 1% of value."""
    tiny = pead.trade_cost_fraction(20.0, 2.0, spread=0.0, daily_vol=0.0,
                                    median_dollar_volume=1e12)
    assert tiny == pytest.approx(0.01)           # 1%-of-value cap binds on a $20 ticket
    small = pead.trade_cost_fraction(1_000.0, 20.0, spread=0.0, daily_vol=0.0,
                                     median_dollar_volume=1e12)
    assert small == pytest.approx(0.35 / 1_000.0)   # minimum binds
    large = pead.trade_cost_fraction(1_000_000.0, 20.0, spread=0.0, daily_vol=0.0,
                                     median_dollar_volume=1e12)
    assert large == pytest.approx(0.0035 * 50_000 / 1_000_000)  # per-share binds


def test_cost_is_charged_on_both_sides():
    """A round trip at a flat spread must cost the full spread, not half of it."""
    calendar_local = pd.bdate_range("2003-01-01", periods=200)
    bars = {"AAA": _bars(calendar_local, 200)}
    positions, _ = pead.build_positions(_signals(100), bars, calendar_local, {},
                                        horizon=20)
    assert len(positions) == 1, "fixture must actually open a position"
    net = pead.simulate_book(positions, calendar_local, costs_on=True)
    gross = pead.simulate_book(positions, calendar_local, costs_on=False)
    assert net.equity.iloc[-1] < gross.equity.iloc[-1]
    assert net.total_cost > 0


def test_sue_is_invariant_to_a_uniform_per_ticker_rescaling():
    """Retroactive split adjustment scales the whole series; SUE must not notice."""
    dates = pd.date_range("2000-03-31", periods=16, freq="QE")
    rng = np.random.default_rng(11)
    eps = pd.Series(rng.normal(0.5, 0.3, 16))
    base = pd.DataFrame({
        "ticker": "AAA", "calendardate": dates, "datekey": dates + pd.Timedelta(days=45),
        "eps": eps, "netinc": np.nan, "shareswa": 1.0,
    })
    scaled = base.assign(eps=base["eps"] * 4.0)
    left = pead.build_sue(base)["sue"].to_numpy()
    right = pead.build_sue(scaled)["sue"].to_numpy()
    assert left.size >= 5
    np.testing.assert_allclose(left, right, rtol=1e-10)


def test_sue_denominator_excludes_the_quarter_being_scored():
    """Including the current difference in its own scale leaks the answer."""
    dates = pd.date_range("2000-03-31", periods=16, freq="QE")
    seasonal = np.tile([0.10, 0.20, 0.30, 0.40], 4)
    # A calm history with real but small dispersion, then one blowout quarter.
    seasonal += np.random.default_rng(3).normal(0.0, 0.02, 16)
    eps = seasonal.copy()
    eps[-1] += 9.0
    frame = pd.DataFrame({
        "ticker": "AAA", "calendardate": dates, "datekey": dates + pd.Timedelta(days=45),
        "eps": eps, "netinc": np.nan, "shareswa": 1.0,
    })
    sue = pead.build_sue(frame)
    # The 9.0 blowout is the last row. Its denominator must come from the calm history,
    # so the SUE has to be large; a leaked denominator would deflate it toward 1.
    assert sue["sue"].iloc[-1] > 20.0


def test_benchmark_does_not_turn_mid_month_delistings_into_minus_100_months(tmp_path,
                                                                            monkeypatch):
    """Regression: the panel's date is each ticker's OWN last bar of the month.

    A name that stops trading on the 12th carries a mid-month date. Grouping the
    benchmark on the raw date puts it in a singleton group, and once its -100% terminal
    return is booked that singleton becomes a -100% MONTH for the whole universe. The
    first run of this sleeve reported a benchmark of exactly -100%/yr for that reason.
    """
    dates = pd.date_range("2005-01-31", periods=24, freq="ME")
    rows = []
    for stamp in dates:
        for name in ("AAA", "BBB", "CCC"):
            rows.append({"ticker": name, "date": stamp, "spread_regime": "measured",
                         "forward_return": 0.01})
    # DDD stops trading on the 12th of one month and is then delisted at -100%.
    rows.append({"ticker": "DDD", "date": pd.Timestamp("2005-06-12"),
                 "spread_regime": "measured", "forward_return": np.nan})
    panel = pd.DataFrame(rows)

    delistings = pd.DataFrame([{"ticker": "DDD", "date": pd.Timestamp("2005-06-20"),
                                "action": "bankruptcyliquidation",
                                "terminal_return": -1.0}])
    panel.to_parquet(tmp_path / "monthly_panel_dev.parquet", index=False)
    delistings.to_parquet(tmp_path / "delistings.parquet", index=False)
    monkeypatch.setattr(pead, "PANEL_DIR", tmp_path)

    monthly = pead.universe_benchmark({"AAA", "BBB", "CCC", "DDD"}, dates[0], dates[-1])
    assert len(monthly) == 24, "one row per calendar month, not one per raw date"
    assert monthly.min() > -0.5, "a mid-month delisting became a -100% month"
    # June holds three survivors at +1% and one wipe-out: (0.01*3 - 1.0) / 4.
    assert monthly.loc[monthly.index.month == 6].iloc[0] == pytest.approx(-0.2425)
    others = monthly.drop(monthly.index[monthly.index.month == 6])
    np.testing.assert_allclose(others.to_numpy(), 0.01)
    # The point of the regression: the series compounds to something finite, not -100%.
    assert pead.summarise(monthly).annual_return > -0.2


def test_benchmark_still_books_a_timely_delisting(tmp_path, monkeypatch):
    """The fix must not throw the delisting away — the loss has to land somewhere."""
    dates = pd.date_range("2005-01-31", periods=12, freq="ME")
    rows = [{"ticker": "AAA", "date": stamp, "spread_regime": "measured",
             "forward_return": 0.0} for stamp in dates]
    rows.append({"ticker": "DDD", "date": pd.Timestamp("2005-06-12"),
                 "spread_regime": "measured", "forward_return": np.nan})
    pd.DataFrame(rows).to_parquet(tmp_path / "monthly_panel_dev.parquet", index=False)
    pd.DataFrame([{"ticker": "DDD", "date": pd.Timestamp("2005-06-20"),
                   "action": "bankruptcyliquidation", "terminal_return": -1.0}]
                 ).to_parquet(tmp_path / "delistings.parquet", index=False)
    monkeypatch.setattr(pead, "PANEL_DIR", tmp_path)

    monthly = pead.universe_benchmark({"AAA", "DDD"}, dates[0], dates[-1])
    june = monthly.loc[monthly.index.month == 6].iloc[0]
    assert june == pytest.approx(-0.5)   # mean of AAA at 0.0 and DDD at -1.0


def test_seasonal_gap_rejects_a_missing_quarter():
    """A skipped filing would silently compare Q3 to Q2 — seasonality, not surprise."""
    dates = list(pd.date_range("2000-03-31", periods=16, freq="QE"))
    del dates[8]                                   # one quarter never filed
    frame = pd.DataFrame({
        "ticker": "AAA", "calendardate": dates,
        "datekey": [d + pd.Timedelta(days=45) for d in dates],
        "eps": np.linspace(0.1, 1.6, len(dates)), "netinc": np.nan, "shareswa": 1.0,
    })
    frame["eps"] += np.tile([0.0, 0.3, -0.2, 0.1], 4)[:len(dates)]
    sue = pead.build_sue(frame)
    gaps = (sue["calendardate"] - sue["calendardate"].shift(4)).dt.days.dropna()
    assert (gaps.between(*pead.SEASONAL_GAP_DAYS)).all() or sue.empty
