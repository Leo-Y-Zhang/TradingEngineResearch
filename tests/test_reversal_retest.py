"""Tests for the short-horizon reversal re-test sleeve.

The numerical machinery this file guards is the vectorised two-bound spread pricing --
the one new calculation in the sleeve that could be silently wrong and would move every
reported number if it were. It is checked against the reference scalar implementation in
`research.spread_estimation`, which is itself covered by the spread-estimation suite.

The rest guards the accounting rules the programme has already paid for: the rebalance
grids execute at the NEXT bar, the delisting terminal return is booked only inside the
window, the return cap binds, and the liquidity-rank universe actually restricts.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from research.sleeves.reversal_retest import (
    UNIVERSE_CUTS,
    PanelMatrices,
    RetestConfig,
    holding_returns,
    rebalance_grid,
    spread_bounds_frame,
    verify_bounds_vectorisation,
)
from research.spread_estimation import bounds_from_estimate


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_vectorised_bounds_match_the_reference_scalar_implementation() -> None:
    """Every branch of `bounds_from_estimate`: measured, upper_bound, unmeasurable, no volume."""
    rows = [
        # A genuine measurement: both bounds are the measurement, the schedule never speaks.
        {"ticker": "M1", "date": "2010-06-30", "close": 40.0, "spread": 0.0090,
         "spread_regime": "measured", "median_dollar_volume": 1.2e8},
        # A liquid unresolved name in the decimal era: the schedule takes it to ~9bps.
        {"ticker": "U1", "date": "2010-06-30", "close": 40.0, "spread": 0.0055,
         "spread_regime": "upper_bound", "median_dollar_volume": 1.2e8},
        # The same name before decimalisation: the era factor and the tick floor bite.
        {"ticker": "U2", "date": "1999-06-30", "close": 20.0, "spread": 0.0300,
         "spread_regime": "upper_bound", "median_dollar_volume": 1.2e8},
        # An unresolved name so illiquid the schedule quotes MORE than the estimate; the
        # min() must keep the estimate, because `upper_bound` bounds the truth from above.
        {"ticker": "U3", "date": "2010-06-30", "close": 5.0, "spread": 0.0040,
         "spread_regime": "upper_bound", "median_dollar_volume": 2.0e5},
        # Untradeable: no honest cost exists, and the schedule must not invent one.
        {"ticker": "X1", "date": "2010-06-30", "close": 12.0, "spread": float("nan"),
         "spread_regime": "unmeasurable", "median_dollar_volume": 3.0e6},
        # No dollar volume: the schedule cannot speak, so it falls back to conservative.
        {"ticker": "U4", "date": "2010-06-30", "close": 12.0, "spread": 0.0070,
         "spread_regime": "upper_bound", "median_dollar_volume": 0.0},
    ]
    out = spread_bounds_frame(_frame(rows))

    for row in out.itertuples(index=False):
        ref = bounds_from_estimate(row.spread, row.spread_regime,
                                   row.median_dollar_volume, price=row.close,
                                   when=row.date)
        for name, got in (("conservative", row.spread_conservative),
                          ("realistic", row.spread_realistic)):
            want = getattr(ref, name)
            if np.isnan(want):
                assert np.isnan(got), f"{row.ticker} {name}"
            else:
                assert got == pytest.approx(want, rel=1e-12, abs=1e-15), f"{row.ticker} {name}"


def test_bounds_never_invert_and_unmeasurable_stays_untradeable() -> None:
    rows = [
        {"ticker": "A", "date": "2004-03-31", "close": 30.0, "spread": 0.002,
         "spread_regime": "upper_bound", "median_dollar_volume": 5.0e8},
        {"ticker": "B", "date": "1998-03-31", "close": 3.0, "spread": 0.05,
         "spread_regime": "upper_bound", "median_dollar_volume": 4.0e5},
        {"ticker": "C", "date": "2013-03-31", "close": 3.0, "spread": float("nan"),
         "spread_regime": "unmeasurable", "median_dollar_volume": 4.0e5},
    ]
    out = spread_bounds_frame(_frame(rows))
    finite = out.dropna(subset=["spread_conservative", "spread_realistic"])
    assert (finite["spread_realistic"] <= finite["spread_conservative"] + 1e-15).all()
    dead = out[out["spread_regime"] == "unmeasurable"]
    assert dead["spread_conservative"].isna().all()
    assert dead["spread_realistic"].isna().all()


def test_the_pre_decimalisation_tick_floor_binds_and_points_the_expensive_way() -> None:
    """A $20 stock in 1999 cannot be quoted inside one sixteenth = 31.25bps."""
    rows = [{"ticker": "T", "date": "1999-09-30", "close": 20.0, "spread": 0.0400,
             "spread_regime": "upper_bound", "median_dollar_volume": 9.0e8}]
    out = spread_bounds_frame(_frame(rows))
    assert out["spread_realistic"].iloc[0] == pytest.approx(0.0625 / 20.0, rel=1e-12)

    rows_modern = [dict(rows[0], date="2005-09-30")]
    modern = spread_bounds_frame(_frame(rows_modern))
    assert modern["spread_realistic"].iloc[0] < out["spread_realistic"].iloc[0]


def test_verify_bounds_vectorisation_raises_on_a_corrupted_bound() -> None:
    rows = [{"ticker": "Z", "date": "2010-06-30", "close": 40.0, "spread": 0.0055,
             "spread_regime": "upper_bound", "median_dollar_volume": 1.2e8}]
    out = spread_bounds_frame(_frame(rows))
    assert verify_bounds_vectorisation(out, n=1) == 1
    out.loc[0, "spread_realistic"] = 0.0001
    with pytest.raises(AssertionError):
        verify_bounds_vectorisation(out, n=1)


def test_rebalance_grids_execute_on_the_next_bar_and_never_the_signal_bar() -> None:
    dates = pd.bdate_range("2005-01-03", "2005-06-30")
    for kind in ("weekly", "fortnightly", "monthly"):
        signal_idx, exec_idx = rebalance_grid(dates, kind)
        assert signal_idx.size > 0
        assert np.all(exec_idx == signal_idx + 1)
        assert np.all(exec_idx < len(dates))
    weekly, _ = rebalance_grid(dates, "weekly")
    fortnightly, _ = rebalance_grid(dates, "fortnightly")
    monthly, _ = rebalance_grid(dates, "monthly")
    assert set(fortnightly).issubset(set(weekly))
    assert len(monthly) < len(fortnightly) < len(weekly)


def test_unregistered_rebalance_grid_is_refused() -> None:
    dates = pd.bdate_range("2005-01-03", "2005-03-31")
    with pytest.raises(ValueError):
        rebalance_grid(dates, "daily")


def _panel(delist_when: str | None, terminal: float = -0.9) -> PanelMatrices:
    dates = pd.DatetimeIndex(pd.bdate_range("2005-01-03", periods=12))
    n = len(dates)
    adj_open = np.full((n, 1), 10.0)
    adj_close = np.full((n, 1), 10.0)
    # The name stops printing after row 4.
    adj_open[5:, 0] = np.nan
    adj_close[6:, 0] = np.nan
    stamp = (np.iinfo(np.int64).min if delist_when is None
             else int(pd.Timestamp(delist_when).to_datetime64()
                      .astype("datetime64[ns]").astype(np.int64)))
    return PanelMatrices(
        dates=dates, tickers=np.array(["A"], dtype=object),
        adj_open=adj_open, adj_close=adj_close,
        raw_open=adj_open.astype(np.float32),
        months=pd.PeriodIndex([], freq="M"),
        dollar_volume=np.zeros((0, 1)), dv_rank=np.zeros((0, 1)),
        spread_basis={}, dv_basis=np.zeros((0, 1)),
        delist_date=np.array([stamp], dtype=np.int64),
        delist_return=np.array([terminal]),
    )


def test_delisting_terminal_return_is_booked_only_inside_the_window() -> None:
    config = RetestConfig()
    inside = _panel(str(_panel(None).dates[6].date()))
    got = holding_returns(inside, 1, 7, config)[0]
    assert got == pytest.approx(-0.9)


def test_a_delisting_far_outside_the_window_is_not_charged() -> None:
    """The defect that charged a 2012 bankruptcy against a 2003 exit."""
    config = RetestConfig()
    far = _panel("2009-01-05")
    assert holding_returns(far, 1, 7, config)[0] == pytest.approx(0.0)


def test_holding_returns_are_capped_at_plus_minus_one_hundred_percent() -> None:
    """The +9,900% bankrupt-shell artefact, and the -100% floor a total loss must respect."""
    config = RetestConfig()
    panel = _panel(None)
    panel.adj_open[7, 0] = 1000.0
    assert holding_returns(panel, 1, 7, config)[0] == pytest.approx(config.return_cap)

    # A price ratio alone can never go below -100%; the floor is reachable only through a
    # terminal delisting return, which is exactly the path that once produced -112%/yr.
    wiped = _panel(str(_panel(None).dates[6].date()), terminal=-1.0)
    assert holding_returns(wiped, 1, 7, config)[0] == pytest.approx(-config.return_cap)


def test_a_seconds_resolution_calendar_still_books_the_delisting() -> None:
    """Regression: `Timestamp.to_datetime64()` carries the timestamp's own resolution.

    A seconds-resolution date index would silently make every delisting-window comparison
    false and hand the book a survivorship flatter that raises nothing.
    """
    config = RetestConfig()
    panel = _panel(str(_panel(None).dates[6].date()))
    panel.dates = pd.DatetimeIndex(panel.dates.astype("datetime64[s]"))
    assert holding_returns(panel, 1, 7, config)[0] == pytest.approx(-0.9)


def test_registered_universe_cuts_are_the_liquid_tail_only() -> None:
    assert UNIVERSE_CUTS == {"top_decile": 0.90, "top_quintile": 0.80}
    assert all(0.5 < v < 1.0 for v in UNIVERSE_CUTS.values())


def test_registered_config_keeps_upper_bound_names_and_excludes_only_unmeasurable() -> None:
    """The whole point of the re-test: `upper_bound` means CHEAP, not unknown."""
    config = RetestConfig()
    assert "upper_bound" in config.allowed_spread_regimes
    assert "measured" in config.allowed_spread_regimes
    assert "unmeasurable" not in config.allowed_spread_regimes
    assert config.min_price == 2.00
    assert config.return_cap == 1.00
    assert config.delisting_grace_days == 62


def test_tick_regime_boundaries_are_honoured_by_the_vectorised_path() -> None:
    dates = ["1997-06-23", "1997-06-24", "2001-04-08", "2001-04-09"]
    price = 10.0
    rows = [{"ticker": f"T{i}", "date": d, "close": price, "spread": 0.5,
             "spread_regime": "upper_bound", "median_dollar_volume": 1e9}
            for i, d in enumerate(dates)]
    out = spread_bounds_frame(_frame(rows))
    got = out["spread_realistic"].to_numpy()
    # At $10 the tick floor dominates the era-scaled mega-cap schedule until decimals,
    # so the first three points ARE the tick and the fourth drops off it.
    assert got[0] == pytest.approx(0.125 / price)
    assert got[1] == pytest.approx(0.0625 / price)
    assert got[2] == pytest.approx(0.0625 / price)
    assert got[3] < got[2]
    for when, expected in zip(dates, got):
        ref = bounds_from_estimate(0.5, "upper_bound", 1e9, price=price,
                                   when=dt.date.fromisoformat(when))
        assert expected == pytest.approx(ref.realistic, rel=1e-12)
