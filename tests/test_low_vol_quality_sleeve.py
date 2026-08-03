"""Adversarial tests for the low-volatility / quality sleeve.

Two of these exist because the capacity study recorded the corresponding bugs (§4 of
`capacity_curve_result.md`): a delisting charged against the wrong year, and a delisted
name re-booked every month forever. Two more exist because this run found its own: a
per-name "month end" grid that manufactured 1,454 pseudo-periods, and a benchmark that
dropped the bankruptcies it charged to the strategy.

The volatility/beta test compares the prefix-sum implementation against a deliberately
slow pandas reference that cannot share a bug with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import research.sleeves.low_vol_quality as sleeve

DATES = pd.date_range("2000-01-31", periods=48, freq="ME")
NO_DELISTINGS = pd.DataFrame(columns=["ticker", "date", "action", "terminal_return"])


# --------------------------------------------------------------------------------------
# Trailing risk features
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def daily_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2010-01-04", periods=400)
    frames = []
    for i in range(12):
        n = len(days) - (0 if i < 8 else 120)  # deliberately ragged histories
        returns = rng.normal(0.0, 0.005 + 0.004 * i, size=n)
        returns[0] = 0.0
        price = 20.0 * np.exp(np.cumsum(returns))
        frames.append(pd.DataFrame({
            "ticker": f"T{i:02d}", "date": days[:n], "close": price, "closeadj": price,
            "volume": rng.integers(1000, 5000, size=n).astype(float),
        }))
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"]).reset_index(drop=True))


def _slow_reference(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    work = prices.copy()
    work["ret"] = work.groupby("ticker")["closeadj"].pct_change().clip(-1.0, 1.0)
    member = (work["ret"].notna() & (work["close"] >= sleeve.MIN_PROXY_PRICE)
              & (work["volume"] > 0))
    work["mkt"] = work["date"].map(work[member].groupby("date")["ret"].mean())

    rows = []
    for ticker, frame in work.groupby("ticker"):
        frame = frame.reset_index(drop=True)
        frame["month"] = frame["date"].values.astype("datetime64[M]")
        for end in frame.groupby("month").tail(1).index:
            span = frame.iloc[max(0, end - window + 1): end + 1]
            both = span[span["ret"].notna() & span["mkt"].notna()]
            if len(both) < 2:
                rows.append((ticker, frame.at[end, "date"], np.nan, np.nan, len(both)))
                continue
            rows.append((
                ticker, frame.at[end, "date"], both["ret"].std(ddof=1),
                both["ret"].cov(both["mkt"]) / both["mkt"].var(ddof=1), len(both),
            ))
    return pd.DataFrame(rows, columns=["ticker", "date", "vol_ref", "beta_ref", "n_ref"])


def test_vol_and_beta_match_a_slow_reference(daily_prices: pd.DataFrame) -> None:
    fast = sleeve.risk_features(daily_prices, window=60, min_observations=5)
    slow = _slow_reference(daily_prices, window=60)

    merged = fast.merge(slow, on=["ticker", "date"], how="outer", indicator=True)
    assert (merged["_merge"] == "both").all(), "month-end grids disagree"
    assert (merged["risk_n_obs"] == merged["n_ref"]).all()
    assert np.allclose(merged["realised_vol"], merged["vol_ref"], atol=1e-12)
    assert np.allclose(merged["beta"], merged["beta_ref"], atol=1e-9)


def test_window_never_reaches_into_the_previous_ticker(daily_prices: pd.DataFrame) -> None:
    fast = sleeve.risk_features(daily_prices, window=60, min_observations=5)
    first = fast.sort_values(["ticker", "date"]).groupby("ticker").head(1)
    # January 2010 has fewer than 25 business days, so a window that stayed inside the
    # ticker cannot possibly have accumulated more observations than that.
    assert (first["risk_n_obs"] <= 25).all()


def test_a_non_trading_shell_is_rejected_not_ranked_best(daily_prices: pd.DataFrame) -> None:
    """The failure mode unique to a low-vol signal: a flat tape looks like zero risk."""
    days = daily_prices["date"].unique()
    flat = pd.DataFrame({"ticker": "FLAT", "date": days, "close": 10.0,
                         "closeadj": 10.0, "volume": 0.0})
    mixed = (pd.concat([daily_prices, flat], ignore_index=True)
             .sort_values(["ticker", "date"]).reset_index(drop=True))
    features = sleeve.risk_features(mixed, window=60, min_observations=5)
    shell = features[features["ticker"] == "FLAT"]
    assert len(shell) > 0
    assert shell["realised_vol"].isna().all(), "a zero-volume shell survived the guard"


# --------------------------------------------------------------------------------------
# Backtest accounting
# --------------------------------------------------------------------------------------
def make_panel(n_names: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_names):
        for date in DATES:
            rows.append({
                "ticker": f"N{i:03d}", "date": date, "close": 10.0,
                "median_dollar_volume": 2_000_000.0, "spread": 0.02,
                "band_group": "B3_1M_5M",
                "forward_return": float(rng.normal(0.01, 0.06)),
                "realised_vol": 0.02 + 0.0001 * i, "beta": 1.0,
                "signal": float(-i),  # deterministic ranking: N000 is always best
            })
    return pd.DataFrame(rows)


def _truncate(panel: pd.DataFrame, ticker: str, last: str) -> pd.DataFrame:
    """Drop a name after ``last`` and blank its final forward return, as the real panel does."""
    out = panel[~((panel["ticker"] == ticker) & (panel["date"] > last))].copy()
    out.loc[out.index[out["ticker"] == ticker][-1], "forward_return"] = np.nan
    return out


def test_holding_the_whole_universe_at_zero_cost_equals_the_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(sleeve, "MIN_CROSS_SECTION", sleeve.N_POSITIONS)
    monkeypatch.setattr(sleeve, "IMPACT_COEFFICIENT", 0.0)
    monkeypatch.setattr(sleeve, "COMMISSION_MIN_PER_ORDER", 0.0)
    monkeypatch.setattr(sleeve, "COMMISSION_PER_SHARE", 0.0)
    monkeypatch.setattr(sleeve, "FX_COST_EACH_WAY", 0.0)

    result = sleeve.run_band(make_panel(sleeve.N_POSITIONS).assign(spread=0.0),
                             "B3_1M_5M", NO_DELISTINGS)
    assert result.net_return_annual == pytest.approx(result.benchmark_return_annual,
                                                     abs=1e-12)


def test_a_delisting_years_after_the_exit_is_not_charged() -> None:
    """Prior bug: `terminal.get(ticker)` booked a 2012 bankruptcy against a 2003 exit."""
    panel = _truncate(make_panel(80), "N000", "2001-06-30")
    late = pd.DataFrame([{"ticker": "N000", "date": pd.Timestamp("2003-11-30"),
                          "action": "bankruptcyliquidation", "terminal_return": -1.0}])
    assert sleeve.run_band(panel, "B3_1M_5M", late).net_return_annual == pytest.approx(
        sleeve.run_band(panel, "B3_1M_5M", NO_DELISTINGS).net_return_annual, abs=1e-12
    )


def test_an_in_window_delisting_is_booked_exactly_once() -> None:
    """Prior bug: a delisted name stayed in the book and lost 100% again every month."""
    panel = _truncate(make_panel(80), "N000", "2001-06-30")
    timely = pd.DataFrame([{"ticker": "N000", "date": pd.Timestamp("2001-07-15"),
                            "action": "bankruptcyliquidation", "terminal_return": -1.0}])
    clean = sleeve.run_band(panel, "B3_1M_5M", NO_DELISTINGS)
    hit = sleeve.run_band(panel, "B3_1M_5M", timely)

    years = hit.n_months / 12.0
    one_position = 1.0 / sleeve.N_POSITIONS / years
    damage = clean.net_return_annual - hit.net_return_annual
    assert 0.0 < damage <= one_position * 1.05, f"re-booked or mis-scaled: {damage}"


def test_the_benchmark_eats_the_same_bankruptcies_as_the_strategy() -> None:
    """A benchmark that drops delisted names is survivorship-biased against the book."""
    panel = _truncate(make_panel(80), "N000", "2001-06-30")
    timely = pd.DataFrame([{"ticker": "N000", "date": pd.Timestamp("2001-07-15"),
                            "action": "bankruptcyliquidation", "terminal_return": -1.0}])
    clean = sleeve.run_band(panel, "B3_1M_5M", NO_DELISTINGS)
    hit = sleeve.run_band(panel, "B3_1M_5M", timely)
    assert hit.benchmark_return_annual < clean.benchmark_return_annual


def test_a_long_only_book_cannot_lose_more_than_everything() -> None:
    panel = make_panel(80)
    doomed = [f"N{i:03d}" for i in range(30)]
    kill = pd.DataFrame([{"ticker": t, "date": DATES[10] + pd.Timedelta(days=20),
                          "action": "bankruptcyliquidation", "terminal_return": -1.0}
                         for t in doomed])
    panel = panel[~(panel["ticker"].isin(doomed) & (panel["date"] > DATES[10]))].copy()
    panel.loc[panel["ticker"].isin(doomed) & (panel["date"] == DATES[10]),
              "forward_return"] = np.nan

    result = sleeve.run_band(panel, "B3_1M_5M", kill)
    assert result.net_return_annual > -1.5
    assert result.max_drawdown <= 1.0 + 1e-9


def test_the_rebalance_grid_is_calendar_monthly() -> None:
    """The defect this run found: per-name month ends became extra pseudo-periods."""
    panel = make_panel(80)
    # Give a handful of names a mid-month final bar, as a delisting does in the real panel.
    stragglers = panel["ticker"].isin(["N005", "N006"]) & (panel["date"] == DATES[20])
    panel.loc[stragglers, "date"] = DATES[20] - pd.Timedelta(days=11)
    result = sleeve.run_band(panel, "B3_1M_5M", NO_DELISTINGS)
    assert result.n_months == len(DATES), "off-grid dates leaked in as extra periods"


def test_costs_respond_to_the_spread() -> None:
    cheap = sleeve.run_band(make_panel(80).assign(spread=0.001), "B3_1M_5M", NO_DELISTINGS)
    dear = sleeve.run_band(make_panel(80).assign(spread=0.10), "B3_1M_5M", NO_DELISTINGS)
    assert dear.cost_drag_annual > cheap.cost_drag_annual > 0.0


# --------------------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------------------
def test_the_composite_prefers_low_volatility_names() -> None:
    rng = np.random.default_rng(11)
    panel = make_panel(60).drop(columns=["signal"])
    panel["beta"] = panel["realised_vol"] * 40.0
    panel["gross_profitability"] = rng.normal(0.3, 0.1, len(panel))
    panel["debt_to_equity"] = rng.normal(0.5, 0.2, len(panel))
    panel["accruals"] = rng.normal(0.0, 0.05, len(panel))

    scored = sleeve.build_signal(panel)
    month = scored[scored["date"] == DATES[5]]
    assert month["signal"].corr(month["realised_vol"]) < -0.6
    picked = month.nlargest(sleeve.N_POSITIONS, "signal")
    assert picked["realised_vol"].mean() < month["realised_vol"].mean()


def test_a_name_missing_a_leg_is_not_ranked() -> None:
    """All three registered legs are required; two of them is a different strategy."""
    rng = np.random.default_rng(5)
    panel = make_panel(60).drop(columns=["signal"])
    panel["beta"] = panel["realised_vol"] * 40.0  # must VARY or its z-score is undefined
    panel["gross_profitability"] = rng.normal(0.3, 0.1, len(panel))
    panel["debt_to_equity"] = rng.normal(0.5, 0.2, len(panel))
    panel["accruals"] = rng.normal(0.0, 0.05, len(panel))
    panel.loc[panel["ticker"] == "N000", "beta"] = np.nan

    scored = sleeve.build_signal(panel)
    assert scored.loc[scored["ticker"] == "N000", "signal"].isna().all()
    # ...and the rest of the cross-section is still ranked, so this is not vacuous.
    assert scored.loc[scored["ticker"] != "N000", "signal"].notna().all()


def test_quality_needs_two_of_its_three_inputs() -> None:
    rng = np.random.default_rng(6)
    panel = make_panel(60).drop(columns=["signal"])
    panel["beta"] = panel["realised_vol"] * 40.0
    panel["gross_profitability"] = rng.normal(0.3, 0.1, len(panel))
    panel["debt_to_equity"] = rng.normal(0.5, 0.2, len(panel))
    panel["accruals"] = rng.normal(0.0, 0.05, len(panel))
    only_one = panel["ticker"] == "N001"
    panel.loc[only_one, ["debt_to_equity", "accruals"]] = np.nan
    two_left = panel["ticker"] == "N002"
    panel.loc[two_left, "accruals"] = np.nan

    scored = sleeve.build_signal(panel)
    assert scored.loc[only_one, "signal"].isna().all(), "ranked on one quality input"
    assert scored.loc[two_left, "signal"].notna().all(), "two of three should suffice"
