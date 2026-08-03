"""Offline tests for the cross-asset carry sleeve.

No network, no files, no fixtures from the real panel. Every test pins a property that
holds independently of the implementation — a closed-form identity, an absence of
lookahead, or a positive/negative control — because a backtest that is merely
self-consistent is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.carry import (
    FX_INSTRUMENTS,
    FxInstrument,
    backtest,
    benchmark_positions,
    carry_positions,
    decompose_pnl,
    drawdown_curve,
    fx_excess_returns,
    newey_west_tstat,
    ols_alpha,
    performance,
    rank_weights,
    realised_dividend_yield,
    scan_quarantine_candidates,
    sharpe_by_decade,
    trailing_vol,
    trend_positions,
    vol_matched_active,
)

MONTHS = pd.date_range("2000-01-31", periods=180, freq="ME")


# ── registry ──────────────────────────────────────────────────────────────────

def test_fx_registry_inversion_flags_match_the_quote_convention():
    """A ticker quoted FOREIGN-per-USD must be inverted; USD-per-FOREIGN must not be."""
    for inst in FX_INSTRUMENTS:
        usd_per_foreign = inst.ticker.endswith("USD=X")
        assert inst.invert is not usd_per_foreign, inst.key
    assert len({i.key for i in FX_INSTRUMENTS}) == len(FX_INSTRUMENTS)
    assert len({i.ccy for i in FX_INSTRUMENTS}) == len(FX_INSTRUMENTS)


# ── rank weights ──────────────────────────────────────────────────────────────

def test_rank_weights_are_dollar_neutral_and_unit_gross():
    w = rank_weights(pd.Series({"a": 0.1, "b": 0.5, "c": -0.2, "d": 0.9}))
    assert abs(w.sum()) < 1e-12
    assert abs(w.abs().sum() - 1.0) < 1e-12
    assert w["d"] > w["b"] > w["a"] > w["c"]


def test_rank_weights_refuse_to_bet_when_every_score_is_identical():
    w = rank_weights(pd.Series({"a": 0.3, "b": 0.3, "c": 0.3}))
    assert (w.abs() < 1e-12).all()


def test_rank_weights_ignore_missing_scores():
    w = rank_weights(pd.Series({"a": 1.0, "b": np.nan, "c": -1.0}))
    assert w["b"] == 0.0
    assert abs(w.abs().sum() - 1.0) < 1e-12


# ── FX carry construction ─────────────────────────────────────────────────────

def _fx_frames(diff_annual: float, spot: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot_ret = pd.DataFrame({"FX_EUR": spot}, index=MONTHS)
    rates = pd.DataFrame({"US": 0.02, "EZ": 0.02 + diff_annual}, index=MONTHS)
    return spot_ret, rates


def test_fx_excess_return_is_spot_plus_the_lagged_differential():
    """A 5%/yr differential with a dead spot must pay exactly 5%/12 a month, one month late."""
    inst = (FxInstrument("FX_EUR", "EURUSD=X", "EZ", False),)
    spot_ret, rates = _fx_frames(0.05, 0.0)
    excess, carry = fx_excess_returns(spot_ret, rates, inst)
    assert np.isnan(excess["FX_EUR"].iloc[0])          # nothing contracted before month 1
    assert excess["FX_EUR"].iloc[1:].sub(0.05 / 12).abs().max() < 1e-15
    assert abs(carry["FX_EUR"].iloc[0] - 0.05) < 1e-15


def test_fx_carry_uses_the_previous_month_rate_not_this_month():
    """Point-in-time: a rate that jumps at month k must not change month k's return."""
    inst = (FxInstrument("FX_EUR", "EURUSD=X", "EZ", False),)
    spot_ret = pd.DataFrame({"FX_EUR": 0.0}, index=MONTHS)
    rates = pd.DataFrame({"US": 0.0, "EZ": 0.0}, index=MONTHS)
    rates.loc[MONTHS[10]:, "EZ"] = 0.12
    excess, _ = fx_excess_returns(spot_ret, rates, inst)
    assert abs(float(excess["FX_EUR"].iloc[10])) < 1e-15        # jump month: still old rate
    assert abs(float(excess["FX_EUR"].iloc[11]) - 0.01) < 1e-15  # next month: 12%/12


def test_fx_excess_returns_skip_instruments_without_a_rate():
    inst = (FxInstrument("FX_XYZ", "XYZUSD=X", "XY", False),)
    spot_ret = pd.DataFrame({"FX_XYZ": 0.0}, index=MONTHS)
    rates = pd.DataFrame({"US": 0.02}, index=MONTHS)
    excess, carry = fx_excess_returns(spot_ret, rates, inst)
    assert excess.empty and carry.empty


# ── dividend yield ────────────────────────────────────────────────────────────

def test_realised_dividend_yield_recovers_a_known_dividend():
    """PR compounding at 0.4%/mo and TR at (1.004*1.0015)-1 must give ~1.8%/yr."""
    monthly_div = 0.0015
    pr = pd.Series(0.004, index=MONTHS)
    tr = (1.0 + pr) * (1.0 + monthly_div) - 1.0
    dy = realised_dividend_yield(tr, pr, window=12)
    expected = (1.0 + monthly_div) ** 12 - 1.0
    assert abs(float(dy.iloc[-1]) - expected) < 1e-12
    assert dy.iloc[:11].isna().all()


# ── volatility: strictly backward-looking ─────────────────────────────────────

def test_trailing_vol_has_no_lookahead():
    rng = np.random.default_rng(7)
    rets = pd.DataFrame({"A": rng.normal(0, 0.03, len(MONTHS))}, index=MONTHS)
    full = trailing_vol(rets)
    truncated = trailing_vol(rets.iloc[:100])
    pd.testing.assert_series_equal(full["A"].iloc[:100], truncated["A"], check_names=False)


def test_trailing_vol_equals_the_window_standard_deviation():
    rng = np.random.default_rng(11)
    rets = pd.DataFrame({"A": rng.normal(0, 0.03, len(MONTHS))}, index=MONTHS)
    got = float(trailing_vol(rets)["A"].iloc[50])
    want = float(rets["A"].iloc[15:51].std()) * np.sqrt(12)
    assert abs(got - want) < 1e-12


# ── positions ─────────────────────────────────────────────────────────────────

def _panel(n_inst: int = 8, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    keys = [f"I{i}" for i in range(n_inst)]
    carry = pd.DataFrame({k: 0.01 * (i + 1) for i, k in enumerate(keys)}, index=MONTHS)
    rets = pd.DataFrame(rng.normal(0.0, 0.03, (len(MONTHS), n_inst)), index=MONTHS, columns=keys)
    return carry, rets


def test_carry_positions_have_no_lookahead():
    carry, rets = _panel()
    full, _, _ = carry_positions(carry, rets)
    trunc, _, _ = carry_positions(carry.iloc[:120], rets.iloc[:120])
    pd.testing.assert_frame_equal(full.iloc[:120], trunc)


def test_carry_positions_are_RISK_neutral_when_live():
    """Inverse-vol scaling deliberately breaks NOTIONAL neutrality; what must survive is
    equal risk on each side, Σ(pos_i·σ_i) = 0."""
    carry, rets = _panel()
    pos, vol, _ = carry_positions(carry, rets)
    live = pos.abs().sum(axis=1) > 0
    assert live.any()
    risk = (pos * vol.fillna(0.0)).sum(axis=1)
    assert risk[live].abs().max() < 1e-12
    assert pos[live].sum(axis=1).abs().max() > 1e-6      # and notional neutrality is NOT held


def test_carry_positions_refuse_a_thin_cross_section():
    carry, rets = _panel(n_inst=5)
    pos, _, n_elig = carry_positions(carry, rets, min_instruments=6)
    assert (pos.abs().sum(axis=1) == 0).all()
    assert n_elig.max() == 5


def test_carry_positions_are_long_high_carry_and_short_low_carry():
    carry, rets = _panel()
    pos, _, _ = carry_positions(carry, rets)
    last = pos.iloc[-1]
    assert last["I7"] > 0 and last["I0"] < 0


def test_permutation_control_keeps_the_book_but_destroys_the_ordering():
    """The control must change WHICH names are held, not how much risk is deployed."""
    carry, rets = _panel()
    live, vol, _ = carry_positions(carry, rets)
    perm, _, _ = carry_positions(carry, rets, permute_seed=42)
    row_live, row_perm, row_vol = live.iloc[-1], perm.iloc[-1], vol.iloc[-1]
    assert row_perm.abs().sum() > 0
    assert abs(float((row_perm * row_vol).sum())) < 1e-12          # still risk neutral
    assert abs(float((row_perm.abs() * row_vol).sum())
               - float((row_live.abs() * row_vol).sum())) < 1e-12  # same risk budget
    assert not np.allclose(row_live.to_numpy(), row_perm.to_numpy())


def test_benchmark_is_long_only_and_only_live_when_the_sleeve_is():
    carry, rets = _panel()
    pos, vol, n_elig = carry_positions(carry, rets)
    bench = benchmark_positions(rets, vol, n_elig)
    assert (bench >= -1e-15).to_numpy().all()
    assert ((bench.abs().sum(axis=1) > 0) == (pos.abs().sum(axis=1) > 0)).all()


def test_trend_positions_follow_the_sign_of_the_trailing_return():
    keys = [f"I{i}" for i in range(8)]
    rets = pd.DataFrame(0.0, index=MONTHS, columns=keys)
    rets.iloc[:, :4] = 0.01
    rets.iloc[:, 4:] = -0.01
    # a dead-flat series has zero vol, so give every column a little noise
    rng = np.random.default_rng(5)
    rets += rng.normal(0.0, 0.002, rets.shape)
    pos, _, _ = trend_positions(rets)
    last = pos.iloc[-1]
    assert (last[keys[:4]] > 0).all()
    assert (last[keys[4:]] < 0).all()


# ── backtest mechanics ────────────────────────────────────────────────────────

def test_pnl_is_booked_one_month_after_the_position_is_set():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    pos = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    rets = pd.DataFrame({"A": [0.10, 0.02, 0.03, 0.04]}, index=idx)
    res = backtest(pos, rets, round_trip_bps=0.0)
    # the first month's +10% happened BEFORE any position existed and must not be earned
    assert list(np.round(res["gross"].to_numpy(), 10)) == [0.02, 0.03, 0.04]


def test_turnover_charges_the_trade_not_the_drift():
    """Holding a position that drifts with the market is not a trade."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    rets = pd.DataFrame({"A": [0.0, 0.10, 0.0]}, index=idx)
    drifted = pd.DataFrame({"A": [1.0, 1.10, 1.10]}, index=idx)
    res = backtest(drifted, rets, round_trip_bps=100.0)
    assert abs(float(res["turnover"].loc[idx[1]])) < 1e-12
    flat = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    res_flat = backtest(flat, rets, round_trip_bps=100.0)
    assert abs(float(res_flat["turnover"].loc[idx[1]]) - 0.10) < 1e-12


def test_zero_cost_makes_net_equal_gross():
    carry, rets = _panel()
    pos, _, _ = carry_positions(carry, rets)
    res = backtest(pos, rets, round_trip_bps=0.0)
    pd.testing.assert_series_equal(res["gross"], res["net"], check_names=False)


def test_cost_is_monotone_in_the_spread_and_never_helps():
    carry, rets = _panel()
    pos, _, _ = carry_positions(carry, rets)
    cheap = backtest(pos, rets, round_trip_bps=3.0)["net"].sum()
    dear = backtest(pos, rets, round_trip_bps=30.0)["net"].sum()
    assert dear < cheap


def test_missing_returns_under_a_live_position_are_counted_not_hidden():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    pos = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    rets = pd.DataFrame({"A": [0.01, np.nan, 0.03, 0.04]}, index=idx)
    res = backtest(pos, rets, round_trip_bps=0.0)
    assert res["n_missing_return_cells"] == 1


# ── decomposition ─────────────────────────────────────────────────────────────

def test_decomposition_is_all_accrual_when_prices_do_not_move():
    keys = [f"I{i}" for i in range(8)]
    carry = pd.DataFrame({k: 0.01 * (i + 1) for i, k in enumerate(keys)}, index=MONTHS)
    # returns EXACTLY equal the carry the position was set on: prices never moved
    rets = carry.shift(1) / 12.0
    rets.iloc[0] = 0.0
    # give the vol estimator something to work with without changing the identity
    vol_source = rets + pd.DataFrame(
        np.random.default_rng(1).normal(0, 1e-6, rets.shape), index=rets.index, columns=keys)
    pos, _, _ = carry_positions(carry, vol_source)
    parts = decompose_pnl(pos, rets, carry)
    assert abs(parts["price_pnl"]) < 1e-9
    assert abs(parts["accrual_share"] - 1.0) < 1e-6


def test_decomposition_adds_up():
    carry, rets = _panel()
    pos, _, _ = carry_positions(carry, rets)
    parts = decompose_pnl(pos, rets, carry)
    assert abs(parts["accrual_pnl"] + parts["price_pnl"] - parts["total_gross_pnl"]) < 1e-9


# ── statistics ────────────────────────────────────────────────────────────────

def test_newey_west_collapses_to_the_iid_t_stat_at_zero_lags():
    rng = np.random.default_rng(19)
    x = pd.Series(rng.normal(0.01, 0.05, 400))
    _, _, t = newey_west_tstat(x, lags=0)
    iid = float(x.mean()) / (float(x.std(ddof=0)) / np.sqrt(len(x)))
    assert abs(t - iid) < 1e-9


def test_newey_west_penalises_positive_autocorrelation():
    rng = np.random.default_rng(23)
    e = rng.normal(0.0, 0.02, 600)
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = 0.7 * x[i - 1] + e[i]
    s = pd.Series(x + 0.01)
    _, _, t_iid = newey_west_tstat(s, lags=0)
    _, _, t_hac = newey_west_tstat(s, lags=6)
    assert abs(t_hac) < abs(t_iid)


def test_ols_alpha_recovers_a_planted_alpha_and_beta():
    rng = np.random.default_rng(29)
    bench = pd.Series(rng.normal(0.005, 0.04, 300))
    strat = 0.002 + 0.6 * bench
    got = ols_alpha(strat, bench)
    assert abs(got["beta"] - 0.6) < 1e-9
    assert abs(got["alpha_annual"] - 0.002 * 12) < 1e-9


def test_performance_reports_arithmetic_not_geometric_as_the_sharpe_numerator():
    """The variance-drag trap: a series whose geometric mean is negative but whose
    arithmetic mean is positive must show a POSITIVE Sharpe and a NEGATIVE CAGR."""
    r = pd.Series([0.5, -0.4] * 60, index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    stats = performance(r)
    assert stats["arithmetic_annual"] > 0
    assert stats["geometric_annual"] < 0
    assert stats["sharpe"] > 0


def test_vol_matched_active_is_invariant_to_levering_the_strategy():
    """THE point of the statistic. The raw difference's t-stat rises with leverage — that
    is how the trend sleeve faked a significant active return — while the vol-matched
    one must not move at all."""
    rng = np.random.default_rng(37)
    idx = pd.date_range("2000-01-31", periods=300, freq="ME")
    bench = pd.Series(rng.normal(0.004, 0.030, 300), index=idx)
    strat = pd.Series(rng.normal(0.005, 0.012, 300), index=idx)
    base = vol_matched_active(strat, bench)
    raw_t, matched_t = [], []
    for k in (1.0, 2.5, 5.0, 10.0):
        got = vol_matched_active(strat * k, bench)
        raw_t.append(got["raw_active_tstat"])
        matched_t.append(got["vol_matched_active_tstat"])
    assert max(matched_t) - min(matched_t) < 1e-9          # invariant
    assert max(raw_t) - min(raw_t) > 0.2                   # the raw one is not
    assert abs(base["benchmark_scale_factor"] - float(strat.std()) / float(bench.std())) < 1e-12


def test_vol_matched_active_recovers_a_planted_difference():
    """Identical shape, strategy shifted up by a known constant ⇒ that constant back."""
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(41)
    bench = pd.Series(rng.normal(0.003, 0.02, 240), index=idx)
    strat = bench + 0.001
    got = vol_matched_active(strat, bench)
    assert abs(got["benchmark_scale_factor"] - 1.0) < 1e-12
    assert abs(got["vol_matched_active_annual"] - 0.012) < 1e-12


def test_vol_matched_active_needs_an_overlap():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    assert vol_matched_active(pd.Series(0.01, index=idx), pd.Series(0.01, index=idx)) == {}


def test_drawdown_curve_matches_a_hand_computation():
    r = pd.Series([0.1, -0.5, 0.2], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    dd = drawdown_curve(r)
    assert abs(float(dd.min()) - (0.55 / 1.1 - 1.0)) < 1e-12


def test_sharpe_by_decade_splits_on_calendar_decades():
    idx = pd.date_range("2005-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(31)
    r = pd.Series(rng.normal(0.01, 0.03, 240), index=idx)
    out = sharpe_by_decade(r)
    assert set(out) == {"2000s", "2010s", "2020s"}
    assert out["2010s"]["n_months"] == 120


# ── controls ──────────────────────────────────────────────────────────────────

LONG_MONTHS = pd.date_range("1970-01-31", periods=600, freq="ME")


def _control_panel(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ten instruments whose expected return IS their carry, plus iid noise.

    600 months so the control is decisive rather than a coin flip: the analytic Sharpe of
    this construction is ~1.31 and the standard error of a Sharpe over 50 years is ~0.14.
    """
    rng = np.random.default_rng(seed)
    keys = [f"I{i}" for i in range(10)]
    carry = pd.DataFrame({k: 0.005 * (i + 1) for i, k in enumerate(keys)}, index=LONG_MONTHS)
    rets = carry / 12.0 + rng.normal(0.0, 0.01, (len(LONG_MONTHS), 10))
    return carry, rets


def test_positive_control_a_real_carry_edge_is_found():
    """Instruments whose returns ARE their carry plus noise: the sleeve must profit."""
    carry, rets = _control_panel(101)
    pos, _, _ = carry_positions(carry, rets)
    res = backtest(pos, rets, round_trip_bps=3.0)
    assert performance(res["net"])["sharpe"] > 0.9


def test_negative_control_permuted_scores_earn_nothing():
    carry, rets = _control_panel(103)
    live = performance(backtest(carry_positions(carry, rets)[0], rets,
                                round_trip_bps=3.0)["net"])["sharpe"]
    shuffled = [
        performance(backtest(carry_positions(carry, rets, permute_seed=s)[0], rets,
                              round_trip_bps=3.0)["net"])["sharpe"]
        for s in (1, 2, 3, 4)
    ]
    assert live > float(np.mean(shuffled)) + 2.0 * float(np.std(shuffled))


# ── quarantine scan ───────────────────────────────────────────────────────────

def _spike_series(spike_date: str, size: float, reverse: bool) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2008-01-01", "2008-12-31")
    lvl = pd.Series(1.0, index=idx)
    pos = int(idx.get_loc(pd.Timestamp(spike_date)))
    lvl.iloc[pos] = 1.0 + size
    if not reverse:                      # a permanent re-rating, not a bad print
        lvl.iloc[pos:] = 1.0 + size
    ret = lvl / lvl.shift(1) - 1.0
    return lvl, ret


def test_quarantine_admits_the_spike_and_refuses_its_own_reversal():
    """The reversal bar also lands in the 8th/9th window and is also >5%, so it is
    REPORTED as a candidate — and must be refused, or the scan would delete a genuine
    two-day window instead of one bad print."""
    lvl, ret = _spike_series("2008-12-08", 0.17, reverse=True)
    rows = scan_quarantine_candidates({"X": lvl}, {"X": ret}, {"X": False})
    admitted = [r for r in rows if r["admitted"]]
    assert [r["date"] for r in admitted] == ["2008-12-08"]
    assert [r["date"] for r in rows] == ["2008-12-08", "2008-12-09"]


def test_quarantine_keeps_a_real_move_that_does_not_round_trip():
    lvl, ret = _spike_series("2008-12-08", 0.17, reverse=False)
    rows = scan_quarantine_candidates({"X": lvl}, {"X": ret}, {"X": False})
    assert len(rows) == 1 and rows[0]["admitted"] is False


def test_quarantine_ignores_dates_outside_the_published_criterion():
    lvl, ret = _spike_series("2008-12-10", 0.17, reverse=True)
    assert scan_quarantine_candidates({"X": lvl}, {"X": ret}, {"X": False}) == []
    lvl2, ret2 = _spike_series("2008-12-08", 0.03, reverse=True)
    assert scan_quarantine_candidates({"X": lvl2}, {"X": ret2}, {"X": False}) == []


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_risk_scaling_never_divides_by_a_non_positive_vol(bad):
    carry, rets = _panel()
    rets.iloc[:, 0] = 0.0                      # zero variance ⇒ zero vol
    pos, vol, _ = carry_positions(carry, rets)
    assert np.isfinite(pos.to_numpy()).all()
    assert (pos["I0"] == 0.0).all()
    assert bad <= 0.0
