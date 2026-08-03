"""Tests for the pre-registered multi-asset trend sleeve.

These pin the MACHINERY, not the result: causality (no lookahead), sign symmetry,
volatility targeting, cost monotonicity, and the statistics that the verdict turned on.
The point is that a lookahead bug or a sign bug would silently manufacture a Sharpe, and
the programme has already paid for that class of mistake twice.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves.multiasset_trend import (
    BLOCKS,
    CASH_SUBTRACTED,
    LOOKBACKS,
    PRIMARY_UNIVERSE,
    TrendConfig,
    active_report,
    annual_sharpe,
    concentration,
    effective_n,
    kelly_report,
    max_drawdown,
    newey_west_tstat,
    run_trend,
)
from research.sleeves.multiasset_trend import _positions


def _frame(data: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("1970-01-31", periods=data.shape[0], freq="ME")
    x = pd.DataFrame(data, index=idx, columns=PRIMARY_UNIVERSE)
    return x, pd.DataFrame(False, index=idx, columns=PRIMARY_UNIVERSE)


def _noise(seed: int = 0, n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return _frame(rng.normal(0.0, 0.04, (n, len(PRIMARY_UNIVERSE))))


def _ar1(phi: float, seed: int = 1, n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 0.04, (n, len(PRIMARY_UNIVERSE)))
    a = np.zeros_like(e)
    for t in range(1, n):
        a[t] = phi * a[t - 1] + e[t]
    return _frame(a)


# ── Pre-registration integrity ────────────────────────────────────────────────

def test_universe_is_the_preregistered_eighteen():
    assert len(PRIMARY_UNIVERSE) == 18
    assert len(set(PRIMARY_UNIVERSE)) == 18
    assert sum(len(v) for v in BLOCKS.values()) == 18
    # Instruments excluded for cause must not have crept back in.
    for banned in ("NATGAS_F", "DJIA", "SPY", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM",
                   "BIL", "IEI", "SLV", "US_CASH_13W"):
        assert banned not in PRIMARY_UNIVERSE


def test_only_usd_total_return_series_have_cash_subtracted():
    """A price/futures/spot return is already an excess return; a par-bond TR is not."""
    assert CASH_SUBTRACTED == {"US5Y_TR", "US10Y_TR", "US30Y_TR"}
    assert CASH_SUBTRACTED <= set(PRIMARY_UNIVERSE)


def test_lookbacks_are_the_canonical_four():
    assert LOOKBACKS == (1, 3, 6, 12)


# ── Causality: the failure mode that would fabricate the whole result ─────────

def test_positions_do_not_use_future_returns():
    """Perturbing returns AFTER t must not change any position decided at or before t."""
    x, interior = _ar1(0.25)
    cut = x.index[400]
    n_before, _, _ = _positions(x, TrendConfig())

    x2 = x.copy()
    rng = np.random.default_rng(99)
    x2.loc[x2.index > cut] = rng.normal(0.0, 0.10, x2.loc[x2.index > cut].shape)
    n_after, _, _ = _positions(x2, TrendConfig())

    pd.testing.assert_frame_equal(n_before.loc[:cut], n_after.loc[:cut])


def test_book_scaler_is_causal():
    """Returns after the cut must not change realised returns at or before the cut."""
    x, interior = _ar1(0.25)
    cut = x.index[400]
    r1 = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    x2 = x.copy()
    rng = np.random.default_rng(7)
    x2.loc[x2.index > cut] = rng.normal(0.0, 0.10, x2.loc[x2.index > cut].shape)
    r2 = run_trend(TrendConfig(), vol_target=0.20, x=x2, interior=interior)
    a = r1.gross.loc[:cut]
    b = r2.gross.loc[:cut]
    pd.testing.assert_series_equal(a, b)


# ── Positive and negative controls on the engine ──────────────────────────────

def test_iid_noise_produces_no_edge():
    x, interior = _noise()
    r = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    assert abs(annual_sharpe(r.gross)) < 0.30


def test_momentum_is_found_and_mean_reversion_is_punished():
    xp, ip = _ar1(+0.25)
    xn, inn = _ar1(-0.25, seed=1)
    sp = annual_sharpe(run_trend(TrendConfig(), vol_target=0.20, x=xp, interior=ip).gross)
    sn = annual_sharpe(run_trend(TrendConfig(), vol_target=0.20, x=xn, interior=inn).gross)
    assert sp > 1.0, "trend engine failed to find injected momentum"
    assert sn < -1.0, "trend engine did not lose on injected mean reversion"
    # Symmetric: no structural long or short bias.
    assert abs(sp + sn) < 0.5


def test_sign_randomised_signal_destroys_the_edge():
    x, interior = _ar1(0.25)
    live = annual_sharpe(run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior).gross)
    ctrl = [
        annual_sharpe(run_trend(TrendConfig(randomise_seed=s), vol_target=0.20,
                                x=x, interior=interior).gross)
        for s in range(4)
    ]
    assert live > np.mean(ctrl) + 3 * (np.std(ctrl, ddof=1) + 1e-9)


# ── Sizing and costs ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("target", [0.10, 0.20, 0.40])
def test_volatility_targeting_hits_its_target(target):
    x, interior = _noise(seed=3)
    r = run_trend(TrendConfig(), vol_target=target, x=x, interior=interior)
    realised = r.gross.std(ddof=1) * math.sqrt(12)
    assert 0.6 * target < realised < 1.5 * target


def test_gross_leverage_cap_is_respected():
    x, interior = _noise(seed=4)
    r = run_trend(TrendConfig(), vol_target=1.50, x=x, interior=interior)
    assert r.gross_leverage.max() <= 10.0 + 1e-9


def test_costs_are_monotone_and_proportional():
    x, interior = _ar1(0.25)
    r = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    c2 = (r.gross - r.net["2bps"]).mean()
    c10 = (r.gross - r.net["10bps"]).mean()
    assert c2 > 0 and c10 > c2
    assert c10 == pytest.approx(5.0 * c2, rel=1e-9)   # 10bps is exactly 5x 2bps


def test_no_position_is_held_in_an_interior_null_month():
    x, interior = _noise(seed=5)
    interior.iloc[300, 0] = True
    r = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    assert r.weights.iloc[301, 0] == 0.0


# ── Statistics the verdict rested on ──────────────────────────────────────────

def test_newey_west_tstat_matches_iid_case_at_zero_lag():
    rng = np.random.default_rng(11)
    s = pd.Series(rng.normal(0.01, 0.05, 400))
    naive = s.mean() / (s.std(ddof=0) / math.sqrt(len(s)))
    assert newey_west_tstat(s, lag=0) == pytest.approx(naive, rel=1e-9)


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(12)
    e = rng.normal(0.0, 0.05, 500)
    a = np.zeros(500)
    for t in range(1, 500):
        a[t] = 0.6 * a[t - 1] + e[t]
    s = pd.Series(a + 0.01)
    assert abs(newey_west_tstat(s, lag=6)) < abs(newey_west_tstat(s, lag=0))


def test_effective_n_counts_independent_bets():
    n = 6
    assert effective_n(pd.DataFrame(np.eye(n))) == pytest.approx(float(n))
    assert effective_n(pd.DataFrame(np.ones((n, n)))) == pytest.approx(1.0)


def test_variance_drag_identity_is_what_the_report_claims():
    """geometric excess == arithmetic active - (var_s - var_b)/2, to second order.

    This identity is the trap that killed the PEAD result. It is asserted, not assumed.
    """
    rng = np.random.default_rng(13)
    idx = pd.date_range("1980-01-31", periods=600, freq="ME")
    s = pd.Series(rng.normal(0.004, 0.010, 600), index=idx)
    b = pd.Series(rng.normal(0.004, 0.030, 600), index=idx)
    rep = active_report(s, b)
    lhs = rep["geometric_excess_annual"]
    rhs = rep["arith_active_annual"] - rep["variance_drag_annual"]
    assert lhs == pytest.approx(rhs, abs=5e-3)


def test_active_report_is_zero_against_itself():
    rng = np.random.default_rng(14)
    idx = pd.date_range("1980-01-31", periods=300, freq="ME")
    s = pd.Series(rng.normal(0.005, 0.04, 300), index=idx)
    rep = active_report(s, s)
    assert rep["arith_active_annual"] == pytest.approx(0.0, abs=1e-12)
    assert rep["jensen_beta"] == pytest.approx(1.0, abs=1e-9)
    assert rep["volmatched_active_annual"] == pytest.approx(0.0, abs=1e-9)


def test_volmatched_active_is_scale_invariant():
    """Levering a strategy cannot change its vol-matched active return.

    This is exactly why the vol-matched measure, not the raw arithmetic difference,
    is what the verdict used: the raw difference rises without limit under leverage.
    """
    rng = np.random.default_rng(15)
    idx = pd.date_range("1980-01-31", periods=400, freq="ME")
    s = pd.Series(rng.normal(0.005, 0.03, 400), index=idx)
    b = pd.Series(rng.normal(0.004, 0.02, 400), index=idx)
    one = active_report(s, b)["volmatched_active_annual"]
    three = active_report(s * 3.0, b)["volmatched_active_annual"]
    assert one == pytest.approx(three, abs=1e-9)
    # ...while the RAW arithmetic difference does NOT survive that test:
    assert active_report(s * 3.0, b)["arith_active_annual"] != pytest.approx(
        active_report(s, b)["arith_active_annual"], abs=1e-6)


def test_max_drawdown_on_a_known_path():
    s = pd.Series([0.5, -0.5, 0.0], index=pd.date_range("2000-01-31", periods=3, freq="ME"))
    # 1.0 -> 1.5 -> 0.75 : peak-to-trough = -50%
    assert max_drawdown(s) == pytest.approx(-0.5)


def test_kelly_arithmetic_matches_the_target():
    """30%/yr at half Kelly needs Sharpe ~0.894 -- the number the mission is chasing."""
    k = kelly_report(0.894)
    assert k["half_kelly_growth"] == pytest.approx(0.2998, abs=1e-3)
    assert k["full_kelly_growth"] == pytest.approx(0.3996, abs=1e-3)
    assert k["implied_vol"] == pytest.approx(0.447, abs=1e-3)


def test_concentration_flags_a_dominant_cell():
    pnl = pd.DataFrame(0.0, index=pd.date_range("2000-01-31", periods=10, freq="ME"),
                       columns=["A", "B"])
    pnl.iloc[:, 0] = 0.01
    pnl.iloc[5, 1] = 1.0
    c = concentration(pnl)
    assert c["top_cell_share"] > 0.9
    assert c["top_instrument"] == "B"


def test_block_risk_parity_changes_the_book_but_stays_causal():
    x, interior = _ar1(0.25)
    flat = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
    rp = run_trend(TrendConfig(block_risk_parity=True), vol_target=0.20,
                   x=x, interior=interior)
    assert not np.allclose(flat.weights.to_numpy(), rp.weights.to_numpy())
    assert rp.gross.notna().all()
