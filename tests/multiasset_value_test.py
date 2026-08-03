"""Tests for the pre-registered cross-asset VALUE sleeve.

These pin the MACHINERY, not the result: causality (no lookahead), the sign convention
(cheap = long), the rank-weight algebra, volatility targeting, cost monotonicity, and the
portfolio arithmetic the verdict turns on. A lookahead bug or a sign bug would silently
manufacture a Sharpe, and this programme has already paid for that class of mistake twice.

Named ``multiasset_value_test.py`` (matching pytest's ``*_test.py`` pattern) rather than
``test_multiasset_value.py`` because a concurrent workflow owns the ``test_multiasset_*``
namespace on this branch.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.sleeves.multiasset_value import (
    BLOCKS,
    MIN_BLOCKS,
    MIN_PER_BLOCK,
    RATE_YIELD,
    REVERSAL_MONTHS,
    VALUE_UNIVERSE,
    ValueConfig,
    annual_sharpe,
    combined_sharpe_equal_risk,
    combined_sharpe_optimal,
    rank_weights,
    rates_value_score,
    reversal_score,
    run_value,
    value_scores,
    year_concentration,
)

N_MONTHS = 660


def _index(n: int = N_MONTHS) -> pd.DatetimeIndex:
    return pd.date_range("1970-01-31", periods=n, freq="ME")


def _spreads(idx: pd.DatetimeIndex, seed: int = 7) -> pd.DataFrame:
    """Synthetic term spreads: three distinct mean-reverting levels."""
    rng = np.random.default_rng(seed)
    out = {}
    for j, key in enumerate(RATE_YIELD):
        e = rng.normal(0.0, 0.0015, len(idx))
        a = np.zeros(len(idx))
        for t in range(1, len(idx)):
            a[t] = 0.97 * a[t - 1] + e[t]
        out[key] = 0.005 * (j + 1) + a
    return pd.DataFrame(out, index=idx)


def _noise(seed: int = 0, n: int = N_MONTHS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _index(n)
    return pd.DataFrame(
        rng.normal(0.0, 0.04, (n, len(VALUE_UNIVERSE))), index=idx, columns=VALUE_UNIVERSE
    )


def _cycles(phi: float = 0.95, seed: int = 3, n: int = N_MONTHS) -> pd.DataFrame:
    """Mean-reverting LEVELS: an OU log-price, so a long-horizon reversal really predicts.

    ``L(t) = phi*L(t-1) + e``, returns are ``L(t) - L(t-1) = -(1-phi)*L(t-1) + e``. The
    trailing 60-month return is ``L(t) - L(t-60)``, and with a ~13-month half-life
    ``L(t-60)`` is nearly independent of ``L(t)``, so the value score ``-(L(t)-L(t-60))``
    is a noisy read on ``-L(t)`` -- which is exactly what the next return is proportional
    to. This is the positive control the sleeve must pass.
    """
    rng = np.random.default_rng(seed)
    idx = _index(n)
    cols = {}
    for key in VALUE_UNIVERSE:
        e = rng.normal(0.0, 0.04, n)
        level = np.zeros(n)
        for t in range(1, n):
            level[t] = phi * level[t - 1] + e[t]
        cols[key] = np.diff(level, prepend=0.0)
    return pd.DataFrame(cols, index=idx)


def _rates_predictive(seed: int = 9, n: int = N_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A panel where the BOND value signal -- the term-spread deviation -- is the predictor.

    The reversal path cannot test the rates block, because bonds are scored on the spread
    and not on their own past returns. Here the spread deviation ``a(t)`` is an AR(1) and
    next month's bond excess return is ``0.012*a(t) + noise``, so a working sleeve must go
    long the bond whose curve is steepest relative to its own history.
    """
    rng = np.random.default_rng(seed)
    idx = _index(n)
    x = pd.DataFrame(
        rng.normal(0.0, 0.02, (n, len(VALUE_UNIVERSE))), index=idx, columns=VALUE_UNIVERSE
    )
    spreads = {}
    for j, key in enumerate(RATE_YIELD):
        e = rng.normal(0.0, 0.30, n)
        a = np.zeros(n)
        for t in range(1, n):
            a[t] = 0.90 * a[t - 1] + e[t]
        spreads[key] = 0.005 * (j + 1) + 0.002 * a
        x[key] = 0.012 * np.concatenate([[0.0], a[:-1]]) + rng.normal(0.0, 0.02, n)
    return x, pd.DataFrame(spreads, index=idx)


def _drifts(seed: int = 5, n: int = N_MONTHS) -> pd.DataFrame:
    """Permanent, heterogeneous drifts: a 5-year reversal signal must LOSE on these."""
    rng = np.random.default_rng(seed)
    idx = _index(n)
    cols = {}
    for j, key in enumerate(VALUE_UNIVERSE):
        mu = 0.001 * (j - len(VALUE_UNIVERSE) / 2.0)
        cols[key] = mu + rng.normal(0.0, 0.01, n)
    return pd.DataFrame(cols, index=idx)


# ── Pre-registration integrity ────────────────────────────────────────────────

def test_universe_is_the_preregistered_fourteen():
    assert len(VALUE_UNIVERSE) == 14
    assert set(VALUE_UNIVERSE) == set(sum((list(v) for v in BLOCKS.values()), []))
    assert len(set(VALUE_UNIVERSE)) == 14


def test_fx_and_natgas_are_absent_by_decision():
    for key in ("USDX", "EURUSD", "GBPUSD", "JPYUSD", "NATGAS_F", "DJIA", "SPY", "EEM"):
        assert key not in VALUE_UNIVERSE


def test_block_and_book_minimums_are_the_preregistered_ones():
    assert MIN_PER_BLOCK == 3
    assert MIN_BLOCKS == 2
    assert ValueConfig().min_blocks == 2          # PRIMARY never relaxes the two-block rule
    assert ValueConfig().skip_months == 0
    assert ValueConfig().uniform_rates is False
    assert REVERSAL_MONTHS == 60


# ── The signal ────────────────────────────────────────────────────────────────

def test_reversal_score_is_the_negative_five_year_log_return():
    idx = _index(80)
    r = pd.Series(np.full(80, 0.01), index=idx)
    v = reversal_score(pd.DataFrame({"A": r}))["A"]
    assert np.isnan(v.iloc[58])                                   # needs 60 observations
    assert v.iloc[59] == pytest.approx(-60.0 * math.log(1.01))
    assert v.iloc[59] < 0                                          # a riser is EXPENSIVE


def test_reversal_score_sign_is_cheap_is_high():
    idx = _index(80)
    up = pd.Series(np.full(80, 0.02), index=idx)
    down = pd.Series(np.full(80, -0.02), index=idx)
    v = reversal_score(pd.DataFrame({"UP": up, "DOWN": down}))
    assert v["DOWN"].iloc[-1] > 0 > v["UP"].iloc[-1]


def test_skip12_ignores_the_most_recent_twelve_months():
    idx = _index(90)
    base = np.full(90, 0.005)
    a = base.copy()
    b = base.copy()
    b[-12:] = 0.20                                                 # only the last year differs
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    plain = reversal_score(df).iloc[-1]
    skipped = reversal_score(df, skip=12).iloc[-1]
    assert plain["A"] != pytest.approx(plain["B"])
    assert skipped["A"] == pytest.approx(skipped["B"])
    # and it still demands a full 60 months of history
    assert np.isnan(reversal_score(df, skip=12).iloc[58]["A"])


def test_rates_value_score_is_deviation_from_its_own_expanding_mean():
    idx = _index(120)
    s = pd.Series(np.linspace(0.00, 0.02, 120), index=idx)
    v = rates_value_score(pd.DataFrame({"US5Y_TR": s}))["US5Y_TR"]
    assert np.isnan(v.iloc[58])
    k = 100
    assert v.iloc[k] == pytest.approx(s.iloc[k] - s.iloc[: k + 1].mean())
    assert v.iloc[-1] > 0                                          # a steepening curve is CHEAP


def test_rates_value_score_uses_no_future_information():
    idx = _index(120)
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.01, 0.003, 120), index=idx)
    a = rates_value_score(pd.DataFrame({"US5Y_TR": s}))["US5Y_TR"]
    s2 = s.copy()
    s2.iloc[-1] = 99.0
    b = rates_value_score(pd.DataFrame({"US5Y_TR": s2}))["US5Y_TR"]
    pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])


def test_rates_use_the_spread_signal_not_the_reversal_unless_d4():
    idx = _index(N_MONTHS)
    x = _noise()
    sp = _spreads(idx)
    primary = value_scores(x, sp)
    uniform = value_scores(x, sp, uniform_rates=True)
    key = "US10Y_TR"
    assert not np.allclose(
        primary[key].dropna().to_numpy()[-50:], uniform[key].dropna().to_numpy()[-50:]
    )
    # D4 must reproduce the plain reversal exactly for bonds
    pd.testing.assert_series_equal(uniform[key], reversal_score(x)[key])
    # and the equity block is untouched by the switch
    pd.testing.assert_series_equal(primary["SPX"], uniform["SPX"])


# ── Rank weighting ────────────────────────────────────────────────────────────

def test_rank_weights_are_dollar_neutral_and_unit_gross():
    idx = _index(3)
    v = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=idx, columns=list("ABCD"))
    e = pd.DataFrame(True, index=idx, columns=list("ABCD"))
    u = rank_weights(v, e)
    assert u.sum(axis=1).abs().max() < 1e-12
    assert u.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert u.loc[idx[0], "D"] > u.loc[idx[0], "C"] > 0 > u.loc[idx[0], "B"] > u.loc[idx[0], "A"]


def test_rank_weights_are_zero_below_the_block_minimum():
    idx = _index(2)
    v = pd.DataFrame([[1.0, 2.0, np.nan, np.nan]] * 2, index=idx, columns=list("ABCD"))
    e = v.notna()
    u = rank_weights(v, e)
    assert (u.abs().sum(axis=1) == 0).all()


def test_rank_weights_ignore_ineligible_instruments():
    idx = _index(1)
    v = pd.DataFrame([[1.0, 2.0, 3.0, 99.0]], index=idx, columns=list("ABCD"))
    e = pd.DataFrame([[True, True, True, False]], index=idx, columns=list("ABCD"))
    u = rank_weights(v, e)
    assert u.loc[idx[0], "D"] == 0.0
    assert u.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert u.loc[idx[0], "B"] == pytest.approx(0.0)                # the median leg is flat


# ── Causality ─────────────────────────────────────────────────────────────────

def test_positions_do_not_use_future_returns():
    idx = _index()
    x = _cycles()
    sp = _spreads(idx)
    base = run_value(x=x, spreads=sp)
    shocked = x.copy()
    shocked.iloc[-1] = shocked.iloc[-1] + 5.0
    after = run_value(x=shocked, spreads=sp)
    pd.testing.assert_frame_equal(base.weights, after.weights)


def test_book_scaler_is_causal():
    idx = _index()
    x = _noise(seed=11)
    sp = _spreads(idx)
    base = run_value(x=x, spreads=sp)
    cut = base.gross.index[len(base.gross) // 2]
    truncated = run_value(x=x.loc[:cut], spreads=sp.loc[:cut])
    common = base.gross.index.intersection(truncated.gross.index)
    assert len(common) > 50
    pd.testing.assert_series_equal(
        base.gross.loc[common], truncated.gross.loc[common], check_names=False
    )


def test_book_is_off_until_two_blocks_are_live():
    idx = _index()
    x = _noise(seed=2)
    # kill the rates and commodity blocks entirely
    x = x.copy()
    for key in list(RATE_YIELD) + list(BLOCKS["commodity"]):
        x[key] = np.nan
    sp = _spreads(idx)
    r = run_value(x=x, spreads=sp)
    assert r.gross.empty


# ── Does it find what it claims to find ───────────────────────────────────────

def test_iid_noise_produces_no_edge():
    sharpes = []
    for seed in range(6):
        x = _noise(seed=seed)
        r = run_value(x=x, spreads=_spreads(x.index, seed=seed))
        sharpes.append(annual_sharpe(r.net["10bps"]))
    assert abs(float(np.mean(sharpes))) < 0.40


_REVERSAL_ONLY = ValueConfig(name="TEST_REVERSAL_BLOCKS", blocks=("equity", "commodity"))


def test_long_horizon_reversal_is_found():
    """Scoped to the two blocks the reversal signal actually drives (prereg §3a)."""
    x = _cycles()
    r = run_value(_REVERSAL_ONLY, x=x, spreads=_spreads(x.index))
    assert annual_sharpe(r.gross) > 0.7


def test_rates_term_spread_signal_is_found():
    x, sp = _rates_predictive()
    r = run_value(ValueConfig(name="TEST_RATES", blocks=("rates",), min_blocks=1),
                  x=x, spreads=sp)
    assert annual_sharpe(r.gross) > 0.7


def test_permanent_drift_is_punished():
    x = _drifts()
    r = run_value(_REVERSAL_ONLY, x=x, spreads=_spreads(x.index))
    assert annual_sharpe(r.gross) < 0.0          # value shorts the winner and loses


def test_sign_randomisation_destroys_the_edge():
    x = _cycles()
    sp = _spreads(x.index)
    live = annual_sharpe(run_value(_REVERSAL_ONLY, x=x, spreads=sp).net["10bps"])
    controls = [
        annual_sharpe(
            run_value(
                ValueConfig(blocks=("equity", "commodity"), randomise_seed=s), x=x, spreads=sp
            ).net["10bps"]
        )
        for s in range(6)
    ]
    assert live > float(np.mean(controls)) + 2.0 * float(np.std(controls, ddof=1))


# ── Book mechanics ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("target", [0.10, 0.20, 0.40])
def test_volatility_targeting_lands_near_its_target(target):
    x = _noise(seed=4)
    r = run_value(x=x, spreads=_spreads(x.index), vol_target=target)
    realised = float(r.gross.std(ddof=1) * math.sqrt(12))
    assert 0.5 * target < realised < 1.8 * target


def test_gross_leverage_cap_is_respected():
    x = _noise(seed=6)
    r = run_value(x=x, spreads=_spreads(x.index), vol_target=0.40)
    assert r.gross_leverage.max() <= 10.0 + 1e-9


def test_costs_are_monotone_and_proportional():
    x = _cycles()
    r = run_value(x=x, spreads=_spreads(x.index))
    gap2 = (r.gross - r.net["2bps"]).sum()
    gap10 = (r.gross - r.net["10bps"]).sum()
    assert gap10 > gap2 > 0
    assert gap10 / gap2 == pytest.approx(5.0, rel=1e-9)


def test_benchmark_is_long_only_equal_weight_of_the_tradable_set():
    x = _cycles()
    r = run_value(x=x, spreads=_spreads(x.index))
    assert r.bench_gross.notna().all()
    # long-only equal weight of N names has an average pairwise-identical exposure: its
    # return must lie inside the cross-sectional range of the instruments each month.
    xz = x.reindex(r.bench_gross.index)
    lo = xz.min(axis=1)
    hi = xz.max(axis=1)
    inside = (r.bench_gross >= lo - 1e-12) & (r.bench_gross <= hi + 1e-12)
    assert bool(inside.all())


def test_strategy_is_close_to_dollar_neutral():
    x = _cycles()
    r = run_value(x=x, spreads=_spreads(x.index))
    # inverse-vol scaling breaks exact neutrality, but net exposure must stay small next
    # to gross leverage -- otherwise this is a directional book wearing a market-neutral label
    assert abs(float(r.net_exposure.mean())) < 0.25 * float(r.gross_leverage.mean())


# ── Portfolio arithmetic ──────────────────────────────────────────────────────

def test_combined_equal_risk_sharpe_matches_the_closed_form():
    assert combined_sharpe_equal_risk(0.4, 0.4, 1.0) == pytest.approx(0.4)
    assert combined_sharpe_equal_risk(0.4, 0.4, 0.0) == pytest.approx(0.4 * math.sqrt(2))
    assert combined_sharpe_equal_risk(0.4, 0.4, -0.5) == pytest.approx(0.4 * 2.0)
    assert combined_sharpe_equal_risk(0.4, 0.4, -0.5) > combined_sharpe_equal_risk(0.4, 0.4, 0.5)


def test_combined_optimal_sharpe_matches_the_closed_form():
    assert combined_sharpe_optimal(0.3, 0.4, 0.0) == pytest.approx(math.hypot(0.3, 0.4))
    assert combined_sharpe_optimal(0.4, 0.4, -0.5) > combined_sharpe_optimal(0.4, 0.4, 0.0)
    assert math.isnan(combined_sharpe_optimal(0.4, 0.4, 1.0))


def test_a_negative_correlation_beats_a_higher_sharpe_correlated_sleeve():
    """The premise of the whole sleeve, asserted as arithmetic rather than assumed."""
    weak_diversifier = combined_sharpe_equal_risk(0.60, 0.35, -0.30)
    strong_correlated = combined_sharpe_equal_risk(0.60, 0.55, 0.60)
    assert weak_diversifier > strong_correlated


def test_year_concentration_flags_a_dominant_year():
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    p = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    p.loc[:, "A"] = 0.001
    p.iloc[6, 0] = 1.0
    out = year_concentration(p)
    assert out["top_year"] == 2000
    assert out["top_year_share"] > 0.9
