"""
Phase 4 Tests — Signal Engines and Alpha-Factory Health
=======================================================
Covers every Phase 4 test target from the build spec:

  - Each sleeve produces valid SignalOutput instances for all fields
  - Momentum: uptrend -> BUY, downtrend -> SELL
  - Mean reversion: oversold -> BUY, overbought -> SELL
  - Stat-arb: a reverting, idiosyncratically cheap residual -> BUY (Avellaneda-Lee)
  - Black-Scholes: put-call parity, monotonicity in sigma, intrinsic at expiry,
    implied-vol round-trip (incl. deep OTM), no-arbitrage guards
  - atm_put_cost surfaces cost with crisis uplift and never auto-executes
  - Alpha-factory health: stability_score < 0.40 is downweighted; a failed
    selection_rule disables the sleeve
  - factor_decay_profile() returns IC at all 5 horizons
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from research.alpha_factory import (
    STABILITY_FLOOR,
    UNVALIDATED_SLEEVE_WEIGHT,
    SignalOutput,
    apply_signal_health,
    factor_decay_profile,
    sleeve_health_weight,
)
from research.validation import ValidationResult
from strategies import black_scholes as bs
from strategies import carry, mean_reversion, momentum, sentiment, stat_arb
from strategies import volatility_overlay as vol_overlay


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _valid_signal(s: SignalOutput) -> bool:
    return (
        isinstance(s, SignalOutput)
        and s.direction in {"BUY", "SELL", "FLAT"}
        and -1.0 <= s.raw_score <= 1.0
        and 0.0 <= s.confidence_proxy <= 1.0
        and s.expected_horizon > 0
        and s.decay_half_life > 0
        and isinstance(s.asof_timestamp, datetime)
    )


def _valid(signals: list[SignalOutput]) -> bool:
    return len(signals) > 0 and all(_valid_signal(s) for s in signals)


class TestVolatilityOverlay:

    def _prices(self, segments, seed: int = 0) -> pd.DataFrame:
        # segments: list of (daily_std, n_days); build a single-symbol price series.
        rng = np.random.default_rng(seed)
        rets = np.concatenate([rng.normal(0.0, std, n) for std, n in segments])
        idx = pd.bdate_range("2022-01-01", periods=len(rets))
        return pd.DataFrame({"X": 100.0 * np.exp(np.cumsum(rets))}, index=idx)

    def test_valid_signals(self):
        assert _valid(vol_overlay.generate_signals(self._prices([(0.01, 90)])))

    def test_vol_expansion_is_defensive(self):
        # calm 60d then volatile 20d -> recent vol > baseline -> defensive (raw < 0)
        sig = vol_overlay.generate_signals(self._prices([(0.005, 60), (0.03, 20)], seed=1))[0]
        assert sig.raw_score < 0.0
        assert sig.direction in ("SELL", "FLAT")

    def test_vol_compression_is_constructive(self):
        # volatile 60d then calm 20d -> recent vol < baseline -> constructive (raw > 0)
        sig = vol_overlay.generate_signals(self._prices([(0.03, 60), (0.005, 20)], seed=2))[0]
        assert sig.raw_score > 0.0
        assert sig.direction in ("BUY", "FLAT")

    def test_short_history_is_flat(self):
        sig = vol_overlay.generate_signals(self._prices([(0.01, 10)]))[0]
        assert sig.direction == "FLAT" and sig.confidence_proxy == 0.0

    def test_deterministic(self):
        prices = self._prices([(0.005, 60), (0.03, 20)], seed=1)
        a = vol_overlay.generate_signals(prices)[0]
        b = vol_overlay.generate_signals(prices)[0]
        assert a.raw_score == b.raw_score and a.confidence_proxy == b.confidence_proxy


class TestCarrySleeve:

    def _prices(self, symbols) -> pd.DataFrame:
        idx = pd.bdate_range("2022-01-01", periods=70)
        # Carry is cross-sectional on dividend yields; price level is irrelevant here.
        return pd.DataFrame({s: np.full(70, 100.0) for s in symbols}, index=idx)

    def test_valid_signals(self):
        sigs = carry.generate_signals(self._prices(["A", "B", "C"]),
                                      dividend_yields={"A": 0.04, "B": 0.02, "C": 0.0})
        assert _valid(sigs)

    def test_high_yield_is_buy_low_is_sell(self):
        sigs = carry.generate_signals(self._prices(["A", "B", "C", "D"]),
                                      dividend_yields={"A": 0.04, "B": 0.02, "C": 0.0, "D": 0.0})
        by = {s.symbol: s for s in sigs}
        assert by["A"].raw_score > 0.0 and by["A"].direction == "BUY"   # high carry
        assert by["C"].raw_score < 0.0                                  # below-average carry

    def test_no_yields_is_flat(self):
        sigs = carry.generate_signals(self._prices(["A", "B"]), dividend_yields=None)
        assert all(s.direction == "FLAT" for s in sigs)

    def test_no_dispersion_is_flat(self):
        sigs = carry.generate_signals(self._prices(["A", "B", "C"]),
                                      dividend_yields={"A": 0.02, "B": 0.02, "C": 0.02})
        assert all(s.direction == "FLAT" for s in sigs)

    def test_deterministic(self):
        prices, ys = self._prices(["A", "B", "C"]), {"A": 0.04, "B": 0.02, "C": 0.0}
        a = [s.raw_score for s in carry.generate_signals(prices, dividend_yields=ys)]
        b = [s.raw_score for s in carry.generate_signals(prices, dividend_yields=ys)]
        assert a == b


def _dir(signals: list[SignalOutput], symbol: str) -> str:
    return next(s.direction for s in signals if s.symbol == symbol)


def _trend_panel(seed: int = 0, n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    up = 100.0 * np.exp(np.cumsum(rng.normal(0.0010, 0.010, n)))
    down = 100.0 * np.exp(np.cumsum(rng.normal(-0.0010, 0.010, n)))
    return pd.DataFrame({"UP": up, "DOWN": down}, index=idx)


def _stat_arb_panel(seed: int = 3, days: int = 160) -> pd.DataFrame:
    """4 factor-driven names + one ('REV') with a reverting residual at a low."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0003, 0.010, days)
    cols = {nm: 100.0 * np.exp(np.cumsum(f + rng.normal(0, 0.002, days)))
            for nm in ("A", "B", "C", "D")}
    ou = np.zeros(days)
    for t in range(1, days):
        ou[t] = 0.85 * ou[t - 1] + rng.normal(0, 0.01)
    ou[-6:] -= 0.05   # force a current negative idiosyncratic deviation
    cols["REV"] = 100.0 * np.exp(np.cumsum(f + np.diff(ou, prepend=0.0)))
    return pd.DataFrame(cols, index=pd.bdate_range("2023-01-01", periods=days))


# ── 1. Momentum sleeve ──────────────────────────────────────────────────────────

class TestMomentum:

    def test_signals_are_valid_and_one_per_symbol(self):
        prices = _trend_panel()
        signals = momentum.generate_signals(prices)
        assert _valid(signals)
        assert len(signals) == prices.shape[1]
        assert all(s.sleeve_name == "momentum" for s in signals)

    def test_uptrend_is_buy_downtrend_is_sell(self):
        signals = momentum.generate_signals(_trend_panel())
        assert _dir(signals, "UP") == "BUY"
        assert _dir(signals, "DOWN") == "SELL"

    def test_short_history_is_flat(self):
        idx = pd.bdate_range("2023-01-01", periods=10)
        prices = pd.DataFrame({"X": np.linspace(100, 110, 10)}, index=idx)
        signals = momentum.generate_signals(prices)
        assert signals[0].direction == "FLAT"
        assert signals[0].confidence_proxy == 0.0

    def test_asof_defaults_to_last_index(self):
        prices = _trend_panel()
        signals = momentum.generate_signals(prices)
        assert signals[0].asof_timestamp == prices.index[-1].to_pydatetime()


# ── 2. Mean-reversion sleeve ─────────────────────────────────────────────────────

class TestMeanReversion:

    def _panel(self, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2023-01-01", periods=60)
        oversold = 100.0 + rng.normal(0, 0.5, 60)
        oversold[-1] -= 3.0          # push the last price well below its mean
        overbought = 100.0 + rng.normal(0, 0.5, 60)
        overbought[-1] += 3.0
        return pd.DataFrame({"OVERSOLD": oversold, "OVERBOUGHT": overbought}, index=idx)

    def test_signals_are_valid(self):
        assert _valid(mean_reversion.generate_signals(self._panel()))

    def test_oversold_is_buy_overbought_is_sell(self):
        signals = mean_reversion.generate_signals(self._panel())
        assert _dir(signals, "OVERSOLD") == "BUY"
        assert _dir(signals, "OVERBOUGHT") == "SELL"

    def test_flat_series_is_flat(self):
        idx = pd.bdate_range("2023-01-01", periods=60)
        prices = pd.DataFrame({"FLATLINE": np.full(60, 100.0)}, index=idx)
        signals = mean_reversion.generate_signals(prices)
        assert signals[0].direction == "FLAT"


# ── 3. Stat-arb sleeve ───────────────────────────────────────────────────────────

class TestStatArb:

    def test_signals_are_valid(self):
        assert _valid(stat_arb.generate_signals(_stat_arb_panel()))

    def test_reverting_cheap_residual_is_buy(self):
        signals = stat_arb.generate_signals(_stat_arb_panel())
        assert _dir(signals, "REV") == "BUY"

    def test_single_symbol_is_flat(self):
        idx = pd.bdate_range("2023-01-01", periods=160)
        prices = pd.DataFrame({"ONLY": 100.0 + np.arange(160) * 0.1}, index=idx)
        signals = stat_arb.generate_signals(prices)
        assert len(signals) == 1
        assert signals[0].direction == "FLAT"

    def test_ou_s_score_negative_for_low_reverting_residual(self):
        rng = np.random.default_rng(3)
        days = 80
        f = rng.normal(0, 0.01, days)
        ou = np.zeros(days)
        for t in range(1, days):
            ou[t] = 0.8 * ou[t - 1] + rng.normal(0, 0.01)
        ou[-5:] -= 0.05
        ri = f + np.diff(ou, prepend=0.0)
        s_score, b = stat_arb._ou_s_score(ri, f)
        assert s_score is not None
        assert s_score < 0.0
        assert 0.0 < b < 1.0


# ── 4. Black-Scholes pricing ─────────────────────────────────────────────────────

class TestBlackScholes:

    S, K, T, r, sigma, q = 100.0, 100.0, 0.5, 0.02, 0.25, 0.01

    def test_put_call_parity(self):
        c = bs.bs_call_price(self.S, self.K, self.T, self.r, self.sigma, self.q)
        p = bs.bs_put_price(self.S, self.K, self.T, self.r, self.sigma, self.q)
        rhs = self.S * math.exp(-self.q * self.T) - self.K * math.exp(-self.r * self.T)
        assert math.isclose(c - p, rhs, abs_tol=1e-9)

    def test_put_monotonic_in_sigma(self):
        p_lo = bs.bs_put_price(self.S, self.K, self.T, self.r, 0.10)
        p_mid = bs.bs_put_price(self.S, self.K, self.T, self.r, 0.30)
        p_hi = bs.bs_put_price(self.S, self.K, self.T, self.r, 0.60)
        assert p_lo < p_mid < p_hi

    def test_intrinsic_value_at_expiry(self):
        assert math.isclose(bs.bs_put_price(95, 100, 0.0, self.r, self.sigma), 5.0)
        assert math.isclose(bs.bs_call_price(105, 100, 0.0, self.r, self.sigma), 5.0)

    def test_vega_positive_atm_and_zero_at_expiry(self):
        assert bs.bs_vega(self.S, self.K, self.T, self.r, self.sigma, self.q) > 0.0
        assert bs.bs_vega(self.S, self.K, 0.0, self.r, self.sigma, self.q) == 0.0

    def test_implied_vol_round_trip_atm(self):
        price = bs.bs_put_price(self.S, self.K, self.T, self.r, 0.2734, self.q)
        iv = bs.implied_vol(price, self.S, self.K, self.T, self.r, "put", self.q)
        assert math.isclose(iv, 0.2734, abs_tol=1e-3)

    def test_implied_vol_round_trip_deep_otm(self):
        price = bs.bs_put_price(100, 70, self.T, self.r, 0.45, self.q)
        iv = bs.implied_vol(price, 100, 70, self.T, self.r, "put", self.q)
        assert math.isclose(iv, 0.45, abs_tol=1e-3)

    def test_implied_vol_below_intrinsic_is_nan(self):
        assert math.isnan(bs.implied_vol(0.01, 90, 100, self.T, self.r, "put"))

    def test_implied_vol_above_upper_bound_is_nan(self):
        # A put cannot be worth more than the discounted strike.
        too_high = 100 * math.exp(-self.r * self.T) + 1.0
        assert math.isnan(bs.implied_vol(too_high, 100, 100, self.T, self.r, "put"))


# ── 5. Tail-hedge (ATM put) cost surface ─────────────────────────────────────────

class TestAtmPutCost:

    def test_surfaces_cost_and_never_auto_executes(self):
        cost = bs.atm_put_cost(100.0, 0.25, 0.25, r=0.02)
        for key in ("strike", "put_price", "cost_pct_of_spot", "auto_execute", "note"):
            assert key in cost
        assert cost["strike"] == 100.0
        assert cost["auto_execute"] is False
        assert cost["put_price"] > 0.0
        assert math.isclose(cost["cost_pct_of_spot"], cost["put_price"] / 100.0)

    def test_crisis_uplift_increases_estimate(self):
        normal = bs.atm_put_cost(100.0, 0.25, 0.25, r=0.02, crisis=False)
        crisis = bs.atm_put_cost(100.0, 0.25, 0.25, r=0.02, crisis=True)
        assert crisis["crisis_adjusted_estimate"] > crisis["put_price"]
        assert normal["crisis_adjusted_estimate"] == normal["put_price"]
        assert crisis["auto_execute"] is False


# ── 6. Alpha-factory signal health ───────────────────────────────────────────────

class TestSignalHealth:

    def _sig(self, conf: float, direction: str = "BUY", raw: float = 0.6) -> SignalOutput:
        return SignalOutput(
            symbol="AAPL", direction=direction, raw_score=raw,
            expected_horizon=5, decay_half_life=3, confidence_proxy=conf,
            sleeve_name="momentum",
            asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        )

    def _passing(self) -> ValidationResult:
        return ValidationResult(
            mean_ic=0.05, mean_rank_ic=0.04, sharpe_net=1.2, turnover=0.02,
            hit_rate=0.55, max_drawdown=-0.05, pbo_proxy=0.15,
            deflated_sharpe_proxy=0.50, cost_drag_bps=5.0, stability_score=0.70,
            deflated_sharpe_ratio=0.99,
            regime_breakdown={"trending": {"sharpe": 0.9}}, leakage_flags=[],
        )

    def _failing(self) -> ValidationResult:
        return ValidationResult(
            mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.1, turnover=0.02,
            hit_rate=0.4, max_drawdown=-0.2, pbo_proxy=0.6,
            deflated_sharpe_proxy=0.0, cost_drag_bps=5.0, stability_score=0.1,
            regime_breakdown={}, leakage_flags=["LEAK"],
        )

    def test_validated_sleeve_full_at_or_above_floor(self):
        assert sleeve_health_weight(0.70, self._passing()) == 1.0
        assert sleeve_health_weight(STABILITY_FLOOR, self._passing()) == 1.0

    def test_validated_sleeve_ramps_below_floor(self):
        assert sleeve_health_weight(0.20, self._passing()) == pytest.approx(0.5)
        assert sleeve_health_weight(0.0, self._passing()) == 0.0

    def test_failed_validation_disables(self):
        assert sleeve_health_weight(0.70, self._failing()) == 0.0
        assert sleeve_health_weight(0.70, self._passing()) == 1.0

    def test_unvalidated_sleeve_is_default_denied(self):
        # SIGNALS-5: an un-validated sleeve (no ValidationResult) is capped at the
        # default-deny floor regardless of this-cycle confidence — never full weight.
        assert sleeve_health_weight(0.70) == UNVALIDATED_SLEEVE_WEIGHT
        assert sleeve_health_weight(0.99) == UNVALIDATED_SLEEVE_WEIGHT
        assert sleeve_health_weight(0.0) == UNVALIDATED_SLEEVE_WEIGHT
        # the LIVE posture (engine passes 0.0) disables it entirely
        assert sleeve_health_weight(0.99, unvalidated_weight=0.0) == 0.0

    def test_apply_downweights_confidence_below_floor(self):
        signals = [self._sig(0.8), self._sig(0.6, "SELL", -0.5)]
        out = apply_signal_health(signals, 0.20, self._passing())  # validated → weight 0.5
        assert [round(s.confidence_proxy, 3) for s in out] == [0.4, 0.3]
        # Directions and raw scores are preserved when merely downweighted
        assert [s.direction for s in out] == ["BUY", "SELL"]

    def test_apply_default_denies_unvalidated_sleeve(self):
        signals = [self._sig(0.8)]
        out = apply_signal_health(signals, 0.99)  # un-validated → floor, not full weight
        assert out[0].confidence_proxy == pytest.approx(0.8 * UNVALIDATED_SLEEVE_WEIGHT)

    def test_apply_disables_to_flat_on_failed_gate(self):
        signals = [self._sig(0.8), self._sig(0.6, "SELL", -0.5)]
        out = apply_signal_health(signals, 0.70, self._failing())
        assert all(s.direction == "FLAT" for s in out)
        assert all(s.raw_score == 0.0 and s.confidence_proxy == 0.0 for s in out)

    def test_inputs_are_not_mutated(self):
        signals = [self._sig(0.8)]
        apply_signal_health(signals, 0.20, self._passing())
        assert signals[0].confidence_proxy == 0.8


# ── 7. factor_decay_profile ──────────────────────────────────────────────────────

class TestFactorDecayProfile:

    def test_returns_all_five_horizons(self):
        idx = pd.bdate_range("2022-01-01", periods=120)
        factor = pd.Series(np.random.default_rng(0).standard_normal(120), index=idx)
        returns = pd.DataFrame(
            {"r": np.random.default_rng(1).standard_normal(120)}, index=idx
        )
        profile = factor_decay_profile(factor, returns)
        assert sorted(profile.keys()) == [1, 3, 5, 10, 20]
        assert all(isinstance(v, float) for v in profile.values())

    def test_measures_real_horizon_decay_not_identical_ic(self):
        # SIGNALS-3: a factor equal to the NEXT-period return (a perfect 1-step predictor) must
        # show the STRONGEST IC at horizon 1 and decay at longer horizons — NOT the same IC at
        # every horizon (the old bug returned the identical contemporaneous IC for all horizons).
        idx = pd.bdate_range("2022-01-01", periods=200)
        r = pd.Series(np.random.default_rng(0).standard_normal(200) * 0.01, index=idx)
        factor = r.shift(-1).fillna(0.0)                      # factor[t] = the 1-period forward return
        profile = factor_decay_profile(factor, pd.DataFrame({"r": r}))
        assert profile[1] > 0.5                               # strong at h=1
        assert profile[1] > profile[20]                       # decays at longer horizons
        assert len({round(v, 3) for v in profile.values()}) > 1   # horizons NOT all identical


class TestSentimentSleeve:

    def _prices(self, symbols=("AAPL", "MSFT", "GOOG"), n=60, seed=3):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        return pd.DataFrame(
            {s: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for s in symbols}, index=idx
        )

    def test_no_scores_means_all_flat(self):
        sigs = sentiment.generate_signals(self._prices())
        assert len(sigs) == 3
        assert all(s.direction == "FLAT" and s.raw_score == 0.0 for s in sigs)

    def test_positive_sentiment_buys_negative_sells(self):
        scores = {"AAPL": 0.8, "MSFT": -0.7, "GOOG": 0.0}
        sigs = sentiment.generate_signals(self._prices(), sentiment_scores=scores)
        by = {s.symbol: s for s in sigs}
        assert by["AAPL"].direction == "BUY" and by["AAPL"].raw_score == pytest.approx(0.8)
        assert by["MSFT"].direction == "SELL" and by["MSFT"].raw_score == pytest.approx(-0.7)
        assert by["GOOG"].direction == "FLAT"

    def test_deadband_suppresses_weak_sentiment(self):
        scores = {"AAPL": 0.05, "MSFT": -0.05, "GOOG": 0.0}
        sigs = sentiment.generate_signals(self._prices(), sentiment_scores=scores)
        assert all(s.direction == "FLAT" for s in sigs)
        assert all(s.confidence_proxy <= 0.2 for s in sigs)

    def test_signal_outputs_are_valid_and_fast_decaying(self):
        asof = datetime(2025, 10, 28, 14, 0, tzinfo=timezone.utc)
        sigs = sentiment.generate_signals(
            self._prices(), asof_timestamp=asof, sentiment_scores={"AAPL": 0.9, "MSFT": -0.9, "GOOG": 0.1}
        )
        for s in sigs:
            assert s.sleeve_name == "sentiment"
            assert s.asof_timestamp == asof
            assert -1.0 <= s.raw_score <= 1.0
            assert 0.0 <= s.confidence_proxy <= 1.0
            assert s.expected_horizon <= 5          # news alpha is fast
            assert s.decay_half_life <= 5

    def test_deterministic(self):
        scores = {"AAPL": 0.4, "MSFT": -0.2, "GOOG": 0.0}
        a = sentiment.generate_signals(self._prices(), sentiment_scores=scores)
        b = sentiment.generate_signals(self._prices(), sentiment_scores=scores)
        assert [(s.symbol, s.direction, s.raw_score) for s in a] == \
               [(s.symbol, s.direction, s.raw_score) for s in b]
