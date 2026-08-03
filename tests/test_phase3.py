"""
Phase 3 Tests — Volatility, Covariance, Regime, and Crisis
==========================================================
Covers every Phase 3 test target from the build spec:

  - GJR-GARCH fallback on non-convergence (gamma forced to 0.0) + min-samples
  - HAR-RV OLS fit + raises on < 60 samples
  - Ensemble switching method field across sample-size ranges
  - vol_ratio_current edge cases (< 5 obs -> 1.0, never < 0.0)
  - rmt_denoise_cov fallbacks (T < 2n, n == 1) and trace preservation
  - HMM regime detect() valid strings, detect_with_probs() probabilities sum to 1
  - recommended_strategy_mix, regime_transition_penalty, infer_execution_regime
  - All 7 crisis detectors return tuple[bool, float] with float in [0, 1]
  - EWMA correlation path, regime-aware drawdown threshold
  - Composite severity level thresholds + defensive_mode gating
  - CrisisStatus.as_dict completeness, singletons, 5-minute cache + reset
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from strategies.volatility_model import (
    MIN_GJR_SAMPLES,
    MIN_HAR_SAMPLES,
    UNKNOWN_DAILY_VAR,
    _fit_garch_fallback,
    fit,
    fit_gjr_garch,
    fit_har_rv,
    forecast_vol,
    rmt_denoise_cov,
    vol_ratio_current,
)
from core import regime_engine as rgm
from core import crisis_manager as cm
from core.crisis_manager import CrisisLevel, level_from_severity


# ── Fixtures / deterministic data builders ──────────────────────────────────────

def _garch_returns(n: int = 400, seed: int = 0,
                   omega: float = 5e-6, alpha: float = 0.08,
                   beta: float = 0.90) -> np.ndarray:
    """Simulate a GARCH(1,1)-style daily return series (deterministic)."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    s2 = omega / max(1.0 - alpha - beta, 1e-3)
    for t in range(1, n):
        s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
        r[t] = rng.normal(0.0, math.sqrt(s2))
    return r


def _price_panel(seed: int = 7, days: int = 300, n_assets: int = 4,
                 switch: int | None = None) -> pd.DataFrame:
    """A multi-asset price panel; optionally calm before `switch`, stressed after."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=days)
    rets = np.zeros((days, n_assets))
    for t in range(days):
        vol = 0.006 if (switch is None or t < switch) else 0.03
        common = rng.normal(0.0, vol)
        for a in range(n_assets):
            rets[t, a] = 0.5 * common + 0.5 * rng.normal(0.0, vol)
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _correlated_returns(seed: int = 1, rows: int = 20, cols: int = 4,
                        common: float = 0.9) -> np.ndarray:
    """Returns matrix with a tunable common factor (high `common` -> high corr)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, 1))
    return common * base + (1.0 - common) * rng.standard_normal((rows, cols))


# ── 1. Volatility fitters ───────────────────────────────────────────────────────

class TestVolatilityFitters:

    def test_fit_gjr_garch_returns_required_keys(self):
        out = fit_gjr_garch(_garch_returns(400, seed=5).tolist())
        for key in ("omega", "alpha", "gamma", "beta", "nu"):
            assert key in out and isinstance(out[key], float)

    def test_fit_gjr_garch_min_samples_raises(self):
        with pytest.raises(ValueError, match="requires"):
            fit_gjr_garch([0.01] * (MIN_GJR_SAMPLES - 1))

    def test_garch_fallback_forces_gamma_zero(self):
        out = _fit_garch_fallback(_garch_returns(200, seed=4))
        assert out["gamma"] == 0.0
        assert out["method"] in ("garch_fallback", "rolling_std")

    def test_fit_gjr_garch_falls_back_on_non_convergence(self, monkeypatch):
        """When arch fails outright, the fitter degrades without crashing and
        emits gamma=0.0 (the symmetric/rolling fallback)."""
        import arch

        def boom(*args, **kwargs):
            raise RuntimeError("forced non-convergence")

        monkeypatch.setattr(arch, "arch_model", boom)
        out = fit_gjr_garch(_garch_returns(100, seed=3).tolist())
        assert out["gamma"] == 0.0
        assert out["method"] == "rolling_std"
        for key in ("omega", "alpha", "gamma", "beta", "nu"):
            assert key in out

    def test_fit_har_rv_returns_keys(self):
        out = fit_har_rv(_garch_returns(300, seed=6).tolist())
        for key in ("c", "beta_d", "beta_w", "beta_m", "r_squared"):
            assert key in out and math.isfinite(out[key])
        assert out["rv_forecast"] >= 0.0

    def test_fit_har_rv_min_samples_raises(self):
        with pytest.raises(ValueError, match="requires"):
            fit_har_rv(_garch_returns(MIN_HAR_SAMPLES - 1, seed=1).tolist())


# ── 2. Ensemble switching + forecast ────────────────────────────────────────────

class TestEnsembleSwitching:

    def setup_method(self):
        self.r = _garch_returns(400, seed=9)

    def test_method_ensemble_when_history_ge_60(self):
        assert fit(self.r.tolist())["method"] == "ensemble"

    def test_method_gjr_only_mid_range(self):
        assert fit(self.r[:45].tolist())["method"] == "gjr_only"

    def test_method_rolling_std_when_short(self):
        assert fit(self.r[:20].tolist())["method"] == "rolling_std"

    def test_too_little_history_is_risky_not_risk_free(self):
        """VREG-1: with fewer than two observations there is no variance to
        measure. Answering 0.0 said RISK-FREE, which a vol-targeting scaler
        reads as unlimited room and levers up to its cap on an instrument it
        knows nothing about. Fail safe to a wide unknown-vol prior instead."""
        for returns in ([], [0.01]):
            out = fit(returns)
            params = out["gjr_params"]
            assert params["cond_var_last"] == pytest.approx(UNKNOWN_DAILY_VAR)
            assert params["cond_var_last"] > 0.0        # never risk-free
        # and the conservative prior is genuinely wide: >= 25% annualised
        assert (UNKNOWN_DAILY_VAR * 252) ** 0.5 >= 0.25

    def test_garch_fallback_with_no_history_is_also_risky(self, monkeypatch):
        """The same guard on the other fallback path (_fit_garch_fallback)."""
        monkeypatch.setattr(
            "strategies.volatility_model.arch_model",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no arch")),
            raising=False,
        )
        out = _fit_garch_fallback(np.asarray([0.01], dtype=float))
        assert out["cond_var_last"] == pytest.approx(UNKNOWN_DAILY_VAR)
        assert out["cond_var_last"] > 0.0

    def test_forecast_vol_positive_for_each_method(self):
        ens = fit(self.r.tolist())
        gjr = fit(self.r[:45].tolist())
        roll = fit(self.r[:20].tolist())
        for f in (ens, gjr, roll):
            v = forecast_vol(f["gjr_params"], f["har_params"], horizon=1)
            assert math.isfinite(v) and v > 0.0

    def test_forecast_vol_horizon_gt5_uses_gjr_recursive(self):
        ens = fit(self.r.tolist())
        v = forecast_vol(ens["gjr_params"], ens["har_params"], horizon=20)
        assert math.isfinite(v) and v > 0.0


# ── 3. vol_ratio_current ────────────────────────────────────────────────────────

class TestVolRatio:

    def test_returns_one_when_insufficient_history(self):
        assert vol_ratio_current([0.01, -0.02, 0.0, 0.015]) == 1.0   # 4 < 5 obs

    def test_never_negative(self):
        r = _garch_returns(120, seed=11).tolist()
        assert vol_ratio_current(r) >= 0.0
        # A wild series must still produce a non-negative ratio
        wild = (np.random.default_rng(2).standard_normal(80) * 0.2).tolist()
        assert vol_ratio_current(wild) >= 0.0


# ── 4. rmt_denoise_cov ──────────────────────────────────────────────────────────

class TestRmtDenoise:

    def _sample_cov(self, n: int = 8, T: int = 200, seed: int = 3) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.cov(rng.standard_normal((n, T)))

    def test_normal_case_shape_symmetric_trace_preserved(self):
        cov = self._sample_cov(8, 200)
        d = rmt_denoise_cov(cov, 200)
        assert d.shape == (8, 8)
        assert np.allclose(d, d.T)
        assert math.isclose(float(np.trace(d)), float(np.trace(cov)), rel_tol=1e-9)

    def test_T_lt_2n_returns_original(self):
        cov = self._sample_cov(8, 200)
        out = rmt_denoise_cov(cov, 10)            # 10 < 2*8
        assert np.array_equal(out, cov)

    def test_n_eq_1_returns_unchanged(self):
        cov = np.array([[2.0]])
        assert rmt_denoise_cov(cov, 100).tolist() == [[2.0]]

    def test_non_psd_input_is_projected_psd(self):
        # Item 3: RMT denoising alone preserves negative eigenvalues (signal
        # eigenvalues are kept verbatim; a negative noise cluster averages to a
        # still-negative value). The PSD projection must guarantee a valid (PSD)
        # covariance feeds the mean-variance / CVaR optimiser.
        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.standard_normal((5, 5)))      # orthonormal basis
        eigs = np.array([0.05, 0.02, 0.01, 0.005, -0.03])     # one negative eigenvalue
        m = (q * eigs) @ q.T
        m = 0.5 * (m + m.T)
        assert np.linalg.eigvalsh(m).min() < 0                # precondition: non-PSD
        out = rmt_denoise_cov(m, 200)
        assert np.linalg.eigvalsh(out).min() >= -1e-12        # guaranteed PSD
        assert np.allclose(out, out.T)


# ── 5. Regime detection (HMM + heuristic) ────────────────────────────────────────

class TestRegimeDetect:

    def setup_method(self):
        rgm.reset_regime_engine()

    def test_detect_returns_valid_string(self):
        engine = rgm.RegimeEngine()
        label = engine.detect(_price_panel(seed=7, days=300, switch=150))
        assert label in rgm.VALID_REGIMES

    def test_detect_with_probs_sum_to_one(self):
        engine = rgm.RegimeEngine()
        label, probs = engine.detect_with_probs(_price_panel(seed=8, days=300, switch=150))
        assert label in rgm.VALID_REGIMES
        assert set(probs) == {"calm", "stressed"}
        assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
        assert all(0.0 <= v <= 1.0 for v in probs.values())

    def test_heuristic_path_short_panel_is_valid(self):
        # Too few rows for the HMM -> heuristic fallback still returns valid output
        engine = rgm.RegimeEngine()
        label, probs = engine.detect_with_probs(_price_panel(seed=4, days=70))
        assert label in rgm.VALID_REGIMES
        assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)

    def test_unmeasurable_volatility_is_stressed_not_calm(self):
        """VREG-2: with no measurable volatility the heuristic used to compute
        ratio = 1.0 and report a confidently BENIGN regime. Benign is the
        AGGRESSIVE answer - it widens the position cap, raises the vol target
        and gives views the largest tau - so an unmeasurable market must size
        down, not up."""
        engine = rgm.RegimeEngine()
        one_row = pd.DataFrame(
            {"AAA": [100.0], "BBB": [50.0]},
            index=pd.date_range("2026-01-01", periods=1, freq="B"),
        )
        flat = pd.DataFrame(
            {"AAA": [100.0] * 90, "BBB": [50.0] * 90},
            index=pd.date_range("2026-01-01", periods=90, freq="B"),
        )
        for panel, why in ((one_row, "too few returns"), (flat, "flat/stale feed")):
            label, probs = engine.detect_with_probs(panel)
            assert label == "high_vol", why
            assert probs["stressed"] > probs["calm"], why
            assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)

    def test_measurable_calm_market_is_still_allowed_to_be_calm(self):
        """Positive control for the guard above: a real, calm-but-moving market
        must NOT be forced into high_vol, or the fail-safe would just pin the
        book to its most defensive settings forever."""
        engine = rgm.RegimeEngine()
        label, probs = engine.detect_with_probs(_price_panel(seed=4, days=200))
        assert label != "high_vol"
        assert probs["stressed"] < rgm._UNKNOWN_STRESSED_PROB

    def test_module_singleton_identity_and_reset(self):
        e1 = rgm.get_regime_engine()
        e2 = rgm.get_regime_engine()
        assert e1 is e2
        rgm.reset_regime_engine()
        assert rgm.get_regime_engine() is not e1


# ── 6. recommended_strategy_mix ──────────────────────────────────────────────────

class TestStrategyMix:

    def test_sums_to_one_and_non_negative(self):
        mix = rgm.recommended_strategy_mix({"calm": 0.7, "stressed": 0.3})
        assert math.isclose(sum(mix.values()), 1.0, rel_tol=1e-9)
        assert all(w >= 0.0 for w in mix.values())
        assert set(mix) == set(rgm.SLEEVE_NAMES)

    def test_overlay_scales_and_renormalises(self):
        mix = rgm.recommended_strategy_mix(
            {"calm": 0.2, "stressed": 0.8}, overlays={"momentum": 0.0}
        )
        assert mix["momentum"] == 0.0
        assert math.isclose(sum(mix.values()), 1.0, rel_tol=1e-9)

    def test_stressed_tilts_to_defensive_sleeves(self):
        stressed = rgm.recommended_strategy_mix({"calm": 0.0, "stressed": 1.0})
        calm = rgm.recommended_strategy_mix({"calm": 1.0, "stressed": 0.0})
        assert stressed["mean_reversion"] > calm["mean_reversion"]
        assert stressed["vol_overlay"] > calm["vol_overlay"]
        assert stressed["momentum"] < calm["momentum"]


# ── 7. regime_transition_penalty ─────────────────────────────────────────────────

class TestTransitionPenalty:

    def test_same_regime_no_penalty(self):
        assert rgm.regime_transition_penalty("trending", "trending") == 1.0

    def test_change_is_penalised(self):
        assert rgm.regime_transition_penalty("trending", "mean_reverting") > 1.0

    def test_high_vol_transition_is_highest(self):
        into_hv = rgm.regime_transition_penalty("mean_reverting", "high_vol")
        ordinary = rgm.regime_transition_penalty("trending", "mean_reverting")
        assert into_hv > ordinary


# ── 8. infer_execution_regime boundaries ─────────────────────────────────────────

class TestExecutionRegime:

    @pytest.mark.parametrize(
        "spread, vol, adv, mins, expected",
        [
            (5.0, 1.0, 0.0, 120.0, "normal_exec"),
            (10.0, 1.0, 0.0, 120.0, "cautious_exec"),   # spread lower band
            (25.0, 1.0, 0.0, 120.0, "cautious_exec"),   # spread upper band (inclusive)
            (26.0, 1.0, 0.0, 120.0, "stressed_exec"),   # spread > 25
            (5.0, 1.5, 0.0, 120.0, "cautious_exec"),    # vol lower band
            (5.0, 2.5, 0.0, 120.0, "cautious_exec"),    # vol upper band (inclusive)
            (5.0, 2.6, 0.0, 120.0, "stressed_exec"),    # vol > 2.5
            (5.0, 1.0, 0.0, 60.0, "cautious_exec"),     # <= 60 min to close
            (5.0, 1.0, 0.0, 61.0, "normal_exec"),       # > 60 min to close
            (5.0, 1.0, 0.0, 14.0, "stressed_exec"),     # < 15 min to close
            (5.0, 1.0, 0.06, 120.0, "cautious_exec"),   # adv escalation
            (5.0, 1.0, 0.11, 120.0, "stressed_exec"),   # adv > 0.10
        ],
    )
    def test_boundaries(self, spread, vol, adv, mins, expected):
        assert rgm.infer_execution_regime(spread, vol, adv, mins) == expected


# ── 9. Crisis detectors ──────────────────────────────────────────────────────────

class TestCrisisDetectors:

    def setup_method(self):
        self.m = cm.CrisisManager()

    def test_all_seven_return_bool_float_in_unit_interval(self):
        results = [
            self.m._detect_correlation_spike(_correlated_returns(common=0.9)),
            self.m._detect_vol_explosion(_garch_returns(80, seed=2).tolist()),
            self.m._detect_drawdown_acceleration(np.array([100.0, 110.0, 90.0])),
            self.m._detect_breadth_collapse(np.array([-1.0, -1.0, 1.0])),
            self.m._detect_liquidity_stress(30.0, 0.1),
            self.m._detect_gap_risk(np.array([0.01, -0.05])),
            self.m._detect_event_risk(0.5),
        ]
        for fired, score in results:
            assert isinstance(fired, bool)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_correlation_spike_high_vs_low(self):
        hi_fired, hi_score = self.m._detect_correlation_spike(
            _correlated_returns(seed=1, common=0.95)
        )
        lo_fired, lo_score = self.m._detect_correlation_spike(
            _correlated_returns(seed=1, common=0.0)
        )
        assert hi_fired is True and hi_score > lo_score
        assert lo_fired is False

    def test_correlation_spike_needs_two_assets_and_ten_rows(self):
        assert self.m._detect_correlation_spike(np.ones((20, 1))) == (False, 0.0)
        assert self.m._detect_correlation_spike(np.ones((5, 4))) == (False, 0.0)
        assert self.m._detect_correlation_spike(None) == (False, 0.0)

    def test_drawdown_threshold_is_regime_aware(self):
        # dd = (110 - 105.6) / 110 = 0.04 -> fires for mean_reverting (0.03), not high_vol (0.075)
        values = np.array([100.0, 110.0, 105.6])
        mr_fired, mr_score = self.m._detect_drawdown_acceleration(values, "mean_reverting")
        hv_fired, hv_score = self.m._detect_drawdown_acceleration(values, "high_vol")
        assert mr_fired is True
        assert hv_fired is False
        assert mr_score > hv_score

    def test_liquidity_boundary_strict(self):
        assert self.m._detect_liquidity_stress(25.0, 0.05)[0] is False   # not > thresholds
        assert self.m._detect_liquidity_stress(26.0, 0.0)[0] is True
        assert self.m._detect_liquidity_stress(0.0, 0.06)[0] is True

    def test_gap_boundary(self):
        assert self.m._detect_gap_risk(np.array([0.02, -0.03]))[0] is False  # not > 0.03
        assert self.m._detect_gap_risk(np.array([0.031]))[0] is True

    def test_event_boundary(self):
        assert self.m._detect_event_risk(None) == (False, 0.0)
        assert self.m._detect_event_risk(3.0)[0] is True
        assert self.m._detect_event_risk(0.5) == (True, 1.0)
        assert self.m._detect_event_risk(8.0)[0] is False

    def test_breadth_boundary(self):
        # 70% losing fires; below does not
        seven_losing = np.array([-1.0] * 7 + [1.0] * 3)
        six_losing = np.array([-1.0] * 6 + [1.0] * 4)
        assert self.m._detect_breadth_collapse(seven_losing)[0] is True
        assert self.m._detect_breadth_collapse(six_losing)[0] is False


# ── 10. Crisis composite + status + caching ──────────────────────────────────────

class TestCrisisComposite:

    def test_level_mapping_boundaries(self):
        assert level_from_severity(0.19) == CrisisLevel.NORMAL
        assert level_from_severity(0.20) == CrisisLevel.ELEVATED
        assert level_from_severity(0.49) == CrisisLevel.ELEVATED
        assert level_from_severity(0.50) == CrisisLevel.CRISIS
        assert level_from_severity(0.74) == CrisisLevel.CRISIS
        assert level_from_severity(0.75) == CrisisLevel.CRITICAL

    def test_crisis_scalars_graduated_bands(self):
        # Item 7: continuous severity -> (vol_target_scalar, cvar_limit_scalar)
        # per upgrade-spec P4. Tighter (smaller) as severity rises; monotone.
        assert cm.crisis_scalars(0.10) == (1.0, 1.0)      # Normal
        assert cm.crisis_scalars(0.45) == (0.80, 0.85)    # Elevated
        assert cm.crisis_scalars(0.70) == (0.60, 0.65)    # Defensive
        assert cm.crisis_scalars(0.90) == (0.50, 0.50)    # Crisis
        # band boundaries fall into the upper (tighter) band
        assert cm.crisis_scalars(0.35) == (0.80, 0.85)
        assert cm.crisis_scalars(0.60) == (0.60, 0.65)
        assert cm.crisis_scalars(0.80) == (0.50, 0.50)
        # clipped at the ends
        assert cm.crisis_scalars(-1.0) == (1.0, 1.0)
        assert cm.crisis_scalars(2.0) == (0.50, 0.50)

    def test_baseline_all_none_is_normal(self):
        status = cm.CrisisManager().assess(use_cache=False)
        assert status.level == CrisisLevel.NORMAL
        assert status.defensive_mode is False
        assert status.severity_score == 0.0
        assert status.signals_fired == []

    def test_single_correlation_fire_is_elevated_not_defensive(self):
        # s_corr ~ 1.0 -> S ~ 0.22 -> ELEVATED, defensive False
        status = cm.CrisisManager().assess(
            returns_matrix=_correlated_returns(seed=1, common=0.97), use_cache=False
        )
        assert status.level == CrisisLevel.ELEVATED
        assert status.defensive_mode is False

    def test_crisis_combination_is_defensive(self):
        # corr(0.22) + drawdown(0.16) + breadth(0.15) = 0.53 -> CRISIS
        status = cm.CrisisManager().assess(
            returns_matrix=_correlated_returns(seed=2, common=0.97),
            portfolio_values=np.array([100.0, 120.0, 60.0]),     # deep drawdown
            position_pnls=np.array([-1.0] * 9 + [1.0]),          # 90% losing
            current_regime="mean_reverting",
            use_cache=False,
        )
        assert status.level in (CrisisLevel.CRISIS, CrisisLevel.CRITICAL)
        assert status.defensive_mode is True

    def test_extreme_inputs_are_critical(self):
        status = cm.CrisisManager().assess(
            returns_matrix=_correlated_returns(seed=3, common=0.98),
            portfolio_values=np.array([100.0, 120.0, 55.0]),
            position_pnls=np.array([-1.0] * 10),
            spread_bps=90.0, adv_ratio=0.3,
            overnight_gaps=np.array([0.12]),
            hours_to_event=0.2,
            current_regime="mean_reverting",
            use_cache=False,
        )
        assert status.level == CrisisLevel.CRITICAL
        assert status.defensive_mode is True

    def test_as_dict_contains_all_fields(self):
        status = cm.CrisisManager().assess(
            spread_bps=90.0, overnight_gaps=np.array([0.1]), hours_to_event=0.5,
            use_cache=False,
        )
        d = status.as_dict()
        for key in (
            "level", "defensive_mode", "signals_fired", "signal_values",
            "severity_score", "liquidity_stress_score", "gap_risk_score",
            "event_risk_score", "timestamp",
        ):
            assert key in d
        assert isinstance(d["level"], str)
        assert d["liquidity_stress_score"] == status.signal_values["liquidity_stress"]

    def test_cache_returns_same_object_until_reset(self):
        m = cm.CrisisManager()
        first = m.assess(spread_bps=90.0, use_cache=True)
        # Different inputs, but the 5-minute cache returns the same object
        cached = m.assess(spread_bps=0.0, use_cache=True)
        assert cached is first
        m.reset_cache()
        recomputed = m.assess(spread_bps=0.0, use_cache=True)
        assert recomputed is not first

    def test_is_defensive_and_current_level_track_last_assessment(self):
        m = cm.CrisisManager()
        assert m.is_defensive() is False
        assert m.current_level() == CrisisLevel.NORMAL
        m.assess(
            returns_matrix=_correlated_returns(seed=5, common=0.98),
            portfolio_values=np.array([100.0, 120.0, 55.0]),
            position_pnls=np.array([-1.0] * 10),
            spread_bps=90.0, overnight_gaps=np.array([0.12]), hours_to_event=0.2,
            use_cache=False,
        )
        assert m.is_defensive() is True
        assert m.current_level() in (CrisisLevel.CRISIS, CrisisLevel.CRITICAL)

    def test_module_singleton_identity(self):
        cm.reset_crisis_manager()
        a = cm.get_crisis_manager()
        assert a is cm.get_crisis_manager()
        cm.reset_crisis_manager()
        assert cm.get_crisis_manager() is not a
