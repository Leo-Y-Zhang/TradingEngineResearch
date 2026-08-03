"""
Phase 6 Tests — Portfolio Optimizer and Risk Manager
====================================================
Covers every Phase 6 test target from the build spec:

  - CVaR exact LP >= Gaussian approximation for negatively-skewed inputs
  - Gaussian fallback used when T < 30
  - Cornish-Fisher path produces a non-zero quantile adjustment for skewed inputs
  - ViewSourceTracker: warm-up active regardless of Sharpe; inactive when
    rolling Sharpe < -0.30
  - ledoit_wolf_cov differs in normal vs crisis mode
  - optimise_portfolio returns all 9 diagnostics keys
  - Black-Litterman posterior is well-formed
  - Drawdown governor thresholds; each of the 10 kill switches; 6 stress
    scenarios; PnL decomposition; RiskSnapshot / KillSwitchStatus instantiable
"""

from __future__ import annotations

import numpy as np
import pytest

from core import risk_manager as rm
from core.engine import optimizer as opt


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _neg_skew_returns(T: int = 160, n: int = 8, seed: int = 0, shocks: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, (T, n))
    shock_days = rng.choice(T, size=shocks, replace=False)
    r[shock_days] -= 0.06
    return r


def _moderate_skew_returns(T: int = 200, n: int = 6, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.01, (T, n))
    shock_days = rng.choice(T, size=10, replace=False)
    r[shock_days] -= 0.025
    return r


# ── 1. CVaR ──────────────────────────────────────────────────────────────────────

class TestCVaR:

    def test_exact_lp_at_least_gaussian_for_neg_skew(self):
        r = _neg_skew_returns()
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        assert opt.portfolio_cvar_exact(w, r) >= opt._gaussian_cvar(w, r)

    def test_gaussian_fallback_when_short(self):
        r = _neg_skew_returns()[:20]
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        assert opt.portfolio_cvar(w, r) == opt._gaussian_cvar(w, r)

    def test_cvar_is_positive_loss_magnitude(self):
        r = _neg_skew_returns()
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        assert opt.portfolio_cvar_exact(w, r) > 0.0

    def test_no_history_cvar_is_conservative_nonzero(self):
        # Item 2: with no returns history the optimiser must NOT report zero tail
        # risk (the old _gaussian_cvar <2-row path silently returned 0.0).
        opt.reset_view_tracker()
        out = opt.optimise_portfolio(["A", "B", "C", "D", "E"])
        assert out["cvar_95"] > 0.0, "no-history CVaR must be conservative non-zero, not 0.0"
        assert out["cvar_95"] < opt._CVAR_LIMIT["normal"], "prior-based CVaR should not spuriously bind"


# ── 2. Cornish-Fisher CVaR ───────────────────────────────────────────────────────

class TestCornishFisher:

    def test_nonzero_adjustment_for_skew(self):
        r = _moderate_skew_returns()
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        out = opt.compute_portfolio_cvar_cf(w, r)
        assert out["method"] == "cornish_fisher"
        assert abs(out["cf_quantile_adjustment"]) > 1e-9
        assert out["cvar"] > 0.0

    def test_extreme_moments_fall_back_to_historical(self):
        r = _neg_skew_returns(shocks=8)   # heavy skew triggers the historical fallback
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        out = opt.compute_portfolio_cvar_cf(w, r)
        assert out["method"] == "historical"

    def test_keys_present(self):
        r = _moderate_skew_returns()
        w = np.full(r.shape[1], 1.0 / r.shape[1])
        out = opt.compute_portfolio_cvar_cf(w, r)
        for key in ("cvar", "var", "portfolio_skew", "portfolio_kurtosis",
                    "cf_quantile_adjustment", "method"):
            assert key in out

    def test_cf_never_below_gaussian_for_neg_skew(self):
        # Item 8: mild negative skew with low kurtosis makes the Cornish-Fisher
        # expansion UNDER-estimate the Gaussian tail (skew=-0.27, kurt=-0.29 here:
        # CF~0.0085 vs Gaussian~0.0112). The Gaussian floor must prevent that.
        rng = np.random.default_rng(0)
        r = rng.normal(0.0004, 0.01, (252, 4))
        r[rng.choice(252, size=5, replace=False)] -= 0.01
        w = np.full(4, 1.0 / 4)
        cf = opt.compute_portfolio_cvar_cf(w, r, method="cornish_fisher")
        g = opt.compute_portfolio_cvar_cf(w, r, method="gaussian")
        assert cf["method"] == "cornish_fisher"          # in-band, no historical fallback
        assert cf["cvar"] >= g["cvar"] - 1e-12, "CF CVaR must never under-estimate Gaussian"


# ── 3. Covariance ────────────────────────────────────────────────────────────────

class TestCovariance:

    def test_normal_and_crisis_blends_differ(self):
        r = _neg_skew_returns()
        cov_n = opt.ledoit_wolf_cov(r, crisis_mode=False)
        cov_c = opt.ledoit_wolf_cov(r, crisis_mode=True)
        assert cov_n.shape == (r.shape[1], r.shape[1])
        assert not np.allclose(cov_n, cov_c)
        assert np.allclose(cov_n, cov_n.T)

    def test_nonlinear_estimator_reports_method(self):
        r = _neg_skew_returns()
        out = opt.estimate_covariance_nonlinear(r)
        for key in ("cov_matrix", "corr_matrix", "shrinkage_intensities",
                    "concentration_ratio", "method_used"):
            assert key in out
        assert out["method_used"] in ("nonlinear_LW", "linear_LW_fallback")


# ── 4. Black-Litterman ───────────────────────────────────────────────────────────

class TestBlackLitterman:

    def test_capm_equilibrium_shape(self):
        r = _neg_skew_returns()
        cov = opt.ledoit_wolf_cov(r)
        pi = opt.capm_equilibrium_returns(cov)
        assert pi.shape == (r.shape[1],) and np.all(np.isfinite(pi))

    def test_posterior_is_finite(self):
        r = _neg_skew_returns()
        cov = opt.ledoit_wolf_cov(r)
        pi = opt.capm_equilibrium_returns(cov)
        n = r.shape[1]
        P = np.eye(n)[:3]
        Q = np.array([0.01, 0.005, -0.008])
        mu_bl = opt.black_litterman_posterior(cov, pi, P, Q, tau=0.05)
        assert mu_bl.shape == (n,) and np.all(np.isfinite(mu_bl))

    def test_no_views_returns_prior(self):
        r = _neg_skew_returns()
        cov = opt.ledoit_wolf_cov(r)
        pi = opt.capm_equilibrium_returns(cov)
        mu_bl = opt.black_litterman_posterior(
            cov, pi, np.empty((0, r.shape[1])), np.array([])
        )
        assert np.allclose(mu_bl, pi)


# ── 5. ViewSourceTracker ─────────────────────────────────────────────────────────

class TestViewSourceTracker:

    def setup_method(self):
        opt.reset_view_tracker()

    def test_warmup_active_regardless_of_sharpe(self):
        tracker = opt.get_view_tracker()
        for _ in range(5):                       # < 20 entries ⇒ warm-up
            tracker.record("ml", 1.0, -0.05)     # terrible predictions
        assert tracker.is_active("ml") is True

    def test_inactive_when_rolling_sharpe_below_floor(self):
        tracker = opt.get_view_tracker()
        actuals = [-0.02 + 0.01 * ((-1) ** i) for i in range(25)]   # mean<0, var>0
        for a in actuals:
            tracker.record("engine", 1.0, a)
        assert tracker.rolling_sharpe("engine") < -0.30
        assert tracker.is_active("engine") is False

    def test_active_when_profitable(self):
        tracker = opt.get_view_tracker()
        actuals = [0.02 + 0.01 * ((-1) ** i) for i in range(25)]
        for a in actuals:
            tracker.record("insider", 1.0, a)
        assert tracker.is_active("insider") is True

    def test_singleton(self):
        assert opt.get_view_tracker() is opt.get_view_tracker()


# ── 6. optimise_portfolio ────────────────────────────────────────────────────────

class TestOptimisePortfolio:

    def setup_method(self):
        opt.reset_view_tracker()

    def test_returns_all_nine_diagnostics(self):
        r = _neg_skew_returns()
        symbols = [f"S{i}" for i in range(r.shape[1])]
        mlp = {s: (0.002 * (i - 3), 0.02, 0.6, 0.1, 0.7) for i, s in enumerate(symbols)}
        res = opt.optimise_portfolio(symbols, ml_predictions=mlp, returns_matrix=r, regime="trending")
        for key in ("weights", "expected_return", "expected_risk", "expected_cost_bps",
                    "cvar_95", "binding_constraints", "turnover_estimate",
                    "capacity_flags", "view_sources_active"):
            assert key in res

    def test_weights_long_only_and_unlevered(self):
        r = _neg_skew_returns()
        symbols = [f"S{i}" for i in range(r.shape[1])]
        res = opt.optimise_portfolio(symbols, returns_matrix=r)
        w = res["weights"]
        assert all(v >= -1e-9 for v in w.values())
        assert sum(w.values()) <= 1.0 + 1e-6
        assert isinstance(res["view_sources_active"], dict)


# ── 6b. Crisis severity scaling (Item 7) ─────────────────────────────────────────

class TestCrisisSeverityScaling:

    def setup_method(self):
        opt.reset_view_tracker()

    def test_higher_severity_delevers(self):
        # Rising continuous crisis severity tightens the vol target, de-levering
        # the book (gross exposure non-increasing in severity, strictly lower at top).
        r = _neg_skew_returns()
        symbols = [f"S{i}" for i in range(r.shape[1])]
        gross = []
        for sev in (0.0, 0.45, 0.70, 0.90):
            opt.reset_view_tracker()
            res = opt.optimise_portfolio(symbols, returns_matrix=r, crisis_severity=sev)
            gross.append(sum(res["weights"].values()))
        assert all(gross[i] >= gross[i + 1] - 1e-9 for i in range(len(gross) - 1)), gross
        assert gross[-1] < gross[0] - 1e-6, gross

    def test_backward_compat_without_severity(self):
        # No crisis_severity arg must reproduce the pre-change normal-mode weights.
        r = _neg_skew_returns()
        symbols = [f"S{i}" for i in range(r.shape[1])]
        opt.reset_view_tracker()
        a = opt.optimise_portfolio(symbols, returns_matrix=r)
        opt.reset_view_tracker()
        b = opt.optimise_portfolio(symbols, returns_matrix=r, crisis_severity=None)
        assert a["weights"] == b["weights"]


# ── 6c. CVaR limit enforcement (Item 1) ──────────────────────────────────────────

class TestCVaREnforcement:

    def test_enforce_scales_down_to_limit(self):
        # A book whose CVaR exceeds the limit must be de-levered toward cash until
        # CVaR <= limit (never levered up); long-only is preserved. Previously the
        # breach was only flagged, never corrected.
        r = _neg_skew_returns(shocks=10)
        n = r.shape[1]
        w = np.full(n, 1.0 / n)
        limit = 0.02
        assert opt.portfolio_cvar(w, r) > limit          # precondition: real breach
        w2, cvar2, binding = opt._enforce_cvar_limit(w, r, None, limit)
        assert binding is True
        assert cvar2 <= limit + 1e-9
        assert opt.portfolio_cvar(w2, r) <= limit + 1e-9  # the returned weights respect it
        assert w2.sum() <= w.sum() + 1e-12                # never levered up
        assert all(x >= -1e-12 for x in w2)               # long-only preserved

    def test_enforce_is_noop_within_limit(self):
        r = _neg_skew_returns(shocks=2)
        n = r.shape[1]
        w = np.full(n, 1.0 / n) * 0.1                     # tiny book -> low CVaR
        limit = 0.05
        assert opt.portfolio_cvar(w, r) <= limit          # precondition: no breach
        w2, _cvar2, binding = opt._enforce_cvar_limit(w, r, None, limit)
        assert binding is False
        assert np.allclose(w2, w)

    def test_optimiser_output_never_exceeds_cvar_limit(self):
        # Invariant: the weights optimise_portfolio returns always satisfy the limit.
        r = _neg_skew_returns(shocks=10)
        symbols = [f"S{i}" for i in range(r.shape[1])]
        opt.reset_view_tracker()
        res = opt.optimise_portfolio(symbols, returns_matrix=r, crisis_severity=0.9)
        w = np.array([res["weights"][s] for s in symbols])
        assert opt.portfolio_cvar(w, r) <= 0.05 * 0.5 + 1e-6   # crisis-tightened limit


# ── 7. Penalty terms ─────────────────────────────────────────────────────────────

class TestPenalties:

    def test_impact_cost_non_negative_and_monotone(self):
        adv = np.array([1.0, 1.0, 1.0])
        sigma = np.array([0.02, 0.02, 0.02])
        small = opt.impact_cost(np.array([0.01, 0.0, 0.0]), adv, sigma)
        large = opt.impact_cost(np.array([0.10, 0.0, 0.0]), adv, sigma)
        assert 0.0 <= small < large

    def test_capacity_penalty_flags_oversize(self):
        weights = np.array([0.5, 0.5])
        adv = np.array([1.0, 1.0])           # tiny ADV vs large notional
        assert opt.capacity_penalty(weights, adv, capital=1_000_000.0) > 0.0

    def test_exposure_penalty_zero_within_cap(self):
        weights = np.array([0.1, 0.1, 0.1])
        sector_map = {0: "tech", 1: "health", 2: "energy"}
        assert opt.exposure_penalty(weights, sector_map) == 0.0
        concentrated = {0: "tech", 1: "tech", 2: "tech"}
        assert opt.exposure_penalty(np.array([0.1, 0.1, 0.1]), concentrated) > 0.0


# ── 8. Drawdown governor ─────────────────────────────────────────────────────────

class TestDrawdownGovernor:

    @pytest.mark.parametrize(
        "dd, expected",
        [
            (0.00, "NORMAL"), (0.049, "NORMAL"),
            (0.05, "SOFT"), (0.079, "SOFT"),
            (0.08, "MEDIUM"), (0.119, "MEDIUM"),
            (0.12, "HARD"), (0.15, "HARD"),
            (0.151, "KILL"), (0.30, "KILL"),
        ],
    )
    def test_thresholds(self, dd, expected):
        assert rm.check_drawdown(dd) == expected


# ── 9. Kill switches ─────────────────────────────────────────────────────────────

class TestKillSwitches:

    def test_default_context_inactive(self):
        assert rm.check_kill_switches({}, mode="LIVE").active is False

    @pytest.mark.parametrize(
        "reason, ctx",
        [
            ("STALE_MARKET_DATA", {"market_data_age_s": 10}),
            ("STALE_FEATURES", {"feature_age_s": 10}),
            ("BROKER_REJECTION_RATE", {"recent_order_rejections": [True, True, True] + [False] * 7}),
            ("SLIPPAGE_SPIKE", {"realized_slippage_bps": 40, "expected_slippage_bps": 10}),
            ("FILL_MISMATCH", {"fill_qty": 90, "expected_qty": 100}),
            ("MISSING_HEARTBEAT", {"heartbeat_age_s": 60}),
            ("NAV_RECONCILIATION", {"nav_reconciliation_error": 0.01}),
            ("ABNORMAL_LOSS", {"intraday_pnl_pct": -0.05}),
            ("MARKET_HALT", {"market_halted": True}),
            ("INTERNAL_EXCEPTION", {"internal_exception": True}),
        ],
    )
    def test_each_trigger(self, reason, ctx):
        status = rm.check_kill_switches(ctx, mode="LIVE")
        assert status.active is True
        assert reason in (status.reason or "")
        assert status.requires_manual_reset is True
        assert status.triggered_at is not None

    def test_stale_data_ignored_outside_live(self):
        assert rm.check_kill_switches({"market_data_age_s": 10}, mode="RESEARCH").active is False


# ── 10. Stress tests ─────────────────────────────────────────────────────────────

class TestStressTests:

    def test_six_scenarios(self):
        w = np.full(5, 0.2)
        r = np.random.default_rng(0).normal(0.0005, 0.01, (120, 5))
        out = rm.run_stress_tests(w, r)
        assert set(out.keys()) == {
            "worst_5_days", "corr_to_1", "vol_x2",
            "overnight_gap", "liquidity_cut", "sector_shock",
        }
        assert all(isinstance(v, float) for v in out.values())


# ── 11. PnL decomposition ────────────────────────────────────────────────────────

class TestPnLDecompose:

    def test_keys_and_reconciliation(self):
        w = np.full(5, 0.2)
        out = rm.decompose_pnl(
            pnl=0.012, weights=w,
            factor_returns=np.array([0.01, 0.02, -0.01, 0.0, 0.03]),
            benchmark_return=0.008,
        )
        for key in ("signal_alpha", "beta_contribution", "factor_contribution",
                    "carry", "costs_slippage", "residual"):
            assert key in out
        assert abs(sum(out.values()) - 0.012) < 1e-9


# ── 12. RiskSnapshot / pre-trade ─────────────────────────────────────────────────

class TestRiskSnapshot:

    def test_dataclasses_instantiable(self):
        from datetime import datetime, timezone
        snap = rm.RiskSnapshot(
            gross_exposure=1.0, net_exposure=1.0, max_single_name_pct=0.03,
            max_sector_pct=0.2, beta_exposure=1.0, target_vol_utilization=0.9,
            cvar_utilization=0.8, drawdown_current=0.02, illiquidity_score=0.1,
            kill_switch_active=False, active_flags=[],
            timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        )
        assert snap.gross_exposure == 1.0
        kss = rm.KillSwitchStatus(active=False, reason=None, triggered_at=None, requires_manual_reset=False)
        assert kss.active is False

    def test_check_pretrade_builds_snapshot(self):
        snap = rm.get_risk_manager().check_pretrade(
            {"A": 0.3, "B": 0.2, "C": 0.1},
            {"drawdown_current": 0.13, "portfolio_vol": 0.12, "cvar_95": 0.04},
        )
        assert isinstance(snap, rm.RiskSnapshot)
        assert "DRAWDOWN_HARD" in snap.active_flags
        assert snap.gross_exposure == pytest.approx(0.6)

    # RISK-1 / ENGINE-1 — STEP-10 independently enforces hard limits (defense in depth).
    def test_check_pretrade_enforces_concentration(self):
        snap = rm.get_risk_manager().check_pretrade({"A": 0.5, "B": 0.1}, {"max_position_weight": 0.2})
        assert snap.kill_switch_active is True
        assert any("CONCENTRATION_BREACH" in f for f in snap.active_flags)

    def test_check_pretrade_enforces_leverage(self):
        snap = rm.get_risk_manager().check_pretrade({"A": 1.5, "B": 1.0}, {"max_gross_leverage": 2.0})
        assert snap.kill_switch_active is True
        assert any("LEVERAGE_BREACH" in f for f in snap.active_flags)

    def test_check_pretrade_enforces_cvar(self):
        snap = rm.get_risk_manager().check_pretrade({"A": 0.3}, {"cvar_95": 0.10, "cvar_limit": 0.05})
        assert snap.kill_switch_active is True
        assert any("CVAR_BREACH" in f for f in snap.active_flags)

    def test_check_pretrade_within_limits_no_block(self):
        snap = rm.get_risk_manager().check_pretrade(
            {"A": 0.15, "B": 0.15},
            {"max_position_weight": 0.2, "max_gross_leverage": 1.0, "cvar_95": 0.02, "cvar_limit": 0.05})
        assert snap.kill_switch_active is False
        assert not any("BREACH" in f for f in snap.active_flags)

    def test_check_pretrade_no_limits_no_enforcement(self):
        # back-compat: no limits supplied -> no enforcement even on a huge concentration
        snap = rm.get_risk_manager().check_pretrade({"A": 0.9}, {})
        assert not any("BREACH" in f for f in snap.active_flags)
        assert snap.kill_switch_active is False

    # RISK-6 — kill-switch latch + hard-stop signal.
    def test_hard_stop_on_kill_drawdown(self):
        snap = rm.get_risk_manager().check_pretrade({"A": 0.3}, {"drawdown_current": 0.20})
        assert snap.hard_stop is True and snap.kill_switch_active is True

    def test_no_hard_stop_on_concentration_breach(self):
        snap = rm.get_risk_manager().check_pretrade({"A": 0.5}, {"max_position_weight": 0.2})
        assert snap.kill_switch_active is True   # blocked this cycle
        assert snap.hard_stop is False           # but NOT a latch-worthy hard stop

    def test_kill_switch_latch_engages_and_is_sticky(self):
        latch = rm.KillSwitchLatch()
        assert not latch.is_latched
        assert latch.engage("DRAWDOWN_KILL", "t0") is True
        assert latch.is_latched
        assert latch.engage("OTHER", "t1") is False     # already latched -> no-op
        assert latch.reason == "DRAWDOWN_KILL"          # original reason preserved

    def test_kill_switch_latch_only_reset_clears(self):
        latch = rm.KillSwitchLatch()
        latch.engage("KILL", "t0")
        assert latch.is_latched                          # stays latched (no auto re-enable)
        assert latch.reset("operator_jane", "t1") is True
        assert not latch.is_latched and latch.reset_by == "operator_jane"

    def test_kill_switch_latch_serialises(self):
        latch = rm.KillSwitchLatch()
        latch.engage("KILL", "t0")
        restored = rm.KillSwitchLatch.from_json(latch.to_json())
        assert restored.is_latched and restored.reason == "KILL" and restored.engaged_at == "t0"
