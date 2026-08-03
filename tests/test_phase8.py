"""
Phase 8 Tests — Monitoring, Registry, and Governance
====================================================
Covers every Phase 8 test target from the build spec:

  - ModelRegistry: register / promote / rollback / latest_live / latest_shadow
  - rollback() restores the previous live model correctly
  - monitoring.snapshot() returns all 4 sections (HEALTH/TRADING/MODEL/RISK)
    with all documented sub-keys
  - alert_list() returns alerts with valid severity values (INFO/WARNING/AMBER/RED)
  - performance_tracker.evaluate_signal() resolves outcomes at all 4 horizons
    (1d/5d/10d/20d) and feeds ml_return_model + view tracker + calibration
  - adaptive_weights.propose_and_validate() rejects new weights when
    selection_rule() fails; frozen mode blocks all updates
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data.data_contracts import FillEvent, PredictionRow
from research.validation import ValidationResult

from core import ml_return_model as mlm
from core.engine import optimizer as opt
from learning import adaptive_weights as aw
from learning import performance_tracker as pt
from ops import model_registry as reg
from ops import monitoring as mon


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vres(*, passing: bool = True) -> ValidationResult:
    """A ValidationResult that passes (or fails) selection_rule()."""
    if passing:
        return ValidationResult(
            mean_ic=0.03, mean_rank_ic=0.05, sharpe_net=1.20, turnover=0.10,
            hit_rate=0.55, max_drawdown=-0.10, pbo_proxy=0.10,
            deflated_sharpe_proxy=0.40, cost_drag_bps=2.0, stability_score=0.70,
            deflated_sharpe_ratio=0.99,
        )
    # sharpe_net <= 0.75 fails the selection rule.
    return ValidationResult(
        mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.0, turnover=0.5,
        hit_rate=0.45, max_drawdown=-0.40, pbo_proxy=0.9,
        deflated_sharpe_proxy=0.0, cost_drag_bps=20.0, stability_score=0.10,
    )


def _record(model_id: str = "m1", model_type: str = "ml_ensemble") -> "reg.ModelRecord":
    return reg.ModelRecord(
        model_id=model_id,
        model_type=model_type,
        training_window=(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
        feature_schema_version="v6",
        hyperparameters={"n_estimators": 200},
        validation_metrics=_vres(),
        calibration_metrics={"brier": 0.18},
        drift_baseline={"psi": 0.02},
        regime_breakdown={"trending": {"sharpe": 1.1}},
        artifact_path=f"/models/{model_id}.pkl",
        promoted_to_live=False,
        promoted_at=None,
        retired_at=None,
    )


# ── 1. Model registry ──────────────────────────────────────────────────────────

class TestModelRegistry:

    def setup_method(self):
        reg.reset_model_registry()

    def test_register_returns_id_and_record_is_shadow(self):
        r = reg.get_model_registry()
        mid = r.register(_record("m1"))
        assert mid == "m1"
        assert r.latest_live() is None
        assert r.latest_shadow().model_id == "m1"   # registered, not yet live

    def test_duplicate_registration_raises(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        with pytest.raises(ValueError, match="already registered"):
            r.register(_record("m1"))

    def test_promote_sets_live(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.promote("m1")
        live = r.latest_live()
        assert live is not None
        assert live.model_id == "m1"
        assert live.promoted_to_live is True
        assert live.promoted_at is not None
        assert r.latest_shadow() is None            # nothing pending now

    def test_promote_unknown_raises(self):
        r = reg.get_model_registry()
        with pytest.raises(ValueError, match="unknown model_id"):
            r.promote("ghost")

    def test_latest_shadow_is_pending_challenger(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.promote("m1")
        r.register(_record("m2"))                   # challenger, not promoted
        assert r.latest_live().model_id == "m1"
        assert r.latest_shadow().model_id == "m2"

    def test_rollback_restores_previous_live(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.promote("m1")
        r.register(_record("m2"))
        r.promote("m2")
        assert r.latest_live().model_id == "m2"

        r.rollback("m2 underperforming in production")

        live = r.latest_live()
        assert live.model_id == "m1"
        assert live.promoted_to_live is True
        assert live.retired_at is None
        # the rolled-back model is demoted and retired
        m2 = r.get("m2")
        assert m2.promoted_to_live is False
        assert m2.retired_at is not None

    def test_rollback_without_history_raises(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.promote("m1")
        with pytest.raises(ValueError, match="no previous live"):
            r.rollback("nothing to roll back to")


# ── 2. Performance tracker (multi-horizon outcome resolution) ────────────────────

_T0 = datetime(2026, 3, 2, tzinfo=timezone.utc)


def _pred(symbol: str = "AAPL", expected_return: float = 0.02,
          p_positive: float = 0.7, asof: datetime = _T0) -> PredictionRow:
    return PredictionRow(
        symbol=symbol, asof_timestamp=asof, model_version="v6.0",
        expected_return=expected_return, risk_estimate=0.15,
        p_positive=p_positive, p_tail_loss=0.05, confidence=0.6,
    )


def _daily_prices(start: datetime, n_days: int, start_price: float = 100.0,
                  daily_ret: float = 0.01) -> list[tuple[datetime, float]]:
    out: list[tuple[datetime, float]] = []
    price = start_price
    for d in range(n_days + 1):
        out.append((start + timedelta(days=d), price))
        price *= 1.0 + daily_ret
    return out


class TestPerformanceTracker:

    def setup_method(self):
        pt.reset_performance_tracker()
        mlm.reset_model()
        opt.reset_view_tracker()

    def _load(self, n_days: int = 25, source: str = "ml") -> "pt.PerformanceTracker":
        t = pt.get_performance_tracker()
        t.record_prediction(_pred(), source=source, sleeve="momentum",
                             regime="trending", execution_regime="normal_exec")
        for ts, price in _daily_prices(_T0, n_days):
            t.record_price("AAPL", ts, price)
        return t

    def test_resolves_all_four_horizons(self):
        t = self._load(25)
        t.evaluate_signal("AAPL", _T0)
        horizons = {o["horizon"] for o in t.outcomes()}
        assert horizons == {1, 5, 10, 20}

    def test_feeds_ml_model_and_view_tracker(self):
        t = self._load(25)
        t.evaluate_signal("AAPL", _T0)
        assert len(mlm.get_model()._prediction_log) == 4
        assert len(opt.get_view_tracker()._prediction_log["ml"]) == 4

    def test_unelapsed_horizons_not_resolved(self):
        t = self._load(8)
        t.evaluate_signal("AAPL", _T0)
        horizons = {o["horizon"] for o in t.outcomes()}
        assert horizons == {1, 5}

    def test_evaluate_is_idempotent(self):
        t = self._load(25)
        t.evaluate_signal("AAPL", _T0)
        t.evaluate_signal("AAPL", _T0)
        assert len(t.outcomes()) == 4

    def test_outcome_dimensions_tracked(self):
        t = self._load(25)
        t.evaluate_signal("AAPL", _T0)
        o = t.outcomes()[0]
        for key in ("symbol", "sleeve", "model_version", "regime",
                    "execution_regime", "horizon", "predicted_return",
                    "actual_return", "raw_return"):
            assert key in o
        assert o["sleeve"] == "momentum"
        assert o["model_version"] == "v6.0"

    def test_fill_adjusted_return_subtracts_cost(self):
        t = self._load(25)
        t.record_fill(FillEvent(
            order_id="AAPL", symbol="AAPL", qty=100.0, fill_price=100.0,
            decision_price=100.0, arrival_price=100.0, slippage_bps=50.0,
            fill_timestamp=_T0,
        ))
        t.evaluate_signal("AAPL", _T0)
        o1 = next(o for o in t.outcomes() if o["horizon"] == 1)
        assert o1["actual_return"] == pytest.approx(o1["raw_return"] - 0.0050, abs=1e-9)

    def test_calibration_report(self):
        t = self._load(25)
        t.evaluate_signal("AAPL", _T0)
        rep = t.calibration_report()
        for key in ("brier_score", "n_samples"):
            assert key in rep
        assert rep["n_samples"] == 4
        assert 0.0 <= rep["brier_score"] <= 1.0


# ── 3. Adaptive weights (validation-gated, frozen mode) ──────────────────────────

class TestAdaptiveWeights:

    def test_accepts_when_selection_rule_passes(self):
        weights = aw.AdaptiveWeights(initial_weights={"momentum": 0.5, "mean_reversion": 0.5})
        new = {"momentum": 0.7, "mean_reversion": 0.3}
        timestamps = [_T0 + timedelta(days=i) for i in range(90)]
        out = weights.propose_and_validate(
            new, validation_result=_vres(passing=True), timestamps=timestamps
        )
        assert out["accepted"] is True
        assert weights.weights == new
        assert out["applied_weights"] == new

    def test_rejects_without_walk_forward_window(self):
        # The purged walk-forward split is MANDATORY before any weight change.
        # Omitting the validation window must reject the change and retain weights,
        # never apply on selection_rule() alone.
        weights = aw.AdaptiveWeights(initial_weights={"momentum": 0.5, "mean_reversion": 0.5})
        out = weights.propose_and_validate(
            {"momentum": 0.7, "mean_reversion": 0.3}, validation_result=_vres(passing=True)
        )
        assert out["accepted"] is False
        assert weights.weights == {"momentum": 0.5, "mean_reversion": 0.5}   # retained
        assert "VALIDATION_FAILED" in out["reason"]

    def test_rejects_when_selection_rule_fails(self):
        weights = aw.AdaptiveWeights(initial_weights={"momentum": 0.5, "mean_reversion": 0.5})
        timestamps = [_T0 + timedelta(days=i) for i in range(90)]
        out = weights.propose_and_validate(
            {"momentum": 0.9, "mean_reversion": 0.1},
            validation_result=_vres(passing=False), timestamps=timestamps,
        )
        assert out["accepted"] is False
        assert weights.weights == {"momentum": 0.5, "mean_reversion": 0.5}   # retained
        assert out["applied_weights"] == {"momentum": 0.5, "mean_reversion": 0.5}
        assert "VALIDATION_FAILED" in out["reason"]

    def test_frozen_blocks_all_updates(self):
        weights = aw.AdaptiveWeights(
            initial_weights={"momentum": 0.5, "mean_reversion": 0.5}, frozen=True
        )
        out = weights.propose_and_validate(
            {"momentum": 1.0, "mean_reversion": 0.0}, validation_result=_vres(passing=True)
        )
        assert out["accepted"] is False
        assert out["reason"] == "FROZEN"
        assert weights.weights == {"momentum": 0.5, "mean_reversion": 0.5}

    def test_freeze_unfreeze_is_reversible(self):
        weights = aw.AdaptiveWeights(initial_weights={"a": 1.0})
        weights.freeze("manual halt for incident review")
        assert weights.frozen is True
        weights.unfreeze("incident resolved")
        assert weights.frozen is False
        timestamps = [_T0 + timedelta(days=i) for i in range(90)]
        out = weights.propose_and_validate(
            {"a": 1.0}, validation_result=_vres(passing=True), timestamps=timestamps
        )
        assert out["accepted"] is True

    def test_runs_purged_walk_forward_on_window(self):
        weights = aw.AdaptiveWeights(initial_weights={"a": 0.5, "b": 0.5})
        timestamps = [_T0 + timedelta(days=i) for i in range(90)]
        out = weights.propose_and_validate(
            {"a": 0.6, "b": 0.4}, validation_result=_vres(passing=True), timestamps=timestamps
        )
        assert out["accepted"] is True
        assert out["n_folds"] >= 1

    def test_negative_weights_raise(self):
        weights = aw.AdaptiveWeights(initial_weights={"a": 1.0})
        with pytest.raises(ValueError, match="non-negative"):
            weights.propose_and_validate(
                {"a": -0.2, "b": 1.2}, validation_result=_vres(passing=True)
            )


# ── 4. Monitoring (4-section snapshot + alerts) ──────────────────────────────────

_VALID_SEVERITIES = {"INFO", "WARNING", "AMBER", "RED"}

_HEALTH_KEYS = ("heartbeat_age_seconds", "market_data_latency_ms", "stale_feature_count",
                "broker_rejection_rate", "failed_prediction_count", "ibkr_connected")
_TRADING_KEYS = ("gross_exposure", "net_exposure", "turnover_today", "fill_rate",
                 "avg_slippage_bps", "expected_vs_realized_cost_delta", "active_kill_switches")
_MODEL_KEYS = ("rolling_ic_20d", "calibration_error", "shadow_vs_live_divergence",
               "drift_flags_active", "last_refit_timestamp", "model_version_live",
               "model_version_shadow")
_RISK_KEYS = ("drawdown_current", "vol_utilization", "cvar_utilization",
              "severity_score", "liquidity_stress_score")


class TestMonitoring:

    def setup_method(self):
        reg.reset_model_registry()

    def test_snapshot_has_all_sections_and_keys(self):
        snap = mon.snapshot()
        assert set(snap) >= {"HEALTH", "TRADING", "MODEL", "RISK"}
        for key in _HEALTH_KEYS:
            assert key in snap["HEALTH"]
        for key in _TRADING_KEYS:
            assert key in snap["TRADING"]
        for key in _MODEL_KEYS:
            assert key in snap["MODEL"]
        for key in _RISK_KEYS:
            assert key in snap["RISK"]

    def test_snapshot_pulls_model_versions_from_registry(self):
        r = reg.get_model_registry()
        r.register(_record("live1"))
        r.promote("live1")
        r.register(_record("shadow1"))
        snap = mon.snapshot()
        assert snap["MODEL"]["model_version_live"] == "live1"
        assert snap["MODEL"]["model_version_shadow"] == "shadow1"

    def test_snapshot_maps_risk_snapshot_fields(self):
        from types import SimpleNamespace
        rs = SimpleNamespace(drawdown_current=0.07, target_vol_utilization=0.80,
                             cvar_utilization=0.50, active_flags=["DRAWDOWN_SOFT", "INTRADAY_LOSS"])
        snap = mon.snapshot({"risk_snapshot": rs})
        assert snap["RISK"]["drawdown_current"] == pytest.approx(0.07)
        assert snap["RISK"]["vol_utilization"] == pytest.approx(0.80)
        assert snap["RISK"]["cvar_utilization"] == pytest.approx(0.50)
        assert snap["TRADING"]["active_kill_switches"] == ["DRAWDOWN_SOFT", "INTRADAY_LOSS"]

    def test_snapshot_maps_crisis_status(self):
        from types import SimpleNamespace
        cs = SimpleNamespace(severity_score=0.70, liquidity_stress_score=0.40)
        snap = mon.snapshot({"crisis_status": cs})
        assert snap["RISK"]["severity_score"] == pytest.approx(0.70)
        assert snap["RISK"]["liquidity_stress_score"] == pytest.approx(0.40)

    def test_alert_list_valid_severities_under_stress(self):
        from types import SimpleNamespace
        state = {
            "risk_snapshot": SimpleNamespace(
                drawdown_current=0.13, target_vol_utilization=1.20,
                cvar_utilization=1.10, active_flags=["KILL:INTRADAY_LOSS"]),
            "crisis_status": SimpleNamespace(severity_score=0.80, liquidity_stress_score=0.60),
            "stale_feature_count": 3, "ibkr_connected": False,
        }
        alerts = mon.alert_list(state)
        assert alerts
        assert all(a["severity"] in _VALID_SEVERITIES for a in alerts)
        assert all("severity" in a and "message" in a for a in alerts)
        assert any(a["severity"] == "RED" for a in alerts)   # kill switch / deep drawdown

    def test_alert_list_clean_state_has_valid_severities(self):
        alerts = mon.alert_list({"ibkr_connected": True})
        assert all(a["severity"] in _VALID_SEVERITIES for a in alerts)

    def test_alert_list_surfaces_risk_events(self):
        from types import SimpleNamespace
        ev = SimpleNamespace(event_type="KILL_SWITCH", severity="RED",
                             description="intraday loss limit breached")
        alerts = mon.alert_list({"risk_events": [ev], "ibkr_connected": True})
        assert any(a["severity"] == "RED" and "intraday" in a["message"].lower() for a in alerts)


class TestTrainingFeedback:
    """Resolved outcomes feed the model's TRAINING BUFFER (real refit loop)."""

    def setup_method(self):
        pt.reset_performance_tracker()
        mlm.reset_model()
        opt.reset_view_tracker()

    def _features(self) -> dict:
        return {n: 0.1 for n in mlm.FEATURE_NAMES}

    def test_horizon1_outcome_feeds_training_buffer(self):
        t = pt.get_performance_tracker()
        t.record_prediction(_pred(), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec",
                            features=self._features())
        for ts, price in _daily_prices(_T0, 25):
            t.record_price("AAPL", ts, price)
        t.evaluate_signal("AAPL", _T0)
        # 4 horizons resolved, but exactly ONE training example (the 1d horizon —
        # the model's prediction target), recorded once (idempotent).
        assert mlm.get_model().training_buffer_size == 1
        t.evaluate_signal("AAPL", _T0)
        assert mlm.get_model().training_buffer_size == 1

    def test_prediction_without_features_resolves_but_does_not_train(self):
        t = pt.get_performance_tracker()
        t.record_prediction(_pred(), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec")
        for ts, price in _daily_prices(_T0, 25):
            t.record_price("AAPL", ts, price)
        t.evaluate_signal("AAPL", _T0)
        assert len(mlm.get_model()._prediction_log) == 4   # outcomes still recorded
        assert mlm.get_model().training_buffer_size == 0   # no features ⇒ no training row


class TestAuditLog:

    def test_append_creates_header_then_rows(self, tmp_path):
        from ops import audit_log
        path = tmp_path / "audit.md"
        summary = {"asof_time": "2025-10-28T14:00:00+00:00", "mode": "PAPER",
                   "regime": "trending", "crisis_level": "NORMAL", "blocked": False,
                   "admitted": 2, "order_intents": 2, "fills": 4,
                   "live_orders_submitted": 0, "alerts": 1}
        audit_log.append_cycle_summary(path, summary)
        audit_log.append_cycle_summary(path, summary)
        text = path.read_text(encoding="utf-8")
        assert text.count("| asof |") == 1               # header written exactly once
        assert text.count("| 2025-10-28T14:00:00+00:00 |") == 2
        assert "PAPER" in text and "NORMAL" in text


class TestPersistence:
    """ROADMAP Phase 4 — durable registry + tracker snapshots (90-day retention)."""

    def setup_method(self):
        pt.reset_performance_tracker()
        mlm.reset_model()
        opt.reset_view_tracker()
        reg.reset_model_registry()

    def test_registry_round_trip_preserves_live_shadow_and_rollback(self, tmp_path):
        from ops import persistence
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.register(_record("m2"))
        r.promote("m1")
        r.promote("m2")                              # m1 demoted into history
        path = tmp_path / "state.json"
        persistence.save_state(path)

        reg.reset_model_registry()
        persistence.restore_state(path)
        restored = reg.get_model_registry()
        assert restored.latest_live().model_id == "m2"
        assert restored.get("m1").validation_metrics.sharpe_net == pytest.approx(1.20)
        restored.rollback("test")                     # history survived the round trip
        assert restored.latest_live().model_id == "m1"

    def test_tracker_round_trip_keeps_resolution_idempotent(self, tmp_path):
        from ops import persistence
        t = pt.get_performance_tracker()
        t.record_prediction(_pred(), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec",
                            features={"signal_score": 0.5})
        for ts, price in _daily_prices(_T0, 25):
            t.record_price("AAPL", ts, price)
        t.evaluate_signal("AAPL", _T0)
        n_before = len(t.outcomes())
        assert n_before == 4

        path = tmp_path / "state.json"
        persistence.save_state(path)
        pt.reset_performance_tracker()
        persistence.restore_state(path)

        t2 = pt.get_performance_tracker()
        assert len(t2.outcomes()) == n_before
        t2.evaluate_signal("AAPL", _T0)               # resolved horizons survived
        assert len(t2.outcomes()) == n_before         # no double-counting

    def test_retention_prunes_stale_history(self, tmp_path):
        from ops import persistence
        t = pt.get_performance_tracker()
        old = _T0 - timedelta(days=200)
        t.record_prediction(_pred(asof=old), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec")
        t.record_prediction(_pred(), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec")
        t.record_price("AAPL", old, 90.0)
        t.record_price("AAPL", _T0, 100.0)
        path = tmp_path / "state.json"
        persistence.save_state(path, retention_days=90)

        pt.reset_performance_tracker()
        persistence.restore_state(path)
        t2 = pt.get_performance_tracker()
        assert len(t2._predictions["AAPL"]) == 1      # 200-day-old prediction pruned
        assert t2._price_val["AAPL"] == [100.0]       # stale price pruned

    def test_restore_missing_file_raises(self, tmp_path):
        from ops import persistence
        with pytest.raises(FileNotFoundError):
            persistence.restore_state(tmp_path / "nope.json")


class TestChallengerLifecycle:
    """ROADMAP Phase 4 — validation-gated promotion + idempotent rollback."""

    def setup_method(self):
        reg.reset_model_registry()

    def test_promotion_is_validation_gated(self):
        r = reg.get_model_registry()
        bad = _record("bad")
        bad.validation_metrics = _vres(passing=False)
        r.register(bad)
        with pytest.raises(ValueError, match="validation"):
            r.promote("bad")
        assert r.latest_live() is None                # nothing went live

    def test_rollback_retry_is_idempotent_with_expect_current(self):
        r = reg.get_model_registry()
        r.register(_record("m1"))
        r.register(_record("m2"))
        r.promote("m1")
        r.promote("m2")
        restored = r.rollback("incident-7", expect_current="m2")
        assert restored == "m1"
        # Operator retry of the SAME incident: current is no longer m2 → no-op.
        assert r.rollback("incident-7 retry", expect_current="m2") is None
        assert r.latest_live().model_id == "m1"

    def test_promotion_candidate_requires_passing_validation(self):
        r = reg.get_model_registry()
        bad = _record("bad")
        bad.validation_metrics = _vres(passing=False)
        r.register(bad)
        assert r.promotion_candidate() is None
        r.register(_record("good"))
        assert r.promotion_candidate().model_id == "good"

    def test_engine_surfaces_candidate_without_auto_promoting(self):
        # STEP 13 spec: "check for promotion criteria (never auto-promote)".
        from ops import monitoring as m
        r = reg.get_model_registry()
        r.register(_record("challenger"))
        alerts = m.alert_list({"shadow_promotion_candidate": "challenger"})
        assert any(a["severity"] == "INFO" and "challenger" in a["message"]
                   for a in alerts)
        assert r.latest_live() is None                # surfaced, never promoted
