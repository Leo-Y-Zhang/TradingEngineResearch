"""
Persistence Backend Tests — ROADMAP Phase 6, item 2
===================================================
The pluggable state-store backends (ops/state_store.py) over the registry +
tracker singletons. The JSON path is already covered by test_phase8.py's
TestPersistence; here we verify the SQLAlchemy/SQLite backend round-trips and
prunes identically (backend parity), the config factory selects correctly, the
schema version guard fires, and the Alembic initial migration builds the schema.

See docs/specs/2026-06-17-persistence-layer-design.md.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import ml_return_model as mlm
from core.config import EngineSettings, make_state_store
from core.engine import optimizer as opt
from data.data_contracts import FillEvent
from data.data_contracts import PredictionRow
from learning import performance_tracker as pt
from ops import model_registry as reg
from ops import persistence
from ops import state_store
from research.validation import ValidationResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_T0 = datetime(2026, 3, 2, tzinfo=timezone.utc)
NO_PRUNE = 100_000  # ~274y retention: nothing pruned (exact round-trip), safe for timedelta


def _settings(**kwargs) -> EngineSettings:
    return EngineSettings(_env_file=None, **kwargs)


def _vres() -> ValidationResult:
    return ValidationResult(
        mean_ic=0.03, mean_rank_ic=0.05, sharpe_net=1.20, turnover=0.10,
        hit_rate=0.55, max_drawdown=-0.10, pbo_proxy=0.10,
        deflated_sharpe_proxy=0.40, cost_drag_bps=2.0, stability_score=0.70,
        deflated_sharpe_ratio=0.97,
    )


def _record(model_id: str = "m1") -> "reg.ModelRecord":
    return reg.ModelRecord(
        model_id=model_id,
        model_type="ml_ensemble",
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


def _pred(symbol: str = "AAPL", asof: datetime = _T0) -> PredictionRow:
    return PredictionRow(
        symbol=symbol, asof_timestamp=asof, model_version="v6.0",
        expected_return=0.02, risk_estimate=0.15,
        p_positive=0.7, p_tail_loss=0.05, confidence=0.6,
    )


def _daily_prices(start: datetime, n_days: int) -> list[tuple[datetime, float]]:
    out, price = [], 100.0
    for d in range(n_days + 1):
        out.append((start + timedelta(days=d), price))
        price *= 1.01
    return out


def _fill(symbol: str = "AAPL") -> FillEvent:
    return FillEvent(
        order_id="o1", symbol=symbol, qty=100.0, fill_price=101.0,
        decision_price=100.0, arrival_price=100.5, slippage_bps=5.0,
        fill_timestamp=_T0,
    )


def _populate() -> None:
    """Realistic registry + tracker state exercising every table."""
    r = reg.get_model_registry()
    r.register(_record("m1"))
    r.register(_record("m2"))
    r.promote("m1")
    r.promote("m2")  # m1 demoted into rollback history

    t = pt.get_performance_tracker()
    t.record_prediction(_pred(), source="ml", sleeve="momentum",
                        regime="trending", execution_regime="normal_exec",
                        features={"signal_score": 0.5})
    for ts, price in _daily_prices(_T0, 25):
        t.record_price("AAPL", ts, price)
    t.evaluate_signal("AAPL", _T0)  # resolves outcomes
    t._fills["AAPL"] = [_fill()]  # exercise the fill table (persistence reads _fills)


@pytest.fixture(autouse=True)
def _reset():
    pytest.importorskip("sqlalchemy")
    for mod in (reg.reset_model_registry, pt.reset_performance_tracker,
                mlm.reset_model, opt.reset_view_tracker):
        mod()
    yield
    for mod in (reg.reset_model_registry, pt.reset_performance_tracker,
                mlm.reset_model, opt.reset_view_tracker):
        mod()


def _reset_singletons() -> None:
    reg.reset_model_registry()
    pt.reset_performance_tracker()
    mlm.reset_model()
    opt.reset_view_tracker()


# ── SQL backend ──────────────────────────────────────────────────────────────────


class TestSqlStateStore:
    def _url(self, tmp_path) -> str:
        return f"sqlite:///{(tmp_path / 'tf.db').as_posix()}"

    def test_sql_round_trip_restores_full_state(self, tmp_path):
        _populate()
        expected = persistence.dump_payload(NO_PRUNE)
        store = state_store.SqlStateStore(self._url(tmp_path))
        store.save(retention_days=NO_PRUNE)

        _reset_singletons()
        state_store.SqlStateStore(self._url(tmp_path)).restore()

        assert persistence.dump_payload(NO_PRUNE) == expected
        # and semantically meaningful invariants survived
        r = reg.get_model_registry()
        assert r.latest_live().model_id == "m2"
        r.rollback("test")  # rollback history round-tripped
        assert r.latest_live().model_id == "m1"
        assert len(pt.get_performance_tracker().outcomes()) == 4

    def test_backend_parity_json_vs_sql(self, tmp_path):
        # Save the SAME live state through both backends (one populate — promote()
        # stamps a wall-clock promoted_at, so re-populating would diverge), then
        # restore each independently and confirm they reconstruct identical state.
        _populate()
        expected = persistence.dump_payload(NO_PRUNE)
        state_store.JsonStateStore(tmp_path / "state.json").save(retention_days=NO_PRUNE)
        state_store.SqlStateStore(self._url(tmp_path)).save(retention_days=NO_PRUNE)

        _reset_singletons()
        state_store.JsonStateStore(tmp_path / "state.json").restore()
        via_json = persistence.dump_payload(NO_PRUNE)

        _reset_singletons()
        state_store.SqlStateStore(self._url(tmp_path)).restore()
        via_sql = persistence.dump_payload(NO_PRUNE)

        assert via_json == expected
        assert via_sql == expected  # both backends restore the identical state

    def test_retention_prunes_identically_to_json(self, tmp_path):
        t = pt.get_performance_tracker()
        old = _T0 - timedelta(days=200)
        t.record_prediction(_pred(asof=old), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec")
        t.record_prediction(_pred(), source="ml", sleeve="momentum",
                            regime="trending", execution_regime="normal_exec")
        t.record_price("AAPL", old, 90.0)
        t.record_price("AAPL", _T0, 100.0)

        state_store.SqlStateStore(self._url(tmp_path)).save(retention_days=90)
        _reset_singletons()
        state_store.SqlStateStore(self._url(tmp_path)).restore()

        t2 = pt.get_performance_tracker()
        assert len(t2._predictions["AAPL"]) == 1   # 200-day-old prediction pruned
        assert t2._price_val["AAPL"] == [100.0]    # stale price pruned

    def test_version_mismatch_on_restore_raises(self, tmp_path):
        from sqlalchemy import create_engine, update

        from ops import sql_models as m

        url = self._url(tmp_path)
        _populate()
        state_store.SqlStateStore(url).save(retention_days=NO_PRUNE)
        with create_engine(url).begin() as conn:
            conn.execute(update(m.SchemaMeta).values(version=999))
        with pytest.raises(ValueError):
            state_store.SqlStateStore(url).restore()

    def test_create_all_builds_expected_tables(self, tmp_path):
        from sqlalchemy import create_engine, inspect

        from ops import sql_models as m

        engine = create_engine(self._url(tmp_path))
        m.Base.metadata.create_all(engine)
        tables = set(inspect(engine).get_table_names())
        assert {
            "schema_meta", "model_record", "registry_meta", "prediction",
            "price", "fill", "outcome", "tracker_meta",
        } <= tables


# ── config factory ───────────────────────────────────────────────────────────────


class TestMakeStateStore:
    def test_default_is_json_backend(self):
        assert isinstance(make_state_store(_settings()), state_store.JsonStateStore)

    def test_sqlite_backend_selected_with_derived_url(self, tmp_path):
        s = _settings(persistence={"backend": "sqlite", "state_dir": str(tmp_path)})
        store = make_state_store(s)
        assert isinstance(store, state_store.SqlStateStore)
        assert store._url == f"sqlite:///{(Path(str(tmp_path)) / 'tradingengineresearch.db').as_posix()}"

    def test_explicit_database_url_honored(self, tmp_path):
        url = f"sqlite:///{(tmp_path / 'custom.db').as_posix()}"
        s = _settings(persistence={"backend": "sqlite", "database_url": url})
        assert make_state_store(s)._url == url

    def test_unknown_backend_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _settings(persistence={"backend": "mongodb"})


# ── Alembic initial migration (skipped if alembic is not installed) ──────────────


class TestAlembicMigration:
    def test_initial_migration_builds_schema(self, tmp_path, monkeypatch):
        pytest.importorskip("alembic")
        from alembic import command
        from alembic.config import Config
        from sqlalchemy import create_engine, inspect

        url = f"sqlite:///{(tmp_path / 'mig.db').as_posix()}"
        monkeypatch.setenv("ENGINE_PERSISTENCE__DATABASE_URL", url)
        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
        command.upgrade(cfg, "head")
        assert "model_record" in inspect(create_engine(url)).get_table_names()
