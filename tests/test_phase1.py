"""
Phase 1 Tests — Data Contracts and Feature Store
=================================================
Covers all test targets specified in the build instructions:

  - All 9 data contract models instantiate correctly with valid data
  - asof_timestamp validation raises ValueError in LIVE mode when missing
  - stale_flag=True blocks risk-taking in LIVE mode
  - validate_train_serve_parity() returns correct mismatch dicts
  - get_features() raises when stale threshold is exceeded in LIVE mode
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

import pandas as pd

from data import feature_store

from data.data_contracts import (
    MarketBar, QuoteSnapshot, NewsEvent, InsiderEvent, FeatureRow,
    PredictionRow, OrderIntent, FillEvent, RiskEvent,
)
from data.feature_store import (
    get_features, validate_train_serve_parity, feature_freshness_report,
    schema_hash, _register_features, _clear_store, FEATURE_SCHEMA_VERSION,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(minutes=2)
STALE = NOW - timedelta(hours=2)


def _fresh_bar(**kwargs) -> MarketBar:
    defaults = dict(
        symbol="AAPL", open=150.0, high=155.0, low=149.0, close=153.0,
        volume=1_000_000.0, event_timestamp=PAST, ingest_timestamp=PAST,
        asof_timestamp=PAST, source="test", freshness_seconds=120.0, stale_flag=False,
    )
    return MarketBar(**(defaults | kwargs))


def _fresh_feature_row(**kwargs) -> FeatureRow:
    defaults = dict(
        symbol="AAPL", asof_timestamp=PAST,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features={"close_1d": 153.0, "returns_1d": 0.01, "vol_realised_20d": 0.20},
        freshness_flags={"close_1d": False, "returns_1d": False, "vol_realised_20d": False},
        missing_count=0,
    )
    return FeatureRow(**(defaults | kwargs))


# ── 1. All 9 models instantiate with valid data ───────────────────────────────

class TestDataContractsInstantiate:

    def test_market_bar(self):
        bar = _fresh_bar()
        assert bar.symbol == "AAPL"
        assert bar.close == 153.0

    def test_quote_snapshot(self):
        q = QuoteSnapshot(
            symbol="AAPL", bid=152.9, ask=153.1, bid_size=100.0, ask_size=200.0,
            event_timestamp=PAST, asof_timestamp=PAST,
            freshness_seconds=10.0, stale_flag=False,
        )
        assert q.bid < q.ask

    def test_news_event(self):
        n = NewsEvent(
            headline="AAPL beats earnings", symbols_mentioned=["AAPL"],
            source="reuters", event_timestamp=PAST, ingest_timestamp=PAST,
            age_minutes=5.0, stale_flag=False,
        )
        assert "AAPL" in n.symbols_mentioned

    def test_insider_event(self):
        i = InsiderEvent(
            symbol="MSFT", insider_name="Jane Smith", transaction_code="P",
            amount_usd=250_000.0, event_timestamp=PAST, age_days=1.0, stale_flag=False,
        )
        assert i.amount_usd > 0

    def test_feature_row(self):
        f = _fresh_feature_row()
        assert f.feature_schema_version == FEATURE_SCHEMA_VERSION
        assert f.missing_count == 0

    def test_prediction_row(self):
        p = PredictionRow(
            symbol="AAPL", asof_timestamp=PAST, model_version="v6.0",
            expected_return=0.005, risk_estimate=0.15,
            p_positive=0.60, p_tail_loss=0.08, confidence=0.72,
        )
        assert 0 <= p.p_positive <= 1

    def test_order_intent(self):
        o = OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.05,
            expected_cost_bps=5.0, urgency="NORMAL",
            alpha_half_life_minutes=120, decision_timestamp=NOW,
            model_version="v6.0", regime_state="trending", risk_approved=True,
        )
        assert o.risk_approved

    def test_fill_event(self):
        f = FillEvent(
            order_id="ord-001", symbol="AAPL", qty=100.0,
            fill_price=153.05, decision_price=153.00, arrival_price=153.02,
            slippage_bps=0.33, fill_timestamp=NOW,
        )
        assert f.slippage_bps >= 0

    def test_fill_event_optional_commission(self):
        # §17 cash leg: commission is OPTIONAL — None/absent when the broker reports none
        # (never invented). A reported value must be a non-negative cost.
        base = dict(order_id="ord-001", symbol="AAPL", qty=100.0,
                    fill_price=153.05, decision_price=153.00, arrival_price=153.02,
                    slippage_bps=0.33, fill_timestamp=NOW)
        assert FillEvent(**base).commission is None
        assert FillEvent(**base, commission=1.25).commission == 1.25
        with pytest.raises(ValueError):
            FillEvent(**base, commission=-0.5)      # a negative "cost" would flip the cash sign

    def test_risk_event(self):
        r = RiskEvent(
            event_type="DRAWDOWN_WARNING", severity="AMBER",
            description="Portfolio drawdown exceeded 8% threshold",
            timestamp=NOW, auto_action="REDUCE_EXPOSURE_30PCT",
        )
        assert r.severity == "AMBER"


# ── 2. asof_timestamp missing raises ValueError in LIVE mode ─────────────────

class TestAsofTimestampLiveMode:

    def test_market_bar_missing_asof_live_raises(self):
        bar = _fresh_bar(asof_timestamp=None)
        with pytest.raises(ValueError, match="asof_timestamp is required in LIVE mode"):
            bar.validate_for_mode("LIVE")

    def test_quote_snapshot_missing_asof_live_raises(self):
        q = QuoteSnapshot(
            symbol="AAPL", bid=100.0, ask=100.1, bid_size=100.0, ask_size=100.0,
            event_timestamp=PAST, asof_timestamp=None,
            freshness_seconds=10.0, stale_flag=False,
        )
        with pytest.raises(ValueError, match="asof_timestamp is required in LIVE mode"):
            q.validate_for_mode("LIVE")

    def test_feature_row_missing_asof_live_raises(self):
        f = FeatureRow(
            symbol="AAPL", asof_timestamp=None,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            features={}, freshness_flags={}, missing_count=0,
        )
        with pytest.raises(ValueError, match="asof_timestamp is required in LIVE mode"):
            f.validate_for_mode("LIVE")

    def test_missing_asof_research_mode_ok(self):
        bar = _fresh_bar(asof_timestamp=None)
        bar.validate_for_mode("RESEARCH")   # must not raise

    def test_missing_asof_paper_mode_ok(self):
        bar = _fresh_bar(asof_timestamp=None)
        bar.validate_for_mode("PAPER")      # must not raise


# ── 3. stale_flag=True blocks risk-taking in LIVE mode ────────────────────────

class TestStaleFlagLiveMode:

    def test_stale_market_bar_live_raises(self):
        bar = _fresh_bar(stale_flag=True)
        with pytest.raises(ValueError, match="stale_flag=True"):
            bar.validate_for_mode("LIVE")

    def test_stale_quote_snapshot_live_raises(self):
        q = QuoteSnapshot(
            symbol="AAPL", bid=100.0, ask=100.1, bid_size=100.0, ask_size=100.0,
            event_timestamp=PAST, asof_timestamp=PAST,
            freshness_seconds=10.0, stale_flag=True,
        )
        with pytest.raises(ValueError, match="stale_flag=True"):
            q.validate_for_mode("LIVE")

    def test_stale_feature_row_live_raises(self):
        f = _fresh_feature_row(freshness_flags={"close_1d": True})
        with pytest.raises(ValueError, match="stale features in LIVE mode"):
            f.validate_for_mode("LIVE")

    def test_stale_flag_paper_mode_ok(self):
        bar = _fresh_bar(stale_flag=True)
        bar.validate_for_mode("PAPER")     # must not raise

    def test_stale_flag_research_mode_ok(self):
        bar = _fresh_bar(stale_flag=True)
        bar.validate_for_mode("RESEARCH")  # must not raise


# ── 4. validate_train_serve_parity ────────────────────────────────────────────

class TestTrainServeParity:

    def _make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_perfect_parity_returns_valid(self):
        df = self._make_df({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = validate_train_serve_parity(df, df.copy())
        assert result["is_valid"] is True
        assert not result["mismatched_columns"]["only_in_train"]
        assert not result["mismatched_columns"]["only_in_serve"]

    def test_column_only_in_train_detected(self):
        train = self._make_df({"a": [1.0], "b": [2.0]})
        serve = self._make_df({"a": [1.0]})
        result = validate_train_serve_parity(train, serve)
        assert "b" in result["mismatched_columns"]["only_in_train"]
        assert result["is_valid"] is False

    def test_column_only_in_serve_detected(self):
        train = self._make_df({"a": [1.0]})
        serve = self._make_df({"a": [1.0], "c": [3.0]})
        result = validate_train_serve_parity(train, serve)
        assert "c" in result["mismatched_columns"]["only_in_serve"]
        assert result["is_valid"] is False

    def test_range_violation_detected(self):
        train = self._make_df({"a": [0.0, 1.0]})
        serve = self._make_df({"a": [5.0]})       # 5.0 > train max of 1.0
        result = validate_train_serve_parity(train, serve)
        assert "a" in result["range_violations"]
        assert result["range_violations"]["a"]["above_train_max"] is True
        assert result["is_valid"] is False

    def test_within_range_no_violation(self):
        train = self._make_df({"a": [0.0, 10.0]})
        serve = self._make_df({"a": [5.0]})
        result = validate_train_serve_parity(train, serve)
        assert "a" not in result["range_violations"]


# ── 5. get_features raises when stale threshold exceeded in LIVE mode ─────────

class TestGetFeaturesStaleThreshold:

    def setup_method(self):
        _clear_store()

    def test_fresh_features_returned_research(self):
        row = _fresh_feature_row()
        _register_features(row)
        df = get_features(["AAPL"], NOW, mode="RESEARCH")
        assert "AAPL" in df.index

    def test_fresh_features_live_within_threshold(self):
        row = _fresh_feature_row(asof_timestamp=NOW - timedelta(seconds=60))
        _register_features(row)
        df = get_features(["AAPL"], NOW, mode="LIVE", stale_threshold_seconds=300.0)
        assert "AAPL" in df.index

    def test_stale_features_live_raises(self):
        row = _fresh_feature_row(asof_timestamp=STALE)   # 2 hours old
        _register_features(row)
        with pytest.raises(ValueError, match="stale in LIVE mode"):
            get_features(["AAPL"], NOW, mode="LIVE", stale_threshold_seconds=300.0)

    def test_stale_features_paper_does_not_raise(self):
        row = _fresh_feature_row(asof_timestamp=STALE)
        _register_features(row)
        df = get_features(["AAPL"], NOW, mode="PAPER", stale_threshold_seconds=300.0)
        assert "AAPL" in df.index

    def test_pit_join_enforced(self):
        # Row with asof_timestamp AFTER decision time must not be returned
        future_row = _fresh_feature_row(asof_timestamp=NOW + timedelta(hours=1))
        _register_features(future_row)
        df = get_features(["AAPL"], NOW, mode="RESEARCH")
        assert df.empty or "AAPL" not in df.index

    def test_missing_symbol_returns_empty(self):
        _clear_store()
        df = get_features(["ZZZZ"], NOW, mode="RESEARCH")
        assert "ZZZZ" not in df.index

    def test_most_recent_row_selected_when_multiple(self):
        older = _fresh_feature_row(
            asof_timestamp=NOW - timedelta(minutes=10),
            features={"close_1d": 100.0},
        )
        newer = _fresh_feature_row(
            asof_timestamp=NOW - timedelta(minutes=2),
            features={"close_1d": 150.0},
        )
        _register_features(older)
        _register_features(newer)
        df = get_features(["AAPL"], NOW, mode="RESEARCH")
        assert df.loc["AAPL", "close_1d"] == 150.0

    def test_per_feature_staleness_live_raises_by_default(self):
        # Row is FRESH at the row level (well within stale_threshold_seconds) but a
        # feature (close_1d, per-feature threshold 300s) is past its own tighter limit.
        row = _fresh_feature_row(asof_timestamp=NOW - timedelta(seconds=1000))
        _register_features(row)
        with pytest.raises(ValueError, match="Per-feature staleness"):
            get_features(["AAPL"], NOW, mode="LIVE", stale_threshold_seconds=100_000.0)

    def test_per_feature_staleness_skipped_when_enforce_disabled(self):
        # enforce_per_feature=False (a daily-data run): only the row-level threshold
        # applies, so the same per-feature-stale row is returned without raising.
        row = _fresh_feature_row(asof_timestamp=NOW - timedelta(seconds=1000))
        _register_features(row)
        df = get_features(["AAPL"], NOW, mode="LIVE",
                          stale_threshold_seconds=100_000.0, enforce_per_feature=False)
        assert "AAPL" in df.index


# ── 6. schema_hash and freshness_report ───────────────────────────────────────

class TestHelpers:

    def test_schema_hash_deterministic(self):
        h1 = schema_hash(["a", "b", "c"])
        h2 = schema_hash(["c", "a", "b"])   # order-invariant
        assert h1 == h2

    def test_schema_hash_changes_on_new_feature(self):
        h1 = schema_hash(["a", "b"])
        h2 = schema_hash(["a", "b", "c"])
        assert h1 != h2

    def test_freshness_report_structure(self):
        _clear_store()
        row = _fresh_feature_row()
        _register_features(row)
        report = feature_freshness_report(["AAPL"], NOW)
        assert "AAPL" in report
        for feature_name, info in report["AAPL"].items():
            assert "age_seconds" in info
            assert "stale" in info
            assert isinstance(info["stale"], bool)


# ── 7. Model-level validation ────────────────────────────────────────────────

class TestModelValidation:

    def test_quote_bid_above_ask_raises(self):
        with pytest.raises(Exception):
            QuoteSnapshot(
                symbol="AAPL", bid=101.0, ask=100.0,  # bid > ask
                bid_size=100.0, ask_size=100.0,
                event_timestamp=PAST, asof_timestamp=PAST,
                freshness_seconds=10.0, stale_flag=False,
            )

    def test_prediction_probability_out_of_range_raises(self):
        with pytest.raises(Exception):
            PredictionRow(
                symbol="AAPL", asof_timestamp=PAST, model_version="v6.0",
                expected_return=0.005, risk_estimate=0.15,
                p_positive=1.5,    # > 1 — invalid
                p_tail_loss=0.08, confidence=0.72,
            )

    def test_order_intent_unapproved_live_raises(self):
        o = OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.05,
            expected_cost_bps=5.0, urgency="NORMAL",
            alpha_half_life_minutes=120, decision_timestamp=NOW,
            model_version="v6.0", regime_state="trending", risk_approved=False,
        )
        with pytest.raises(ValueError, match="risk_approved must be True"):
            o.validate_for_mode("LIVE")

    def test_order_intent_unapproved_paper_ok(self):
        o = OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.05,
            expected_cost_bps=5.0, urgency="NORMAL",
            alpha_half_life_minutes=120, decision_timestamp=NOW,
            model_version="v6.0", regime_state="trending", risk_approved=False,
        )
        o.validate_for_mode("PAPER")  # must not raise


class TestSchemaCrossValidation:
    """ROADMAP Phase 5 — the store's feature metadata must cover the model schema."""

    def test_every_model_feature_has_metadata_coverage(self):
        from core.ml_return_model import FEATURE_NAMES
        report = feature_store.validate_schema_against_model(FEATURE_NAMES)
        assert report["missing_freshness"] == []
        assert report["missing_imputation"] == []
        assert report["ok"] is True

    def test_unknown_feature_is_reported(self):
        report = feature_store.validate_schema_against_model(["not_a_real_feature"])
        assert report["missing_freshness"] == ["not_a_real_feature"]
        assert report["missing_imputation"] == ["not_a_real_feature"]
        assert report["ok"] is False

    def test_risk_features_impute_conservatively(self):
        # A missing risk input must never make a name look SAFER than reality:
        # zero idio-vol or zero spread would inflate sizes / understate costs.
        assert feature_store.IMPUTATION_RULES["idiosyncratic_vol"] > 0.0
        assert feature_store.IMPUTATION_RULES["spread_bps"] > 0.0
        assert feature_store.IMPUTATION_RULES["volume_ratio"] == 1.0   # neutral ratio


class TestData1AbsentFeatureImputation:
    """DATA-1: a model feature absent from EVERY retrieved row must still be imputed
    CONSERVATIVELY (via ``required_features``) — never silently zeroed downstream by the
    model's ``features.get(name, 0.0)`` (0 idio-vol → oversized positions; 0 spread →
    understated costs). Previously such a feature never became a column, so the impute
    loop never touched it."""

    def setup_method(self):
        feature_store._clear_store()

    def teardown_method(self):
        feature_store._clear_store()

    def _register_sparse(self, symbol="AAPL"):
        # a row that carries only a couple of features — the RISK inputs are absent
        ts = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        feats = {"signal_score": 0.2, "momentum_12_1": 0.05}
        feature_store._register_features(FeatureRow(
            symbol=symbol, asof_timestamp=ts,
            feature_schema_version=feature_store.FEATURE_SCHEMA_VERSION,
            features=feats, freshness_flags={k: False for k in feats}, missing_count=0,
        ))
        return ts

    def test_absent_risk_features_imputed_conservatively_not_zeroed(self):
        from core.ml_return_model import FEATURE_NAMES
        self._register_sparse()
        asof = datetime(2026, 6, 1, 14, 1, tzinfo=timezone.utc)
        df = feature_store.get_features(["AAPL"], asof, mode="RESEARCH",
                                        required_features=FEATURE_NAMES)
        # every model feature is now a column...
        assert all(name in df.columns for name in FEATURE_NAMES)
        # ...and the risk inputs carry the CONSERVATIVE value (the exact downstream
        # value the model now sees), not the 0.0 it would have defaulted to.
        assert df.loc["AAPL", "idiosyncratic_vol"] == 0.30
        assert df.loc["AAPL", "spread_bps"] == 10.0
        assert df.loc["AAPL", "earnings_proximity_days"] == 10.0
        # a feature that WAS present is untouched
        assert df.loc["AAPL", "signal_score"] == 0.2

    def test_without_required_features_is_backward_compatible(self):
        # the legacy call (no required_features) is unchanged — absent features are NOT
        # fabricated; this is exactly the gap the engine now closes by passing the schema.
        self._register_sparse()
        asof = datetime(2026, 6, 1, 14, 1, tzinfo=timezone.utc)
        df = feature_store.get_features(["AAPL"], asof, mode="RESEARCH")
        assert "idiosyncratic_vol" not in df.columns

    def test_dataless_symbol_is_not_fabricated(self):
        # a symbol with NO row at all must not get a fabricated conservative row — only
        # symbols that already have data get their absent features filled.
        from core.ml_return_model import FEATURE_NAMES
        self._register_sparse("AAPL")
        asof = datetime(2026, 6, 1, 14, 1, tzinfo=timezone.utc)
        df = feature_store.get_features(["AAPL", "ZZZZ"], asof, mode="RESEARCH",
                                        required_features=FEATURE_NAMES)
        assert "AAPL" in df.index and "ZZZZ" not in df.index


class TestLiveFailClosed:
    """ROADMAP Phase 5 — LIVE fails closed on missing symbols and ambiguous times."""

    def setup_method(self):
        feature_store._clear_store()

    def teardown_method(self):
        feature_store._clear_store()

    def _register(self, symbol="AAPL"):
        ts = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        feature_store._register_features(FeatureRow(
            symbol=symbol, asof_timestamp=ts,
            feature_schema_version=feature_store.FEATURE_SCHEMA_VERSION,
            features={"momentum_12_1": 0.05},
            freshness_flags={"momentum_12_1": False}, missing_count=0,
        ))
        return ts

    def test_live_missing_symbol_raises(self):
        ts = self._register("AAPL")
        with pytest.raises(ValueError, match="GOOG"):
            feature_store.get_features(["AAPL", "GOOG"], ts, "LIVE")

    def test_research_missing_symbol_warns_and_continues(self):
        ts = self._register("AAPL")
        df = feature_store.get_features(["AAPL", "GOOG"], ts, "RESEARCH")
        assert list(df.index) == ["AAPL"]              # degraded, not fatal

    def test_live_naive_asof_time_raises(self):
        self._register("AAPL")
        naive = datetime(2026, 6, 1, 14, 0)            # no tzinfo: ambiguous boundary
        with pytest.raises(ValueError, match="timezone"):
            feature_store.get_features(["AAPL"], naive, "LIVE")

    def test_research_naive_asof_time_is_coerced_to_utc(self):
        self._register("AAPL")
        naive = datetime(2026, 6, 1, 14, 0)
        df = feature_store.get_features(["AAPL"], naive, "RESEARCH")
        assert "momentum_12_1" in df.columns


class TestVersionedRetrievalAndDrift:
    """ROADMAP Phase 5 — versioned PIT retrieval + schema-hash drift warnings."""

    def setup_method(self):
        feature_store._clear_store()

    def teardown_method(self):
        feature_store._clear_store()

    def _row(self, version: str, features: dict, symbol: str = "AAPL") -> FeatureRow:
        return FeatureRow(
            symbol=symbol, asof_timestamp=datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc),
            feature_schema_version=version, features=features,
            freshness_flags={k: False for k in features}, missing_count=0,
        )

    def test_get_features_retrieves_a_specific_schema_version(self):
        feature_store._register_features(self._row("v6.0", {"momentum_12_1": 0.05}))
        feature_store._register_features(self._row("v7.0-test", {"momentum_12_1": 0.99}))
        ts = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
        v6 = feature_store.get_features(["AAPL"], ts, "RESEARCH")
        v7 = feature_store.get_features(["AAPL"], ts, "RESEARCH", schema_version="v7.0-test")
        assert v6.loc["AAPL", "momentum_12_1"] == pytest.approx(0.05)
        assert v7.loc["AAPL", "momentum_12_1"] == pytest.approx(0.99)

    def test_schema_drift_is_warned_once_per_new_feature_set(self, caplog):
        import logging as _logging
        feature_store._register_features(self._row("v6.0", {"momentum_12_1": 0.05}))
        with caplog.at_level(_logging.WARNING, logger="data.feature_store"):
            feature_store._register_features(
                self._row("v6.0", {"momentum_12_1": 0.06, "brand_new_feature": 1.0}))
        assert any("schema drift" in r.message.lower() for r in caplog.records)

    def test_same_feature_set_does_not_warn(self, caplog):
        import logging as _logging
        feature_store._register_features(self._row("v6.0", {"momentum_12_1": 0.05}))
        with caplog.at_level(_logging.WARNING, logger="data.feature_store"):
            feature_store._register_features(self._row("v6.0", {"momentum_12_1": 0.07}))
        assert not any("schema drift" in r.message.lower() for r in caplog.records)


class TestDistributionalParity:
    """ROADMAP Phase 5 — train/serve parity beyond min/max: PSI shift detection."""

    def test_identical_distributions_pass(self):
        import numpy as np
        rng = np.random.default_rng(0)
        values = rng.uniform(0, 1, 500)
        train = pd.DataFrame({"x": values})
        serve = pd.DataFrame({"x": rng.permutation(values)})   # same dist, reordered
        result = feature_store.validate_train_serve_parity(train, serve)
        assert result["distribution_shift"] == {}
        assert result["is_valid"] is True

    def test_in_range_but_shifted_distribution_is_caught(self):
        import numpy as np
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"x": rng.uniform(0, 1, 500)})
        # Every serve value inside the train range — the old min/max check passes —
        # but concentrated in the top decile: a major distributional shift.
        serve = pd.DataFrame({"x": rng.uniform(0.93, 0.97, 500)})
        result = feature_store.validate_train_serve_parity(train, serve)
        assert "x" in result["distribution_shift"]
        assert result["distribution_shift"]["x"]["psi"] > 0.25
        assert result["is_valid"] is False


class TestPortfolioAndBrokerState:
    """ROADMAP Phase 5 — PortfolioState / BrokerState contracts (positions + NAV)."""

    _TS = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)

    def _portfolio(self, **kw):
        from data.data_contracts import PortfolioState
        base = dict(asof_timestamp=self._TS, nav_gbp=1_000_000.0, cash_gbp=400_000.0,
                    positions={"AAPL": 3000.0, "MSFT": -500.0},
                    weights={"AAPL": 0.45, "MSFT": -0.08})
        base.update(kw)
        return PortfolioState(**base)

    def _broker(self, **kw):
        from data.data_contracts import BrokerState
        base = dict(broker="IBKR", connected=True, account_id="DU123",
                    asof_timestamp=self._TS, nav_gbp=1_000_000.0, cash_gbp=400_000.0,
                    positions={"AAPL": 3000.0, "MSFT": -500.0})
        base.update(kw)
        return BrokerState(**base)

    def test_portfolio_exposures(self):
        p = self._portfolio()
        assert p.gross_exposure == pytest.approx(0.53)
        assert p.net_exposure == pytest.approx(0.37)
        p.validate_for_mode("LIVE")                        # clean state passes LIVE

    def test_live_blocks_non_positive_nav_and_stale(self):
        with pytest.raises(ValueError, match="NAV"):
            self._portfolio(nav_gbp=0.0).validate_for_mode("LIVE")
        with pytest.raises(ValueError, match="stale"):
            self._portfolio(stale_flag=True).validate_for_mode("LIVE")
        self._portfolio(nav_gbp=0.0).validate_for_mode("RESEARCH")   # research tolerant

    def test_live_requires_connected_broker(self):
        with pytest.raises(ValueError, match="connected"):
            self._broker(connected=False).validate_for_mode("LIVE")
        self._broker(connected=False).validate_for_mode("PAPER")     # fine off-LIVE

    def test_position_divergence_reconciliation(self):
        from data.data_contracts import position_divergence
        p = self._portfolio()
        b = self._broker(positions={"AAPL": 2990.0, "MSFT": -500.0, "GOOG": 10.0})
        div = position_divergence(p.positions, b.positions, tolerance=1.0)
        assert set(div) == {"AAPL", "GOOG"}                 # MSFT matches within tol
        assert div["AAPL"] == {"internal": 3000.0, "broker": 2990.0}
        assert div["GOOG"] == {"internal": 0.0, "broker": 10.0}


class TestContractsFailClosedOnNonFinite:
    """Security review 2026-06-10 — NaN/inf must fail closed, never slip through."""

    _TS = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)

    def test_portfolio_rejects_non_finite_positions_and_weights(self):
        from data.data_contracts import PortfolioState
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                PortfolioState(nav_gbp=1e6, cash_gbp=0.0, positions={"AAPL": bad})
            with pytest.raises(ValueError, match="finite"):
                PortfolioState(nav_gbp=1e6, cash_gbp=0.0, weights={"AAPL": bad})

    def test_broker_rejects_non_finite_money_and_positions(self):
        from data.data_contracts import BrokerState
        with pytest.raises(ValueError, match="finite"):
            BrokerState(broker="IBKR", connected=True, nav_gbp=float("nan"))
        with pytest.raises(ValueError, match="finite"):
            BrokerState(broker="IBKR", connected=True, positions={"AAPL": float("inf")})

    def test_broker_live_gate_requires_positive_nav_and_buying_power(self):
        from data.data_contracts import BrokerState
        base = dict(broker="IBKR", connected=True, asof_timestamp=self._TS)
        with pytest.raises(ValueError, match="NAV"):
            BrokerState(**base).validate_for_mode("LIVE")             # NAV missing
        with pytest.raises(ValueError, match="NAV"):
            BrokerState(**base, nav_gbp=0.0).validate_for_mode("LIVE")
        with pytest.raises(ValueError, match="buying_power"):
            BrokerState(**base, nav_gbp=1e6,
                        buying_power_gbp=-1.0).validate_for_mode("LIVE")
        BrokerState(**base, nav_gbp=1e6).validate_for_mode("LIVE")    # clean passes
        BrokerState(**base).validate_for_mode("PAPER")                # off-LIVE tolerant

    def test_divergence_treats_non_finite_as_divergent(self):
        from data.data_contracts import position_divergence
        # Raw dicts (not via contracts) can still carry NaN — never "matches".
        div = position_divergence({"AAPL": float("nan")}, {"AAPL": float("nan")},
                                  tolerance=1e9)
        assert "AAPL" in div
        with pytest.raises(ValueError, match="tolerance"):
            position_divergence({}, {}, tolerance=float("nan"))
        with pytest.raises(ValueError, match="tolerance"):
            position_divergence({}, {}, tolerance=-1.0)
