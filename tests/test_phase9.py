"""
Phase 9 Tests — TradingEngineResearch Engine Integration (13-step ``_run_cycle``)
=================================================================
Covers the Phase 9 acceptance targets:

  - All 13 v6 pipeline steps run, in exact order
  - STEP 1 (ingest + validate) blocks LIVE risk-taking on stale data
  - STEP 10 (pre-trade risk gate) fails CLOSED: a kill switch / KILL drawdown
    halts new orders; an active drawdown governor scales exposure down
  - STEP 13 (post-trade learning) produces a 4-section monitoring snapshot with
    only valid alert severities
  - RESEARCH end-to-end: full pipeline runs, NO orders placed
  - PAPER smoke: full pipeline runs, ZERO live orders, decisions recorded
  - Mode gate: an admitted trade reaches the market only in LIVE (with a broker)
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core import crisis_manager, ml_return_model, regime_engine, risk_manager
from core.engine import engine as eng
from core.engine import optimizer
from core.engine.engine import CycleInputs, TradingEngine
from research.validation import ValidationResult
from data.data_contracts import FillEvent, MarketBar
from execution import tca
from learning import performance_tracker
from ops import model_registry

_T = datetime(2025, 10, 28, 14, 0, tzinfo=timezone.utc)
_SYMBOLS = ["AAPL", "MSFT", "GOOG"]


# ── helpers ──────────────────────────────────────────────────────────────────────

def _prices(symbols: list[str] = _SYMBOLS, n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))) for s in symbols},
        index=idx,
    )


def _inputs(**overrides) -> CycleInputs:
    prices = overrides.pop("prices", _prices())
    rets = prices.pct_change().dropna().to_numpy()
    base = dict(
        asof_time=_T,
        symbols=list(prices.columns),
        prices=prices,
        returns_matrix=rets,
        portfolio_returns=rets.mean(axis=1),
        portfolio_values=(1.0 + rets.mean(axis=1)).cumprod(),
        capital_gbp=1_000_000.0,
        market_microstructure={
            s: {"spread_bps": 6.0, "adv": 2.0e7, "price": float(prices[s].iloc[-1]),
                "participation": 0.02}
            for s in prices.columns
        },
    )
    base.update(overrides)
    return CycleInputs(**base)


def _fresh_bar(stale_flag: bool = False) -> MarketBar:
    return MarketBar(
        symbol="AAPL", open=100.0, high=101.0, low=99.0, close=100.5, volume=1_000_000.0,
        event_timestamp=_T, ingest_timestamp=_T, asof_timestamp=_T,
        source="test", freshness_seconds=10.0, stale_flag=stale_flag,
    )


class _MockBroker:
    """Records submissions and returns a fill per child order (LIVE only)."""

    connected = True                      # BrokerProtocol requires it (the §15 lifecycle probes it)

    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, child_plans: list, mode: str) -> list:
        self.submitted.extend(child_plans)
        return [
            FillEvent(order_id=f"{p.symbol}-live-{i}", symbol=p.symbol, qty=max(float(p.qty), 1.0),
                      fill_price=100.0, decision_price=100.0, arrival_price=100.0,
                      slippage_bps=2.0, fill_timestamp=_T)
            for i, p in enumerate(child_plans) if float(p.qty) > 0.0
        ]


def _reset_singletons() -> None:
    ml_return_model.reset_model()
    regime_engine.reset_regime_engine()
    crisis_manager.reset_crisis_manager()
    risk_manager.reset_risk_manager()
    tca.reset_tca_model()
    optimizer.reset_view_tracker()
    model_registry.reset_model_registry()
    performance_tracker.reset_performance_tracker()


@pytest.fixture(autouse=True)
def _isolation():
    _reset_singletons()
    yield
    _reset_singletons()


# ── STEP 1 — ingest and validate ──────────────────────────────────────────────────

class TestStep1Ingest:

    def test_live_stale_data_blocks_risk_taking(self):
        e = TradingEngine(mode="LIVE")
        out = e._step1_ingest_and_validate(_inputs(contracts_to_validate=[_fresh_bar(stale_flag=True)]))
        assert out["blocked"] is True
        assert out["stale_fields"] == ["MarketBar"]

    def test_research_does_not_block_on_stale(self):
        e = TradingEngine(mode="RESEARCH")
        out = e._step1_ingest_and_validate(_inputs(contracts_to_validate=[_fresh_bar(stale_flag=True)]))
        assert out["blocked"] is False

    def test_live_clean_data_passes(self):
        e = TradingEngine(mode="LIVE")
        out = e._step1_ingest_and_validate(_inputs(contracts_to_validate=[_fresh_bar(stale_flag=False)]))
        assert out["blocked"] is False
        assert out["validation_errors"] == []


# ── STEP 10 — pre-trade risk gate (fails CLOSED) ───────────────────────────────────

class TestStep10RiskGate:

    def _market(self):
        return {"regime_label": "mean_reverting", "execution_regime": "normal_exec",
                "defensive_mode": False, "crisis": {"liquidity_stress_score": 0.0}}

    def test_normal_drawdown_approves_and_keeps_exposure(self):
        e = TradingEngine(mode="PAPER")
        snap, approved, scaled = e._step10_pretrade_risk_gate(
            {"AAPL": 0.1, "MSFT": 0.1}, _inputs(drawdown_current=0.0), self._market(), {"cvar_95": 0.0}
        )
        assert approved is True
        assert scaled == {"AAPL": 0.1, "MSFT": 0.1}

    def test_kill_level_drawdown_halts_orders(self):
        e = TradingEngine(mode="LIVE")
        snap, approved, scaled = e._step10_pretrade_risk_gate(
            {"AAPL": 0.1}, _inputs(drawdown_current=0.20), self._market(), {"cvar_95": 0.0}
        )
        assert snap.kill_switch_active is True
        assert approved is False
        assert scaled == {}                      # no weights survive a halted gate

    def test_soft_drawdown_scales_exposure_down(self):
        e = TradingEngine(mode="PAPER")
        _snap, approved, scaled = e._step10_pretrade_risk_gate(
            {"AAPL": 1.0}, _inputs(drawdown_current=0.06), self._market(), {"cvar_95": 0.0}
        )
        assert approved is True
        assert scaled["AAPL"] == pytest.approx(0.80)   # SOFT governor → 20% reduction


# ── STEP 13 — post-trade learning + monitoring ─────────────────────────────────────

class TestStep13Monitoring:

    def test_snapshot_has_all_four_sections_and_valid_alerts(self):
        e = TradingEngine(mode="PAPER")
        inp = _inputs()
        market = e._step2_build_market_state(inp)
        risk_snap, _approved, _scaled = e._step10_pretrade_risk_gate({}, inp, market, {"cvar_95": 0.0})
        preds = {s: ml_return_model.SAFE_FALLBACK for s in inp.symbols}
        snap, alerts = e._step13_post_trade_learning(inp, preds, {}, [], risk_snap, market, [])
        assert sorted(snap.keys()) == ["HEALTH", "MODEL", "RISK", "TRADING"]
        assert all(a["severity"] in {"INFO", "WARNING", "AMBER", "RED"} for a in alerts)


# ── STEP 8 — ex-ante cost (volatility slot, not price) ─────────────────────────────

class TestStep8Cost:

    _MARKET = {"execution_regime": "normal_exec", "regime_label": "trending"}

    def test_ex_ante_cost_is_price_invariant_and_realistic(self):
        # The cost model's 5th arg is VOLATILITY, not price: the expected cost must
        # not change when only the share price changes, and must stay realistic.
        e = TradingEngine(mode="PAPER")
        pred = {"AAPL": (0.001, 0.20, 0.6, 0.05, 0.5)}
        lo = _inputs(market_microstructure={"AAPL": {"spread_bps": 6.0, "adv": 2e7, "price": 100.0, "participation": 0.02}})
        hi = _inputs(market_microstructure={"AAPL": {"spread_bps": 6.0, "adv": 2e7, "price": 400.0, "participation": 0.02}})
        _d1, c_lo = e._step8_meta_label(lo, pred, self._MARKET)
        _d2, c_hi = e._step8_meta_label(hi, pred, self._MARKET)
        assert c_lo["AAPL"] == pytest.approx(c_hi["AAPL"])     # price must not affect cost
        assert c_lo["AAPL"] < 20.0                             # ~half-spread + fee + small impact


# ── STEP 11 — deltas against the current held book ─────────────────────────────────

class TestStep11Deltas:

    _MARKET = {"execution_regime": "normal_exec", "regime_label": "trending"}

    def _decision(self):
        return eng.meta_labeler.TradeDecision(
            take_trade=True, size_multiplier=1.0, hold_horizon_override=None,
            rejection_reason=None, expected_net_edge_bps=30.0,
        )

    def test_buy_when_target_above_current(self):
        e = TradingEngine(mode="PAPER")
        intents, _ = e._step11_execution_planning(
            _inputs(current_weights={"AAPL": 0.0}), {"AAPL": 0.10},
            {"AAPL": self._decision()}, {"AAPL": 5.0}, True, self._MARKET,
        )
        assert [i.direction for i in intents] == ["BUY"]

    def test_sell_when_target_below_current(self):
        e = TradingEngine(mode="PAPER")
        intents, _ = e._step11_execution_planning(
            _inputs(current_weights={"AAPL": 0.30}), {"AAPL": 0.10},
            {"AAPL": self._decision()}, {"AAPL": 5.0}, True, self._MARKET,
        )
        assert [i.direction for i in intents] == ["SELL"]      # holding 0.30, target 0.10 → SELL

    def test_engine_keeps_no_cross_cycle_book_state(self):
        e = TradingEngine(mode="RESEARCH")
        r1 = e.run_cycle(_inputs())
        r2 = e.run_cycle(_inputs())
        assert not hasattr(e, "_prev_weights")                 # book is supplied per-cycle, not accumulated
        assert r1.order_intents == [] and r2.order_intents == []


# ── RESEARCH determinism / reproducibility (ROADMAP Phase 2) ───────────────────────

class TestDeterminism:

    def _run_once(self):
        _reset_singletons()
        return TradingEngine(mode="RESEARCH").run_cycle(_inputs())

    def test_research_cycle_is_bit_reproducible(self):
        # Same inputs + a clean singleton reset must reproduce the cycle exactly
        # (fixed-seed RNG, data-stamped audit, deterministic solvers).
        a = self._run_once()
        b = self._run_once()
        assert a.regime_label == b.regime_label
        assert a.optimizer_result.get("weights", {}) == b.optimizer_result.get("weights", {})
        assert a.target_weights == b.target_weights
        assert a.predictions == b.predictions
        assert [s["name"] for s in a.audit] == [s["name"] for s in b.audit]


# ── RESEARCH end-to-end + PAPER smoke ──────────────────────────────────────────────

class TestEndToEnd:

    def test_research_runs_all_13_steps_and_places_no_orders(self):
        res = TradingEngine(mode="RESEARCH").run_cycle(_inputs())
        assert [a["step"] for a in res.audit] == list(range(1, 14))   # exact order, no merge/skip
        assert res.order_intents == []
        assert res.live_orders_submitted == 0
        assert res.mode == "RESEARCH"

    def test_paper_smoke_runs_full_pipeline_with_zero_live_orders(self):
        res = TradingEngine(mode="PAPER").run_cycle(_inputs())
        assert len(res.audit) == 13
        assert res.live_orders_submitted == 0
        assert res.decisions                                        # every candidate recorded a decision
        assert sorted(res.monitoring_snapshot.keys()) == ["HEALTH", "MODEL", "RISK", "TRADING"]


# ── Mode gate with an admitted trade ───────────────────────────────────────────────

def _force_admit(monkeypatch) -> None:
    """Force the meta-labeller to admit every candidate so execution is exercised."""
    decision = eng.meta_labeler.TradeDecision(
        take_trade=True, size_multiplier=0.5, hold_horizon_override=None,
        rejection_reason=None, expected_net_edge_bps=30.0,
    )
    monkeypatch.setattr(eng.meta_labeler, "compute", lambda **kw: decision)


class TestModeGate:

    def test_research_places_no_orders_even_when_admitted(self, monkeypatch):
        _force_admit(monkeypatch)
        res = TradingEngine(mode="RESEARCH").run_cycle(_inputs())
        assert any(d.take_trade for d in res.decisions.values())     # trades WERE admitted
        assert res.order_intents == []                               # but RESEARCH plans none
        assert res.live_orders_submitted == 0

    def test_paper_plans_orders_but_submits_zero_live(self, monkeypatch):
        _force_admit(monkeypatch)
        res = TradingEngine(mode="PAPER").run_cycle(_inputs())
        assert len(res.order_intents) > 0                            # orders planned
        assert res.live_orders_submitted == 0                        # but ZERO live orders
        assert len(res.fills) > 0                                    # paper fills recorded

    def test_live_with_broker_submits_orders(self, monkeypatch):
        _force_admit(monkeypatch)
        broker = _MockBroker()
        res = TradingEngine(mode="LIVE", broker=broker).run_cycle(_inputs())
        assert len(res.order_intents) > 0
        assert res.live_orders_submitted > 0
        assert len(broker.submitted) > 0

    def test_live_without_broker_submits_nothing(self, monkeypatch):
        _force_admit(monkeypatch)
        res = TradingEngine(mode="LIVE", broker=None).run_cycle(_inputs())
        assert res.live_orders_submitted == 0                        # fail-safe: no broker → no orders


# ── Phase 1 engine wiring (size_multiplier ordering, prior, crisis severity) ───────

class TestPhase1EngineWiring:

    def test_size_multiplier_applied_before_risk_gate(self, monkeypatch):
        # Item 6: the meta-label size_multiplier (0.5) must scale the weights the
        # pre-trade risk gate SEES (spec STEP 8), applied exactly ONCE. After the fix
        # the final order targets equal the weights the gate evaluated; previously the
        # gate saw the un-scaled weights and STEP 11 tacked the multiplier on after.
        _force_admit(monkeypatch)                                    # size_multiplier = 0.5
        captured: dict = {}
        rm_singleton = risk_manager.get_risk_manager()
        real = rm_singleton.check_pretrade

        def _spy(weights, market_state):
            captured["weights"] = dict(weights)
            return real(weights, market_state)

        monkeypatch.setattr(rm_singleton, "check_pretrade", _spy)
        res = TradingEngine(mode="PAPER").run_cycle(_inputs(drawdown_current=0.0))
        assert captured.get("weights"), "risk gate was not exercised"
        assert res.order_intents, "no orders were planned"
        targets = {i.symbol: i.target_weight for i in res.order_intents}
        for sym, tgt in targets.items():
            # single application: gate-seen weight == final target (not 2x it)
            assert captured["weights"][sym] == pytest.approx(tgt, abs=1e-9)

    def test_step7_populates_cross_sectional_prior(self):
        # Item 5 wiring: STEP 7 batch-predicts so the model sets its cross-sectional
        # shrinkage prior from the universe's mean raw view, not the hard-coded 0.0.
        ml_return_model.reset_model()
        try:
            model = ml_return_model.get_model()
            rng = np.random.default_rng(0)
            X = rng.standard_normal((200, 18))
            yr = 0.01 * X[:, 2] + rng.normal(0, 0.01, 200)
            yv = 0.02 + 0.005 * np.abs(rng.standard_normal(200))
            model.fit(X, yr, yv)
            TradingEngine(mode="RESEARCH").run_cycle(_inputs())       # 3-symbol universe
            assert model._cross_sectional_prior != 0.0
        finally:
            ml_return_model.reset_model()

    def test_step9_passes_crisis_severity_to_optimiser(self, monkeypatch):
        # Item 7 wiring: STEP 9 threads the continuous crisis severity into the
        # optimiser so the CVaR limit / vol target can tighten gradually.
        _force_admit(monkeypatch)
        captured: dict = {}
        real = optimizer.optimise_portfolio

        def _spy(**kwargs):
            captured.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(optimizer, "optimise_portfolio", _spy)
        TradingEngine(mode="PAPER").run_cycle(_inputs())
        assert "crisis_severity" in captured
        assert isinstance(captured["crisis_severity"], float) and captured["crisis_severity"] >= 0.0

    def test_cycle_result_exposes_risk_approved_target_weights(self, monkeypatch):
        # Phase 2 (harness): the risk-approved book is a first-class CycleResult field
        # so the backtester reads the intended allocation directly (order_intents omit
        # exited / untraded-hold names and are an unreliable book source).
        _force_admit(monkeypatch)                                    # size_multiplier 0.5, admitted
        res = TradingEngine(mode="PAPER").run_cycle(_inputs(drawdown_current=0.0))
        assert isinstance(res.target_weights, dict) and res.target_weights
        targets = {i.symbol: i.target_weight for i in res.order_intents}
        for sym, tgt in targets.items():
            assert res.target_weights[sym] == pytest.approx(tgt, abs=1e-9)


# ── Validation-driven signal health (ROADMAP Phase 3) ──────────────────────────────

def _val_result(**overrides) -> ValidationResult:
    defaults = dict(
        mean_ic=0.05, mean_rank_ic=0.04, sharpe_net=1.2, turnover=0.02, hit_rate=0.55,
        max_drawdown=-0.05, pbo_proxy=0.15, deflated_sharpe_proxy=0.50, cost_drag_bps=5.0,
        stability_score=0.70, deflated_sharpe_ratio=0.99, regime_breakdown={"trending": {"sharpe": 0.9}}, leakage_flags=[],
    )
    return ValidationResult(**(defaults | overrides))


class TestSignalHealthValidationGate:

    def setup_method(self):
        eng.alpha_factory.reset_sleeve_validation()

    def teardown_method(self):
        eng.alpha_factory.reset_sleeve_validation()

    def _sig(self, symbol: str, sleeve: str):
        return eng.alpha_factory.SignalOutput(
            symbol=symbol, direction="BUY", raw_score=0.5, expected_horizon=5,
            decay_half_life=3, confidence_proxy=0.8, sleeve_name=sleeve, asof_timestamp=_T,
        )

    def test_registry_register_get_reset(self):
        af = eng.alpha_factory
        assert af.get_sleeve_validation("momentum") is None
        af.register_sleeve_validation("momentum", _val_result())
        assert af.get_sleeve_validation("momentum") is not None
        af.reset_sleeve_validation()
        assert af.get_sleeve_validation("momentum") is None

    def test_failing_validation_disables_sleeve(self):
        # A sleeve whose registered ValidationResult fails selection_rule is disabled
        # at STEP 5 — its symbols contribute nothing; an un-validated sleeve still
        # contributes (SIGNALS-5: at the default-deny floor, not full weight).
        eng.alpha_factory.register_sleeve_validation("momentum", _val_result(sharpe_net=0.1))
        engine = TradingEngine(mode="RESEARCH")
        raw = {"momentum": [self._sig("AAA", "momentum")],
               "mean_reversion": [self._sig("BBB", "mean_reversion")]}
        scores = engine._step5_signal_health(_inputs(), raw, {})
        assert "AAA" not in scores            # momentum gated off by failed validation
        assert "BBB" in scores                # un-validated sleeve still contributes (floored)

    def test_passing_validation_keeps_sleeve(self):
        eng.alpha_factory.register_sleeve_validation("momentum", _val_result(stability_score=0.9))
        engine = TradingEngine(mode="RESEARCH")
        scores = engine._step5_signal_health(_inputs(), {"momentum": [self._sig("AAA", "momentum")]}, {})
        assert "AAA" in scores


class TestCarrySleeveWiring:

    def test_carry_in_step4_and_uses_dividend_yields(self):
        # The carry sleeve is wired into STEP 4 and tilts toward the high-yield name.
        engine = TradingEngine(mode="RESEARCH")
        raw = engine._step4_raw_signals(_inputs(dividend_yields={"AAPL": 0.04, "MSFT": 0.0, "GOOG": 0.0}))
        assert "carry" in raw
        by = {s.symbol: s for s in raw["carry"]}
        assert by["AAPL"].direction == "BUY"          # highest cross-sectional yield
        assert by["AAPL"].sleeve_name == "carry"

    def test_carry_flat_without_yields(self):
        engine = TradingEngine(mode="RESEARCH")
        raw = engine._step4_raw_signals(_inputs())     # no dividend_yields supplied
        assert all(s.direction == "FLAT" for s in raw["carry"])


class TestSentimentWiring:
    """ROADMAP Phase 3 — NLP sentiment wired end-to-end (news → sleeve + ML feature)."""

    @pytest.fixture(autouse=True)
    def _offline_scorer(self):
        # Force the lexicon fallback so no model download / network is ever attempted.
        from nlp import finbert_scorer
        finbert_scorer.reset_scorer()
        finbert_scorer.get_scorer()._available = False
        yield
        finbert_scorer.reset_scorer()

    def _news(self):
        return [
            {"headline": "AAPL profit surges to record as growth beats", "age_minutes": 5.0,
             "symbol": "AAPL"},
            {"headline": "MSFT warns of losses amid lawsuit and probe", "age_minutes": 5.0,
             "symbol": "MSFT"},
        ]

    def test_step4_sentiment_sleeve_uses_news(self):
        engine = TradingEngine(mode="RESEARCH")
        raw = engine._step4_raw_signals(_inputs(news_items=self._news()))
        assert "sentiment" in raw
        by = {s.symbol: s for s in raw["sentiment"]}
        assert by["AAPL"].direction == "BUY"      # bullish headline
        assert by["MSFT"].direction == "SELL"     # bearish headline
        assert by["GOOG"].direction == "FLAT"     # no news

    def test_step4_sentiment_flat_without_news(self):
        engine = TradingEngine(mode="RESEARCH")
        raw = engine._step4_raw_signals(_inputs())
        assert all(s.direction == "FLAT" for s in raw["sentiment"])

    def test_step6_sentiment_score_feature_is_real(self):
        # With news supplied, the ML feature row carries the computed sentiment_score.
        engine = TradingEngine(mode="RESEARCH")
        inputs = _inputs(news_items=self._news())
        scores = engine._sentiment_scores(inputs)
        features = engine._step6_build_features(inputs, sentiment_scores=scores)
        assert "sentiment_score" in features.columns
        assert features.loc["AAPL", "sentiment_score"] > 0.0
        assert features.loc["MSFT", "sentiment_score"] < 0.0
        assert features.loc["GOOG", "sentiment_score"] == 0.0
        # And it matches what the pipeline computed (single source of truth).
        assert features.loc["AAPL", "sentiment_score"] == pytest.approx(scores["AAPL"])

    def test_full_research_cycle_with_news_runs_clean(self):
        engine = TradingEngine(mode="RESEARCH")
        result = engine.run_cycle(_inputs(news_items=self._news()))
        assert result.live_orders_submitted == 0
        assert [a["step"] for a in result.audit] == list(range(1, 14))


class TestRefitWiring:
    """ROADMAP Phase 4 — the learning loop actually closes inside the engine."""

    def test_step13_records_prices_into_the_tracker(self):
        engine = TradingEngine(mode="RESEARCH")
        inputs = _inputs()
        engine.run_cycle(inputs)
        tracker = performance_tracker.get_performance_tracker()
        last_close = float(inputs.prices["AAPL"].iloc[-1])
        assert tracker._price_asof("AAPL", _T) == pytest.approx(last_close)

    def test_step13_threads_features_to_the_tracker(self):
        from data import feature_store
        from data.data_contracts import FeatureRow
        feature_store._clear_store()
        feature_store._register_features(FeatureRow(
            symbol="AAPL", asof_timestamp=_T,
            feature_schema_version=feature_store.FEATURE_SCHEMA_VERSION,
            features={"momentum_12_1": 0.05, "reversal_5d": -0.01},
            freshness_flags={"momentum_12_1": False, "reversal_5d": False},
            missing_count=0,
        ))
        try:
            engine = TradingEngine(mode="RESEARCH")
            engine.run_cycle(_inputs())
            tracker = performance_tracker.get_performance_tracker()
            records = tracker._predictions.get("AAPL", [])
            assert records, "STEP 13 must record a prediction for AAPL"
            assert records[-1].features is not None        # training-loop input captured
            assert records[-1].features["momentum_12_1"] == pytest.approx(0.05)
        finally:
            feature_store._clear_store()

    def test_refit_is_synchronous_in_research_and_paper(self):
        calls: list[str] = []
        model = ml_return_model.get_model()
        model.refit = lambda: calls.append(__import__("threading").current_thread().name)  # type: ignore[method-assign]
        for mode in ("RESEARCH", "PAPER"):
            engine = TradingEngine(mode=mode)
            engine._spawn_refit(model)
        assert calls == ["MainThread", "MainThread"]       # deterministic replay/backtest

    def test_refit_is_background_in_live(self):
        calls: list[str] = []
        model = ml_return_model.get_model()
        model.refit = lambda: calls.append(__import__("threading").current_thread().name)  # type: ignore[method-assign]
        engine = TradingEngine(mode="LIVE")
        engine._spawn_refit(model)
        assert engine._refit_thread is not None
        engine._refit_thread.join(timeout=5.0)
        assert calls and calls[0] != "MainThread"          # never blocks a LIVE cycle

    def test_step7_triggers_initial_fit_when_buffer_ready(self, monkeypatch):
        model = ml_return_model.get_model()
        monkeypatch.setattr(type(model), "ready_for_initial_fit",
                            property(lambda self: True))
        called = []
        model.refit = lambda: called.append(True)          # type: ignore[method-assign]
        engine = TradingEngine(mode="RESEARCH")
        engine.run_cycle(_inputs())
        assert called == [True]


class TestBookReconciliation:
    """ROADMAP Phase 4 — multi-cycle delta accounting (size to the DELTA, exit
    dropped names explicitly, reconcile the achieved book from fills)."""

    _MARKET = {"execution_regime": "normal_exec", "regime_label": "trending"}

    def _decision(self):
        return eng.meta_labeler.TradeDecision(
            take_trade=True, size_multiplier=1.0, hold_horizon_override=None,
            rejection_reason=None, expected_net_edge_bps=30.0,
        )

    def test_child_orders_are_sized_to_the_delta_not_the_target(self):
        # Holding 30%, target 10% → the trade is 20% of capital, NOT 10%.
        e = TradingEngine(mode="PAPER", capital_gbp=1_000_000.0)
        inputs = _inputs(current_weights={"AAPL": 0.30})
        price = float(inputs.prices["AAPL"].iloc[-1])
        _, child_plans = e._step11_execution_planning(
            inputs, {"AAPL": 0.10}, {"AAPL": self._decision()}, {"AAPL": 5.0},
            True, self._MARKET,
        )
        total_qty = sum(float(p.qty) for p in child_plans if p.symbol == "AAPL")
        expected = 0.20 * 1_000_000.0 / price
        assert total_qty == pytest.approx(expected, rel=0.02)

    def test_dropped_holding_gets_an_explicit_exit_order(self):
        # MSFT is held but absent from the target book → SELL-to-zero intent.
        e = TradingEngine(mode="PAPER")
        intents, _ = e._step11_execution_planning(
            _inputs(current_weights={"AAPL": 0.10, "MSFT": 0.10}),
            {"AAPL": 0.10}, {"AAPL": self._decision()}, {"AAPL": 5.0},
            True, self._MARKET,
        )
        by = {i.symbol: i for i in intents}
        assert "MSFT" in by
        assert by["MSFT"].direction == "SELL"
        assert by["MSFT"].target_weight == pytest.approx(0.0)

    def test_achieved_weights_reconcile_from_fills(self):
        e = TradingEngine(mode="PAPER", capital_gbp=1_000_000.0)
        inputs = _inputs(current_weights={"AAPL": 0.05})
        intents = [eng.OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.10, expected_cost_bps=5.0,
            urgency="NORMAL", alpha_half_life_minutes=60, decision_timestamp=_T,
            model_version="v6.0", regime_state="trending", risk_approved=True,
        )]
        fills = [FillEvent(
            order_id="AAPL-paper-0", symbol="AAPL", qty=500.0, fill_price=100.0,
            decision_price=100.0, arrival_price=100.0, slippage_bps=2.0,
            fill_timestamp=_T,
        )]
        achieved = e._achieved_weights(inputs, intents, fills)
        # 500 shares @ 100 = 50k = 5% of capital, bought on top of the held 5%.
        assert achieved["AAPL"] == pytest.approx(0.10, rel=1e-6)

    def test_unfilled_intent_leaves_the_held_weight_unchanged(self):
        e = TradingEngine(mode="PAPER")
        inputs = _inputs(current_weights={"AAPL": 0.05})
        intents = [eng.OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.10, expected_cost_bps=5.0,
            urgency="NORMAL", alpha_half_life_minutes=60, decision_timestamp=_T,
            model_version="v6.0", regime_state="trending", risk_approved=True,
        )]
        achieved = e._achieved_weights(inputs, intents, [])
        assert achieved["AAPL"] == pytest.approx(0.05)


class TestExec4RealisticFills:
    """EXEC-4: PAPER fills model square-root market impact + partial fills, not the
    frictionless full-at-half-spread sim (so fill_rate / slippage are realistic)."""

    def _plan(self, qty, side="BUY"):
        from types import SimpleNamespace
        return SimpleNamespace(symbol="AAPL", qty=float(qty), side=side)

    def _micro(self, adv=1_000_000.0, price=100.0, spread=6.0):
        return _inputs(market_microstructure={
            "AAPL": {"spread_bps": spread, "adv": adv, "price": price, "participation": 0.02}})

    def test_impact_scales_with_order_size(self):
        e = TradingEngine(mode="PAPER")                     # defaults: coef 10, cap 0.10
        inp = self._micro()
        small = e._simulate_fills(inp, [], [self._plan(100)])    # participation 0.01
        large = e._simulate_fills(inp, [], [self._plan(800)])    # participation 0.08
        assert small[0].slippage_bps < large[0].slippage_bps     # bigger order → worse price
        assert large[0].fill_price > small[0].fill_price > 100.0  # a BUY pays up

    def test_partial_fill_caps_at_max_participation(self):
        e = TradingEngine(mode="PAPER")                     # default cap 0.10 of ADV
        inp = self._micro(adv=1_000_000.0, price=100.0)
        fills = e._simulate_fills(inp, [], [self._plan(2000)])   # wants 20% of ADV
        assert fills[0].qty == pytest.approx(1000.0)             # capped to 10% of ADV → partial

    def test_frictionless_mode_matches_legacy(self):
        # coef 0 + no cap reproduces the old full-at-half-spread sim exactly.
        e = TradingEngine(mode="PAPER", fill_impact_coef=0.0, fill_max_participation=None)
        inp = self._micro(adv=1_000_000.0, price=100.0, spread=6.0)
        fills = e._simulate_fills(inp, [], [self._plan(5000)])
        assert fills[0].qty == pytest.approx(5000.0)             # full fill, no partial
        assert fills[0].slippage_bps == pytest.approx(3.0)       # spread/2, no impact
        assert fills[0].fill_price == pytest.approx(100.0 * (1 + 3.0 / 1e4))

    def test_research_cycle_achieved_equals_current(self):
        e = TradingEngine(mode="RESEARCH")
        result = e.run_cycle(_inputs(current_weights={"AAPL": 0.07}))
        assert result.achieved_weights == {"AAPL": 0.07}     # no orders in RESEARCH


class TestMonitoringPopulated:
    """ROADMAP Phase 4 — STEP 13 populates MODEL/TRADING/HEALTH with real cycle data."""

    def test_model_and_health_sections_from_a_real_cycle(self):
        e = TradingEngine(mode="PAPER")
        result = e.run_cycle(_inputs())
        snap = result.monitoring_snapshot
        # Unfitted model ⇒ every prediction was the safe fallback, and that is
        # surfaced as a HEALTH signal instead of silently reading 0.
        assert snap["HEALTH"]["failed_prediction_count"] == len(_SYMBOLS)
        assert snap["HEALTH"]["ibkr_connected"] is True       # no broker needed off-LIVE
        assert snap["MODEL"]["model_version_live"] == "untrained"
        assert "needs_refit" not in snap["MODEL"]["drift_flags_active"]

    def test_stale_inputs_surface_in_health(self):
        e = TradingEngine(mode="PAPER")
        result = e.run_cycle(_inputs(contracts_to_validate=[_fresh_bar(stale_flag=True)]))
        assert result.monitoring_snapshot["HEALTH"]["stale_feature_count"] == 1

    def test_trading_section_from_fills_and_book(self):
        e = TradingEngine(mode="PAPER")
        inp = _inputs(current_weights={"AAPL": 0.10})
        market = e._step2_build_market_state(inp)
        risk_snap, _a, _s = e._step10_pretrade_risk_gate({}, inp, market, {"cvar_95": 0.0})
        preds = {s: ml_return_model.SAFE_FALLBACK for s in inp.symbols}
        fills = [
            FillEvent(order_id="AAPL-paper-0", symbol="AAPL", qty=500.0, fill_price=100.0,
                      decision_price=100.0, arrival_price=100.0, slippage_bps=2.0,
                      fill_timestamp=_T),
            FillEvent(order_id="AAPL-paper-1", symbol="AAPL", qty=250.0, fill_price=100.0,
                      decision_price=100.0, arrival_price=100.0, slippage_bps=4.0,
                      fill_timestamp=_T),
        ]
        from types import SimpleNamespace
        plans = [SimpleNamespace(symbol="AAPL", qty=1000.0, side="BUY")]
        snap, _alerts = e._step13_post_trade_learning(
            inp, preds, {}, fills, risk_snap, market, [], None,
            cycle_stats={
                "stale_feature_count": 0,
                "child_plans": plans,
                "expected_cost_bps_by_symbol": {"AAPL": 5.0},
                "achieved_weights": {"AAPL": 0.175},
                "exec_reports": [{"realized_cost_bps": 8.0}],
            },
        )
        trading = snap["TRADING"]
        assert trading["fill_rate"] == pytest.approx(0.75)            # 750 of 1000 filled
        assert trading["avg_slippage_bps"] == pytest.approx(3.0)
        assert trading["turnover_today"] == pytest.approx(0.075)      # |0.175 - 0.10|
        assert trading["expected_vs_realized_cost_delta"] == pytest.approx(3.0)


class TestCycleAuditPersistence:
    """ROADMAP Phase 4 — spec STEP 13 'append cycle summary to the audit trail'."""

    def test_engine_appends_one_summary_line_per_cycle(self, tmp_path):
        path = tmp_path / "CYCLE_AUDIT.md"
        e = TradingEngine(mode="RESEARCH", audit_log_path=str(path))
        e.run_cycle(_inputs())
        e.run_cycle(_inputs())
        text = path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.startswith("| 2025-")]
        assert len(lines) == 2
        assert "RESEARCH" in lines[0]

    def test_no_audit_file_without_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)                      # any stray write would land here
        e = TradingEngine(mode="RESEARCH")
        e.run_cycle(_inputs())
        assert list(tmp_path.iterdir()) == []            # no I/O unless opted in

    def test_audit_write_failure_never_breaks_the_cycle(self):
        e = TradingEngine(mode="RESEARCH", audit_log_path=r"Z:\no\such\dir\audit.md")
        result = e.run_cycle(_inputs())                  # must not raise
        assert [a["step"] for a in result.audit] == list(range(1, 14))
