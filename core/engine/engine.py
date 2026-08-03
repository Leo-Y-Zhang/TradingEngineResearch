"""
TradingEngineResearch — TradingEngineResearch v6 Decision Engine
=====================================
The integrated 13-step ``_run_cycle()`` pipeline (Part 20 of the master prompt).

This module is the *integration layer*: it owns no quant logic of its own. Each
step delegates to the module built in Phases 1-8 and threads the result into the
next step. The thirteen steps are documented internal boundaries — they are
never merged or reordered.

Mode discipline (Rule 7 / Section 3.3): the TRADING_MODE is explicit at every
point and never inferred. RESEARCH places **no** orders at all; PAPER runs the
full pipeline but submits **zero** live orders (decisions are recorded only);
LIVE is the only mode that may reach a broker, and only for risk-approved orders.

Failure policy: the two critical steps — STEP 1 (ingest/validate) and STEP 10
(pre-trade risk gate) — fail **closed** (a failure blocks new risk-taking).
Non-critical steps degrade gracefully, but every degradation is logged loudly to
the cycle audit trail; nothing is ever silently swallowed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from core import crisis_manager, meta_labeler, ml_return_model, regime_engine, risk_manager
from core.engine import microstructure, optimizer
from data import feature_store
from data.data_contracts import (
    DiscoveredFill,
    FillEvent,
    OrderIntent,
    PredictionRow,
    TradingMode,
    normalize_mode,
)
from execution import execution_engine, tca
from execution.broker_adapter import OrderManagerBrokerAdapter
from execution.order_lifecycle import OrderLifecycle, OrderStatus, RESUBMIT_BLOCKING_STATES, TERMINAL_STATES
from execution.order_manager import OrderManager
from learning import performance_tracker
from ops import audit_log, model_registry, monitoring
from nlp import sentiment_pipeline
from research import alpha_factory
from strategies import (
    carry,
    mean_reversion,
    momentum,
    sentiment,
    stat_arb,
    volatility_model,
    volatility_overlay,
)

logger = logging.getLogger(__name__)

__all__ = ["CycleInputs", "CycleResult", "TradingEngine"]

# Sleeves run in STEP 4. The sentiment sleeve consumes per-symbol news sentiment
# (FinBERT loads lazily inside the scorer, with an offline lexicon fallback, so
# the engine stays importable without the optional transformers stack).
_SLEEVES = {
    "momentum": momentum.generate_signals,
    "mean_reversion": mean_reversion.generate_signals,
    "stat_arb": stat_arb.generate_signals,
    "volatility_overlay": volatility_overlay.generate_signals,
    "carry": carry.generate_signals,
    "sentiment": sentiment.generate_signals,
}

_MODEL_VERSION = ml_return_model.FEATURE_SCHEMA_VERSION if hasattr(ml_return_model, "FEATURE_SCHEMA_VERSION") else "v6.0"


# ── Cycle I/O ────────────────────────────────────────────────────────────────────

@dataclass
class CycleInputs:
    """Everything a single decision cycle consumes. Most fields default so a
    minimal RESEARCH cycle can be constructed for tests and dry-runs."""

    asof_time: datetime
    symbols: list[str]
    prices: pd.DataFrame                                   # columns=symbols, index=datetime
    returns_matrix: Optional[np.ndarray] = None           # T x n, aligned to symbols
    portfolio_returns: Optional[Any] = None               # 1-D return series
    portfolio_values: Optional[Any] = None                # equity curve
    position_pnls: Optional[Any] = None
    overnight_gaps: Optional[Any] = None
    current_weights: dict = field(default_factory=dict)    # symbol -> current weight
    capital_gbp: float = 1_000_000.0
    drawdown_current: float = 0.0
    market_microstructure: dict = field(default_factory=dict)  # symbol -> {spread_bps, adv, price, ofi_norm, ofi_data, volatility}
    engine_returns: dict = field(default_factory=dict)     # optimizer view source
    insider_flows: dict = field(default_factory=dict)      # optimizer view source
    sector_map: dict = field(default_factory=dict)
    dividend_yields: dict = field(default_factory=dict)    # symbol -> trailing dividend yield (carry sleeve)
    news_items: list = field(default_factory=list)         # dicts: headline/text, age_minutes, symbol(s) — sentiment
    betas: Optional[list] = None
    hours_to_event: Optional[float] = None
    minutes_to_close: float = 390.0
    contracts_to_validate: list = field(default_factory=list)  # objects exposing validate_for_mode(mode)
    kill_context: dict = field(default_factory=dict)


@dataclass
class CycleResult:
    """The full, auditable output of one ``_run_cycle()`` call."""

    mode: str
    asof_time: datetime
    blocked: bool                                         # STEP 1/10 hard block on new risk-taking
    regime_label: str
    regime_probs: dict
    crisis: dict                                          # CrisisStatus.as_dict()
    execution_regime: str
    vol_forecasts: dict
    signal_scores: dict
    predictions: dict                                     # symbol -> 5-tuple
    decisions: dict                                       # symbol -> TradeDecision
    optimizer_result: dict
    risk_snapshot: dict
    target_weights: dict = field(default_factory=dict)   # risk-approved book (backtester / monitoring)
    achieved_weights: Optional[dict] = None              # held book reconciled from FILLS (what we actually hold)
    order_intents: list = field(default_factory=list)
    child_order_plans: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    execution_reports: list = field(default_factory=list)
    live_orders_submitted: int = 0                        # MUST be 0 unless mode == LIVE
    monitoring_snapshot: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    audit: list = field(default_factory=list)            # one entry per step


# ── Engine ───────────────────────────────────────────────────────────────────────

class TradingEngine:
    """Runs the integrated v6 decision pipeline for a single explicit mode."""

    def __init__(
        self,
        mode: str = "RESEARCH",
        capital_gbp: float = 1_000_000.0,
        broker: Any = None,
        stale_threshold_seconds: float = 300.0,
        enforce_per_feature_freshness: bool = True,
        audit_log_path: Optional[str] = None,
        baseline_deploy_enabled: bool = True,
        baseline_in_crisis: bool = False,
        target_vol: Optional[float] = None,
        max_gross_leverage: float = 1.0,
        max_lever_up_step: float | None = None,
        max_position_weight: Optional[float] = None,
        cvar_limit: Optional[float] = None,
        signal_tilt_strength: float = 5e-4,
        fill_impact_coef: float = 10.0,
        fill_max_participation: Optional[float] = 0.10,
        max_retained_terminal_orders: int = 5000,
    ) -> None:
        self.mode = normalize_mode(mode)
        self.capital_gbp = float(capital_gbp)
        self.broker = broker
        self.stale_threshold_seconds = float(stale_threshold_seconds)
        # Per-feature LIVE freshness guard (intraday-tuned); set false for a daily-data run so only
        # the row-level stale_threshold_seconds applies (see core.config for the rationale).
        self.enforce_per_feature_freshness = bool(enforce_per_feature_freshness)
        # Long-biased baseline deployment (returns fix): deploy the equilibrium +
        # signal tilt when ML admits nothing, instead of sitting in cash. Stays in
        # cash during a detected crisis unless baseline_in_crisis is set.
        self.baseline_deploy_enabled = bool(baseline_deploy_enabled)
        self.baseline_in_crisis = bool(baseline_in_crisis)
        # Risk budget (returns aggressiveness): None/1.0 → the optimizer's conservative
        # constants. The run-loop/backtester pass aggressive values to push returns.
        self.target_vol = target_vol
        self.max_gross_leverage = float(max_gross_leverage)
        # OPT-1: None = unchanged (the scaler may reach full leverage in one step).
        self.max_lever_up_step = (
            None if max_lever_up_step is None else float(max_lever_up_step)
        )
        self.max_position_weight = max_position_weight
        self.cvar_limit = cvar_limit
        self.signal_tilt_strength = float(signal_tilt_strength)
        # EXEC-4: PAPER fill realism. `fill_impact_coef` = bps of square-root market
        # impact per unit sqrt(participation), added to the half-spread; `fill_max_participation`
        # caps how much of a name's ADV one cycle can take (the remainder is unfilled →
        # fill_rate < 1). 0.0 / None reproduce the old frictionless full-at-half-spread sim.
        self.fill_impact_coef = float(fill_impact_coef)
        self.fill_max_participation = fill_max_participation
        # Durable cycle audit trail (spec STEP 13) — opt-in: the PAPER/LIVE
        # run-loop sets a path; replays/tests default to zero disk I/O.
        self.audit_log_path = audit_log_path
        self._refit_thread: Optional[threading.Thread] = None
        # Phase 6(b): the §15 order-lifecycle safety machine for the LIVE execute path.
        # Built lazily on the FIRST live submit (so RESEARCH/PAPER never instantiate it)
        # and owned by the engine so uncertain/unknown orders persist ACROSS cycles for
        # reconnect resync. The run-loop's record_cycle stays the sole ledger writer, so
        # this OrderManager runs with ledger=None (no double-recording).
        self._order_lifecycle: Optional[OrderLifecycle] = None
        self._order_manager: Optional[OrderManager] = None
        # LIVE6B-4: bound the in-memory lifecycle's TERMINAL-order retention over a long run.
        self.max_retained_terminal_orders = int(max_retained_terminal_orders)
        # LIVE6B-2: set by the run-loop to gate ALL new LIVE submits until a clean
        # post-restart/reconnect resync has run (fail-closed).
        self.live_submits_blocked = False

    # ── public entry point ───────────────────────────────────────────────────────

    def run_cycle(self, inputs: CycleInputs) -> CycleResult:
        """Public alias for the documented internal pipeline."""
        return self._run_cycle(inputs)

    def _run_cycle(self, inputs: CycleInputs) -> CycleResult:
        """The 13-step v6 decision pipeline, executed in exact order."""
        audit: list[dict] = []

        # STEP 1 — INGEST AND VALIDATE
        ingest = self._step1_ingest_and_validate(inputs)
        audit.append({"step": 1, "name": "ingest_and_validate", **ingest})
        hard_block = bool(ingest["blocked"])

        # STEP 2 — BUILD MARKET STATE
        market = self._step2_build_market_state(inputs)
        audit.append({"step": 2, "name": "build_market_state",
                      "regime": market["regime_label"], "execution_regime": market["execution_regime"],
                      "crisis_level": market["crisis"]["level"]})

        # STEP 3 — VOLATILITY AND RISK FORECASTS
        forecasts = self._step3_forecasts(inputs, market)
        audit.append({"step": 3, "name": "forecasts", **{k: forecasts[k] for k in ("vol_1d", "vol_5d", "vol_ratio")}})

        # Per-symbol news sentiment, computed ONCE per cycle and shared by STEP 4
        # (sentiment sleeve) and STEP 6 (the ML model's sentiment_score feature).
        sentiment_scores = self._sentiment_scores(inputs)

        # STEP 4 — GENERATE RAW SIGNALS
        raw_signals = self._step4_raw_signals(inputs, sentiment_scores)
        audit.append({"step": 4, "name": "raw_signals",
                      "counts": {k: len(v) for k, v in raw_signals.items()}})

        # STEP 5 — SIGNAL HEALTH FILTER
        signal_scores = self._step5_signal_health(inputs, raw_signals, market)
        audit.append({"step": 5, "name": "signal_health", "scored_symbols": len(signal_scores)})

        # STEP 6 — BUILD FEATURES
        features = self._step6_build_features(inputs, sentiment_scores)
        audit.append({"step": 6, "name": "build_features", "rows": int(features.shape[0])})

        # STEP 7 — ML PREDICTION
        predictions = self._step7_ml_predict(inputs, features, market["regime_label"])
        audit.append({"step": 7, "name": "ml_predict", "n_predictions": len(predictions)})

        # STEP 8 — META-LABEL TRADE ADMISSION
        decisions, costs = self._step8_meta_label(inputs, predictions, market)
        admitted = [s for s, d in decisions.items() if d.take_trade]
        audit.append({"step": 8, "name": "meta_label", "candidates": len(decisions), "admitted": admitted})

        # STEP 9 — PORTFOLIO OPTIMIZATION
        opt = self._step9_optimize(inputs, admitted, predictions, signal_scores, market)
        # Item 6: apply the meta-label size_multiplier to the optimised weights HERE
        # (spec STEP 8), so the pre-trade risk gate (STEP 10) evaluates the conviction-
        # scaled book. Applied exactly once — STEP 11 no longer re-multiplies. The
        # multiplier is clipped to [0, 1] by the meta-labeller, so it only de-levers
        # and cannot breach the CVaR limit enforced inside optimise_portfolio.
        # CRITICAL: only ADMITTED names carry a conviction multiplier. Baseline-deploy
        # names also have a (take_trade=False, size_multiplier=0.0) decision, so
        # multiplying them would re-collapse the long-biased baseline book to cash —
        # they keep a multiplier of 1.0.
        admitted_set = set(admitted)
        opt["weights"] = {
            s: float(w) * (float(getattr(decisions.get(s), "size_multiplier", 1.0))
                           if s in admitted_set else 1.0)
            for s, w in opt.get("weights", {}).items()
        }
        audit.append({"step": 9, "name": "optimize",
                      "binding_constraints": opt.get("binding_constraints", []),
                      "turnover": opt.get("turnover_estimate", 0.0)})

        # STEP 10 — PRE-TRADE RISK GATE
        risk_snap, approved, scaled_weights = self._step10_pretrade_risk_gate(
            opt.get("weights", {}), inputs, market, opt
        )
        approved = approved and not hard_block
        audit.append({"step": 10, "name": "pretrade_risk_gate",
                      "approved": approved, "kill_switch_active": risk_snap.kill_switch_active,
                      "active_flags": list(risk_snap.active_flags)})

        # STEP 11 — EXECUTION PLANNING
        order_intents, child_plans = self._step11_execution_planning(
            inputs, scaled_weights, decisions, costs, approved, market
        )
        audit.append({"step": 11, "name": "execution_planning",
                      "order_intents": len(order_intents), "child_orders": len(child_plans)})

        # STEP 12 — EXECUTE AND TCA
        fills, exec_reports, live_count = self._step12_execute_and_tca(
            inputs, order_intents, child_plans, market
        )
        # Reconcile the HELD book from what was actually filled — the caller carries
        # this (not the target) into the next cycle, so partial fills and unfilled
        # orders never silently corrupt the multi-cycle delta accounting.
        achieved_weights = self._achieved_weights(inputs, order_intents, fills)
        audit.append({"step": 12, "name": "execute_and_tca",
                      "fills": len(fills), "live_orders_submitted": live_count})

        # STEP 13 — POST-TRADE LEARNING AND MONITORING
        snap, alerts = self._step13_post_trade_learning(
            inputs, predictions, decisions, fills, risk_snap, market, order_intents, features,
            cycle_stats={
                "stale_feature_count": len(ingest.get("stale_fields", [])),
                "child_plans": child_plans,
                "expected_cost_bps_by_symbol": costs,
                "achieved_weights": achieved_weights,
                "exec_reports": exec_reports,
            },
        )
        audit.append({"step": 13, "name": "post_trade_learning",
                      "alerts": len(alerts), "snapshot_sections": sorted(snap.keys())})

        result = CycleResult(
            mode=self.mode,
            asof_time=inputs.asof_time,
            blocked=hard_block or not approved,
            regime_label=market["regime_label"],
            regime_probs=market["regime_probs"],
            crisis=market["crisis"],
            execution_regime=market["execution_regime"],
            vol_forecasts=forecasts,
            signal_scores=signal_scores,
            predictions=predictions,
            decisions=decisions,
            optimizer_result=opt,
            risk_snapshot=_risk_as_dict(risk_snap),
            target_weights=scaled_weights,
            achieved_weights=achieved_weights,
            order_intents=order_intents,
            child_order_plans=child_plans,
            fills=fills,
            execution_reports=exec_reports,
            live_orders_submitted=live_count,
            monitoring_snapshot=snap,
            alerts=alerts,
            audit=audit,
        )

        if self.audit_log_path:
            _safe(lambda: audit_log.append_cycle_summary(self.audit_log_path, {
                "asof_time": inputs.asof_time.isoformat(),
                "mode": self.mode,
                "regime": result.regime_label,
                "crisis_level": result.crisis.get("level", "NORMAL"),
                "blocked": result.blocked,
                "admitted": len(admitted),
                "order_intents": len(order_intents),
                "fills": len(fills),
                "live_orders_submitted": live_count,
                "alerts": len(alerts),
            }), default=None, what="cycle audit persistence")
        return result

    # ── STEP 1 ────────────────────────────────────────────────────────────────────

    def _step1_ingest_and_validate(self, inputs: CycleInputs) -> dict:
        """Validate every input contract for the active mode and compute feature
        freshness. In LIVE, critical stale data blocks new risk-taking (fail-closed)."""
        input_timestamps: list = []
        stale_fields: list[str] = []
        validation_errors: list[str] = []

        for contract in inputs.contracts_to_validate:
            ts = getattr(contract, "event_timestamp", None) or getattr(contract, "asof_timestamp", None)
            if ts is not None:
                input_timestamps.append(ts)
            if getattr(contract, "stale_flag", False):
                stale_fields.append(type(contract).__name__)
            try:
                contract.validate_for_mode(self.mode)
            except ValueError as exc:
                # In LIVE this is a hard block; in RESEARCH/PAPER it is recorded and skipped.
                validation_errors.append(f"{type(contract).__name__}: {exc}")

        try:
            freshness = feature_store.feature_freshness_report(
                inputs.symbols, inputs.asof_time
            )
        except Exception as exc:  # noqa: BLE001 — degrade, but surface loudly
            logger.warning("STEP1 freshness report unavailable (%s); treating as no freshness data.", exc)
            freshness = {}

        blocked = self.mode == "LIVE" and bool(validation_errors)
        if blocked:
            logger.warning("RISK_EVENT RED: STEP1 blocked LIVE risk-taking — %s", "; ".join(validation_errors))

        return {
            "input_timestamps": [str(t) for t in input_timestamps],
            "stale_fields": stale_fields,
            "validation_errors": validation_errors,
            "freshness_available": bool(freshness),
            "blocked": blocked,
        }

    # ── STEP 2 ────────────────────────────────────────────────────────────────────

    def _step2_build_market_state(self, inputs: CycleInputs) -> dict:
        regime_label, regime_probs = _safe(
            lambda: regime_engine.get_regime_engine().detect_with_probs(inputs.prices),
            default=("mean_reverting", {"calm": 0.5, "stressed": 0.5}),
            what="regime detection",
        )

        vr = self._portfolio_vol_ratio(inputs)
        spread_bps, adv_participation = self._aggregate_microstructure(inputs)

        crisis = _safe(
            lambda: crisis_manager.get_crisis_manager().assess(
                returns_matrix=inputs.returns_matrix,
                portfolio_returns=inputs.portfolio_returns,
                portfolio_values=inputs.portfolio_values,
                position_pnls=inputs.position_pnls,
                spread_bps=spread_bps,
                adv_ratio=adv_participation,
                overnight_gaps=inputs.overnight_gaps,
                hours_to_event=inputs.hours_to_event,
                current_regime=regime_label,
                use_cache=False,
            ),
            default=None,
            what="crisis assessment",
        )
        if crisis is not None:
            crisis_dict = crisis.as_dict()
            defensive = bool(crisis.defensive_mode)
        else:
            # Crisis assessment FAILED (exception swallowed by _safe). Fail CLOSED:
            # treat the cycle as maximally defensive so NO new risk-taking happens
            # while the crisis signal is unavailable. This keeps the long-biased
            # baseline (STEP 9) in cash and forces the admitted optimiser path onto
            # the tightest crisis vol/CVaR envelope — never fail open on lost signal.
            crisis_dict = {
                "level": "UNKNOWN", "defensive_mode": True, "severity_score": 1.0,
                "liquidity_stress_score": 0.0, "signals_fired": ["crisis_assessment_failed"],
            }
            defensive = True

        execution_regime = regime_engine.infer_execution_regime(
            spread_bps=spread_bps, vol_ratio=vr,
            adv_participation=adv_participation, minutes_to_close=inputs.minutes_to_close,
        )

        return {
            "regime_label": regime_label,
            "regime_probs": regime_probs,
            "crisis": crisis_dict,
            "crisis_status_obj": crisis,
            "defensive_mode": defensive,
            "execution_regime": execution_regime,
            "spread_bps": spread_bps,
            "vol_ratio": vr,
        }

    # ── STEP 3 ────────────────────────────────────────────────────────────────────

    def _step3_forecasts(self, inputs: CycleInputs, market: dict) -> dict:
        gjr = har = None
        if inputs.portfolio_returns is not None:
            fitted = _safe(lambda: volatility_model.fit(inputs.portfolio_returns),
                           default=None, what="volatility fit")
            if fitted is not None:
                gjr, har = fitted.get("gjr_params"), fitted.get("har_params")

        vol_1d = _safe(lambda: volatility_model.forecast_vol(gjr, har, horizon=1),
                       default=0.0, what="vol forecast 1d") if gjr is not None else 0.0
        vol_5d = _safe(lambda: volatility_model.forecast_vol(gjr, har, horizon=5),
                       default=0.0, what="vol forecast 5d") if gjr is not None else 0.0

        cov_denoised = None
        if inputs.returns_matrix is not None:
            arr = np.asarray(inputs.returns_matrix, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                cov_denoised = _safe(
                    lambda: volatility_model.rmt_denoise_cov(np.cov(arr, rowvar=False), arr.shape[0]),
                    default=None, what="rmt denoise",
                )

        return {
            "vol_1d": float(vol_1d),
            "vol_5d": float(vol_5d),
            "vol_ratio": float(market["vol_ratio"]),
            "gap_risk_score": float(market.get("crisis", {}).get("gap_risk_score", 0.0)),
            "cov_denoised_available": cov_denoised is not None,
        }

    # ── STEP 4 ────────────────────────────────────────────────────────────────────

    def _step4_raw_signals(self, inputs: CycleInputs,
                           sentiment_scores: Optional[dict] = None) -> dict:
        if sentiment_scores is None:
            sentiment_scores = self._sentiment_scores(inputs)
        out: dict[str, list] = {}
        for name, fn in _SLEEVES.items():
            if name == "carry":
                # carry ranks cross-sectional dividend yields, not just price series.
                out[name] = _safe(
                    lambda: carry.generate_signals(
                        inputs.prices, asof_timestamp=inputs.asof_time,
                        dividend_yields=inputs.dividend_yields),
                    default=[], what="carry signals",
                )
            elif name == "sentiment":
                # sentiment consumes aggregated news scores, not the price series.
                out[name] = _safe(
                    lambda: sentiment.generate_signals(
                        inputs.prices, asof_timestamp=inputs.asof_time,
                        sentiment_scores=sentiment_scores),
                    default=[], what="sentiment signals",
                )
            else:
                out[name] = _safe(lambda fn=fn: fn(inputs.prices, asof_timestamp=inputs.asof_time),
                                  default=[], what=f"{name} signals")
        return out

    # ── STEP 5 ────────────────────────────────────────────────────────────────────

    def _step5_signal_health(self, inputs: CycleInputs, raw_signals: dict, market: dict) -> dict:
        """Apply per-sleeve health weighting and the OFI microstructure veto, then
        collapse to a per-symbol signal score in [-1, 1] for the optimizer tilt."""
        scores: dict[str, float] = {}
        weights: dict[str, float] = {}
        # SIGNALS-5 default-deny: an un-validated sleeve is downweighted to a small floor
        # off-LIVE and DISABLED (0.0) in LIVE — no un-validated sleeve drives real money
        # (golden rule 5). A validated sleeve uses its validated stability_score (and is
        # disabled if it fails selection_rule).
        unvalidated_weight = 0.0 if self.mode == "LIVE" else alpha_factory.UNVALIDATED_SLEEVE_WEIGHT
        for sleeve, signals in raw_signals.items():
            validation = alpha_factory.get_sleeve_validation(sleeve)
            stability = float(validation.stability_score) if validation is not None else 0.0
            healthy = _safe(
                lambda signals=signals, stability=stability, validation=validation:
                    alpha_factory.apply_signal_health(
                        signals, stability_score=stability, validation=validation,
                        unvalidated_weight=unvalidated_weight),
                default=[], what=f"signal_health {sleeve}",
            )
            for sig in healthy:
                if sig.direction == "FLAT" or sig.confidence_proxy <= 0.0:
                    continue
                ofi_norm = self._symbol_ofi(inputs, sig.symbol)
                if not microstructure.ofi_filter_gate(sig.direction, ofi_norm):
                    logger.info("STEP5 OFI veto: %s %s rejected (ofi=%.3f)", sleeve, sig.symbol, ofi_norm)
                    continue
                contribution = sig.raw_score * sig.confidence_proxy
                scores[sig.symbol] = scores.get(sig.symbol, 0.0) + contribution
                weights[sig.symbol] = weights.get(sig.symbol, 0.0) + sig.confidence_proxy
        return {s: float(np.clip(scores[s] / weights[s], -1.0, 1.0)) for s in scores if weights.get(s, 0.0) > 0}

    # ── STEP 6 ────────────────────────────────────────────────────────────────────

    def _step6_build_features(self, inputs: CycleInputs,
                              sentiment_scores: Optional[dict] = None) -> pd.DataFrame:
        features = _safe(
            lambda: feature_store.get_features(
                inputs.symbols, inputs.asof_time, cast(TradingMode, self.mode),
                stale_threshold_seconds=self.stale_threshold_seconds,
                enforce_per_feature=self.enforce_per_feature_freshness,
                # DATA-1: pass the model's full schema so features absent from the data
                # are conservatively imputed in the store, not silently zeroed by the
                # model (0 idio-vol → oversized positions; 0 spread → understated cost).
                required_features=ml_return_model.FEATURE_NAMES,
            ),
            default=pd.DataFrame(index=inputs.symbols),
            what="feature retrieval",
        )
        # Real sentiment_score (model FEATURE_NAMES) from this cycle's news. Only
        # when news was actually supplied — otherwise the store's value (or the
        # model's 0.0 default for an absent feature) is left untouched. Symbols the
        # store has no row for still get their sentiment (other features stay NaN
        # and are dropped per-row; the model defaults absent features to 0.0).
        if inputs.news_items and sentiment_scores:
            features = features.reindex(features.index.union([str(s) for s in inputs.symbols]))
            features["sentiment_score"] = [
                float(sentiment_scores.get(str(s), 0.0)) for s in features.index
            ]
        return features

    # ── STEP 7 ────────────────────────────────────────────────────────────────────

    def _step7_ml_predict(self, inputs: CycleInputs, features: pd.DataFrame, regime_label: str) -> dict:
        model = ml_return_model.get_model()
        feats = [self._feature_row(features, symbol) for symbol in inputs.symbols]
        # Batch predict (Item 5): lets the model set its cross-sectional shrinkage
        # prior from the universe's mean raw view before per-name prediction, instead
        # of shrinking every mu toward a hard-coded 0.0. predict() already returns the
        # safe fallback per symbol on error, so the batch is fail-safe.
        tuples = _safe(
            lambda: model.predict_batch(feats, current_regime=regime_label),
            default=[ml_return_model.SAFE_FALLBACK] * len(feats), what="ml predict batch",
        )
        predictions: dict[str, tuple] = dict(zip(inputs.symbols, tuples))

        # Refit governance: a fitted-but-degraded model refits, and an unfitted
        # model bootstraps its FIRST fit once enough live outcomes have accumulated
        # in the training buffer (otherwise the learning loop can never start).
        if getattr(model, "needs_refit", False) or getattr(model, "ready_for_initial_fit", False):
            self._spawn_refit(model)
        return predictions

    # ── STEP 8 ────────────────────────────────────────────────────────────────────

    def _step8_meta_label(self, inputs: CycleInputs, predictions: dict, market: dict) -> tuple[dict, dict]:
        """Admit/reject each candidate. Returns ``(decisions, expected_cost_bps_by_symbol)``."""
        decisions: dict[str, Any] = {}
        costs: dict[str, float] = {}
        execution_regime = market["execution_regime"]
        regime_label = market["regime_label"]
        for symbol, pred in predictions.items():
            mu, sigma, p_positive, p_tail_loss, confidence = pred
            micro = inputs.market_microstructure.get(symbol, {})
            # NOTE: the 5th arg of ex_ante_cost_model is VOLATILITY, not price. Fall back to
            # the model's predicted sigma when no explicit per-name volatility is supplied.
            expected_cost_bps = _safe(
                lambda micro=micro, mu=mu, sigma=sigma: tca.ex_ante_cost_model(
                    symbol,
                    qty=float(micro.get("target_qty", 1000.0)),
                    side="BUY" if mu >= 0 else "SELL",
                    spread_bps=float(micro.get("spread_bps", 5.0)),
                    volatility=float(micro.get("volatility", abs(sigma))),
                    adv=float(micro.get("adv", 5_000_000.0)),
                    participation=float(micro.get("participation", 0.02)),
                ),
                default=10.0, what=f"ex_ante cost {symbol}",
            )
            costs[symbol] = float(expected_cost_bps)
            decisions[symbol] = meta_labeler.compute(
                mu=mu, sigma=sigma, p_positive=p_positive, p_tail_loss=p_tail_loss,
                confidence=confidence, expected_cost_bps=expected_cost_bps,
                execution_regime=execution_regime,
                crowding_score=float(micro.get("crowding_score", 0.0)),
                liquidity_score=float(micro.get("liquidity_score", 1.0)),
                regime=regime_label,
            )
        return decisions, costs

    # ── STEP 9 ────────────────────────────────────────────────────────────────────

    def _step9_optimize(self, inputs: CycleInputs, admitted: list, predictions: dict,
                        signal_scores: dict, market: dict) -> dict:
        empty = {"weights": {}, "binding_constraints": [], "turnover_estimate": 0.0,
                 "expected_return": 0.0, "expected_risk": 0.0, "expected_cost_bps": 0.0,
                 "cvar_95": 0.0, "capacity_flags": [], "view_sources_active": {}}

        if admitted:
            # High-conviction path: optimise over the ML-admitted names with their views.
            universe, ml_preds, baseline = admitted, {s: predictions[s] for s in admitted}, False
        elif self.baseline_deploy_enabled and signal_scores:
            # Long-biased baseline (returns fix): when ML admits nothing, do NOT sit
            # in cash — deploy the optimizer's CAPM-equilibrium prior tilted by the
            # validated signal sleeves at the vol target. Empty ml_predictions makes
            # Black-Litterman fall back to π; every risk protection (vol/CVaR/caps in
            # the optimizer, the STEP-10 fail-closed gate, drawdown governor) still
            # runs on the produced book. Stay in cash in a detected crisis unless
            # explicitly allowed.
            if market["defensive_mode"] and not self.baseline_in_crisis:
                return {**empty, "baseline_deployment": False}
            # Intersect with the cycle universe: signal_scores keys come from the
            # sleeves (inputs.prices columns); a name outside inputs.symbols has no
            # returns/microstructure and would deploy on a fabricated prior.
            cycle_symbols = set(inputs.symbols)
            universe = sorted(s for s in signal_scores if s in cycle_symbols)
            if not universe:
                return {**empty, "baseline_deployment": False}
            ml_preds, baseline = {}, True
        else:
            return {**empty, "baseline_deployment": False}

        adv = {s: float(inputs.market_microstructure.get(s, {}).get("adv", np.inf)) for s in universe}
        result = _safe(
            lambda: optimizer.optimise_portfolio(
                symbols=universe,
                ml_predictions=ml_preds,
                signal_scores={s: signal_scores.get(s, 0.0) for s in universe},
                engine_returns=inputs.engine_returns,
                insider_flows=inputs.insider_flows,
                capital_gbp=self.capital_gbp,
                returns_matrix=self._returns_for(inputs, universe),
                crisis_mode=market["defensive_mode"],
                crisis_severity=float(market.get("crisis", {}).get("severity_score", 0.0)),
                regime=market["regime_label"],
                w_prev=inputs.current_weights or None,
                adv=adv,
                sector_map=inputs.sector_map or None,
                target_vol=self.target_vol,
                max_gross_leverage=self.max_gross_leverage,
                max_lever_up_step=self.max_lever_up_step,
                max_position_weight=self.max_position_weight,
                cvar_limit_override=self.cvar_limit,
                signal_tilt_strength=self.signal_tilt_strength,
            ),
            default={**empty, "binding_constraints": ["optimizer_error"]},
            what="portfolio optimization",
        )
        result["baseline_deployment"] = baseline
        return result

    # ── STEP 10 ───────────────────────────────────────────────────────────────────

    def _step10_pretrade_risk_gate(self, weights: dict, inputs: CycleInputs,
                                   market: dict, opt: dict) -> tuple[Any, bool, dict]:
        """The pre-trade risk gate. Fails CLOSED: any kill switch or KILL-level
        drawdown blocks all new orders; an active drawdown governor scales exposure."""
        market_state = {
            "betas": inputs.betas,
            "sector_map": inputs.sector_map or None,
            "drawdown_current": inputs.drawdown_current,
            "cvar_95": float(opt.get("cvar_95", 0.0)),
            "illiquidity_score": float(market.get("crisis", {}).get("liquidity_stress_score", 0.0)),
            "kill_context": inputs.kill_context,
            "mode": self.mode,
        }
        # ENGINE-1 / RISK-1: pass the configured hard limits so STEP-10 enforces them
        # INDEPENDENTLY of the optimizer (defense in depth, directive §16). The book is
        # already within these caps after STEP-9, so this is normally a no-op — it only
        # blocks if the optimizer is bypassed or returns an over-limit book.
        market_state["max_gross_leverage"] = float(self.max_gross_leverage)
        if self.cvar_limit is not None:
            market_state["cvar_limit"] = float(self.cvar_limit)
        if self.max_position_weight is not None:
            # Leverage-aware concentration cap (OPT-3 / RISK-1): the optimizer caps each
            # name at max_position_weight PRE-leverage (STEP-9), then the vol scaler levers
            # the whole book up to max_gross_leverage — so a name's legitimate POST-leverage
            # weight reaches max_position_weight × max_gross_leverage. STEP-10 inspects the
            # post-leverage book, so its independent cap must be scaled by the allowed
            # leverage; comparing a post-leverage weight to the pre-leverage cap would halt
            # every normal levered+concentrated cycle (leaving the book under-invested).
            market_state["max_position_weight"] = (
                float(self.max_position_weight) * max(1.0, float(self.max_gross_leverage))
            )
        try:
            snap = risk_manager.get_risk_manager().check_pretrade(weights, market_state)
        except Exception as exc:  # noqa: BLE001 — risk gate fails CLOSED
            logger.warning("RISK_EVENT RED: STEP10 risk gate error (%s); blocking all new orders.", exc)
            snap = risk_manager.RiskSnapshot(
                gross_exposure=0.0, net_exposure=0.0, max_single_name_pct=0.0, max_sector_pct=0.0,
                beta_exposure=0.0, target_vol_utilization=0.0, cvar_utilization=0.0,
                drawdown_current=inputs.drawdown_current, illiquidity_score=0.0,
                kill_switch_active=True, active_flags=["RISK_GATE_ERROR"],
            )

        approved = not snap.kill_switch_active
        if not approved:
            logger.warning("RISK_EVENT RED: kill switch active (%s); halting new orders.",
                           "; ".join(snap.active_flags))

        # Drawdown governor → graduated exposure scale-down.
        scale = _drawdown_scale(risk_manager.check_drawdown(inputs.drawdown_current))
        scaled = {s: float(w) * scale for s, w in (weights or {}).items()} if approved else {}
        return snap, approved, scaled

    # ── STEP 11 ───────────────────────────────────────────────────────────────────

    def _step11_execution_planning(self, inputs: CycleInputs, target_weights: dict, decisions: dict,
                                   costs: dict, approved: bool, market: dict) -> tuple[list, list]:
        """Build OrderIntents from target deltas and slice them into regime-aware child
        orders. Deltas are taken against the CURRENT held book (``inputs.current_weights``).
        The meta-label ``size_multiplier`` has ALREADY been applied to the target weights
        at STEP 9 (before the risk gate, per spec STEP 8), so it is NOT re-applied here
        (doing so would double-count it). RESEARCH never plans real orders; an unapproved
        cycle plans none."""
        held = {str(s): float(w) for s, w in (inputs.current_weights or {}).items()}
        if not approved or self.mode == "RESEARCH" or (not target_weights and not held):
            return [], []
        # LIVE6B-1: fold in-flight (non-terminal) order exposure into the held book so the
        # delta is sized against TRUE exposure (held + pending) and an unconfirmed order is
        # never re-traded. Inert (held unchanged) when nothing is pending.
        for sym, w in self._pending_overlay(inputs).items():
            held[sym] = held.get(sym, 0.0) + w

        execution_regime = market["execution_regime"]
        order_intents: list[OrderIntent] = []
        child_plans: list = []
        # The trade universe is the UNION of the target book and the held book: a
        # held name absent from the targets gets an explicit SELL-to-zero exit —
        # otherwise dropped positions linger forever in LIVE (and exit "for free",
        # untraded, in any book that just copies the targets).
        universe = list(target_weights) + [s for s in held if s not in target_weights]
        for symbol in universe:
            target = float(np.clip(float(target_weights.get(symbol, 0.0)), -1.0, 1.0))
            delta = target - held.get(symbol, 0.0)
            if abs(delta) < 1e-6:
                continue
            micro = inputs.market_microstructure.get(symbol, {})
            price = float(micro.get("price", 100.0))
            intent = OrderIntent(
                symbol=symbol,
                direction="BUY" if delta > 0 else "SELL",
                target_weight=target,
                expected_cost_bps=max(float(costs.get(symbol, 5.0)), 0.0),
                urgency="NORMAL",
                alpha_half_life_minutes=int(micro.get("alpha_half_life_minutes", 60)),
                decision_timestamp=inputs.asof_time,
                model_version=_MODEL_VERSION,
                regime_state=market["regime_label"],
                risk_approved=True,                       # STEP 10 approved this cycle
            )
            order_intents.append(intent)
            plan_state = {
                "execution_regime": execution_regime,
                "capital_gbp": self.capital_gbp,
                "price": price,
                # Size the trade to the DELTA, never the full target: holding 30%
                # with a 10% target is a 20%-of-capital trade. Without an explicit
                # target_qty the scheduler falls back to |target_weight| x capital,
                # re-trading the entire position on every rebalance.
                "target_qty": abs(delta) * self.capital_gbp / price if price > 0 else 0.0,
                "spread_bps": float(micro.get("spread_bps", 5.0)),
                "time_to_close": float(inputs.minutes_to_close),
                "max_participation": float(micro.get("participation", 0.05)),
            }
            child_plans.extend(_safe(
                lambda intent=intent, plan_state=plan_state: execution_engine.schedule_order(intent, plan_state, mode=self.mode),
                default=[], what=f"schedule {symbol}",
            ))
        return order_intents, child_plans

    # ── STEP 12 ───────────────────────────────────────────────────────────────────

    def _step12_execute_and_tca(self, inputs: CycleInputs, order_intents: list,
                                child_plans: list, market: dict) -> tuple[list, list, int]:
        """Submit/manage child orders and run post-trade TCA. RESEARCH and PAPER
        submit ZERO live orders; only LIVE (with a broker) reaches the market."""
        if not child_plans:
            return [], [], 0

        live_count = 0
        fills: list[FillEvent] = []

        if self.mode == "LIVE":
            if self.broker is None:
                logger.warning("RISK_EVENT RED: LIVE cycle with no broker wired; submitting nothing.")
                return [], [], 0
            fills, live_count = self._submit_live_via_lifecycle(inputs, child_plans)
        elif self.mode == "PAPER":
            # Paper fills are simulated locally and are NOT live orders.
            fills = self._simulate_fills(inputs, order_intents, child_plans)

        if not fills:
            return [], [], live_count

        ex_post = _safe(lambda: tca.ex_post_cost_analysis(fills, order_intents),
                        default={}, what="ex_post TCA")
        if ex_post:
            _safe(lambda: tca.update_cost_priors(ex_post), default=None, what="update_cost_priors")

        reports = [{"symbol": f.symbol, "order_id": f.order_id,
                    "slippage_bps": f.slippage_bps,
                    "realized_cost_bps": ex_post.get("total_realized_cost_bps", 0.0)} for f in fills]
        return fills, reports, live_count

    # ── STEP 12 (LIVE): §15 order-lifecycle routing ───────────────────────────────────

    def _submit_live_via_lifecycle(self, inputs: CycleInputs,
                                   child_plans: list) -> tuple[list, int]:
        """Route each LIVE child slice through the §15 OrderManager / OrderLifecycle.

        One ``OrderManager.place()`` per child slice (the unit the broker executes and
        the lifecycle tracks), driving create→approve→submit and applying the broker's
        fills idempotently and clamped to the approved qty. The lifecycle is engine-owned
        (built lazily here) so uncertain/unknown orders persist across cycles for reconnect
        resync. Returns ``(fills, submitted_count)``: ``fills`` are the broker's genuine
        FillEvents (clamped to the lifecycle's ``filled_qty`` on an over/duplicate fill so
        the achieved book can never exceed the approved delta), and ``submitted_count``
        counts slices that actually reached the broker (i.e. not the disconnected
        BROKER_UNKNOWN pre-submit short-circuit). Each ``place`` is wrapped in ``_safe`` so
        one bad slice degrades to the audit instead of aborting the LIVE cycle."""
        # LIVE6B-2: until a clean post-restart/reconnect resync has cleared it, place NO new
        # LIVE order (fail-closed). The lifecycle is left untouched (nothing created).
        if self.live_submits_blocked:
            logger.warning("RISK_EVENT AMBER: LIVE submits blocked (pending resync); "
                           "placing no orders this cycle.")
            return [], 0
        # Lifecycle (stateful) is created once and reused; the adapter is rebuilt every
        # call so a run-loop reconnect that swaps self.broker can never leave a stale
        # adapter pointing at a dead session.
        lifecycle = self._order_lifecycle or OrderLifecycle()
        self._order_lifecycle = lifecycle
        adapter = OrderManagerBrokerAdapter(self.broker)
        manager = self._order_manager
        if manager is None:
            manager = OrderManager(lifecycle, adapter, mode="LIVE", ledger=None)
            self._order_manager = manager
        else:
            manager.broker = adapter

        ts = inputs.asof_time.isoformat()
        fills: list = []
        submitted = 0
        # LIVE6B-1 fail-closed backstop: a symbol with an UNRESOLVED pending order from a PRIOR
        # cycle never gets a new order (no stacking). Computed BEFORE the loop so this cycle's
        # own slices for a symbol do not block each other.
        blocked_symbols = {rec.symbol for rec in lifecycle.all()
                           if rec.status in RESUBMIT_BLOCKING_STATES and rec.symbol}
        for plan in child_plans:
            qty = abs(float(getattr(plan, "qty", 0.0)))
            if qty <= 0.0:
                continue
            symbol = str(getattr(plan, "symbol", ""))
            side = str(getattr(plan, "side", "BUY"))
            if symbol in blocked_symbols:
                logger.warning("RISK_EVENT AMBER: live submit skipped — symbol %s has an "
                               "unresolved pending order (no stacking).", symbol)
                continue
            # Order id keyed on the slice's STABLE economic identity (PIT asof + symbol +
            # side + slice_index), NOT its flat-list position — so a same-asof re-drive
            # with a shifted plan list maps each slice to the SAME id.
            order_id = f"{ts}|{symbol}|{side}|{getattr(plan, 'slice_index', 0)}"
            # No blind re-submission (directive §15): a slice already tracked by the
            # engine-owned lifecycle (e.g. a same-asof replay) is NEVER re-sent. A fresh
            # cycle uses a new asof, so ids are unique and nothing is skipped here.
            if self._is_tracked(order_id):
                logger.warning("RISK_EVENT AMBER: live submit skipped — order %s already "
                               "tracked (no blind resubmit).", order_id)
                continue
            ref_price = float(inputs.market_microstructure.get(symbol, {}).get("price", 0.0) or 0.0)
            placed = _safe(
                lambda order_id=order_id, symbol=symbol, side=side, qty=qty, ref_price=ref_price:
                    manager.place(order_id, symbol, side, qty, ts, ref_price=ref_price),
                default=None, what=f"live submit {symbol}",
            )
            if placed is None:
                continue
            rec, order_fills = placed
            if rec.status != OrderStatus.BROKER_UNKNOWN:
                submitted += 1
            fills.extend(self._clamp_fills(order_fills, float(rec.filled_qty)))
        # LIVE6B-4: bound in-memory growth — evict only fully-resolved TERMINAL orders.
        lifecycle.prune_terminal(self.max_retained_terminal_orders)
        return fills, submitted

    def _is_tracked(self, order_id: str) -> bool:
        """True iff the engine-owned lifecycle already holds ``order_id`` — the replay /
        no-blind-resubmit guard for the LIVE submit path (directive §15)."""
        if self._order_lifecycle is None:
            return False
        try:
            self._order_lifecycle.get(order_id)
            return True
        except KeyError:
            return False

    @staticmethod
    def _clamp_fills(order_fills: list, filled_qty: float) -> list:
        """Clamp a slice's broker fills to the lifecycle's authoritative ``filled_qty``
        (which already deduped/clamped its own record) WITHOUT losing real per-execution
        prices: keep fills in order, accept qty up to ``filled_qty``, trim the boundary
        fill and drop the rest. A normal (non-over) fill set is returned unchanged, so
        TCA and the achieved book see every genuine execution price/slippage."""
        raw_qty = sum(float(f.qty) for f in order_fills)
        if not order_fills or raw_qty <= filled_qty + 1e-9:
            return order_fills
        capped: list = []
        remaining = filled_qty
        for f in order_fills:
            if remaining <= 1e-9:
                break
            take = min(float(f.qty), remaining)
            capped.append(f if take >= float(f.qty) - 1e-9 else f.model_copy(update={"qty": take}))
            remaining -= take
        return capped

    def resync_open_orders(self, broker_open_orders, timestamp: str) -> list:
        """Reconnect resynchronisation entry point (directive §15): reconcile the engine's
        uncertain/unknown/resting LIVE orders against the broker's open-order truth. The
        run-loop's ``_maybe_resync`` calls this each LIVE cycle (LIVE6B-1/3) with
        ``broker.open_orders()``. Returns [] only when there is no lifecycle to reconcile."""
        if self._order_manager is None:
            if self._order_lifecycle is None:
                return []
            # A restart restored the lifecycle but no submit has rebuilt the OrderManager yet —
            # build it now (read-only reconcile; ledger=None) so the post-restart resync GENUINELY
            # runs (else _needs_resync would clear on a no-op and trade an unreconciled book).
            self._order_manager = OrderManager(
                self._order_lifecycle, OrderManagerBrokerAdapter(self.broker), mode="LIVE", ledger=None)
        return self._order_manager.reconcile_open_orders(broker_open_orders, timestamp)

    def drain_discovered_fills(self) -> list[DiscoveredFill]:
        """Return and CLEAR the reconnect-resync-discovered disconnect-fills accumulated on the
        OrderManager outbox (a fill that landed during a disconnect — the broker filled more than
        we locally booked). ``[]`` when no manager exists. The run-loop drains this each LIVE cycle
        after resync to raise durable OPEN reconciliation items (held-book flow); it is NEVER
        auto-applied to the book (operator-gated — directive Section 2/17)."""
        m = self._order_manager
        if m is None:
            return []
        drained = list(m.discovered_fills)
        m.discovered_fills.clear()
        return drained

    def book_reconciled_fill(self, order_id: str, timestamp: str) -> None:
        """Advance an operator-acknowledged resync-discovered fill out of RECONCILIATION_HOLD to
        its true broker state (FILLED if fully filled, else PARTIALLY_FILLED) so its symbol
        unblocks. The run-loop performs the audited ledger + held-book update; this only advances
        the lifecycle. Idempotent (a re-call is a no-op via the transition guard); a missing
        lifecycle/order is a safe no-op."""
        lc = self._order_lifecycle
        if lc is None:
            return
        try:
            rec = lc.get(order_id)
        except KeyError:
            return
        target = (OrderStatus.FILLED if rec.filled_qty >= rec.approved_qty - 1e-9
                  else OrderStatus.PARTIALLY_FILLED)
        lc.transition(order_id, target, timestamp, "operator-reconciled: booked into held book")

    def cancel_reconciled_order(self, order_id: str, timestamp: str) -> None:
        """REJECT path of a resync-discovered fill: the operator declares the discovered fill
        spurious/duplicate, so cancel the parked RECONCILIATION_HOLD order out (-> CANCELLED,
        unfreezing its symbol). No fill is booked. Idempotent (a terminal order is a no-op); a
        missing lifecycle/order is a safe no-op."""
        lc = self._order_lifecycle
        if lc is None:
            return
        try:
            rec = lc.get(order_id)
        except KeyError:
            return
        if rec.status in TERMINAL_STATES:
            return
        lc.transition(order_id, OrderStatus.CANCELLED, timestamp,
                      "operator-reconciled: rejected (spurious/duplicate fill)")

    def has_pending_orders(self) -> bool:
        """True if the engine-owned lifecycle holds any non-terminal (in-flight) order."""
        lc = self._order_lifecycle
        return lc is not None and any(r.status not in TERMINAL_STATES for r in lc.all())

    def snapshot_open_orders(self) -> list:
        """Non-terminal orders to persist across a restart (LIVE6B-2); [] when no lifecycle
        exists (RESEARCH/PAPER and pre-first-LIVE-submit persist nothing)."""
        lc = self._order_lifecycle
        return lc.snapshot_nonterminal() if lc is not None else []

    def restore_open_orders(self, records: list) -> None:
        """Restore persisted non-terminal orders into the engine-owned lifecycle (built lazily
        if needed) on a LIVE restart, so uncertain/working orders are remembered, not forgotten."""
        if not records:
            return
        lc = self._order_lifecycle or OrderLifecycle()
        self._order_lifecycle = lc
        lc.restore(records)

    def _pending_overlay(self, inputs: CycleInputs) -> dict:
        """LIVE6B-1: signed weight of every non-terminal order's UNFILLED residual, valued at
        the cycle's PIT price (the SAME basis STEP-11 sizes on), so the next delta is taken
        against held + in-flight exposure and a pending order is never re-traded. Empty when no
        lifecycle exists or nothing is pending — so RESEARCH/PAPER and no-pending LIVE cycles
        leave the held book byte-identical."""
        lc = self._order_lifecycle
        if lc is None:
            return {}
        capital = self.capital_gbp if self.capital_gbp > 0 else 1.0
        overlay: dict = {}
        for rec in lc.all():
            if rec.status in TERMINAL_STATES:
                continue
            remaining = float(rec.approved_qty) - float(rec.filled_qty)
            if remaining <= 0.0:
                continue
            side = rec.side.upper()
            if side == "BUY":
                sign = 1.0
            elif side == "SELL":
                sign = -1.0
            else:
                logger.warning("pending overlay: order %s has unknown side %r; exposure not folded "
                               "(symbol still blocked by the per-symbol guard).", rec.order_id, rec.side)
                continue
            price = float(inputs.market_microstructure.get(rec.symbol, {}).get("price", 0.0) or 0.0)
            if price <= 0.0:
                logger.warning("pending overlay: no price for pending %s; in-flight exposure not "
                               "folded this cycle (symbol still blocked by the per-symbol guard).",
                               rec.symbol)
                continue
            overlay[rec.symbol] = overlay.get(rec.symbol, 0.0) + sign * remaining * price / capital
        return overlay

    # ── STEP 13 ───────────────────────────────────────────────────────────────────

    def _step13_post_trade_learning(self, inputs: CycleInputs, predictions: dict, decisions: dict,
                                    fills: list, risk_snap: Any, market: dict, order_intents: list,
                                    features: Optional[pd.DataFrame] = None,
                                    cycle_stats: Optional[dict] = None) -> tuple[dict, list]:
        tracker = performance_tracker.get_performance_tracker()

        # Feed this cycle's prices into the tracker — without them no horizon can
        # ever elapse and the learning loop never closes (predictions pile up
        # unresolved). The price timestamp is the bar's own time (PIT-safe).
        for symbol in inputs.symbols:
            _safe(lambda symbol=symbol: self._record_cycle_price(tracker, inputs, symbol),
                  default=None, what=f"record_price {symbol}")

        # Log predictions (with their feature rows — the future training examples)
        # + resolve any elapsed outcomes.
        for symbol, pred in predictions.items():
            mu, sigma, p_positive, p_tail_loss, confidence = pred
            row = _safe(lambda: PredictionRow(
                symbol=symbol, asof_timestamp=inputs.asof_time, model_version=_MODEL_VERSION,
                expected_return=float(mu), risk_estimate=float(abs(sigma)),
                p_positive=float(p_positive), p_tail_loss=float(p_tail_loss), confidence=float(confidence),
            ), default=None, what=f"PredictionRow {symbol}")
            if row is not None:
                feat_row = self._feature_row(features, symbol) if features is not None else {}
                _safe(lambda row=row, market=market, feat_row=feat_row: tracker.record_prediction(
                    row, source="ml", sleeve="blended",
                    regime=market["regime_label"], execution_regime=market["execution_regime"],
                    features=feat_row or None),
                    default=None, what=f"record_prediction {symbol}")
        for fill in fills:
            _safe(lambda fill=fill: tracker.record_fill(fill), default=None, what="record_fill")
        for symbol in inputs.symbols:
            _safe(lambda symbol=symbol: tracker.evaluate_signal(symbol, inputs.asof_time),
                  default=None, what=f"evaluate_signal {symbol}")

        # Monitoring snapshot + alerts — ALL four sections populated from real
        # cycle data (MODEL from the model's own reports, TRADING from this
        # cycle's fills/book, HEALTH from ingest/prediction outcomes).
        state = _safe(
            lambda: self._monitoring_state(inputs, predictions, fills, risk_snap,
                                           market, cycle_stats or {}),
            default={
                "risk_snapshot": risk_snap,
                "crisis_status": market.get("crisis_status_obj"),
                "active_kill_switches": list(risk_snap.active_flags),
            },
            what="monitoring state",
        )
        snap = _safe(lambda: monitoring.snapshot(state), default={}, what="monitoring snapshot")
        alerts = _safe(lambda: monitoring.alert_list(state), default=[], what="alert_list")
        return snap, alerts

    def _monitoring_state(self, inputs: CycleInputs, predictions: dict, fills: list,
                          risk_snap: Any, market: dict, cycle_stats: dict) -> dict:
        """Assemble the full monitoring state for STEP 13 from this cycle's data."""
        model = ml_return_model.get_model()
        cal = model.calibration_report()
        drift = model.drift_report()

        drift_flags = []
        if drift.get("drift_flag"):
            drift_flags.append("feature_drift")
        if drift.get("needs_refit"):
            drift_flags.append("needs_refit")
        tail_err = cal.get("tail_calibration_error")
        if tail_err is not None and tail_err > 0.10:
            drift_flags.append("tail_miscalibration")

        child_plans = cycle_stats.get("child_plans", []) or []
        planned_qty = float(sum(abs(float(getattr(p, "qty", 0.0))) for p in child_plans))
        filled_qty = float(sum(abs(float(f.qty)) for f in fills))
        fill_rate = min(filled_qty / planned_qty, 1.0) if planned_qty > 0 else 0.0
        avg_slippage = (float(np.mean([abs(float(f.slippage_bps)) for f in fills]))
                        if fills else 0.0)

        held = {str(s): float(w) for s, w in (inputs.current_weights or {}).items()}
        achieved = cycle_stats.get("achieved_weights") or held
        turnover = float(sum(abs(achieved.get(s, 0.0) - held.get(s, 0.0))
                             for s in set(achieved) | set(held)))

        expected_costs = cycle_stats.get("expected_cost_bps_by_symbol", {}) or {}
        exec_reports = cycle_stats.get("exec_reports", []) or []
        realized = [float(r.get("realized_cost_bps", 0.0)) for r in exec_reports
                    if isinstance(r, dict)]
        cost_delta = ((float(np.mean(realized)) - float(np.mean(list(expected_costs.values()))))
                      if realized and expected_costs else 0.0)

        return {
            "risk_snapshot": risk_snap,
            "crisis_status": market.get("crisis_status_obj"),
            "active_kill_switches": list(risk_snap.active_flags),
            # MODEL — from the model's own governance reports.
            "rolling_ic_20d": float(drift.get("rolling_ic", 0.0)),
            "calibration_error": float(cal.get("calibration_error") or 0.0),
            "drift_flags_active": drift_flags,
            "last_refit_timestamp": getattr(model, "_fit_time", None),
            "model_version_live": model.active_model_version,
            # TRADING — from this cycle's fills and book.
            "gross_exposure": float(getattr(risk_snap, "gross_exposure", 0.0)),
            "net_exposure": float(getattr(risk_snap, "net_exposure", 0.0)),
            "turnover_today": turnover,
            "fill_rate": fill_rate,
            "avg_slippage_bps": avg_slippage,
            "expected_vs_realized_cost_delta": cost_delta,
            # HEALTH — from ingest + prediction outcomes. ibkr_connected is "as
            # needed": off-LIVE no broker is required, in LIVE it must be wired.
            "stale_feature_count": int(cycle_stats.get("stale_feature_count", 0)),
            "failed_prediction_count": sum(
                1 for p in predictions.values() if tuple(p) == ml_return_model.SAFE_FALLBACK
            ),
            "ibkr_connected": (self.mode != "LIVE") or (self.broker is not None),
            # Shadow/challenger check (spec STEP 13): surface a validated
            # candidate for manual promotion — never auto-promote.
            "shadow_promotion_candidate": getattr(
                _safe(lambda: model_registry.get_model_registry().promotion_candidate(),
                      default=None, what="promotion candidate check"),
                "model_id", None),
        }

    # ── helpers ───────────────────────────────────────────────────────────────────

    def _sentiment_scores(self, inputs: CycleInputs) -> dict:
        """Aggregate this cycle's news into per-symbol sentiment in [-1, 1].
        No news ⇒ empty dict (the sleeve goes FLAT and the feature is untouched)."""
        if not inputs.news_items:
            return {}
        return _safe(
            lambda: sentiment_pipeline.compute_sentiment_scores(
                inputs.news_items, inputs.symbols),
            default={}, what="sentiment scoring",
        )

    def _achieved_weights(self, inputs: CycleInputs, order_intents: list, fills: list) -> dict:
        """Reconcile the held book from the fills actually achieved this cycle:
        ``held + signed fill notional / capital`` per symbol. No fills (RESEARCH,
        blocked, or unfilled orders) ⇒ the held book carries unchanged. Dust
        below 1bp of capital is dropped so spread residuals never churn exits."""
        achieved = {str(s): float(w) for s, w in (inputs.current_weights or {}).items()}
        if fills:
            sign_by_symbol = {
                i.symbol: (1.0 if i.direction == "BUY" else -1.0) for i in order_intents
            }
            capital = self.capital_gbp if self.capital_gbp > 0 else 1.0
            for f in fills:
                sign = sign_by_symbol.get(f.symbol)
                if sign is None:
                    logger.warning("achieved-book: fill for %s has no matching intent; skipped.", f.symbol)
                    continue
                notional = float(f.qty) * float(f.fill_price)
                if not np.isfinite(notional) or notional < 0.0:
                    continue
                achieved[f.symbol] = achieved.get(f.symbol, 0.0) + sign * notional / capital
        return {s: w for s, w in achieved.items() if abs(w) >= 1e-4}

    def _record_cycle_price(self, tracker: Any, inputs: CycleInputs, symbol: str) -> None:
        """Record the symbol's latest PIT price into the performance tracker so
        prediction horizons can elapse and outcomes resolve."""
        if symbol not in inputs.prices.columns:
            return
        series = inputs.prices[symbol].dropna()
        if series.empty:
            return
        price = float(series.iloc[-1])
        if not np.isfinite(price) or price <= 0.0:
            return
        ts = series.index[-1]
        when = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else inputs.asof_time
        tracker.record_price(symbol, when, price)

    def _spawn_refit(self, model: Any) -> None:
        """Run the model's no-arg ``refit`` hook. LIVE runs it on a background
        daemon thread (a refit must never block a live cycle); RESEARCH and PAPER
        run it synchronously so replays and backtests stay deterministic."""
        if self._refit_thread is not None and self._refit_thread.is_alive():
            return
        refit = getattr(model, "refit", None)
        if not callable(refit):
            logger.info("STEP7 ml_return_model flagged needs_refit; no refit hook — deferring.")
            return

        def _run() -> None:
            try:
                refit()
            except Exception as exc:  # noqa: BLE001 — a refit must never crash the cycle
                logger.warning("ML refit failed (%s).", exc)

        if self.mode == "LIVE":
            self._refit_thread = threading.Thread(target=_run, name="ml-refit", daemon=True)
            self._refit_thread.start()
        else:
            _run()

    def _simulate_fills(self, inputs: CycleInputs, order_intents: list, child_plans: list) -> list:
        """Deterministic paper fills from child-order plans (PAPER mode only).

        EXEC-4: models realistic execution rather than frictionless full-at-half-spread.
        (1) PARTIAL FILLS — a cycle cannot take more than ``fill_max_participation`` of a
        name's ADV; the remainder is left unfilled (fill_rate < 1). (2) square-root MARKET
        IMPACT (``fill_impact_coef`` bps × √participation) added to the half-spread, so a
        bigger order fills at a worse price. Both are deterministic (PAPER replay parity)
        and reduce EXACTLY to the old sim when ``fill_impact_coef=0`` and no participation cap.
        """
        fills: list[FillEvent] = []
        for i, plan in enumerate(child_plans):
            qty = float(getattr(plan, "qty", 0.0))
            if qty <= 0.0:
                continue
            micro = inputs.market_microstructure.get(plan.symbol, {})
            price = float(micro.get("price", 100.0))
            spread = float(micro.get("spread_bps", 5.0))
            adv = float(micro.get("adv", 0.0))
            signed = 1.0 if plan.side == "BUY" else -1.0

            # Partial fill: cap the cycle's take at fill_max_participation of ADV.
            filled_qty = qty
            participation = 0.0
            if adv > 0.0:
                participation = (qty * price) / adv
                cap = self.fill_max_participation
                if cap is not None and participation > cap:
                    filled_qty = (cap * adv) / price
                    participation = cap
            if filled_qty <= 0.0:
                continue

            # Half-spread + square-root impact (bps); reduces to the old half-spread at coef 0.
            slippage_bps = spread / 2.0 + self.fill_impact_coef * (participation ** 0.5)
            fill_price = max(price * (1.0 + signed * slippage_bps / 10_000.0), 1e-6)
            fills.append(FillEvent(
                order_id=f"{plan.symbol}-paper-{i}",
                symbol=plan.symbol, qty=filled_qty, fill_price=fill_price,
                decision_price=price, arrival_price=price, slippage_bps=slippage_bps,
                fill_timestamp=inputs.asof_time,
            ))
        return fills

    def _portfolio_vol_ratio(self, inputs: CycleInputs) -> float:
        if inputs.portfolio_returns is None:
            return 1.0
        return _safe(lambda: volatility_model.vol_ratio_current(inputs.portfolio_returns),
                     default=1.0, what="vol_ratio_current")

    def _aggregate_microstructure(self, inputs: CycleInputs) -> tuple[float, float]:
        micros = list(inputs.market_microstructure.values())
        if not micros:
            return 5.0, 0.0
        spread = float(np.mean([float(m.get("spread_bps", 5.0)) for m in micros]))
        part = float(np.mean([float(m.get("participation", 0.0)) for m in micros]))
        return spread, part

    def _symbol_ofi(self, inputs: CycleInputs, symbol: str) -> float:
        micro = inputs.market_microstructure.get(symbol, {})
        if "ofi_norm" in micro:
            return float(micro["ofi_norm"])
        if "ofi_data" in micro:
            return _safe(lambda: microstructure.compute_ofi(micro["ofi_data"]), default=0.0, what="compute_ofi")
        return 0.0

    @staticmethod
    def _feature_row(features: pd.DataFrame, symbol: str) -> dict:
        # Numeric values only: get_features also returns an asof_timestamp column,
        # which must never reach the model's feature vector (it would crash the
        # float() vectorisation — and a timestamp is not a feature).
        if isinstance(features, pd.DataFrame) and symbol in features.index:
            return {
                k: float(v) for k, v in features.loc[symbol].to_dict().items()
                if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v)
            }
        return {}

    @staticmethod
    def _returns_for(inputs: CycleInputs, admitted: list) -> Optional[np.ndarray]:
        if inputs.returns_matrix is None:
            return None
        arr = np.asarray(inputs.returns_matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(inputs.symbols):
            return None
        idx = [inputs.symbols.index(s) for s in admitted if s in inputs.symbols]
        if len(idx) != len(admitted):
            return None
        return arr[:, idx]


# ── module-level helpers ─────────────────────────────────────────────────────────

def _safe(fn, default, what: str):
    """Run ``fn``; on failure log loudly (never silent) and return ``default``."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — integration-layer graceful degradation
        logger.warning("Cycle step degraded — %s failed (%s); using fallback.", what, exc)
        return default


def _drawdown_scale(level: str) -> float:
    """Exposure multiplier for the drawdown-governor level (STEP 10)."""
    return {"NORMAL": 1.0, "SOFT": 0.80, "MEDIUM": 0.70, "HARD": 0.40, "KILL": 0.0}.get(level, 1.0)


def _risk_as_dict(snap: Any) -> dict:
    return {
        "gross_exposure": snap.gross_exposure,
        "net_exposure": snap.net_exposure,
        "max_single_name_pct": snap.max_single_name_pct,
        "target_vol_utilization": snap.target_vol_utilization,
        "cvar_utilization": snap.cvar_utilization,
        "drawdown_current": snap.drawdown_current,
        "kill_switch_active": snap.kill_switch_active,
        "hard_stop": getattr(snap, "hard_stop", False),
        "active_flags": list(snap.active_flags),
    }
