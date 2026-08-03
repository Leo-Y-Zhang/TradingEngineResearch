"""Long-biased baseline deployment (returns fix).

When ML conviction is weak (nothing admitted), the engine must deploy the
optimiser's CAPM-equilibrium prior tilted by the validated signal sleeves —
instead of sitting in cash — while every risk protection still runs. These tests
pin that behaviour at the STEP-9 seam and confirm the safety guards (crisis gate,
config switch, long-only/unlevered, empty-universe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.harness import _reset_engine_state
from core.engine.engine import CycleInputs, TradingEngine
from core.engine.optimizer import capm_equilibrium_returns, optimise_portfolio

SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def _inputs(n: int = 140, seed: int = 3) -> CycleInputs:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = {}
    for i, s in enumerate(SYMBOLS):
        steps = rng.normal(0.0003, 0.011 + 0.002 * i, size=n)
        cols[s] = 100.0 * (1 + 0.1 * i) * np.exp(np.cumsum(steps))
    prices = pd.DataFrame(cols, index=idx)
    rmat = prices.pct_change().dropna().to_numpy()
    last = prices.iloc[-1]
    micro = {s: {"spread_bps": 6.0, "adv": 5.0e7, "price": float(last[s]), "participation": 0.01}
             for s in SYMBOLS}
    return CycleInputs(
        asof_time=idx[-1].to_pydatetime(), symbols=list(SYMBOLS), prices=prices,
        returns_matrix=rmat, market_microstructure=micro,
    )


def _market(defensive: bool = False, severity: float = 0.0) -> dict:
    return {"defensive_mode": defensive, "regime_label": "mean_reverting",
            "crisis": {"severity_score": severity}}


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(11)
    yield


def _gross(book: dict) -> float:
    return float(sum(abs(v) for v in book.values()))


# ── the core fix: deploy a baseline when nothing is admitted ──────────────────────────

def test_step9_deploys_baseline_when_admitted_empty():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True)
    scores = {s: 0.2 for s in SYMBOLS}
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={}, signal_scores=scores, market=_market())
    assert out.get("baseline_deployment") is True
    assert _gross(out["weights"]) > 0.1            # actually deployed, not cash
    assert all(w >= -1e-9 for w in out["weights"].values())          # long-only
    assert sum(out["weights"].values()) <= 1.0 + 1e-6               # unlevered


def test_baseline_disabled_by_config_stays_in_cash():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=False)
    scores = {s: 0.2 for s in SYMBOLS}
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={}, signal_scores=scores, market=_market())
    assert out["weights"] == {}
    assert out.get("baseline_deployment") is False


def test_baseline_gated_off_in_crisis():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True, baseline_in_crisis=False)
    scores = {s: 0.2 for s in SYMBOLS}
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={}, signal_scores=scores,
                              market=_market(defensive=True, severity=0.7))
    assert out["weights"] == {}                    # stays in cash during a crisis
    assert out.get("baseline_deployment") is False


def test_baseline_deploys_in_crisis_when_allowed():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True, baseline_in_crisis=True)
    scores = {s: 0.2 for s in SYMBOLS}
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={}, signal_scores=scores,
                              market=_market(defensive=True, severity=0.7))
    assert out.get("baseline_deployment") is True
    # crisis tightens the vol target → smaller but non-zero book
    assert _gross(out["weights"]) > 0.0
    assert sum(out["weights"].values()) <= 1.0 + 1e-6


def test_empty_signal_scores_yields_empty_book():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True)
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={}, signal_scores={}, market=_market())
    assert out["weights"] == {}                    # no validated names → true no-op


# ── empty-universe guards (no crashes) ────────────────────────────────────────────────

def test_capm_equilibrium_returns_handles_empty_universe():
    out = capm_equilibrium_returns(np.zeros((0, 0)))
    assert out.shape == (0,)                        # no ZeroDivision


def test_optimise_portfolio_empty_symbols_returns_empty_book():
    out = optimise_portfolio(symbols=[])
    assert out["weights"] == {}                     # explicit empty book, no crash


def test_risk_budget_leverage_and_unlevered_default():
    rmat = _inputs().returns_matrix
    scores = {s: 0.3 for s in SYMBOLS}
    # Default (max_gross_leverage=1.0): long-only unlevered, sum <= 1.
    base = optimise_portfolio(symbols=SYMBOLS, signal_scores=scores, returns_matrix=rmat)
    assert sum(base["weights"].values()) <= 1.0 + 1e-6
    # Aggressive (high target_vol + leverage): the vol scaler levers UP past 1.0.
    lev = optimise_portfolio(symbols=SYMBOLS, signal_scores=scores, returns_matrix=rmat,
                             target_vol=0.60, max_gross_leverage=2.0)
    assert sum(lev["weights"].values()) > sum(base["weights"].values())
    assert all(w >= -1e-9 for w in lev["weights"].values())   # still long-only


# ── full-cycle: baseline deploys but never breaches mode discipline ───────────────────

def test_crisis_assessment_failure_fails_closed(monkeypatch):
    # If crisis assessment RAISES, the engine must fail CLOSED: defensive_mode True
    # and the baseline stays in cash (never deploy with the crisis signal unavailable).
    import core.crisis_manager as cm

    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True)

    def _boom(**kwargs):
        raise RuntimeError("crisis detector blew up")

    monkeypatch.setattr(cm.get_crisis_manager(), "assess", _boom)
    market = eng._step2_build_market_state(_inputs())
    assert market["defensive_mode"] is True
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={},
                              signal_scores={s: 0.2 for s in SYMBOLS}, market=market)
    assert out["weights"] == {}
    assert out.get("baseline_deployment") is False


def test_baseline_universe_intersects_inputs_symbols():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True)
    scores = {s: 0.2 for s in SYMBOLS}
    scores["ZZZ_OFF_UNIVERSE"] = 0.9          # a name not in inputs.symbols
    out = eng._step9_optimize(_inputs(), admitted=[], predictions={},
                              signal_scores=scores, market=_market())
    assert "ZZZ_OFF_UNIVERSE" not in out["weights"]   # never deploy an off-universe name
    assert _gross(out["weights"]) > 0.0               # in-universe names still deploy


def test_full_research_cycle_deploys_without_orders():
    eng = TradingEngine(mode="RESEARCH", baseline_deploy_enabled=True)
    result = eng.run_cycle(_inputs())
    # The engine now deploys a long-biased book in RESEARCH instead of sitting in cash...
    assert _gross(result.target_weights) > 0.0
    # ...but RESEARCH still places ZERO live orders (golden rule 1 intact).
    assert result.live_orders_submitted == 0


# ── OPT-1 / OPT-4 risk-appetite levers (default = unchanged behaviour) ────────────────

def test_leverage_ramp_defaults_to_off_and_changes_nothing():
    """OPT-1: the lever must be inert unless the operator turns it on, or it is
    a unilateral change to the risk budget dressed up as a bug fix."""
    rmat = _inputs().returns_matrix
    scores = {s: 0.3 for s in SYMBOLS}
    kw = dict(symbols=SYMBOLS, signal_scores=scores, returns_matrix=rmat,
              target_vol=0.60, max_gross_leverage=2.0)
    assert optimise_portfolio(**kw) == optimise_portfolio(**kw, max_lever_up_step=None)


def test_leverage_ramp_bounds_how_fast_the_book_levers_up():
    """With no previous book the first levered step is capped at 1 + step, so a
    single low-vol reading cannot take the book straight to max_gross_leverage."""
    rmat = _inputs().returns_matrix
    scores = {s: 0.3 for s in SYMBOLS}
    kw = dict(symbols=SYMBOLS, signal_scores=scores, returns_matrix=rmat,
              target_vol=0.60, max_gross_leverage=2.0)

    unramped = optimise_portfolio(**kw)
    ramped = optimise_portfolio(**kw, max_lever_up_step=0.25)

    gross_unramped = sum(abs(w) for w in unramped["weights"].values())
    gross_ramped = sum(abs(w) for w in ramped["weights"].values())
    assert gross_unramped > 1.0                      # non-vacuity: it really did lever
    assert gross_ramped < gross_unramped             # the ramp bit
    assert gross_ramped <= 1.25 + 1e-6               # and bit at the documented bound
    assert "leverage_ramp" in ramped["binding_constraints"]   # and said so


def test_leverage_ramp_never_slows_de_levering():
    """De-levering must stay instant: a ramp that also throttled shrinking would
    make the book slowest to reduce risk exactly when it needs to."""
    rmat = _inputs().returns_matrix
    scores = {s: 0.3 for s in SYMBOLS}
    # target_vol far BELOW realised vol -> the scaler must cut, hard.
    kw = dict(symbols=SYMBOLS, signal_scores=scores, returns_matrix=rmat,
              target_vol=0.01, max_gross_leverage=2.0)
    assert optimise_portfolio(**kw) == optimise_portfolio(**kw, max_lever_up_step=0.01)


def test_view_gate_knobs_default_to_shipped_behaviour():
    """OPT-4: same discipline — the defaults must reproduce what ships today."""
    from core.engine.optimizer import ViewSourceTracker

    default = ViewSourceTracker()
    assert (default.sharpe_floor, default.warmup, default.gate_during_warmup) == (-0.30, 20, False)
    # An unproven source is trusted by default (that is the leniency OPT-4 names)...
    assert default.is_active("ml") is True
    # ...and withheld when the operator asks for the strict posture.
    assert ViewSourceTracker(gate_during_warmup=True).is_active("ml") is False


def test_view_gate_floor_is_configurable():
    """A source that loses money mildly but persistently passes the shipped -0.30
    floor and keeps full influence; raising the floor to 0.0 gates it."""
    from core.engine.optimizer import ViewSourceTracker

    lenient = ViewSourceTracker()
    strict = ViewSourceTracker(sharpe_floor=0.0)
    for tracker in (lenient, strict):
        for i in range(40):
            # Noisy but slightly loss-making: mean -0.001 against a 0.019 spread,
            # i.e. a raw Sharpe near -0.05 — squarely inside the -0.30..0 gap the
            # shipped floor leaves open.
            tracker.record("ml", 1.0, -0.02 if i % 2 == 0 else 0.018)
    sharpe = lenient.rolling_sharpe("ml")
    assert -0.30 < sharpe < 0.0          # non-vacuity: mildly negative, inside the gap
    assert lenient.is_active("ml") is True
    assert strict.is_active("ml") is False


def test_leverage_ramp_is_reachable_from_config_and_off_by_default():
    """A lever the operator cannot pull is dead code. Pin the whole path:
    settings field -> engine kwargs -> engine attribute."""
    from core.config import EngineSettings, engine_kwargs as build_engine_kwargs
    from core.engine.engine import TradingEngine

    default = EngineSettings()
    assert default.max_lever_up_step is None                      # ships OFF
    assert build_engine_kwargs(default)["max_lever_up_step"] is None

    tuned = EngineSettings(max_lever_up_step=0.25)
    kwargs = build_engine_kwargs(tuned)
    assert kwargs["max_lever_up_step"] == 0.25
    assert TradingEngine(mode="RESEARCH", max_lever_up_step=0.25).max_lever_up_step == 0.25
    assert TradingEngine(mode="RESEARCH").max_lever_up_step is None
