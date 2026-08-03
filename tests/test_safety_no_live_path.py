"""§23 safety net — assert NO config or code path can reach a live account unless mode
is **explicitly** LIVE *and* properly armed.

This is the paper/shadow guarantee the IBKR master directive forbids ever weakening
(golden rule 1: mode is explicit, never inferred). It consolidates, in one auditable
place, the four independent lines of defence:

  A. mode validation is default-deny (unknown modes are rejected, never coerced safe);
  B. arming LIVE requires explicit ``confirm_live`` + ``audit_log_path`` (fail-closed);
  C. the money boundary (``make_broker``) refuses to build a real-money broker from an
     unconfirmed/unaudited config — even one that bypassed the settings validator;
  D. the engine's STEP-12 execution gate calls ``broker.submit`` ONLY in LIVE, and in
     LIVE exactly once PER non-zero child slice (Phase 6(b): per-order §15 routing)
     (RESEARCH plans none; PAPER simulates fills locally and never reaches a broker).

If any assertion here ever fails, a path to live money has opened — treat it as a P0.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backtesting.harness import _reset_engine_state
from core.config import EngineSettings, make_broker
from core.engine.engine import TradingEngine
from data.data_contracts import normalize_mode
from ops.run_loop import build_cycle_inputs

SYMBOLS = ["AAA", "BBB", "CCC"]


def _prices(n: int = 180, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {s: 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(rng.normal(0.0005, 0.011, size=n)))
            for i, s in enumerate(SYMBOLS)}
    return pd.DataFrame(data, index=idx)


@pytest.fixture(autouse=True)
def _reset():
    _reset_engine_state(123)
    yield


class _SpyBroker:
    """A broker that LOUDLY records any submit attempt. Wired into non-LIVE engines so a
    single accidental submission would be caught (submit_calls > 0)."""

    def __init__(self) -> None:
        self.submit_calls = 0
        self.is_paper = False  # deliberately NOT paper — a strict, live-like broker

    @property
    def connected(self) -> bool:
        return True

    def submit(self, child_plans, mode):
        self.submit_calls += 1
        return []


# ── A. mode validation is default-deny ────────────────────────────────────────────────

def test_normalize_mode_rejects_unknown():
    # Note: surrounding whitespace/case are normalised (so "live " IS valid → LIVE);
    # these are the genuinely-unrecognised strings that must be refused.
    for bad in ["", "prod", "paper-ish", "REAL", "l1ve", "research2", "LIVE!", "papertrade"]:
        with pytest.raises(ValueError):
            normalize_mode(bad)


def test_normalize_mode_accepts_only_the_three_known():
    assert normalize_mode(" live ") == "LIVE"
    assert normalize_mode("Research") == "RESEARCH"
    assert normalize_mode("paper") == "PAPER"


def test_settings_reject_unknown_mode():
    with pytest.raises(ValidationError):
        EngineSettings(mode="prod", universe=SYMBOLS)


# ── B. arming LIVE requires explicit confirmation + audit (fail-closed) ───────────────

def test_live_requires_confirm_live():
    with pytest.raises(ValidationError):
        EngineSettings(mode="LIVE", universe=SYMBOLS)


def test_live_requires_audit_log_path():
    with pytest.raises(ValidationError):
        EngineSettings(mode="LIVE", confirm_live=True, universe=SYMBOLS)


def test_live_armed_only_with_confirm_and_audit():
    s = EngineSettings(mode="LIVE", confirm_live=True,
                             audit_log_path="audit.log", universe=SYMBOLS)
    assert s.mode == "LIVE"


def test_lone_mode_flip_cannot_arm_live():
    # validate_assignment re-runs the fail-closed gate, so flipping a live mode onto an
    # already-built PAPER config (without confirm/audit) is rejected.
    s = EngineSettings(mode="PAPER", universe=SYMBOLS)
    with pytest.raises(ValidationError):
        s.mode = "LIVE"


# ── C. the money boundary (make_broker) refuses unless LIVE is properly armed ──────────

def test_research_builds_no_broker_at_all():
    assert make_broker(EngineSettings(mode="RESEARCH", universe=SYMBOLS)) is None


def test_paper_builds_a_paper_only_broker():
    b = make_broker(EngineSettings(mode="PAPER", universe=SYMBOLS))
    assert getattr(b, "is_paper", False) is True


def test_make_broker_refuses_unvalidated_live_config():
    # Bypass the settings validator entirely (model_construct) — the money boundary must
    # STILL refuse to build a real-money broker from an unconfirmed/unaudited config.
    rogue = EngineSettings.model_construct(
        mode="LIVE", confirm_live=False, audit_log_path=None)
    with pytest.raises(ValueError):
        make_broker(rogue)


def test_live_broker_is_never_built_anonymously():
    # confirmed + audited, but no account id from settings or vault → refuse.
    s = EngineSettings(mode="LIVE", confirm_live=True,
                             audit_log_path="audit.log", universe=SYMBOLS)
    with pytest.raises(ValueError):
        make_broker(s, vault=None)


# ── D. the engine reaches a broker ONLY in LIVE ───────────────────────────────────────

def test_step12_gate_submits_only_in_live():
    """The decisive unit-level proof: identical child plans, three modes. RESEARCH and
    PAPER never call broker.submit; LIVE does (the positive control that shows the spy
    isn't trivially zero)."""
    plans = [SimpleNamespace(symbol="AAA", qty=100.0, side="BUY")]
    prices = _prices()
    inputs = build_cycle_inputs(prices, prices.index[-1].to_pydatetime(), SYMBOLS, {}, 1e6)

    for mode in ("RESEARCH", "PAPER"):
        spy = _SpyBroker()
        engine = TradingEngine(mode=mode, broker=spy)
        engine._step12_execute_and_tca(inputs, [], plans, {})
        assert spy.submit_calls == 0, f"{mode} must never reach a broker"

    spy = _SpyBroker()
    engine = TradingEngine(mode="LIVE", broker=spy)
    _fills, _reports, _live = engine._step12_execute_and_tca(inputs, [], plans, {})
    # Phase 6(b): LIVE now routes per child slice through the §15 OrderManager, so
    # broker.submit is called once PER non-zero slice. One plan here -> exactly 1 call.
    assert spy.submit_calls == 1  # LIVE is the ONLY mode that reaches a broker


def test_step12_live_submits_once_per_child_slice_zero_in_non_live():
    """Phase 6(b) strengthens the guarantee: LIVE calls broker.submit exactly once PER
    non-zero child slice (per-order §15 routing), and RESEARCH/PAPER still call it ZERO
    times. The live-money invariant is unweakened — only the LIVE batching granularity
    changed from one-list-call to one-call-per-slice."""
    plans = [SimpleNamespace(symbol="AAA", qty=100.0, side="BUY"),
             SimpleNamespace(symbol="BBB", qty=50.0, side="BUY"),
             SimpleNamespace(symbol="CCC", qty=0.0, side="BUY")]    # zero-qty slice is skipped
    prices = _prices()
    inputs = build_cycle_inputs(prices, prices.index[-1].to_pydatetime(), SYMBOLS, {}, 1e6)

    for mode in ("RESEARCH", "PAPER"):
        spy = _SpyBroker()
        TradingEngine(mode=mode, broker=spy)._step12_execute_and_tca(inputs, [], plans, {})
        assert spy.submit_calls == 0, f"{mode} must never reach a broker"

    spy = _SpyBroker()
    TradingEngine(mode="LIVE", broker=spy)._step12_execute_and_tca(inputs, [], plans, {})
    assert spy.submit_calls == 2  # one per NON-ZERO slice (the zero-qty CCC is skipped)


def _run_full_cycle(mode: str, broker):
    prices = _prices()
    engine = TradingEngine(mode=mode, capital_gbp=1_000_000.0, broker=broker)
    inputs = build_cycle_inputs(prices, prices.index[-1].to_pydatetime(), SYMBOLS, {}, 1_000_000.0)
    return engine.run_cycle(inputs)


def test_research_full_cycle_never_submits_even_with_broker_wired():
    spy = _SpyBroker()
    result = _run_full_cycle("RESEARCH", spy)
    assert spy.submit_calls == 0
    assert result.live_orders_submitted == 0


def test_paper_full_cycle_never_submits_even_with_broker_wired():
    spy = _SpyBroker()
    result = _run_full_cycle("PAPER", spy)
    # PAPER runs the FULL pipeline (plans + simulates fills locally) but the broker is
    # never asked to submit — the live_orders count is 0 by construction.
    assert spy.submit_calls == 0
    assert result.live_orders_submitted == 0
