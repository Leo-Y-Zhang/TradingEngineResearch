"""Fuzz the Pydantic data contracts across their constraint space (ROADMAP Phase 7
item 5).

The original per-field validators used ``< 0`` / ``<= 0`` / ``== 0`` comparisons,
which NaN and ±inf silently evade (every NaN comparison is False; inf passes a
``<= 0`` test). These tests pin the fail-closed contract surface: non-finite is
rejected for every float field (scalar and dict-valued), `normalize_mode` is
default-deny over arbitrary text, bounded fields enforce their ranges, and
`position_divergence` stays fail-closed on non-finite quantities.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from data.data_contracts import (
    _KNOWN_MODES,
    BrokerState,
    FeatureRow,
    FillEvent,
    InsiderEvent,
    MarketBar,
    NewsEvent,
    OrderIntent,
    PortfolioState,
    PredictionRow,
    QuoteSnapshot,
    normalize_mode,
    position_divergence,
)

DT = datetime(2023, 1, 2)
NONFINITE = [float("nan"), float("inf"), float("-inf")]

# (model, valid kwargs, [scalar float fields to fuzz])
BASELINES: dict[str, tuple] = {
    "MarketBar": (MarketBar, dict(
        symbol="X", open=10.0, high=11.0, low=9.0, close=10.5, volume=1000.0,
        event_timestamp=DT, ingest_timestamp=DT, asof_timestamp=DT, source="t",
        freshness_seconds=1.0, stale_flag=False),
        ["open", "high", "low", "close", "volume", "freshness_seconds"]),
    "QuoteSnapshot": (QuoteSnapshot, dict(
        symbol="X", bid=10.0, ask=10.2, bid_size=100.0, ask_size=100.0,
        event_timestamp=DT, asof_timestamp=DT, freshness_seconds=1.0, stale_flag=False),
        ["bid", "ask", "bid_size", "ask_size", "freshness_seconds"]),
    "NewsEvent": (NewsEvent, dict(
        headline="h", symbols_mentioned=["X"], source="s", event_timestamp=DT,
        ingest_timestamp=DT, age_minutes=5.0, stale_flag=False),
        ["age_minutes"]),
    "InsiderEvent": (InsiderEvent, dict(
        symbol="X", insider_name="a", transaction_code="P", amount_usd=1000.0,
        event_timestamp=DT, age_days=2.0, stale_flag=False),
        ["amount_usd", "age_days"]),
    "PredictionRow": (PredictionRow, dict(
        symbol="X", asof_timestamp=DT, model_version="v", expected_return=0.01,
        risk_estimate=0.2, p_positive=0.5, p_tail_loss=0.1, confidence=0.5),
        ["expected_return", "risk_estimate"]),
    "OrderIntent": (OrderIntent, dict(
        symbol="X", direction="BUY", target_weight=0.1, expected_cost_bps=5.0,
        urgency="NORMAL", alpha_half_life_minutes=10, decision_timestamp=DT,
        model_version="v", regime_state="N", risk_approved=True),
        ["expected_cost_bps"]),
    "FillEvent": (FillEvent, dict(
        order_id="1", symbol="X", qty=10.0, fill_price=10.0, decision_price=10.0,
        arrival_price=10.0, slippage_bps=1.0, fill_timestamp=DT),
        ["qty", "fill_price", "decision_price", "arrival_price", "slippage_bps"]),
    "FeatureRow": (FeatureRow, dict(
        symbol="X", asof_timestamp=DT, feature_schema_version="v6.0",
        features={"f1": 0.5}, freshness_flags={"f1": False}, missing_count=0),
        []),
    "PortfolioState": (PortfolioState, dict(
        asof_timestamp=DT, nav_gbp=1.0e6, cash_gbp=1.0e3,
        positions={"X": 10.0}, weights={"X": 0.1}, stale_flag=False),
        ["nav_gbp", "cash_gbp"]),
    "BrokerState": (BrokerState, dict(
        broker="IBKR", connected=True, account_id="A", asof_timestamp=DT,
        nav_gbp=1.0e6, cash_gbp=1.0e3, buying_power_gbp=1.0e6,
        positions={"X": 10.0}, stale_flag=False),
        ["nav_gbp", "cash_gbp", "buying_power_gbp"]),
}

DICT_FLOAT_FIELDS = [
    ("FeatureRow", "features", "f1"),
    ("PortfolioState", "positions", "X"),
    ("PortfolioState", "weights", "X"),
    ("BrokerState", "positions", "X"),
]


# ── baselines are valid (so the rejection tests below are meaningful) ─────────────────

@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_constructs(name):
    model, kwargs, _ = BASELINES[name]
    model(**kwargs)


# ── non-finite is rejected for every float field (the core fail-closed property) ──────

@pytest.mark.parametrize("name", list(BASELINES))
def test_scalar_float_fields_reject_nonfinite(name):
    model, kwargs, float_fields = BASELINES[name]
    for field in float_fields:
        for bad in NONFINITE:
            broken = dict(kwargs)
            broken[field] = bad
            with pytest.raises(ValidationError):
                model(**broken)


@pytest.mark.parametrize("name,field,key", DICT_FLOAT_FIELDS)
def test_dict_float_fields_reject_nonfinite(name, field, key):
    model, kwargs, _ = BASELINES[name]
    for bad in NONFINITE:
        broken = dict(kwargs)
        d = dict(broken[field])
        d[key] = bad
        broken[field] = d
        with pytest.raises(ValidationError):
            model(**broken)


# ── normalize_mode is default-deny ───────────────────────────────────────────────────

@given(st.text())
def test_normalize_mode_default_deny(s):
    try:
        result = normalize_mode(s)
    except ValueError:
        return  # rejecting an unknown mode is the correct, safe behaviour
    assert result in _KNOWN_MODES
    assert result == result.strip().upper()


@given(st.sampled_from(sorted(_KNOWN_MODES)), st.sampled_from(["", " ", "  ", "\t"]))
def test_normalize_mode_roundtrips_known(mode, pad):
    assert normalize_mode(pad + mode.lower() + pad) == mode


# ── bounded fields enforce their range (and reject non-finite) ────────────────────────

@given(st.floats(allow_nan=True, allow_infinity=True))
def test_order_intent_weight_bounds(w):
    kwargs = dict(BASELINES["OrderIntent"][1])
    kwargs["target_weight"] = w
    if math.isfinite(w) and -1.0 <= w <= 1.0:
        assert OrderIntent(**kwargs).target_weight == w
    else:
        with pytest.raises(ValidationError):
            OrderIntent(**kwargs)


@given(st.floats(allow_nan=True, allow_infinity=True))
def test_prediction_probability_bounds(p):
    kwargs = dict(BASELINES["PredictionRow"][1])
    kwargs["p_positive"] = p
    if math.isfinite(p) and 0.0 <= p <= 1.0:
        assert PredictionRow(**kwargs).p_positive == p
    else:
        with pytest.raises(ValidationError):
            PredictionRow(**kwargs)


# ── valid finite inputs still round-trip ──────────────────────────────────────────────

@given(st.floats(min_value=1e-3, max_value=1e7, allow_nan=False, allow_infinity=False))
def test_fillevent_valid_prices_roundtrip(price):
    fe = FillEvent(order_id="1", symbol="X", qty=1.0, fill_price=price,
                   decision_price=price, arrival_price=price, slippage_bps=0.0,
                   fill_timestamp=DT)
    assert fe.fill_price == price


# ── position_divergence stays fail-closed ─────────────────────────────────────────────

_FINITE = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
_BOOK = st.dictionaries(st.text(min_size=1, max_size=4), _FINITE, max_size=5)


@given(_BOOK, _BOOK)
def test_position_divergence_reports_finite_gaps(internal, broker):
    out = position_divergence(internal, broker, tolerance=0.0)
    for sym in set(internal) | set(broker):
        if abs(internal.get(sym, 0.0) - broker.get(sym, 0.0)) > 0.0:
            assert sym in out


def test_position_divergence_nonfinite_always_divergent():
    # NaN compares False against any threshold — must still be flagged (fail-closed).
    assert "X" in position_divergence({"X": float("nan")}, {"X": float("nan")}, tolerance=1e9)
    assert "Y" in position_divergence({"Y": float("inf")}, {"Y": 1.0}, tolerance=1e9)


@given(st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False))
def test_position_divergence_rejects_negative_tolerance(tol):
    with pytest.raises(ValueError):
        position_divergence({}, {}, tolerance=tol)


def test_position_divergence_rejects_nonfinite_tolerance():
    for bad in NONFINITE:
        with pytest.raises(ValueError):
            position_divergence({}, {}, tolerance=bad)
