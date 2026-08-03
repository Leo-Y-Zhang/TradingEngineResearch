"""
Phase 6(b) — STEP-12 LIVE execution routed through the §15 OrderManager lifecycle.

Two kinds of test here:
  • GOLDEN MASTERS that pin the CURRENT observable STEP-12 / achieved-book contract so
    the refactor (routing LIVE submits through OrderManager) cannot silently drift it.
  • NEW behaviour tests proving the §15 safety on the live path: timeout->UNCERTAIN
    (not reject), disconnect->BROKER_UNKNOWN, idempotent + clamped fills, cross-cycle
    lifecycle persistence, deterministic order ids, and the no-OrderManager-outside-LIVE
    object-graph invariant.

RESEARCH/PAPER are never touched by this change — the no-live-path guarantee
(tests/test_safety_no_live_path.py) stays intact.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.engine.engine import TradingEngine
from data.data_contracts import FillEvent
from execution.order_lifecycle import OrderLifecycle, OrderRecord, OrderStatus, TERMINAL_STATES
from ops.run_loop import build_cycle_inputs

_T = datetime(2024, 1, 15)
_CAPITAL = 1_000_000.0


def _fe(symbol: str, qty: float, price: float = 100.0, *, order_id: str = "x", slip: float = 2.0) -> FillEvent:
    return FillEvent(order_id=order_id, symbol=symbol, qty=qty, fill_price=price,
                     decision_price=price, arrival_price=price, slippage_bps=slip, fill_timestamp=_T)


def _inputs(asof: datetime | None = None, symbols=("AAPL", "MSFT"),
            capital: float = _CAPITAL, weights: dict | None = None):
    idx = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(3)
    data = {s: 100.0 * (1 + i * 0.05) * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 60)))
            for i, s in enumerate(symbols)}
    prices = pd.DataFrame(data, index=idx)
    asof = asof or idx[-1].to_pydatetime()
    return build_cycle_inputs(prices, asof, list(symbols), weights or {}, capital)


def _plan(symbol: str, qty: float, side: str = "BUY", slice_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, side=side, qty=qty, slice_index=slice_index)


# ── broker fakes (BrokerProtocol-shaped: `connected` + submit(child_plans, mode)) ─────

class _FillBroker:
    """Connected broker that returns one fill per plan (the deterministic LIVE stub)."""
    connected = True

    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, child_plans, mode):
        self.submitted.extend(child_plans)
        return [_fe(p.symbol, float(p.qty), order_id=f"{p.symbol}-live-{i}")
                for i, p in enumerate(child_plans) if float(p.qty) > 0.0]


class _TimeoutBroker:
    connected = True

    def submit(self, child_plans, mode):
        raise TimeoutError("no acknowledgement from broker")


class _DisconnectedBroker:
    connected = False

    def submit(self, child_plans, mode):  # pragma: no cover - must never be reached
        raise AssertionError("submit must not be called while disconnected")


class _DupFillBroker:
    """Returns the SAME fill_id twice for the one plan (a duplicate broker callback)."""
    connected = True

    def submit(self, child_plans, mode):
        p = child_plans[0]
        return [_fe(p.symbol, float(p.qty), order_id="DUP"), _fe(p.symbol, float(p.qty), order_id="DUP")]


class _OverfillBroker:
    """Returns more than the approved qty for the one plan."""
    connected = True

    def submit(self, child_plans, mode):
        p = child_plans[0]
        return [_fe(p.symbol, float(p.qty) * 2.0, order_id="OF")]


# ── GOLDEN MASTERS (pin the current contract; must survive the refactor) ──────────────

def test_step12_live_output_is_pinned():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    plans = [_plan("AAPL", 100.0), _plan("MSFT", 50.0)]
    fills, reports, live_count = e._step12_execute_and_tca(_inputs(), [], plans, {})
    got = sorted((f.symbol, round(f.qty, 6), round(f.fill_price, 6), round(f.slippage_bps, 6)) for f in fills)
    assert got == [("AAPL", 100.0, 100.0, 2.0), ("MSFT", 50.0, 100.0, 2.0)]
    assert live_count == 2
    assert sorted(r["symbol"] for r in reports) == ["AAPL", "MSFT"]


def test_achieved_weights_multi_slice_is_pinned():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    inputs = _inputs(weights={"AAPL": 0.05})
    intents = [SimpleNamespace(symbol="AAPL", direction="BUY"),
               SimpleNamespace(symbol="MSFT", direction="BUY")]
    fills = [_fe("AAPL", 500.0), _fe("AAPL", 500.0), _fe("MSFT", 200.0)]  # 50k + 50k ; 20k
    aw = e._achieved_weights(inputs, intents, fills)
    # AAPL: held 0.05 + 100k/1M = 0.15 ; MSFT: 0 + 20k/1M = 0.02
    assert aw == {"AAPL": pytest.approx(0.15), "MSFT": pytest.approx(0.02)}


# ── NEW behaviour: §15 safety on the LIVE path ────────────────────────────────────────

def test_live_timeout_marks_order_uncertain_not_reject():
    e = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    fills, _reports, live_count = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert fills == []                                   # nothing confirmed filled
    rec = e._order_lifecycle.all()[0]
    assert rec.status == OrderStatus.SUBMISSION_UNCERTAIN  # NOT a rejection
    assert e._order_manager.can_resubmit(rec.order_id) is False
    assert live_count == 1                               # a submit WAS attempted


def test_live_disconnect_marks_broker_unknown_no_fills():
    e = TradingEngine(mode="LIVE", broker=_DisconnectedBroker(), capital_gbp=_CAPITAL)
    fills, _reports, live_count = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert fills == [] and live_count == 0               # never reached the broker
    rec = e._order_lifecycle.all()[0]
    assert rec.status == OrderStatus.BROKER_UNKNOWN
    assert e._order_manager.can_resubmit(rec.order_id) is False


def test_live_duplicate_fill_is_idempotent():
    e = TradingEngine(mode="LIVE", broker=_DupFillBroker(), capital_gbp=_CAPITAL)
    fills, _reports, _live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    rec = e._order_lifecycle.all()[0]
    assert rec.filled_qty == pytest.approx(100.0)            # applied once, not doubled
    assert sum(f.qty for f in fills) == pytest.approx(100.0)  # engine emits the deduped qty


def test_live_overfill_is_clamped_in_emitted_fill():
    e = TradingEngine(mode="LIVE", broker=_OverfillBroker(), capital_gbp=_CAPITAL)
    fills, _reports, _live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    rec = e._order_lifecycle.all()[0]
    assert "OVERFILL_CLAMPED" in rec.flags
    assert sum(f.qty for f in fills) == pytest.approx(100.0)  # never exceeds the approved delta


def test_cross_cycle_lifecycle_persists_on_engine():
    e = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    e._step12_execute_and_tca(_inputs(asof=datetime(2024, 3, 1)), [], [_plan("AAPL", 100.0)], {})  # -> uncertain
    uncertain = [r.order_id for r in e._order_lifecycle.all() if r.status == OrderStatus.SUBMISSION_UNCERTAIN]
    assert len(uncertain) == 1
    e.broker = _FillBroker()                                                       # reconnect: new session
    e._step12_execute_and_tca(_inputs(asof=datetime(2024, 3, 2)), [], [_plan("MSFT", 50.0)], {})  # cycle 2, diff symbol
    assert e._order_lifecycle.get(uncertain[0]).status == OrderStatus.SUBMISSION_UNCERTAIN  # AAPL still tracked
    assert len(e._order_lifecycle.all()) == 2                                      # AAPL (uncertain) + MSFT


# ── LIVE6B-1: pending-exposure overlay + per-symbol no-stacking block ──────────────────

def _pending_working(e, symbol, qty, side="BUY", oid="p1"):
    lc = e._order_lifecycle or OrderLifecycle()
    lc.create(oid, qty, "2024-01-01T00:00:00", symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition(oid, st, "2024-01-01T00:00:00")
    e._order_lifecycle = lc
    return lc


def test_pending_overlay_signed_residual():
    e = TradingEngine(mode="LIVE", capital_gbp=1_000_000.0)
    lc = _pending_working(e, "AAPL", 1000.0, "BUY", "b")
    lc.record_fill("b", "f", 400.0, "2024-01-01T00:00:00")            # 600 remaining
    lc.create("s", 500.0, "2024-01-01T00:00:00", symbol="MSFT", side="SELL")
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition("s", st, "2024-01-01T00:00:00")
    inputs = _inputs()
    inputs.market_microstructure["AAPL"]["price"] = 100.0
    inputs.market_microstructure["MSFT"]["price"] = 200.0
    ov = e._pending_overlay(inputs)
    assert ov["AAPL"] == pytest.approx(600 * 100 / 1_000_000)         # +0.06 remaining BUY residual
    assert ov["MSFT"] == pytest.approx(-500 * 200 / 1_000_000)        # -0.20 SELL


def test_pending_overlay_cancels_redundant_buy_in_step11():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    inputs = _inputs()
    market = {"execution_regime": "NORMAL", "regime_label": "NORMAL"}
    base, _ = e._step11_execution_planning(inputs, {"AAPL": 0.10}, {}, {}, True, market)
    assert any(i.symbol == "AAPL" and i.direction == "BUY" for i in base)   # baseline: a BUY is planned
    price = float(inputs.market_microstructure["AAPL"]["price"])
    _pending_working(e, "AAPL", 0.10 * e.capital_gbp / price, "BUY")        # in-flight covers the full target
    intents, _ = e._step11_execution_planning(inputs, {"AAPL": 0.10}, {}, {}, True, market)
    assert not any(i.symbol == "AAPL" for i in intents)                     # delta cancelled -> no new order


def test_per_symbol_block_does_not_stack_on_unresolved_order():
    e = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    e._step12_execute_and_tca(_inputs(asof=datetime(2024, 8, 1)), [], [_plan("AAPL", 100.0)], {})  # -> uncertain
    assert e._order_lifecycle.all()[0].status == OrderStatus.SUBMISSION_UNCERTAIN
    e.broker = _FillBroker()
    fills, _r, live = e._step12_execute_and_tca(_inputs(asof=datetime(2024, 8, 2)), [], [_plan("AAPL", 100.0)], {})
    assert fills == [] and live == 0                                        # blocked: no second AAPL order
    assert len([r for r in e._order_lifecycle.all() if r.symbol == "AAPL"]) == 1


# ── LIVE6B-2: engine snapshot/restore + fail-closed submit gate ────────────────────────

def test_engine_snapshot_and_restore_open_orders():
    e1 = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    e1._step12_execute_and_tca(_inputs(asof=datetime(2024, 9, 1)), [], [_plan("AAPL", 100.0)], {})  # uncertain
    snap = e1.snapshot_open_orders()
    assert len(snap) == 1
    e2 = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)   # fresh process
    e2.restore_open_orders(snap)
    assert e2.has_pending_orders()                                          # remembered, not forgotten
    assert e2._order_lifecycle.get(snap[0]["order_id"]).status == OrderStatus.SUBMISSION_UNCERTAIN


def test_non_live_snapshot_open_orders_is_empty():
    e = TradingEngine(mode="PAPER", capital_gbp=_CAPITAL)
    e.run_cycle(_inputs())
    assert e.snapshot_open_orders() == []                                   # nothing to persist off-LIVE


def test_live_submits_blocked_gate_places_nothing():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    e.live_submits_blocked = True                                          # e.g. pending post-restart resync
    fills, _r, live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert fills == [] and live == 0                                        # gated -> nothing submitted
    assert e._order_lifecycle is None                                       # no order even created


def test_deterministic_order_ids_and_replay_does_not_crash():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    inputs = _inputs(asof=datetime(2024, 4, 1))
    plan = [_plan("AAPL", 100.0)]
    e._step12_execute_and_tca(inputs, [], plan, {})
    ids1 = sorted(r.order_id for r in e._order_lifecycle.all())
    e._step12_execute_and_tca(inputs, [], plan, {})        # same asof+plan -> same ids, must NOT crash
    ids2 = sorted(r.order_id for r in e._order_lifecycle.all())
    assert ids1 == ids2 and len(ids1) == 1                 # deterministic; no duplicate order created


def test_resync_open_orders_passthrough():
    e = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    assert e.resync_open_orders({}, "t0") == []            # inert before any submit
    e._step12_execute_and_tca(_inputs(asof=datetime(2024, 5, 1)), [], [_plan("AAPL", 100.0)], {})
    oid = e._order_lifecycle.all()[0].order_id
    changed = e.resync_open_orders({oid: "FILLED"}, "t1")
    assert oid in changed and e._order_lifecycle.get(oid).status == OrderStatus.FILLED


@pytest.mark.parametrize("mode", ["RESEARCH", "PAPER"])
def test_non_live_modes_never_build_an_order_manager(mode):
    e = TradingEngine(mode=mode, capital_gbp=_CAPITAL)
    e.run_cycle(_inputs())
    assert e._order_manager is None and e._order_lifecycle is None


# ── review fixes: no-blind-resubmit on a shifted same-asof replay, faithful clamp ─────

class _TwoPriceOverfillBroker:
    """One plan -> two DISTINCT executions at DIFFERENT prices, together OVER the approved qty."""
    connected = True

    def submit(self, child_plans, mode):
        p = child_plans[0]
        return [_fe(p.symbol, 60.0, 100.0, order_id="OF1", slip=1.0),
                _fe(p.symbol, 60.0, 110.0, order_id="OF2", slip=99.0)]   # 120 vs approved 100


class _TwoPartialBroker:
    """One plan -> two DISTINCT partial fills that exactly sum to the approved qty."""
    connected = True

    def submit(self, child_plans, mode):
        p = child_plans[0]
        half = float(p.qty) / 2.0
        return [_fe(p.symbol, half, 100.0, order_id="A"), _fe(p.symbol, half, 101.0, order_id="B")]


class _AckBroker:
    connected = True

    def submit(self, child_plans, mode):
        return []                                    # accepted, resting, no fills yet


def test_same_asof_replay_with_shifted_plans_does_not_resubmit():
    # FINDING #1/#6: a same-asof re-drive with a shifted plan list must NOT re-send an
    # already-placed slice (no blind resubmission — directive §15).
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    inputs = _inputs(asof=datetime(2024, 6, 1))
    e._step12_execute_and_tca(inputs, [], [_plan("AAPL", 100.0)], {})           # cycle 1: AAPL
    e._step12_execute_and_tca(inputs, [], [_plan("MSFT", 50.0), _plan("AAPL", 100.0)], {})  # shifted
    aapl_submits = sum(1 for p in e.broker.submitted if p.symbol == "AAPL")
    aapl_orders = [r for r in e._order_lifecycle.all() if "|AAPL|" in r.order_id]
    assert aapl_submits == 1                              # AAPL sent exactly once, never re-sent
    assert len(aapl_orders) == 1
    assert sum(r.filled_qty for r in aapl_orders) == pytest.approx(100.0)   # not 150


def test_multi_price_overfill_clamp_preserves_real_prices():
    # FINDING #2/#4/#11: clamping an over-fill must keep each execution's REAL price/slippage
    # (TCA + achieved book depend on them), not collapse to the first fill's price.
    e = TradingEngine(mode="LIVE", broker=_TwoPriceOverfillBroker(), capital_gbp=_CAPITAL)
    fills, _reports, _live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert sum(f.qty for f in fills) == pytest.approx(100.0)                     # clamped to approved
    assert sum(f.qty * f.fill_price for f in fills) == pytest.approx(10_400.0)   # 60@100 + 40@110
    assert {round(f.slippage_bps, 1) for f in fills} == {1.0, 99.0}             # both real, not just f0


def test_two_distinct_partial_fills_are_both_preserved():
    e = TradingEngine(mode="LIVE", broker=_TwoPartialBroker(), capital_gbp=_CAPITAL)
    fills, _reports, live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert len(fills) == 2 and sum(f.qty for f in fills) == pytest.approx(100.0)
    assert {round(f.fill_price, 1) for f in fills} == {100.0, 101.0}            # both kept, real prices
    assert live == 1                                                            # one SLICE, not two fills


def test_live_count_counts_submitted_slices_not_fills():
    e = TradingEngine(mode="LIVE", broker=_AckBroker(), capital_gbp=_CAPITAL)
    fills, _reports, live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0)], {})
    assert fills == [] and live == 1                  # acked-no-fill still counts as submitted


def test_live_sell_slice_routes_through_lifecycle():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    fills, _reports, live = e._step12_execute_and_tca(_inputs(), [], [_plan("AAPL", 100.0, side="SELL")], {})
    assert live == 1 and [f.symbol for f in fills] == ["AAPL"]
    rec = e._order_lifecycle.all()[0]
    assert "|SELL|" in rec.order_id and rec.status == OrderStatus.FILLED


def test_overfill_clamp_flows_through_achieved_weights():
    # FINDING #13: the real submit -> clamp -> _achieved_weights chain must book only the
    # APPROVED delta, never the broker's inflated qty.
    e = TradingEngine(mode="LIVE", broker=_OverfillBroker(), capital_gbp=_CAPITAL)   # returns 2x
    inputs = _inputs(weights={})
    fills, _reports, _live = e._step12_execute_and_tca(inputs, [], [_plan("AAPL", 100.0)], {})
    aw = e._achieved_weights(inputs, [SimpleNamespace(symbol="AAPL", direction="BUY")], fills)
    assert aw == {"AAPL": pytest.approx(0.01)}        # 100 @ 100 = 10k = 1% of 1M, not 2%


# ── LIVE6B-4: bounded lifecycle growth (prune TERMINAL only) ───────────────────────────

def _rec(order_id: str, status: OrderStatus, ts: str) -> OrderRecord:
    r = OrderRecord(order_id=order_id, approved_qty=10.0, status=status)
    r.history.append((ts, status.value, "created"))
    return r


def test_prune_terminal_drops_oldest_keeps_nonterminal():
    lc = OrderLifecycle()
    for i in range(5):                                  # 5 terminal (FILLED), oldest -> newest
        r = _rec(f"t{i}", OrderStatus.FILLED, f"2024-01-1{i}T00:00:00")
        lc._orders[r.order_id] = r
    for i, st in enumerate((OrderStatus.SUBMISSION_UNCERTAIN, OrderStatus.WORKING)):
        r = _rec(f"n{i}", st, "2024-01-01T00:00:00")
        lc._orders[r.order_id] = r
    assert lc.prune_terminal(2) == 3                    # 5 terminal - cap 2
    remaining = {r.order_id for r in lc.all()}
    assert {"t3", "t4", "n0", "n1"} == remaining        # newest 2 terminal + BOTH non-terminal
    assert len([r for r in lc.all() if r.status in TERMINAL_STATES]) == 2


def test_prune_terminal_noop_below_cap():
    lc = OrderLifecycle()
    lc._orders["t0"] = _rec("t0", OrderStatus.FILLED, "2024-01-01T00:00:00")
    assert lc.prune_terminal(5000) == 0 and len(lc.all()) == 1


def test_engine_bounds_terminal_orders_across_live_cycles():
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL, max_retained_terminal_orders=1)
    for day in (1, 2, 3):                               # 3 distinct-asof LIVE cycles -> 3 FILLED orders
        e._step12_execute_and_tca(_inputs(asof=datetime(2024, 7, day)), [], [_plan("AAPL", 100.0)], {})
    assert len([r for r in e._order_lifecycle.all() if r.status in TERMINAL_STATES]) == 1  # bounded


def test_resync_after_restore_builds_manager_and_reconciles():
    # review fix (CRITICAL): a restart restores the lifecycle but NOT the OrderManager; resync
    # must still build it and GENUINELY reconcile, else _needs_resync clears on a no-op and the
    # engine trades an unreconciled book after restart.
    src = OrderLifecycle()
    src.create("o1", 100.0, "t", symbol="AAPL", side="BUY")
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING):
        src.transition("o1", st, "t")
    src.mark_submission_uncertain("o1", "t")
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    e.restore_open_orders(src.snapshot_nonterminal())
    assert e._order_manager is None                       # restore alone does not build the manager
    changed = e.resync_open_orders(
        [{"order_ref": "o1", "broker_order_id": "B", "status": "WORKING", "symbol": "AAPL", "filled_qty": 0.0}], "t2")
    assert "o1" in changed                                # resync GENUINELY ran (not a no-op)
    assert e._order_lifecycle.get("o1").status == OrderStatus.WORKING


def test_pending_overlay_skips_unknown_side():
    # review fix: a restored/legacy record with no side must NOT be treated as a SELL by the overlay.
    e = TradingEngine(mode="LIVE", capital_gbp=1_000_000.0)
    lc = OrderLifecycle()
    lc.create("o1", 1000.0, "t", symbol="AAPL", side="")  # malformed/legacy: empty side
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition("o1", st, "t")
    e._order_lifecycle = lc
    inputs = _inputs()
    inputs.market_microstructure["AAPL"]["price"] = 100.0
    assert e._pending_overlay(inputs) == {}               # unknown side -> skipped, not signed as SELL


# ── slice 4 (held-book reconciliation): engine plumbing ───────────────────────────────

def _hold_order(e, oid="o1", symbol="AAPL", side="BUY", approved=100.0, broker_filled=100.0):
    """An engine-owned lifecycle order parked in RECONCILIATION_HOLD with a broker-discovered
    filled_qty (the disconnect-fill shape: filled_qty set WITHOUT the normal fill transition)."""
    lc = OrderLifecycle()
    lc.create(oid, approved, "t", symbol=symbol, side=side)
    for st in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
               OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition(oid, st, "t")
    lc.reconcile_broker_fill(oid, broker_filled, "t")     # set filled_qty, no transition
    lc.transition(oid, OrderStatus.RECONCILIATION_HOLD, "t")
    e._order_lifecycle = lc
    return lc


def test_book_reconciled_fill_resolves_full_hold_to_filled():
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    _hold_order(e, broker_filled=100.0)                   # fully filled at the broker
    e.book_reconciled_fill("o1", "t2")
    assert e._order_lifecycle.get("o1").status == OrderStatus.FILLED


def test_book_reconciled_fill_resolves_partial_hold_to_partially_filled():
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    _hold_order(e, broker_filled=70.0)                    # partially filled at the broker
    e.book_reconciled_fill("o1", "t2")
    assert e._order_lifecycle.get("o1").status == OrderStatus.PARTIALLY_FILLED


def test_book_reconciled_fill_is_idempotent():
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    _hold_order(e, broker_filled=100.0)
    e.book_reconciled_fill("o1", "t2")
    e.book_reconciled_fill("o1", "t3")                    # second call: no crash, stays FILLED
    assert e._order_lifecycle.get("o1").status == OrderStatus.FILLED


def test_book_reconciled_fill_missing_order_is_noop():
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    e.book_reconciled_fill("nope", "t2")                  # no lifecycle / unknown id -> no crash


def test_drain_discovered_fills_returns_then_clears():
    e = TradingEngine(mode="LIVE", broker=_TimeoutBroker(), capital_gbp=_CAPITAL)
    assert e.drain_discovered_fills() == []               # inert before any manager
    e._step12_execute_and_tca(_inputs(asof=datetime(2024, 6, 1)), [], [_plan("AAPL", 100.0)], {})  # -> uncertain
    oid = e._order_lifecycle.all()[0].order_id
    # broker now reports a partial fill that landed during the gap
    e.resync_open_orders([{"order_ref": oid, "status": "WORKING", "symbol": "AAPL", "filled_qty": 30.0}], "t2")
    drained = e.drain_discovered_fills()
    assert len(drained) == 1 and drained[0]["delta_qty"] == pytest.approx(30.0)
    assert e.drain_discovered_fills() == []               # drained -> empty


def test_cancel_reconciled_order_cancels_hold():
    # REJECT path: a spurious discovered fill cancels the parked HOLD order out (unfreezes symbol).
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    _hold_order(e, broker_filled=70.0)
    e.cancel_reconciled_order("o1", "t2")
    assert e._order_lifecycle.get("o1").status == OrderStatus.CANCELLED


def test_cancel_reconciled_missing_order_is_noop():
    e = TradingEngine(mode="LIVE", capital_gbp=_CAPITAL)
    e.cancel_reconciled_order("nope", "t2")               # no lifecycle / unknown id -> no crash


def test_hold_freezes_symbol_until_book_reconciled_fill():
    # the core safety property: a parked RECONCILIATION_HOLD order freezes its symbol from new
    # orders (fail-closed); the operator-gated book_reconciled_fill advances it FILLED -> unfreezes.
    e = TradingEngine(mode="LIVE", broker=_FillBroker(), capital_gbp=_CAPITAL)
    _hold_order(e, oid="o1", symbol="AAPL", approved=100.0, broker_filled=100.0)   # AAPL parked HOLD
    fills, submitted = e._submit_live_via_lifecycle(_inputs(), [_plan("AAPL", 50.0, slice_index=5)])
    assert submitted == 0 and fills == []                 # frozen — the new AAPL slice is blocked
    e.book_reconciled_fill("o1", "t2")                    # operator resolves -> HOLD -> FILLED
    _fills2, submitted2 = e._submit_live_via_lifecycle(_inputs(), [_plan("AAPL", 50.0, slice_index=6)])
    assert submitted2 == 1                                # symbol unblocked -> the new slice submits
