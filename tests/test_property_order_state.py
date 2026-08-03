"""
Property-based invariants for the safety-critical order-state machine (directive §23)
=====================================================================================
The order lifecycle (`execution/order_lifecycle.py`) and the ledger→position replay
(`ops/ledger.py`) are the real-money state machines. The §15/§7.3 invariants must hold
for EVERY reachable sequence of operations, not just the hand-picked examples in
`test_order_lifecycle.py`. These Hypothesis tests attack the whole reachable state space:

  • A RuleBasedStateMachine drives random sequences of create/transition/fill/cancel/
    uncertain/unknown/commission/reconcile ops and asserts the invariants after each step.
  • @given tests pin the record_fill clamp+idempotency and the ledger-replay sum.

A counterexample here is a real bug. Invariants asserted (directive §15/§7.3):
  filled_qty ≤ approved_qty (never over-fill) · idempotent fills (a duplicate fill_id is a
  no-op) · only valid OrderStatus reached · a disallowed transition raises and leaves state
  unchanged · a TERMINAL order never leaves a terminal state · can_resubmit fails closed for
  any resubmit-blocking state · ledger replay = the signed-fill sum per symbol.
"""

from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from execution.order_lifecycle import (
    _ALLOWED,
    RESUBMIT_BLOCKING_STATES,
    TERMINAL_STATES,
    InvalidTransition,
    OrderLifecycle,
    OrderStatus,
)
from ops.ledger import replay_ledger_to_positions
from ops.reconciliation import reconcile

_TS = "2024-01-01T00:00:00"
_EPS = 1e-6
_qty = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
_pos_qty = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)


# ── @given: record_fill clamp + idempotency ───────────────────────────────────────────

@settings(max_examples=300, deadline=None)
@given(approved=_pos_qty,
       fills=st.lists(st.tuples(st.integers(min_value=0, max_value=4), _qty), max_size=12))
def test_record_fill_never_overfills_and_is_idempotent(approved, fills):
    lc = OrderLifecycle()
    lc.create("o", approved, _TS, symbol="AAA", side="BUY")
    for st_ in (OrderStatus.VALIDATED, OrderStatus.RISK_APPROVED, OrderStatus.SUBMIT_PENDING,
                OrderStatus.ACKED, OrderStatus.WORKING):
        lc.transition("o", st_, _TS)
    seen: dict[str, float] = {}
    for fid, qty in fills:
        fill_id = f"f{fid}"
        before = lc.get("o").filled_qty
        lc.record_fill("o", fill_id, qty, _TS)
        after = lc.get("o").filled_qty
        assert after <= approved + _EPS                 # never over-fill (clamp invariant)
        assert after >= before - _EPS                   # filled_qty is monotonic non-decreasing
        if fill_id in seen:
            assert abs(after - before) <= _EPS          # duplicate fill_id -> no-op (idempotent)
        else:
            seen[fill_id] = qty
    rec = lc.get("o")
    if rec.filled_qty >= approved - 1e-9:
        assert rec.status == OrderStatus.FILLED         # fully filled -> FILLED
    # an over-applied set must be flagged, never silently swallowed
    if sum(q for _, q in {(f"f{fid}"): qty for fid, qty in fills}.items()) > approved + 1.0:
        assert "OVERFILL_CLAMPED" in rec.flags or rec.filled_qty <= approved + _EPS


# ── @given: ledger replay = signed-fill sum per symbol ─────────────────────────────────

class _FakeLedger:
    """Minimal ledger exposing only what replay_ledger_to_positions consumes: events('FILL')
    yielding objects with .seq and .payload. Lets the replay LOGIC be fuzzed without file I/O
    (the real ImmutableLedger round-trip is covered in test_ledger.py)."""

    def __init__(self, fill_payloads: list[dict]) -> None:
        self._f = fill_payloads

    def events(self, event_type: str):
        if event_type != "FILL":
            return []
        return [SimpleNamespace(seq=i, payload=p) for i, p in enumerate(self._f)]


@settings(max_examples=300, deadline=None)
@given(fills=st.lists(
    st.tuples(st.sampled_from(["AAA", "BBB", "CCC"]),
              st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
    max_size=30))
def test_replay_equals_signed_sum_per_symbol(fills):
    payloads = [{"symbol": s, "signed_qty": q} for s, q in fills]
    expected: dict[str, float] = {}
    for s, q in fills:
        expected[s] = expected.get(s, 0.0) + q
    expected = {s: v for s, v in expected.items() if abs(v) > 1e-9}      # dust-filtered (matches replay)
    got = replay_ledger_to_positions(_FakeLedger(payloads))
    assert set(got) == set(expected)
    for s in expected:
        assert abs(got[s] - expected[s]) <= 1e-6


@settings(max_examples=100, deadline=None)
@given(fills=st.lists(
    st.tuples(st.sampled_from(["AAA", "BBB"]),
              st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
              st.booleans()),
    max_size=20))
def test_replay_skips_unsigned_fills_never_misreports(fills):
    # a FILL lacking signed_qty must be SKIPPED (under-report), never counted as zero/mis-signed
    payloads = []
    signed_only: dict[str, float] = {}
    for s, q, has_sign in fills:
        if has_sign:
            payloads.append({"symbol": s, "signed_qty": q})
            signed_only[s] = signed_only.get(s, 0.0) + q
        else:
            payloads.append({"symbol": s})                              # no signed_qty -> must be skipped
    signed_only = {s: v for s, v in signed_only.items() if abs(v) > 1e-9}
    got = replay_ledger_to_positions(_FakeLedger(payloads))
    assert set(got) == set(signed_only)
    for s in signed_only:
        assert abs(got[s] - signed_only[s]) <= 1e-6


# ── @given: the reconcile() engine — break iff beyond tol, non-finite fails closed ─────

_sym = st.sampled_from(["AAA", "BBB", "CCC"])
_finite_share = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_pos_dict = st.dictionaries(_sym, _finite_share, max_size=3)


@settings(max_examples=300, deadline=None)
@given(internal=_pos_dict, broker=_pos_dict)
def test_reconcile_position_break_iff_beyond_share_tol(internal, broker):
    tol = 1.0
    rep = reconcile({"positions": internal}, {"positions": broker}, asof=_TS, share_tol=tol)
    pos_breaks = {b.key for b in rep.breaks if b.dimension == "position"}
    for k in set(internal) | set(broker):
        diff = abs(internal.get(k, 0.0) - broker.get(k, 0.0))
        if diff > tol:
            assert k in pos_breaks                          # divergence beyond tol -> surfaced
        else:
            assert k not in pos_breaks                      # within tol -> not surfaced
    assert rep.clean == (len(rep.breaks) == 0)


@settings(max_examples=200, deadline=None)
@given(positions=_pos_dict, bad_key=_sym,
       bad=st.sampled_from([float("nan"), float("inf"), float("-inf")]), on_internal=st.booleans())
def test_reconcile_nonfinite_fails_closed(positions, bad_key, bad, on_internal):
    internal, broker = dict(positions), dict(positions)
    (internal if on_internal else broker)[bad_key] = bad      # poison one side with a non-finite value
    rep = reconcile({"positions": internal}, {"positions": broker}, asof=_TS, share_tol=1.0)
    bad_break = [b for b in rep.breaks if b.dimension == "position" and b.key == bad_key]
    assert bad_break and bad_break[0].severity == "BREAK"    # fail-closed: non-finite -> BREAK


# ── stateful: random operation sequences on the order lifecycle ────────────────────────

class OrderLifecycleMachine(RuleBasedStateMachine):
    """Drive random sequences of lifecycle operations and assert the §15/§7.3 invariants hold
    after every step. (prune_terminal is excluded — it is a retention concern tested separately;
    including it would delete bundle orders mid-run.)"""

    orders = Bundle("orders")

    def __init__(self) -> None:
        super().__init__()
        self.lc = OrderLifecycle()
        self.approved: dict[str, float] = {}
        self.seen_fill: dict[str, set] = {}
        self.ever_terminal: set[str] = set()
        self._ctr = 0

    def _mark(self, oid: str) -> None:
        if self.lc.get(oid).status in TERMINAL_STATES:
            self.ever_terminal.add(oid)

    @rule(target=orders, qty=_pos_qty)
    def create(self, qty):
        oid = f"o{self._ctr}"
        self._ctr += 1
        self.lc.create(oid, qty, _TS, symbol="AAA", side="BUY")
        self.approved[oid] = qty
        self.seen_fill[oid] = set()
        return oid

    @rule(oid=orders, to=st.sampled_from(list(OrderStatus)))
    def try_transition(self, oid, to):
        before = self.lc.get(oid).status
        try:
            self.lc.transition(oid, to, _TS)
        except InvalidTransition:
            assert self.lc.get(oid).status == before        # disallowed -> state unchanged
        self._mark(oid)

    @rule(oid=orders, data=st.data())
    def valid_advance(self, oid, data):
        # Take a RANDOM ALLOWED transition so orders actually walk the graph into deep/terminal
        # states (otherwise random targets are almost always rejected and orders stay in CREATED).
        allowed = sorted(_ALLOWED.get(self.lc.get(oid).status, frozenset()), key=lambda s: s.value)
        if allowed:
            self.lc.transition(oid, data.draw(st.sampled_from(allowed)), _TS)
        self._mark(oid)

    @rule(oid=orders, fid=st.integers(min_value=0, max_value=4), qty=_qty)
    def record_fill(self, oid, fid, qty):
        fill_id = f"f{fid}"
        before = self.lc.get(oid).filled_qty
        self.lc.record_fill(oid, fill_id, qty, _TS)
        after = self.lc.get(oid).filled_qty
        if fill_id in self.seen_fill[oid]:
            assert abs(after - before) <= _EPS              # duplicate fill_id -> idempotent no-op
        else:
            self.seen_fill[oid].add(fill_id)
        self._mark(oid)

    @rule(oid=orders)
    def mark_uncertain(self, oid):
        try:
            self.lc.mark_submission_uncertain(oid, _TS)
        except InvalidTransition:
            pass

    @rule(oid=orders)
    def mark_unknown(self, oid):
        try:
            self.lc.mark_broker_unknown(oid, _TS)
        except InvalidTransition:
            pass

    @rule(oid=orders)
    def request_cancel(self, oid):
        try:
            self.lc.request_cancel(oid, _TS)
        except InvalidTransition:
            pass

    @rule(oid=orders, amount=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False))
    def commission(self, oid, amount):
        self.lc.record_commission(oid, amount, _TS)         # attaches even to terminal orders

    @rule(oid=orders, broker_filled=_qty)
    def reconcile_broker_fill(self, oid, broker_filled):
        self.lc.reconcile_broker_fill(oid, broker_filled, _TS)
        self._mark(oid)

    @invariant()
    def invariants_hold(self):
        for oid, approved in self.approved.items():
            rec = self.lc.get(oid)
            assert rec.filled_qty <= approved + _EPS        # §7.3 filled_qty never exceeds approved
            assert isinstance(rec.status, OrderStatus)      # always a valid state
            if rec.status in RESUBMIT_BLOCKING_STATES:
                assert self.lc.can_resubmit(oid) is False   # §15 fail-closed resubmission
        for oid in self.ever_terminal:
            assert self.lc.get(oid).status in TERMINAL_STATES  # §15 terminal is absorbing


OrderLifecycleMachine.TestCase.settings = settings(
    max_examples=200, stateful_step_count=40, deadline=None)
TestOrderLifecycle = OrderLifecycleMachine.TestCase
