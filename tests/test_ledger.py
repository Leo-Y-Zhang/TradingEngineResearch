"""
Phase 3 — immutable hash-chained event ledger tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ops.ledger import GENESIS_HASH, ImmutableLedger, LedgerEvent


def _two() -> ImmutableLedger:
    led = ImmutableLedger()
    led.append("FILL", {"symbol": "AAPL", "qty": 10}, "2026-01-01T00:00:00")
    led.append("CASH", {"ccy": "USD", "amount": -1500.0}, "2026-01-02T00:00:00")
    return led


class TestAppendAndChain:
    def test_appends_and_chain_valid(self):
        led = _two()
        assert len(led) == 2
        assert led.events()[0].prev_hash == GENESIS_HASH
        assert led.events()[1].prev_hash == led.events()[0].hash
        assert led.verify_chain()

    def test_unknown_type_and_bad_payload_raise(self):
        led = ImmutableLedger()
        with pytest.raises(ValueError):
            led.append("NOT_A_TYPE", {}, "2026-01-01T00:00:00")
        with pytest.raises(TypeError):
            led.append("FILL", ["not", "a", "dict"], "2026-01-01T00:00:00")  # type: ignore[arg-type]

    def test_determinism(self):
        a, b = ImmutableLedger(), ImmutableLedger()
        for led in (a, b):
            led.append("FILL", {"q": 1}, "2026-01-01T00:00:00")
            led.append("CASH", {"amount": 100}, "2026-01-02T00:00:00")
        assert a.head_hash == b.head_hash


class TestTamperEvidence:
    def test_in_memory_tamper_detected(self):
        led = _two()
        assert led.verify_chain()
        # alter event 0's payload but keep its old hash -> chain must break
        orig = led._events[0]
        led._events[0] = LedgerEvent(
            seq=orig.seq, timestamp=orig.timestamp, event_type=orig.event_type,
            payload={"symbol": "AAPL", "qty": 99999}, prev_hash=orig.prev_hash, hash=orig.hash,
        )
        assert not led.verify_chain()

    def test_file_tamper_detected_on_load(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        led = ImmutableLedger(p)
        led.append("FILL", {"qty": 10}, "2026-01-01T00:00:00")
        led.append("FILL", {"qty": 20}, "2026-01-02T00:00:00")
        lines = p.read_text(encoding="utf-8").splitlines()
        d = json.loads(lines[0])
        d["payload"]["qty"] = 9999
        lines[0] = json.dumps(d, sort_keys=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            ImmutableLedger(p)


class TestDurability:
    def test_reload_reconstructs_exact_trail(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        led = ImmutableLedger(p)
        for i in range(5):
            led.append("CASH", {"i": i}, f"2026-01-0{i+1}T00:00:00")
        head = led.head_hash
        led2 = ImmutableLedger(p)
        assert len(led2) == 5
        assert led2.verify_chain()
        assert led2.head_hash == head


class TestCorrections:
    def test_correction_is_append_only(self):
        led = ImmutableLedger()
        led.append("FILL", {"qty": 10}, "2026-01-01T00:00:00")
        c = led.correct(0, "qty was wrong", {"qty": -10}, "2026-01-02T00:00:00")
        assert c.event_type == "CORRECTION"
        assert c.payload["corrects_seq"] == 0 and c.payload["reason"] == "qty was wrong"
        # the original event is untouched
        assert led.events("FILL")[0].payload["qty"] == 10
        assert led.verify_chain() and len(led) == 2


class TestRecordCycle:
    def test_records_fills_position_and_recon(self):
        from types import SimpleNamespace

        from ops.ledger import record_cycle
        fill = SimpleNamespace(order_id="o1", symbol="AAPL", qty=10.0, fill_price=190.0, slippage_bps=2.0)
        result = SimpleNamespace(fills=[fill], achieved_weights={"AAPL": 0.5},
                                 target_weights={}, blocked=False, live_orders_submitted=0)
        led = ImmutableLedger()
        n = record_cycle(led, result, "2026-01-01T00:00:00",
                         recon_alert={"severity": "WARNING", "kind": "reconciliation"})
        assert n == 3
        assert [e.event_type for e in led.events()] == ["FILL", "POSITION", "RECONCILIATION"]
        assert led.events("FILL")[0].payload["symbol"] == "AAPL"
        assert led.events("POSITION")[0].payload["book"] == {"AAPL": 0.5}
        assert led.verify_chain()

    def test_position_only_when_no_fills(self):
        from types import SimpleNamespace

        from ops.ledger import record_cycle
        result = SimpleNamespace(fills=[], achieved_weights=None,
                                 target_weights={"MSFT": 1.0}, blocked=True, live_orders_submitted=0)
        led = ImmutableLedger()
        n = record_cycle(led, result, "2026-01-01T00:00:00")
        assert n == 1
        assert led.events("POSITION")[0].payload["book"] == {"MSFT": 1.0}
        assert led.events("POSITION")[0].payload["blocked"] is True

    def test_records_broker_reported_commissions(self):
        # §17 cash leg (a): a fill carrying a broker-REPORTED commission appends a COMMISSION
        # event; a fill without one appends nothing (record only facts, never invent numbers).
        from types import SimpleNamespace

        from ops.ledger import record_cycle
        billed = SimpleNamespace(order_id="e1", symbol="AAPL", qty=10.0, fill_price=190.0,
                                 slippage_bps=2.0, commission=1.32)
        free = SimpleNamespace(order_id="e2", symbol="MSFT", qty=5.0, fill_price=300.0,
                               slippage_bps=1.0, commission=None)
        result = SimpleNamespace(fills=[billed, free], achieved_weights={}, target_weights={},
                                 blocked=False, live_orders_submitted=2)
        led = ImmutableLedger()
        n = record_cycle(led, result, "2026-01-01T00:00:00")
        comms = led.events("COMMISSION")
        assert len(comms) == 1
        assert comms[0].payload["order_id"] == "e1" and comms[0].payload["symbol"] == "AAPL"
        assert comms[0].payload["amount"] == pytest.approx(1.32)
        assert comms[0].payload["source"] == "broker_fill"
        assert n == 4                                     # 2 FILLs + 1 COMMISSION + 1 POSITION

    def test_commission_recording_is_idempotent_on_re_record(self):
        # dedup-safe across a same-cycle replay/re-record: anchored on the DURABLE ledger by the
        # broker's own fill id, the same commission is never double-counted into the cash leg.
        from types import SimpleNamespace

        from ops.ledger import record_cycle
        fill = SimpleNamespace(order_id="e1", symbol="AAPL", qty=10.0, fill_price=190.0,
                               slippage_bps=2.0, commission=1.32)
        result = SimpleNamespace(fills=[fill], achieved_weights={}, target_weights={},
                                 blocked=False, live_orders_submitted=1)
        led = ImmutableLedger()
        record_cycle(led, result, "2026-01-01T00:00:00")
        record_cycle(led, result, "2026-01-01T00:00:00")   # replayed re-record of the same cycle
        assert len(led.events("COMMISSION")) == 1

    def test_recorded_commission_reduces_replayed_cash(self):
        # end-to-end: record_cycle -> COMMISSION event -> replay_ledger_to_balances cash leg.
        from types import SimpleNamespace

        from ops.ledger import record_cycle, replay_ledger_to_balances
        led = ImmutableLedger()
        led.append("CASH", {"action": "deposit", "amount": 100_000.0}, "2026-01-01T00:00:00")
        fill = SimpleNamespace(order_id="e1", symbol="AAPL", qty=10.0, fill_price=100.0,
                               slippage_bps=2.0, commission=1.5)
        result = SimpleNamespace(fills=[fill],
                                 order_intents=[SimpleNamespace(symbol="AAPL", direction="BUY")],
                                 achieved_weights={}, target_weights={}, blocked=False,
                                 live_orders_submitted=1)
        record_cycle(led, result, "2026-01-02T00:00:00")
        # 100,000 − 1,000 (buy) − 1.5 (commission) = 98,998.5
        assert replay_ledger_to_balances(led)["cash"]["GBP"] == pytest.approx(98_998.5)


# ── Phase 3 (§17): replay the immutable trail into the internal position book ──────────

def _intent(symbol: str, direction: str) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, direction=direction)


def _fill(symbol: str, qty: float, price: float = 100.0) -> SimpleNamespace:
    return SimpleNamespace(order_id=f"{symbol}-x", symbol=symbol, qty=qty,
                           fill_price=price, slippage_bps=1.0)


def _cycle(led, fills, intents, ts):
    from ops.ledger import record_cycle
    record_cycle(led, SimpleNamespace(fills=fills, order_intents=intents, achieved_weights={},
                                      target_weights={}, blocked=False, live_orders_submitted=0), ts)


class TestReplayPositions:
    def test_record_cycle_signs_fills_from_intents(self):
        led = ImmutableLedger()
        _cycle(led, [_fill("AAPL", 10.0)], [_intent("AAPL", "BUY")], "2026-01-01T00:00:00")
        _cycle(led, [_fill("MSFT", 4.0)], [_intent("MSFT", "SELL")], "2026-01-02T00:00:00")
        fills = led.events("FILL")
        assert fills[0].payload["side"] == "BUY" and fills[0].payload["signed_qty"] == 10.0
        assert fills[1].payload["side"] == "SELL" and fills[1].payload["signed_qty"] == -4.0
        # the unsigned qty is preserved (back-compat)
        assert fills[1].payload["qty"] == 4.0

    def test_replay_reconstructs_net_positions(self):
        led = ImmutableLedger()
        _cycle(led, [_fill("AAPL", 10.0), _fill("MSFT", 5.0)],
               [_intent("AAPL", "BUY"), _intent("MSFT", "BUY")], "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 4.0)], [_intent("AAPL", "SELL")], "2026-01-02T00:00:00")
        from ops.ledger import replay_ledger_to_positions
        assert replay_ledger_to_positions(led) == {"AAPL": 6.0, "MSFT": 5.0}

    def test_replay_drops_fully_closed_name(self):
        led = ImmutableLedger()
        _cycle(led, [_fill("AAPL", 10.0)], [_intent("AAPL", "BUY")], "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 10.0)], [_intent("AAPL", "SELL")], "2026-01-02T00:00:00")
        from ops.ledger import replay_ledger_to_positions
        assert "AAPL" not in replay_ledger_to_positions(led)

    def test_replay_skips_unsigned_fills_failsafe(self):
        # a raw FILL with no signed_qty (older trail / no matching intent) is skipped,
        # never silently treated as a buy
        led = ImmutableLedger()
        led.append("FILL", {"symbol": "AAPL", "qty": 10}, "2026-01-01T00:00:00")
        from ops.ledger import replay_ledger_to_positions
        assert replay_ledger_to_positions(led) == {}

    def test_replay_survives_reload(self, tmp_path):
        p = tmp_path / "ledger.jsonl"
        led = ImmutableLedger(p)
        _cycle(led, [_fill("AAPL", 7.0)], [_intent("AAPL", "BUY")], "2026-01-01T00:00:00")
        from ops.ledger import replay_ledger_to_positions
        assert replay_ledger_to_positions(ImmutableLedger(p)) == {"AAPL": 7.0}

    def test_replay_feeds_reconcile(self):
        from ops.ledger import replay_ledger_to_positions
        from ops.reconciliation import reconcile
        led = ImmutableLedger()
        _cycle(led, [_fill("AAPL", 10.0)], [_intent("AAPL", "BUY")], "2026-01-01T00:00:00")
        internal = {"positions": replay_ledger_to_positions(led)}
        # broker agrees → clean
        assert reconcile(internal, {"positions": {"AAPL": 10.0}},
                         asof="2026-01-01T00:00:00", share_tol=1.0).clean
        # broker disagrees → a position break
        report = reconcile(internal, {"positions": {"AAPL": 3.0}},
                           asof="2026-01-01T00:00:00", share_tol=1.0)
        assert not report.clean
        assert report.breaks[0].dimension == "position"


class TestReplayBalances:
    """§17 cash leg: reconstruct {positions, cash} from the immutable trail."""

    def test_reconstructs_cash_from_deposit_and_trades(self):
        from ops.ledger import replay_ledger_to_balances
        led = ImmutableLedger()
        led.append("CASH", {"action": "deposit", "amount": 1_000_000.0, "ccy": "GBP"},
                   "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 10.0, price=190.0)], [_intent("AAPL", "BUY")], "2026-01-02T00:00:00")
        _cycle(led, [_fill("AAPL", 4.0, price=200.0)], [_intent("AAPL", "SELL")], "2026-01-03T00:00:00")
        bal = replay_ledger_to_balances(led)
        assert bal["positions"] == {"AAPL": 6.0}                       # 10 bought, 4 sold
        # 1,000,000 − (10×190 buy) + (4×200 sell) = 998,900
        assert bal["cash"]["GBP"] == pytest.approx(998_900.0)

    def test_commissions_and_fees_reduce_cash(self):
        from ops.ledger import replay_ledger_to_balances
        led = ImmutableLedger()
        led.append("CASH", {"amount": 100_000.0}, "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 10.0, price=100.0)], [_intent("AAPL", "BUY")], "2026-01-02T00:00:00")
        led.append("COMMISSION", {"amount": 5.0}, "2026-01-02T00:00:00")
        led.append("FEE", {"amount": 2.0}, "2026-01-02T00:00:00")
        bal = replay_ledger_to_balances(led)
        # 100,000 − 1000 (buy) − 5 (commission) − 2 (fee) = 98,993
        assert bal["cash"]["GBP"] == pytest.approx(98_993.0)

    def test_feeds_reconcile_cash_leg(self):
        from ops.ledger import replay_ledger_to_balances
        from ops.reconciliation import reconcile
        led = ImmutableLedger()
        led.append("CASH", {"amount": 50_000.0}, "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 100.0, price=100.0)], [_intent("AAPL", "BUY")], "2026-01-02T00:00:00")
        internal = replay_ledger_to_balances(led)   # positions {AAPL:100}, cash {GBP: 40,000}
        assert internal["cash"]["GBP"] == pytest.approx(40_000.0)
        # broker agrees on positions AND cash → clean
        assert reconcile(internal, {"positions": {"AAPL": 100.0}, "cash": {"GBP": 40_000.0}},
                         asof="2026-01-02T00:00:00", share_tol=1.0, cash_tol=1.0).clean
        # broker cash disagrees → a cash break
        rep = reconcile(internal, {"positions": {"AAPL": 100.0}, "cash": {"GBP": 39_000.0}},
                        asof="2026-01-02T00:00:00", share_tol=1.0, cash_tol=1.0)
        assert not rep.clean and rep.breaks[0].dimension == "cash"

    def test_cash_complete_flags_a_fully_priced_trail(self):
        # §17(b): the replay self-reports whether its cash figure is reliable, so the LIVE
        # _reconcile can SKIP (not false-break) the cash leg on an incomplete trail.
        from ops.ledger import replay_ledger_to_balances
        led = ImmutableLedger()
        led.append("CASH", {"amount": 50_000.0}, "2026-01-01T00:00:00")
        _cycle(led, [_fill("AAPL", 10.0, price=100.0)], [_intent("AAPL", "BUY")], "2026-01-02T00:00:00")
        assert replay_ledger_to_balances(led)["cash_complete"] is True

    def test_cash_complete_false_when_a_fill_lacks_sign_or_price(self):
        from ops.ledger import replay_ledger_to_balances
        led = ImmutableLedger()
        led.append("CASH", {"amount": 50_000.0}, "2026-01-01T00:00:00")
        led.append("FILL", {"symbol": "AAPL", "signed_qty": 10.0}, "2026-01-02T00:00:00")  # no price
        bal = replay_ledger_to_balances(led)
        assert bal["cash_complete"] is False
        assert bal["cash"]["GBP"] == pytest.approx(50_000.0)      # the unpriceable fill is skipped
