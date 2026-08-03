"""
Phase 7 Tests — Execution Engine, TCA, and Capacity
===================================================
Covers every Phase 7 test target from the build spec:

  - Order state machine: complete valid transitions, no invalid transitions
  - ex_ante_cost_model monotonically increasing in participation rate
  - update_cost_priors converges k1/k2 toward observed values (EMA) and feeds
    back into ex_ante_cost_model
  - capacity_score returns 0.0 below the minimum ADV threshold
  - ExecutionReport / ChildOrderPlan instantiable
  - schedule_order regime behaviour (stressed halves size; URGENT_DERISK uses
    market orders; normal/cautious stay passive; cautious halves participation)
  - ex_post_cost_analysis returns the documented keys; slippage monotone
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.data_contracts import FillEvent, OrderIntent
from execution import capacity_model as cap
from execution import execution_engine as ee
from execution import slippage_model as slip
from execution import tca
from execution.execution_engine import OrderState as S


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _order(urgency: str = "NORMAL", weight: float = 0.02) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL", direction="BUY", target_weight=weight, expected_cost_bps=8.0,
        urgency=urgency, alpha_half_life_minutes=60,
        decision_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        model_version="v6.0", regime_state="trending", risk_approved=True,
    )


def _fill(qty: float, fill_price: float, slippage_bps: float) -> FillEvent:
    return FillEvent(
        order_id="AAPL", symbol="AAPL", qty=qty, fill_price=fill_price,
        decision_price=100.0, arrival_price=100.2, slippage_bps=slippage_bps,
        fill_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
    )


_MS = {"target_qty": 10000, "max_participation": 0.05, "spread_bps": 6, "time_to_close": 390}


# ── 1. Slippage model ────────────────────────────────────────────────────────────

class TestSlippageModel:

    def test_monotone_in_participation(self):
        vals = [slip.estimate_slippage(1000, 5e7, 150.0, 5.0, participation=p)
                for p in (0.01, 0.05, 0.10, 0.25)]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_non_negative_and_at_least_half_spread(self):
        s = slip.estimate_slippage(1000, 5e7, 150.0, 8.0, participation=0.05)
        assert s >= 4.0     # half of an 8 bps spread


# ── 2. TCA ───────────────────────────────────────────────────────────────────────

class TestTCA:

    def setup_method(self):
        tca.reset_tca_model()

    def test_ex_ante_monotone_in_participation(self):
        costs = [tca.ex_ante_cost_model("AAPL", 1000, "BUY", 5.0, 150.0, 5e7, p)
                 for p in (0.01, 0.02, 0.05, 0.10, 0.20)]
        assert all(costs[i] < costs[i + 1] for i in range(len(costs) - 1))

    def test_update_cost_priors_ema_convergence(self):
        model = tca.get_tca_model()
        for _ in range(60):
            tca.update_cost_priors({"observed_k1": 0.5, "observed_k2": 0.2})
        assert model.k1 == pytest.approx(0.5, abs=1e-3)
        assert model.k2 == pytest.approx(0.2, abs=1e-3)

    def test_priors_feed_back_into_ex_ante(self):
        before = tca.ex_ante_cost_model("AAPL", 1000, "BUY", 5.0, 150.0, 5e7, 0.1)
        for _ in range(60):
            tca.update_cost_priors({"observed_k1": 1.0, "observed_k2": 0.5})
        after = tca.ex_ante_cost_model("AAPL", 1000, "BUY", 5.0, 150.0, 5e7, 0.1)
        assert after > before   # higher impact coefficients ⇒ higher predicted cost

    def test_ex_post_returns_documented_keys(self):
        fills = [_fill(5000, 100.5, 5.0), _fill(5000, 100.3, 3.0)]
        out = tca.ex_post_cost_analysis(fills, [_order()])
        for key in ("realized_spread_cost_bps", "realized_impact_bps", "realized_fee_bps",
                    "total_realized_cost_bps", "vs_expected_delta_bps", "passive_fill_ratio"):
            assert key in out

    def test_ex_post_empty_fills(self):
        out = tca.ex_post_cost_analysis([], [_order()])
        assert out["total_realized_cost_bps"] == 0.0

    def test_ex_post_sell_side_is_signed_correctly(self):
        # An adverse SELL — price falls from the decision (100.0) through arrival
        # (99.8) to the fill (99.6) — is a genuine cost to the seller and must
        # register as a POSITIVE realised cost, not a phantom gain. Decisions and
        # fills are matched by symbol (OrderIntent has no order_id).
        sell = OrderIntent(
            symbol="MSFT", direction="SELL", target_weight=-0.02, expected_cost_bps=8.0,
            urgency="NORMAL", alpha_half_life_minutes=60,
            decision_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
            model_version="v6.0", regime_state="trending", risk_approved=True,
        )
        fill = FillEvent(
            order_id="MSFT-1", symbol="MSFT", qty=5000, fill_price=99.6,
            decision_price=100.0, arrival_price=99.8, slippage_bps=40.0,
            fill_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        )
        out = tca.ex_post_cost_analysis([fill], [sell])
        assert out["total_realized_cost_bps"] > 0.0
        assert out["realized_impact_bps"] > 0.0
        assert out["passive_fill_ratio"] == 0.0   # adverse impact ⇒ not a passive fill


# ── 3. Capacity model ────────────────────────────────────────────────────────────

class TestCapacityModel:

    def test_zero_below_min_adv(self):
        assert cap.capacity_score("X", 0.01, 1_000_000.0, {"adv": 1_000_000.0}) == 0.0

    def test_full_capacity_when_ample(self):
        score = cap.capacity_score(
            "X", 0.01, 1_000_000.0, {"adv": 5e7, "spread_bps": 5, "volatility": 100}
        )
        assert score == pytest.approx(1.0)

    def test_score_scales_down_for_oversize(self):
        # Required notional far exceeds single-day capacity ⇒ score < 1.
        score = cap.capacity_score(
            "X", 0.9, 1_000_000_000.0, {"adv": 6_000_000.0, "spread_bps": 5, "volatility": 100}
        )
        assert 0.0 <= score < 1.0

    def test_portfolio_report_flags(self):
        report = cap.portfolio_capacity_report(
            {"A": 0.02, "B": 0.02}, 1_000_000.0,
            {"A": {"adv": 5e7}, "B": {"adv": 1_000_000.0}},
        )
        for key in ("per_symbol", "aggregate_score", "n_tradeable", "flags"):
            assert key in report
        assert "B:below_min_adv" in report["flags"]


# ── 4. Order state machine ───────────────────────────────────────────────────────

class TestOrderStateMachine:

    def test_full_valid_lifecycle(self):
        sm = ee.OrderStateMachine()
        for state in (S.STAGED, S.WORKING, S.PARTIAL, S.FILLED):
            sm.transition(state)
        assert sm.state == S.FILLED and sm.is_terminal()
        assert sm.history == [S.NEW, S.STAGED, S.WORKING, S.PARTIAL, S.FILLED]

    @pytest.mark.parametrize(
        "frm, to",
        [
            (S.FILLED, S.NEW), (S.NEW, S.FILLED), (S.NEW, S.WORKING),
            (S.REJECTED, S.WORKING), (S.CANCELLED, S.FILLED), (S.STAGED, S.PARTIAL),
        ],
    )
    def test_invalid_transitions_raise(self, frm, to):
        sm = ee.OrderStateMachine(frm)
        with pytest.raises(ValueError, match="Invalid order-state transition"):
            sm.transition(to)

    def test_terminal_states_have_no_exits(self):
        for terminal in ee.TERMINAL_STATES:
            sm = ee.OrderStateMachine(terminal)
            assert sm.is_terminal()
            assert all(not sm.can_transition(s) for s in S)

    def test_partial_can_repeat(self):
        sm = ee.OrderStateMachine(S.PARTIAL)
        assert sm.can_transition(S.PARTIAL)   # successive partial fills


# ── 5. Child-order scheduling ────────────────────────────────────────────────────

class TestScheduleOrder:

    def test_normal_is_passive_limit_full_size(self):
        plans = ee.schedule_order(_order(), {**_MS, "execution_regime": "normal_exec"})
        assert plans and all(p.order_type == "LIMIT" for p in plans)
        assert sum(p.qty for p in plans) == pytest.approx(10000.0)

    def test_cautious_halves_participation(self):
        normal = ee.schedule_order(_order(), {**_MS, "execution_regime": "normal_exec"})
        cautious = ee.schedule_order(_order(), {**_MS, "execution_regime": "cautious_exec"})
        assert cautious[0].participation == pytest.approx(0.5 * normal[0].participation)

    def test_stressed_halves_size_and_stays_passive(self):
        plans = ee.schedule_order(_order(), {**_MS, "execution_regime": "stressed_exec"})
        assert sum(p.qty for p in plans) == pytest.approx(5000.0)
        assert all(p.order_type == "LIMIT" for p in plans)

    def test_urgent_derisk_uses_market_at_full_size(self):
        plans = ee.schedule_order(_order("URGENT_DERISK"), {**_MS, "execution_regime": "stressed_exec"})
        assert all(p.order_type == "MARKET" for p in plans)
        assert sum(p.qty for p in plans) == pytest.approx(10000.0)   # de-risk is NOT downsized
        assert all(p.tag == "derisk" for p in plans)

    def test_qty_derived_from_weight_when_no_target_qty(self):
        ms = {"capital_gbp": 1_000_000.0, "price": 100.0, "execution_regime": "normal_exec"}
        plans = ee.schedule_order(_order(weight=0.02), ms)
        # 0.02 * 1,000,000 / 100 = 200 shares
        assert sum(p.qty for p in plans) == pytest.approx(200.0)

    def test_stressed_caps_participation_at_two_pct(self):
        # stressed_exec must never exceed the 2% ADV participation cap, even when
        # the caller hands in a higher max_participation.
        plans = ee.schedule_order(_order(), {**_MS, "max_participation": 0.05, "execution_regime": "stressed_exec"})
        assert all(p.participation <= 0.02 + 1e-12 for p in plans)
        assert plans[0].participation == pytest.approx(0.02)

    def test_normal_caps_participation_at_five_pct(self):
        # normal_exec must never exceed the 5% ADV participation cap.
        plans = ee.schedule_order(_order(), {**_MS, "max_participation": 0.10, "execution_regime": "normal_exec"})
        assert all(p.participation <= 0.05 + 1e-12 for p in plans)
        assert plans[0].participation == pytest.approx(0.05)


# ── 5b. Execution mode gate (Rule 7: mode is explicit, never inferred) ────────────

class TestExecutionModeGate:

    def test_research_mode_is_default_and_passes(self):
        plans = ee.schedule_order(_order(), {**_MS, "execution_regime": "normal_exec"})
        assert plans  # RESEARCH default never blocks scheduling

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown trading mode"):
            ee.schedule_order(_order(), {**_MS, "execution_regime": "normal_exec"}, mode="TURBO")

    def test_live_requires_risk_approval(self):
        unapproved = OrderIntent(
            symbol="AAPL", direction="BUY", target_weight=0.02, expected_cost_bps=8.0,
            urgency="NORMAL", alpha_half_life_minutes=60,
            decision_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
            model_version="v6.0", regime_state="trending", risk_approved=False,
        )
        with pytest.raises(ValueError, match="risk_approved"):
            ee.schedule_order(unapproved, {**_MS, "execution_regime": "normal_exec"}, mode="LIVE")

    def test_live_with_risk_approval_passes(self):
        plans = ee.schedule_order(_order(), {**_MS, "execution_regime": "normal_exec"}, mode="LIVE")
        assert plans and all(p.order_type == "LIMIT" for p in plans)


# ── 6. Execution report ──────────────────────────────────────────────────────────

class TestExecutionReport:

    def test_dataclass_instantiable(self):
        report = ee.ExecutionReport(
            symbol="AAPL", order_id="o1", expected_cost_bps=8.0, realized_cost_bps=9.0,
            fill_rate=1.0, avg_slippage_bps=4.0, passive_fill_ratio=0.6,
            implementation_shortfall_bps=5.0, warnings=[], execution_regime_used="normal_exec",
        )
        assert report.symbol == "AAPL"
        plan = ee.ChildOrderPlan("AAPL", "BUY", 100.0, "LIMIT", -3.0, 0.0, 0.05, 0, "passive")
        assert plan.order_type == "LIMIT"

    def test_compute_from_fills(self):
        fills = [_fill(5000, 100.5, 5.0), _fill(5000, 100.3, 3.0)]
        report = ee.compute_execution_report(
            _order(), fills, target_qty=10000, execution_regime="normal_exec", expected_cost_bps=8.0,
        )
        assert report.fill_rate == pytest.approx(1.0)
        assert report.avg_slippage_bps == pytest.approx(4.0)
        assert report.execution_regime_used == "normal_exec"

    def test_incomplete_fill_warns(self):
        report = ee.compute_execution_report(
            _order(), [_fill(3000, 100.5, 5.0)], target_qty=10000,
        )
        assert report.fill_rate < 1.0
        assert "INCOMPLETE_FILL" in report.warnings

    def test_no_fills_warns(self):
        report = ee.compute_execution_report(_order(), [], target_qty=10000)
        assert "NO_FILLS" in report.warnings
        assert report.fill_rate == 0.0
