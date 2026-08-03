"""
Phase 2 Tests — Backtest / Walk-Forward Harness
===============================================
Covers backtesting.metrics (pure performance functions, known-answer) and
backtesting.harness (PAPER-mode engine replay, net-of-cost, walk-forward,
determinism, drawdown-governor behaviour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core.engine.engine as eng
from backtesting import metrics as m
from backtesting.harness import Backtester
from research.validation import PurgedWalkForwardSplitter


def _prices(n: int = 260, seed: int = 11, cols=("AAA", "BBB", "CCC")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {c: 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))) for c in cols},
        index=idx,
    )


# ── 1. Pure performance metrics (known answers) ──────────────────────────────────

class TestMetrics:

    def test_max_drawdown_known(self):
        # equity 1 -> 1.2 -> 0.9 -> 1.0; worst peak-to-trough = (0.9-1.2)/1.2 = -0.25
        returns = pd.Series([0.2, -0.25, 1.0 / 0.9 - 1.0])
        assert m.max_drawdown(returns) == pytest.approx(0.25, abs=1e-9)

    def test_max_drawdown_monotonic_up_is_zero(self):
        assert m.max_drawdown(pd.Series([0.01, 0.02, 0.01])) == pytest.approx(0.0, abs=1e-12)

    def test_hit_rate(self):
        # positives: 0.01, 0.02 -> 2 of 5 (0.0 is not a hit)
        assert m.hit_rate(pd.Series([0.01, -0.01, 0.02, 0.0, -0.03])) == pytest.approx(0.4)

    def test_sharpe_zero_mean_is_zero(self):
        assert m.sharpe(pd.Series([0.01, -0.01, 0.01, -0.01])) == pytest.approx(0.0, abs=1e-12)

    def test_sharpe_matches_formula(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.01, 500))
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert m.sharpe(r, periods_per_year=252) == pytest.approx(expected, rel=1e-9)

    def test_ann_vol_matches_formula(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.0, 0.01, 1000))
        assert m.ann_vol(r, periods_per_year=252) == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-9)

    def test_ann_return_is_geometric(self):
        r = pd.Series([0.01] * 252)
        assert m.ann_return(r, periods_per_year=252) == pytest.approx(1.01 ** 252 - 1.0, rel=1e-9)

    def test_sortino_target_downside_deviation(self):
        # mean 0.005; downside dev = sqrt(mean(min(r,0)^2)) = sqrt((0.01^2+0.01^2)/4)
        r = pd.Series([0.02, -0.01, 0.02, -0.01])
        dd = np.sqrt((0.01 ** 2 + 0.01 ** 2) / 4.0)
        assert m.sortino(r, periods_per_year=1) == pytest.approx(0.005 / dd, rel=1e-9)

    def test_calmar_is_ann_return_over_max_drawdown(self):
        r = pd.Series([0.10, -0.20, 0.05, 0.03, -0.04])
        assert m.calmar(r, periods_per_year=12) == pytest.approx(
            m.ann_return(r, periods_per_year=12) / m.max_drawdown(r), rel=1e-9
        )

    def test_summarize_has_all_keys(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.0005, 0.01, 300))
        out = m.summarize(r, periods_per_year=252)
        for key in ("ann_return", "ann_vol", "sharpe", "sortino",
                    "max_drawdown", "calmar", "hit_rate"):
            assert key in out and isinstance(out[key], float)


# ── 2. Backtester (PAPER-mode engine replay) ─────────────────────────────────────

class TestBacktester:

    def test_run_produces_result_with_metrics(self):
        res = Backtester(rebalance="W", warmup=40, seed=7).run(_prices())
        assert isinstance(res.equity_curve, pd.Series)
        assert isinstance(res.returns, pd.Series)
        assert isinstance(res.weights_history, pd.DataFrame)
        assert res.n_rebalances > 0
        assert len(res.returns) == res.n_rebalances
        for key in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
                    "calmar", "hit_rate", "avg_turnover", "total_cost_bps", "n_rebalances"):
            assert key in res.metrics
        # Long-only at every rebalance; effectively unlevered. The optimiser TARGET
        # book is exactly unlevered (sum <= 1); the PAPER achieved book reconciled
        # here drifts a few bps above 1.0 with mark-to-market on carried holdings
        # between rebalances (not leverage), so allow a small tolerance.
        assert (res.weights_history.sum(axis=1) <= 1.02).all()
        assert (res.weights_history.to_numpy() >= -1e-9).all()
        assert isinstance(res.summary(), str)

    def test_deterministic(self):
        prices = _prices()
        a = Backtester(rebalance="W", warmup=40, seed=7).run(prices)
        b = Backtester(rebalance="W", warmup=40, seed=7).run(prices)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)

    def test_costs_reduce_terminal_equity(self):
        prices = _prices()
        with_cost = Backtester(rebalance="W", warmup=40, cost_bps_per_turnover=50.0, seed=7).run(prices)
        no_cost = Backtester(rebalance="W", warmup=40, cost_bps_per_turnover=0.0, seed=7).run(prices)
        assert with_cost.equity_curve.iloc[-1] <= no_cost.equity_curve.iloc[-1] + 1e-12
        assert with_cost.metrics["total_cost_bps"] >= 0.0

    def test_financing_reduces_levered_returns(self):
        # OPT-2 / METH-4: a LEVERED book pays borrow on the levered notional, so its
        # net-of-cost return is strictly lower with financing than without.
        prices = _prices()
        common = dict(rebalance="W", warmup=40, seed=7, target_vol=0.60, max_gross_leverage=2.0)
        financed = Backtester(financing_rate_annual=0.06, **common).run(prices)
        free = Backtester(financing_rate_annual=0.0, **common).run(prices)
        # the book actually levered (gross > 1 on at least one rebalance)...
        assert (financed.weights_history.abs().sum(axis=1) > 1.05).any()
        # ...so financing strictly lowered terminal equity and raised reported cost.
        assert financed.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
        assert financed.metrics["total_cost_bps"] > free.metrics["total_cost_bps"]

    def test_financing_does_not_touch_unlevered_book(self):
        # An unlevered config (default max_gross_leverage=1.0) never pays borrow — even
        # when mark-to-market drifts gross a few bps above 1.0 — so the financing rate
        # has no effect on the equity curve.
        prices = _prices()
        common = dict(rebalance="W", warmup=40, seed=7)  # default leverage 1.0
        a = Backtester(financing_rate_annual=0.06, **common).run(prices)
        b = Backtester(financing_rate_annual=0.0, **common).run(prices)
        pd.testing.assert_series_equal(a.equity_curve, b.equity_curve)

    def test_walk_forward_reports_per_split(self):
        splitter = PurgedWalkForwardSplitter(
            train_size=6, valid_size=2, test_size=3, embargo_size=1, label_horizon=1
        )
        res = Backtester(rebalance="W", warmup=16, seed=7).run(_prices(), splitter=splitter)
        assert isinstance(res.per_split, list) and len(res.per_split) >= 1
        for seg in res.per_split:
            assert "sharpe" in seg and "max_drawdown" in seg

    def test_feeds_rising_drawdown_into_cycle(self, monkeypatch):
        # The harness must compute drawdown_current from its running strategy equity
        # and feed it into each CycleInputs (so the governor can engage). We isolate
        # that responsibility with a stub that holds a full long book through a crash,
        # so the strategy equity genuinely falls (the real engine would de-risk — its
        # governor behaviour is tested separately in test_phase9).
        from types import SimpleNamespace

        seen: list[float] = []

        def _stub_run_cycle(self, inputs):
            seen.append(float(inputs.drawdown_current))
            n = len(inputs.symbols)
            return SimpleNamespace(
                blocked=False, target_weights={s: 1.0 / n for s in inputs.symbols}
            )

        monkeypatch.setattr(eng.TradingEngine, "run_cycle", _stub_run_cycle)
        idx = pd.date_range("2022-01-01", periods=160, freq="D", tz="UTC")
        crash = np.concatenate([np.full(80, 0.0005), np.full(80, -0.03)])   # calm then crash
        prices = pd.DataFrame(
            {c: 100.0 * np.exp(np.cumsum(crash)) for c in ("AAA", "BBB")}, index=idx
        )
        Backtester(rebalance="W", warmup=20, seed=7, cost_bps_per_turnover=0.0).run(prices)
        assert seen, "engine was never invoked"
        assert seen[0] == pytest.approx(0.0, abs=1e-9)     # flat at the start
        assert max(seen) > 0.05                             # full book in a crash → deep drawdown


class TestAchievedBook:

    def test_paper_book_carries_achieved_weights(self, monkeypatch):
        # When the engine reports an achieved book (fills), the harness must carry
        # THAT into the next cycle — not assume the target was reached.
        from types import SimpleNamespace

        seen_books: list[dict] = []

        def _stub_run_cycle(self, inputs):
            seen_books.append(dict(inputs.current_weights))
            n = len(inputs.symbols)
            return SimpleNamespace(
                blocked=False,
                target_weights={s: 1.0 / n for s in inputs.symbols},
                achieved_weights={s: 0.5 / n for s in inputs.symbols},   # half-filled
            )

        monkeypatch.setattr(eng.TradingEngine, "run_cycle", _stub_run_cycle)
        idx = pd.date_range("2022-01-01", periods=120, freq="D", tz="UTC")
        rng = np.random.default_rng(5)
        prices = pd.DataFrame(
            {c: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))) for c in ("AAA", "BBB")},
            index=idx,
        )
        Backtester(rebalance="W", warmup=20, seed=7).run(prices)
        assert len(seen_books) >= 2
        assert seen_books[1]["AAA"] == pytest.approx(0.25)   # achieved, not the 0.5 target

    def test_stub_without_achieved_weights_still_works(self, monkeypatch):
        # Backward-compatible: results lacking achieved_weights fall back to targets.
        from types import SimpleNamespace

        def _stub_run_cycle(self, inputs):
            n = len(inputs.symbols)
            return SimpleNamespace(
                blocked=False, target_weights={s: 1.0 / n for s in inputs.symbols}
            )

        monkeypatch.setattr(eng.TradingEngine, "run_cycle", _stub_run_cycle)
        idx = pd.date_range("2022-01-01", periods=120, freq="D", tz="UTC")
        rng = np.random.default_rng(6)
        prices = pd.DataFrame(
            {c: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))) for c in ("AAA", "BBB")},
            index=idx,
        )
        res = Backtester(rebalance="W", warmup=20, seed=7).run(prices)
        assert res.n_rebalances >= 1
