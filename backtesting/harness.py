"""
TradingEngineResearch — Backtest / Walk-Forward Harness
============================================
Replays the real TradingEngineResearch engine (PAPER mode) over a price history, net of cost, on
optional purged walk-forward splits, and reports risk-adjusted performance.

At each rebalance date ``t`` the harness builds point-in-time-safe ``CycleInputs``
from data strictly ≤ ``t`` (so no future leaks in), runs the full 13-step pipeline,
takes the risk-approved book (``CycleResult.target_weights``; the carried book if the
cycle was blocked), holds it to the next rebalance, and books the net-of-cost return.
This exercises regime detection, crisis tightening, the optimiser + CVaR enforcement,
the fail-closed risk gate + drawdown governor, and TCA — the same code that trades.

Determinism: timestamps come from the data (never wall-clock) and the engine's
stateful singletons are reset at the start of every ``run`` with a fixed seed, so two
runs on the same inputs are bit-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtesting import metrics as _metrics
from core.engine.engine import CycleInputs, TradingEngine

logger = logging.getLogger(__name__)

__all__ = ["Backtester", "BacktestResult"]


def _reset_engine_state(seed: int) -> None:
    """Reset the engine's process-wide singletons so each run is self-contained
    and reproducible. Missing singletons are ignored (best-effort)."""
    np.random.seed(seed)
    for module_name, fn_name in (
        ("core.engine.optimizer", "reset_view_tracker"),
        ("core.ml_return_model", "reset_model"),
        ("core.regime_engine", "reset_regime_engine"),
        ("core.crisis_manager", "reset_crisis_manager"),
        ("core.risk_manager", "reset_risk_manager"),
        ("execution.tca", "reset_tca_model"),
        ("ops.model_registry", "reset_model_registry"),
        ("learning.performance_tracker", "reset_performance_tracker"),
    ):
        try:
            module = __import__(module_name, fromlist=[fn_name])
            getattr(module, fn_name)()
        except Exception:  # noqa: BLE001 — optional singleton reset
            pass


@dataclass
class BacktestResult:
    """The full output of a backtest run."""

    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.Series
    cost_history: pd.Series
    metrics: dict
    per_split: list = field(default_factory=list)
    n_rebalances: int = 0

    def summary(self) -> str:
        m = self.metrics
        return (
            f"Backtest: {self.n_rebalances} rebalances | "
            f"ann_return={m.get('ann_return', 0.0):.2%} "
            f"ann_vol={m.get('ann_vol', 0.0):.2%} "
            f"sharpe={m.get('sharpe', 0.0):.2f} "
            f"max_dd={m.get('max_drawdown', 0.0):.2%} "
            f"turnover={m.get('avg_turnover', 0.0):.2f} "
            f"cost={m.get('total_cost_bps', 0.0):.1f}bps"
        )


class Backtester:
    """Replays ``TradingEngine`` over a price history; see the module docstring."""

    def __init__(
        self,
        mode: str = "PAPER",
        capital_gbp: float = 1_000_000.0,
        rebalance: str = "W",
        warmup: int = 60,
        cost_bps_per_turnover: float = 10.0,
        periods_per_year: int | None = None,
        stale_threshold_seconds: float = 300.0,
        seed: int = 42,
        target_vol: float | None = None,
        max_gross_leverage: float = 1.0,
        max_position_weight: float | None = None,
        cvar_limit: float | None = None,
        signal_tilt_strength: float = 5e-4,
        max_lever_up_step: float | None = None,
        financing_rate_annual: float = 0.06,
    ) -> None:
        self.mode = mode
        self.capital_gbp = float(capital_gbp)
        self.rebalance = rebalance
        self.warmup = int(warmup)
        self.cost_bps_per_turnover = float(cost_bps_per_turnover)
        self.periods_per_year = periods_per_year
        self.stale_threshold_seconds = float(stale_threshold_seconds)
        self.seed = int(seed)
        # Risk-budget passthrough to the engine (aggressiveness for returns tuning).
        self.target_vol = target_vol
        self.max_gross_leverage = float(max_gross_leverage)
        self.max_position_weight = max_position_weight
        self.cvar_limit = cvar_limit
        self.signal_tilt_strength = float(signal_tilt_strength)
        # OPT-1: None = unchanged. Lets a backtest MEASURE the leverage ramp
        # before anyone decides whether to run it live.
        self.max_lever_up_step = (
            None if max_lever_up_step is None else float(max_lever_up_step)
        )
        # OPT-2 / METH-4: borrow/financing cost on the LEVERED portion of the book
        # (gross > 1), charged per holding period. Without this, a levered backtest
        # books levered gross returns but pays no borrow — overstating net-of-cost for
        # the 2-3.5x tiers. Default ~6%/yr (an IBKR-ish margin rate); 0.0 disables it.
        # An UNLEVERED book (gross <= 1) pays nothing, so unlevered runs are unaffected.
        self.financing_rate_annual = float(financing_rate_annual)

    # ── rebalance schedule ────────────────────────────────────────────────────────

    def _rebalance_dates(self, index: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Last actual trading date in each ``rebalance`` period, at/after warmup."""
        naive = index.tz_localize(None) if index.tz is not None else index
        periods = naive.to_period(self.rebalance)
        last_pos: dict[object, int] = {}
        for pos, per in enumerate(periods):
            last_pos[per] = pos                       # keep the last position per period
        return [index[p] for p in sorted(last_pos.values()) if p >= self.warmup]

    def _infer_ppy(self, dates: list[pd.Timestamp]) -> int:
        if self.periods_per_year is not None:
            return int(self.periods_per_year)
        if len(dates) < 2:
            return 252
        gaps = np.diff([d.value for d in dates]) / 8.64e13   # ns → days
        median_days = float(np.median(gaps))
        return max(1, int(round(365.25 / median_days))) if median_days > 0 else 252

    # ── inputs ────────────────────────────────────────────────────────────────────

    def _build_inputs(
        self, prices_hist: pd.DataFrame, asof: pd.Timestamp,
        book: dict, drawdown: float,
    ) -> CycleInputs:
        symbols = list(prices_hist.columns)
        rets = prices_hist.pct_change().dropna().to_numpy()
        last = prices_hist.iloc[-1]
        micro = {
            s: {"spread_bps": 6.0, "adv": 2.0e7, "price": float(last[s]), "participation": 0.02}
            for s in symbols
        }
        return CycleInputs(
            asof_time=asof.to_pydatetime(),
            symbols=symbols,
            prices=prices_hist,
            returns_matrix=rets if rets.size else None,
            portfolio_returns=rets.mean(axis=1) if rets.size else None,
            portfolio_values=(1.0 + rets.mean(axis=1)).cumprod() if rets.size else None,
            current_weights=dict(book),
            capital_gbp=self.capital_gbp,
            drawdown_current=float(drawdown),
            market_microstructure=micro,
        )

    # ── run ────────────────────────────────────────────────────────────────────────

    def run(
        self, prices: pd.DataFrame,
        splitter: Any = None,           # any object exposing .split(timestamps) -> [(train, valid, test)]
    ) -> BacktestResult:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must be indexed by a DatetimeIndex.")
        prices = prices.sort_index()
        reb_dates = self._rebalance_dates(prices.index)
        if len(reb_dates) < 2:
            raise ValueError(
                f"Not enough rebalance dates ({len(reb_dates)}) after warmup={self.warmup}; "
                "supply more history, lower warmup, or use a finer rebalance frequency."
            )

        _reset_engine_state(self.seed)
        engine = TradingEngine(
            mode=self.mode, capital_gbp=self.capital_gbp,
            stale_threshold_seconds=self.stale_threshold_seconds,
            target_vol=self.target_vol, max_gross_leverage=self.max_gross_leverage,
            max_position_weight=self.max_position_weight, cvar_limit=self.cvar_limit,
            signal_tilt_strength=self.signal_tilt_strength,
            max_lever_up_step=self.max_lever_up_step,
        )

        book: dict[str, float] = {}
        equity = 1.0
        peak = 1.0
        ret_dates: list[pd.Timestamp] = []
        ret_vals: list[float] = []
        eq_vals: list[float] = []
        w_rows: list[dict] = []
        w_dates: list[pd.Timestamp] = []
        turnover_vals: list[float] = []
        cost_vals: list[float] = []

        for t_now, t_next in zip(reb_dates[:-1], reb_dates[1:]):
            drawdown = 0.0 if peak <= 0 else max(0.0, 1.0 - equity / peak)
            inputs = self._build_inputs(prices.loc[:t_now], t_now, book, drawdown)
            result = engine.run_cycle(inputs)

            # Carry the current book on a blocked cycle. Otherwise carry what the
            # engine ACHIEVED (reconciled from fills) — partial fills must not be
            # booked as if the target had been reached. RESEARCH plans no orders,
            # so there the target book is the (hypothetical) replay book.
            if result.blocked:
                new_book = dict(book)
            else:
                achieved = getattr(result, "achieved_weights", None)
                src = achieved if (achieved is not None and self.mode != "RESEARCH") \
                    else result.target_weights
                new_book = {s: float(w) for s, w in src.items()}

            symbols = set(new_book) | set(book)
            turnover = float(sum(abs(new_book.get(s, 0.0) - book.get(s, 0.0)) for s in symbols))
            cost_return = turnover * self.cost_bps_per_turnover * 1e-4

            # OPT-2 / METH-4: borrow cost on the levered notional (gross - 1) over the
            # actual holding period. Only for a config that INTENTIONALLY levers
            # (max_gross_leverage > 1) — an unlevered book's gross may drift a few bps
            # above 1.0 from mark-to-market (appreciation, not borrowing), which must
            # not be charged financing.
            financing_return = 0.0
            if self.max_gross_leverage > 1.0:
                gross_book = float(sum(abs(w) for w in new_book.values()))
                period_years = max((t_next - t_now).days, 1) / 365.25
                financing_return = max(gross_book - 1.0, 0.0) * self.financing_rate_annual * period_years
            total_cost_return = cost_return + financing_return

            fwd = (prices.loc[t_next] / prices.loc[t_now] - 1.0)
            gross_return = float(sum(new_book.get(s, 0.0) * float(fwd[s]) for s in new_book))
            net_return = gross_return - total_cost_return

            equity *= (1.0 + net_return)
            peak = max(peak, equity)
            book = new_book

            w_rows.append(dict(new_book))
            w_dates.append(t_now)
            ret_dates.append(t_next)
            ret_vals.append(net_return)
            eq_vals.append(equity)
            turnover_vals.append(turnover)
            cost_vals.append(total_cost_return * 1e4)       # bps (trading + financing)

        returns = pd.Series(ret_vals, index=pd.DatetimeIndex(ret_dates), name="returns")
        equity_curve = pd.Series(eq_vals, index=returns.index, name="equity")
        all_symbols = sorted(set().union(*w_rows)) if w_rows else []
        weights_history = pd.DataFrame(w_rows, index=pd.DatetimeIndex(w_dates)).reindex(
            columns=all_symbols
        ).fillna(0.0)
        turnover_history = pd.Series(turnover_vals, index=returns.index, name="turnover")
        cost_history = pd.Series(cost_vals, index=returns.index, name="cost_bps")

        ppy = self._infer_ppy(reb_dates)
        metrics = _metrics.summarize(returns, periods_per_year=ppy)
        metrics["avg_turnover"] = float(np.mean(turnover_vals)) if turnover_vals else 0.0
        metrics["total_cost_bps"] = float(np.sum(cost_vals))
        metrics["n_rebalances"] = float(len(returns))

        per_split: list[dict] = []
        if splitter is not None and len(returns) > 0:
            try:
                for _train, _valid, test_idx in splitter.split(returns.index):
                    seg = returns.iloc[np.asarray(test_idx, dtype=int)]
                    per_split.append(_metrics.summarize(seg, periods_per_year=ppy))
            except Exception as exc:  # noqa: BLE001 — splitter is optional/diagnostic
                logger.warning("walk-forward split failed (%s); per_split left empty.", exc)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            weights_history=weights_history,
            turnover_history=turnover_history,
            cost_history=cost_history,
            metrics=metrics,
            per_split=per_split,
            n_rebalances=len(returns),
        )
