"""LOW-VOLATILITY / QUALITY, RE-TESTED with both cost fixes actually applied.

Pre-registration: `research/sleeves/lowvol_retest_prereg.md`, committed at `0b12f93`
BEFORE this module existed. One configuration, run once, no tuning.

WHAT THIS IS
============
Iteration 1 measured this sleeve as DEAD (B2 net Sharpe 0.324, excess -5.54%/yr) under two
cost defects that have since been fixed and validated against positive controls. Iteration
4 then **re-priced** that book arithmetically -- 119.5 -> 59.6/49.9bps one-way, net Sharpe
0.324 -> 0.715/0.779, excess -5.54%/yr -> -0.13%/+0.75%. **The books were never re-run.**
The universe bias was still baked into WHICH names were held, and the impact model was
still being fed a reference volatility instead of each name's own. This is the real run.

WHAT IS UNCHANGED
-----------------
The signal is not redesigned; that would be a different hypothesis. `build_signal`,
`risk_features`, the band grouping and every registered constant are imported verbatim
from `research.sleeves.low_vol_quality`. Long-only, top 30 by composite, equal weight,
monthly rebalance, no rebalance below 60 rankable names, +/-100% return clip, $2 price
floor and the 90%-trading-fraction filter (both inherited from the panel), delisting
returns applied by DATE within 62 days of the position closing and the name REMOVED from
holdings once its exit is booked.

WHAT CHANGES
------------
1. **Universe.** `spread_regime in {measured, upper_bound}`. Iteration 1 kept only
   `measured` and deleted 525,933 of 922,652 eligible cells panel-wide. `upper_bound`
   means the true spread is BELOW the estimator's resolution floor -- the name is CHEAP.
   In bands B2..B6 the corrected universe is 801,341 cells against 302,538 (+164.9%).
2. **Spread cost.** `spread_estimation.bounds_from_estimate` -- BOTH bounds, per cell.
3. **Impact cost.** `capacity_study.impact_cost_bounds` -- BOTH bounds, fed the name's OWN
   252-day realised daily volatility, which the signal already computes. The old model had
   no volatility term at all and charged a flat 100bps/side.
4. **Reporting.** Vol-matched active return, the benchmark through the DSR gate, Sharpe per
   decade, and P&L / gross-notional concentration. None of these were run in iteration 1.

ONE BOOK, TWO PRICES
--------------------
The signal cannot see costs, so the HOLDINGS are identical under both bounds and only the
cost stream differs. Both are therefore accumulated in a single pass, which makes it
impossible for the two reported books to have drifted apart.

THE ONE-MONTH DATING DEFECT (found 2026-07-28, `date_convention` below)
----------------------------------------------------------------------
`run_band` originally labelled every monthly slot with the FORMATION month -- the panel
row date the signal was ranked on -- but filled it with ``forward_return``, the
close-to-close return of the FOLLOWING month. Every slot was therefore dated ONE MONTH
EARLY.

**No within-series statistic can see this.** Mean, volatility, Sharpe, drawdown,
Newey-West t and the vol-matched active return are all invariant to shifting every
observation by a constant number of periods, which is why an independent bit-for-bit
re-implementation reproduced the series exactly and still did not catch it. It only bites
when the series is JOINED TO ANOTHER SERIES BY DATE. Measured with
`research.alignment.probe_alignment`: the band-B2 benchmark correlates with SPX at +0.189
contemporaneously and **+0.769 against SPX(t+1)** -- the signature of a series dated a
month early.

`date_convention` is the switch. ``FORMATION`` is the DEFAULT and reproduces every banked
number bit-for-bit; ``REALISATION`` shifts ``months`` (and ``pnl_by_name_month``) forward
by one so the index means the month the return was EARNED. Nothing else moves: the return
arrays are untouched, so every within-series statistic is identical under both. Only
`_decade_sharpes` can differ, and only for the handful of months that cross a decade
boundary.

THE TRAP THIS SLEEVE IS IN
--------------------------
`vol_matched_active` scales the benchmark by ``k = sigma_strategy / sigma_benchmark``, so
its mean is exactly ``(Sharpe_strategy - Sharpe_benchmark) * sigma_strategy``. This sleeve
runs at roughly 0.64x its benchmark's volatility BY CONSTRUCTION, so vol-matching
de-levers the benchmark and flatters the strategy -- the same variance-drag mechanism that
killed PEAD, pointing the other way. It is still the right statistic (it is the only one
invariant to leverage) but the raw arithmetic and geometric excesses are reported beside
it, and the verdict is really a statement about Sharpe against Sharpe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.capacity_study import (
    COMMISSION_MAX_FRACTION,
    COMMISSION_MIN_PER_ORDER,
    COMMISSION_PER_SHARE,
    FX_COST_EACH_WAY,
    IMPACT_COEFFICIENT_CONSERVATIVE,
    IMPACT_COEFFICIENT_REALISTIC,
    impact_fraction,
)
from research.alignment import FORMATION, REALISATION
from research.delisting import CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW
from research.delisting import in_window, in_window_mask
from research.multiasset.carry import newey_west_tstat, vol_matched_active
from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.low_vol_quality import (
    BAND_GROUPS,
    FORWARD_RETURN_CLIP,
    MIN_CROSS_SECTION,
    MONTHS_PER_YEAR,
    N_POSITIONS,
    PARTICIPATION_LIMIT,
)
from research.spread_estimation import bounds_from_estimate, bracket_verdict
from research.validation import deflated_sharpe_ratio

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDS",
    "CORRECTED_DELISTING_WINDOW",
    "DATE_CONVENTIONS",
    "GATE_DSR_TARGET",
    "GATE_EXCESS",
    "GATE_TSTAT",
    "REGISTERED_DATE_CONVENTION",
    "REGISTERED_DELISTING_WINDOW",
    "BandBooks",
    "attach_spread_bounds",
    "evaluate_band",
    "run_band",
    "verdict_for",
]

# Registered gate (prereg section 6). Evaluated on the CONSERVATIVE bound.
GATE_EXCESS = 0.02        # vol-matched active return, per year
GATE_TSTAT = 2.0          # Newey-West t on that active return
GATE_DSR_TARGET = 0.95    # the DSR level the Sharpe bar is inverted from
N_TRIALS = 38             # 36 registered + this study + a concurrent seasonality study

BOUNDS = ("conservative", "realistic")

# How `BandBooks.months` is labelled. See "THE ONE-MONTH DATING DEFECT" above.
#   FORMATION   -- the month the signal was ranked on. The slot holds the NEXT month's
#                  return, so the index is one month EARLY and must never be joined to
#                  another series by date. This is the REGISTERED behaviour and the
#                  DEFAULT, so every banked number reproduces bit-for-bit.
#   REALISATION -- the month the return was actually EARNED. Use this for anything that
#                  joins by date: correlation, portfolio construction, regime work.
DATE_CONVENTIONS = (FORMATION, REALISATION)
REGISTERED_DATE_CONVENTION = FORMATION

_IMPACT_COEFFICIENT = {
    "conservative": IMPACT_COEFFICIENT_CONSERVATIVE,
    "realistic": IMPACT_COEFFICIENT_REALISTIC,
}


# --------------------------------------------------------------------------------------
# Cost inputs
# --------------------------------------------------------------------------------------
def attach_spread_bounds(universe: pd.DataFrame) -> pd.DataFrame:
    """Add ``spread_conservative`` / ``spread_realistic`` to every cell.

    Calls `research.spread_estimation.bounds_from_estimate` row by row rather than
    re-deriving its logic vectorised. The universe is 801k rows and the call takes ~7s
    total; a hand-rolled reimplementation would be faster and would be the exact kind of
    silent divergence from the audited cost model that this whole re-test exists because
    of.
    """
    required = {"spread", "spread_regime", "median_dollar_volume", "close", "date"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"universe missing columns {sorted(missing)}")

    conservative = np.empty(len(universe), dtype=float)
    realistic = np.empty(len(universe), dtype=float)
    for i, row in enumerate(universe.itertuples()):
        bounds = bounds_from_estimate(row.spread, row.spread_regime,
                                      row.median_dollar_volume, row.close, row.date)
        conservative[i] = bounds.conservative
        realistic[i] = bounds.realistic

    inverted = int(np.sum(realistic > conservative + 1e-12))
    if inverted:
        raise ValueError(f"{inverted} cells have realistic > conservative; not a bracket")

    out = universe.copy()
    out["spread_conservative"] = conservative
    out["spread_realistic"] = realistic
    return out


def _commission_fraction(trade_value: float, price: float) -> float:
    """IBKR tiered commission + FX, as a fraction of trade value. Bound-independent."""
    if trade_value <= 0 or price <= 0:
        return FX_COST_EACH_WAY
    shares = trade_value / price
    commission = max(COMMISSION_MIN_PER_ORDER, shares * COMMISSION_PER_SHARE)
    commission = min(commission, trade_value * COMMISSION_MAX_FRACTION)
    return commission / trade_value + FX_COST_EACH_WAY


# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
@dataclass
class BandBooks:
    """One book, priced twice. Monthly series are aligned to ``months``."""

    band: str
    deployable_capital: float
    position_value: float
    months: list[pd.Period]
    gross: np.ndarray
    cost_conservative: np.ndarray
    cost_realistic: np.ndarray
    spread_cost: dict[str, np.ndarray]
    impact_cost: dict[str, np.ndarray]
    commission_cost: np.ndarray
    benchmark: np.ndarray
    benchmark_rankable: np.ndarray
    n_rebalances: int
    n_positions_mean: float
    median_cross_section: float
    legs_traded: int
    forced_exit_share: float
    delisting_drag_annual: float
    mean_holding_vol: float
    mean_universe_vol: float
    upper_bound_share_universe: float
    upper_bound_share_held: float
    fallback_volatility_legs: int
    unpriced_exit_legs: int
    charged_unpriced_exit_legs: int
    pnl_by_name_month: list[tuple[str, pd.Period, float]]
    max_gross_weight: float
    top3_gross_weight: float
    notes: list[str] = field(default_factory=list)
    #: what `months` (and the month in `pnl_by_name_month`) MEANS. See DATE_CONVENTIONS.
    date_convention: str = REGISTERED_DATE_CONVENTION


# The REGISTERED delisting window, expressed as day offsets from the exit date. (1, 62)
# reproduces iteration 1's `exit < delisted_on <= exit + 62 days` exactly, because dates
# carry day resolution. It is the default so the registered run is bit-identical.
#
# DEFECT FOUND BY VERIFICATION, NOT BY A TEST (see lowvol_retest_result.md section 6): the
# ACTIONS delisting date is typically the SAME DAY as the ticker's last SEP bar -- median
# gap 0 days -- so the strict `>` excludes almost every real delisting. In the corrected
# universe it fires 39 times against 3,018 available. `CORRECTED_DELISTING_WINDOW` is the
# one-character repair and is used ONLY in the declared sensitivity, never in the headline.
#
# BOTH now come from `research.delisting`, which is the single definition shared with
# `low_vol_quality`, `capacity_study`, `institutional_flow` and `insider_clustering` --
# all of which carried their own copy of the same off-by-one.


def run_band(panel: pd.DataFrame, band: str, delistings: pd.DataFrame,
             delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
             charge_unpriced_exits: bool = False,
             date_convention: str = REGISTERED_DATE_CONVENTION,
             ) -> BandBooks | None:
    """Backtest the registered construction in one band, pricing both cost bounds.

    ``date_convention`` only relabels ``months``; the return arrays are byte-identical
    under both settings, so every within-series statistic is unchanged. See
    DATE_CONVENTIONS and the module docstring.
    """
    if date_convention not in DATE_CONVENTIONS:
        raise ValueError(f"date_convention must be one of {DATE_CONVENTIONS}")
    rows = panel[panel["band_group"] == band].copy()
    if rows.empty:
        return None

    notes: list[str] = []
    delist_date_by_ticker = {row.ticker: row.date for row in delistings.itertuples()}
    delist_value_by_ticker = {row.ticker: float(row.terminal_return)
                              for row in delistings.itertuples()}

    def exit_return(ticker: str, at: pd.Timestamp) -> float:
        delisted_on = delist_date_by_ticker.get(ticker)
        if not in_window(at, delisted_on, delisting_window):
            return 0.0
        return float(delist_value_by_ticker.get(ticker, 0.0))

    # ONE realised-return column read identically by strategy and benchmark, so the
    # comparison cannot be won on accounting asymmetry (erratum 3 of iteration 1).
    forward = rows["forward_return"].clip(-FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP)
    delist_date = pd.to_datetime(rows["ticker"].map(delist_date_by_ticker), errors="coerce")
    delist_value = pd.to_numeric(rows["ticker"].map(delist_value_by_ticker), errors="coerce")
    within = in_window_mask(rows["date"], delist_date, delisting_window)
    rows["terminal_on_exit"] = np.where(within, delist_value.fillna(0.0), 0.0)
    rows["forward_clipped"] = forward
    rows["realised_return"] = forward.where(forward.notna(), rows["terminal_on_exit"])

    deployable = N_POSITIONS * PARTICIPATION_LIMIT * float(
        rows["median_dollar_volume"].median()
    )
    position_value = deployable / N_POSITIONS

    rows["month"] = rows["date"].dt.to_period("M")
    duplicated = int(rows.duplicated(["ticker", "month"]).sum())
    if duplicated:
        raise ValueError(f"{duplicated} (ticker, month) duplicates; the grid is not monthly")
    by_month = {month: frame for month, frame in rows.groupby("month", sort=True)}
    months = sorted(by_month)

    holdings: set[str] = set()
    last_seen: dict[str, pd.Timestamp] = {}
    equity_conservative = 1.0

    gross: list[float] = []
    cost: dict[str, list[float]] = {b: [] for b in BOUNDS}
    spread_cost: dict[str, list[float]] = {b: [] for b in BOUNDS}
    impact_cost: dict[str, list[float]] = {b: [] for b in BOUNDS}
    commission_cost: list[float] = []
    benchmark: list[float] = []
    benchmark_rankable: list[float] = []
    holding_vols: list[float] = []
    universe_vols: list[float] = []
    position_counts: list[int] = []
    cross_sections: list[int] = []
    delisting_contributions: list[float] = []
    held_upper_bound: list[float] = []
    pnl_by_name_month: list[tuple[str, pd.Period, float]] = []

    # Last observed cost inputs per held name, so an exit that happens after the name has
    # left the tradable universe can still be priced (see `charge_unpriced_exits`).
    last_cost: dict[str, tuple[float, float, float | None, dict[str, float]]] = {}
    n_rebalances = 0
    legs_traded_total = 0
    forced_exits = 0
    discretionary_exits = 0
    fallback_volatility_legs = 0
    unpriced_exit_legs = 0
    charged_unpriced_exit_legs = 0
    max_gross_weight = 0.0
    top3_gross_weight = 0.0

    for month in months:
        cross_section = by_month[month]
        rankable = cross_section[cross_section["signal"].notna()]
        cross_sections.append(len(rankable))

        period = {b: {"spread": 0.0, "impact": 0.0} for b in BOUNDS}
        period_commission = 0.0
        traded: set[str] = set()

        if len(rankable) >= MIN_CROSS_SECTION:
            n_rebalances += 1
            target = set(rankable.nlargest(N_POSITIONS, "signal")["ticker"])
            traded = target ^ holdings
            still_rankable = set(rankable["ticker"])
            for ticker in holdings - target:
                if ticker in still_rankable:
                    discretionary_exits += 1
                else:
                    forced_exits += 1
            weight = 1.0 / max(len(target), 1)
            priced = cross_section.set_index("ticker")
            for ticker in traded:
                if ticker not in priced.index:
                    # The name has left the TRADABLE universe -- its price fell through
                    # the $2 floor, its dollar volume left the band, or its spread stopped
                    # resolving. Iteration 1 skipped these legs entirely, which UNDERSTATES
                    # cost: the position still has to be sold. `charge_unpriced_exits`
                    # prices it at the name's LAST OBSERVED cost inputs, which is the
                    # nearest honest estimate available and is certainly too cheap (a name
                    # that just fell out of the universe trades worse, not better).
                    # Counted UNCONDITIONALLY: how many legs would be free must not
                    # depend on whether this run chose to charge them.
                    unpriced_exit_legs += 1
                    if not (charge_unpriced_exits and ticker in last_cost):
                        continue
                    mdv, price, daily_vol, spreads = last_cost[ticker]
                    period_commission += weight * _commission_fraction(position_value,
                                                                       price)
                    for bound in BOUNDS:
                        impact = impact_fraction(position_value, mdv, daily_vol,
                                                 _IMPACT_COEFFICIENT[bound])
                        period[bound]["spread"] += weight * spreads[bound] / 2.0
                        period[bound]["impact"] += weight * (impact
                                                             if np.isfinite(impact) else 0.0)
                    charged_unpriced_exit_legs += 1
                    continue
                row = priced.loc[ticker]
                mdv = float(row["median_dollar_volume"])
                price = float(row["close"])
                daily_vol = float(row["realised_vol"])
                if not np.isfinite(daily_vol) or daily_vol <= 0.0:
                    # Only reachable on an EXIT leg of a name that stopped being rankable.
                    # None routes impact_fraction to REFERENCE_DAILY_VOLATILITY, which is
                    # the documented fallback and the dearer choice for a quiet name.
                    daily_vol = None
                    fallback_volatility_legs += 1
                period_commission += weight * _commission_fraction(position_value, price)
                for bound in BOUNDS:
                    half_spread = float(row[f"spread_{bound}"]) / 2.0
                    impact = impact_fraction(position_value, mdv, daily_vol,
                                             _IMPACT_COEFFICIENT[bound])
                    if not np.isfinite(half_spread):
                        raise ValueError(f"{ticker} {month}: non-finite {bound} spread in "
                                         "a tradable regime")
                    if not np.isfinite(impact):
                        impact = 0.0
                    period[bound]["spread"] += weight * half_spread
                    period[bound]["impact"] += weight * impact
            legs_traded_total += len(traded)
            holdings = target
            if target:
                # Equal weight, so this is 1/n by construction. Measured anyway: an
                # inverse-vol sleeve put 65% of gross notional into 3 names.
                max_gross_weight = max(max_gross_weight, weight)
                top3_gross_weight = max(top3_gross_weight, weight * min(3, len(target)))

        for bound in BOUNDS:
            spread_cost[bound].append(period[bound]["spread"])
            impact_cost[bound].append(period[bound]["impact"])
            cost[bound].append(period[bound]["spread"] + period[bound]["impact"]
                               + period_commission)
        commission_cost.append(period_commission)
        position_counts.append(len(holdings))

        # Registered benchmark: equal-weight buy-and-hold of the band's whole TRADABLE
        # universe. `rankable` is the declared diagnostic (names the strategy could have
        # picked), which is the harder comparison whenever SF1 coverage tilts to better
        # names.
        universe = cross_section["realised_return"].dropna()
        step = float(universe.mean()) if len(universe) else 0.0
        rankable_universe = rankable["realised_return"].dropna()
        rankable_step = float(rankable_universe.mean()) if len(rankable_universe) else step
        benchmark.append(step)
        benchmark_rankable.append(rankable_step)

        universe_vol = cross_section["realised_vol"].dropna()
        if len(universe_vol):
            universe_vols.append(float(universe_vol.mean()))

        if not holdings:
            gross.append(0.0)
            equity_conservative *= 1.0 - cost["conservative"][-1]
            continue

        indexed = cross_section.set_index("ticker")
        realised: list[float] = []
        contributions: list[tuple[str, float]] = []
        closing_out: list[str] = []
        vols: list[float] = []
        upper_bound_flags: list[float] = []
        month_terminal = 0.0
        for ticker in holdings:
            if ticker not in indexed.index:
                # Left the tradable universe. Book the exit ONCE and drop the name --
                # leaving it in `holdings` re-books the terminal return every month, which
                # is how a long-only book once "lost" 112%/yr. Dated at the LAST OBSERVED
                # bar so the 62-day delisting window runs from when the position closed.
                exit_date = last_seen.get(ticker, month.to_timestamp(how="end"))
                value = exit_return(ticker, exit_date)
                realised.append(value)
                contributions.append((ticker, value))
                month_terminal += value
                closing_out.append(ticker)
                continue
            row = indexed.loc[ticker]
            last_seen[ticker] = row["date"]
            held_vol = float(row["realised_vol"])
            last_cost[ticker] = (
                float(row["median_dollar_volume"]),
                float(row["close"]),
                held_vol if np.isfinite(held_vol) and held_vol > 0.0 else None,
                {b: float(row[f"spread_{b}"]) for b in BOUNDS},
            )
            value = float(row["realised_return"])
            realised.append(value)
            contributions.append((ticker, value))
            upper_bound_flags.append(float(row["spread_regime"] == "upper_bound"))
            if pd.isna(row["forward_clipped"]):
                month_terminal += value
                closing_out.append(ticker)
                continue
            vols.append(float(row["realised_vol"]))
        holdings.difference_update(closing_out)

        delisting_contributions.append(month_terminal / max(len(realised), 1))
        if upper_bound_flags:
            held_upper_bound.append(float(np.mean(upper_bound_flags)))

        weight = 1.0 / len(realised)
        period_gross = max(float(np.mean(realised)), -1.0)
        gross.append(period_gross)
        # Dollar P&L attribution on the conservative equity path, for the concentration
        # test. Costs are portfolio-level here, so this attributes GROSS P&L by name.
        for ticker, value in contributions:
            pnl_by_name_month.append((ticker, month, equity_conservative * weight * value))
        equity_conservative *= 1.0 + max(period_gross - cost["conservative"][-1], -1.0)

        if vols:
            holding_vols.append(float(np.mean(vols)))

    if len(gross) < 24:
        return None
    if n_rebalances < 12:
        notes.append("fewer than 12 rebalances; turnover and cost estimates are unstable")

    # THE DATING FIX. `months` was built from the FORMATION date but every slot holds
    # `forward_return`, the FOLLOWING month's return, so the index is one month early.
    # Relabelling here and nowhere else keeps the loop (which keys `by_month` on the
    # formation month) untouched and leaves every return array byte-identical.
    if date_convention == REALISATION:
        months = [month + 1 for month in months]
        pnl_by_name_month = [(ticker, month + 1, value)
                             for ticker, month, value in pnl_by_name_month]

    years = len(gross) / MONTHS_PER_YEAR
    gross_array = np.asarray(gross, dtype=float)
    cost_arrays = {b: np.asarray(cost[b], dtype=float) for b in BOUNDS}
    net_floor = {b: np.maximum(gross_array - cost_arrays[b], -1.0) for b in BOUNDS}

    return BandBooks(
        band=band,
        deployable_capital=deployable,
        position_value=position_value,
        months=months,
        gross=gross_array,
        cost_conservative=gross_array - net_floor["conservative"],
        cost_realistic=gross_array - net_floor["realistic"],
        spread_cost={b: np.asarray(spread_cost[b], dtype=float) for b in BOUNDS},
        impact_cost={b: np.asarray(impact_cost[b], dtype=float) for b in BOUNDS},
        commission_cost=np.asarray(commission_cost, dtype=float),
        benchmark=np.asarray(benchmark, dtype=float),
        benchmark_rankable=np.asarray(benchmark_rankable, dtype=float),
        n_rebalances=n_rebalances,
        n_positions_mean=float(np.mean(position_counts)) if position_counts else 0.0,
        median_cross_section=float(np.median(cross_sections)) if cross_sections else 0.0,
        legs_traded=legs_traded_total,
        forced_exit_share=(forced_exits / (forced_exits + discretionary_exits)
                           if (forced_exits + discretionary_exits) else float("nan")),
        delisting_drag_annual=float(np.sum(delisting_contributions)) / years,
        mean_holding_vol=(float(np.mean(holding_vols)) * np.sqrt(252.0)
                          if holding_vols else float("nan")),
        mean_universe_vol=(float(np.mean(universe_vols)) * np.sqrt(252.0)
                           if universe_vols else float("nan")),
        upper_bound_share_universe=float(
            (rows["spread_regime"] == "upper_bound").mean()
        ),
        upper_bound_share_held=(float(np.mean(held_upper_bound))
                                if held_upper_bound else float("nan")),
        fallback_volatility_legs=fallback_volatility_legs,
        unpriced_exit_legs=unpriced_exit_legs,
        charged_unpriced_exit_legs=charged_unpriced_exit_legs,
        pnl_by_name_month=pnl_by_name_month,
        max_gross_weight=max_gross_weight,
        top3_gross_weight=top3_gross_weight,
        notes=notes,
        date_convention=date_convention,
    )


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def _stream_stats(returns: np.ndarray, n_trials: int = N_TRIALS) -> dict[str, float]:
    series = pd.Series(returns, dtype=float)
    mean, _se, tstat = newey_west_tstat(series)
    volatility = float(series.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
    annual = float(series.mean()) * MONTHS_PER_YEAR
    equity = float((1.0 + series).prod())
    years = len(series) / MONTHS_PER_YEAR
    curve = (1.0 + series).cumprod().to_numpy()
    peak = np.maximum.accumulate(curve)
    return {
        "annual_arithmetic": annual,
        "cagr": float(equity ** (1.0 / years) - 1.0) if equity > 0 else float("nan"),
        "volatility": volatility,
        "sharpe": annual / volatility if volatility > 0 else float("nan"),
        "tstat": tstat,
        "max_drawdown": float(np.max(1.0 - curve / peak)),
        "dsr": float(deflated_sharpe_ratio(series.to_numpy(), n_trials=n_trials)),
    }


def _decade_sharpes(months: list[pd.Period], returns: np.ndarray) -> dict[str, dict]:
    frame = pd.DataFrame({"year": [m.year for m in months], "r": returns})
    frame["decade"] = (frame["year"] // 10) * 10
    out: dict[str, dict] = {}
    for decade, block in frame.groupby("decade"):
        values = block["r"].to_numpy(dtype=float)
        if values.size < 12:
            out[f"{decade}s"] = {"n_months": int(values.size), "sharpe": float("nan"),
                                 "annual": float(values.mean() * MONTHS_PER_YEAR)}
            continue
        volatility = float(values.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
        annual = float(values.mean()) * MONTHS_PER_YEAR
        out[f"{decade}s"] = {
            "n_months": int(values.size),
            "annual": annual,
            "sharpe": annual / volatility if volatility > 0 else float("nan"),
        }
    return out


def _concentration(books: BandBooks) -> dict[str, float]:
    """P&L concentration and gross-notional concentration."""
    if not books.pnl_by_name_month:
        return {}
    values = np.array([value for _t, _m, value in books.pnl_by_name_month], dtype=float)
    total = float(values.sum())
    absolute = float(np.abs(values).sum())
    order = np.argsort(-np.abs(values))
    top10 = float(values[order[:10]].sum())
    return {
        "n_name_months": int(values.size),
        "largest_abs_share_of_gross_pnl": (float(np.abs(values).max()) / absolute
                                           if absolute > 0 else float("nan")),
        "largest_share_of_net_pnl": (float(values[order[0]]) / total
                                     if total != 0 else float("nan")),
        "top10_share_of_net_pnl": top10 / total if total != 0 else float("nan"),
        "max_gross_notional_weight": books.max_gross_weight,
        "top3_gross_notional_weight": books.top3_gross_weight,
    }


def evaluate_band(books: BandBooks, n_trials: int = N_TRIALS) -> dict:
    """Every registered statistic for one band, both bounds."""
    years = len(books.gross) / MONTHS_PER_YEAR
    benchmark_stats = _stream_stats(books.benchmark, n_trials)
    rankable_stats = _stream_stats(books.benchmark_rankable, n_trials)
    gross_stats = _stream_stats(books.gross, n_trials)
    bar = dsr_sharpe_bar(years, n_trials=n_trials, target=GATE_DSR_TARGET)

    out: dict = {
        "band": books.band,
        "date_convention": books.date_convention,
        "n_months": len(books.gross),
        "years": years,
        "n_rebalances": books.n_rebalances,
        "n_positions_mean": books.n_positions_mean,
        "median_cross_section": books.median_cross_section,
        "deployable_capital": books.deployable_capital,
        "position_value": books.position_value,
        "turnover_annual": books.legs_traded / max(N_POSITIONS, 1) / years,
        "forced_exit_share": books.forced_exit_share,
        "delisting_drag_annual": books.delisting_drag_annual,
        "mean_holding_vol": books.mean_holding_vol,
        "mean_universe_vol": books.mean_universe_vol,
        "upper_bound_share_universe": books.upper_bound_share_universe,
        "upper_bound_share_held": books.upper_bound_share_held,
        "fallback_volatility_legs": books.fallback_volatility_legs,
        "unpriced_exit_legs": books.unpriced_exit_legs,
        "charged_unpriced_exit_legs": books.charged_unpriced_exit_legs,
        "dsr_sharpe_bar": bar,
        "n_trials": n_trials,
        "gross": gross_stats,
        "benchmark": benchmark_stats,
        "benchmark_rankable": rankable_stats,
        "benchmark_decades": _decade_sharpes(books.months, books.benchmark),
        "concentration": _concentration(books),
        "notes": list(books.notes),
        "bounds": {},
    }

    passes: dict[str, bool] = {}
    for bound in BOUNDS:
        costs = books.cost_conservative if bound == "conservative" else books.cost_realistic
        net = np.maximum(books.gross - costs, -1.0)
        stats = _stream_stats(net, n_trials)
        matched = vol_matched_active(pd.Series(net), pd.Series(books.benchmark))
        matched_rankable = vol_matched_active(pd.Series(net),
                                              pd.Series(books.benchmark_rankable))
        gate = (
            matched.get("vol_matched_active_annual", float("nan")) > GATE_EXCESS
            and matched.get("vol_matched_active_tstat", float("nan")) > GATE_TSTAT
        )
        passes[bound] = bool(gate)
        out["bounds"][bound] = {
            "net": stats,
            "cost_annual_total": float(costs.sum()) / years,
            "cost_annual_spread": float(books.spread_cost[bound].sum()) / years,
            "cost_annual_impact": float(books.impact_cost[bound].sum()) / years,
            "cost_annual_commission": float(books.commission_cost.sum()) / years,
            "cost_one_way_bps": (float(costs.sum()) / max(books.legs_traded, 1)
                                 * N_POSITIONS * 1e4),
            "excess_arithmetic": stats["annual_arithmetic"]
                                 - benchmark_stats["annual_arithmetic"],
            "excess_geometric": stats["cagr"] - benchmark_stats["cagr"],
            "excess_vs_rankable_arithmetic": stats["annual_arithmetic"]
                                             - rankable_stats["annual_arithmetic"],
            "vol_matched": matched,
            "vol_matched_vs_rankable": matched_rankable,
            "decades": _decade_sharpes(books.months, net),
            "gate_excess_pass": bool(
                matched.get("vol_matched_active_annual", float("nan")) > GATE_EXCESS
            ),
            "gate_tstat_pass": bool(
                matched.get("vol_matched_active_tstat", float("nan")) > GATE_TSTAT
            ),
            "gate_dsr_bar_pass": bool(stats["sharpe"] >= bar),
            "gate_beats_benchmark_dsr": bool(stats["dsr"] > benchmark_stats["dsr"]),
        }

    try:
        out["bracket_verdict"] = bracket_verdict(passes["conservative"],
                                                 passes["realistic"])
    except ValueError:
        # `bracket_verdict` refuses a conservative-pass / realistic-fail pair because the
        # two bounds cannot invert on COST. They can still invert on the t-statistic,
        # because the two books have very slightly different volatilities. Recorded
        # rather than crashed, and flagged so it is never read as a clean pass.
        out["bracket_verdict"] = "inverted"
        out["notes"].append(
            "gate outcome inverted across the bounds (t-statistic, not cost); the "
            "conservative bound is the one that counts"
        )
    return out


def verdict_for(evaluated: dict) -> str:
    """The pre-committed decision rule of `lowvol_retest_prereg.md` section 6."""
    conservative = evaluated["bounds"]["conservative"]
    realistic = evaluated["bounds"]["realistic"]
    if (conservative["gate_excess_pass"] and conservative["gate_tstat_pass"]
            and conservative["gate_dsr_bar_pass"]
            and conservative["gate_beats_benchmark_dsr"]):
        return "PROMOTE"
    if conservative["gate_excess_pass"] and conservative["gate_tstat_pass"]:
        return "MARGINAL"
    if realistic["vol_matched"].get("vol_matched_active_annual", float("nan")) <= 0.0:
        return "DEAD"
    if conservative["gate_excess_pass"] or realistic["gate_excess_pass"]:
        return "UNDETERMINED"
    return "DEAD"


def overall_verdict(evaluations: list[dict]) -> str:
    """Best verdict across the capacity curve, in the registered ordering."""
    ranking = ["PROMOTE", "MARGINAL", "UNDETERMINED", "DEAD"]
    reached = [verdict_for(e) for e in evaluations]
    for candidate in ranking:
        if candidate in reached:
            return candidate
    return "DEAD"


BAND_ORDER = tuple(BAND_GROUPS)
