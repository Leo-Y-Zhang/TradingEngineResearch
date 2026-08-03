"""The capacity-curve study: does the fundamental edge survive where big money cannot go?

Registered design: `research/medallion_style_alpha_search/capacity_curve_prereg.md`
(+ errata 1 and 2). This module is the measurement; `scripts/run_capacity_study.py` runs
it and prints the verdict.

The primary statistic is a SINGLE number: the Spearman rank correlation between a band's
deployable capital and its net Sharpe. Testing the ladder as one ordered hypothesis --
rather than six separate band tests -- is what keeps the whole study at one trial instead
of six, which matters because deflation is already charging 23 prior trials against it.

The signal is a FIXED equal-weight composite of the 14 registered fundamental factors,
sign-aligned by documented economic direction. Nothing is fitted. That is deliberate:
a learned combiner adds researcher degrees of freedom and, in the prior programme, the
learned ridge UNDERPERFORMED the naive composite in every tradable universe
(`sharadar_dev_log.md` entry 3). A fixed composite is both the cheaper and the better-
supported choice, and it makes the band comparison clean because every band sees exactly
the same signal definition.

**The cost model is the thing most likely to decide this study's answer, so it is
bracketed rather than point-estimated and every component is validated against ground
truth before it is trusted.** Spreads come from `research.spread_estimation`
(`spread_cost_bounds`, controlled by `scripts/spread_positive_control.py`); market impact
is the square-root law calibrated on live institutional executions and controlled by
`scripts/impact_positive_control.py`. See the block above `FIM_ANCHOR_PARTICIPATION` for
the impact calibration and for the 100bps-a-side defect it replaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from research.delisting import CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW
from research.delisting import in_window

logger = logging.getLogger(__name__)

__all__ = [
    "BandResult",
    "CORRECTED_DELISTING_WINDOW",
    "FACTOR_SIGNS",
    "FIM_ANCHOR_DAILY_VOLATILITY",
    "FIM_ANCHOR_DOLLAR_VOLUME",
    "FIM_ANCHOR_HALF_SPREAD_BPS",
    "FIM_ANCHOR_PARTICIPATION",
    "FIM_LARGE_CAP_MEAN_BPS",
    "FIM_LARGE_CAP_MEDIAN_BPS",
    "FIM_NYSE_AMEX_MEDIAN_BPS",
    "FIM_SMALL_CAP_MEDIAN_BPS",
    "IMPACT_COEFFICIENT",
    "IMPACT_COEFFICIENT_CONSERVATIVE",
    "IMPACT_COEFFICIENT_REALISTIC",
    "IMPACT_EXPONENT",
    "ImpactBounds",
    "REFERENCE_DAILY_VOLATILITY",
    "REGISTERED_DELISTING_WINDOW",
    "capacity_statistic",
    "composite_signal",
    "deployable_capital",
    "impact_cost_bounds",
    "impact_fraction",
    "round_trip_cost",
    "run_band",
]

# Documented economic direction of each registered factor. +1 means a higher value is
# expected to predict a HIGHER return. These are the standard signs from the asset-
# pricing literature, fixed before any result was computed; they are not fitted.
FACTOR_SIGNS: dict[str, int] = {
    "earnings_yield": +1,
    "book_to_price": +1,
    "sales_to_price": +1,
    "roe": +1,
    "roa": +1,
    "gross_profitability": +1,
    "operating_margin": +1,
    "revenue_growth": +1,
    "earnings_growth": +1,
    "momentum_12_1": +1,
    "asset_growth": -1,
    "net_share_issuance": -1,
    "accruals": -1,
    "debt_to_equity": -1,
}

N_POSITIONS = 30
REBALANCE_MONTHS = 3
ENTRY_QUANTILE = 0.90
EXIT_QUANTILE = 0.70
PARTICIPATION_LIMIT = 0.01

# Interactive Brokers tiered US equities, the realistic retail schedule.
COMMISSION_PER_SHARE = 0.0035
COMMISSION_MIN_PER_ORDER = 0.35
COMMISSION_MAX_FRACTION = 0.01
FX_COST_EACH_WAY = 0.00002

# ---------------------------------------------------------------------------
# Market impact
# ---------------------------------------------------------------------------
#
# **THE DEFECT THIS REPLACES (2026-07-28).** Until now this module carried
# ``IMPACT_COEFFICIENT = 0.1`` with ``impact = 0.1 * sqrt(participation)``. At the
# registered 1%-of-daily-volume position cap that is ``0.1 * sqrt(0.01)`` = **100bps per
# side, 200bps round trip, from market impact alone** -- before spread, before commission,
# and identical for a placid mega-cap and a wild micro-cap because the formula carries no
# volatility term at all. Iteration 1 measured total round-trip costs of 117-236bps across
# six sleeves (the internal research log). Impact was not a component of that
# bill; it very nearly WAS the bill.
#
# **THE FUNCTIONAL FORM.** The square-root law is kept, because it is the best-supported
# shape in the literature, but written the conventional way -- with the volatility term the
# old version was missing:
#
#     impact_one_way = Y * sigma_daily * sqrt(Q / V)
#
# Tóth, Lempérière, Deremble, de Lataillade, Kockelkoren & Bouchaud (2011), "Anomalous
# Price Impact and the Critical Nature of Liquidity in Financial Markets", *Physical
# Review X* 1, 021006, eq. (1): "the price change ... is well described by the so-called
# 'square-root' law: Delta(Q) = Y sigma sqrt(Q/V), where sigma is the daily volatility of
# the asset, and V the daily traded volume, both quantities measured contemporaneously to
# the trade. The numerical constant Y is of order unity." The exponent is supported across
# independent datasets: delta ~= 0.6 on 700,000 Citigroup US equity orders (Almgren, Thum,
# Hauptmann & Li 2005), ~0.5 on Madrid and ~0.7 on London (Moro et al.), and ~0.5/0.6 on
# ~500,000 CFM futures metaorders. Frazzini, Israel & Moskowitz (2018) likewise approximate
# price impact with a square root in trade size. 0.5 is the central value and the one used.
#
# **THE COEFFICIENT IS SOURCED, NOT GUESSED, AND IT IS BRACKETED.** The anchor is Frazzini,
# Israel & Moskowitz (2018), "Trading Costs", Table II Panel A: $1.7tn of live US
# institutional executions, Aug 1998 - Jun 2016, average trade 0.9% of daily volume,
# US large-cap MEDIAN all-in one-way cost 5.54bps (mean 8.90), NYSE-Amex median 5.06,
# small cap median 13.53.
#
# **The honest difficulty, and how it is handled.** FIM measure ALL-IN cost -- spread plus
# impact plus delay -- so the impact COMPONENT must be strictly smaller than 5.54bps, and
# the published table gives no decomposition. Calibrating impact to the whole 5.54 would
# double-charge, because this module charges spread and commission separately on top. The
# decomposition is also unstable: subtracting this repo's own liquid-name spread schedule
# (4.50bps per side for a name of the anchor's liquidity) leaves a residual of ~1bps, and a
# 10% error in the spread schedule moves that residual by 43%. There is no defensible point
# estimate. So, exactly as `research.spread_estimation.spread_cost_bounds` does for spreads,
# the answer is two bounds and never one:
#
#   (a) CONSERVATIVE -- attribute the ENTIRE FIM MEAN all-in cost (8.90bps) to impact.
#       Impact is a strict subset of all-in cost, so 100% of it is the largest share
#       arithmetically available, and the mean is the dearer of the two published
#       statistics. Spread and commission are then charged AGAIN on top, so this
#       deliberately double-counts. **A result that passes under (a) is REAL.**
#   (b) REALISTIC -- attribute the FIM MEDIAN all-in cost (5.54bps) LESS the half-spread
#       this repo's own documented schedule charges a name of the anchor's liquidity
#       (4.50bps per side). This credits the spread model with everything it claims and
#       gives impact only the residual, which is the smallest defensible share.
#       **A result that fails under (b) is DEAD.** In between it is UNDETERMINED
#       (`research.spread_estimation.bracket_verdict`).
#
# `realistic <= conservative` holds by construction, so the pair can never invert.
#
# **THE CONTROL CAME FIRST** (`scripts/impact_positive_control.py`). The coefficients above
# are fixed on the LARGE-CAP anchor alone and then tested, without refitting, against FIM's
# SMALL-CAP number, which the bracket must contain. A cost model that cannot reproduce a
# known execution cost cannot be trusted on an unknown one -- and skipping that check is how
# the spread model went wrong twice.

# --- the published anchor (FIM 2018 Table II Panel A), in basis points, one way ---------
FIM_ANCHOR_PARTICIPATION = 0.009
FIM_LARGE_CAP_MEDIAN_BPS = 5.54
FIM_LARGE_CAP_MEAN_BPS = 8.90
FIM_NYSE_AMEX_MEDIAN_BPS = 5.06
FIM_SMALL_CAP_MEDIAN_BPS = 13.53

# --- what that population looks like on OUR tape, MEASURED not assumed -----------------
# Median daily log-return volatility and median dollar volume of the 591 Sharadar SEP names
# with at least 500 bars and median dollar volume >= $50M/day over 1998-01-01..2015-12-31
# (28,992,477 bars, 11,198 names with >= 500 bars). $50M/day is the mapping chosen for
# FIM's "US large cap": it is the liquidity at which a $1.7tn manager's large-cap book
# actually sits. Reproduced and re-checked by `scripts/impact_positive_control.py`.
FIM_ANCHOR_DOLLAR_VOLUME = 9.6283e7
FIM_ANCHOR_DAILY_VOLATILITY = 0.0262

# Half of `research.spread_estimation.liquid_name_spread(FIM_ANCHOR_DOLLAR_VOLUME)`, the
# repo's own documented liquid-name schedule (Ardia-Guidotti-Kroencke Table 4 Panel C top
# quintile, 9bps full effective spread, era factor 1.0). Hard-coded rather than imported so
# this calibration cannot drift silently when the spread schedule is edited; the positive
# control reads the live schedule and fails if the two disagree by more than 1bps.
FIM_ANCHOR_HALF_SPREAD_BPS = 4.50

# Fallback daily volatility for callers that cannot supply one. Median daily volatility of
# the 6,754 names with median dollar volume >= $200k/day -- the capacity study's own
# eligibility floor -- over the same window. Using a fallback is always worse than using
# the name's own volatility and callers should pass `daily_volatility`; this exists so that
# a legacy call site degrades to a documented central value rather than to a number with no
# volatility in it at all.
REFERENCE_DAILY_VOLATILITY = 0.0335

IMPACT_EXPONENT = 0.5

_ANCHOR_IMPACT_UNIT = FIM_ANCHOR_DAILY_VOLATILITY * (
    FIM_ANCHOR_PARTICIPATION ** IMPACT_EXPONENT
)

# (a) the whole measured all-in MEAN charged as impact, then spread charged again on top.
IMPACT_COEFFICIENT_CONSERVATIVE = (FIM_LARGE_CAP_MEAN_BPS / 1e4) / _ANCHOR_IMPACT_UNIT

# (b) the measured all-in MEDIAN less the spread this repo already charges the same name.
IMPACT_COEFFICIENT_REALISTIC = (
    (FIM_LARGE_CAP_MEDIAN_BPS - FIM_ANCHOR_HALF_SPREAD_BPS) / 1e4
) / _ANCHOR_IMPACT_UNIT

# DEPRECATED, and kept only so that call sites written against the old flat form
# (`IMPACT_COEFFICIENT * sqrt(participation)`, with no volatility term) do not silently
# keep charging 100bps a side. It is the CONSERVATIVE coefficient evaluated at the
# reference volatility, i.e. exactly what the flat form means once sigma is held fixed;
# the conservative branch is chosen because a caller that cannot express a bracket should
# be left on the overstating side. New code must call `impact_cost_bounds` or pass
# `daily_volatility` to `impact_fraction`, both of which use the name's OWN volatility.
IMPACT_COEFFICIENT = IMPACT_COEFFICIENT_CONSERVATIVE * REFERENCE_DAILY_VOLATILITY


@dataclass(frozen=True)
class ImpactBounds:
    """The two bracketing market-impact charges for one trade, one way.

    Deliberately shaped like `research.spread_estimation.SpreadBounds`: two bounds, a
    `determined` property, and the invariant ``conservative >= realistic`` holding by
    construction so a caller can always read the pair as a genuine bracket.
    """

    conservative: float
    realistic: float
    participation: float
    daily_volatility: float

    @property
    def determined(self) -> bool:
        """True when the two bounds agree to within a basis point."""
        if not (np.isfinite(self.conservative) and np.isfinite(self.realistic)):
            return False
        return abs(self.conservative - self.realistic) < 1e-4


@dataclass
class BandResult:
    band: str
    deployable_capital: float
    n_rebalances: int
    n_positions_mean: float
    gross_return_annual: float
    net_return_annual: float
    net_volatility: float
    net_sharpe: float
    max_drawdown: float
    benchmark_return_annual: float
    benchmark_sharpe: float
    turnover_annual: float
    cost_drag_annual: float
    measured_share: float
    excluded_upper_bound: int
    notes: list[str] = field(default_factory=list)
    # The same book priced under the REALISTIC impact bound. Reported only -- the equity
    # path, and therefore every Sharpe above, is built on the CONSERVATIVE bound, so a
    # band that clears its gate here clears it on the expensive side of the bracket.
    cost_drag_annual_realistic: float = 0.0


def composite_signal(features: pd.DataFrame) -> pd.Series:
    """Equal-weight, sign-aligned composite of whichever factors are present.

    Averaging over available factors rather than requiring all 14 keeps names with
    partial fundamental coverage in the cross-section. Small caps are exactly where
    coverage is patchiest, so demanding a complete row would quietly bias the universe
    toward larger, better-covered names -- the opposite of what this study is measuring.
    """
    present = [name for name in FACTOR_SIGNS if name in features.columns]
    if not present:
        raise ValueError("no registered factors present in the feature frame")

    aligned = pd.DataFrame(
        {name: features[name] * FACTOR_SIGNS[name] for name in present},
        index=features.index,
    )
    coverage = aligned.notna().sum(axis=1)
    signal = aligned.mean(axis=1, skipna=True)
    # Require at least a third of the factor set, so a name is never ranked on one
    # lucky number.
    return signal.where(coverage >= max(len(present) // 3, 3))


def _commission_fraction(trade_value: float, price: float) -> float:
    """Commission as a fraction of trade value, including the per-order minimum.

    The $0.35 floor is the reason small accounts cannot trade like large ones: on a
    £300 position it is over 9bps before any spread is paid, and it is charged again on
    exit.
    """
    if trade_value <= 0 or price <= 0:
        return 0.0
    shares = trade_value / price
    commission = max(COMMISSION_MIN_PER_ORDER, shares * COMMISSION_PER_SHARE)
    commission = min(commission, trade_value * COMMISSION_MAX_FRACTION)
    return commission / trade_value


def _participation(trade_value: float, median_dollar_volume: float) -> float:
    """Trade size as a fraction of the name's own median daily dollar volume."""
    if not np.isfinite(median_dollar_volume) or median_dollar_volume <= 0:
        return float("nan")
    if not np.isfinite(trade_value):
        return float("nan")
    return max(float(trade_value), 0.0) / float(median_dollar_volume)


def impact_fraction(
    trade_value: float,
    median_dollar_volume: float,
    daily_volatility: float | None = None,
    coefficient: float = IMPACT_COEFFICIENT_CONSERVATIVE,
) -> float:
    """Square-root market impact, one way, as a fraction of trade value.

    ``coefficient * daily_volatility * sqrt(trade_value / median_dollar_volume)``.

    Args:
        trade_value: Notional traded, in the same currency as ``median_dollar_volume``.
        median_dollar_volume: Trailing median daily dollar volume of the name.
        daily_volatility: The name's OWN daily return volatility. Pass it. ``None`` falls
            back to `REFERENCE_DAILY_VOLATILITY`, which prices every name identically and
            is exactly the blindness this recalibration exists to remove.
        coefficient: `IMPACT_COEFFICIENT_CONSERVATIVE` (default) or
            `IMPACT_COEFFICIENT_REALISTIC`. Use `impact_cost_bounds` to get both.

    Returns:
        The one-way impact fraction, or NaN when dollar volume is missing or
        non-positive -- which means untradeable, not free.
    """
    participation = _participation(trade_value, median_dollar_volume)
    if not np.isfinite(participation):
        return float("nan")
    volatility = (REFERENCE_DAILY_VOLATILITY if daily_volatility is None
                  else float(daily_volatility))
    if not np.isfinite(volatility) or volatility < 0.0:
        return float("nan")
    return float(coefficient) * volatility * (participation ** IMPACT_EXPONENT)


def impact_cost_bounds(
    trade_value: float,
    median_dollar_volume: float,
    daily_volatility: float | None = None,
) -> ImpactBounds:
    """Price one trade's market impact under BOTH bounds. This is the API to use.

    Mirrors `research.spread_estimation.spread_cost_bounds`. Report both numbers for every
    result: a result is REAL only if it survives ``conservative``; it is DEAD if it fails
    ``realistic``; anything in between is UNDETERMINED
    (`research.spread_estimation.bracket_verdict`).
    """
    participation = _participation(trade_value, median_dollar_volume)
    volatility = (REFERENCE_DAILY_VOLATILITY if daily_volatility is None
                  else float(daily_volatility))
    return ImpactBounds(
        conservative=impact_fraction(trade_value, median_dollar_volume,
                                     daily_volatility,
                                     IMPACT_COEFFICIENT_CONSERVATIVE),
        realistic=impact_fraction(trade_value, median_dollar_volume,
                                  daily_volatility,
                                  IMPACT_COEFFICIENT_REALISTIC),
        participation=participation,
        daily_volatility=volatility,
    )


def _impact_fraction(
    trade_value: float,
    median_dollar_volume: float,
    daily_volatility: float | None = None,
) -> float:
    """Back-compatible alias for `impact_fraction`; new code should call that.

    Kept because `research/sleeves/institutional_flow.py` imports this private name. It
    now routes to the calibrated model on the CONSERVATIVE branch, so a legacy caller gets
    a sourced number instead of the flat 0.1 coefficient, and gets it on the overstating
    side.
    """
    return impact_fraction(trade_value, median_dollar_volume, daily_volatility,
                           IMPACT_COEFFICIENT_CONSERVATIVE)


def round_trip_cost(
    spread: float,
    trade_value: float,
    price: float,
    median_dollar_volume: float,
    daily_volatility: float | None = None,
    impact_coefficient: float = IMPACT_COEFFICIENT_CONSERVATIVE,
) -> float:
    """One-way cost fraction: half-spread + impact + commission + FX.

    The name is historic and misleading -- this is ONE side of a round trip. Left alone
    because several sleeves import it; call it twice for a round trip.

    Pass ``impact_coefficient=IMPACT_COEFFICIENT_REALISTIC`` for the other bound.
    """
    return (
        spread / 2.0
        + impact_fraction(trade_value, median_dollar_volume, daily_volatility,
                          impact_coefficient)
        + _commission_fraction(trade_value, price)
        + FX_COST_EACH_WAY
    )


def deployable_capital(median_dollar_volume: float,
                       n_positions: int = N_POSITIONS) -> float:
    """Capital the band supports at the registered 1%-of-median-volume position cap."""
    return n_positions * PARTICIPATION_LIMIT * median_dollar_volume


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak)) if len(equity) else 0.0


def run_band(
    panel: pd.DataFrame,
    band: str,
    delistings: pd.DataFrame | None = None,
    delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
) -> BandResult | None:
    """Backtest the registered long-only construction inside one liquidity band.

    Quarterly rebalance with a no-trade band: a held name is sold only when it drops out
    of the top 30% of the ranking, not merely because it left the top decile. Turnover,
    not signal strength, is what killed the prior programme's construction.

    `delisting_window` defaults to the REGISTERED window, which reproduces every banked
    number bit-for-bit. Its lower edge is STRICT and therefore rejects the modal
    delisting, which Sharadar dates on the ticker's last traded bar -- see
    `research.delisting`. `CORRECTED_DELISTING_WINDOW` is the repair.
    """
    rows = panel[(panel["band"] == band) & panel["signal"].notna()].copy()
    if rows.empty:
        return None

    notes: list[str] = []
    total_eligible = len(rows)

    # Only names whose spread is genuinely MEASURED can be honestly costed. Names at the
    # resolution floor are excluded rather than costed at that floor, which would
    # manufacture cost from an absence of information (erratum 2).
    measured = rows[rows["spread_regime"] == "measured"]
    excluded = total_eligible - len(measured)
    measured_share = len(measured) / total_eligible if total_eligible else 0.0
    if measured.empty:
        return None
    rows = measured

    # Map ticker -> (delisting date, terminal return). The DATE is essential: a name
    # can leave the measurable universe years before it actually delists (its band
    # changes, or its spread stops being resolvable). Applying the terminal return on
    # mere absence would book a 2012 bankruptcy against a 2003 exit, and would do it
    # for thousands of names -- which is exactly how a long-only book in a universe
    # returning +12%/yr ends up "losing" 60%/yr.
    terminal: dict[str, tuple[pd.Timestamp, float]] = {}
    if delistings is not None and not delistings.empty:
        terminal = {
            row.ticker: (row.date, row.terminal_return)
            for row in delistings.itertuples()
        }

    def exit_return(ticker: str, at: pd.Timestamp) -> float:
        """Terminal return if the name delists imminently, else a flat exit."""
        entry = terminal.get(ticker)
        if entry is None:
            return 0.0
        delisted_on, value = entry
        # The position is closed in the month following `at`, so only a delisting
        # inside that window is the reason it closed.
        if in_window(at, delisted_on, delisting_window):
            return float(value)
        return 0.0

    dates = sorted(rows["date"].unique())
    holdings: dict[str, float] = {}
    equity = [1.0]
    benchmark_equity = [1.0]
    turnovers: list[float] = []
    costs: list[float] = []
    costs_realistic: list[float] = []
    position_counts: list[int] = []
    gross_returns: list[float] = []
    # The panel does not carry a volatility column yet (`research/capacity_panel.py`
    # builds it without one). Use it when a caller supplies it, and fall back to the
    # documented reference volatility otherwise rather than silently pricing every name
    # as though volatility did not exist.
    has_volatility = "daily_volatility" in rows.columns

    for index, date in enumerate(dates):
        cross_section = rows[rows["date"] == date]
        if len(cross_section) < 10:
            continue

        rebalancing = index % REBALANCE_MONTHS == 0
        if rebalancing:
            ranks = cross_section["signal"].rank(pct=True)
            entry = set(cross_section.loc[ranks >= ENTRY_QUANTILE, "ticker"])
            hold_ok = set(cross_section.loc[ranks >= EXIT_QUANTILE, "ticker"])

            kept = {t for t in holdings if t in hold_ok}
            room = N_POSITIONS - len(kept)
            additions = [t for t in
                         cross_section.loc[ranks >= ENTRY_QUANTILE]
                         .sort_values("signal", ascending=False)["ticker"]
                         if t not in kept][:max(room, 0)]
            new_holdings = list(kept) + list(additions)
            if not new_holdings:
                new_holdings = list(entry)[:N_POSITIONS]

            traded = (set(new_holdings) ^ set(holdings)) & set(cross_section["ticker"])
            turnover = len(traded) / max(len(new_holdings), 1)
            turnovers.append(turnover)

            weight = 1.0 / max(len(new_holdings), 1)
            capital = deployable_capital(
                float(cross_section["median_dollar_volume"].median())
            )
            period_cost = 0.0
            period_cost_realistic = 0.0
            for ticker in traded:
                row = cross_section[cross_section["ticker"] == ticker]
                if row.empty:
                    continue
                row = row.iloc[0]
                trade_value = capital * weight
                volatility = (float(row["daily_volatility"]) if has_volatility
                              else None)
                period_cost += weight * round_trip_cost(
                    float(row["spread"]), trade_value, float(row["close"]),
                    float(row["median_dollar_volume"]), volatility,
                    IMPACT_COEFFICIENT_CONSERVATIVE,
                )
                period_cost_realistic += weight * round_trip_cost(
                    float(row["spread"]), trade_value, float(row["close"]),
                    float(row["median_dollar_volume"]), volatility,
                    IMPACT_COEFFICIENT_REALISTIC,
                )
            costs.append(period_cost)
            costs_realistic.append(period_cost_realistic)
            holdings = {t: weight for t in new_holdings}
        else:
            costs.append(0.0)
            costs_realistic.append(0.0)

        position_counts.append(len(holdings))

        returns = []
        closing_out: list[str] = []
        for ticker, weight in holdings.items():
            row = cross_section[cross_section["ticker"] == ticker]
            if row.empty:
                # The name left the measurable universe. Book its exit ONCE and drop
                # it. Leaving it in `holdings` would re-book the same terminal return
                # every subsequent month -- which is a -100% per month loop for any
                # delisted name, and would sink every band regardless of the signal.
                returns.append((weight, exit_return(ticker, date)))
                closing_out.append(ticker)
                continue

            forward = row.iloc[0]["forward_return"]
            if np.isfinite(forward):
                returns.append((weight, float(forward)))
                continue

            # Last observation for this name: there is no next bar to return into.
            # If it delisted, the registered terminal return applies here; otherwise
            # it is exited at the last traded price.
            returns.append((weight, exit_return(ticker, date)))
            closing_out.append(ticker)

        for ticker in closing_out:
            holdings.pop(ticker, None)

        gross = sum(w * r for w, r in returns)
        # Weights are renormalised implicitly by the equal-weight construction, but a
        # long-only book cannot lose more than everything.
        gross = max(gross, -1.0)
        gross_returns.append(gross)
        net = max(gross - costs[-1], -1.0)
        equity.append(equity[-1] * (1.0 + net))

        # An all-NaN forward column (the final rebalance date, which has no next bar)
        # would turn the whole benchmark series into NaN, so carry it flat instead.
        band_forward = cross_section["forward_return"].dropna()
        benchmark_step = float(band_forward.mean()) if len(band_forward) else 0.0
        benchmark_equity.append(benchmark_equity[-1] * (1.0 + benchmark_step))

    if len(equity) < 24:
        return None

    net_series = np.diff(equity) / equity[:-1]
    benchmark_series = np.diff(benchmark_equity) / benchmark_equity[:-1]
    periods_per_year = 12.0

    net_vol = float(np.std(net_series, ddof=1) * np.sqrt(periods_per_year))
    net_annual = float(np.mean(net_series) * periods_per_year)
    gross_annual = float(np.mean(gross_returns) * periods_per_year)
    benchmark_annual = float(np.mean(benchmark_series) * periods_per_year)
    benchmark_vol = float(np.std(benchmark_series, ddof=1) * np.sqrt(periods_per_year))

    if len(turnovers) < 4:
        notes.append("fewer than four rebalances; turnover estimate is unstable")

    return BandResult(
        band=band,
        deployable_capital=deployable_capital(
            float(rows["median_dollar_volume"].median())
        ),
        n_rebalances=len(turnovers),
        n_positions_mean=float(np.mean(position_counts)) if position_counts else 0.0,
        gross_return_annual=gross_annual,
        net_return_annual=net_annual,
        net_volatility=net_vol,
        net_sharpe=net_annual / net_vol if net_vol > 0 else float("nan"),
        max_drawdown=_max_drawdown(np.array(equity)),
        benchmark_return_annual=benchmark_annual,
        benchmark_sharpe=(benchmark_annual / benchmark_vol
                          if benchmark_vol > 0 else float("nan")),
        turnover_annual=(float(np.mean(turnovers)) * (12.0 / REBALANCE_MONTHS)
                         if turnovers else 0.0),
        cost_drag_annual=float(np.sum(costs)) / max(len(equity) - 1, 1) * 12.0,
        measured_share=measured_share,
        excluded_upper_bound=excluded,
        notes=notes,
        cost_drag_annual_realistic=(float(np.sum(costs_realistic))
                                    / max(len(equity) - 1, 1) * 12.0),
    )


def capacity_statistic(results: list[BandResult],
                       n_permutations: int = 100_000,
                       seed: int = 42,
                       use_excess: bool = False) -> dict:
    """The single registered primary test: is net Sharpe decreasing in capacity?

    Spearman rho between deployable capital and net Sharpe, with a one-sided permutation
    p-value. Permutation rather than the asymptotic p because there are only a handful
    of bands and the asymptotic distribution is not trustworthy at that n.
    """
    usable = [r for r in results if np.isfinite(r.net_sharpe)]
    if len(usable) < 3:
        return {"rho": float("nan"), "p_value": float("nan"),
                "n_bands": len(usable),
                "verdict": "INSUFFICIENT BANDS - hypothesis not testable"}

    capital = np.array([r.deployable_capital for r in usable])
    if use_excess:
        # Excess over each band's OWN equal-weight buy-and-hold. The registered
        # statistic uses raw net Sharpe, but raw returns fall with capacity partly
        # because the small-cap premium does, so the registered version cannot by
        # itself separate a capacity effect from a size effect. This variant is
        # reported alongside it, never instead of it.
        sharpe = np.array([
            (r.net_return_annual - r.benchmark_return_annual) / r.net_volatility
            if r.net_volatility > 0 else np.nan for r in usable
        ])
    else:
        sharpe = np.array([r.net_sharpe for r in usable])
    if not np.all(np.isfinite(sharpe)):
        return {"rho": float("nan"), "p_value": float("nan"),
                "n_bands": len(usable), "verdict": "NON-FINITE INPUTS"}
    rho = float(stats.spearmanr(capital, sharpe).statistic)

    rng = np.random.default_rng(seed)
    permuted = np.array([
        stats.spearmanr(capital, rng.permutation(sharpe)).statistic
        for _ in range(n_permutations)
    ])
    # One-sided: H1 predicts a NEGATIVE correlation.
    p_value = float(np.mean(permuted <= rho))

    if rho >= 0:
        verdict = "H1 REFUTED - net performance does not decline with capacity"
    elif p_value < 0.05:
        verdict = "H1 SUPPORTED - net performance declines with capacity"
    else:
        verdict = "H1 NOT SUPPORTED - direction is right but not significant"

    return {"rho": rho, "p_value": p_value, "n_bands": len(usable),
            "verdict": verdict}
