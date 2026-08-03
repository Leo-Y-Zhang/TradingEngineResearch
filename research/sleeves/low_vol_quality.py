"""LOW-VOLATILITY / QUALITY sleeve, measured inside each tradable capacity band.

PRE-REGISTRATION
================
Written before the run. ONE configuration, run ONCE. Nothing below was chosen after
seeing a result; if it fails, the failure is the result.

Hypothesis
----------
The low-volatility anomaly (low-risk stocks earn higher *risk-adjusted* returns) is
usually explained by leverage constraints: investors who want more return but cannot
borrow bid up high-beta names instead, so high beta is over-priced and low beta is
under-priced. That mechanism is an ARBITRAGE-COST story, so it predicts the anomaly is
strongest exactly where arbitrage is dearest -- small, illiquid names. This is the one
anomaly whose mechanism predicts survival in the capacity band this project can reach.

H1: net-of-cost EXCESS return over each band's own equal-weight buy-and-hold is positive
    in at least one band, and is larger in the less liquid bands.

The prior capacity study (`capacity_curve_result.md`) found a fundamental-composite
ordering that was real but lost to buy-and-hold in EVERY band. The only thing being
tested here is whether a low-risk signal, whose economic mechanism is different, does
something the fundamental composite could not.

Signal (fixed, unfitted)
------------------------
Three legs, cross-sectionally winsorised (2% tails) then z-scored WITHIN (band, month),
then averaged with EQUAL weight. A name needs all three legs to be ranked.

  (a) LOW VOLATILITY   z(-realised_vol_252)
      Trailing 252 trading-day sample standard deviation of daily simple returns on the
      split/dividend-adjusted close.
  (b) LOW BETA         z(-beta_252)
      Trailing 252-day OLS beta against an EQUAL-WEIGHT market proxy built from this
      panel (cross-sectional mean daily return of every name priced >= $2 with non-zero
      volume that day). Equal weight, not cap weight, because the constraint story is
      about the cross-section of names, not about index membership.
  (c) QUALITY          equal-weight mean of z(gross_profitability), z(-debt_to_equity),
      z(-accruals), requiring at least 2 of the 3.
      gross_profitability = gp / assets       (Novy-Marx 2013)
      debt_to_equity      = debt / equity     (low leverage = quality)
      accruals            = (netinc - ncfo) / assets   (Sloan 1996; low = quality)
      All three from Sharadar SF1, dimension **ART** (As-Reported, trailing twelve
      months). AR* is the original filing, never restated. TTM rather than a single
      quarter because gp is a flow and quarterly gp/assets is dominated by fiscal
      seasonality across firms with different year ends.

Point-in-time
-------------
SF1 is attached by ``datekey`` (the publication date), never ``calendardate``, via a
backward ``merge_asof`` onto the rebalance grid. Volatility and beta at a month-end use
only bars up to and including that month-end. No bar after 2015-12-31 is ever read; the
DEV guard lives in ``research.capacity_panel.load_prices``.

Universe and bands
------------------
Cells from the cached monthly panel with ``spread_regime == "measured"`` -- i.e. the
name's EDGE spread is a genuine measurement rather than an upper bound at the resolution
floor. Names at the floor are EXCLUDED, never costed at the floor. That filter also
carries the panel's artefact filters (close >= $2, high>low and volume>0 on >= 90% of the
trailing 63 days).

Four bands on trailing-63-day median dollar volume, run SEPARATELY:
    $200k-$1M, $1M-$5M, $5M-$25M, >$25M   (the last is B5+B6 of the panel's ladder).

Extra artefact filter specific to THIS sleeve
---------------------------------------------
A low-volatility signal has an obvious failure mode the prior studies did not: a stock
that does not trade prints the same close every day, its realised volatility is ~0, and
it therefore ranks BEST on the headline leg. So a name is also required to have
    - at least 200 valid daily returns in its 252-day window, and
    - fewer than 50% of those returns exactly zero, and
    - realised volatility strictly positive.
Forward returns are clipped to +/-100% for BOTH the strategy and the benchmark.

Construction
------------
Monthly rebalance, long-only, top 30 by composite, equal weight. No no-trade band (the
prior study's quarterly no-trade-band construction still traded too much; this one is
deliberately the plain construction so the cost verdict is not confounded by a smoothing
rule that was itself never validated). A month with fewer than 60 rankable names does not
rebalance -- picking 30 from fewer than 60 is closer to owning the universe than to
selecting from it.

Costs (mandatory, per name, identical model to the capacity study for comparability)
------------------------------------------------------------------------------------
One-way, charged on every leg traded (entries and exits alike):
    half the name's own EDGE spread
  + square-root market impact, 0.1 * sqrt(trade value / that name's median dollar volume)
  + IBKR tiered commission: $0.0035/share, $0.35 per-order minimum, capped at 1% of value
  + $0.00002 FX each way
Position size = deployable capital / 30, where deployable capital = 30 * 1% of the band's
median dollar volume. Every band therefore trades at its own capacity, which is what
makes the bands comparable.

Benchmark
---------
Equal-weight buy-and-hold of THIS band's own measurable universe, reconstituted monthly,
with IDENTICAL return accounting (same +/-100% clip, same delisting terminal returns).
Reported GROSS of costs, which is the conservative direction for the strategy. The
headline number is EXCESS over that benchmark, not raw return.

Delistings
----------
The terminal return from ``delistings.parquet`` is applied only if the delisting date
falls in ``(exit_date, exit_date + 62 days]``. A name is REMOVED from the book the moment
its exit is booked. Both rules exist because both were violated once before and produced
-60%/yr and -112%/yr respectively.

Breadth
-------
Reported, because the whole programme's Sharpe ceiling is sqrt(BR). Measured as
    N_eff = (average cross-sectional variance of holding residuals) /
            (time-series variance of the portfolio residual)
    breadth = 12 * N_eff
i.e. the effective number of INDEPENDENT bets the 30-name book actually makes each month,
derived from the realised co-movement of its own positions, not asserted.

Decision rule (pre-committed)
-----------------------------
PROMISING  -- some band has net excess over its own buy-and-hold > +2%/yr AND net
              Sharpe >= 0.75 (the programme's promotion gate).
MARGINAL   -- some band has positive net excess but misses the Sharpe gate.
DEAD       -- no band has positive net excess after costs.
No band edges, horizons, weights or filters will be adjusted afterwards to change which
of these three lands.

ERRATUM 1 -- harness defect found on the first run and fixed (2026-07-28)
-------------------------------------------------------------------------
The first execution reported 681 monthly periods inside a 213-month window, a median
cross-section of zero, and a benchmark of 3.1-4.0%/yr on a universe the prior capacity
study measured at 8.4-12.0%/yr. All three are impossible, so the harness was wrong.

Cause: the cached panel's "month end" is each NAME's own last trading bar of the month,
not a shared grid date. A name that stopped trading on the 14th carries a date no other
name shares. Iterating over distinct dates therefore created 1,454 extra pseudo-periods
(0.48% of cells, median cross-section 1 name), and on each of them every holding was
absent from the cross-section, was closed out, and was rebought at full cost the next
month. Turnover, cost drag and the number of periods were all inflated; the arithmetic
annualisation was diluted by a factor of ~3.

Fix: iterate over CALENDAR MONTHS. A name has at most one panel row per calendar month by
construction, so this is exact rather than an approximation, and it is asserted at run
time. A position's exit is now dated at the name's last observed bar rather than at a
calendar month end, so the 62-day delisting window is measured from when the position
actually closed.

Nothing about the SIGNAL, the universe, the cost model, the benchmark or the decision rule
changed. This is the same class of defect the capacity study recorded in its §4 -- found
because the first output was impossible, not because a test caught it.

ERRATUM 3 -- benchmark survivorship asymmetry found and fixed (2026-07-28)
--------------------------------------------------------------------------
The strategy booked a delisted holding's terminal return (-100% for a bankruptcy), while
the benchmark dropped that same cell because its forward return was missing. Every
bankruptcy in the universe therefore cost the strategy everything and cost the benchmark
nothing -- exactly the survivorship bias the panel exists to avoid, pointed at the
comparison rule the verdict depends on.

Fix: the terminal return is resolved ONCE per cell into a single ``realised_return``
column that BOTH sides read. Reported honestly because the fix moves the number in the
strategy's favour: the annualised delisting drag it charges is reported per band so the
size of the correction is visible rather than absorbed.

One second-order asymmetry remains and is NOT corrected, because it also runs against the
strategy: a holding that leaves the MEASURABLE universe (its spread stops resolving)
is exited by the strategy and may have a terminal return charged, while the benchmark
simply stops including it.

ERRATUM 2 -- second benchmark declared (2026-07-28, before the corrected run)
-----------------------------------------------------------------------------
The registered benchmark is the band's whole measurable universe. But ~15% of measurable
cells have no SF1 coverage and are therefore un-pickable, so that benchmark is not quite
the opportunity set the strategy chooses from. The equal-weight return of the RANKABLE
subset is reported alongside it as a declared diagnostic. It is not a softer test: names
with fundamental coverage are typically the better-established ones, so it is the harder
of the two to beat. The headline excess remains the registered one.
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
    IMPACT_COEFFICIENT,
)
from research.delisting import CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW
from research.delisting import DELISTING_WINDOW_DAYS, in_window, in_window_mask

logger = logging.getLogger(__name__)

__all__ = [
    "BAND_GROUPS",
    "CORRECTED_DELISTING_WINDOW",
    "DELISTING_WINDOW_DAYS",
    "MIN_CROSS_SECTION",
    "N_POSITIONS",
    "REGISTERED_DELISTING_WINDOW",
    "RISK_WINDOW",
    "SleeveResult",
    "band_group",
    "build_signal",
    "risk_features",
    "run_band",
]

# --- registered constants -------------------------------------------------------------
RISK_WINDOW = 252            # trading days for realised vol and beta
MIN_RISK_OBSERVATIONS = 200  # valid daily returns required inside that window
MAX_ZERO_RETURN_FRACTION = 0.50
N_POSITIONS = 30
MIN_CROSS_SECTION = 60       # 2x n_positions; below this the "selection" is the universe
PARTICIPATION_LIMIT = 0.01   # position cap as a share of the name's median dollar volume
FORWARD_RETURN_CLIP = 1.00
MIN_PROXY_PRICE = 2.00       # price floor for membership of the equal-weight market proxy
WINSOR_QUANTILE = 0.02
# DELISTING_WINDOW_DAYS (=62) now has ONE definition, in `research.delisting`, and is
# imported above and re-exported here for every module that already reads it from this
# one. Two copies of a registered constant is how they drift.
MONTHS_PER_YEAR = 12.0

# The four registered bands. The panel's B5/B6 are merged because the sleeve is specified
# over ">$25M" as a single band.
BAND_GROUPS: dict[str, tuple[str, ...]] = {
    "B2_200k_1M": ("B2_200k_1M",),
    "B3_1M_5M": ("B3_1M_5M",),
    "B4_5M_25M": ("B4_5M_25M",),
    "B5_25M_plus": ("B5_25M_200M", "B6_200M_plus"),
}

QUALITY_LEGS: dict[str, int] = {
    # column -> sign that makes "higher is better quality"
    "gross_profitability": +1,
    "debt_to_equity": -1,
    "accruals": -1,
}


def band_group(band: str | None) -> str | None:
    """Map a panel band label onto this sleeve's registered band grouping."""
    for group, members in BAND_GROUPS.items():
        if band in members:
            return group
    return None


# --------------------------------------------------------------------------------------
# Trailing risk features
# --------------------------------------------------------------------------------------
def _segment_starts(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(first-row-index-of-this-ticker per row, is-first-row-of-a-ticker mask)``.

    The price frame is sorted by (ticker, date), so a ticker occupies one contiguous
    block. Knowing where a block starts is what stops a rolling window reaching back into
    the PREVIOUS ticker's bars -- which would silently fabricate returns.
    """
    n = codes.size
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = codes[1:] != codes[:-1]
    starts = np.maximum.accumulate(np.where(is_new, np.arange(n, dtype=np.int64), 0))
    return starts, is_new


def _window_sums(values: np.ndarray, at: np.ndarray, starts: np.ndarray,
                 window: int) -> np.ndarray:
    """Trailing-``window`` sums of ``values``, evaluated only at rows ``at``.

    A prefix-sum difference rather than ``groupby().rolling()``: the price frame has 29M
    rows and only ~1.3M of them are rebalance dates, so materialising a rolling result for
    every row costs 20x the memory for no information. The window is clamped at the
    ticker's own first bar via ``starts``.
    """
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    left = np.maximum(at - window + 1, starts[at])
    return prefix[at + 1] - prefix[left]


def risk_features(
    prices: pd.DataFrame,
    window: int = RISK_WINDOW,
    min_observations: int = MIN_RISK_OBSERVATIONS,
) -> pd.DataFrame:
    """Trailing realised volatility and beta at every (ticker, month-end).

    Both are computed from daily simple returns on ``closeadj`` (so splits and dividends
    are already handled) against an equal-weight market proxy built from the same panel.
    The value at a month-end uses only bars up to and including that month-end, so a
    strategy acting on it at the close is not reading its own future.

    Returns a frame of ``ticker, date, realised_vol, beta, n_obs, zero_fraction`` with one
    row per (ticker, month-end).
    """
    required = {"ticker", "date", "close", "closeadj", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price frame missing columns {sorted(missing)}")

    ticker_codes = pd.factorize(prices["ticker"], sort=False)[0]
    starts, is_new = _segment_starts(ticker_codes)

    # Month-end rows: the last bar of each (ticker, calendar month), which is exactly the
    # grid `build_monthly_panel` used, so these join to the panel one-for-one. Identified
    # first so the ticker/date labels can be captured and the wide columns released --
    # the price frame is 29M rows and this machine has 8 GB free.
    month = prices["date"].to_numpy().astype("datetime64[M]")
    last_of_month = np.empty(len(prices), dtype=bool)
    last_of_month[-1] = True
    # Row i ends a holding period if the NEXT row starts a new ticker, or is a new month.
    last_of_month[:-1] = is_new[1:] | (month[1:] != month[:-1])
    del month
    at = np.flatnonzero(last_of_month)
    out_tickers = np.asarray(prices["ticker"])[at]
    out_dates = prices["date"].to_numpy()[at]
    del last_of_month

    close_adjusted = prices["closeadj"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.empty_like(close_adjusted)
        returns[0] = np.nan
        returns[1:] = close_adjusted[1:] / np.where(close_adjusted[:-1] > 0.0,
                                                    close_adjusted[:-1], np.nan) - 1.0
    del close_adjusted
    # First bar of every ticker has no predecessor inside its own block.
    returns[is_new] = np.nan
    # Same +/-100% artefact clip the forward returns get. A 949x daily print is a data
    # error, and left alone it would dominate both the volatility and the beta estimate.
    np.clip(returns, -FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP, out=returns)

    # Equal-weight market proxy: cross-sectional mean daily return over names that were
    # genuinely investable that day. Without the price floor the proxy is dominated by
    # sub-$1 shells, which is not a market anyone could have held.
    valid_own = np.isfinite(returns)
    proxy_member = (
        valid_own
        & (prices["close"].to_numpy(dtype=np.float64) >= MIN_PROXY_PRICE)
        & (prices["volume"].to_numpy(dtype=np.float64) > 0.0)
    )
    date_codes, date_values = pd.factorize(prices["date"], sort=True)
    proxy_sum = np.bincount(date_codes, weights=np.where(proxy_member, returns, 0.0),
                            minlength=len(date_values))
    proxy_count = np.bincount(date_codes, weights=proxy_member.astype(np.float64),
                              minlength=len(date_values))
    del proxy_member
    with np.errstate(invalid="ignore", divide="ignore"):
        proxy_by_date = np.where(proxy_count > 0, proxy_sum / proxy_count, np.nan)
    market = proxy_by_date[date_codes]
    del date_codes
    logger.info("equal-weight market proxy: %s trading days, median %s names/day",
                f"{len(date_values):,}", f"{np.median(proxy_count):,.0f}")

    usable = valid_own & np.isfinite(market)
    del valid_own
    own = np.where(usable, returns, 0.0)
    mkt = np.where(usable, market, 0.0)
    del market

    count = _window_sums(usable.astype(np.float64), at, starts, window)
    sum_own = _window_sums(own, at, starts, window)
    sum_own_sq = _window_sums(own * own, at, starts, window)
    sum_mkt = _window_sums(mkt, at, starts, window)
    sum_mkt_sq = _window_sums(mkt * mkt, at, starts, window)
    sum_cross = _window_sums(own * mkt, at, starts, window)
    zero_count = _window_sums((usable & (returns == 0.0)).astype(np.float64),
                              at, starts, window)

    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = np.where(count > 1, count - 1.0, np.nan)
        variance_own = (sum_own_sq - sum_own * sum_own / count) / denominator
        variance_mkt = (sum_mkt_sq - sum_mkt * sum_mkt / count) / denominator
        covariance = (sum_cross - sum_own * sum_mkt / count) / denominator
        volatility = np.sqrt(np.maximum(variance_own, 0.0))
        beta = np.where(variance_mkt > 0.0, covariance / variance_mkt, np.nan)
        zero_fraction = np.where(count > 0, zero_count / count, np.nan)

    # The artefact guard this sleeve specifically needs: a name that barely trades has
    # near-zero realised volatility and would otherwise rank BEST on the headline leg.
    unusable = (
        (count < min_observations)
        | ~np.isfinite(volatility)
        | (volatility <= 0.0)
        | (zero_fraction >= MAX_ZERO_RETURN_FRACTION)
    )
    volatility = np.where(unusable, np.nan, volatility)
    beta = np.where(unusable, np.nan, beta)

    return pd.DataFrame({
        "ticker": out_tickers,
        "date": out_dates,
        "realised_vol": volatility,
        "beta": beta,
        "risk_n_obs": count,
        "zero_return_fraction": zero_fraction,
    })


# --------------------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------------------
def _winsorised_z(block: pd.Series) -> pd.Series:
    """2%-winsorised z-score of one cross-section. NaN if it cannot be defined."""
    values = pd.to_numeric(block, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return pd.Series(np.nan, index=block.index, dtype=float)
    low, high = finite.quantile(WINSOR_QUANTILE), finite.quantile(1.0 - WINSOR_QUANTILE)
    clipped = values.clip(lower=low, upper=high)
    finite_clipped = clipped[np.isfinite(clipped)]
    spread = float(finite_clipped.std(ddof=0))
    if not np.isfinite(spread) or spread == 0.0:
        return pd.Series(np.nan, index=block.index, dtype=float)
    return (clipped - float(finite_clipped.mean())) / spread


def build_signal(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the three registered legs and their equal-weight composite.

    Normalisation is strictly within (band group, date): the sleeve is run separately per
    band, so a name must be ranked against the names it actually competes with for a slot,
    not against a market-wide cross-section it will never be compared to.
    """
    work = panel.copy()
    keys = [work["band_group"].to_numpy(), work["date"].to_numpy()]

    work["leg_low_vol"] = (
        (-work["realised_vol"]).groupby(keys, sort=False).transform(_winsorised_z)
    )
    work["leg_low_beta"] = (
        (-work["beta"]).groupby(keys, sort=False).transform(_winsorised_z)
    )

    quality_parts = []
    for column, sign in QUALITY_LEGS.items():
        z = (sign * work[column]).groupby(keys, sort=False).transform(_winsorised_z)
        quality_parts.append(z.rename(column))
    quality = pd.concat(quality_parts, axis=1)
    # At least 2 of 3 quality inputs: SF1 coverage is patchiest in exactly the small
    # illiquid names this sleeve is about, and demanding all three would quietly tilt the
    # universe toward larger, better-covered names.
    work["leg_quality"] = quality.mean(axis=1, skipna=True).where(
        quality.notna().sum(axis=1) >= 2
    )

    legs = work[["leg_low_vol", "leg_low_beta", "leg_quality"]]
    # All three legs required: each is a third of a signal that was registered as a
    # three-way composite, and a name scored on two of them is a different strategy.
    work["signal"] = legs.mean(axis=1).where(legs.notna().all(axis=1))
    return work


# --------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------
@dataclass
class SleeveResult:
    band: str
    deployable_capital: float
    n_months: int
    n_rebalances: int
    n_positions_mean: float
    median_cross_section: float
    gross_return_annual: float
    gross_sharpe: float
    net_return_annual: float
    net_cagr: float
    net_volatility: float
    net_sharpe: float
    max_drawdown: float
    net_ex_impact_annual: float
    net_ex_impact_sharpe: float
    excess_ex_impact: float
    benchmark_max_drawdown: float
    forced_exit_share: float
    delisting_drag_annual: float
    benchmark_return_annual: float
    benchmark_cagr: float
    benchmark_volatility: float
    benchmark_rankable_annual: float
    excess_annual: float
    excess_cagr: float
    excess_vs_rankable: float
    turnover_annual: float
    cost_drag_annual: float
    cost_spread_annual: float
    cost_impact_annual: float
    cost_commission_annual: float
    breadth_per_year: float
    effective_bets_per_rebalance: float
    mean_holding_vol: float
    mean_universe_vol: float
    notes: list[str] = field(default_factory=list)
    #: exit legs counted in TURNOVER but charged NOTHING because the name had left the
    #: tradable universe. A name that leaves still has to be sold. See
    #: `charge_unpriced_exits`.
    unpriced_exit_legs: int = 0
    #: how many of those were actually charged, at the name's LAST OBSERVED cost inputs
    charged_unpriced_exit_legs: int = 0


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def _cost_components(spread: float, trade_value: float, price: float,
                     median_dollar_volume: float) -> tuple[float, float, float]:
    """One-way (spread, impact, commission+FX) fractions for a single leg."""
    half_spread = spread / 2.0
    impact = (IMPACT_COEFFICIENT * np.sqrt(max(trade_value / median_dollar_volume, 0.0))
              if median_dollar_volume > 0 else np.nan)
    if trade_value <= 0 or price <= 0:
        commission = 0.0
    else:
        shares = trade_value / price
        raw = max(COMMISSION_MIN_PER_ORDER, shares * COMMISSION_PER_SHARE)
        commission = min(raw, trade_value * COMMISSION_MAX_FRACTION) / trade_value
    return half_spread, impact, commission + FX_COST_EACH_WAY


def run_band(
    panel: pd.DataFrame,
    band: str,
    delistings: pd.DataFrame,
    delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
    charge_unpriced_exits: bool = False,
) -> SleeveResult | None:
    """Backtest the registered construction inside one band.

    Two switches, BOTH defaulting to the registered behaviour so the banked run
    reproduces bit-for-bit:

    `delisting_window` -- see `research.delisting`. The registered lower edge is STRICT,
    and Sharadar dates a delisting ON the ticker's last traded bar (median gap 0 days),
    so the registered window rejects the modal case. `CORRECTED_DELISTING_WINDOW` is the
    repair and must be declared, never slipped into a headline.

    `charge_unpriced_exits` -- the registered run counts an exit leg in TURNOVER but
    charges it NOTHING when the name has left the tradable universe (its price fell
    through the floor, its dollar volume left the band, or its spread stopped resolving).
    A name that leaves the universe still has to be SOLD. Setting this True prices the
    leg at the name's LAST OBSERVED inputs, which is the nearest honest estimate
    available and is certainly too cheap -- a name that just fell out of the universe
    trades worse, not better. `unpriced_exit_legs` counts them either way, so the size of
    the omission is always visible.
    """
    rows = panel[(panel["band_group"] == band)].copy()
    if rows.empty:
        return None

    notes: list[str] = []

    # Terminal return keyed by ticker AND date. Absence from the universe is NOT evidence
    # of delisting -- a name leaves because its band changed or its spread stopped
    # resolving far more often than because it died.
    terminal = {
        row.ticker: (row.date, float(row.terminal_return))
        for row in delistings.itertuples()
    }

    def exit_return(ticker: str, at: pd.Timestamp) -> float:
        entry = terminal.get(ticker)
        if entry is None:
            return 0.0
        delisted_on, value = entry
        if in_window(at, delisted_on, delisting_window):
            return value
        return 0.0

    # One realised-return column used IDENTICALLY by the strategy and the benchmark, so
    # the comparison cannot be won or lost on accounting asymmetry.
    #
    # A cell with no forward return is a name's LAST observation. Dropping those from the
    # benchmark while charging them to the strategy would give the benchmark exactly the
    # survivorship bias this panel was built to avoid: every bankruptcy would cost the
    # strategy 100% and cost the benchmark nothing. So the terminal return is resolved
    # here, once, for every cell, and both sides read the same column.
    forward = rows["forward_return"].clip(-FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP)
    delist_date = pd.to_datetime(
        rows["ticker"].map({row.ticker: row.date for row in delistings.itertuples()}),
        errors="coerce",
    )
    delist_value = pd.to_numeric(
        rows["ticker"].map(
            {row.ticker: float(row.terminal_return) for row in delistings.itertuples()}
        ),
        errors="coerce",
    )
    within = in_window_mask(rows["date"], delist_date, delisting_window)
    rows["terminal_on_exit"] = np.where(within, delist_value.fillna(0.0), 0.0)
    rows["forward_clipped"] = forward
    rows["realised_return"] = forward.where(forward.notna(), rows["terminal_on_exit"])

    deployable = N_POSITIONS * PARTICIPATION_LIMIT * float(
        rows["median_dollar_volume"].median()
    )
    position_value = deployable / N_POSITIONS

    # Rebalance on a CALENDAR-MONTH grid, not on the panel's raw dates.
    #
    # The panel's "month end" is each name's OWN last trading bar of the month, so a name
    # that stopped trading on the 14th carries a date no other name shares. Treating every
    # distinct date as a period turned 213 real months into 1,667 pseudo-periods, 1,454 of
    # which held a single dying name -- and on each of those the whole book vanished from
    # the cross-section, was closed out, and was rebought at full cost the next month.
    # Symptoms: 681 "months" inside a 213-month window, a median cross-section of zero,
    # and a benchmark of 3%/yr where the same universe is known to return ~12%/yr.
    # A name has at most one row per calendar month by construction, so grouping by month
    # is exact, not an approximation.
    rows["month"] = rows["date"].dt.to_period("M")
    duplicated = int(rows.duplicated(["ticker", "month"]).sum())
    if duplicated:
        raise ValueError(f"{duplicated} (ticker, month) duplicates; the grid is not monthly")
    by_month = {month: frame for month, frame in rows.groupby("month", sort=True)}
    months = sorted(by_month)

    holdings: set[str] = set()
    # The date a held name was last observed. Used as the EXIT date when the name later
    # disappears, so the delisting window is measured from when the position actually
    # closed rather than from an arbitrary calendar month end.
    last_seen: dict[str, pd.Timestamp] = {}
    equity = [1.0]
    benchmark_equity = [1.0]
    gross_returns: list[float] = []
    net_returns: list[float] = []
    benchmark_returns: list[float] = []
    benchmark_rankable_returns: list[float] = []
    active_returns: list[float] = []
    residual_variance: list[float] = []
    holding_vols: list[float] = []
    universe_vols: list[float] = []
    costs: list[float] = []
    cost_spread: list[float] = []
    cost_impact: list[float] = []
    cost_commission: list[float] = []
    legs_traded: list[int] = []
    position_counts: list[int] = []
    cross_sections: list[int] = []
    delisting_contributions: list[float] = []
    n_rebalances = 0
    # Why a position was sold. A signal that barely moves should trade rarely; if most
    # exits are FORCED -- the name simply stopped being rankable this month, because its
    # band flipped or its spread stopped resolving -- then the measured turnover is a
    # property of the measurement panel, not of the strategy, and the cost estimate
    # inherits that.
    forced_exits = 0
    discretionary_exits = 0
    unpriced_exit_legs = 0
    charged_unpriced_exit_legs = 0
    # Last observed cost inputs per held name, so an exit that happens after the name has
    # left the tradable universe can still be priced (see `charge_unpriced_exits`).
    last_cost: dict[str, tuple[float, float, float]] = {}

    for month in months:
        cross_section = by_month[month]
        rankable = cross_section[cross_section["signal"].notna()]
        cross_sections.append(len(rankable))

        period_cost = 0.0
        period_spread = 0.0
        period_impact = 0.0
        period_commission = 0.0
        traded: set[str] = set()

        if len(rankable) >= MIN_CROSS_SECTION:
            n_rebalances += 1
            target = set(
                rankable.nlargest(N_POSITIONS, "signal")["ticker"]
            )
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
                    # A held name that has vanished from the universe has no measurable
                    # price THIS month -- but it still has to be sold, so skipping it
                    # UNDERSTATES cost while `legs_traded` still counts the leg. The
                    # exit-return path below books the RETURN, never a COST.
                    unpriced_exit_legs += 1
                    if not (charge_unpriced_exits and ticker in last_cost):
                        continue
                    spread_last, price_last, mdv_last = last_cost[ticker]
                    spread_part, impact_part, commission_part = _cost_components(
                        spread_last, position_value, price_last, mdv_last,
                    )
                    if not np.isfinite(impact_part):
                        impact_part = 0.0
                    period_spread += weight * spread_part
                    period_impact += weight * impact_part
                    period_commission += weight * commission_part
                    charged_unpriced_exit_legs += 1
                    continue
                row = priced.loc[ticker]
                spread_part, impact_part, commission_part = _cost_components(
                    float(row["spread"]), position_value, float(row["close"]),
                    float(row["median_dollar_volume"]),
                )
                if not np.isfinite(impact_part):
                    impact_part = 0.0
                period_spread += weight * spread_part
                period_impact += weight * impact_part
                period_commission += weight * commission_part
            period_cost = period_spread + period_impact + period_commission
            holdings = target

        costs.append(period_cost)
        cost_spread.append(period_spread)
        cost_impact.append(period_impact)
        cost_commission.append(period_commission)
        legs_traded.append(len(traded))
        position_counts.append(len(holdings))

        # The registered benchmark: equal-weight buy-and-hold of the band's whole
        # MEASURABLE universe. `rankable_step` is the declared diagnostic -- the same
        # thing restricted to names the strategy could actually have chosen, which is the
        # harder comparison whenever fundamental coverage tilts toward better names.
        universe = cross_section["realised_return"].dropna()
        step = float(universe.mean()) if len(universe) else 0.0
        rankable_universe = rankable["realised_return"].dropna()
        rankable_step = (float(rankable_universe.mean())
                         if len(rankable_universe) else step)
        benchmark_returns.append(step)
        benchmark_rankable_returns.append(rankable_step)
        benchmark_equity.append(benchmark_equity[-1] * (1.0 + step))

        if not holdings:
            gross_returns.append(0.0)
            net_returns.append(-period_cost)
            equity.append(equity[-1] * (1.0 - period_cost))
            continue

        indexed = cross_section.set_index("ticker")
        realised: list[float] = []
        closing_out: list[str] = []
        vols: list[float] = []
        month_terminal = 0.0
        for ticker in holdings:
            if ticker not in indexed.index:
                # Left the measurable universe. Book the exit ONCE and drop the name;
                # leaving it in `holdings` re-books the same terminal return every month
                # thereafter, which is how a long-only book once "lost" 112%/yr. The exit
                # is dated at the name's LAST OBSERVED bar, so a delisting only counts if
                # it happened within 62 days of the position actually closing.
                exit_date = last_seen.get(ticker, month.to_timestamp(how="end"))
                value = exit_return(ticker, exit_date)
                realised.append(value)
                month_terminal += value
                closing_out.append(ticker)
                continue
            row = indexed.loc[ticker]
            last_seen[ticker] = row["date"]
            last_cost[ticker] = (float(row["spread"]), float(row["close"]),
                                 float(row["median_dollar_volume"]))
            realised.append(float(row["realised_return"]))
            if pd.isna(row["forward_clipped"]):
                # Last observation for this name: the position closes here, at the
                # registered terminal return already resolved into `realised_return`.
                month_terminal += float(row["realised_return"])
                closing_out.append(ticker)
                continue
            vols.append(float(row["realised_vol"]))
        holdings.difference_update(closing_out)

        delisting_contributions.append(month_terminal / max(len(realised), 1))

        gross = float(np.mean(realised))
        gross = max(gross, -1.0)
        net = max(gross - period_cost, -1.0)
        gross_returns.append(gross)
        net_returns.append(net)
        equity.append(equity[-1] * (1.0 + net))

        # Breadth inputs: the residual of each holding against the universe it was picked
        # from, and the residual of the portfolio as a whole.
        residuals = np.asarray(realised, dtype=float) - step
        if residuals.size > 1:
            residual_variance.append(float(np.mean(residuals ** 2)))
            active_returns.append(gross - step)
        if vols:
            holding_vols.append(float(np.mean(vols)))
        universe_vol = cross_section["realised_vol"].dropna()
        if len(universe_vol):
            universe_vols.append(float(universe_vol.mean()))

    if len(net_returns) < 24:
        return None

    net_series = np.asarray(net_returns, dtype=float)
    gross_series = np.asarray(gross_returns, dtype=float)
    benchmark_series = np.asarray(benchmark_returns, dtype=float)
    equity_curve = np.asarray(equity, dtype=float)
    benchmark_curve = np.asarray(benchmark_equity, dtype=float)
    years = len(net_series) / MONTHS_PER_YEAR

    net_vol = float(net_series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    gross_vol = float(gross_series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    benchmark_vol = float(benchmark_series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    net_annual = float(net_series.mean() * MONTHS_PER_YEAR)
    gross_annual = float(gross_series.mean() * MONTHS_PER_YEAR)
    benchmark_annual = float(benchmark_series.mean() * MONTHS_PER_YEAR)
    benchmark_rankable_annual = float(
        np.mean(benchmark_rankable_returns) * MONTHS_PER_YEAR
    )
    net_cagr = float(equity_curve[-1] ** (1.0 / years) - 1.0)
    benchmark_cagr = float(benchmark_curve[-1] ** (1.0 / years) - 1.0)

    # Declared diagnostic, NOT the verdict: the same book with market impact set to zero,
    # i.e. traded at infinitesimal size. It separates "the signal cannot pay the spread"
    # from "the position size assumed here is too large for the band". Zero impact is not
    # attainable -- a position IS 1% of the name's daily volume by construction -- so this
    # is an upper bound on what the sleeve could ever net, never a deployable figure.
    ex_impact_series = gross_series - (np.asarray(cost_spread, dtype=float)
                                       + np.asarray(cost_commission, dtype=float))
    ex_impact_vol = float(ex_impact_series.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    ex_impact_annual = float(ex_impact_series.mean() * MONTHS_PER_YEAR)

    # Effective independent bets per rebalance: the ratio of the average variance of ONE
    # holding's residual to the realised variance of the portfolio's residual. If the 30
    # positions were independent this is 30; if they are one common bet it is 1.
    if residual_variance and len(active_returns) > 2:
        individual = float(np.mean(residual_variance))
        portfolio = float(np.var(np.asarray(active_returns, dtype=float), ddof=1))
        effective = individual / portfolio if portfolio > 0 else float("nan")
    else:
        effective = float("nan")

    if n_rebalances < 12:
        notes.append("fewer than 12 rebalances; turnover and cost estimates are unstable")

    return SleeveResult(
        band=band,
        deployable_capital=deployable,
        n_months=len(net_series),
        n_rebalances=n_rebalances,
        n_positions_mean=float(np.mean(position_counts)) if position_counts else 0.0,
        median_cross_section=float(np.median(cross_sections)) if cross_sections else 0.0,
        gross_return_annual=gross_annual,
        gross_sharpe=gross_annual / gross_vol if gross_vol > 0 else float("nan"),
        net_return_annual=net_annual,
        net_cagr=net_cagr,
        net_volatility=net_vol,
        net_sharpe=net_annual / net_vol if net_vol > 0 else float("nan"),
        max_drawdown=_max_drawdown(equity_curve),
        net_ex_impact_annual=ex_impact_annual,
        net_ex_impact_sharpe=(ex_impact_annual / ex_impact_vol
                              if ex_impact_vol > 0 else float("nan")),
        excess_ex_impact=ex_impact_annual - benchmark_annual,
        benchmark_max_drawdown=_max_drawdown(benchmark_curve),
        forced_exit_share=(forced_exits / (forced_exits + discretionary_exits)
                           if (forced_exits + discretionary_exits) else float("nan")),
        delisting_drag_annual=float(np.sum(delisting_contributions)) / years,
        benchmark_return_annual=benchmark_annual,
        benchmark_cagr=benchmark_cagr,
        benchmark_volatility=benchmark_vol,
        benchmark_rankable_annual=benchmark_rankable_annual,
        excess_annual=net_annual - benchmark_annual,
        excess_cagr=net_cagr - benchmark_cagr,
        excess_vs_rankable=net_annual - benchmark_rankable_annual,
        turnover_annual=float(np.sum(legs_traded)) / max(N_POSITIONS, 1) / years,
        cost_drag_annual=float(np.sum(costs)) / years,
        cost_spread_annual=float(np.sum(cost_spread)) / years,
        cost_impact_annual=float(np.sum(cost_impact)) / years,
        cost_commission_annual=float(np.sum(cost_commission)) / years,
        breadth_per_year=effective * MONTHS_PER_YEAR,
        effective_bets_per_rebalance=effective,
        mean_holding_vol=(float(np.mean(holding_vols)) * np.sqrt(252.0)
                          if holding_vols else float("nan")),
        mean_universe_vol=(float(np.mean(universe_vols)) * np.sqrt(252.0)
                           if universe_vols else float("nan")),
        notes=notes,
        unpriced_exit_legs=unpriced_exit_legs,
        charged_unpriced_exit_legs=charged_unpriced_exit_legs,
    )
