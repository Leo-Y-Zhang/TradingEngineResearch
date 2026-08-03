"""Sleeve: multi-timeframe TIME-SERIES MOMENTUM on the liquid US single-name cross-section.

PRE-SPECIFICATION
=================
Everything in this section was fixed before the first backtest was run and is executed
exactly once. There are two declared configurations (PRIMARY and SENSITIVITY-B); both are
reported whatever they say. No parameter below was chosen by looking at a result.

Hypothesis
----------
An instrument's own trailing return predicts the sign of its next return (Moskowitz, Ooi
& Pedersen 2012). The claim under test is NOT that the signal is strong -- it is that
BREADTH (many instruments x several quasi-independent timeframes) can lift the
information ratio of a mediocre signal far enough to matter, per
`docs/project-control/specs/2026-07-28-the-breadth-lever.md`.

Why this is expected to be hard on THIS data
--------------------------------------------
Breadth wants INDEPENDENT instruments. US single names are not independent: they share
one dominant market factor. The effective number of independent instruments is therefore
measured and reported as a first-class result rather than assumed.

Universe (PRIMARY)
------------------
At each month-end t, from `monthly_panel_dev.parquet` (already point-in-time: liquidity
and the EDGE spread are computed on the trailing 63 sessions ENDING at t):
  * artefact filters inherited from the panel builder: close >= $2.00, non-zero volume on
    >= 90% of the trailing 63 sessions, a liquidity band assigned;
  * spread_regime == "measured" -- names whose EDGE estimate is an `upper_bound` or is
    `unmeasurable` are EXCLUDED, never costed at the floor (rule 3);
  * >= 252 trailing daily bars, so all three lookbacks exist;
  * the 200 survivors with the largest trailing-63d median dollar volume.
Plus SECTOR AGGREGATES: equal-weight baskets of those 200 names grouped by Sharadar
`sector`, minimum 8 live constituents. A basket's history is built from its CURRENT
constituents' past bars, which is point-in-time legal (today's membership, yesterday's
prices) but carries a mild current-members tilt, disclosed in the output.

Known and disclosed defect of the PRIMARY universe
--------------------------------------------------
The resolution test admits a name only when its EDGE estimate exceeds 1.5x the noise
floor (0.262 x daily vol). For a genuinely liquid name the true spread sits far BELOW
that floor, so the only way such a name passes is on an upward noise draw. The PRIMARY
universe is therefore the right tail of estimation error among liquid names, and its
spreads (median ~70-100bps on $50M/day stocks) are biased high by roughly an order of
magnitude for the post-decimalisation era. Rule 3 is non-negotiable so PRIMARY governs,
but a sleeve killed only by that bias would be killed by an artefact, hence:

ERRATUM 1 -- written BEFORE the single registered run, after a universe diagnostic
----------------------------------------------------------------------------------
Measuring the PRIMARY universe before running it: month-to-month membership retention is
**54%**. Nearly half the book is forced to turn over every month, not because any name
became illiquid but because the EDGE resolution FLAG flickers between `measured` and
`upper_bound` on a name whose liquidity did not change. (Same measurement on the
unfiltered top-200: **95%** retention.) That is 46%/month of pure artefact turnover paid
at a ~95bps median spread -- roughly 5%/yr of cost, before leverage, before the strategy
expresses a single view. PRIMARY as literally specified therefore measures the stability
of a spread-estimator flag, not the trend hypothesis.

Rule 3 is non-negotiable, so PRIMARY still runs and still governs formally. But a third
configuration is declared here, before any full-sample result has been seen, which obeys
rule 3's actual requirement -- no name is ever costed at the floor, and no name without a
genuine measurement is ever held -- without treating a flag flicker as a sell signal:

Universe (PRIMARY-STICKY, erratum 1, rule-3 compliant)
------------------------------------------------------
A name is eligible if it passes the panel's artefact filters AND carries a `measured`
EDGE spread from some session within the trailing 365 days. It is costed at that most
recent MEASURED spread -- a real measurement, possibly a few months stale, never a floor
and never a substitute. Top 200 by trailing dollar volume as before.

Disclosure of what was known when: this erratum was written after a 1998-2001 smoke test
of the plumbing had been executed. The justification above rests only on the 54%-vs-95%
retention statistic, which is a property of the universe and independent of any return.
Both numbers are reported below so the amendment can be checked.

Universe (SENSITIVITY-B, declared in advance, NOT gate-eligible)
---------------------------------------------------------------
Identical, except the spread_regime filter is dropped and `upper_bound` names are costed
at a FIXED 20bps full effective spread (10bps half). 20bps is above the true effective
spread of a $20M+/day US large cap after decimalisation and is roughly the 1/16 tick on a
$40 stock before it. This is the "substitute a liquid-name cost schedule" treatment the
`spread_with_resolution` docstring prescribes; it conflicts with rule 3, which is exactly
why it is secondary and reported separately.

Signal
------
s_i(t) = mean over L in {21, 63, 252} of sign( closeadj_i(t)/closeadj_i(t-L) - 1 ),
a value in {-1, -1/3, +1/3, +1}. Each lookback is one quasi-independent bet.

Sizing
------
b_i = s_i / sigma_i, with sigma_i the annualised stdev of the trailing 63 daily returns,
floored at 10% and capped at 150%. Normalised so sum|b_i| = 1.
Book scaling: sigma_book(t) = annualised stdev of b(t) applied to the trailing 126 daily
return vectors (point-in-time: today's weights, yesterday's returns).
k(t) = target_vol / sigma_book(t), capped at 8x gross. Targets: 15%, 30%, 45%.
Rebalanced monthly; positions drift with prices in between.

Costs (mandatory, per-name, charged on the NET executed trade per stock)
-----------------------------------------------------------------------
  * spread: 0.5 x the EDGE spread known at t x traded notional, per stock;
  * commission: $0.0035/share, $0.35 per-order minimum, capped at 1% of trade value;
  * a basket's trade is netted into its constituents and costed there -- a synthetic
    sector never trades more cheaply than the stocks inside it;
  * stock borrow: 1.00%/yr on short notional, accrued daily;
  * margin financing: 3.50%/yr on max(0, long notional - equity), accrued daily;
  * idle cash earns nothing.
Borrow and financing are flat assumptions and are disclosed as such; the spread, which
dominates, is per-name.

Delistings
----------
A held name that stops printing bars is force-exited on the next session. Its
`terminal_return` is applied ONLY if the delisting event falls within 62 calendar days
after its last bar, and the name is then removed from the book permanently -- it can
never be re-booked.

Artefact filters
----------------
Daily returns capped at +/-100% (count reported). The $2 price floor and the 90%
non-zero-volume rule are inherited from the panel builder.

Benchmark
---------
Equal-weight, monthly-rebalanced, long-only holding of the SAME universe, through the
SAME cost model. Excess is reported against the NET benchmark (apples to apples); the
gross benchmark is printed alongside.

Reported
--------
Gross and net return/vol/Sharpe, max drawdown, annual turnover, cost drag, excess, the
Sharpe of 1998-2005 and 2006-2015 SEPARATELY (a full-sample pass with a failing decade is
a FAILURE), and realised breadth:

    bets_per_year = N_eff_instruments x sum_over_timeframes(sign flips per year)

with N_eff the participation ratio (sum lambda)^2 / sum(lambda^2) of the universe's daily
return correlation matrix -- the honest count of independent instruments, not the
nominal one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.capacity_panel import DATA_DIR, DEV_CUTOFF, PANEL_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-specified constants. None was chosen by looking at a result.
# ---------------------------------------------------------------------------
LOOKBACKS: tuple[int, ...] = (21, 63, 252)
UNIVERSE_SIZE = 200
MIN_HISTORY = 252
VOL_WINDOW = 63
BOOK_VOL_WINDOW = 126
INST_VOL_FLOOR = 0.10
INST_VOL_CAP = 1.50
MAX_GROSS_LEVERAGE = 8.0
TARGET_VOLS: tuple[float, ...] = (0.15, 0.30, 0.45)

COMMISSION_PER_SHARE = 0.0035
COMMISSION_MIN_ORDER = 0.35
COMMISSION_CAP_FRACTION = 0.01

# The commission schedule is NOT scale-invariant: a $0.35 order minimum and a 1%-of-value
# cap only mean anything relative to an account size, so one has to be stated. $1,000,000
# spread over ~200 names is ~$5,000 a position, at which the minimum is under 1bp and the
# per-share charge dominates -- the regime a real book would run in. The smaller size is
# reported alongside because below roughly $250k the 1% cap starts binding on the smaller
# rebalancing trades and the sleeve becomes a different, worse strategy.
#
# ERRATUM 2 (implementation defect, found before any result was accepted): the first
# execution of this script ran on a notional equity of $1.00, so EVERY trade hit the
# 1%-of-value commission cap and the book paid 100bps a side on everything. The reported
# 27-72%/yr "cost drag" of that run was that bug, not a measurement. Nothing about the
# registered signal, universe or sizing changed in the fix.
STARTING_EQUITY = 1_000_000.0
SMALL_ACCOUNT_EQUITY = 100_000.0
BORROW_RATE = 0.0100
FINANCING_RATE = 0.0350
TRADING_DAYS = 252

# SENSITIVITY-B only: flat liquid-name spread substituted for `upper_bound` names.
LIQUID_SPREAD_SUBSTITUTE = 0.0020

DELISTING_WINDOW_DAYS = 62
DAILY_RETURN_CAP = 1.00
MIN_SECTOR_CONSTITUENTS = 8

DECADES: tuple[tuple[str, str, str], ...] = (
    ("1998-2005", "1998-01-01", "2005-12-31"),
    ("2006-2015", "2006-01-01", "2015-12-31"),
)


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
@dataclass
class Universe:
    """Monthly membership plus the point-in-time cost input each member carries."""

    dates: list[pd.Timestamp]
    members: dict[pd.Timestamp, list[str]]
    spread: dict[tuple[pd.Timestamp, str], float]
    sectors: dict[pd.Timestamp, dict[str, list[str]]]
    n_upper_bound_costed: int = 0
    n_measured: int = 0


MODES = ("measured_only", "sticky_measured", "liquid_schedule")

# Erratum 1: how stale a MEASURED spread may be and still be used to cost a trade.
STICKY_SPREAD_MAX_AGE_DAYS = 365


def build_universe(
    panel: pd.DataFrame,
    sector_of: dict[str, str],
    *,
    mode: str,
) -> Universe:
    """Top-`UNIVERSE_SIZE` liquid names per month with their sector groupings.

    `mode` selects the declared cost/eligibility treatment:
      * ``measured_only``  -- PRIMARY, rule 3 read literally: hold a name only in the
        months its EDGE spread resolves. Costs at that month's measurement.
      * ``sticky_measured`` -- PRIMARY-STICKY (erratum 1): hold a name whose spread has
        resolved at some point in the trailing year, costed at that measurement. Still
        never the floor, still never a name that has no measurement at all.
      * ``liquid_schedule`` -- SENSITIVITY-B, NOT gate-eligible: no regime filter,
        unresolved names costed at a flat liquid-name spread.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")

    frame = panel.copy()
    if mode == "measured_only":
        frame = frame[frame["spread_regime"] == "measured"]
    else:
        # `ineligible` rows failed the $2 floor or the 90% non-zero-volume test and can
        # never be held under any mode.
        frame = frame[frame["spread_regime"].isin(["measured", "upper_bound"])]

    # One row per (ticker, month): the LAST panel row inside that calendar month, i.e.
    # the last session the name traded that month. Nothing is read forward.
    frame = frame.sort_values(["ticker", "date"])
    frame["month"] = frame["date"].values.astype("datetime64[M]")
    frame = frame.groupby(["ticker", "month"], as_index=False).tail(1)

    if mode == "sticky_measured":
        # Carry each name's most recent MEASURED spread forward, with its age. This is a
        # backward fill in time only -- it can never see a future measurement.
        frame = frame.sort_values(["ticker", "date"])
        measured = frame["spread"].where(frame["spread_regime"] == "measured")
        frame["last_measured"] = measured.groupby(frame["ticker"]).ffill()
        stamp = frame["date"].where(frame["spread_regime"] == "measured")
        frame["last_measured_date"] = stamp.groupby(frame["ticker"]).ffill()
        age = (frame["date"] - frame["last_measured_date"]).dt.days
        frame = frame[frame["last_measured"].notna()
                      & (age <= STICKY_SPREAD_MAX_AGE_DAYS)]

    frame["rank"] = frame.groupby("month")["median_dollar_volume"].rank(
        ascending=False, method="first"
    )
    frame = frame[frame["rank"] <= UNIVERSE_SIZE]

    universe = Universe(dates=[], members={}, spread={}, sectors={})
    for _month, group in frame.groupby("month"):
        # The cohort's last session is the rebalance day: every name acts on the same
        # date. A name's own inputs still come from its own row.
        rebalance_date = pd.Timestamp(group["date"].max())
        tickers = group["ticker"].tolist()
        universe.dates.append(rebalance_date)
        universe.members[rebalance_date] = tickers
        if mode == "sticky_measured":
            for ticker, spread in zip(group["ticker"], group["last_measured"],
                                      strict=True):
                universe.spread[(rebalance_date, ticker)] = float(spread)
                universe.n_measured += 1
            universe.sectors[rebalance_date] = _sector_buckets(tickers, sector_of)
            continue
        for ticker, spread, regime in zip(
            group["ticker"], group["spread"], group["spread_regime"], strict=True
        ):
            if regime == "measured":
                universe.spread[(rebalance_date, ticker)] = float(spread)
                universe.n_measured += 1
            else:
                universe.spread[(rebalance_date, ticker)] = LIQUID_SPREAD_SUBSTITUTE
                universe.n_upper_bound_costed += 1

        universe.sectors[rebalance_date] = _sector_buckets(tickers, sector_of)

    universe.dates.sort()
    return universe


def _sector_buckets(tickers: list[str], sector_of: dict[str, str]
                    ) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for ticker in tickers:
        sector = sector_of.get(ticker)
        if sector:
            buckets.setdefault(sector, []).append(ticker)
    return {
        name: members
        for name, members in buckets.items()
        if len(members) >= MIN_SECTOR_CONSTITUENTS
    }


def monthly_retention(universe: Universe) -> float:
    """Mean fraction of members surviving into the next month.

    Reported because erratum 1 turns on it: a universe that churns half its names every
    month pays enormous cost for no economic reason.
    """
    retained = []
    for earlier, later in zip(universe.dates[:-1], universe.dates[1:], strict=True):
        before = set(universe.members[earlier])
        after = set(universe.members[later])
        if before:
            retained.append(len(before & after) / len(before))
    return float(np.mean(retained)) if retained else float("nan")


# ---------------------------------------------------------------------------
# Daily matrix with delisting outcomes attached once
# ---------------------------------------------------------------------------
@dataclass
class DailyMatrix:
    dates: pd.DatetimeIndex
    tickers: list[str]
    index_of: dict[str, int]
    returns: np.ndarray  # (D, N), NaN where the name has no bar
    close: np.ndarray  # (D, N), forward-filled, used only for share counts
    last_bar: np.ndarray
    first_bar: np.ndarray
    terminal: np.ndarray
    has_terminal: np.ndarray
    n_returns_capped: int = 0
    n_terminal_names: int = 0


def build_daily_matrix(
    prices: pd.DataFrame,
    tickers: set[str],
    delistings: pd.DataFrame,
) -> DailyMatrix:
    """Daily total returns per name with delisting outcomes attached, once each.

    The terminal return is attached to the session AFTER the final bar and only when the
    delisting event lands within `DELISTING_WINDOW_DAYS` of that final bar. A name whose
    bars stop for any other reason (band change, spread stops resolving, data gap) exits
    flat. The backtester zeroes the position on the terminal day, so nothing can be
    booked twice.
    """
    subset = prices[prices["ticker"].isin(tickers)]
    subset = subset.sort_values(["ticker", "date"])

    adjusted = subset.pivot(index="date", columns="ticker", values="closeadj").sort_index()
    raw = subset.pivot(index="date", columns="ticker", values="close")
    raw = raw.reindex(index=adjusted.index, columns=adjusted.columns).ffill()

    values = adjusted.to_numpy(dtype=float)
    returns = np.full_like(values, np.nan)
    returns[1:] = values[1:] / values[:-1] - 1.0
    n_capped = int((np.isfinite(returns) & (np.abs(returns) > DAILY_RETURN_CAP)).sum())
    returns = np.clip(returns, -DAILY_RETURN_CAP, DAILY_RETURN_CAP)

    valid = np.isfinite(values)
    n_dates, n_names = values.shape
    rows = np.arange(n_dates)[:, None]
    last_bar = np.where(valid, rows, -1).max(axis=0)
    first_bar = np.where(valid, rows, n_dates).min(axis=0)

    terminal = np.zeros(n_names)
    has_terminal = np.zeros(n_names, dtype=bool)
    events = (
        delistings.sort_values("date").drop_duplicates("ticker", keep="first")
        .set_index("ticker")
    )
    dates = adjusted.index
    for position, ticker in enumerate(adjusted.columns):
        if ticker not in events.index or last_bar[position] < 0:
            continue
        row = events.loc[ticker]
        delta = (pd.Timestamp(row["date"]) - dates[last_bar[position]]).days
        # Rule 4: only a delisting that actually happened around THIS exit counts. A
        # 2012 bankruptcy must never be charged against a 2003 exit.
        if -5 <= delta <= DELISTING_WINDOW_DAYS:
            terminal[position] = float(row["terminal_return"])
            has_terminal[position] = True

    return DailyMatrix(
        dates=dates,
        tickers=list(adjusted.columns),
        index_of={t: i for i, t in enumerate(adjusted.columns)},
        returns=returns,
        close=raw.to_numpy(dtype=float),
        last_bar=last_bar,
        first_bar=first_bar,
        terminal=terminal,
        has_terminal=has_terminal,
        n_returns_capped=n_capped,
        n_terminal_names=int(has_terminal.sum()),
    )


# ---------------------------------------------------------------------------
# Book planning: signals, sizing, book volatility. Independent of the vol target,
# so it is computed once and reused across the three targets and the gross/net pair.
# ---------------------------------------------------------------------------
@dataclass
class Rebalance:
    day: int
    date: pd.Timestamp
    base_weights: np.ndarray  # sum|w| == 1 (or == 1 long-only for the benchmark)
    book_vol: float
    spreads: np.ndarray  # per-stock full spread known at this date, NaN if unknown
    n_instruments: int


@dataclass
class BookPlan:
    rebalances: list[Rebalance]
    flips_per_year: dict[int, float]
    mean_instruments: float
    mean_singles: float
    mean_baskets: float


def _basket_series(matrix: DailyMatrix, legs: np.ndarray, end: int,
                   window: int) -> np.ndarray:
    """Equal-weight daily returns of `legs` over the trailing `window` ending at `end`.

    Only bars at or before `end` are touched, so this is point-in-time safe.
    """
    start = max(0, end - window + 1)
    block = matrix.returns[start : end + 1, legs]
    mask = np.isfinite(block)
    counts = mask.sum(axis=1)
    totals = np.where(mask, block, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def plan_book(
    matrix: DailyMatrix,
    universe: Universe,
    *,
    long_only_equal_weight: bool = False,
) -> BookPlan:
    """Build the point-in-time weight path once.

    `long_only_equal_weight=True` produces the benchmark: equal weights over the same
    single-name universe, no signal, no shorts, no baskets.
    """
    date_position = {date: i for i, date in enumerate(matrix.dates)}
    n_names = len(matrix.tickers)
    rebalances: list[Rebalance] = []
    flip_counts = {lookback: 0 for lookback in LOOKBACKS}
    flip_slots = {lookback: 0 for lookback in LOOKBACKS}
    previous_signal: dict[int, dict[str, float]] = {lb: {} for lb in LOOKBACKS}
    singles_seen: list[int] = []
    baskets_seen: list[int] = []
    last_spread = np.full(n_names, np.nan)

    for rebalance_date in universe.dates:
        day = date_position.get(rebalance_date)
        if day is None:
            continue
        members = universe.members[rebalance_date]

        for ticker in members:
            position = matrix.index_of.get(ticker)
            if position is not None:
                last_spread[position] = universe.spread[(rebalance_date, ticker)]

        # Instruments = eligible single names + sector baskets of those names.
        instruments: list[tuple[str, np.ndarray]] = []
        eligible: dict[str, int] = {}
        for ticker in members:
            position = matrix.index_of.get(ticker)
            if position is None or matrix.last_bar[position] < day:
                continue
            if day - matrix.first_bar[position] < MIN_HISTORY:
                continue
            eligible[ticker] = position
            instruments.append((ticker, np.array([position])))
        n_singles = len(instruments)

        if not long_only_equal_weight:
            for sector, tickers in universe.sectors[rebalance_date].items():
                legs = np.array([eligible[t] for t in tickers if t in eligible], dtype=int)
                if legs.size >= MIN_SECTOR_CONSTITUENTS:
                    instruments.append((f"SECTOR::{sector}", legs))
        n_baskets = len(instruments) - n_singles

        if not instruments:
            continue
        singles_seen.append(n_singles)
        baskets_seen.append(n_baskets)

        base = np.zeros(n_names)
        if long_only_equal_weight:
            share = 1.0 / n_singles
            for _name, legs in instruments:
                base[legs[0]] = share
            book_vol = float("nan")
        else:
            raw: list[tuple[np.ndarray, float]] = []
            gross_sum = 0.0
            for name, legs in instruments:
                signal_values = []
                for lookback in LOOKBACKS:
                    if day - lookback < 0:
                        signal_values.append(0.0)
                        continue
                    series = _basket_series(matrix, legs, day, lookback)
                    if np.isnan(series).any():
                        value = 0.0
                    else:
                        value = float(np.sign(np.prod(1.0 + series) - 1.0))
                    signal_values.append(value)
                    previous = previous_signal[lookback].get(name)
                    if previous is not None:
                        flip_slots[lookback] += 1
                        if value != previous:
                            flip_counts[lookback] += 1
                    previous_signal[lookback][name] = value

                signal = float(np.mean(signal_values))
                if signal == 0.0:
                    continue
                vol_series = _basket_series(matrix, legs, day, VOL_WINDOW)
                vol_series = vol_series[np.isfinite(vol_series)]
                if vol_series.size < VOL_WINDOW // 2:
                    continue
                vol = float(np.std(vol_series, ddof=1)) * np.sqrt(TRADING_DAYS)
                vol = float(np.clip(vol, INST_VOL_FLOOR, INST_VOL_CAP))
                value = signal / vol
                raw.append((legs, value))
                gross_sum += abs(value)

            if gross_sum <= 0.0:
                continue
            for legs, value in raw:
                base[legs] += (value / gross_sum) / legs.size

            # Point-in-time book volatility: today's weights on the trailing 126
            # sessions. Never touches a future bar.
            start = max(0, day - BOOK_VOL_WINDOW + 1)
            history = matrix.returns[start : day + 1]
            history = np.where(np.isfinite(history), history, 0.0)
            path = history @ base
            book_vol = float(np.std(path, ddof=1)) * np.sqrt(TRADING_DAYS)

        rebalances.append(
            Rebalance(
                day=day,
                date=rebalance_date,
                base_weights=base,
                book_vol=book_vol,
                spreads=last_spread.copy(),
                n_instruments=len(instruments),
            )
        )

    flips = {
        lookback: (flip_counts[lookback] / max(flip_slots[lookback], 1)) * 12.0
        for lookback in LOOKBACKS
    }
    return BookPlan(
        rebalances=rebalances,
        flips_per_year=flips,
        mean_instruments=float(np.mean([r.n_instruments for r in rebalances]))
        if rebalances else 0.0,
        mean_singles=float(np.mean(singles_seen)) if singles_seen else 0.0,
        mean_baskets=float(np.mean(baskets_seen)) if baskets_seen else 0.0,
    )


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def _commission(trade_notional: np.ndarray, price: np.ndarray) -> np.ndarray:
    """IBKR-style: $0.0035/share, $0.35 order minimum, capped at 1% of trade value."""
    safe_price = np.where(np.isfinite(price) & (price > 0.0), price, np.nan)
    shares = trade_notional / safe_price
    raw = np.maximum(COMMISSION_PER_SHARE * shares, COMMISSION_MIN_ORDER)
    capped = np.minimum(raw, COMMISSION_CAP_FRACTION * trade_notional)
    return np.where(np.isfinite(capped), capped, COMMISSION_CAP_FRACTION * trade_notional)


def execution_cost(
    trade: np.ndarray, spreads: np.ndarray, prices: np.ndarray
) -> float:
    """Spread plus commission on the NET executed trade in each stock.

    Netting at the stock level is what an execution desk actually does; costing each
    instrument (single name AND the sector basket containing it) separately would
    double-charge a stock that appears in both.
    """
    traded = np.abs(trade)
    active = traded > 0.0
    if not active.any():
        return 0.0
    notional = traded[active]
    spread = spreads[active]
    # An unknown spread means the name left the measurable universe. It still has to be
    # sold, and pricing that exit at zero would be a free lunch, so the last liquid-name
    # schedule is used as a floor rather than dropping the cost.
    spread = np.where(np.isfinite(spread), spread, LIQUID_SPREAD_SUBSTITUTE)
    cost = 0.5 * spread * notional
    cost = cost + _commission(notional, prices[active])
    return float(cost.sum())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    dates: pd.DatetimeIndex
    returns: np.ndarray
    turnover_per_year: float
    mean_leverage: float
    mean_gross_exposure: float
    mean_net_exposure: float
    n_terminal_booked: int
    total_costs: float


def simulate(
    matrix: DailyMatrix,
    plan: BookPlan,
    target_vol: float | None,
    *,
    charge_costs: bool,
    starting_equity: float = STARTING_EQUITY,
) -> SimResult:
    """Daily-marked book, monthly rebalanced, positions drifting in between.

    `target_vol=None` means take the plan's weights as they are (the benchmark case).
    Running with `charge_costs=False` reproduces the identical weight path, because
    weights are fractions of equity, which is what makes cost drag a clean difference.
    """
    n_dates = len(matrix.dates)
    n_names = len(matrix.tickers)
    by_day = {rebalance.day: rebalance for rebalance in plan.rebalances}

    positions: np.ndarray = np.zeros(n_names)
    cash = starting_equity
    equity = starting_equity
    booked = np.zeros(n_names, dtype=bool)
    last_spread: np.ndarray = np.full(n_names, np.nan)

    returns = np.full(n_dates, np.nan)
    traded_total = 0.0
    cost_total = 0.0
    leverages: list[float] = []
    gross_exposures: list[float] = []
    net_exposures: list[float] = []
    n_terminal = 0
    started = False

    for day in range(n_dates):
        previous_equity = equity

        step = matrix.returns[day]
        step = np.where(np.isfinite(step), step, 0.0)
        held = positions != 0.0
        if held.any():
            positions[held] *= 1.0 + step[held]

            # Force-exit names whose bars stopped. Terminal return booked ONCE, then the
            # name is removed from the book permanently.
            dead = np.flatnonzero(held & (matrix.last_bar < day))
            if dead.size:
                for position in dead:
                    if matrix.has_terminal[position] and not booked[position]:
                        positions[position] *= 1.0 + matrix.terminal[position]
                        booked[position] = True
                        n_terminal += 1
                exit_trade = np.zeros(n_names)
                exit_trade[dead] = -positions[dead]
                book_value = cash + float(positions.sum())
                if charge_costs:
                    cost = execution_cost(exit_trade, last_spread, matrix.close[day])
                    cash -= cost
                    if book_value > 0.0:
                        cost_total += cost / book_value
                # Turnover is measured against CONTEMPORANEOUS equity. Accumulating raw
                # dollars against a book whose equity is changing makes the ratio
                # meaningless -- a losing book would report falling turnover.
                if book_value > 0.0:
                    traded_total += float(np.abs(exit_trade).sum()) / book_value
                cash += float(positions[dead].sum())
                positions[dead] = 0.0

        equity = cash + float(positions.sum())
        if charge_costs and started:
            long_notional = float(positions[positions > 0].sum())
            short_notional = float(-positions[positions < 0].sum())
            carry = (
                short_notional * BORROW_RATE
                + max(0.0, long_notional - equity) * FINANCING_RATE
            ) / TRADING_DAYS
            cash -= carry
            if equity > 0.0:
                cost_total += carry / equity
            equity = cash + float(positions.sum())

        rebalance = by_day.get(day)
        if rebalance is not None and equity > 0.0:
            last_spread = np.where(
                np.isfinite(rebalance.spreads), rebalance.spreads, last_spread
            )
            if target_vol is None:
                weights = rebalance.base_weights
                leverage = 1.0
            elif not np.isfinite(rebalance.book_vol) or rebalance.book_vol <= 0.0:
                weights = np.zeros(n_names)
                leverage = 0.0
            else:
                leverage = min(target_vol / rebalance.book_vol, MAX_GROSS_LEVERAGE)
                weights = rebalance.base_weights * leverage
            leverages.append(leverage)

            target_dollars = weights * equity
            trade = target_dollars - positions
            if charge_costs:
                cost = execution_cost(trade, last_spread, matrix.close[day])
                cash -= cost
                cost_total += cost / equity
                equity -= cost
                target_dollars = weights * equity
                trade = target_dollars - positions
            traded_total += float(np.abs(trade).sum()) / equity
            positions = target_dollars
            cash = equity - float(positions.sum())
            gross_exposures.append(float(np.abs(positions).sum()) / equity)
            net_exposures.append(float(positions.sum()) / equity)
            started = True

        equity = cash + float(positions.sum())
        if started and previous_equity > 0.0:
            returns[day] = equity / previous_equity - 1.0
        if equity <= 0.0:
            # A wiped-out book stops trading rather than compounding into nonsense.
            positions[:] = 0.0
            cash = 0.0
            equity = 0.0

    years = n_dates / TRADING_DAYS
    return SimResult(
        dates=matrix.dates,
        returns=returns,
        turnover_per_year=traded_total / years,
        mean_leverage=float(np.mean(leverages)) if leverages else 1.0,
        mean_gross_exposure=float(np.mean(gross_exposures)) if gross_exposures else 0.0,
        mean_net_exposure=float(np.mean(net_exposures)) if net_exposures else 0.0,
        n_terminal_booked=n_terminal,
        total_costs=cost_total,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def annualised(returns: np.ndarray) -> tuple[float, float, float]:
    """(CAGR, annualised volatility, annualised Sharpe) on a daily return series."""
    clean = returns[np.isfinite(returns)]
    if clean.size < 2:
        return float("nan"), float("nan"), float("nan")
    equity = float(np.prod(1.0 + clean))
    years = clean.size / TRADING_DAYS
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0.0 else -1.0
    vol = float(np.std(clean, ddof=1)) * np.sqrt(TRADING_DAYS)
    sharpe = float(np.mean(clean)) * TRADING_DAYS / vol if vol > 0.0 else float("nan")
    return cagr, vol, sharpe


def max_drawdown(returns: np.ndarray) -> float:
    clean = np.where(np.isfinite(returns), returns, 0.0)
    equity = np.cumprod(1.0 + clean)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def effective_instruments(matrix: DailyMatrix, universe: Universe) -> float:
    """Participation ratio of the universe's daily return correlation matrix.

    (sum lambda)^2 / sum(lambda^2). This is how many INDEPENDENT instruments the book
    actually holds; for US single names it is far below the nominal count, which is the
    whole question the breadth lever turns on. Computed on one rebalance per calendar
    year and averaged.
    """
    ratios: list[float] = []
    date_position = {date: i for i, date in enumerate(matrix.dates)}
    by_year: dict[int, pd.Timestamp] = {}
    for rebalance_date in universe.dates:
        by_year.setdefault(rebalance_date.year, rebalance_date)
    for _year, rebalance_date in by_year.items():
        end = date_position.get(rebalance_date)
        if end is None:
            continue
        start = max(0, end - TRADING_DAYS + 1)
        positions = [
            matrix.index_of[t]
            for t in universe.members[rebalance_date]
            if t in matrix.index_of
        ]
        block = matrix.returns[start : end + 1, positions]
        block = block[:, np.isfinite(block).all(axis=0)]
        if block.shape[1] < 10 or block.shape[0] < 60:
            continue
        correlation = np.nan_to_num(np.corrcoef(block, rowvar=False), nan=0.0)
        eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
        total = eigenvalues.sum()
        if total > 0.0:
            ratios.append(float(total**2 / float(np.sum(eigenvalues**2))))
    return float(np.mean(ratios)) if ratios else float("nan")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Monthly panel, delistings and sector labels -- DEV window only."""
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    if pd.Timestamp(panel["date"].max()) > DEV_CUTOFF:
        raise ValueError("monthly panel contains bars past the DEV cutoff")
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    tickers = pd.read_csv(
        Path(DATA_DIR) / "TICKERS.csv",
        usecols=["table", "ticker", "sector"],
        low_memory=False,
    )
    tickers = tickers[tickers["table"] == "SEP"].dropna(subset=["sector"])
    tickers = tickers.drop_duplicates("ticker", keep="first")
    sector_of = dict(zip(tickers["ticker"], tickers["sector"], strict=True))
    return panel, delistings, sector_of
