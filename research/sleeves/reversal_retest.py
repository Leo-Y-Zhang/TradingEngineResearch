"""Sleeve: short-horizon reversal RE-TESTED on the corrected, liquidity-first universe.

Registered design: `research/sleeves/reversal_retest_prereg.md`, written and frozen
BEFORE this module was run. Nothing here was chosen by looking at a result.

What changed from iteration 1 (`short_horizon_reversal.py`, which is left untouched as
the historical record):

1. **The universe INCLUDES `upper_bound` names.** Iteration 1 deleted them. They are the
   CHEAP half of the tape -- `upper_bound` means the true spread lies *below* the
   estimate -- and deleting them removed 525,933 of 922,652 eligible (name, month) cells
   at 6.4x the dollar volume and 0.24x the spread of the cells that were kept.
2. **The universe is LIQUIDITY-FIRST**: the top decile (PRIMARY) or top quintile
   (SECONDARY) by trailing 63-day median dollar volume, ranked cross-sectionally WITHIN
   each month so the cut is not a secular time series of the market's growth.
3. **Every number is produced under BOTH cost bounds** (`spread_estimation.
   bounds_from_estimate`), never one.
4. **Three rebalance frequencies** are run on an identical signal, because cost scales
   linearly with frequency and gross Sharpe scales with sqrt(frequency), so the shape of
   that curve is the deliverable even when every point on it is negative.

Everything else -- execution at the next open, drifted-weight turnover, the 62-day
delisting rule with removal from the book, the +/-100% return cap, the $2 price floor,
the forward-filled cost basis so exits are charged, the own-universe benchmark -- is
carried over unchanged, because those are the accounting defects the programme has
already paid for.
"""

from __future__ import annotations

import datetime as _dt
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research.spread_estimation import (
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
    AGK_LIQUIDITY_ANCHOR_SPREAD,
    TICK_REGIMES,
    bounds_from_estimate,
    era_multiplier,
)

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO / "_data" / "sharadar" / "panel"
DEV_CUTOFF = pd.Timestamp("2015-12-31")

# Registered universe cuts: rank percentile of median dollar volume within the month.
UNIVERSE_CUTS: dict[str, float] = {"top_decile": 0.90, "top_quintile": 0.80}

# Registered rebalance grids. Nominal frequency only -- annualisation uses the REALISED
# periods per year so a ragged grid cannot inflate a Sharpe.
FREQUENCIES: tuple[str, ...] = ("weekly", "fortnightly", "monthly")

# Registered cost treatments. `zero_cost` is the diagnostic ceiling of prereg 9.7: it is
# physically impossible and is never a headline.
COST_TREATMENTS: tuple[str, ...] = ("conservative", "realistic", "zero_cost")


@dataclass(frozen=True)
class RetestConfig:
    """The registered configuration. Frozen before the run; nothing here is fitted."""

    # --- universe -------------------------------------------------------------
    min_price: float = 2.00
    # Borrow is only plausible where there is a real securities-lending market.
    shortable_dollar_volume: float = 2.5e7
    allowed_spread_regimes: tuple[str, ...] = ("measured", "upper_bound")
    min_trading_fraction: float = 0.90

    # --- signal ---------------------------------------------------------------
    lookback_days: int = 5

    # --- construction ---------------------------------------------------------
    decile: float = 0.10
    max_names_per_leg: int = 100
    min_names_per_leg: int = 20
    min_shortable_names: int = 60
    long_leg_exposure: float = 1.00
    short_leg_exposure: float = 1.00

    # --- costs ----------------------------------------------------------------
    equity: float = 1_000_000.0
    commission_per_share: float = 0.0035
    commission_min_per_order: float = 0.35
    commission_cap_fraction: float = 0.01
    impact_coefficient: float = 1.0
    impact_vol_window: int = 21
    short_borrow_annual: float = 0.01

    # --- accounting -----------------------------------------------------------
    return_cap: float = 1.00
    delisting_grace_days: int = 62
    cost_ladder: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


# ---------------------------------------------------------------------------
# Vectorised two-bound spread pricing
# ---------------------------------------------------------------------------


def _tick_size_vec(dates: pd.Series) -> np.ndarray:
    """Minimum quoting increment in force on each date. Mirrors `spread_estimation.tick_size`."""
    out: np.ndarray = np.full(len(dates), TICK_REGIMES[0][1])
    stamps = pd.DatetimeIndex(dates)
    for start, value in TICK_REGIMES:
        # The first regime's sentinel start (year 1) is outside pandas' ns range and is
        # already the initial fill, so it is skipped rather than compared.
        if start < _dt.date(1700, 1, 1):
            continue
        out = np.where(stamps >= pd.Timestamp(start), value, out)
    return out


def _era_multiplier_vec(dates: pd.Series) -> np.ndarray:
    """Era factor per date. Looked up once per distinct YEAR, not per row."""
    years = pd.DatetimeIndex(dates).year.to_numpy()
    lookup = {
        int(y): era_multiplier(_dt.date(int(y), 7, 1)) for y in np.unique(years)
    }
    return np.array([lookup[int(y)] for y in years])


def spread_bounds_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add `spread_conservative` / `spread_realistic` columns to an eligible panel.

    A vectorised reimplementation of `spread_estimation.bounds_from_estimate` -- there
    are ~185,000 cells and the scalar function builds a frozen dataclass per call. It is
    asserted cell-for-cell against the reference function by `verify_bounds_vectorisation`
    BEFORE any return is computed, and the run aborts on a mismatch.
    """
    estimate = monthly["spread"].to_numpy(dtype=float)
    price = monthly["close"].to_numpy(dtype=float)
    volume = monthly["median_dollar_volume"].to_numpy(dtype=float)
    regime = monthly["spread_regime"].to_numpy()

    tick = _tick_size_vec(monthly["date"])
    with np.errstate(invalid="ignore", divide="ignore"):
        tick_floor = np.where(np.isfinite(price) & (price > 0.0), tick / price, np.nan)
    floor = np.where(np.isfinite(tick_floor), tick_floor, 0.0)

    conservative = np.maximum(estimate, floor)

    # The documented liquid-name schedule: log-interpolated AGK Table 4 Panel C quintile
    # medians, CLAMPED at both ends (np.interp does not extrapolate), era-scaled, then
    # tick-floored.
    with np.errstate(invalid="ignore", divide="ignore"):
        log_volume = np.where(np.isfinite(volume) & (volume > 0.0), np.log(volume), np.nan)
    base = np.exp(np.interp(
        log_volume,
        np.log(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME),
        np.log(AGK_LIQUIDITY_ANCHOR_SPREAD),
    ))
    schedule = base * _era_multiplier_vec(monthly["date"])
    schedule = np.where(np.isfinite(tick_floor), np.maximum(schedule, tick_floor), schedule)
    schedule = np.where(np.isfinite(log_volume), schedule, np.nan)

    is_upper = regime == "upper_bound"
    have_schedule = is_upper & np.isfinite(schedule)
    realistic = np.where(
        have_schedule,
        np.maximum(np.minimum(estimate, schedule), floor),
        conservative,
    )

    dead = (regime == "unmeasurable") | ~np.isfinite(estimate)
    conservative = np.where(dead, np.nan, conservative)
    realistic = np.where(dead, np.nan, realistic)

    out = monthly.copy()
    out["spread_conservative"] = conservative
    out["spread_realistic"] = realistic
    return out


def verify_bounds_vectorisation(frame: pd.DataFrame, n: int = 4000, seed: int = 20260728) -> int:
    """Assert the vectorised bounds equal the reference scalar function. Raises on mismatch."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
    sample = frame.iloc[idx]
    for row in sample.itertuples(index=False):
        ref = bounds_from_estimate(
            row.spread, row.spread_regime, row.median_dollar_volume,
            price=row.close, when=row.date,
        )
        for name, got in (("conservative", row.spread_conservative),
                          ("realistic", row.spread_realistic)):
            want = getattr(ref, name)
            if np.isnan(want) and np.isnan(got):
                continue
            if not np.isclose(want, got, rtol=1e-12, atol=1e-15):
                raise AssertionError(
                    f"vectorised {name} bound disagrees with the reference: "
                    f"{row.ticker} {row.date} got {got!r} want {want!r}"
                )
        if np.isfinite(row.spread_realistic) and np.isfinite(row.spread_conservative):
            if row.spread_realistic > row.spread_conservative + 1e-15:
                raise AssertionError("bounds inverted: realistic exceeds conservative")
    return len(sample)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class PanelMatrices:
    """Wide, calendar-aligned matrices. Rows are trading days, columns are tickers."""

    dates: pd.DatetimeIndex
    tickers: np.ndarray
    adj_open: np.ndarray
    adj_close: np.ndarray
    raw_open: np.ndarray
    months: pd.PeriodIndex
    dollar_volume: np.ndarray        # [month, ticker], NaN where not eligible
    dv_rank: np.ndarray              # [month, ticker] percentile rank within the month
    # Cost basis: forward-filled over ELIGIBLE months so an exit is always priced. The
    # eligibility decision never reads these -- a name that left the universe must still
    # be sold, and reading the current month's NaN there would hand it a free liquidation.
    spread_basis: dict[str, np.ndarray]
    dv_basis: np.ndarray
    delist_date: np.ndarray
    delist_return: np.ndarray


def _eligible_monthly(monthly: pd.DataFrame, config: RetestConfig) -> pd.DataFrame:
    """Cells this study is allowed to see, before the liquidity-rank restriction."""
    mask = (
        monthly["spread_regime"].isin(config.allowed_spread_regimes)
        & (monthly["median_dollar_volume"] > 0.0)
        & (monthly["close"] >= config.min_price)
        & (monthly["trading_fraction"] >= config.min_trading_fraction)
        & np.isfinite(monthly["spread"])
    )
    return monthly.loc[mask].copy()


def build_matrices(config: RetestConfig) -> PanelMatrices:
    """Read the panel once and build every matrix both universe cuts need.

    The ticker union is every name that reaches the WIDER cut (top quintile) at least
    once. The narrower cut is a subset of it, so one read serves both. Cost-basis
    forward-fill uses each of those names' full ELIGIBLE history, including months when
    it sat outside the quintile -- that is the accurate cost, and it is only ever used to
    price a trade, never to admit a name.
    """
    monthly = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    eligible = _eligible_monthly(monthly, config)
    del monthly

    month_period = eligible["date"].dt.to_period("M")
    eligible["dv_rank"] = (
        eligible.groupby(month_period)["median_dollar_volume"].rank(pct=True)
    )
    eligible = spread_bounds_frame(eligible)
    checked = verify_bounds_vectorisation(eligible)
    logger.info("bounds vectorisation verified against the reference on %d cells", checked)

    widest = min(UNIVERSE_CUTS.values())
    union_tickers = sorted(set(eligible.loc[eligible["dv_rank"] > widest, "ticker"]))
    logger.info("union universe: %d tickers", len(union_tickers))
    ticker_set = set(union_tickers)
    eligible = eligible[eligible["ticker"].isin(ticker_set)]
    month_period = eligible["date"].dt.to_period("M")

    prices = pd.read_parquet(
        PANEL_DIR / "prices_to_2015-12-31.parquet",
        columns=["ticker", "date", "open", "close", "closeadj"],
    )
    prices = prices[prices["ticker"].isin(ticker_set)]
    if prices["date"].max() > DEV_CUTOFF:
        raise RuntimeError("price panel contains bars past the DEV cutoff")

    factor = prices["closeadj"].to_numpy() / prices["close"].to_numpy()
    prices["adj_open"] = prices["open"].to_numpy() * factor

    dates = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    date_pos = pd.Series(np.arange(len(dates)), index=dates)
    tick_pos = pd.Series(np.arange(len(union_tickers)), index=union_tickers)

    rows = date_pos.reindex(prices["date"]).to_numpy()
    cols = tick_pos.reindex(prices["ticker"]).to_numpy()
    shape = (len(dates), len(union_tickers))

    def _fill(values: np.ndarray, dtype: type) -> np.ndarray:
        out: np.ndarray = np.full(shape, np.nan, dtype=dtype)
        out[rows, cols] = values
        return out

    adj_open = _fill(prices["adj_open"].to_numpy(), np.float64)
    adj_close = _fill(prices["closeadj"].to_numpy(), np.float64)
    raw_open = _fill(prices["open"].to_numpy(), np.float32)
    del prices

    delist = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    delist = delist.drop_duplicates("ticker", keep="first").set_index("ticker")
    delist = delist.reindex(union_tickers)
    delist_date = delist["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    delist_date = np.where(delist["date"].isna().to_numpy(), np.iinfo(np.int64).min,
                           delist_date)
    delist_return = delist["terminal_return"].to_numpy(dtype=float)

    # Rows are keyed on the calendar MONTH: the panel's "month end" is each ticker's own
    # last trading day of the month, so it is not a shared calendar and indexing on the
    # raw date would line up almost no tickers with each other.
    months = pd.PeriodIndex(np.sort(month_period.unique()))
    month_pos = pd.Series(np.arange(len(months)), index=months)
    m_rows = month_pos.reindex(month_period).to_numpy()
    m_cols = tick_pos.reindex(eligible["ticker"]).to_numpy()
    m_shape = (len(months), len(union_tickers))

    def _monthly(values: np.ndarray) -> np.ndarray:
        out = np.full(m_shape, np.nan)
        out[m_rows, m_cols] = values
        return out

    dollar_volume = _monthly(eligible["median_dollar_volume"].to_numpy())
    dv_rank = _monthly(eligible["dv_rank"].to_numpy())
    spread_basis = {
        "conservative": pd.DataFrame(
            _monthly(eligible["spread_conservative"].to_numpy())).ffill().to_numpy(),
        "realistic": pd.DataFrame(
            _monthly(eligible["spread_realistic"].to_numpy())).ffill().to_numpy(),
    }
    spread_basis["zero_cost"] = np.zeros_like(spread_basis["realistic"])
    dv_basis = pd.DataFrame(dollar_volume).ffill().to_numpy()

    return PanelMatrices(
        dates=dates,
        tickers=np.asarray(union_tickers, dtype=object),
        adj_open=adj_open,
        adj_close=adj_close,
        raw_open=raw_open,
        months=months,
        dollar_volume=dollar_volume,
        dv_rank=dv_rank,
        spread_basis=spread_basis,
        dv_basis=dv_basis,
        delist_date=delist_date,
        delist_return=delist_return,
    )


# ---------------------------------------------------------------------------
# Rebalance grids
# ---------------------------------------------------------------------------


def rebalance_grid(dates: pd.DatetimeIndex, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Signal-date and execution-date row indices for a registered rebalance grid.

    Execution is always the NEXT trading day's open, never the close the signal came from.
    """
    if kind in ("weekly", "fortnightly"):
        period = dates.to_period("W")
    elif kind == "monthly":
        period = dates.to_period("M")
    else:
        raise ValueError(f"unregistered rebalance grid: {kind}")

    last = pd.Series(np.arange(len(dates)), index=period).groupby(level=0).max()
    signal_idx = np.sort(last.to_numpy())
    if kind == "fortnightly":
        signal_idx = signal_idx[::2]

    exec_idx = signal_idx + 1
    keep = exec_idx < len(dates)
    return signal_idx[keep], exec_idx[keep]


def month_row_for(panel: PanelMatrices, signal_idx: np.ndarray) -> np.ndarray:
    """Row of the monthly panel demonstrably complete at each signal date: the PREVIOUS month."""
    signal_months = panel.dates[signal_idx].to_period("M")
    lookup = pd.Series(np.arange(len(panel.months)), index=panel.months)
    return lookup.reindex(signal_months - 1).to_numpy(dtype=float, na_value=-1.0).astype(int)


# ---------------------------------------------------------------------------
# Backtest primitives
# ---------------------------------------------------------------------------


@dataclass
class BookResult:
    dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    gross_return: np.ndarray = field(default_factory=lambda: np.array([]))
    cost: np.ndarray = field(default_factory=lambda: np.array([]))
    turnover: np.ndarray = field(default_factory=lambda: np.array([]))
    n_names: np.ndarray = field(default_factory=lambda: np.array([]))
    spread_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    impact_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    commission_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    borrow_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    # P&L attribution: signed gross contribution per ticker, and the single largest
    # (ticker, period) contribution seen. Prereg 9.5.
    pnl_by_ticker: np.ndarray = field(default_factory=lambda: np.array([]))
    max_cell_pnl: float = 0.0
    max_cell_label: str = ""


def _as_ns(stamp: pd.Timestamp) -> np.int64:
    """Nanoseconds since the epoch, whatever resolution the timestamp happens to carry."""
    return stamp.to_datetime64().astype("datetime64[ns]").astype(np.int64)


def holding_returns(
    panel: PanelMatrices,
    entry_row: int,
    exit_row: int,
    config: RetestConfig,
) -> np.ndarray:
    """Adjusted open-to-open return per ticker over one holding period.

    Delisting terminal returns are booked ONLY when the delisting date lies inside the
    window extended by the grace period AND the name actually stopped printing prices.
    """
    entry = panel.adj_open[entry_row]
    exit_price = panel.adj_open[exit_row].copy()

    missing = ~np.isfinite(exit_price) & np.isfinite(entry)
    if missing.any():
        window = panel.adj_close[entry_row: exit_row + 1, missing]
        valid = np.isfinite(window)
        has_any = valid.any(axis=0)
        last_row = np.where(has_any, valid.shape[0] - 1 - valid[::-1].argmax(axis=0), 0)
        last_close = np.where(has_any, window[last_row, np.arange(window.shape[1])], np.nan)
        exit_price[missing] = last_close

    with np.errstate(invalid="ignore", divide="ignore"):
        returns = exit_price / entry - 1.0
    stalled = ~np.isfinite(returns) & np.isfinite(entry)
    returns = np.where(stalled, 0.0, returns)

    grace = np.timedelta64(config.delisting_grace_days, "D").astype("timedelta64[ns]")
    # `Timestamp.to_datetime64()` returns whatever RESOLUTION the timestamp carries, and
    # `delist_date` is nanoseconds. A seconds-resolution index would silently make every
    # comparison here false and book no delisting at all -- a survivorship flatter that
    # would not raise. The cast is explicit so the units cannot drift apart.
    window_start = int(_as_ns(panel.dates[entry_row]))
    window_end = int(_as_ns(panel.dates[exit_row])) + int(grace)
    in_window = (panel.delist_date >= window_start) & (panel.delist_date <= window_end)
    booked = in_window & (missing | stalled)
    if booked.any():
        terminal = panel.delist_return[booked]
        returns[booked] = (1.0 + returns[booked]) * (1.0 + terminal) - 1.0

    return np.clip(returns, -config.return_cap, config.return_cap)


def trailing_vol(panel: PanelMatrices, row: int, window: int) -> np.ndarray:
    lo = max(0, row - window)
    block = panel.adj_close[lo: row + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_returns = np.diff(np.log(block), axis=0)
    if log_returns.shape[0] < 5:
        return np.full(panel.adj_close.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(log_returns, axis=0, ddof=1)


@dataclass
class PeriodData:
    """Per-rebalance quantities that do not depend on which leg is being run."""

    volatility: list[np.ndarray]
    returns: list[np.ndarray]


def precompute_periods(
    panel: PanelMatrices,
    config: RetestConfig,
    signal_idx: np.ndarray,
    exec_idx: np.ndarray,
) -> PeriodData:
    vols, rets = [], []
    last_row = panel.adj_close.shape[0] - 1
    for k, (s_row, e_row) in enumerate(zip(signal_idx, exec_idx)):
        vols.append(trailing_vol(panel, s_row, config.impact_vol_window))
        exit_row = exec_idx[k + 1] if k + 1 < len(exec_idx) else last_row
        rets.append(holding_returns(panel, e_row, exit_row, config))
    return PeriodData(volatility=vols, returns=rets)


def one_way_cost(
    panel: PanelMatrices,
    exec_row: int,
    month_row: int,
    traded_notional: np.ndarray,
    volatility: np.ndarray,
    config: RetestConfig,
    spread_basis: np.ndarray,
    charge_impact: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dollar cost of trading `traded_notional` per name: (spread, impact, commission)."""
    trading = traded_notional > 0.0
    spread = spread_basis[month_row]
    dollar_volume = panel.dv_basis[month_row]
    price = panel.raw_open[exec_row].astype(np.float64)

    if not np.isfinite(spread[trading]).all():
        raise RuntimeError("a traded name has no spread on any prior month")

    spread_cost = np.where(trading, traded_notional * spread / 2.0, 0.0)

    if charge_impact:
        with np.errstate(invalid="ignore", divide="ignore"):
            participation = np.where(dollar_volume > 0,
                                     traded_notional / dollar_volume, np.nan)
            impact_rate = config.impact_coefficient * volatility * np.sqrt(participation)
        impact_cost = np.where(trading,
                               traded_notional * np.nan_to_num(impact_rate, nan=0.0), 0.0)
    else:
        impact_cost = np.zeros_like(traded_notional)

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(price > 0, traded_notional / price, 0.0)
    commission = np.maximum(config.commission_min_per_order,
                            config.commission_per_share * shares)
    commission = np.minimum(commission, config.commission_cap_fraction * traded_notional)
    commission = np.where(trading, commission, 0.0)

    return spread_cost, impact_cost, commission


def run_leg(
    panel: PanelMatrices,
    config: RetestConfig,
    selections: list[np.ndarray],
    signal_idx: np.ndarray,
    exec_idx: np.ndarray,
    month_rows: np.ndarray,
    periods: PeriodData,
    exposure: float,
    spread_basis: np.ndarray,
    periods_per_year: float,
    charge_impact: bool = True,
    borrow_annual: float = 0.0,
) -> BookResult:
    """Walk one leg forward, tracking DRIFTED weights so turnover is the real thing."""
    n_tickers = panel.adj_close.shape[1]
    previous_weights = np.zeros(n_tickers)
    pnl_by_ticker = np.zeros(n_tickers)
    max_cell, max_label = 0.0, ""

    gross, costs, turnovers, counts = [], [], [], []
    spread_costs, impact_costs, commission_costs, borrow_costs = [], [], [], []
    period_dates = []

    for k, (e_row, m_row, chosen) in enumerate(zip(exec_idx, month_rows, selections)):
        target = np.zeros(n_tickers)
        if chosen.size:
            target[chosen] = exposure / chosen.size

        traded_weight = np.abs(target - previous_weights)
        traded_notional = traded_weight * config.equity
        spread_cost, impact_cost, commission = one_way_cost(
            panel, e_row, m_row, traded_notional, periods.volatility[k], config,
            spread_basis, charge_impact,
        )

        period_return = periods.returns[k]
        held = target > 0
        contribution = target[held] * period_return[held]
        leg_return = float(np.sum(contribution))

        if contribution.size:
            held_idx = np.flatnonzero(held)
            pnl_by_ticker[held_idx] += contribution
            j = int(np.argmax(np.abs(contribution)))
            if abs(float(contribution[j])) > abs(max_cell):
                max_cell = float(contribution[j])
                max_label = (f"{panel.tickers[held_idx[j]]}@"
                             f"{panel.dates[e_row].date()}")

        borrow = borrow_annual * exposure / periods_per_year
        total_cost = float(spread_cost.sum() + impact_cost.sum() + commission.sum())
        total_cost = total_cost / config.equity + borrow

        gross.append(leg_return)
        costs.append(total_cost)
        turnovers.append(float(traded_weight.sum() / 2.0 / max(exposure, 1e-12)))
        counts.append(int(held.sum()))
        spread_costs.append(float(spread_cost.sum()) / config.equity)
        impact_costs.append(float(impact_cost.sum()) / config.equity)
        commission_costs.append(float(commission.sum()) / config.equity)
        borrow_costs.append(borrow)
        period_dates.append(panel.dates[e_row])

        drifted = target * (1.0 + period_return)
        previous_weights = np.where(np.isfinite(drifted), drifted, 0.0)

    return BookResult(
        dates=pd.DatetimeIndex(period_dates),
        gross_return=np.asarray(gross),
        cost=np.asarray(costs),
        turnover=np.asarray(turnovers),
        n_names=np.asarray(counts),
        spread_cost=np.asarray(spread_costs),
        impact_cost=np.asarray(impact_costs),
        commission_cost=np.asarray(commission_costs),
        borrow_cost=np.asarray(borrow_costs),
        pnl_by_ticker=pnl_by_ticker,
        max_cell_pnl=max_cell,
        max_cell_label=max_label,
    )


def build_selections(
    panel: PanelMatrices,
    config: RetestConfig,
    rank_threshold: float,
    signal_idx: np.ndarray,
    exec_idx: np.ndarray,
    month_rows: np.ndarray,
) -> dict[str, list]:
    """Per-rebalance ticker indices for the long leg, the short leg and the universe.

    Signal = negative trailing 5-day return on adjusted closes through the signal date.
    The cross-sectional z-score is computed for the record (prereg 3); a decile sort is
    rank-invariant, so it cannot and does not change which names are picked.
    """
    longs, shorts, universes, signals, zsignals = [], [], [], [], []

    for s_row, e_row, m_row in zip(signal_idx, exec_idx, month_rows):
        dollar_volume = panel.dollar_volume[m_row]
        rank = panel.dv_rank[m_row]

        past = panel.adj_close[s_row - config.lookback_days]
        now = panel.adj_close[s_row]
        with np.errstate(invalid="ignore", divide="ignore"):
            ret5 = now / past - 1.0
        signal = -ret5

        tradable = (
            np.isfinite(rank)
            & (rank > rank_threshold)
            & np.isfinite(dollar_volume)
            & np.isfinite(signal)
            & np.isfinite(panel.adj_open[e_row])
            & (panel.raw_open[e_row] >= config.min_price)
        )
        universe = np.flatnonzero(tradable)
        universes.append(universe)
        signals.append(np.where(tradable, signal, np.nan))

        z = np.full_like(signal, np.nan)
        if universe.size > 1:
            block = signal[universe]
            sd = float(np.std(block, ddof=1))
            z[universe] = (block - float(block.mean())) / sd if sd > 0 else 0.0
        zsignals.append(z)

        if universe.size < config.min_names_per_leg * 3:
            longs.append(np.array([], dtype=int))
            shorts.append(np.array([], dtype=int))
            continue

        order = universe[np.argsort(signal[universe], kind="stable")]
        n_long = int(np.floor(universe.size * config.decile))
        n_long = min(max(n_long, 0), config.max_names_per_leg)
        long_leg = (order[-n_long:] if n_long >= config.min_names_per_leg
                    else np.array([], dtype=int))
        longs.append(long_leg)

        shortable = universe[dollar_volume[universe] >= config.shortable_dollar_volume]
        if shortable.size >= config.min_shortable_names:
            s_order = shortable[np.argsort(signal[shortable], kind="stable")]
            n_short = int(np.floor(shortable.size * config.decile))
            n_short = min(max(n_short, 0), config.max_names_per_leg)
            short_leg = (s_order[:n_short] if n_short >= config.min_names_per_leg
                         else np.array([], dtype=int))
        else:
            short_leg = np.array([], dtype=int)
        shorts.append(short_leg)

    return {"long": longs, "short": shorts, "universe": universes,
            "signal": signals, "zsignal": zsignals}
