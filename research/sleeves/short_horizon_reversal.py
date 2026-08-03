"""Sleeve: short-horizon cross-sectional reversal, weekly, on liquid US equities.

PRE-SPECIFICATION
=================
Written and frozen BEFORE the run. Everything in `ReversalConfig` below is the
registered configuration. One run, both books (long/short and long-only) reported,
whichever way they come out. No parameter in this file was chosen by looking at a
result.

Why this sleeve exists
----------------------
`docs/project-control/specs/2026-07-28-the-breadth-lever.md` diagnoses every prior
study in this programme as breadth-starved: 4-12 rebalances a year, one cross-section
each. Grinold says IR ~= IC * sqrt(BR), so a decade of work on IC could not have
reached the target no matter how good the signal got. This sleeve pulls the other
lever: 52 rebalances a year on the daily panel, a 4-13x multiplier on realised
rebalance count, i.e. a 2-3.6x multiplier on sqrt(BR) for identical per-bet skill.

The same doc names the reason it might not work: "Costs scale LINEARLY with turnover;
Sharpe scales only with sqrt(BR)." A weekly book replaces itself ~52 times a year
against ~3.4x for the capacity study. So the sleeve is a race between a sqrt gain and
a linear cost, and the universe is deliberately restricted to names where the linear
term is smallest ($5M+ median dollar volume) -- the one region the prior small-cap
work could not use.

Hypothesis (H1): the equal-weighted top decile of the negative trailing 5-day return,
rebalanced weekly in the >$5M/day universe, earns a positive net-of-cost excess over
an equal-weight buy-and-hold of that same universe.

The two universes, and why there are two
----------------------------------------
`spread_with_resolution` classifies every (name, month) as `measured`, `upper_bound`
or `unmeasurable`. The governing rule for this study is that `upper_bound` and
`unmeasurable` names are EXCLUDED and never costed at the floor -- costing a name at
the noise floor manufactures a cheap trade out of an absence of information, which is
the failure mode the whole cost apparatus exists to prevent.

That rule has a consequence which must be stated in advance, because it is not
neutral. In the >$5M/day universe only 27% of cells resolve, and the resolved ones
have a MEDIAN spread of 100bps against 52bps for the unresolved ones. This is not
noise: a name resolves precisely when its true spread is wide enough to clear 1.5x the
EDGE noise floor. So the `measured` universe is a systematically EXPENSIVE subsample
of the liquid universe, and running only on it prices a weekly strategy at roughly
double the spread its actual universe would pay. Under a 52x turnover that difference
is worth tens of percent a year and could decide the verdict on its own.

Therefore two universes are registered in advance, both reported, neither selected:

  PRIMARY (governing, rule-compliant):  `measured` cells only.
  SECONDARY (diagnostic, declared non-gate-eligible): `measured` + `upper_bound`,
      with `upper_bound` names costed AT their upper-bound estimate.

The secondary does not violate the spirit of the exclusion rule. The prohibited move
is costing an unresolved name at the FLOOR, which understates cost. Costing it at its
upper bound OVERSTATES cost -- `spread_with_resolution`'s own docstring says so -- by
roughly 3-10x for a modern liquid name whose true spread is 5-15bps. The secondary is
therefore strictly more conservative per name than the truth while covering 3.6x more
names. Its purpose is to answer one question the primary cannot: is a failure caused
by the signal, or by the cost model's inability to resolve spreads in the liquid band?

Point-in-time discipline
------------------------
* Signal uses closes through the signal date t only.
* Execution is at the OPEN of t+1, never at the close the signal was computed from.
  Short-horizon reversal is the construction most exposed to this; taking the same
  close for both is the standard way this effect is manufactured.
* Universe membership, spread and dollar volume come from the last month-end panel
  row at or before t, so they were knowable a month ahead at worst.
* `load_prices` refuses bars after 2015-12-31; nothing here asks it to.

Delisting accounting (the two bugs from `capacity_curve_result.md` §4)
----------------------------------------------------------------------
* A terminal return is applied ONLY if the delisting date falls inside the position's
  own holding window extended by `DELISTING_GRACE_DAYS`. Asking "did this ticker ever
  delist" charged a 2012 bankruptcy against a 2003 exit and produced -60%/yr.
* The book is re-formed from scratch every week from the current universe, so a name
  booked as terminal cannot be re-booked: it is not in next week's universe. The prior
  bug re-charged -100% every month forever and produced -112%/yr on a long-only book.

Costs (mandatory, per name, never flat)
---------------------------------------
one-way cost = spread/2 + impact + commission

* spread/2 : EDGE targets the EFFECTIVE spread, whose conventional one-way cost from
  mid is half of it. Per name, per month, point-in-time.
* impact : C * sigma_daily * sqrt(notional / median_dollar_volume), the square-root
  law, C = 1.0 (the conservative end of the usual 0.5-1.0). At the registered $1M of
  equity this is small by construction, which is the point of the liquidity floor.
* commission : IBKR, $0.0035/share, $0.35 per-order minimum, capped at 1% of value.
* short book additionally pays a flat 100bps/yr general-collateral borrow fee, and is
  credited NO interest on short proceeds. Both are the conservative direction.

Artefact filters (all inherited from the registered monthly panel, re-checked here)
----------------------------------------------------------------------------------
price floor $2, non-zero volume on >=90% of trailing 63 days, per-name holding
returns capped at +/-100%. A prior study booked +9,900% on a zero-volume bankrupt
shell that was 13% of its P&L.

Benchmark
---------
Equal-weight buy-and-hold of THIS sleeve's own universe, rebalanced on the same
weekly grid, gross of costs. Not the S&P. A strategy that loses to passive ownership
of the names it picks from has no edge whatever its raw return, which is exactly the
verdict `capacity_curve_result.md` reached.

Robustness ladder (not a tuning knob)
-------------------------------------
Net Sharpe is reported at 0.5x, 1x, 1.5x, 2x and 3x the estimated cost. The ladder is
reported in full; the 1x rung is the registered answer. No rung is selected.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO / "_data" / "sharadar" / "panel"
DEV_CUTOFF = pd.Timestamp("2015-12-31")


@dataclass(frozen=True)
class ReversalConfig:
    """The registered configuration. Frozen before the run; nothing here is fitted."""

    # --- universe -------------------------------------------------------------
    min_dollar_volume: float = 5e6
    # Borrow is only plausible where there is a real securities-lending market. $25M/day
    # is the registered threshold; below it a short leg is a fiction.
    shortable_dollar_volume: float = 2.5e7
    min_price: float = 2.00
    # PRIMARY excludes upper_bound; the SECONDARY diagnostic includes it (see docstring).
    allowed_spread_regimes: tuple[str, ...] = ("measured",)

    # --- signal ---------------------------------------------------------------
    lookback_days: int = 5           # trailing 5-day return; signal = its negative
    # --- construction ---------------------------------------------------------
    decile: float = 0.10
    max_names_per_leg: int = 100
    min_names_per_leg: int = 20
    # Reg-T retail maximum for a market-neutral book: 100% long + 100% short of equity.
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
    periods_per_year: float = 52.0
    cost_ladder: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)

    # --- book selector --------------------------------------------------------
    label: str = "PRIMARY_measured_only"


SECONDARY_CONFIG = ReversalConfig(
    allowed_spread_regimes=("measured", "upper_bound"),
    label="SECONDARY_upper_bound_costed_at_bound",
)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class PanelMatrices:
    """Wide, calendar-aligned matrices. Rows are trading days, columns are tickers.

    A wide layout is used rather than the long panel because every calculation here is
    a cross-section on a date: "the top decile of a column vector" is a one-liner on a
    matrix and a groupby-per-date on a long frame. NaN means the name had no bar that
    day, and NaN propagation through the return arithmetic is the desired behaviour --
    a name with no bar is a name we could not have traded.
    """

    dates: pd.DatetimeIndex
    tickers: np.ndarray            # object array of ticker symbols
    adj_open: np.ndarray           # split/dividend adjusted open
    adj_close: np.ndarray          # split/dividend adjusted close
    raw_open: np.ndarray           # unadjusted open, needed for share counts
    months: pd.PeriodIndex         # rows of the monthly panel matrices
    spread: np.ndarray             # [month, ticker] EDGE spread, NaN where not allowed
    dollar_volume: np.ndarray      # [month, ticker] trailing-63d median dollar volume
    month_close: np.ndarray        # [month, ticker] month-end close (price-floor check)
    # Forward-filled copies used ONLY to price trades, never to decide eligibility. A
    # name that leaves the universe must still be SOLD, and its sale must be costed:
    # reading the current month's NaN there charges nothing for the exit, which hands
    # the strategy a free liquidation of every name whose spread stopped resolving.
    spread_cost_basis: np.ndarray
    dv_cost_basis: np.ndarray
    delist_date: np.ndarray        # [ticker] int64 ns, or iNaT
    delist_return: np.ndarray      # [ticker] terminal return


def _eligible_monthly(monthly: pd.DataFrame, config: ReversalConfig) -> pd.DataFrame:
    """Monthly cells this configuration is allowed to see.

    The panel already applied the $2 price floor and the 90%-trading-days filter before
    it would estimate a spread at all (cells failing them carry regime 'ineligible'),
    so selecting on regime enforces those too. They are re-asserted explicitly anyway:
    a filter that is only implied is a filter that silently stops applying when an
    upstream file changes.
    """
    mask = (
        monthly["spread_regime"].isin(config.allowed_spread_regimes)
        & (monthly["median_dollar_volume"] > config.min_dollar_volume)
        & (monthly["close"] >= config.min_price)
        & (monthly["trading_fraction"] >= 0.90)
        & np.isfinite(monthly["spread"])
    )
    return monthly.loc[mask]


def build_matrices(configs: tuple[ReversalConfig, ...]) -> dict[str, PanelMatrices]:
    """Build one PanelMatrices per config, sharing a single read of the 29M-bar panel.

    The daily panel is 2.5 GB in memory, so it is read once, restricted to the union of
    tickers any config could ever hold, pivoted, and dropped.
    """
    monthly = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    eligible = {c.label: _eligible_monthly(monthly, c) for c in configs}

    union_tickers = sorted(set().union(*(set(e["ticker"]) for e in eligible.values())))
    logger.info("union universe: %d tickers", len(union_tickers))

    prices = pd.read_parquet(
        PANEL_DIR / "prices_to_2015-12-31.parquet",
        columns=["ticker", "date", "open", "close", "closeadj"],
    )
    prices = prices[prices["ticker"].isin(set(union_tickers))]
    if prices["date"].max() > DEV_CUTOFF:
        raise RuntimeError("price panel contains bars past the DEV cutoff")

    # closeadj/close is the cumulative split+dividend factor Sharadar has already
    # applied to the close; applying the same factor to the open puts entry and exit
    # prices on one consistent basis. Comparing a raw open against an adjusted close
    # across a split is how phantom 50% moves get manufactured.
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

    # The monthly panel's "month end" is the last trading day OF THAT TICKER in that
    # month, so it is not a shared calendar: a name that stopped trading on the 12th
    # carries the 12th. Indexing on the raw date would therefore line up almost no
    # tickers with each other. Rows are keyed on the calendar MONTH instead, and
    # `month_row_for` then reads the previous month, which every ticker's row is
    # complete by whatever day within the month it happens to carry.
    months = pd.PeriodIndex(np.sort(monthly["date"].dt.to_period("M").unique()))
    month_pos = pd.Series(np.arange(len(months)), index=months)

    result: dict[str, PanelMatrices] = {}
    for config in configs:
        frame = eligible[config.label]
        m_rows = month_pos.reindex(frame["date"].dt.to_period("M")).to_numpy()
        m_cols = tick_pos.reindex(frame["ticker"]).to_numpy()
        m_shape = (len(months), len(union_tickers))

        spread = np.full(m_shape, np.nan)
        dollar_volume = np.full(m_shape, np.nan)
        month_close = np.full(m_shape, np.nan)
        spread[m_rows, m_cols] = frame["spread"].to_numpy()
        dollar_volume[m_rows, m_cols] = frame["median_dollar_volume"].to_numpy()
        month_close[m_rows, m_cols] = frame["close"].to_numpy()

        spread_basis = pd.DataFrame(spread).ffill().to_numpy()
        dv_basis = pd.DataFrame(dollar_volume).ffill().to_numpy()

        result[config.label] = PanelMatrices(
            dates=dates,
            tickers=np.asarray(union_tickers, dtype=object),
            adj_open=adj_open,
            adj_close=adj_close,
            raw_open=raw_open,
            months=months,
            spread=spread,
            dollar_volume=dollar_volume,
            month_close=month_close,
            spread_cost_basis=spread_basis,
            dv_cost_basis=dv_basis,
            delist_date=delist_date,
            delist_return=delist_return,
        )
    return result


def weekly_grid(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Signal-date and execution-date row indices for a weekly rebalance.

    The signal date is the last trading day of each calendar week; execution is the
    next trading day's open. Returning row INDICES rather than dates keeps every
    downstream lookup an array index instead of a merge.
    """
    week = dates.to_period("W")
    last_of_week = pd.Series(np.arange(len(dates)), index=week).groupby(level=0).max()
    signal_idx = np.sort(last_of_week.to_numpy())
    # Execution is the following bar; the final signal date has no bar after it.
    exec_idx = signal_idx + 1
    keep = exec_idx < len(dates)
    return signal_idx[keep], exec_idx[keep]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass
class BookResult:
    """Per-period record for one book (long-only, short-only, or the universe)."""

    dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    gross_return: np.ndarray = field(default_factory=lambda: np.array([]))
    cost: np.ndarray = field(default_factory=lambda: np.array([]))
    turnover: np.ndarray = field(default_factory=lambda: np.array([]))
    n_names: np.ndarray = field(default_factory=lambda: np.array([]))
    spread_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    impact_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    commission_cost: np.ndarray = field(default_factory=lambda: np.array([]))


def _holding_returns(
    panel: PanelMatrices,
    entry_row: int,
    exit_row: int,
    config: ReversalConfig,
) -> np.ndarray:
    """Adjusted open-to-open return per ticker over one holding period.

    A name whose exit open is missing did not trade on the exit date. Its position is
    closed at the last adjusted close inside the window, and a terminal return is
    applied ONLY when the delisting date lies inside the window extended by the grace
    period -- the check whose absence charged a 2012 bankruptcy to a 2003 exit.
    """
    entry = panel.adj_open[entry_row]
    exit_price = panel.adj_open[exit_row].copy()

    missing = ~np.isfinite(exit_price) & np.isfinite(entry)
    if missing.any():
        window = panel.adj_close[entry_row: exit_row + 1, missing]
        # Last finite close in the window, or NaN if the name never traded again.
        valid = np.isfinite(window)
        has_any = valid.any(axis=0)
        last_row = np.where(has_any, valid.shape[0] - 1 - valid[::-1].argmax(axis=0), 0)
        last_close = np.where(has_any, window[last_row, np.arange(window.shape[1])],
                              np.nan)
        exit_price[missing] = last_close

    returns = exit_price / entry - 1.0
    # A name that stopped trading with no close at all in the window is held flat until
    # its terminal event is (or is not) booked below.
    stalled = ~np.isfinite(returns) & np.isfinite(entry)
    returns = np.where(stalled, 0.0, returns)

    grace = np.timedelta64(config.delisting_grace_days, "D").astype("timedelta64[ns]")
    window_start = panel.dates[entry_row].to_datetime64().astype(np.int64)
    window_end = panel.dates[exit_row].to_datetime64().astype(np.int64) + int(grace)
    in_window = (panel.delist_date >= window_start) & (panel.delist_date <= window_end)
    # Only names that actually stopped printing prices are treated as delisted. A name
    # still trading on the exit date was not delisted inside the window, whatever the
    # ACTIONS file says about a later corporate event.
    booked = in_window & (missing | stalled)
    if booked.any():
        terminal = panel.delist_return[booked]
        returns[booked] = (1.0 + returns[booked]) * (1.0 + terminal) - 1.0

    # Artefact cap. A prior study found +9,900% on a bankrupt zero-volume shell.
    return np.clip(returns, -config.return_cap, config.return_cap)


def _trailing_vol(panel: PanelMatrices, row: int, window: int) -> np.ndarray:
    """Trailing daily volatility of adjusted closes ending at ``row`` (inclusive)."""
    lo = max(0, row - window)
    block = panel.adj_close[lo: row + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_returns = np.diff(np.log(block), axis=0)
    if log_returns.shape[0] < 5:
        return np.full(panel.adj_close.shape[1], np.nan)
    with warnings.catch_warnings():
        # Names with fewer than two bars in the window return NaN here; the impact term
        # treats NaN as "no impact estimate", and every universe name has >=90% trading
        # days by construction, so this only fires for columns we never trade.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(log_returns, axis=0, ddof=1)


def _one_way_cost(
    panel: PanelMatrices,
    exec_row: int,
    month_row: int,
    traded_notional: np.ndarray,
    volatility: np.ndarray,
    config: ReversalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dollar cost of trading ``traded_notional`` per name. Returns (spread, impact, commission)."""
    trading = traded_notional > 0.0
    spread = panel.spread_cost_basis[month_row]
    dollar_volume = panel.dv_cost_basis[month_row]
    price = panel.raw_open[exec_row].astype(np.float64)

    # A name being traded must have a priceable spread. If the forward fill cannot
    # supply one the position is untradeable and the study is wrong, not cheap -- so
    # this raises rather than defaulting to zero.
    if not np.isfinite(spread[trading]).all():
        raise RuntimeError("a traded name has no spread on any prior month")

    spread_cost = np.where(trading, traded_notional * spread / 2.0, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        participation = np.where(dollar_volume > 0, traded_notional / dollar_volume,
                                 np.nan)
        impact_rate = config.impact_coefficient * volatility * np.sqrt(participation)
    impact_cost = np.where(trading, traded_notional * np.nan_to_num(impact_rate, nan=0.0),
                           0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(price > 0, traded_notional / price, 0.0)
    commission = np.maximum(config.commission_min_per_order,
                            config.commission_per_share * shares)
    commission = np.minimum(commission, config.commission_cap_fraction * traded_notional)
    commission = np.where(trading, commission, 0.0)

    return spread_cost, impact_cost, commission


def _run_leg(
    panel: PanelMatrices,
    config: ReversalConfig,
    selections: list[np.ndarray],
    signal_idx: np.ndarray,
    exec_idx: np.ndarray,
    month_rows: np.ndarray,
    exposure: float,
    borrow_annual: float = 0.0,
) -> BookResult:
    """Walk one leg forward, tracking drifted weights so turnover is the real thing.

    Turnover is measured against the weights the previous period DRIFTED to, not
    against the previous target. A name that stays in the book and merely drifts is not
    re-bought, and charging it as if it were would overstate cost; a name that leaves is
    fully sold. Getting this wrong in either direction moves a 52x-turnover strategy by
    tens of percent a year.
    """
    n_tickers = panel.adj_close.shape[1]
    previous_weights = np.zeros(n_tickers)

    gross, costs, turnovers, counts = [], [], [], []
    spread_costs, impact_costs, commission_costs = [], [], []
    period_dates = []

    for k, (s_row, e_row, m_row, chosen) in enumerate(
        zip(signal_idx, exec_idx, month_rows, selections)
    ):
        target = np.zeros(n_tickers)
        if chosen.size:
            target[chosen] = exposure / chosen.size

        traded_weight = np.abs(target - previous_weights)
        traded_notional = traded_weight * config.equity
        volatility = _trailing_vol(panel, s_row, config.impact_vol_window)
        spread_cost, impact_cost, commission = _one_way_cost(
            panel, e_row, m_row, traded_notional, volatility, config
        )

        # Exit is at the next period's execution open; the next period's turnover term
        # charges the sale. The final period is closed out explicitly after the loop.
        exit_row = exec_idx[k + 1] if k + 1 < len(exec_idx) else panel.adj_close.shape[0] - 1
        period_return = _holding_returns(panel, e_row, exit_row, config)

        held = target > 0
        leg_return = float(np.sum(target[held] * period_return[held]))

        borrow = borrow_annual * exposure / config.periods_per_year
        total_cost = float(spread_cost.sum() + impact_cost.sum() + commission.sum())
        total_cost = total_cost / config.equity + borrow

        gross.append(leg_return)
        costs.append(total_cost)
        turnovers.append(float(traded_weight.sum() / 2.0 / max(exposure, 1e-12)))
        counts.append(int(held.sum()))
        spread_costs.append(float(spread_cost.sum()) / config.equity)
        impact_costs.append(float(impact_cost.sum()) / config.equity)
        commission_costs.append(float(commission.sum()) / config.equity)
        period_dates.append(panel.dates[e_row])

        # Drift: the weights we arrive at next rebalance, before trading.
        drifted = target * (1.0 + period_return)
        drifted = np.where(np.isfinite(drifted), drifted, 0.0)
        previous_weights = drifted

    return BookResult(
        dates=pd.DatetimeIndex(period_dates),
        gross_return=np.asarray(gross),
        cost=np.asarray(costs),
        turnover=np.asarray(turnovers),
        n_names=np.asarray(counts),
        spread_cost=np.asarray(spread_costs),
        impact_cost=np.asarray(impact_costs),
        commission_cost=np.asarray(commission_costs),
    )


def build_selections(
    panel: PanelMatrices,
    config: ReversalConfig,
    signal_idx: np.ndarray,
    exec_idx: np.ndarray,
    month_rows: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    """Per-rebalance ticker indices for the long leg, the short leg and the universe.

    Signal = negative trailing 5-day return on adjusted closes through the signal date.
    Ranking is by decile, so the cross-sectional z-score is computed for diagnostics
    (IC) only -- a decile sort is rank-invariant and the z-score cannot change it.
    """
    longs, shorts, universes, signals = [], [], [], []

    for s_row, e_row, m_row in zip(signal_idx, exec_idx, month_rows):
        spread = panel.spread[m_row]
        dollar_volume = panel.dollar_volume[m_row]

        past = panel.adj_close[s_row - config.lookback_days]
        now = panel.adj_close[s_row]
        with np.errstate(invalid="ignore", divide="ignore"):
            ret5 = now / past - 1.0
        signal = -ret5

        tradable = (
            np.isfinite(spread)
            & (dollar_volume > config.min_dollar_volume)
            & np.isfinite(signal)
            & np.isfinite(panel.adj_open[e_row])
            & (panel.raw_open[e_row] >= config.min_price)
        )
        universe = np.flatnonzero(tradable)
        universes.append(universe)
        full_signal = np.where(tradable, signal, np.nan)
        signals.append(full_signal)

        if universe.size < config.min_names_per_leg * 3:
            longs.append(np.array([], dtype=int))
            shorts.append(np.array([], dtype=int))
            continue

        order = universe[np.argsort(signal[universe], kind="stable")]
        n_long = int(np.floor(universe.size * config.decile))
        n_long = min(max(n_long, 0), config.max_names_per_leg)
        long_leg = order[-n_long:] if n_long >= config.min_names_per_leg else np.array([], dtype=int)
        longs.append(long_leg)

        # The short leg lives only where borrow is plausible, so it is a decile of the
        # SHORTABLE subset, not the shortable part of the full universe's decile.
        shortable = universe[dollar_volume[universe] > config.shortable_dollar_volume]
        if shortable.size >= config.min_names_per_leg * 3:
            s_order = shortable[np.argsort(signal[shortable], kind="stable")]
            n_short = int(np.floor(shortable.size * config.decile))
            n_short = min(max(n_short, 0), config.max_names_per_leg)
            short_leg = s_order[:n_short] if n_short >= config.min_names_per_leg else np.array([], dtype=int)
        else:
            short_leg = np.array([], dtype=int)
        shorts.append(short_leg)

    return {"long": longs, "short": shorts, "universe": universes, "signal": signals}


def month_row_for(panel: PanelMatrices, signal_idx: np.ndarray) -> np.ndarray:
    """Row of the monthly panel that was demonstrably complete at each signal date.

    The PREVIOUS calendar month, never the current one. Each ticker's monthly row is
    stamped with its own last trading day of that month, which can be any day in it, so
    the current month's row is not knowable until the month is over. Reading the prior
    month costs up to eight weeks of staleness in the liquidity and spread estimates --
    the conservative direction, and how a real system would run anyway.
    """
    signal_months = panel.dates[signal_idx].to_period("M")
    lookup = pd.Series(np.arange(len(panel.months)), index=panel.months)
    return lookup.reindex(signal_months - 1).to_numpy(dtype=float, na_value=-1.0).astype(int)
