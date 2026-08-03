"""Sleeve: institutional ownership FLOW from Sharadar SF3 13F holdings.

Registered design: `research/sleeves/institutional_flow_prereg.md`, written before this
module produced a number. `scripts/run_institutional_flow_sleeve.py` runs it.

The signal is the quarter-on-quarter change in the fraction of shares outstanding held by
13F filers, cross-sectionally z-scored. The hypothesis is that institutional accumulation
is informed and persists past the filing date.

**The one rule that decides whether any of this means anything.** SF3 carries no filing
date. Joining it on `calendardate` reads holdings six weeks before they were public.
Every join here is on `calendardate + 45 days`, and the rebalance date is the first
month-end at or after that, which pins the schedule to the February / May / August /
November month-ends. `scripts/build_sf3_ownership.py` computes the availability date once
so no consumer can forget.

The portfolio engine, cost model and delisting accounting are lifted from
`research/capacity_study.py` unchanged rather than rewritten. That module's two accounting
defects (terminal returns booked against the wrong year; delisted names re-booked every
month forever) were found the hard way and are already fixed there; a fresh implementation
would be a fresh opportunity to reintroduce them. The benchmark runs through the SAME
engine, holding the whole universe instead of the top 50, so strategy and benchmark cannot
differ through an accounting asymmetry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from research.capacity_study import (
    FX_COST_EACH_WAY,
    _commission_fraction,
    _impact_fraction,
    _max_drawdown,
)
from research.delisting import (  # noqa: F401  -- re-exported: the declared repair
    CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW,
)
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW
from research.delisting import in_window, in_window_mask

logger = logging.getLogger(__name__)

# --- registered parameters (prereg §3-§5). Not tunable after the fact. ---------------

FILING_LAG_DAYS = 45
N_POSITIONS = 50
N_POSITIONS_TERCILE = 20
ENTRY_QUANTILE = 0.90
EXIT_QUANTILE = 0.70
BOOK_SIZE = 250_000.0
PARTICIPATION_LIMIT = 0.01
MIN_DOLLAR_VOLUME = BOOK_SIZE / N_POSITIONS / PARTICIPATION_LIMIT  # $500,000
FORWARD_RETURN_CAP = 1.00
MAX_OWNERSHIP_RATIO = 1.50
WINSOR_QUANTILE = 0.01
# DELISTING_WINDOW_DAYS and the window edges now live in `research.delisting`, imported
# above. This module carried its own copy of the same off-by-one lower edge.
DELISTING_WINDOW_DAYS = REGISTERED_DELISTING_WINDOW[1]
MIN_CROSS_SECTION = 50
HOLDING_MONTHS = 3
PERIODS_PER_YEAR = 12.0
REBALANCES_PER_YEAR = 4.0

# Verdict gates, fixed in advance (prereg §9).
GATE_EXCESS = 0.02
GATE_SHARPE = 0.75
GATE_IC_T = 2.0


@dataclass
class PortfolioRun:
    """Realised path of one book. Costs are already inside `net_returns`."""

    dates: list[pd.Timestamp]
    net_returns: np.ndarray
    gross_returns: np.ndarray
    costs: np.ndarray
    equity: np.ndarray
    turnovers: list[float]
    position_counts: list[int]
    cost_components: dict[str, float]
    #: exit legs counted in TURNOVER but charged NOTHING -- the name had left both the
    #: selection universe AND the accrual frame, or never had a measured spread
    unpriced_exit_legs: int = 0
    #: how many of those were charged at the name's LAST OBSERVED inputs
    charged_unpriced_exit_legs: int = 0

    @property
    def annual_return(self) -> float:
        return float(np.mean(self.net_returns) * PERIODS_PER_YEAR)

    @property
    def annual_gross(self) -> float:
        return float(np.mean(self.gross_returns) * PERIODS_PER_YEAR)

    @property
    def annual_volatility(self) -> float:
        if len(self.net_returns) < 2:
            return float("nan")
        return float(np.std(self.net_returns, ddof=1) * np.sqrt(PERIODS_PER_YEAR))

    @property
    def sharpe(self) -> float:
        vol = self.annual_volatility
        return self.annual_return / vol if vol > 0 else float("nan")

    @property
    def max_drawdown(self) -> float:
        return _max_drawdown(self.equity)

    @property
    def annual_cost_drag(self) -> float:
        if not len(self.costs):
            return 0.0
        return float(np.mean(self.costs) * PERIODS_PER_YEAR)

    @property
    def annual_turnover(self) -> float:
        if not self.turnovers:
            return 0.0
        return float(np.mean(self.turnovers)) * REBALANCES_PER_YEAR


@dataclass
class SleeveResult:
    strategy: PortfolioRun
    benchmark: PortfolioRun
    ic_mean: float
    ic_std_error: float
    ic_t_stat: float
    ic_count: int
    ic_by_date: pd.Series
    ic_1m_mean: float
    ic_1m_t_stat: float
    long_short_annual: float
    long_short_sharpe: float
    rebalance_dates: list[pd.Timestamp]
    universe_size_mean: float
    diagnostics: dict[str, float] = field(default_factory=dict)
    tercile_results: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def excess_annual(self) -> float:
        return self.strategy.annual_return - self.benchmark.annual_return

    @property
    def gross_excess_annual(self) -> float:
        """Excess BEFORE costs: what the signal contributed, separate from what it cost.

        Reported because the two failures are different failures. A negative gross excess
        means the signal is wrong; a positive gross excess swamped by costs means the
        signal is real but the construction cannot afford it.
        """
        return self.strategy.annual_gross - self.benchmark.annual_gross

    @property
    def sharpe_standard_error(self) -> float:
        """Approximate standard error of the annualised Sharpe, ~sqrt((1+S^2/2)/years).

        At 2.17 years this is ~0.68, which is the single most important number in the
        whole report: every Sharpe here is one standard error from zero.
        """
        years = len(self.strategy.net_returns) / PERIODS_PER_YEAR
        if years <= 0:
            return float("nan")
        sharpe = self.strategy.sharpe
        if not np.isfinite(sharpe):
            return float("nan")
        return float(np.sqrt((1.0 + sharpe ** 2 / 2.0) / years))

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.excess_annual) or self.excess_annual <= 0:
            return "DEAD"
        gates = (
            self.excess_annual > GATE_EXCESS
            and np.isfinite(self.strategy.sharpe)
            and self.strategy.sharpe >= GATE_SHARPE
            and np.isfinite(self.ic_t_stat)
            and self.ic_t_stat >= GATE_IC_T
        )
        return "PROMISING" if gates else "MARGINAL"


# --- schedule -----------------------------------------------------------------------


def market_month_ends(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """The last TRADING day of each calendar month, market-wide.

    `build_monthly_panel` stamps each name's last bar of the month, which for a name that
    stopped trading mid-month is a mid-month date. Treating the raw set of panel dates as
    "month ends" therefore produces phantom cross-sections containing only the handful of
    names that delisted that day -- the first run of this sleeve rebalanced into a
    one-name cross-section on 2014-02-14 for exactly that reason. The market-wide maximum
    per calendar month is the real grid.
    """
    dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    grid = pd.Series(dates, index=dates).groupby(
        dates.values.astype("datetime64[M]")).max()
    return pd.DatetimeIndex(grid.values).sort_values()


def rebalance_schedule(
    quarters: pd.DatetimeIndex,
    month_ends: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each SF3 quarter to the first tradable month-end at or after its 13F deadline.

    Returns a frame of (quarter, prev_quarter, available_date, rebalance_date). Quarters
    whose availability date falls past the last month-end in the price panel are dropped:
    they are signals that could have been formed but never acted on inside the DEV window.
    """
    quarters = pd.DatetimeIndex(sorted(pd.unique(quarters)))
    month_ends = pd.DatetimeIndex(sorted(pd.unique(month_ends)))

    rows = []
    for index in range(1, len(quarters)):
        quarter = quarters[index]
        previous = quarters[index - 1]
        # Consecutive quarters only. A gap means the "change" would span six months and
        # would not be the quantity the hypothesis is about.
        gap = (quarter - previous).days
        if not 80 <= gap <= 100:
            continue
        available = quarter + pd.Timedelta(days=FILING_LAG_DAYS)
        later = month_ends[month_ends >= available]
        if len(later) == 0:
            continue
        rows.append({
            "quarter": quarter,
            "prev_quarter": previous,
            "available_date": available,
            "rebalance_date": later[0],
        })
    return pd.DataFrame(rows)


# --- signal -------------------------------------------------------------------------


def ownership_by_quarter(ownership: pd.DataFrame,
                         marketcap: pd.DataFrame) -> pd.Series:
    """Institutional ownership FRACTION per (ticker, quarter), split-invariant.

        ownership = sum(SF3.value) / DAILY.marketcap

    Both sides are dollar amounts in USD millions struck at the same quarter-end price,
    so the ratio is the fraction of the company held by 13F filers and no share count
    enters anywhere.

    **This is erratum 1 of the pre-registration and it is the difference between a study
    and a mistake.** The registered denominator was SF1 `sharesbas`. SF1 restates share
    counts onto TODAY's split basis; SF3 `units` are as reported at the time. AAPL's
    2015-09-30 shares read 22.3 billion (its true 5.575 billion times the 4:1 split of
    August 2020) against 13F units that are unadjusted, giving 0.01% institutional
    ownership where the truth is 58%. The distortion is per name, scaled by each stock's
    own POST-SAMPLE split history, so it survives a cross-sectional z-score and it
    corrupts the quarter-on-quarter difference for exactly those names that split -- and
    stocks split after they rise. The first run of this sleeve was made on that basis and
    is void; it is reported anyway, because a discarded run is only honest if it is shown.
    """
    merged = ownership.merge(marketcap[["ticker", "calendardate", "marketcap"]],
                             on=["ticker", "calendardate"], how="inner")
    merged = merged[merged["marketcap"] > 0]
    merged["ownership"] = merged["inst_value_musd"] / merged["marketcap"]
    return merged.set_index(["ticker", "calendardate"])["ownership"]


def build_signal_panel(
    panel: pd.DataFrame,
    ownership: pd.DataFrame,
    marketcap: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (rebalance date, eligible ticker) carrying the z-scored signal.

    Eligibility is the registered universe (prereg §3): a genuinely MEASURED spread, and
    enough dollar volume that a $5,000 position stays under 1% of median daily volume.
    Names at the EDGE resolution floor are excluded outright rather than costed at the
    floor, because costing an upper bound as if it were a measurement is the bias the
    spread module exists to refuse.
    """
    fraction = ownership_by_quarter(ownership, marketcap)

    frames: list[pd.DataFrame] = []
    for row in schedule.itertuples():
        rebalance = row.rebalance_date
        cross_section = panel[
            (panel["date"] == rebalance)
            & (panel["spread_regime"] == "measured")
            & (panel["median_dollar_volume"] >= MIN_DOLLAR_VOLUME)
        ].copy()
        if cross_section.empty:
            continue

        tickers = cross_section["ticker"]
        cross_section["own_q"] = tickers.map(
            lambda t, q=row.quarter: fraction.get((t, q), np.nan)).to_numpy()
        cross_section["own_prev"] = tickers.map(
            lambda t, q=row.prev_quarter: fraction.get((t, q), np.nan)).to_numpy()

        # Differencing the FRACTION, not the share count: a 2:1 split doubles the shares
        # an institution reports holding, and a raw share-count difference would read
        # every split as enormous accumulation. Splits are not randomly distributed
        # across the cross-section -- they follow price rises.
        valid = (
            cross_section["own_q"].between(0, MAX_OWNERSHIP_RATIO, inclusive="right")
            & cross_section["own_prev"].between(0, MAX_OWNERSHIP_RATIO,
                                                inclusive="right")
        )
        cross_section = cross_section[valid].copy()
        if len(cross_section) < MIN_CROSS_SECTION:
            logger.warning("cross-section at %s has only %d names",
                           rebalance.date(), len(cross_section))
            if cross_section.empty:
                continue

        delta = cross_section["own_q"] - cross_section["own_prev"]
        low, high = delta.quantile([WINSOR_QUANTILE, 1.0 - WINSOR_QUANTILE])
        delta = delta.clip(low, high)
        spread_sd = float(delta.std(ddof=1))
        cross_section["delta_own"] = delta
        cross_section["signal"] = (
            (delta - delta.mean()) / spread_sd if spread_sd > 0 else np.nan
        )
        cross_section["quarter"] = row.quarter
        frames.append(cross_section)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- portfolio engine ----------------------------------------------------------------


def _one_way_cost(
    spread: float,
    trade_value: float,
    price: float,
    median_dollar_volume: float,
) -> tuple[float, float, float, float]:
    """One-way cost fraction, decomposed into (total, spread, impact, commission)."""
    half_spread = spread / 2.0
    impact = _impact_fraction(trade_value, median_dollar_volume)
    commission = _commission_fraction(trade_value, price)
    total = half_spread + impact + commission + FX_COST_EACH_WAY
    return total, half_spread, impact, commission


def run_portfolio(
    accrual: pd.DataFrame,
    signals: pd.DataFrame,
    select: Callable[[pd.Timestamp, pd.DataFrame, dict[str, float]], list[str]],
    delistings: pd.DataFrame,
    charge_costs: bool = True,
    n_positions: int = N_POSITIONS,
    delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
    charge_unpriced_exits: bool = False,
) -> PortfolioRun:
    """Monthly-accrual, quarterly-rebalanced long-only book.

    `accrual` is the FULL monthly panel: once a name is owned it keeps accruing its return
    even if its spread estimate stops resolving that month. Selling a name because the
    cost model lost sight of it would be an artefact of the measurement, not a decision a
    live book would take. Trades are only ever initiated on names whose spread IS measured
    (that is the selection universe), so every trade is honestly costed.

    `select` receives (date, that date's signal cross-section, current holdings) and
    returns the new holding list.
    """
    terminal = {
        row.ticker: (row.date, row.terminal_return)
        for row in delistings.itertuples()
    }

    def exit_return(ticker: str, at: pd.Timestamp) -> float:
        """Terminal return only if the delisting is what closed the position.

        `terminal.get(ticker)` alone asks whether a name delisted EVER, which books a
        2012 bankruptcy against a 2003 exit. The date window is the fix.
        """
        entry = terminal.get(ticker)
        if entry is None:
            return 0.0
        delisted_on, value = entry
        if in_window(at, delisted_on, delisting_window):
            return float(value)
        return 0.0

    rebalance_dates = pd.DatetimeIndex(signals["date"].unique()).sort_values()
    if not len(rebalance_dates):
        raise ValueError("no rebalance dates in the signal frame")
    first, last = rebalance_dates[0], accrual["date"].max()
    grid = pd.DatetimeIndex(accrual["date"].unique()).sort_values()
    months = list(grid[(grid >= first) & (grid <= last)])

    accrual_by_date = {date: frame.set_index("ticker")
                       for date, frame in accrual.groupby("date", sort=False)}
    signals_by_date = {date: frame for date, frame in signals.groupby("date",
                                                                     sort=False)}
    # Last measured spread per name, carried forward. A holding is at most three months
    # old, so the carried value is at most three months stale, and it is only ever used
    # to cost the SALE of a name that was measured when it was bought.
    last_spread: dict[str, float] = {}
    # Last observed (spread, price, median dollar volume) per name, so an exit that
    # happens after the name has left BOTH frames can still be priced. Without it the
    # leg is counted in `turnovers` and charged nothing -- a free liquidation.
    last_inputs: dict[str, tuple[float, float, float]] = {}
    unpriced_exit_legs = 0
    charged_unpriced_exit_legs = 0

    holdings: dict[str, float] = {}
    equity = [1.0]
    net_returns: list[float] = []
    gross_returns: list[float] = []
    costs: list[float] = []
    turnovers: list[float] = []
    position_counts: list[int] = []
    components = {"spread": 0.0, "impact": 0.0, "commission": 0.0}

    for date in months:
        month_costs = 0.0
        cross_section = signals_by_date.get(date)
        if cross_section is not None and not cross_section.empty:
            for row in cross_section.itertuples():
                last_spread[row.ticker] = float(row.spread)
                if np.isfinite(row.spread):
                    last_inputs[row.ticker] = (float(row.spread), float(row.close),
                                               float(row.median_dollar_volume))

            new_holdings = select(date, cross_section, holdings)
            traded = set(new_holdings) ^ set(holdings)
            turnovers.append(len(traded) / max(len(new_holdings), 1))

            weight = 1.0 / max(len(new_holdings), 1)
            if charge_costs:
                reference = cross_section.set_index("ticker")
                for ticker in traded:
                    if ticker in reference.index:
                        row = reference.loc[ticker]
                        price = float(row["close"])
                        mdv = float(row["median_dollar_volume"])
                        spread = float(row["spread"])
                    else:
                        # A name being SOLD that is no longer in the selection universe.
                        held = accrual_by_date.get(date)
                        spread = last_spread.get(ticker, np.nan)
                        if (held is not None and ticker in held.index
                                and np.isfinite(spread)):
                            row = held.loc[ticker]
                            price = float(row["close"])
                            mdv = float(row["median_dollar_volume"])
                        else:
                            # It has left BOTH frames, or was never measured. The leg is
                            # still counted in `turnovers`, so skipping it here is a FREE
                            # liquidation. Price it at the last observed inputs when
                            # asked to; either way, count it.
                            unpriced_exit_legs += 1
                            if not (charge_unpriced_exits and ticker in last_inputs):
                                continue
                            spread, price, mdv = last_inputs[ticker]
                            charged_unpriced_exit_legs += 1
                    trade_value = BOOK_SIZE * weight
                    total, half, impact, commission = _one_way_cost(
                        spread, trade_value, price, mdv)
                    month_costs += weight * total
                    components["spread"] += weight * half
                    components["impact"] += weight * impact
                    components["commission"] += weight * commission
            holdings = {t: weight for t in new_holdings}

        position_counts.append(len(holdings))

        month_frame = accrual_by_date.get(date)
        realised: list[tuple[float, float]] = []
        closing: list[str] = []
        for ticker, weight in holdings.items():
            if month_frame is None or ticker not in month_frame.index:
                # Left the panel entirely. Book the exit ONCE and drop it: leaving it in
                # `holdings` re-books the same terminal return every month thereafter,
                # which is how a long-only book "loses" 112% a year.
                realised.append((weight, exit_return(ticker, date)))
                closing.append(ticker)
                continue
            forward = month_frame.at[ticker, "forward_return"]
            if isinstance(forward, pd.Series):
                forward = float(forward.iloc[0])
            if np.isfinite(forward):
                realised.append((weight, float(np.clip(forward,
                                                       -FORWARD_RETURN_CAP,
                                                       FORWARD_RETURN_CAP))))
                continue
            realised.append((weight, exit_return(ticker, date)))
            closing.append(ticker)

        for ticker in closing:
            holdings.pop(ticker, None)

        gross = max(sum(w * r for w, r in realised), -1.0)
        net = max(gross - month_costs, -1.0)
        gross_returns.append(gross)
        costs.append(month_costs)
        net_returns.append(net)
        equity.append(equity[-1] * (1.0 + net))

    return PortfolioRun(
        dates=list(months),
        net_returns=np.array(net_returns),
        gross_returns=np.array(gross_returns),
        costs=np.array(costs),
        equity=np.array(equity),
        turnovers=turnovers,
        position_counts=position_counts,
        cost_components=components,
        unpriced_exit_legs=unpriced_exit_legs,
        charged_unpriced_exit_legs=charged_unpriced_exit_legs,
    )


def top_n_selector(n_positions: int) -> Callable:
    """Top-N by signal with the capacity study's no-trade band.

    A held name is sold only when it drops out of the top 30%, not merely because it left
    the top decile. Turnover, not signal strength, is what killed the prior programme's
    construction, so the band is inherited rather than re-derived.
    """

    def select(date: pd.Timestamp, cross_section: pd.DataFrame,
               holdings: dict[str, float]) -> list[str]:
        ranked = cross_section.dropna(subset=["signal"])
        if ranked.empty:
            return list(holdings)
        pct = ranked["signal"].rank(pct=True)
        entry = ranked.loc[pct >= ENTRY_QUANTILE].sort_values("signal",
                                                              ascending=False)
        hold_ok = set(ranked.loc[pct >= EXIT_QUANTILE, "ticker"])

        kept = [t for t in holdings if t in hold_ok]
        room = max(n_positions - len(kept), 0)
        additions = [t for t in entry["ticker"] if t not in kept][:room]
        new_holdings = kept + additions
        if not new_holdings:
            new_holdings = list(entry["ticker"])[:n_positions]
        return new_holdings

    return select


def equal_weight_universe_selector() -> Callable:
    """Buy-and-hold the whole eligible universe: the benchmark of prereg §6."""

    def select(date: pd.Timestamp, cross_section: pd.DataFrame,
               holdings: dict[str, float]) -> list[str]:
        return list(cross_section["ticker"])

    return select


# --- diagnostics ---------------------------------------------------------------------


def forward_horizon_return(
    panel: pd.DataFrame,
    delistings: pd.DataFrame,
    months: int = HOLDING_MONTHS,
    delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
) -> pd.DataFrame:
    """Realised `months`-ahead total return per (ticker, date), delisting-aware.

    Names that stop trading inside the window get their booked terminal return rather than
    being dropped: dropping them would quietly remove the bankruptcies from the IC, which
    is a survivorship bias in the direction that flatters any signal correlated with
    distress.
    """
    frame = panel.sort_values(["ticker", "date"]).copy()
    grouped = frame.groupby("ticker", sort=False)
    future = grouped["closeadj"].shift(-months)
    future_date = grouped["date"].shift(-months)
    horizon = future / frame["closeadj"] - 1.0

    # Only count it as a genuine `months`-ahead return if the future bar really is that
    # far ahead; a gap in the panel would otherwise silently become a longer horizon.
    span = (future_date - frame["date"]).dt.days
    horizon = horizon.where(span.between(months * 25, months * 37))

    terminal = delistings.set_index("ticker")
    missing = horizon.isna()
    if missing.any():
        subset = frame.loc[missing, ["ticker", "date"]]
        delist_date = subset["ticker"].map(terminal["date"])
        delist_value = subset["ticker"].map(terminal["terminal_return"])
        # The horizon window is `months` ahead plus one month of grace, so only its
        # LOWER edge comes from `delisting_window`; the upper one is horizon-specific.
        window = in_window_mask(subset["date"], delist_date,
                                (delisting_window[0], months * 31 + 31))
        horizon.loc[missing] = np.where(window, delist_value, np.nan)

    frame["forward_horizon_return"] = np.clip(horizon, -FORWARD_RETURN_CAP,
                                              FORWARD_RETURN_CAP)
    return frame[["ticker", "date", "forward_horizon_return"]]


def information_coefficient(signals: pd.DataFrame,
                            column: str) -> tuple[pd.Series, float, float, float, int]:
    """Per-cross-section Spearman IC, plus its mean, standard error and t-statistic."""
    by_date: dict[pd.Timestamp, float] = {}
    for date, frame in signals.groupby("date", sort=True):
        usable = frame.dropna(subset=["signal", column])
        if len(usable) < MIN_CROSS_SECTION:
            continue
        rho = stats.spearmanr(usable["signal"], usable[column]).statistic
        if np.isfinite(rho):
            by_date[date] = float(rho)

    series = pd.Series(by_date).sort_index()
    if len(series) < 2:
        return series, float("nan"), float("nan"), float("nan"), len(series)
    mean = float(series.mean())
    std_error = float(series.std(ddof=1) / np.sqrt(len(series)))
    t_stat = mean / std_error if std_error > 0 else float("nan")
    return series, mean, std_error, t_stat, len(series)


def long_short_spread(signals: pd.DataFrame, column: str) -> tuple[float, float]:
    """Top-decile minus bottom-decile equal-weight return, gross of borrow.

    NOT deployable: small-cap borrow is neither available nor costed here. It is reported
    only as a second read on whether the signal orders returns at all.
    """
    periods: list[float] = []
    for _, frame in signals.groupby("date", sort=True):
        usable = frame.dropna(subset=["signal", column])
        if len(usable) < MIN_CROSS_SECTION:
            continue
        pct = usable["signal"].rank(pct=True)
        top = usable.loc[pct >= 0.90, column].mean()
        bottom = usable.loc[pct <= 0.10, column].mean()
        if np.isfinite(top) and np.isfinite(bottom):
            periods.append(float(top - bottom))
    if len(periods) < 2:
        return float("nan"), float("nan")
    values = np.array(periods)
    annual = float(values.mean() * REBALANCES_PER_YEAR)
    vol = float(values.std(ddof=1) * np.sqrt(REBALANCES_PER_YEAR))
    return annual, (annual / vol if vol > 0 else float("nan"))
