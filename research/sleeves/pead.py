"""Post-earnings-announcement drift — signal, universe, cost model and book.

Registered design: `research/sleeves/pead_prereg.md`, written before this module was
run. Every constant below is the registered one; none may be adjusted after a result is
seen.

WHY this sleeve exists at all: `docs/project-control/specs/2026-07-28-the-breadth-lever.md`
argues the programme has been improving IC while leaving sqrt(BR) at its floor — ten
studies, all at one cross-section per month. PEAD is the cheapest available test of the
opposite regime, because entries are triggered by filings landing continuously rather
than by a rebalance grid. The point is to measure realised breadth AND what it costs to
harvest, not to assume breadth is free.

WHY the entry is at the close of datekey+1: `datekey` is the SEC filing date, and a
filing can be accepted after the close. Acting on the filing date is therefore a
look-ahead. Entering at the close of the NEXT session also forgoes the whole
announcement-day jump, which is the direction that understates the strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR, load_prices
from research.spread_estimation import spread_with_resolution

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# REGISTERED CONFIGURATION (prereg sections 3-7). Frozen before the run.
# ---------------------------------------------------------------------------

SCREEN_WINDOW = 63          # trailing bars for price/liquidity/spread, prereg s4
SEASONAL_LAG = 4            # quarters, prereg s3
SUE_HISTORY = 8             # trailing seasonal differences used for the denominator
SUE_MIN_HISTORY = 6         # minimum non-null differences required
SUE_MIN_DENOM = 0.01        # $0.01: below this the denominator is noise, not dispersion
SEASONAL_GAP_DAYS = (330, 400)
FILING_LAG_DAYS = (0, 180)  # datekey - calendardate sanity band

TOP_DECILE = 0.90           # prereg s1: buy the top surprise decile
BREAKPOINT_MONTHS = 12      # trailing window for the PIT decile breakpoint

MIN_PRICE = 2.00
MIN_TRADING_FRACTION = 0.90
MIN_DOLLAR_VOLUME = 5e4     # capacity_panel.BANDS lowest floor
RETURN_CAP = 1.00

HOLDING_HORIZONS = (20, 40, 60)   # ALL THREE reported, prereg s5

START_CAPITAL = 1_000_000.0
MAX_POSITION_FRACTION = 1.0 / 200.0

COMMISSION_PER_SHARE = 0.0035
COMMISSION_MINIMUM = 0.35
COMMISSION_CAP_FRACTION = 0.01
IMPACT_COEFFICIENT = 1.0    # square-root law, coefficient on daily volatility

DELISTING_WINDOW_DAYS = 62  # prereg s7 — the guard against the 2003/2012 mis-booking

MIN_ENTRY_DATE = pd.Timestamp("1998-04-01")


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def load_sf1_arq(cutoff: pd.Timestamp = DEV_CUTOFF) -> pd.DataFrame:
    """SF1 ARQ filings visible inside the DEV window.

    The DEV guard is on ``datekey`` (the SEC filing date) rather than ``calendardate``,
    because ``datekey`` is when the strategy could act. A quarter ending inside the
    window but filed after it is correctly excluded.
    """
    path = PANEL_DIR / "sf1_arq_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/build_pead_inputs.py first"
        )
    sf1 = pd.read_parquet(path)
    for column in ("calendardate", "datekey", "reportperiod"):
        sf1[column] = pd.to_datetime(sf1[column])
    sf1 = sf1[sf1["datekey"] <= cutoff]

    lag = (sf1["datekey"] - sf1["calendardate"]).dt.days
    sf1 = sf1[lag.between(*FILING_LAG_DAYS)]

    # Restatements share a calendardate. The ORIGINAL filing is the announcement; a
    # later restatement is information the market did not have on the day.
    sf1 = sf1.sort_values(["ticker", "calendardate", "datekey"])
    sf1 = sf1.drop_duplicates(["ticker", "calendardate"], keep="first")
    return sf1.reset_index(drop=True)


def build_sue(sf1: pd.DataFrame) -> pd.DataFrame:
    """Standardised unexpected earnings per filing.

    SUE = (eps_t - eps_{t-4}) / stdev of the trailing 8 seasonal differences. Both
    numerator and denominator are in the same units, so the ratio is invariant to any
    per-ticker rescaling — which is exactly what Sharadar's retroactive split adjustment
    of per-share fields is. A split therefore cannot manufacture a surprise here.
    """
    frame = sf1.copy()
    eps = frame["eps"].astype(float)
    fallback = frame["netinc"].astype(float) / frame["shareswa"].replace(0.0, np.nan)
    frame["eps_used"] = eps.where(eps.notna(), fallback)

    grouped = frame.groupby("ticker", sort=False)
    frame["eps_lag4"] = grouped["eps_used"].shift(SEASONAL_LAG)
    gap = (frame["calendardate"]
           - grouped["calendardate"].shift(SEASONAL_LAG)).dt.days
    # A missing quarter would silently turn a 4-quarter lag into a 5-quarter one and
    # compare Q3 against Q2, which is a seasonality artefact rather than a surprise.
    valid_gap = gap.between(*SEASONAL_GAP_DAYS)
    frame["seasonal_diff"] = np.where(valid_gap, frame["eps_used"] - frame["eps_lag4"],
                                      np.nan)

    diffs = frame.groupby("ticker", sort=False)["seasonal_diff"]
    # shift(1) so the denominator uses only quarters STRICTLY BEFORE the one being
    # scored; including the current difference would leak the answer into its own scale.
    prior = diffs.shift(1)
    frame["diff_std"] = prior.groupby(frame["ticker"], sort=False).rolling(
        SUE_HISTORY, min_periods=SUE_MIN_HISTORY
    ).std(ddof=1).reset_index(level=0, drop=True)

    usable = (
        frame["seasonal_diff"].notna()
        & frame["diff_std"].notna()
        & (frame["diff_std"] >= SUE_MIN_DENOM)
    )
    frame = frame[usable].copy()
    frame["sue"] = frame["seasonal_diff"] / frame["diff_std"]
    return frame[["ticker", "calendardate", "datekey", "eps_used", "seasonal_diff",
                  "diff_std", "sue"]].reset_index(drop=True)


def decile_breakpoints(sue: pd.DataFrame,
                       quantile: float = TOP_DECILE,
                       months: int = BREAKPOINT_MONTHS) -> pd.Series:
    """PIT top-decile breakpoint for each entry month.

    The breakpoint applied in month *m* is computed from filings made in the 12 months
    ending at the close of month *m-1*. Using the contemporaneous cross-section would
    require knowing every filing of the month before the month is over.
    """
    monthly = sue.set_index("datekey")["sue"].sort_index()
    by_month = monthly.groupby(pd.Grouper(freq="ME"))

    values: dict[pd.Timestamp, np.ndarray] = {
        stamp: group.to_numpy() for stamp, group in by_month
    }
    stamps = sorted(values)
    out: dict[pd.Timestamp, float] = {}
    for index, stamp in enumerate(stamps):
        window = stamps[max(0, index - months + 1): index + 1]
        pool = np.concatenate([values[s] for s in window]) if window else np.array([])
        if pool.size < 100:
            continue
        # Keyed by the month the breakpoint becomes USABLE, i.e. the following month.
        usable_from = (stamp + pd.offsets.MonthBegin(1)).normalize()
        out[usable_from] = float(np.quantile(pool, quantile))
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Universe screen — computed at each candidate entry, on trailing bars only
# ---------------------------------------------------------------------------

@dataclass
class TickerBars:
    """Per-ticker daily arrays, aligned to a global trading-day index."""

    day_index: np.ndarray      # position in the global trading calendar
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    closeadj: np.ndarray
    volume: np.ndarray
    dollar_volume: np.ndarray


def build_ticker_bars(prices: pd.DataFrame,
                      calendar: pd.DatetimeIndex) -> dict[str, TickerBars]:
    """Split the price panel into per-ticker arrays indexed against ``calendar``."""
    day_lookup = pd.Series(np.arange(len(calendar)), index=calendar)
    prices = prices.copy()
    prices["day"] = day_lookup.reindex(prices["date"]).to_numpy()

    bars: dict[str, TickerBars] = {}
    for ticker, frame in prices.groupby("ticker", sort=False):
        bars[ticker] = TickerBars(
            day_index=frame["day"].to_numpy(np.int32),
            open_=frame["open"].to_numpy(np.float64),
            high=frame["high"].to_numpy(np.float64),
            low=frame["low"].to_numpy(np.float64),
            close=frame["close"].to_numpy(np.float64),
            closeadj=frame["closeadj"].to_numpy(np.float64),
            volume=frame["volume"].to_numpy(np.float64),
            dollar_volume=frame["dollar_volume"].to_numpy(np.float64),
        )
    return bars


@dataclass
class Screen:
    """Outcome of the registered universe screen at one candidate entry."""

    passed: bool
    reason: str
    spread: float = float("nan")
    daily_vol: float = float("nan")
    median_dollar_volume: float = float("nan")
    entry_row: int = -1


def screen_at_filing(bars: TickerBars, filing_day: int) -> Screen:
    """Apply prereg s4 using only bars on or before ``filing_day``.

    ``filing_day`` is the global index of the announcement date. The screen window ends
    at the last bar <= filing_day; the entry is the first bar strictly after it.
    """
    positions = bars.day_index
    # Last bar at or before the filing.
    last = int(np.searchsorted(positions, filing_day, side="right")) - 1
    if last < SCREEN_WINDOW - 1:
        return Screen(False, "insufficient_history")
    entry_row = last + 1
    if entry_row >= len(positions):
        return Screen(False, "no_bar_after_filing")

    start = last - SCREEN_WINDOW + 1
    window_high = bars.high[start:last + 1]
    window_low = bars.low[start:last + 1]
    window_volume = bars.volume[start:last + 1]
    window_close = bars.close[start:last + 1]
    window_open = bars.open_[start:last + 1]
    window_dv = bars.dollar_volume[start:last + 1]

    if float(bars.close[last]) < MIN_PRICE:
        return Screen(False, "price_floor")

    traded = (window_high > window_low) & (window_volume > 0)
    if float(traded.mean()) < MIN_TRADING_FRACTION:
        return Screen(False, "thin_trading")

    median_dv = float(np.median(window_dv))
    if not np.isfinite(median_dv) or median_dv < MIN_DOLLAR_VOLUME:
        return Screen(False, "illiquid")

    spread, regime = spread_with_resolution(window_open, window_high, window_low,
                                            window_close)
    if regime != "measured":
        # prereg s4.4: an unresolved spread means the cost is unknown, and an unknown
        # cost costed at the floor is the free lunch this programme exists to refuse.
        return Screen(False, f"spread_{regime}")

    log_returns = np.diff(np.log(window_close[window_close > 0]))
    daily_vol = float(np.std(log_returns, ddof=1)) if log_returns.size > 1 else np.nan
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        return Screen(False, "no_volatility")

    return Screen(True, "ok", spread=spread, daily_vol=daily_vol,
                  median_dollar_volume=median_dv, entry_row=entry_row)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@dataclass
class PositionSet:
    """Every position the registered rule opens, with its realised daily path.

    Stored column-wise because the book simulation walks the trading calendar and needs
    day-indexed slices, not per-object attribute access. The daily marks live in three
    flat parallel arrays (``mark_day`` / ``mark_position`` / ``mark_return``) rather than
    per-position lists: the book has to look up "every position that moved on day d",
    which is a sort-and-slice on a flat array and a dictionary walk otherwise.
    """

    ticker: list[str] = field(default_factory=list)
    entry_day: list[int] = field(default_factory=list)
    exit_day: list[int] = field(default_factory=list)
    entry_price: list[float] = field(default_factory=list)
    exit_price: list[float] = field(default_factory=list)
    gross_return: list[float] = field(default_factory=list)
    spread: list[float] = field(default_factory=list)
    daily_vol: list[float] = field(default_factory=list)
    median_dollar_volume: list[float] = field(default_factory=list)
    sue: list[float] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    delisted: list[bool] = field(default_factory=list)

    mark_day: list[np.ndarray] = field(default_factory=list)
    mark_return: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ticker)

    def to_frame(self) -> pd.DataFrame:
        skip = {"mark_day", "mark_return"}
        return pd.DataFrame({name: getattr(self, name)
                             for name in self.__annotations__ if name not in skip})

    def flat_marks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(day, position index, daily return) for every position-day, sorted by day."""
        if not self.mark_day:
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_i, np.empty(0, dtype=np.float64)
        days = np.concatenate(self.mark_day)
        rets = np.concatenate(self.mark_return)
        owner = np.repeat(np.arange(len(self.mark_day), dtype=np.int64),
                          [len(a) for a in self.mark_day])
        order = np.argsort(days, kind="stable")
        return days[order], owner[order], rets[order]


def build_positions(
    signals: pd.DataFrame,
    bars: dict[str, TickerBars],
    calendar: pd.DatetimeIndex,
    terminal: dict[str, tuple[pd.Timestamp, float]],
    horizon: int,
) -> tuple[PositionSet, dict[str, int]]:
    """Open one position per qualifying filing and walk it to its exit.

    ``terminal`` maps ticker -> (delisting date, terminal return). It is consulted ONLY
    when the price series ends early, and only when the delisting falls inside
    ``DELISTING_WINDOW_DAYS`` after the last traded bar. Both halves of that rule are the
    two defects recorded in capacity_curve_result.md s4: a name that merely leaves the
    universe is not a delisting, and a name that IS delisted must be booked once and
    removed, not re-booked forever.
    """
    positions = PositionSet()
    rejects: dict[str, int] = {}
    last_day = len(calendar) - 1

    for row in signals.itertuples(index=False):
        ticker = row.ticker
        frame = bars.get(ticker)
        if frame is None:
            rejects["no_price_data"] = rejects.get("no_price_data", 0) + 1
            continue

        screen = screen_at_filing(frame, int(row.filing_day))
        if not screen.passed:
            rejects[screen.reason] = rejects.get(screen.reason, 0) + 1
            continue

        entry_row = screen.entry_row
        entry_day = int(frame.day_index[entry_row])
        if calendar[entry_day] < MIN_ENTRY_DATE:
            rejects["before_start"] = rejects.get("before_start", 0) + 1
            continue

        planned_exit_day = entry_day + horizon
        if planned_exit_day > last_day:
            # The horizon would run past the DEV cutoff. Truncating it instead would
            # quietly shorten the holding period at the end of the sample.
            rejects["horizon_past_cutoff"] = rejects.get("horizon_past_cutoff", 0) + 1
            continue

        exit_row = entry_row + horizon
        truncated = exit_row >= len(frame.day_index)
        if truncated:
            exit_row = len(frame.day_index) - 1
        if exit_row <= entry_row:
            rejects["no_holding_bars"] = rejects.get("no_holding_bars", 0) + 1
            continue

        path_adj = frame.closeadj[entry_row:exit_row + 1].astype(np.float64)
        if not np.all(np.isfinite(path_adj)) or np.any(path_adj <= 0):
            rejects["bad_adjusted_price"] = rejects.get("bad_adjusted_price", 0) + 1
            continue

        path_returns = path_adj[1:] / path_adj[:-1] - 1.0
        mark_days = frame.day_index[entry_row + 1:exit_row + 1].astype(np.int64)
        exit_day = int(frame.day_index[exit_row])

        delisted = False
        if truncated and ticker in terminal:
            event_date, terminal_return = terminal[ticker]
            gap = (event_date - calendar[exit_day]).days
            if 0 <= gap <= DELISTING_WINDOW_DAYS:
                # Booked ONCE, on the exit day, and the position is then removed from
                # the book by simulate_book. Both halves matter — see the docstring.
                path_returns = path_returns.copy()
                path_returns[-1] = (1.0 + path_returns[-1]) * (1.0 + terminal_return) - 1.0
                delisted = True

        gross_uncapped = float(np.prod(1.0 + path_returns) - 1.0)
        gross = float(np.clip(gross_uncapped, -RETURN_CAP, RETURN_CAP))
        if gross != gross_uncapped and gross_uncapped > -1.0 and gross > -1.0:
            # Rescale the whole path uniformly so the marked-to-market curve compounds
            # to the capped figure. Capping only the endpoint would leave a daily path
            # that disagrees with the return the book actually books.
            exponent = np.log1p(gross) / np.log1p(gross_uncapped)
            path_returns = np.expm1(exponent * np.log1p(path_returns))
        elif gross <= -1.0:
            path_returns = np.full_like(path_returns, 0.0)
            path_returns[-1] = -1.0

        positions.mark_day.append(mark_days)
        positions.mark_return.append(path_returns)
        positions.ticker.append(ticker)
        positions.entry_day.append(entry_day)
        positions.exit_day.append(exit_day)
        positions.entry_price.append(float(frame.close[entry_row]))
        positions.exit_price.append(float(frame.close[exit_row]))
        positions.gross_return.append(gross)
        positions.spread.append(screen.spread)
        positions.daily_vol.append(screen.daily_vol)
        positions.median_dollar_volume.append(screen.median_dollar_volume)
        positions.sue.append(float(row.sue))
        positions.truncated.append(bool(truncated))
        positions.delisted.append(delisted)

    return positions, rejects


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def trade_cost_fraction(notional: float, price: float, spread: float,
                        daily_vol: float, median_dollar_volume: float) -> float:
    """Per-side cost as a fraction of notional (prereg s6).

    Three components, all of which bind somewhere in this universe: the half-spread
    dominates for small illiquid names, the square-root impact term dominates once a
    position is a meaningful slice of a day's volume, and the $0.35 order minimum
    dominates for very small tickets.
    """
    if notional <= 0 or price <= 0:
        return 0.0
    half_spread = 0.5 * spread
    participation = notional / median_dollar_volume if median_dollar_volume > 0 else 1.0
    impact = IMPACT_COEFFICIENT * daily_vol * float(np.sqrt(participation))
    shares = notional / price
    commission = min(max(COMMISSION_PER_SHARE * shares, COMMISSION_MINIMUM),
                     COMMISSION_CAP_FRACTION * notional)
    return half_spread + impact + commission / notional


# ---------------------------------------------------------------------------
# Book simulation
# ---------------------------------------------------------------------------

@dataclass
class BookResult:
    equity: pd.Series
    exposure_start: pd.Series   # position notional carried INTO each day
    entries: int
    total_bought: float
    total_sold: float
    mean_concurrent: float
    mean_cash_weight: float
    mean_position_notional: float
    entry_days: int
    total_cost: float
    open_at_end: int


def simulate_book(positions: PositionSet, calendar: pd.DatetimeIndex,
                  costs_on: bool = True) -> BookResult:
    """Walk the trading calendar holding every position the rule opened.

    Accounting is cash-explicit: equity = cash + positions marked to market EVERY day.
    Marking only at exit would compress a 60-day position's whole P&L into one month and
    make the monthly volatility — and therefore the Sharpe — meaningless.

    WHY new positions are scaled down rather than skipped when cash is short: skipping
    would silently truncate realised breadth, which is the one quantity this study
    exists to measure.
    """
    count = len(positions)
    entry_day = np.asarray(positions.entry_day, dtype=np.int64)
    exit_day = np.asarray(positions.exit_day, dtype=np.int64)
    entry_price = np.asarray(positions.entry_price, dtype=np.float64)
    exit_price = np.asarray(positions.exit_price, dtype=np.float64)
    spread = np.asarray(positions.spread, dtype=np.float64)
    daily_vol = np.asarray(positions.daily_vol, dtype=np.float64)
    mdv = np.asarray(positions.median_dollar_volume, dtype=np.float64)

    mark_days, mark_owner, mark_return = positions.flat_marks()
    mark_starts = np.searchsorted(mark_days, np.arange(len(calendar)), side="left")
    mark_ends = np.searchsorted(mark_days, np.arange(len(calendar)), side="right")

    entries_by_day: dict[int, list[int]] = {}
    for index in np.argsort(entry_day, kind="stable"):
        entries_by_day.setdefault(int(entry_day[index]), []).append(int(index))
    exits_by_day: dict[int, list[int]] = {}
    for index in range(count):
        exits_by_day.setdefault(int(exit_day[index]), []).append(index)

    notional = np.zeros(count)
    is_open = np.zeros(count, dtype=bool)
    cash = START_CAPITAL
    invested = 0.0
    open_count = 0

    equity_path = np.empty(len(calendar))
    exposure_start = np.empty(len(calendar))
    concurrent = np.empty(len(calendar))
    cash_weight = np.empty(len(calendar))
    total_bought = total_sold = total_cost = 0.0
    notionals_used: list[float] = []

    for day in range(len(calendar)):
        # Exposure carried INTO the day, before anything moves. This is the denominator
        # for the sleeve's return on invested capital, which is the statistic that
        # separates signal quality from how much cash the sizing rule happened to leave
        # idle.
        exposure_start[day] = invested

        # --- mark open positions to market.
        start, end = mark_starts[day], mark_ends[day]
        if end > start:
            owners = mark_owner[start:end]
            before = notional[owners]
            after = before * (1.0 + mark_return[start:end])
            notional[owners] = after
            invested += float((after - before).sum())

        # --- exits settle next: they free the cash the day's entries can use.
        for index in exits_by_day.get(day, ()):
            if not is_open[index]:
                continue
            proceeds = notional[index]
            if costs_on and proceeds > 0:
                cost = trade_cost_fraction(proceeds, exit_price[index], spread[index],
                                           daily_vol[index], mdv[index])
                total_cost += proceeds * cost
                proceeds *= (1.0 - cost)
            cash += proceeds
            invested -= notional[index]
            total_sold += notional[index]
            # Removed from the book here, permanently. A position that has paid out must
            # never be marked again — the -112%/yr defect in capacity_curve_result.md s4.
            notional[index] = 0.0
            is_open[index] = False
            open_count -= 1

        # --- entries at the close.
        todays = entries_by_day.get(day, ())
        if todays:
            equity_now = cash + invested
            target = MAX_POSITION_FRACTION * equity_now
            per_position = min(target, cash / len(todays)) if cash > 0 else 0.0
            if per_position > 0:
                for index in todays:
                    size = per_position
                    if costs_on:
                        cost = trade_cost_fraction(size, entry_price[index],
                                                   spread[index], daily_vol[index],
                                                   mdv[index])
                        total_cost += size * cost
                        notional[index] = size * (1.0 - cost)
                    else:
                        notional[index] = size
                    cash -= size
                    invested += notional[index]
                    is_open[index] = True
                    total_bought += size
                    notionals_used.append(size)
                    open_count += 1

        equity_path[day] = cash + invested
        concurrent[day] = open_count
        cash_weight[day] = cash / equity_path[day] if equity_path[day] > 0 else 1.0

    equity = pd.Series(equity_path, index=calendar, name="equity")
    return BookResult(
        equity=equity,
        exposure_start=pd.Series(exposure_start, index=calendar, name="exposure"),
        entries=count,
        total_bought=total_bought,
        total_sold=total_sold,
        mean_concurrent=float(np.mean(concurrent)),
        mean_cash_weight=float(np.mean(cash_weight)),
        mean_position_notional=float(np.mean(notionals_used)) if notionals_used else 0.0,
        entry_days=len(entries_by_day),
        total_cost=total_cost,
        open_at_end=int(is_open.sum()),
    )


# ---------------------------------------------------------------------------
# Benchmark and statistics
# ---------------------------------------------------------------------------

def universe_benchmark(tickers: set[str], start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.Series:
    """Equal-weight, monthly-rebalanced, zero-cost buy-and-hold of the OWN universe.

    Uses exactly the eligibility the sleeve uses (``spread_regime == 'measured'``, which
    the panel only assigns to cells that already cleared the price floor and the
    trading-fraction test) restricted to names with SF1 ARQ coverage. Delisting terminal
    returns are booked on the same 62-day rule, so the benchmark carries the same
    survivorship treatment as the strategy rather than a flattering one.
    """
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    panel = panel[(panel["spread_regime"] == "measured")
                  & panel["ticker"].isin(tickers)
                  & panel["date"].between(start, end)].copy()

    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    terminal = {row.ticker: (row.date, row.terminal_return)
                for row in delistings.itertuples(index=False)}

    forward = panel["forward_return"].to_numpy(np.float64)
    missing = ~np.isfinite(forward)
    if missing.any():
        replacement = np.full(missing.sum(), np.nan)
        subset = panel.loc[missing, ["ticker", "date"]].itertuples(index=False)
        for slot, item in enumerate(subset):
            event = terminal.get(item.ticker)
            if event is None:
                continue
            gap = (event[0] - item.date).days
            if 0 <= gap <= DELISTING_WINDOW_DAYS:
                replacement[slot] = event[1]
        forward[missing] = replacement

    panel["forward_return"] = np.clip(forward, -RETURN_CAP, RETURN_CAP)
    panel = panel[np.isfinite(panel["forward_return"])]

    # Group by calendar MONTH, not by the raw date. The panel's date is each ticker's
    # OWN last bar of the month, so a name that stops trading on the 12th carries a
    # mid-month date. Grouping on the raw date puts that name in a singleton "month" of
    # its own, and once its delisting return is booked that singleton is a -100% month
    # in the benchmark series. That is how this function first returned -100%/yr.
    month = panel["date"].dt.to_period("M")
    monthly = panel.groupby(month)["forward_return"].mean().sort_index()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "date"
    return monthly


@dataclass
class Stats:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    months: int


def summarise(monthly_returns: pd.Series, daily_equity: pd.Series | None = None
              ) -> Stats:
    """Annualised statistics from MONTHLY returns; drawdown from the daily curve.

    Sharpe is return/volatility with no risk-free deduction, matching the convention in
    `capacity_curve_result.md` so the two studies are directly comparable.
    """
    returns = monthly_returns.dropna()
    if returns.empty:
        return Stats(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    total_growth = float(np.prod(1.0 + returns.to_numpy()))
    years = len(returns) / 12.0
    annual_return = total_growth ** (1.0 / years) - 1.0 if total_growth > 0 else -1.0
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(12.0))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else float("nan")

    curve = daily_equity if daily_equity is not None else (1.0 + returns).cumprod()
    peak = curve.cummax()
    max_drawdown = float((1.0 - curve / peak).max())
    return Stats(annual_return, annual_volatility, sharpe, max_drawdown, len(returns))


def monthly_from_equity(equity: pd.Series) -> pd.Series:
    """Month-end equity to monthly simple returns."""
    month_end = equity.resample("ME").last()
    return month_end.pct_change().dropna()


__all__ = [
    "HOLDING_HORIZONS",
    "BookResult",
    "PositionSet",
    "Stats",
    "build_positions",
    "build_sue",
    "build_ticker_bars",
    "decile_breakpoints",
    "load_prices",
    "load_sf1_arq",
    "monthly_from_equity",
    "screen_at_filing",
    "simulate_book",
    "summarise",
    "trade_cost_fraction",
    "universe_benchmark",
]
