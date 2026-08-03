"""PEAD re-tested on the CORRECTED universe, priced under both cost bounds.

Registered design: `research/sleeves/pead_retest_prereg.md`, written before this module
was run. n_trials 32 -> 33.

Exactly two things differ from `research/sleeves/pead.py`, and both are corrections to the
measurement apparatus rather than to the strategy:

  1. **The universe.** Iteration 1 excluded every name whose EDGE spread regime was
     ``upper_bound``. That deleted 525,933 of 922,652 eligible (name, month) cells -- the
     CHEAP half of the tape, at 6.4x the dollar volume and 0.24x the spread of the ones
     kept. ``upper_bound`` means the true spread lies BELOW the estimator's resolution
     floor, i.e. the name is cheap, not unknown. Those names are admitted here. Only
     ``unmeasurable`` is still excluded.
  2. **Cost is a bracket.** Every position is priced twice via
     `research.spread_estimation.bounds_from_estimate`: (a) conservative charges the EDGE
     estimate, which overstates, so a pass there is REAL; (b) realistic charges the
     documented Ardia-Guidotti-Kroencke liquid-name schedule, so a failure there is DEAD.

Everything else -- signal, screens, sizing, commissions, impact coefficient, delisting
rule, horizons -- is the iteration-1 constant, deliberately untouched. The impact
coefficient in particular is NOT corrected here even though it is suspected too high:
correcting two cost terms in one run would make the result uninterpretable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.sleeves import pead
from research.spread_estimation import bounds_from_estimate, spread_with_resolution

logger = logging.getLogger(__name__)

# Registered constants are imported, never redeclared, so the two studies cannot drift.
SCREEN_WINDOW = pead.SCREEN_WINDOW
MIN_PRICE = pead.MIN_PRICE
MIN_TRADING_FRACTION = pead.MIN_TRADING_FRACTION
MIN_DOLLAR_VOLUME = pead.MIN_DOLLAR_VOLUME
RETURN_CAP = pead.RETURN_CAP
HOLDING_HORIZONS = pead.HOLDING_HORIZONS
HEADLINE_HORIZON = 40                    # prereg s6, declared before the run
START_CAPITAL = pead.START_CAPITAL
MAX_POSITION_FRACTION = pead.MAX_POSITION_FRACTION
COMMISSION_PER_SHARE = pead.COMMISSION_PER_SHARE
COMMISSION_MINIMUM = pead.COMMISSION_MINIMUM
COMMISSION_CAP_FRACTION = pead.COMMISSION_CAP_FRACTION
IMPACT_COEFFICIENT = pead.IMPACT_COEFFICIENT
DELISTING_WINDOW_DAYS = pead.DELISTING_WINDOW_DAYS
MIN_ENTRY_DATE = pead.MIN_ENTRY_DATE
TOP_DECILE = pead.TOP_DECILE

PROMOTION_GATE_SHARPE = 0.75             # prereg s6
CONCENTRATION_ALARM = 0.03               # prereg s9.3 -- 3% of total P&L in one name-month

# The two regimes that are TRADABLE under the corrected universe. `unmeasurable` stays
# out: the schedule prices cheap names, not absent ones.
TRADABLE_REGIMES = frozenset({"measured", "upper_bound"})


# ---------------------------------------------------------------------------
# Screen -- identical to iteration 1 except at the spread step
# ---------------------------------------------------------------------------

@dataclass
class ScreenBounds:
    """Outcome of the registered universe screen, carrying BOTH cost bounds."""

    passed: bool
    reason: str
    spread_conservative: float = float("nan")
    spread_realistic: float = float("nan")
    regime: str = ""
    daily_vol: float = float("nan")
    median_dollar_volume: float = float("nan")
    entry_row: int = -1
    screen_price: float = float("nan")


def screen_at_filing(bars: pead.TickerBars, filing_day: int,
                     calendar: pd.DatetimeIndex) -> ScreenBounds:
    """Apply prereg s4 using only bars on or before ``filing_day``.

    The only change from `pead.screen_at_filing` is the spread step: an ``upper_bound``
    name is ADMITTED and priced under two bounds instead of being deleted. The era factor
    and tick regime are keyed on the ENTRY date, which is when the cost is actually paid.
    """
    positions = bars.day_index
    last = int(np.searchsorted(positions, filing_day, side="right")) - 1
    if last < SCREEN_WINDOW - 1:
        return ScreenBounds(False, "insufficient_history")
    entry_row = last + 1
    if entry_row >= len(positions):
        return ScreenBounds(False, "no_bar_after_filing")

    start = last - SCREEN_WINDOW + 1
    window_high = bars.high[start:last + 1]
    window_low = bars.low[start:last + 1]
    window_volume = bars.volume[start:last + 1]
    window_close = bars.close[start:last + 1]
    window_open = bars.open_[start:last + 1]
    window_dv = bars.dollar_volume[start:last + 1]

    screen_price = float(bars.close[last])
    if screen_price < MIN_PRICE:
        return ScreenBounds(False, "price_floor")

    traded = (window_high > window_low) & (window_volume > 0)
    if float(traded.mean()) < MIN_TRADING_FRACTION:
        return ScreenBounds(False, "thin_trading")

    median_dv = float(np.median(window_dv))
    if not np.isfinite(median_dv) or median_dv < MIN_DOLLAR_VOLUME:
        return ScreenBounds(False, "illiquid")

    estimate, regime = spread_with_resolution(window_open, window_high, window_low,
                                              window_close)
    if regime not in TRADABLE_REGIMES:
        return ScreenBounds(False, f"spread_{regime}")

    entry_date = calendar[int(bars.day_index[entry_row])]
    bounds = bounds_from_estimate(estimate, regime, median_dv, price=screen_price,
                                  when=entry_date)
    if not bounds.tradable:
        return ScreenBounds(False, "spread_untradable")
    if not (np.isfinite(bounds.conservative) and np.isfinite(bounds.realistic)):
        return ScreenBounds(False, "spread_nan")
    if bounds.realistic > bounds.conservative + 1e-12:
        # Impossible by construction. If it ever happens the two numbers did not come
        # from the same run and the result would be a wiring bug reported as a finding.
        raise ValueError("spread bracket inverted")

    log_returns = np.diff(np.log(window_close[window_close > 0]))
    daily_vol = float(np.std(log_returns, ddof=1)) if log_returns.size > 1 else np.nan
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        return ScreenBounds(False, "no_volatility")

    return ScreenBounds(True, "ok",
                        spread_conservative=float(bounds.conservative),
                        spread_realistic=float(bounds.realistic),
                        regime=regime, daily_vol=daily_vol,
                        median_dollar_volume=median_dv, entry_row=entry_row,
                        screen_price=screen_price)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@dataclass
class PositionSet:
    """Every position the registered rule opens, with both cost bounds attached."""

    ticker: list[str] = field(default_factory=list)
    entry_day: list[int] = field(default_factory=list)
    exit_day: list[int] = field(default_factory=list)
    entry_price: list[float] = field(default_factory=list)
    exit_price: list[float] = field(default_factory=list)
    gross_return: list[float] = field(default_factory=list)
    spread_conservative: list[float] = field(default_factory=list)
    spread_realistic: list[float] = field(default_factory=list)
    regime: list[str] = field(default_factory=list)
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
        if not self.mark_day:
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_i, np.empty(0, dtype=np.float64)
        days = np.concatenate(self.mark_day)
        rets = np.concatenate(self.mark_return)
        owner = np.repeat(np.arange(len(self.mark_day), dtype=np.int64),
                          [len(a) for a in self.mark_day])
        order = np.argsort(days, kind="stable")
        return days[order], owner[order], rets[order]


def screen_all(signals: pd.DataFrame, bars: dict[str, pead.TickerBars],
               calendar: pd.DatetimeIndex
               ) -> tuple[dict[tuple[str, int], ScreenBounds], dict[str, int]]:
    """Screen every candidate filing ONCE.

    The screen is horizon-independent, so computing it per horizon would run the EDGE
    estimator three times over the same 63 bars. This is a pure caching change: the
    inputs, the rule and the outputs are identical to screening inside the position
    builder.
    """
    screens: dict[tuple[str, int], ScreenBounds] = {}
    rejects: dict[str, int] = {}
    for row in signals.itertuples(index=False):
        key = (row.ticker, int(row.filing_day))
        if key in screens:
            continue
        frame = bars.get(row.ticker)
        if frame is None:
            screens[key] = ScreenBounds(False, "no_price_data")
            continue
        screens[key] = screen_at_filing(frame, int(row.filing_day), calendar)
    for screen in screens.values():
        if not screen.passed:
            rejects[screen.reason] = rejects.get(screen.reason, 0) + 1
    return screens, rejects


def build_positions(
    signals: pd.DataFrame,
    bars: dict[str, pead.TickerBars],
    calendar: pd.DatetimeIndex,
    terminal: dict[str, tuple[pd.Timestamp, float]],
    horizon: int,
    screens: dict[tuple[str, int], ScreenBounds],
) -> tuple[PositionSet, dict[str, int]]:
    """Open one position per qualifying filing and walk it to its exit.

    ``terminal`` maps ticker -> (delisting date, terminal return). It is consulted ONLY
    when the price series ends early, and only when the delisting falls inside
    ``DELISTING_WINDOW_DAYS`` after the last traded bar. The position is then removed from
    the book by `simulate_book`. Both halves are the defects that produced -60%/yr and
    -112%/yr in earlier studies.
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

        screen = screens[(ticker, int(row.filing_day))]
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
                path_returns = path_returns.copy()
                path_returns[-1] = (1.0 + path_returns[-1]) * (1.0 + terminal_return) - 1.0
                delisted = True

        gross_uncapped = float(np.prod(1.0 + path_returns) - 1.0)
        gross = float(np.clip(gross_uncapped, -RETURN_CAP, RETURN_CAP))
        if gross != gross_uncapped and gross_uncapped > -1.0 and gross > -1.0:
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
        positions.spread_conservative.append(screen.spread_conservative)
        positions.spread_realistic.append(screen.spread_realistic)
        positions.regime.append(screen.regime)
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
                        daily_vol: float, median_dollar_volume: float
                        ) -> tuple[float, bool]:
    """Per-side cost as a fraction of notional, and whether the $0.35 minimum bound.

    Identical arithmetic to `pead.trade_cost_fraction`; it additionally reports whether
    the IBKR order minimum was the binding commission term, because that is the term that
    makes a small account a different strategy (prereg s5).
    """
    if notional <= 0 or price <= 0:
        return 0.0, False
    half_spread = 0.5 * spread
    participation = notional / median_dollar_volume if median_dollar_volume > 0 else 1.0
    impact = IMPACT_COEFFICIENT * daily_vol * float(np.sqrt(participation))
    shares = notional / price
    per_share = COMMISSION_PER_SHARE * shares
    commission = min(max(per_share, COMMISSION_MINIMUM),
                     COMMISSION_CAP_FRACTION * notional)
    minimum_bound = (per_share < COMMISSION_MINIMUM
                     and COMMISSION_MINIMUM <= COMMISSION_CAP_FRACTION * notional)
    return half_spread + impact + commission / notional, minimum_bound


# ---------------------------------------------------------------------------
# Book simulation
# ---------------------------------------------------------------------------

@dataclass
class BookResult:
    equity: pd.Series
    exposure_start: pd.Series
    entries: int
    total_bought: float
    total_sold: float
    mean_concurrent: float
    mean_cash_weight: float
    mean_position_notional: float
    median_position_notional: float
    entry_days: int
    total_cost: float
    total_commission: float
    total_spread_cost: float
    total_impact_cost: float
    open_at_end: int
    orders: int
    orders_at_minimum: int
    position_pnl: np.ndarray          # realised dollars per position, entry to exit
    position_notional: np.ndarray     # notional actually deployed at entry


def simulate_book(positions: PositionSet, calendar: pd.DatetimeIndex,
                  spread: np.ndarray | None) -> BookResult:
    """Walk the trading calendar holding every position the rule opened.

    ``spread`` selects the cost bound: pass the conservative array, the realistic array,
    or ``None`` for the zero-cost gross book. Nothing else differs between the runs, so
    the difference between two books is attributable to the spread term alone.

    Accounting is cash-explicit and positions are marked to market EVERY day; marking
    only at exit would compress a 60-day position's P&L into one month and make the
    monthly volatility, and therefore the Sharpe, meaningless.
    """
    count = len(positions)
    entry_day = np.asarray(positions.entry_day, dtype=np.int64)
    exit_day = np.asarray(positions.exit_day, dtype=np.int64)
    entry_price = np.asarray(positions.entry_price, dtype=np.float64)
    exit_price = np.asarray(positions.exit_price, dtype=np.float64)
    daily_vol = np.asarray(positions.daily_vol, dtype=np.float64)
    mdv = np.asarray(positions.median_dollar_volume, dtype=np.float64)
    costs_on = spread is not None
    spread_arr = (np.asarray(spread, dtype=np.float64) if costs_on
                  else np.zeros(count, dtype=np.float64))

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
    position_pnl = np.zeros(count)
    position_notional = np.zeros(count)
    cash = START_CAPITAL
    invested = 0.0
    open_count = 0

    equity_path = np.empty(len(calendar))
    exposure_start = np.empty(len(calendar))
    concurrent = np.empty(len(calendar))
    cash_weight = np.empty(len(calendar))
    total_bought = total_sold = total_cost = 0.0
    total_commission = total_spread_cost = total_impact_cost = 0.0
    orders = orders_at_minimum = 0
    notionals_used: list[float] = []

    for day in range(len(calendar)):
        exposure_start[day] = invested

        start, end = mark_starts[day], mark_ends[day]
        if end > start:
            owners = mark_owner[start:end]
            before = notional[owners]
            after = before * (1.0 + mark_return[start:end])
            notional[owners] = after
            invested += float((after - before).sum())

        for index in exits_by_day.get(day, ()):
            if not is_open[index]:
                continue
            proceeds = notional[index]
            if costs_on and proceeds > 0:
                cost, at_minimum = trade_cost_fraction(
                    proceeds, exit_price[index], spread_arr[index],
                    daily_vol[index], mdv[index])
                total_cost += proceeds * cost
                total_spread_cost += proceeds * 0.5 * spread_arr[index]
                shares = proceeds / exit_price[index]
                commission = min(max(COMMISSION_PER_SHARE * shares, COMMISSION_MINIMUM),
                                 COMMISSION_CAP_FRACTION * proceeds)
                total_commission += commission
                total_impact_cost += proceeds * cost - proceeds * 0.5 * spread_arr[index] \
                    - commission
                orders += 1
                orders_at_minimum += int(at_minimum)
                proceeds *= (1.0 - cost)
            elif proceeds > 0:
                orders += 1
            cash += proceeds
            invested -= notional[index]
            total_sold += notional[index]
            position_pnl[index] += proceeds
            # Removed from the book here, permanently.
            notional[index] = 0.0
            is_open[index] = False
            open_count -= 1

        todays = entries_by_day.get(day, ())
        if todays:
            equity_now = cash + invested
            target = MAX_POSITION_FRACTION * equity_now
            per_position = min(target, cash / len(todays)) if cash > 0 else 0.0
            if per_position > 0:
                for index in todays:
                    size = per_position
                    if costs_on:
                        cost, at_minimum = trade_cost_fraction(
                            size, entry_price[index], spread_arr[index],
                            daily_vol[index], mdv[index])
                        total_cost += size * cost
                        total_spread_cost += size * 0.5 * spread_arr[index]
                        shares = size / entry_price[index]
                        commission = min(
                            max(COMMISSION_PER_SHARE * shares, COMMISSION_MINIMUM),
                            COMMISSION_CAP_FRACTION * size)
                        total_commission += commission
                        total_impact_cost += (size * cost - size * 0.5
                                              * spread_arr[index] - commission)
                        orders += 1
                        orders_at_minimum += int(at_minimum)
                        notional[index] = size * (1.0 - cost)
                    else:
                        orders += 1
                        notional[index] = size
                    cash -= size
                    position_pnl[index] -= size
                    position_notional[index] = size
                    invested += notional[index]
                    is_open[index] = True
                    total_bought += size
                    notionals_used.append(size)
                    open_count += 1

        equity_path[day] = cash + invested
        concurrent[day] = open_count
        cash_weight[day] = cash / equity_path[day] if equity_path[day] > 0 else 1.0

    equity = pd.Series(equity_path, index=calendar, name="equity")
    used = np.asarray(notionals_used, dtype=np.float64)
    return BookResult(
        equity=equity,
        exposure_start=pd.Series(exposure_start, index=calendar, name="exposure"),
        entries=count,
        total_bought=total_bought,
        total_sold=total_sold,
        mean_concurrent=float(np.mean(concurrent)),
        mean_cash_weight=float(np.mean(cash_weight)),
        mean_position_notional=float(used.mean()) if used.size else 0.0,
        median_position_notional=float(np.median(used)) if used.size else 0.0,
        entry_days=len(entries_by_day),
        total_cost=total_cost,
        total_commission=total_commission,
        total_spread_cost=total_spread_cost,
        total_impact_cost=total_impact_cost,
        open_at_end=int(is_open.sum()),
        orders=orders,
        orders_at_minimum=orders_at_minimum,
        position_pnl=position_pnl,
        position_notional=position_notional,
    )


# ---------------------------------------------------------------------------
# Benchmark -- the SAME corrected universe
# ---------------------------------------------------------------------------

def universe_benchmark(tickers: set[str], start: pd.Timestamp,
                       end: pd.Timestamp) -> pd.Series:
    """Equal-weight, monthly-rebalanced, zero-cost buy-and-hold of the CORRECTED universe.

    The eligibility test is the strategy's own: ``spread_regime in {measured,
    upper_bound}`` (the panel only assigns either to cells that already cleared the $2
    price floor, the 90% trading-fraction test and the $50k/day band floor), restricted to
    names with SF1 ARQ coverage.

    **This benchmark MOVES relative to iteration 1 and that is the point.** Admitting the
    liquid half of the tape changes what "the same names" means, so the iteration-1
    benchmark return is not comparable and the excess must be recomputed.
    """
    panel = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet")
    panel = panel[panel["spread_regime"].isin(TRADABLE_REGIMES)
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

    # Group by calendar MONTH, not by the raw date: the panel's date is each ticker's OWN
    # last bar of the month, so grouping on the raw date puts a name that stopped trading
    # mid-month into a singleton "month" whose delisting return becomes a -100% month for
    # the whole benchmark. That is how this function once returned -100%/yr.
    month = panel["date"].dt.to_period("M")
    monthly = panel.groupby(month)["forward_return"].mean().sort_index()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "date"
    return monthly


# ---------------------------------------------------------------------------
# Diagnostics required by the pre-registration
# ---------------------------------------------------------------------------

def sharpe_by_decade(monthly_returns: pd.Series) -> dict[str, dict[str, float]]:
    """Annualised return / volatility / Sharpe per calendar decade (prereg s9.2).

    A decade with fewer than 24 months is flagged ``thin`` and must not be read as
    evidence; it is reported anyway because suppressing it would be selection.
    """
    out: dict[str, dict[str, float]] = {}
    returns = monthly_returns.dropna()
    if returns.empty:
        return out
    decade = (returns.index.year // 10) * 10
    for label, group in returns.groupby(decade):
        stats = pead.summarise(group)
        out[f"{int(label)}s"] = {
            "months": int(stats.months),
            "annual_return": float(stats.annual_return),
            "annual_volatility": float(stats.annual_volatility),
            "sharpe": float(stats.sharpe),
            "thin": bool(stats.months < 24),
        }
    return out


def pnl_concentration(positions: PositionSet, book: BookResult,
                      calendar: pd.DatetimeIndex, top: int = 10) -> dict:
    """Share of total net P&L carried by the single largest (name, month) (prereg s9.3).

    Each position's realised dollar P&L is attributed to (ticker, calendar month of
    exit). The shares sum to 1 by construction because idle cash earns nothing, so the
    book's whole P&L is the sum of its positions'.
    """
    pnl = np.asarray(book.position_pnl, dtype=np.float64)
    total = float(pnl.sum())
    exit_month = pd.DatetimeIndex(
        [calendar[d] for d in positions.exit_day]).to_period("M").astype(str)
    frame = pd.DataFrame({"ticker": positions.ticker, "month": exit_month, "pnl": pnl})
    grouped = frame.groupby(["ticker", "month"], sort=False)["pnl"].sum()

    absolute = float(np.abs(grouped.to_numpy()).sum())
    share = grouped / total if total != 0 else grouped * np.nan
    ordered = share.reindex(share.abs().sort_values(ascending=False).index)
    biggest = ordered.head(top)

    return {
        "total_net_pnl": total,
        "max_name_month_share_of_total": float(ordered.iloc[0]) if len(ordered) else 0.0,
        "max_name_month_share_of_absolute": (
            float(np.abs(grouped).max() / absolute) if absolute > 0 else 0.0),
        "exceeds_alarm": bool(len(ordered)
                              and abs(float(ordered.iloc[0])) > CONCENTRATION_ALARM),
        "alarm_threshold": CONCENTRATION_ALARM,
        "name_months_over_alarm": int((ordered.abs() > CONCENTRATION_ALARM).sum()),
        "top": [
            {"ticker": key[0], "month": key[1], "pnl": float(grouped.loc[key]),
             "share_of_total": float(value)}
            for key, value in biggest.items()
        ],
    }


def verdict_for(excess: float, net_sharpe: float, bound: str) -> bool:
    """Primary gate G (prereg s6): positive excess AND net Sharpe at or above 0.75."""
    del bound
    return bool(np.isfinite(excess) and np.isfinite(net_sharpe)
                and excess > 0.0 and net_sharpe >= PROMOTION_GATE_SHARPE)


def registered_verdict(excess_conservative: float, sharpe_conservative: float,
                       excess_realistic: float, sharpe_realistic: float
                       ) -> tuple[str, bool]:
    """The four-way verdict fixed in prereg s6, plus a bound-inversion flag.

    Returns ``(verdict, inverted)``. EXCESS cannot invert: the realistic bound charges a
    spread no larger than the conservative one, so its net return is weakly higher against
    the same benchmark. SHARPE is a ratio and is *not* guaranteed monotone -- a cheaper
    cost can raise the return and the volatility together. If the conservative bound
    clears the gate and the realistic one does not, that is reported as an inversion and
    the verdict still reads PROMISING, because refusing to report the stronger result
    would be selection in the flattering direction's favour.
    """
    passes_conservative = verdict_for(excess_conservative, sharpe_conservative, "a")
    passes_realistic = verdict_for(excess_realistic, sharpe_realistic, "b")
    inverted = bool(passes_conservative and not passes_realistic)
    if passes_conservative:
        return "PROMISING", inverted
    if passes_realistic:
        return "UNDETERMINED", inverted
    if np.isfinite(excess_realistic) and excess_realistic > 0.0:
        return "MARGINAL", inverted
    return "DEAD", inverted


__all__ = [
    "BookResult",
    "CONCENTRATION_ALARM",
    "HEADLINE_HORIZON",
    "HOLDING_HORIZONS",
    "PROMOTION_GATE_SHARPE",
    "PositionSet",
    "ScreenBounds",
    "TRADABLE_REGIMES",
    "build_positions",
    "pnl_concentration",
    "registered_verdict",
    "screen_all",
    "screen_at_filing",
    "sharpe_by_decade",
    "simulate_book",
    "trade_cost_fraction",
    "universe_benchmark",
    "verdict_for",
]
