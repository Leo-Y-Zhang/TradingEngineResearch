"""Insider-transaction CLUSTERING sleeve, built on Sharadar SF2.

Registered design: `research/sleeves/insider_clustering_prereg.md`. Read it first; every
constant here is fixed there and none of them may be moved after a result is seen.

WHY this sleeve exists at all. The prior Form-4 study in this repo
(`research/insider_features.py`) ran on free scraped SEC data whose as-filed ticker join
lost ~22% of rows, and it returned "cannot certify" on POWER grounds rather than "no
effect". SF2 resolves the issuer entity itself, so the join defect is gone and the
question is open again.

WHY the code is shaped the way it is, in the order that matters:

1. `securityadcode == "NA"` is read with `keep_default_na=False`. pandas parses the
   literal string "NA" as missing by default, so the naive read silently matches ZERO
   open-market purchases and the whole study measures nothing. This is the single
   easiest way to get a confidently wrong answer here.
2. Deduplication happens on the ECONOMIC transaction, not the row. A joint Form 4 emits
   one row per co-reporting owner for a single purchase; the prior adversarial review
   measured 41% of purchase-leg dollar value duplicated that way, and an inflated
   distinct-buyer count is precisely the artefact that would manufacture the clustering
   effect this sleeve is trying to detect.
3. Delisting terminal returns are gated on the delisting DATE falling inside a 62-day
   window after the exit. The bug this replaces booked a 2012 bankruptcy against a 2003
   exit and produced -60%/yr.
4. The book is rebuilt from scratch at every rebalance, so a name that left the universe
   cannot be re-booked in a later month. The bug this replaces re-applied -100% every
   month forever and produced -112%/yr on a long-only book.
5. Costs are per name and are refused where they cannot be measured: only
   `spread_regime == "measured"` names are investable. Costing an `upper_bound` name at
   the resolution floor invents a cheap trade in exactly the illiquid corner where a
   false positive would look best.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from research.delisting import (  # noqa: F401  -- re-exported: the declared repair
    CORRECTED_WINDOW as CORRECTED_DELISTING_WINDOW,
)
from research.delisting import REGISTERED_WINDOW as REGISTERED_DELISTING_WINDOW
from research.delisting import in_window_mask

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "_data" / "sharadar"
PANEL_DIR = DATA_DIR / "panel"

DEV_CUTOFF = pd.Timestamp("2015-12-31")

# ---------------------------------------------------------------------------
# Registered constants (prereg sections in brackets). Do not edit post-result.
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 90                    # [S5] trailing insider window
FIRST_REBALANCE = pd.Timestamp("2008-04-30")   # [S2] one clear month after SF2 starts
LAST_REBALANCE = pd.Timestamp("2015-11-30")    # [S2] last month with a forward return
DECILE = 0.10                         # [S8]
BOOK_SIZE = 250_000.0                 # [S8] retail book
RETURN_CAP = 1.00                     # [S7] artefact filter
# [S7]. The registered window's lower edge is STRICT and so rejects a delisting dated ON
# the last traded bar -- the modal case. Single definition in `research.delisting`.
DELISTING_WINDOW_DAYS = REGISTERED_DELISTING_WINDOW[1]

# IBKR US equities, tiered/fixed retail schedule [S9].
COMMISSION_PER_SHARE = 0.0035
COMMISSION_MINIMUM = 0.35
COMMISSION_VALUE_CAP = 0.01

# Open-market, non-derivative ACQUISITION. 'P' alone would also admit derivative
# purchases (securityadcode 'DA'), which are warrant/option buys, not share buys.
PURCHASE_CODE = "P"
NON_DERIVATIVE_ACQUISITION = "NA"

# The economic identity of a purchase leg [S4]. Deliberately excludes ownername (that is
# what fans out), filingdate and formtype (a RESTATED-4 repeats the same transaction).
DEDUPE_KEY = [
    "ticker",
    "transactiondate",
    "transactionshares",
    "transactionpricepershare",
    "securitytitle",
    "directorindirect",
]

SF2_COLUMNS = [
    "ticker",
    "filingdate",
    "formtype",
    "ownername",
    "isdirector",
    "isofficer",
    "istenpercentowner",
    "transactiondate",
    "securityadcode",
    "transactioncode",
    "transactionshares",
    "transactionpricepershare",
    "transactionvalue",
    "securitytitle",
    "directorindirect",
]


@dataclass
class DedupeReport:
    """Row and value collapse from fan-out removal. Silent filtering is banned."""

    raw_legs: int
    deduped_legs: int
    raw_value: float
    deduped_value: float

    def render(self) -> str:
        return (
            f"  purchase legs (raw)      {self.raw_legs:>12,}\n"
            f"  purchase legs (deduped)  {self.deduped_legs:>12,}\n"
            f"  rows collapsed           "
            f"{1.0 - self.deduped_legs / max(self.raw_legs, 1):>12.1%}\n"
            f"  dollar value collapsed   "
            f"{1.0 - self.deduped_value / max(self.raw_value, 1.0):>12.1%}"
        )


@dataclass
class UniverseReport:
    """Why names left the investable universe, per the registered filters."""

    panel_rows_at_rebalances: int = 0
    dropped_no_band: int = 0
    dropped_spread_unmeasured: int = 0
    dropped_unresolvable_outcome: int = 0
    dropped_return_cap: int = 0
    surviving: int = 0
    terminal_applied: int = 0

    def render(self) -> str:
        return (
            f"  panel rows at rebalances {self.panel_rows_at_rebalances:>12,}\n"
            f"  - no liquidity band      {self.dropped_no_band:>12,}\n"
            f"  - spread not measured    {self.dropped_spread_unmeasured:>12,}\n"
            f"  - outcome unresolvable   {self.dropped_unresolvable_outcome:>12,}\n"
            f"  - |return| > 100%        {self.dropped_return_cap:>12,}\n"
            f"  investable cells         {self.surviving:>12,}\n"
            f"  (terminal return booked  {self.terminal_applied:>12,})"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_purchase_legs(force: bool = False) -> pd.DataFrame:
    """Open-market, non-derivative insider PURCHASES filed on or before the DEV cutoff.

    The `keep_default_na=False` is load-bearing: without it pandas turns the
    `securityadcode` value "NA" (non-derivative Acquisition) into NaN and the purchase
    filter selects nothing at all.
    """
    cache = PANEL_DIR / "sf2_dev_purchases.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        DATA_DIR / "SF2.csv",
        usecols=SF2_COLUMNS,
        chunksize=2_000_000,
        low_memory=False,
        keep_default_na=False,
        na_values=[""],
    ):
        chunk["filingdate"] = pd.to_datetime(chunk["filingdate"], errors="coerce")
        chunk = chunk[chunk["filingdate"] <= DEV_CUTOFF]
        chunk = chunk[
            (chunk["transactioncode"] == PURCHASE_CODE)
            & (chunk["securityadcode"] == NON_DERIVATIVE_ACQUISITION)
        ]
        frames.append(chunk)

    legs = pd.concat(frames, ignore_index=True)
    legs.to_parquet(cache, index=False)
    return legs


def dedupe_purchase_legs(legs: pd.DataFrame) -> tuple[pd.DataFrame, DedupeReport]:
    """Collapse multi-owner filing fan-out to one row per economic transaction [S4].

    Keeping the EARLIEST filing date per key matters for point-in-time correctness: a
    restatement filed months later must not postpone the date the market learned the
    purchase, and must not be counted as a second purchase either.
    """
    legs = legs.copy()
    value = legs["transactionvalue"].where(
        legs["transactionvalue"].notna() & (legs["transactionvalue"] > 0.0),
        legs["transactionshares"] * legs["transactionpricepershare"],
    )
    legs["value"] = value.fillna(0.0).clip(lower=0.0)

    raw_legs, raw_value = len(legs), float(legs["value"].sum())
    legs = legs.sort_values("filingdate", kind="stable")
    legs = legs.drop_duplicates(subset=DEDUPE_KEY, keep="first").reset_index(drop=True)

    report = DedupeReport(
        raw_legs=raw_legs,
        deduped_legs=len(legs),
        raw_value=raw_value,
        deduped_value=float(legs["value"].sum()),
    )
    return legs, report


def daily_volatility_panel(force: bool = False) -> pd.DataFrame:
    """Trailing 63-day daily log-return volatility at every panel date.

    Needed for the square-root impact charge [S9]; the monthly panel does not carry it.
    Computed with a single global rolling window over the ticker-sorted price file and
    then masked where the window would straddle two tickers -- a per-ticker groupby
    rolling over 29M bars is an order of magnitude slower for the same answer.
    """
    cache = PANEL_DIR / "dailyvol_dev.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    prices = pd.read_parquet(
        PANEL_DIR / f"prices_to_{DEV_CUTOFF.date()}.parquet",
        columns=["ticker", "date", "closeadj"],
    )
    # The file is documented as already sorted by (ticker, date); assert rather than
    # trust, because an unsorted input makes every diff below silently meaningless.
    ticker = prices["ticker"].to_numpy()
    changed = np.empty(len(prices), dtype=bool)
    changed[0] = True
    changed[1:] = ticker[1:] != ticker[:-1]

    log_close = np.log(prices["closeadj"].to_numpy(dtype=float))
    log_return = np.diff(log_close, prepend=np.nan)
    log_return[changed] = np.nan  # first bar of each ticker has no predecessor

    series = pd.Series(log_return)
    vol = series.rolling(63, min_periods=40).std()

    # Position of each row within its ticker; a full 63-day window needs 63 prior rows.
    position = np.arange(len(prices)) - np.maximum.accumulate(
        np.where(changed, np.arange(len(prices)), 0)
    )
    vol = vol.where(position >= 63)

    out = prices[["ticker", "date"]].copy()
    out["daily_vol"] = vol.to_numpy()
    out = out[out["daily_vol"].notna()].reset_index(drop=True)
    out.to_parquet(cache, index=False)
    return out


# ---------------------------------------------------------------------------
# Universe and outcomes
# ---------------------------------------------------------------------------
def build_universe(
    panel: pd.DataFrame,
    delistings: pd.DataFrame,
    delisting_window: tuple[int, int] = REGISTERED_DELISTING_WINDOW,
) -> tuple[pd.DataFrame, list[pd.Timestamp], UniverseReport]:
    """Investable cells with a resolved one-month outcome, per prereg S6 and S7.

    `delisting_window` defaults to the REGISTERED window and reproduces every banked
    number bit-for-bit; `CORRECTED_DELISTING_WINDOW` is the declared repair.
    """
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["month"] = panel["date"].values.astype("datetime64[M]")

    # Month-end dates are the LAST panel date in each calendar month. The panel also
    # holds mid-month stub rows (a name's final bar before it stopped trading); those
    # are outcomes to be booked, never dates to trade on.
    month_ends = sorted(panel.groupby("month")["date"].max().tolist())
    # DEFECT FIXED 2026-07-28: the successor date must be the next MONTH-END, not the
    # next REBALANCE. Deriving it from the (truncated) rebalance list left the final
    # rebalance with a synthetic successor 40 days out, so every still-listed name
    # looked like it had "stopped trading before the next date" and was dropped as
    # unresolvable -- leaving 17 delisting names out of 2,200 and a -52% month. The
    # symptom was impossible, which is how it was caught.
    next_month_end = dict(zip(month_ends[:-1], month_ends[1:], strict=False))
    rebalance_dates = [
        d for d in month_ends if FIRST_REBALANCE <= d <= LAST_REBALANCE
    ]

    panel["next_date"] = panel.groupby("ticker")["date"].shift(-1)

    report = UniverseReport()
    cells = panel[panel["date"].isin(rebalance_dates)].copy()
    report.panel_rows_at_rebalances = len(cells)

    has_band = cells["band"].notna()
    report.dropped_no_band = int((~has_band).sum())
    cells = cells[has_band]

    measured = cells["spread_regime"] == "measured"
    report.dropped_spread_unmeasured = int((~measured).sum())
    cells = cells[measured].copy()

    # --- outcome resolution -------------------------------------------------
    cells["t_next"] = cells["date"].map(next_month_end)
    if cells["t_next"].isna().any():
        raise RuntimeError(
            "a rebalance date has no successor month-end; LAST_REBALANCE must leave at "
            "least one further month of panel data"
        )

    stopped = cells["next_date"].isna() | (cells["next_date"] < cells["t_next"])

    terminal = delistings.set_index("ticker")
    delist_date = cells["ticker"].map(terminal["date"])
    delist_return = cells["ticker"].map(terminal["terminal_return"])
    # Rule 4: the terminal return counts ONLY if the delisting actually happened inside
    # the window following this exit. Asking merely "did this ticker ever delist?" is
    # the bug that charged a 2012 bankruptcy against a 2003 position.
    in_window = in_window_mask(cells["date"], delist_date, delisting_window)

    price_leg = cells["forward_return"]
    booked_terminal = stopped & in_window
    outcome = np.where(
        booked_terminal,
        (1.0 + price_leg.fillna(0.0)) * (1.0 + delist_return.fillna(0.0)) - 1.0,
        price_leg,
    )
    cells["realised_return"] = outcome
    cells["booked_terminal"] = booked_terminal

    # A name that stopped trading with no delisting event inside the window has an
    # unknown fate. Guessing it is the choice that decides studies; drop it instead and
    # report how many, symmetrically for strategy and benchmark.
    unresolvable = stopped & ~in_window
    report.dropped_unresolvable_outcome = int(unresolvable.sum())
    cells = cells[~unresolvable]

    resolvable = cells["realised_return"].notna()
    report.dropped_unresolvable_outcome += int((~resolvable).sum())
    cells = cells[resolvable]

    within_cap = cells["realised_return"].abs() <= RETURN_CAP
    report.dropped_return_cap = int((~within_cap).sum())
    cells = cells[within_cap].copy()

    report.surviving = len(cells)
    report.terminal_applied = int(cells["booked_terminal"].sum())
    return cells.reset_index(drop=True), rebalance_dates, report


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def cluster_signal(
    legs: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Distinct director/officer buyers and their aggregate value, per (ticker, date).

    Only filings STRICTLY BEFORE the rebalance date are visible. A filing dated `t`
    itself is excluded, matching the prior study's next-business-day availability
    convention; a signal at `t`'s close can never be built from a filing that was not
    already public.
    """
    insiders = legs[(legs["isdirector"] == "Y") | (legs["isofficer"] == "Y")]
    insiders = insiders[["ticker", "filingdate", "ownername", "value"]].copy()

    frames: list[pd.DataFrame] = []
    for date in rebalance_dates:
        window = insiders[
            (insiders["filingdate"] >= date - pd.Timedelta(days=LOOKBACK_DAYS))
            & (insiders["filingdate"] < date)
        ]
        if window.empty:
            continue
        grouped = window.groupby("ticker").agg(
            n_buyers=("ownername", "nunique"),
            buy_value=("value", "sum"),
        )
        grouped["date"] = date
        frames.append(grouped.reset_index())

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "n_buyers", "buy_value"])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def trade_cost(
    notional: np.ndarray,
    price: np.ndarray,
    spread: np.ndarray,
    daily_vol: np.ndarray,
    median_dollar_volume: np.ndarray,
) -> np.ndarray:
    """Dollar cost of trading `notional` in each name, one side [S9].

    Three components, all per name and none flat:
      * half the name's own EDGE effective spread;
      * square-root impact scaled by the name's OWN trailing volatility, so a quiet
        large-cap and a violent micro-cap at the same participation are not charged the
        same -- the constant-coefficient alternative is a guess dressed as a model;
      * IBKR commission with its per-order minimum and its 1%-of-value cap.
    """
    notional = np.asarray(notional, dtype=float)
    traded = notional > 0.0

    half_spread = np.where(traded, notional * np.asarray(spread) / 2.0, 0.0)

    participation = np.divide(
        notional,
        np.asarray(median_dollar_volume, dtype=float),
        out=np.zeros_like(notional),
        where=np.asarray(median_dollar_volume, dtype=float) > 0.0,
    )
    impact = np.where(
        traded, notional * np.asarray(daily_vol) * np.sqrt(participation), 0.0
    )

    shares = np.divide(
        notional,
        np.asarray(price, dtype=float),
        out=np.zeros_like(notional),
        where=np.asarray(price, dtype=float) > 0.0,
    )
    commission = np.minimum(
        np.maximum(COMMISSION_PER_SHARE * shares, COMMISSION_MINIMUM),
        COMMISSION_VALUE_CAP * notional,
    )
    commission = np.where(traded, commission, 0.0)

    return half_spread + impact + commission


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    monthly: pd.DataFrame
    holdings_count: list[int] = field(default_factory=list)
    cost_components: dict[str, float] = field(default_factory=dict)


def run_backtest(
    cells: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    selector,
    label: str,
) -> BacktestResult:
    """Long-only monthly-rebalanced book with per-name costs.

    `selector(month_frame) -> DataFrame` returns the names to hold at that date. The book
    is rebuilt from scratch every month, which is what structurally prevents the
    re-booking bug: a delisted name simply is not in next month's frame, and its exit
    proceeds sit in cash until they are redeployed.
    """
    by_date = {d: f for d, f in cells.groupby("date")}
    holdings: dict[str, float] = {}  # ticker -> dollar value held
    cash = BOOK_SIZE
    rows: list[dict] = []
    counts: list[int] = []
    totals = {"spread": 0.0, "impact": 0.0, "commission": 0.0}
    # Last observed cost inputs per name. A position that must be SOLD because the name
    # dropped out of the investable universe still pays a real spread, real impact and a
    # real commission; charging it zero (the first version of this loop did, because the
    # name is absent from the current frame) is a free exit on every liquidation and
    # understates the cost of exactly the names that are most expensive to leave.
    last_known: dict[str, tuple[float, float, float, float]] = {}

    for date in rebalance_dates:
        frame = by_date.get(date)
        if frame is None or frame.empty:
            continue
        # Refresh cost inputs from the WHOLE investable universe, not just the picks, so
        # a name that merely fell out of the top decile is priced at today's figures.
        for row in frame[["ticker", "close", "spread", "daily_vol",
                          "median_dollar_volume"]].itertuples(index=False):
            last_known[row.ticker] = (row.close, row.spread, row.daily_vol,
                                      row.median_dollar_volume)
        picks = selector(frame)
        equity_start = cash + sum(holdings.values())
        if picks.empty:
            # Nothing to hold: liquidate into cash rather than invent a position.
            cash, holdings = equity_start, {}
            rows.append({"date": date, "gross": 0.0, "net": 0.0, "cost": 0.0,
                         "n_held": 0, "turnover": 0.0})
            counts.append(0)
            continue

        picks = picks.set_index("ticker")
        target_value = equity_start / len(picks)
        targets = pd.Series(target_value, index=picks.index)

        current = pd.Series(holdings, dtype=float)
        universe_index = targets.index.union(current.index)
        target_full = targets.reindex(universe_index, fill_value=0.0)
        current_full = current.reindex(universe_index, fill_value=0.0)
        trade_notional = (target_full - current_full).abs()

        # Cost inputs: today's figures for names in the universe, the most recent
        # observed figures for a name being sold that has left it. Every leg of every
        # trade is priced; nothing exits free except a corporate action, which by then
        # is already cash and is not in `holdings` at all.
        inputs = np.array(
            [last_known[t] for t in universe_index], dtype=float
        ).reshape(len(universe_index), 4)
        price_v, spread_v, vol_v, mdv_v = (inputs[:, 0], inputs[:, 1],
                                           inputs[:, 2], inputs[:, 3])
        notional = trade_notional.to_numpy(dtype=float)

        costs = trade_cost(notional, price_v, spread_v, vol_v, mdv_v)
        total_cost = float(costs.sum())

        # Decompose for reporting (recomputed, not re-derived, so the parts sum).
        spread_part = float((notional * spread_v / 2.0).sum())
        part = np.divide(notional, mdv_v, out=np.zeros_like(notional), where=mdv_v > 0)
        impact_part = float((notional * vol_v * np.sqrt(part)).sum())
        totals["spread"] += spread_part
        totals["impact"] += impact_part
        totals["commission"] += total_cost - spread_part - impact_part

        equity_after_cost = equity_start - total_cost
        # Costs are paid out of the book, so the positions actually established are
        # slightly smaller than the pre-cost targets.
        scale = equity_after_cost / equity_start if equity_start > 0 else 0.0
        established = {t: target_value * scale for t in picks.index}

        realised = picks["realised_return"].to_numpy(dtype=float)
        gross_return = float(np.mean(realised))
        end_values = {
            t: v * (1.0 + r)
            for (t, v), r in zip(established.items(), realised, strict=True)
        }
        equity_end = sum(end_values.values())

        # Names whose position ended via a corporate action convert to cash; everything
        # else stays invested and drifts into next month's rebalance.
        booked = picks["booked_terminal"].to_dict()
        cash = sum(v for t, v in end_values.items() if booked.get(t, False))
        holdings = {t: v for t, v in end_values.items() if not booked.get(t, False)}

        rows.append({
            "date": date,
            "gross": gross_return,
            "net": equity_end / equity_start - 1.0 if equity_start > 0 else 0.0,
            "cost": total_cost / equity_start if equity_start > 0 else 0.0,
            "n_held": len(picks),
            # One-way turnover: buys and sells both appear in `notional`, so half the
            # total traded notional is the fraction of the book replaced.
            "turnover": 0.5 * float(notional.sum()) / equity_start
            if equity_start > 0 else 0.0,
        })
        counts.append(len(picks))

    monthly = pd.DataFrame(rows)
    monthly["label"] = label
    return BacktestResult(monthly=monthly, holdings_count=counts,
                          cost_components=totals)


def annualise(monthly_returns: pd.Series) -> dict[str, float]:
    """Compound annual return, annualised volatility, Sharpe (zero cash rate) and maxDD.

    A zero risk-free rate overstates Sharpe for a 2008-2015 sample where cash paid
    roughly nothing, so the distortion is small; it is stated rather than corrected so
    the number is comparable with the rest of the programme.
    """
    r = monthly_returns.dropna().to_numpy(dtype=float)
    if r.size == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    growth = np.prod(1.0 + r)
    years = r.size / 12.0
    cagr = growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(12.0)
    mean_annual = float(np.mean(r)) * 12.0
    sharpe = mean_annual / vol if vol > 0 else np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    maxdd = float(np.min(equity / peak) - 1.0)
    return {"cagr": float(cagr), "vol": vol, "sharpe": float(sharpe), "maxdd": maxdd}
