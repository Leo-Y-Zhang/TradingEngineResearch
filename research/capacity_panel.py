"""Build the point-in-time panel the capacity-curve study runs on.

Registered design: `research/medallion_style_alpha_search/capacity_curve_prereg.md`.

Turns the raw Sharadar export into one monthly panel with, per name and rebalance date:
prices, trailing liquidity, an EDGE spread estimate with its resolution regime, the
delisting outcome, and the liquidity band the name belonged to *at that date*. Everything
here is point-in-time by construction; nothing reads a bar the strategy could not have
seen.

The expensive steps are cached to parquet under ``_data/sharadar/panel/`` because the
raw SEP export is 3.2 GB and a study that has to re-read it on every run will not get
re-run. Cached files are derived row-for-row from licensed Data and are covered by the
purge obligation (`scripts/purge_sharadar_data.py` already globs ``*.parquet``).

**The DEV/CONFIRM split is enforced here, at the loader.** ``load_prices`` refuses to
return any bar after the DEV cutoff unless explicitly asked, so no tool built on top of
this can accidentally read the confirmation window that
`sharadar_confirmatory_prereg.md` is holding unfired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.spread_estimation import spread_with_resolution

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "_data" / "sharadar"
PANEL_DIR = DATA_DIR / "panel"

# The physical split from sharadar_confirmatory_prereg.md. Development work may only
# see bars on or before this date.
DEV_CUTOFF = pd.Timestamp("2015-12-31")

# Registered band edges (prereg §3), on median trailing-63-day dollar volume.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("B1_50k_200k", 5e4, 2e5),
    ("B2_200k_1M", 2e5, 1e6),
    ("B3_1M_5M", 1e6, 5e6),
    ("B4_5M_25M", 5e6, 2.5e7),
    ("B5_25M_200M", 2.5e7, 2e8),
    ("B6_200M_plus", 2e8, np.inf),
)

LIQUIDITY_WINDOW = 63
MIN_PRICE = 2.00
MIN_TRADING_FRACTION = 0.90
FORWARD_RETURN_CAP = 1.00

# Delisting treatment by ACTIONS event type (prereg §7). Dropping delistings biases
# returns UP; assigning -100% to all of them biases DOWN. Neither is acceptable when
# 8,247 of them are acquisitions that pay a premium and 3,347 are bankruptcies.
TERMINAL_LOSS_ACTIONS = frozenset({
    "bankruptcyliquidation", "regulatorydelisting", "voluntarydelisting", "delisted",
})
ACQUISITION_ACTIONS = frozenset({
    "acquisitionby", "acquisitionof", "mergerto", "mergerfrom",
})


@dataclass(frozen=True)
class FilterReport:
    """How many rows each artefact filter removed. Silent filtering is banned."""

    starting_rows: int
    dropped_min_price: int
    dropped_thin_trading: int
    dropped_return_cap: int
    dropped_split_inconsistent: int
    surviving_rows: int

    def render(self) -> str:
        return (
            f"  starting rows            {self.starting_rows:>12,}\n"
            f"  - price below ${MIN_PRICE:.2f}      {self.dropped_min_price:>12,}\n"
            f"  - thin trading           {self.dropped_thin_trading:>12,}\n"
            f"  - forward return capped  {self.dropped_return_cap:>12,}\n"
            f"  - split inconsistent     {self.dropped_split_inconsistent:>12,}\n"
            f"  surviving rows           {self.surviving_rows:>12,}"
        )


def _cache_path(name: str) -> Path:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    return PANEL_DIR / f"{name}.parquet"


def load_prices(
    cutoff: pd.Timestamp = DEV_CUTOFF,
    allow_confirmation_window: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Daily bars up to ``cutoff``, chronologically sorted, cached to parquet.

    The raw export is NOT chronologically ordered -- dates run backwards within a
    ticker. Every consecutive-bar calculation downstream (spreads, returns) is silently
    meaningless on unsorted input, so sorting happens here, once, rather than being left
    to each caller to remember.

    Args:
        cutoff: Latest bar date to return.
        allow_confirmation_window: Must be True to read past ``DEV_CUTOFF``. Guards the
            unfired confirmation window against accidental use by development tools.
        force: Rebuild the cache even if it exists.
    """
    if cutoff > DEV_CUTOFF and not allow_confirmation_window:
        raise ValueError(
            f"cutoff {cutoff.date()} is past the DEV cutoff {DEV_CUTOFF.date()}. The "
            "2016+ confirmation window is pre-registered and unfired; pass "
            "allow_confirmation_window=True only when deliberately firing it."
        )

    cache = _cache_path(f"prices_to_{cutoff.date()}")
    if cache.exists() and not force:
        logger.info("loading cached prices from %s", cache.name)
        return pd.read_parquet(cache)

    logger.info("building price panel from SEP.csv (this reads ~3.2 GB)")
    columns = ["ticker", "date", "open", "high", "low", "close", "closeadj", "volume"]
    frames = []
    for chunk in pd.read_csv(DATA_DIR / "SEP.csv", usecols=columns,
                             chunksize=4_000_000):
        chunk["date"] = pd.to_datetime(chunk["date"])
        frames.append(chunk[chunk["date"] <= cutoff])
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["dollar_volume"] = prices["close"] * prices["volume"]

    prices.to_parquet(cache, index=False)
    logger.info("cached %s rows to %s", f"{len(prices):,}", cache.name)
    return prices


def load_actions() -> pd.DataFrame:
    """Corporate actions, with the delisting outcome resolved per event type."""
    actions = pd.read_csv(DATA_DIR / "ACTIONS.csv",
                          usecols=["date", "action", "ticker", "value"])
    actions["date"] = pd.to_datetime(actions["date"])
    return actions


def delisting_returns(actions: pd.DataFrame) -> pd.DataFrame:
    """Terminal return for each delisted name, by event type.

    Bankruptcy and involuntary delisting are booked at -100%. Acquisitions and mergers
    are booked at the last traded price (0% incremental), which is *conservative*: real
    deals typically close at a premium, so this understates the return to holding a
    name that gets bought. Understating a gain is the safe direction; overstating one
    would flatter any strategy that happens to hold acquisition targets.
    """
    terminal = actions[actions["action"].isin(TERMINAL_LOSS_ACTIONS)].copy()
    terminal["terminal_return"] = -1.0

    acquired = actions[actions["action"].isin(ACQUISITION_ACTIONS)].copy()
    acquired["terminal_return"] = 0.0

    combined = pd.concat([terminal, acquired], ignore_index=True)
    # A name can appear under several codes; the earliest terminal event is the one
    # that actually ended the position.
    combined = combined.sort_values("date").drop_duplicates("ticker", keep="first")
    return combined[["ticker", "date", "action", "terminal_return"]]


def assign_band(dollar_volume: float) -> str | None:
    for label, low, high in BANDS:
        if low <= dollar_volume < high:
            return label
    return None


def build_monthly_panel(
    prices: pd.DataFrame,
    min_history: int = LIQUIDITY_WINDOW,
) -> pd.DataFrame:
    """One row per (ticker, month-end): price, liquidity, band, spread, forward return.

    Liquidity and spread are computed from the trailing window ENDING at the rebalance
    date, so both are known to the strategy at the moment it would act on them.
    """
    prices = prices.copy()
    prices["month"] = prices["date"].values.astype("datetime64[M]")

    rows: list[dict] = []
    for ticker, frame in prices.groupby("ticker", sort=False):
        if len(frame) < min_history * 2:
            continue
        frame = frame.reset_index(drop=True)
        month_end_positions = frame.groupby("month").tail(1).index

        for position in month_end_positions:
            if position < min_history:
                continue
            window = frame.iloc[position - min_history + 1: position + 1]

            traded = (window["high"] > window["low"]) & (window["volume"] > 0)
            trading_fraction = float(traded.mean())
            close = float(frame.at[position, "close"])
            median_dollar_volume = float(window["dollar_volume"].median())

            # Apply the cheap registered filters BEFORE the expensive spread estimate.
            # A name that fails the price floor or the trading-fraction test can never
            # enter the universe, so estimating its spread is wasted work -- and there
            # are roughly four million (name, month) cells to get through.
            band = assign_band(median_dollar_volume)
            eligible = (
                band is not None
                and close >= MIN_PRICE
                and trading_fraction >= MIN_TRADING_FRACTION
            )
            if eligible:
                spread, regime = spread_with_resolution(
                    window["open"], window["high"], window["low"], window["close"]
                )
            else:
                spread, regime = float("nan"), "ineligible"

            rows.append({
                "ticker": ticker,
                "date": frame.at[position, "date"],
                "close": close,
                "closeadj": float(frame.at[position, "closeadj"]),
                "median_dollar_volume": median_dollar_volume,
                "trading_fraction": trading_fraction,
                "spread": spread,
                "spread_regime": regime,
                "band": band,
            })

    panel = pd.DataFrame(rows)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Forward one-month return on the ADJUSTED close, so dividends and splits are
    # already handled and only genuine price change is measured.
    panel["forward_return"] = (
        panel.groupby("ticker")["closeadj"].shift(-1) / panel["closeadj"] - 1.0
    )
    return panel
