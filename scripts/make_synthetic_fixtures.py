"""
Synthetic test-fixture generator (NO real market data)
=======================================================
Regenerates ``tests/fixtures/prices_sample.csv`` and
``tests/fixtures/dividends_sample.csv`` as fully SYNTHETIC data.

Why: the previous fixtures were recorded from yfinance, i.e. Yahoo! Finance
data, whose Terms of Service do not permit redistribution (flagged in an
internal licence audit, 2026-07-14). The
fixtures below are generated from a seeded geometric-Brownian-motion process —
they contain NO Yahoo-derived values and only reuse ticker STRINGS as labels.
Price levels are deliberately unrealistic round-number starts (20/40/60/80/100)
so they cannot be mistaken for, or reverse-engineered into, real quotes.

What the tests need (tests/test_ingestion.py) and what this guarantees:
  * schema ``date,symbol,open,high,low,close,volume`` / ``date,symbol,dividend``
  * 5 symbols (AAPL, AMZN, GOOG, JPM, MSFT), business days 2022-01-03..2023-12-29
    (>2000 rows total, >253 rows/symbol so the 252-day momentum window fills)
  * OHLC invariants: 0 < low <= min(open, close) <= max(open, close) <= high
  * strictly positive integer volume
  * dividend payers = {AAPL, JPM, MSFT} only (AMZN/GOOG stay non-payers),
    quarterly schedule across 2022-2023, all amounts > 0, trailing yields
    landing well inside the tests' sane band (0, 0.5)

Deterministic: a fixed numpy seed — rerunning reproduces byte-identical CSVs.

Usage (from the repo root):
    .venv/Scripts/python.exe scripts/make_synthetic_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260714
START, END = "2022-01-03", "2023-12-29"

# Clearly-synthetic parameters per symbol: round-number start prices, generic
# drift/vol, and volume bases far below the real names' typical share counts.
SYMBOL_PARAMS: dict[str, dict[str, float]] = {
    "AAPL": {"start": 20.0, "mu": 0.06, "sigma": 0.22, "vol_base": 5_000_000},
    "AMZN": {"start": 40.0, "mu": 0.03, "sigma": 0.30, "vol_base": 3_000_000},
    "GOOG": {"start": 60.0, "mu": 0.05, "sigma": 0.26, "vol_base": 2_000_000},
    "JPM": {"start": 80.0, "mu": 0.04, "sigma": 0.20, "vol_base": 1_500_000},
    "MSFT": {"start": 100.0, "mu": 0.07, "sigma": 0.24, "vol_base": 4_000_000},
}

_TRADING_DAYS = 252.0
_GAP_SIGMA = 0.003      # overnight open-vs-prev-close gap
_WICK_SIGMA = 0.004     # high/low extension beyond the open/close body
_VOLUME_SIGMA = 0.35    # lognormal day-to-day volume dispersion

# Synthetic quarterly dividend calendars: (anchor months, day-of-month, amounts).
# Amounts step up over time like a typical payer, but are arbitrary numbers.
DIVIDEND_SCHEDULES: dict[str, dict] = {
    "AAPL": {"months": (2, 5, 8, 11), "day": 15,
             "amounts": [0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13]},
    "JPM": {"months": (1, 4, 7, 10), "day": 10,
            "amounts": [0.50, 0.50, 0.50, 0.50, 0.55, 0.55, 0.55, 0.55]},
    "MSFT": {"months": (3, 6, 9, 12), "day": 15,
             "amounts": [0.30, 0.30, 0.32, 0.32, 0.34, 0.34, 0.36, 0.36]},
}


def make_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Seeded GBM close path per symbol + gapped opens, wicked highs/lows,
    lognormal positive integer volume. OHLC invariants are enforced after
    rounding so no float edge case can invert them."""
    dates = pd.bdate_range(START, END)
    n = len(dates)
    frames: list[pd.DataFrame] = []
    for symbol, p in SYMBOL_PARAMS.items():
        mu_d = p["mu"] / _TRADING_DAYS
        sigma_d = p["sigma"] / np.sqrt(_TRADING_DAYS)
        log_returns = rng.normal(mu_d - 0.5 * sigma_d**2, sigma_d, size=n)
        close = p["start"] * np.exp(np.cumsum(log_returns))
        prev_close = np.concatenate(([p["start"]], close[:-1]))
        open_ = prev_close * np.exp(rng.normal(0.0, _GAP_SIGMA, size=n))
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        high = body_hi * (1.0 + np.abs(rng.normal(0.0, _WICK_SIGMA, size=n)))
        low = body_lo * (1.0 - np.abs(rng.normal(0.0, _WICK_SIGMA, size=n)))
        volume = np.maximum(
            1, np.round(p["vol_base"] * rng.lognormal(0.0, _VOLUME_SIGMA, size=n))
        ).astype(np.int64)

        o, h = np.round(open_, 4), np.round(high, 4)
        lo_r, c = np.round(low, 4), np.round(close, 4)
        h = np.maximum.reduce([h, o, c])            # rounding must not break
        lo_r = np.minimum.reduce([lo_r, o, c])      # low <= o,c <= high
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "symbol": symbol,
            "open": o, "high": h, "low": lo_r, "close": c, "volume": volume,
        }))
    return pd.concat(frames, ignore_index=True)


def make_dividends() -> pd.DataFrame:
    """Fixed quarterly synthetic dividend calendar for the payer subset,
    weekend dates rolled forward to the next business day."""
    rows: list[dict] = []
    for symbol, sched in DIVIDEND_SCHEDULES.items():
        events = [pd.Timestamp(year=year, month=month, day=sched["day"])
                  for year in (2022, 2023) for month in sched["months"]]
        for date, amount in zip(events, sched["amounts"], strict=True):
            rolled = np.busday_offset(date.date(), 0, roll="forward")
            rows.append({"date": pd.Timestamp(rolled).strftime("%Y-%m-%d"),
                         "symbol": symbol, "dividend": amount})
    return pd.DataFrame(rows, columns=["date", "symbol", "dividend"])


def _check(prices: pd.DataFrame, dividends: pd.DataFrame) -> None:
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (prices["low"] > 0).all() and (prices["volume"] > 0).all()
    assert prices["symbol"].nunique() == 5 and len(prices) > 2000
    assert set(dividends["symbol"]) == set(DIVIDEND_SCHEDULES)
    assert (dividends["dividend"] > 0).all()


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    rng = np.random.default_rng(SEED)
    prices, dividends = make_prices(rng), make_dividends()
    _check(prices, dividends)
    prices.to_csv(out_dir / "prices_sample.csv", index=False)
    dividends.to_csv(out_dir / "dividends_sample.csv", index=False)
    print(f"wrote {out_dir / 'prices_sample.csv'} ({len(prices)} rows, "
          f"{prices['symbol'].nunique()} symbols, {prices['date'].min()}..{prices['date'].max()})")
    print(f"wrote {out_dir / 'dividends_sample.csv'} ({len(dividends)} rows, "
          f"payers={sorted(set(dividends['symbol']))})")


if __name__ == "__main__":
    main()
