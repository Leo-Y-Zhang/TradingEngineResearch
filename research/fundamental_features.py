"""
TradingEngineResearch — PIT-safe cross-sectional FUNDAMENTAL factor library
==================================================================
A library of well-motivated, point-in-time-safe cross-sectional FUNDAMENTAL
factors — the family with the strongest *out-of-sample* academic track record
(value, profitability/quality, investment, earnings quality, leverage, plus a
price 12-1 momentum control). Each factor is a **pure function** of a tidy panel
and ``compute_features`` assembles them into a tidy ``(ticker, date, feature...)``
frame of per-date cross-sectionally normalized features.

EXPECTED PANEL (the contract a fundamentals ingestion such as ``data/sharadar_ingestion.py``
produces): a *tidy* (one row per ``ticker`` × ``date``) ``pd.DataFrame`` whose
``date`` is the POINT-IN-TIME date — the date on which every value in that row was
*already knowable* (the close price on ``date`` and the latest fundamentals filed on
or before ``date``). Columns use Sharadar SF1 field names where relevant:

    ticker, date,                        # identifiers (required)
    price,                               # adjusted close on `date`           (momentum, eps/price)
    marketcap (or mktcap),               # market capitalisation              (value ratios)
    netinc, eps,                         # net income (TTM), EPS              (earnings yield, ROE/ROA)
    equity, assets,                      # book equity, total assets          (book/price, ROE/ROA)
    revenue, gp, ebit,                   # revenue, gross profit, op. income  (sales/price, GP/A, op. margin)
    ncfo,                                # net cash flow from operations      (accruals)
    debt,                                # total debt                         (leverage)
    sharesbas                            # basic shares outstanding (optional → net share issuance)

POINT-IN-TIME SAFETY (the whole game; golden rule 3):
  * Every factor is computed **only** from values on its own row (row date) or from
    EARLIER rows of the SAME ticker (year-on-year growth, momentum). No factor ever
    reads a future row.
  * Year-on-year / momentum lookbacks resolve to the most recent observation AT OR
    BEFORE a target calendar offset in the past (a BACKWARD asof match, bounded
    tolerance) — the matched row is strictly earlier than the current date for any
    positive offset, so a lag can never read the current or any future row regardless
    of the tolerance.
  * Normalization is **strictly per-date**: winsorize then rank/z-score WITHIN each
    cross-section only. There is NO full-sample (cross-date) statistic anywhere, so a
    row's feature cannot move when other dates' data change — no look-ahead leakage.
  * NaNs are never imputed with information that would not have been known: a missing
    input yields ``NaN`` (dropped from the cross-section), never a fabricated value.

Sign convention: raw factors are reported as *measured* (not pre-signed to "higher =
better"). The documented expected direction of each is noted on its function; the
downstream learner (``research.alpha_factory.learn_signal_weights``) discovers the sign.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    # value
    "earnings_yield",
    "book_to_price",
    "sales_to_price",
    # quality / profitability
    "roe",
    "roa",
    "gross_profitability",
    "operating_margin",
    # growth
    "revenue_growth",
    "earnings_growth",
    # investment
    "asset_growth",
    "net_share_issuance",
    # earnings quality
    "accruals",
    # leverage
    "debt_to_equity",
    # momentum control
    "momentum_12_1",
    # assembly
    "compute_features",
    "raw_features",
    "FACTOR_FUNCTIONS",
    "FEATURE_NAMES",
]

# Market-cap column aliases (Sharadar uses ``marketcap``; ``mktcap`` accepted too).
_MKTCAP_COLUMNS: tuple[str, ...] = ("marketcap", "mktcap")

# Year-on-year lookback: nearest observation to ~1y prior, within ±185d (so quarterly
# panels match ~4 quarters back and annual panels match the prior year; the matched
# date lies in [t-550d, t-180d] — always strictly before t).
_YOY_LOOKBACK_DAYS: int = 365
_YOY_TOL_DAYS: int = 185

# Price 12-1 momentum: skip the most recent ~1 month (avoid short-term reversal), then
# measure the return back to ~12 months prior. Both legs resolve via a BACKWARD asof match
# (matched observation strictly before the target, hence strictly before `date`), so neither
# leg can ever read the price on `date` or later — regardless of tolerance. The skip target
# is 25d (inside the 28d shortest calendar month) so on a monthly grid the backward match
# lands on the PRIOR month-end (the genuine "1 month ago" observation), not two months back.
_MOM_SKIP_DAYS: int = 25
_MOM_SKIP_TOL_DAYS: int = 20
_MOM_LOOKBACK_DAYS: int = 365
_MOM_LOOKBACK_TOL_DAYS: int = 60

# Minimum finite names required to define a cross-section (mathematical floor for a
# z-score). Production callers may raise this; tiny hand-checkable fixtures need it low.
_MIN_XS_OBS: int = 2

# Default per-date winsorization quantile (clip each tail at 2%).
_WINSOR_QUANTILE: float = 0.02


# --------------------------------------------------------------------------- #
# Column / arithmetic helpers (None means "input column absent" → all-NaN factor)
# --------------------------------------------------------------------------- #
def _nan_series(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(np.full(len(panel), np.nan), index=panel.index, dtype=float)


def _num(panel: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """A panel column coerced to float, or ``None`` if the column is absent."""
    if name not in panel.columns:
        return None
    return pd.to_numeric(panel[name], errors="coerce")


def _market_cap(panel: pd.DataFrame) -> Optional[pd.Series]:
    for name in _MKTCAP_COLUMNS:
        if name in panel.columns:
            return pd.to_numeric(panel[name], errors="coerce")
    return None


def _safe_div(num: Optional[pd.Series], den: Optional[pd.Series]) -> Optional[pd.Series]:
    """``num / den`` with zero/NaN denominators → NaN. ``None`` if either input absent."""
    if num is None or den is None:
        return None
    return num / den.replace(0.0, np.nan)


def _coalesce(panel: pd.DataFrame, *parts: Optional[pd.Series]) -> pd.Series:
    """First non-NaN of ``parts`` per row (earlier parts win)."""
    out = _nan_series(panel)
    for part in parts:
        if part is not None:
            out = out.where(out.notna(), part)
    return out


def _finalize(panel: pd.DataFrame, series: Optional[pd.Series]) -> pd.Series:
    """Coerce to a float Series aligned to ``panel`` with ±inf scrubbed to NaN."""
    if series is None:
        return _nan_series(panel)
    return series.astype(float).replace([np.inf, -np.inf], np.nan)


def _asof_lag(panel: pd.DataFrame, col: str, lookback_days: int, tolerance_days: int) -> pd.Series:
    """For each row, the value of ``col`` from the SAME ticker at the most recent
    observation AT OR BEFORE ``date - lookback_days`` (within ``tolerance_days`` of that
    target).

    PIT-safe by the backward-match guarantee: with ``direction="backward"`` the matched
    observation's date is ``<= date - lookback_days < date`` for any ``lookback_days > 0``,
    so a lag can NEVER read the current or a future row — this holds regardless of the
    ``tolerance`` value (it does not rely on the unenforced ``tolerance < lookback``
    invariant that a "nearest" match would). ``tolerance_days`` only bounds how stale the
    matched observation may be; rows with no in-tolerance prior observation get ``NaN``
    (never imputed)."""
    n = len(panel)
    if n == 0 or col not in panel.columns or "date" not in panel.columns or "ticker" not in panel.columns:
        return _nan_series(panel)
    base = pd.DataFrame(
        {
            "__pos": np.arange(n),
            "ticker": panel["ticker"].to_numpy(),
            "date": pd.to_datetime(panel["date"]).to_numpy(),
            "__val": pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float),
        }
    )
    base = base[base["date"].notna()]
    if base.empty:
        return _nan_series(panel)
    left = base[["__pos", "ticker", "date"]].copy()
    left["__target"] = left["date"] - pd.Timedelta(days=lookback_days)
    left = left.sort_values("__target", kind="mergesort")
    right = base[["ticker", "date", "__val"]].sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="__target",
        right_on="date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
        suffixes=("", "_r"),
    )
    lag = pd.Series(merged["__val"].to_numpy(dtype=float), index=merged["__pos"].to_numpy())
    return lag.reindex(np.arange(n)).set_axis(panel.index)


def _yoy_growth(panel: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Year-on-year growth ``(x_t - x_{t-1y}) / |x_{t-1y}|``. ``|·|`` in the denominator
    keeps the sign sensible when the base is negative (common for earnings)."""
    cur = _num(panel, col)
    if cur is None:
        return None
    prior = _asof_lag(panel, col, _YOY_LOOKBACK_DAYS, _YOY_TOL_DAYS)
    return (cur - prior) / prior.abs().replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# VALUE factors (cheapness; higher ⇒ cheaper ⇒ higher expected return)
# --------------------------------------------------------------------------- #
def earnings_yield(panel: pd.DataFrame) -> pd.Series:
    """Earnings yield = netinc / marketcap (per-row fallback to eps / price)."""
    primary = _safe_div(_num(panel, "netinc"), _market_cap(panel))
    fallback = _safe_div(_num(panel, "eps"), _num(panel, "price"))
    return _finalize(panel, _coalesce(panel, primary, fallback))


def book_to_price(panel: pd.DataFrame) -> pd.Series:
    """Book-to-price = equity / marketcap (the classic HML value signal)."""
    return _finalize(panel, _safe_div(_num(panel, "equity"), _market_cap(panel)))


def sales_to_price(panel: pd.DataFrame) -> pd.Series:
    """Sales-to-price = revenue / marketcap (robust value ratio, less earnings noise)."""
    return _finalize(panel, _safe_div(_num(panel, "revenue"), _market_cap(panel)))


# --------------------------------------------------------------------------- #
# QUALITY / PROFITABILITY factors (higher ⇒ better ⇒ higher expected return)
# --------------------------------------------------------------------------- #
def roe(panel: pd.DataFrame) -> pd.Series:
    """Return on equity = netinc / equity."""
    return _finalize(panel, _safe_div(_num(panel, "netinc"), _num(panel, "equity")))


def roa(panel: pd.DataFrame) -> pd.Series:
    """Return on assets = netinc / assets."""
    return _finalize(panel, _safe_div(_num(panel, "netinc"), _num(panel, "assets")))


def gross_profitability(panel: pd.DataFrame) -> pd.Series:
    """Gross profitability = gp / assets (Novy-Marx 2013: the 'other side of value')."""
    return _finalize(panel, _safe_div(_num(panel, "gp"), _num(panel, "assets")))


def operating_margin(panel: pd.DataFrame) -> pd.Series:
    """Operating margin = ebit / revenue."""
    return _finalize(panel, _safe_div(_num(panel, "ebit"), _num(panel, "revenue")))


# --------------------------------------------------------------------------- #
# GROWTH factors (year-on-year)
# --------------------------------------------------------------------------- #
def revenue_growth(panel: pd.DataFrame) -> pd.Series:
    """Year-on-year revenue growth."""
    return _finalize(panel, _yoy_growth(panel, "revenue"))


def earnings_growth(panel: pd.DataFrame) -> pd.Series:
    """Year-on-year net-income growth."""
    return _finalize(panel, _yoy_growth(panel, "netinc"))


# --------------------------------------------------------------------------- #
# INVESTMENT factors (higher ⇒ more aggressive ⇒ LOWER expected return)
# --------------------------------------------------------------------------- #
def asset_growth(panel: pd.DataFrame) -> pd.Series:
    """Year-on-year total-asset growth (Cooper-Gulen-Schill 2008 investment factor)."""
    return _finalize(panel, _yoy_growth(panel, "assets"))


def net_share_issuance(panel: pd.DataFrame) -> pd.Series:
    """Year-on-year growth in basic shares outstanding (net issuance; Pontiff-Woodgate
    2008). All-NaN if ``sharesbas`` is not in the panel ('if available')."""
    return _finalize(panel, _yoy_growth(panel, "sharesbas"))


# --------------------------------------------------------------------------- #
# EARNINGS-QUALITY factor (higher accruals ⇒ lower quality ⇒ LOWER expected return)
# --------------------------------------------------------------------------- #
def accruals(panel: pd.DataFrame) -> pd.Series:
    """Cash-flow-statement accruals = (netinc - ncfo) / assets (Sloan 1996)."""
    ni = _num(panel, "netinc")
    ncfo = _num(panel, "ncfo")
    if ni is None or ncfo is None:
        return _nan_series(panel)
    return _finalize(panel, _safe_div(ni - ncfo, _num(panel, "assets")))


# --------------------------------------------------------------------------- #
# LEVERAGE factor
# --------------------------------------------------------------------------- #
def debt_to_equity(panel: pd.DataFrame) -> pd.Series:
    """Leverage = debt / equity."""
    return _finalize(panel, _safe_div(_num(panel, "debt"), _num(panel, "equity")))


# --------------------------------------------------------------------------- #
# MOMENTUM control (price 12-1; Jegadeesh-Titman, skipping the most recent month)
# --------------------------------------------------------------------------- #
def momentum_12_1(panel: pd.DataFrame) -> pd.Series:
    """Price 12-1 momentum: trailing return from ~12 months ago to ~1 month ago, i.e.
    ``price_{t-1m} / price_{t-12m} - 1`` (skips the last month to avoid reversal)."""
    if "price" not in panel.columns:
        return _nan_series(panel)
    short_leg = _asof_lag(panel, "price", _MOM_SKIP_DAYS, _MOM_SKIP_TOL_DAYS)
    long_leg = _asof_lag(panel, "price", _MOM_LOOKBACK_DAYS, _MOM_LOOKBACK_TOL_DAYS)
    return _finalize(panel, (short_leg - long_leg) / long_leg.replace(0.0, np.nan))


# --------------------------------------------------------------------------- #
# Registry — canonical feature order
# --------------------------------------------------------------------------- #
FACTOR_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "earnings_yield": earnings_yield,
    "book_to_price": book_to_price,
    "sales_to_price": sales_to_price,
    "roe": roe,
    "roa": roa,
    "gross_profitability": gross_profitability,
    "operating_margin": operating_margin,
    "revenue_growth": revenue_growth,
    "earnings_growth": earnings_growth,
    "asset_growth": asset_growth,
    "net_share_issuance": net_share_issuance,
    "accruals": accruals,
    "debt_to_equity": debt_to_equity,
    "momentum_12_1": momentum_12_1,
}
FEATURE_NAMES: list[str] = list(FACTOR_FUNCTIONS)


# --------------------------------------------------------------------------- #
# Per-date cross-sectional normalization (STRICTLY within each date — no leakage)
# --------------------------------------------------------------------------- #
def _winsorize(block: pd.Series, q: float) -> pd.Series:
    if q is None or q <= 0.0:
        return block.astype(float)
    finite = block[np.isfinite(block)]
    if finite.empty:
        return block.astype(float)
    lo = float(finite.quantile(q))
    hi = float(finite.quantile(1.0 - q))
    return block.clip(lower=lo, upper=hi)


def _zscore_normalize(block: pd.Series) -> pd.Series:
    arr = block.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    sd = float(finite.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return pd.Series(np.nan, index=block.index, dtype=float)
    return (block - float(finite.mean())) / sd


def _rank_normalize(block: pd.Series) -> pd.Series:
    # Percentile rank (NaNs kept) mapped to [-1, 1]; ties averaged.
    ranked = block.rank(method="average", na_option="keep", pct=True)
    return ranked * 2.0 - 1.0


def _normalize_block(block: pd.Series, method: str, q: float, min_obs: int) -> pd.Series:
    finite = int(np.isfinite(block.to_numpy(dtype=float)).sum())
    if finite < min_obs:
        return pd.Series(np.nan, index=block.index, dtype=float)
    winsorized = _winsorize(block, q)
    if method == "zscore":
        return _zscore_normalize(winsorized)
    if method == "rank":
        return _rank_normalize(winsorized)
    raise ValueError(f"unknown normalization method {method!r} (expected 'zscore' or 'rank')")


def _normalize_cross_section(
    values: pd.Series, dates: pd.Series, method: str, q: float, min_obs: int
) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for _date, block in s.groupby(np.asarray(dates)):
        out.loc[block.index] = _normalize_block(block, method, q, min_obs)
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    for col in ("ticker", "date"):
        if col not in panel.columns:
            raise ValueError(f"panel missing required column {col!r}")
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"])
    return work.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def raw_features(panel: pd.DataFrame) -> pd.DataFrame:
    """RAW (un-normalized) factor values as a tidy ``(ticker, date, <FEATURE_NAMES>)``
    frame, sorted by ``(date, ticker)``. Useful for inspection/debugging; the deployable
    features come from :func:`compute_features`."""
    work = _prepare(panel)
    data: dict[str, np.ndarray] = {
        "ticker": work["ticker"].to_numpy(),
        "date": work["date"].to_numpy(),
    }
    for name, fn in FACTOR_FUNCTIONS.items():
        data[name] = fn(work).to_numpy(dtype=float)
    return pd.DataFrame(data)


def compute_features(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
    winsor_quantile: float = _WINSOR_QUANTILE,
    min_obs: int = _MIN_XS_OBS,
) -> pd.DataFrame:
    """Compute all fundamental factors and normalize them STRICTLY per date.

    Parameters
    ----------
    panel : tidy ``(ticker, date, <fundamentals + price>)`` frame (see module docstring).
    method : ``"zscore"`` (default; per-date population z-score) or ``"rank"`` (per-date
        percentile rank mapped to ``[-1, 1]``).
    winsor_quantile : per-date tail clip before normalization (``0`` disables).
    min_obs : minimum finite names for a date's cross-section to be normalized; below
        this the whole date is ``NaN`` for that feature (never fabricated).

    Returns
    -------
    Tidy ``(ticker, date, <FEATURE_NAMES>)`` frame sorted by ``(date, ticker)``. Each
    feature is winsorized then normalized using ONLY same-date values — there is no
    cross-date statistic, so no row can leak information from another date. Missing
    inputs propagate as ``NaN`` (dropped from the cross-section, never imputed).

    Raises
    ------
    ValueError
        If ``winsor_quantile`` is not in ``[0.0, 0.5)`` (a tail clip must be a
        non-negative quantile strictly below the median; ``0`` disables winsorization).
    """
    if not (0.0 <= winsor_quantile < 0.5):
        raise ValueError(
            f"winsor_quantile must be in [0.0, 0.5), got {winsor_quantile!r}"
        )
    work = _prepare(panel)
    out = pd.DataFrame(
        {"ticker": work["ticker"].to_numpy(), "date": work["date"].to_numpy()}
    )
    dates = work["date"]
    for name, fn in FACTOR_FUNCTIONS.items():
        raw = fn(work)
        out[name] = _normalize_cross_section(
            raw, dates, method, winsor_quantile, min_obs
        ).to_numpy(dtype=float)
    return out
