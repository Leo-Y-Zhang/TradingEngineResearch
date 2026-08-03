"""
TradingEngineResearch — Sharadar (Nasdaq Data Link) point-in-time, survivorship-free ingestion
======================================================================================
Loads the two Sharadar tables most useful for cross-sectional equity research from a
**configurable LOCAL path** (a CSV file, several CSVs, or a directory of CSVs) — never a
live API — so the loader is fully testable offline and drops straight onto a bulk export.

  • **SF1** — fundamentals (one row per ticker / dimension / filing period).
  • **SEP** — daily prices (one row per ticker / date).

Two properties are the whole point of paying for this data and they are enforced here:

POINT-IN-TIME (PIT) CORRECTNESS — the most important rule
---------------------------------------------------------
A fundamental is only knowable once the filing that contained it was *published*. Sharadar
gives that publication date as ``datekey``; ``calendardate`` is merely the accounting
period the figure DESCRIBES. Using ``calendardate`` leaks the future (you would "know" a
Q2 number on the last day of Q2, weeks before it was filed). **Every accessor here keys on
``datekey`` and uses only rows with ``datekey <= asof``** — `calendardate` is carried for
reference and grouping but is NEVER used to decide visibility.

SURVIVORSHIP-FREEDOM
--------------------
The panel is built from EVERY ticker present in the price table, including delisted /
acquired / bankrupt names. We never restrict to a "currently listed" universe, so a
backtest sees the same dead tickers a trader saw in real time (the classic survivorship
bias that inflates historical returns is avoided).

The headline output, :func:`build_panel`, is a tidy LONG panel keyed ``(ticker, date)``:
each price date carries the most recent filing whose ``datekey <= date`` (a forward-fill of
known fundamentals onto the price grid via a grouped ``merge_asof``). Dates before a
ticker's first known filing carry NaN fundamentals (fail-open as missing — nothing is
fabricated).

Assumed Sharadar columns (and how to adapt if your export differs)
------------------------------------------------------------------
Column names are lower-cased on load, so a differently-cased export (``Ticker`` /
``DATEKEY``) is handled automatically.

SF1 (fundamentals) — required keys + the fundamentals we retain:
  required : ``ticker``, ``dimension``, ``datekey``, ``calendardate``
  retained : ``revenue, netinc, equity, assets, liabilities, eps, ebit, ebitda, gp,
              ncfo, debt, sharesbas`` (the intersection actually present is kept)
  ``dimension`` selects the view. Sharadar ships ``AR*`` (As-Reported — the *original*
  filing, never restated) and ``MR*`` (Most-Recent — restated/back-filled) variants in
  ``Q`` (quarterly), ``T`` (trailing-twelve-month) and ``Y`` (annual). For a PIT backtest
  you almost always want an **AR** dimension (default ``"ARQ"``): the MR* views silently
  rewrite history and reintroduce look-ahead even though ``datekey`` looks honest.

SEP (prices) — required keys + price/volume:
  required : ``ticker``, ``date``
  price    : ``closeadj`` (split/dividend-adjusted; preferred) else ``close`` (raw)
  optional : ``volume``

To adapt to a renamed export: either rename your columns to the names above before
loading, OR pass ``value_columns=`` to override which fundamentals are retained. If your
publication-date column is named something other than ``datekey``, rename it to ``datekey``
first — PIT correctness here is defined entirely by that column.

This module reads files only; the live Nasdaq-Data-Link pull is intentionally out of scope
(do it once with the official client to produce the local CSV/bulk export this consumes).
For a very large SEP export, read it once and persist the wide close matrix; ``pandas``
``read_csv`` is used directly here for simplicity and offline determinism.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SF1_KEY_COLUMNS",
    "SF1_FUNDAMENTAL_COLUMNS",
    "SEP_KEY_COLUMNS",
    "AS_REPORTED_DIMENSIONS",
    "DEFAULT_DIMENSION",
    "load_sf1",
    "load_sep",
    "pit_fundamentals",
    "pit_value",
    "build_panel",
]

# ── Schema constants (see the module docstring for adaptation guidance) ──────────────
SF1_KEY_COLUMNS = ("ticker", "dimension", "datekey", "calendardate")
SF1_FUNDAMENTAL_COLUMNS = (
    "revenue", "netinc", "equity", "assets", "liabilities",
    "eps", "ebit", "ebitda", "gp", "ncfo", "debt", "sharesbas",
)
SEP_KEY_COLUMNS = ("ticker", "date")
# As-Reported dimensions preserve the ORIGINAL filing (PIT-safe); MR* are restated.
AS_REPORTED_DIMENSIONS = ("ARQ", "ART", "ARY")
DEFAULT_DIMENSION = "ARQ"

# A path to a single CSV, a directory of CSVs, or an iterable of CSV paths.
PathLike = Union[str, Path]
PathSource = Union[PathLike, Iterable[PathLike]]


def _read_frames(
    source: PathSource,
    pattern: str = "*.csv",
    usecols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Read a single CSV, every ``pattern`` CSV in a directory, or an iterable of CSV
    paths, and concatenate them into one DataFrame (column union preserved).

    ``usecols`` prunes columns AT READ TIME (case-insensitive name match): the full SF1
    export carries ~110 columns and SEP several GB of never-used ones — materializing
    them first would exhaust memory on the real export before the loaders' own column
    selection ran. Columns not present in a file are simply absent (the loaders'
    required-column checks still fire on the result)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = sorted(path.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No files matching {pattern!r} under {path}")
        elif path.exists():
            files = [path]
        else:
            raise FileNotFoundError(f"No such file or directory: {path}")
    else:
        files = [Path(p) for p in source]
        if not files:
            raise ValueError("Empty path iterable passed to Sharadar loader")

    selector = None
    if usecols is not None:
        wanted = {str(c).strip().lower() for c in usecols}
        def selector(name: str) -> bool:  # noqa: E306 - tiny read-time closure
            return str(name).strip().lower() in wanted

    frames = [pd.read_csv(f, usecols=selector) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case + strip column names so a differently-cased export still matches."""
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame


def load_sf1(
    source: PathSource,
    dimension: Optional[str] = None,
    value_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load Sharadar **SF1** fundamentals from a local CSV / directory / list of CSVs.

    Returns a tidy DataFrame with columns ``ticker, dimension, datekey, calendardate`` plus
    every retained fundamental that is present (see :data:`SF1_FUNDAMENTAL_COLUMNS`, or pass
    ``value_columns`` to override). ``datekey`` (the PUBLICATION date — the PIT timestamp)
    and ``calendardate`` (the accounting period) are parsed to timestamps; ``ticker`` and
    ``dimension`` are upper-cased; fundamentals are coerced to numeric (bad cells → NaN).
    Rows without a parseable ``datekey`` are dropped (they cannot be placed in time). If
    ``dimension`` is given, only that view is kept (default ``None`` = keep all; pass
    ``"ARQ"`` for the PIT-safe as-reported quarterly view). Sorted by
    ``(ticker, datekey, calendardate)``.
    """
    wanted_values = tuple(value_columns) if value_columns is not None else SF1_FUNDAMENTAL_COLUMNS
    frame = _normalise_columns(
        _read_frames(source, usecols=(*SF1_KEY_COLUMNS, *wanted_values))
    )
    missing = [c for c in SF1_KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Sharadar SF1 CSV missing required columns {missing}")

    frame["datekey"] = pd.to_datetime(frame["datekey"], errors="coerce")
    frame["calendardate"] = pd.to_datetime(frame["calendardate"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["dimension"] = frame["dimension"].astype(str).str.upper()

    present = [c for c in wanted_values if c in frame.columns]
    out = frame[[*SF1_KEY_COLUMNS, *present]].copy()
    for col in present:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _select_dimension(out, dimension)

    out = out.dropna(subset=["datekey"])
    return out.sort_values(["ticker", "datekey", "calendardate"]).reset_index(drop=True)


def load_sep(source: PathSource, use_adjusted: bool = True) -> pd.DataFrame:
    """Load Sharadar **SEP** prices from a local CSV / directory / list of CSVs.

    Returns a tidy DataFrame ``ticker, date, close, volume``. ``close`` is the
    split/dividend-adjusted ``closeadj`` when present and ``use_adjusted`` is True, else the
    raw ``close``. ``date`` is parsed to a timestamp, ``ticker`` upper-cased, ``close`` /
    ``volume`` coerced to numeric (``volume`` NaN if the export omits it). Rows without a
    parseable ``date`` are dropped. **Survivorship-free**: every ticker present — including
    delisted ones — is retained; nothing is filtered to a current universe. Sorted by
    ``(ticker, date)``.
    """
    frame = _normalise_columns(
        _read_frames(source, usecols=(*SEP_KEY_COLUMNS, "close", "closeadj", "volume"))
    )
    missing = [c for c in SEP_KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Sharadar SEP CSV missing required columns {missing}")

    if use_adjusted and "closeadj" in frame.columns:
        price_col = "closeadj"
    elif "close" in frame.columns:
        price_col = "close"
    elif "closeadj" in frame.columns:
        price_col = "closeadj"
    else:
        raise ValueError("Sharadar SEP CSV needs a 'closeadj' or 'close' column")

    out = pd.DataFrame({
        "ticker": frame["ticker"].astype(str).str.upper(),
        "date": pd.to_datetime(frame["date"], errors="coerce"),
        "close": pd.to_numeric(frame[price_col], errors="coerce"),
    })
    out["volume"] = (
        pd.to_numeric(frame["volume"], errors="coerce")
        if "volume" in frame.columns
        # float("nan") (NOT pd.NA): pandas 2.2 cannot fill a float64 buffer with the
        # pd.NA scalar (TypeError), and an all-NaN float64 column is what callers expect.
        else pd.Series(float("nan"), index=frame.index, dtype="float64")
    )
    out = out.dropna(subset=["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _select_dimension(frame: pd.DataFrame, dimension: Optional[str]) -> pd.DataFrame:
    """Restrict an SF1 frame to a single ``dimension``. ``None`` or ``"ALL"`` is the
    explicit OPT-OUT — it keeps every view (and so reintroduces restated MR* rows); any
    other value selects exactly that view. The PIT-facing accessors default to the
    As-Reported quarterly view, so opting out is a deliberate, visible choice."""
    if dimension is None or str(dimension).upper() == "ALL":
        return frame
    return frame[frame["dimension"] == str(dimension).upper()]


def _order_filings(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic 'latest-known' filing order: ascending by ``(datekey, calendardate)``
    and, for rows tied on BOTH, As-Reported (``AR*``) rows sort LAST so the caller's
    ``.iloc[-1]`` prefers the original filing over a restated ``MR*`` row sharing the same
    publication timestamp (a restatement must never out-rank the original it replaced)."""
    ar_pref = frame["dimension"].astype(str).str.startswith("AR").astype(int)
    return (
        frame.assign(_ar_pref=ar_pref)
        .sort_values(["datekey", "calendardate", "_ar_pref"], kind="mergesort")
        .drop(columns="_ar_pref")
    )


def pit_fundamentals(
    sf1: pd.DataFrame,
    ticker: str,
    asof,
    dimension: Optional[str] = DEFAULT_DIMENSION,
) -> Optional[pd.Series]:
    """The latest SF1 filing for ``ticker`` **known on or before** ``asof``.

    Filters on ``datekey <= asof`` (NEVER ``calendardate`` — that would leak the future)
    and returns the most recent row (ties on ``datekey`` broken by the latest
    ``calendardate``, then by preferring an As-Reported ``AR*`` row over a restated ``MR*``
    row) as a ``pd.Series``. Returns ``None`` if nothing had been filed by ``asof``
    (fail-closed: no fabricated fundamental).

    ``dimension`` defaults to :data:`DEFAULT_DIMENSION` (``"ARQ"``) — **fail-closed**: the
    PIT-safe as-reported quarterly view. Pass ``dimension=None`` or ``"ALL"`` to explicitly
    opt out and consider every view (which reintroduces look-ahead via restated MR* rows).
    """
    asof_ts = pd.Timestamp(asof)
    sub = sf1[(sf1["ticker"] == str(ticker).upper()) & (sf1["datekey"] <= asof_ts)]
    sub = _select_dimension(sub, dimension)
    if sub.empty:
        return None
    return _order_filings(sub).iloc[-1]


def pit_value(
    sf1: pd.DataFrame,
    ticker: str,
    column: str,
    asof,
    dimension: Optional[str] = DEFAULT_DIMENSION,
) -> Optional[float]:
    """One fundamental ``column`` from the latest filing known by ``asof`` (PIT-safe).

    Thin wrapper over :func:`pit_fundamentals` (same fail-closed ``dimension`` default).
    Returns ``None`` if no filing was known by ``asof`` or the value is missing/NaN."""
    row = pit_fundamentals(sf1, ticker, asof, dimension=dimension)
    if row is None or column not in row.index:
        return None
    val = row[column]
    return None if pd.isna(val) else float(val)


def build_panel(
    sf1: pd.DataFrame,
    sep: pd.DataFrame,
    dimension: Optional[str] = DEFAULT_DIMENSION,
    value_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Tidy LONG panel keyed ``(ticker, date)`` = prices + latest KNOWN fundamentals.

    For every price row, the most recent SF1 filing whose ``datekey <= date`` is attached
    (a grouped, backward ``merge_asof`` — i.e. a forward-fill of known fundamentals onto the
    price grid). This is **PIT-safe** (the filing's ``datekey`` never exceeds the price
    ``date``) and **survivorship-free** (the universe is exactly the tickers in ``sep``,
    delisted names included — they are never dropped).

    Output columns: ``ticker, date, close, volume`` then the retained fundamentals, plus
    ``filed_datekey`` (the publication date of the attached filing) and
    ``fundamental_calendardate`` (its accounting period) for auditability. Price dates
    before a ticker's first known filing carry NaN fundamentals / NaT ``filed_datekey``
    (nothing fabricated). One row per ``(ticker, date)``; sorted by ``(ticker, date)``.

    ``dimension`` defaults to :data:`DEFAULT_DIMENSION` (``"ARQ"``) — **fail-closed** to the
    PIT-safe As-Reported quarterly view; pass ``None`` or ``"ALL"`` to opt out and carry
    every view (reintroducing restated MR* rows). ``value_columns`` overrides which
    fundamentals are carried.
    """
    funds = _select_dimension(sf1, dimension)

    wanted = tuple(value_columns) if value_columns is not None else SF1_FUNDAMENTAL_COLUMNS
    present = [c for c in wanted if c in funds.columns]

    # merge_asof needs the right frame globally sorted by its on-key (datekey). Sort
    # additionally by calendardate (so a datekey tie resolves IDENTICALLY to
    # pit_fundamentals regardless of caller input order — determinism) and by AR-preference
    # (so the backward merge_asof, which picks the LAST row among a datekey tie, prefers an
    # original AR* filing over a restated MR* one — agreeing with pit_fundamentals).
    right = (
        funds[["ticker", "datekey", "calendardate", "dimension", *present]]
        .dropna(subset=["datekey"])
        .assign(_ar_pref=lambda d: d["dimension"].astype(str).str.startswith("AR").astype(int))
        .sort_values(["datekey", "calendardate", "_ar_pref"], kind="mergesort")
        .drop(columns=["dimension", "_ar_pref"])
        .reset_index(drop=True)
    )
    left = (
        sep[["ticker", "date", "close", "volume"]]
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="datekey",
        by="ticker",
        direction="backward",
    )
    merged = merged.rename(
        columns={"datekey": "filed_datekey", "calendardate": "fundamental_calendardate"}
    )
    ordered = ["ticker", "date", "close", "volume", *present,
               "filed_datekey", "fundamental_calendardate"]
    merged = merged[ordered]
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)
