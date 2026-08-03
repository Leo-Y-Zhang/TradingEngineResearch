"""
TradingEngineResearch — PIT-safe monthly INSIDER-TRANSACTION feature panels
=================================================================
Turns the tidy Form-4 transactions frame produced by ``data.insider_ingestion`` into
monthly ``(ticker, date, feature...)`` panels, mirroring the style and normalization
conventions of ``research.fundamental_features`` (per-date winsorize + z-score via the
SAME helpers — no cross-date statistic anywhere).

FEATURE SET — FIXED A PRIORI (pre-registered in
``research/medallion_style_alpha_search/insider_study_prereg.md``; do NOT add features
after seeing results):

  * ``net_buy_ratio_6m``      (n_buys - n_sells) / (n_buys + n_sells), trailing 6 months,
                              open-market P/S by OFFICERS+DIRECTORS only, count-based.
  * ``net_buy_value_6m``      same, dollar-value weighted (shares * price; transactions
                              with no reported price contribute counts but no value).
                              A multi-owner filing's fan-out (one row per co-reporting
                              owner, same accession) contributes its dollar value ONCE —
                              the 2026-07 adversarial review measured 41% of purchase-leg
                              value duplicated without accession-level dedup. COUNT
                              features keep their registered per-owner-row semantics.
  * ``cluster_buying_3m``     number of DISTINCT officer/director owner CIKs with a P
                              purchase in the trailing 3 months (0 when there is P/S
                              activity but no buyers; NaN when there is no activity).
  * ``opportunistic_buy_6m``  like ``net_buy_ratio_6m`` but EXCLUDING routine insiders —
                              Cohen-Malloy-Pomorski (2012) simplified: a trade is
                              "routine" if the same owner filed a P purchase for the same
                              issuer in the SAME calendar month in EACH of the 3 preceding
                              years. Detection is PIT by construction: it only queries
                              strictly EARLIER years' purchases.
  * ``buy_intensity_6m``      total P-purchase dollar value over the trailing 6 months
                              (the per-date cross-sectional z-score does the scaling; no
                              shares-outstanding data required).

POINT-IN-TIME DISCIPLINE (golden rule 3):
  * The ONLY availability timestamp is ``filing_date``; a transaction becomes usable at
    ``filing_date + 1 BUSINESS day`` (conservatism against late-day EDGAR postings).
  * A filing enters the month whose month-end is >= that availability date — so a filing
    ON the month-end day lands in the NEXT month.
  * Belt-and-braces holiday guard: if the availability date falls AFTER that month's
    PANEL date (e.g. a Memorial-Day May 31st when the last trading day was the 28th),
    the filing is pushed to the NEXT month — a signal at panel date ``t`` can never see
    a filing that was not public strictly before ``t``'s close.
  * Amendments (``is_amendment=True``) are excluded — as-filed data only.
  * Symbols/months with no qualifying activity are ``NaN`` (never fabricated); the
    runner neutral-fills 0.0 AFTER normalization, matching ``research_free_alpha``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from research.fundamental_features import (
    _MIN_XS_OBS,
    _WINSOR_QUANTILE,
    _normalize_cross_section,
)

__all__ = [
    "INSIDER_FEATURES",
    "is_officer_or_director",
    "raw_insider_features",
    "compute_insider_features",
]

INSIDER_FEATURES: list[str] = [
    "net_buy_ratio_6m",
    "net_buy_value_6m",
    "cluster_buying_3m",
    "opportunistic_buy_6m",
    "buy_intensity_6m",
]

_NET_WINDOW_M = 6         # trailing window (calendar months, inclusive) for 6m features
_CLUSTER_WINDOW_M = 3     # trailing window for cluster_buying
_ROUTINE_YEARS = 3        # same-calendar-month purchases in EACH of the 3 preceding years

_REQUIRED_COLUMNS = ("ticker", "filing_date", "trans_code", "shares", "price",
                     "owner_cik", "relationship", "is_amendment", "accession",
                     "trans_date", "shrs_owned_after")

# One ECONOMIC transaction (fanned out once per co-reporting owner under the same
# accession) is identified by everything except the owner: dollar value is counted once
# per identity, while per-owner rows stay for the count/cluster features.
_VALUE_IDENTITY_COLUMNS = ["accession", "trans_date", "trans_code", "shares", "price",
                           "shrs_owned_after"]


def is_officer_or_director(relationship: pd.Series) -> pd.Series:
    """True where the (uppercased) SEC relationship string names an officer or director
    (combined strings such as ``DIRECTOR,OFFICER`` or ``DIRECTOROTHER`` match; pure
    ``TENPERCENTOWNER`` / ``OTHER`` do not)."""
    rel = relationship.fillna("").astype(str).str.upper()
    return rel.str.contains("OFFICER") | rel.str.contains("DIRECTOR")


def _availability_dates(filing_dates: pd.Series) -> pd.Series:
    """``filing_date + 1 business day`` — the first date a filing may influence a signal.
    Weekend filings roll back to the prior business day first, so Sat/Sun filings become
    usable on Monday (still strictly after the filing hit EDGAR)."""
    days = filing_dates.to_numpy(dtype="datetime64[D]")
    avail = np.busday_offset(days, 1, roll="backward")
    return pd.Series(pd.DatetimeIndex(avail), index=filing_dates.index)


def _panel_dates(dates: Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
    if len(idx) == 0:
        raise ValueError("insider features: at least one panel date is required")
    periods = idx.to_period("M")
    if periods.duplicated().any():
        raise ValueError("insider features: panel dates must contain at most one date "
                         "per calendar month (month-end grid)")
    return idx


def _effective_periods(avail: pd.Series, panel_idx: pd.DatetimeIndex) -> pd.PeriodIndex:
    """Month bucket per availability date: its own calendar month, pushed to the NEXT
    month when the availability date falls after that month's panel date (the holiday
    month-end guard described in the module docstring)."""
    periods = pd.PeriodIndex(avail, freq="M")
    panel_by_period = pd.Series(panel_idx, index=panel_idx.to_period("M"))
    panel_date_for_row = panel_by_period.reindex(periods).to_numpy()
    late = pd.notna(panel_date_for_row) & (avail.to_numpy() > panel_date_for_row)
    return pd.PeriodIndex(np.where(late, periods + 1, periods), freq="M")


def _prepare_events(
    transactions: pd.DataFrame, panel_idx: pd.DatetimeIndex, symbols: list[str]
) -> pd.DataFrame:
    """Qualifying events: as-filed (non-amendment) open-market P/S transactions by
    officers/directors of the requested symbols, with availability month buckets and
    the PIT routine flag attached."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"insider features: transactions frame missing column(s) {missing}")

    ev = transactions.copy()
    ev["ticker"] = ev["ticker"].astype(str).str.strip().str.upper()
    shares = pd.to_numeric(ev["shares"], errors="coerce")
    code = ev["trans_code"].astype(str).str.strip().str.upper()
    keep = (
        ev["ticker"].isin(set(symbols))
        & ~ev["is_amendment"].astype(bool)
        & code.isin(("P", "S"))
        & is_officer_or_director(ev["relationship"])
        & pd.to_datetime(ev["filing_date"]).notna()
        & np.isfinite(shares.fillna(np.nan))
        & (shares > 0)
    )
    ev = ev[keep].copy()
    if ev.empty:
        return pd.DataFrame(columns=["ticker", "period", "is_buy", "value", "owner_cik",
                                     "routine", "value_once"])

    ev["filing_date"] = pd.to_datetime(ev["filing_date"])
    avail = _availability_dates(ev["filing_date"])
    ev["period"] = _effective_periods(avail, panel_idx)
    ev["is_buy"] = ev["trans_code"].astype(str).str.strip().str.upper() == "P"
    price = pd.to_numeric(ev["price"], errors="coerce")
    ev["value"] = pd.to_numeric(ev["shares"], errors="coerce") * price
    ev["owner_cik"] = ev["owner_cik"].fillna("").astype(str).str.strip()
    # Accession-level value dedup: the first row of each economic transaction carries
    # the dollar value; its per-owner siblings keep counting but contribute no value.
    ev["value_once"] = ~ev.duplicated(subset=_VALUE_IDENTITY_COLUMNS, keep="first")

    # Routine flag (PIT): a trade in calendar month m of year y is routine iff the same
    # owner filed a P purchase for the same issuer bucketed in month m of years
    # y-1 .. y-3. Only strictly earlier years are ever queried, so knowing the full
    # purchase history cannot leak future information into the flag.
    purchase_keys = {
        (row.owner_cik, row.ticker, row.period.month, row.period.year)
        for row in ev[ev["is_buy"] & (ev["owner_cik"] != "")].itertuples()
    }
    ev["routine"] = [
        bool(row.owner_cik)
        and all(
            (row.owner_cik, row.ticker, row.period.month, row.period.year - k)
            in purchase_keys
            for k in range(1, _ROUTINE_YEARS + 1)
        )
        for row in ev.itertuples()
    ]
    # Buckets after the last panel month can never be observed on this grid.
    ev = ev[ev["period"] <= panel_idx[-1].to_period("M")]
    return ev[["ticker", "period", "is_buy", "value", "owner_cik", "routine",
               "value_once"]]


def _wide(agg: pd.DataFrame, col: str, periods: pd.PeriodIndex,
          symbols: list[str]) -> pd.DataFrame:
    w = agg.pivot(index="period", columns="ticker", values=col)
    return w.reindex(index=periods, columns=symbols).fillna(0.0).astype(float)


def _rolling_sum(w: pd.DataFrame, window: int) -> pd.DataFrame:
    return w.rolling(window, min_periods=1).sum()


def _ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    return num / den.where(den > 0.0)


def _distinct_buyers(
    events: pd.DataFrame, periods: pd.PeriodIndex, symbols: list[str], window: int
) -> pd.DataFrame:
    """(period × ticker) count of DISTINCT buying owner CIKs over a trailing window of
    ``window`` months (set-union — the same owner buying twice counts once)."""
    buyers = (
        events[events["is_buy"] & (events["owner_cik"] != "")]
        .groupby(["ticker", "period"])["owner_cik"]
        .agg(set)
    )
    out = np.zeros((len(periods), len(symbols)))
    for j, sym in enumerate(symbols):
        recent: list[set[str]] = []
        for i, p in enumerate(periods):
            recent.append(buyers.get((sym, p), set()))
            if len(recent) > window:
                recent.pop(0)
            out[i, j] = float(len(set().union(*recent)))
    return pd.DataFrame(out, index=periods, columns=symbols)


def raw_insider_features(
    transactions: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    symbols: Sequence[str],
) -> pd.DataFrame:
    """RAW (un-normalized) insider features on the full ``dates × symbols`` grid.

    Returns a tidy ``(ticker, date, <INSIDER_FEATURES>)`` frame with one row per
    (panel date, symbol). Cells with no qualifying activity in the trailing window are
    ``NaN`` — never fabricated. See the module docstring for the PIT rules."""
    panel_idx = _panel_dates(dates)
    syms = [str(s).strip().upper() for s in symbols]
    grid = pd.DataFrame({
        "ticker": np.tile(np.asarray(syms, dtype=object), len(panel_idx)),
        "date": np.repeat(panel_idx.to_numpy(), len(syms)),
    })

    events = _prepare_events(transactions, panel_idx, syms)
    if events.empty:
        for name in INSIDER_FEATURES:
            grid[name] = np.nan
        return grid

    panel_periods = panel_idx.to_period("M")
    start = min(events["period"].min(), panel_periods[0] - (_NET_WINDOW_M - 1))
    periods = pd.period_range(start=start, end=panel_periods[-1], freq="M")

    flat = events.assign(
        n_buy=events["is_buy"].astype(float),
        n_sell=(~events["is_buy"]).astype(float),
        buy_val=events["value"].where(events["is_buy"] & events["value_once"]).fillna(0.0),
        sell_val=events["value"].where(~events["is_buy"] & events["value_once"]).fillna(0.0),
        opp_buy=(events["is_buy"] & ~events["routine"]).astype(float),
        opp_sell=(~events["is_buy"] & ~events["routine"]).astype(float),
    )
    agg = (
        flat.groupby(["ticker", "period"], as_index=False)
        [["n_buy", "n_sell", "buy_val", "sell_val", "opp_buy", "opp_sell"]]
        .sum()
    )

    b6 = _rolling_sum(_wide(agg, "n_buy", periods, syms), _NET_WINDOW_M)
    s6 = _rolling_sum(_wide(agg, "n_sell", periods, syms), _NET_WINDOW_M)
    bv6 = _rolling_sum(_wide(agg, "buy_val", periods, syms), _NET_WINDOW_M)
    sv6 = _rolling_sum(_wide(agg, "sell_val", periods, syms), _NET_WINDOW_M)
    ob6 = _rolling_sum(_wide(agg, "opp_buy", periods, syms), _NET_WINDOW_M)
    os6 = _rolling_sum(_wide(agg, "opp_sell", periods, syms), _NET_WINDOW_M)
    act3 = _rolling_sum(_wide(agg, "n_buy", periods, syms)
                        + _wide(agg, "n_sell", periods, syms), _CLUSTER_WINDOW_M)
    act6 = b6 + s6

    cluster = _distinct_buyers(events, periods, syms, _CLUSTER_WINDOW_M)
    feats = {
        "net_buy_ratio_6m": _ratio(b6 - s6, act6),
        "net_buy_value_6m": _ratio(bv6 - sv6, bv6 + sv6),
        "cluster_buying_3m": cluster.where(act3 > 0.0),
        "opportunistic_buy_6m": _ratio(ob6 - os6, ob6 + os6),
        "buy_intensity_6m": bv6.where(act6 > 0.0),
    }
    for name in INSIDER_FEATURES:
        grid[name] = feats[name].loc[panel_periods].to_numpy(dtype=float).ravel()
    return grid


def compute_insider_features(
    transactions: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    symbols: Sequence[str],
    *,
    method: str = "zscore",
    winsor_quantile: float = _WINSOR_QUANTILE,
    min_obs: int = _MIN_XS_OBS,
) -> pd.DataFrame:
    """Insider features normalized STRICTLY per date (winsorize then z-score/rank WITHIN
    each cross-section via the ``research.fundamental_features`` helpers — identical
    conventions, no cross-date statistic, NaNs preserved).

    Returns a tidy ``(ticker, date, <INSIDER_FEATURES>)`` frame covering the full
    ``dates × symbols`` grid. Symbols with no insider activity remain ``NaN``; the
    runner neutral-fills 0.0 at the combination layer only."""
    if not (0.0 <= winsor_quantile < 0.5):
        raise ValueError(f"winsor_quantile must be in [0.0, 0.5), got {winsor_quantile!r}")
    raw = raw_insider_features(transactions, dates, symbols)
    out = raw[["ticker", "date"]].copy()
    for name in INSIDER_FEATURES:
        out[name] = _normalize_cross_section(
            raw[name], raw["date"], method, winsor_quantile, min_obs
        ).to_numpy(dtype=float)
    return out
