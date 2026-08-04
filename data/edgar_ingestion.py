"""
TradingEngineResearch — SEC EDGAR fundamentals ingestion (Stage B / #1: richer cross-sectional data)
===========================================================================================
Free per-stock fundamentals from SEC EDGAR's XBRL company-facts API. Fundamental
characteristics (value / quality / profitability) are historically the STRONGEST
cross-sectional signals — far richer than price or Fama-French factor loadings.

POINT-IN-TIME CORRECTNESS (the whole game): a fundamental is only knowable once it has
been **filed**, not on its accounting period-end. Every accessor here is keyed on the
SEC ``filed`` date and uses only facts with ``filed <= asof`` — so a backtest can never
trade on an earnings number before it was public (golden rule 3; prevents the classic
look-ahead that inflates fundamental-factor backtests).

Live fetch hits ``data.sec.gov`` (requires a descriptive User-Agent per SEC policy) and
is excluded from coverage; the suite runs against the committed offline fixture
``tests/fixtures/edgar_fundamentals_sample.csv`` and never touches the network.

SURVIVORSHIP / TICKER-REASSIGNMENT CAVEAT (any real run inherits these): the ticker->CIK map
(:func:`ticker_to_cik_map`) is the CURRENT SEC mapping — it lists only currently registered
issuers (delisted / acquired / bankrupt names are gone, so any universe built from it is
survivorship-biased) AND tickers are recycled over time, so a present-day symbol can resolve
to a CIK whose history actually belongs to a different, now-defunct company (mis-attribution).
Free data cannot fix either; they are disclosed, not corrected.

DETERMINISM: :func:`extract_company_facts` is a pure, JSON-order-independent function of the
companyfacts payload (duration-period selection, concept coalescing, multi-class summation and
the debt proxy below all resolve deterministically), so a live run is reproducible.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "load_fundamentals",
    "ticker_to_cik_map",
    "fetch_concept_facts",
    "fetch_company_facts",
    "extract_company_facts",
    "company_facts_funds",
    "build_edgar_panel",
    "pit_fundamental",
    "fundamental_features",
    "DEFAULT_TAGS",
    "COMPANY_FACTS_TAGS",
    "EDGAR_PANEL_COLUMNS",
]

DEFAULT_TAGS = ("StockholdersEquity", "NetIncomeLoss", "Assets")
_UA = {"User-Agent": "TradingEngineResearch research 268190724+Leo-Y-Zhang@users.noreply.github.com"}

# ── companyfacts → 14-factor library bridge (Stage B / #1, free SEC fundamentals) ─────
#
# Map each PANEL column the 14-factor library consumes (research/fundamental_features.py) to the
# raw XBRL concepts that source it. companyfacts is messier than a flat tag list can express, so
# the bridge is SPEC-DRIVEN (:class:`_ColumnSpec`) to defend three documented correctness hazards:
#
#   • DURATION vs INSTANT periodicity. Income-statement / cash-flow concepts (NetIncomeLoss,
#     Revenues, GrossProfit, OperatingIncomeLoss, NCFO, EPS) are DURATIONS: companyfacts co-files
#     the 3-month, the year-to-date AND the trailing-12-month figure at the SAME (filed, end).
#     Dropping ``start`` and de-duplicating on (filed, end) alone would pick among them by JSON
#     order, silently mixing period lengths across issuers on different fiscal calendars and
#     corrupting every fundamental factor. We deterministically select ONE consistent ~quarterly
#     period (see :func:`_select_quarterly`). Balance-sheet concepts (StockholdersEquity, Assets,
#     debt, shares) are INSTANTS (no ``start``) and are left alone.
#   • CONCEPT SWITCHES. A leg's candidates are coalesced PER (filed, end) — e.g. the ASC-606
#     ``Revenues`` → ``RevenueFromContractWithCustomerExcludingAssessedTax`` (~2018) transition
#     is stitched across the switch instead of truncating one era to NaN.
#   • MULTI-CLASS SHARES. CommonStockSharesOutstanding is filed once per share class (GOOGL/META…);
#     the per-class rows are SUMMED per (filed, end) so the count is the consolidated total
#     (keeping a single row halves it and ~doubles the reconstructed market cap).
#
# A ``taxonomy:Concept`` selector picks a non-default taxonomy (``dei:`` rather than us-gaap:).

_DURATION = "duration"   # a flow over a period (income statement / cash flow): has start + end
_INSTANT = "instant"     # a balance-sheet stock at a point in time: has end only
_SHARES = "shares"       # an instant share COUNT, filed per share class → SUMMED per period

# Duration-period selection tolerances. A fiscal quarter is ~91 days; co-filed YTD / annual facts
# span ~180 / 270 / 365 days. Treat facts in [80, 100] days as "quarterly"; for a period with no
# quarterly fact (e.g. an annual-only 10-K, or a Q4 that only carries the 12-month figure) take
# the SHORTEST available duration and SCALE it to a quarter-equivalent so the periodicity stays
# consistent across the cross-section.
_QUARTER_DAYS = 365.25 / 4.0
_QUARTER_MIN_DAYS = 80.0
_QUARTER_MAX_DAYS = 100.0


@dataclass(frozen=True)
class _ColumnSpec:
    """How to source ONE canonical panel column from raw XBRL concepts.

    ``kind`` is the period semantics (:data:`_DURATION` / :data:`_INSTANT` / :data:`_SHARES`).
    ``legs`` is a tuple of *coalesce groups*; each group is an ordered tuple of candidate raw
    concepts and resolves, per (filed, end), to the FIRST candidate that reports a value (the
    per-period concept-switch stitch). The resolved legs are SUMMED to form the column — one leg
    is a simple coalesced concept; several legs compose an aggregate (interest-bearing debt =
    current debt + noncurrent debt). ``fallback`` is a last-resort coalesce group, used only for
    periods where EVERY leg is absent.
    """

    kind: str
    legs: tuple[tuple[str, ...], ...]
    fallback: tuple[str, ...] = ()


_CONCEPT_SPECS: dict[str, _ColumnSpec] = {
    "netinc": _ColumnSpec(_DURATION, (("NetIncomeLoss",),)),
    "equity": _ColumnSpec(_INSTANT, (("StockholdersEquity",),)),
    "assets": _ColumnSpec(_INSTANT, (("Assets",),)),
    "revenue": _ColumnSpec(
        _DURATION,
        (("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"),),
    ),
    "gp": _ColumnSpec(_DURATION, (("GrossProfit",),)),
    "ebit": _ColumnSpec(_DURATION, (("OperatingIncomeLoss",),)),
    "ncfo": _ColumnSpec(_DURATION, (("NetCashProvidedByUsedInOperatingActivities",),)),
    # Interest-bearing DEBT proxy = current debt + noncurrent debt. NOT total Liabilities (which
    # sweeps in payables / deferred revenue / operating leases — not borrowings). DebtCurrent and
    # LongTermDebtNoncurrent are mutually-exclusive portions (no double count); ShortTermBorrowings
    # / LongTermDebt are per-leg fallbacks (LongTermDebt can include the current maturity → a mild
    # double count only when it stands in for a missing LongTermDebtNoncurrent). Total Liabilities
    # is the LAST-RESORT proxy, used only when a company reports no debt line at all.
    "debt": _ColumnSpec(
        _INSTANT,
        (("DebtCurrent", "ShortTermBorrowings"), ("LongTermDebtNoncurrent", "LongTermDebt")),
        fallback=("Liabilities",),
    ),
    # Share COUNT, SUMMED per (filed, end) across share classes; us-gaap balance-sheet count
    # preferred, dei cover-page (consolidated) count the fallback.
    "sharesbas": _ColumnSpec(
        _SHARES,
        (("CommonStockSharesOutstanding", "dei:EntityCommonStockSharesOutstanding"),),
    ),
    "eps": _ColumnSpec(_DURATION, (("EarningsPerShareDiluted", "EarningsPerShareBasic"),)),
}

# Public, flattened {column: ordered candidate concepts} view (documentation / back-compat / the
# panel's value-column set). DERIVED from :data:`_CONCEPT_SPECS` so the two can never drift.
COMPANY_FACTS_TAGS: dict[str, tuple[str, ...]] = {
    col: tuple(dict.fromkeys(tag for leg in spec.legs for tag in leg)) + spec.fallback
    for col, spec in _CONCEPT_SPECS.items()
}

# Tidy panel column order build_edgar_panel emits — exactly the columns
# research.fundamental_features.compute_features consumes (no marketcap: SEC is free and
# does not publish prices/market cap; the value ratios that need it fall back or stay NaN).
EDGAR_PANEL_COLUMNS: tuple[str, ...] = (
    "ticker", "date", "price",
    "netinc", "equity", "assets", "revenue", "gp", "ebit", "ncfo", "debt", "sharesbas", "eps",
)
_EDGAR_VALUE_COLUMNS: tuple[str, ...] = tuple(COMPANY_FACTS_TAGS)  # the 10 fundamental columns


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    """Load the EDGAR fundamentals fixture (``ticker,tag,filed,period_end,value``).
    ``filed`` / ``period_end`` are parsed to timestamps; rows sorted by filing date."""
    df = pd.read_csv(path, parse_dates=["filed", "period_end"])
    need = ["ticker", "tag", "filed", "period_end", "value"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"EDGAR fundamentals CSV missing columns {missing}")
    return df[need].sort_values(["ticker", "tag", "filed", "period_end"]).reset_index(drop=True)


def ticker_to_cik_map() -> dict[str, int]:  # pragma: no cover - network
    """Fetch SEC's ticker → CIK map (~10k companies).

    CAVEAT: this is the CURRENT mapping — it contains only currently registered issuers
    (delisted / acquired / bankrupt names are absent → any universe built from it is
    survivorship-biased), and tickers are REASSIGNED over time, so a symbol can resolve to a
    CIK whose filing history belongs to a different, now-defunct company. Neither is fixable
    from free data; both are disclosed by the research runners, not corrected."""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=_UA)
    data = json.loads(urllib.request.urlopen(req, timeout=25).read())
    return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}


def fetch_concept_facts(cik: int, tag: str) -> pd.DataFrame:  # pragma: no cover - network
    """Fetch one us-gaap concept's full fact history for a company. Returns rows of
    ``(filed, period_end, value)`` (filing-date keyed → PIT-safe downstream)."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{int(cik):010d}/us-gaap/{tag}.json"
    cc = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=25).read())
    rows = []
    for _unit, facts in cc.get("units", {}).items():
        for u in facts:
            if u.get("filed") and u.get("end") and u.get("val") is not None:
                rows.append((pd.Timestamp(u["filed"]), pd.Timestamp(u["end"]), float(u["val"])))
    return pd.DataFrame(rows, columns=["filed", "period_end", "value"]).sort_values("filed")


def pit_fundamental(funds: pd.DataFrame, ticker: str, tag: str, asof) -> Optional[float]:
    """The latest value of ``tag`` for ``ticker`` that was **filed on or before** ``asof``
    — i.e. the most recent figure an investor could actually have known. Ties on filing
    date are broken by the most recent accounting period. Returns ``None`` if nothing was
    filed by ``asof`` (fail-closed: no fabricated fundamental)."""
    asof_ts = pd.Timestamp(asof)
    sub = funds[(funds["ticker"] == ticker) & (funds["tag"] == tag) & (funds["filed"] <= asof_ts)]
    if sub.empty:
        return None
    row = sub.sort_values(["filed", "period_end"]).iloc[-1]
    return float(row["value"])


def fundamental_features(
    funds: pd.DataFrame, ticker: str, asof, market_cap: Optional[float] = None
) -> dict[str, float]:
    """PIT-safe per-stock fundamental features as of ``asof``:

      • ``roe``  = NetIncome / StockholdersEquity     (profitability / quality)
      • ``roa``  = NetIncome / Assets                  (quality)
      • ``book_yield``     = Equity / market_cap       (value)   [needs market_cap]
      • ``earnings_yield`` = NetIncome / market_cap    (value)   [needs market_cap]

    Only fundamentals filed on or before ``asof`` are used. NOTE: NetIncomeLoss is taken
    as the latest filed value (a TTM-summation refinement is a follow-up); zero/None
    denominators are skipped (fail-closed)."""
    eq = pit_fundamental(funds, ticker, "StockholdersEquity", asof)
    ni = pit_fundamental(funds, ticker, "NetIncomeLoss", asof)
    assets = pit_fundamental(funds, ticker, "Assets", asof)
    feats: dict[str, float] = {}
    if ni is not None and eq not in (None, 0):
        feats["roe"] = ni / eq                       # type: ignore[operator]
    if ni is not None and assets not in (None, 0):
        feats["roa"] = ni / assets                   # type: ignore[operator]
    if market_cap and market_cap > 0:
        if eq is not None:
            feats["book_yield"] = eq / market_cap
        if ni is not None:
            feats["earnings_yield"] = ni / market_cap
    return feats


# --------------------------------------------------------------------------- #
# companyfacts: ALL us-gaap/dei concepts for a company in ONE request
# --------------------------------------------------------------------------- #
def fetch_company_facts(cik: int) -> dict:  # pragma: no cover - network
    """Fetch the ENTIRE XBRL fact set for one company in a SINGLE request.

    Hits ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json``, which returns
    every us-gaap (and dei) concept the company has ever reported — far fewer round-trips
    than :func:`fetch_concept_facts` (one call per company vs. one per concept). SEC requires
    a descriptive ``User-Agent`` (see :data:`_UA`); requests without one are throttled/denied.

    Returns the parsed JSON as a nested dict (``facts → taxonomy → concept → units → unit →
    [facts]``); pass it to :func:`extract_company_facts` to obtain a PIT-safe tidy frame.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
    req = urllib.request.Request(url, headers=_UA)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _facts_for_concept(
    facts: dict, raw_tag: str
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    """All ``(filed, start, period_end, value)`` rows for one XBRL concept across every unit.

    ``raw_tag`` is either a bare us-gaap concept (e.g. ``"Assets"``) or a ``"taxonomy:Concept"``
    selector (e.g. ``"dei:EntityCommonStockSharesOutstanding"``). Each fact must carry a filing
    date (``filed``), a period end (``end``) and a value (``val``); incomplete facts are skipped
    (fail-closed — nothing fabricated). ``start`` is captured when present (it is supplied for
    DURATION facts and absent → ``NaT`` for INSTANT facts) so :func:`_select_quarterly` can keep
    a consistent period length for flow concepts."""
    taxonomy, concept = raw_tag.split(":", 1) if ":" in raw_tag else ("us-gaap", raw_tag)
    node = facts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return []
    rows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]] = []
    for _unit, items in node.get("units", {}).items():
        for it in items:
            filed, end, val = it.get("filed"), it.get("end"), it.get("val")
            if filed and end and val is not None:
                start = it.get("start")
                rows.append(
                    (
                        pd.Timestamp(filed),
                        pd.Timestamp(start) if start else pd.NaT,
                        pd.Timestamp(end),
                        float(val),
                    )
                )
    return rows


def _select_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse co-filed DURATION facts to ONE quarter-consistent value per (filed, end).

    companyfacts files the 3-month, the year-to-date and the trailing-12-month figure for a flow
    at the SAME (filed, end). For each (filed, end): if any fact spans ~a quarter (period_end −
    start in [80, 100] days) keep the one closest to a calendar quarter (value unchanged);
    otherwise take the SHORTEST available duration and SCALE it to a quarter-equivalent
    (value × ~91.3 / duration_days) so an annual-only / YTD-only period lands on the same
    quarterly basis as the rest of the cross-section. Facts with no ``start`` keep their value as
    is. Selection is by a total ordering on (distance-to-a-quarter, duration, value), so the
    result is fully deterministic and independent of input row order."""
    work = df.copy()
    work["filed"] = pd.to_datetime(work["filed"], errors="coerce")
    work["start"] = pd.to_datetime(work["start"], errors="coerce")
    work["period_end"] = pd.to_datetime(work["period_end"], errors="coerce")
    dur_days = (work["period_end"] - work["start"]).dt.total_seconds() / 86_400.0
    work["__dur"] = dur_days
    work["__is_q"] = dur_days.between(_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS)
    work["__qdist"] = (dur_days - _QUARTER_DAYS).abs()
    records: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    for (filed, end), grp in work.groupby(["filed", "period_end"], sort=True):
        quarterly = grp[grp["__is_q"]]
        if not quarterly.empty:
            pick = quarterly.sort_values(["__qdist", "__dur", "value"], kind="mergesort").iloc[0]
            value = float(pick["value"])
        else:
            dated = grp[grp["__dur"] > 0.0]
            if not dated.empty:
                pick = dated.sort_values(["__dur", "value"], kind="mergesort").iloc[0]
                value = float(pick["value"]) * (_QUARTER_DAYS / float(pick["__dur"]))
            else:                                   # no usable duration (start absent): raw value
                pick = grp.sort_values("value", kind="mergesort").iloc[-1]
                value = float(pick["value"])
        records.append((pd.Timestamp(filed), pd.Timestamp(end), value))
    return pd.DataFrame(records, columns=["filed", "period_end", "value"])


def _concept_period_frame(facts: dict, raw_tag: str, kind: str) -> pd.DataFrame:
    """One representative ``(filed, period_end, value)`` row per (filed, end) for ONE concept.

    DURATION → the quarter-consistent value (:func:`_select_quarterly`); SHARES → the per-period
    SUM across share classes (fixes the multi-class under-count); INSTANT → the balance-sheet
    stock with duplicate contexts collapsed deterministically. An empty (typed) frame is returned
    if the concept is absent."""
    rows = _facts_for_concept(facts, raw_tag)
    if not rows:
        return pd.DataFrame(columns=["filed", "period_end", "value"])
    df = pd.DataFrame(rows, columns=["filed", "start", "period_end", "value"])
    if kind == _DURATION:
        return _select_quarterly(df)
    if kind == _SHARES:
        summed = df.groupby(["filed", "period_end"], as_index=False, sort=True)["value"].sum()
        return summed[["filed", "period_end", "value"]].reset_index(drop=True)
    # INSTANT: one stock per (filed, end); collapse duplicate contexts deterministically
    # (identical values at a given (filed, end) → no-op; any restated values → keep the largest).
    collapsed = df.sort_values(
        ["filed", "period_end", "value"], kind="mergesort"
    ).drop_duplicates(subset=["filed", "period_end"], keep="last")
    return collapsed[["filed", "period_end", "value"]].reset_index(drop=True)


def _coalesce_leg(facts: dict, candidates: tuple[str, ...], kind: str) -> pd.DataFrame:
    """Per (filed, end), the value from the FIRST candidate concept (priority order) that reports
    one — stitching concept switches (e.g. the ASC-606 ``Revenues`` →
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` transition) PER PERIOD rather than
    truncating a whole era to NaN (the bug where ``break`` dropped the second era)."""
    out = pd.DataFrame(columns=["filed", "period_end", "value"])
    for raw_tag in candidates:
        sub = _concept_period_frame(facts, raw_tag, kind)
        if sub.empty:
            continue
        if out.empty:
            out = sub
            continue
        have = out[["filed", "period_end"]].assign(__have=True)
        merged = sub.merge(have, on=["filed", "period_end"], how="left")
        fresh = merged.loc[merged["__have"].isna(), ["filed", "period_end", "value"]]
        out = pd.concat([out, fresh], ignore_index=True)
    if out.empty:
        return out
    return out.sort_values(["filed", "period_end"], kind="mergesort").reset_index(drop=True)


def _resolve_column(facts: dict, spec: _ColumnSpec) -> pd.DataFrame:
    """Resolve one panel column's ``(filed, period_end, value)`` from ``spec``: coalesce each
    leg's candidate concepts, SUM the legs per (filed, end) (a missing leg contributes 0 where at
    least one leg is present), and fall back to ``spec.fallback`` only when EVERY leg is absent."""
    legs = [_coalesce_leg(facts, leg, spec.kind) for leg in spec.legs]
    legs = [lf for lf in legs if not lf.empty]
    if not legs:
        if spec.fallback:
            return _coalesce_leg(facts, spec.fallback, spec.kind)
        return pd.DataFrame(columns=["filed", "period_end", "value"])
    if len(legs) == 1:
        return legs[0]
    combined = legs[0].rename(columns={"value": "__v0"})
    for i, leg in enumerate(legs[1:], start=1):
        combined = combined.merge(
            leg.rename(columns={"value": f"__v{i}"}), on=["filed", "period_end"], how="outer"
        )
    vcols = [c for c in combined.columns if c.startswith("__v")]
    combined["value"] = combined[vcols].sum(axis=1, min_count=1)
    out = combined.loc[combined["value"].notna(), ["filed", "period_end", "value"]]
    return out.sort_values(["filed", "period_end"], kind="mergesort").reset_index(drop=True)


def extract_company_facts(facts: dict, ticker: str) -> pd.DataFrame:
    """Tidy ``(ticker, tag, filed, period_end, value)`` frame for one company's companyfacts.

    ``tag`` is the CANONICAL panel column name (``netinc``/``equity``/… — see
    :data:`_CONCEPT_SPECS` / :data:`COMPANY_FACTS_TAGS`), NOT the raw XBRL concept. Each column is
    resolved per the documented correctness rules — consistent quarterly periodicity for flows
    (:func:`_select_quarterly`), per-period concept-switch coalescing (:func:`_coalesce_leg`),
    summed multi-class shares, and an interest-bearing debt proxy (:func:`_resolve_column`) — to
    exactly ONE value per ``(ticker, tag, filed, period_end)``. Filing-date keyed → PIT-safe
    downstream via :func:`build_edgar_panel`. ``ticker`` is attached by the caller (companyfacts
    carries only ``cik``/``entityName``). An empty (typed) frame is returned if the company
    reports none of the wanted concepts. The function is a pure, JSON-fact-order-INDEPENDENT
    function of ``facts`` (the whole point: a live run is deterministic and reproducible)."""
    records: list[tuple[str, str, pd.Timestamp, pd.Timestamp, float]] = []
    for col, spec in _CONCEPT_SPECS.items():
        resolved = _resolve_column(facts, spec)
        for filed, end, val in zip(
            resolved["filed"], resolved["period_end"], resolved["value"], strict=True
        ):
            records.append((ticker, col, pd.Timestamp(filed), pd.Timestamp(end), float(val)))
    out = pd.DataFrame(records, columns=["ticker", "tag", "filed", "period_end", "value"])
    if out.empty:
        return out
    return out.sort_values(
        ["ticker", "tag", "filed", "period_end"], kind="mergesort"
    ).reset_index(drop=True)


def company_facts_funds(cik: int, ticker: str) -> pd.DataFrame:  # pragma: no cover - network
    """Convenience: :func:`fetch_company_facts` then :func:`extract_company_facts` for one
    company. Returns the tidy canonical-tag frame ready to feed :func:`build_edgar_panel`."""
    return extract_company_facts(fetch_company_facts(cik), ticker)


# --------------------------------------------------------------------------- #
# Tidy panel for the 14-factor library (PIT forward-fill on the FILING date)
# --------------------------------------------------------------------------- #
def _tidy_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``prices`` to a tidy ``(ticker, date, price)`` frame.

    Accepts either a tidy frame (columns ``ticker``, ``date`` and ``price``/``close``,
    any case) or a WIDE close matrix (a ``date`` index or column, one column per ticker)
    which is melted to long form. ``date`` → timestamp, ``price`` → numeric."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    lower = {str(c).lower(): c for c in prices.columns}
    if "ticker" in lower and "date" in lower:
        price_key = next((lower[k] for k in ("price", "close", "closeadj") if k in lower), None)
        if price_key is None:
            raise ValueError("tidy prices frame needs a 'price' (or 'close'/'closeadj') column")
        out = prices[[lower["ticker"], lower["date"], price_key]].rename(
            columns={lower["ticker"]: "ticker", lower["date"]: "date", price_key: "price"}
        )
    else:
        wide = prices.set_index(lower["date"]) if "date" in lower else prices.copy()
        wide = wide.rename_axis("date")
        out = wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="price")
    out = out.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["date"])
    return out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def build_edgar_panel(funds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Tidy ``(ticker, date, price, <fundamentals>)`` panel for the 14-factor library.

    Emits exactly :data:`EDGAR_PANEL_COLUMNS` — the schema
    ``research.fundamental_features.compute_features`` consumes. For every price ``(ticker,
    date)`` each fundamental column carries the latest value whose **FILING date ≤ date** (a
    grouped, backward ``merge_asof`` — a forward-fill of KNOWN fundamentals onto the price
    grid). This is strict point-in-time: visibility is decided by ``filed`` (when the figure
    became public), NEVER by ``period_end`` (the accounting period it describes) — so a fact
    filed after ``date`` is never used at ``date`` (golden rule 3; the classic look-ahead
    that inflates fundamental backtests). Ties on ``filed`` resolve to the latest
    ``period_end`` (matching :func:`pit_fundamental`).

    Parameters
    ----------
    funds : tidy ``(ticker, tag, filed, value)`` frame with CANONICAL ``tag`` names (the
        output of :func:`extract_company_facts`; an optional ``period_end`` column refines
        same-``filed`` ties). Tickers/tags absent from ``funds`` yield an all-NaN column
        (never fabricated).
    prices : tidy ``(ticker, date, price)`` frame or a wide close matrix (see
        :func:`_tidy_prices`).

    Returns
    -------
    Tidy panel, one row per ``(ticker, date)`` in ``prices``, sorted by ``(ticker, date)``.
    Dates before a ticker's first known filing carry NaN fundamentals (fail-open as missing).

    NOTE (mild value-factor bias). The value ratios the panel feeds (book/price, sales/price,
    earnings yield) reconstruct ``marketcap = price[t] * sharesbas`` from the SAME ``price[t]``
    that the research runner uses as the BASE of the forward return ``price[t+1]/price[t] - 1``.
    A common ``price[t]`` therefore appears in both the value factor and the return denominator,
    a mild mechanical coupling (it does not violate point-in-time — both legs use only data
    known at ``t``) that should be kept in mind when reading value-factor results.
    """
    panel = _tidy_prices(prices)[["ticker", "date", "price"]]
    left = panel.sort_values("date", kind="mergesort").reset_index(drop=True)

    f = funds.copy()
    # Guard an empty / column-less funds frame (e.g. a universe where no name resolved) so we emit
    # the all-NaN fundamental columns rather than KeyError-ing on a missing tag/filed/value column.
    usable = (not f.empty) and {"ticker", "tag", "filed", "value"}.issubset(f.columns)
    sort_keys = ["ticker", "filed"]
    if usable:
        f["ticker"] = f["ticker"].astype(str)
        f["filed"] = pd.to_datetime(f["filed"], errors="coerce")
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        if "period_end" in f.columns:
            f["period_end"] = pd.to_datetime(f["period_end"], errors="coerce")
            sort_keys.append("period_end")

    for col in _EDGAR_VALUE_COLUMNS:
        if not usable:
            left[col] = np.nan
            continue
        sub = f[f["tag"] == col].dropna(subset=["filed", "value"])
        if sub.empty:
            left[col] = np.nan
            continue
        # One row per (ticker, filed): the latest period_end wins (mirror pit_fundamental's
        # ["filed", "period_end"] tie-break) so the backward merge_asof is unambiguous.
        right = (
            sub.sort_values(sort_keys, kind="mergesort")
            .drop_duplicates(subset=["ticker", "filed"], keep="last")[["ticker", "filed", "value"]]
            .sort_values("filed", kind="mergesort")
            .reset_index(drop=True)
        )
        merged = pd.merge_asof(
            left, right, left_on="date", right_on="filed", by="ticker", direction="backward"
        )
        left[col] = merged["value"].to_numpy(dtype=float)

    out = left[list(EDGAR_PANEL_COLUMNS)]
    return out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
