"""
TradingEngineResearch — rename-safe mapping of insider transactions onto a ticker universe
================================================================================
The SEC Form-4 tidy frame identifies issuers by the *free-text trading symbol as filed*.
Joining that symbol directly against a current-listed universe silently loses every
issuer that ever renamed (GOOG→GOOGL, FB→META, UTX→RTX, PCLN→BKNG, WLP/ANTM→ELV,
MHP/MHFI→SPGI, FPL→NEE, KFT→MDLZ) and every formatting variant (pre-2009 ``(KO)``,
``GE:``) — the adversarial review of the first insider study measured ~22% of the
universe's matched rows lost this way, with Alphabet effectively absent for 20 years.

This module bridges by ISSUER CIK instead:

  1. Symbols are NORMALIZED (uppercase, non-alphanumerics stripped) so ``(KO)`` == ``KO``.
  2. A CIK is a CANDIDATE for universe ticker ``T`` when it EVER filed under normalized
     ``T``; ALL of that CIK's rows (any as-filed symbol, any era) then map to ``T``.
  3. A CIK claiming several universe tickers is assigned to the one it filed under most
     RECENTLY (no row is ever double-counted).
  4. Rows with no issuer CIK (older vintages) fall back to the normalized-symbol match.
  5. ``extra_ciks`` adds audited predecessor CIKs a symbol bridge cannot reach (corporate
     reorgs that changed CIK, e.g. Google Inc → Alphabet Inc); ``exclude_ciks`` removes
     audited ticker-recycling pollution (an unrelated company that once used the symbol).

The bridge is intentionally conservative-by-audit: :func:`build_universe_cik_map`
produces the per-(ticker, CIK) evidence table (row counts, filing-date ranges, issuer
names, symbols used) that a human/agent audit reviews before a study run banks results.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping

import pandas as pd

__all__ = [
    "normalize_symbol",
    "build_universe_cik_map",
    "map_transactions_to_universe",
]

logger = logging.getLogger(__name__)

_REQUIRED = ("ticker", "filing_date", "issuer_cik")


def normalize_symbol(symbol: str) -> str:
    """Uppercase and strip every non-alphanumeric character (``'(ko)'`` → ``'KO'``)."""
    return re.sub(r"[^A-Z0-9]", "", str(symbol).upper())


def _check_columns(transactions: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED if c not in transactions.columns]
    if missing:
        raise ValueError(
            f"insider universe mapping: transactions frame missing column(s) {missing} "
            "(re-ingest with the current data.insider_ingestion schema)"
        )


def _norm_universe(universe: Iterable[str]) -> dict[str, str]:
    """normalized symbol → canonical universe ticker (as given). Collisions are refused
    loudly — two universe tickers must never normalize to the same key."""
    out: dict[str, str] = {}
    for t in universe:
        key = normalize_symbol(t)
        if not key:
            raise ValueError(f"insider universe mapping: unusable universe ticker {t!r}")
        if key in out and out[key] != t:
            raise ValueError(
                f"insider universe mapping: universe tickers {out[key]!r} and {t!r} "
                f"both normalize to {key!r}"
            )
        out[key] = t
    return out


def _cik_assignments(
    transactions: pd.DataFrame,
    norm_map: Mapping[str, str],
    extra_ciks: Mapping[str, Iterable[str]] | None,
    exclude_ciks: Iterable[str] | None,
) -> dict[str, str]:
    """issuer_cik → canonical universe ticker. A CIK is a candidate for every universe
    ticker it ever filed under (normalized); conflicts resolve to the ticker of the
    CIK's most recent such filing. ``extra_ciks`` entries win over discovery; excluded
    CIKs are never assigned."""
    excluded = {str(c).lstrip("0") for c in (exclude_ciks or ())}

    with_cik = transactions[transactions["issuer_cik"] != ""]
    norm_symbols = with_cik["ticker"].map(normalize_symbol)
    in_universe = norm_symbols.map(norm_map.get)
    hits = with_cik[in_universe.notna()].assign(_uni=in_universe.dropna())

    assignment: dict[str, str] = {}
    if not hits.empty:
        latest = (
            hits.groupby(["issuer_cik", "_uni"])["filing_date"].max()
            .reset_index()
            .sort_values(["issuer_cik", "filing_date", "_uni"])
        )
        n_multi = int(latest["issuer_cik"].duplicated().sum())
        if n_multi:
            logger.warning(
                "insider universe mapping: %d CIK(s) filed under multiple universe "
                "tickers; assigning each to its most recent one", n_multi,
            )
        # sorted ascending by filing_date → the LAST row per CIK is the most recent.
        assignment = dict(latest.groupby("issuer_cik")["_uni"].last())

    for uni_ticker, ciks in (extra_ciks or {}).items():
        for cik in ciks:
            assignment[str(cik).lstrip("0")] = uni_ticker

    return {cik: t for cik, t in assignment.items() if cik not in excluded}


def map_transactions_to_universe(
    transactions: pd.DataFrame,
    universe: Iterable[str],
    *,
    extra_ciks: Mapping[str, Iterable[str]] | None = None,
    exclude_ciks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Transactions restricted to the universe with ``ticker`` REWRITTEN to the canonical
    universe symbol, joined rename-safely by issuer CIK (see module docstring).

    Rows whose issuer CIK is assigned to a universe ticker map with their FULL history
    (whatever symbol was on the filing); CIK-less rows map only on the normalized-symbol
    match. The result is safe to hand to ``research.insider_features`` — one canonical
    ticker per issuer across its whole filing history (which also makes the trailing
    windows and the routine flag continuous across renames)."""
    _check_columns(transactions)
    norm_map = _norm_universe(universe)
    assignment = _cik_assignments(transactions, norm_map, extra_ciks, exclude_ciks)

    by_cik = transactions["issuer_cik"].map(assignment)
    by_symbol = transactions["ticker"].map(
        lambda s: norm_map.get(normalize_symbol(s))
    ).where(transactions["issuer_cik"] == "")
    canonical = by_cik.fillna(by_symbol)

    mapped = transactions[canonical.notna()].copy()
    mapped["ticker"] = canonical.dropna()

    n_naive = int(transactions["ticker"].isin(set(norm_map.values())).sum())
    logger.info(
        "insider universe mapping: %d rows mapped (naive exact-symbol join: %d; "
        "recovered %+d) across %d universe names",
        len(mapped), n_naive, len(mapped) - n_naive, mapped["ticker"].nunique(),
    )
    return mapped


def build_universe_cik_map(
    transactions: pd.DataFrame,
    universe: Iterable[str],
    *,
    extra_ciks: Mapping[str, Iterable[str]] | None = None,
    exclude_ciks: Iterable[str] | None = None,
) -> pd.DataFrame:
    """The AUDIT TABLE behind :func:`map_transactions_to_universe`: one row per
    (universe ticker, issuer CIK) with row counts, filing-date range, the as-filed
    symbols observed and the issuer names — everything a reviewer needs to spot
    ticker-recycling pollution or a missing predecessor CIK before a study is banked."""
    _check_columns(transactions)
    norm_map = _norm_universe(universe)
    assignment = _cik_assignments(transactions, norm_map, extra_ciks, exclude_ciks)
    if not assignment:
        return pd.DataFrame(columns=["universe_ticker", "issuer_cik", "n_rows",
                                     "first_filing", "last_filing", "symbols_filed",
                                     "issuer_names"])

    rows = transactions[transactions["issuer_cik"].map(assignment).notna()].copy()
    rows["_uni"] = rows["issuer_cik"].map(assignment)
    name_col = "issuer_name" if "issuer_name" in rows.columns else "ticker"
    audit = (
        rows.groupby(["_uni", "issuer_cik"])
        .agg(
            n_rows=("ticker", "size"),
            first_filing=("filing_date", "min"),
            last_filing=("filing_date", "max"),
            symbols_filed=("ticker", lambda s: sorted(set(s))),
            issuer_names=(name_col, lambda s: sorted({x for x in s if x})[:5]),
        )
        .reset_index()
        .rename(columns={"_uni": "universe_ticker"})
        .sort_values(["universe_ticker", "first_filing"], kind="mergesort")
        .reset_index(drop=True)
    )
    return audit
