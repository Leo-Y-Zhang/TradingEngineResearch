"""
TradingEngineResearch — SEC Form 3/4/5 insider-transactions ingestion (quarterly TSV ZIPs)
================================================================================
Loads the SEC's quarterly *insider transactions* data sets (``{YYYY}q{Q}_form345.zip``,
each containing tab-delimited UTF-8 TSV tables) into ONE tidy transactions frame:

    ticker, filing_date, trans_date, trans_code, shares, price, owner_cik,
    relationship, direct_indirect, is_amendment, shrs_owned_after,
    issuer_cik, issuer_name, accession

``issuer_cik`` (leading zeros stripped) is the issuer's STABLE identity across ticker
renames (GOOG→GOOGL, FB→META, UTX→RTX, ...) — an adversarial review of the first insider
study found a plain as-filed-ticker join silently dropped ~22% of the universe's insider
rows across such renames. ``issuer_name`` is kept for map auditability and ``accession``
lets consumers dedup the per-owner fan-out (one economic transaction is repeated once per
co-reporting owner). Both ISSUERCIK/ISSUERNAME are treated as OPTIONAL TSV columns
(empty-string when a vintage lacks them) so a schema drift can never silently drop a
quarter.

POINT-IN-TIME DISCIPLINE (the whole game; golden rule 3):
  * The ONLY availability timestamp is ``FILING_DATE`` — the date the form hit EDGAR.
    A transaction is knowable strictly AFTER its ``filing_date``; consumers must apply
    a ``filing_date + 1 business day`` conservatism (``research.insider_features`` does).
    ``TRANS_DATE`` is the economic event date and must NEVER be used for availability.
  * AS-FILED ONLY: ``DOCUMENT_TYPE == '4'`` exactly. ``4/A`` amendments are EXCLUDED —
    an amendment's content was not knowable at the original filing date, and stamping it
    with its own (later) filing date would let corrected/restated numbers silently
    overwrite the as-filed record the market actually saw. Forms 3/5 (+ /A) are excluded
    too: Form 3 is an initial-holdings statement (no open-market signal) and Form 5 is an
    annual catch-up filed up to 45 days after fiscal year end (stale by construction).
    A plain ``4`` whose ``DATE_OF_ORIG_SUB`` is populated is a resubmission oddity — it
    is KEPT but flagged ``is_amendment=True`` so consumers can (and do) drop it.
  * Officers/directors/10%-owners are ALL kept in the tidy frame; the officer+director
    restriction is a FILTER FLAG applied at the feature layer, not here.

DEFENSIVE PARSING: quoting is disabled (SEC TSVs are unquoted; remarks contain quotes),
malformed lines are skipped and counted, rows with missing/unparseable ticker, dates or
shares are dropped and counted, encodings are decoded with replacement — never crash.
Missing PRICES are kept as NaN (legitimate for some transaction codes).

CACHE: the parsed result is stored as one parquet (default
``<raw_dir>/../insider_transactions.parquet`` → ``_data/insider/insider_transactions.parquet``)
plus a tiny sidecar meta JSON holding the quarter count. The cache is rebuilt when it is
missing or when the number of quarterly ZIPs in ``raw_dir`` has changed (or ``force=True``).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "TIDY_COLUMNS",
    "QUARTER_ZIP_GLOB",
    "parse_quarter_zip",
    "load_insider_transactions",
]

logger = logging.getLogger(__name__)

# ── Tidy output schema (documented contract; consumers rely on these names) ──────────
TIDY_COLUMNS: list[str] = [
    "ticker",
    "filing_date",
    "trans_date",
    "trans_code",
    "shares",
    "price",
    "owner_cik",
    "relationship",
    "direct_indirect",
    "is_amendment",
    "shrs_owned_after",
    "issuer_cik",
    "issuer_name",
    "accession",
]

QUARTER_ZIP_GLOB = "*_form345.zip"

# SEC uses these placeholders where no trading symbol exists.
_TICKER_PLACEHOLDERS = {"", "NONE", "N/A", "NA"}

# Exact (uppercase) SEC column names — see the in-archive readme.
_SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DATE_OF_ORIG_SUB", "DOCUMENT_TYPE",
             "ISSUERTRADINGSYMBOL"]
# Optional SUBMISSION columns: filled with "" (and warned) when a vintage lacks them —
# their absence must degrade the join, never silently drop the quarter.
_SUB_OPTIONAL_COLS = ["ISSUERCIK", "ISSUERNAME"]
_OWN_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]
_TRN_COLS = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
             "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS", "DIRECT_INDIRECT_OWNERSHIP"]


def _empty_tidy() -> pd.DataFrame:
    frame = pd.DataFrame(columns=TIDY_COLUMNS)
    frame["filing_date"] = pd.to_datetime(frame["filing_date"])
    frame["trans_date"] = pd.to_datetime(frame["trans_date"])
    for col in ("shares", "price", "shrs_owned_after"):
        frame[col] = frame[col].astype(float)
    frame["is_amendment"] = frame["is_amendment"].astype(bool)
    return frame


def _read_tsv(
    archive: zipfile.ZipFile,
    member: str,
    usecols: list[str],
    optional: list[str] | None = None,
) -> pd.DataFrame:
    """One TSV table from the ZIP → string DataFrame of ``usecols`` (+ ``optional``).
    Malformed lines (wrong field count) are SKIPPED and counted; quoting is disabled (SEC
    TSVs are unquoted — remarks legitimately contain quote characters); undecodable bytes
    are replaced. Missing table or missing REQUIRED columns → empty frame (logged), never
    a crash; missing OPTIONAL columns are filled with "" (logged) so a vintage's schema
    drift degrades the affected fields instead of silently dropping the quarter."""
    optional = optional or []
    out_cols = usecols + optional
    if member not in archive.namelist():
        logger.warning("insider ingestion: %s missing table %s", archive.filename, member)
        return pd.DataFrame(columns=out_cols)
    bad_lines: list[int] = []

    def _on_bad_line(fields: list[str]) -> None:
        bad_lines.append(1)
        return None

    with archive.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        try:
            table = pd.read_csv(
                text,
                sep="\t",
                dtype=str,
                engine="python",
                quoting=csv.QUOTE_NONE,
                on_bad_lines=_on_bad_line,
            )
        except Exception:                                     # pragma: no cover - defensive
            logger.exception("insider ingestion: unreadable table %s in %s",
                             member, archive.filename)
            return pd.DataFrame(columns=out_cols)
    if bad_lines:
        logger.warning("insider ingestion: %s/%s skipped %d malformed line(s)",
                       archive.filename, member, len(bad_lines))
    missing = [c for c in usecols if c not in table.columns]
    if missing:
        logger.warning("insider ingestion: %s/%s missing column(s) %s",
                       archive.filename, member, missing)
        return pd.DataFrame(columns=out_cols)
    missing_opt = [c for c in optional if c not in table.columns]
    if missing_opt:
        logger.warning("insider ingestion: %s/%s missing OPTIONAL column(s) %s "
                       "(filled with empty strings)", archive.filename, member, missing_opt)
        table = table.copy()
        for col in missing_opt:
            table[col] = ""
    return table[out_cols]


def _parse_sec_date(values: pd.Series) -> pd.Series:
    """SEC ``DD-MON-YYYY`` (e.g. ``24-MAR-2010``) → datetime64, unparseable → NaT.
    ``%b`` matches month abbreviations case-insensitively."""
    return pd.to_datetime(values.astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")


def _clean_str(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def parse_quarter_zip(path: Path | str) -> pd.DataFrame:
    """Parse ONE quarterly ``form345`` ZIP into the tidy transactions frame.

    Keeps only original Form 4 submissions (see module docstring for the as-filed PIT
    rationale), joins ``NONDERIV_TRANS`` to ``SUBMISSION`` (inner, on accession) and fans
    out per ``REPORTINGOWNER`` (left join — a multi-owner filing yields one row per
    owner). Rows with a missing/placeholder ticker, an unparseable filing/transaction
    date, or missing/non-positive shares are DROPPED (counts logged). Prices may be
    missing (NaN). Never raises on bad data; a structurally unusable archive yields an
    empty frame."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            subs = _read_tsv(archive, "SUBMISSION.tsv", _SUB_COLS,
                             optional=_SUB_OPTIONAL_COLS)
            owners = _read_tsv(archive, "REPORTINGOWNER.tsv", _OWN_COLS)
            trans = _read_tsv(archive, "NONDERIV_TRANS.tsv", _TRN_COLS)
    except (zipfile.BadZipFile, OSError):
        logger.exception("insider ingestion: unreadable archive %s", path)
        return _empty_tidy()
    if subs.empty or trans.empty:
        logger.warning("insider ingestion: %s has no usable SUBMISSION/NONDERIV_TRANS rows",
                       path.name)
        return _empty_tidy()

    n_subs_total = len(subs)
    doc_type = _clean_str(subs["DOCUMENT_TYPE"])
    subs = subs[doc_type == "4"].copy()            # as-filed PIT: no 4/A, no Form 3/5
    logger.info("insider ingestion: %s submissions total=%d form4=%d (excluded=%d "
                "amendments/form3/form5)", path.name, n_subs_total, len(subs),
                n_subs_total - len(subs))

    subs["ticker"] = _clean_str(subs["ISSUERTRADINGSYMBOL"]).str.upper()
    subs["filing_date"] = _parse_sec_date(subs["FILING_DATE"])
    subs["is_amendment"] = _clean_str(subs["DATE_OF_ORIG_SUB"]) != ""
    # Issuer identity: CIK normalized like owner_cik (leading zeros stripped) so the
    # SAME issuer matches across quarters; name kept as filed for map auditability.
    subs["issuer_cik"] = _clean_str(subs["ISSUERCIK"]).str.lstrip("0")
    subs["issuer_name"] = _clean_str(subs["ISSUERNAME"])
    subs["accession"] = _clean_str(subs["ACCESSION_NUMBER"])

    n_before = len(subs)
    subs = subs[~subs["ticker"].isin(_TICKER_PLACEHOLDERS) & subs["filing_date"].notna()]
    if len(subs) < n_before:
        logger.warning("insider ingestion: %s dropped %d submission(s) with bad "
                       "ticker/filing_date", path.name, n_before - len(subs))

    trans = trans.copy()
    trans["trans_date"] = _parse_sec_date(trans["TRANS_DATE"])
    trans["trans_code"] = _clean_str(trans["TRANS_CODE"]).str.upper()
    trans["shares"] = pd.to_numeric(trans["TRANS_SHARES"], errors="coerce")
    trans["price"] = pd.to_numeric(trans["TRANS_PRICEPERSHARE"], errors="coerce")
    trans["shrs_owned_after"] = pd.to_numeric(trans["SHRS_OWND_FOLWNG_TRANS"],
                                              errors="coerce")
    trans["direct_indirect"] = _clean_str(trans["DIRECT_INDIRECT_OWNERSHIP"]).str.upper()

    n_before = len(trans)
    good_shares = trans["shares"].notna() & np.isfinite(trans["shares"]) & (trans["shares"] > 0)
    trans = trans[trans["trans_date"].notna() & good_shares]
    if len(trans) < n_before:
        logger.warning("insider ingestion: %s dropped %d transaction(s) with bad "
                       "trans_date/shares", path.name, n_before - len(trans))
    trans.loc[trans["price"] < 0, "price"] = np.nan

    if owners.empty:
        owners = pd.DataFrame(columns=_OWN_COLS)
    owners = owners.copy()
    # CIK normalized by stripping leading zeros so the SAME owner matches across files.
    owners["owner_cik"] = _clean_str(owners["RPTOWNERCIK"]).str.lstrip("0")
    owners["relationship"] = _clean_str(owners["RPTOWNER_RELATIONSHIP"]).str.upper()
    owners = owners[["ACCESSION_NUMBER", "owner_cik", "relationship"]]

    merged = trans.merge(
        subs[["ACCESSION_NUMBER", "ticker", "filing_date", "is_amendment",
              "issuer_cik", "issuer_name", "accession"]],
        on="ACCESSION_NUMBER", how="inner",
    ).merge(owners, on="ACCESSION_NUMBER", how="left")
    merged["owner_cik"] = merged["owner_cik"].fillna("")
    merged["relationship"] = merged["relationship"].fillna("")

    tidy = merged[TIDY_COLUMNS].reset_index(drop=True)
    tidy["is_amendment"] = tidy["is_amendment"].astype(bool)
    logger.info("insider ingestion: %s -> %d tidy transaction row(s)", path.name, len(tidy))
    return tidy


# --------------------------------------------------------------------------- #
# Cache lifecycle
# --------------------------------------------------------------------------- #
def _default_cache_path(raw_dir: Path) -> Path:
    return raw_dir.parent / "insider_transactions.parquet"


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _cached_quarter_count(cache_path: Path) -> Optional[int]:
    meta = _meta_path(cache_path)
    if not (cache_path.exists() and meta.exists()):
        return None
    try:
        return int(json.loads(meta.read_text(encoding="utf-8"))["n_quarters"])
    except (ValueError, KeyError, OSError):
        return None


def load_insider_transactions(
    raw_dir: Path | str,
    cache_path: Path | str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """All quarterly ZIPs under ``raw_dir`` → ONE tidy transactions frame, cached.

    The parquet cache (default ``<raw_dir>/../insider_transactions.parquet``) is reused
    when the number of ``*_form345.zip`` archives is unchanged; it is rebuilt when the
    cache is missing/corrupt, the quarter count changed, or ``force=True`` — nothing
    fancier, matching the brief. Works against however many quarters exist (a partial
    download parses fine and the cache refreshes as more quarters land)."""
    raw_dir = Path(raw_dir)
    cache = Path(cache_path) if cache_path is not None else _default_cache_path(raw_dir)
    zips = sorted(raw_dir.glob(QUARTER_ZIP_GLOB))

    if not force and _cached_quarter_count(cache) == len(zips):
        try:
            frame = pd.read_parquet(cache)
            if list(frame.columns) == TIDY_COLUMNS:
                logger.info("insider ingestion: cache hit (%d quarters) %s", len(zips), cache)
                return frame
            logger.warning("insider ingestion: cache schema mismatch; rebuilding %s", cache)
        except Exception:
            logger.exception("insider ingestion: unreadable cache; rebuilding %s", cache)

    frames = [parse_quarter_zip(z) for z in zips]
    frames = [f for f in frames if not f.empty]
    tidy = (
        pd.concat(frames, ignore_index=True).sort_values(
            ["filing_date", "ticker"], kind="mergesort").reset_index(drop=True)
        if frames
        else _empty_tidy()
    )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_parquet(cache, index=False)
        _meta_path(cache).write_text(json.dumps({"n_quarters": len(zips)}),
                                     encoding="utf-8")
    except Exception:                                          # pragma: no cover - defensive
        logger.exception("insider ingestion: could not write cache %s "
                         "(continuing uncached)", cache)
    logger.info("insider ingestion: parsed %d quarter(s) -> %d row(s)", len(zips), len(tidy))
    return tidy
