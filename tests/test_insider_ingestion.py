"""
TradingEngineResearch — tests for the SEC Form 3/4/5 insider-transactions ingestion
(``data.insider_ingestion``). Offline, deterministic, NO network.

The properties under test (house style of ``tests/test_edgar_ingestion.py``):

  1. A synthetic quarterly ZIP (exact SEC TSV headers) parses into the documented
     tidy schema with correct values, dtypes and DD-MON-YYYY date handling.
  2. AS-FILED point-in-time discipline: only ``DOCUMENT_TYPE == '4'`` survives —
     ``4/A`` amendments and Form 3/5 (+ their /A) are excluded; a plain ``4`` with
     ``DATE_OF_ORIG_SUB`` set is kept but flagged ``is_amendment=True``.
  3. Defensive parsing: bad tickers / dates / shares are DROPPED (never crash),
     malformed lines are skipped, prices may be missing (NaN) without dropping.
  4. Multi-owner filings fan out to one row per reporting owner.
  5. The parquet cache: built once, reused when the quarter count is unchanged,
     rebuilt when a new quarterly ZIP appears (or on ``force=True``).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.insider_ingestion import (
    TIDY_COLUMNS,
    load_insider_transactions,
    parse_quarter_zip,
)

# ── Exact SEC TSV headers (verified against the real 2010q1 archive) ─────────────────
SUB_HEADER = [
    "ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB",
    "NO_SECURITIES_OWNED", "NOT_SUBJECT_SEC16", "FORM3_HOLDINGS_REPORTED",
    "FORM4_TRANS_REPORTED", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
    "ISSUERTRADINGSYMBOL", "REMARKS",
]
OWN_HEADER = [
    "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP",
    "RPTOWNER_TITLE", "RPTOWNER_TXT", "RPTOWNER_STREET1", "RPTOWNER_STREET2",
    "RPTOWNER_CITY", "RPTOWNER_STATE", "RPTOWNER_ZIPCODE", "RPTOWNER_STATE_DESC",
    "FILE_NUMBER",
]
TRANS_HEADER = [
    "ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "SECURITY_TITLE_FN",
    "TRANS_DATE", "TRANS_DATE_FN", "DEEMED_EXECUTION_DATE", "DEEMED_EXECUTION_DATE_FN",
    "TRANS_FORM_TYPE", "TRANS_CODE", "EQUITY_SWAP_INVOLVED", "EQUITY_SWAP_TRANS_CD_FN",
    "TRANS_TIMELINESS", "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN",
    "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN", "TRANS_ACQUIRED_DISP_CD",
    "TRANS_ACQUIRED_DISP_CD_FN", "SHRS_OWND_FOLWNG_TRANS", "SHRS_OWND_FOLWNG_TRANS_FN",
    "VALU_OWND_FOLWNG_TRANS", "VALU_OWND_FOLWNG_TRANS_FN", "DIRECT_INDIRECT_OWNERSHIP",
    "DIRECT_INDIRECT_OWNERSHIP_FN", "NATURE_OF_OWNERSHIP", "NATURE_OF_OWNERSHIP_FN",
]


def _tsv(header: list[str], rows: list[dict[str, str]]) -> str:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(row.get(col, "") for col in header))
    return "\n".join(lines) + "\n"


def write_quarter_zip(
    path: Path,
    submissions: list[dict[str, str]],
    owners: list[dict[str, str]],
    transactions: list[dict[str, str]],
    *,
    raw_tables: dict[str, str] | None = None,
) -> Path:
    """Build a synthetic ``{YYYY}q{Q}_form345.zip`` with the exact SEC TSV layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("SUBMISSION.tsv", raw_tables.get("SUBMISSION.tsv") if raw_tables and "SUBMISSION.tsv" in raw_tables else _tsv(SUB_HEADER, submissions))
        z.writestr("REPORTINGOWNER.tsv", raw_tables.get("REPORTINGOWNER.tsv") if raw_tables and "REPORTINGOWNER.tsv" in raw_tables else _tsv(OWN_HEADER, owners))
        z.writestr("NONDERIV_TRANS.tsv", raw_tables.get("NONDERIV_TRANS.tsv") if raw_tables and "NONDERIV_TRANS.tsv" in raw_tables else _tsv(TRANS_HEADER, transactions))
    return path


def _sub(acc: str, filing: str, doc: str = "4", ticker: str = "AAA",
         orig_sub: str = "") -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": acc, "FILING_DATE": filing, "PERIOD_OF_REPORT": filing,
        "DATE_OF_ORIG_SUB": orig_sub, "DOCUMENT_TYPE": doc,
        "ISSUERCIK": "0000123456", "ISSUERNAME": "Test Issuer",
        "ISSUERTRADINGSYMBOL": ticker,
    }


def _own(acc: str, cik: str = "0001111111", rel: str = "Officer") -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": acc, "RPTOWNERCIK": cik, "RPTOWNERNAME": "Doe Jane",
        "RPTOWNER_RELATIONSHIP": rel, "RPTOWNER_TITLE": "CEO",
    }


def _trn(acc: str, sk: str, date: str, code: str = "P", shares: str = "100.0",
         price: str = "10.0", ad: str = "A", owned: str = "1000.0",
         di: str = "D") -> dict[str, str]:
    return {
        "ACCESSION_NUMBER": acc, "NONDERIV_TRANS_SK": sk,
        "SECURITY_TITLE": "Common Stock", "TRANS_DATE": date, "TRANS_FORM_TYPE": "4",
        "TRANS_CODE": code, "EQUITY_SWAP_INVOLVED": "0", "TRANS_SHARES": shares,
        "TRANS_PRICEPERSHARE": price, "TRANS_ACQUIRED_DISP_CD": ad,
        "SHRS_OWND_FOLWNG_TRANS": owned, "DIRECT_INDIRECT_OWNERSHIP": di,
    }


# --------------------------------------------------------------------------- #
# 1. Happy-path parsing
# --------------------------------------------------------------------------- #
def test_parses_minimal_zip_to_tidy_schema(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2010q1_form345.zip",
        submissions=[_sub("A1", "24-MAR-2010"), _sub("A2", "31-MAR-2010", ticker="bbb ")],
        owners=[_own("A1", "0001111111", "Officer"),
                _own("A2", "0002222222", "Director")],
        transactions=[
            _trn("A1", "1", "22-MAR-2010", code="P", shares="2000.0", price="8.5",
                 ad="A", owned="29000.0", di="D"),
            _trn("A2", "2", "30-MAR-2010", code="S", shares="3891.0", price="13.08",
                 ad="D", owned="4000.0", di="I"),
        ],
    )
    tidy = parse_quarter_zip(zp)

    assert list(tidy.columns) == TIDY_COLUMNS
    assert len(tidy) == 2
    tidy = tidy.sort_values("ticker").reset_index(drop=True)

    row = tidy.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["filing_date"] == pd.Timestamp("2010-03-24")
    assert row["trans_date"] == pd.Timestamp("2010-03-22")
    assert row["trans_code"] == "P"
    assert row["shares"] == pytest.approx(2000.0)
    assert row["price"] == pytest.approx(8.5)
    assert row["owner_cik"] == "1111111"          # leading zeros normalized
    assert row["relationship"] == "OFFICER"
    assert row["direct_indirect"] == "D"
    assert bool(row["is_amendment"]) is False
    assert row["shrs_owned_after"] == pytest.approx(29000.0)

    row2 = tidy.iloc[1]
    assert row2["ticker"] == "BBB"                 # upper + strip
    assert row2["trans_code"] == "S"
    assert row2["relationship"] == "DIRECTOR"
    assert row2["direct_indirect"] == "I"


def test_dd_mon_yyyy_dates_parse_case_insensitively(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2011q2_form345.zip",
        submissions=[_sub("A1", "01-Jun-2011")],
        owners=[_own("A1")],
        transactions=[_trn("A1", "1", "31-may-2011")],
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1
    assert tidy.iloc[0]["filing_date"] == pd.Timestamp("2011-06-01")
    assert tidy.iloc[0]["trans_date"] == pd.Timestamp("2011-05-31")


def test_issuer_cik_name_and_accession_kept_for_rename_safe_joins(tmp_path: Path) -> None:
    """The tidy frame must retain the issuer's CIK (leading zeros stripped), issuer name
    and the accession number — a free-text ticker join silently loses renamed issuers
    (GOOG->GOOGL, FB->META, ...) and the accession is the only key that lets value
    aggregation dedup multi-owner fan-out."""
    zp = write_quarter_zip(
        tmp_path / "2020q1_form345.zip",
        submissions=[_sub("ACC-777", "10-JAN-2020")],
        owners=[_own("ACC-777")],
        transactions=[_trn("ACC-777", "1", "09-JAN-2020")],
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1
    row = tidy.iloc[0]
    assert row["issuer_cik"] == "123456"           # leading zeros normalized
    assert row["issuer_name"] == "Test Issuer"
    assert row["accession"] == "ACC-777"


def test_submission_missing_issuer_cik_column_still_parses(tmp_path: Path) -> None:
    """Vintage drift guard: a SUBMISSION.tsv without the optional ISSUERCIK/ISSUERNAME
    columns must still parse (empty strings), NOT silently drop the whole quarter."""
    header = [c for c in SUB_HEADER if c not in ("ISSUERCIK", "ISSUERNAME")]
    subs = _sub("A1", "05-NOV-2014")
    raw = _tsv(header, [{k: v for k, v in subs.items() if k in header}])
    zp = write_quarter_zip(
        tmp_path / "2014q2_form345.zip",
        submissions=[], owners=[_own("A1")],
        transactions=[_trn("A1", "1", "04-NOV-2014")],
        raw_tables={"SUBMISSION.tsv": raw},
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1
    assert tidy.iloc[0]["issuer_cik"] == ""
    assert tidy.iloc[0]["issuer_name"] == ""
    assert tidy.iloc[0]["ticker"] == "AAA"


# --------------------------------------------------------------------------- #
# 2. AS-FILED PIT: document-type filtering + amendment flag
# --------------------------------------------------------------------------- #
def test_only_original_form4_survives(tmp_path: Path) -> None:
    docs = ["4", "4/A", "3", "5", "3/A", "5/A"]
    zp = write_quarter_zip(
        tmp_path / "2012q3_form345.zip",
        submissions=[_sub(f"A{i}", "15-AUG-2012", doc=d) for i, d in enumerate(docs)],
        owners=[_own(f"A{i}") for i in range(len(docs))],
        transactions=[_trn(f"A{i}", str(i), "14-AUG-2012") for i in range(len(docs))],
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1                          # only the plain Form 4


def test_form4_with_orig_sub_date_is_flagged_amendment(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2013q1_form345.zip",
        submissions=[_sub("A1", "10-JAN-2013"),
                     _sub("A2", "11-JAN-2013", orig_sub="09-JAN-2013")],
        owners=[_own("A1"), _own("A2")],
        transactions=[_trn("A1", "1", "09-JAN-2013"), _trn("A2", "2", "08-JAN-2013")],
    )
    tidy = parse_quarter_zip(zp).sort_values("filing_date").reset_index(drop=True)
    assert len(tidy) == 2
    assert bool(tidy.iloc[0]["is_amendment"]) is False
    assert bool(tidy.iloc[1]["is_amendment"]) is True


# --------------------------------------------------------------------------- #
# 3. Defensive parsing: bad rows dropped, never crash
# --------------------------------------------------------------------------- #
def test_bad_ticker_date_shares_rows_dropped(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2014q4_form345.zip",
        submissions=[
            _sub("GOOD", "05-NOV-2014"),
            _sub("NOTICK", "05-NOV-2014", ticker=""),        # missing ticker
            _sub("NONE1", "05-NOV-2014", ticker="NONE"),     # SEC 'NONE' placeholder
            _sub("BADFD", "not-a-date"),                     # unparseable filing date
            _sub("BADTD", "05-NOV-2014"),
            _sub("BADSH", "05-NOV-2014"),
        ],
        owners=[_own(a) for a in ("GOOD", "NOTICK", "NONE1", "BADFD", "BADTD", "BADSH")],
        transactions=[
            _trn("GOOD", "1", "04-NOV-2014"),
            _trn("NOTICK", "2", "04-NOV-2014"),
            _trn("NONE1", "3", "04-NOV-2014"),
            _trn("BADFD", "4", "04-NOV-2014"),
            _trn("BADTD", "5", "99-XXX-2014"),               # unparseable trans date
            _trn("BADSH", "6", "04-NOV-2014", shares=""),    # missing shares
        ],
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1
    assert tidy.iloc[0]["ticker"] == "AAA"


def test_malformed_lines_skipped_without_crash(tmp_path: Path) -> None:
    good = _tsv(SUB_HEADER, [_sub("A1", "05-NOV-2014")])
    corrupted = good + "this line has\tway too few fields\n"
    zp = write_quarter_zip(
        tmp_path / "2015q1_form345.zip",
        submissions=[], owners=[_own("A1")],
        transactions=[_trn("A1", "1", "04-NOV-2014")],
        raw_tables={"SUBMISSION.tsv": corrupted},
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1                          # good row survives, bad line skipped


def test_missing_price_kept_as_nan(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2016q2_form345.zip",
        submissions=[_sub("A1", "17-MAY-2016")],
        owners=[_own("A1")],
        transactions=[_trn("A1", "1", "16-MAY-2016", price="")],
    )
    tidy = parse_quarter_zip(zp)
    assert len(tidy) == 1
    assert np.isnan(tidy.iloc[0]["price"])
    assert tidy.iloc[0]["shares"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# 4. Multi-owner filings fan out
# --------------------------------------------------------------------------- #
def test_multi_owner_filing_yields_one_row_per_owner(tmp_path: Path) -> None:
    zp = write_quarter_zip(
        tmp_path / "2017q3_form345.zip",
        submissions=[_sub("A1", "20-JUL-2017")],
        owners=[_own("A1", "0001111111", "Officer"),
                _own("A1", "0002222222", "Director,Officer")],
        transactions=[_trn("A1", "1", "19-JUL-2017")],
    )
    tidy = parse_quarter_zip(zp).sort_values("owner_cik").reset_index(drop=True)
    assert len(tidy) == 2
    assert set(tidy["owner_cik"]) == {"1111111", "2222222"}
    assert tidy.iloc[1]["relationship"] == "DIRECTOR,OFFICER"


# --------------------------------------------------------------------------- #
# 5. Parquet cache lifecycle
# --------------------------------------------------------------------------- #
def _one_quarter(tmp_path: Path, name: str, acc: str, filing: str) -> Path:
    return write_quarter_zip(
        tmp_path / "raw" / name,
        submissions=[_sub(acc, filing)],
        owners=[_own(acc)],
        transactions=[_trn(acc, "1", filing)],
    )


def test_cache_built_reused_and_rebuilt_on_new_quarter(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="insider parquet cache requires the optional ingestion extra (pyarrow)")
    raw = tmp_path / "raw"
    _one_quarter(tmp_path, "2018q1_form345.zip", "A1", "05-FEB-2018")

    first = load_insider_transactions(raw)
    cache = tmp_path / "insider_transactions.parquet"
    assert cache.exists()
    assert len(first) == 1

    # Unchanged quarter count → cache is REUSED (parquet not rewritten).
    mtime = cache.stat().st_mtime_ns
    second = load_insider_transactions(raw)
    assert cache.stat().st_mtime_ns == mtime
    pd.testing.assert_frame_equal(first, second)

    # A new quarterly ZIP → quarter count changed → rebuilt with the new rows.
    _one_quarter(tmp_path, "2018q2_form345.zip", "B1", "07-MAY-2018")
    third = load_insider_transactions(raw)
    assert len(third) == 2
    assert cache.stat().st_mtime_ns > mtime


def test_cache_force_rebuild(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="insider parquet cache requires the optional ingestion extra (pyarrow)")
    raw = tmp_path / "raw"
    _one_quarter(tmp_path, "2019q1_form345.zip", "A1", "05-FEB-2019")
    load_insider_transactions(raw)
    cache = tmp_path / "insider_transactions.parquet"
    mtime = cache.stat().st_mtime_ns
    load_insider_transactions(raw, force=True)
    assert cache.stat().st_mtime_ns >= mtime       # rewritten (>= guards coarse clocks)


def test_empty_raw_dir_returns_empty_tidy_frame(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    tidy = load_insider_transactions(raw)
    assert list(tidy.columns) == TIDY_COLUMNS
    assert tidy.empty
