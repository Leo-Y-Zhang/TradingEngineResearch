"""
TradingEngineResearch — tests for the rename-safe insider→universe mapping
(``research.insider_universe``). Offline, deterministic, NO network.

Properties under test (each traces to a confirmed finding of the 2026-07 adversarial
review of the first insider study):

  1. Same-CIK renames are bridged: an issuer's OLD-symbol rows map to the universe's
     CURRENT ticker (the GOOG→GOOGL / FB→META class of silent join loss).
  2. Formatting variants normalize: pre-2009 ``(KO)``-style symbols match ``KO``.
  3. A CIK that filed under several universe tickers is assigned ONLY to its most
     recent one — no row is ever double-counted.
  4. Ticker recycling is auditable and excludable: an unrelated CIK that once used a
     universe symbol appears in the audit table and vanishes with ``exclude_ciks``.
  5. Audited predecessor CIKs (``extra_ciks``) pull in history a symbol bridge cannot
     reach (cross-CIK reorgs like Google Inc → Alphabet Inc).
  6. CIK-less rows (older vintages) still match by exact normalized symbol.
  7. Non-universe rows never leak in; missing schema columns fail LOUDLY.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.insider_universe import (
    build_universe_cik_map,
    map_transactions_to_universe,
    normalize_symbol,
)


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows = (as_filed_ticker, filing_date, issuer_cik) → minimal tidy-like frame."""
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "filing_date": pd.to_datetime([r[1] for r in rows]),
            "issuer_cik": [r[2] for r in rows],
            "issuer_name": [f"Issuer {r[2] or 'unknown'}" for r in rows],
        }
    )


def test_normalize_symbol_strips_formatting() -> None:
    assert normalize_symbol("(ko)") == "KO"
    assert normalize_symbol("GE:") == "GE"
    assert normalize_symbol("CMCSA]") == "CMCSA"
    assert normalize_symbol("brk.b") == "BRKB"


def test_same_cik_rename_bridged_to_current_ticker() -> None:
    txns = _frame([
        ("FB", "2015-05-01", "1326801"),        # old symbol, same issuer
        ("FB", "2020-02-01", "1326801"),
        ("META", "2023-08-01", "1326801"),      # current symbol
        ("ZZZZ", "2019-01-01", "999"),          # unrelated issuer
    ])
    mapped = map_transactions_to_universe(txns, ["META", "AAPL"])
    assert len(mapped) == 3                      # ZZZZ excluded
    assert set(mapped["ticker"]) == {"META"}     # ALL rows carry the canonical ticker
    assert mapped["filing_date"].min() == pd.Timestamp("2015-05-01")


def test_punctuation_variant_matches_via_normalization() -> None:
    txns = _frame([
        ("(KO)", "2007-03-01", "21344"),
        ("KO", "2015-06-01", "21344"),
    ])
    mapped = map_transactions_to_universe(txns, ["KO"])
    assert len(mapped) == 2
    assert set(mapped["ticker"]) == {"KO"}


def test_cik_claiming_two_universe_tickers_goes_to_most_recent_only() -> None:
    # One CIK filed under two symbols that are BOTH in the universe (e.g. a share-class
    # pair): every row must land on exactly one canonical ticker — the most recent.
    txns = _frame([
        ("OLD", "2010-01-01", "777"),
        ("NEW", "2024-01-01", "777"),
    ])
    mapped = map_transactions_to_universe(txns, ["OLD", "NEW"])
    assert len(mapped) == 2                      # no duplication
    assert set(mapped["ticker"]) == {"NEW"}      # most recent filing wins


def test_recycled_ticker_appears_in_audit_and_is_excludable() -> None:
    txns = _frame([
        ("META", "2008-04-01", "555"),           # unrelated company once used META
        ("META", "2023-01-01", "1326801"),       # the real issuer
    ])
    audit = build_universe_cik_map(txns, ["META"])
    assert set(audit["issuer_cik"]) == {"555", "1326801"}   # pollution is VISIBLE

    mapped = map_transactions_to_universe(txns, ["META"], exclude_ciks=["555"])
    assert list(mapped["issuer_cik"]) == ["1326801"]        # and removable


def test_extra_ciks_pull_in_audited_predecessor_history() -> None:
    txns = _frame([
        ("GOOG", "2008-06-01", "1288776"),       # Google Inc era (never filed GOOGL)
        ("GOOGL", "2020-06-01", "1652044"),      # Alphabet Inc
    ])
    plain = map_transactions_to_universe(txns, ["GOOGL"])
    assert len(plain) == 1                       # bridge alone cannot reach 1288776

    mapped = map_transactions_to_universe(
        txns, ["GOOGL"], extra_ciks={"GOOGL": ["1288776"]}
    )
    assert len(mapped) == 2
    assert set(mapped["ticker"]) == {"GOOGL"}


def test_cikless_rows_match_by_exact_normalized_symbol_only() -> None:
    txns = _frame([
        ("AAPL", "2006-02-01", ""),              # old vintage without ISSUERCIK
        ("FB", "2006-03-01", ""),                # cannot be bridged without a CIK
    ])
    mapped = map_transactions_to_universe(txns, ["AAPL", "META"])
    assert list(mapped["ticker"]) == ["AAPL"]


def test_missing_schema_columns_fail_loudly() -> None:
    with pytest.raises(ValueError, match="missing column"):
        map_transactions_to_universe(pd.DataFrame({"ticker": ["A"]}), ["A"])


def test_universe_normalization_collision_is_refused() -> None:
    with pytest.raises(ValueError, match="normalize"):
        map_transactions_to_universe(_frame([("A", "2020-01-01", "1")]), ["BRK.B", "BRKB"])


def test_audit_table_shape_and_evidence() -> None:
    txns = _frame([
        ("FB", "2015-05-01", "1326801"),
        ("META", "2023-08-01", "1326801"),
    ])
    audit = build_universe_cik_map(txns, ["META"])
    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["universe_ticker"] == "META"
    assert row["n_rows"] == 2
    assert row["symbols_filed"] == ["FB", "META"]
    assert row["first_filing"] == pd.Timestamp("2015-05-01")
    assert row["last_filing"] == pd.Timestamp("2023-08-01")
