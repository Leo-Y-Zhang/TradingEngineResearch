"""
Stage B / #1 — SEC EDGAR *companyfacts* → 14-factor-library bridge (offline, NO network).
=========================================================================================
These tests exercise the additive companyfacts path of ``data/edgar_ingestion.py``
(``extract_company_facts`` / ``build_edgar_panel``) entirely against SYNTHETIC, in-memory
companyfacts JSON — the live ``fetch_company_facts`` round-trip is never touched.

They prove the three properties the bridge exists to guarantee:

  (1) FILING-DATE POINT-IN-TIME safety — a fact whose ``filed`` date is AFTER a price date
      D is NOT used at D (visibility is keyed on ``filed``, never on the accounting
      ``period_end``); and dropping future filings is a no-op.
  (2) TAG FALLBACK — when a company reports an alternate concept variant (e.g. only
      ``RevenueFromContractWithCustomerExcludingAssessedTax`` rather than ``Revenues``, or
      the ``dei`` shares concept, or basic rather than diluted EPS) the canonical panel
      column is still populated; and when BOTH the primary and a fallback concept are
      present, the PRIMARY wins.
  (3) ``build_edgar_panel`` emits exactly the columns/shape the 14-factor library consumes
      (and the panel flows cleanly through ``research.fundamental_features.compute_features``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.edgar_ingestion import (
    COMPANY_FACTS_TAGS,
    EDGAR_PANEL_COLUMNS,
    build_edgar_panel,
    extract_company_facts,
)
from research.fundamental_features import FEATURE_NAMES, compute_features

# ── Synthetic companyfacts builders ──────────────────────────────────────────────────
#
# Two quarterly filings per company: Q1 (period end 2020-03-31) becomes PUBLIC on its
# 2020-05-01 filing date; Q2 (period end 2020-06-30) becomes public on 2020-08-01. The gap
# between Q2's period end (06-30) and its filing date (08-01) is exactly what the PIT tests
# exploit — on 2020-07-15 the period has ended but the filing has NOT been published.
_FILINGS: list[tuple[str, str]] = [("2020-05-01", "2020-03-31"), ("2020-08-01", "2020-06-30")]


def _concept(values: list[float]) -> dict:
    """An XBRL concept node: one fact per filing in ``_FILINGS`` carrying ``values``."""
    return {
        "units": {
            "USD": [
                {"filed": filed, "end": end, "val": val}
                for (filed, end), val in zip(_FILINGS, values, strict=True)
            ]
        }
    }


def _aaa_facts() -> dict:
    """AAA exercises every FALLBACK concept: revenue via the ASC-606 contract concept (no
    ``Revenues``), shares via the ``dei`` concept, EPS via Basic (no Diluted)."""
    return {
        "cik": 1,
        "entityName": "AAA Co",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": _concept([10.0, 12.0]),
                "StockholdersEquity": _concept([200.0, 210.0]),
                "Assets": _concept([500.0, 520.0]),
                "RevenueFromContractWithCustomerExcludingAssessedTax": _concept([100.0, 120.0]),
                "GrossProfit": _concept([40.0, 48.0]),
                "OperatingIncomeLoss": _concept([13.0, 15.0]),
                "NetCashProvidedByUsedInOperatingActivities": _concept([11.0, 13.0]),
                "Liabilities": _concept([300.0, 310.0]),
                "EarningsPerShareBasic": _concept([0.10, 0.12]),
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": _concept([1000.0, 1000.0]),
            },
        },
    }


def _bbb_facts() -> dict:
    """BBB reports the PRIMARY concepts AND, for revenue, ALSO a fallback variant — so the
    'primary wins' tie-break is exercised (``Revenues`` 300/330, never the 999 fallback)."""
    return {
        "cik": 2,
        "entityName": "BBB Co",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": _concept([30.0, 33.0]),
                "StockholdersEquity": _concept([400.0, 410.0]),
                "Assets": _concept([900.0, 920.0]),
                "Revenues": _concept([300.0, 330.0]),
                "RevenueFromContractWithCustomerExcludingAssessedTax": _concept([999.0, 999.0]),
                "GrossProfit": _concept([120.0, 132.0]),
                "OperatingIncomeLoss": _concept([39.0, 43.0]),
                "NetCashProvidedByUsedInOperatingActivities": _concept([33.0, 36.0]),
                "Liabilities": _concept([500.0, 510.0]),
                "CommonStockSharesOutstanding": _concept([2000.0, 2000.0]),
                "EarningsPerShareDiluted": _concept([0.30, 0.33]),
            },
        },
    }


_PRICE_DATES = [
    "2020-04-15", "2020-05-15", "2020-06-15", "2020-07-15", "2020-08-15", "2020-09-15",
]


def _tidy_prices() -> pd.DataFrame:
    rows = []
    for tic, base in (("AAA", 10.0), ("BBB", 20.0)):
        for i, d in enumerate(_PRICE_DATES):
            rows.append({"ticker": tic, "date": d, "price": base + i})
    return pd.DataFrame(rows)


def _funds() -> pd.DataFrame:
    return pd.concat(
        [extract_company_facts(_aaa_facts(), "AAA"), extract_company_facts(_bbb_facts(), "BBB")],
        ignore_index=True,
    )


# ── (1) extraction: canonical tags + fallback ─────────────────────────────────────────

class TestExtract:
    def test_columns_and_all_canonical_tags(self):
        df = extract_company_facts(_aaa_facts(), "AAA")
        assert list(df.columns) == ["ticker", "tag", "filed", "period_end", "value"]
        assert df["filed"].dtype.kind == "M" and df["period_end"].dtype.kind == "M"
        # AAA reports (some variant of) every wanted column.
        assert set(df["tag"]) == set(COMPANY_FACTS_TAGS)
        assert (df["ticker"] == "AAA").all()

    def test_tag_fallback_populates_canonical_columns(self):
        df = extract_company_facts(_aaa_facts(), "AAA")

        def vals(tag: str) -> set[float]:
            return set(df.loc[df["tag"] == tag, "value"])

        # revenue came from RevenueFromContractWithCustomerExcludingAssessedTax (no "Revenues")
        assert vals("revenue") == {100.0, 120.0}
        # sharesbas came from dei:EntityCommonStockSharesOutstanding
        assert vals("sharesbas") == {1000.0}
        # eps came from EarningsPerShareBasic (no Diluted)
        assert vals("eps") == {0.10, 0.12}

    def test_primary_concept_wins_over_fallback(self):
        # BBB reports BOTH Revenues (300/330) and the contract concept (999). Primary wins.
        df = extract_company_facts(_bbb_facts(), "BBB")
        assert set(df.loc[df["tag"] == "revenue", "value"]) == {300.0, 330.0}
        assert 999.0 not in set(df["value"])

    def test_missing_company_yields_empty_typed_frame(self):
        empty = extract_company_facts({"facts": {"us-gaap": {}}}, "AAA")
        assert empty.empty
        assert list(empty.columns) == ["ticker", "tag", "filed", "period_end", "value"]


# ── (2) point-in-time forward fill (filing date, never period end) ─────────────────────

class TestPointInTime:
    def test_fact_filed_after_date_is_not_used(self):
        panel = build_edgar_panel(_funds(), _tidy_prices())
        aaa = panel[panel["ticker"] == "AAA"].set_index("date")
        # Before any filing was public → NaN (nothing fabricated).
        assert pd.isna(aaa.loc[pd.Timestamp("2020-04-15"), "assets"])
        # Q1 known from its 2020-05-01 filing; Q2 (period end 06-30) is NOT visible on
        # 07-15 because it was not FILED until 08-01 — a period-end-keyed loader would leak.
        assert aaa.loc[pd.Timestamp("2020-05-15"), "assets"] == 500.0
        assert aaa.loc[pd.Timestamp("2020-07-15"), "assets"] == 500.0   # still Q1
        assert aaa.loc[pd.Timestamp("2020-08-15"), "assets"] == 520.0   # Q2 now filed
        assert aaa.loc[pd.Timestamp("2020-07-15"), "revenue"] == 100.0
        assert aaa.loc[pd.Timestamp("2020-08-15"), "revenue"] == 120.0

    def test_dropping_future_filings_is_a_noop(self):
        funds, prices = _funds(), _tidy_prices()
        asof = pd.Timestamp("2020-07-15")
        full = build_edgar_panel(funds, prices)
        masked = build_edgar_panel(funds[funds["filed"] <= asof], prices)
        keys = ["ticker", "date", *COMPANY_FACTS_TAGS]
        row_full = full[full["date"] == asof].reset_index(drop=True)[keys]
        row_masked = masked[masked["date"] == asof].reset_index(drop=True)[keys]
        pd.testing.assert_frame_equal(row_full, row_masked)


# ── (3) panel shape / schema / integration with the 14-factor library ─────────────────

class TestPanel:
    def test_columns_shape_and_keys(self):
        panel = build_edgar_panel(_funds(), _tidy_prices())
        assert list(panel.columns) == list(EDGAR_PANEL_COLUMNS)
        assert len(panel) == len(_PRICE_DATES) * 2                 # one row per (ticker, date)
        assert not panel.duplicated(subset=["ticker", "date"]).any()
        assert set(panel["ticker"]) == {"AAA", "BBB"}
        # sorted by (ticker, date)
        assert panel.equals(panel.sort_values(["ticker", "date"]).reset_index(drop=True))

    def test_accepts_wide_close_matrix(self):
        wide = _tidy_prices().pivot(index="date", columns="ticker", values="price")
        panel = build_edgar_panel(_funds(), wide)
        assert list(panel.columns) == list(EDGAR_PANEL_COLUMNS)
        assert len(panel) == len(_PRICE_DATES) * 2
        aaa = panel[panel["ticker"] == "AAA"].set_index("date")
        assert aaa.loc[pd.Timestamp("2020-07-15"), "assets"] == 500.0   # PIT preserved

    def test_missing_ticker_column_is_all_nan(self):
        # funds with no 'gp' rows at all → the gp column exists but is entirely NaN.
        funds = _funds()
        funds = funds[funds["tag"] != "gp"]
        panel = build_edgar_panel(funds, _tidy_prices())
        assert "gp" in panel.columns
        assert panel["gp"].isna().all()

    def test_flows_through_compute_features(self):
        panel = build_edgar_panel(_funds(), _tidy_prices())
        feats = compute_features(panel)
        assert set(FEATURE_NAMES).issubset(feats.columns)
        # A 2-name cross-section with distinct ROE (AAA 12/210, BBB 33/410) z-scores finite.
        assert np.isfinite(feats["roe"]).any()
        assert np.isfinite(feats["earnings_yield"]).any()   # eps/price fallback (no marketcap)


# ── (4) data-correctness defences against the REAL companyfacts API shapes ────────────
#
# These exercise the messy shapes the live companyfacts API actually emits — co-filed
# 3M/YTD/annual durations, a mid-history concept switch, multi-class share rows, and the
# debt-line hierarchy — and lock the deterministic, period-consistent resolution.

def _company(concepts: dict, *, dei: dict | None = None, cik: int = 99) -> dict:
    """A minimal companyfacts payload from ``{concept: units-node}`` maps."""
    facts: dict = {"us-gaap": concepts}
    if dei is not None:
        facts["dei"] = dei
    return {"cik": cik, "entityName": "Test Co", "facts": facts}


def _dur(rows: list[tuple[str, str, str, float]]) -> dict:
    """A DURATION concept node: rows of (filed, start, end, val)."""
    return {"units": {"USD": [
        {"filed": f, "start": s, "end": e, "val": v} for (f, s, e, v) in rows
    ]}}


def _inst(rows: list[tuple[str, str, float]], unit: str = "USD") -> dict:
    """An INSTANT concept node: rows of (filed, end, val) — no ``start``."""
    return {"units": {unit: [{"filed": f, "end": e, "val": v} for (f, e, v) in rows]}}


def _tags(df, tag: str) -> list[float]:
    return sorted(df.loc[df["tag"] == tag, "value"].tolist())


class TestDurationCollision:
    """Fix 1 — co-filed 3M / YTD / 12M durations at the SAME (filed, end)."""

    @staticmethod
    def _facts(order: str) -> dict:
        # Q3-2020 filing: a 3-month (~92d), a 9-month YTD (~273d) and a 12-month (~365d) flow,
        # ALL with the same filed (2020-11-01) and the same end (2020-09-30).
        q3 = ("2020-11-01", "2020-07-01", "2020-09-30", 25.0)    # ~91 days  → the quarter
        ytd = ("2020-11-01", "2020-01-01", "2020-09-30", 70.0)   # ~273 days → year-to-date
        ttm = ("2020-11-01", "2019-10-01", "2020-09-30", 100.0)  # ~365 days → trailing 12-month
        rows = {"natural": [q3, ytd, ttm],
                "reversed": [ttm, ytd, q3],
                "shuffled": [ytd, ttm, q3]}[order]
        return _company({"NetIncomeLoss": _dur(rows), "StockholdersEquity": _inst(
            [("2020-11-01", "2020-09-30", 500.0)])})

    def test_picks_the_quarterly_fact_not_ytd_or_annual(self):
        df = extract_company_facts(self._facts("natural"), "FLW")
        # The ~quarterly value (25) is chosen — NEVER the YTD (70) or the 12-month (100).
        assert _tags(df, "netinc") == [25.0]

    def test_selection_is_independent_of_json_fact_order(self):
        a = extract_company_facts(self._facts("natural"), "FLW")
        b = extract_company_facts(self._facts("reversed"), "FLW")
        c = extract_company_facts(self._facts("shuffled"), "FLW")
        assert _tags(a, "netinc") == _tags(b, "netinc") == _tags(c, "netinc") == [25.0]

    def test_panel_carries_the_quarterly_value(self):
        df = extract_company_facts(self._facts("natural"), "FLW")
        prices = pd.DataFrame([{"ticker": "FLW", "date": "2020-12-15", "price": 10.0}])
        panel = build_edgar_panel(df, prices)
        assert panel.loc[panel["ticker"] == "FLW", "netinc"].iloc[0] == 25.0

    def test_annual_only_filer_is_scaled_to_a_quarter_equivalent(self):
        # An annual-only filer reports ONLY the 12-month figure for the period → no quarterly
        # fact exists, so it is scaled to a quarter-equivalent (~/4) for a consistent basis.
        facts = _company({"NetIncomeLoss": _dur(
            [("2021-02-15", "2020-01-01", "2020-12-31", 400.0)])})       # 365d only
        df = extract_company_facts(facts, "ANN")
        (val,) = df.loc[df["tag"] == "netinc", "value"].tolist()
        assert 90.0 < val < 110.0                                       # ~100, not the raw 400


class TestConceptSwitch:
    """Fix 2 — a mid-history concept switch must be stitched, not truncated."""

    def test_revenue_concept_switch_is_stitched_across_eras(self):
        # Revenues for 2016-2017, then the ASC-606 contract concept for 2018-2019. The old
        # `break`-on-first-non-empty logic truncated the second era to NaN; coalescing per
        # (filed, end) keeps BOTH eras continuous.
        facts = _company({
            "Revenues": _dur([
                ("2016-05-01", "2016-01-01", "2016-03-31", 100.0),
                ("2017-05-01", "2017-01-01", "2017-03-31", 110.0),
            ]),
            "RevenueFromContractWithCustomerExcludingAssessedTax": _dur([
                ("2018-05-01", "2018-01-01", "2018-03-31", 120.0),
                ("2019-05-01", "2019-01-01", "2019-03-31", 130.0),
            ]),
        })
        df = extract_company_facts(facts, "SWX")
        assert _tags(df, "revenue") == [100.0, 110.0, 120.0, 130.0]     # all four, not truncated

    def test_primary_still_wins_where_both_concepts_overlap(self):
        # Where a period has BOTH concepts, the primary (Revenues) wins; the fallback is used
        # only to fill the era the primary does not cover.
        facts = _company({
            "Revenues": _dur([("2018-05-01", "2018-01-01", "2018-03-31", 100.0)]),
            "RevenueFromContractWithCustomerExcludingAssessedTax": _dur([
                ("2018-05-01", "2018-01-01", "2018-03-31", 999.0),       # same period → ignored
                ("2019-05-01", "2019-01-01", "2019-03-31", 130.0),       # new period → kept
            ]),
        })
        df = extract_company_facts(facts, "SWX")
        assert _tags(df, "revenue") == [100.0, 130.0]
        assert 999.0 not in df["value"].tolist()


class TestMultiClassShares:
    """Fix 3 — multi-class share rows at one (filed, end) must be SUMMED."""

    def test_shares_are_summed_across_classes(self):
        # Class A 600 + Class C 400 = 1000 (same filed + end). Keeping one row would report 600
        # or 400 and ~double the reconstructed market cap.
        facts = _company({"CommonStockSharesOutstanding": _inst([
            ("2020-02-01", "2019-12-31", 600.0),
            ("2020-02-01", "2019-12-31", 400.0),
        ], unit="shares")})
        df = extract_company_facts(facts, "DUL")
        assert _tags(df, "sharesbas") == [1000.0]

    def test_sum_is_independent_of_row_order(self):
        rows = [("2020-02-01", "2019-12-31", 400.0), ("2020-02-01", "2019-12-31", 600.0)]
        facts = _company({"CommonStockSharesOutstanding": _inst(rows, unit="shares")})
        df = extract_company_facts(facts, "DUL")
        assert _tags(df, "sharesbas") == [1000.0]


class TestInterestBearingDebt:
    """Fix 4 — debt is interest-bearing borrowings, not total liabilities."""

    def test_debt_sums_current_and_noncurrent_not_total_liabilities(self):
        facts = _company({
            "DebtCurrent": _inst([("2020-02-01", "2019-12-31", 50.0)]),
            "LongTermDebtNoncurrent": _inst([("2020-02-01", "2019-12-31", 150.0)]),
            "Liabilities": _inst([("2020-02-01", "2019-12-31", 900.0)]),     # must NOT be used
        })
        df = extract_company_facts(facts, "LVR")
        assert _tags(df, "debt") == [200.0]                                 # 50 + 150

    def test_debt_uses_per_leg_fallback_concepts(self):
        # No DebtCurrent / LongTermDebtNoncurrent → fall to ShortTermBorrowings + LongTermDebt.
        facts = _company({
            "ShortTermBorrowings": _inst([("2020-02-01", "2019-12-31", 30.0)]),
            "LongTermDebt": _inst([("2020-02-01", "2019-12-31", 170.0)]),
        })
        df = extract_company_facts(facts, "LVR")
        assert _tags(df, "debt") == [200.0]                                 # 30 + 170

    def test_debt_falls_back_to_liabilities_only_as_last_resort(self):
        facts = _company({"Liabilities": _inst([("2020-02-01", "2019-12-31", 900.0)])})
        df = extract_company_facts(facts, "NOD")
        assert _tags(df, "debt") == [900.0]                                 # last-resort proxy

    def test_single_leg_company_reports_that_leg(self):
        # Only noncurrent debt reported (no current portion) → debt is that leg alone.
        facts = _company({"LongTermDebtNoncurrent": _inst([("2020-02-01", "2019-12-31", 150.0)])})
        df = extract_company_facts(facts, "LVR")
        assert _tags(df, "debt") == [150.0]
