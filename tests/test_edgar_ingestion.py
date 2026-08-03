"""
Stage B / #1 — SEC EDGAR fundamentals tests (offline; uses the committed fixture,
never the network). The critical property under test is FILING-DATE point-in-time
safety: a fundamental must only be visible after it was filed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.edgar_ingestion import (
    fundamental_features,
    load_fundamentals,
    pit_fundamental,
)

FIXTURE = Path(__file__).parent / "fixtures" / "edgar_fundamentals_sample.csv"


def _funds() -> pd.DataFrame:
    return load_fundamentals(FIXTURE)


class TestLoad:
    def test_fixture_loads(self):
        f = _funds()
        assert set(["ticker", "tag", "filed", "period_end", "value"]).issubset(f.columns)
        assert f["filed"].dtype.kind == "M" and f["period_end"].dtype.kind == "M"
        assert {"AAPL", "MSFT", "JPM", "XOM"}.issubset(set(f["ticker"]))
        assert len(f) > 1000


class TestPointInTime:
    def test_latest_value_positive(self):
        f = _funds()
        eq = pit_fundamental(f, "AAPL", "StockholdersEquity", "2025-12-31")
        assert eq is not None and eq > 0

    def test_none_before_any_filing(self):
        f = _funds()
        assert pit_fundamental(f, "AAPL", "StockholdersEquity", "2010-01-01") is None

    def test_filing_date_pit_safe(self):
        f = _funds()
        early = pit_fundamental(f, "AAPL", "StockholdersEquity", "2016-06-01")
        late = pit_fundamental(f, "AAPL", "StockholdersEquity", "2025-12-31")
        assert early is not None and late is not None
        assert early != late                      # the figure is time-aware (changed over years)
        # explicit PIT property: dropping FUTURE filings cannot change the as-of value
        asof = pd.Timestamp("2016-06-01")
        masked = f[f["filed"] <= asof]
        assert pit_fundamental(f, "AAPL", "StockholdersEquity", asof) == \
            pit_fundamental(masked, "AAPL", "StockholdersEquity", asof)


class TestFeatures:
    def test_features_finite(self):
        f = _funds()
        feats = fundamental_features(f, "AAPL", "2025-12-31", market_cap=3.0e12)
        for k in ("roe", "roa", "book_yield", "earnings_yield"):
            assert k in feats and np.isfinite(feats[k])

    def test_features_without_market_cap(self):
        f = _funds()
        feats = fundamental_features(f, "JPM", "2025-12-31")   # no market_cap
        assert "roe" in feats and "roa" in feats
        assert "book_yield" not in feats and "earnings_yield" not in feats
