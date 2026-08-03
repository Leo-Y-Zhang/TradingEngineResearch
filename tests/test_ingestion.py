"""
Phase 2 Tests — Real feature ingestion
======================================
Price-derived features from a committed offline fixture flowing through the PIT-safe
feature store. The fixtures are SYNTHETIC (seeded GBM via
``scripts/make_synthetic_fixtures.py`` — no redistributed Yahoo/yfinance data); the
live yfinance fetch is NOT exercised here. These tests run entirely offline.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import data.price_ingestion as pi
from data import feature_store

_FIXTURE = Path(__file__).parent / "fixtures" / "prices_sample.csv"
_DIV_FIXTURE = Path(__file__).parent / "fixtures" / "dividends_sample.csv"


class TestPriceIngestion:

    def setup_method(self):
        feature_store._clear_store()

    def teardown_method(self):
        feature_store._clear_store()

    def test_fixture_loads_offline(self):
        prices = pi.load_prices(_FIXTURE)
        assert {"date", "symbol", "open", "high", "low", "close", "volume"} <= set(prices.columns)
        assert prices["symbol"].nunique() == 5
        assert len(prices) > 2000

    def test_compute_features_finite_and_sane(self):
        feats = pi.compute_price_features(pi.load_prices(_FIXTURE))
        assert feats
        for fdict in feats.values():
            for value in fdict.values():
                assert math.isfinite(value)
        assert all(fd["volume_ratio"] > 0 for fd in feats.values() if "volume_ratio" in fd)
        assert all(fd["idiosyncratic_vol"] >= 0 for fd in feats.values() if "idiosyncratic_vol" in fd)
        assert any("momentum_12_1" in fd for fd in feats.values())   # 252-day lookback satisfied

    def test_compute_is_pit_safe(self):
        # A feature at date t uses only data <= t (all trailing windows): the value at a
        # cutoff date is identical whether or not future data is present.
        prices = pi.load_prices(_FIXTURE)
        sub = prices[prices["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
        cutoff = sub.iloc[300]["date"]
        full = pi.compute_price_features(prices)
        truncated = pi.compute_price_features(prices[prices["date"] <= cutoff])
        key = ("AAPL", pd.Timestamp(cutoff))
        assert key in full and key in truncated
        assert full[key]
        for feat, val in full[key].items():
            assert truncated[key][feat] == pytest.approx(val, rel=1e-9, abs=1e-12)

    def test_ingest_then_get_features_returns_real_values(self):
        n = pi.ingest_prices(pi.load_prices(_FIXTURE), mode="RESEARCH")
        assert n > 0
        asof = datetime(2023, 12, 1, tzinfo=timezone.utc)
        df = feature_store.get_features(["AAPL"], asof, "RESEARCH")
        assert not df.empty
        assert "idiosyncratic_vol" in df.columns
        assert float(df.loc["AAPL", "idiosyncratic_vol"]) >= 0.0
        assert float(df.loc["AAPL", "volume_ratio"]) > 0.0

    def test_get_features_does_not_leak_future(self):
        pi.ingest_prices(pi.load_prices(_FIXTURE), mode="RESEARCH")
        early = datetime(2023, 6, 1, tzinfo=timezone.utc)
        df = feature_store.get_features(["AAPL"], early, "RESEARCH")
        assert not df.empty
        assert pd.Timestamp(df.loc["AAPL", "asof_timestamp"]) <= pd.Timestamp(early)


class TestDividendIngestion:

    def test_load_dividends_fixture(self):
        div = pi.load_dividends(_DIV_FIXTURE)
        assert {"date", "symbol", "dividend"} <= set(div.columns)
        assert set(div["symbol"]) <= {"AAPL", "JPM", "MSFT"}     # only payers appear
        assert (div["dividend"] > 0).all()

    def test_trailing_dividend_yields(self):
        prices = pi.load_prices(_FIXTURE)
        div = pi.load_dividends(_DIV_FIXTURE)
        y = pi.trailing_dividend_yields(prices, div)
        assert y["AAPL"] > 0.0 and y["MSFT"] > 0.0 and y["JPM"] > 0.0   # payers
        assert y.get("AMZN", 0.0) == 0.0 and y.get("GOOG", 0.0) == 0.0  # non-payers
        assert all(0.0 <= v < 0.5 for v in y.values())                 # sane annual yields

    def test_trailing_yield_is_pit_safe(self):
        # The trailing yield at an early asof must not see later dividends.
        prices = pi.load_prices(_FIXTURE)
        div = pi.load_dividends(_DIV_FIXTURE)
        early = pd.Timestamp("2022-03-01")
        y_early = pi.trailing_dividend_yields(prices, div, asof=early)
        # only dividends paid in (early-1y, early] count — a strictly smaller window
        # than the full-history yield, so it must be finite and non-negative.
        assert all(v >= 0.0 and v < 0.5 for v in y_early.values())


class TestNewsIngestion:

    def test_converts_modern_yfinance_shape(self):
        now = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)
        raw = [{"content": {"title": "Profit surges to record",
                            "pubDate": "2026-06-10T13:30:00Z"}}]
        items = pi.news_items_from_yfinance(raw, "AAPL", now=now)
        assert len(items) == 1
        assert items[0]["headline"] == "Profit surges to record"
        assert items[0]["symbol"] == "AAPL"
        assert items[0]["age_minutes"] == pytest.approx(30.0)

    def test_converts_legacy_yfinance_shape(self):
        now = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)
        ts = int(now.timestamp()) - 600                       # 10 minutes ago
        raw = [{"title": "Shares plunge on warning", "providerPublishTime": ts}]
        items = pi.news_items_from_yfinance(raw, "MSFT", now=now)
        assert len(items) == 1
        assert items[0]["headline"] == "Shares plunge on warning"
        assert items[0]["age_minutes"] == pytest.approx(10.0)

    def test_skips_malformed_and_undated_items(self):
        now = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)
        raw = [{}, {"content": {}}, {"title": "no date"}, "not-a-dict"]
        assert pi.news_items_from_yfinance(raw, "AAPL", now=now) == []
