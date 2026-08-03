"""
Stage B — Fama-French factor ingestion tests (offline; uses the committed fixture,
never the network).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.factor_ingestion import (
    FF_FACTORS,
    factor_loadings,
    factor_momentum_features,
    load_fama_french,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fama_french_daily_sample.csv"


def _ff() -> pd.DataFrame:
    return load_fama_french(FIXTURE)


class TestLoad:
    def test_fixture_loads_clean(self):
        ff = _ff()
        assert list(ff.columns) == ["mkt_rf", "smb", "hml", "rf"]
        assert isinstance(ff.index, pd.DatetimeIndex)
        assert ff.index.is_monotonic_increasing
        assert np.isfinite(ff.to_numpy()).all()
        assert len(ff) > 1000          # ~2848 rows in the 2015+ fixture
        # decimal returns, not percent
        assert ff["mkt_rf"].abs().max() < 0.5


class TestMomentumFeatures:
    def test_features_present_and_finite(self):
        ff = _ff()
        feats = factor_momentum_features(ff, ff.index[-1])
        for col in FF_FACTORS:
            assert f"ff_{col}_cumret_21d" in feats
        assert all(np.isfinite(v) for v in feats.values())

    def test_point_in_time_safe(self):
        ff = _ff()
        asof = ff.index[500]
        feats = factor_momentum_features(ff, asof, windows=(21,))
        # recompute on a frame that has bogus FUTURE rows appended — must be identical
        future = ff.copy()
        future.loc[ff.index[-1] + pd.Timedelta(days=5)] = [9.0, 9.0, 9.0, 9.0]
        feats2 = factor_momentum_features(future.sort_index(), asof, windows=(21,))
        assert feats == feats2


class TestFactorLoadings:
    def test_recovers_known_betas(self):
        ff = _ff()
        rng = np.random.default_rng(0)
        # synthetic stock built as a known linear combination of the FF factors
        syn = (ff["rf"] + 1.2 * ff["mkt_rf"] + 0.5 * ff["smb"] - 0.3 * ff["hml"]
               + rng.normal(0.0, 1e-4, len(ff)))
        stock = pd.DataFrame({"SYN": syn}, index=ff.index)
        loadings = factor_loadings(stock, ff, ff.index[-1], window=126)
        b = loadings["SYN"]
        assert abs(b["beta_mkt"] - 1.2) < 0.05
        assert abs(b["beta_smb"] - 0.5) < 0.10
        assert abs(b["beta_hml"] + 0.3) < 0.10

    def test_point_in_time_safe(self):
        ff = _ff()
        rng = np.random.default_rng(1)
        stock = pd.DataFrame({"A": rng.normal(0, 0.01, len(ff))}, index=ff.index)
        asof = ff.index[800]
        b1 = factor_loadings(stock, ff, asof, window=126)
        # corrupt everything AFTER asof — loadings as-of must not change
        stock2 = stock.copy()
        stock2.loc[stock2.index > asof, "A"] = 99.0
        b2 = factor_loadings(stock2, ff, asof, window=126)
        assert b1["A"] == b2["A"]

    def test_insufficient_history_skips(self):
        ff = _ff()
        stock = pd.DataFrame({"A": np.full(len(ff), np.nan)}, index=ff.index)
        # all-NaN stock -> no usable obs -> skipped (fail-closed, no fabricated beta)
        assert factor_loadings(stock, ff, ff.index[-1], window=126) == {}
