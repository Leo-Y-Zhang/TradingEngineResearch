"""
TradingEngineResearch — tests for the PIT-safe fundamental factor library
(``research.fundamental_features``). Offline, deterministic, NO network.

The properties under test (mirroring the house style of ``tests/test_edgar_ingestion.py``
and ``tests/test_signal_learner.py``):

  1. Each raw factor computes the textbook value on a tiny hand-checkable example.
  2. Year-on-year growth and price 12-1 momentum use only PAST same-ticker rows
     (no look-ahead): dropping FUTURE rows cannot change a row's value.
  3. Cross-sectional normalization is STRICTLY per-date — a date's features are
     identical whether or not other dates are present (no cross-date leakage), and
     winsorization is applied within each date.
  4. NaN handling: a missing input yields NaN (never imputed); a date with too few
     finite names is all-NaN for that feature; finite peers are unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.fundamental_features import (
    FACTOR_FUNCTIONS,
    FEATURE_NAMES,
    _winsorize,
    accruals,
    asset_growth,
    book_to_price,
    compute_features,
    debt_to_equity,
    earnings_growth,
    earnings_yield,
    gross_profitability,
    momentum_12_1,
    net_share_issuance,
    operating_margin,
    raw_features,
    revenue_growth,
    roa,
    roe,
    sales_to_price,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _single_date_panel() -> pd.DataFrame:
    """One date, three hand-pickable companies with round-number fundamentals."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "date": pd.to_datetime(["2022-12-31"] * 3),
            "price": [10.0, 20.0, 50.0],
            "marketcap": [1000.0, 2000.0, 5000.0],
            "netinc": [100.0, 100.0, -250.0],
            "eps": [1.0, 1.0, -2.5],
            "equity": [500.0, 1000.0, 2500.0],
            "assets": [2000.0, 4000.0, 10000.0],
            "revenue": [800.0, 1000.0, 4000.0],
            "gp": [400.0, 200.0, 1000.0],
            "ebit": [200.0, 150.0, 400.0],
            "ncfo": [120.0, 90.0, 50.0],
            "debt": [250.0, 2000.0, 5000.0],
            "sharesbas": [100.0, 100.0, 100.0],
        }
    )


def _annual_panel() -> pd.DataFrame:
    """Two tickers, three year-ends exactly 365 days apart — clean YoY arithmetic."""
    dates = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"])
    rows = []
    for tkr, rev0, ni0, ast0, sh0 in [("AAA", 100.0, 50.0, 1000.0, 1000.0),
                                      ("BBB", 200.0, -40.0, 4000.0, 500.0)]:
        for k, d in enumerate(dates):
            rows.append(
                {
                    "ticker": tkr,
                    "date": d,
                    "revenue": rev0 * (1.10 ** k),
                    "netinc": ni0 + 10.0 * k,
                    "assets": ast0 * (1.20 ** k),
                    "sharesbas": sh0 * (1.05 ** k),
                }
            )
    return pd.DataFrame(rows)


def _monthly_momentum_panel() -> pd.DataFrame:
    """One ticker, 14 consecutive month-ends (2021-01..2022-02, no leap year), prices
    geometric at 1% / month so 12-1 momentum is exactly computable."""
    dates = pd.date_range("2021-01-31", periods=14, freq="ME")
    prices = 100.0 * (1.01 ** np.arange(14))
    return pd.DataFrame({"ticker": "ZZZ", "date": dates, "price": prices})


# --------------------------------------------------------------------------- #
# 1. Hand-checkable raw factor values (row-wise factors)
# --------------------------------------------------------------------------- #
class TestRawValuesHandChecked:
    def test_value_factors(self):
        p = _single_date_panel()
        # earnings_yield = netinc / marketcap
        assert np.allclose(earnings_yield(p), [100 / 1000, 100 / 2000, -250 / 5000])
        # book_to_price = equity / marketcap
        assert np.allclose(book_to_price(p), [500 / 1000, 1000 / 2000, 2500 / 5000])
        # sales_to_price = revenue / marketcap
        assert np.allclose(sales_to_price(p), [800 / 1000, 1000 / 2000, 4000 / 5000])

    def test_earnings_yield_falls_back_to_eps_over_price(self):
        # No marketcap/netinc → must fall back to eps / price.
        p = _single_date_panel().drop(columns=["marketcap", "netinc"])
        assert np.allclose(earnings_yield(p), [1.0 / 10.0, 1.0 / 20.0, -2.5 / 50.0])

    def test_quality_factors(self):
        p = _single_date_panel()
        assert np.allclose(roe(p), [100 / 500, 100 / 1000, -250 / 2500])
        assert np.allclose(roa(p), [100 / 2000, 100 / 4000, -250 / 10000])
        assert np.allclose(gross_profitability(p), [400 / 2000, 200 / 4000, 1000 / 10000])
        assert np.allclose(operating_margin(p), [200 / 800, 150 / 1000, 400 / 4000])

    def test_earnings_quality_and_leverage(self):
        p = _single_date_panel()
        # accruals = (netinc - ncfo) / assets
        assert np.allclose(accruals(p), [(100 - 120) / 2000, (100 - 90) / 4000, (-250 - 50) / 10000])
        # debt_to_equity = debt / equity
        assert np.allclose(debt_to_equity(p), [250 / 500, 2000 / 1000, 5000 / 2500])

    def test_zero_denominator_is_nan_not_inf(self):
        p = _single_date_panel()
        p.loc[0, "equity"] = 0.0
        out = roe(p)
        assert np.isnan(out.iloc[0])               # 100 / 0 → NaN, never +inf
        assert np.isfinite(out.iloc[1]) and np.isfinite(out.iloc[2])


# --------------------------------------------------------------------------- #
# 2. Growth / momentum: hand-checked AND PIT (no look-ahead)
# --------------------------------------------------------------------------- #
class TestGrowthAndMomentum:
    def test_yoy_growth_hand_checked(self):
        p = _annual_panel()
        rg = revenue_growth(p)
        ag = asset_growth(p)
        ni = earnings_growth(p)
        nsi = net_share_issuance(p)
        # First year of each ticker has no prior observation → NaN.
        first_rows = p.groupby("ticker", sort=False).head(1).index
        assert rg.loc[first_rows].isna().all()
        # AAA 2021 & 2022: revenue +10% YoY, assets +20% YoY, shares +5% YoY.
        aaa = p.index[(p["ticker"] == "AAA") & (p["date"] > "2020-12-31")]
        assert np.allclose(rg.loc[aaa], 0.10)
        assert np.allclose(ag.loc[aaa], 0.20)
        assert np.allclose(nsi.loc[aaa], 0.05)
        # earnings_growth with a NEGATIVE base (BBB ni: -40, -30, -20): (cur-prior)/|prior|.
        bbb = p.index[(p["ticker"] == "BBB") & (p["date"] == "2021-12-31")]
        assert np.allclose(ni.loc[bbb], (-30.0 - -40.0) / 40.0)   # = +0.25

    def test_growth_is_pit_dropping_future_does_not_change_past(self):
        p = _annual_panel()
        full = revenue_growth(p)
        # Truncate to <= 2021 (drop the 2022 rows entirely).
        past = p[p["date"] <= "2021-12-31"].copy()
        truncated = revenue_growth(past)
        for idx in past.index:
            a, b = full.loc[idx], truncated.loc[idx]
            assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b)

    def test_momentum_12_1_hand_checked(self):
        p = _monthly_momentum_panel()
        mom = momentum_12_1(p)
        # Last row (2022-02-28): short leg ≈ 1 month prior (2022-01-31, idx 12),
        # long leg ≈ 12 months prior (2021-02-28, idx 1) → 1.01**12 / 1.01**1 - 1.
        expected = (1.01 ** 12) / (1.01 ** 1) - 1.0
        assert np.isclose(mom.iloc[-1], expected)
        # Earliest rows have no ~12-month-prior observation → NaN (never fabricated).
        assert mom.iloc[0:2].isna().all()

    def test_momentum_never_uses_current_or_future_price(self):
        p = _monthly_momentum_panel().reset_index(drop=True)
        mom_full = momentum_12_1(p)
        # Pick an INTERIOR row (not the last) that has a defined 12-1 momentum, so there are
        # genuinely strictly-future rows to perturb. The momentum at i must be a pure
        # function of rows whose date is < date[i]; nothing dated >= date[i] may touch it.
        i = next(k for k in range(len(p)) if np.isfinite(mom_full.iloc[k]))
        baseline = mom_full.iloc[i]
        date_i = p.loc[i, "date"]

        # (a) Spike the CURRENT row's price 100× — the lag legs both sit strictly before
        # date_i, so momentum at i must be unchanged (never reads the price on its own date).
        spiked_current = p.copy()
        spiked_current.loc[i, "price"] *= 100.0
        assert np.isclose(momentum_12_1(spiked_current).iloc[i], baseline)

        # (b) REAL future perturbation: 100× every row strictly AFTER date_i. A forward-
        # looking lag would change momentum at i; a backward-only lag cannot. Assert
        # unchanged (proving the match window never crosses into the future).
        future_mask = p["date"] > date_i
        assert future_mask.sum() > 0                            # there ARE future rows to spike
        spiked_future = p.copy()
        spiked_future.loc[future_mask, "price"] *= 100.0
        assert np.isclose(momentum_12_1(spiked_future).iloc[i], baseline)

    def test_net_share_issuance_absent_column(self):
        p = _annual_panel().drop(columns=["sharesbas"])
        assert net_share_issuance(p).isna().all()   # 'if available' → all-NaN, no error


# --------------------------------------------------------------------------- #
# 3. Cross-sectional normalization is STRICTLY per-date (no leakage)
# --------------------------------------------------------------------------- #
class TestCrossSectionalNormalization:
    def _two_date_panel(self) -> pd.DataFrame:
        rows = []
        # Date 1: small-scale ROE inputs; Date 2: 1000× larger scale, different names.
        specs = {
            "2022-12-31": {"AAA": 0.10, "BBB": 0.20, "CCC": 0.30, "DDD": 0.40},
            "2023-12-31": {"AAA": 100.0, "BBB": 200.0, "CCC": 300.0, "DDD": 400.0},
        }
        for d, m in specs.items():
            for tkr, r in m.items():
                rows.append(
                    {"ticker": tkr, "date": d, "netinc": r, "equity": 1.0,
                     "assets": 1.0, "marketcap": 1.0}
                )
        return pd.DataFrame(rows)

    def test_zscore_is_mean0_std1_within_each_date(self):
        feats = compute_features(self._two_date_panel(), winsor_quantile=0.0)
        for _d, g in feats.groupby("date"):
            col = g["roe"].to_numpy()
            assert np.isclose(col.mean(), 0.0, atol=1e-9)
            assert np.isclose(col.std(ddof=0), 1.0, atol=1e-9)

    def test_no_cross_date_leakage(self):
        panel = self._two_date_panel()
        both = compute_features(panel, winsor_quantile=0.0)
        # Compute features for ONLY date 1's rows.
        only_d1 = compute_features(panel[panel["date"] == "2022-12-31"], winsor_quantile=0.0)
        merged = both[both["date"] == pd.Timestamp("2022-12-31")].merge(
            only_d1, on=["ticker", "date"], suffixes=("_both", "_solo")
        )
        for feat in FEATURE_NAMES:
            a = merged[f"{feat}_both"].to_numpy()
            b = merged[f"{feat}_solo"].to_numpy()
            both_nan = np.isnan(a) & np.isnan(b)
            assert np.all(both_nan | np.isclose(a, b)), feat
        # And changing date 2's RAW inputs leaves date 1's features byte-identical.
        bumped = panel.copy()
        mask = bumped["date"] == "2023-12-31"
        bumped.loc[mask, "netinc"] = bumped.loc[mask, "netinc"] * 7.0 + 13.0
        after = compute_features(bumped, winsor_quantile=0.0)
        d1_before = both[both["date"] == pd.Timestamp("2022-12-31")].reset_index(drop=True)
        d1_after = after[after["date"] == pd.Timestamp("2022-12-31")].reset_index(drop=True)
        pd.testing.assert_frame_equal(d1_before, d1_after)

    def test_rank_normalization_is_monotonic_per_date(self):
        feats = compute_features(self._two_date_panel(), method="rank", winsor_quantile=0.0)
        for _d, g in feats.groupby("date"):
            roe_vals = g.sort_values("ticker")["roe"].to_numpy()
            # ROE inputs increase AAA<BBB<CCC<DDD on both dates → ranks strictly increase.
            assert np.all(np.diff(roe_vals) > 0)
            assert roe_vals.min() >= -1.0 and roe_vals.max() <= 1.0

    def test_winsorize_clips_per_date_tails(self):
        # Direct hand-check of the per-date clip on [1, 2, 3, 100] at q=0.25.
        block = pd.Series([1.0, 2.0, 3.0, 100.0])
        w = _winsorize(block, 0.25)
        assert np.isclose(w.iloc[0], 1.75)         # lower 25% quantile (linear interp)
        assert np.isclose(w.iloc[3], 27.25)        # upper outlier clipped to 75% quantile
        assert np.isclose(w.iloc[1], 2.0) and np.isclose(w.iloc[2], 3.0)


# --------------------------------------------------------------------------- #
# 4. NaN handling
# --------------------------------------------------------------------------- #
class TestNanHandling:
    def test_missing_input_is_nan_peers_unaffected(self):
        p = _single_date_panel()
        p.loc[1, "equity"] = np.nan               # BBB book value unknown
        out = compute_features(p, winsor_quantile=0.0)
        bbb = out.loc[out["ticker"] == "BBB", "roe"].iloc[0]
        assert np.isnan(bbb)                       # never imputed
        # The two finite peers are still standardized (mean over the 2 finite names = 0).
        finite = out.loc[out["ticker"] != "BBB", "roe"].to_numpy()
        assert np.all(np.isfinite(finite))
        assert np.isclose(np.nanmean(finite), 0.0, atol=1e-9)

    def test_too_few_finite_names_is_all_nan_for_that_date(self):
        p = _single_date_panel()
        # Wipe ROE inputs for all but one name → only 1 finite < min_obs(2) → all NaN.
        p.loc[[0, 1], "netinc"] = np.nan
        out = compute_features(p, winsor_quantile=0.0)
        assert out["roe"].isna().all()
        # A different feature with full, varying data on the same date is unaffected.
        assert out["operating_margin"].notna().all()

    def test_zero_variance_date_is_nan(self):
        # All names share the same ROE → std 0 → z-score undefined → NaN (not 0/0=nan-safe).
        p = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "date": pd.to_datetime(["2022-12-31"] * 3),
                "netinc": [50.0, 50.0, 50.0],
                "equity": [100.0, 100.0, 100.0],
                "assets": [1.0, 2.0, 3.0],
            }
        )
        out = compute_features(p, winsor_quantile=0.0)
        assert out["roe"].isna().all()             # zero cross-sectional variance
        assert out["roa"].notna().all()            # roa varies → fine


# --------------------------------------------------------------------------- #
# 5. Assembly contract
# --------------------------------------------------------------------------- #
class TestAssemblyContract:
    def test_feature_names_match_registry_and_output_columns(self):
        assert FEATURE_NAMES == list(FACTOR_FUNCTIONS)
        assert len(FEATURE_NAMES) == 14
        out = compute_features(_single_date_panel())
        assert list(out.columns) == ["ticker", "date"] + FEATURE_NAMES
        assert len(out) == 3

    def test_raw_features_tidy_shape(self):
        raw = raw_features(_annual_panel())
        assert list(raw.columns) == ["ticker", "date"] + FEATURE_NAMES
        assert len(raw) == len(_annual_panel())

    def test_missing_required_columns_raise(self):
        with pytest.raises(ValueError):
            compute_features(pd.DataFrame({"ticker": ["A"], "netinc": [1.0]}))
        with pytest.raises(TypeError):
            compute_features([1, 2, 3])  # type: ignore[arg-type]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            compute_features(_single_date_panel(), method="bogus")

    @pytest.mark.parametrize("bad_q", [-0.01, 0.5, 0.75, 1.0])
    def test_invalid_winsor_quantile_raises(self, bad_q):
        # A tail clip must be a non-negative quantile strictly below the median.
        with pytest.raises(ValueError):
            compute_features(_single_date_panel(), winsor_quantile=bad_q)

    @pytest.mark.parametrize("ok_q", [0.0, 0.02, 0.49])
    def test_valid_winsor_quantile_accepted(self, ok_q):
        out = compute_features(_single_date_panel(), winsor_quantile=ok_q)
        assert list(out.columns) == ["ticker", "date"] + FEATURE_NAMES
