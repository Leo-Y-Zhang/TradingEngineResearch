"""
TradingEngineResearch — tests for the insider-alpha runner's price-date handling
(``scripts.research_insider_alpha``). Offline, deterministic, NO network.

Guards the PIT month-labelling defect found in adversarial review: yfinance
``interval='1mo'`` bars are LABELLED at the first of the month while their Close
is the month-END price, so the runner must relabel the price matrix to month-end
before the insider feature bucketing keys off those labels. Without the fix a
mid-month filing is pushed an extra month forward (silent extra lag), and the
whole test/selftest suite (which uses ``freq='ME'`` grids) never exercises the
grid the real run actually produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research_insider_alpha import (
    _drop_partial_last_bar,
    _to_month_end_labels,
    run_research,
)


def _monthly_prices(labels: pd.DatetimeIndex, symbols: list[str],
                    seed: int) -> pd.DataFrame:
    """A tidy (ticker, date, price) frame of one bar per month at the given labels.
    The price series is identical regardless of the label convention — only the
    index labels differ between the two callers."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in symbols:
        lvl = 100.0
        for d in labels:
            lvl *= 1.0 + rng.normal(0.01, 0.05)
            rows.append({"ticker": s, "date": d, "price": lvl})
    return pd.DataFrame(rows)


class TestMonthEndRelabelling:
    def test_month_start_labels_map_to_month_end(self):
        # yfinance-style: first-of-month labels.
        idx = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"])
        px = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
        out = _to_month_end_labels(px)
        assert list(out.index) == [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-02-29"),
            pd.Timestamp("2020-03-31"),
        ]
        # Values ride along with their bar unchanged.
        assert out.loc[pd.Timestamp("2020-01-31"), "AAA"] == 1.0

    def test_idempotent_on_month_end_index(self):
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="ME")
        px = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
        out = _to_month_end_labels(px)
        assert list(out.index) == list(idx)

    def test_mid_month_labels_snap_to_month_end(self):
        idx = pd.DatetimeIndex(["2020-01-15", "2020-02-10"])
        px = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
        out = _to_month_end_labels(px)
        assert list(out.index) == [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-02-29"),
        ]

    def test_multiple_bars_in_one_month_raise(self):
        # A daily/mixed index must not pass silently as if monthly.
        idx = pd.DatetimeIndex(["2020-01-05", "2020-01-20", "2020-02-10"])
        px = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
        with pytest.raises(ValueError, match="one bar per calendar month"):
            _to_month_end_labels(px)


class TestLabelConventionDoesNotChangeResult:
    """The decisive regression: identical monthly closes fed with month-START vs
    month-END labels must yield an identical research report. Before the fix the
    month-start grid shifted every filing a month, changing the feature/return
    alignment and thus the verdict; after the fix the label convention is
    normalised away and the two runs are bit-for-bit equivalent."""

    def test_month_start_and_month_end_grids_agree(self):
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        # 30 months so the walk-forward splitter has enough dates.
        n = 30
        end_labels = pd.date_range("2020-01-31", periods=n, freq="ME")
        start_labels = pd.DatetimeIndex(
            [d.replace(day=1) for d in end_labels]
        )
        prices_end = _monthly_prices(end_labels, symbols, seed=7)
        prices_start = _monthly_prices(start_labels, symbols, seed=7)

        # A mid-month insider purchase in each month for AAA — the case that a
        # month-start label would misbucket by a month.
        txn_rows = []
        for d in end_labels:
            txn_rows.append({
                "ticker": "AAA",
                "filing_date": d.replace(day=16),
                "trans_date": d.replace(day=14),
                "trans_code": "P", "shares": 500.0, "price": 50.0,
                "owner_cik": "1001", "relationship": "OFFICER",
                "direct_indirect": "D", "is_amendment": False,
                "shrs_owned_after": 10000.0,
                "issuer_cik": "42", "issuer_name": "AAA Corp",
                "accession": f"ACC-{d:%Y%m}",
            })
        txns = pd.DataFrame(txn_rows)

        rep_end = run_research(txns, prices_end, warmup_days=0)
        rep_start = run_research(txns, prices_start, warmup_days=0)

        assert rep_start.n_rebalances == rep_end.n_rebalances
        assert rep_start.date_start == rep_end.date_start
        assert rep_start.date_end == rep_end.date_end
        assert rep_start.weights == pytest.approx(rep_end.weights)
        assert rep_start.result.deflated_sharpe_ratio == pytest.approx(
            rep_end.result.deflated_sharpe_ratio, abs=1e-9
        )
        assert rep_start.result.mean_ic == pytest.approx(
            rep_end.result.mean_ic, abs=1e-9
        )
        assert rep_start.pbo == pytest.approx(rep_end.pbo, abs=1e-9)

class TestPartialLastBarDropped:
    """The 2026-07-11 run's final OOS point was a ~7-trading-day July return
    mislabeled as a full month (yfinance serves the current month as a partial
    bar; the month-end relabel stamps it with a future month-end)."""

    def test_future_labeled_trailing_bar_is_dropped(self):
        idx = pd.date_range("2026-04-30", "2026-07-31", freq="ME")
        px = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        out = _drop_partial_last_bar(px, asof=pd.Timestamp("2026-07-11"))
        assert list(out.index) == list(idx[:-1])

    def test_completed_trailing_bar_is_kept(self):
        idx = pd.date_range("2026-03-31", "2026-06-30", freq="ME")
        px = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        out = _drop_partial_last_bar(px, asof=pd.Timestamp("2026-07-11"))
        assert list(out.index) == list(idx)

    def test_empty_frame_passes_through(self):
        px = pd.DataFrame(index=pd.DatetimeIndex([]))
        assert _drop_partial_last_bar(px, asof=pd.Timestamp("2026-07-11")).empty


class TestRenameSafeJoinInRunner:
    """run_research must recover an issuer's OLD-symbol insider history: the same
    CIK filing under a pre-rename symbol carries the SAME features as filing under
    the current one (the P1 silent join loss of the first run)."""

    def test_old_symbol_history_equals_current_symbol_history(self):
        symbols = ["NEWT", "OTHR"]
        n = 30
        labels = pd.date_range("2020-01-31", periods=n, freq="ME")
        prices = _monthly_prices(labels, symbols, seed=11)

        def _rows(as_filed: list[str]) -> pd.DataFrame:
            out = []
            for i, d in enumerate(labels):
                out.append({
                    "ticker": as_filed[i], "filing_date": d.replace(day=16),
                    "trans_date": d.replace(day=14), "trans_code": "P",
                    "shares": 500.0, "price": 50.0, "owner_cik": "1001",
                    "relationship": "OFFICER", "direct_indirect": "D",
                    "is_amendment": False, "shrs_owned_after": 10000.0,
                    "issuer_cik": "42", "issuer_name": "NewCo",
                    "accession": f"ACC-{d:%Y%m}",
                })
            return pd.DataFrame(out)

        # First half filed under the OLD symbol, then the issuer renamed to NEWT.
        renamed = _rows(["OLDT"] * (n // 2) + ["NEWT"] * (n - n // 2))
        always_new = _rows(["NEWT"] * n)

        rep_renamed = run_research(renamed, prices, warmup_days=0)
        rep_new = run_research(always_new, prices, warmup_days=0)

        assert rep_renamed.weights == pytest.approx(rep_new.weights)
        assert rep_renamed.result.mean_ic == pytest.approx(
            rep_new.result.mean_ic, abs=1e-12
        )
        assert rep_renamed.result.deflated_sharpe_ratio == pytest.approx(
            rep_new.result.deflated_sharpe_ratio, abs=1e-12
        )
