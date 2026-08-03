"""Tests for the panel return-convention repair (research/multiasset/convention.py).

Offline and synthetic throughout: every fixture is constructed so the right answer is
known in closed form, which is what lets these tests fail for the right reason. The
network controls that run against real data live in ``scripts/run_convention_repair.py``
and are recorded in ``research/multiasset/convention_repair_result.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.convention import (
    BRACKET_BOUNDS,
    EQUITY_CORRECTIONS,
    Provenance,
    assert_bracket_ordering,
    bracket_dividend_yields,
    correct_equity,
    correct_panel,
    local_total_return,
    measured_dividend_yield,
    measured_fraction,
    provenance_frame,
    rates_block_unchanged,
)

MONTHS = 12


def idx(n: int, start: str = "1990-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


class TestLocalTotalReturn:
    def test_de_dollarising_inverts_the_currency_leg_exactly(self) -> None:
        i = idx(24)
        rng = np.random.default_rng(11)
        local = pd.Series(rng.normal(0.006, 0.04, 24), index=i)
        fx = pd.Series(rng.normal(0.001, 0.02, 24), index=i)
        usd = (1.0 + local) * (1.0 + fx) - 1.0

        got = local_total_return(usd, fx)

        pd.testing.assert_series_equal(got, local, check_names=False)

    def test_no_fx_leg_passes_through(self) -> None:
        i = idx(6)
        usd = pd.Series([0.01, -0.02, 0.03, 0.0, 0.05, -0.01], index=i)
        pd.testing.assert_series_equal(local_total_return(usd, None), usd,
                                       check_names=False)


class TestMeasuredDividendYield:
    def test_recovers_a_known_constant_yield(self) -> None:
        """A total return that is the price return plus a fixed monthly dividend must
        read back as that dividend, annualised, once the window is full."""
        i = idx(36)
        rng = np.random.default_rng(3)
        price = pd.Series(rng.normal(0.005, 0.03, 36), index=i)
        q_annual = 0.03
        total = (1.0 + price) * (1.0 + q_annual / MONTHS) - 1.0

        got = measured_dividend_yield(total, price)

        # Compounded over 12 months the credit is (1+q/12)^12 - 1, not q.
        expected = (1.0 + q_annual / MONTHS) ** MONTHS - 1.0
        assert got.iloc[:11].isna().all(), "a partial window must not report a yield"
        assert got.iloc[11:].notna().all()
        assert got.iloc[11:].sub(expected).abs().max() < 1e-9

    def test_months_without_a_reference_are_nan_not_zero(self) -> None:
        """The whole bracket exists to cover these months, so they must be visibly
        absent rather than quietly reading as a zero dividend."""
        i = idx(36)
        price = pd.Series(0.004, index=i)
        total = pd.Series(0.006, index=i)
        total.iloc[:20] = np.nan

        got = measured_dividend_yield(total, price)

        assert got.iloc[:20].isna().all()
        assert got.iloc[-1] == pytest.approx((1.006 / 1.004) ** MONTHS - 1.0, rel=1e-9)

    def test_a_total_return_index_reads_back_as_zero_yield(self) -> None:
        """The DAX case: if the 'price' series already contains the dividends, the
        measured gap is zero -- which is what control D checks against real data."""
        i = idx(30)
        rng = np.random.default_rng(5)
        series = pd.Series(rng.normal(0.006, 0.05, 30), index=i)

        got = measured_dividend_yield(series, series)

        assert got.dropna().abs().max() < 1e-12


class TestBracket:
    def test_ordering_holds_when_the_era_path_exceeds_its_modern_mean(self) -> None:
        """The exact case the registered bracket was amended for: pre-window US yields
        ran well above their modern mean, and a flat realistic bound would cross."""
        i = idx(48, "1980-01-31")
        measured = pd.Series(np.nan, index=i)
        measured.iloc[24:] = 0.02
        us = pd.Series(0.02, index=i)
        us.iloc[:24] = 0.05          # the old era, 2.5x the modern level

        bounds = bracket_dividend_yields(measured, us, bias_budget=0.008)

        assert_bracket_ordering({k: v.to_frame("x") for k, v in bounds.items()})
        assert bounds["conservative"].iloc[0] == 0.0
        assert bounds["central"].iloc[0] > bounds["conservative"].iloc[0]
        assert bounds["realistic"].iloc[0] >= bounds["central"].iloc[0]

    def test_inside_the_window_every_bound_uses_the_measurement(self) -> None:
        i = idx(36)
        measured = pd.Series(np.nan, index=i)
        measured.iloc[12:] = 0.025
        us = pd.Series(0.02, index=i)

        bounds = bracket_dividend_yields(measured, us, bias_budget=0.008)

        assert bounds["conservative"].iloc[20] == pytest.approx(0.025)
        assert bounds["central"].iloc[20] == pytest.approx(0.025)
        assert bounds["realistic"].iloc[20] == pytest.approx(0.033)

    def test_nothing_measured_yields_a_zero_bracket_not_a_crash(self) -> None:
        i = idx(12)
        bounds = bracket_dividend_yields(pd.Series(np.nan, index=i),
                                         pd.Series(0.02, index=i))
        for name in BRACKET_BOUNDS:
            assert (bounds[name] == 0.0).all()

    def test_a_crossing_bracket_is_rejected(self) -> None:
        i = idx(6)
        low = pd.DataFrame({"a": [0.0] * 6}, index=i)
        mid = pd.DataFrame({"a": [0.01] * 6}, index=i)
        high = pd.DataFrame({"a": [0.02] * 6}, index=i)
        high.iloc[3, 0] = 0.005      # overtakes the central bound

        with pytest.raises(AssertionError, match="bracket ordering violated"):
            assert_bracket_ordering({"conservative": low, "central": mid,
                                     "realistic": high})

    def test_ordering_reports_how_many_cells_it_checked(self) -> None:
        i = idx(10)
        frame = pd.DataFrame({"a": np.linspace(0, 1, 10), "b": np.linspace(0, 1, 10)},
                             index=i)
        got = assert_bracket_ordering({"conservative": frame, "central": frame + 0.1,
                                       "realistic": frame + 0.2})
        assert got["pairs_checked"] == 40
        assert got["ordered"] is True


class TestCorrectEquity:
    def test_is_price_minus_the_bill_plus_the_dividend(self) -> None:
        i = idx(4)
        price = pd.Series([0.02, -0.01, 0.03, 0.00], index=i)
        rf = pd.Series(0.004, index=i)
        q = pd.Series(0.024, index=i)

        got = correct_equity(price, rf, q)

        pd.testing.assert_series_equal(got, price - 0.004 + 0.002, check_names=False)

    def test_a_total_return_index_pays_the_full_bill(self) -> None:
        """DAX: zero dividend credit, so the correction is exactly minus the bill."""
        i = idx(4)
        price = pd.Series([0.02, -0.01, 0.03, 0.00], index=i)
        rf = pd.Series(0.004, index=i)

        pd.testing.assert_series_equal(correct_equity(price, rf, 0.0), price - 0.004,
                                       check_names=False)

    def test_a_missing_dividend_month_does_not_silently_delete_the_month(self) -> None:
        i = idx(4)
        price = pd.Series([0.02, -0.01, 0.03, 0.00], index=i)
        q = pd.Series([np.nan, 0.024, 0.024, 0.024], index=i)

        got = correct_equity(price, pd.Series(0.004, index=i), q)

        assert got.notna().all()
        assert got.iloc[0] == pytest.approx(0.02 - 0.004)


class TestCorrectPanelLeavesAloneWhatItMust:
    def _panel(self) -> tuple[pd.DataFrame, pd.Series]:
        i = idx(24)
        rng = np.random.default_rng(7)
        panel = pd.DataFrame(
            {"SPX": rng.normal(0.006, 0.04, 24),
             "US10Y_TR": rng.normal(0.002, 0.015, 24),
             "GOLD_F": rng.normal(0.003, 0.05, 24)},
            index=i)
        return panel, pd.Series(0.003, index=i)

    def test_the_rates_block_is_byte_identical(self) -> None:
        """Prereg control B, the anti-rigging leg: a repair that improves every block
        is a repair that is measuring its own wishes."""
        panel, rf = self._panel()

        out = correct_panel(panel, rf, {"SPX": pd.Series(0.02, index=panel.index)},
                            already_excess=("US10Y_TR",), uncorrected=("GOLD_F",))

        assert rates_block_unchanged(panel, out, ("US10Y_TR",))["max_abs_diff"] == 0.0
        pd.testing.assert_series_equal(out["GOLD_F"], panel["GOLD_F"])
        assert not out["SPX"].equals(panel["SPX"])

    def test_the_pipeline_invents_no_correction_for_an_already_excess_series(self) -> None:
        """Prereg control F. A pipeline that finds a dividend yield inside a bond total
        return is broken, and this is the test that would say so."""
        panel, rf = self._panel()

        out = correct_panel(panel, rf, {}, already_excess=("US10Y_TR", "SPX", "GOLD_F"))

        pd.testing.assert_frame_equal(out, panel)

    def test_a_moved_rates_cell_is_caught(self) -> None:
        panel, _ = self._panel()
        tampered = panel.copy()
        tampered.iloc[5, tampered.columns.get_loc("US10Y_TR")] += 1e-9

        with pytest.raises(AssertionError, match="byte-identical"):
            rates_block_unchanged(panel, tampered, ("US10Y_TR",))

    def test_a_changed_null_pattern_is_caught(self) -> None:
        panel, _ = self._panel()
        tampered = panel.copy()
        tampered.iloc[5, tampered.columns.get_loc("US10Y_TR")] = np.nan

        with pytest.raises(AssertionError, match="null pattern changed"):
            rates_block_unchanged(panel, tampered, ("US10Y_TR",))


class TestProvenance:
    def test_measured_and_bracketed_months_are_distinguished(self) -> None:
        i = idx(12)
        panel = pd.DataFrame({"SPX": 0.01, "DAX": 0.01, "US10Y_TR": 0.001,
                              "GOLD_F": 0.02}, index=i)
        measured = pd.Series(np.nan, index=i)
        measured.iloc[6:] = 0.02

        prov = provenance_frame(panel, {"SPX": measured}, exempt=("DAX",),
                                already_excess=("US10Y_TR",), uncorrected=("GOLD_F",))

        assert (prov["SPX"].iloc[:6] == Provenance.BRACKETED.value).all()
        assert (prov["SPX"].iloc[6:] == Provenance.MEASURED.value).all()
        assert (prov["DAX"] == Provenance.EXEMPT.value).all()
        assert (prov["US10Y_TR"] == Provenance.ALREADY_EXCESS.value).all()
        assert (prov["GOLD_F"] == Provenance.UNCORRECTED.value).all()

    def test_measured_fraction_ignores_cells_the_panel_never_had(self) -> None:
        """The panel is unbalanced. A series that did not exist yet must not dilute the
        share of the book that rests on measurement."""
        i = idx(10)
        live = pd.DataFrame({"SPX": [0.01] * 10, "ASX200": [np.nan] * 5 + [0.01] * 5},
                            index=i)
        measured = pd.Series([np.nan] * 5 + [0.02] * 5, index=i)
        prov = provenance_frame(live, {"SPX": measured, "ASX200": measured})

        got = measured_fraction(prov, live)

        assert got["n_live_cells"] == 15
        assert got["n_measured"] == 10          # SPX last 5 + ASX200 last 5
        assert got["frac_measured"] == pytest.approx(10 / 15)


class TestRegistry:
    def test_every_equity_instrument_in_the_panel_has_a_registered_treatment(self) -> None:
        from research.sleeves.multiasset_trend import BLOCKS

        registered = {c.key for c in EQUITY_CORRECTIONS}
        assert registered == set(BLOCKS["equity"])

    def test_dax_is_the_only_total_return_index_and_it_has_no_reference(self) -> None:
        tr = [c for c in EQUITY_CORRECTIONS if c.total_return_index]
        assert [c.key for c in tr] == ["DAX"]
        assert tr[0].reference is None, "a total-return index needs no dividend source"

    def test_every_other_instrument_names_a_measurement_source(self) -> None:
        for c in EQUITY_CORRECTIONS:
            if not c.total_return_index:
                assert c.reference, f"{c.key} has no total-return source"
