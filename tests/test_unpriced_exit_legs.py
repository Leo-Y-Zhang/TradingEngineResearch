"""UNPRICED SELL LEGS — turnover counted them, the cost model did not.

When a held name leaves the TRADABLE universe — its price falls through the $2 floor, its
dollar volume moves it into another band, or its spread stops resolving — it is absent
from that month's priced cross-section. Every long-only sleeve `continue`d past the cost
accumulation at that point while STILL counting the leg in turnover. **A name that leaves
the universe still has to be sold.** 777 sell legs in one band of the low-vol book were
counted and charged nothing.

The repair prices the leg at the name's LAST OBSERVED inputs. That is the nearest honest
estimate available and is certainly too cheap — a name that has just fallen out of the
universe trades worse, not better — so the corrected cost is still a lower bound. Both
counters are reported under BOTH settings, so the size of the omission is never hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

BAND = "B2_200k_1M"
OTHER_BAND = "B3_1M_5M"


def _panel_with_a_band_migration(n_months: int = 48, n_names: int = 80,
                                 leaves_at: int = 20, seed: int = 20260728):
    """A panel in which one HELD name leaves the band while still trading.

    It keeps a forward return throughout, so it is not a delisting: it is exactly the
    case the cost model skipped — a position that must be SOLD at a price the tradable
    universe no longer quotes.
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range("1990-01", periods=n_months, freq="M")
    leaver = f"T{n_names - 1:03d}"

    rows = []
    for i, month in enumerate(months):
        tickers = [f"T{j:03d}" for j in range(n_names)]
        bands = [OTHER_BAND if (t == leaver and i >= leaves_at) else BAND
                 for t in tickers]
        rows.append(pd.DataFrame({
            "ticker": tickers,
            "date": month.to_timestamp(how="end").normalize(),
            "band_group": bands,
            "band": bands,
            "forward_return": rng.normal(0.008, 0.05, n_names),
            # the leaver is ranked top so the book is certain to be holding it
            "signal": [9.0 if t == leaver else float(rng.normal()) for t in tickers],
            "median_dollar_volume": 5.0e5,
            "close": 25.0,
            "realised_vol": 0.02,
            "spread_regime": "measured",
            "spread": 0.004,
            "spread_conservative": 0.004,
            "spread_realistic": 0.002,
        }))
    panel = pd.concat(rows, ignore_index=True)
    delistings = pd.DataFrame({"ticker": pd.Series(dtype="object"),
                               "date": pd.Series(dtype="datetime64[ns]"),
                               "terminal_return": pd.Series(dtype="float")})
    return panel, delistings, leaver


@pytest.fixture(scope="module")
def migration_panel():
    return _panel_with_a_band_migration()


class TestLowvolRetest:

    def test_the_registered_run_leaves_the_sell_leg_free(self, migration_panel):
        from research.sleeves import lowvol_retest as LV

        panel, delistings, _leaver = migration_panel
        books = LV.run_band(panel, BAND, delistings)
        assert books is not None
        assert books.unpriced_exit_legs > 0, "the setup did not produce a free sell leg"
        assert books.charged_unpriced_exit_legs == 0

    def test_the_free_leg_was_counted_in_turnover(self, migration_panel):
        """The defect precisely: turnover counts the leg, the cost model does not."""
        from research.sleeves import lowvol_retest as LV

        panel, delistings, _leaver = migration_panel
        priced = LV.run_band(panel, BAND, delistings, charge_unpriced_exits=True)
        free = LV.run_band(panel, BAND, delistings)
        assert priced is not None and free is not None
        assert priced.legs_traded == free.legs_traded, (
            "the leg is in turnover under BOTH settings; only the COST differs")

    def test_charging_the_leg_raises_cost_under_both_bounds(self, migration_panel):
        from research.sleeves import lowvol_retest as LV

        panel, delistings, _leaver = migration_panel
        free = LV.run_band(panel, BAND, delistings)
        priced = LV.run_band(panel, BAND, delistings, charge_unpriced_exits=True)
        assert free is not None and priced is not None
        assert priced.charged_unpriced_exit_legs == priced.unpriced_exit_legs
        for bound in ("conservative", "realistic"):
            costs = getattr(priced, f"cost_{bound}")
            was = getattr(free, f"cost_{bound}")
            assert costs.sum() > was.sum(), bound
        assert priced.gross.tolist() == free.gross.tolist(), (
            "the signal cannot see costs, so the HOLDINGS must be identical")

    def test_the_default_is_the_registered_free_exit(self, migration_panel):
        import inspect

        from research.sleeves import lowvol_retest as LV

        panel, delistings, _leaver = migration_panel
        parameter = inspect.signature(LV.run_band).parameters["charge_unpriced_exits"]
        assert parameter.default is False
        default = LV.run_band(panel, BAND, delistings)
        explicit = LV.run_band(panel, BAND, delistings, charge_unpriced_exits=False)
        assert default is not None and explicit is not None
        assert np.array_equal(default.cost_conservative, explicit.cost_conservative)


class TestLowVolQuality:
    """The direct ancestor of `lowvol_retest`, which had the same free exit."""

    def test_the_registered_run_leaves_the_sell_leg_free(self, migration_panel):
        from research.sleeves import low_vol_quality as LVQ

        panel, delistings, _leaver = migration_panel
        result = LVQ.run_band(panel, BAND, delistings)
        assert result is not None
        assert result.unpriced_exit_legs > 0
        assert result.charged_unpriced_exit_legs == 0

    def test_charging_the_leg_raises_the_cost_drag(self, migration_panel):
        from research.sleeves import low_vol_quality as LVQ

        panel, delistings, _leaver = migration_panel
        free = LVQ.run_band(panel, BAND, delistings)
        priced = LVQ.run_band(panel, BAND, delistings, charge_unpriced_exits=True)
        assert free is not None and priced is not None
        assert priced.charged_unpriced_exit_legs == priced.unpriced_exit_legs
        assert priced.cost_drag_annual > free.cost_drag_annual
        assert priced.net_return_annual < free.net_return_annual
        assert priced.turnover_annual == free.turnover_annual, (
            "turnover is unchanged; only the COST of the same legs moves")
        assert priced.gross_return_annual == free.gross_return_annual

    def test_the_default_is_the_registered_free_exit(self, migration_panel):
        import inspect

        from research.sleeves import low_vol_quality as LVQ

        parameter = inspect.signature(LVQ.run_band).parameters["charge_unpriced_exits"]
        assert parameter.default is False
        panel, delistings, _leaver = migration_panel
        default = LVQ.run_band(panel, BAND, delistings)
        explicit = LVQ.run_band(panel, BAND, delistings, charge_unpriced_exits=False)
        assert default is not None and explicit is not None
        assert default.cost_drag_annual == explicit.cost_drag_annual


class TestEveryPerNameCostingSleeveExposesTheSwitch:
    """The audit. Sleeves with no per-name costing are out of scope by construction."""

    @pytest.mark.parametrize("module_path,function", [
        ("research.sleeves.lowvol_retest", "run_band"),
        ("research.sleeves.low_vol_quality", "run_band"),
        ("research.sleeves.institutional_flow", "run_portfolio"),
    ])
    def test_the_switch_exists_and_defaults_to_the_registered_behaviour(
            self, module_path, function):
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        parameters = inspect.signature(getattr(module, function)).parameters
        assert "charge_unpriced_exits" in parameters, (
            f"{module_path}.{function} cannot charge a name it can no longer price")
        assert parameters["charge_unpriced_exits"].default is False

    @pytest.mark.parametrize("module_path,function,fields", [
        ("research.sleeves.lowvol_retest", "BandBooks",
         ("unpriced_exit_legs", "charged_unpriced_exit_legs")),
        ("research.sleeves.low_vol_quality", "SleeveResult",
         ("unpriced_exit_legs", "charged_unpriced_exit_legs")),
        ("research.sleeves.institutional_flow", "PortfolioRun",
         ("unpriced_exit_legs", "charged_unpriced_exit_legs")),
    ])
    def test_the_omission_is_always_counted_even_when_it_is_not_charged(
            self, module_path, function, fields):
        import dataclasses
        import importlib

        module = importlib.import_module(module_path)
        names = {f.name for f in dataclasses.fields(getattr(module, function))}
        assert set(fields) <= names, f"{module_path}.{function} hides the count"
