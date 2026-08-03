"""THE 12-MONTH LEVERAGE BUG.

Every volatility-targeted sleeve sizes the book with::

    k = pd.concat([k_raw, k_cap], axis=1).min(axis=1)

`DataFrame.min(axis=1)` defaults to ``skipna=True``, and `k_raw` is NaN for the book's
first ``BOOK_VOL_MIN = 12`` months because the trailing volatility window has not filled.
So `k` silently becomes `k_cap` alone and the book runs at exactly ``GROSS_CAP = 10x``
with no volatility estimate behind it.

The diagnostic hid it: ``cap_binding = (k_raw > k_cap) & k_raw.notna() & ...`` excludes
precisely the months the cap is the ONLY thing setting leverage, so it reports 0% binding
for 12 months that are 100% cap-driven.

These tests pin the mechanism, the fall-through, the concealment, the repair, and the two
sleeves that were immune to it for two different reasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.book_scaler import (
    NO_ESTIMATE_CAP,
    NO_ESTIMATE_FLAT,
    REGISTERED_NO_ESTIMATE,
    book_scaler,
)

N = 24
IDX = pd.date_range("1970-01-31", periods=N, freq="ME")


def _scalers(missing: int = 12, cap: float = 5.0, raw: float = 2.0):
    k_raw = pd.Series([np.nan] * missing + [raw] * (N - missing), index=IDX)
    k_cap = pd.Series(cap, index=IDX)
    return k_raw, k_cap


class TestTheMechanism:

    def test_dataframe_min_skips_nan_which_is_the_whole_bug(self):
        k_raw, k_cap = _scalers(missing=1)
        combined = pd.concat([k_raw, k_cap], axis=1).min(axis=1)
        assert combined.iloc[0] == 5.0, "the cap alone survives where the estimate is NaN"

    def test_series_clip_propagates_nan_which_is_why_riskparity_was_immune(self):
        """`riskparity.levered` uses `clip(upper=cap)`, which keeps the NaN and goes flat."""
        k_raw, _ = _scalers(missing=1)
        assert pd.isna(k_raw.clip(upper=5.0).iloc[0])


class TestRegisteredPolicy:

    def test_registered_is_the_default(self):
        assert REGISTERED_NO_ESTIMATE == NO_ESTIMATE_CAP

    def test_the_book_runs_at_the_full_cap_with_no_estimate(self):
        k_raw, k_cap = _scalers()
        scaler = book_scaler(k_raw, k_cap)
        assert (scaler.k.iloc[:12] == 5.0).all()
        assert (scaler.k.iloc[12:] == 2.0).all()

    def test_the_registered_k_is_exactly_the_original_expression(self):
        k_raw, k_cap = _scalers()
        original = pd.concat([k_raw, k_cap], axis=1).min(axis=1)
        assert book_scaler(k_raw, k_cap).k.equals(original)

    def test_cap_binding_hides_the_no_estimate_months(self):
        """THE CONCEALMENT: 12 fully cap-driven months, and cap_binding reports zero."""
        k_raw, k_cap = _scalers()
        scaler = book_scaler(k_raw, k_cap)
        assert scaler.cap_binding.iloc[:12].sum() == 0
        assert scaler.no_estimate.iloc[:12].all()
        assert scaler.cap_or_no_estimate.iloc[:12].all()

    def test_cap_binding_still_reports_a_genuine_binding(self):
        k_raw, k_cap = _scalers(missing=0, cap=1.0, raw=2.0)
        scaler = book_scaler(k_raw, k_cap)
        assert scaler.cap_binding.all()
        assert not scaler.no_estimate.any()


class TestFlatRepair:

    def test_no_estimate_means_no_position(self):
        k_raw, k_cap = _scalers()
        scaler = book_scaler(k_raw, k_cap, no_estimate=NO_ESTIMATE_FLAT)
        assert scaler.k.iloc[:12].isna().all()
        assert (scaler.k.iloc[12:] == 2.0).all()

    def test_the_repair_touches_only_the_no_estimate_months(self):
        k_raw, k_cap = _scalers()
        registered = book_scaler(k_raw, k_cap).k
        repaired = book_scaler(k_raw, k_cap, no_estimate=NO_ESTIMATE_FLAT).k
        differ = ~(registered.eq(repaired) | (registered.isna() & repaired.isna()))
        assert list(differ) == [True] * 12 + [False] * (N - 12)

    def test_the_masks_are_reported_under_both_policies(self):
        k_raw, k_cap = _scalers()
        for policy in (NO_ESTIMATE_CAP, NO_ESTIMATE_FLAT):
            scaler = book_scaler(k_raw, k_cap, no_estimate=policy)
            assert scaler.no_estimate.sum() == 12
            assert scaler.policy == policy

    def test_an_unknown_policy_is_refused(self):
        k_raw, k_cap = _scalers()
        with pytest.raises(ValueError, match="no_estimate"):
            book_scaler(k_raw, k_cap, no_estimate="whatever")


class TestLiveMask:

    def test_a_flat_month_is_not_counted_as_cap_driven(self):
        k_raw, k_cap = _scalers()
        live = pd.Series([False] * 6 + [True] * (N - 6), index=IDX)
        scaler = book_scaler(k_raw, k_cap, live=live)
        assert scaler.no_estimate.iloc[:6].sum() == 0
        assert scaler.no_estimate.iloc[6:12].all()

    def test_masks_are_boolean_and_never_nan(self):
        k_raw, k_cap = _scalers()
        scaler = book_scaler(k_raw, k_cap)
        for mask in (scaler.cap_binding, scaler.no_estimate, scaler.cap_or_no_estimate):
            assert mask.dtype == bool
            assert not mask.isna().any()


# ── the sleeves, end to end ───────────────────────────────────────────────────
def _trend_panel(seed: int = 3, n: int = 600):
    from research.sleeves.multiasset_trend import PRIMARY_UNIVERSE

    rng = np.random.default_rng(seed)
    index = pd.date_range("1970-01-31", periods=n, freq="ME")
    x = pd.DataFrame(rng.normal(0.0, 0.04, (n, len(PRIMARY_UNIVERSE))),
                     index=index, columns=PRIMARY_UNIVERSE)
    return x, pd.DataFrame(False, index=index, columns=PRIMARY_UNIVERSE)


class TestTrendSleeve:

    def test_the_first_twelve_months_run_at_exactly_the_gross_cap(self):
        from research.sleeves.multiasset_trend import (
            BOOK_VOL_MIN,
            GROSS_CAP,
            TrendConfig,
            run_trend,
        )

        x, interior = _trend_panel()
        result = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
        at_cap = np.isclose(result.gross_leverage.to_numpy(), GROSS_CAP)
        assert int(at_cap.sum()) == BOOK_VOL_MIN
        assert at_cap[:BOOK_VOL_MIN].all(), "and they are the FIRST months of the book"

    def test_cap_binding_reports_zero_for_those_months(self):
        from research.sleeves.multiasset_trend import TrendConfig, run_trend

        x, interior = _trend_panel()
        result = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
        assert int(result.cap_binding.sum()) == 0
        assert int(result.no_vol_estimate.sum()) > 0, "the honest count is now reported"

    def test_the_repair_goes_flat_and_changes_nothing_else(self):
        from research.sleeves.multiasset_trend import (
            BOOK_VOL_MIN,
            TrendConfig,
            run_trend,
        )

        x, interior = _trend_panel()
        registered = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
        repaired = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior,
                             no_vol_estimate=NO_ESTIMATE_FLAT)
        assert len(repaired.gross) == len(registered.gross) - BOOK_VOL_MIN
        common = repaired.gross.index
        assert np.allclose(repaired.gross.to_numpy(),
                           registered.gross.reindex(common).to_numpy())

    def test_the_default_is_the_registered_policy(self):
        from research.sleeves.multiasset_trend import TrendConfig, run_trend

        x, interior = _trend_panel()
        default = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior)
        explicit = run_trend(TrendConfig(), vol_target=0.20, x=x, interior=interior,
                             no_vol_estimate=NO_ESTIMATE_CAP)
        assert default.gross.equals(explicit.gross)
        assert default.no_vol_estimate_policy == REGISTERED_NO_ESTIMATE


class TestDefensiveSleeve:

    def _run(self, **kwargs):
        from research.sleeves.multiasset_defensive import (
            PRIMARY_UNIVERSE,
            DefensiveConfig,
            run_defensive,
        )

        rng = np.random.default_rng(5)
        index = pd.date_range("1970-01-31", periods=600, freq="ME")
        x = pd.DataFrame(rng.normal(0.0, 0.04, (600, len(PRIMARY_UNIVERSE))),
                         index=index, columns=PRIMARY_UNIVERSE)
        interior = pd.DataFrame(False, index=index, columns=PRIMARY_UNIVERSE)
        return run_defensive(DefensiveConfig(), vol_target=0.20, x=x,
                             interior=interior, **kwargs)

    def test_it_has_the_same_twelve_month_hole(self):
        from research.sleeves.multiasset_trend import BOOK_VOL_MIN

        result = self._run()
        assert int(result.no_vol_estimate.sum()) == BOOK_VOL_MIN

    def test_this_sleeve_already_counted_it_in_cap_binding(self):
        """Defensive folds the hole into cap_binding; trend and value do not."""
        result = self._run()
        assert bool((result.no_vol_estimate & ~result.cap_binding).sum() == 0)

    def test_the_repair_goes_flat(self):
        from research.sleeves.multiasset_trend import BOOK_VOL_MIN

        registered = self._run()
        repaired = self._run(no_vol_estimate=NO_ESTIMATE_FLAT)
        assert len(repaired.gross) == len(registered.gross) - BOOK_VOL_MIN


class TestValueSleeve:

    def _run(self, **kwargs):
        import tests.multiasset_value_test as harness
        from research.sleeves.multiasset_value import ValueConfig, run_value

        index = harness._index()
        return run_value(ValueConfig(), vol_target=0.20, x=harness._cycles(),
                         spreads=harness._spreads(index), **kwargs)

    def test_it_has_the_same_twelve_month_hole(self):
        from research.sleeves.multiasset_value import BOOK_VOL_MIN

        assert int(self._run().no_vol_estimate.sum()) == BOOK_VOL_MIN

    def test_cap_binding_hides_it_here_too(self):
        result = self._run()
        assert int((result.no_vol_estimate & result.cap_binding).sum()) == 0

    def test_the_repair_goes_flat(self):
        from research.sleeves.multiasset_value import BOOK_VOL_MIN

        registered = self._run()
        repaired = self._run(no_vol_estimate=NO_ESTIMATE_FLAT)
        assert len(repaired.gross) == len(registered.gross) - BOOK_VOL_MIN


class TestEverySleeveOnThatPathExposesTheSwitch:

    @pytest.mark.parametrize("module_path,function", [
        ("research.sleeves.multiasset_trend", "run_trend"),
        ("research.sleeves.multiasset_value", "run_value"),
        ("research.sleeves.multiasset_defensive", "run_defensive"),
        ("research.sleeves.multiasset_seasonal", "_book_scaler"),
    ])
    def test_the_policy_is_a_parameter_defaulting_to_registered(self, module_path, function):
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        parameters = inspect.signature(getattr(module, function)).parameters
        assert "no_vol_estimate" in parameters, (
            f"{module_path}.{function} does not expose the no-estimate policy")
        assert parameters["no_vol_estimate"].default == REGISTERED_NO_ESTIMATE

    def test_no_sleeve_hardcodes_the_fall_through_any_more(self):
        """A grep guard: the raw two-column `min` idiom must not come back."""
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        idiom = re.compile(r"pd\.concat\(\[k_raw\w*,\s*k_cap\]\s*,\s*axis=1\)\.min\(")
        offenders = []
        for path in sorted((repo / "research").rglob("*.py")):
            if path.name == "book_scaler.py":     # the module that documents it
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if idiom.search(line):
                    offenders.append(f"{path.relative_to(repo).as_posix()}:{number}")
        assert not offenders, ("the un-switchable scaler idiom is back at: "
                               + ", ".join(offenders))
