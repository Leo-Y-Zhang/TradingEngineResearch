"""THE DELISTING OFF-BY-ONE.

Every sleeve gated its terminal (delisting) return on a window running from the exit bar,
and every one of them wrote the lower edge STRICT::

    at < delisted_on <= at + 62 days

Sharadar dates a delisting ON the ticker's last traded SEP bar — median gap **0 days** —
so the strict edge rejects the MODAL case. Measured on the low-vol B2..B5 universe the
window fired **39 times out of 3,018**, and **6,322** last-observation cells carried a
delisting record (median terminal return **-1.00**) that was never booked.
`delisting_drag_annual = 0.0` was a dead code path, not a finding.

These tests pin the difference to exactly the gap-0 case, prove the registered default is
unchanged everywhere, and prove that every call site can actually be switched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.delisting import (
    CORRECTED_WINDOW,
    DELISTING_WINDOW_DAYS,
    REGISTERED_WINDOW,
    in_window,
    in_window_mask,
    window_for,
)

EXIT = pd.Timestamp("2009-03-31")


class TestTheWindowItself:

    def test_the_registered_window_rejects_the_modal_case(self):
        """gap 0 — a delisting dated ON the last traded bar. This is the defect."""
        assert not in_window(EXIT, EXIT, REGISTERED_WINDOW)
        assert in_window(EXIT, EXIT, CORRECTED_WINDOW)

    @pytest.mark.parametrize("gap", [1, 2, 30, 61, 62])
    def test_the_two_windows_agree_everywhere_else_inside(self, gap: int):
        on = EXIT + pd.Timedelta(days=gap)
        assert in_window(EXIT, on, REGISTERED_WINDOW)
        assert in_window(EXIT, on, CORRECTED_WINDOW)

    @pytest.mark.parametrize("gap", [-30, -1, 63, 400])
    def test_the_two_windows_agree_everywhere_outside(self, gap: int):
        on = EXIT + pd.Timedelta(days=gap)
        assert not in_window(EXIT, on, REGISTERED_WINDOW)
        assert not in_window(EXIT, on, CORRECTED_WINDOW)

    def test_gap_zero_is_the_only_disagreement(self):
        """Exhaustive over the whole neighbourhood: the repair changes ONE day."""
        differ = [g for g in range(-120, 200)
                  if in_window(EXIT, EXIT + pd.Timedelta(days=g), REGISTERED_WINDOW)
                  != in_window(EXIT, EXIT + pd.Timedelta(days=g), CORRECTED_WINDOW)]
        assert differ == [0]

    def test_a_name_that_never_delisted_is_never_in_the_window(self):
        assert not in_window(EXIT, None, CORRECTED_WINDOW)
        assert not in_window(EXIT, pd.NaT, CORRECTED_WINDOW)

    def test_window_for_matches_the_named_constants(self):
        assert window_for(corrected=False) == REGISTERED_WINDOW
        assert window_for(corrected=True) == CORRECTED_WINDOW
        assert REGISTERED_WINDOW == (1, DELISTING_WINDOW_DAYS)
        assert CORRECTED_WINDOW == (0, DELISTING_WINDOW_DAYS)

    def test_the_registered_window_reproduces_the_original_expression(self):
        """`(1, 62)` must be exactly `at < on <= at + 62 days`, day resolution."""
        rng = np.random.default_rng(20260728)
        for offset in rng.integers(-200, 400, 500):
            on = EXIT + pd.Timedelta(days=int(offset))
            original = bool(EXIT < on <= EXIT + pd.Timedelta(days=62))
            assert in_window(EXIT, on, REGISTERED_WINDOW) is original


class TestVectorisedMask:

    def _frame(self) -> tuple[pd.Series, pd.Series]:
        exits = pd.Series([EXIT] * 5)
        delists = pd.Series([EXIT,                                # gap 0 — the modal case
                             EXIT + pd.Timedelta(days=1),
                             EXIT + pd.Timedelta(days=62),
                             EXIT + pd.Timedelta(days=63),
                             pd.NaT])
        return exits, delists

    def test_mask_matches_the_scalar_predicate(self):
        exits, delists = self._frame()
        for window in (REGISTERED_WINDOW, CORRECTED_WINDOW):
            mask = in_window_mask(exits, delists, window)
            expected = [in_window(a, b, window) for a, b in zip(exits, delists)]
            assert list(mask) == expected

    def test_mask_is_boolean_and_never_nan(self):
        exits, delists = self._frame()
        mask = in_window_mask(exits, delists, CORRECTED_WINDOW)
        assert mask.dtype == bool
        assert not mask.isna().any()

    def test_the_repair_books_the_modal_case_and_only_it(self):
        exits, delists = self._frame()
        registered = in_window_mask(exits, delists, REGISTERED_WINDOW)
        corrected = in_window_mask(exits, delists, CORRECTED_WINDOW)
        gained = corrected & ~registered
        assert gained.tolist() == [True, False, False, False, False]
        assert not (registered & ~corrected).any()

    def test_mask_keeps_the_callers_index(self):
        exits, delists = self._frame()
        exits.index = pd.RangeIndex(100, 105)
        delists.index = pd.RangeIndex(100, 105)
        mask = in_window_mask(exits, delists, CORRECTED_WINDOW)
        assert list(mask.index) == list(range(100, 105))


# ── every call site can actually be switched ──────────────────────────────────
def _panel(n_months: int = 40, n_names: int = 80, seed: int = 11):
    """A panel whose last name STOPS TRADING, with the delisting dated on its last bar."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("2005-01", periods=n_months, freq="M")
    dead = "T079"
    dead_last = months[n_months // 2].to_timestamp(how="end").normalize()

    rows = []
    for i, month in enumerate(months):
        date = month.to_timestamp(how="end").normalize()
        tickers = [f"T{j:03d}" for j in range(n_names)]
        if date > dead_last:
            tickers = [t for t in tickers if t != dead]
        forward = rng.normal(0.008, 0.05, len(tickers))
        if date == dead_last:
            # a name's LAST observation has no forward return
            forward[tickers.index(dead)] = np.nan
        rows.append(pd.DataFrame({
            "ticker": tickers,
            "date": date,
            "band_group": "B2_200k_1M",
            "band": "B2_200k_1M",
            "forward_return": forward,
            # the dead name is ranked top so the book is certain to hold it
            "signal": [9.0 if t == dead else float(rng.normal()) for t in tickers],
            "median_dollar_volume": 5.0e5,
            "close": 25.0,
            "realised_vol": 0.02,
            "spread_regime": "measured",
            "spread": 0.004,
            "spread_conservative": 0.004,
            "spread_realistic": 0.002,
        }))
        _ = i
    panel = pd.concat(rows, ignore_index=True)
    delistings = pd.DataFrame({"ticker": [dead], "date": [dead_last],
                               "terminal_return": [-1.0]})
    return panel, delistings, dead, dead_last


class TestLowvolRetestCallSite:

    def test_the_registered_window_misses_a_gap_zero_delisting(self):
        from research.sleeves import lowvol_retest as LV

        panel, delistings, _dead, _last = _panel()
        registered = LV.run_band(panel, "B2_200k_1M", delistings)
        corrected = LV.run_band(panel, "B2_200k_1M", delistings,
                                delisting_window=LV.CORRECTED_DELISTING_WINDOW)
        assert registered is not None and corrected is not None
        assert registered.delisting_drag_annual == 0.0, (
            "the registered window books NOTHING for a delisting dated on the last bar")
        assert corrected.delisting_drag_annual < 0.0
        assert not np.array_equal(registered.gross, corrected.gross)

    def test_the_default_is_the_registered_window(self):
        from research.sleeves import lowvol_retest as LV

        panel, delistings, _dead, _last = _panel()
        default = LV.run_band(panel, "B2_200k_1M", delistings)
        explicit = LV.run_band(panel, "B2_200k_1M", delistings,
                               delisting_window=LV.REGISTERED_DELISTING_WINDOW)
        assert default is not None and explicit is not None
        assert np.array_equal(default.gross, explicit.gross)
        assert LV.REGISTERED_DELISTING_WINDOW == REGISTERED_WINDOW
        assert LV.CORRECTED_DELISTING_WINDOW == CORRECTED_WINDOW


class TestEverySleeveExposesTheSwitch:
    """The audit: no live sleeve may hard-code the strict lower edge any more."""

    @pytest.mark.parametrize("module_path,function", [
        ("research.sleeves.lowvol_retest", "run_band"),
        ("research.sleeves.low_vol_quality", "run_band"),
        ("research.capacity_study", "run_band"),
        ("research.sleeves.institutional_flow", "run_portfolio"),
        ("research.sleeves.institutional_flow", "forward_horizon_return"),
        ("research.sleeves.insider_clustering", "build_universe"),
    ])
    def test_call_site_takes_the_window_and_defaults_to_registered(self, module_path, function):
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        signature = inspect.signature(getattr(module, function))
        assert "delisting_window" in signature.parameters, (
            f"{module_path}.{function} does not expose the delisting window")
        assert signature.parameters["delisting_window"].default == REGISTERED_WINDOW, (
            f"{module_path}.{function} does not default to the REGISTERED window")

    @pytest.mark.parametrize("module_path", [
        "research.sleeves.lowvol_retest",
        "research.sleeves.low_vol_quality",
        "research.capacity_study",
        "research.sleeves.institutional_flow",
        "research.sleeves.insider_clustering",
    ])
    def test_module_reuses_the_shared_definition(self, module_path):
        import importlib

        module = importlib.import_module(module_path)
        assert module.REGISTERED_DELISTING_WINDOW is REGISTERED_WINDOW
        assert module.CORRECTED_DELISTING_WINDOW is CORRECTED_WINDOW

    def test_no_live_sleeve_hardcodes_a_strict_lower_edge(self):
        """A grep-level guard: the defective idiom must not come back."""
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        # `_lowvol_verify` deliberately replicates iteration 1 bug-for-bug for forensics;
        # `delisting.py` is the module that DOCUMENTS the defective idiom.
        skip = ("_lowvol_verify", "delisting.py")
        strict = re.compile(
            r"(delist\w*\s*>\s*\w+\[[\"']date[\"']\]"
            r"|at\s*<\s*delisted_on"
            r"|at\s*<\s*on\s*<=)"
        )
        offenders = []
        for path in sorted((repo / "research").rglob("*.py")):
            if any(part in path.parts for part in skip):
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if strict.search(line):
                    offenders.append(f"{path.relative_to(repo).as_posix()}:{number}")
        assert not offenders, ("strict delisting lower edge reintroduced at: "
                               + ", ".join(offenders))
