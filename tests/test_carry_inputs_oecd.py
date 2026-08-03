"""Offline tests for the OECD short-rate route and its overwrite gate.

`build_carry_inputs` now prefers OECD (the publisher) over FRED (the mirror), because FRED
IP-blocked this machine on 2026-07-31. The risk that introduces is a SILENT change to a
rate panel the whole repo depends on, so the gate that refuses to overwrite on drift is the
thing most worth testing — no network required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.multiasset.carry import (
    FRED_SHORT_RATES,
    OECD_DATAFLOW,
    OECD_OVERNIGHT_MEASURE,
    OECD_SHORT_MEASURE,
    OECD_SHORT_RATES,
)
from scripts.build_carry_inputs import RATE_DRIFT_LIMIT, compare_to_existing


def _panel(values: dict[str, list[float]], n: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame({k: pd.Series(v, index=idx) for k, v in values.items()})


# ------------------------------------------------------------------- registry parity


def test_oecd_and_fred_cover_exactly_the_same_currencies() -> None:
    """A route swap must not silently drop or add a currency."""
    assert set(OECD_SHORT_RATES) == set(FRED_SHORT_RATES)
    assert len(OECD_SHORT_RATES) == 10


def test_oecd_coordinates_are_the_registered_ones() -> None:
    assert OECD_SHORT_MEASURE == "IR3TIB"
    assert OECD_OVERNIGHT_MEASURE == "IRSTCI"
    assert "DSD_STES@DF_FINMARK" in OECD_DATAFLOW
    # the euro area is an aggregate, not a member state - getting this wrong would
    # silently substitute one country's rate for the bloc's
    assert OECD_SHORT_RATES["EZ"] == "EA20"


# ------------------------------------------------------------------ the overwrite gate


def test_identical_panels_report_zero_drift() -> None:
    p = _panel({"US": [0.05] * 6, "EZ": [0.02] * 6})
    report = compare_to_existing(p, p.copy())
    assert all(r["max_abs_diff"] == 0.0 for r in report.values())
    assert report["US"]["n_overlap"] == 6


def test_a_single_drifted_month_is_caught() -> None:
    old = _panel({"US": [0.05] * 6})
    new = old.copy()
    new.iloc[3, 0] = 0.05 + 0.01
    report = compare_to_existing(new, old)
    assert report["US"]["max_abs_diff"] == pytest.approx(0.01)
    assert report["US"]["max_abs_diff"] > RATE_DRIFT_LIMIT      # would refuse the write


def test_drift_below_the_limit_is_tolerated() -> None:
    """The measured OECD-vs-FRED difference must not trip the gate.

    EZ was measured at 1.244e-04 and CA at 2.5e-05; the limit is set above both on
    purpose, so a legitimate transport swap is not mistaken for corruption.
    """
    old = _panel({"EZ": [0.02] * 6})
    new = old.copy()
    new.iloc[2, 0] = 0.02 + 1.244e-4
    report = compare_to_existing(new, old)
    assert report["EZ"]["max_abs_diff"] <= RATE_DRIFT_LIMIT


def test_the_limit_sits_above_the_measured_transport_difference() -> None:
    """Regression pin: if someone tightens the limit below the known difference, the
    OECD route would start refusing every legitimate rebuild."""
    assert RATE_DRIFT_LIMIT > 1.244e-4


def test_a_new_column_is_reported_rather_than_compared() -> None:
    old = _panel({"US": [0.05] * 6})
    new = _panel({"US": [0.05] * 6, "NZ": [0.03] * 6})
    report = compare_to_existing(new, old)
    assert report["NZ"]["note"] == "new column"
    assert report["NZ"]["n_overlap"] == 0


def test_only_the_overlap_is_compared() -> None:
    """A longer new panel must not be penalised for months the old one never had."""
    old = _panel({"US": [0.05] * 6}, n=6)
    new = _panel({"US": [0.05] * 10}, n=10)
    report = compare_to_existing(new, old)
    assert report["US"]["n_overlap"] == 6
    assert report["US"]["max_abs_diff"] == 0.0


def test_missing_months_do_not_manufacture_a_difference() -> None:
    old = _panel({"US": [0.05] * 6})
    new = old.copy()
    new.iloc[1, 0] = np.nan
    report = compare_to_existing(new, old)
    assert report["US"]["n_overlap"] == 5
    assert report["US"]["max_abs_diff"] == 0.0
