"""THE REFUTED SMALL-CAP LIQUIDITY CONSTANT.

`scripts/impact_positive_control.py` registered FIM's "US small cap" bucket as
``SMALL_CAP_DOLLAR_VOLUME_RANGE = ($10M, $50M)/day``. That is **refuted**. FIM define the
bucket by market-cap RANK — "below the Russell 1000 in market cap, typically within the
Russell 2000 universe" — and a pure rank mapping reproduces their published index volumes
out of sample on this tape. Ranks 1001-3000 then measure a **median of $3.31M/day**, IQR
$1.16M-$8.14M. $10M-$50M/day is the bottom half of the **Russell 1000** here, median
$52.1M/day.

`tests/test_spread_estimation.py::TestFimSizeBucketMapping` already pins the measurement.
What is pinned HERE is the consequence for the impact control, which is the part that was
only ever recorded in a log entry:

  **Re-running check B at the correct liquidity FAILS containment, and it fails on the
  HALF-SPREAD term, not on the impact coefficients.** The half-spread is 33.1 of the
  ~36bps floor — the E5 residual `spread_positive_control.py` already discloses and
  deliberately leaves standing. Impact alone stays inside FIM's 13.53bps all-in figure.

That distinction is the whole reason the bucket was not simply swapped: moving it without
resolving E5 turns a passing control into a failing one, and correctly so.
"""

from __future__ import annotations

import pytest

from research.spread_estimation import (
    FIM_SMALL_CAP_DOLLAR_VOLUME_IQR,
    FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME,
    FIM_SMALL_CAP_RANK_RANGE,
)
from scripts.impact_positive_control import (
    FIM_SMALL_CAP_MEDIAN_BPS,
    SMALL_CAP_DOLLAR_VOLUME_RANGE,
    SMALL_CAP_DOLLAR_VOLUME_RANGE_MEASURED,
    SMALL_CAP_DOLLAR_VOLUME_RANGE_REFUTED,
    SMALL_CAP_MEDIAN_DOLLAR_VOLUME_MEASURED,
    check_b_measured_liquidity_disclosure,
    modelled_total_bps,
)


class TestTheConstantIsMarkedRefuted:

    def test_the_registered_range_is_flagged(self):
        assert SMALL_CAP_DOLLAR_VOLUME_RANGE == (1.0e7, 5.0e7)
        assert SMALL_CAP_DOLLAR_VOLUME_RANGE_REFUTED is True

    def test_the_measured_replacement_comes_from_the_registered_measurement(self):
        """One definition, in `research.spread_estimation`, not a second copy."""
        assert SMALL_CAP_DOLLAR_VOLUME_RANGE_MEASURED is FIM_SMALL_CAP_DOLLAR_VOLUME_IQR
        assert SMALL_CAP_MEDIAN_DOLLAR_VOLUME_MEASURED is FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME

    def test_the_two_buckets_do_not_even_overlap(self):
        """The registered range sits entirely ABOVE the measured one."""
        registered_low = SMALL_CAP_DOLLAR_VOLUME_RANGE[0]
        measured_high = SMALL_CAP_DOLLAR_VOLUME_RANGE_MEASURED[1]
        assert measured_high < registered_low
        assert SMALL_CAP_MEDIAN_DOLLAR_VOLUME_MEASURED < registered_low / 3.0

    def test_the_bucket_is_a_rank_not_a_dollar_volume(self):
        assert FIM_SMALL_CAP_RANK_RANGE == (1001, 3000)


class TestContainmentFailsAtTheCorrectLiquidity:

    def test_check_b_would_fail(self):
        report = check_b_measured_liquidity_disclosure()
        assert report["contained"] is False
        low, high = report["modelled_bracket_bps"]
        assert FIM_SMALL_CAP_MEDIAN_BPS < low, (
            "FIM's measured cost sits BELOW the modelled bracket, not inside it")

    def test_the_failure_is_the_half_spread_and_not_the_impact_model(self):
        """The load-bearing distinction. Impact alone stays inside FIM's all-in figure."""
        report = check_b_measured_liquidity_disclosure()
        assert report["impact_inside_fim_all_in"] is True
        assert report["impact_conservative_bps"] < FIM_SMALL_CAP_MEDIAN_BPS
        assert report["half_spread_bps"] > FIM_SMALL_CAP_MEDIAN_BPS
        # the half-spread is essentially the whole of the modelled floor
        floor = report["modelled_bracket_bps"][0]
        assert report["half_spread_bps"] / floor > 0.95

    def test_the_disclosure_is_not_a_gate(self):
        """It must never be able to flip the control's verdict."""
        assert check_b_measured_liquidity_disclosure()["gated"] is False

    def test_the_measured_half_spread_matches_the_recorded_residual(self):
        """the internal research log iteration 9 records 33.1bps per side at this liquidity."""
        report = check_b_measured_liquidity_disclosure()
        assert report["half_spread_bps"] == pytest.approx(33.1, abs=0.1)

    def test_the_registered_bucket_is_what_made_the_control_pass(self):
        """At $10M-$50M/day the model does contain FIM's figure; at $3.31M it does not."""
        from research.capacity_study import FIM_ANCHOR_DAILY_VOLATILITY, FIM_ANCHOR_PARTICIPATION

        midpoint = sum(SMALL_CAP_DOLLAR_VOLUME_RANGE) / 2.0
        total_c, total_r, _, _ = modelled_total_bps(
            midpoint, FIM_ANCHOR_DAILY_VOLATILITY, FIM_ANCHOR_PARTICIPATION)
        assert total_r <= FIM_SMALL_CAP_MEDIAN_BPS <= total_c

        total_c, total_r, _, _ = modelled_total_bps(
            SMALL_CAP_MEDIAN_DOLLAR_VOLUME_MEASURED, FIM_ANCHOR_DAILY_VOLATILITY,
            FIM_ANCHOR_PARTICIPATION)
        assert not (total_r <= FIM_SMALL_CAP_MEDIAN_BPS <= total_c)
