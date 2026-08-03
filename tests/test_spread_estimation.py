"""Tests for the daily-OHLC bid-ask spread estimators.

The estimator is the load-bearing component of the capacity-curve study's cost model
(`research/medallion_style_alpha_search/capacity_curve_prereg.md` §6). Sharadar SEP
carries no quotes, so the spread every position pays must be *estimated* from daily
high/low. If that estimate is wrong the whole study is void, so the estimator is tested
against ground truth it cannot see.

Three estimators are covered. Corwin-Schultz and Abdi-Ranaldo were both tried and both
failed the registered positive control; EDGE (Ardia, Guidotti & Kroencke 2024) passed
and is what the cost model uses. The earlier two are retained because their failure
modes are the tests: a floor that fakes a spread, and a dilution that fakes a free lunch.

The failure mode that matters most is silent and specific: a name that did not trade has
``high == low``, which drives a naive estimator's inputs to zero and yields a spread of
*zero* -- a free lunch, manufactured precisely in the illiquid band the study is about.
That case must return NaN (unknown, therefore untradeable), never 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.spread_estimation import (
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_DEV_WINDOW,
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED,
    AGK_LIQUIDITY_ANCHOR_SPREAD,
    EDGE_FLOOR_PER_UNIT_VOL,
    FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME,
    FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME_MEASURED,
    FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME,
    FIM_SMALL_CAP_DOLLAR_VOLUME_IQR,
    FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME,
    FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME_MEASURED,
    FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME,
    FIM_SMALL_CAP_RANK_RANGE,
    bounds_from_estimate,
    bracket_verdict,
    corwin_schultz_spread,
    edge_spread,
    era_multiplier,
    liquid_name_spread,
    minimum_quoted_spread,
    resolution_floor,
    rolling_spread_estimate,
    spread_cost_bounds,
    spread_with_resolution,
    tick_size,
)


def simulate_bid_ask_bounce(
    n_days: int,
    true_spread: float,
    daily_vol: float = 0.02,
    ticks_per_day: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Daily OHLC generated from a known spread, so the estimate has ground truth.

    A true log price random-walks intraday; every observed transaction prints at the
    bid or the ask with equal probability. The daily high and low are therefore the
    extremes of a mid-price range *widened by the spread* -- which is exactly the
    signal Corwin-Schultz exploits to separate spread from volatility.
    """
    rng = np.random.default_rng(seed)
    step = daily_vol / np.sqrt(ticks_per_day)
    highs, lows, closes = [], [], []
    log_price = np.log(100.0)

    for _ in range(n_days):
        increments = rng.normal(0.0, step, ticks_per_day)
        intraday_true = log_price + np.cumsum(increments)
        side = rng.choice([-1.0, 1.0], ticks_per_day)
        observed = np.exp(intraday_true) * (1.0 + side * true_spread / 2.0)
        highs.append(observed.max())
        lows.append(observed.min())
        closes.append(observed[-1])
        log_price = intraday_true[-1]

    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


class TestGroundTruthRecovery:
    """The estimator must recover a spread it was never told."""

    @pytest.mark.parametrize("true_spread", [0.005, 0.02, 0.05])
    def test_recovers_known_spread(self, true_spread: float) -> None:
        ohlc = simulate_bid_ask_bounce(n_days=1500, true_spread=true_spread)
        estimate = corwin_schultz_spread(ohlc["high"], ohlc["low"])

        assert np.isfinite(estimate)
        # Corwin-Schultz is noisy per-pair but roughly unbiased in aggregate. A factor
        # of two either way is a meaningful test: it distinguishes a working estimator
        # from one that is off by an order of magnitude, which is what would actually
        # void the cost model.
        assert true_spread / 2.0 < estimate < true_spread * 2.0, (
            f"estimated {estimate:.5f} for a true spread of {true_spread:.5f}"
        )

    def test_is_monotonic_in_the_true_spread(self) -> None:
        estimates = [
            corwin_schultz_spread(
                *simulate_bid_ask_bounce(1200, s)[["high", "low"]].T.values
            )
            for s in (0.002, 0.01, 0.04)
        ]
        assert estimates[0] < estimates[1] < estimates[2]

    def test_high_volatility_is_not_mistaken_for_wide_spread(self) -> None:
        """The estimator's entire purpose: separate volatility from spread.

        A naive high-low range would report the volatile stock as far more expensive.
        """
        calm = simulate_bid_ask_bounce(1500, true_spread=0.01, daily_vol=0.01)
        wild = simulate_bid_ask_bounce(1500, true_spread=0.01, daily_vol=0.06)

        calm_estimate = corwin_schultz_spread(calm["high"], calm["low"])
        wild_estimate = corwin_schultz_spread(wild["high"], wild["low"])

        naive_calm = float(np.mean(calm["high"] / calm["low"] - 1.0))
        naive_wild = float(np.mean(wild["high"] / wild["low"] - 1.0))
        assert naive_wild > naive_calm * 3, "the naive range should be badly fooled"

        assert abs(wild_estimate - calm_estimate) < 0.02


class TestNonTradingDays:
    """The failure mode that would manufacture a free lunch in the illiquid band."""

    def test_zero_range_days_do_not_report_a_zero_spread(self) -> None:
        """high == low means the stock did not trade, not that trading was free."""
        flat = pd.Series([10.0] * 60)
        result = corwin_schultz_spread(flat, flat)
        assert np.isnan(result), f"expected NaN for a non-trading name, got {result}"

    @pytest.mark.parametrize("flat_fraction", [0.05, 0.25])
    def test_scattered_flat_days_are_excluded_not_treated_as_zero(
        self, flat_fraction: float
    ) -> None:
        """A thinly-traded name is priced off the days it actually traded.

        The two fractions are measured from the real SEP tape: names with median
        volume of 10k-100k shares print ``high == low`` on 4.4% of days, and names at
        1k-10k shares on 23.1%. Both sit inside the study's tradable bands, so both
        must yield a finite estimate that is NOT dragged toward zero.
        """
        ohlc = simulate_bid_ask_bounce(n_days=1500, true_spread=0.03)
        stale = ohlc.copy()
        rng = np.random.default_rng(11)
        flat_days = rng.random(len(stale)) < flat_fraction
        stale.loc[flat_days, "high"] = stale.loc[flat_days, "close"]
        stale.loc[flat_days, "low"] = stale.loc[flat_days, "close"]

        clean_estimate = corwin_schultz_spread(ohlc["high"], ohlc["low"])
        stale_estimate = corwin_schultz_spread(stale["high"], stale["low"])

        assert np.isfinite(stale_estimate)
        # Folding flat days in as zero-spread observations would pull the mean toward
        # zero in proportion to flat_fraction. It must not move materially at all.
        assert stale_estimate > clean_estimate * 0.85

    def test_a_name_with_no_consecutive_trading_days_is_unmeasurable(self) -> None:
        """Strict alternation leaves zero consecutive pairs -- NaN is the right answer.

        Corwin-Schultz compares a one-day range against the range over the *adjacent*
        two days; volatility scales with elapsed time, so pairing day t with day t+2
        would silently attribute an extra day of volatility to the spread. A name that
        never trades twice in a row therefore has no honest estimate, and the study
        must treat it as untradeable rather than invent a number for it. Real names
        with ~0 median volume print flat on 51% of days and land here.
        """
        ohlc = simulate_bid_ask_bounce(n_days=1200, true_spread=0.03)
        alternating = ohlc.copy()
        even = alternating.index % 2 == 0
        alternating.loc[even, "high"] = alternating.loc[even, "close"]
        alternating.loc[even, "low"] = alternating.loc[even, "close"]

        assert np.isnan(corwin_schultz_spread(alternating["high"],
                                              alternating["low"]))

    def test_insufficient_valid_pairs_returns_nan(self) -> None:
        assert np.isnan(corwin_schultz_spread(pd.Series([1.0]), pd.Series([1.0])))
        assert np.isnan(corwin_schultz_spread(pd.Series([], dtype=float),
                                              pd.Series([], dtype=float)))

    def test_non_positive_prices_are_rejected_not_logged(self) -> None:
        highs = pd.Series([10.0, 0.0, 11.0, 12.0] * 20)
        lows = pd.Series([9.0, 0.0, 10.0, 11.0] * 20)
        result = corwin_schultz_spread(highs, lows)
        assert np.isfinite(result), "zero-priced rows should be dropped, not poison it"


class TestEstimatorMechanics:
    def test_overnight_gap_is_adjusted_away(self) -> None:
        """An overnight gap inflates the two-day range and fakes a wide spread."""
        ohlc = simulate_bid_ask_bounce(n_days=800, true_spread=0.01, seed=7)
        gapped = ohlc.copy()
        # A clean 20% overnight jump partway through, with no change in spread.
        gapped.loc[400:, ["high", "low", "close"]] *= 1.20

        base = corwin_schultz_spread(ohlc["high"], ohlc["low"])
        with_gap = corwin_schultz_spread(gapped["high"], gapped["low"])
        assert abs(with_gap - base) < 0.005

    def test_negative_estimates_are_floored_at_zero_per_paper(self) -> None:
        """Corwin & Schultz (2012) set negative two-day estimates to zero.

        Flooring is applied per pair *before* averaging; without it the mean is
        biased downward by noise, which would understate cost.
        """
        rng = np.random.default_rng(3)
        mids = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
        # Zero true spread: roughly half the raw pair estimates will come out negative.
        highs = pd.Series(mids * 1.001)
        lows = pd.Series(mids * 0.999)
        estimate = corwin_schultz_spread(highs, lows)
        assert estimate >= 0.0

    def test_rolling_estimate_is_point_in_time(self) -> None:
        """A rolling estimate must never use a bar the strategy has not seen."""
        ohlc = simulate_bid_ask_bounce(n_days=400, true_spread=0.02)
        rolling = rolling_spread_estimate(ohlc["high"], ohlc["low"], window=63)

        assert len(rolling) == len(ohlc)
        assert rolling.iloc[:62].isna().all(), "no estimate before the window fills"

        truncated = rolling_spread_estimate(
            ohlc["high"].iloc[:200], ohlc["low"].iloc[:200], window=63
        )
        pd.testing.assert_series_equal(
            rolling.iloc[:200], truncated, check_names=False
        )


def simulate_ohlc(n_days, true_spread, daily_vol=0.02, overnight_frac=0.5,
                  ticks_per_day=100, seed=42):
    """As above, but emitting an open price and realistic overnight gaps.

    EDGE needs the open, and the overnight gap is the stylised fact that separated
    working estimators from broken ones here.
    """
    rng = np.random.default_rng(seed)
    intraday = daily_vol * np.sqrt(1.0 - overnight_frac)
    overnight = daily_vol * np.sqrt(overnight_frac)
    step = intraday / np.sqrt(ticks_per_day)
    opens, highs, lows, closes = [], [], [], []
    log_price = np.log(100.0)
    for _ in range(n_days):
        log_price += rng.normal(0.0, overnight)
        path = log_price + np.cumsum(rng.normal(0.0, step, ticks_per_day))
        side = rng.choice([-1.0, 1.0], ticks_per_day)
        observed = np.exp(path) * (1.0 + side * true_spread / 2.0)
        opens.append(observed[0])
        highs.append(observed.max())
        lows.append(observed.min())
        closes.append(observed[-1])
        log_price = path[-1]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes})


class TestEdgeEstimator:
    """EDGE is the estimator the cost model actually uses (prereg erratum 1)."""

    @pytest.mark.parametrize("true_bps", [100, 200, 400])
    @pytest.mark.parametrize("daily_vol", [0.02, 0.06])
    def test_recovers_known_spread_across_volatility(self, true_bps, daily_vol):
        """The property that made EDGE the choice: accuracy holds as vol rises.

        Abdi-Ranaldo degrades to 18-22% error at 4-6% daily vol; EDGE stays within a
        few percent, which matters because the study's illiquid bands are volatile.
        """
        floor_bps = EDGE_FLOOR_PER_UNIT_VOL * daily_vol * 1e4
        if true_bps < floor_bps:
            pytest.skip(
                f"a {true_bps}bps spread is below the {floor_bps:.0f}bps resolution "
                f"floor at {daily_vol:.0%} daily vol -- not recoverable from daily "
                f"bars, which is an information limit rather than an estimator defect"
            )
        estimates = [
            edge_spread(*simulate_ohlc(750, true_bps / 1e4, daily_vol, seed=s)
                        [["open", "high", "low", "close"]].T.values) * 1e4
            for s in range(4)
        ]
        estimate = float(np.mean(estimates))
        assert abs(estimate - true_bps) / true_bps < 0.10, (
            f"EDGE returned {estimate:.1f}bps for a true {true_bps}bps "
            f"at {daily_vol:.0%} daily vol"
        )

    def test_requires_three_observations(self):
        tiny = pd.Series([1.0, 1.1])
        assert np.isnan(edge_spread(tiny, tiny, tiny, tiny))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            edge_spread(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]),
                        pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))

    def test_non_positive_prices_do_not_poison_the_estimate(self):
        frame = simulate_ohlc(400, 0.02)
        frame.loc[50, ["open", "high", "low", "close"]] = 0.0
        result = edge_spread(frame["open"], frame["high"], frame["low"],
                             frame["close"])
        assert np.isfinite(result)


class TestResolutionRegime:
    """The guard that stops a noise floor being sold as a measured trading cost."""

    def test_floor_scales_linearly_with_volatility(self):
        assert resolution_floor(0.02) == pytest.approx(2 * resolution_floor(0.01))
        assert resolution_floor(0.01) == pytest.approx(EDGE_FLOOR_PER_UNIT_VOL * 0.01)

    def test_floor_is_nan_for_degenerate_volatility(self):
        assert np.isnan(resolution_floor(0.0))
        assert np.isnan(resolution_floor(float("nan")))

    def test_a_wide_spread_is_reported_as_measured(self):
        frame = simulate_ohlc(750, 0.03, daily_vol=0.02)
        value, regime = spread_with_resolution(
            frame["open"], frame["high"], frame["low"], frame["close"]
        )
        assert regime == "measured"
        assert abs(value * 1e4 - 300) / 300 < 0.15

    def test_a_negligible_spread_is_reported_as_an_upper_bound(self):
        """A near-zero true spread must never be sold as a measurement."""
        frame = simulate_ohlc(750, 0.0002, daily_vol=0.02)
        _, regime = spread_with_resolution(
            frame["open"], frame["high"], frame["low"], frame["close"]
        )
        assert regime == "upper_bound"

    def test_upper_bound_is_not_inflated_to_the_floor(self):
        """Raising an upper bound to the floor would manufacture cost from ignorance.

        It would also do so hardest for volatile names -- exactly the population the
        capacity study is about -- so it must return the estimate itself.
        """
        frame = simulate_ohlc(750, 0.0002, daily_vol=0.02)
        value, regime = spread_with_resolution(
            frame["open"], frame["high"], frame["low"], frame["close"]
        )
        raw = edge_spread(frame["open"], frame["high"], frame["low"], frame["close"])
        assert regime == "upper_bound"
        assert value == pytest.approx(raw)

    def test_a_non_trading_name_is_unmeasurable_not_free(self):
        flat = pd.Series([10.0] * 80)
        value, regime = spread_with_resolution(flat, flat, flat, flat)
        assert regime == "unmeasurable"
        assert np.isnan(value)


MEGA_CAP_DOLLAR_VOLUME = 5e9
MEGA_CAP_PRICE = 150.0
MODERN = "2020-06-30"


def a_mega_cap_frame() -> pd.DataFrame:
    """A liquid name: real volatility, a spread far below what daily bars can resolve."""
    return simulate_ohlc(750, 0.0002, daily_vol=0.02, seed=5)


class TestTickRegime:
    """The regulatory floor. A quote cannot be narrower than one tick, ever."""

    @pytest.mark.parametrize(
        ("date", "expected"),
        [
            ("1996-01-02", 0.125),      # eighths
            ("1997-06-23", 0.125),      # last day of eighths
            ("1997-06-24", 0.0625),     # NYSE moves to sixteenths
            ("2001-04-08", 0.0625),     # last day before full decimalisation
            ("2001-04-09", 0.01),       # decimalisation complete
            ("2024-01-02", 0.01),       # Reg NMS Rule 612
        ],
    )
    def test_tick_size_by_era(self, date, expected):
        assert tick_size(date) == expected

    def test_undated_callers_get_the_cheapest_regime(self):
        """An undated caller must not be charged a 1996 tick it may not deserve."""
        assert tick_size(None) == 0.01

    def test_minimum_quoted_spread_is_one_tick_over_price(self):
        assert minimum_quoted_spread(20.0, "2015-01-02") == pytest.approx(0.0005)
        # The same $20 stock in 1999 could not legally trade inside 31bps.
        assert minimum_quoted_spread(20.0, "1999-01-04") == pytest.approx(0.003125)

    @pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
    def test_no_price_means_no_floor_is_known_not_that_there_is_none(self, price):
        assert np.isnan(minimum_quoted_spread(price, MODERN))


class TestEraMultiplier:
    def test_pre_decimalisation_eras_are_wider(self):
        assert era_multiplier("1999-06-30") > era_multiplier("2014-06-30")
        assert era_multiplier("1999-06-30") == pytest.approx(0.0168 / 0.0076)

    def test_the_factor_is_never_a_discount(self):
        """A uniform post-2003 discount would quietly cheapen modern small caps.

        The compression in the source table is concentrated in large caps, so passing it
        through uniformly understates the cost of exactly the names the programme trades.
        Refusing the discount cannot flatter any strategy; granting it could.
        """
        for year in range(1993, 2027):
            assert era_multiplier(f"{year}-06-30") >= 1.0
        assert era_multiplier("2014-06-30") == 1.0
        assert era_multiplier(None) == 1.0


class TestLiquidNameSchedule:
    """Bound (b): the documented cost of a name the estimator cannot resolve."""

    def test_reproduces_its_source_anchors(self):
        """The schedule must return the published quintile medians at its anchors."""
        for volume, spread in zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
                                  AGK_LIQUIDITY_ANCHOR_SPREAD):
            assert liquid_name_spread(volume, price=None, when=MODERN) == pytest.approx(
                spread, rel=1e-9
            )

    def test_is_monotonically_cheaper_as_liquidity_rises(self):
        volumes = [1e5, 5e5, 2e6, 1e7, 5e7, 5e9]
        values = [liquid_name_spread(v, price=None, when=MODERN) for v in volumes]
        assert all(a >= b for a, b in zip(values, values[1:])), values

    def test_is_clamped_and_never_extrapolated(self):
        """Past the table's support the schedule holds the endpoint, it does not fit.

        Extrapolating a power law would make a $5bn/day name cheaper than the largest
        quintile the source ever measured -- inventing liquidity the data never showed.
        """
        top = liquid_name_spread(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME[-1],
                                 price=None, when=MODERN)
        assert liquid_name_spread(1e12, price=None, when=MODERN) == pytest.approx(top)
        bottom = liquid_name_spread(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME[0],
                                    price=None, when=MODERN)
        assert liquid_name_spread(1.0, price=None, when=MODERN) == pytest.approx(bottom)

    def test_the_tick_floor_binds_before_decimalisation(self):
        """A 1999 mega-cap cannot be charged a 2020 spread. The tick forbids it."""
        modern = liquid_name_spread(5e9, price=20.0, when="2015-06-30")
        legacy = liquid_name_spread(5e9, price=20.0, when="1999-06-30")
        assert modern == pytest.approx(AGK_LIQUIDITY_ANCHOR_SPREAD[-1])
        assert legacy == pytest.approx(minimum_quoted_spread(20.0, "1999-06-30"))
        # 31.25bps against 9bps: the sixteenth tick alone makes the same name 3.5x
        # dearer in 1999 than the modern schedule would charge it.
        assert legacy > modern * 3.0

    @pytest.mark.parametrize("volume", [0.0, -1.0, float("nan"), None])
    def test_no_dollar_volume_means_no_quote(self, volume):
        """Missing liquidity is not free liquidity."""
        assert np.isnan(liquid_name_spread(volume, price=MEGA_CAP_PRICE, when=MODERN))


class TestTwoCostBounds:
    """The universe-bias fix: price `upper_bound` names, do not delete them."""

    def test_mega_caps_are_cheap_under_b_and_expensive_under_a(self):
        """The registered positive control, in unit-test form.

        This is the whole finding. The same mega-cap is ~50bps under the bound that
        overstates cost and 9bps under the documented one, and iteration 1 resolved that
        disagreement by throwing the name away -- which is how six strategies ended up
        trading only the expensive tail of the market.
        """
        frame = a_mega_cap_frame()
        bounds = spread_cost_bounds(
            frame["open"], frame["high"], frame["low"], frame["close"],
            median_dollar_volume=MEGA_CAP_DOLLAR_VOLUME,
            price=MEGA_CAP_PRICE, when=MODERN,
        )

        assert bounds.regime == "upper_bound"
        assert bounds.tradable, "the cheapest names in the market must be tradable"

        per_side_realistic = bounds.realistic / 2.0 * 1e4
        assert 1.0 <= per_side_realistic <= 5.0, (
            f"bound (b) put a mega-cap at {per_side_realistic:.2f}bps per side; the "
            f"registered window is 1-5bps"
        )

        per_side_conservative = bounds.conservative / 2.0 * 1e4
        assert per_side_conservative > 10.0, (
            f"bound (a) must still charge the noise floor, got "
            f"{per_side_conservative:.2f}bps per side"
        )
        assert bounds.conservative > bounds.realistic * 2.0

    @pytest.mark.parametrize("volume", [5e4, 5e5, 5e6, 5e7, 5e9])
    @pytest.mark.parametrize("regime", ["measured", "upper_bound"])
    def test_the_bracket_can_never_invert(self, volume, regime):
        """`realistic <= conservative` is what makes the pair readable as a bracket."""
        for estimate in (0.0001, 0.001, 0.01, 0.05):
            bounds = bounds_from_estimate(estimate, regime, volume,
                                          price=MEGA_CAP_PRICE, when=MODERN)
            assert bounds.realistic <= bounds.conservative + 1e-12, bounds

    def test_the_schedule_never_overrides_a_measurement(self):
        """`upper_bound` asserts the truth is BELOW the estimate. Honour that.

        A thin, genuinely cheap name can draw an estimate under the schedule's number.
        Quoting the schedule anyway would contradict a measurement we trust and would
        re-introduce the very over-charging this change exists to remove.
        """
        estimate = 0.004
        bounds = bounds_from_estimate(estimate, "upper_bound", 2e5,
                                      price=20.0, when=MODERN)
        assert liquid_name_spread(2e5, price=20.0, when=MODERN) > estimate
        assert bounds.realistic == pytest.approx(estimate)
        assert bounds.determined

    def test_measured_names_are_priced_exactly_as_before(self):
        """The change must be strictly additive: nothing already honest may move.

        If the measured universe re-priced, no result from iteration 1 would be
        comparable and the size of the bias could not be read off the difference.
        """
        estimate = 0.0126
        bounds = bounds_from_estimate(estimate, "measured", 2e6,
                                      price=20.0, when=MODERN)
        assert bounds.conservative == pytest.approx(estimate)
        assert bounds.realistic == pytest.approx(estimate)
        assert bounds.determined
        assert np.isnan(bounds.schedule), "the schedule must not touch a measurement"

    def test_an_unmeasurable_name_stays_untradable_under_both_bounds(self):
        """The schedule prices CHEAP names, not ABSENT ones.

        A name with no genuine trading days would otherwise be handed a mega-cap cost
        purely because its dollar volume looks large -- a free lunch in exactly the
        illiquid band the study is about, which is the original bug in a new costume.
        """
        bounds = bounds_from_estimate(float("nan"), "unmeasurable",
                                      MEGA_CAP_DOLLAR_VOLUME,
                                      price=MEGA_CAP_PRICE, when=MODERN)
        assert bounds.regime == "unmeasurable"
        assert not bounds.tradable
        assert np.isnan(bounds.conservative) and np.isnan(bounds.realistic)

    def test_a_name_without_dollar_volume_falls_back_to_the_dear_bound(self):
        """No liquidity figure means no schedule, and silence must cost, not discount."""
        estimate = 0.005
        bounds = bounds_from_estimate(estimate, "upper_bound", float("nan"),
                                      price=20.0, when=MODERN)
        assert bounds.tradable
        assert bounds.realistic == pytest.approx(bounds.conservative)
        assert bounds.realistic == pytest.approx(estimate)

    def test_both_bounds_respect_the_pre_decimalisation_tick(self):
        """In 1999 a $10 stock cost at least 62.5bps whatever any estimator said."""
        floor = minimum_quoted_spread(10.0, "1999-06-30")
        bounds = bounds_from_estimate(0.0002, "upper_bound", 5e9,
                                      price=10.0, when="1999-06-30")
        assert floor == pytest.approx(0.00625)
        assert bounds.realistic == pytest.approx(floor)
        assert bounds.conservative == pytest.approx(floor)

    def test_the_deleted_names_really_were_the_cheap_ones(self):
        """The bias, demonstrated end to end on ground truth.

        Two synthetic names, identical but for their true spread. The wide one resolves
        and iteration 1 kept it; the tight one does not resolve and iteration 1 deleted
        it. The rule therefore selected the EXPENSIVE name and discarded the CHEAP one --
        which is the mechanism, not merely a correlation observed in the panel.
        """
        wide = simulate_ohlc(750, 0.0300, daily_vol=0.02, seed=5)
        tight = simulate_ohlc(750, 0.0002, daily_vol=0.02, seed=5)

        wide_bounds = spread_cost_bounds(
            wide["open"], wide["high"], wide["low"], wide["close"],
            median_dollar_volume=MEGA_CAP_DOLLAR_VOLUME, price=MEGA_CAP_PRICE,
            when=MODERN,
        )
        tight_bounds = spread_cost_bounds(
            tight["open"], tight["high"], tight["low"], tight["close"],
            median_dollar_volume=MEGA_CAP_DOLLAR_VOLUME, price=MEGA_CAP_PRICE,
            when=MODERN,
        )

        assert wide_bounds.regime == "measured"      # kept by iteration 1
        assert tight_bounds.regime == "upper_bound"  # deleted by iteration 1
        assert tight_bounds.realistic < wide_bounds.realistic / 10.0
        assert tight_bounds.tradable and wide_bounds.tradable


class TestBracketVerdict:
    def test_the_three_honest_verdicts(self):
        assert bracket_verdict(True, True) == "real"
        assert bracket_verdict(False, True) == "undetermined"
        assert bracket_verdict(False, False) == "dead"

    def test_an_impossible_bracket_raises_rather_than_reporting_a_pass(self):
        """Passing (a) but failing (b) cannot happen, so it means the wiring is wrong.

        Silently returning "real" there would report a result built from two different
        runs as though it had cleared the conservative bound.
        """
        with pytest.raises(ValueError, match="cannot invert"):
            bracket_verdict(True, False)


def _schedule_with(volume: float, anchors: tuple[float, ...]) -> float:
    """`liquid_name_spread`'s interpolation against an arbitrary anchor set."""
    return float(np.exp(np.interp(
        np.log(volume), np.log(anchors), np.log(AGK_LIQUIDITY_ANCHOR_SPREAD))))


class TestFimSizeBucketMapping:
    """Where Frazzini-Israel-Moskowitz's published size buckets sit on our axis.

    the internal research log iteration 4 recorded a suspected third cost defect: the schedule
    charging ~3.6-4.4x FIM's small-cap all-in cost at $1M-$10M/day. Two explanations were
    open -- either FIM's "small cap" is more liquid than that band (in which case the
    schedule is fine), or the schedule is too dear there. These tests pin the answer so
    it cannot be re-litigated from memory. The measurement itself lives in
    `scripts/fim_size_bucket_control.py`; what is pinned here is the CONCLUSION.
    """

    def test_fim_small_cap_is_the_russell_2000_by_market_cap_rank(self):
        """FIM define the bucket by benchmark, so it is a rank, not a dollar volume.

        "for the US large cap is the Russell 1000 and small cap is below the Russell
        1000 in market cap, typically within the Russell 2000 universe" -- Table II note.
        """
        assert FIM_SMALL_CAP_RANK_RANGE == (1001, 3000)

    def test_the_ten_to_fifty_million_mapping_is_refuted(self):
        """Explanation (a) is FALSE, and this is the test that says so.

        `scripts/impact_positive_control.py` registered FIM's small cap as $10M-$50M/day.
        Measured on this tape over FIM's own sample window, the median constituent of
        market-cap ranks 1001-3000 trades $3.31M/day -- inside the $1M-$10M band the
        discrepancy was reported in, not above it. $10M-$50M/day is the bottom half of
        the RUSSELL 1000 here, whose median is $52.1M/day.
        """
        assert FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME < 1.0e7
        assert 1.0e6 < FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME < 5.0e6
        low, high = FIM_SMALL_CAP_DOLLAR_VOLUME_IQR
        assert low < FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME < high
        assert high < 1.0e7
        # ... and the large-cap bucket is an order of magnitude clear of it.
        assert FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME > 10.0 * FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME

    def test_the_rank_mapping_reproduces_two_published_index_volumes(self):
        """The out-of-sample validation, in unit-test form.

        FIM Table IX Panel A publishes the capitalisation-weighted average daily dollar
        volume of the S&P 500 and the Russell 2000. This repo calibrates nothing on that
        table, so reproducing both from a pure market-cap rank is what licenses reading
        their size buckets off this tape at all.
        """
        for published, measured in (
            (FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME, FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME_MEASURED),
            (FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME, FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME_MEASURED),
        ):
            assert abs(measured - published) / published < 0.15

    def test_the_published_index_volume_is_far_above_the_median_constituent(self):
        """Both numbers are true and they are not the same number.

        A capitalisation-weighted average over a right-skewed universe sits well above
        the median name. Conflating the two is how a bucket mismatch gets argued either
        way, so the gap is pinned rather than left to intuition -- and it is why the
        schedule passes the check at FIM's published Russell 2000 liquidity while still
        looking dear at the median constituent.
        """
        assert (FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME
                > 3.0 * FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME)

    def test_half_spread_stays_inside_fims_measured_all_in_cost(self):
        """Half-spread is a strict subset of what FIM measured, so it must be smaller.

        FIM Table II Panel A: median all-in one-way market impact of 5.54bps for large
        caps and 13.53bps for small caps, on $1.7tn of live executions. Charging more
        than that in spread ALONE, at the liquidity FIM themselves publish for those
        universes, would mean the schedule cannot be right.
        """
        for volume, all_in_bps in ((FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME, 5.54),
                                   (FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME, 13.53)):
            half_spread_bps = liquid_name_spread(volume, price=None, when=MODERN) / 2 * 1e4
            assert 0.0 < half_spread_bps < all_in_bps


class TestAnchorUniverseCorrection:
    """The anchors must be AGK's market-cap quintiles, not some other cross-section."""

    def test_the_superseded_anchors_are_rejected_by_their_own_source(self):
        """Control of the control: the mapping this replaced must FAIL.

        AGK Table 4 Panel C quintiles an explicitly UNSCREENED CRSP-TAQ universe by
        market capitalisation. The superseded anchors were the dollar-volume quintiles of
        the capacity study's own liquidity-SCREENED universe, which is a strictly more
        liquid set of names, so every spread level was pinned too far to the right.
        Priced at the liquidity AGK actually measured, they overstate.
        """
        errors_bps = [
            abs(_schedule_with(volume, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)
                - published) * 1e4
            for volume, published in zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
                                         AGK_LIQUIDITY_ANCHOR_SPREAD)
        ]
        assert max(errors_bps) > 25.0
        # ... and the shipped anchors reproduce the same table exactly.
        for volume, published in zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
                                     AGK_LIQUIDITY_ANCHOR_SPREAD):
            assert liquid_name_spread(volume, price=None, when=MODERN) == pytest.approx(
                published, rel=1e-9)

    def test_the_defect_ran_in_one_direction_and_worst_where_the_screen_bit(self):
        """A liquidity screen removes illiquid names, so it inflates the LOW quintiles.

        Q1-Q4 were all placed at more liquid names than AGK measured, by 3.4% at Q4
        rising to 583% at Q1 -- exactly the gradient a bottom-end screen produces. Q5 is
        the one that went the other way (the shipped anchor is 12.8% MORE liquid), which
        is why the correction is not a uniform discount and is not described as one.
        """
        ratios = [superseded / shipped for shipped, superseded
                  in zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
                         AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)]
        assert all(ratio > 1.0 for ratio in ratios[:4])
        assert ratios[0] > ratios[1] > ratios[2] > ratios[3]
        assert ratios[0] > 6.0
        assert ratios[4] < 1.0

    def test_the_dev_window_alternative_would_have_been_cheaper(self):
        """The arguable choice was made the conservative way, and that is testable.

        The anchors are measured over 1998-2020 to match AGK's own sample. Measuring over
        the study's 1998-2015 DEV window instead yields a strictly more liquid anchor
        set, i.e. a CHEAPER schedule everywhere it matters. Shipping that would have
        flattered every strategy, so it was not shipped.
        """
        for volume in (2e5, 1e6, 3.3e6, 1e7):
            dev = _schedule_with(volume, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_DEV_WINDOW)
            shipped = liquid_name_spread(volume, price=None, when=MODERN)
            assert dev < shipped

    def test_the_correction_only_moves_bound_b(self):
        """Bound (a) is not on this function's path, so it cannot move.

        The whole guarantee of the two-bound model rests on this: a result that passed
        the conservative bound before the recalibration must still pass it after.
        """
        estimate, regime = 0.0040, "upper_bound"
        for volume in (2e5, 1e6, 3.3e6, 1e7, 5e7, 1e9):
            for anchors in (AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
                            AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED):
                schedule = _schedule_with(volume, anchors)
                realistic = max(min(estimate, schedule), 0.0)
                assert realistic <= estimate
        bounds = bounds_from_estimate(estimate, regime, 3.3e6, price=20.0, when=MODERN)
        assert bounds.conservative == pytest.approx(estimate)
        assert bounds.realistic <= bounds.conservative

    def test_the_correction_is_a_reduction_below_thirteen_million_a_day(self):
        """Where it changes, it changes in the documented direction and size.

        Below ~$13.5M/day the schedule got cheaper -- up to 41bps per side at $200k/day.
        Between there and the top quintile's anchor it got at most 0.40bps per side
        DEARER, because Q5's measured liquidity sits above where it was placed. Both are
        what the measurement says; neither was chosen, and neither is hidden.
        """
        for volume in (2e5, 1e6, 3.3e6, 1e7):
            old = _schedule_with(volume, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)
            new = liquid_name_spread(volume, price=None, when=MODERN)
            assert new < old
        cheapening_bps = (
            _schedule_with(2e5, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)
            - liquid_name_spread(2e5, price=None, when=MODERN)) / 2 * 1e4
        assert 35.0 < cheapening_bps < 45.0

        grid = np.exp(np.linspace(np.log(1e3), np.log(1e10), 3000))
        deltas_bps = np.array([
            (liquid_name_spread(v, price=None, when=MODERN)
             - _schedule_with(v, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)) / 2 * 1e4
            for v in grid
        ])
        assert deltas_bps.max() < 0.5
        dearer = grid[deltas_bps > 1e-9]
        assert dearer.min() > 1.3e7
        assert dearer.max() < 6.4e7
