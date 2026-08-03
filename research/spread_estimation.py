"""Effective bid-ask spread estimated from daily high/low prices.

Sharadar SEP carries no quotes, so the spread a position actually pays has to be
inferred from the daily bar. This implements Corwin & Schultz (2012), "A Simple Way to
Estimate Bid-Ask Spreads from Daily High and Low Prices", *Journal of Finance* 67(2),
719-759.

The idea: a single day's high-low range reflects both volatility and the spread, but
volatility scales with the *square root of time* while the spread does not. Comparing
the range over one day against the range over two consecutive days therefore separates
the two. Everything below is that comparison.

Why this matters here: the capacity-curve study
(`research/medallion_style_alpha_search/capacity_curve_prereg.md` §6) prices trades in
bands down to $200k/day, where a flat 10bps assumption is meaningless -- real spreads
run 100-300bps. The cost model is the most likely thing to decide that study's outcome,
so it is validated against mega-caps of known spread before it is trusted on anything
else (`scripts/spread_positive_control.py`).

**The failure mode this module exists to avoid.** A name that did not trade prints
``high == low``. Fed naively through the algebra that yields a spread estimate of
*zero* -- a free lunch, manufactured exactly in the illiquid band the study is about,
and pointing exactly the way that would make a false result look good. Such days are
excluded, and a name with too few genuine two-day observations returns ``NaN``
(unknown, therefore untradeable) rather than a number.

**THE UNIVERSE BIAS, and the two-bound fix (2026-07-28).** ``spread_with_resolution``
labels a name ``upper_bound`` when its EDGE estimate sits at or below the volatility-
scaled noise floor. Every iteration-1 sleeve was then told to EXCLUDE those names. That
instruction was backwards. ``upper_bound`` does not mean "cost unknown"; it means "the
true spread is somewhere BELOW this number", i.e. **the name is CHEAP**. Excluding them
deleted the cheapest half of the tape and forced six strategies into the expensive tail,
which is why measured round-trip cost came back at 117-236bps against 15-256bps of gross
alpha (the internal research log, iteration 1).

The estimator's own authors say so in as many words. Ardia, Guidotti & Kroencke (2024)
§4: "following the proliferation of electronic trading between 2001-2005, we find that
the spreads for mid and large caps have become too small to be reliably estimated from a
monthly sample of daily data". Non-resolution in the liquid part of the cross-section is
the documented, expected behaviour of this estimator -- not a defect, and not a reason to
delete those names.

The fix is NOT to lower costs, which would manufacture a pass. It is to price every
``upper_bound`` name TWICE and report both numbers:

  (a) ``conservative`` -- charge the EDGE estimate itself. The true spread is below it,
      so this OVERSTATES cost. **A result that passes under (a) is real.**
  (b) ``realistic``   -- charge a documented liquid-name schedule keyed on median dollar
      volume (`liquid_name_spread`, sourced below), capped at the (a) figure because the
      measurement genuinely bounds the truth from above, and floored at the regulatory
      minimum tick. **A result that fails under (b) is dead.**

Between the two the result is UNDETERMINED and must be reported as such
(`bracket_verdict`). Construction guarantees ``realistic <= conservative`` for every
name, so the bracket can never invert, and a cheaper (b) can only ever move a verdict
from "dead" to "undetermined" -- never to "real".

**THE ANCHOR-UNIVERSE DEFECT, and what it was actually worth (2026-07-28).**
the internal research log iteration 4 recorded a suspicion that this schedule charged 3.6-4.4x
too much at $1M-$10M/day, judged against Frazzini, Israel & Moskowitz (2018) Table II
Panel A. Investigating it found ONE real defect and ONE framing error.

The real defect is in `AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME`. Ardia-Guidotti-Kroencke
Table 4 Panel C is keyed on MARKET-CAPITALISATION quintiles of the whole CRSP-TAQ
universe -- "all NYSE, AMEX, and NASDAQ stocks with CRSP share codes of 10 or 11 ...
No other data pre-processing is performed to maintain all the complexity of empirical
data and especially of the highly illiquid stocks" (AGK §3.1). The anchors placed those
five spread levels at the DOLLAR-VOLUME quintiles of the capacity study's own ELIGIBLE
universe, which carries a liquidity screen. Quintile k of a screened universe is not
quintile k of an unscreened one, so every spread level was pinned to a name more liquid
than the one it was measured on, and the schedule read too dear at every dollar volume
below ~$13M/day. Measured (see `scripts/fim_size_bucket_control.py`), the registered
anchors sat 1.03x-6.83x above the market-cap quintiles they claimed to represent.

The framing error is the 3.6-4.4x itself. It compared the RAW SCHEDULE against FIM,
but bound (b) is ``min(estimate, schedule)`` and in that band the EDGE estimate is
already the cheaper of the two 70-84% of the time. Re-pricing the 801,341 eligible
name-months in the study's four capacity bands, correcting the anchors moves the
realistic bound by only
**-1.5bps round trip at $200k-$1M, -3.5bps at $1M-$5M, -0.9bps at $5M-$25M and +0.2bps
above $25M** -- and the conservative bound not at all. Even a schedule cut all the way
to the minimum legal tick would only relieve a further 5-16bps per side. **The spread
schedule was never what made the illiquid bands expensive; the EDGE measurement is.**
"""

from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "AGK_ERA_MEDIAN_SPREAD",
    "AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME",
    "AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_DEV_WINDOW",
    "AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED",
    "AGK_LIQUIDITY_ANCHOR_SPREAD",
    "AGK_POOLED_MEDIAN_SPREAD",
    "AR_FLOOR_PER_UNIT_VOL",
    "FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME",
    "FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME_MEASURED",
    "FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME",
    "FIM_SMALL_CAP_DOLLAR_VOLUME_IQR",
    "FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME",
    "FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME_MEASURED",
    "FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME",
    "FIM_SMALL_CAP_RANK_RANGE",
    "SpreadBounds",
    "TICK_REGIMES",
    "abdi_ranaldo_spread",
    "bracket_verdict",
    "bounds_from_estimate",
    "corwin_schultz_spread",
    "edge_spread",
    "era_multiplier",
    "liquid_name_spread",
    "minimum_quoted_spread",
    "resolution_floor",
    "rolling_spread_estimate",
    "spread_cost_bounds",
    "spread_with_resolution",
    "tick_size",
]

# Calibrated by simulation (true spread = 0, 750 days, 50% of variance overnight,
# 8 seeds per point): the Abdi-Ranaldo estimate a name returns when its true spread is
# ZERO, expressed per unit of daily volatility. Measured at daily vols of 1/2/3/5/8%,
# the ratio was 1120/1115/1117/1116/1116 -- linear to within 0.5% across a factor of 8.
#
# This is NOT additive noise. It is RECTIFICATION BIAS: the estimator averages a
# signed quantity and floors the mean at zero, so pure sampling noise is rectified into
# a positive number whenever there is no real spread to measure. Where a genuine spread
# IS present the estimator is close to unbiased (true 50/100/200/400 bps recovered as
# 41/97/198/399 at 2% daily vol), which is why the correct treatment is a
# resolution THRESHOLD and not a subtraction. Subtracting it in quadrature was tried
# and made the error worse (29% mean relative error, zeroing genuine 50bps spreads).
AR_FLOOR_PER_UNIT_VOL = 0.1116

# The same calibration for EDGE, the estimator actually used: 11.80 bps per 1% of daily
# volatility, constant to two decimal places across daily vols of 1/2/3/5/8%.
#
# This floor is NOT a defect of any estimator -- it is an information limit of daily
# bars. A 2bps spread cannot be recovered from open/high/low/close when the stock moves
# 2% a day; the signal is simply not in the data. What separates EDGE is its accuracy
# ABOVE the floor: 1-2% error at true spreads of 100 and 300bps at every volatility
# tested, where Abdi-Ranaldo degrades to 18-22% error once daily vol reaches 4-6%.
EDGE_SIMULATED_FLOOR_PER_UNIT_VOL = 0.1180

# The floor MEASURED ON REAL DATA, which is what the cost model actually uses.
# Calibrated on 8 mega-caps (AAPL/MSFT/JPM/XOM/KO/PG/JNJ/WMT) across 2010-2025. Their
# true effective spreads sat at 1-3bps throughout while their volatility varied by more
# than a factor of two, so essentially the whole EDGE estimate for them IS floor. The
# ratio came out at 26.2 bps per 1% of daily volatility (25 excluding the 2020 COVID
# outlier at 45.9), with ~15% year-to-year dispersion.
#
# It is 2.2x the idealised simulation because a real tape has volatility clustering and
# a U-shaped intraday volatility profile, neither of which a constant-volatility random
# walk reproduces. Calibrating against securities of KNOWN spread rather than against a
# simulation is the difference between a cost model that survives contact with the data
# and one that does not -- the simulated constant would have classified AAPL's 56bps
# estimate as a genuine measurement.
EDGE_FLOOR_PER_UNIT_VOL = 0.2620

# How far above the floor an estimate must sit to count as a measurement rather than an
# upper bound. The floor is NOT additive: synthetic runs show a true 100bps spread
# recovered as 98.2bps against a 71bps floor (ratio 1.4), i.e. accuracy returns as soon
# as a real spread is comparable to the floor, whereas a true 20bps against the same
# floor came back as 76.6 and is pure noise. 1.5 sits just above the point where
# accuracy was demonstrated.
RESOLUTION_MULTIPLE = 1.5

# (3 - 2*sqrt(2)), the constant arising from the two-day/one-day range comparison
# (Corwin & Schultz 2012, eq. 14).
_K = 3.0 - 2.0 * np.sqrt(2.0)

# A name needs at least this many usable consecutive-day pairs for an estimate to mean
# anything. Below it the estimator is dominated by noise, and a noisy cost estimate in
# the illiquid band is worse than an honest refusal to quote one.
MIN_VALID_PAIRS = 20


def _pairwise_alpha(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Per-pair alpha (eq. 18), with the paper's overnight-gap adjustment applied.

    Returns an array of length ``len(high) - 1``; entries are NaN where the pair is
    unusable.
    """
    h_t, h_next = high[:-1].astype(float).copy(), high[1:].astype(float).copy()
    l_t, l_next = low[:-1].astype(float).copy(), low[1:].astype(float).copy()

    # Overnight gap adjustment (Corwin & Schultz §II.C). A price jump between the
    # close and the next open widens the two-day range without widening the spread,
    # so an unadjusted estimator reads gaps as trading costs. Shift the second day
    # back onto the first day's range whenever the two do not overlap.
    gap_up = l_next > h_t
    gap = np.where(gap_up, l_next - h_t, 0.0)
    h_next -= gap
    l_next -= gap

    gap_down = h_next < l_t
    gap = np.where(gap_down, l_t - h_next, 0.0)
    h_next += gap
    l_next += gap

    with np.errstate(divide="ignore", invalid="ignore"):
        # beta: the sum of two single-day squared log ranges (eq. 15).
        log_range_t = np.log(h_t / l_t)
        log_range_next = np.log(h_next / l_next)
        beta = log_range_t**2 + log_range_next**2

        # gamma: the squared log range across both days combined (eq. 16).
        two_day_high = np.maximum(h_t, h_next)
        two_day_low = np.minimum(l_t, l_next)
        gamma = np.log(two_day_high / two_day_low) ** 2

        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)

    return alpha


def _usable_pairs(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Boolean mask over consecutive-day pairs that carry real information.

    A pair is usable only if BOTH days have strictly positive prices and a strictly
    positive range. ``high == low`` means the name did not trade that day; folding it
    in as a zero-spread observation is the free-lunch bug this module is built to
    prevent, so it is excluded rather than counted.
    """
    positive = (high > 0.0) & (low > 0.0)
    traded = high > low
    valid_day = positive & traded & np.isfinite(high) & np.isfinite(low)
    return valid_day[:-1] & valid_day[1:]


def corwin_schultz_spread(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    min_pairs: int = MIN_VALID_PAIRS,
) -> float:
    """Mean proportional effective spread over the sample, or NaN if unmeasurable.

    Args:
        high: Daily high prices, chronologically ordered.
        low: Daily low prices, chronologically ordered and index-aligned to ``high``.
        min_pairs: Minimum usable consecutive-day pairs required to return a number.

    Returns:
        The proportional spread (0.01 == 100bps == 1% of price), or ``float('nan')``
        when too few days carry real trading. **NaN means untradeable, not free.**
    """
    high_values = np.asarray(high, dtype=float).ravel()
    low_values = np.asarray(low, dtype=float).ravel()
    if high_values.shape != low_values.shape:
        raise ValueError("high and low must be the same length")
    if high_values.size < 2:
        return float("nan")

    usable = _usable_pairs(high_values, low_values)
    if int(usable.sum()) < min_pairs:
        return float("nan")

    alpha = _pairwise_alpha(high_values, low_values)[usable]
    alpha = alpha[np.isfinite(alpha)]
    if alpha.size < min_pairs:
        return float("nan")

    # Spread from alpha (eq. 14).
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    # Corwin & Schultz set negative two-day estimates to zero BEFORE averaging (§II.B).
    # Averaging the raw values instead lets estimation noise cancel real cost and
    # biases the result downward -- i.e. it understates what trading costs, which is
    # the direction that would flatter a strategy.
    spread = np.maximum(spread, 0.0)

    return float(np.mean(spread))


def abdi_ranaldo_spread(
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    min_pairs: int = MIN_VALID_PAIRS,
) -> float:
    """Close-High-Low spread estimator, Abdi & Ranaldo (2017).

    "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices",
    *Review of Financial Studies* 30(12), 4437-4480.

    Where Corwin-Schultz compares one-day and two-day *ranges*, this compares the close
    against the mid-range ``(log high + log low) / 2``. The mid-range is a low-noise
    proxy for the efficient price, so the covariance between consecutive
    close-minus-midrange deviations isolates bid-ask bounce:

        S^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})]

    The expectation is taken BEFORE the square root (as the paper specifies): squaring
    each pair first would rectify noise into signal and bias the estimate upward, which
    is the failure mode that makes Corwin-Schultz unusable on liquid names here.

    Adopted after Corwin-Schultz failed the registered positive control
    (`scripts/spread_positive_control.py`), overstating mega-cap spreads by 20-40x.

    Returns:
        The proportional spread, or NaN when too few days carry real trading.
        **NaN means untradeable, not free.**
    """
    high_values = np.asarray(high, dtype=float).ravel()
    low_values = np.asarray(low, dtype=float).ravel()
    close_values = np.asarray(close, dtype=float).ravel()
    if not (high_values.shape == low_values.shape == close_values.shape):
        raise ValueError("high, low and close must be the same length")
    if high_values.size < 2:
        return float("nan")

    # Same exclusion as Corwin-Schultz: high == low means the name did not trade, and
    # counting it as a zero-spread observation manufactures a free lunch in exactly the
    # illiquid band this study is about.
    usable_day = (
        (high_values > 0.0)
        & (low_values > 0.0)
        & (close_values > 0.0)
        & (high_values > low_values)
        & np.isfinite(high_values)
        & np.isfinite(low_values)
        & np.isfinite(close_values)
    )
    usable = usable_day[:-1] & usable_day[1:]
    if int(usable.sum()) < min_pairs:
        return float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close_values)
        mid_range = (np.log(high_values) + np.log(low_values)) / 2.0
        terms = 4.0 * (log_close[:-1] - mid_range[:-1]) * (log_close[:-1]
                                                           - mid_range[1:])

    terms = terms[usable]
    terms = terms[np.isfinite(terms)]
    if terms.size < min_pairs:
        return float("nan")

    mean_squared_spread = float(np.mean(terms))
    # A negative mean is a genuinely possible sampling outcome when the true spread is
    # near zero. The paper floors it: the spread cannot be imaginary.
    return float(np.sqrt(max(mean_squared_spread, 0.0)))


def edge_spread(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    signed: bool = False,
) -> float:
    """Efficient Discrete Generalized Estimator (EDGE) of the effective spread.

    Ardia, Guidotti & Kroencke (2024), "Efficient Estimation of Bid-Ask Spreads from
    Open, High, Low, and Close Prices", *Journal of Financial Economics* 161, 103916.
    Implementation follows the authors' reference code at
    https://github.com/eguidotti/bidask.

    Adopted after Corwin-Schultz and Abdi-Ranaldo both failed the registered positive
    control (prereg erratum 1). The property that makes EDGE different in kind, rather
    than merely better tuned, is that it estimates the *probabilities* ``po`` and ``pc``
    that the open (or previous close) coincides with the period's high or low, and
    divides by them. Those probabilities fall as a name trades less, which inflates the
    estimate to compensate -- so a thin tape self-corrects instead of diluting the
    estimate toward zero. That is precisely the mechanism the other two estimators
    lacked, and it is why they collapsed toward a volatility-driven noise floor.

    The estimator targets the root-mean-square effective spread over the sample and
    needs at least three observations.

    Args:
        open_: Open prices, chronological.
        high: High prices, chronological.
        low: Low prices, chronological.
        close: Close prices, chronological.
        signed: Return the signed root. The squared-spread estimate can come out
            negative in small samples; the unsigned default takes its absolute value,
            while ``signed=True`` preserves the sign for diagnostics.

    Returns:
        The proportional spread (0.01 == 1% == 100bps), or NaN when unmeasurable.
    """
    o_arr = np.asarray(open_, dtype=float).ravel()
    h_arr = np.asarray(high, dtype=float).ravel()
    l_arr = np.asarray(low, dtype=float).ravel()
    c_arr = np.asarray(close, dtype=float).ravel()
    if not (o_arr.shape == h_arr.shape == l_arr.shape == c_arr.shape):
        raise ValueError("open, high, low and close must be the same length")
    if o_arr.size < 3:
        return float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        # Non-positive prices would produce -inf under the log and silently poison
        # every downstream mean, so they are marked missing rather than logged.
        def _log(values: np.ndarray) -> np.ndarray:
            return np.log(np.where(values > 0.0, values, np.nan))

        o_log, h_log, l_log, c_log = (_log(o_arr), _log(h_arr),
                                      _log(l_arr), _log(c_arr))
        m_log = (h_log + l_log) / 2.0

        h1, l1, c1, m1 = h_log[:-1], l_log[:-1], c_log[:-1], m_log[:-1]
        o_log, h_log, l_log, c_log, m_log = (o_log[1:], h_log[1:], l_log[1:],
                                             c_log[1:], m_log[1:])

        r1 = m_log - o_log
        r2 = o_log - m1
        r3 = m_log - c1
        r4 = c1 - m1
        r5 = o_log - c1

        # tau marks periods carrying information: either the bar had a genuine range,
        # or it moved away from the previous close. A bar that is completely flat AND
        # equal to the prior close tells us nothing and is weighted out here rather
        # than being counted as evidence of a zero spread.
        tau = np.where(np.isnan(h_log) | np.isnan(l_log) | np.isnan(c1),
                       np.nan, (h_log != l_log) | (l_log != c1))
        po1 = tau * np.where(np.isnan(o_log) | np.isnan(h_log), np.nan,
                             o_log != h_log)
        po2 = tau * np.where(np.isnan(o_log) | np.isnan(l_log), np.nan,
                             o_log != l_log)
        pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
        pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob_tau = np.nanmean(tau)
            prob_open = np.nanmean(po1) + np.nanmean(po2)
            prob_close = np.nanmean(pc1) + np.nanmean(pc2)

            if np.nansum(tau) < 2 or prob_open == 0 or prob_close == 0:
                return float("nan")

            d1 = r1 - np.nanmean(r1) / prob_tau * tau
            d3 = r3 - np.nanmean(r3) / prob_tau * tau
            d5 = r5 - np.nanmean(r5) / prob_tau * tau

            # Two moment conditions for the squared spread, each unbiased on its own.
            x1 = -4.0 / prob_open * d1 * r2 + -4.0 / prob_close * d3 * r4
            x2 = -4.0 / prob_open * d1 * r5 + -4.0 / prob_close * d5 * r4

            e1, e2 = np.nanmean(x1), np.nanmean(x2)
            v1 = np.nanmean(x1**2) - e1**2
            v2 = np.nanmean(x2**2) - e2**2

    # GMM-optimal combination: weight each moment by the OTHER's variance, so the
    # noisier one contributes less. Fall back to an equal weighting if the total
    # variance is degenerate.
    total_variance = v1 + v2
    squared_spread = ((v2 * e1 + v1 * e2) / total_variance
                      if total_variance > 0 else (e1 + e2) / 2.0)

    if not np.isfinite(squared_spread):
        return float("nan")

    spread = float(np.sqrt(abs(squared_spread)))
    if signed:
        spread *= float(np.sign(squared_spread))
    return spread


def resolution_floor(
    daily_volatility: float,
    per_unit_vol: float = EDGE_FLOOR_PER_UNIT_VOL,
) -> float:
    """The spread a name would report if its true spread were exactly zero.

    Below roughly twice this value an estimate is rectified noise, not a measurement.
    It scales linearly with volatility, so a volatile name needs a wider genuine spread
    before it can be measured at all -- which is why illiquid, volatile names are the
    hardest case and must be reported honestly rather than costed off a noisy number.
    """
    if not np.isfinite(daily_volatility) or daily_volatility <= 0.0:
        return float("nan")
    return per_unit_vol * float(daily_volatility)


def spread_with_resolution(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    min_pairs: int = MIN_VALID_PAIRS,
) -> tuple[float, str]:
    """Spread estimate plus an explicit statement of what it means.

    Returns ``(value, regime)`` where regime is one of:

    ``"measured"``
        The estimate comfortably exceeds the noise floor and is a genuine measurement
        (simulation: within ~5% at 2% daily vol, degrading to ~20% at 6%).
    ``"upper_bound"``
        The estimate sits at or near the floor. The true spread is *somewhere below*
        the returned value; the number is an upper bound, not a measurement. Costing a
        trade at this value OVERSTATES cost, which for liquid names is the direction
        that would flatter an illiquid-band strategy by comparison -- so callers must
        substitute a liquid-name cost schedule here rather than use this figure.
    ``"unmeasurable"``
        Too few genuine trading days. The name is untradeable, not free.

    The regime is returned rather than folded into the number because the capacity
    study's validity depends on which bands are measured and which are merely bounded,
    and a caller that silently treats an upper bound as a measurement would reintroduce
    exactly the bias the positive control was built to catch.
    """
    estimate = edge_spread(open_, high, low, close)
    if not np.isfinite(estimate):
        return float("nan"), "unmeasurable"

    # A genuine trading history is still required: EDGE tolerates flat bars via its tau
    # weighting, but a name with almost no real trading days has no honest cost at all.
    high_values = np.asarray(high, dtype=float).ravel()
    low_values = np.asarray(low, dtype=float).ravel()
    traded = (high_values > low_values) & (low_values > 0.0)
    if int(traded.sum()) < min_pairs:
        return float("nan"), "unmeasurable"

    close_values = np.asarray(close, dtype=float).ravel()
    close_values = close_values[np.isfinite(close_values) & (close_values > 0.0)]
    if close_values.size < 2:
        return float("nan"), "unmeasurable"

    returns = np.diff(np.log(close_values))
    daily_volatility = float(np.std(returns, ddof=1)) if returns.size > 1 else np.nan
    floor = resolution_floor(daily_volatility)
    if not np.isfinite(floor):
        return estimate, "upper_bound"

    if estimate > RESOLUTION_MULTIPLE * floor:
        return estimate, "measured"
    # The estimate itself IS the upper bound -- the true spread lies somewhere below it.
    # Raising it to the floor would manufacture cost out of an absence of information,
    # and does so hardest for volatile names, which is exactly the population the
    # capacity study is about.
    return estimate, "upper_bound"


def rolling_spread_estimate(
    high: pd.Series,
    low: pd.Series,
    window: int = 63,
    min_pairs: int = MIN_VALID_PAIRS,
) -> pd.Series:
    """Trailing-window spread estimate, aligned so it is point-in-time safe.

    The value at index ``t`` uses only bars up to and including ``t``, so a backtest
    reading it at ``t`` to price a trade is not using tomorrow's tape. The first
    ``window - 1`` entries are NaN.
    """
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")

    high_values = np.asarray(high, dtype=float)
    low_values = np.asarray(low, dtype=float)
    out = np.full(len(high_values), np.nan)

    for end in range(window, len(high_values) + 1):
        start = end - window
        out[end - 1] = corwin_schultz_spread(
            high_values[start:end], low_values[start:end], min_pairs=min_pairs
        )

    index = high.index if isinstance(high, pd.Series) else None
    return pd.Series(out, index=index, name="spread_estimate")


# ---------------------------------------------------------------------------
# The liquid-name cost schedule -- bound (b), the REALISTIC bound.
# ---------------------------------------------------------------------------
#
# SOURCE. Ardia, D., Guidotti, E. & Kroencke, T. A. (2024), "Efficient estimation of
# bid-ask spreads from open, high, low, and close prices", *Journal of Financial
# Economics* 161, 103916, **Table 4**. The table reports the MEDIAN TAQ EFFECTIVE SPREAD
# per group over the CRSP-TAQ merged sample 1993-2020 (N = 1,626,448 stock-months).
# TAQ effective spreads are the transaction-level ground truth: 2*|price - midpoint|,
# computed by the WRDS implementation of Holden & Jacobsen (2014).
#
# This is the right source rather than a convenient one. It is the ground-truth benchmark
# against which EDGE -- the estimator this module already uses -- was validated in the
# paper the estimator comes from, so bound (b) is calibrated on the same yardstick that
# certified bound (a), in the region where bound (a) stops resolving.
#
# Table 4 Panel C, median effective spread by market-capitalisation quintile:
#     Q1 (smallest) 3.14%   Q2 2.09%   Q3 1.08%   Q4 0.30%   Q5 (largest) 0.09%
AGK_LIQUIDITY_ANCHOR_SPREAD = (0.0314, 0.0209, 0.0108, 0.0030, 0.0009)

# The source table is keyed on market capitalisation; our panel is keyed on median daily
# dollar volume, which is what actually determines what a trade pays and what the study
# bands on. So each quintile's spread has to be placed at the dollar volume that
# quintile ACTUALLY TRADES AT -- and it must be measured on a universe built the way AGK
# built theirs, not on some other cross-section that happens to be to hand.
#
# CORRECTED 2026-07-28. The previous anchors -- (1.5385e5, 7.3255e5, 2.758144e6,
# 9.557127e6, 5.504659e7) -- were the DOLLAR-VOLUME quintiles of the capacity study's own
# ELIGIBLE universe, which carries a liquidity screen. AGK's quintiles are
# MARKET-CAPITALISATION quintiles of an explicitly UNSCREENED universe (§3.1: "all NYSE,
# AMEX, and NASDAQ stocks with CRSP share codes of 10 or 11 ... No other data
# pre-processing is performed to maintain all the complexity of empirical data and
# especially of the highly illiquid stocks with only a few observations per month").
# Quintile k of a screened universe is a MORE LIQUID set of names than quintile k of an
# unscreened one, so every spread level was pinned to a name that trades more than the
# one it was measured on, and the schedule read too dear everywhere below ~$13M/day --
# by 1.03x at Q4 rising to 6.83x at Q1.
#
# The values below are the median of each monthly quintile's median daily dollar volume,
# over the closest reproduction of AGK's universe this tape supports: Sharadar
# domestic-common-stock issues on NYSE / NASDAQ / AMEX, month-end market capitalisation
# from DAILY, monthly medians of close x volume from SEP, 1998-2020, 1,364,189
# (name, month) cells at a median 4,898 names per month -- against AGK's 1,626,448 cells
# over 1993-2020, ~4,840 names per month.
#
# TWO CHOICES HERE WERE MADE THE CONSERVATIVE WAY, and both are stated because both are
# arguable. (i) The window is 1998-2020, matching AGK's, rather than the study's own
# 1998-2015 DEV window; the DEV window yields (1.5669e4, 1.59981e5, 1.369061e6,
# 7.127419e6, 5.084771e7), which is CHEAPER in every band the study trades and was
# therefore rejected as the flattering choice. (ii) AGK's sample also covers 1993-1997,
# which this tape does not reach; those years carried lower nominal dollar volume, so
# including them would lower these anchors further. Both omissions leave the schedule
# DEARER than a perfect replication, which is the only direction a cost model may err in.
#
# NOTE what this mapping is NOT: it is not fitted to any return, any strategy result or
# any cost outcome. It is a liquidity measurement, and it is pinned by
# `scripts/fim_size_bucket_control.py` and `scripts/spread_positive_control.py` check E.
AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME = (2.2511e4, 2.07734e5, 1.760933e6,
                                      9.246530e6, 6.309893e7)

# The same measurement on the DEV window alone. Kept so the sensitivity above is a
# number in the code rather than a claim in a comment, and so the control can show that
# shipping it would have been cheaper.
AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_DEV_WINDOW = (1.5669e4, 1.59981e5, 1.369061e6,
                                                 7.127419e6, 5.084771e7)

# The anchors as they stood before the correction, kept ONLY so the positive control can
# demonstrate that they FAIL the check the new ones pass. A gate that only the shipped
# constants can fail is not a gate.
AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED = (1.5385e5, 7.3255e5, 2.758144e6,
                                                 9.557127e6, 5.504659e7)

# ---------------------------------------------------------------------------
# Where Frazzini-Israel-Moskowitz's published size buckets sit on this axis.
# ---------------------------------------------------------------------------
#
# The capacity study's cost numbers are checked against Frazzini, Israel & Moskowitz
# (2018), "Trading Costs" -- $1.7tn of live institutional executions, Aug 1998 - Jun 2016,
# average trade 0.9% of daily volume. That check is only meaningful if their size buckets
# are placed at the right liquidity, and until 2026-07-28 they were not: the impact
# control mapped "FIM small cap" onto $10M-$50M/day. The constants below replace that
# assumption with what the paper states and what this tape measures.
#
# WHAT THE PAPER STATES (Table II note, and §II.A almost verbatim again): "The distinction
# between large and small cap is based on the portfolio's benchmark (e.g. for the US large
# cap is the Russell 1000 and small cap is below the Russell 1000 in market cap, typically
# within the Russell 2000 universe)." So the bucket is a MARKET-CAP RANK, not a dollar
# volume. The paper does NOT state a dollar volume for the trades in that bucket, and no
# number here pretends otherwise -- what it does state, in Table IX Panel A, is the
# portfolio-weighted average daily volume of the two indices it prices, and those two
# published figures are what pin the mapping below.
FIM_SMALL_CAP_RANK_RANGE = (1001, 3000)

# FIM Table IX Panel A, "Average daily volume (Million USD)", as published. These are
# capitalisation-weighted averages over the index, so they sit far above the MEDIAN
# constituent -- which is exactly why they are useful: they are an independent, published
# statement of where these two universes sit, on a table this repo calibrates nothing on.
FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME = 6.6283e8    # S&P 500
FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME = 1.476e7     # Russell 2000

# The same two quantities MEASURED on this tape over 2011-2016 (Table IX is calibrated on
# 2016 figures): top-500-by-market-cap and rank 1001-3000, capitalisation-weighted median
# daily dollar volume. $694.4M against a published $662.8M (+4.8%) and $14.54M against a
# published $14.76M (-1.5%). Two published numbers reproduced out of sample is what
# licenses reading FIM's size buckets off this tape at all.
FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME_MEASURED = 6.943877e8
FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME_MEASURED = 1.453985e7

# And the number the whole question turned on: the MEDIAN constituent of FIM's small-cap
# universe. Market-cap ranks 1001-3000 of the domestic-common-stock cross-section over
# 1998-2015 (FIM's own sample window), 410,000 (name, month) cells -- median daily dollar
# volume $3.31M, interquartile range $1.16M-$8.14M.
#
# **This refutes the assumption it replaces.** $3.31M/day is not $10M-$50M/day; the band
# the impact control called "FIM small cap" is closer to the BOTTOM HALF OF THE RUSSELL
# 1000 on this tape (rank 1-1000 has a median of $52.1M/day and a first quartile of
# $23.4M/day). FIM's small-cap bucket really does live in $1M-$10M/day, so the
# 3.6-4.4x discrepancy recorded in internal research log iteration 4 could NOT be explained away
# as a bucket mismatch, and was not.
FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME = 3.312622e6
FIM_SMALL_CAP_DOLLAR_VOLUME_IQR = (1.157248e6, 8.139061e6)
FIM_LARGE_CAP_MEDIAN_DOLLAR_VOLUME = 5.214041e7

# Table 4 Panel B, median effective spread by sub-period (all stocks). Spreads collapsed
# by an order of magnitude across the sample; a single pooled number applied to 1999
# would understate cost enormously.
AGK_ERA_MEDIAN_SPREAD: tuple[tuple[int, int, float], ...] = (
    (1993, 1996, 0.0249),
    (1997, 2000, 0.0168),
    (2001, 2002, 0.0125),
    (2003, 2007, 0.0031),
    (2008, 2011, 0.0025),
    (2012, 2015, 0.0018),
    (2016, 2020, 0.0018),
)

# Table 3, median of the same TAQ effective-spread benchmark pooled over 1993-2020.
# The era factor is each period's median divided by this.
AGK_POOLED_MEDIAN_SPREAD = 0.0076

# Minimum quoting increment for a US NMS stock priced at or above $1.00, by date.
# NYSE moved from eighths to sixteenths on 1997-06-24; decimal pricing was phased in from
# 2000-08-28 and completed across all NYSE and Nasdaq issues on 2001-04-09; SEC Reg NMS
# Rule 612 (17 CFR 242.612, effective 2005) then codified the $0.01 minimum.
#
# This matters more than it looks. A quoted spread cannot be narrower than one tick, so
# in 1999 a $20 stock could not trade inside 31bps however liquid it was. Applying a
# modern liquid-name number to the pre-decimalisation half of the DEV window would
# understate cost by an order of magnitude -- exactly the error this whole exercise
# exists to stop, pointing the other way.
#
# The 2024 amendments to Rule 612 (Release 34-101070) introduce a $0.005 increment for
# tick-constrained stocks from November 2025. It is deliberately NOT modelled: a smaller
# tick can only make trading cheaper, so ignoring it keeps the floor conservative.
TICK_REGIMES: tuple[tuple[_dt.date, float], ...] = (
    (_dt.date(1, 1, 1), 0.125),
    (_dt.date(1997, 6, 24), 0.0625),
    (_dt.date(2001, 4, 9), 0.01),
)


def _as_date(when: object) -> _dt.date | None:
    """Coerce whatever a caller passes as a date, or None if there isn't one."""
    if when is None:
        return None
    if isinstance(when, _dt.datetime):
        return when.date()
    if isinstance(when, _dt.date):
        return when
    try:
        stamp = pd.Timestamp(when)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.date()


def tick_size(when: object = None) -> float:
    """Minimum quoting increment in dollars in force on ``when``.

    ``None`` means "assume the decimal era", the cheapest regime, so an undated caller is
    never charged a pre-2001 tick it may not deserve.
    """
    date = _as_date(when)
    if date is None:
        return TICK_REGIMES[-1][1]
    size = TICK_REGIMES[0][1]
    for start, value in TICK_REGIMES:
        if date >= start:
            size = value
    return size


def minimum_quoted_spread(price: float, when: object = None) -> float:
    """One tick expressed as a proportion of price -- the narrowest legal quote.

    Returns NaN for a non-positive or missing price, which callers must treat as "no
    floor known", never as "no floor".
    """
    try:
        value = float(price)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return tick_size(when) / value


def era_multiplier(when: object = None) -> float:
    """How much wider spreads were in ``when``'s era than over 1993-2020 as a whole.

    Ardia-Guidotti-Kroencke Table 4 Panel B divided by the Table 3 pooled median, and
    then **floored at 1.0**. The floor is deliberate and it is the conservative choice:
    the post-2003 compression in that table is concentrated in large caps, so passing the
    discount through uniformly would quietly cheapen modern SMALL-cap trading, which is
    the direction that flatters a strategy. Refusing a discount cannot flatter anything.

    Dates before the table's coverage take the earliest (widest) factor, since spreads
    were wider still; dates after it take 1.0.
    """
    date = _as_date(when)
    if date is None:
        return 1.0
    year = date.year
    if year < AGK_ERA_MEDIAN_SPREAD[0][0]:
        median = AGK_ERA_MEDIAN_SPREAD[0][2]
    else:
        median = AGK_POOLED_MEDIAN_SPREAD
        for start, end, value in AGK_ERA_MEDIAN_SPREAD:
            if start <= year <= end:
                median = value
                break
    return max(1.0, median / AGK_POOLED_MEDIAN_SPREAD)


def liquid_name_spread(
    median_dollar_volume: float,
    price: float | None = None,
    when: object = None,
) -> float:
    """Documented effective spread for a name the estimator cannot resolve.

    Log-linear interpolation of the Ardia-Guidotti-Kroencke Table 4 Panel C quintile
    medians against ``AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME``, scaled by ``era_multiplier``
    and floored at ``minimum_quoted_spread``.

    **The schedule is CLAMPED at both ends and never extrapolated.** A $5bn/day name is
    charged the top quintile's 9bps rather than whatever a power law would extrapolate to
    below it. That is the honest reading of a table of quintile medians -- the source says
    nothing about what happens past its own support -- and it is also the conservative
    one, because extrapolation would make mega-caps cheaper still.

    The anchors were corrected on 2026-07-28 (see the block above
    ``AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME``). The correction is a REDUCTION of up to
    41bps per side below $13.5M/day and an increase of at most 0.40bps per side between
    $13.5M and $63.1M/day, where the top quintile's anchor moved right. Both directions
    are what the measurement says; neither was chosen. Only bound (b) can move, because
    this function is not on bound (a)'s path at all -- so no result that passed the
    conservative bound can fail because of it.

    Args:
        median_dollar_volume: Trailing median daily dollar volume, in dollars.
        price: Share price, used only for the minimum-tick floor. Optional.
        when: Date of the estimate, used for the era factor and the tick regime.

    Returns:
        The full proportional effective spread (0.0009 == 9bps), or NaN when dollar
        volume is missing or non-positive -- which means untradeable, not free.
    """
    try:
        volume = float(median_dollar_volume)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(volume) or volume <= 0.0:
        return float("nan")

    base = float(np.exp(np.interp(
        np.log(volume),
        np.log(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME),
        np.log(AGK_LIQUIDITY_ANCHOR_SPREAD),
    )))
    value = base * era_multiplier(when)

    floor = minimum_quoted_spread(price, when) if price is not None else float("nan")
    if np.isfinite(floor):
        value = max(value, floor)
    return value


@dataclass(frozen=True)
class SpreadBounds:
    """The two bracketing spread bounds for one name, plus how they were reached.

    ``conservative >= realistic`` holds by construction for every name, so a caller can
    always read the pair as a genuine bracket.
    """

    conservative: float
    realistic: float
    regime: str
    tradable: bool
    estimate: float
    schedule: float
    tick_floor: float

    @property
    def determined(self) -> bool:
        """True when the two bounds agree to within a basis point."""
        if not (np.isfinite(self.conservative) and np.isfinite(self.realistic)):
            return False
        return abs(self.conservative - self.realistic) < 1e-4


def bounds_from_estimate(
    estimate: float,
    regime: str,
    median_dollar_volume: float,
    price: float | None = None,
    when: object = None,
) -> SpreadBounds:
    """Bracket an already-computed EDGE estimate. See `spread_cost_bounds`.

    Split out so a caller holding a panel of pre-computed (estimate, regime) pairs --
    there are ~1.3 million of them -- can bracket it without re-running the estimator
    over every trailing window again.
    """
    tick_floor = minimum_quoted_spread(price, when) if price is not None else float("nan")
    floor = tick_floor if np.isfinite(tick_floor) else 0.0

    if regime == "unmeasurable" or not np.isfinite(estimate):
        # Too few genuine trading days. There is no honest cost, and inventing one from
        # the schedule would hand a free tradable universe to names that never traded.
        return SpreadBounds(
            conservative=float("nan"), realistic=float("nan"), regime="unmeasurable",
            tradable=False, estimate=float("nan"), schedule=float("nan"),
            tick_floor=tick_floor,
        )

    conservative = max(float(estimate), floor)

    if regime != "upper_bound":
        # A genuine measurement. Both bounds are the measurement; the schedule has no
        # business overriding data. Iteration-1's measured universe is therefore priced
        # identically to before, which is what makes this change strictly additive.
        return SpreadBounds(
            conservative=conservative, realistic=conservative, regime=regime,
            tradable=True, estimate=float(estimate), schedule=float("nan"),
            tick_floor=tick_floor,
        )

    schedule = liquid_name_spread(median_dollar_volume, price, when)
    if not np.isfinite(schedule):
        # No dollar volume, so the schedule cannot speak. Fall back to the conservative
        # bound for both, which prices the name exactly as iteration 1 would have.
        return SpreadBounds(
            conservative=conservative, realistic=conservative, regime=regime,
            tradable=True, estimate=float(estimate), schedule=float("nan"),
            tick_floor=tick_floor,
        )

    # min() with the estimate is not a safety belt, it is the definition of the regime:
    # `upper_bound` states that the true spread lies BELOW the estimate, so a schedule
    # that quoted more than the estimate would contradict a measurement we trust. max()
    # with the tick then refuses to price any name inside the narrowest legal quote.
    realistic = max(min(float(estimate), schedule), floor)
    return SpreadBounds(
        conservative=conservative, realistic=realistic, regime=regime,
        tradable=True, estimate=float(estimate), schedule=schedule,
        tick_floor=tick_floor,
    )


def spread_cost_bounds(
    open_: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray,
    low: pd.Series | np.ndarray,
    close: pd.Series | np.ndarray,
    median_dollar_volume: float,
    price: float | None = None,
    when: object = None,
    min_pairs: int = MIN_VALID_PAIRS,
) -> SpreadBounds:
    """Price one name under BOTH cost bounds. This is the API strategies should use.

    Replaces the iteration-1 pattern of calling `spread_with_resolution` and dropping
    every ``upper_bound`` name. Dropping them was the universe bias: it deleted the
    cheapest names in the market and left every sleeve trading the expensive tail.

    Returns a `SpreadBounds`. Report both numbers for every result. A result is REAL only
    if it survives ``conservative``; it is DEAD if it fails ``realistic``; anything in
    between is UNDETERMINED (`bracket_verdict`).
    """
    estimate, regime = spread_with_resolution(open_, high, low, close,
                                              min_pairs=min_pairs)
    if price is None:
        closes = np.asarray(close, dtype=float).ravel()
        closes = closes[np.isfinite(closes) & (closes > 0.0)]
        price = float(closes[-1]) if closes.size else None
    return bounds_from_estimate(estimate, regime, median_dollar_volume, price, when)


def bracket_verdict(passes_conservative: bool, passes_realistic: bool) -> str:
    """Turn a pair of pass/fail flags into the only three verdicts that are honest.

    ``"real"``          survives the bound that overstates cost.
    ``"dead"``          fails even the bound that is generous to it.
    ``"undetermined"``  in between: the cost model cannot separate the two, and the
                        result must be reported as undetermined rather than as a pass.

    Raises:
        ValueError: if the caller reports passing the conservative bound but failing the
            realistic one. That is arithmetically impossible when both come from
            `spread_cost_bounds`, so it means the two figures were computed from
            different runs -- a wiring bug that would otherwise be reported as a result.
    """
    if passes_conservative and not passes_realistic:
        raise ValueError(
            "passed the conservative bound but failed the realistic one; the bounds "
            "cannot invert, so these two numbers did not come from the same run"
        )
    if passes_conservative:
        return "real"
    if passes_realistic:
        return "undetermined"
    return "dead"
