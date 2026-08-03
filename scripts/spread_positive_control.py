"""Positive control for the capacity study's cost model.

Registered in `research/medallion_style_alpha_search/capacity_curve_prereg.md` §6 (and
amended by its 2026-07-27 erratum) as a gate on the whole study: a cost model that
cannot price a stock whose spread is known cannot be trusted to price one whose spread
is not. If this fails, no capacity-curve result may be reported.

Three checks, each able to fail independently:

  A. ACCURACY IN THE MEASURED REGIME (synthetic, ground truth known by construction).
     EDGE must recover true spreads of 100-400bps -- the range the study's
     illiquid bands actually live in -- to within 20% at realistic volatility and with
     realistic overnight gaps.

  B. HONEST DEGRADATION ON LIQUID NAMES (real data). Mega-caps trade at 1-3bps, far
     below the estimator's resolution. The estimator must therefore classify them as
     ``upper_bound``, NOT report a large number as though it were measured. This is the
     check the original Corwin-Schultz model failed: it reported ~40bps for AAPL as if
     that were a measurement, which would have overstated liquid-band costs ~20x and
     biased the study toward its own hypothesis.

  C. CROSS-SECTIONAL MONOTONICITY (real data, a genuine falsifiable prediction).
     Estimated spreads must rise as liquidity falls. If they do not, the estimator is
     not tracking trading costs on this dataset at all.

  D. THE LIQUID-NAME SCHEDULE IS RIGHT WHERE THE ESTIMATOR IS BLIND (real data).
     Added 2026-07-28 with the two-bound cost model. Check B establishes that the
     estimator correctly DECLINES to measure a mega-cap. That leaves the question B
     cannot answer: what should a mega-cap actually be charged? Bound (b) must put
     AAPL/MSFT/JPM/XOM/KO at 1-5bps per side over 2016-2026 -- their real, documented
     cost -- while bound (a) must still charge them the noise floor. If the schedule
     misses that window it is wrong, the whole two-bound model is void, and NO strategy
     may be re-run under it. This is the gate: it fails closed.

     It is the only check permitted to read bars after 2015-12-31, and it may because it
     touches no strategy, no signal and no forward return -- it measures the cost of five
     named securities whose true spreads are public knowledge.

  E. THE SCHEDULE AGAINST INDEPENDENTLY PUBLISHED COSTS AT STATED LIQUIDITY (arithmetic).
     Added 2026-07-28. Check D pins the schedule at ONE liquidity level -- mega-caps --
     and that is where it was least likely to be wrong. the internal research log iteration 4
     recorded a suspicion that it was 3.6-4.4x too dear four orders of magnitude further
     down, at $1M-$10M/day, and nothing in checks A-D could have caught that. E is the
     check that can.

     It uses two published sources and nothing this repo fitted:

       - Ardia, Guidotti & Kroencke (2024) JFE Table 4 Panel C, the median TAQ effective
         spread of each MARKET-CAP quintile of the unscreened CRSP-TAQ universe. The
         schedule must return those five numbers at the dollar volumes those five
         quintiles actually trade at (E1). This is an implementation check and is
         labelled as one -- the anchors are set to satisfy it. Its teeth are E2.
       - Frazzini, Israel & Moskowitz (2018) Table IX Panel A, the published average daily
         dollar volume of the S&P 500 ($662.83M) and the Russell 2000 ($14.76M), paired
         with Table II Panel A's measured all-in one-way market impact for large caps
         (5.54bps) and small caps (13.53bps). FIM's small cap IS the Russell 2000 -- the
         paper says so twice. Half-spread is a strict SUBSET of what FIM measured, so at
         each of those two liquidity levels our half-spread must sit strictly below their
         figure (E3). Nothing in this repo is calibrated on Table IX, so E3 is out of
         sample in the strict sense.

     E2 is the control of the control: the SUPERSEDED anchors must FAIL E1. E4 is a
     property check -- the schedule must stay inside its source's support and never
     extrapolate. E5 records, and does NOT gate, the residual: at the MEDIAN name of
     FIM's small-cap universe the schedule is still ~2.5x their all-in figure. That gap
     is real, its causes are named in the printout, and it is deliberately left standing
     rather than closed by a factor nobody measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.spread_estimation import (  # noqa: E402
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME,
    AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED,
    AGK_LIQUIDITY_ANCHOR_SPREAD,
    FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME,
    FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME,
    FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME,
    bounds_from_estimate,
    edge_spread,
    liquid_name_spread,
    resolution_floor,
    spread_with_resolution,
)

SEP = REPO / "_data" / "sharadar" / "SEP.csv"
MEGA_CAPS = ["AAPL", "MSFT", "JPM", "XOM", "KO"]
ACCURACY_TOLERANCE = 0.20

# The registered acceptance window for check D, in basis points PER SIDE. `round_trip_cost`
# charges `spread / 2` on each side of a trade, so this is half the schedule's full
# effective spread. Set from what these five names demonstrably cost: one cent on a
# $50-400 share is 0.25-2bps of quoted spread, and Frazzini, Israel & Moskowitz (2018)
# Table II Panel A measures a MEDIAN of 5.54bps of ALL-IN one-way cost -- spread plus
# market impact -- on $1.7tn of live US large-cap institutional executions at an average
# trade size of 0.9% of daily volume. A pure spread cost above 5bps per side for these
# names is therefore not credible, and below 1bps would be cheaper than the tape.
MEGA_CAP_MIN_BPS_PER_SIDE = 1.0
MEGA_CAP_MAX_BPS_PER_SIDE = 5.0


def simulate(n_days, true_spread, daily_vol, overnight_frac=0.5, ticks=100, seed=0):
    """Daily OHLC from a known spread, with realistic overnight gaps."""
    rng = np.random.default_rng(seed)
    intraday = daily_vol * np.sqrt(1.0 - overnight_frac)
    overnight = daily_vol * np.sqrt(overnight_frac)
    step = intraday / np.sqrt(ticks)
    opens, highs, lows, closes = [], [], [], []
    log_price = np.log(100.0)
    for _ in range(n_days):
        log_price += rng.normal(0.0, overnight)
        path = log_price + np.cumsum(rng.normal(0.0, step, ticks))
        side = rng.choice([-1.0, 1.0], ticks)
        observed = np.exp(path) * (1.0 + side * true_spread / 2.0)
        opens.append(observed[0])
        highs.append(observed.max())
        lows.append(observed.min())
        closes.append(observed[-1])
        log_price = path[-1]
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes})


def check_a_accuracy() -> bool:
    print("=" * 74)
    print("A. ACCURACY IN THE MEASURED REGIME (synthetic, ground truth known)")
    print("=" * 74)
    print(f"{'daily vol':>10} {'true bps':>9} {'estimate':>10} {'rel err':>9}  regime")
    errors, regimes_ok = [], True
    for vol in (0.02, 0.04, 0.06):
        for true_bps in (100, 200, 400):
            frames = [simulate(750, true_bps / 1e4, vol, seed=s)
                      for s in range(6)]
            estimates = [
                edge_spread(f["open"], f["high"], f["low"], f["close"]) * 1e4
                for f in frames
            ]
            estimate = float(np.mean(estimates))
            _, regime = spread_with_resolution(
                frames[0]["open"], frames[0]["high"], frames[0]["low"],
                frames[0]["close"]
            )
            relative = abs(estimate - true_bps) / true_bps
            errors.append(relative)
            # Only require the "measured" classification where the true spread is
            # comfortably above the floor; near the limit, "upper_bound" is the
            # correct and honest answer, not a failure.
            # Demand the "measured" classification only where the true spread is
            # comfortably above the real-world floor (26.2bps per 1% daily vol).
            # Near or below it, "upper_bound" is the correct answer, not a failure.
            floor_bps = 26.2 * vol * 100
            if true_bps > 1.5 * floor_bps and regime != "measured":
                regimes_ok = False
            print(f"{vol:>10.2f} {true_bps:>9} {estimate:>10.1f} {relative:>8.1%}  "
                  f"{regime}")
    worst = max(errors)
    passed = worst <= ACCURACY_TOLERANCE and regimes_ok
    print(f"\n  worst relative error {worst:.1%} (tolerance {ACCURACY_TOLERANCE:.0%}); "
          f"all classified 'measured': {regimes_ok}")
    print(f"  CHECK A: {'PASS' if passed else 'FAIL'}\n")
    return passed


def load_names(tickers: set[str], columns: list[str]) -> pd.DataFrame:
    frames = []
    reader = pd.read_csv(SEP, usecols=columns, chunksize=2_000_000)
    for chunk in reader:
        hit = chunk[chunk["ticker"].isin(tickers)]
        if not hit.empty:
            frames.append(hit)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def check_b_liquid_names() -> bool:
    print("=" * 74)
    print("B. HONEST DEGRADATION ON LIQUID NAMES (real data, 2016-2026)")
    print("=" * 74)
    data = load_names(set(MEGA_CAPS),
                      ["ticker", "date", "open", "high", "low", "close"])
    window = data[data["date"] >= "2016-01-01"]
    print(f"{'name':>6} {'estimate':>10} {'floor':>8} {'regime':>13}   verdict")
    all_bounded = True
    for ticker in MEGA_CAPS:
        series = window[window["ticker"] == ticker]
        value, regime = spread_with_resolution(
            series["open"], series["high"], series["low"], series["close"]
        )
        returns = np.diff(np.log(series["close"].to_numpy()))
        floor = resolution_floor(float(np.std(returns, ddof=1))) * 1e4
        ok = regime == "upper_bound"
        all_bounded &= ok
        verdict = "ok (below resolution)" if ok else "WRONGLY CLAIMS MEASUREMENT"
        print(f"{ticker:>6} {value * 1e4:>10.1f} {floor:>8.1f} {regime:>13}   "
              f"{verdict}")
    print("\n  Mega-caps truly trade at 1-3bps. The estimator must decline to claim a\n"
          "  measurement here rather than report its noise floor as a cost.")
    print(f"  CHECK B: {'PASS' if all_bounded else 'FAIL'}\n")
    return all_bounded


def check_c_monotonicity() -> bool:
    print("=" * 74)
    print("C. CROSS-SECTIONAL MONOTONICITY (real data, falsifiable prediction)")
    print("=" * 74)
    print("  Sampling the tape to build liquidity buckets...")
    columns = ["ticker", "date", "open", "high", "low", "close", "volume"]
    sample = pd.read_csv(SEP, usecols=columns, nrows=6_000_000)
    sample["date"] = pd.to_datetime(sample["date"])
    sample = sample[sample["date"] >= "2010-01-01"]
    # The raw SEP export is NOT in chronological order (dates run backwards within a
    # ticker). Every estimator here compares consecutive bars, so unsorted input is
    # silently meaningless rather than merely noisy.
    sample = sample.sort_values(["ticker", "date"]).reset_index(drop=True)
    sample["dollar_volume"] = sample["close"] * sample["volume"]

    stats = sample.groupby("ticker").agg(
        days=("close", "size"), dollar_volume=("dollar_volume", "median")
    )
    stats = stats[(stats["days"] >= 250) & (stats["dollar_volume"] > 0)]

    buckets = [
        ("$50k-$200k", 5e4, 2e5),
        ("$200k-$1M", 2e5, 1e6),
        ("$1M-$5M", 1e6, 5e6),
        ("$5M-$25M", 5e6, 2.5e7),
        ("$25M-$200M", 2.5e7, 2e8),
        (">$200M", 2e8, np.inf),
    ]
    print(f"\n{'band':>14} {'names':>7} {'median spread':>15} {'measured %':>12}")
    medians = []
    for label, low, high in buckets:
        names = stats[(stats["dollar_volume"] >= low)
                      & (stats["dollar_volume"] < high)].index
        names = list(names)[:60]
        values, measured = [], 0
        for ticker in names:
            series = sample[sample["ticker"] == ticker]
            value, regime = spread_with_resolution(
                series["open"], series["high"], series["low"], series["close"]
            )
            if np.isfinite(value):
                values.append(value * 1e4)
                measured += regime == "measured"
        if not values:
            print(f"{label:>14} {0:>7} {'-':>15} {'-':>12}")
            continue
        median = float(np.median(values))
        medians.append(median)
        print(f"{label:>14} {len(values):>7} {median:>14.0f}bps "
              f"{measured / len(values):>11.0%}")

    declining = all(a >= b for a, b in zip(medians, medians[1:]))
    print(f"\n  Spread falls monotonically as liquidity rises: {declining}")
    print(f"  CHECK C: {'PASS' if declining else 'FAIL'}\n")
    return declining


def check_d_liquid_schedule() -> bool:
    """Bound (b) must price the five mega-caps at 1-5bps per side; bound (a) must not."""
    print("=" * 74)
    print("D. LIQUID-NAME SCHEDULE, BOUND (b) vs BOUND (a) (real data, 2016-2026)")
    print("=" * 74)
    data = load_names(set(MEGA_CAPS),
                      ["ticker", "date", "open", "high", "low", "close", "volume"])
    window = data[data["date"] >= "2016-01-01"].copy()
    window["dollar_volume"] = window["close"] * window["volume"]
    print(f"  bars 2016-01-01 to {window['date'].max().date()}, "
          f"{len(window):,} rows across {window['ticker'].nunique()} names\n")
    print(f"{'name':>6} {'$vol/day':>12} {'price':>8} {'(a) cons':>10} {'(b) real':>10} "
          f"{'(b)/side':>10}  verdict")

    all_ok = True
    for ticker in MEGA_CAPS:
        series = window[window["ticker"] == ticker]
        estimate, regime = spread_with_resolution(
            series["open"], series["high"], series["low"], series["close"]
        )
        median_dollar_volume = float(series["dollar_volume"].median())
        price = float(series["close"].median())
        bounds = bounds_from_estimate(
            estimate, regime, median_dollar_volume, price,
            when=series["date"].max(),
        )
        per_side = bounds.realistic / 2.0 * 1e4
        # The point of the whole exercise: (b) must be cheap AND (a) must still be dear,
        # so that the two genuinely bracket rather than collapsing onto one number.
        cheap = MEGA_CAP_MIN_BPS_PER_SIDE <= per_side <= MEGA_CAP_MAX_BPS_PER_SIDE
        brackets = bounds.conservative > bounds.realistic * 2.0
        ok = cheap and brackets
        all_ok &= ok
        if not cheap:
            verdict = f"SCHEDULE WRONG ({per_side:.2f}bps/side)"
        elif not brackets:
            verdict = "BOUNDS COLLAPSED"
        else:
            verdict = "ok"
        print(f"{ticker:>6} {median_dollar_volume / 1e6:>11.0f}M {price:>8.2f} "
              f"{bounds.conservative * 1e4:>9.1f}b {bounds.realistic * 1e4:>9.1f}b "
              f"{per_side:>9.2f}b  {verdict}")

    print(f"\n  Required: bound (b) between {MEGA_CAP_MIN_BPS_PER_SIDE:.0f} and "
          f"{MEGA_CAP_MAX_BPS_PER_SIDE:.0f} bps per side, and bound (a) at least 2x it.")
    print("  If (b) is outside that window the schedule is wrong and NOTHING may be")
    print("  re-run under it -- this check fails closed by design.")
    print(f"  CHECK D: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok


# ---------------------------------------------------------------------------
# Check E -- the schedule against independently published costs
# ---------------------------------------------------------------------------

# Frazzini, Israel & Moskowitz (2018), "Trading Costs", Table II Panel A: median all-in
# one-way MARKET IMPACT, being execution price against the arrival price at the moment the
# first order was submitted. It bundles spread, impact and the trader's own patience; the
# separate implementation-shortfall row adds delay on top and is NOT what is used here.
#
# These two numbers are deliberately restated rather than imported from
# `research.capacity_study`. A control that shares constants with the model it is meant to
# catch out is not independent of it, and this one must survive that module changing.
FIM_LARGE_CAP_ALL_IN_BPS = 5.54
FIM_SMALL_CAP_ALL_IN_BPS = 13.53

# How exactly the schedule must land on AGK's published quintile medians, in bps of full
# effective spread. Registered here before the numbers are printed.
ANCHOR_TOLERANCE_BPS = 0.5


def _interpolated_schedule(volume: float, anchors: tuple[float, ...]) -> float:
    """`liquid_name_spread`'s interpolation against an arbitrary anchor set.

    Exists only so check E2 can price the SUPERSEDED anchors without mutating module
    state. E1 verifies it agrees with the real function on the shipped anchors, so it
    cannot silently drift into being a different schedule.
    """
    return float(np.exp(np.interp(
        np.log(volume),
        np.log(anchors),
        np.log(AGK_LIQUIDITY_ANCHOR_SPREAD),
    )))


def check_e_published_costs() -> bool:
    print("=" * 74)
    print("E. THE SCHEDULE AGAINST INDEPENDENTLY PUBLISHED COSTS AT STATED LIQUIDITY")
    print("=" * 74)

    results: dict[str, bool] = {}

    # --- E1: the shipped anchors put AGK's quintile medians where AGK measured them ---
    print("  E1 (implementation). Ardia-Guidotti-Kroencke 2024 Table 4 Panel C: the median")
    print("      TAQ effective spread of each MARKET-CAP quintile of the unscreened")
    print("      CRSP-TAQ universe, placed at the dollar volume that quintile trades at.\n")
    print(f"{'':>6}{'quintile':<10}{'$vol/day':>14}{'AGK median':>13}{'schedule':>11}"
          f"{'error':>9}")
    e1_ok = True
    for index, (volume, published) in enumerate(
            zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME, AGK_LIQUIDITY_ANCHOR_SPREAD), start=1):
        modelled = liquid_name_spread(volume)
        helper = _interpolated_schedule(volume, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME)
        error = abs(modelled - published) * 1e4
        ok = bool(error <= ANCHOR_TOLERANCE_BPS and np.isclose(modelled, helper))
        e1_ok = e1_ok and ok
        print(f"{'':>6}Q{index:<9}{volume / 1e6:>13,.3f}M{published * 1e4:>12.0f}b"
              f"{modelled * 1e4:>10.0f}b{error:>8.2f}b")
    results["E1 schedule reproduces AGK Table 4 Panel C at AGK's own liquidity"] = e1_ok

    # --- E2: the anchors this replaced must FAIL E1 ---
    print("\n  E2 (control of the control). The SUPERSEDED anchors -- dollar-volume")
    print("      quintiles of the capacity study's own liquidity-SCREENED universe -- must")
    print("      fail E1. If they pass, E1 cannot tell a right mapping from a wrong one.\n")
    print(f"{'':>6}{'quintile':<10}{'$vol/day':>14}{'AGK median':>13}{'superseded':>12}"
          f"{'error':>9}")
    superseded_errors = []
    for index, (volume, published) in enumerate(
            zip(AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME, AGK_LIQUIDITY_ANCHOR_SPREAD), start=1):
        modelled = _interpolated_schedule(volume, AGK_LIQUIDITY_ANCHOR_DOLLAR_VOLUME_SUPERSEDED)
        error = abs(modelled - published) * 1e4
        superseded_errors.append(error)
        print(f"{'':>6}Q{index:<9}{volume / 1e6:>13,.3f}M{published * 1e4:>12.0f}b"
              f"{modelled * 1e4:>11.0f}b{error:>8.2f}b")
    detected = max(superseded_errors) > ANCHOR_TOLERANCE_BPS
    print(f"\n      worst superseded error {max(superseded_errors):.1f}bps against a "
          f"{ANCHOR_TOLERANCE_BPS}bps tolerance: rejected = {detected}")
    results["E2 the superseded anchors are rejected by E1"] = bool(detected)

    # --- E3: half-spread must sit inside a published live all-in cost ---
    print("\n  E3 (out of sample, and this is the one with teeth). Frazzini-Israel-Moskowitz")
    print("      2018 Table IX Panel A publishes the average daily dollar volume of the two")
    print("      indices it prices; Table II Panel A measures what trading them cost. Our")
    print("      half-spread is a strict SUBSET of that measured cost, so it must sit")
    print("      strictly below it at each stated liquidity. Nothing here is calibrated on")
    print("      Table IX.\n")
    print(f"{'':>6}{'universe':<30}{'published $vol':>16}{'half-spread':>13}"
          f"{'FIM all-in':>12}   verdict")
    e3_ok = True
    for label, volume, all_in in (
        ("S&P 500 / FIM large cap", FIM_LARGE_CAP_INDEX_DOLLAR_VOLUME,
         FIM_LARGE_CAP_ALL_IN_BPS),
        ("Russell 2000 / FIM small cap", FIM_SMALL_CAP_INDEX_DOLLAR_VOLUME,
         FIM_SMALL_CAP_ALL_IN_BPS),
    ):
        half = liquid_name_spread(volume) / 2.0 * 1e4
        ok = bool(half < all_in)
        e3_ok = e3_ok and ok
        print(f"{'':>6}{label:<30}{volume / 1e6:>15,.1f}M{half:>12.2f}b{all_in:>11.2f}b   "
              f"{'ok' if ok else 'SCHEDULE EXCEEDS A MEASURED ALL-IN COST'}")
    results["E3 half-spread is inside FIM's measured all-in cost at both published levels"] \
        = e3_ok

    # --- E4: the schedule never leaves its source's support ---
    print("\n  E4 (property). A table of quintile medians says nothing outside its own")
    print("      support, so the schedule must clamp rather than extrapolate, and it must")
    print("      fall monotonically as liquidity rises.\n")
    grid = np.exp(np.linspace(np.log(1e2), np.log(1e11), 2000))
    curve = np.array([liquid_name_spread(volume) for volume in grid])
    monotone = bool(np.all(np.diff(curve) <= 1e-15))
    clamped = bool(curve.max() <= AGK_LIQUIDITY_ANCHOR_SPREAD[0] + 1e-12
                   and curve.min() >= AGK_LIQUIDITY_ANCHOR_SPREAD[-1] - 1e-12)
    print(f"{'':>6}monotonically cheaper as liquidity rises: {monotone}")
    print(f"{'':>6}never priced outside "
          f"[{AGK_LIQUIDITY_ANCHOR_SPREAD[-1] * 1e4:.0f}bps, "
          f"{AGK_LIQUIDITY_ANCHOR_SPREAD[0] * 1e4:.0f}bps]: {clamped}")
    results["E4 clamped to the source's support and monotone"] = bool(monotone and clamped)

    # --- E5: the residual, reported and NOT gated ---
    print("\n  E5 (REPORTED, NOT A GATE). At the MEDIAN name of FIM's small-cap universe the")
    print("      schedule is still well above their all-in figure, and that is left")
    print("      standing rather than closed by a factor nobody measured.\n")
    median_half = liquid_name_spread(FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME) / 2.0 * 1e4
    print(f"{'':>6}median Russell-2000 constituent  "
          f"${FIM_SMALL_CAP_MEDIAN_DOLLAR_VOLUME / 1e6:.2f}M/day  ->  "
          f"{median_half:.1f}bps per side")
    print(f"{'':>6}FIM small-cap all-in one-way     {FIM_SMALL_CAP_ALL_IN_BPS:.2f}bps  "
          f"->  ratio {median_half / FIM_SMALL_CAP_ALL_IN_BPS:.1f}x")
    print("\n      Why it is NOT gated: FIM's 13.53bps is the median across their TRADES,")
    print("      and the paper does not state the liquidity of those trades. It states the")
    print("      universe (Russell 2000) and, separately, that universe's")
    print("      capitalisation-weighted daily volume ($14.76M) -- where E3 does pass. The")
    print("      median CONSTITUENT is the far end of that range. Three known effects sit")
    print("      in the gap and none is measured here: AGK's quintile medians pool")
    print("      1993-2020 while FIM's sample is weighted to post-2006; `era_multiplier` is")
    print("      floored at 1.0 and so refuses the modern compression; and FIM's executions")
    print("      are patient algorithmic ones that need not cross the full half-spread.")
    print("      Charging less on the strength of any of those would be a guess, and a")
    print("      guess in the direction that flatters every strategy.")

    print()
    for label, ok in results.items():
        print(f"    {'ok  ' if ok else 'FAIL'}  {label}")
    passed = all(results.values())
    print(f"\n  CHECK E: {'PASS' if passed else 'FAIL'}\n")
    return passed


def main() -> int:
    if not SEP.exists():
        print(f"ERROR: {SEP} not found. Run scripts/download_sharadar_data.py first.")
        return 2

    results = {
        "A accuracy (synthetic)": check_a_accuracy(),
        "B honest degradation (real)": check_b_liquid_names(),
        "C monotonicity (real)": check_c_monotonicity(),
        "D liquid schedule (real)": check_d_liquid_schedule(),
        "E published costs (arithmetic)": check_e_published_costs(),
    }

    print("=" * 74)
    for name, passed in results.items():
        print(f"  {name:<32} {'PASS' if passed else 'FAIL'}")
    all_passed = all(results.values())
    print()
    if all_passed:
        print("  POSITIVE CONTROL PASSED - the two-bound EDGE cost model may be used.")
        print("  'upper_bound' names are TRADABLE and must be priced under BOTH bounds;")
        print("  every result carries both numbers. Excluding them was the universe bias.")
    else:
        print("  POSITIVE CONTROL FAILED - per prereg the cost model is VOID and no")
        print("  capacity-curve result may be reported using it.")
    print("=" * 74)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
