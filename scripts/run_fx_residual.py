"""Run the registered FX-residual decomposition.

Pre-registration: `research/multiasset/fx_residual_prereg.md`, written before any
decomposition existed. This script evaluates predictions P1-P6 in that document and
writes the result JSON. It corrects nothing, promotes nothing, and touches no panel
series, no strategy and no gate — it tests whether the currency-deposit ETFs used to
*verify* the FX correction are a fair yardstick.

The residual reproduced here is exactly the one `run_convention_repair.py` Control C
measured:

    diff_t = fx_excess_t - (ETF_ret_t - cash_t)

**Timing convention, which is load-bearing.** `fx_excess` credits the differential
contracted at ``t-1`` (`carry.fx_excess_returns` lags it by design, and that is what makes
it point-in-time safe), while the trust accrues the overnight rate prevailing during
``t`` and the bill accrues in ``t``. So the two panel-side rates enter LAGGED and the two
benchmark-side quantities enter CONTEMPORANEOUS. `tests/test_fx_residual.py::
test_margin_is_recovered_only_under_the_runners_lag_convention` pins this: under
time-varying rates the injected margin is recovered exactly under this convention and
provably not under the naive one.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.carry import (  # noqa: E402
    FX_INSTRUMENTS,
    FxInstrument,
    fx_excess_returns,
)
from research.multiasset.fx_residual import (  # noqa: E402
    CONSTRUCTIONS,
    OECD_DATAFLOW,
    OECD_OVERNIGHT_MEASURE,
    OECD_SHORT_MEASURE,
    FX_PAIRS,
    HEADLINE_CONSTRUCTION,
    LOW_RATE_THRESHOLD,
    MAX_ASYMMETRY,
    MAX_BRACKET_DISAGREEMENT,
    MAX_CROSS_CURRENCY_SPREAD,
    MONTHS_PER_YEAR,
    OVERNIGHT_SERIES,
    SPONSOR_FEE,
    TOL_FX,
    Prediction,
    annualise,
    decompose,
    evaluate_null_control,
    evaluate_sign_discipline,
    regime_split,
    verify_against_fred_anchors,
)
from research.sleeves.multiasset_trend import BLOCKS, load_excess_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_data" / "multiasset"
CONV = DATA / "convention"
RATE_CACHE = ROOT / "_data" / "carry"
OUT_DIR = ROOT / "research" / "multiasset" / "_fx_residual"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
#: Base retry backoff, doubled per attempt, and the polite gap between series. FRED
#: throttles bursts; the first run of this script tripped that and died on attempt 3.
BACKOFF_SECONDS = 20
INTER_SERIES_SECONDS = 8

#: The residuals Control C measured. P5 must reproduce these. Taken from the COMMITTED
#: `_convention/convention_repair.json`, not from the rounded figures in the write-up:
#: against "0.94" the null control would be passing on my own rounding rather than on a
#: real reproduction.
PUBLISHED_RESIDUALS_PCT = {"EURUSD": 0.9356, "GBPUSD": 0.8580, "JPYUSD": 0.4960}
#: The regime asymmetries the same run measured, in pp/yr. P1 must shrink these.
PUBLISHED_ASYMMETRY_PP = {"EURUSD": 0.9163, "GBPUSD": 0.3578, "JPYUSD": 0.0791}


def fetch_oecd(measure: str, *, use_cache: bool, retries: int = 3,
               areas: tuple[str, ...] | None = None,
               columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """One OECD measure for all four reference areas, as a percent-per-annum frame.

    Columns are the panel's currency codes (EZ/GB/JP/US), rows are PERIOD STRINGS
    ("1994-01"), values are percent per annum exactly as OECD publishes them. Converting
    to decimals and to month-end stamps is deliberately left to the caller, so that the
    anchor check in `verify_against_fred_anchors` compares like-for-like against the
    FRED figures recorded before the block.

    Replaces the FRED pull (prereg amendment 2026-07-31). OECD is the publisher of these
    series; FRED mirrors them. `sdmx.oecd.org` is a different CDN from the Akamai edge
    that IP-blocked this machine, and it answers in ~0.1 s.
    """
    # `areas`/`columns` let a caller pull other reference areas (the FXF/FXA/FXC holdout)
    # without disturbing the default four, which must stay byte-reproducible for P0/P5.
    if areas is None or columns is None:
        columns = ("EZ", "GB", "JP", "US")
        areas = tuple(OVERNIGHT_SERIES[c] for c in columns)
        suffix = ""
    else:
        if len(areas) != len(columns):
            raise ValueError("areas and columns must be the same length")
        suffix = "_" + "-".join(columns)

    path = RATE_CACHE / f"OECD_{measure}{suffix}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)

    area_key = "+".join(areas)
    url = (f"https://sdmx.oecd.org/public/rest/data/{OECD_DATAFLOW}/"
           f"{area_key}.M.{measure}.PA.....?format=csvfile")

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "replace")
            break
        except Exception as exc:  # noqa: BLE001 - retried with backoff, then raised
            last = exc
            if attempt < retries - 1:
                time.sleep(BACKOFF_SECONDS * (2 ** attempt))
    else:
        raise RuntimeError(f"OECD {measure}: fetch failed after {retries} attempts ({last})")

    rows = list(csv.DictReader(io.StringIO(raw)))
    by_ccy: dict[str, pd.Series] = {}
    for col, area in zip(columns, areas, strict=True):
        pairs = {r["TIME_PERIOD"]: float(r["OBS_VALUE"]) for r in rows
                 if r["REF_AREA"] == area and r["OBS_VALUE"] not in ("", "NaN")}
        by_ccy[col] = pd.Series(pairs).sort_index()
    frame = pd.DataFrame(by_ccy).sort_index()

    RATE_CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def to_month_end_decimals(series_pct: pd.Series) -> pd.Series:
    """OECD period strings + percent -> month-end timestamps + decimals.

    Both conversions have bitten this repo before and are therefore explicit: the cached
    short-rate parquet holds DECIMALS (US 0.0532 = 5.32%), and the panel is stamped at
    calendar MONTH END. Handled exactly as `build_carry_inputs.py` handles the 3-month
    family, so overnight and 3-month stay directly differenceable.
    """
    s = pd.Series(series_pct).dropna()
    s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp(how="end").normalize()
    return s[~s.index.duplicated(keep="last")].sort_index() / 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cached FRED pulls instead of re-fetching")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "prereg": "research/multiasset/fx_residual_prereg.md",
        "what_this_changes": "Nothing. No panel series, strategy, gate or headline "
                             "number is touched. This tests the yardstick, not the ruler.",
        "sponsor_fee_pct_yr": SPONSOR_FEE * 100.0,
        "low_rate_threshold_pct": LOW_RATE_THRESHOLD * 100.0,
    }

    # -- rebuild Control C's residual exactly ---------------------------------
    old, _interior = load_excess_panel()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"].reindex(old.index)
    ref = pd.read_parquet(CONV / "reference_returns_monthly.parquet").reindex(old.index)
    rates_short = pd.read_parquet(RATE_CACHE / "short_rates_monthly.parquet")

    panel_fx: list[FxInstrument] = []
    for key in tuple(BLOCKS["fx"]):
        if key == "USDX":
            continue
        match = next((i for i in FX_INSTRUMENTS if i.key.endswith(key[:3])), None)
        ccy = {"EURUSD": "EZ", "GBPUSD": "GB", "JPYUSD": "JP"}.get(key)
        if ccy is None:
            continue
        panel_fx.append(FxInstrument(key, match.ticker if match else key, ccy, False,
                                     f"{key} as carried in the trend panel"))
    fx_excess, _carry = fx_excess_returns(old[[i.key for i in panel_fx]],
                                          rates_short.reindex(old.index),
                                          tuple(panel_fx))

    # -- overnight counterparts ------------------------------------------------
    raw_overnight = fetch_oecd(OECD_OVERNIGHT_MEASURE, use_cache=args.use_cache)

    # -- P0: the substitution must not have changed the data -------------------
    p0 = verify_against_fred_anchors(
        {c: raw_overnight[c].dropna() for c in raw_overnight.columns})
    results["P0_source_identity"] = p0.as_dict()
    if not p0.passed:
        results["verdict"] = "VOID - OECD pull does not reproduce the registered series"
        (OUT_DIR / "fx_residual.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("P0 FAILED - the transport substitution changed the data. Run void.")
        return 1

    overnight = {c: to_month_end_decimals(raw_overnight[c]) for c in raw_overnight.columns}
    results["overnight_sources"] = {
        c: {"oecd_ref_area": OVERNIGHT_SERIES[c], "n": int(len(s)),
            "first": str(s.index.min().date()), "last": str(s.index.max().date())}
        for c, s in overnight.items()
    }
    results["source_note"] = (
        "Overnight legs from OECD SDMX direct (dataflow "
        f"{OECD_DATAFLOW}, MEASURE={OECD_OVERNIGHT_MEASURE}); FRED was IP-blocked "
        "mid-run. OECD publishes these series, FRED mirrors them. See the prereg "
        "amendment and P0."
    )

    # -- transport cross-check: OECD's own 3-month vs the repo's FRED-sourced one --
    try:
        raw_short = fetch_oecd(OECD_SHORT_MEASURE, use_cache=args.use_cache)
        xcheck: dict[str, dict] = {}
        for ccy in ("EZ", "GB", "JP", "US"):
            oecd_dec = to_month_end_decimals(raw_short[ccy])
            both = pd.concat([oecd_dec.rename("oecd"),
                              rates_short[ccy].rename("repo")], axis=1).dropna()
            if len(both):
                d = (both["oecd"] - both["repo"]).abs()
                xcheck[ccy] = {"n_overlap": int(len(both)),
                               "max_abs_diff": float(d.max()),
                               "mean_abs_diff": float(d.mean()),
                               "identical": bool(d.max() < 1e-6)}
        results["transport_crosscheck_3m"] = {
            "question": "Does OECD-direct carry the same 3-month data the repo already "
                        "holds from FRED? If yes, the overnight substitution is transport, "
                        "not source.",
            "per_currency": xcheck,
        }
    except Exception as exc:  # noqa: BLE001 - a cross-check failure must not void the run
        results["transport_crosscheck_3m"] = {"skipped": f"{type(exc).__name__}: {exc}"}

    # -- P6: sign discipline, an integrity GATE -------------------------------
    rates_idx = rates_short.index
    spreads = {}
    for ccy in OVERNIGHT_SERIES:
        pair = pd.concat([rates_short[ccy].rename("m3"),
                          overnight[ccy].reindex(rates_idx).rename("on")],
                         axis=1).dropna()
        spreads[ccy] = float((pair["m3"] - pair["on"]).mean()) if len(pair) else float("nan")
    p6 = evaluate_sign_discipline(spreads)
    results["P6_sign_discipline"] = p6.as_dict()
    if not p6.passed:
        results["verdict"] = "VOID - series pairing error, see P6"
        (OUT_DIR / "fx_residual.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("P6 FAILED - run is void, pairing error:", p6.detail["negative_offenders"])
        return 1

    # -- assemble per-pair inputs under the runner's lag convention ------------
    inputs: dict[str, dict] = {}
    diffs: dict[str, pd.Series] = {}
    for key, (etf, ccy) in FX_PAIRS.items():
        if etf not in ref.columns or key not in fx_excess.columns:
            continue
        diffs[key] = (fx_excess[key] - (ref[etf] - cash)).dropna()
        inputs[key] = {
            # panel side: the rate CONTRACTED at t-1, exactly what fx_excess credited
            "i3m_foreign": rates_short[ccy].reindex(old.index).shift(1),
            "i3m_us": rates_short["US"].reindex(old.index).shift(1),
            # benchmark side: what the trust and the bill accrued DURING t
            "overnight_foreign": overnight[ccy].reindex(old.index),
            "cash": cash,
        }

    # -- P5: null control ------------------------------------------------------
    reproduced = {}
    for key, diff in diffs.items():
        null = decompose(diff, construction="published", fee=0.0,
                         i3m_foreign=inputs[key]["i3m_foreign"],
                         overnight_foreign=inputs[key]["i3m_foreign"],   # spread -> 0
                         i3m_us=inputs[key]["i3m_us"],
                         cash=inputs[key]["i3m_us"] / MONTHS_PER_YEAR)   # TED -> 0
        reproduced[key] = round(annualise(null["remainder"]) * 100.0, 4)
    p5 = evaluate_null_control(reproduced, PUBLISHED_RESIDUALS_PCT)
    results["P5_null_control"] = p5.as_dict()

    # -- the decomposition, over all three registered constructions ------------
    by_construction: dict[str, dict] = {}
    for construction in CONSTRUCTIONS:
        rows: dict[str, dict] = {}
        for key, diff in diffs.items():
            frame = decompose(diff, construction=construction, **inputs[key])
            rows[key] = {
                "etf": FX_PAIRS[key][0],
                "n": int(len(frame)),
                "first": str(frame.index.min().date()),
                "last": str(frame.index.max().date()),
                "measured_residual_pct_yr": round(annualise(frame["diff"]) * 100.0, 4),
                "components_pct_yr": {
                    "A1_foreign_tenor": round(
                        annualise(frame["a1_foreign_tenor"]) * 100.0, 4),
                    "B_sponsor_fee": round(annualise(frame["b_sponsor_fee"]) * 100.0, 4),
                    "C_us_ted_subtracted": round(annualise(frame["c_us_ted"]) * 100.0, 4),
                },
                "predicted_pct_yr": round(annualise(frame["predicted"]) * 100.0, 4),
                "remainder_pct_yr": round(annualise(frame["remainder"]) * 100.0, 4),
                "remainder_inside_budget": bool(
                    abs(annualise(frame["remainder"])) <= TOL_FX),
                "regime_before": regime_split(frame, "diff"),
                "regime_after": regime_split(frame, "remainder"),
            }
        by_construction[construction] = rows
    results["by_construction"] = by_construction
    results["headline_construction"] = HEADLINE_CONSTRUCTION
    head = by_construction[HEADLINE_CONSTRUCTION]

    # -- P1 / P2 / P3 / P4 -----------------------------------------------------
    asym_after = {k: v["regime_after"]["asymmetry_pct_yr"] for k, v in head.items()}
    asym_before = {k: v["regime_before"]["asymmetry_pct_yr"] for k, v in head.items()}
    target = ("EURUSD", "GBPUSD")
    p1_ok = all(asym_after.get(k) is not None
                and asym_after[k] <= MAX_ASYMMETRY * 100.0 for k in target)
    widened = [k for k in target
               if asym_after.get(k) is not None and asym_before.get(k) is not None
               and asym_after[k] > asym_before[k]]
    results["P1_asymmetry_collapse"] = Prediction(
        "P1", f"regime asymmetry <= {MAX_ASYMMETRY * 100.0} pp/yr for EUR and GBP "
              f"(published: EUR {PUBLISHED_ASYMMETRY_PP['EURUSD']}, "
              f"GBP {PUBLISHED_ASYMMETRY_PP['GBPUSD']})",
        p1_ok,
        {"asymmetry_before_pct_yr": asym_before, "asymmetry_after_pct_yr": asym_after,
         "widened_and_therefore_refuted": widened},
    ).as_dict()

    p2_ok = all(head[k]["remainder_inside_budget"] for k in target if k in head)
    results["P2_level_inside_budget"] = Prediction(
        "P2", f"adjusted remainder inside TOL_FX = {TOL_FX * 100.0}%/yr for EUR and GBP",
        p2_ok,
        {"remainder_pct_yr": {k: head[k]["remainder_pct_yr"] for k in head}},
    ).as_dict()

    rems = [v["remainder_pct_yr"] for v in head.values()]
    spread_pp = round(max(rems) - min(rems), 4) if rems else None
    results["P3_cross_currency_consistency"] = Prediction(
        "P3", f"the three remainders lie within {MAX_CROSS_CURRENCY_SPREAD * 100.0} pp/yr "
              f"of one another (informative, not decisive)",
        spread_pp is not None and spread_pp <= MAX_CROSS_CURRENCY_SPREAD * 100.0,
        {"remainder_spread_pp": spread_pp},
    ).as_dict()

    jpy_ok = head.get("JPYUSD", {}).get("remainder_inside_budget", False)
    results["P4_do_no_harm"] = Prediction(
        "P4", f"JPY, the leg that already passed, stays inside {TOL_FX * 100.0}%/yr",
        bool(jpy_ok),
        {"jpy_remainder_pct_yr": head.get("JPYUSD", {}).get("remainder_pct_yr")},
    ).as_dict()

    # -- bracket disagreement (prereg 4) --------------------------------------
    bracket = {}
    for key in head:
        vals = [by_construction[c][key]["remainder_pct_yr"] for c in CONSTRUCTIONS]
        bracket[key] = {"low": min(vals), "high": max(vals),
                        "range_pp": round(max(vals) - min(vals), 4)}
    widest = max((b["range_pp"] for b in bracket.values()), default=0.0)
    results["bracket"] = {
        "per_pair": bracket, "widest_range_pp": widest,
        "max_allowed_pp": MAX_BRACKET_DISAGREEMENT * 100.0,
        "constructions_agree": bool(widest <= MAX_BRACKET_DISAGREEMENT * 100.0),
    }

    explained = bool(p5.passed and p6.passed and p1_ok and p2_ok and jpy_ok)
    partial = bool(p5.passed and p6.passed and p1_ok and not p2_ok)
    results["verdict"] = (
        "EXPLAINED" if explained
        else "MECHANISM IDENTIFIED, GAP NOT CLOSED" if partial
        else "NOT EXPLAINED"
    )
    results["decision_rule"] = ("EXPLAINED requires P1 and P2 and P4, with P5 and P6 "
                                "passing as preconditions. P3 cannot rescue a failure.")

    (OUT_DIR / "fx_residual.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"P5 null control: {'PASS' if p5.passed else 'FAIL'}  {reproduced}")
    print(f"P6 sign discipline: {'PASS' if p6.passed else 'FAIL'}")
    print(f"\nheadline construction: {HEADLINE_CONSTRUCTION}")
    for key, row in head.items():
        print(f"  {key} vs {row['etf']}: residual {row['measured_residual_pct_yr']:+.3f} "
              f"-> remainder {row['remainder_pct_yr']:+.3f} %/yr  "
              f"(A1 {row['components_pct_yr']['A1_foreign_tenor']:+.3f}, "
              f"B {row['components_pct_yr']['B_sponsor_fee']:+.3f}, "
              f"C {row['components_pct_yr']['C_us_ted_subtracted']:+.3f})  "
              f"asym {row['regime_before']['asymmetry_pct_yr']} -> "
              f"{row['regime_after']['asymmetry_pct_yr']} pp")
    print(f"\nP1 {'PASS' if p1_ok else 'FAIL'} | P2 {'PASS' if p2_ok else 'FAIL'} | "
          f"P4 {'PASS' if jpy_ok else 'FAIL'}")
    print(f"VERDICT: {results['verdict']}")
    print(f"\nWrote {OUT_DIR / 'fx_residual.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
