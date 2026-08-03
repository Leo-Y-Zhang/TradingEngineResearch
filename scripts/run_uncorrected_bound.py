"""Put a number on the uncorrected 21.2%: by how much is 0.7834 an upper bound?

Pre-registration: `research/multiasset/uncorrected_bound_prereg.md`, written before any
charged panel or re-run book existed.

Two components, treated differently and never conflated:

* **USDX is CORRECTED** — `spot - (i_basket - i_US)_{t-1}/12` at the published DXY
  weights. This executes a step the original convention-repair prereg registered and the
  implementation skipped (`run_convention_repair.py` short-circuits USDX at line 285 and
  stamps it UNCORRECTED at line 413).
* **Commodities are CHARGED, not corrected** — a constant drag equal to the measured gap
  against a tradable reference. Nothing about the roll is reconstructed. The charged panel
  must never be described as corrected.

Chooses no method for the book itself: `evaluate` is imported from
`run_convention_repair_book` and called exactly as that script calls it, so the comparison
against 0.7834 is like for like. **No sleeve is re-selected, re-tuned or re-gated.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.panel import (  # noqa: E402
    clean_levels,
    monthly_returns,
    simple_returns,
    wide_panel,
)
from research.sleeves.multiasset_trend import load_excess_panel  # noqa: E402
from scripts.build_multiasset_panel import fetch_one  # noqa: E402
from scripts.run_convention_repair_book import DATA, evaluate  # noqa: E402

PANEL_RAW = DATA / "raw"
RESULT_DIR = Path(__file__).resolve().parents[1] / "research" / "multiasset" / "_uncorrected"
MPY = 12

#: Committed figure this must reproduce with zero charge (B1).
CENTRAL_BOOK_SHARPE = 0.7834

#: panel series -> (reference ETF, why it was chosen, is the reference roll-free?)
COMMODITY_REFS: dict[str, tuple[str, str, bool]] = {
    "GOLD_F": ("GLD", "physically backed bullion - no roll at all", True),
    "SILVER_F": ("SLV", "physically backed - no roll at all", True),
    "WTI_F": ("USO", "WTI-only futures ETF with a published roll", False),
    "COPPER_F": ("CPER", "copper-only futures ETF; only single-metal option", False),
}

#: Published DXY basket weights, held constant, as the original prereg registered.
DXY_WEIGHTS: dict[str, float] = {
    "EZ": 0.576, "JP": 0.136, "GB": 0.119, "CA": 0.091, "SE": 0.042, "CH": 0.036,
}

BOUNDS = ("roll_free_only", "overlap_only", "full_sample", "full_sample_upper")
#: The prereg (section 2) fixed the ordering: GLD and SLV are roll-FREE references, so
#: their gaps cannot be blamed on the reference performing a different roll. USO and
#: CPER roll themselves and are the weaker two. This cut isolates the unambiguous core.
ROLL_FREE = ("GOLD_F", "SILVER_F")
HEADLINE_BOUND = "full_sample"
BRACKET_DISAGREEMENT_LIMIT = 0.05     # prereg 4


def reference_excess(ticker: str, cash: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Monthly EXCESS return of a reference ETF, built the way the panel builds returns."""
    raw = fetch_one(ticker, PANEL_RAW, use_cache=True)
    lvl, _ = clean_levels(raw["Close"])
    ret, _ = simple_returns(lvl)
    monthly = monthly_returns(wide_panel({ticker: ret}))[ticker]
    return (monthly.reindex(index) - cash.reindex(index)).rename(ticker)


def measure_gap(panel_series: pd.Series, ref_excess: pd.Series) -> dict:
    """Annualised gap `panel - reference`, with a 95% interval on the mean.

    A POSITIVE gap means the panel series sits above a tradable reference, i.e. the panel
    overstates and the charge is positive.
    """
    both = pd.concat([panel_series.rename("p"), ref_excess.rename("r")], axis=1).dropna()
    if len(both) < 24:
        return {"n": int(len(both)), "verdict": "INSUFFICIENT_OVERLAP"}
    d = both["p"] - both["r"]
    mean_m = float(d.mean())
    se_m = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "n": int(len(both)),
        "first": str(both.index.min().date()), "last": str(both.index.max().date()),
        "gap_pct_yr": round(mean_m * MPY * 100, 4),
        "gap_upper95_pct_yr": round((mean_m + 1.96 * se_m) * MPY * 100, 4),
        "corr": round(float(both["p"].corr(both["r"])), 5),
        "_mean_monthly": mean_m,
        "_upper_monthly": mean_m + 1.96 * se_m,
        "_overlap_index": both.index,
    }


def correct_usdx(panel: pd.DataFrame, rates: pd.DataFrame) -> tuple[pd.Series, dict]:
    """USDX excess return with the basket interest differential applied.

    Long USD against the basket earns the US rate and forgoes the basket rate, so the
    correction SUBTRACTS `(i_basket - i_US)`. A month is corrected only where every
    basket rate is present; otherwise it is left untouched and counted as uncorrected.
    """
    idx = panel.index
    r = rates.reindex(idx)
    have = r[list(DXY_WEIGHTS)].notna().all(axis=1) & r["US"].notna()
    basket = sum(w * r[c] for c, w in DXY_WEIGHTS.items())
    delta = ((basket - r["US"]).shift(1) / MPY).where(have.shift(1).fillna(False), 0.0)
    corrected = panel["USDX"] - delta
    n_corr = int((delta != 0.0).sum())
    return corrected, {
        "weights": DXY_WEIGHTS,
        "n_months_corrected": n_corr,
        "n_months_live": int(panel["USDX"].notna().sum()),
        "frac_corrected": round(n_corr / max(int(panel["USDX"].notna().sum()), 1), 4),
        "mean_charge_pct_yr": round(float(delta[delta != 0.0].mean()) * MPY * 100, 4)
        if n_corr else 0.0,
    }


def charge_panel(panel: pd.DataFrame, gaps: dict, bound: str) -> pd.DataFrame:
    """Apply the commodity drag under one registered bound. Never called 'correcting'."""
    out = panel.copy()
    for key, g in gaps.items():
        if "_mean_monthly" not in g or key not in out.columns:
            continue
        if bound == "roll_free_only" and key not in ROLL_FREE:
            continue
        per_month = g["_upper_monthly"] if bound == "full_sample_upper" else g["_mean_monthly"]
        live = out[key].notna()
        mask = live & out.index.isin(g["_overlap_index"]) if bound == "overlap_only" else live
        out.loc[mask, key] = out.loc[mask, key] - per_month
    return out


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    old, interior = load_excess_panel()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"].reindex(old.index)
    rates = pd.read_parquet(
        Path(__file__).resolve().parents[1] / "_data" / "carry" / "short_rates_monthly.parquet")

    central = pd.read_parquet(DATA / "returns_monthly_corrected_central.parquet")
    results: dict = {
        "prereg": "research/multiasset/uncorrected_bound_prereg.md",
        "question": "By how much is the corrected book Sharpe of 0.7834 an upper bound?",
        "what_this_changes": "The honesty statement attached to 0.7834. It promotes "
                             "nothing, re-opens no gate, and re-selects no sleeve.",
    }

    # -- B1: reproduction anchor -------------------------------------------------------
    base = evaluate("central_uncharged", central, interior, cash)
    reproduces = abs(base["sharpe_book"] - CENTRAL_BOOK_SHARPE) < 5e-4
    results["B1_reproduction_anchor"] = {
        "measured": base["sharpe_book"], "recorded": CENTRAL_BOOK_SHARPE,
        "reproduces": bool(reproduces),
    }
    if not reproduces:
        results["verdict"] = "VOID - the uncharged central panel does not reproduce 0.7834"
        (RESULT_DIR / "uncorrected_bound.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"B1 FAILED: {base['sharpe_book']} vs {CENTRAL_BOOK_SHARPE}. Run void.")
        return 1

    # -- measure the commodity gaps ----------------------------------------------------
    gaps: dict[str, dict] = {}
    for key, (ticker, why, rollfree) in COMMODITY_REFS.items():
        ref = reference_excess(ticker, cash, central.index)
        g = measure_gap(central[key], ref)
        g.update({"reference": ticker, "why": why, "reference_is_roll_free": rollfree})
        gaps[key] = g
    results["commodity_gaps"] = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in gaps.items()
    }

    # -- USDX, corrected ---------------------------------------------------------------
    usdx_corrected, usdx_info = correct_usdx(central, rates)
    results["usdx_correction"] = usdx_info
    usdx_panel = central.copy()
    usdx_panel["USDX"] = usdx_corrected
    usdx_only = evaluate("usdx_only", usdx_panel, interior, cash)
    results["B4_usdx_standalone"] = {
        "sharpe_book": usdx_only["sharpe_book"],
        "delta_vs_uncharged": round(base["sharpe_book"] - usdx_only["sharpe_book"], 6),
    }

    # -- the three registered bounds ---------------------------------------------------
    runs: dict[str, dict] = {"central_uncharged": base}
    for bound in BOUNDS:
        panel = charge_panel(usdx_panel, gaps, bound)
        runs[bound] = evaluate(bound, panel, interior, cash)
    results["runs"] = {k: {kk: v[kk] for kk in
                           ("sharpe_book", "sharpe_trend", "sharpe_passive",
                            "vol_matched_active_pct_yr", "vol_matched_active_tstat",
                            "sharpe_since_2010", "leverage_ladder")}
                       for k, v in runs.items()}

    per_instrument: dict[str, dict] = {}
    for key in COMMODITY_REFS:
        if "_mean_monthly" not in gaps[key]:
            continue
        one = usdx_panel.copy()
        live = one[key].notna()
        one.loc[live, key] = one.loc[live, key] - gaps[key]["_mean_monthly"]
        s_one = evaluate(key, one, interior, cash)["sharpe_book"]
        per_instrument[key] = {
            "sharpe_book": s_one,
            "delta_vs_usdx_only": round(usdx_only["sharpe_book"] - s_one, 6),
            "gap_pct_yr": gaps[key]["gap_pct_yr"],
            "reference": gaps[key]["reference"],
            "reference_is_roll_free": gaps[key]["reference_is_roll_free"],
        }
    results["per_instrument_charge"] = per_instrument

    deltas = {b: round(base["sharpe_book"] - runs[b]["sharpe_book"], 6) for b in BOUNDS}
    results["B3_deltas_sharpe"] = deltas
    results["B2_direction_all_charges_lower_the_book"] = bool(all(d >= 0 for d in deltas.values()))
    spread = max(runs[b]["sharpe_book"] for b in BOUNDS) - min(
        runs[b]["sharpe_book"] for b in BOUNDS)
    results["bracket_spread_sharpe"] = round(float(spread), 6)
    results["bracket_bounds_agree"] = bool(spread <= BRACKET_DISAGREEMENT_LIMIT)

    harshest = max(deltas.values())
    robust = deltas.get("roll_free_only", float("nan"))
    results["headline"] = {
        "uncharged_book_sharpe": base["sharpe_book"],
        "headline_bound": HEADLINE_BOUND,
        "charged_book_sharpe": runs[HEADLINE_BOUND]["sharpe_book"],
        "upper_bound_by_at_most_sharpe": round(float(harshest), 4),
        "robust_core_delta_roll_free_only": round(float(robust), 4),
        "statement": (f"0.7834 is an upper bound by at most {harshest:.4f} Sharpe; "
                      f"the headline charged figure is "
                      f"{runs[HEADLINE_BOUND]['sharpe_book']:.4f}. The ROBUST core, using "
                      f"only the two roll-free references, is {robust:.4f}."),
        "material_at_0.01_sharpe": bool(harshest >= 0.01),
    }
    if not results["B2_direction_all_charges_lower_the_book"]:
        results["verdict"] = "VOID - a charge RAISED the book, which means a sign error"
    else:
        results["verdict"] = ("MATERIAL" if harshest >= 0.01 else
                              "NOT MATERIAL - the uncorrected 21.2% does not move the headline")

    (RESULT_DIR / "uncorrected_bound.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"B1 reproduction: {base['sharpe_book']:.6f} vs {CENTRAL_BOOK_SHARPE} -> "
          f"{'REPRODUCES' if reproduces else 'FAILS'}\n")
    print("commodity gaps (panel minus tradable reference, +ve = panel overstates):")
    for k, g in gaps.items():
        if "gap_pct_yr" in g:
            print(f"  {k:9s} vs {g['reference']:5s} {g['gap_pct_yr']:+8.3f} %/yr "
                  f"(upper95 {g['gap_upper95_pct_yr']:+7.3f})  n={g['n']:4d}  "
                  f"corr={g['corr']:.3f}  roll_free_ref={g['reference_is_roll_free']}")
    print(f"\nUSDX: {usdx_info['n_months_corrected']} of {usdx_info['n_months_live']} months "
          f"corrected ({usdx_info['frac_corrected']:.1%}), mean charge "
          f"{usdx_info['mean_charge_pct_yr']:+.3f}%/yr, standalone dSharpe "
          f"{results['B4_usdx_standalone']['delta_vs_uncharged']:+.4f}")
    print(f"\n  {'bound':<20}{'book':>9}{'dSharpe':>10}{'lev@DD50':>10}{'CAGR%':>9}{'x0.877':>9}")
    for name in ("central_uncharged", *BOUNDS):
        r = runs[name]
        dd = r["leverage_ladder"]["dd50"]
        d = "" if name == "central_uncharged" else f"{deltas[name]:>10.4f}"
        print(f"  {name:<20}{r['sharpe_book']:>9.4f}{d:>10}{dd['leverage']:>10.2f}"
              f"{dd['cagr_pct']:>9.2f}{dd['cagr_after_reconciliation_pct']:>9.2f}")
    print(f"\nbracket spread {spread:.4f} (limit {BRACKET_DISAGREEMENT_LIMIT}) -> "
          f"{'agree' if results['bracket_bounds_agree'] else 'DISAGREE, report as a range'}")
    print(f"\n{results['headline']['statement']}")
    print(f"VERDICT: {results['verdict']}")
    print(f"\nWrote {RESULT_DIR / 'uncorrected_bound.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
