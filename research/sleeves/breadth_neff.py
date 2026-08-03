"""MEASURE the correlation-effective N of the expanded panel. This is the deliverable.

Iteration 11 measured N_eff = 5.26 on the original 18 instruments. Iteration 12 showed
that the growth ceiling is ``S^2/2`` with ``S = s * sqrt(N_eff)``, so N_eff is the only
quantity left that can move it. This module measures the new one.

    .venv/Scripts/python.exe -m research.sleeves.breadth_neff

THE DEFINITION, stated explicitly (``multiasset_trend.effective_n``, unmodified)
================================================================================
Let ``C`` be the Pearson correlation matrix of the month-end EXCESS returns of the
universe and ``lambda_1..lambda_n`` its eigenvalues. Then::

    N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

the participation ratio of the eigenvalue spectrum. For ``n`` uncorrelated instruments
every eigenvalue is 1 and ``N_eff = n``; for ``n`` instruments that are all the same bet
one eigenvalue is ``n`` and the rest are 0, so ``N_eff = 1``. It answers "how many
independent bets is this really", which is not the same question as "how many tickers".

THE CONTROL
===========
The number is only comparable if it is produced the same way, so this module first
reproduces iteration 11's **5.26** on the original 18 instruments over the same window
(1996 onward, pairwise-complete correlations) and refuses to report anything else if it
cannot. A tolerance of 0.005 is enforced in code, not by eye.

THE CONFOUND, and how it is handled
===================================
The added instruments start between 2000 and 2020, so the expanded panel is SHORTER than
the original. Correlations are not stationary — they rise in crises, and a 2008-onward
window is crisis-dense — so comparing 5.26 (measured from 1996) against an expanded
number measured from 2008 would confound breadth with window. Every comparison here is
therefore run on a MATCHED window, with the original 18 re-measured on that same window
as the baseline. The headline gain is always ``expanded - original ON THE SAME WINDOW``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.sleeves.breadth_universe import (
    ALL_BLOCKS,
    BREADTH_ADDITIONS,
    EXPANDED,
    ORIGINAL_18,
    UNIVERSES,
    load_combined_panel,
)
from research.sleeves.multiasset_trend import MONTHS, annual_sharpe, effective_n

_OUT = Path("research/sleeves/_breadth")

CONTROL_WINDOW = "1996"
CONTROL_TARGET = 5.2602648349
CONTROL_TOL = 0.005

# Matched windows. Each is the first month at which a further group of additions is
# fully live, so each row prices the breadth it buys in years of history given up.
WINDOWS: dict[str, str] = {
    "1996+": "1996",     # iteration 11's window; only the original 18 are live
    "2001+": "2001-04",  # all agriculture and livestock live
    "2008+": "2008-02",  # + credit, REITs, gilts, Bunds
    "2011+": "2011-02",  # + volatility (VIXY)
    "2016+": "2016-07",  # + JGBs
    "2020+": "2020-09",  # + freight and carbon: everything live
}


def _neff_both(x: pd.DataFrame) -> dict[str, float]:
    """N_eff under both correlation conventions, because they can disagree.

    ``pairwise`` is what iteration 11 used (pandas' default: each pair uses the months
    both series have). ``complete`` uses only months where EVERY series is present. With
    staggered start dates the pairwise matrix need not be positive semi-definite, so the
    two are reported side by side rather than one being chosen silently.
    """
    if x.shape[1] < 2 or len(x) < 12:
        return {"pairwise": float("nan"), "complete": float("nan"), "months": len(x),
                "months_complete": 0, "n_instruments": int(x.shape[1])}
    cc = x.dropna(how="any")
    return {
        "pairwise": effective_n(x.corr()),
        "complete": effective_n(cc.corr()) if len(cc) >= 12 else float("nan"),
        "months": int(len(x)),
        "months_complete": int(len(cc)),
        "n_instruments": int(x.shape[1]),
    }


def _mean_abs_corr(x: pd.DataFrame) -> float:
    c = x.corr().to_numpy(dtype=float)
    iu = np.triu_indices_from(c, k=1)
    v = c[iu]
    v = v[np.isfinite(v)]
    return float(np.abs(v).mean()) if v.size else float("nan")


def implied(n_eff: float, per_bet_sharpe: float, efficiency: float) -> dict[str, float]:
    """Turn an N_eff into the numbers that decide the question.

    ``S = s * sqrt(N_eff)``; theoretical max growth ``S^2/2`` (full Kelly) and
    ``3S^2/8`` (half). ``efficiency`` is iteration 11's MEASURED ratio of achieved peak
    compound return to the theoretical ``S^2/2`` — 0.71 — which is where financing,
    variance drag and trading costs go. Everything here is arithmetic on measured
    inputs; nothing here is a backtest result.
    """
    s = per_bet_sharpe * math.sqrt(n_eff)
    return {
        "portfolio_sharpe": s,
        "kelly_full_pct": 100.0 * s * s / 2.0,
        "kelly_half_pct": 100.0 * 3.0 * s * s / 8.0,
        "expected_measured_peak_pct": 100.0 * efficiency * s * s / 2.0,
    }


def n_eff_required(target: float, per_bet_sharpe: float, efficiency: float) -> dict[str, float]:
    """The N_eff a given compound target needs, on the clean and the honest arithmetic."""
    s2 = per_bet_sharpe ** 2
    return {
        # clean idealisations, which UNDERSTATE the requirement
        "n_eff_full_kelly": (2.0 * target) / s2,
        "n_eff_half_kelly": (8.0 * target / 3.0) / s2,
        # the honest one: a MEASURED 30% needs a theoretical 30/efficiency, and the
        # theoretical figure that iteration 11's ladder actually realised is the
        # half-Kelly one. This is the number to judge against.
        "n_eff_half_kelly_measured_efficiency": (8.0 * target / 3.0 / efficiency) / s2,
        "n_eff_full_kelly_measured_efficiency": (2.0 * target / efficiency) / s2,
    }


def main() -> dict:
    out: dict = {"definition": "N_eff = (sum eigenvalues)^2 / sum(eigenvalues^2) of the "
                               "Pearson correlation matrix of month-end excess returns"}

    # ── THE CONTROL ───────────────────────────────────────────────────────────
    x_orig, _ = load_combined_panel(ORIGINAL_18)
    control = effective_n(x_orig.loc[CONTROL_WINDOW:].corr())
    out["control"] = {
        "universe": "original_18", "window": f"{CONTROL_WINDOW}+",
        "measured": control, "recorded_iteration_11": CONTROL_TARGET,
        "abs_error": abs(control - CONTROL_TARGET), "tolerance": CONTROL_TOL,
        "reproduced": bool(abs(control - CONTROL_TARGET) <= CONTROL_TOL),
    }
    if not out["control"]["reproduced"]:
        raise AssertionError(
            f"CONTROL FAILED: N_eff {control:.6f} != {CONTROL_TARGET:.6f}. "
            "The expanded number is not comparable; nothing else is reported.")
    print(f"CONTROL OK  original 18, {CONTROL_WINDOW}+  N_eff = {control:.4f} "
          f"(recorded {CONTROL_TARGET:.4f})")

    # ── coverage of the additions ─────────────────────────────────────────────
    x_all, _ = load_combined_panel(EXPANDED)
    out["coverage"] = {
        k: {"first_month": (str(x_all[k].first_valid_index().date())
                            if x_all[k].first_valid_index() is not None else None),
            "months": int(x_all[k].notna().sum()),
            "annual_sharpe": annual_sharpe(x_all[k]),
            "mean_excess_annual_pct": 100.0 * float(x_all[k].mean() * MONTHS),
            "vol_annual_pct": 100.0 * float(x_all[k].std(ddof=1) * math.sqrt(MONTHS))}
        for k in BREADTH_ADDITIONS
    }

    # ── N_eff by universe x window, ALWAYS matched ────────────────────────────
    by_window: dict[str, dict] = {}
    for wname, wstart in WINDOWS.items():
        row: dict = {}
        for uname, keys in UNIVERSES.items():
            xu, _ = load_combined_panel(tuple(keys))
            sl = xu.loc[wstart:]
            # drop instruments with no data at all in this window
            sl = sl.loc[:, [c for c in sl.columns if sl[c].notna().sum() >= 12]]
            if sl.shape[1] < 2:
                continue
            row[uname] = {**_neff_both(sl), "mean_abs_corr": _mean_abs_corr(sl)}
        base = row.get("original_18", {}).get("pairwise", float("nan"))
        for uname, v in row.items():
            v["gain_vs_original_same_window"] = v["pairwise"] - base
        by_window[wname] = row
    out["n_eff_by_window"] = by_window

    # ── the headline pair ─────────────────────────────────────────────────────
    head_w = "2011+"
    h = by_window[head_w]
    out["headline"] = {
        "window": head_w,
        "original_18": h["original_18"]["pairwise"],
        "expanded": h["expanded_37"]["pairwise"],
        "gain": h["expanded_37"]["pairwise"] - h["original_18"]["pairwise"],
        "n_instruments_original": h["original_18"]["n_instruments"],
        "n_instruments_expanded": h["expanded_37"]["n_instruments"],
    }

    # ── per-block: what did each addition actually buy? ───────────────────────
    blocks: dict[str, dict] = {}
    for b in ALL_BLOCKS:
        key_alone = f"block_{b}" if f"block_{b}" in UNIVERSES else None
        key_plus = f"orig_plus_{b}" if f"orig_plus_{b}" in UNIVERSES else None
        rec: dict = {"instruments": list(ALL_BLOCKS[b])}
        for wname in ("2011+", "2020+"):
            w = by_window[wname]
            rec[wname] = {
                "block_alone_neff": w.get(key_alone, {}).get("pairwise") if key_alone else None,
                "orig_plus_block_neff": (w.get(key_plus, {}).get("pairwise")
                                         if key_plus else None),
                "marginal_vs_original": (
                    w[key_plus]["pairwise"] - w["original_18"]["pairwise"]
                    if key_plus and key_plus in w else None),
                "n_added": len(ALL_BLOCKS[b]) if key_plus else 0,
                "efficiency_pct": (
                    100.0 * (w[key_plus]["pairwise"] - w["original_18"]["pairwise"])
                    / len(ALL_BLOCKS[b]) if key_plus and key_plus in w else None),
            }
        blocks[b] = rec
    out["per_block"] = blocks

    # ── the arithmetic that decides the question ─────────────────────────────
    # Inputs are iteration 11's MEASURED numbers, not new estimates.
    sharpe_18 = 0.6678
    neff_18 = control
    per_bet = sharpe_18 / math.sqrt(neff_18)
    efficiency = 0.1583 / (sharpe_18 ** 2 / 2.0)
    out["arithmetic_inputs"] = {
        "portfolio_sharpe_original_18_measured": sharpe_18,
        "n_eff_original_18_measured": neff_18,
        "per_bet_sharpe_implied": per_bet,
        "measured_peak_compound_original_18": 0.1583,
        "theoretical_peak_s2_over_2": sharpe_18 ** 2 / 2.0,
        "efficiency_measured": efficiency,
    }
    out["required_for_30pct"] = n_eff_required(0.30, per_bet, efficiency)
    # WARNING, and the point of the whole study. The projection below is iteration 12's
    # arithmetic applied to the new N_eff, and it holds the per-bet Sharpe FIXED at the
    # value measured on the original 18. That assumption is what `breadth_ladder.py`
    # tests, and it is REFUTED: the added bets are independent AND worse, so the measured
    # per-bet Sharpe falls from 0.279 to 0.169 as N_eff rises from 5.14 to 8.39. These
    # rows are therefore a COUNTERFACTUAL — what the expansion would have been worth if
    # the new bets had been as good as the old ones — not a forecast.
    out["projection_note"] = (
        "COUNTERFACTUAL: holds per-bet Sharpe fixed at the original panel's 0.2912. "
        "breadth_ladder.py measures that it does not hold. Judge on the ladder.")
    out["projection"] = {
        name: {"n_eff": v, **implied(v, per_bet, efficiency)}
        for name, v in (
            ("original_18_1996", control),
            ("original_18_2011", h["original_18"]["pairwise"]),
            ("expanded_2011", h["expanded_37"]["pairwise"]),
            ("expanded_2020", by_window["2020+"]["expanded_37"]["pairwise"]),
        )
    }

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "neff.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str),
                                    encoding="utf-8")

    # ── print ────────────────────────────────────────────────────────────────
    print("\n=== N_eff BY WINDOW (pairwise / complete-case), matched ===")
    hdr = f"{'universe':26s}" + "".join(f"{w:>16s}" for w in WINDOWS)
    print(hdr)
    for uname in ("original_18", "expanded_37", "expanded_no_livestock",
                  "expanded_no_rollcontam", "expanded_no_vol",
                  "expanded_long_history_only"):
        line = f"{uname:26s}"
        for win in WINDOWS:
            v = by_window[win].get(uname)
            line += (f"{v['pairwise']:7.2f}/{v['complete']:<8.2f}" if v else f"{'-':>16s}")
        print(line)
    print("\n=== PER BLOCK, 2011+ ===")
    print(f"{'block':20s}{'n':>4s}{'alone':>9s}{'orig+blk':>10s}{'marginal':>10s}{'per bet':>9s}")
    for b, rec in blocks.items():
        r = rec["2011+"]
        if r["orig_plus_block_neff"] is None:
            continue
        print(f"{b:20s}{len(rec['instruments']):>4d}"
              f"{(r['block_alone_neff'] or float('nan')):>9.2f}"
              f"{r['orig_plus_block_neff']:>10.2f}{r['marginal_vs_original']:>10.2f}"
              f"{r['efficiency_pct'] / 100.0:>9.2f}")
    print("\n=== WHAT THE N_eff IMPLIES ===")
    print(f"per-bet Sharpe (measured, from S=0.6678 over N_eff=5.26): {per_bet:.4f}")
    print(f"measured efficiency vs S^2/2 (0.1583 / {sharpe_18**2/2:.4f}): {efficiency:.3f}")
    for name, v in out["projection"].items():
        print(f"  {name:22s} N_eff={v['n_eff']:6.2f}  S={v['portfolio_sharpe']:.4f}  "
              f"half-Kelly={v['kelly_half_pct']:6.2f}%  "
              f"expected MEASURED peak={v['expected_measured_peak_pct']:6.2f}%")
    req = out["required_for_30pct"]
    print(f"\nN_eff required for a MEASURED 30%/yr: "
          f"{req['n_eff_half_kelly_measured_efficiency']:.2f} "
          f"(clean half-Kelly arithmetic understates it at "
          f"{req['n_eff_half_kelly']:.2f})")
    print(f"ACHIEVED on the expanded panel (2011+): "
          f"{out['headline']['expanded']:.2f}  -> SHORTFALL "
          f"{req['n_eff_half_kelly_measured_efficiency'] - out['headline']['expanded']:.2f}")
    return out


if __name__ == "__main__":
    main()
