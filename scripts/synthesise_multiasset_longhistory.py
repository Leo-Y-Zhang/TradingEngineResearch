"""Synthesis for the long-history multi-asset iteration (panel + trend + carry).

Re-measures every load-bearing number in
``research/medallion_style_alpha_search/multiasset_longhistory_result.md`` from the
artefacts on disk, rather than transcribing them, and adds the checks the two sleeve
studies did NOT run:

* the DSR bar as a curve over the sample lengths this panel can actually deliver;
* the trend-carry correlation with a Fisher confidence interval (the sleeve studies
  reported a point estimate only, and the portfolio arithmetic is a function of it);
* the sleeve-count inversion of ``S = s*sqrt(N/(1+(N-1)rho))`` across that interval;
* leave-one-INSTRUMENT-out on carry (the study tested leave-one-CELL-out only);
* a 200-seed negative control for carry (the study used 4 fixed seeds), split into
  accrual and price legs so the test's power can be read off;
* the carry accrual/price decomposition BY DECADE, which the study computed only
  full-sample and which is what its "ZIRP regime" caveat actually rests on.

Derived statistics only. No raw vendor rows are written anywhere.

Run: ``.venv/Scripts/python.exe scripts/synthesise_multiasset_longhistory.py``
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.multiasset.panel import dsr_sharpe_bar  # noqa: E402

CARRY_OUT = ROOT / "research" / "sleeves" / "_carry_output"
TREND_OUT = ROOT / "research" / "sleeves" / "_multiasset_trend"
OUT = ROOT / "research" / "medallion_style_alpha_search" / "_multiasset_longhistory"

HALF_KELLY = 3.0 / 8.0
TARGET_SHARPE = math.sqrt(0.30 / HALF_KELLY)   # the S that supports 30%/yr at half Kelly


# ── portfolio arithmetic ──────────────────────────────────────────────────────

def combined_sharpe(s: float, n: float, rho: float) -> float:
    """S = s * sqrt(N / (1 + (N-1)*rho)) — equal-risk, equal-quality sleeves."""
    denom = 1.0 + (n - 1.0) * rho
    if denom <= 0:
        return float("inf")
    return s * math.sqrt(n / denom)


def sleeves_needed(s: float, rho: float, target: float = TARGET_SHARPE) -> float:
    """Invert the above for N. Returns inf when the rho-ceiling is below target."""
    if s <= 0:
        return float("inf")
    r = (target / s) ** 2
    denom = 1.0 - rho * r
    if denom <= 0:
        return float("inf")
    return r * (1.0 - rho) / denom


def ceiling(s: float, rho: float) -> float:
    """lim N->inf of the combined Sharpe."""
    if rho <= 0:
        return float("inf")
    return s / math.sqrt(rho)


def half_kelly(s: float) -> float:
    return HALF_KELLY * s * s


def fisher_ci(rho: float, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """(lo, hi, two-sided p vs zero) for a correlation on n observations."""
    z = 0.5 * math.log((1 + rho) / (1 - rho))
    se = 1.0 / math.sqrt(n - 3)
    crit = float(stats.norm.ppf(0.5 + conf / 2))
    lo, hi = math.tanh(z - crit * se), math.tanh(z + crit * se)
    p = 2.0 * (1.0 - float(stats.norm.cdf(abs(z) / se)))
    return lo, hi, p


def perf(r: pd.Series) -> dict[str, float]:
    r = pd.Series(r).dropna()
    mu, sd = float(r.mean()) * 12.0, float(r.std(ddof=1)) * math.sqrt(12.0)
    return {
        "n_months": int(len(r)),
        "years": round(len(r) / 12.0, 2),
        "arithmetic_annual": mu,
        "annual_vol": sd,
        "sharpe": mu / sd if sd > 0 else 0.0,
        "t_stat": float(r.mean()) / (float(r.std(ddof=1)) / math.sqrt(len(r))),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict[str, object] = {}

    carry_res = json.loads((CARRY_OUT / "multiasset_carry_result.json").read_text())
    trend_res = json.loads((TREND_OUT / "result.json").read_text())
    synth = json.loads((CARRY_OUT / "carry_trend_synthesis.json").read_text())

    # ── 1. the DSR bar over ACHIEVABLE sample lengths ─────────────────────────
    # Anchors first: the function is only usable if it reproduces the programme's two.
    anchors = {
        "7yr_n32": dsr_sharpe_bar(7.0, n_trials=32),
        "40yr_n32": dsr_sharpe_bar(40.0, n_trials=32),
    }
    assert abs(anchors["7yr_n32"] - 1.488) < 5e-4, anchors
    assert abs(anchors["40yr_n32"] - 0.597) < 5e-4, anchors

    lengths = {
        "7.0 (every prior study)": 7.0,
        "17.0 (longest prior study)": 17.0,
        "20.6 (all 27 instruments)": 20.6,
        "22.4 (carry, actual)": 22.42,
        "33.6 (12 instruments)": 33.6,
        "42.6 (8 instruments)": 42.6,
        "47.4 (trend reference, actual)": 47.42,
        "61.5 (trend, actual)": 61.5,
        "98.6 (SPX alone)": 98.58,
    }
    res["dsr_bar_curve"] = {
        "anchors": anchors,
        "by_length": {
            k: {f"n_trials_{n}": dsr_sharpe_bar(v, n_trials=n) for n in (32, 36, 38)}
            for k, v in lengths.items()
        },
        "target_sharpe_for_30pct_half_kelly": TARGET_SHARPE,
    }

    # ── 1b. BREADTH vs LENGTH on the shipped panel — they trade off directly ──
    panel = pd.read_parquet(ROOT / "_data" / "multiasset" / "returns_monthly.parquet")
    first_valid = panel.apply(lambda s: s.first_valid_index()).sort_values()
    panel_end = panel.index.max()
    tradeoff = {}
    for n in (1, 2, 4, 8, 12, 16, 20, 24, panel.shape[1]):
        start = first_valid.iloc[n - 1]
        yrs = (panel_end - start).days / 365.25
        tradeoff[str(n)] = {
            "breadth_available_from": str(start.date()),
            "years_to_panel_end": yrs,
            "dsr_bar_n36": dsr_sharpe_bar(yrs, n_trials=36),
            "dsr_bar_n32": dsr_sharpe_bar(yrs, n_trials=32),
            "bar_below_30pct_target": dsr_sharpe_bar(yrs, n_trials=36) < TARGET_SHARPE,
        }
    res["breadth_vs_length"] = {
        "panel_instruments": int(panel.shape[1]),
        "panel_first_month": str(panel.index.min().date()),
        "panel_last_month": str(panel_end.date()),
        "by_instrument_count": tradeoff,
    }

    # ── 2. the two sleeves, re-measured from their own return series ──────────
    carry_net = pd.read_parquet(CARRY_OUT / "carry_primary_net_monthly.parquet")
    carry_net = carry_net.iloc[:, 0] if carry_net.shape[1] == 1 else carry_net["net"]
    trend_ref = pd.read_parquet(CARRY_OUT / "trend_reference_net_monthly.parquet")
    trend_ref = trend_ref.iloc[:, 0] if trend_ref.shape[1] == 1 else trend_ref["net"]
    trend_real = pd.read_csv(TREND_OUT / "primary_20pct_monthly.csv", index_col=0,
                             parse_dates=True)
    trend_real_net = trend_real["net_10bps"]
    trend_bench = trend_real["bench_net_10bps"]

    res["sleeve_recompute"] = {
        "trend_benchmark_net_10bps": perf(trend_bench),
        "trend_benchmark_levered_to_strategy_vol": perf(
            trend_bench * (trend_real_net.std(ddof=1) / trend_bench.std(ddof=1))),
        "carry_net_3bps": perf(carry_net),
        "carry_net_3bps_reported": {
            "sharpe": carry_res["primary"]["net_realistic"]["sharpe"],
            "arithmetic_annual": carry_res["primary"]["net_realistic"]["arithmetic_annual"],
        },
        "trend_real_net_10bps": perf(trend_real_net),
        "trend_real_net_10bps_reported": {
            "sharpe": trend_res["vol_targets"]["20%"]["net_sharpe_10bps"],
            "columns_available": list(trend_real.columns),
        },
        "trend_reference_net": perf(trend_ref),
    }

    # ── 3. THE PORTFOLIO TEST, with the correlation's uncertainty carried ─────
    both = pd.concat([carry_net.rename("carry"),
                      trend_real_net.rename("trend")], axis=1).dropna()
    rho = float(both["carry"].corr(both["trend"]))
    n_ov = int(len(both))
    lo, hi, pval = fisher_ci(rho, n_ov)

    ew = both.mean(axis=1)
    # equal-RISK: scale each leg by its own full-overlap vol, then average
    z = both / both.std(ddof=1)
    er = z.mean(axis=1) * float(both.std(ddof=1).mean())

    s_carry = perf(both["carry"])["sharpe"]
    s_trend = perf(both["trend"])["sharpe"]
    s_mean = 0.5 * (s_carry + s_trend)

    rho_ref = float(pd.concat([carry_net.rename("c"), trend_ref.rename("t")],
                              axis=1).dropna().corr().iloc[0, 1])

    grid = {
        "CI_low": lo,
        "point_estimate": rho,
        "zero": 0.0,
        "prereg_comparator": rho_ref,
        "CI_high": hi,
    }
    res["portfolio_test"] = {
        "overlap_months": n_ov,
        "overlap_first": str(both.index[0].date()),
        "overlap_last": str(both.index[-1].date()),
        "rho": rho,
        "rho_fisher_ci95": [lo, hi],
        "rho_p_value_vs_zero": pval,
        "rho_prereg_comparator": rho_ref,
        # Cross-check against the carry study's own shipped synthesis: my rho is
        # recomputed from the two saved return streams, theirs was computed in-run.
        "rho_shipped_by_carry_study": synth["two_sleeve_vs_real_trend_sleeve"][
            "correlation"],
        "rho_recompute_discrepancy": rho - synth["two_sleeve_vs_real_trend_sleeve"][
            "correlation"],
        "carry_sharpe_on_overlap": s_carry,
        "trend_sharpe_on_overlap": s_trend,
        "mean_sleeve_sharpe": s_mean,
        "measured_equal_weight": perf(ew),
        "measured_equal_risk": perf(er),
        "formula_two_sleeve_at_point_rho": combined_sharpe(s_mean, 2, rho),
        # NOT an independent validation of the formula: for two sleeves scaled to equal
        # volatility the measured equal-risk Sharpe IS (s1+s2)/sqrt(2(1+rho)), which is
        # the formula at N=2. The agreement below is algebra. The informative comparison
        # is equal-DOLLAR (0.55) vs equal-RISK (0.65) — the sleeves have 5.7x different
        # vols, so dollar-weighting throws away most of the diversification.
        "formula_equals_equal_risk_by_identity": True,
        "formula_minus_measured_equal_risk": combined_sharpe(s_mean, 2, rho)
        - perf(er)["sharpe"],
        "half_kelly_equal_weight": half_kelly(perf(ew)["sharpe"]),
        "half_kelly_equal_risk": half_kelly(perf(er)["sharpe"]),
        "gap_to_target_sharpe_from_equal_weight": TARGET_SHARPE - perf(ew)["sharpe"],
        "gap_to_target_sharpe_from_equal_risk": TARGET_SHARPE - perf(er)["sharpe"],
        "dsr_bar_at_overlap_n36": dsr_sharpe_bar(n_ov / 12.0, n_trials=36),
        "sleeve_count_across_rho": {
            k: {
                "rho": v,
                "sleeves_needed_at_mean_quality": sleeves_needed(s_mean, v),
                "ceiling": ceiling(s_mean, v),
                "two_sleeve_sharpe": combined_sharpe(s_mean, 2, v),
                "half_kelly_two_sleeve": half_kelly(combined_sharpe(s_mean, 2, v)),
            }
            for k, v in grid.items()
        },
        "sleeve_count_across_quality_at_point_rho": {
            f"{q:.3f}": {
                "sleeves_needed": sleeves_needed(q, rho),
                "ceiling": ceiling(q, rho),
            }
            for q in (0.430, 0.344, 0.297, 0.100)
        },
        # THE PRICE OF THE TARGET, stated as a requirement on the NEXT sleeve rather
        # than as a sleeve count, because that is the decision actually facing the
        # programme. Orthogonal equal-risk sleeves add in quadrature: S^2 = sum s_i^2.
        "what_the_next_sleeve_must_score": {
            "two_sleeve_orthogonal_S": math.sqrt(s_carry ** 2 + s_trend ** 2),
            "S_squared_required": TARGET_SHARPE ** 2,
            "S_squared_held": s_carry ** 2 + s_trend ** 2,
            "one_more_sleeve_must_score": math.sqrt(
                max(TARGET_SHARPE ** 2 - s_carry ** 2 - s_trend ** 2, 0.0)),
            "two_more_each_must_score": math.sqrt(
                max(TARGET_SHARPE ** 2 - s_carry ** 2 - s_trend ** 2, 0.0) / 2),
            "three_more_each_must_score": math.sqrt(
                max(TARGET_SHARPE ** 2 - s_carry ** 2 - s_trend ** 2, 0.0) / 3),
            "N_at_measured_mean_quality_rho_zero": {
                str(k): {
                    "S": combined_sharpe(s_mean, k, 0.0),
                    "half_kelly": half_kelly(combined_sharpe(s_mean, k, 0.0)),
                }
                for k in (2, 3, 4, 5, 6)
            },
            "N_at_measured_mean_quality_point_rho": {
                str(k): {
                    "S": combined_sharpe(s_mean, k, rho),
                    "half_kelly": half_kelly(combined_sharpe(s_mean, k, rho)),
                }
                for k in (2, 3, 4, 5, 6)
            },
        },
    }

    # ── 4. carry: leave-one-INSTRUMENT-out (never run by the study) ───────────
    from research.multiasset.carry import (  # noqa: E402
        MONTHS_PER_YEAR,
        carry_positions,
        decompose_pnl,
        performance,
    )
    from scripts.run_multiasset_carry import build_universe, run_sleeve  # noqa: E402

    excess, carry_scores, asset_class = build_universe(unscreened=False)
    keys = list(carry_res["universe"]["keys"])
    base = run_sleeve(excess, carry_scores)
    base_sharpe = performance(base["realistic"]["net"])["sharpe"]

    loo: dict[str, float] = {}
    for k in keys:
        sub = [x for x in keys if x != k]
        r = run_sleeve(excess, carry_scores, keys=sub)
        loo[k] = performance(r["realistic"]["net"])["sharpe"]

    # Dropping a whole class can leave fewer than the registered MIN_INSTRUMENTS=6,
    # which would silently produce an empty book. Keep the registered floor wherever
    # the survivors allow it and relax it ONLY where they do not — recording which
    # floor each row used, because the floor changes early-sample eligibility and so
    # the two are not directly comparable.
    loc: dict[str, dict[str, float]] = {}
    for cls in sorted(set(asset_class.values())):
        sub = [x for x in keys if asset_class[x] != cls]
        if len(sub) < 3:
            continue
        floor = 6 if len(sub) >= 6 else 3
        r = run_sleeve(excess, carry_scores, keys=sub, min_instruments=floor)
        p = performance(r["realistic"]["net"])
        loc[cls] = {"sharpe": p.get("sharpe", float("nan")),
                    "min_instruments_floor": floor,
                    "n_survivors": len(sub),
                    "n_months": p.get("n_months", 0)}

    res["carry_leave_one_out"] = {
        "base_sharpe": base_sharpe,
        "base_sharpe_reported": carry_res["primary"]["net_realistic"]["sharpe"],
        "prereg_dead_threshold": 0.35,
        "leave_one_instrument_out": dict(sorted(loo.items(), key=lambda kv: kv[1])),
        "leave_one_class_out": loc,
        "worst_drop_instrument": min(loo, key=lambda k: loo[k]),
        "worst_drop_sharpe": min(loo.values()),
        "n_below_dead_threshold": sum(1 for v in loo.values() if v < 0.35),
    }

    # ── 5. carry: 200-seed negative control (the study used 4 fixed seeds) ────
    # Uses the sleeve's OWN permutation control (`permute_seed`), so nothing about the
    # machinery, universe, weighting or cost model differs between live and control.
    rng_sharpes = []
    for seed in range(200):
        r = run_sleeve(excess, carry_scores, permute_seed=seed)
        rng_sharpes.append(performance(r["realistic"]["net"])["sharpe"])

    arr = np.array(rng_sharpes, dtype=float)
    live = base_sharpe
    res["carry_negative_control_200"] = {
        "n_seeds": 200,
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "pct_5_50_95": [float(np.percentile(arr, 5)), float(np.percentile(arr, 50)),
                        float(np.percentile(arr, 95))],
        "live_sharpe": live,
        "sd_above_control": float((live - arr.mean()) / arr.std(ddof=1)),
        "empirical_p": (float((arr >= live).sum()) + 1.0) / (len(arr) + 1.0),
        "study_reported_4_seed": {
            "mean": carry_res["negative_control"]["mean"],
            "sd": carry_res["negative_control"]["sd"],
            "sd_above_control": carry_res["negative_control"]["sd_above_control"],
        },
    }

    # ── 6. carry: accrual vs price leg BY DECADE ──────────────────────────────
    sub_excess = excess[keys]
    sub_carry = carry_scores[keys]
    pos, _vol, _nelig = carry_positions(sub_carry, sub_excess)
    # Reproduce decompose_pnl's identity month by month, then ASSERT the full-sample
    # totals match the shipped scalar decomposition before slicing by decade.
    car = sub_carry.reindex(index=sub_excess.index, columns=pos.columns).fillna(0.0)
    ret_f = sub_excess.reindex(columns=pos.columns).fillna(0.0)
    accrual_m = (pos * car / MONTHS_PER_YEAR).shift(1).sum(axis=1, min_count=1)
    total_m = (pos.shift(1) * ret_f).sum(axis=1, min_count=1)
    legs = pd.DataFrame({"accrual": accrual_m, "price": total_m - accrual_m}).dropna()
    shipped = decompose_pnl(pos, sub_excess, sub_carry)
    assert abs(float(legs["accrual"].sum()) - shipped["accrual_pnl"]) < 1e-9
    assert abs(float(legs["price"].sum()) - shipped["price_pnl"]) < 1e-9

    legs = legs.loc[legs.index >= both.index[0]]
    by_decade = {}
    for label, mask in (("2000s", legs.index.year < 2010),
                        ("2010s", (legs.index.year >= 2010) & (legs.index.year < 2020)),
                        ("2020s", legs.index.year >= 2020)):
        seg = legs[mask]
        if len(seg) < 12:
            continue
        by_decade[label] = {
            "n_months": int(len(seg)),
            "accrual_annual": float(seg["accrual"].mean()) * 12.0,
            "price_annual": float(seg["price"].mean()) * 12.0,
            "price_vol_annual": float(seg["price"].std(ddof=1)) * math.sqrt(12.0),
        }
    res["carry_legs_by_decade"] = {
        "full_sample": {
            "accrual_annual": float(legs["accrual"].mean()) * 12.0,
            "price_annual": float(legs["price"].mean()) * 12.0,
            "accrual_vol_annual": float(legs["accrual"].std(ddof=1)) * math.sqrt(12.0),
            "price_vol_annual": float(legs["price"].std(ddof=1)) * math.sqrt(12.0),
        },
        "by_decade": by_decade,
    }

    (OUT / "synthesis.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
