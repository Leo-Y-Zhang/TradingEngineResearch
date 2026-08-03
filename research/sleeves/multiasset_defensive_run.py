"""Run the pre-registered DEFENSIVE / BAB sleeve ONCE and write every receipt.

Six declared arms x three volatility targets x two cost brackets, all reported, no
selection. See ``research/sleeves/multiasset_defensive_prereg.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.multiasset_defensive import (
    COST_BRACKETS,
    N_TRIALS,
    VOL_TARGETS,
    DefensiveConfig,
    combined_sharpe,
    describe,
    kelly_reality,
    realised_beta,
    run_defensive,
    year_concentration,
)
from research.sleeves.multiasset_trend import (
    BLOCKS as BLOCKS_LOCAL,
    MONTHS,
    active_report,
    annual_sharpe,
    concentration,
    decade_sharpe,
    effective_n,
    load_excess_panel,
    newey_west_tstat,
    PRIMARY_UNIVERSE,
)

OUT = Path("research/sleeves/_defensive")
TREND_CSV = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")
CARRY_PARQUET = Path("research/sleeves/_carry_output/carry_primary_net_monthly.parquet")

ARMS: tuple[DefensiveConfig, ...] = (
    DefensiveConfig(name="PRIMARY"),
    DefensiveConfig(name="S1_WITHIN_BLOCK", within_block=True),
    DefensiveConfig(name="S2_HEDGED", hedged=True),
    DefensiveConfig(name="S3_OVERLAP_REMOVED", overlap_lag=12),
    DefensiveConfig(name="S4_UNSCREENED", unscreened=True),
    DefensiveConfig(name="S5_PLACEBO", placebo_seed=20260728),
)


def _bench_levered_active(strat: pd.Series, bench: pd.Series) -> dict[str, float]:
    """The brief's convention: lever the BENCHMARK up to the strategy's own volatility.

    ``active_report`` scales the STRATEGY down to the benchmark's volatility instead.
    The two differ by a positive constant, so the t-stat is identical and only the
    magnitude changes; both are reported so neither convention can flatter anything.
    """
    a, b = strat.align(bench, join="inner")
    sd_s = float(a.std(ddof=1))
    sd_b = float(b.std(ddof=1))
    if sd_b <= 0:
        return {}
    lev = sd_s / sd_b
    d = a - b * lev
    return {
        "bench_leverage_factor": lev,
        "bench_levered_active_annual": float(d.mean() * MONTHS),
        "bench_levered_active_tstat": newey_west_tstat(d),
    }


def _diagnostics(res, x: pd.DataFrame) -> dict:
    d = res.diagnostics.reindex(res.gross.index)
    u = res.unscaled.reindex(res.gross.index)
    gross_abs = u.abs().sum(axis=1)
    short_abs = u.clip(upper=0.0).abs().sum(axis=1)
    on = gross_abs > 0
    w = res.weights.reindex(res.gross.index)

    def leg(prefix: str = "") -> dict[str, Any]:
        return {
            "mean_n_eligible": float(d[prefix + "n_elig"].mean()),
            "mean_betaL": float(d[prefix + "betaL"].mean()),
            "mean_betaH": float(d[prefix + "betaH"].mean()),
            "mean_rho": float(d[prefix + "rho"].mean()),
            "median_rho": float(d[prefix + "rho"].median()),
            "pct_rho_clipped_at_zero": float(
                100.0 * d[prefix + "rho_clipped_low"].fillna(0).mean()),
            "pct_rho_clipped_at_cap": float(
                100.0 * d[prefix + "rho_clipped_high"].fillna(0).mean()),
            "pct_months_off_no_beta_spread": float(
                100.0 * d[prefix + "off_spread"].fillna(0).mean()),
            "mean_exante_book_beta": float(d[prefix + "book_beta"].mean()),
        }

    if "n_elig" in d.columns:
        legs = leg()
    else:
        prefixes = sorted({c.rsplit("_", 1)[0] for c in d.columns if c.endswith("_rho")})
        legs = {"per_block": {p: leg(p + "_") for p in prefixes}}

    return {
        "months_book_on": int(on.sum()),
        "months_total": int(len(res.gross)),
        **legs,
        "short_leg_share_of_gross": float(short_abs[on].sum() / gross_abs[on].sum()),
        "mean_gross_leverage": float(res.gross_leverage.mean()),
        "max_gross_leverage": float(res.gross_leverage.max()),
        "pct_months_gross_cap_binding": float(100.0 * res.cap_binding.mean()),
        "pct_months_no_vol_estimate_full_cap": float(100.0 * res.no_vol_estimate.mean()),
        "turnover_per_year": float(res.turnover.mean() * MONTHS),
        "mean_net_exposure": float(w.sum(axis=1).mean()),
        "gross_leverage_share_by_instrument": {
            c: float(v) for c, v in (w.abs().mean() / w.abs().mean().sum())
            .sort_values(ascending=False).round(4).items()
        },
    }


def run_arm(cfg: DefensiveConfig, x, interior, x_uns, int_uns) -> dict:
    xx, ii = (x_uns, int_uns) if cfg.unscreened else (x, interior)
    out: dict = {"arm": cfg.name, "targets": {}}
    for vt in VOL_TARGETS:
        res = run_defensive(cfg, vol_target=vt, x=xx, interior=ii)
        if len(res.gross) < 24:
            out["targets"][f"{int(vt*100)}pct"] = {"note": "book never turned on"}
            continue
        blk: dict = {
            "gross": describe(res.gross),
            "bench_gross": describe(res.bench_gross),
        }
        for label in COST_BRACKETS:
            blk[f"net_{label}"] = describe(res.net[label])
            blk[f"bench_net_{label}"] = describe(res.bench_net[label])
            blk[f"active_{label}"] = {
                **active_report(res.net[label], res.bench_net[label]),
                **_bench_levered_active(res.net[label], res.bench_net[label]),
            }
        blk["decade_sharpe_net_10bps"] = decade_sharpe(res.net["10bps"])
        blk["decade_sharpe_bench_net_10bps"] = decade_sharpe(res.bench_net["10bps"])
        blk["concentration"] = concentration(res.pnl.loc[res.gross.index])
        blk["year_concentration"] = year_concentration(res.pnl.loc[res.gross.index])
        blk["realised_beta_net_10bps"] = realised_beta(res.net["10bps"], res.proxy)
        blk["realised_beta_bench_net_10bps"] = realised_beta(res.bench_net["10bps"], res.proxy)
        blk["diagnostics"] = _diagnostics(res, xx)
        blk["cost_drag_annual"] = {
            label: float((res.gross - res.net[label]).mean() * MONTHS)
            for label in COST_BRACKETS
        }
        out["targets"][f"{int(vt*100)}pct"] = blk
        if abs(vt - 0.20) < 1e-9:
            out["_res20"] = res
    return out


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    x, interior = load_excess_panel(universe=PRIMARY_UNIVERSE)
    x_uns, int_uns = load_excess_panel(universe=PRIMARY_UNIVERSE, unscreened=True)

    report: dict = {
        "n_trials": N_TRIALS,
        "universe": list(PRIMARY_UNIVERSE),
        "panel": {
            "months": int(len(x)),
            "first": str(x.index.min().date()),
            "last": str(x.index.max().date()),
        },
        "dsr_anchors": {
            "7yr_n32": round(dsr_sharpe_bar(7.0, n_trials=32), 4),
            "40yr_n32": round(dsr_sharpe_bar(40.0, n_trials=32), 4),
            "expected": {"7yr_n32": 1.488, "40yr_n32": 0.597},
        },
        "arms": {},
    }

    res20: dict = {}
    for cfg in ARMS:
        arm = run_arm(cfg, x, interior, x_uns, int_uns)
        res20[cfg.name] = arm.pop("_res20", None)
        report["arms"][cfg.name] = arm

    prim = res20["PRIMARY"]
    s3 = res20["S3_OVERLAP_REMOVED"]

    # ── Ruin check: a levered book that loses >100% in a month is bankrupt, not down ──
    ruin: dict = {}
    for name, arm in report["arms"].items():
        for tgt, blk in arm["targets"].items():
            if "net_10bps" not in blk:
                continue
            ruin[f"{name}@{tgt}"] = {
                "worst_month_net_10bps": blk["net_10bps"]["worst_month"],
                "max_drawdown_net_10bps": blk["net_10bps"]["max_drawdown"],
                "below_minus_100pct": bool(blk["net_10bps"]["worst_month"] <= -1.0),
            }
    report["ruin_check"] = ruin

    # ── Why the within-block arm's betas are not comparable across blocks ─────
    #  The FX block's equal-weight proxy is (EURUSD+GBPUSD+JPYUSD+USDX)/4, and USDX is
    #  approximately MINUS the basket of the other three, so the proxy nearly cancels.
    #  A beta whose denominator is a near-zero variance is not an estimate.
    from research.sleeves.multiasset_defensive import panel_proxy as _pp
    blk_diag = {}
    for name, keys in BLOCKS_LOCAL.items():
        cols = [k for k in keys if k in x.columns]
        pr = _pp(x[cols]).dropna()
        mem = x[cols].loc[pr.index]
        blk_diag[name] = {
            "n_instruments": len(cols),
            "proxy_vol_annual": float(pr.std(ddof=1) * math.sqrt(MONTHS)),
            "mean_member_vol_annual": float(mem.std(ddof=1).mean() * math.sqrt(MONTHS)),
            "proxy_to_member_vol_ratio": float(
                pr.std(ddof=1) / mem.std(ddof=1).mean()),
            "mean_pairwise_member_corr": float(
                mem.corr().where(~np.eye(len(cols), dtype=bool)).stack().mean()),
        }
    report["block_proxy_diagnostics"] = blk_diag

    # ── Is the S4 cleaning-sensitivity arm even capable of saying anything? ───
    #  The quarantine drops a corrupt daily CLOSE, so the two daily returns straddling it
    #  become one valid two-day return. Compounded to a month, that is the SAME number.
    #  If the two month-end panels are identical, S4 is a no-op by construction and no
    #  future month-end sleeve should spend an arm on it.
    diff = (x.fillna(0.0) - x_uns.fillna(0.0)).abs()
    report["s4_is_a_noop"] = {
        "differing_month_cells": int((diff > 1e-12).to_numpy().sum()),
        "max_abs_difference": float(diff.to_numpy().max()),
        "note": (
            "The screened and unscreened MONTH-END panels are identical over this "
            "universe. The quarantine drops a daily level, and the genuine move across "
            "the corrupt print survives as one two-day return, so monthly compounding is "
            "unchanged. Arm S4 therefore cannot distinguish anything and its agreement "
            "with PRIMARY is not evidence of robustness."
        ),
    }

    # ── Correlations to the other two sleeves, and the mechanical-vs-economic test ──
    trend = pd.read_csv(TREND_CSV, parse_dates=["date"]).set_index("date")
    carry = pd.read_parquet(CARRY_PARQUET)["net"]

    def corr_block(series: pd.Series, label: str) -> dict:
        out = {}
        for other_name, other in (("trend", trend["net_10bps"]), ("carry", carry)):
            a, b = series.align(other, join="inner")
            a, b = a.dropna(), b.reindex(a.index)
            out[other_name] = {
                "months": int(len(a)),
                "first": str(a.index.min().date()) if len(a) else None,
                "last": str(a.index.max().date()) if len(a) else None,
                "pearson": float(a.corr(b)),
                "spearman": float(stats.spearmanr(a, b).statistic),
            }
        return {"series": label, **out}

    corr: dict[str, Any] = {
        "primary_net_10bps": corr_block(prim.net["10bps"], "PRIMARY net 10bps"),
        "primary_gross": corr_block(prim.gross, "PRIMARY gross"),
        "s3_overlap_removed_net_10bps": corr_block(
            s3.net["10bps"], "S3 overlap-removed net 10bps"),
    }
    for other in ("trend", "carry"):
        corr[f"delta_{other}_from_overlap_removal"] = float(
            corr["s3_overlap_removed_net_10bps"][other]["pearson"]
            - corr["primary_net_10bps"][other]["pearson"]
        )
    # Does the SHARED vol-targeting machinery alone create co-movement?
    # NOTE: the direct test -- correlating the two books' k(t) scalers -- is NOT
    # implemented, and it is BLOCKED rather than merely skipped. Only `prim` is
    # computed in process and exposes `.scaler`; the trend book arrives from
    # TREND_CSV, whose columns are date, net_10bps, gross, bench_net_10bps -- no
    # scaler is persisted. Answering it properly means re-running the trend book
    # to emit k(t), which is a study, not a diagnostic. What is reported below is
    # the |return| correlation, a PROXY, and the `note` field says so. VERIFY-2.
    tr_r = trend["net_10bps"]
    a, b = prim.net["10bps"].align(tr_r, join="inner")
    corr["primary_vs_trend_abs_return_corr"] = float(a.abs().corr(b.abs()))
    corr["note"] = (
        "S3 lags EVERY position input (beta and the inverse-vol sizing) by 12 months, so "
        "no input to that book has seen the window trend's signal is computed from. The "
        "book-vol scaler k(t) still uses recent realised book returns, which is why the "
        "correlation of |returns| is reported separately."
    )
    report["correlations"] = corr

    # ── Three-sleeve arithmetic on MEASURED correlations ──────────────────────
    dser = prim.net["10bps"]
    idx = dser.index.intersection(trend.index).intersection(carry.index)
    mat = pd.DataFrame({
        "defensive": dser.reindex(idx),
        "trend": trend["net_10bps"].reindex(idx),
        "carry": carry.reindex(idx),
    }).dropna()
    cmat = mat.corr()
    sharpes = {c: annual_sharpe(mat[c]) for c in mat.columns}
    # Equal-RISK blend, measured. Rescaled to a 20% annual volatility so that the
    # geometric return and the drawdown are quantities a book could actually run; the
    # Sharpe is scale-invariant and unaffected.
    def _eq_risk(m: pd.DataFrame, target: float = 0.20) -> pd.Series:
        z = m.div(m.std(ddof=1), axis=1).mean(axis=1)
        return z * (target / (float(z.std(ddof=1)) * math.sqrt(MONTHS)))

    eq = _eq_risk(mat)
    report["three_sleeve"] = {
        "overlap_months": int(len(mat)),
        "overlap_first": str(mat.index.min().date()) if len(mat) else None,
        "overlap_last": str(mat.index.max().date()) if len(mat) else None,
        "correlation_matrix": {a: {b: float(cmat.loc[a, b]) for b in cmat.columns}
                               for a in cmat.index},
        "sharpes_on_overlap": sharpes,
        "effective_n": effective_n(cmat),
        **combined_sharpe(sharpes, cmat),
        "measured_equal_risk_blend": describe(eq),
    }
    report["three_sleeve"]["blend_note"] = (
        "Both blends are equal-RISK (each sleeve divided by its own volatility, then "
        "averaged) and rescaled to 20%/yr volatility. Sharpe is scale-invariant; the "
        "rescaling exists so the geometric return and drawdown are runnable numbers."
    )
    report["three_sleeve"]["measured_two_sleeve_blend_same_window"] = describe(
        _eq_risk(mat[["trend", "carry"]]))
    # What Sharpe would a THIRD sleeve have needed for the trio to reach 30%/yr?
    # Equal-risk: S = sum(s_i)/sqrt(1'C1). Solve for s3 at the measured correlations,
    # and again at the hypothetical rho = 0, holding trend and carry at their measured
    # Sharpes ON THIS OVERLAP (not their headline full-sample numbers).
    TARGET = math.sqrt(8.0 * 0.30 / 3.0)          # half-Kelly g = 3S^2/8 = 30%/yr
    s_tc = sharpes["trend"] + sharpes["carry"]
    r_tc = float(cmat.loc["trend", "carry"])

    def _needed(r_dt: float, r_dc: float) -> float:
        q = math.sqrt(3.0 + 2.0 * (r_tc + r_dt + r_dc))
        return TARGET * q - s_tc

    report["three_sleeve"]["target_sharpe_for_30pct_half_kelly"] = TARGET
    report["three_sleeve"]["required_third_sleeve_sharpe"] = {
        "at_measured_correlations": _needed(
            float(cmat.loc["defensive", "trend"]), float(cmat.loc["defensive", "carry"])),
        "if_uncorrelated_to_both": _needed(0.0, 0.0),
        "defensive_measured_sharpe_on_overlap": sharpes["defensive"],
        "note": (
            "Trend and carry are held at their Sharpes ON THIS 22.4-year overlap "
            f"({sharpes['trend']:.3f} and {sharpes['carry']:.3f}), not their headline "
            "full-sample numbers, because that is the only window all three exist on."
        ),
    }
    report["three_sleeve"]["kelly_reality"] = {
        "trend_plus_carry": kelly_reality(_eq_risk(mat[["trend", "carry"]])),
        "all_three": kelly_reality(eq),
        "defensive_alone": kelly_reality(dser),
        "trend_plus_carry_at_the_30pct_target": {
            "target_sharpe": TARGET,
            "required_vol": TARGET / 2.0,
            "leverage_on_trend_carry_blend": (TARGET / 2.0) / 0.20,
            "implied_max_drawdown": float(
                kelly_reality(_eq_risk(mat[["trend", "carry"]]))["measured_max_drawdown"]
                * ((TARGET / 2.0) / 0.20)),
            "note": (
                "What it would take to run the EXISTING pair at the 30%/yr target "
                "volatility, before any third sleeve exists."
            ),
        },
        "rule": (
            "Programme standing rule 2026-07-28: half-Kelly growth is reported only "
            "with the volatility it requires, the leverage that implies on the "
            "series own volatility, and the measured max drawdown scaled by that "
            "leverage. Drawdown is scaled LINEARLY, which is optimistic."
        ),
    }
    report["three_sleeve"]["measured_pairs"] = {
        f"{a}+{b}": describe(_eq_risk(mat[[a, b]]))["sharpe"]
        for a, b in (("trend", "carry"), ("trend", "defensive"), ("carry", "defensive"))
    }

    # ── Receipts ──────────────────────────────────────────────────────────────
    frame = pd.DataFrame({
        "gross": prim.gross,
        "net_2bps": prim.net["2bps"],
        "net_10bps": prim.net["10bps"],
        "bench_gross": prim.bench_gross,
        "bench_net_10bps": prim.bench_net["10bps"],
    })
    frame.index.name = "date"
    frame.to_csv(OUT / "primary_20pct_monthly.csv")
    pd.DataFrame({"net": prim.net["10bps"]}).to_parquet(
        OUT / "defensive_primary_net_monthly.parquet")
    pd.DataFrame({"net": res20["S1_WITHIN_BLOCK"].net["10bps"]}).to_parquet(
        OUT / "defensive_within_block_net_monthly.parquet")
    prim.diagnostics.to_csv(OUT / "primary_diagnostics.csv")
    prim.beta.round(4).to_csv(OUT / "primary_betas.csv")

    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=str),
                                     encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    r = main()
    print(json.dumps(r["arms"]["PRIMARY"]["targets"]["20pct"]["net_10bps"], indent=2,
                     default=str))
