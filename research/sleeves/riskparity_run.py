"""Run the pre-registered RISK PARITY study ONCE and write ``_riskparity/result.json``.

Pre-registration: ``research/sleeves/riskparity_prereg.md`` (commit ``d895110``).
Nothing here is searched. Every constant comes from the prereg or from the trend sleeve.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves.multiasset_trend import (
    BLOCKS,
    MONTHS,
    PRIMARY_UNIVERSE,
    TrendConfig,
    active_report,
    annual_sharpe,
    concentration,
    decade_sharpe,
    inverse_vol,
    load_excess_panel,
    run_trend,
)
from research.sleeves.riskparity import (
    BOND_BULL,
    COSTS,
    FINANCING,
    LEGACY_FLAT,
    RATES_KEYS,
    VOL_TARGETS,
    build_book,
    drawdown_report,
    ladder,
    levered,
    weight_concentration,
)

_OUT = Path("research/sleeves/_riskparity")
SCHEMES = ("ew", "rp_naive", "rp_bucket")
HEAD_COST = "10bps"                      # the conservative bound is the headline
HEAD_FIN = "primary_bill_plus_150bp"
N_TRIALS = (32, 46, 56, 304)


def _cash() -> pd.Series:
    return pd.read_parquet("_data/multiasset/cash_monthly.parquet")["US_CASH_13W"]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else round(v, 10)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    out: dict = {"prereg": "research/sleeves/riskparity_prereg.md", "commit_of_prereg": "d895110"}
    cash = _cash()
    x, interior = load_excess_panel()
    books = {s: build_book(s, x, interior, s) for s in SCHEMES}

    span = books["ew"].excess.dropna()
    years = len(span) / MONTHS
    out["sample"] = {
        "first": str(span.index[0].date()), "last": str(span.index[-1].date()),
        "months": int(len(span)), "years": years,
        "n_instruments": len(PRIMARY_UNIVERSE),
        "mean_eligible": float(books["ew"].elig_count.reindex(span.index).mean()),
    }

    # ── unlevered books, both cost brackets ───────────────────────────────────
    one = pd.Series(1.0, index=x.index)
    unlev: dict = {}
    for s, bk in books.items():
        row: dict[str, Any] = {}
        for cl, c in COSTS.items():
            L = levered(bk, cash, tau=0.0, cost=c, spread=0.0, k_override=one)
            row[cl] = {
                "sharpe": annual_sharpe(L["net_excess"]),
                "mean_excess_annual": float(L["net_excess"].mean() * MONTHS),
                "vol_annual": float(L["net_excess"].std(ddof=1) * math.sqrt(MONTHS)),
                "max_drawdown_total": drawdown_report(L["total"])["max_drawdown"],
                "compound_annual_total": drawdown_report(L["total"])["compound_annual"],
                "turnover_per_year": float(L["turnover"].mean() * MONTHS),
                "financing_charged": float(L["financing"].abs().sum()),
            }
        row["gross_sharpe"] = annual_sharpe(bk.excess)
        unlev[s] = row
    out["unlevered"] = unlev

    # the pre-declared "better of W1/W2" -- decided mechanically on net 10bps Sharpe
    rp_key = max(("rp_naive", "rp_bucket"), key=lambda s: unlev[s][HEAD_COST]["sharpe"])
    out["rp_selected"] = rp_key

    # ── the leverage ladder, both books, both cost brackets ───────────────────
    out["ladder"] = {
        s: {cl: ladder(books[s], cash, cost_label=cl, financing_label=HEAD_FIN)
            for cl in COSTS}
        for s in SCHEMES
    }

    # ── financing sensitivity at every rung (headline cost bracket) ───────────
    fin_sens: dict = {}
    for label, spread in FINANCING.items():
        fin_sens[label] = {}
        for tau in VOL_TARGETS:
            L = levered(books[rp_key], cash, tau=tau, cost=COSTS[HEAD_COST], spread=spread)
            d = drawdown_report(L["total"])
            fin_sens[label][f"{tau:.2f}"] = {
                "compound_annual": d["compound_annual"], "max_drawdown": d["max_drawdown"],
                "sharpe_net": annual_sharpe(L["net_excess"]),
                "financing_drag_annual": float(L["financing"].mean() * MONTHS),
            }
    fin_sens["legacy_flat_6pct"] = {}
    for tau in VOL_TARGETS:
        L = levered(books[rp_key], cash, tau=tau, cost=COSTS[HEAD_COST],
                    spread=None, flat_rate=LEGACY_FLAT)
        d = drawdown_report(L["total"])
        fin_sens["legacy_flat_6pct"][f"{tau:.2f}"] = {
            "compound_annual": d["compound_annual"], "max_drawdown": d["max_drawdown"],
            "sharpe_net": annual_sharpe(L["net_excess"]),
            "financing_drag_annual": float(L["financing"].mean() * MONTHS),
        }
    out["financing_sensitivity"] = fin_sens
    out["mean_bill_rate_annual"] = float(cash.reindex(span.index).mean() * MONTHS)

    # ── THE HEADLINE: highest compound return at a survivable drawdown ────────
    surv: dict = {}
    for limit in (0.35, 0.50, 0.60):
        best = None
        for s in SCHEMES:
            for tau, row in out["ladder"][s][HEAD_COST].items():
                if row["max_drawdown"] >= -limit and not row["ruin"]:
                    if best is None or row["compound_annual"] > best["compound_annual"]:
                        best = {"book": s, "target_vol": float(tau), **row}
        surv[f"mdd_le_{int(limit * 100)}pct"] = best
    out["survivable"] = surv

    # supplementary, DESCRIPTIVE ONLY (leverage is a pure scaling, not a searched
    # parameter): the exact vol target at which the drawdown constraint binds.
    fine: dict = {}
    for s in SCHEMES:
        rows = []
        for tau_v in np.arange(0.05, 0.605, 0.01):
            L = levered(books[s], cash, tau=float(tau_v), cost=COSTS[HEAD_COST],
                        spread=FINANCING[HEAD_FIN])
            d = drawdown_report(L["total"])
            rows.append((float(tau_v), d["compound_annual"], d["max_drawdown"],
                         d["ruin"]))
        ok = [r for r in rows if r[2] >= -0.50 and not r[3]]
        fine[s] = {
            "boundary_tau_mdd_50": max((r[0] for r in ok), default=None),
            "compound_at_boundary": max(((r[1], r[0]) for r in ok), default=(None, None))[0],
            "mdd_at_boundary": next((r[2] for r in rows
                                     if ok and abs(r[0] - max(q[0] for q in ok)) < 1e-9), None),
            "curve": [{"tau": r[0], "compound": r[1], "mdd": r[2], "ruin": r[3]} for r in rows],
        }
    out["fine_grid"] = fine

    # ── matched-volatility comparison, RP vs EW at the same target ────────────
    matched: dict = {}
    for tau in VOL_TARGETS:
        Lr = levered(books[rp_key], cash, tau=tau, cost=COSTS[HEAD_COST],
                     spread=FINANCING[HEAD_FIN])
        Le = levered(books["ew"], cash, tau=tau, cost=COSTS[HEAD_COST],
                     spread=FINANCING[HEAD_FIN])
        matched[f"{tau:.2f}"] = active_report(Lr["net_excess"], Le["net_excess"])
    out["matched_vol_active"] = matched

    # unlevered matched comparison too (no financing in either leg)
    Lr1 = levered(books[rp_key], cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0,
                  k_override=one)
    Le1 = levered(books["ew"], cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0,
                  k_override=one)
    out["unlevered_active"] = active_report(Lr1["net_excess"], Le1["net_excess"])

    # breakeven round-trip cost for the RP-vs-EW vol-matched active return
    grid = []
    for bp in range(0, 201):
        c = bp * 1e-4
        a = levered(books[rp_key], cash, tau=0.0, cost=c, spread=0.0, k_override=one)
        b = levered(books["ew"], cash, tau=0.0, cost=c, spread=0.0, k_override=one)
        grid.append((c, active_report(a["net_excess"], b["net_excess"])["volmatched_active_annual"]))
    be = next((grid[i][0] for i in range(1, len(grid))
               if grid[i][1] * grid[0][1] < 0), None)
    out["breakeven_cost_roundtrip"] = be

    # ── DSR gate, applied to RP *and* to the benchmark ────────────────────────
    bars = {str(n): dsr_sharpe_bar(years, n_trials=n) for n in N_TRIALS}
    out["dsr"] = {
        "years": years, "bars": bars,
        "sharpe_needed_for_30pct_half_kelly": math.sqrt(0.30 * 8.0 / 3.0),
        "clears": {
            s: {cl: {str(n): bool(unlev[s][cl]["sharpe"] >= bars[str(n)]) for n in N_TRIALS}
                for cl in COSTS}
            for s in SCHEMES
        },
    }

    # ── decades ───────────────────────────────────────────────────────────────
    out["decades"] = {
        s: decade_sharpe(
            levered(books[s], cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0,
                    k_override=one)["net_excess"]
        )
        for s in SCHEMES
    }
    ec = books["ew"].elig_count.reindex(span.index)
    out["mean_eligible_by_decade"] = {
        f"{int(d)}s": float(g.mean()) for d, g in ec.groupby((ec.index.year // 10) * 10)
    }

    # ── THE BOND BULL MARKET ──────────────────────────────────────────────────
    bb: dict = {}
    lo, hi = pd.Timestamp(BOND_BULL[0]), pd.Timestamp(BOND_BULL[1])
    for s in SCHEMES:
        L = levered(books[s], cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0, k_override=one)
        ne, tot = L["net_excess"], L["total"]
        inside = (ne.index >= lo) & (ne.index <= hi)
        segs = {
            "full": (ne, tot),
            "excl_bond_bull": (ne[~inside], tot[~inside]),
            "inside_bond_bull": (ne[inside], tot[inside]),
            "pre_1981_10": (ne[ne.index < lo], tot[tot.index < lo]),
            "post_2021_12": (ne[ne.index > hi], tot[tot.index > hi]),
        }
        bb[s] = {
            k: {
                "months": int(len(v[0])),
                "sharpe": annual_sharpe(v[0]),
                "mean_excess_annual": float(v[0].mean() * MONTHS) if len(v[0]) else None,
                "compound_annual_spliced": drawdown_report(v[1])["compound_annual"],
                "max_drawdown_spliced": drawdown_report(v[1])["max_drawdown"],
            }
            for k, v in segs.items()
        }
    out["bond_bull"] = bb

    # bond instruments' own excess-return record, for attribution
    out["rates_excess_by_segment"] = {}
    for key in RATES_KEYS:
        r = x[key].dropna()
        ins = (r.index >= lo) & (r.index <= hi)
        out["rates_excess_by_segment"][key] = {
            "full_sharpe": annual_sharpe(r), "full_mean_annual": float(r.mean() * MONTHS),
            "inside_sharpe": annual_sharpe(r[ins]),
            "inside_mean_annual": float(r[ins].mean() * MONTHS),
            "outside_sharpe": annual_sharpe(r[~ins]),
            "outside_mean_annual": float(r[~ins].mean() * MONTHS),
        }

    # ── rates removed entirely (15 instruments, full pipeline re-run) ─────────
    uni15 = tuple(k for k in PRIMARY_UNIVERSE if k not in RATES_KEYS)
    blocks15 = {b: v for b, v in BLOCKS.items() if b != "rates"}
    x15, int15 = load_excess_panel(universe=uni15)
    nr: dict = {}
    for s in SCHEMES:
        bk15 = build_book(s, x15, int15, s, blocks=blocks15)
        L = levered(bk15, cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0, k_override=one)
        d = drawdown_report(L["total"])
        nr[s] = {
            "months": int(len(L["net_excess"])),
            "first": str(L["net_excess"].index[0].date()),
            "sharpe": annual_sharpe(L["net_excess"]),
            "compound_annual": d["compound_annual"], "max_drawdown": d["max_drawdown"],
            "vol_annual": d["vol_annual"],
            "ladder_10bps": ladder(bk15, cash, cost_label=HEAD_COST, financing_label=HEAD_FIN),
        }
    out["rates_excluded"] = nr

    # ── concentration ─────────────────────────────────────────────────────────
    out["weight_concentration"] = {
        s: weight_concentration(books[s].w, books[s].live) for s in SCHEMES
    }
    conc_dec: dict = {}
    for s in SCHEMES:
        ww = books[s].w.loc[books[s].live.reindex(books[s].w.index).fillna(False)]
        ww = ww.loc[ww.abs().sum(axis=1) > 0]
        arr = np.sort(ww.to_numpy(), axis=1)[:, ::-1]
        t3 = pd.Series(arr[:, :3].sum(axis=1), index=ww.index)
        conc_dec[s] = {f"{int(d)}s": float(g.mean())
                       for d, g in t3.groupby((t3.index.year // 10) * 10)}
    out["top3_share_by_decade"] = conc_dec

    # ── P&L concentration at the headline rung ────────────────────────────────
    pnl_conc: dict = {}
    for s in SCHEMES:
        L = levered(books[s], cash, tau=0.20, cost=COSTS[HEAD_COST],
                    spread=FINANCING[HEAD_FIN])
        pnl_conc[s] = concentration(L["pnl"].fillna(0.0))
        by_block = {
            b: float(L["pnl"][[k for k in keys if k in L["pnl"].columns]].sum().sum()
                     / L["pnl"].sum().sum())
            for b, keys in BLOCKS.items()
        }
        pnl_conc[s]["block_share"] = by_block
    out["pnl_concentration"] = pnl_conc

    # ── verification controls (prereg §8) ─────────────────────────────────────
    ver: dict = {}
    tr = run_trend(TrendConfig(), vol_target=0.10)
    ver["trend_benchmark_reproduction"] = {
        cl: annual_sharpe(v) for cl, v in tr.bench_net.items()
    }
    ver["trend_benchmark_gross"] = annual_sharpe(tr.bench_gross)
    ver["our_ew"] = {cl: unlev["ew"][cl]["sharpe"] for cl in COSTS}
    ver["ew_matches_recorded_0.7065"] = bool(
        abs(unlev["ew"]["2bps"]["sharpe"] - 0.7065) <= 0.03
        or abs(unlev["ew"]["10bps"]["sharpe"] - 0.7065) <= 0.03
    )

    gross_by_tau = {
        s: [out["ladder"][s][HEAD_COST][f"{t:.2f}"]["sharpe_gross"] for t in VOL_TARGETS]
        for s in SCHEMES
    }
    ver["leverage_invariance_gross_sharpe"] = gross_by_tau
    ver["leverage_invariance_ok"] = {
        s: bool(np.nanmax(v) - np.nanmin(v) < 1e-9) for s, v in gross_by_tau.items()
    }
    ver["cap_binding_months_by_tau"] = {
        s: {f"{t:.2f}": out["ladder"][s][HEAD_COST][f"{t:.2f}"]["cap_binding_months"]
            for t in VOL_TARGETS}
        for s in SCHEMES
    }

    # point-in-time audit: weights rebuilt on a panel truncated at t must match row t
    audit_dates = list(span.index[::92])[:10]
    pit_max = 0.0
    for d in audit_dates:
        xt, it = x.loc[:d], interior.loc[:d]
        for s in SCHEMES:
            bkt = build_book(s, xt, it, s)
            diff = float((bkt.w.loc[d] - books[s].w.loc[d]).abs().max())
            pit_max = max(pit_max, diff)
    ver["pit_audit_max_weight_diff"] = pit_max
    ver["pit_audit_dates"] = [str(d.date()) for d in audit_dates]
    ver["pit_audit_ok"] = bool(pit_max < 1e-12)

    # sign-flip control: volatility is sign-invariant, so the WEIGHTS must be identical
    # and the book's Sharpe must collapse -- proving the return is the assets' risk
    # premia and not the sizing machinery.
    flips = []
    for seed in (11, 23, 47, 91):
        rng = np.random.default_rng(seed)
        sgn = pd.Series(rng.choice([-1.0, 1.0], size=x.shape[1]), index=x.columns)
        xf = x.mul(sgn, axis=1)
        bkf = build_book(rp_key, xf, interior, rp_key)
        wdiff = float((bkf.w - books[rp_key].w).abs().max().max())
        Lf = levered(bkf, cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0, k_override=one)
        flips.append({"seed": seed, "weight_max_diff": wdiff,
                      "sharpe": annual_sharpe(Lf["net_excess"])})
    ver["sign_flip_control"] = flips
    ver["sign_flip_weights_identical"] = bool(max(f["weight_max_diff"] for f in flips) < 1e-12)
    ver["sign_flip_mean_sharpe"] = float(np.mean([f["sharpe"] for f in flips]))

    # financing arithmetic at k == 1
    Lk1 = levered(books[rp_key], cash, tau=0.0, cost=COSTS[HEAD_COST],
                  spread=FINANCING[HEAD_FIN], k_override=one)
    ver["financing_zero_at_k1"] = float(Lk1["financing"].abs().max())
    raw = books[rp_key].excess.reindex(Lk1["net_excess"].index)
    ver["k1_net_excess_identity_max_err"] = float(
        (Lk1["net_excess"] - (raw - Lk1["cost"].reindex(raw.index))).abs().max()
    )
    ver["weights_sum_to_one_max_err"] = {
        s: float((books[s].w.sum(axis=1)
                  .loc[books[s].w.abs().sum(axis=1) > 0] - 1.0).abs().max())
        for s in SCHEMES
    }
    ver["dsr_anchor_7yr_n32"] = dsr_sharpe_bar(7.0, n_trials=32)
    ver["dsr_anchor_40yr_n32"] = dsr_sharpe_bar(40.0, n_trials=32)

    # leverage invariance with the 10x gross cap LIFTED -- the cap is the only thing that
    # can break it, so this is the decisive form of the control.
    nocap = {
        s: [annual_sharpe(levered(books[s], cash, tau=t, cost=0.0, spread=0.0,
                                  cap=1e9)["gross_excess"]) for t in VOL_TARGETS]
        for s in SCHEMES
    }
    ver["leverage_invariance_nocap"] = nocap
    ver["leverage_invariance_nocap_ok"] = {
        s: bool(np.nanmax(v) - np.nanmin(v) < 1e-9) for s, v in nocap.items()
    }

    # Why our monthly EW book is 0.668 and the recorded benchmark is 0.7065: the recorded
    # figure comes from a DAILY-rebalanced equal-weight book. Rebuild that here.
    dr = pd.read_parquet("_data/multiasset/returns_daily.parquet")
    dc = pd.read_parquet("_data/multiasset/cash_daily.parquet")["US_CASH_13W"]
    dx = dr.loc[:, list(PRIMARY_UNIVERSE)].copy()
    for key in ("US5Y_TR", "US10Y_TR", "US30Y_TR"):
        dx[key] = dx[key] - dc.reindex(dx.index)
    elig_m = (books["ew"].w.abs() > 0)
    elig_d = elig_m.reindex(dx.index, method="ffill").fillna(False)
    nd = elig_d.sum(axis=1)
    wd = elig_d.astype(float).div(nd.replace(0, np.nan), axis=0).fillna(0.0)
    rd = (wd * dx.fillna(0.0)).sum(axis=1).where(nd > 0)
    m_from_daily = (1.0 + rd.dropna()).groupby(
        pd.Grouper(freq="ME")).prod().sub(1.0)
    m_from_daily = m_from_daily.loc[books["ew"].excess.dropna().index[0]:]
    ver["ew_daily_rebalanced_sharpe"] = annual_sharpe(m_from_daily)
    ver["ew_daily_rebalanced_months"] = int(len(m_from_daily))
    ver["ew_monthly_vs_daily_gap"] = float(
        annual_sharpe(m_from_daily) - unlev["ew"]["10bps"]["sharpe"])
    ver["ew_daily_matches_recorded_0.7065"] = bool(
        abs(annual_sharpe(m_from_daily) - 0.7065) <= 0.03)
    out["verification"] = ver

    # ── diagnostics (reporting only; no strategy parameter is changed) ────────
    diag: dict = {}

    # 1. return attribution at every rung of the headline book and of EW
    attr: dict = {}
    for s in ("ew", rp_key):
        attr[s] = {}
        for tau in VOL_TARGETS:
            L = levered(books[s], cash, tau=tau, cost=COSTS[HEAD_COST],
                        spread=FINANCING[HEAD_FIN])
            idx = L["net_excess"].index
            d = drawdown_report(L["total"])
            arith = float(L["total"].mean() * MONTHS)
            attr[s][f"{tau:.2f}"] = {
                "cash_contribution": float(cash.reindex(idx).mean() * MONTHS),
                "levered_gross_excess": float(L["gross_excess"].mean() * MONTHS),
                "trading_cost": -float(L["cost"].mean() * MONTHS),
                "financing": -float(L["financing"].mean() * MONTHS),
                "arithmetic_total": arith,
                "compound_total": d["compound_annual"],
                "variance_drag": arith - d["compound_annual"],
            }
    diag["attribution"] = attr

    # 2. compound return BY DECADE at the survivable rung, with the bill rate
    dec_comp: dict = {}
    for s in ("ew", rp_key):
        L = levered(books[s], cash, tau=0.15, cost=COSTS[HEAD_COST],
                    spread=FINANCING[HEAD_FIN])
        t = L["total"]
        dec_comp[s] = {}
        for d10, g in t.groupby((t.index.year // 10) * 10):
            n = len(g)
            dec_comp[s][f"{int(d10)}s"] = {
                "months": int(n),
                "compound_annual": float((1.0 + g).prod() ** (MONTHS / n) - 1.0),
                "max_drawdown": drawdown_report(g)["max_drawdown"],
            }
    cs = cash.reindex(span.index)
    dec_comp["bill_rate"] = {
        f"{int(d10)}s": float(g.mean() * MONTHS)
        for d10, g in cs.groupby((cs.index.year // 10) * 10)
    }
    diag["compound_by_decade_tau15"] = dec_comp

    # 3. why RP loses to EW at matched vol: a ZERO-financing counterfactual.
    #    NOT a deployable assumption -- leverage is never free. It exists only to
    #    decompose the gap.
    zero_fin: dict = {}
    for tau in VOL_TARGETS:
        a = levered(books[rp_key], cash, tau=tau, cost=COSTS[HEAD_COST], spread=0.0)
        b = levered(books["ew"], cash, tau=tau, cost=COSTS[HEAD_COST], spread=0.0)
        zero_fin[f"{tau:.2f}"] = {
            "rp_sharpe": annual_sharpe(a["net_excess"]),
            "ew_sharpe": annual_sharpe(b["net_excess"]),
            "volmatched_active_annual":
                active_report(a["net_excess"], b["net_excess"])["volmatched_active_annual"],
            "volmatched_active_tstat":
                active_report(a["net_excess"], b["net_excess"])["volmatched_active_tstat"],
        }
    diag["diagnostic_zero_financing"] = zero_fin

    # vol-matched active at ZERO trading cost (unlevered), since the breakeven scan
    # found no crossing
    a0 = levered(books[rp_key], cash, tau=0.0, cost=0.0, spread=0.0, k_override=one)
    b0 = levered(books["ew"], cash, tau=0.0, cost=0.0, spread=0.0, k_override=one)
    diag["unlevered_active_zero_cost"] = active_report(a0["net_excess"], b0["net_excess"])

    # 4. block detail for the bucketed book -- the FX block is internally hedged
    W = books["rp_bucket"].extra["block_weights"]
    rb = books["rp_bucket"].extra["block_returns"]
    sig_b = rb.rolling(36, min_periods=12).std(ddof=1) * math.sqrt(MONTHS)
    live_rows = books["rp_bucket"].live.reindex(W.index).fillna(False)
    diag["block_detail"] = {
        b: {
            "mean_block_weight": float(W.loc[live_rows, b].mean()),
            "mean_block_vol": float(sig_b.loc[live_rows, b].mean()),
            "block_sharpe": annual_sharpe(rb[b]),
            "block_mean_excess_annual": float(rb[b].mean() * MONTHS),
            "sum_of_member_vols": float(
                inverse_vol(x)[[k for k in BLOCKS[b] if k in x.columns]]
                .loc[live_rows].mean(axis=1).mean()),
        }
        for b in BLOCKS
    }

    # 5. correlation-based effective N -- four FX pairs that are all the dollar
    from research.sleeves.multiasset_trend import effective_n
    corr_all = x.loc["1996":].corr()
    diag["effective_n_correlation"] = {
        "universe_18": effective_n(corr_all),
        **{f"block_{b}": effective_n(x.loc["1996":, [k for k in keys if k in x.columns]].corr())
           for b, keys in BLOCKS.items()},
    }

    # 6. rates-excluded vs full universe on the SAME window
    start15 = pd.Timestamp(nr["ew"]["first"])
    same: dict = {}
    for s in SCHEMES:
        L = levered(books[s], cash, tau=0.0, cost=COSTS[HEAD_COST], spread=0.0,
                    k_override=one)
        ne = L["net_excess"].loc[start15:]
        same[s] = {"sharpe_full_universe_same_window": annual_sharpe(ne),
                   "sharpe_no_rates": nr[s]["sharpe"], "months": int(len(ne))}
    diag["rates_excluded_matched_window"] = same
    out["diagnostics"] = diag

    _OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(out), indent=2, sort_keys=True)
    (_OUT / "result.json").write_text(payload, encoding="utf-8")
    print("md5", hashlib.md5(payload.encode()).hexdigest())
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps(_jsonable({
        "sample": r["sample"],
        "unlevered": {k: v[HEAD_COST]["sharpe"] for k, v in r["unlevered"].items()},
        "survivable": r["survivable"],
        "verification_ok": {
            k: v for k, v in r["verification"].items() if k.endswith("_ok")
        },
    }), indent=2))
