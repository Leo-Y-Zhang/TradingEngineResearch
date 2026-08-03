"""STANDALONE ADVERSARIAL VERIFICATION OF THE ONE SURVIVOR: `trend + passive`.

Every DEAD candidate in this programme got a dedicated verifier and several died only
because of it. The one survivor never did — its result came from inside the portfolio
study that produced it. This module is that missing verifier, and it assumes the claim is
wrong until the measurements say otherwise.

THE CLAIM UNDER ATTACK
    `trend + passive`, inverse-vol weighted, **Sharpe 0.9033 over 738 months**, clearing
    DSR at n_trials 46/104/304, positive in all seven decades, beating its own benchmark
    at matched volatility by +2.11%/yr (t +2.34) where trend alone loses.

THE ATTACKS
    A1  IS IT JUST PASSIVE?     capital / risk / return decomposition; the marginal
                                contribution of the trend leg and its t-stat; the
                                mean-variance spanning test; a Sharpe-difference
                                bootstrap; and CAUSAL weights, because the published
                                inverse-vol weights are full-sample.
    A2  THE DECADE PROBLEM      Sharpe by decade for the book AND both legs; rolling
                                10-year windows; and the modern-era split.
    A3  THE BOND BULL           bond-bull exclusion; P&L attribution by asset block for
                                BOTH legs; the book with rates removed, as an attribution
                                and as a re-run.
    A4  SURVIVORSHIP            jackknife every instrument; drop the top contributors;
                                add back the two instruments the prereg excluded; and
                                check the claim that validation instruments delisted.
    A5  DATING AND DELISTING    the alignment probe on every series, plus a direct
                                position-lag ladder that trusts no audit.
    A6  THE 12-MONTH LEVERAGE   re-run under the repaired scaler and measure the delta.
    A7  COSTS                   re-price the whole book on a cost grid, price the
                                unpriced equal-weight rebalance leg, and solve breakeven.
    A8  THE CI                  block bootstrap and Lo (2002); P(S<0.894), P(S<0.75).
    A9  RETURN CONVENTIONS      the panel calls seven equity PRICE indices "excess
                                returns". Measure the gap where measurable; solve the
                                breakeven drag.
    A10 THE HONEST NUMBER       the survivable-drawdown compound return after all of it.

    .venv/Scripts/python.exe -m research.sleeves._survivor.survivor_verification
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.alignment import probe_alignment
from research.book_scaler import NO_ESTIMATE_CAP, NO_ESTIMATE_FLAT
from research.multiasset.carry import vol_matched_active
from research.multiasset.panel import dsr_sharpe_bar
from research.trial_ledger import cumulative_trials
from research.sleeves.multiasset_trend import (
    BLOCKS,
    PRIMARY_UNIVERSE,
    TrendConfig,
    _positions,
    load_excess_panel,
    newey_west_tstat,
    run_trend,
)

OUT_DIR = Path(__file__).resolve().parent
DATA = Path("_data/multiasset")
TREND_CSV = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")
MPY = 12
VOL_TARGET = 0.20
COST_HEADLINE = 0.0010
NW_LAG = 6
RNG_SEED = 20260728
N_BOOT = 10_000
BLOCK = 12
BULL_FIRST, BULL_LAST = "1981-10-31", "2021-12-31"
TARGET = 0.894
FINANCING = {"optimistic_bill_plus_50bp": 0.0050,
             "primary_bill_plus_150bp": 0.0150,
             "retail_bill_plus_300bp": 0.0300}


# ── primitives ────────────────────────────────────────────────────────────────
def sharpe(x) -> float:
    a = pd.Series(x).dropna()
    if len(a) < 8 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * math.sqrt(MPY))


def ann_vol(x) -> float:
    return float(pd.Series(x).dropna().std(ddof=1) * math.sqrt(MPY))


def ann_mean(x) -> float:
    return float(pd.Series(x).dropna().mean() * MPY)


def is_ruined(x) -> bool:
    return bool(np.min(np.asarray(x, dtype=float)) <= -1.0)


def max_dd(total) -> float:
    a = np.asarray(total, dtype=float)
    if is_ruined(a):
        return -1.0
    curve = np.cumprod(1.0 + a)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def cagr(total) -> float:
    a = np.asarray(total, dtype=float)
    if is_ruined(a):
        return -1.0
    return float(np.prod(1.0 + a) ** (MPY / a.size) - 1.0)


def inv_vol_weights(frame: pd.DataFrame) -> np.ndarray:
    iv = 1.0 / frame.std(ddof=1).to_numpy()
    return iv / iv.sum()


def book_from(trend: pd.Series, passive: pd.Series) -> tuple[pd.Series, np.ndarray]:
    f = pd.concat({"trend": trend, "passive": passive}, axis=1).dropna()
    w = inv_vol_weights(f)
    return pd.Series(f.to_numpy() @ w, index=f.index), w


def vm(strat: pd.Series, bench: pd.Series) -> dict:
    """The programme's own vol-matched active: the BENCHMARK is scaled to the strategy."""
    r = vol_matched_active(pd.Series(strat), pd.Series(bench), lags=NW_LAG)
    return {"annual": float(r.get("vol_matched_active_annual", float("nan"))),
            "tstat": float(r.get("vol_matched_active_tstat", float("nan")))}


def circular_blocks(n: int, block: int, rng: np.random.Generator, reps: int) -> np.ndarray:
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(reps, nb))
    offs = np.arange(block)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(reps, nb * block) % n
    return idx[:, :n]


def boot_sharpe(x: pd.Series, reps: int = N_BOOT, seed: int = RNG_SEED) -> np.ndarray:
    a = x.dropna().to_numpy(dtype=float)
    s = a[circular_blocks(a.size, BLOCK, np.random.default_rng(seed), reps)]
    sd = s.std(axis=1, ddof=1)
    return s.mean(axis=1) / np.where(sd > 0, sd, np.nan) * math.sqrt(MPY)


def lo_sharpe_se(x: pd.Series, lag: int = NW_LAG) -> float:
    """Lo (2002) autocorrelation-adjusted SE of the ANNUALISED Sharpe."""
    a = x.dropna().to_numpy(dtype=float)
    n = a.size
    s_m = a.mean() / a.std(ddof=1)
    rho = [float(np.corrcoef(a[k:], a[:-k])[0, 1]) for k in range(1, lag + 1)]
    q = max(1.0 + 2.0 * sum((1.0 - k / (lag + 1.0)) * rho[k - 1]
                            for k in range(1, lag + 1)), 1e-9)
    return float(math.sqrt((1.0 + 0.5 * s_m ** 2) / n * q) * math.sqrt(MPY))


def levered_total(excess: np.ndarray, cash: np.ndarray, lev: float,
                  spread: float) -> np.ndarray:
    """L units of notional funded by 1 equity + (L-1) borrowed at bill + `spread`."""
    return lev * excess - max(lev - 1.0, 0.0) * spread / MPY + cash


def dd_ladder(excess: np.ndarray, cash: np.ndarray, spread: float,
              caps=(0.35, 0.50)) -> dict:
    rows: dict[str, Any] = {}
    grid = np.arange(0.05, 8.0001, 0.05)
    best = {c: (0.0, float("nan"), float("nan")) for c in caps}
    peak = (0.0, -9.9)
    ruin = None
    for lev in grid:
        tot = levered_total(excess, cash, float(lev), spread)
        if is_ruined(tot):
            ruin = float(lev)
            break
        g, dd = cagr(tot), max_dd(tot)
        if g > peak[1]:
            peak = (float(lev), g)
        for c in caps:
            if abs(dd) <= c:
                best[c] = (float(lev), g, dd)
    for c in caps:
        cap_lev, cap_g, cap_dd = best[c]
        rows[f"dd{int(c * 100)}"] = {"leverage": round(cap_lev, 2), "cagr": cap_g,
                                     "max_dd": cap_dd}
    rows["peak"] = {"leverage": peak[0], "cagr": peak[1]}
    rows["ruin_leverage"] = ruin
    return rows


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    out: dict = {"_meta": {"claim": "trend+passive inverse-vol, Sharpe 0.9033, n=738",
                           "bootstrap": {"seed": RNG_SEED, "n": N_BOOT,
                                         "block_months": BLOCK}}}

    saved = pd.read_csv(TREND_CSV, index_col=0, parse_dates=True)
    trend = saved["net_10bps"].dropna()
    passive = saved["bench_net_10bps"].dropna()
    gross_trend = saved["gross"].dropna()
    cash = pd.read_parquet(DATA / "cash_monthly.parquet")["US_CASH_13W"]

    book, w = book_from(trend, passive)
    f = pd.concat({"trend": trend, "passive": passive}, axis=1).dropna()
    cs = cash.reindex(book.index)

    out["A0_baseline"] = {
        "n": int(len(book)), "years": len(book) / MPY,
        "sharpe_book": sharpe(book), "sharpe_trend": sharpe(trend),
        "sharpe_passive": sharpe(passive),
        "vol_book": ann_vol(book), "vol_trend": ann_vol(trend),
        "vol_passive": ann_vol(passive),
        "weights_capital": {"trend": float(w[0]), "passive": float(w[1])},
        "rho_trend_passive": float(np.corrcoef(f["trend"], f["passive"])[0, 1]),
        "reproduces_0.9033": bool(abs(sharpe(book) - 0.9033) < 5e-4),
    }

    # ── A1. IS IT JUST PASSIVE? ───────────────────────────────────────────────
    cov = f.cov(ddof=1).to_numpy() * MPY
    vp = math.sqrt(w @ cov @ w)
    rc = w * ((cov @ w) / vp)
    mu = f.mean().to_numpy() * MPY
    ret_c = w * mu

    X = np.column_stack([np.ones(len(f)), f["passive"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, f["trend"].to_numpy(), rcond=None)
    resid = f["trend"].to_numpy() - X @ coef
    span_t = newey_west_tstat(pd.Series(resid + coef[0], index=f.index), NW_LAG)

    idx = circular_blocks(len(f), BLOCK, np.random.default_rng(RNG_SEED), N_BOOT)
    bt, bp = f["trend"].to_numpy()[idx], f["passive"].to_numpy()[idx]
    bb = bt * w[0] + bp * w[1]
    diff = (bb.mean(axis=1) / bb.std(axis=1, ddof=1)
            - bp.mean(axis=1) / bp.std(axis=1, ddof=1)) * math.sqrt(MPY)

    # CAUSAL weights: only information available at t-1 sets the weight held at t.
    vols = f.rolling(60, min_periods=36).std(ddof=1).shift(1)
    wc = (1.0 / vols).div((1.0 / vols).sum(axis=1), axis=0)
    book_causal = (f * wc).sum(axis=1).where(wc.notna().all(axis=1)).dropna()
    exp_vol = f.expanding(min_periods=36).std(ddof=1).shift(1)
    we = (1.0 / exp_vol).div((1.0 / exp_vol).sum(axis=1), axis=0)
    book_exp = (f * we).sum(axis=1).where(we.notna().all(axis=1)).dropna()
    fixed = {}
    for wp in (0.5, 0.6, 0.7218, 0.8, 0.9):
        b = f["trend"] * (1 - wp) + f["passive"] * wp
        fixed[f"passive_{wp:.4f}"] = {"sharpe": sharpe(b), "vm": vm(b, f["passive"])}

    out["A1_is_it_just_passive"] = {
        "capital_weight_passive": float(w[1]),
        "risk_share": {"trend": float(rc[0] / rc.sum()),
                       "passive": float(rc[1] / rc.sum())},
        "return_contribution_annual": {"trend": float(ret_c[0]),
                                       "passive": float(ret_c[1])},
        "return_share": {"trend": float(ret_c[0] / ret_c.sum()),
                         "passive": float(ret_c[1] / ret_c.sum())},
        "book_mean_annual": float(ret_c.sum()),
        "variance_share": {"trend": float(w[0] ** 2 * cov[0, 0] / vp ** 2),
                           "passive": float(w[1] ** 2 * cov[1, 1] / vp ** 2),
                           "interaction": float(2 * w[0] * w[1] * cov[0, 1] / vp ** 2)},
        "book_vs_passive": vm(book, passive),
        "trend_vs_passive": vm(trend, passive),
        "spanning_alpha_annual": float(coef[0] * MPY),
        "spanning_beta": float(coef[1]),
        "spanning_alpha_tstat_nw": float(span_t),
        "sharpe_gain_over_passive": sharpe(book) - sharpe(passive),
        "sharpe_gain_boot_ci95": [float(np.percentile(diff, 2.5)),
                                  float(np.percentile(diff, 97.5))],
        "p_sharpe_gain_le_zero": float((diff <= 0).mean()),
        "causal_weights_rolling60m": {"n": int(len(book_causal)),
                                      "sharpe": sharpe(book_causal),
                                      "vm": vm(book_causal,
                                               f["passive"].reindex(book_causal.index)),
                                      "mean_passive_weight": float(
                                          wc["passive"].reindex(book_causal.index).mean())},
        "causal_weights_expanding": {"n": int(len(book_exp)), "sharpe": sharpe(book_exp),
                                     "vm": vm(book_exp,
                                              f["passive"].reindex(book_exp.index)),
                                     "mean_passive_weight": float(
                                         we["passive"].reindex(book_exp.index).mean())},
        "fixed_weight_grid": fixed,
    }

    # ── A2. THE DECADE PROBLEM ────────────────────────────────────────────────
    dec: dict = {}
    for d, blk in book.groupby(book.index.year // 10 * 10):
        if len(blk) < 24:
            continue
        p = f["passive"].reindex(blk.index)
        dec[str(int(d))] = {"n": int(len(blk)), "book": sharpe(blk),
                            "trend": sharpe(f["trend"].reindex(blk.index)),
                            "passive": sharpe(p), "book_annual": ann_mean(blk),
                            "vm_vs_passive": vm(blk, p)}
    rs = book.rolling(120).apply(lambda a: a.mean() / a.std(ddof=1) * math.sqrt(MPY))
    rs = rs.dropna()
    rp = passive.reindex(book.index).rolling(120).apply(
        lambda a: a.mean() / a.std(ddof=1) * math.sqrt(MPY)).dropna()
    eras = {}
    for lab, lo, hi in (("1965-2009", "1965-01-01", "2009-12-31"),
                        ("2010-2026", "2010-01-01", "2026-12-31"),
                        ("1996-2026", "1996-01-01", "2026-12-31"),
                        ("2000-2026", "2000-01-01", "2026-12-31")):
        m = (book.index >= lo) & (book.index <= hi)
        eras[lab] = {"n": int(m.sum()), "book": sharpe(book[m]),
                     "trend": sharpe(f["trend"][m]), "passive": sharpe(f["passive"][m]),
                     "vm_vs_passive": vm(book[m], f["passive"][m])}
    out["A2_decades"] = {
        "by_decade": dec,
        "n_below_target": sum(1 for v in dec.values() if v["book"] < TARGET),
        "rolling10y_min": float(rs.min()), "rolling10y_min_at": str(rs.idxmin().date()),
        "rolling10y_frac_below_target": float((rs < TARGET).mean()),
        "rolling10y_median": float(rs.median()),
        "rolling10y_passive_median": float(rp.median()),
        "rolling10y_beats_passive_frac": float((rs.reindex(rp.index) > rp).mean()),
        "eras": eras,
    }

    # ── A3. THE BOND BULL AND DURATION ────────────────────────────────────────
    mask = ~((book.index >= BULL_FIRST) & (book.index <= BULL_LAST))
    res = run_trend(TrendConfig(), vol_target=VOL_TARGET)
    x, interior = load_excess_panel()
    _n, eligible, _c = _positions(x, TrendConfig())
    live = res.weights.abs().sum(axis=1) > 0
    held = interior.reindex_like(res.weights).fillna(False)
    elig_shift = eligible.shift(1).astype(float).fillna(0.0).astype(bool) & ~held
    wb = elig_shift.astype(float).div(
        elig_shift.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    wb = wb.where(live.reindex(wb.index).fillna(False), 0.0)
    xz = x.fillna(0.0)
    bench_pnl = wb * xz
    bench_to = wb.diff().abs().sum(axis=1).where(live)
    rebuilt = (bench_pnl.sum(axis=1).where(live)
               - 0.5 * COST_HEADLINE * bench_to).dropna()
    rebuild_err = float((rebuilt - passive).abs().max())

    def block_of(k: str) -> str:
        for b, keys in BLOCKS.items():
            if k in keys:
                return b
        return "other"

    tp = res.pnl.reindex(columns=list(PRIMARY_UNIVERSE)).fillna(0.0).reindex(trend.index)
    bpn = bench_pnl.reindex(columns=list(PRIMARY_UNIVERSE)).reindex(passive.index)
    attribution: dict = {}
    for b in BLOCKS:
        cols = [k for k in PRIMARY_UNIVERSE if block_of(k) == b]
        attribution[b] = {
            "instruments": cols,
            "trend_pnl_annual": ann_mean(tp[cols].sum(axis=1)),
            "passive_pnl_annual": ann_mean(bpn[cols].sum(axis=1)),
            "passive_share": float(bpn[cols].sum(axis=1).sum() / bpn.sum(axis=1).sum()),
            "trend_share": float(tp[cols].sum(axis=1).sum() / tp.sum(axis=1).sum()),
            "trend_pnl_in_bull": ann_mean(tp[cols].sum(axis=1)[~mask]),
            "trend_pnl_ex_bull": ann_mean(tp[cols].sum(axis=1)[mask]),
            "passive_pnl_in_bull": ann_mean(bpn[cols].sum(axis=1)[~mask]),
            "passive_pnl_ex_bull": ann_mean(bpn[cols].sum(axis=1)[mask]),
        }

    rate_cols = list(BLOCKS["rates"])
    keep = [k for k in PRIMARY_UNIVERSE if k not in rate_cols]
    trend_ex = (trend - tp[rate_cols].sum(axis=1)).dropna()
    pass_ex = (passive - bpn[rate_cols].sum(axis=1)).dropna()
    book_ex, w_ex = book_from(trend_ex, pass_ex)
    res_nr = run_trend(TrendConfig(name="NO_RATES"), vol_target=VOL_TARGET,
                       x=x[keep], interior=interior[keep])
    book_nr, w_nr = book_from(res_nr.net["10bps"], res_nr.bench_net["10bps"])
    m_nr = ~((book_nr.index >= BULL_FIRST) & (book_nr.index <= BULL_LAST))

    out["A3_bond_bull"] = {
        "passive_leg_rebuild_max_abs_err": rebuild_err,
        "book_full": sharpe(book), "book_ex_bull": sharpe(book[mask]),
        "trend_ex_bull": sharpe(f["trend"][mask]),
        "passive_ex_bull": sharpe(f["passive"][mask]),
        "n_outside_bull": int(mask.sum()),
        "book_ex_bull_vm_vs_passive": vm(book[mask], f["passive"][mask]),
        "attribution": attribution,
        "ex_rates_attribution": {
            "n": int(len(book_ex)), "sharpe": sharpe(book_ex),
            "weights": {"trend": float(w_ex[0]), "passive": float(w_ex[1])},
            "sharpe_trend": sharpe(trend_ex), "sharpe_passive": sharpe(pass_ex),
            "vm_vs_passive": vm(book_ex, pass_ex)},
        "ex_rates_rerun": {
            "n": int(len(book_nr)), "sharpe": sharpe(book_nr),
            "first_month": str(book_nr.index[0].date()),
            "weights": {"trend": float(w_nr[0]), "passive": float(w_nr[1])},
            "sharpe_trend": sharpe(res_nr.net["10bps"]),
            "sharpe_passive": sharpe(res_nr.bench_net["10bps"]),
            "vm_vs_passive": vm(book_nr, res_nr.bench_net["10bps"]),
            "ex_bull": sharpe(book_nr[m_nr])},
    }

    # ── A4. SURVIVORSHIP ──────────────────────────────────────────────────────
    jack: dict = {}
    for k in PRIMARY_UNIVERSE:
        b_j, _ = book_from((trend - tp[[k]].sum(axis=1)).dropna(),
                           (passive - bpn[[k]].sum(axis=1)).dropna())
        jack[k] = {"sharpe": sharpe(b_j),
                   "passive_pnl_share": float(bpn[k].sum() / bpn.sum(axis=1).sum()),
                   "trend_pnl_share": float(tp[k].sum() / tp.sum(axis=1).sum()),
                   "own_annual_return": ann_mean(x[k].dropna())}
    order = sorted(PRIMARY_UNIVERSE, key=lambda k: -jack[k]["passive_pnl_share"])
    drops = {}
    for kd in (1, 2, 3, 5):
        gone = order[:kd]
        b_d, _ = book_from((trend - tp[gone].sum(axis=1)).dropna(),
                           (passive - bpn[gone].sum(axis=1)).dropna())
        drops[str(kd)] = {"dropped": gone, "sharpe": sharpe(b_d)}
    added = {}
    for extra in (("NATGAS_F",), ("DJIA",), ("NATGAS_F", "DJIA")):
        uni = tuple(PRIMARY_UNIVERSE) + extra
        xa, ia = load_excess_panel(universe=uni)
        ra = run_trend(TrendConfig(name="ADD"), vol_target=VOL_TARGET, x=xa, interior=ia)
        ba, wa = book_from(ra.net["10bps"], ra.bench_net["10bps"])
        added["+".join(extra)] = {"n": int(len(ba)), "sharpe": sharpe(ba),
                                  "sharpe_trend": sharpe(ra.net["10bps"]),
                                  "sharpe_passive": sharpe(ra.bench_net["10bps"]),
                                  "vm_vs_passive": vm(ba, ra.bench_net["10bps"])}
    all_ret = pd.read_parquet(DATA / "returns_all_monthly.parquet")
    last_obs = {c: str(all_ret[c].last_valid_index().date()) for c in all_ret.columns}
    out["A4_survivorship"] = {
        "n_instruments": len(PRIMARY_UNIVERSE), "jackknife": jack,
        "jackknife_min": min(v["sharpe"] for v in jack.values()),
        "jackknife_max": max(v["sharpe"] for v in jack.values()),
        "drop_top_passive_contributors": drops,
        "add_back_prereg_exclusions": added,
        "n_series_ending_before_2026": int(sum(1 for v in last_obs.values()
                                               if v < "2026-01-01")),
        "instrument_last_observation": last_obs,
        "all_instruments_alive_today": all(v >= "2026-06-30" for v in last_obs.values()),
    }

    # ── A5. DATING AND DELISTING ──────────────────────────────────────────────
    spx = x["SPX"].dropna()
    probes = {n_: probe_alignment(s, spx, name=n_, reference_name="SPX",
                                  lags=(-2, -1, 0, 1, 2)).to_dict()
              for n_, s in {"trend": trend, "passive": passive, "book": book}.items()}
    lookahead: dict = {}
    n_raw = _n.mask(interior.reindex_like(_n).fillna(False), 0.0)
    for lag in (0, 1, 2):
        pos = n_raw.shift(lag).fillna(0.0)
        b_raw = (pos * xz).sum(axis=1).where(pos.abs().sum(axis=1) > 0)
        sig = b_raw.rolling(36, min_periods=12).std(ddof=1) * math.sqrt(MPY)
        gu = n_raw.abs().sum(axis=1)
        k = pd.concat([VOL_TARGET / sig.replace(0.0, np.nan),
                       10.0 / gu.replace(0.0, np.nan)], axis=1).min(axis=1)
        ww = n_raw.mul(k, axis=0).shift(lag).fillna(0.0)
        lv = ww.abs().sum(axis=1) > 0
        net = ((ww * xz).sum(axis=1)
               - 0.5 * COST_HEADLINE * ww.diff().abs().sum(axis=1)).where(lv).dropna()
        es = eligible.shift(lag).astype(float).fillna(0.0).astype(bool) & ~held
        wl = es.astype(float).div(es.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        wl = wl.where(lv.reindex(wl.index).fillna(False), 0.0)
        bn = ((wl * xz).sum(axis=1)
              - 0.5 * COST_HEADLINE * wl.diff().abs().sum(axis=1)).where(lv).dropna()
        bk, _ = book_from(net, bn)
        lookahead[str(lag)] = {"sharpe_trend": sharpe(net), "sharpe_passive": sharpe(bn),
                              "sharpe_book": sharpe(bk), "n": int(len(bk))}
    src = Path("research/sleeves/multiasset_trend.py").read_text(encoding="utf-8")
    out["A5_dating_delisting"] = {
        "alignment_probes": probes, "lookahead_ladder": lookahead,
        "trend_module_imports_delisting": "research.delisting" in src,
        "panel_note": ("18 exchange indices / continuous futures / FX spot from one "
                       "vendor; none delist, the sleeve never imports research.delisting, "
                       "and every series in the file runs to 2026-06-30"),
    }

    # ── A6. THE 12-MONTH LEVERAGE BUG ─────────────────────────────────────────
    res_flat = run_trend(TrendConfig(), vol_target=VOL_TARGET,
                         no_vol_estimate=NO_ESTIMATE_FLAT)
    book_flat, _ = book_from(res_flat.net["10bps"], res_flat.bench_net["10bps"])
    n_no = int(res.no_vol_estimate.sum())
    out["A6_leverage_bug"] = {
        "policy_in_headline": res.no_vol_estimate_policy,
        "headline_is_post_fix": res.no_vol_estimate_policy != NO_ESTIMATE_CAP,
        "months_with_no_vol_estimate": n_no,
        "gross_leverage_in_those_months": [float(v)
                                           for v in res.gross_leverage.head(n_no)],
        "sharpe_book_registered": sharpe(book),
        "sharpe_book_repaired": sharpe(book_flat),
        "delta": sharpe(book_flat) - sharpe(book),
        "n_repaired": int(len(book_flat)),
        "vm_repaired": vm(book_flat, res_flat.bench_net["10bps"]),
    }
    vt = {}
    for target in (0.10, 0.20, 0.40, 0.60):
        rt = run_trend(TrendConfig(), vol_target=target, x=x, interior=interior)
        bt_, _ = book_from(rt.net["10bps"], rt.bench_net["10bps"])
        vt[f"{target:.2f}"] = {"sharpe_book": sharpe(bt_),
                               "sharpe_trend": sharpe(rt.net["10bps"]),
                               "vm": vm(bt_, rt.bench_net["10bps"])}
    out["A6_leverage_bug"]["vol_target_grid"] = vt

    # ── A7. COSTS ─────────────────────────────────────────────────────────────
    turn = res.turnover.reindex(trend.index)
    bturn = bench_to.reindex(passive.index)
    # the equal-weight benchmark charges only weight-VECTOR changes; drifting back to
    # 1/N every month is a real trade and is unpriced. Price it.
    drift = wb.shift(1) * (1.0 + xz)
    drift = drift.div(drift.sum(axis=1).replace(0, np.nan), axis=0)
    true_bturn = (wb - drift).abs().sum(axis=1).where(live).reindex(passive.index)
    true_bturn = true_bturn.fillna(bturn)

    def repriced(bps: float, bturn_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        c = bps / 10_000.0
        t_c = (gross_trend - 0.5 * c * turn).dropna()
        p_c = (passive + 0.5 * COST_HEADLINE * bturn
               - 0.5 * c * bturn_series).dropna()
        return t_c, p_c

    grid = {}
    for bps in (0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100):
        t_c, p_c = repriced(bps, bturn)
        b_c, w_c = book_from(t_c, p_c)
        t_r, p_r = repriced(bps, true_bturn)
        b_r, _ = book_from(t_r, p_r)
        grid[str(bps)] = {"sharpe_book": sharpe(b_c), "sharpe_trend": sharpe(t_c),
                          "sharpe_passive": sharpe(p_c),
                          "weight_passive": float(w_c[1]), "vm": vm(b_c, p_c),
                          "sharpe_book_with_true_rebalance_cost": sharpe(b_r)}
    lo_bp, hi_bp = 0.0, 400.0
    for _ in range(50):
        mid = (lo_bp + hi_bp) / 2
        t_c, p_c = repriced(mid, true_bturn)
        b_c, _ = book_from(t_c, p_c)
        if sharpe(b_c) >= TARGET:
            lo_bp = mid
        else:
            hi_bp = mid
    out["A7_costs"] = {
        "headline_cost_bps": 10.0,
        "convention": "half-spread x turnover, charged to BOTH legs",
        "trend_turnover_units_per_year": float(turn.mean() * MPY),
        "passive_turnover_as_charged_per_year": float(bturn.mean() * MPY),
        "passive_true_rebalance_turnover_per_year": float(true_bturn.mean() * MPY),
        "unpriced_rebalance_cost_at_10bps_annual": float(
            0.5 * COST_HEADLINE * (true_bturn - bturn).mean() * MPY),
        "annual_cost_at_10bps_trend": float(0.5 * COST_HEADLINE * turn.mean() * MPY),
        "grid": grid,
        "breakeven_bps_for_sharpe_0894": float(lo_bp),
    }

    # ── A8. THE CI ────────────────────────────────────────────────────────────
    bs, bs_p = boot_sharpe(book), boot_sharpe(passive)
    se = lo_sharpe_se(book)
    out["A8_confidence"] = {
        "sharpe": sharpe(book), "boot_mean": float(np.nanmean(bs)),
        "boot_ci95": [float(np.nanpercentile(bs, 2.5)),
                      float(np.nanpercentile(bs, 97.5))],
        "p_below_0894": float(np.nanmean(bs < TARGET)),
        "p_below_075": float(np.nanmean(bs < 0.75)),
        "p_below_passive_alone": float(np.nanmean(bs < sharpe(passive))),
        "p_below_060": float(np.nanmean(bs < 0.60)),
        "lo2002_se": se,
        "lo2002_ci95": [sharpe(book) - 1.96 * se, sharpe(book) + 1.96 * se],
        "p_below_0894_lo": float(0.5 * (1 + math.erf(
            (TARGET - sharpe(book)) / (se * math.sqrt(2))))),
        "p_below_075_lo": float(0.5 * (1 + math.erf(
            (0.75 - sharpe(book)) / (se * math.sqrt(2))))),
        "passive_boot_ci95": [float(np.nanpercentile(bs_p, 2.5)),
                              float(np.nanpercentile(bs_p, 97.5))],
        # the ledger is the authority on the programme's cumulative count; the rest of
        # the ladder is a declared sensitivity, not a claim about how many were run
        "dsr_bars": {str(n): dsr_sharpe_bar(len(book) / MPY, n_trials=n)
                     for n in (cumulative_trials(),
                               cumulative_trials() + 58,      # the v2 subset search
                               cumulative_trials() + 58 + 40,  # + this verification
                               1000)},
        "ledger_cumulative_trials": cumulative_trials(),
    }

    # ── A9. RETURN CONVENTIONS ────────────────────────────────────────────────
    def gap(tr: str, px: str) -> dict:
        d = pd.concat([all_ret[tr], all_ret[px]], axis=1).dropna()
        c = cash.reindex(d.index)
        return {"n": int(len(d)), "from": str(d.index[0].date()),
                "to": str(d.index[-1].date()),
                "total_return_annual": ann_mean(d[tr]),
                "panel_series_annual": ann_mean(d[px]),
                "gap_annual": ann_mean(d[tr] - d[px]), "cash_annual": ann_mean(c),
                "panel_overstates_true_excess_by": ann_mean(c) - ann_mean(d[tr] - d[px])}

    conv = {p: gap(*p.split("_vs_")) for p in
            ("SPY_vs_SPX", "GLD_vs_GOLD_F", "SLV_vs_SILVER_F", "TLT_vs_US30Y_TR",
             "IEF_vs_US10Y_TR", "IEI_vs_US5Y_TR", "DBC_vs_WTI_F")}
    eq = list(BLOCKS["equity"])
    sens = {}
    for bps in (0, 75, 150, 250, 400):
        xa = x.copy()
        xa[eq] = xa[eq] - bps / 10_000.0 / MPY
        ra = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=xa, interior=interior)
        ba, _ = book_from(ra.net["10bps"], ra.bench_net["10bps"])
        sens[str(bps)] = {"sharpe_book": sharpe(ba),
                          "sharpe_trend": sharpe(ra.net["10bps"]),
                          "sharpe_passive": sharpe(ra.bench_net["10bps"]),
                          "vm": vm(ba, ra.bench_net["10bps"])}
    lo_d, hi_d = 0.0, 1200.0
    for _ in range(30):
        mid = (lo_d + hi_d) / 2
        xa = x.copy()
        xa[eq] = xa[eq] - mid / 10_000.0 / MPY
        ra = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=xa, interior=interior)
        ba, _ = book_from(ra.net["10bps"], ra.bench_net["10bps"])
        if sharpe(ba) >= TARGET:
            lo_d = mid
        else:
            hi_d = mid
    # the fully-honest variant: charge cash - measured dividend on the equity block,
    # month by month, using the ONE dividend estimate this panel can measure.
    div = conv["SPY_vs_SPX"]["gap_annual"]
    xh = x.copy()
    xh[eq] = xh[eq].sub(cash.reindex(x.index) - div / MPY, axis=0)
    rh = run_trend(TrendConfig(), vol_target=VOL_TARGET, x=xh, interior=interior)
    bh, wh = book_from(rh.net["10bps"], rh.bench_net["10bps"])
    out["A9_return_conventions"] = {
        "panel_convention": ("price / futures / spot returns are treated as ALREADY "
                             "excess; only the three rates series have cash subtracted"),
        "equity_instruments_treated_as_excess": eq,
        "measured_gaps": conv,
        "equity_drag_sensitivity_bps": sens,
        "breakeven_equity_drag_bps_for_0894": float(lo_d),
        "full_correction_cash_minus_measured_div": {
            "measured_dividend_yield_annual": div,
            "n": int(len(bh)), "sharpe_book": sharpe(bh),
            "sharpe_trend": sharpe(rh.net["10bps"]),
            "sharpe_passive": sharpe(rh.bench_net["10bps"]),
            "weights": {"trend": float(wh[0]), "passive": float(wh[1])},
            "vm": vm(bh, rh.bench_net["10bps"]),
            "mean_charge_annual": float((cash.reindex(x.index).mean() * MPY) - div)},
    }

    # ── A10. THE HONEST NUMBER ────────────────────────────────────────────────
    ex, csn = book.to_numpy(), cs.to_numpy()
    ladder = {lab: dd_ladder(ex, csn, sp) for lab, sp in FINANCING.items()}
    bidx = circular_blocks(len(ex), BLOCK, np.random.default_rng(RNG_SEED + 1), 2000)
    boot_best = {}
    for cap in (0.35, 0.50):
        chosen = 0.0
        for lev in np.arange(0.05, 5.0001, 0.05):
            tot = levered_total(ex, csn, float(lev), FINANCING["primary_bill_plus_150bp"])
            paths = tot[bidx]
            curve = np.cumprod(1.0 + paths, axis=1)
            dd = (curve / np.maximum.accumulate(curve, axis=1) - 1.0).min(axis=1)
            if abs(float(np.percentile(dd, 5))) <= cap:
                chosen = float(lev)
            else:
                break
        tot = levered_total(ex, csn, chosen, FINANCING["primary_bill_plus_150bp"])
        boot_best[f"dd{int(cap * 100)}"] = {"leverage": round(chosen, 2),
                                            "cagr": cagr(tot),
                                            "dd_observed": max_dd(tot)}
    # and the same, on the book after the A9 correction
    ex_h = bh.reindex(book.index).dropna()
    cs_h = cash.reindex(ex_h.index)
    ladder_h = dd_ladder(ex_h.to_numpy(), cs_h.to_numpy(),
                         FINANCING["primary_bill_plus_150bp"])
    out["A10_honest_number"] = {
        "unlevered_cagr_total": cagr(levered_total(ex, csn, 1.0, 0.0150)),
        "unlevered_max_dd": max_dd(levered_total(ex, csn, 1.0, 0.0150)),
        "cash_mean_annual": ann_mean(cs),
        "ladder_observed_path": ladder,
        "bootstrap_p95_primary_financing": boot_best,
        "iteration11_reconciliation_factor": 0.877,
        "dd50_boot_after_reconciliation": boot_best["dd50"]["cagr"] * 0.877,
        "dd35_boot_after_reconciliation": boot_best["dd35"]["cagr"] * 0.877,
        "after_A9_correction_ladder": ladder_h,
        "after_A9_correction_dd50_reconciled": ladder_h["dd50"]["cagr"] * 0.877,
    }

    (OUT_DIR / "survivor_verification.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", OUT_DIR / "survivor_verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
