"""THE DECISION LAYER: Kelly, drawdown, DSR and sampling uncertainty on the best combos.

Consumes `portfolio_correlation_result.json`; adds nothing new to the return series.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_decision
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar
from research.validation import deflated_sharpe_ratio

from research.trial_ledger import cumulative_trials
from research.sleeves._portfolio.portfolio_correlation import (
    ALL_NAMES, MPY, SOURCES, _load, ann_vol,
    block_bootstrap_sharpe, erc_weights, inverse_variance_weights,
    inverse_vol_weights, max_dd, sharpe,
)

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "research" / "sleeves" / "_portfolio"
RNG = np.random.default_rng(20260728)

TARGET_SHARPE_30PCT = 0.894      # 30%/yr at half Kelly
DD_TOLERANCE = 0.60              # a book whose implied DD exceeds this is not reachable
# FROZEN, AND WRONG. This study deflated against 38 while the programme ledger already
# stood at 47 (`research.trial_ledger`), and 31 undercounts a combination search that was
# actually 189-234 configurations. Both are kept verbatim so the banked
# `portfolio_decision.json` stays reproducible, and the ledger bar is now computed BESIDE
# them under separate keys so the understatement is visible rather than inferred by
# inverting a bar. They are registered as STALE in `trial_ledger.FROZEN_TRIAL_COUNTS`.
N_TRIALS_PROGRAMME = 38          # the count the low-vol gate was evaluated at
N_COMBOS_SEARCHED = 31           # 2^5 - 1 sleeve subsets examined here


def kelly_block(port: np.ndarray, label: str) -> dict:
    """Half-Kelly growth, the volatility it needs, the leverage, the drawdown that implies."""
    s = sharpe(port)
    vol = ann_vol(port)
    mu = float(np.mean(port) * MPY)
    dd_1x = max_dd(port)

    # Full Kelly: L = S / sigma  -> levered vol = S, growth = S^2/2
    # Half Kelly: L = S / (2 sigma) -> levered vol = S/2, growth = 3 S^2/8
    lev_half = s / (2.0 * vol)
    vol_half = s / 2.0
    growth_half = 3.0 * s * s / 8.0
    growth_full = s * s / 2.0

    # Drawdown at that leverage, two ways: the naive linear scaling the brief asks for,
    # and the honest one -- recompound the actually-levered monthly series.
    dd_linear = dd_1x * lev_half
    dd_compounded = max_dd(port * lev_half)

    # Largest leverage whose RECOMPOUNDED drawdown stays inside DD_TOLERANCE, and the
    # compound growth actually attainable there.
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(max_dd(port * mid)) > DD_TOLERANCE:
            hi = mid
        else:
            lo = mid
    lev_dd_capped = lo
    r_capped = port * lev_dd_capped
    cagr_capped = float(np.prod(1.0 + r_capped) ** (MPY / len(r_capped)) - 1.0)

    return {
        "label": label,
        "n_months": int(len(port)),
        "years": len(port) / MPY,
        "sharpe": s,
        "vol_1x": vol,
        "mean_annual_1x": mu,
        "cagr_1x": float(np.prod(1.0 + port) ** (MPY / len(port)) - 1.0),
        "max_dd_1x": dd_1x,
        "half_kelly_growth": growth_half,
        "half_kelly_required_vol": vol_half,
        "half_kelly_leverage": lev_half,
        "half_kelly_dd_linear_scaled": dd_linear,
        "half_kelly_dd_recompounded": dd_compounded,
        "half_kelly_reachable": bool(abs(dd_compounded) <= DD_TOLERANCE),
        "full_kelly_growth": growth_full,
        "leverage_at_60pct_dd": lev_dd_capped,
        "cagr_at_60pct_dd": cagr_capped,
        "vol_at_60pct_dd": ann_vol(r_capped),
        "clears_0894": bool(s >= TARGET_SHARPE_30PCT),
    }


def main() -> int:
    series = {k: _load(p, c) for k, (p, c, _) in SOURCES.items()}
    out: dict = {}

    # -- rebuild every combination's portfolio return stream (inverse-vol + ERC) -------
    def build(combo: list[str], index=None) -> pd.DataFrame:
        f = pd.concat({c: series[c] for c in combo}, axis=1).dropna()
        return f if index is None else f.reindex(index).dropna()

    headline = [
        ["lowvol", "trend"],
        ["lowvol", "trend", "carry"],
        ["lowvol", "trend", "defensive"],
        ["lowvol", "trend", "carry", "defensive"],
        ["trend", "carry"],
        ["lowvol"],
        ["passive"],
    ]
    blocks = []
    for combo in headline:
        f = build(combo)
        for scheme, wfun in (("inverse_vol", inverse_vol_weights),
                             ("erc", erc_weights),
                             ("inverse_variance", inverse_variance_weights)):
            w = wfun(f) if len(combo) > 1 else np.array([1.0])
            port = f.to_numpy() @ w
            blk = kelly_block(port, f"{'+'.join(combo)} [{scheme}]")
            blk["combo"] = combo
            blk["scheme"] = scheme
            blk["weights"] = {c: float(x) for c, x in zip(f.columns, w)}
            blk["first"] = str(f.index.min().date())
            blk["last"] = str(f.index.max().date())
            lo, hi = block_bootstrap_sharpe(port)
            blk["sharpe_ci95_block_boot"] = [lo, hi]
            # Analytic Lo (2002) SE of an annualised Sharpe, iid-Gaussian.
            n = len(port)
            se_ann = float(np.sqrt((1 + 0.5 * (blk["sharpe"] / np.sqrt(MPY)) ** 2) / n)
                           * np.sqrt(MPY))
            blk["sharpe_se_analytic"] = se_ann
            blk["sharpe_ci95_analytic"] = [blk["sharpe"] - 1.96 * se_ann,
                                           blk["sharpe"] + 1.96 * se_ann]
            blk["dsr_bar_programme_trials"] = dsr_sharpe_bar(
                blk["years"], n_trials=N_TRIALS_PROGRAMME)
            blk["dsr_bar_incl_combo_search"] = dsr_sharpe_bar(
                blk["years"], n_trials=N_TRIALS_PROGRAMME + N_COMBOS_SEARCHED)
            blk["dsr_programme_trials"] = float(
                deflated_sharpe_ratio(port, n_trials=N_TRIALS_PROGRAMME))
            blk["dsr_incl_combo_search"] = float(
                deflated_sharpe_ratio(port, n_trials=N_TRIALS_PROGRAMME + N_COMBOS_SEARCHED))
            # The same two statistics at the LEDGER count, which is the honest one.
            blk["n_trials_ledger"] = cumulative_trials()
            blk["dsr_bar_ledger_trials"] = dsr_sharpe_bar(blk["years"], n_trials=None)
            blk["dsr_ledger_trials"] = float(
                deflated_sharpe_ratio(port, n_trials=None))
            blocks.append(blk)
    out["headline"] = blocks

    # -- SENSITIVITY: low-vol re-scaled to its own two corrections (Sharpe 0.878 -> 0.677)
    #    Holds the CORRELATION structure fixed and scales only the mean, which is what the
    #    two accounting defects did. Labelled an assumption, not a measurement.
    lv = series["lowvol"]
    s_reg, s_cor = sharpe(lv), 0.677
    lv_cor = (lv - lv.mean()) + lv.mean() * (s_cor / s_reg)
    ser2 = dict(series)
    ser2["lowvol"] = lv_cor
    sens = []
    for combo in [["lowvol", "trend"], ["lowvol", "trend", "carry"],
                  ["lowvol", "trend", "defensive"]]:
        f = pd.concat({c: ser2[c] for c in combo}, axis=1).dropna()
        for scheme, wfun in (("inverse_vol", inverse_vol_weights), ("erc", erc_weights)):
            w = wfun(f)
            port = f.to_numpy() @ w
            b = kelly_block(port, f"{'+'.join(combo)} [{scheme}] LOWVOL-CORRECTED")
            b["combo"], b["scheme"] = combo, scheme
            sens.append(b)
    out["lowvol_corrected_sensitivity"] = {
        "assumption": ("low-vol mean scaled so its standalone Sharpe is 0.677 (the value "
                       "after the two accounting defects the study itself found); its "
                       "correlations to every other sleeve are held at the measured values"),
        "lowvol_sharpe_registered": s_reg,
        "lowvol_sharpe_corrected": s_cor,
        "blocks": sens,
    }

    # -- what the BEST combination would need to reach 0.894 --------------------------
    best = max((b for b in blocks if b["scheme"] in ("inverse_vol", "erc")),
               key=lambda b: b["sharpe"])
    out["best"] = best

    # -- exhaustive scan: does ANY subset (incl. passive) clear 0.894, and on what n? ---
    scan = []
    for size in range(1, len(ALL_NAMES) + 1):
        for subset in itertools.combinations(ALL_NAMES, size):
            f = pd.concat({c: series[c] for c in subset}, axis=1).dropna()
            if len(f) < 24:
                continue
            for scheme, wfun in (("inverse_vol", inverse_vol_weights),
                                 ("erc", erc_weights),
                                 ("inverse_variance", inverse_variance_weights)):
                w = wfun(f) if size > 1 else np.array([1.0])
                port = f.to_numpy() @ w
                s = sharpe(port)
                scan.append({"combo": list(combo), "scheme": scheme, "n_months": int(len(f)),
                             "first": str(f.index.min().date()),
                             "last": str(f.index.max().date()),
                             "sharpe": s, "clears_0894": bool(s >= TARGET_SHARPE_30PCT)})
    out["scan"] = sorted(scan, key=lambda e: -e["sharpe"])
    out["n_clearing_0894"] = sum(1 for e in scan if e["clears_0894"])
    out["n_scanned"] = len(scan)

    (OUT_DIR / "portfolio_decision.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"best": best["label"], "sharpe": best["sharpe"],
                      "n_clearing": out["n_clearing_0894"],
                      "n_scanned": out["n_scanned"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
