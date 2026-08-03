"""Synthesis arithmetic for the six-sleeve breadth hunt (DEV window, 2026-07-28).

WHY this script exists rather than a spreadsheet: every number in
`research/medallion_style_alpha_search/breadth_sleeve_hunt_result.md` has to be
reproducible and auditable, including the two that are easy to fudge — the
breadth-vs-Sharpe rank correlation (n=6, so a single reordering moves it a lot,
and an exact permutation p-value is cheap) and the DSR-required Sharpe at the
new trial count.

WHY the sleeve scalars are hard-coded here: five of the six sleeves wrote result
JSON with different schemas and one wrote only a text file, so there is no common
loader. Every scalar below was read back off disk and cross-checked before being
pasted in; `verify_against_disk()` re-checks the ones that ARE machine-readable
so a stale paste cannot survive a re-run silently.

Reads nothing outside the DEV artefacts; downloads nothing; touches no price bar.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1. The six sleeves, as reported and verified
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Sleeve:
    key: str
    name: str
    # Breadth as the sleeve REPORTED it (its headline figure).
    breadth_reported: float
    # WHY two breadth columns: a "Grinold-implied" breadth is back-solved from
    # BR = (IR/IC)^2, which makes IR = IC*sqrt(BR) an identity rather than a
    # test. Only a STRUCTURAL estimator (N_eff from a correlation/residual
    # matrix, or a statutory event count) is independent of the realised Sharpe
    # and can therefore be regressed against it.
    breadth_structural: float | None
    breadth_structural_basis: str
    breadth_is_circular: bool
    gross_sharpe: float
    net_sharpe: float
    excess_annual: float
    cost_drag_annual: float
    turnover_annual: float
    # Gross return used for the per-round-trip economics. For long-only books
    # measured against their own universe this is the gross EXCESS (the part the
    # signal earned); for the dollar-neutral reversal book it is the gross
    # return, which is already an excess by construction.
    gross_alpha_annual: float
    gross_alpha_basis: str
    years: float
    ic: float | None
    gate_eligible_headline: bool


SLEEVES: tuple[Sleeve, ...] = (
    Sleeve(
        key="reversal",
        name="Short-horizon cross-sectional reversal (weekly, dollar-neutral)",
        breadth_reported=577.16,
        breadth_structural=3172.40,  # 52 rebalances x 61.0 names traded, naive
        breadth_structural_basis="naive 52 x 61.0 names (no N_eff computed)",
        breadth_is_circular=True,  # headline 577 = (IR/IC)^2
        gross_sharpe=0.5193,
        net_sharpe=-3.6189,
        excess_annual=-0.5281,
        cost_drag_annual=0.7181,
        turnover_annual=45.29,
        gross_alpha_annual=0.06925,
        gross_alpha_basis="gross long/short return (already an excess)",
        years=17.65,
        ic=0.021617,
        gate_eligible_headline=True,
    ),
    Sleeve(
        key="pead",
        name="Post-earnings-announcement drift (SF1 ARQ SUE, 40d hold)",
        breadth_reported=476.94,
        breadth_structural=119.55,  # distinct entry DAYS/yr, the honest bound
        breadth_structural_basis="119.5 distinct entry days/yr (477 entry events)",
        breadth_is_circular=False,
        gross_sharpe=1.0749,
        net_sharpe=0.3422,
        excess_annual=-0.0297,
        cost_drag_annual=0.0557,
        turnover_annual=2.3895,
        gross_alpha_annual=0.0256 * 2.3895,  # +256bps/bet at 2.39 round trips/yr
        gross_alpha_basis="gross alpha per bet 256bps x 2.39 turnover",
        years=17.73,
        ic=None,
        gate_eligible_headline=True,
    ),
    Sleeve(
        key="tsmom",
        name="Time-series momentum, multi-timeframe (200 names + sector baskets)",
        breadth_reported=98.0,
        breadth_structural=98.0,  # N_eff 9.29 from the return correlation matrix
        breadth_structural_basis="N_eff 9.29 of 206 x signal flips 10.55/yr",
        breadth_is_circular=False,
        gross_sharpe=0.4455,
        net_sharpe=0.0576,
        excess_annual=-0.0235,
        cost_drag_annual=0.0665,
        turnover_annual=21.19,
        gross_alpha_annual=-0.0044 + 0.0665,  # net + cost = gross
        gross_alpha_basis="net return + cost drag (SENSITIVITY-B)",
        years=17.0,
        ic=None,  # the reported 0.045 was back-solved from IR/sqrt(BR)
        gate_eligible_headline=False,  # headline is SENSITIVITY-B, flat 20bps
    ),
    Sleeve(
        key="instflow",
        name="Institutional ownership flow (SF3 13F QoQ change)",
        breadth_reported=4.0,
        breadth_structural=4.0,  # statutory quarterly filing
        breadth_structural_basis="4 statutory 13F quarters/yr",
        breadth_is_circular=False,
        gross_sharpe=0.0808,
        net_sharpe=-0.4447,
        excess_annual=-0.065369,
        cost_drag_annual=0.077131,
        turnover_annual=6.6133,
        gross_alpha_annual=0.0118,
        gross_alpha_basis="gross excess over own universe",
        years=2.17,
        ic=0.0072,
        gate_eligible_headline=True,
    ),
    Sleeve(
        key="insider",
        name="Insider transaction clustering (SF2 distinct 90d buyers)",
        breadth_reported=1162.2,
        breadth_structural=1162.2,  # N_eff 96.85 residual-variance x 12
        breadth_structural_basis="N_eff 96.85 of 151 held x 12 rebalances",
        breadth_is_circular=False,
        gross_sharpe=0.53,
        net_sharpe=-0.12,
        excess_annual=-0.0869,
        cost_drag_annual=0.1373,
        turnover_annual=5.83,
        gross_alpha_annual=0.0545,
        gross_alpha_basis="gross excess over own universe",
        years=7.67,
        ic=0.0134,
        gate_eligible_headline=True,
    ),
    Sleeve(
        key="lowvol",
        name="Low-volatility / quality composite (B2 $200k-$1M band)",
        breadth_reported=93.5,
        breadth_structural=93.46,  # 7.788 effective bets/rebalance x 12
        breadth_structural_basis="7.79 effective bets/rebalance x 12",
        breadth_is_circular=False,
        gross_sharpe=1.116,
        net_sharpe=0.3244,
        excess_annual=-0.05544,
        cost_drag_annual=0.10819,
        turnover_annual=9.0573,
        gross_alpha_annual=0.15316 - 0.10041,
        gross_alpha_basis="gross return 15.32% - benchmark 10.04%",
        years=17.75,
        ic=None,
        gate_eligible_headline=True,
    ),
)


def verify_against_disk() -> list[str]:
    """Re-read the machine-readable artefacts and assert the pasted scalars match.

    WHY: this file is a transcription of six other agents' outputs. A silent
    transcription error would propagate into the synthesis unchallenged.
    """
    notes: list[str] = []
    tol = 1e-3

    p = REPO / "reports" / "reversal_sleeve_result.json"
    d = json.loads(p.read_text())["PRIMARY_measured_only"]
    assert abs(d["long_short"]["net"]["sharpe"] - (-3.6189)) < tol
    assert abs(d["long_short"]["gross"]["sharpe"] - 0.5193) < tol
    assert abs(d["breadth"]["grinold_implied_bets_per_year"] - 577.16) < 0.01
    assert abs(d["ic"]["mean"] - 0.021617) < 1e-5
    notes.append("reversal: net/gross Sharpe, breadth, IC verified on disk")

    p = REPO / "research" / "sleeves" / "_pead_output" / "pead_results.json"
    rows = json.loads(p.read_text())
    r40 = next(r for r in rows if r["book"] == "top_decile" and r["horizon_days"] == 40)
    assert abs(r40["net_sharpe"] - 0.3422) < tol
    assert abs(r40["gross_alpha_per_bet"] - 0.0256) < 1e-4
    assert abs(r40["cost_per_roundtrip_bps"] - 219.14) < 0.01
    notes.append("pead: 40d net Sharpe, alpha/bet, cost/round-trip verified on disk")

    p = REPO / "research" / "sleeves" / "_out" / "tsmom_multitimeframe_result.json"
    cfgs = json.loads(p.read_text())["configurations"]
    sb = next(c for c in cfgs if c["mode"] == "SENSITIVITY_B" or "SENSITIVITY-B" in c["label"])
    assert sb["gate_eligible"] is False, "tsmom headline config must be flagged non-gate-eligible"
    t15 = next(t for t in sb["targets"] if abs(t["target_vol"] - 0.15) < 1e-9)
    assert abs(t15["net_sharpe"] - 0.0576) < tol
    assert abs(sb["breadth"]["bets_per_year"] - 98.0) < 0.1
    notes.append("tsmom: headline is SENSITIVITY-B, gate_eligible=False, verified on disk")

    p = REPO / "research" / "sleeves" / "low_vol_quality_result.json"
    b2 = json.loads(p.read_text())["bands"][0]
    assert abs(b2["net_sharpe"] - 0.3244) < tol
    assert abs(b2["excess_annual"] - (-0.05544)) < tol
    assert abs(b2["breadth_per_year"] - 93.46) < 0.01
    notes.append("lowvol: B2 net Sharpe, excess, breadth verified on disk")

    return notes


# --------------------------------------------------------------------------
# 2. Rank correlation with an EXACT permutation p-value (n=6 -> 720 orderings)
# --------------------------------------------------------------------------
def _rank(x: list[float]) -> np.ndarray:
    order = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    return order.astype(float) + 1.0


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(rx @ ry / math.sqrt((rx @ rx) * (ry @ ry)))


def spearman_exact_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Two-sided exact permutation p-value. n=6 so all 720 orderings are cheap."""
    rho = spearman(x, y)
    ry = list(_rank(y))
    hits = 0
    total = 0
    for perm in permutations(ry):
        total += 1
        if abs(spearman(x, list(perm))) >= abs(rho) - 1e-12:
            hits += 1
    return rho, hits / total


# --------------------------------------------------------------------------
# 3. DSR-required Sharpe
# --------------------------------------------------------------------------
def required_annual_sharpe(years: float, n_trials: int, dsr: float = 0.95,
                           periods_per_year: int = 12) -> float:
    """Annualised Sharpe at which DSR = `dsr`, assuming normal monthly returns.

    Same algebra as research/validation.py::deflated_sharpe_ratio with g3=0 and
    g4=3 (normal), inverted for the Sharpe. WHY closed form rather than a
    numeric search on the repo function: the repo function takes a RETURN
    SERIES, and any synthetic series carries its own sample skew/kurtosis noise,
    which would make the published bar depend on a random seed. The normality
    assumption is what the recorded DSR table used and reproduces its anchor.

        sigma^2 = (1 + 0.5 s^2) / (T - 1)          [g3=0, g4=3]
        s       = sigma * (k(N) + z_dsr)           where k(N) is the expected
                                                   max of N trials
    """
    T = int(round(years * periods_per_year))
    n = max(int(n_trials), 1)
    gamma = 0.5772156649015329  # Euler-Mascheroni, as in validation.py
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    k = (1.0 - gamma) * z1 + gamma * z2
    a = (k + float(norm.ppf(dsr))) ** 2 / (T - 1)
    # s^2 = a * (1 + 0.5 s^2)  ->  s^2 (1 - a/2) = a
    denom = 1.0 - a / 2.0
    if denom <= 0:
        return float("inf")
    s_monthly = math.sqrt(a / denom)
    return s_monthly * math.sqrt(periods_per_year)


def cross_check_with_repo_dsr() -> str:
    """Confirm the closed form agrees with research/validation.py on a real series."""
    import sys
    sys.path.insert(0, str(REPO))
    from research.validation import deflated_sharpe_ratio  # noqa: PLC0415

    years, n_trials = 7.0, 26
    s_ann = required_annual_sharpe(years, n_trials)
    T = int(years * 12)
    # Normal scores: zero sample skew by construction, so the only residual
    # difference from the closed form is the (small) kurtosis of order stats.
    z = norm.ppf((np.arange(1, T + 1) - 0.5) / T)
    z = (z - z.mean()) / z.std(ddof=1)
    r = z + s_ann / math.sqrt(12.0)
    got = deflated_sharpe_ratio(r, n_trials=n_trials)
    return (f"closed form says Sharpe {s_ann:.3f} at {years:.0f}yr/n={n_trials}; "
            f"repo deflated_sharpe_ratio on a normal-scores series at that Sharpe "
            f"returns DSR {got:.4f} (target 0.9500)")


# --------------------------------------------------------------------------
# 4. Portfolio arithmetic
# --------------------------------------------------------------------------
def combined_sharpe(s: float, n: int, rho: float) -> float:
    """S = s * sqrt(N / (1 + (N-1) rho)) -- equal Sharpe, equal pairwise rho."""
    return s * math.sqrt(n / (1.0 + (n - 1) * rho))


def half_kelly_growth(s: float) -> float:
    """Half-Kelly compound growth = 3 S^2 / 8, at volatility S/2."""
    return 3.0 * s * s / 8.0


# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("DISK VERIFICATION")
    for n in verify_against_disk():
        print("  OK  " + n)

    print("\n" + "=" * 78)
    print("PER-ROUND-TRIP ECONOMICS  (the unifying number)")
    print(f"{'sleeve':<12}{'turnover':>9}{'alpha/turn':>12}{'cost/turn':>11}{'cover':>8}")
    for s in SLEEVES:
        a = s.gross_alpha_annual / s.turnover_annual * 1e4
        c = s.cost_drag_annual / s.turnover_annual * 1e4
        print(f"{s.key:<12}{s.turnover_annual:>9.2f}{a:>10.1f}bp{c:>9.1f}bp{a / c:>8.2f}")

    print("\n" + "=" * 78)
    print("BREADTH PER UNIT OF TURNOVER  (how expensively each sleeve bought breadth)")
    for s in sorted(SLEEVES, key=lambda z: -z.breadth_reported / z.turnover_annual):
        print(f"  {s.key:<12}{s.breadth_reported / s.turnover_annual:>8.1f} bets per unit turnover"
              f"   (BR {s.breadth_reported:.0f}, turnover {s.turnover_annual:.2f}x)")

    print("\n" + "=" * 78)
    print("CENTRAL QUESTION: does breadth predict Sharpe across the six sleeves?")
    br_rep = [s.breadth_reported for s in SLEEVES]
    br_str = [s.breadth_structural for s in SLEEVES]
    gs = [s.gross_sharpe for s in SLEEVES]
    ns = [s.net_sharpe for s in SLEEVES]
    for label, br in (("reported breadth", br_rep), ("structural breadth", br_str)):
        for tgt_label, tgt in (("GROSS Sharpe", gs), ("NET Sharpe", ns)):
            rho, p = spearman_exact_p(br, tgt)
            print(f"  {label:<20} vs {tgt_label:<13} rho = {rho:+.4f}   exact p = {p:.4f}")
    rho, p = spearman_exact_p(br_rep, [s.cost_drag_annual for s in SLEEVES])
    print(f"  {'reported breadth':<20} vs {'cost drag':<13} rho = {rho:+.4f}   exact p = {p:.4f}")
    rho, p = spearman_exact_p([s.turnover_annual for s in SLEEVES],
                              [s.cost_drag_annual for s in SLEEVES])
    print(f"  {'turnover':<20} vs {'cost drag':<13} rho = {rho:+.4f}   exact p = {p:.4f}")

    print("\n  Grinold check, IR = IC*sqrt(BR), where an IC was measured:")
    for s in SLEEVES:
        if s.ic is None:
            continue
        pred = s.ic * math.sqrt(s.breadth_reported)
        flag = "  <-- CIRCULAR (BR was solved from IR and IC)" if s.breadth_is_circular else ""
        print(f"    {s.key:<12} IC {s.ic:.4f} x sqrt({s.breadth_reported:.0f}) = "
              f"{pred:.3f}  vs realised gross {s.gross_sharpe:.3f}{flag}")

    print("\n" + "=" * 78)
    print("PORTFOLIO ARITHMETIC")
    survivors = [s for s in SLEEVES if s.excess_annual > 0 and s.net_sharpe >= 0.75]
    print(f"  sleeves with POSITIVE excess AND net Sharpe >= 0.75 gate: {len(survivors)}")
    print("  -> S = s*sqrt(N/(1+(N-1)rho)) is undefined at N = 0. Nothing to combine.")
    print("\n  Counterfactual A (NOT achievable): combine the 2 least-bad NET Sharpes")
    two = sorted(SLEEVES, key=lambda z: -z.net_sharpe)[:2]
    s_bar = sum(t.net_sharpe for t in two) / 2
    print(f"    inputs: {two[0].key} {two[0].net_sharpe:.4f}, {two[1].key} {two[1].net_sharpe:.4f}"
          f"  -> mean s = {s_bar:.4f}")
    for rho_ in (0.0, 0.2, 0.5):
        S = combined_sharpe(s_bar, 2, rho_)
        g = half_kelly_growth(S)
        print(f"    rho={rho_:.1f} -> S={S:.3f}, half-Kelly growth {g * 100:.2f}%/yr "
              f"at {S / 2 * 100:.1f}% vol")
    print("\n  Counterfactual B (physically impossible): all six GROSS Sharpes, zero cost")
    s_bar_g = sum(t.gross_sharpe for t in SLEEVES) / len(SLEEVES)
    print(f"    mean gross s = {s_bar_g:.4f} over N = {len(SLEEVES)}")
    for rho_ in (0.0, 0.2, 0.5):
        S = combined_sharpe(s_bar_g, len(SLEEVES), rho_)
        g = half_kelly_growth(S)
        print(f"    rho={rho_:.1f} -> S={S:.3f}, half-Kelly growth {g * 100:.2f}%/yr "
              f"at {S / 2 * 100:.1f}% vol")

    print("\n" + "=" * 78)
    print("TRIAL ACCOUNTING")
    print("  " + cross_check_with_repo_dsr())
    print(f"\n  {'OOS years':>10}{'n=26':>10}{'n=32':>10}{'delta':>9}")
    for y in (7.0, 7.67, 10.0, 15.0, 17.7, 20.0, 30.0, 40.0, 50.0):
        a = required_annual_sharpe(y, 26)
        b = required_annual_sharpe(y, 32)
        print(f"  {y:>10.2f}{a:>10.2f}{b:>10.2f}{b - a:>9.3f}")

    print("\n  Each sleeve against the bar its OWN sample length demands (n=32):")
    for s in SLEEVES:
        bar = required_annual_sharpe(s.years, 32)
        print(f"    {s.key:<12} {s.years:>5.2f}yr  bar {bar:>5.2f}  "
              f"gross {s.gross_sharpe:>6.3f} {'PASS' if s.gross_sharpe >= bar else 'fail'}  "
              f"net {s.net_sharpe:>7.3f} {'PASS' if s.net_sharpe >= bar else 'fail'}")


if __name__ == "__main__":
    main()
