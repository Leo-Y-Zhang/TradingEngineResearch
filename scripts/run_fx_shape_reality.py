"""Run the third registered FX-residual test: is the regime asymmetry real?

Registered in ``research/multiasset/fx_shape_reality_prereg.md``, committed at
``f8a10a2`` with no statistic attached.

The frames are rebuilt here rather than refactored out of ``run_fx_residual.py``,
which is reviewed research code. The rebuild is **self-validating**: it must
reproduce the committed headline remainders before any test statistic is read, so a
loader mistake shows up as a reproduction failure rather than as a wrong verdict.

    python scripts/run_fx_shape_reality.py --use-cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.carry import (  # noqa: E402
    FX_INSTRUMENTS,
    FxInstrument,
    fx_excess_returns,
)
from research.multiasset.fx_residual import (  # noqa: E402
    FX_PAIRS,
    HEADLINE_CONSTRUCTION,
    annualise,
    decompose,
)
from research.multiasset.fx_shape_reality import (  # noqa: E402
    asymmetry,
    block_bootstrap_null,
    circular_shift_null,
    minimum_detectable_effect,
    p_value,
    pooled_statistic,
    regime_mask,
    verdict,
)
from research.sleeves.multiasset_trend import BLOCKS, load_excess_panel  # noqa: E402

import scripts.run_fx_residual as base  # noqa: E402

OUT_DIR = base.OUT_DIR

#: Headline remainders the committed run produced, %/yr. The rebuild must match these
#: to within rounding or the run is void -- this is the loader's own null control.
COMMITTED_REMAINDER_PCT = {"EURUSD": 0.743, "GBPUSD": 0.490, "JPYUSD": 0.216}
REBUILD_TOL_PP = 0.02


def build_frames(use_cache: bool) -> dict[str, pd.DataFrame]:
    """Rebuild the per-leg monthly decomposition exactly as Control C computes it."""
    old, _interior = load_excess_panel()
    cash = pd.read_parquet(base.DATA / "cash_monthly.parquet")["US_CASH_13W"].reindex(old.index)
    ref = pd.read_parquet(base.CONV / "reference_returns_monthly.parquet").reindex(old.index)
    rates_short = pd.read_parquet(base.RATE_CACHE / "short_rates_monthly.parquet")

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

    raw_overnight = base.fetch_oecd(base.OECD_OVERNIGHT_MEASURE, use_cache=use_cache)
    overnight = {c: base.to_month_end_decimals(raw_overnight[c])
                 for c in raw_overnight.columns}

    frames: dict[str, pd.DataFrame] = {}
    for key, (etf, ccy) in FX_PAIRS.items():
        if etf not in ref.columns or key not in fx_excess.columns:
            continue
        diff = (fx_excess[key] - (ref[etf] - cash)).dropna()
        frames[key] = decompose(
            diff,
            construction=HEADLINE_CONSTRUCTION,
            i3m_foreign=rates_short[ccy].reindex(old.index).shift(1),
            i3m_us=rates_short["US"].reindex(old.index).shift(1),
            overnight_foreign=overnight[ccy].reindex(old.index),
            cash=cash,
        )
    return frames


def run_nulls(legs, draws, seed):
    shift = circular_shift_null(legs, draws=draws, seed=seed)
    block = block_bootstrap_null(legs, draws=draws, seed=seed)
    return shift, block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260731)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "prereg": "research/multiasset/fx_shape_reality_prereg.md",
        "what_this_changes": "Nothing. No panel series, gate, ledger row, live path "
                             "or headline number is touched. 0.7834 is unaffected.",
        "draws": args.draws,
        "seed": args.seed,
    }

    frames = build_frames(args.use_cache)

    # -- the loader's own null control: reproduce the committed remainders -----
    rebuilt = {k: round(annualise(f["remainder"]) * 100.0, 4) for k, f in frames.items()}
    worst = max(abs(rebuilt[k] - v) for k, v in COMMITTED_REMAINDER_PCT.items()
                if k in rebuilt)
    results["C0_rebuild_reproduces_committed"] = {
        "committed_pct_yr": COMMITTED_REMAINDER_PCT,
        "rebuilt_pct_yr": rebuilt,
        "worst_abs_diff_pp": round(worst, 5),
        "tolerance_pp": REBUILD_TOL_PP,
        "passed": bool(worst <= REBUILD_TOL_PP),
    }
    if worst > REBUILD_TOL_PP:
        results["verdict"] = "VOID - the rebuild does not reproduce the committed run"
        (OUT_DIR / "fx_shape_reality.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("C0 FAILED - rebuild does not match the committed remainders. Void.")
        return 1
    print(f"C0 rebuild reproduces committed remainders (worst {worst:.5f} pp)")

    legs = {k: (f["remainder"].to_numpy(dtype=float), regime_mask(f))
            for k, f in frames.items()}
    per_leg = {k: asymmetry(v, low) for k, (v, low) in legs.items()}
    s_obs = pooled_statistic(per_leg)
    results["observed"] = {"per_leg_asymmetry_pct_yr": {k: round(v, 4)
                                                        for k, v in per_leg.items()},
                           "pooled_S_pct_yr": round(s_obs, 4),
                           "n_months": {k: int(len(f)) for k, f in frames.items()}}
    print(f"observed S = {s_obs:.4f} %/yr   per leg " +
          ", ".join(f"{k}={v:+.3f}" for k, v in per_leg.items()))

    shift, block = run_nulls(legs, args.draws, args.seed)
    p_shift, p_block = p_value(s_obs, shift), p_value(s_obs, block)
    v = verdict(p_shift, p_block)
    results["N1_circular_shift"] = {"p": round(p_shift, 5),
                                    "null_median": round(float(np.median(shift)), 4),
                                    "null_q95": round(float(np.quantile(shift, 0.95)), 4)}
    results["N2_block_bootstrap"] = {"p": round(p_block, 5),
                                     "null_median": round(float(np.median(block)), 4),
                                     "null_q95": round(float(np.quantile(block, 0.95)), 4)}
    results["verdict"] = v
    results["minimum_detectable_pct_yr"] = round(
        minimum_detectable_effect(legs, shift), 4)
    print(f"N1 circular shift p={p_shift:.4f}   N2 block bootstrap p={p_block:.4f}")
    print(f"VERDICT: {v}   (test can see effects above "
          f"{results['minimum_detectable_pct_yr']:.3f} %/yr)")

    # -- C1 POWER: an injected real effect must come back REAL -----------------
    inject = 0.005 / 12.0            # 0.5 %/yr concentrated in high-rate months
    powered = {}
    for k, (v_arr, low) in legs.items():
        bumped = v_arr.copy()
        bumped[~low] += inject
        powered[k] = (bumped, low)
    s_pow = pooled_statistic({k: asymmetry(a, m) for k, (a, m) in powered.items()})
    ps, pb = run_nulls(powered, max(2000, args.draws // 5), args.seed)
    c1 = verdict(p_value(s_pow, ps), p_value(s_pow, pb))
    results["C1_power"] = {"injected_pct_yr": 0.5, "S": round(s_pow, 4),
                           "verdict": c1, "passed": c1 == "REAL"}
    print(f"C1 power (inject 0.5%/yr): S={s_pow:.4f} -> {c1}")

    # -- C2 SIZE: matched-variance white noise must come back ARTEFACT ---------
    rng = np.random.default_rng(args.seed + 99)
    noise = {k: (rng.normal(0.0, v_arr.std(), size=len(v_arr)), low)
             for k, (v_arr, low) in legs.items()}
    s_noise = pooled_statistic({k: asymmetry(a, m) for k, (a, m) in noise.items()})
    ns, nb = run_nulls(noise, max(2000, args.draws // 5), args.seed)
    c2 = verdict(p_value(s_noise, ns), p_value(s_noise, nb))
    results["C2_size"] = {"S": round(s_noise, 4), "verdict": c2,
                          "passed": c2 != "REAL"}
    print(f"C2 size (white noise): S={s_noise:.4f} -> {c2}")

    # -- C3 DETERMINISM --------------------------------------------------------
    shift2, block2 = run_nulls(legs, args.draws, args.seed)
    same = bool(np.array_equal(shift, shift2) and np.array_equal(block, block2))
    results["C3_determinism"] = {"passed": same}
    print(f"C3 determinism: {'identical' if same else 'DIFFERS'}")

    results["controls_all_passed"] = bool(
        results["C0_rebuild_reproduces_committed"]["passed"]
        and results["C1_power"]["passed"]
        and results["C2_size"]["passed"]
        and same
    )
    (OUT_DIR / "fx_shape_reality.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\ncontrols all passed: {results['controls_all_passed']}")
    print(f"wrote {OUT_DIR / 'fx_shape_reality.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
