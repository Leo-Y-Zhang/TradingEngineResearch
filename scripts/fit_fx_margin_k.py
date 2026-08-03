"""Estimate the rate-proportional margin `k` on the ORIGINAL three legs only.

This is the FIT step of a fit-on-A / test-on-B design. `k` is estimated here on EUR, GBP
and JPY — data whose answer is already known, so fitting on it proves nothing by itself —
and is then FROZEN into a pre-registration that predicts the residuals of three
CurrencyShares trusts the panel has never touched (FXF/FXA/FXC). Only that second step
is a test.

Model. The `zero_floored` construction assumed the trust earns `max(0, overnight)` before
its fee. If instead the depository keeps a FRACTION `k` of the rate,

    earned_t = max(0, overnight_t) * (1 - k)

then the shortfall against the earlier assumption is `max(0, overnight_t) * k`, which is
proportional to the rate level — the shape the result doc found and a constant cannot make.
So the already-measured remainder should satisfy

    remainder_t  ~=  k * max(0, overnight_t) / 12

Fitted POOLED across the three legs with **no intercept**: a free intercept per currency
would let the model absorb the level separately from the slope, which is precisely the
overfitting this design exists to avoid. The intercept is reported as a diagnostic only.

Writes `research/multiasset/_fx_residual/margin_fit.json`. Touches no panel series.
"""

from __future__ import annotations

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
    MONTHS_PER_YEAR,
    annualise,
    decompose,
)
from research.sleeves.multiasset_trend import BLOCKS, load_excess_panel  # noqa: E402
from scripts.run_fx_residual import (  # noqa: E402
    CONV,
    DATA,
    OUT_DIR,
    RATE_CACHE,
    fetch_oecd,
    to_month_end_decimals,
)
from research.multiasset.fx_residual import OECD_OVERNIGHT_MEASURE  # noqa: E402


def main() -> int:
    old, _ = load_excess_panel()
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
        panel_fx.append(FxInstrument(key, match.ticker if match else key, ccy, False, ""))
    fx_excess, _ = fx_excess_returns(old[[i.key for i in panel_fx]],
                                     rates_short.reindex(old.index), tuple(panel_fx))

    raw_on = fetch_oecd(OECD_OVERNIGHT_MEASURE, use_cache=True)
    overnight = {c: to_month_end_decimals(raw_on[c]) for c in raw_on.columns}

    pooled_x: list[np.ndarray] = []
    pooled_y: list[np.ndarray] = []
    per_ccy: dict[str, dict] = {}

    for key, (etf, ccy) in FX_PAIRS.items():
        diff = (fx_excess[key] - (ref[etf] - cash)).dropna()
        frame = decompose(
            diff,
            i3m_foreign=rates_short[ccy].reindex(old.index).shift(1),
            i3m_us=rates_short["US"].reindex(old.index).shift(1),
            overnight_foreign=overnight[ccy].reindex(old.index),
            cash=cash,
            construction=HEADLINE_CONSTRUCTION,
        )
        x = (frame["overnight_foreign"].clip(lower=0.0) / MONTHS_PER_YEAR).to_numpy()
        y = frame["remainder"].to_numpy()
        pooled_x.append(x)
        pooled_y.append(y)
        k_i = float(x @ y / (x @ x)) if float(x @ x) > 0 else float("nan")
        per_ccy[key] = {
            "n": int(len(x)),
            "k_if_fitted_alone": round(k_i, 6),
            "mean_remainder_pct_yr": round(annualise(frame["remainder"]) * 100.0, 4),
            "mean_positive_overnight_pct": round(
                float(frame["overnight_foreign"].clip(lower=0.0).mean()) * 100.0, 4),
        }

    X = np.concatenate(pooled_x)
    Y = np.concatenate(pooled_y)
    k = float(X @ Y / (X @ X))
    resid = Y - k * X
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2_about_mean = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    # no-intercept R^2 (fraction of raw sum of squares explained), the honest one here
    r2_raw = 1.0 - float((resid ** 2).sum()) / float((Y ** 2).sum())

    # Diagnostic only, NOT used downstream: what a free intercept would absorb.
    A = np.column_stack([X, np.ones_like(X)])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)

    out = {
        "purpose": "FIT step only. k is frozen from here into the holdout prereg; the "
                   "holdout residuals have NOT been computed at the time of writing.",
        "model": "earned = max(0, overnight) * (1 - k); remainder ~= k * max(0, overnight)/12",
        "fitted_on": ["EURUSD", "GBPUSD", "JPYUSD"],
        "k_pooled_no_intercept": round(k, 6),
        "k_pct_of_rate_kept_by_depository": round(k * 100.0, 3),
        "r2_raw_no_intercept": round(r2_raw, 5),
        "r2_about_mean": round(r2_about_mean, 5),
        "n_pooled_months": int(len(X)),
        "per_currency": per_ccy,
        "diagnostic_free_intercept": {
            "slope": round(float(coef[0]), 6),
            "intercept_pct_yr": round(float(coef[1]) * MONTHS_PER_YEAR * 100.0, 4),
            "note": "reported to show how much level a free intercept would absorb; the "
                    "frozen k above is the NO-INTERCEPT fit, which is the harder one",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "margin_fit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"k (pooled, no intercept) = {k:.6f}  -> depository keeps "
          f"{k * 100:.2f}% of the overnight rate")
    print(f"  raw R^2 = {r2_raw:.4f}   R^2 about mean = {r2_about_mean:.4f}   n = {len(X)}")
    for key, row in per_ccy.items():
        print(f"  {key}: k_alone={row['k_if_fitted_alone']:.4f}  "
              f"remainder={row['mean_remainder_pct_yr']:+.3f}%/yr  n={row['n']}")
    print(f"  [diagnostic] free-intercept slope={coef[0]:.4f}, "
          f"intercept={coef[1] * MONTHS_PER_YEAR * 100:.4f}%/yr")
    print(f"\nWrote {OUT_DIR / 'margin_fit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
