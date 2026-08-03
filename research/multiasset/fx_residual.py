"""Decomposing the EUR/GBP FX residual the convention repair left open.

`convention_repair_result.md` §1 measured the corrected FX legs against currency-deposit
ETFs and found EUR and GBP sitting 0.94%/yr and 0.86%/yr above their benchmarks, outside
the registered 0.75%/yr budget, with the registered zero-rate-fee explanation **refuted
by its own diagnostic** — the residual is larger in normal-rate months, not smaller.

This module implements the replacement hypothesis registered in `fx_residual_prereg.md`:
the residual is a **benchmark-construction artefact** with three separately identifiable
parts. Writing the measured residual as the convention-repair runner computes it,

    diff_t = fx_excess_t - (ETF_ret_t - cash_t)
    fx_excess_t = spot_t + (i3m_foreign - i3m_US)_{t-1} / 12

and substituting what a CurrencyShares trust actually returns,
``ETF_ret_t = spot_t + net_t / 12`` with ``net`` its annualised interest-and-fee credit:

    diff_t = [ (i3m_foreign - earned) / 12 ]   (A1) foreign leg over-credit
           + [ fee / 12 ]                      (B)  sponsor's fee
           - [ i3m_US / 12 - cash_t ]          (C)  US leg: minus the TED spread
           + remainder                         (A2) depository margin, unobservable

The panel credits a **3-month interbank** differential on both legs by design, so that no
maturity or basis is mixed *inside* the differential (`build_carry_inputs` docstring). The
benchmark breaks that symmetry: its foreign leg earns an **overnight deposit rate less an
unpublished margin** and its US leg is a **government bill**. (A1) and (C) are the two
halves of that broken symmetry and both scale with the rate level, which is why the
residual is larger in normal-rate months and why a fee-only story could never produce it.

**No value for the depository margin is assumed anywhere.** It is what is left after the
published fee and the two measured rate spreads are removed. A test that fitted it could
not fail.

Nothing here changes a panel series, a strategy, a gate or a headline number. It tests
whether the *yardstick* used to verify the FX correction is a fair one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MONTHS_PER_YEAR = 12

#: Sponsor's fee for FXE, FXB and FXY. **Published, not fitted**: 0.40%/yr, accrued daily
#: and paid out of the trust's interest (Invesco CurrencyShares prospectus / 10-K).
SPONSOR_FEE: float = 0.0040

#: The foreign 3-month rate at or below which a month counts as "low-rate". Identical to
#: the split Control C used, so the two diagnostics are directly comparable.
LOW_RATE_THRESHOLD: float = 0.005

#: Registered budget the adjusted remainder must come inside (prereg P2). Same constant
#: as `run_convention_repair.TOL_FX`.
TOL_FX: float = 0.0075

#: Pre-registered bars. Fixed in `fx_residual_prereg.md` §3 before any run.
MAX_ASYMMETRY: float = 0.0025          # P1, 0.25 pp/yr
MAX_CROSS_CURRENCY_SPREAD: float = 0.0035   # P3, 0.35 pp/yr
MAX_BRACKET_DISAGREEMENT: float = 0.0025    # prereg §4, 0.25 pp/yr

#: The three registered constructions of the trust's *pre-fee* earned rate (prereg §4).
CONSTRUCTIONS: tuple[str, ...] = ("published", "zero_floored", "fee_first")
HEADLINE_CONSTRUCTION: str = "zero_floored"

#: OECD overnight ("immediate rates: call money/interbank, < 24 hours") counterparts of
#: the 3-month family already in `_data/carry/short_rates_monthly.parquet`.
#: Same source family on purpose: splicing EONIA/*STR/SONIA/TONA would mix sources inside
#: a spread and reintroduce the very error this module exists to measure.
#:
#: Taken from OECD DIRECTLY rather than via FRED (prereg amendment 2026-07-31): FRED
#: became unreachable from this machine at the IP level mid-run. OECD publishes these;
#: FRED mirrors them. Verified equal against the pre-block FRED values -- see
#: OVERNIGHT_FRED_ANCHORS and `verify_against_fred_anchors`.
OVERNIGHT_SERIES: dict[str, str] = {
    "EZ": "EA20",
    "GB": "GBR",
    "JP": "JPN",
    "US": "USA",
}

#: OECD SDMX coordinates. `IRSTCI` is the overnight measure, `IR3TIB` the 3-month one.
OECD_DATAFLOW = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0"
OECD_OVERNIGHT_MEASURE = "IRSTCI"
OECD_SHORT_MEASURE = "IR3TIB"

#: FRED values recorded on 2026-07-31 BEFORE the block, used to prove the OECD pull is the
#: same series rather than merely a plausible substitute. (n, first, last, last_value).
#: EA20 legitimately runs 5 months longer -- FRED's mirror lags -- so its check is on the
#: overlap; the other three must match exactly.
OVERNIGHT_FRED_ANCHORS: dict[str, tuple[int, str, str, float]] = {
    "GB": (582, "1978-01", "2026-06", 3.7298),
    "JP": (492, "1985-07", "2026-06", 0.841),
    "US": (864, "1954-07", "2026-06", 3.625172),
    "EZ": (385, "1994-01", "2026-01", 1.931671),
}


def verify_against_fred_anchors(series_pct: dict[str, "pd.Series"],
                                *, tol: float = 5e-6) -> Prediction:
    """P0 - the OECD pull must BE the registered series, not merely resemble it.

    Each series is truncated to the anchor's last month before comparison, so EA20's
    five extra months of newer data cannot mask a mismatch on the registered window.
    A failure here voids the run exactly as P6 does: it would mean the substitution
    changed the data, which is the one thing the amendment promises it does not.
    """
    rows: dict[str, dict] = {}
    ok = True
    for ccy, (n, first, last, last_value) in OVERNIGHT_FRED_ANCHORS.items():
        s = series_pct.get(ccy)
        if s is None or not len(s):
            rows[ccy] = {"matched": False, "why": "series missing"}
            ok = False
            continue
        idx = [str(x) for x in s.index]
        trimmed = s[[i <= last for i in idx]]
        got_n, got_first = len(trimmed), (idx[0] if idx else "")
        got_last = str(trimmed.index[-1]) if len(trimmed) else ""
        got_value = float(trimmed.iloc[-1]) if len(trimmed) else float("nan")
        matched = (got_n == n and got_first == first and got_last == last
                   and abs(got_value - last_value) <= tol)
        ok = ok and matched
        rows[ccy] = {"matched": bool(matched),
                     "n": got_n, "expected_n": n,
                     "first": got_first, "expected_first": first,
                     "last": got_last, "expected_last": last,
                     "last_value": got_value, "expected_last_value": last_value}
    return Prediction(
        key="P0_source_identity",
        statement="the OECD pull reproduces the pre-block FRED series exactly on the "
                  "registered window; otherwise the transport substitution changed the "
                  "data and the run is void",
        passed=ok,
        detail={"per_currency": rows},
    )

#: Panel key -> (benchmark ETF, short-rate column). Mirrors `run_convention_repair`.
FX_PAIRS: dict[str, tuple[str, str]] = {
    "EURUSD": ("FXE", "EZ"),
    "GBPUSD": ("FXB", "GB"),
    "JPYUSD": ("FXY", "JP"),
}


def annualise(series: pd.Series) -> float:
    """Annualised arithmetic mean of a monthly series. Arithmetic, never geometric.

    Identical to `run_convention_repair.ann`, so P5's null control can compare like
    with like.
    """
    a = pd.Series(series).dropna()
    return float(a.mean() * MONTHS_PER_YEAR) if len(a) else float("nan")


def earned_rate(overnight: pd.Series, *, construction: str,
                fee: float = SPONSOR_FEE) -> pd.Series:
    """The trust's annualised **pre-fee** earned deposit rate, per registered construction.

    * ``published`` — the overnight rate unmodified.
    * ``zero_floored`` — ``max(0, overnight)``. The trust cannot be credited a negative
      deposit rate it did not pay; FXB's rate in effect at 2021-12-31 was 0.00%, which is
      the evidence this case is real rather than assumed.
    * ``fee_first`` — ``max(fee, overnight)``, so that the *net* credit
      ``earned - fee`` floors at zero: the fee is taken from interest first and the
      shortfall is not charged onward.

    Keeping the fee out of this function is what lets term (B) stay a published constant
    in all three constructions instead of being absorbed into a fitted one.
    """
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"unregistered construction {construction!r}; "
                         f"expected one of {CONSTRUCTIONS}")
    o = pd.Series(overnight).astype(float)
    if construction == "published":
        return o
    if construction == "zero_floored":
        return o.clip(lower=0.0)
    return o.clip(lower=fee)


def decompose(
    diff: pd.Series,
    *,
    i3m_foreign: pd.Series,
    overnight_foreign: pd.Series,
    i3m_us: pd.Series,
    cash: pd.Series,
    construction: str = HEADLINE_CONSTRUCTION,
    fee: float = SPONSOR_FEE,
) -> pd.DataFrame:
    """Split the measured residual into A1 + B - C + remainder.

    All rate arguments are **annualised decimals** (0.0532 = 5.32%); ``cash`` and ``diff``
    are **monthly** returns, matching the panel. Returns one row per month over the
    intersection of every input, with the identity
    ``predicted + remainder == diff`` true to floating point by construction.
    """
    frame = pd.concat(
        {
            "diff": pd.Series(diff).astype(float),
            "i3m_foreign": pd.Series(i3m_foreign).astype(float),
            "overnight_foreign": pd.Series(overnight_foreign).astype(float),
            "i3m_us": pd.Series(i3m_us).astype(float),
            "cash": pd.Series(cash).astype(float),
        },
        axis=1,
    ).dropna()

    earned = earned_rate(frame["overnight_foreign"], construction=construction, fee=fee)
    frame["earned"] = earned
    frame["a1_foreign_tenor"] = (frame["i3m_foreign"] - earned) / MONTHS_PER_YEAR
    frame["b_sponsor_fee"] = fee / MONTHS_PER_YEAR
    frame["c_us_ted"] = frame["i3m_us"] / MONTHS_PER_YEAR - frame["cash"]
    frame["predicted"] = (frame["a1_foreign_tenor"] + frame["b_sponsor_fee"]
                          - frame["c_us_ted"])
    frame["remainder"] = frame["diff"] - frame["predicted"]
    return frame


def regime_split(frame: pd.DataFrame, column: str,
                 *, threshold: float = LOW_RATE_THRESHOLD,
                 min_months: int = 12) -> dict:
    """Annualised ``column`` in low-rate vs normal-rate months, and the asymmetry.

    The split is on the **foreign 3-month rate**, exactly as Control C split it, so the
    'before' and 'after' asymmetries are comparable without re-deriving anything.
    """
    low = frame["i3m_foreign"] <= threshold
    n_low, n_high = int(low.sum()), int((~low).sum())
    gap_low = annualise(frame.loc[low, column]) if n_low >= min_months else None
    gap_high = annualise(frame.loc[~low, column]) if n_high >= min_months else None
    asym = (abs(gap_high - gap_low)
            if gap_low is not None and gap_high is not None else None)
    return {
        "n_low_rate_months": n_low,
        "n_normal_months": n_high,
        "gap_low_pct_yr": None if gap_low is None else round(gap_low * 100.0, 4),
        "gap_high_pct_yr": None if gap_high is None else round(gap_high * 100.0, 4),
        "asymmetry_pct_yr": None if asym is None else round(asym * 100.0, 4),
    }


@dataclass
class Prediction:
    """One registered prediction and whether the measured data met it."""

    key: str
    statement: str
    passed: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"prediction": self.statement, "passed": self.passed, **self.detail}


def evaluate_sign_discipline(spreads: dict[str, float]) -> Prediction:
    """P6 — a 3-month interbank rate below overnight on average means wrong pairing.

    This is an integrity gate, not a finding: if it fails the run is void and the pairing
    error is the only thing to report.
    """
    offenders = {k: round(v * 100.0, 4) for k, v in spreads.items() if v < 0.0}
    return Prediction(
        key="P6_sign_discipline",
        statement="mean(i3m - overnight) >= 0 for every currency; otherwise the two "
                  "series are paired wrongly and the run is void",
        passed=not offenders,
        detail={"mean_3m_minus_overnight_pct_yr":
                {k: round(v * 100.0, 4) for k, v in spreads.items()},
                "negative_offenders": offenders},
    )


def evaluate_null_control(measured: dict[str, float],
                          published: dict[str, float],
                          *, tol_pp: float = 0.01) -> Prediction:
    """P5 — with the fee and both rate spreads zeroed, reproduce the published residuals.

    ``measured`` and ``published`` are annualised percents. If this fails, the
    decomposition is not measuring the quantity Control C measured and nothing else in
    the run may be believed.
    """
    deltas = {k: round(abs(measured[k] - published[k]), 6)
              for k in published if k in measured}
    return Prediction(
        key="P5_null_control",
        statement=f"fee=0 and both rate spreads zeroed reproduces the published "
                  f"residuals to within {tol_pp} pp",
        passed=bool(deltas) and all(d <= tol_pp for d in deltas.values()),
        detail={"published_pct_yr": published, "reproduced_pct_yr": measured,
                "abs_delta_pp": deltas},
    )
