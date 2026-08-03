"""DATING ALIGNMENT — does a dated return series' index mean what it says?

WHY THIS MODULE EXISTS
======================
`research/sleeves/lowvol_retest.py::run_band` labelled every monthly slot with the
FORMATION month (the panel row date the signal was ranked on) but filled it with
``forward_return`` — the close-to-close return of the FOLLOWING month. Every slot was
therefore dated ONE MONTH EARLY.

**That defect is invisible to every WITHIN-series statistic.** Mean, volatility, Sharpe,
drawdown, Newey-West t and the vol-matched active return are all invariant to shifting
every observation by a constant number of periods. An independent bit-for-bit
re-implementation reproduced the series exactly and still could not see it. It only
becomes visible when the series is JOINED TO ANOTHER SERIES BY DATE — correlation,
portfolio construction, regime conditioning, event studies — at which point it silently
destroys the answer.

THE PROBE
=========
Correlate the series against a reference whose dating is known to be correct, at lags
-1, 0 and +1 months, and look at WHERE the largest |rho| sits.

    lag k  compares  series(t)  against  reference(t + k)

    k =  0   contemporaneous — where a correctly-dated series peaks
    k = +1   the series leads the reference: slot t holds month t+1's return, so the
             series is dated ONE MONTH EARLY and its index must be shifted **+1 month**
    k = -1   the series lags: dated one month LATE, shift **-1 month**

``AlignmentProbe.suggested_shift_months`` is exactly the number of months to ADD to the
index to correct it, so a repair is ``index + probe.suggested_shift_months``.

POWER, AND WHY "ALIGNED" IS NOT THE SAME AS "PROVEN"
====================================================
A market-neutral long/short book has no market exposure, so its correlation against an
equity reference is noise at every lag and the probe cannot say anything. That is
``has_power is False``, and it is reported as ``UNINFORMATIVE`` rather than dressed up as
a pass. A registry that asserts alignment for such a series must either supply a
reference the series IS exposed to, or record explicitly that the probe has no power.

The verdict rules, in order:

  INSUFFICIENT_OVERLAP  no lag has ``min_overlap`` joined observations
  ALIGNED               the largest |rho| is at lag 0
  MISALIGNED            the largest |rho| is elsewhere AND the evidence is material:
                        |rho| at the best lag is at least ``min_abs_rho`` and exceeds
                        |rho| at lag 0 by more than ``sigma_margin`` standard errors
  UNINFORMATIVE         the largest |rho| is elsewhere but the evidence is not material

Reference dating is a claim, not an axiom. Anchor it on something external and
independently dated (a market index) before trusting the verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ALIGNED",
    "DEFAULT_LAGS",
    "FORMATION",
    "INSUFFICIENT_OVERLAP",
    "MISALIGNED",
    "MIN_ABS_RHO",
    "MIN_OVERLAP",
    "REALISATION",
    "SIGMA_MARGIN",
    "UNINFORMATIVE",
    "AlignmentProbe",
    "MisalignedSeriesError",
    "assert_aligned",
    "lag_correlations",
    "month_end_index",
    "probe_alignment",
    "shift_months",
    "to_month_end",
]

# ── verdicts ──────────────────────────────────────────────────────────────────
ALIGNED = "aligned"
MISALIGNED = "misaligned"
UNINFORMATIVE = "uninformative"
INSUFFICIENT_OVERLAP = "insufficient_overlap"

# ── index conventions a producer may declare ──────────────────────────────────
#: the slot is labelled by the month the return was EARNED — the only convention that
#: can be joined to another series by date
REALISATION = "realisation"
#: the slot is labelled by the month the SIGNAL was formed, and holds the NEXT month's
#: return; joining it to anything by date is off by one
FORMATION = "formation"

DEFAULT_LAGS: tuple[int, ...] = (-1, 0, 1)
MIN_OVERLAP = 24
#: |rho| below this is not material market exposure — the probe has no power
MIN_ABS_RHO = 0.20
#: how many standard errors the best lag must beat lag 0 by before we call it misaligned
SIGMA_MARGIN = 2.0


class MisalignedSeriesError(AssertionError):
    """A series' index does not mean what it says."""


# ── index handling ────────────────────────────────────────────────────────────
def month_end_index(index) -> pd.DatetimeIndex:
    """Normalise any monthly-ish index onto month-END timestamps.

    Two series that label the same month differently — month start, month end, the last
    trading day — must still join. Everything is pushed through a monthly `Period` so
    the join is on the MONTH and not on a timestamp that happens to differ by a few days.
    """
    return pd.DatetimeIndex(
        pd.DatetimeIndex(index).to_period("M").to_timestamp(how="end").normalize()
    )


def to_month_end(series: pd.Series) -> pd.Series:
    """A float series re-indexed onto month ends, NaNs dropped, duplicates refused."""
    out = pd.Series(np.asarray(series, dtype=float), index=month_end_index(series.index))
    duplicated = int(out.index.duplicated().sum())
    if duplicated:
        raise ValueError(f"{duplicated} duplicate months in the index; not a monthly series")
    return out.dropna()


def shift_months(series: pd.Series, months: int) -> pd.Series:
    """Add ``months`` to every label. This is the REPAIR, not a positional `.shift`.

    Positional shifting is wrong here: a series with a missing month would move every
    later observation by a different amount. Calendar arithmetic on the period index
    cannot do that.
    """
    out = to_month_end(series)
    out.index = pd.DatetimeIndex(
        (out.index.to_period("M") + months).to_timestamp(how="end").normalize()
    )
    return out


def lag_correlations(
    series: pd.Series,
    reference: pd.Series,
    *,
    lags: Sequence[int] = DEFAULT_LAGS,
    min_overlap: int = MIN_OVERLAP,
) -> tuple[dict[int, float], dict[int, int]]:
    """rho between ``series(t)`` and ``reference(t + k)``, plus the overlap, for each k.

    Implemented by relabelling the REFERENCE (an observation earned in month ``u`` is
    given the label ``u - k``) rather than by `.shift`, so gaps in either series cannot
    corrupt the comparison.
    """
    left = to_month_end(series)
    rho: dict[int, float] = {}
    overlap: dict[int, int] = {}
    for k in lags:
        right = shift_months(reference, -int(k))
        joined = pd.concat({"s": left, "r": right}, axis=1, join="inner").dropna()
        n = int(len(joined))
        overlap[int(k)] = n
        if n < min_overlap:
            rho[int(k)] = float("nan")
            continue
        a = joined["s"].to_numpy()
        b = joined["r"].to_numpy()
        if float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
            rho[int(k)] = float("nan")
            continue
        rho[int(k)] = float(np.corrcoef(a, b)[0, 1])
    return rho, overlap


# ── the probe ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlignmentProbe:
    """One series' dating, measured against one correctly-dated reference."""

    name: str
    reference: str
    rho: dict[int, float]
    n_overlap: dict[int, int]
    best_lag: int
    rho_at_zero: float
    max_abs_rho: float
    has_power: bool
    verdict: str
    notes: list[str] = field(default_factory=list)

    @property
    def suggested_shift_months(self) -> int:
        """Months to ADD to the index to correct it. 0 when nothing is wrong."""
        return int(self.best_lag) if self.verdict == MISALIGNED else 0

    @property
    def is_misaligned(self) -> bool:
        return self.verdict == MISALIGNED

    def describe(self) -> str:
        lags = " ".join(f"k={k:+d}:{self.rho[k]:+.4f}" for k in sorted(self.rho))
        return (f"{self.name} vs {self.reference}: {lags} | best k={self.best_lag:+d} "
                f"| power={'yes' if self.has_power else 'no'} | {self.verdict.upper()}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "reference": self.reference,
            "rho": {str(k): v for k, v in sorted(self.rho.items())},
            "n_overlap": {str(k): v for k, v in sorted(self.n_overlap.items())},
            "best_lag": self.best_lag,
            "rho_at_zero": self.rho_at_zero,
            "max_abs_rho": self.max_abs_rho,
            "has_power": self.has_power,
            "verdict": self.verdict,
            "suggested_shift_months": self.suggested_shift_months,
            "notes": list(self.notes),
        }


def probe_alignment(
    series: pd.Series,
    reference: pd.Series,
    *,
    name: str = "series",
    reference_name: str = "reference",
    lags: Sequence[int] = DEFAULT_LAGS,
    min_overlap: int = MIN_OVERLAP,
    min_abs_rho: float = MIN_ABS_RHO,
    sigma_margin: float = SIGMA_MARGIN,
) -> AlignmentProbe:
    """Measure where a series' correlation with a correctly-dated reference peaks."""
    if 0 not in set(int(k) for k in lags):
        raise ValueError("lags must include 0; there is nothing to compare against")
    rho, overlap = lag_correlations(series, reference, lags=lags, min_overlap=min_overlap)
    finite = {k: v for k, v in rho.items() if np.isfinite(v)}
    notes: list[str] = []

    if not finite:
        return AlignmentProbe(
            name=name, reference=reference_name, rho=rho, n_overlap=overlap,
            best_lag=0, rho_at_zero=float("nan"), max_abs_rho=float("nan"),
            has_power=False, verdict=INSUFFICIENT_OVERLAP,
            notes=[f"no lag reached {min_overlap} joined observations"],
        )

    best_lag = max(finite, key=lambda k: abs(finite[k]))
    max_abs_rho = abs(finite[best_lag])
    rho_at_zero = float(finite.get(0, float("nan")))
    has_power = bool(max_abs_rho >= min_abs_rho)

    if best_lag == 0:
        verdict = ALIGNED
        if not has_power:
            notes.append(
                f"peak is at lag 0 but |rho| is only {max_abs_rho:.4f}; the reference has "
                "little explanatory power over this series, so ALIGNED is weak evidence"
            )
    else:
        n = max(overlap.get(best_lag, 0), 4)
        se = 1.0 / np.sqrt(n)
        margin = abs(rho_at_zero) + sigma_margin * se if np.isfinite(rho_at_zero) else 0.0
        material = has_power and max_abs_rho > margin
        verdict = MISALIGNED if material else UNINFORMATIVE
        if material:
            notes.append(
                f"|rho| peaks at k={best_lag:+d} ({max_abs_rho:.4f}) rather than at lag 0 "
                f"({rho_at_zero:+.4f}); add {best_lag:+d} month(s) to the index"
            )
        else:
            notes.append(
                f"|rho| peaks at k={best_lag:+d} but only at {max_abs_rho:.4f}; below the "
                f"{min_abs_rho:.2f} materiality floor or within {sigma_margin:.0f} SE of "
                "lag 0, so the probe has no power here"
            )

    return AlignmentProbe(
        name=name, reference=reference_name, rho=rho, n_overlap=overlap,
        best_lag=int(best_lag), rho_at_zero=rho_at_zero, max_abs_rho=float(max_abs_rho),
        has_power=has_power, verdict=verdict, notes=notes,
    )


def assert_aligned(
    series: pd.Series,
    reference: pd.Series,
    *,
    name: str = "series",
    reference_name: str = "reference",
    require_power: bool = False,
    **kwargs,
) -> AlignmentProbe:
    """Raise `MisalignedSeriesError` if ``series`` is misaligned against ``reference``.

    ``require_power=True`` also refuses a verdict the probe could not actually establish,
    which is what stops a market-neutral series from passing on noise.
    """
    probe = probe_alignment(series, reference, name=name,
                            reference_name=reference_name, **kwargs)
    if probe.verdict == MISALIGNED:
        raise MisalignedSeriesError(probe.describe() + " | " + "; ".join(probe.notes))
    if require_power and not probe.has_power:
        raise MisalignedSeriesError(
            probe.describe() + " | the probe has no power against this reference, so "
            "alignment is UNPROVEN rather than established"
        )
    if require_power and probe.verdict == INSUFFICIENT_OVERLAP:
        raise MisalignedSeriesError(probe.describe())
    return probe
