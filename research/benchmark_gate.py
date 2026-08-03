"""BENCHMARK-RELATIVE ELIGIBILITY — the Form-B criterion adopted 2026-07-28.

Implements, exactly, the registered wording of
``docs/project-control/specs/2026-07-28-benchmark-relative-gate-review.md`` section 5
(verdict: ADOPT-WITH-CONDITIONS, preconditions C1-C9). The review's own summary of what
this buys: it is a control, not a lever — it rescues no rejected sleeve, raises no
Sharpe, and converts the run's human 3am matched-volatility rule into a machine rule.

Guardrails carried by this module:

- **C1 — scope.** Sleeve-promotion path ONLY. This is a SEPARATE function, deliberately
  NOT wired into ``selection_rule``: the shared callers at ``ops/model_registry.py`` and
  ``learning/adaptive_weights.py`` are untouched and must stay that way.
- **Conjunctive only** (registered wording, item 7). This criterion is ADDED to the seven
  absolute criteria. It replaces nothing, relaxes nothing, and may not be cited as
  grounds for relaxing any absolute criterion. Cumulative ``n_trials`` and the
  ``DSR >= 0.95`` threshold are not amended by this change.
- **C2 — no permissive default.** The verdict is always computed; ``UNDETERMINED``
  FORBIDS promotion. Nothing in this module defaults to a pass.
- **C3 — registration.** The nominated benchmark (construction rule, instrument set,
  rebalancing frequency, cost bracket, excess-return convention, observation window)
  must be registered in a committed file BEFORE any candidate result is inspected.
  That is procedural and cannot be verified here; callers carry it, and the git
  timestamp of the registration is the evidence.
- **C4 — identical window.** Asserted, not assumed: an index mismatch raises.
- **C5 — paired.** Significance uses a paired stationary bootstrap resampling BOTH
  series on a shared index; ``rho(candidate, benchmark)`` is always reported.
- **C6 — three-way verdict.** ``BEATS`` / ``LOSES`` / ``UNDETERMINED``; only ``BEATS``
  permits promotion.
- **C7 — panel.** The verdict is recomputed against every member of the registered
  benchmark panel (for this programme the registered minimum is three: own-universe
  equal weight, passive monthly EW of the 18 instruments, passive daily EW of the
  same). A candidate that BEATS its nominated benchmark but not every panel member is
  flagged ``benchmark_sensitive`` and is NOT promotable without an explicit written
  justification.

Registered wording implemented (section 5 "Exact registered wording"):
(1) identical window, asserted; (2) direction — candidate net Sharpe strictly exceeds
the benchmark's under the same estimator and cost bracket; (3) significance — the
one-sided 90% lower confidence bound on the Sharpe DIFFERENCE from a paired stationary
bootstrap (expected block length 6 months, B >= 10,000) is >= 0; (4) three-way verdict;
(5) panel check; (6) separate function, promotion path only; (7) conjunctive only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "BenchmarkComparison",
    "BenchmarkRelativeVerdict",
    "benchmark_relative_rule",
    "paired_sharpe_comparison",
]

#: Registered bootstrap parameters (section 5, criterion 3).
EXPECTED_BLOCK_MONTHS: int = 6
MIN_BOOTSTRAP_DRAWS: int = 10_000


@dataclass(frozen=True)
class BenchmarkComparison:
    """One candidate-vs-benchmark comparison under the registered criterion."""

    verdict: str                # "BEATS" | "LOSES" | "UNDETERMINED"
    candidate_sharpe: float     # annualised, same estimator both sides
    benchmark_sharpe: float
    sharpe_gap: float           # candidate - benchmark, point estimate
    rho: float                  # Pearson correlation on the shared window (C5: reported)
    diff_lower_90: float        # one-sided 90% LOWER bound of the Sharpe difference
    diff_upper_90: float        # one-sided 90% UPPER bound (used for LOSES)
    n_obs: int


@dataclass(frozen=True)
class BenchmarkRelativeVerdict:
    """The full registered outcome: nominated benchmark plus the panel check."""

    nominated: BenchmarkComparison
    panel: dict[str, BenchmarkComparison]
    benchmark_sensitive: bool   # BEATS nominated but not BEATS against a panel member
    promotable: bool            # BEATS nominated AND BEATS every panel member


def _annualised_sharpe(x: np.ndarray, periods_per_year: int) -> float:
    sd = float(np.std(x, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(x) / sd) * float(np.sqrt(periods_per_year))


def _stationary_bootstrap_indices(
    n: int, draws: int, expected_block: int, rng: np.random.Generator
) -> np.ndarray:
    """(draws, n) circular stationary-bootstrap index matrix (Politis-Romano 1994).

    Each row is a resampled index path: with probability 1/expected_block the next
    index restarts uniformly, otherwise it continues the current block (circularly).
    Vectorised across draws; the loop is over n only.
    """
    p = 1.0 / float(expected_block)
    out = np.empty((draws, n), dtype=np.int64)
    out[:, 0] = rng.integers(0, n, size=draws)
    restart = rng.random((draws, n)) < p
    fresh = rng.integers(0, n, size=(draws, n))
    for t in range(1, n):
        out[:, t] = np.where(restart[:, t], fresh[:, t], (out[:, t - 1] + 1) % n)
    return out


def paired_sharpe_comparison(
    candidate: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int,
    *,
    n_boot: int = MIN_BOOTSTRAP_DRAWS,
    expected_block: int = EXPECTED_BLOCK_MONTHS,
    seed: int = 20260728,
) -> BenchmarkComparison:
    """The registered criterion for ONE benchmark: direction + paired significance.

    Both series must share an identical observation index (C4 — raises otherwise) and
    be net returns under the same cost bracket (the caller's registration duty, C3).
    The bootstrap resamples BOTH series on a SHARED index path (C5 — paired), so the
    correlation between them tightens the confidence interval exactly as it should.

    Verdict (C6):
      - ``BEATS``        — gap > 0 AND the one-sided 90% lower bound of the difference
                           is >= 0. Permits promotion (subject to the panel check).
      - ``LOSES``        — gap < 0 AND the one-sided 90% upper bound is <= 0
                           (significantly worse).
      - ``UNDETERMINED`` — anything else. Permits continued research; FORBIDS promotion.
    """
    if not candidate.index.equals(benchmark.index):
        raise ValueError(
            "benchmark_gate: candidate and benchmark must share an IDENTICAL "
            "observation window (C4 - assert, do not assume). Align them explicitly "
            "and re-register the window before calling."
        )
    if candidate.isna().any() or benchmark.isna().any():
        raise ValueError("benchmark_gate: NaNs in inputs - align and clean explicitly.")
    if n_boot < MIN_BOOTSTRAP_DRAWS:
        raise ValueError(f"benchmark_gate: B >= {MIN_BOOTSTRAP_DRAWS} is registered; got {n_boot}.")
    n = len(candidate)
    if n < 24:
        raise ValueError("benchmark_gate: fewer than 24 shared observations is not a comparison.")

    cand = candidate.to_numpy(dtype=float)
    bench = benchmark.to_numpy(dtype=float)

    sr_c = _annualised_sharpe(cand, periods_per_year)
    sr_b = _annualised_sharpe(bench, periods_per_year)
    gap = sr_c - sr_b
    rho = float(np.corrcoef(cand, bench)[0, 1]) if n > 2 else 0.0

    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_indices(n, n_boot, expected_block, rng)
    c_draws = cand[idx]
    b_draws = bench[idx]

    def _sharpes(mat: np.ndarray) -> np.ndarray:
        mean = mat.mean(axis=1)
        sd = mat.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(sd > 0.0, mean / sd, 0.0)
        return raw * np.sqrt(float(periods_per_year))

    diffs = _sharpes(c_draws) - _sharpes(b_draws)
    lower90 = float(np.quantile(diffs, 0.10))
    upper90 = float(np.quantile(diffs, 0.90))

    if gap > 0.0 and lower90 >= 0.0:
        verdict = "BEATS"
    elif gap < 0.0 and upper90 <= 0.0:
        verdict = "LOSES"
    else:
        verdict = "UNDETERMINED"

    return BenchmarkComparison(
        verdict=verdict,
        candidate_sharpe=sr_c,
        benchmark_sharpe=sr_b,
        sharpe_gap=gap,
        rho=rho,
        diff_lower_90=lower90,
        diff_upper_90=upper90,
        n_obs=n,
    )


def benchmark_relative_rule(
    candidate: pd.Series,
    nominated_benchmark: pd.Series,
    panel: Mapping[str, pd.Series],
    periods_per_year: int,
    *,
    n_boot: int = MIN_BOOTSTRAP_DRAWS,
    expected_block: int = EXPECTED_BLOCK_MONTHS,
    seed: int = 20260728,
) -> BenchmarkRelativeVerdict:
    """The full registered benchmark-relative eligibility check (C1-C7).

    ``panel`` is the REGISTERED benchmark panel (C7) and must be non-empty — the check
    runs every time, not on request. For this programme the registered minimum panel is
    (i) own-universe equal weight, (ii) passive monthly EW of the 18 instruments,
    (iii) passive daily EW of the same.

    ``promotable`` is True only when the candidate BEATS the nominated benchmark AND
    every panel member. ``benchmark_sensitive`` marks the exact failure mode the review
    demonstrated (a book that beats its nominated benchmark but loses to another
    registered opportunity cost); it may not be promoted without an explicit written
    justification, which this function cannot grant.
    """
    if not panel:
        raise ValueError(
            "benchmark_gate: the registered benchmark PANEL is required (C7 - the "
            "benchmark-shopping detector runs every time)."
        )
    nominated = paired_sharpe_comparison(
        candidate, nominated_benchmark, periods_per_year,
        n_boot=n_boot, expected_block=expected_block, seed=seed,
    )
    panel_results: dict[str, BenchmarkComparison] = {}
    for name, series in panel.items():
        panel_results[name] = paired_sharpe_comparison(
            candidate, series, periods_per_year,
            n_boot=n_boot, expected_block=expected_block, seed=seed,
        )

    beats_nominated = nominated.verdict == "BEATS"
    beats_all_panel = all(r.verdict == "BEATS" for r in panel_results.values())
    return BenchmarkRelativeVerdict(
        nominated=nominated,
        panel=panel_results,
        benchmark_sensitive=beats_nominated and not beats_all_panel,
        promotable=beats_nominated and beats_all_panel,
    )
