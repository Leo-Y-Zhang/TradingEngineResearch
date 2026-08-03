"""Is the FX residual's regime asymmetry real, or an alignment artefact?

Registered in ``fx_shape_reality_prereg.md``, committed before any statistic here
was computed. This is the THIRD registered test on the residual and deliberately
attacks a different premise from the first two: they assumed a shape existed and
hunted its cause, this asks whether the shape is distinguishable from an
autocorrelated, near-zero-drift series that happens to line up with the rate regime.

The nulls preserve the residual's own serial dependence and destroy only its
ALIGNMENT with the regime labels. That is the whole idea: a margin of any form must
survive the alignment being broken, an artefact cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12

#: Committed elsewhere in this package; re-stated so this module is readable alone.
LOW_RATE_THRESHOLD: float = 0.005


def regime_mask(frame: pd.DataFrame, *, threshold: float = LOW_RATE_THRESHOLD) -> np.ndarray:
    """True where the month is LOW-rate, on the committed foreign 3-month rule."""
    return (frame["i3m_foreign"] <= threshold).to_numpy()


def asymmetry(values: np.ndarray, low: np.ndarray) -> float:
    """Annualised high-minus-low gap in percent per year.

    Arithmetic annualisation, matching ``fx_residual.annualise``, so this is
    comparable with every number already committed.
    """
    if low.all() or (~low).all():
        return float("nan")
    gap = values[~low].mean() - values[low].mean()
    return float(gap * MONTHS_PER_YEAR * 100.0)


def pooled_statistic(per_leg: dict[str, float]) -> float:
    """S = mean over legs of |A_L|, as registered."""
    vals = [abs(v) for v in per_leg.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _circular_shift(values: np.ndarray, tau: int) -> np.ndarray:
    return np.concatenate([values[-tau:], values[:-tau]])


def circular_shift_null(
    legs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    draws: int = 10_000,
    seed: int = 20260731,
) -> np.ndarray:
    """Null distribution of S under random re-alignment.

    Each leg is shifted by its OWN random offset, which breaks the alignment between
    residual and regime while preserving each residual's autocorrelation exactly.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=float)
    for b in range(draws):
        per_leg = {}
        for name, (values, low) in legs.items():
            n = len(values)
            tau = int(rng.integers(1, n)) if n > 1 else 0
            per_leg[name] = asymmetry(_circular_shift(values, tau), low)
        out[b] = pooled_statistic(per_leg)
    return out


def _stationary_bootstrap_index(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, wrapping at the end."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def block_bootstrap_null(
    legs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    draws: int = 10_000,
    mean_block: float = 6.0,
    seed: int = 20260731,
) -> np.ndarray:
    """Second opinion with different assumptions, as registered.

    The residual is resampled in blocks (keeping short-range dependence) and laid
    against the UNCHANGED regime labels, so again only the alignment is destroyed.
    """
    rng = np.random.default_rng(seed + 1)
    out = np.empty(draws, dtype=float)
    for b in range(draws):
        per_leg = {}
        for name, (values, low) in legs.items():
            idx = _stationary_bootstrap_index(len(values), mean_block, rng)
            per_leg[name] = asymmetry(values[idx], low)
        out[b] = pooled_statistic(per_leg)
    return out


def p_value(observed: float, null: np.ndarray) -> float:
    """Fraction of null draws at least as extreme, with the usual +1 correction."""
    ge = int(np.sum(null >= observed))
    return float((ge + 1) / (len(null) + 1))


def verdict(p_shift: float, p_block: float) -> str:
    """The decision rule, fixed in the pre-registration before any run."""
    if p_shift < 0.05 and p_block < 0.05:
        return "REAL"
    if p_shift >= 0.10 and p_block >= 0.10:
        return "ARTEFACT"
    return "UNDETERMINED"


def minimum_detectable_effect(
    legs: dict[str, tuple[np.ndarray, np.ndarray]],
    null: np.ndarray,
    *,
    quantile: float = 0.95,
) -> float:
    """Smallest injected high-minus-low gap (%/yr) that would clear the null.

    The prereg requires reporting what the test can SEE, not merely its verdict: a
    null result from an underpowered test is worth nothing.
    """
    return float(np.quantile(null, quantile))
