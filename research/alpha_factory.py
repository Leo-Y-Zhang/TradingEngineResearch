"""
TradingEngineResearch — Alpha Factory and Signal Health
============================================
The gatekeeper for signal promotion. A signal that does not pass
selection_rule() is never deployed, regardless of raw performance.

Provides:
  - SignalOutput  : standardised dataclass returned by every strategy sleeve
  - evaluate_factor   : run purged walk-forward validation on a factor
  - factor_decay_profile : IC at horizons [1, 3, 5, 10, 20]
  - orthogonalize_factor : Gram-Schmidt orthogonalization
  - cluster_factors      : hierarchical clustering by correlation
  - promote_factor       : full promotion gate (selection_rule + diversity checks)
  - sleeve_health_weight / apply_signal_health : pipeline STEP-5 signal-health
                           downweighting (low stability) and disabling (failed gate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from research.validation import (
    PurgedWalkForwardSplitter,
    ValidationResult,
    deflated_sharpe_ratio,
    evaluate_alpha_stability,
    leakage_guard,
    selection_rule,
)

logger = logging.getLogger(__name__)


# ── SignalOutput ──────────────────────────────────────────────────────────────

@dataclass
class SignalOutput:
    """
    Standardised output from every strategy sleeve.

    Every strategy module must return list[SignalOutput].
    No raw floats, dicts, or scores — only this dataclass.
    The pipeline consumes this format; deviating breaks the pipeline.
    """
    symbol: str
    direction: str              # "BUY" | "SELL" | "FLAT"
    raw_score: float            # [-1, 1] — negative = bearish signal
    expected_horizon: int       # expected holding period in days
    decay_half_life: int        # signal decay half-life in days
    confidence_proxy: float     # [0, 1] — 0 = no confidence
    sleeve_name: str            # e.g. "momentum", "mean_reversion"
    asof_timestamp: datetime    # PIT timestamp when signal was generated

    def __post_init__(self) -> None:
        valid_directions = {"BUY", "SELL", "FLAT"}
        if self.direction not in valid_directions:
            raise ValueError(
                f"direction must be one of {valid_directions}, got '{self.direction}'"
            )
        if not -1.0 <= self.raw_score <= 1.0:
            raise ValueError(
                f"raw_score must be in [-1, 1], got {self.raw_score}"
            )
        if not 0.0 <= self.confidence_proxy <= 1.0:
            raise ValueError(
                f"confidence_proxy must be in [0, 1], got {self.confidence_proxy}"
            )
        if self.expected_horizon <= 0:
            raise ValueError(
                f"expected_horizon must be > 0, got {self.expected_horizon}"
            )
        if self.decay_half_life <= 0:
            raise ValueError(
                f"decay_half_life must be > 0, got {self.decay_half_life}"
            )


# ── Default splitter parameters ───────────────────────────────────────────────

_DEFAULT_SPLITTER = PurgedWalkForwardSplitter(
    train_size=252,
    valid_size=63,
    test_size=63,
    embargo_size=5,
    label_horizon=5,
)


# ── evaluate_factor ───────────────────────────────────────────────────────────

def evaluate_factor(
    factor_series: pd.Series,
    returns_df: pd.DataFrame,
    costs_bps: float = 5.0,
    splitter: Optional[PurgedWalkForwardSplitter] = None,
    regime_labels: Optional[pd.Series] = None,
    n_trials: Optional[int] = None,
) -> ValidationResult:
    """
    Run purged walk-forward validation on a factor signal.

    Parameters
    ----------
    factor_series : Series indexed by datetime — the raw factor values
    returns_df    : DataFrame indexed by datetime — forward returns per symbol
    costs_bps     : round-trip transaction cost in bps (used for net Sharpe)
    splitter      : optional custom PurgedWalkForwardSplitter; uses default if None
    regime_labels : optional Series indexed by datetime — regime label per bar
    n_trials      : true programme-level trial count for DSR deflation. None (default)
                    reads the cumulative trial ledger — NEVER the fold count (GATE-1 fix
                    per the 2026-07-28 gate review, section 1.4). An explicit value is
                    floored at 1; the ledger default is floored at len(splits).

    Returns
    -------
    ValidationResult with all fields populated
    """
    if splitter is None:
        splitter = _DEFAULT_SPLITTER

    # Align factor and returns on common timestamps
    common_idx = factor_series.index.intersection(returns_df.index)
    if len(common_idx) < splitter.train_size + splitter.valid_size + splitter.test_size:
        logger.warning(
            "evaluate_factor: insufficient data (%d observations, need %d)",
            len(common_idx),
            splitter.train_size + splitter.valid_size + splitter.test_size,
        )
        return ValidationResult(
            mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.0, turnover=0.0,
            hit_rate=0.0, max_drawdown=0.0, pbo_proxy=0.5,
            deflated_sharpe_proxy=0.0, cost_drag_bps=costs_bps,
            stability_score=0.0,
            leakage_flags=["INSUFFICIENT_DATA"],
        )

    factor_aligned = factor_series.loc[common_idx]
    returns_aligned = returns_df.loc[common_idx]

    # SIGNALS-4 fix: evaluate the factor PREDICTIVELY. The factor at bar t is scored
    # against the FORWARD market return over the next ``label_horizon`` bars
    # (t+1 .. t+h), not the contemporaneous same-bar return. Without this the IC was a
    # look-ahead/market-timing artefact, and random junk could be spuriously
    # "validated". The unlabelable tail bars become NaN and are dropped per fold.
    h = max(int(splitter.label_horizon), 1)
    fwd_mkt = returns_aligned.mean(axis=1).rolling(h).sum().shift(-h).to_numpy()

    ts_index = pd.DatetimeIndex(common_idx)
    splits = splitter.split(ts_index)

    if not splits:
        return ValidationResult(
            mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.0, turnover=0.0,
            hit_rate=0.0, max_drawdown=0.0, pbo_proxy=0.5,
            deflated_sharpe_proxy=0.0, cost_drag_bps=costs_bps,
            stability_score=0.0,
            leakage_flags=["NO_VALID_SPLITS"],
        )

    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    net_returns: list[float] = []
    oos_period_returns: list[float] = []   # per-bar long-book net returns across all OOS folds (for DSR)

    for train_idx, valid_idx, _test_idx in splits:
        # OOS forward-return IC on the validation fold; drop unlabelable tail bars
        # (NaN forward return) so the score is strictly predictive.
        vf = factor_aligned.iloc[valid_idx].to_numpy(dtype=float)
        vr = fwd_mkt[valid_idx]
        finite = np.isfinite(vf) & np.isfinite(vr)
        val_factor = vf[finite]
        val_returns = vr[finite]

        if len(val_factor) < 3:
            continue

        # Pearson IC
        if np.std(val_factor) > 0 and np.std(val_returns) > 0:
            ic = float(np.corrcoef(val_factor, val_returns)[0, 1])
        else:
            ic = 0.0
        ic_values.append(ic)

        # Rank IC (Spearman)
        rank_ic, _ = spearmanr(val_factor, val_returns)
        rank_ic_values.append(float(rank_ic) if np.isfinite(rank_ic) else 0.0)

        # Net return (simplified: long top-decile, penalise by cost)
        top_threshold = np.percentile(val_factor, 80)
        long_mask = val_factor >= top_threshold
        gross_ret = float(np.mean(val_returns[long_mask])) if long_mask.any() else 0.0
        cost_daily = (costs_bps / 10_000) / max(splitter.label_horizon, 1)
        net_returns.append(gross_ret - cost_daily)
        if long_mask.any():
            oos_period_returns.extend((val_returns[long_mask] - cost_daily).tolist())

    if not ic_values:
        return ValidationResult(
            mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.0, turnover=0.0,
            hit_rate=0.0, max_drawdown=0.0, pbo_proxy=0.5,
            deflated_sharpe_proxy=0.0, cost_drag_bps=costs_bps,
            stability_score=0.0,
            leakage_flags=["NO_IC_COMPUTED"],
        )

    ic_arr = np.array(ic_values)
    rank_ic_arr = np.array(rank_ic_values)
    net_ret_arr = np.array(net_returns)

    mean_ic       = float(np.mean(ic_arr))
    mean_rank_ic  = float(np.mean(rank_ic_arr))
    hit_rate      = float(np.mean(ic_arr > 0))
    stability     = evaluate_alpha_stability(pd.Series(ic_arr))

    # Net Sharpe (annualised, using fold-level net returns as observations)
    net_sharpe = 0.0
    if len(net_ret_arr) >= 2 and np.std(net_ret_arr) > 0:
        fold_sharpe = float(np.mean(net_ret_arr) / np.std(net_ret_arr))
        net_sharpe = fold_sharpe * np.sqrt(252 / max(splitter.valid_size, 1))

    # Max drawdown of cumulative IC
    cumulative_ic = np.cumsum(ic_arr)
    peak = np.maximum.accumulate(cumulative_ic)
    drawdown = (cumulative_ic - peak) / (np.abs(peak) + 1e-9)
    max_dd = float(np.min(drawdown))

    # Deflate against the PROGRAMME's cumulative trial count, never the fold count
    # (GATE-1 fix per the 2026-07-28 gate review, section 1.4: folds are not trials, and
    # deflating at 8 folds made the DSR criterion ~0.14 Sharpe too lenient). The floor of
    # len(splits) means the ledger default can never deflate by less than the folds looked at.
    if n_trials is None:
        from research.trial_ledger import cumulative_trials
        n_trials = max(cumulative_trials(), len(splits), 1)
    else:
        n_trials = max(n_trials, 1)
    deflated_sharpe = net_sharpe / np.sqrt(1 + np.log(n_trials))

    # PBO proxy: fraction of splits where net return < 0
    pbo_proxy = float(np.mean(net_ret_arr < 0))

    # Real Deflated Sharpe Ratio (Bailey & Lopez de Prado): probability the long-book's
    # Sharpe is genuine after deflating for sample length, non-normality, and the number
    # of trials (the cumulative programme ledger by default — GATE-1). Computed on the per-bar OOS returns;
    # hardens the gate against small-sample / multiple-testing overfitting (SIGNALS-6).
    from research.validation import deflated_sharpe_ratio
    dsr = deflated_sharpe_ratio(np.asarray(oos_period_returns, dtype=float), n_trials=n_trials)

    # Regime breakdown
    regime_breakdown: dict[str, dict[str, float]] = {}
    if regime_labels is not None:
        for regime in regime_labels.unique():
            mask = regime_labels.loc[common_idx] == regime
            regime_ic_vals = []
            for train_idx, valid_idx, _ in splits:
                val_idx_dates = common_idx[valid_idx]
                regime_mask = mask.loc[val_idx_dates]
                if not regime_mask.any():
                    continue
                rf = factor_aligned.loc[val_idx_dates][regime_mask].values
                rr = returns_aligned.loc[val_idx_dates][regime_mask].mean(axis=1).values
                if len(rf) >= 2 and np.std(rf) > 0 and np.std(rr) > 0:
                    regime_ic_vals.append(float(np.corrcoef(rf, rr)[0, 1]))
            if regime_ic_vals:
                r_arr = np.array(regime_ic_vals)
                r_sharpe = 0.0
                if np.std(r_arr) > 0:
                    r_sharpe = float(np.mean(r_arr) / np.std(r_arr)) * np.sqrt(252 / max(splitter.valid_size, 1))
                regime_breakdown[str(regime)] = {
                    "ic": float(np.mean(r_arr)),
                    "sharpe": r_sharpe,
                }

    # Leakage check
    leakage_flags = leakage_guard(
        pd.DataFrame(factor_aligned),
        returns_df.loc[common_idx],
        splitter.label_horizon,
    )

    return ValidationResult(
        mean_ic=round(mean_ic, 6),
        mean_rank_ic=round(mean_rank_ic, 6),
        sharpe_net=round(net_sharpe, 4),
        turnover=0.0,           # computed separately in the pipeline
        hit_rate=round(hit_rate, 4),
        max_drawdown=round(max_dd, 4),
        pbo_proxy=round(pbo_proxy, 4),
        deflated_sharpe_proxy=round(deflated_sharpe, 4),
        deflated_sharpe_ratio=round(float(dsr), 4),
        cost_drag_bps=costs_bps,
        stability_score=round(stability, 4),
        regime_breakdown=regime_breakdown,
        leakage_flags=leakage_flags,
    )


# ── factor_decay_profile ──────────────────────────────────────────────────────

def factor_decay_profile(
    factor_series: pd.Series,
    returns_df: pd.DataFrame,
) -> dict[int, float]:
    """
    Compute rank IC at each of the 5 canonical horizons.

    Parameters
    ----------
    factor_series : Series indexed by datetime
    returns_df    : DataFrame of forward returns at various horizons;
                    columns are expected to be horizon days (1, 3, 5, 10, 20)
                    OR a single-column DataFrame for all horizons.

    Returns
    -------
    dict mapping horizon (days) to mean rank IC at that horizon
    """
    horizons = [1, 3, 5, 10, 20]
    profile: dict[int, float] = {}

    common_idx = factor_series.index.intersection(returns_df.index)
    if len(common_idx) < 20:
        logger.warning(
            "factor_decay_profile: fewer than 20 common observations; "
            "profile will be unreliable."
        )
        return {h: 0.0 for h in horizons}

    factor_aligned = factor_series.loc[common_idx]
    # When the caller does NOT supply horizon-named columns, returns_df holds PER-PERIOD
    # returns; build the TRUE h-period FORWARD return (compounded over the next h periods)
    # so each horizon measures real decay — not the identical contemporaneous IC (SIGNALS-3).
    per_period = returns_df.loc[common_idx].mean(axis=1)

    for h in horizons:
        if h in returns_df.columns:
            fwd = returns_df.loc[common_idx, h]                       # pre-computed horizon-h return
        else:
            fwd = (1.0 + per_period).rolling(h).apply(np.prod, raw=True).shift(-h) - 1.0
        pair = pd.concat([factor_aligned.rename("f"), fwd.rename("r")], axis=1).dropna()
        if len(pair) < 2:
            profile[h] = 0.0
            continue
        rank_ic, _ = spearmanr(pair["f"].to_numpy(), pair["r"].to_numpy())
        profile[h] = round(float(rank_ic) if not np.isnan(rank_ic) else 0.0, 6)

    return profile


# ── orthogonalize_factor ──────────────────────────────────────────────────────

def orthogonalize_factor(
    candidate: np.ndarray,
    live_matrix: np.ndarray,
) -> np.ndarray:
    """
    Gram-Schmidt orthogonalization of a candidate factor against a live library.

    Removes the components of `candidate` that are explained by any live factor,
    returning the residual component that is orthogonal to all live factors.

    Parameters
    ----------
    candidate   : 1-D array of length T
    live_matrix : 2-D array of shape (T, n_live_factors)

    Returns
    -------
    orthogonalized factor of length T
    """
    candidate = np.array(candidate, dtype=float)

    if live_matrix.ndim == 1:
        live_matrix = live_matrix.reshape(-1, 1)

    if live_matrix.shape[0] != len(candidate):
        raise ValueError(
            f"candidate length ({len(candidate)}) must match "
            f"live_matrix rows ({live_matrix.shape[0]})"
        )

    if live_matrix.shape[1] == 0:
        return candidate.copy()

    # Full Gram-Schmidt: first orthogonalise the live columns against each other,
    # then project the candidate onto each orthogonal basis and subtract.
    # This ensures correct removal even when live factors are mutually correlated.
    n_live = live_matrix.shape[1]
    ortho_basis = np.zeros_like(live_matrix, dtype=float)
    for j in range(n_live):
        b = live_matrix[:, j].astype(float).copy()
        for k in range(j):
            prev = ortho_basis[:, k]
            norm_sq_prev = float(np.dot(prev, prev))
            if norm_sq_prev > 1e-12:
                b -= float(np.dot(b, prev)) / norm_sq_prev * prev
        ortho_basis[:, j] = b

    residual = candidate.copy()
    for j in range(n_live):
        basis = ortho_basis[:, j]
        norm_sq = float(np.dot(basis, basis))
        if norm_sq > 1e-12:
            residual -= float(np.dot(residual, basis)) / norm_sq * basis

    # Renormalise to unit variance for consistency
    std = float(np.std(residual))
    if std > 1e-12:
        residual = residual / std

    return residual


# ── cluster_factors ───────────────────────────────────────────────────────────

def cluster_factors(
    factor_matrix: np.ndarray,
    factor_names: Optional[list[str]] = None,
    distance_threshold: float = 0.50,
) -> dict[str, list[str]]:
    """
    Hierarchical clustering of factors by correlation distance.

    Distance = 1 - |correlation| so factors with |corr| > 0.5 are in the
    same cluster (below the threshold distance).

    Parameters
    ----------
    factor_matrix      : 2-D array of shape (T, n_factors)
    factor_names       : optional list of n_factors names
    distance_threshold : factors within this distance are grouped

    Returns
    -------
    dict mapping cluster_id (str) to list of factor names/indices
    """
    n = factor_matrix.shape[1] if factor_matrix.ndim == 2 else 1

    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(n)]

    if len(factor_names) != n:
        raise ValueError(
            f"factor_names length ({len(factor_names)}) must match "
            f"number of columns in factor_matrix ({n})"
        )

    if n == 1:
        return {"cluster_0": [factor_names[0]]}

    # Compute correlation matrix and convert to distance matrix
    corr = np.corrcoef(factor_matrix.T)
    np.fill_diagonal(corr, 1.0)
    # Distance: 1 - |corr|  (0 = identical, 1 = orthogonal)
    dist_matrix = 1.0 - np.abs(corr)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

    # Convert to condensed distance vector for scipy
    condensed = squareform(dist_matrix, checks=False)

    # Ward linkage — compact, well-separated clusters
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    clusters: dict[str, list[str]] = {}
    for i, cluster_id in enumerate(labels):
        key = f"cluster_{cluster_id}"
        clusters.setdefault(key, []).append(factor_names[i])

    return clusters


# ── promote_factor ────────────────────────────────────────────────────────────

_LIVE_FACTOR_MATRIX: np.ndarray = np.empty((0, 0))
_LIVE_FACTOR_NAMES: list[str] = []


def promote_factor(
    name: str,
    result: ValidationResult,
    live_matrix: np.ndarray,
    candidate_series: Optional[np.ndarray] = None,
    max_correlation: float = 0.80,
) -> bool:
    """
    Full promotion gate for a factor signal.

    Returns True and logs promotion only if ALL conditions pass:
      1. selection_rule(result) passes
      2. Max |correlation| with any live factor < max_correlation (0.80)
      3. Positive net contribution after costs (sharpe_net > 0)
      4. Improves ensemble diversity (not a near-duplicate of existing)

    Parameters
    ----------
    name             : factor name (for logging)
    result           : ValidationResult from evaluate_factor()
    live_matrix      : 2-D array (T, n_live) of currently-live factor values
    candidate_series : 1-D array of length T for the candidate factor
    max_correlation  : correlation threshold (default 0.80 per spec)

    Returns
    -------
    True if promoted, False otherwise
    """
    # Gate 1: selection rule
    if not selection_rule(result):
        logger.info("promote_factor(%s) REJECTED: selection_rule failed", name)
        return False

    # Gate 2: net positive contribution
    if result.sharpe_net <= 0:
        logger.info(
            "promote_factor(%s) REJECTED: sharpe_net=%.4f <= 0 "
            "(no positive net contribution after costs)", name, result.sharpe_net
        )
        return False

    # Gate 3: correlation check against live library
    if (
        candidate_series is not None
        and live_matrix.ndim == 2
        and live_matrix.shape[1] > 0
        and len(candidate_series) == live_matrix.shape[0]
    ):
        for j in range(live_matrix.shape[1]):
            live_col = live_matrix[:, j].astype(float)
            cand = candidate_series.astype(float)
            std_live = float(np.std(live_col))
            std_cand = float(np.std(cand))
            if std_live < 1e-12 or std_cand < 1e-12:
                continue
            corr = float(np.corrcoef(cand, live_col)[0, 1])
            if abs(corr) >= max_correlation:
                logger.info(
                    "promote_factor(%s) REJECTED: |correlation|=%.4f >= %.2f "
                    "with live factor %d (near-duplicate)", name, abs(corr), max_correlation, j
                )
                return False

    # Gate 4: diversity check via clustering
    if (
        candidate_series is not None
        and live_matrix.ndim == 2
        and live_matrix.shape[1] > 0
        and len(candidate_series) == live_matrix.shape[0]
    ):
        combined = np.column_stack([live_matrix, candidate_series.reshape(-1, 1)])
        n_live = live_matrix.shape[1]
        names = [f"live_{j}" for j in range(n_live)] + [name]
        clusters = cluster_factors(combined, factor_names=names)
        # If candidate clusters with any live factor it's too similar
        for cluster_members in clusters.values():
            if name in cluster_members and any(
                m.startswith("live_") for m in cluster_members
            ):
                logger.info(
                    "promote_factor(%s) REJECTED: clusters with existing live factor "
                    "(ensemble diversity would not improve)", name
                )
                return False

    logger.info(
        "promote_factor(%s) APPROVED: rank_ic=%.4f, sharpe_net=%.4f, "
        "stability=%.4f, deflated_sharpe=%.4f",
        name, result.mean_rank_ic, result.sharpe_net,
        result.stability_score, result.deflated_sharpe_proxy,
    )
    return True


# ── Live factor library + end-to-end promotion pipeline ─────────────────────────

@dataclass
class PromotionOutcome:
    """Per-candidate result of the promotion pipeline."""

    name: str
    promoted: bool
    passed_selection_rule: bool
    result: ValidationResult
    reason: str


def reset_live_factors() -> None:
    """Clear the live factor library (research sessions / tests)."""
    global _LIVE_FACTOR_MATRIX, _LIVE_FACTOR_NAMES
    _LIVE_FACTOR_MATRIX = np.empty((0, 0))
    _LIVE_FACTOR_NAMES = []


def get_live_factors() -> tuple[list[str], np.ndarray]:
    """Return ``(names, matrix)`` for the currently-promoted live factor library."""
    return list(_LIVE_FACTOR_NAMES), _LIVE_FACTOR_MATRIX.copy()


def _append_live_factor(name: str, column: np.ndarray) -> None:
    global _LIVE_FACTOR_MATRIX, _LIVE_FACTOR_NAMES
    col = np.asarray(column, dtype=float).reshape(-1, 1)
    if _LIVE_FACTOR_MATRIX.size and _LIVE_FACTOR_MATRIX.shape[0] == col.shape[0]:
        _LIVE_FACTOR_MATRIX = np.column_stack([_LIVE_FACTOR_MATRIX, col])
    else:
        # Empty library, or a different timeline (new universe) → start fresh.
        _LIVE_FACTOR_MATRIX = col
        _LIVE_FACTOR_NAMES = []
    _LIVE_FACTOR_NAMES.append(name)


def promote_candidates(
    candidates: dict[str, pd.Series],
    returns_df: pd.DataFrame,
    costs_bps: float = 5.0,
    splitter: Optional[PurgedWalkForwardSplitter] = None,
    regime_labels: Optional[pd.Series] = None,
) -> dict[str, PromotionOutcome]:
    """End-to-end factor promotion: evaluate → ``selection_rule`` + diversity gate →
    grow the live factor library.

    Each candidate is validated with :func:`evaluate_factor` (purged walk-forward),
    then gated by :func:`promote_factor` (``selection_rule`` + net contribution +
    ``|corr| < 0.80`` and cluster-diversity vs the already-promoted set). Promoted
    factors are appended to the live library so later candidates are gated against
    them. Returns a per-candidate :class:`PromotionOutcome`. The library accumulates
    within one consistent time index; reset it between universes
    (:func:`reset_live_factors`).
    """
    outcomes: dict[str, PromotionOutcome] = {}
    index = returns_df.index
    for name, factor_series in candidates.items():
        result = evaluate_factor(factor_series, returns_df, costs_bps, splitter, regime_labels)
        passed = selection_rule(result)
        column = factor_series.reindex(index).ffill().fillna(0.0).to_numpy(dtype=float)
        promoted = promote_factor(name, result, _LIVE_FACTOR_MATRIX, candidate_series=column)
        if promoted:
            _append_live_factor(name, column)
            reason = "promoted"
        elif not passed:
            reason = "selection_rule_failed"
        else:
            reason = "rejected_by_diversity_or_contribution"
        outcomes[name] = PromotionOutcome(
            name=name, promoted=promoted, passed_selection_rule=passed,
            result=result, reason=reason,
        )
    return outcomes


# ── Per-sleeve validation registry (signal-health gate) ─────────────────────────

_SLEEVE_VALIDATION: dict[str, ValidationResult] = {}


def register_sleeve_validation(sleeve: str, result: ValidationResult) -> None:
    """Record a sleeve's validated quality (from purged walk-forward) for the STEP-5
    signal-health gate. A sleeve whose result fails ``selection_rule`` is disabled."""
    _SLEEVE_VALIDATION[sleeve] = result


def get_sleeve_validation(sleeve: str) -> Optional[ValidationResult]:
    """Return the registered ``ValidationResult`` for a sleeve, or None if un-validated."""
    return _SLEEVE_VALIDATION.get(sleeve)


def reset_sleeve_validation() -> None:
    """Clear the per-sleeve validation registry (research sessions / tests)."""
    _SLEEVE_VALIDATION.clear()


# ── Signal health filter (pipeline STEP 5) ──────────────────────────────────────

STABILITY_FLOOR = 0.40   # sleeves below this stability score are downweighted

# SIGNALS-5 default-deny floor: an un-validated sleeve (no registered ValidationResult)
# has NOT passed purged walk-forward + selection_rule, so it must not drive allocation at
# full weight (golden rule 5). It is capped here regardless of this-cycle confidence. A
# small non-zero floor keeps the sleeve pipeline alive/testable off-LIVE; the engine
# tightens this to 0.0 in LIVE (no un-validated sleeve touches real money).
UNVALIDATED_SLEEVE_WEIGHT = 0.25


def sleeve_health_weight(
    stability_score: float,
    validation: Optional[ValidationResult] = None,
    unvalidated_weight: float = UNVALIDATED_SLEEVE_WEIGHT,
) -> float:
    """
    Health multiplier in [0, 1] for a sleeve's signals.

      - DEFAULT-DENY (SIGNALS-5): if NO ValidationResult is registered, the sleeve is
        un-validated — capped at ``unvalidated_weight`` (a small floor; pass 0.0, the LIVE
        posture, to disable it) REGARDLESS of this-cycle confidence. It can never earn
        full weight on un-validated signals.
      - 0.0 (disabled) if a ValidationResult is supplied and selection_rule fails.
      - else (validated & passes): 1.0 if stability_score >= STABILITY_FLOOR (0.40), or a
        linear ramp ``stability_score / STABILITY_FLOOR`` below the floor.
    """
    if validation is None:
        return float(np.clip(unvalidated_weight, 0.0, 1.0))
    if not selection_rule(validation):
        return 0.0
    if stability_score >= STABILITY_FLOOR:
        return 1.0
    return float(np.clip(stability_score / STABILITY_FLOOR, 0.0, 1.0))


def apply_signal_health(
    signals: list[SignalOutput],
    stability_score: float,
    validation: Optional[ValidationResult] = None,
    unvalidated_weight: float = UNVALIDATED_SLEEVE_WEIGHT,
) -> list[SignalOutput]:
    """
    Apply a sleeve's health weight to its signals (pipeline STEP 5).

    Scales each signal's ``confidence_proxy`` by the health weight. A fully
    disabled sleeve (weight 0 — ``selection_rule`` fails, or an un-validated sleeve with
    ``unvalidated_weight=0.0``) is forced to FLAT with zero score and confidence. Returns
    new ``SignalOutput`` instances; the inputs are not mutated.
    """
    weight = sleeve_health_weight(stability_score, validation, unvalidated_weight)

    adjusted: list[SignalOutput] = []
    for signal in signals:
        if weight <= 0.0:
            adjusted.append(
                replace(signal, direction="FLAT", raw_score=0.0, confidence_proxy=0.0)
            )
        else:
            new_conf = float(np.clip(signal.confidence_proxy * weight, 0.0, 1.0))
            adjusted.append(replace(signal, confidence_proxy=new_conf))

    if weight < 1.0 and signals:
        logger.info(
            "apply_signal_health: %s sleeve downweighted "
            "(stability=%.3f, weight=%.3f) across %d signals",
            signals[0].sleeve_name, stability_score, weight, len(signals),
        )
    return adjusted


# ── Learned signal combination (alpha-slice: replace the naive equal-weight blend) ──

def _failing_learn_result(names: list[str], cost_drag_bps: float) -> "tuple[dict[str, float], ValidationResult]":
    return ({n: 0.0 for n in names}, ValidationResult(
        mean_ic=0.0, mean_rank_ic=0.0, sharpe_net=0.0, turnover=0.0, hit_rate=0.0,
        max_drawdown=0.0, pbo_proxy=0.0, deflated_sharpe_proxy=0.0,
        cost_drag_bps=float(cost_drag_bps), stability_score=0.0, deflated_sharpe_ratio=0.0,
        leakage_flags=["insufficient_data"]))


def learn_signal_weights(
    signal_panel: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    splitter: Optional[PurgedWalkForwardSplitter] = None,
    l2: float = 1.0,
    n_trials: Optional[int] = None,
    cost_drag_bps: float = 10.0,
    periods_per_year: int = 252,
) -> "tuple[dict[str, float], ValidationResult]":
    """Learn a REGULARISED cross-sectional combination of signals via purged walk-forward.

    Returns ``({signal_name: weight}, ValidationResult)``. The weights are ridge
    coefficients averaged across out-of-sample folds; the ``ValidationResult`` (OOS IC,
    rank-IC, net Sharpe, real Deflated Sharpe) lets :func:`selection_rule` GATE deployment
    — default-deny (golden rule 5 / SIGNALS-5): a combination that does not survive
    deflation must NOT replace the incumbent equal-weight blend. This is the credible
    alpha-slice vehicle (a learned, regularised, OOS-validated combination) — it captures
    a real edge if one exists and honestly rejects noise if it does not.

    Inputs are aligned panels: ``signal_panel[name]`` and ``forward_returns`` are
    (dates × symbols); ``forward_returns.loc[t, s]`` is the return earned AFTER date ``t``
    (the caller shifts it, so a signal at ``t`` predicts it — no look-ahead). Ridge is
    closed-form (deterministic). Fails closed (zero weights + a failing result) on
    degenerate input."""
    names = sorted(signal_panel.keys())
    if not names:
        return _failing_learn_result(names, cost_drag_bps)

    dates = forward_returns.index
    symbols = forward_returns.columns
    for df in signal_panel.values():
        dates = dates.intersection(df.index)
        symbols = symbols.intersection(df.columns)
    dates = dates.sort_values()
    symbols = list(symbols)
    if len(dates) < 8 or len(symbols) < 3:
        return _failing_learn_result(names, cost_drag_bps)

    S = {n: signal_panel[n].reindex(index=dates, columns=symbols).to_numpy(dtype=float) for n in names}
    Y = forward_returns.reindex(index=dates, columns=symbols).to_numpy(dtype=float)
    k = len(names)

    if splitter is None:
        n = len(dates)
        test = max(2, n // 5)
        splitter = PurgedWalkForwardSplitter(
            train_size=max(4, n - 3 * test), valid_size=test, test_size=test,
            embargo_size=1, label_horizon=1)
    try:
        folds = splitter.split(dates)
    except ValueError:
        return _failing_learn_result(names, cost_drag_bps)
    if not folds:
        return _failing_learn_result(names, cost_drag_bps)

    fold_weights: list[np.ndarray] = []
    oos_ic: list[float] = []
    oos_rank_ic: list[float] = []
    oos_ret: list[float] = []
    for train_idx, _valid_idx, test_idx in folds:
        xs, ys = [], []
        for ti in train_idx:
            sig = np.column_stack([S[n][ti] for n in names])     # (n_symbols, k)
            yv = Y[ti]
            mask = np.isfinite(yv) & np.all(np.isfinite(sig), axis=1)
            if mask.any():
                xs.append(sig[mask])
                ys.append(yv[mask])
        if not xs:
            continue
        X = np.vstack(xs)
        y = np.concatenate(ys)
        if X.shape[0] < k + 2:
            continue
        try:
            w = np.linalg.solve(X.T @ X + l2 * np.eye(k), X.T @ y)   # ridge, closed-form
        except np.linalg.LinAlgError:
            continue
        fold_weights.append(w)
        for ti in test_idx:
            sig = np.column_stack([S[n][ti] for n in names])
            yv = Y[ti]
            mask = np.isfinite(yv) & np.all(np.isfinite(sig), axis=1)
            if int(mask.sum()) < 3:
                continue
            combined = sig[mask] @ w
            yvm = yv[mask]
            if np.std(combined) <= 0.0 or np.std(yvm) <= 0.0:
                continue
            oos_ic.append(float(np.corrcoef(combined, yvm)[0, 1]))
            rho = spearmanr(combined, yvm).correlation
            if np.isfinite(rho):
                oos_rank_ic.append(float(rho))
            wcs = combined - combined.mean()
            denom = float(np.abs(wcs).sum())
            if denom > 0.0:
                wcs = wcs / denom                                # dollar-neutral, gross 1
                oos_ret.append(float(wcs @ yvm) - cost_drag_bps / 1e4)

    if not fold_weights or len(oos_ic) < 4:
        return _failing_learn_result(names, cost_drag_bps)

    weights = np.mean(np.vstack(fold_weights), axis=0)
    weights_dict = {n: float(weights[i]) for i, n in enumerate(names)}

    ic = np.asarray(oos_ic, dtype=float)
    ret = np.asarray(oos_ret, dtype=float)
    mean_ic = float(np.mean(ic))
    mean_rank_ic = float(np.mean(oos_rank_ic)) if oos_rank_ic else 0.0
    sd = float(np.std(ret, ddof=1)) if ret.size > 1 else 0.0
    sharpe_net = float(np.mean(ret) / sd * np.sqrt(periods_per_year)) if sd > 0.0 else 0.0
    hit_rate = float(np.mean(ic > 0.0))
    stability = float(np.mean(np.sign(ic) == np.sign(mean_ic))) if mean_ic != 0.0 else 0.0
    if n_trials is None:
        # GATE-1: the default deflation is the programme's cumulative trial ledger.
        from research.trial_ledger import cumulative_trials
        n_trials = cumulative_trials()
    dsr = float(deflated_sharpe_ratio(ret, n_trials=n_trials))
    if ret.size:
        cum = np.cumprod(1.0 + ret)
        peak = np.maximum.accumulate(cum)
        mdd = float(np.max((peak - cum) / np.where(peak > 0, peak, 1.0)))
    else:
        mdd = 0.0

    result = ValidationResult(
        mean_ic=mean_ic, mean_rank_ic=mean_rank_ic, sharpe_net=sharpe_net,
        turnover=1.0, hit_rate=hit_rate, max_drawdown=mdd, pbo_proxy=0.0,
        deflated_sharpe_proxy=dsr, cost_drag_bps=float(cost_drag_bps),
        stability_score=stability, deflated_sharpe_ratio=dsr, leakage_flags=[])
    return weights_dict, result
