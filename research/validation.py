"""
TradingEngineResearch — Research Validation and Leakage Controls
======================================================
The only permitted cross-validation implementation in TradingEngineResearch.

All model training that reports performance metrics MUST use
PurgedWalkForwardSplitter with embargo. No standard k-fold,
no TimeSeriesSplit, no train/test split in research paths.

Reference: López de Prado (2018), Advances in Financial Machine Learning,
Chapter 7.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

logger = logging.getLogger(__name__)


# ── ValidationResult ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Full validation output for a strategy signal or ML model.
    Used by selection_rule() and promote_factor().
    """
    mean_ic: float                          # mean information coefficient
    mean_rank_ic: float                     # mean rank IC (Spearman)
    sharpe_net: float                       # net-of-cost annualised Sharpe
    turnover: float                         # average daily turnover (fraction)
    hit_rate: float                         # fraction of periods with positive IC
    max_drawdown: float                     # max drawdown of cumulative IC series
    pbo_proxy: float                        # probability of backtest overfitting proxy
    deflated_sharpe_proxy: float            # Sharpe adjusted for multiple testing
    cost_drag_bps: float                    # estimated cost drag in bps per trade
    stability_score: float                  # IC consistency [0, 1]
    deflated_sharpe_ratio: float = 0.0      # real Deflated Sharpe (probability); 0.0 = default-DENY
    # (GATE-2 fix per docs/project-control/specs/2026-07-28-benchmark-relative-gate-review.md
    #  section 1.4: the old default of 1.0 was a default-ALLOW on a criterion documented as
    #  default-deny. A result that never computed its real DSR must fail the gate, not pass.)
    regime_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    # e.g. {"trending": {"sharpe": 1.2, "ic": 0.03}, "mean_reverting": {...}}
    leakage_flags: list[str] = field(default_factory=list)
    # Non-empty means a leakage problem was detected — signal cannot be promoted


# ── PurgedWalkForwardSplitter ─────────────────────────────────────────────────

class PurgedWalkForwardSplitter:
    """
    Walk-forward cross-validation with purging and embargo.

    Purging removes training observations whose label window
    [t_i, t_i + label_horizon] overlaps with any timestamp in
    the validation or test window. Embargo removes an additional
    `embargo_size` bars after each training window to prevent
    look-ahead leakage through lagged features.

    Parameters
    ----------
    train_size    : number of observations in each training window
    valid_size    : number of observations in each validation window
    test_size     : number of observations in each test window
    embargo_size  : bars to exclude after training window ends
    label_horizon : forward-return window length (used for purging)
    """

    def __init__(
        self,
        train_size: int,
        valid_size: int,
        test_size: int,
        embargo_size: int,
        label_horizon: int,
    ) -> None:
        if train_size <= 0 or valid_size <= 0 or test_size <= 0:
            raise ValueError("train_size, valid_size and test_size must all be > 0")
        if embargo_size < 0:
            raise ValueError("embargo_size must be >= 0")
        if label_horizon <= 0:
            raise ValueError("label_horizon must be > 0")

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.embargo_size = embargo_size
        self.label_horizon = label_horizon

    def split(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, valid_idx, test_idx) tuples.

        Each index array contains integer positional indices into `timestamps`.
        Purging and embargo are applied to the train_idx of each fold.

        Parameters
        ----------
        timestamps : ordered DatetimeIndex of all observations

        Returns
        -------
        list of (train_idx, valid_idx, test_idx) — one tuple per fold
        """
        n = len(timestamps)
        window = self.train_size + self.valid_size + self.test_size
        if n < window:
            raise ValueError(
                f"Not enough observations ({n}) for one fold "
                f"(need at least {window})."
            )

        splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # Slide the window forward, step = test_size
        start = 0
        while start + window <= n:
            train_end = start + self.train_size                  # exclusive
            valid_end = train_end + self.valid_size              # exclusive
            test_end  = valid_end + self.test_size               # exclusive

            raw_train_idx = np.arange(start, train_end)
            valid_idx = np.arange(train_end, valid_end)
            test_idx  = np.arange(valid_end, test_end)

            # The earliest position in the val/test window.
            eval_start_pos = int(valid_idx[0])

            # Purge: drop training obs whose forward-label window (label_horizon
            # bars) reaches into the evaluation window. Purged POSITIONALLY so a
            # business-day index is handled correctly (SIGNALS-2).
            purged_train_idx = self._purge(raw_train_idx, eval_start_pos)

            # Embargo: additionally remove the `embargo_size` bars immediately
            # before the evaluation window (look-ahead through lagged features).
            purged_train_idx = self._embargo(
                purged_train_idx, train_end
            )

            if len(purged_train_idx) == 0:
                logger.warning(
                    "Fold starting at index %d has zero training observations "
                    "after purging — skipped.", start
                )
                start += self.test_size
                continue

            splits.append((purged_train_idx, valid_idx, test_idx))
            start += self.test_size

        return splits

    def _purge(
        self,
        train_idx: np.ndarray,
        eval_start_pos: int,
    ) -> np.ndarray:
        """
        Remove training observations whose forward-label window overlaps the eval
        window, purging by POSITIONAL distance (SIGNALS-2 fix).

        ``label_horizon`` is a bar count (number of observations), so an obs at
        position ``i`` carries a forward label spanning bars ``[i, i+label_horizon]``
        and overlaps the evaluation window (which starts at ``eval_start_pos``)
        unless ``i + label_horizon < eval_start_pos``. Purging by CALENDAR days
        (``Timedelta(days=label_horizon)``) under-purges on a business-day index -
        e.g. 5 calendar days ~ 3-4 trading days - leaking the tail of the label
        window into the evaluation set.
        """
        idx = np.asarray(train_idx, dtype=int)
        return idx[idx + self.label_horizon < int(eval_start_pos)]

    def _embargo(
        self,
        train_idx: np.ndarray,
        train_end: int,
    ) -> np.ndarray:
        """
        Remove the last `embargo_size` bars of the training window.
        These bars are close enough to the eval window that lagged features
        could carry information forward.
        """
        if self.embargo_size == 0 or len(train_idx) == 0:
            return train_idx
        cutoff = train_end - self.embargo_size
        return train_idx[train_idx < cutoff]


# ── leakage_guard ─────────────────────────────────────────────────────────────

def leakage_guard(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    label_horizon: int,
) -> list[str]:
    """
    Detect timestamp overlap and feature-label contamination.

    Checks:
      1. Feature timestamps must not be later than the corresponding label
         generation date (i.e. no feature computed using future data).
      2. Label and feature DataFrames must share a compatible index
         without future timestamps leaking into the feature set.
      3. The label_horizon must not reach beyond the last available feature.

    Parameters
    ----------
    feature_df    : DataFrame indexed by observation datetime
    label_df      : DataFrame indexed by observation datetime, columns = label names
    label_horizon : forward-return horizon (days)

    Returns
    -------
    list of leakage flag strings — empty means clean.
    """
    flags: list[str] = []

    if feature_df.empty or label_df.empty:
        flags.append("EMPTY_INPUT: feature_df or label_df is empty")
        return flags

    feat_idx  = pd.DatetimeIndex(feature_df.index)
    label_idx = pd.DatetimeIndex(label_df.index)

    # Check 1: feature index must not contain timestamps later than the last label
    if len(feat_idx) > 0 and len(label_idx) > 0:
        if feat_idx.max() > label_idx.max():
            flags.append(
                f"FEATURE_AHEAD_OF_LABELS: latest feature timestamp "
                f"({feat_idx.max()}) > latest label timestamp ({label_idx.max()})"
            )

    # Check 2 (REMOVED - SIGNALS-1 fix): overlapping forward-label windows are the
    # NORMAL property of dense (e.g. daily) sampling with label_horizon > 1, NOT
    # leakage. They are handled by PurgedWalkForwardSplitter's purge+embargo (which
    # drops training obs whose label window reaches into the eval window). Flagging
    # every contiguous panel as LABEL_WINDOW_OVERLAP made selection_rule reject ALL
    # real factors - the promotion gate was a brick wall. Genuine look-ahead is
    # caught by Check 1 (feature dated after labels) and by the splitter's purge.
    label_horizon_td = pd.Timedelta(days=label_horizon)

    # Check 3 (SIGNALS-1 fix): a forward label needs label_horizon bars AFTER the
    # feature. The last ~label_horizon features are simply unlabelable and dropped
    # by the caller - that is the normal forward-return tail, not leakage. Only flag
    # the degenerate case where NOT EVEN THE FIRST feature can be labelled (labels
    # end before feat_start + horizon -> there is no usable training label at all).
    if len(feat_idx) > 0 and len(label_idx) > 0:
        first_labelable = feat_idx.min() + label_horizon_td
        if first_labelable > label_idx.max():
            flags.append(
                f"LABEL_DATA_INSUFFICIENT: labels end {label_idx.max().date()} before "
                f"the earliest feature could be labelled ({first_labelable.date()}) - "
                f"no usable forward labels for horizon={label_horizon}"
            )

    return flags


# ── evaluate_alpha_stability ──────────────────────────────────────────────────

def evaluate_alpha_stability(ic_series: pd.Series) -> float:
    """
    Compute a stability score [0, 1] for a series of per-fold IC values.

    A higher score means the IC is consistently positive across folds
    (not just good on average). Uses:
      - Fraction of folds with positive IC (hit rate component)
      - IC t-statistic proxy (signal strength component)

    Both components are combined and clipped to [0, 1].

    Parameters
    ----------
    ic_series : Series of per-fold IC values

    Returns
    -------
    stability_score in [0, 1]
    """
    if ic_series.empty or len(ic_series) < 2:
        return 0.0

    values = ic_series.dropna().values
    n = len(values)
    if n < 2:
        return 0.0

    # Component 1: fraction of folds with positive IC
    hit_rate = float(np.mean(values > 0))

    # Component 2: IC t-stat proxy — scaled to [0, 1]
    mean_ic = float(np.mean(values))
    std_ic  = float(np.std(values, ddof=1))
    if std_ic > 0:
        t_stat = mean_ic / (std_ic / math.sqrt(n))
        # Map t-stat to [0,1]: t=0 → 0.5, t=2 → ~0.97 (sigmoid-like)
        t_component = 1.0 / (1.0 + math.exp(-t_stat / 2.0))
    else:
        t_component = 1.0 if mean_ic > 0 else 0.0

    # Combine: geometric mean of the two components
    stability = math.sqrt(hit_rate * t_component)
    return float(np.clip(stability, 0.0, 1.0))


# ── Deflated Sharpe Ratio & Probability of Backtest Overfitting ────────────────

def deflated_sharpe_ratio(returns, n_trials: int | None = 1, sr_benchmark=None) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the probability in [0, 1]
    that a strategy's TRUE Sharpe exceeds a benchmark, after deflating the observed
    Sharpe for (a) sample length, (b) non-normality (skew/kurtosis), and (c) the number
    of independent trials tried (multiple-testing / selection bias).

        DSR = Phi[ (SR_hat - SR*) / sigma_SR ],   with
        sigma_SR^2 = (1 - g3*SR + (g4-1)/4 * SR^2) / (T - 1)
        SR*        = sigma_SR * [ (1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N e)) ]

    g3 = skew, g4 = non-excess kurtosis, gamma = Euler-Mascheroni, Phi = N(0,1) cdf.
    DSR >= ~0.95 means the Sharpe is significant after deflation. Returns 0.0 on
    degenerate input (too few / zero-variance observations) — fail-closed.

    NOTE: ``n_trials`` should be the number of strategy CONFIGURATIONS tried during the
    research (not folds). Passing a smaller count is a conservative lower bound.

    ``n_trials=None`` reads the programme's cumulative count from
    `research.trial_ledger`, which is the single source of truth. New work should use it
    rather than restating a number. The default of 1 is UNCHANGED, because it is baked
    into every banked result; a default that moved with the ledger would silently
    re-deflate all of them.
    """
    if n_trials is None:
        from research.trial_ledger import cumulative_trials
        n_trials = cumulative_trials()
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = int(r.size)
    if T < 4:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return 0.0
    sr = float(np.mean(r) / sd)                          # non-annualised Sharpe
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))               # non-excess kurtosis (normal = 3)
    sr_var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / (T - 1)
    if sr_var <= 0.0:
        return 0.0
    sigma = math.sqrt(sr_var)
    if sr_benchmark is None:
        n = max(int(n_trials), 1)
        if n > 1:
            gamma = 0.5772156649015329                  # Euler-Mascheroni
            z1 = float(norm.ppf(1.0 - 1.0 / n))
            z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
            sr_benchmark = sigma * ((1.0 - gamma) * z1 + gamma * z2)
        else:
            sr_benchmark = 0.0
    return float(norm.cdf((sr - float(sr_benchmark)) / sigma))


def probability_of_backtest_overfitting(performance, n_splits: int = 16) -> float:
    """
    Probability of Backtest Overfitting (PBO) via Combinatorial Symmetric Cross-
    Validation (Bailey, Borwein, Lopez de Prado, Zhu, 2017).

    ``performance`` is a (T x N) matrix of per-period performance (e.g. returns) for N
    candidate configurations over T periods. PBO estimates the probability that the
    config which looks BEST in-sample lands in the WORSE half out-of-sample — i.e. how
    overfit the *selection* is. PBO near 0.5+ ⇒ the research process is overfitting.

    CSCV: split T into S equal blocks; for every choice of S/2 blocks as in-sample,
    rank configs by IS performance, take the best, find its OOS rank, form the logit
    of its relative OOS rank; PBO = fraction of combinations whose logit <= 0.
    """
    from itertools import combinations

    M = np.asarray(performance, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < 4:
        return 0.0
    T, N = M.shape
    S = max(2, int(n_splits) - (int(n_splits) % 2))     # even, >= 2
    while S > 2 and T // S < 1:
        S -= 2
    blocks = [M[idx, :] for idx in np.array_split(np.arange(T), S)]
    half = S // 2
    logits: list[float] = []
    for is_sel in combinations(range(S), half):
        in_set = set(is_sel)
        is_perf = np.concatenate([blocks[b] for b in range(S) if b in in_set]).sum(axis=0)
        oos_perf = np.concatenate([blocks[b] for b in range(S) if b not in in_set]).sum(axis=0)
        best = int(np.argmax(is_perf))
        oos_rank = int(np.sum(oos_perf <= oos_perf[best]))   # 1 = worst, N = best OOS
        w = min(max(oos_rank / (N + 1.0), 1e-6), 1.0 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))
    if not logits:
        return 0.0
    return float(np.mean(np.asarray(logits) <= 0.0))


# ── selection_rule ────────────────────────────────────────────────────────────

#: GATE-3 bar. ``probability_of_backtest_overfitting`` documents its own failure point:
#: "PBO near 0.5+ => the research process is overfitting". The bar is therefore the value
#: the estimator itself names, NOT a stricter number invented after seeing results — a
#: threshold chosen to make a particular candidate pass or fail is not a gate.
PBO_MAX: float = 0.50


def selection_rule(result: ValidationResult) -> bool:
    """
    Gate for signal/model promotion to live.

    Returns True only if ALL of the conditions below hold simultaneously.
    Any single failure blocks promotion — partial passes are not acceptable.

    Conditions (build spec Part 6.3, plus the gate-review fixes):
      1. mean_rank_ic > 0.01
      2. sharpe_net > 0.75
      3. stability_score > 0.60
      4. deflated_sharpe_proxy > 0.25
      5. deflated_sharpe_ratio >= 0.95    (the REAL DSR; default-DENY — GATE-2)
      6. pbo_proxy < PBO_MAX              (GATE-3, closed 2026-07-31)
      7. len(leakage_flags) == 0
      8. No single regime shows Sharpe < -0.50

    This docstring previously claimed "six conditions" and listed neither the real DSR nor
    PBO. It was wrong in both directions: the DSR condition existed but was undocumented,
    while PBO was described in the specs as a standing safeguard and was **never checked
    at all**. Two independent reviews (2026-07-27 and 2026-07-28) recorded that as GATE-3.
    """
    failures: list[str] = []

    if result.mean_rank_ic <= 0.01:
        failures.append(
            f"mean_rank_ic={result.mean_rank_ic:.4f} <= 0.01 (minimum edge threshold)"
        )

    if result.sharpe_net <= 0.75:
        failures.append(
            f"sharpe_net={result.sharpe_net:.4f} <= 0.75 (minimum Sharpe net of costs)"
        )

    if result.stability_score <= 0.60:
        failures.append(
            f"stability_score={result.stability_score:.4f} <= 0.60 (IC inconsistent)"
        )

    if result.deflated_sharpe_proxy <= 0.25:
        failures.append(
            f"deflated_sharpe_proxy={result.deflated_sharpe_proxy:.4f} <= 0.25 "
            "(high probability of backtest overfitting)"
        )

    if result.pbo_proxy >= PBO_MAX:
        failures.append(
            f"pbo_proxy={result.pbo_proxy:.4f} >= {PBO_MAX:.2f} "
            "(the SELECTION is overfit: the config that looks best in-sample lands in "
            "the worse half out-of-sample too often)"
        )

    if result.deflated_sharpe_ratio < 0.95:
        failures.append(
            f"deflated_sharpe_ratio={result.deflated_sharpe_ratio:.4f} < 0.95 "
            "(Sharpe not significant after multiple-testing / non-normality deflation)"
        )

    if result.leakage_flags:
        failures.append(
            f"leakage_flags not empty: {result.leakage_flags} "
            "(data leakage detected — cannot promote)"
        )

    for regime, metrics in result.regime_breakdown.items():
        regime_sharpe = metrics.get("sharpe", 0.0)
        if regime_sharpe < -0.50:
            failures.append(
                f"regime '{regime}' has Sharpe={regime_sharpe:.4f} < -0.50 "
                "(unacceptable regime-conditional performance)"
            )

    if failures:
        logger.info(
            "selection_rule FAILED (%d conditions): %s",
            len(failures), "; ".join(failures),
        )
        return False

    logger.info("selection_rule PASSED — signal meets all promotion criteria")
    return True
