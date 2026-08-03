"""
TradingEngineResearch — ML Return Model
===========================
The expected-return ensemble. It maps an 18-feature vector to the canonical
5-tuple ``(expected_return, risk_estimate, p_positive, p_tail_loss, confidence)``
and governs its own trustworthiness through adaptive shrinkage, isotonic
calibration, rolling-IC tracking, and drift/refit governance.

Three complementary learners vote on the return: an ElasticNet baseline, a
tree ensemble (ExtraTrees), and a boosted ensemble (LightGBM if available, else
gradient boosting). Their mean is the raw view; their dispersion feeds the
shrinkage and confidence. A separate model predicts realised volatility, and the
hit / tail-loss probabilities are derived from the predictive distribution and
then isotonically calibrated.

Safety: ``predict()`` *always* returns the safe fallback
``(0.0, 0.15, 0.50, 0.10, 0.0)`` when the model is unfitted, stale, or raises —
it never propagates an exception into the trading pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "SAFE_FALLBACK",
    "MLReturnModel",
    "get_model",
    "reset_model",
]

# The 18 features, in exact order (master prompt Part 12.2).
FEATURE_NAMES: list[str] = [
    "insider_flow",
    "engine_expected_return",
    "signal_score",
    "volume_ratio",
    "sentiment_score",
    "regime_encoded",
    "market_cap_log",
    "momentum_12_1",
    "reversal_5d",
    "overnight_gap_mean",
    "spread_bps",
    "adv_ratio",
    "idiosyncratic_vol",
    "sector_relative_strength",
    "earnings_proximity_days",
    "ofi_signal",
    "insider_flow_age_days",
    "news_age_minutes",
]

FEATURE_SCHEMA_VERSION = "v6.0"

# (expected_return, risk_estimate, p_positive, p_tail_loss, confidence)
SAFE_FALLBACK: tuple[float, float, float, float, float] = (0.0, 0.15, 0.50, 0.10, 0.0)

_MAX_LOG = 60                # rolling prediction-log length
_IC_WINDOW = 20             # observations used for the rolling IC
_MIN_FIT_SAMPLES = 20
_DEFAULT_TAIL_THRESHOLD = -0.05    # a "tail loss" is a return below −5%
_DEFAULT_MAX_AGE_DAYS = 30.0
_DISAGREEMENT_SCALE = 0.02         # ensemble std (return units) that maps to score 1.0
_MIN_CAL_SAMPLES = 10              # min held-out rows for an honest OOS calibration fold
_CAL_FRACTION = 0.30               # fraction of (time-ordered) rows used as the calibration fold
_CAL_EMBARGO = 1                   # rows purged between the train and calibration folds
_TAIL_CAL_MIN_SAMPLES = 20         # min live outcomes before a live tail rate is reported
_TAIL_MISCAL_THRESHOLD = 0.10      # |live − base| tail-rate gap that triggers a refit
_TRAINING_BUFFER_MAX = 2000        # rolling (features, outcome) training window
_INITIAL_FIT_MIN = 40              # resolved outcomes before the first from-scratch fit


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _make_booster(random_state: int):
    """LightGBM if installed, else sklearn GradientBoosting (spec Part 12.2)."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, random_state=random_state, verbose=-1)
    except Exception:  # noqa: BLE001 — optional dependency; gradient boosting is the fallback
        return GradientBoostingRegressor(random_state=random_state)


class MLReturnModel:
    """An adaptively-shrunk, calibrated return ensemble with refit governance."""

    def __init__(
        self,
        max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
        tail_threshold: float = _DEFAULT_TAIL_THRESHOLD,
        random_state: int = 42,
        training_buffer_max: int = _TRAINING_BUFFER_MAX,
    ) -> None:
        self.max_age_days = max_age_days
        self.tail_threshold = tail_threshold
        self.random_state = random_state
        self.training_buffer_max = int(training_buffer_max)
        self._training_buffer: list[tuple[np.ndarray, float, float]] = []

        self._return_models: list = []
        self._vol_model: Any = None
        self._hit_calibrator: Optional[IsotonicRegression] = None
        self._tail_calibrator: Optional[IsotonicRegression] = None

        self._fitted = False
        self._stale = False
        self._prediction_log: list[tuple[float, float]] = []
        self._rolling_ic = 0.0
        self._prev_rolling_ic = 0.0

        self._brier_score: Optional[float] = None
        self._calibration_error: Optional[float] = None
        self._prev_calibration_error: Optional[float] = None
        self._tail_outcome_log: list[float] = []          # realized tail events (1.0/0.0)
        self._base_tail_rate: Optional[float] = None       # training tail frequency

        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._drift_flag = False

        self.last_regime: Optional[str] = None
        self._current_regime: Optional[str] = None
        self.active_model_version = "untrained"
        self.shadow_model_version: Optional[str] = None
        self._fit_time: Optional[float] = None
        self._cross_sectional_prior = 0.0

        self._refit_lock = threading.Lock()

    # ── vectorisation ───────────────────────────────────────────────────────

    def _vectorize(self, features: dict) -> np.ndarray:
        return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)

    def _ensemble_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(x) for m in self._return_models])

    # ── fitting ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X,
        y_returns,
        y_vols,
        y_hit=None,
        regime: Optional[str] = None,
    ) -> None:
        """Fit the return ensemble, the vol model, and the probability calibrators."""
        with self._refit_lock:
            X = np.asarray(X, dtype=float)
            y_returns = np.asarray(y_returns, dtype=float).ravel()
            y_vols = np.asarray(y_vols, dtype=float).ravel()

            if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
                raise ValueError(
                    f"X must be (n, {len(FEATURE_NAMES)}); got {X.shape}."
                )
            if X.shape[0] < _MIN_FIT_SAMPLES:
                raise ValueError(
                    f"need >= {_MIN_FIT_SAMPLES} samples to fit, got {X.shape[0]}."
                )

            self._return_models = [
                ElasticNet(alpha=1e-3, random_state=self.random_state, max_iter=5000),
                ExtraTreesRegressor(n_estimators=200, random_state=self.random_state),
                _make_booster(self.random_state),
            ]
            for model in self._return_models:
                model.fit(X, y_returns)

            self._vol_model = ExtraTreesRegressor(
                n_estimators=200, random_state=self.random_state
            )
            self._vol_model.fit(X, np.abs(y_vols))

            # Out-of-sample calibration (Item 4): fit the probability calibrators on
            # a held-out, time-ordered tail fold (purged from the train fold by an
            # embargo) and score Brier / ECE on that fold — never in-sample, which
            # the tree ensemble memorises (yielding absurdly optimistic metrics:
            # ~0.004 Brier on pure noise). The point return/vol models above stay
            # full-sample; only calibration honesty changes. Spec Part 12:
            # "calibrate p_positive and p_tail_loss via isotonic regression ON VALID
            # FOLDS." When there are too few rows for an honest split we still fit
            # the calibrators (so predict() works) but report None, not a fabricated
            # optimistic number.
            n_obs = X.shape[0]
            y_hit_full = (
                (y_returns > 0).astype(float)
                if y_hit is None
                else np.asarray(y_hit, dtype=float).ravel()
            )
            n_cal = max(_MIN_CAL_SAMPLES, int(round(_CAL_FRACTION * n_obs)))
            train_end = n_obs - n_cal - _CAL_EMBARGO
            have_oos = train_end >= _MIN_FIT_SAMPLES and n_cal >= _MIN_CAL_SAMPLES

            if have_oos:
                cal_models = [
                    ElasticNet(alpha=1e-3, random_state=self.random_state, max_iter=5000),
                    ExtraTreesRegressor(n_estimators=200, random_state=self.random_state),
                    _make_booster(self.random_state),
                ]
                for cm in cal_models:
                    cm.fit(X[:train_end], y_returns[:train_end])
                cal_vol = ExtraTreesRegressor(
                    n_estimators=200, random_state=self.random_state
                ).fit(X[:train_end], np.abs(y_vols[:train_end]))
                x_cal = X[n_obs - n_cal:]
                mu_hat = np.column_stack([cm.predict(x_cal) for cm in cal_models]).mean(axis=1)
                sigma_hat = np.clip(cal_vol.predict(x_cal), 1e-6, None)
                hit_cal = y_hit_full[n_obs - n_cal:]
                tail_cal = (y_returns[n_obs - n_cal:] < self.tail_threshold).astype(float)
            else:
                mu_hat = self._ensemble_predictions(X).mean(axis=1)
                sigma_hat = np.clip(self._vol_model.predict(X), 1e-6, None)
                hit_cal = y_hit_full
                tail_cal = (y_returns < self.tail_threshold).astype(float)

            p_pos_raw = norm.cdf(mu_hat / sigma_hat)
            self._hit_calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_pos_raw, hit_cal)
            p_tail_raw = norm.cdf((self.tail_threshold - mu_hat) / sigma_hat)
            self._tail_calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_tail_raw, tail_cal)

            self._prev_calibration_error = self._calibration_error
            if have_oos:
                p_pos_cal = self._hit_calibrator.predict(p_pos_raw)
                self._brier_score = float(np.mean((p_pos_cal - hit_cal) ** 2))
                self._calibration_error = _expected_calibration_error(p_pos_cal, hit_cal)
            else:
                self._brier_score = None
                self._calibration_error = None

            self._base_tail_rate = float(np.mean(y_returns < self.tail_threshold))
            self._feature_means = X.mean(axis=0)
            self._feature_stds = X.std(axis=0) + 1e-9
            self._drift_flag = False

            self._fitted = True
            self._stale = False
            self.last_regime = regime
            self._fit_time = time.time()
            self.active_model_version = self.register_model_version()
            logger.info(
                "MLReturnModel fitted on %d samples (version=%s, brier=%s)",
                X.shape[0], self.active_model_version,
                f"{self._brier_score:.4f}" if self._brier_score is not None else "n/a (OOS fold too small)",
            )

    # ── prediction ──────────────────────────────────────────────────────────

    def predict(
        self, features: dict, current_regime: Optional[str] = None
    ) -> tuple[float, float, float, float, float]:
        """Return ``(expected_return, risk_estimate, p_positive, p_tail_loss, confidence)``."""
        if current_regime is not None:
            self._current_regime = current_regime
        if not self._fitted or self._stale:
            return SAFE_FALLBACK
        try:
            x = self._vectorize(features).reshape(1, -1)
            preds = self._ensemble_predictions(x)[0]
            mu_raw = float(np.mean(preds))
            disagreement = float(np.std(preds))
            sigma = float(np.clip(self._vol_model.predict(x)[0], 1e-4, None))

            ic = self._rolling_ic
            calibration_score = self._calibration_score()
            disagreement_score = _clip01(disagreement / _DISAGREEMENT_SCALE)

            shrinkage = float(
                np.clip(
                    0.35 + 1.5 * ic + 0.20 * calibration_score - 0.15 * disagreement_score,
                    0.20, 0.80,
                )
            )
            mu = shrinkage * mu_raw + (1.0 - shrinkage) * self._cross_sectional_prior

            p_positive = self._calibrate(self._hit_calibrator, float(norm.cdf(mu / sigma)))
            p_tail_loss = self._calibrate(
                self._tail_calibrator, float(norm.cdf((self.tail_threshold - mu) / sigma))
            )

            confidence = _clip01(
                0.5 * max(ic, 0.0)
                + 0.3 * calibration_score
                + 0.2 * (1.0 - disagreement_score)
            )
            return (mu, sigma, p_positive, p_tail_loss, confidence)
        except Exception as exc:  # noqa: BLE001 — never propagate into the pipeline
            logger.warning("MLReturnModel.predict failed (%s); using safe fallback.", exc)
            return SAFE_FALLBACK

    def set_cross_sectional_prior(self, prior: float) -> None:
        """Set the shrinkage anchor (Item 5). Non-finite values fall back to 0.0."""
        self._cross_sectional_prior = float(prior) if np.isfinite(prior) else 0.0

    def predict_batch(
        self, feature_dicts: list[dict], current_regime: Optional[str] = None
    ) -> list[tuple[float, float, float, float, float]]:
        # Item 5: anchor the adaptive-shrinkage prior at the CROSS-SECTIONAL MEAN of
        # this cycle's RAW ensemble expected returns (James-Stein style). The convex
        # blend then shrinks each name toward the universe's average view rather than
        # toward a hard-coded 0.0 (which silently attenuated every signal by up to
        # 80%). A single-symbol universe ends up un-shrunk (prior == its own mu_raw).
        if self._fitted and not self._stale and feature_dicts:
            try:
                x_all = np.vstack([self._vectorize(f) for f in feature_dicts])
                mu_raw_all = self._ensemble_predictions(x_all).mean(axis=1)
                self.set_cross_sectional_prior(float(np.mean(mu_raw_all)))
            except Exception as exc:  # noqa: BLE001 — never break the cycle on the prior
                logger.warning("cross-sectional prior failed (%s); using 0.0.", exc)
                self.set_cross_sectional_prior(0.0)
        return [self.predict(f, current_regime) for f in feature_dicts]

    @staticmethod
    def _calibrate(calibrator: Optional[IsotonicRegression], raw: float) -> float:
        if calibrator is None:
            return _clip01(raw)
        return _clip01(float(calibrator.predict([raw])[0]))

    # ── outcome tracking / rolling IC ────────────────────────────────────────

    def record_outcome(
        self, predicted: float, actual: float, tail_event: Optional[bool] = None
    ) -> None:
        actual = float(actual)
        self._prediction_log.append((float(predicted), actual))
        if len(self._prediction_log) > _MAX_LOG:
            self._prediction_log = self._prediction_log[-_MAX_LOG:]
        # Tail calibration from live outcomes: honor an explicit tail_event, else derive
        # it from the realized return vs the model's tail threshold.
        tail = bool(tail_event) if tail_event is not None else (actual < self.tail_threshold)
        self._tail_outcome_log.append(1.0 if tail else 0.0)
        if len(self._tail_outcome_log) > _MAX_LOG:
            self._tail_outcome_log = self._tail_outcome_log[-_MAX_LOG:]
        self._update_rolling_ic()

    def live_tail_rate(self) -> Optional[float]:
        """Realized tail-loss frequency over recent live outcomes (None until enough)."""
        if len(self._tail_outcome_log) < _TAIL_CAL_MIN_SAMPLES:
            return None
        return float(np.mean(self._tail_outcome_log))

    def tail_calibration_error(self) -> Optional[float]:
        """``|live realized tail rate − training base tail rate|`` (None if unavailable)."""
        live = self.live_tail_rate()
        if live is None or self._base_tail_rate is None:
            return None
        return float(abs(live - self._base_tail_rate))

    def _update_rolling_ic(self) -> None:
        self._prev_rolling_ic = self._rolling_ic
        if len(self._prediction_log) < _IC_WINDOW:
            self._rolling_ic = 0.0
            return
        preds = [p for p, _ in self._prediction_log[-_IC_WINDOW:]]
        acts = [a for _, a in self._prediction_log[-_IC_WINDOW:]]
        if np.std(preds) > 0 and np.std(acts) > 0:
            ic, _ = spearmanr(preds, acts)
            self._rolling_ic = float(ic) if not np.isnan(ic) else 0.0
        else:
            self._rolling_ic = 0.0

    def _calibration_score(self) -> float:
        if self._calibration_error is None:
            return 0.0
        return _clip01(1.0 - self._calibration_error)

    # ── training buffer / real refit (ROADMAP Phase 4) ───────────────────────

    def record_training_example(
        self,
        features: dict,
        realized_return: float,
        realized_vol: Optional[float] = None,
    ) -> None:
        """Append one resolved (features → realized outcome) row to the rolling
        training window. ``realized_vol`` defaults to ``|realized_return|`` (the
        vol model trains on absolute outcomes)."""
        ret = float(realized_return)
        if not np.isfinite(ret):
            return
        vol = float(realized_vol) if realized_vol is not None else abs(ret)
        self._training_buffer.append((self._vectorize(features), ret, vol))
        if len(self._training_buffer) > self.training_buffer_max:
            self._training_buffer = self._training_buffer[-self.training_buffer_max:]

    @property
    def training_buffer_size(self) -> int:
        return len(self._training_buffer)

    @property
    def ready_for_initial_fit(self) -> bool:
        """True when the model has never been fitted but enough live outcomes have
        accumulated to bootstrap it (the engine then triggers the first fit)."""
        return not self._fitted and len(self._training_buffer) >= _INITIAL_FIT_MIN

    def refit(self) -> bool:
        """Retrain from the accumulated training buffer. Safe no-op (returns
        False) when there is too little data or the fit fails — a refit must
        never take down a cycle; the previous model keeps serving."""
        if len(self._training_buffer) < _MIN_FIT_SAMPLES:
            logger.info(
                "refit deferred: %d/%d training examples.",
                len(self._training_buffer), _MIN_FIT_SAMPLES,
            )
            return False
        X = np.vstack([x for x, _, _ in self._training_buffer])
        y_ret = np.array([r for _, r, _ in self._training_buffer], dtype=float)
        y_vol = np.array([v for _, _, v in self._training_buffer], dtype=float)
        try:
            self.fit(X, y_ret, y_vol, regime=self._current_regime or self.last_regime)
        except Exception as exc:  # noqa: BLE001 — keep serving the previous model
            logger.warning("refit failed (%s); keeping the previous model.", exc)
            return False
        return True

    # ── drift / refit governance ─────────────────────────────────────────────

    def mark_feature_drift(self, drifted: bool = True) -> None:
        """Externally raise/clear the feature-drift flag (set by the monitor)."""
        self._drift_flag = bool(drifted)

    def check_feature_drift(self, recent_X, z_threshold: float = 3.0) -> bool:
        """Flag drift when recent feature means deviate > ``z_threshold`` σ from baseline."""
        if not self._fitted or self._feature_means is None:
            return False
        recent = np.asarray(recent_X, dtype=float)
        if recent.ndim == 1:
            recent = recent.reshape(1, -1)
        if recent.shape[1] != len(FEATURE_NAMES):
            return False
        z = np.abs(recent.mean(axis=0) - self._feature_means) / self._feature_stds
        self._drift_flag = bool(np.any(z > z_threshold))
        return self._drift_flag

    def model_age_days(self) -> float:
        if self._fit_time is None:
            return 0.0
        return (time.time() - self._fit_time) / 86400.0

    @property
    def needs_refit(self) -> bool:
        """True when any governance trigger fires (see master prompt Part 12.2)."""
        if not self._fitted:
            return False
        if self._drift_flag:
            return True
        if self.model_age_days() > self.max_age_days:
            return True
        if (
            self._prev_calibration_error is not None
            and self._calibration_error is not None
            and self._calibration_error > self._prev_calibration_error * 1.20
        ):
            return True
        if (
            self.last_regime is not None
            and self._current_regime is not None
            and self._current_regime != self.last_regime
            and self._rolling_ic < self._prev_rolling_ic
        ):
            return True
        tail_err = self.tail_calibration_error()
        if tail_err is not None and tail_err > _TAIL_MISCAL_THRESHOLD:
            return True                          # live tails persistently diverge from base
        return False

    # ── reporting / versioning ───────────────────────────────────────────────

    def calibration_report(self) -> dict:
        return {
            "brier_score": self._brier_score,
            "calibration_error": self._calibration_error,
            "prev_calibration_error": self._prev_calibration_error,
            "n_outcomes": len(self._prediction_log),
            "is_calibrated": self._hit_calibrator is not None,
            "live_tail_rate": self.live_tail_rate(),
            "base_tail_rate": self._base_tail_rate,
            "tail_calibration_error": self.tail_calibration_error(),
        }

    def drift_report(self) -> dict:
        return {
            "drift_flag": self._drift_flag,
            "rolling_ic": self._rolling_ic,
            "prev_rolling_ic": self._prev_rolling_ic,
            "model_age_days": self.model_age_days(),
            "last_regime": self.last_regime,
            "current_regime": self._current_regime,
            "needs_refit": self.needs_refit,
        }

    def register_model_version(self) -> str:
        stamp = int(self._fit_time) if self._fit_time is not None else 0
        version = f"{FEATURE_SCHEMA_VERSION}-{stamp}"
        self.active_model_version = version
        return version


def _expected_calibration_error(probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
    """Binned expected calibration error in [0, 1]."""
    probs = np.asarray(probs, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = probs.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(probs[mask]))
        bin_acc = float(np.mean(actuals[mask]))
        ece += (np.sum(mask) / total) * abs(bin_conf - bin_acc)
    return float(ece)


# ── Module-level singleton ──────────────────────────────────────────────────────

_MODEL: Optional[MLReturnModel] = None


def get_model() -> MLReturnModel:
    """Return the process-wide MLReturnModel singleton (created on first use)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = MLReturnModel()
    return _MODEL


def reset_model() -> None:
    """Drop the singleton (tests / restarts)."""
    global _MODEL
    _MODEL = None
