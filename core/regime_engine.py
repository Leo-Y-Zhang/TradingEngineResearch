"""
TradingEngineResearch — Regime Engine
=========================
The single source of truth for market regime.

A 2-state Gaussian Hidden Markov Model (``hmmlearn``) is fit on a rolling 252-day
window of standardised observation features. State 0 is calm/trending and state 1
is stressed/high-vol — but because the HMM assigns latent-state indices
arbitrarily, fitted states are mapped to "calm"/"stressed" by their realised-vol
means (the higher-volatility state is always treated as "stressed"), so the
labelling is stable across refits.

Public surface
--------------
  - ``detect(prices)``             → "high_vol" | "trending" | "mean_reverting"
  - ``detect_with_probs(prices)``  → (label, {"calm": p, "stressed": p})
  - ``recommended_strategy_mix()`` → regime-adapted sleeve weights
  - ``regime_transition_penalty()``→ turnover cost multiplier across transitions
  - ``infer_execution_regime()``   → "normal_exec" | "cautious_exec" | "stressed_exec"
  - ``get_regime_engine()``        → process-wide singleton (caches the HMM)

If there is too little history for a stable HMM fit, or the fit fails, a
transparent volatility/momentum heuristic is used instead (logged as a WARNING),
so ``detect()`` always returns a valid regime string and never crashes the cycle.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "RegimeEngine",
    "detect",
    "detect_with_probs",
    "recommended_strategy_mix",
    "regime_transition_penalty",
    "infer_execution_regime",
    "get_regime_engine",
    "reset_regime_engine",
    "VALID_REGIMES",
    "SLEEVE_NAMES",
]

# ── Constants ──────────────────────────────────────────────────────────────────

VALID_REGIMES: frozenset[str] = frozenset({"high_vol", "trending", "mean_reverting"})

# Strategy sleeves the optimizer allocates across (Section 1 of the master prompt).
SLEEVE_NAMES: tuple[str, ...] = (
    "momentum", "mean_reversion", "stat_arb", "carry", "event", "vol_overlay",
)

_REGIME_FEATURE_COLS: tuple[str, ...] = (
    "log_vol_20d",
    "log_vol_60d_ratio",
    "momentum_sign",
    "cross_sectional_dispersion",
    "mean_pairwise_correlation",
)

# Posterior stressed-probability above which the regime is "high_vol".
_STRESSED_PROB_THRESHOLD: float = 0.65
# |20-day return| above which a non-stressed regime is labelled "trending".
_TREND_RETURN_THRESHOLD: float = 0.03

_EPS: float = 1e-8

# VREG-2 fail-safe. When volatility cannot be measured at all, the heuristic's
# arithmetic used to produce ratio = 1.0 -> p_stressed ~0.16 -> a confidently
# BENIGN regime. Benign is the aggressive answer here: it widens the position
# cap, raises the vol target, and hands views the largest tau. Assume stressed
# instead when we cannot see, so an unmeasurable market sizes DOWN rather than up.
# Sits above _STRESSED_PROB_THRESHOLD so the label resolves to "high_vol".
_UNKNOWN_STRESSED_PROB: float = 0.75


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


# ── Feature construction ────────────────────────────────────────────────────────

def _as_price_frame(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        return prices.to_frame(name=prices.name or "asset")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame or Series.")
    return prices


def _rolling_mean_pairwise_corr(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """Mean upper-triangular pairwise correlation over a trailing window."""
    n, ncols = returns.shape
    if ncols < 2:
        return pd.Series(np.zeros(n), index=returns.index)

    vals = returns.to_numpy()
    iu = np.triu_indices(ncols, k=1)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        block = vals[t - window + 1 : t + 1]
        block = block[~np.isnan(block).any(axis=1)]
        if block.shape[0] < 3:
            continue
        corr = np.corrcoef(block, rowvar=False)
        out[t] = float(np.nanmean(corr[iu]))
    return pd.Series(out, index=returns.index)


def _build_regime_features(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Build the standardisable observation features from a price panel.

    ``spread_proxy`` from the spec is omitted: it is not derivable from a price
    series and is only included "if available".
    """
    px = _as_price_frame(prices)
    returns = px.pct_change(fill_method=None)
    port_ret = returns.mean(axis=1)

    vol_20d = port_ret.rolling(20).std()
    vol_60d = port_ret.rolling(60).std()

    price_mean = px.mean(axis=1)
    ret_20d = price_mean.pct_change(20, fill_method=None)

    if px.shape[1] > 1:
        dispersion = returns.std(axis=1)
    else:
        dispersion = pd.Series(np.zeros(len(px)), index=px.index)

    features = pd.DataFrame(
        {
            "log_vol_20d": np.log(vol_20d + _EPS),
            "log_vol_60d_ratio": np.log((vol_20d + _EPS) / (vol_60d + _EPS)),
            "momentum_sign": np.sign(ret_20d),
            "cross_sectional_dispersion": dispersion,
            "mean_pairwise_correlation": _rolling_mean_pairwise_corr(returns, 20),
        },
        index=px.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


# ── Regime engine ───────────────────────────────────────────────────────────────

class RegimeEngine:
    """Stateful HMM regime detector with a refit-every-N-cycles cache."""

    def __init__(
        self,
        n_components: int = 2,
        window: int = 252,
        refit_every: int = 5,
        random_state: int = 42,
        min_fit_rows: int = 40,
    ) -> None:
        self.n_components = n_components
        self.window = window
        self.refit_every = max(int(refit_every), 1)
        self.random_state = random_state
        self.min_fit_rows = min_fit_rows

        self._model: Any = None
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._state_to_regime: dict[int, str] = {}
        self._cycle = 0

    # -- internal helpers --------------------------------------------------------

    def _standardize(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self._scaler_mean = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            std[std < 1e-9] = 1.0
            self._scaler_std = std
        assert self._scaler_mean is not None and self._scaler_std is not None
        return (x - self._scaler_mean) / self._scaler_std

    def _fit(self, window_df: pd.DataFrame) -> None:
        from hmmlearn.hmm import GaussianHMM

        x_std = self._standardize(window_df.to_numpy(dtype=float), fit=True)
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
            init_params="stmc",
        )
        model.fit(x_std)

        # Map latent states to calm/stressed by realised-vol mean (feature 0).
        vol_means = model.means_[:, 0]
        stressed_state = int(np.argmax(vol_means))
        self._state_to_regime = {
            s: ("stressed" if s == stressed_state else "calm")
            for s in range(self.n_components)
        }
        self._model = model

    def _maybe_fit(self, window_df: pd.DataFrame) -> None:
        if self._model is None or (self._cycle % self.refit_every == 0):
            self._fit(window_df)
        self._cycle += 1

    def _heuristic(self, prices: pd.DataFrame | pd.Series) -> tuple[str, dict[str, float]]:
        """Transparent vol/momentum fallback when the HMM is unavailable."""
        px = _as_price_frame(prices)
        port_ret = px.pct_change(fill_method=None).mean(axis=1).dropna()
        price_mean = px.mean(axis=1)

        # VREG-2: no measurable volatility -> assume stressed, never calm. Two
        # cases reach here: too few returns to compute a standard deviation at
        # all, and a 60-day vol of ~0, which on real prices means a flat or
        # stale feed rather than a genuinely riskless market. Both used to fall
        # through to ratio = 1.0 and label a benign regime with full confidence.
        vol_20 = float(port_ret.tail(20).std()) if len(port_ret) >= 2 else 0.0
        vol_60 = float(port_ret.tail(60).std()) if len(port_ret) >= 2 else 0.0
        if len(port_ret) < 2 or vol_60 <= _EPS:
            return "high_vol", {
                "calm": 1.0 - _UNKNOWN_STRESSED_PROB,
                "stressed": _UNKNOWN_STRESSED_PROB,
            }
        ratio = vol_20 / vol_60

        ret_20 = price_mean.pct_change(20, fill_method=None).iloc[-1] if len(price_mean) > 20 else 0.0
        ret_20 = 0.0 if not np.isfinite(ret_20) else float(ret_20)

        p_stressed = _clip01(1.0 / (1.0 + math.exp(-(ratio - 1.5) / 0.3)))
        probs = {"calm": 1.0 - p_stressed, "stressed": p_stressed}

        if p_stressed > _STRESSED_PROB_THRESHOLD:
            label = "high_vol"
        elif abs(ret_20) >= _TREND_RETURN_THRESHOLD:
            label = "trending"
        else:
            label = "mean_reverting"
        return label, probs

    @staticmethod
    def _label(probs: dict[str, float], trend_strong: bool) -> str:
        if probs["stressed"] > _STRESSED_PROB_THRESHOLD:
            return "high_vol"
        if trend_strong:
            return "trending"
        return "mean_reverting"

    # -- public API --------------------------------------------------------------

    def detect_with_probs(
        self, prices: pd.DataFrame | pd.Series
    ) -> tuple[str, dict[str, float]]:
        features = _build_regime_features(prices).dropna()
        if features.shape[0] < self.min_fit_rows:
            return self._heuristic(prices)

        window_df = features.iloc[-self.window :]
        try:
            self._maybe_fit(window_df)
            x_std = self._standardize(window_df.to_numpy(dtype=float), fit=False)
            posterior = self._model.predict_proba(x_std)[-1]
            p_stressed = float(
                sum(
                    posterior[s]
                    for s in range(self.n_components)
                    if self._state_to_regime.get(s) == "stressed"
                )
            )
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never crash
            logger.warning(
                "HMM regime inference failed (%s); using heuristic fallback.", exc
            )
            return self._heuristic(prices)

        p_stressed = _clip01(p_stressed)
        probs = {"calm": 1.0 - p_stressed, "stressed": p_stressed}

        price_mean = _as_price_frame(prices).mean(axis=1)
        ret_20 = price_mean.pct_change(20, fill_method=None).iloc[-1]
        trend_strong = bool(np.isfinite(ret_20) and abs(ret_20) >= _TREND_RETURN_THRESHOLD)

        return self._label(probs, trend_strong), probs

    def detect(self, prices: pd.DataFrame | pd.Series) -> str:
        return self.detect_with_probs(prices)[0]


# ── Module-level singleton ──────────────────────────────────────────────────────

_ENGINE: Optional[RegimeEngine] = None


def get_regime_engine() -> RegimeEngine:
    """Return the process-wide RegimeEngine singleton (created on first use)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RegimeEngine()
    return _ENGINE


def reset_regime_engine() -> None:
    """Drop the singleton so the next call refits from scratch (tests / restarts)."""
    global _ENGINE
    _ENGINE = None


def detect(prices: pd.DataFrame | pd.Series) -> str:
    """Backward-compatible string regime label for ``prices``."""
    return get_regime_engine().detect(prices)


def detect_with_probs(prices: pd.DataFrame | pd.Series) -> tuple[str, dict[str, float]]:
    """Regime label plus calm/stressed posterior probabilities (summing to 1.0)."""
    return get_regime_engine().detect_with_probs(prices)


# ── Pure regime utilities ───────────────────────────────────────────────────────

# Equal-ish base mixes; the live mix is a probability-weighted blend of the two.
_BASE_MIX_CALM: dict[str, float] = {
    "momentum": 0.35, "mean_reversion": 0.15, "stat_arb": 0.15,
    "carry": 0.15, "event": 0.10, "vol_overlay": 0.10,
}
_BASE_MIX_STRESSED: dict[str, float] = {
    "momentum": 0.10, "mean_reversion": 0.30, "stat_arb": 0.20,
    "carry": 0.05, "event": 0.05, "vol_overlay": 0.30,
}


def recommended_strategy_mix(
    regime_probs: dict[str, float],
    overlays: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Regime-adapted sleeve weights, normalised to sum to 1.0.

    ``regime_probs`` is the ``{"calm", "stressed"}`` posterior. The mix linearly
    interpolates between a calm base (momentum-heavy) and a stressed base
    (mean-reversion / vol-overlay-heavy). ``overlays`` optionally scales
    individual sleeves multiplicatively (e.g. a crisis overlay damping momentum)
    before renormalisation.
    """
    p_stressed = _clip01(float(regime_probs.get("stressed", 0.5)))
    p_calm = 1.0 - p_stressed

    mix = {
        sleeve: p_calm * _BASE_MIX_CALM[sleeve] + p_stressed * _BASE_MIX_STRESSED[sleeve]
        for sleeve in SLEEVE_NAMES
    }

    if overlays:
        for sleeve, scale in overlays.items():
            if sleeve in mix:
                mix[sleeve] = max(mix[sleeve] * float(scale), 0.0)

    total = sum(mix.values())
    if total <= 0.0:
        # Degenerate overlays zeroed everything — fall back to an equal weight.
        return {sleeve: 1.0 / len(SLEEVE_NAMES) for sleeve in SLEEVE_NAMES}
    return {sleeve: weight / total for sleeve, weight in mix.items()}


def regime_transition_penalty(prev_regime: str, new_regime: str) -> float:
    """
    Turnover cost multiplier for trading through a regime change.

    No change → 1.0. Any change is penalised; transitions into or out of the
    high-vol regime are penalised most heavily because turnover there is the most
    expensive and the most likely to be adverse.
    """
    if prev_regime == new_regime:
        return 1.0
    if prev_regime not in VALID_REGIMES or new_regime not in VALID_REGIMES:
        return 1.2
    if "high_vol" in (prev_regime, new_regime):
        return 1.5
    return 1.2


def infer_execution_regime(
    spread_bps: float,
    vol_ratio: float,
    adv_participation: float,
    minutes_to_close: float,
) -> str:
    """
    Execution-mode overlay used by the meta-labeller and execution engine.

      - ``stressed_exec``: spread > 25bps, or vol_ratio > 2.5, or < 15min to close
      - ``cautious_exec``: spread 10–25bps, or vol_ratio 1.5–2.5, or ≤ 60min to close
      - ``normal_exec``:   spread < 10bps, vol_ratio < 1.5, and > 60min to close

    ``adv_participation`` escalates conservatively (> 0.10 → stressed,
    > 0.05 → cautious) without overriding the documented spread/vol/time bands.
    """
    if (
        spread_bps > 25.0
        or vol_ratio > 2.5
        or minutes_to_close < 15.0
        or adv_participation > 0.10
    ):
        return "stressed_exec"
    if (
        spread_bps >= 10.0
        or vol_ratio >= 1.5
        or minutes_to_close <= 60.0
        or adv_participation > 0.05
    ):
        return "cautious_exec"
    return "normal_exec"
