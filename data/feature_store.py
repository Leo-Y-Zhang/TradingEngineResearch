"""
TradingEngineResearch — Feature Store
=========================
The single source of truth for feature retrieval.

All feature access goes through this module. Point-in-time safety is enforced
on every call: no feature value may have been computed using information that
would not have been available at asof_time.

Rules:
  - PIT joins strictly enforced (asof_timestamp <= asof_time for every row)
  - In LIVE mode, raises if any feature exceeds the staleness threshold
  - train/serve parity must be validated before model promotion
  - Missing values are handled deterministically (no random imputation)
  - Feature schema is versioned; hash changes are surfaced as warnings
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from data.data_contracts import FeatureRow, TradingMode

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_SCHEMA_VERSION = "v6.0"

# Maximum age (seconds) before a feature is considered stale in LIVE mode.
# Individual feature sources may override this via FEATURE_FRESHNESS_THRESHOLDS.
DEFAULT_STALE_THRESHOLD_SECONDS = 300.0   # 5 minutes

# A "daily" feature computed from the previous close must survive a weekend +
# bank-holiday gap before LIVE flags it stale (Friday close → Tuesday open).
_DAILY = 4 * 86400.0

FEATURE_FRESHNESS_THRESHOLDS: dict[str, float] = {
    # ── model schema (core/ml_return_model.FEATURE_NAMES) ──
    # Microstructure — must be very fresh.
    "ofi_signal":           30.0,
    "spread_bps":           30.0,
    # Intraday — 5-minute tolerance.
    "signal_score":         300.0,
    "engine_expected_return": 300.0,
    "volume_ratio":         300.0,
    # Sentiment / news — 60-minute tolerance.
    "sentiment_score":      3600.0,
    "news_age_minutes":     3600.0,
    # Regime — hourly.
    "regime_encoded":       3600.0,
    # Daily, from the prior close (weekend-tolerant).
    "insider_flow":         _DAILY,
    "insider_flow_age_days": _DAILY,
    "market_cap_log":       _DAILY,
    "momentum_12_1":        _DAILY,
    "reversal_5d":          _DAILY,
    "overnight_gap_mean":   _DAILY,
    "adv_ratio":            _DAILY,
    "idiosyncratic_vol":    _DAILY,
    "sector_relative_strength": _DAILY,
    "earnings_proximity_days":  _DAILY,
    # ── legacy/store-internal names (kept for back-compat) ──
    "ofi_norm":             30.0,
    "bid_ask_spread_bps":   30.0,
    "close_1d":             300.0,
    "volume_1d":            300.0,
    "returns_1d":           300.0,
    "returns_5d":           300.0,
    "returns_20d":          300.0,
    "vol_realised_20d":     300.0,
    "vol_ratio":            300.0,
    "news_sentiment_60m":   3600.0,
    "regime_state_encoded": 3600.0,
    "crisis_severity_score": 300.0,
}

# Missing-value imputation rules — deterministic, never random. Risk-relevant
# features impute CONSERVATIVELY: a missing input must never make a name look
# safer (zero idio-vol → oversized positions; zero spread → understated costs).
IMPUTATION_RULES: dict[str, float] = {
    # ── model schema ──
    "insider_flow":          0.0,
    "engine_expected_return": 0.0,
    "signal_score":          0.0,
    "volume_ratio":          1.0,     # neutral activity ratio
    "sentiment_score":       0.0,
    "regime_encoded":        1.0,     # default to TRENDING (regime 1)
    "market_cap_log":        23.0,    # ~ln(£1e10): neutral large-cap, not ln(£1)
    "momentum_12_1":         0.0,
    "reversal_5d":           0.0,
    "overnight_gap_mean":    0.0,
    "spread_bps":            10.0,    # conservative: assume a wide-ish spread
    "adv_ratio":             1.0,     # neutral liquidity trend
    "idiosyncratic_vol":     0.30,    # conservative: assume a risky name
    "sector_relative_strength": 0.0,
    "earnings_proximity_days": 10.0,  # cautious: event possibly near
    "ofi_signal":            0.0,
    "insider_flow_age_days": 30.0,    # assume 30-day old signal = cold
    "news_age_minutes":      60.0,    # assume 60-minute old news
    # ── legacy/store-internal names ──
    "ofi_norm":              0.0,
    "news_sentiment_60m":    0.0,
    "crisis_severity_score": 0.0,
    "regime_state_encoded":  1.0,
}
DEFAULT_IMPUTATION = 0.0              # fallback for any feature not in the table


def validate_schema_against_model(feature_names: list[str]) -> dict:
    """Cross-validate the store's feature metadata against a model's schema.

    Returns ``{"ok", "missing_freshness", "missing_imputation"}`` listing model
    features with no freshness threshold / no imputation rule. A miss means the
    feature silently falls back to the defaults — surfaced loudly here instead.
    """
    missing_fresh = [n for n in feature_names if n not in FEATURE_FRESHNESS_THRESHOLDS]
    missing_impute = [n for n in feature_names if n not in IMPUTATION_RULES]
    ok = not missing_fresh and not missing_impute
    if not ok:
        logger.warning(
            "Feature-schema mismatch vs model: missing freshness=%s, imputation=%s",
            missing_fresh, missing_impute,
        )
    return {"ok": ok, "missing_freshness": missing_fresh, "missing_imputation": missing_impute}


# ── Feature store (IN-MEMORY ONLY) ─────────────────────────────────────────────
# DATA-2 (honesty fix): this store is in-memory only. There is NO SQLAlchemy
# `feature_store` table — none exists in ops/sql_models.py. Features are therefore
# NOT durable across process restarts, so the point-in-time history of record is
# only as complete as what the running process has ingested. A durable
# `FeatureRecord` table is a Phase-3 reproducibility task; until then this
# limitation is documented explicitly rather than implied away.
# Keyed by (symbol, feature_schema_version).
_store: dict[tuple[str, str], list[FeatureRow]] = {}

# Per-schema-version union of feature names ever registered — used to surface
# schema drift (a row introducing names this version has never carried).
_schema_features: dict[str, set] = {}


def _register_features(row: FeatureRow) -> None:
    """Insert a FeatureRow into the in-memory store (test/dev use only).
    Surfaces schema drift: a row that introduces feature names its schema
    version has never carried changes the version's schema hash — warned loudly."""
    version = row.feature_schema_version
    seen = _schema_features.get(version)
    names = set(row.features)
    if seen is None:
        _schema_features[version] = set(names)
    elif not names <= seen:
        new = sorted(names - seen)
        old_hash = schema_hash(sorted(seen))
        seen.update(names)
        logger.warning(
            "Feature schema drift in %s: new feature(s) %s (schema_hash %s -> %s)",
            version, new, old_hash, schema_hash(sorted(seen)),
        )
    key = (row.symbol, version)
    _store.setdefault(key, []).append(row)


def _clear_store() -> None:
    """Clear the in-memory store (test/dev use only)."""
    _store.clear()
    _schema_features.clear()


# ── Public API ────────────────────────────────────────────────────────────────

def get_features(
    symbols: list[str],
    asof_time: datetime,
    mode: TradingMode,
    stale_threshold_seconds: float = DEFAULT_STALE_THRESHOLD_SECONDS,
    schema_version: Optional[str] = None,
    required_features: Optional[list[str]] = None,
    enforce_per_feature: bool = True,
) -> pd.DataFrame:
    """
    Point-in-time-safe feature retrieval.

    Returns a DataFrame with one row per symbol, columns = feature names.
    PIT safety: only rows with asof_timestamp <= asof_time are eligible.
    Most-recent eligible row is used for each symbol.

    In LIVE mode, raises ValueError if any feature exceeds stale_threshold_seconds.

    Parameters
    ----------
    symbols              : list of symbols to retrieve
    asof_time            : the decision time — no future data may leak through
    mode                 : RESEARCH / PAPER / LIVE
    stale_threshold_seconds : maximum allowed feature age in LIVE mode
    required_features    : the consumer's full feature schema (e.g. the model's
        ``FEATURE_NAMES``). Any of these absent from the retrieved data is added as
        a column so the conservative imputation below applies (DATA-1). Without this,
        a feature missing from EVERY row never becomes a column, never gets imputed,
        and the downstream model silently fills it with 0.0 — which for risk inputs
        is dangerously optimistic (0 idio-vol → oversized positions; 0 spread →
        understated costs). Only added for symbols that already have a row; symbols
        with no data at all are left absent (the consumer's own fallback applies).
    """
    if asof_time.tzinfo is None:
        # A naive boundary is ambiguous (which market's midnight?). In LIVE that
        # ambiguity could leak future data — fail closed. Off-LIVE, coerce to UTC.
        if mode == "LIVE":
            raise ValueError(
                "asof_time must be timezone-aware in LIVE mode (naive boundary is ambiguous)"
            )
        asof_time = asof_time.replace(tzinfo=timezone.utc)

    version = schema_version or FEATURE_SCHEMA_VERSION
    rows: list[dict] = []
    missing_symbols: list[str] = []

    for symbol in symbols:
        best_row: Optional[FeatureRow] = None
        best_ts: Optional[datetime] = None

        # PIT scan: find the most recent row where asof_timestamp <= asof_time
        for key in [(symbol, version)]:
            candidates = _store.get(key, [])
            for candidate in candidates:
                cts = candidate.asof_timestamp
                if cts is None:
                    continue                     # PIT-unsafe row (no asof) — never eligible
                if cts.tzinfo is None:
                    cts = cts.replace(tzinfo=timezone.utc)
                if cts <= asof_time:
                    if best_ts is None or cts > best_ts:
                        best_row = candidate
                        best_ts = cts

        if best_row is None:
            # Mode-aware missing-symbol handling: trading LIVE on a name with no
            # eligible features means trading blind — fail closed. RESEARCH/PAPER
            # degrade (the symbol is skipped) with a loud warning.
            if mode == "LIVE":
                raise ValueError(
                    f"No features for {symbol} at asof_time={asof_time} in LIVE mode "
                    "(fail-closed: refusing to trade a symbol with no feature row)"
                )
            missing_symbols.append(symbol)
            logger.warning(
                "No features found for %s at asof_time=%s (schema=%s)",
                symbol, asof_time, version,
            )
            continue

        # Row-level staleness check (overall threshold)
        age_seconds = (asof_time - best_ts).total_seconds()  # type: ignore[operator]
        if mode == "LIVE" and age_seconds > stale_threshold_seconds:
            raise ValueError(
                f"Features for {symbol} are stale in LIVE mode: "
                f"age={age_seconds:.0f}s exceeds threshold={stale_threshold_seconds:.0f}s. "
                "Halt trading or reduce exposure until features are refreshed."
            )

        # Per-feature staleness check using per-feature thresholds.
        # e.g. microstructure features have a 30s threshold; violating it in
        # LIVE mode is a hard error regardless of the row-level threshold.
        # Skipped when enforce_per_feature is False (a daily-data run, where the intraday
        # per-feature thresholds are inapplicable — only the row-level threshold applies).
        if mode == "LIVE" and enforce_per_feature:
            per_feature_violations: list[str] = []
            for feature_name in best_row.features:
                threshold = FEATURE_FRESHNESS_THRESHOLDS.get(
                    feature_name, stale_threshold_seconds
                )
                if age_seconds > threshold:
                    per_feature_violations.append(
                        f"{feature_name} (age={age_seconds:.0f}s > {threshold:.0f}s)"
                    )
            if per_feature_violations:
                raise ValueError(
                    f"Per-feature staleness violation in LIVE mode for {symbol}: "
                    f"{per_feature_violations}. Refresh the feature store."
                )

        # Enforce FeatureRow data contract (stale_flag check per contract)
        best_row.validate_for_mode(mode)

        row_dict: dict[str, object] = {"symbol": symbol, "asof_timestamp": best_ts}
        row_dict.update(best_row.features)
        rows.append(row_dict)

    if not rows:
        logger.warning("get_features returned no rows for symbols=%s", symbols)
        return pd.DataFrame(columns=["symbol", "asof_timestamp"])

    df = pd.DataFrame(rows)
    df = df.set_index("symbol")

    # DATA-1: ensure the consumer's full feature schema is present BEFORE imputing.
    # A feature absent from every retrieved row is otherwise never a column, so the
    # conservative impute loop below never touches it and the model defaults it to 0.0.
    # Adding it as an (all-NaN) column routes it through the conservative rules instead.
    if required_features:
        absent = [f for f in required_features if f not in df.columns]
        if absent:
            df = df.reindex(columns=list(df.columns) + absent)
            logger.warning("get_features: %d model feature(s) absent from all rows — "
                           "imputing conservatively (not 0.0): %s", len(absent), absent)

    # Deterministic missing-value imputation
    for col in df.columns:
        if col == "asof_timestamp":
            continue
        if df[col].isna().any():
            impute_val = IMPUTATION_RULES.get(col, DEFAULT_IMPUTATION)
            df[col] = df[col].fillna(impute_val)
            logger.debug("Imputed missing %s with %s", col, impute_val)

    return df


def validate_train_serve_parity(
    train_df: pd.DataFrame,
    serve_df: pd.DataFrame,
) -> dict:
    """
    Compare training and serving feature sets for parity.

    Returns a dict with:
      mismatched_columns   : columns present in one but not the other
      dtype_mismatches     : columns with different dtypes between train and serve
      range_violations     : columns where serve values fall outside the training range
      distribution_shift   : columns whose serve distribution shifted vs train
                             (PSI > 0.25 — catches in-range shifts min/max misses)
    """
    train_cols = set(train_df.columns) - {"asof_timestamp"}
    serve_cols = set(serve_df.columns) - {"asof_timestamp"}

    only_in_train = train_cols - serve_cols
    only_in_serve = serve_cols - train_cols
    mismatched_columns = {
        "only_in_train": sorted(only_in_train),
        "only_in_serve": sorted(only_in_serve),
    }

    # Dtype mismatches on common columns
    common = train_cols & serve_cols
    dtype_mismatches: dict[str, dict] = {}
    for col in common:
        t_dtype = str(train_df[col].dtype)
        s_dtype = str(serve_df[col].dtype)
        if t_dtype != s_dtype:
            dtype_mismatches[col] = {"train": t_dtype, "serve": s_dtype}

    # Range violations: serve values outside [train_min, train_max]
    range_violations: dict[str, dict] = {}
    for col in common:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        t_min = float(train_df[col].min())
        t_max = float(train_df[col].max())
        s_min = float(serve_df[col].min())
        s_max = float(serve_df[col].max())

        if s_min < t_min or s_max > t_max:
            range_violations[col] = {
                "train_range": (t_min, t_max),
                "serve_range": (s_min, s_max),
                "below_train_min": s_min < t_min,
                "above_train_max": s_max > t_max,
            }

    # Distributional shift (PSI): a serve distribution can sit entirely inside
    # the training range and still be unrecognisable to the model (e.g. all mass
    # in the top decile). PSI > 0.25 is the conventional "major shift" line.
    distribution_shift: dict[str, dict] = {}
    for col in common:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        psi = _population_stability_index(train_df[col], serve_df[col])
        if psi is not None and psi > 0.25:
            distribution_shift[col] = {"psi": round(psi, 4), "severity": "major"}

    result = {
        "mismatched_columns": mismatched_columns,
        "dtype_mismatches": dtype_mismatches,
        "range_violations": range_violations,
        "distribution_shift": distribution_shift,
        "is_valid": (
            not only_in_train
            and not only_in_serve
            and not dtype_mismatches
            and not range_violations
            and not distribution_shift
        ),
    }

    if not result["is_valid"]:
        logger.warning(
            "Train/serve parity check failed: %d column mismatches, "
            "%d dtype mismatches, %d range violations",
            len(only_in_train) + len(only_in_serve),
            len(dtype_mismatches),
            len(range_violations),
        )

    return result


def feature_freshness_report(
    symbols: list[str],
    asof_time: datetime,
) -> dict:
    """
    Return per-feature freshness age and stale flags for each symbol.

    Returns dict keyed by symbol, each value is a dict:
      { feature_name: { age_seconds: float, stale: bool } }
    """
    if asof_time.tzinfo is None:
        asof_time = asof_time.replace(tzinfo=timezone.utc)

    report: dict[str, dict] = {}

    for symbol in symbols:
        symbol_report: dict[str, dict] = {}

        best_row: Optional[FeatureRow] = None
        best_ts: Optional[datetime] = None
        for key in [(symbol, FEATURE_SCHEMA_VERSION)]:
            for candidate in _store.get(key, []):
                cts = candidate.asof_timestamp
                if cts is None:
                    continue
                if cts.tzinfo is None:
                    cts = cts.replace(tzinfo=timezone.utc)
                if cts <= asof_time:
                    if best_ts is None or cts > best_ts:
                        best_row = candidate
                        best_ts = cts

        if best_row is None:
            report[symbol] = {}
            continue

        age_seconds = (asof_time - best_ts).total_seconds()  # type: ignore[operator]

        for feature_name in best_row.features:
            threshold = FEATURE_FRESHNESS_THRESHOLDS.get(
                feature_name, DEFAULT_STALE_THRESHOLD_SECONDS
            )
            is_stale = age_seconds > threshold or best_row.freshness_flags.get(feature_name, False)
            symbol_report[feature_name] = {
                "age_seconds": round(age_seconds, 1),
                "threshold_seconds": threshold,
                "stale": is_stale,
            }

        report[symbol] = symbol_report

    return report


def _population_stability_index(
    train: pd.Series, serve: pd.Series, n_bins: int = 10
) -> Optional[float]:
    """PSI of ``serve`` vs ``train`` over train-quantile bins (None if degenerate)."""
    t = pd.to_numeric(train, errors="coerce").dropna()
    s = pd.to_numeric(serve, errors="coerce").dropna()
    if len(t) < n_bins or len(s) < n_bins:
        return None
    edges = t.quantile([i / n_bins for i in range(n_bins + 1)]).to_numpy()
    edges[0], edges[-1] = -float("inf"), float("inf")
    edges = pd.unique(edges)                       # constant features collapse bins
    if len(edges) < 3:
        return None
    eps = 1e-6
    t_frac = pd.cut(t, edges).value_counts(normalize=True, sort=False).to_numpy() + eps
    s_frac = pd.cut(s, edges).value_counts(normalize=True, sort=False).to_numpy() + eps
    return float(np.sum((s_frac - t_frac) * np.log(s_frac / t_frac)))


def schema_hash(feature_names: list[str]) -> str:
    """
    Deterministic hash of a feature schema for versioning.

    The hash changes whenever the set or order of feature names changes.
    Use this to detect train/serve schema drift.
    """
    canonical = ",".join(sorted(feature_names))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
