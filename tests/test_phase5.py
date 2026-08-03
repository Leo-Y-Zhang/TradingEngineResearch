"""
Phase 5 Tests — ML Prediction, Calibration, Drift, Meta-Labelling, OFI, NLP
==========================================================================
Covers every Phase 5 test target from the build spec:

  - predict() returns the safe fallback when the model is not ready; a valid
    5-tuple once fitted
  - needs_refit fires under each documented trigger; _refit_lock serialises fit
  - calibration_report / drift_report expose the documented keys
  - TradeDecision admission: all four conditions independently, size clip,
    stressed_exec marginal block
  - compute_ofi L1/L2 clip to [-1, 1]; unavailable -> 0.0; ofi_filter_gate
    rejects the two cases
  - FinBERT temperature-scaling and raw-softmax both yield valid dicts; offline
    lexicon fallback never crashes; batch_score defers under CPU load
  - Sentiment aggregator returns 0.0 with no in-window headlines; time decay
    weights fresher news more
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from core import meta_labeler as ml
from core import ml_filter as mlf
from core import ml_return_model as mrm
from core.engine import microstructure as ms
from nlp import finbert_scorer as fb
from nlp import sentiment_aggregator as sa
from nlp import sentiment_pipeline as sp


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _ml_training_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 18))
    y_ret = 0.01 * X[:, 2] + 0.006 * X[:, 7] + rng.normal(0, 0.01, n)
    y_vol = 0.02 + 0.01 * np.abs(X[:, 12]) + 0.002 * rng.standard_normal(n)
    return X, y_ret, y_vol


def _feature_dict(row: np.ndarray) -> dict:
    return {name: float(v) for name, v in zip(mrm.FEATURE_NAMES, row)}


# ── 1. Microstructure / OFI ──────────────────────────────────────────────────────

class TestMicrostructure:

    def test_l2_buying_pressure_is_positive(self):
        data = {
            "bid_sizes": [1200, 800, 600, 400, 200],
            "prev_bid_sizes": [1000, 800, 600, 400, 200],
            "ask_sizes": [900, 700, 500, 300, 100],
            "prev_ask_sizes": [1000, 700, 500, 300, 100],
            "total_volume_5min": 5000,
        }
        ofi = ms.compute_ofi(data)
        assert 0.0 < ofi <= 1.0

    def test_l2_clips_to_unit_interval(self):
        data = {
            "delta_bid_sizes": [-500, 0, 0, 0, 0],
            "delta_ask_sizes": [500, 0, 0, 0, 0],
            "total_volume_5min": 100,
        }
        assert ms.compute_ofi(data) == -1.0

    def test_l1_fallback(self):
        data = {"bid_size": 1000, "ask_size": 1000, "bid_size_change": 200, "ask_size_change": -100}
        ofi = ms.compute_ofi(data)
        assert -1.0 <= ofi <= 1.0 and ofi > 0.0

    def test_unavailable_returns_zero(self):
        assert ms.compute_ofi({}) == 0.0
        assert ms.compute_ofi(None) == 0.0

    def test_l2_partial_nan_level_preserves_signal(self):
        # A NaN at one level must NOT zero the whole signal — the valid levels still
        # contribute (previously a single NaN propagated and the output collapsed to 0).
        data = {
            "delta_bid_sizes": [10.0, float("nan"), 5.0],
            "delta_ask_sizes": [2.0, 3.0, 1.0],
            "total_volume_5min": 100.0,
        }
        ofi = ms.compute_ofi(data)
        assert -1.0 <= ofi <= 1.0
        assert ofi > 0.0                                  # bid pressure from valid levels

    def test_l2_mismatched_level_counts(self):
        data = {
            "delta_bid_sizes": [10.0, 5.0, 3.0, 2.0, 1.0],     # 5 levels
            "delta_ask_sizes": [2.0, 1.0],                      # 2 levels
            "total_volume_5min": 100.0,
        }
        assert -1.0 <= ms.compute_ofi(data) <= 1.0             # pairs the common 2 levels

    def test_l2_all_nan_levels_neutral(self):
        data = {
            "delta_bid_sizes": [float("nan"), float("nan")],
            "delta_ask_sizes": [float("nan"), float("nan")],
            "total_volume_5min": 100.0,
        }
        assert ms.compute_ofi(data) == 0.0                     # no valid level -> neutral

    def test_l2_all_nan_falls_back_to_l1(self):
        # When no L2 level is usable, fall through to L1 instead of returning a
        # degenerate 0 from the L2 path.
        data = {
            "delta_bid_sizes": [float("nan")],
            "delta_ask_sizes": [float("nan")],
            "total_volume_5min": 100.0,
            "bid_size": 1000, "ask_size": 1000,
            "bid_size_change": 200, "ask_size_change": -100,
        }
        assert ms.compute_ofi(data) > 0.0                      # L1 buy pressure used

    def test_gate_rejects_buy_into_selling_and_sell_into_buying(self):
        assert ms.ofi_filter_gate("BUY", -0.40) is False
        assert ms.ofi_filter_gate("SELL", 0.40) is False

    def test_gate_passes_otherwise_and_at_boundary(self):
        assert ms.ofi_filter_gate("BUY", 0.0) is True
        assert ms.ofi_filter_gate("SELL", 0.0) is True
        assert ms.ofi_filter_gate("BUY", -0.30) is True     # strict threshold
        assert ms.ofi_filter_gate("SELL", 0.30) is True


# ── 2. Meta-labeller ─────────────────────────────────────────────────────────────

class TestMetaLabeler:

    def _good(self, **kw):
        params = dict(
            mu=0.004, sigma=0.02, p_positive=0.62, p_tail_loss=0.10, confidence=0.6,
            expected_cost_bps=5.0, execution_regime="normal_exec",
            crowding_score=0.2, liquidity_score=0.9, regime="trending",
        )
        params.update(kw)
        return ml.compute(**params)

    def test_admits_good_candidate(self):
        d = self._good()
        assert d.take_trade is True
        assert 0.0 < d.size_multiplier <= 1.0

    def test_edge_formula(self):
        d = self._good(mu=0.004, expected_cost_bps=5.0, p_tail_loss=0.10)
        assert math.isclose(d.expected_net_edge_bps, 10_000 * 0.004 - 5.0 - 0.10 * 50)

    def test_low_p_positive_rejects(self):
        assert self._good(p_positive=0.50).take_trade is False

    def test_high_p_tail_loss_rejects(self):
        assert self._good(p_tail_loss=0.30).take_trade is False

    def test_low_confidence_rejects(self):
        assert self._good(confidence=0.30).take_trade is False

    def test_insufficient_edge_rejects(self):
        assert self._good(mu=0.0005).take_trade is False

    def test_stressed_exec_blocks_marginal(self):
        # edge = 26 - 1 - 5 = 20 == min_edge(20) but < 1.5*min (30) -> blocked
        d = self._good(mu=0.0026, expected_cost_bps=1.0, execution_regime="stressed_exec")
        assert math.isclose(d.expected_net_edge_bps, 20.0)
        assert d.take_trade is False

    def test_size_multiplier_clips_to_unit_interval(self):
        d = self._good(p_positive=0.99, p_tail_loss=0.0, confidence=1.0,
                       liquidity_score=1.0, crowding_score=0.0)
        assert 0.0 <= d.size_multiplier <= 1.0
        rejected = self._good(p_positive=0.50)
        assert rejected.size_multiplier == 0.0


# ── 3. ML return model ───────────────────────────────────────────────────────────

class TestMLReturnModel:

    def test_feature_schema(self):
        assert len(mrm.FEATURE_NAMES) == 18
        assert mrm.FEATURE_SCHEMA_VERSION == "v6.0"

    def test_unfitted_returns_safe_fallback(self):
        model = mrm.MLReturnModel()
        feats = {n: 0.0 for n in mrm.FEATURE_NAMES}
        assert model.predict(feats) == mrm.SAFE_FALLBACK
        assert model.predict(feats) == (0.0, 0.15, 0.50, 0.10, 0.0)

    def test_fitted_returns_valid_5_tuple(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol, regime="trending")
        out = model.predict(_feature_dict(X[0]), current_regime="trending")
        assert len(out) == 5
        mu, sigma, p_pos, p_tail, conf = out
        assert -1.0 < mu < 1.0
        assert sigma > 0.0
        assert 0.0 <= p_pos <= 1.0
        assert 0.0 <= p_tail <= 1.0
        assert 0.0 <= conf <= 1.0

    def test_fit_rejects_wrong_shape(self):
        model = mrm.MLReturnModel()
        with pytest.raises(ValueError):
            model.fit(np.zeros((50, 5)), np.zeros(50), np.ones(50))

    def test_needs_refit_triggers(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol, regime="trending")
        assert model.needs_refit is False

        model.mark_feature_drift(True)
        assert model.needs_refit is True
        model.mark_feature_drift(False)

        import time
        model._fit_time = time.time() - 40 * 86400
        assert model.needs_refit is True
        model._fit_time = time.time()

        model._prev_calibration_error = 0.05
        model._calibration_error = 0.10
        assert model.needs_refit is True
        model._calibration_error = 0.05

        model.last_regime = "trending"
        model._current_regime = "high_vol"
        model._rolling_ic = -0.1
        model._prev_rolling_ic = 0.2
        assert model.needs_refit is True

    def test_refit_lock_serialises_fit(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data(n=25)
        assert hasattr(model._refit_lock, "acquire")

        model._refit_lock.acquire()
        done = threading.Event()

        def _run():
            model.fit(X, y_ret, y_vol)
            done.set()

        worker = threading.Thread(target=_run)
        worker.start()
        try:
            assert done.wait(timeout=1.0) is False   # blocked while lock held
        finally:
            model._refit_lock.release()
        assert done.wait(timeout=10.0) is True
        worker.join(timeout=10.0)

    def test_record_outcome_caps_log_and_sets_ic(self):
        model = mrm.MLReturnModel()
        rng = np.random.default_rng(1)
        for _ in range(80):
            model.record_outcome(predicted=rng.normal(), actual=rng.normal())
        assert len(model._prediction_log) <= 60

    def test_reports_have_documented_keys(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol)
        cal = model.calibration_report()
        for key in ("brier_score", "calibration_error", "n_outcomes", "is_calibrated"):
            assert key in cal
        drift = model.drift_report()
        for key in ("drift_flag", "rolling_ic", "model_age_days", "needs_refit"):
            assert key in drift

    def test_predict_batch_and_singleton(self):
        mrm.reset_model()
        model = mrm.get_model()
        assert model is mrm.get_model()
        feats = {n: 0.0 for n in mrm.FEATURE_NAMES}
        assert model.predict_batch([feats, feats]) == [mrm.SAFE_FALLBACK, mrm.SAFE_FALLBACK]

    def test_calibration_is_out_of_sample(self):
        # Item 4: on data with NO real signal, IN-SAMPLE calibration looks absurd
        # (brier ~0.004) because the tree ensemble memorises; honest OUT-OF-SAMPLE
        # calibration is ~a coin flip (brier ~0.25). Reported metrics must reflect
        # the held-out fold, not the training rows.
        rng = np.random.default_rng(3)
        X = rng.standard_normal((200, 18))
        y_ret = rng.normal(0.0, 0.01, 200)            # independent of X -> no signal
        y_vol = 0.02 + 0.005 * np.abs(rng.standard_normal(200))
        model = mrm.MLReturnModel()
        model.fit(X, y_ret, y_vol)
        assert model._brier_score is not None
        assert model._brier_score >= 0.15, model._brier_score   # not in-sample-optimistic
        assert model._hit_calibrator is not None                # predict() still works
        out = model.predict(_feature_dict(X[0]))
        assert len(out) == 5 and 0.0 <= out[2] <= 1.0

    def test_calibration_metrics_none_on_tiny_fit(self):
        # Item 4: too few samples for an honest OOS fold -> report None, not a
        # fabricated optimistic number (calibrators are still fit so predict works).
        X, y_ret, y_vol = _ml_training_data(n=25)
        model = mrm.MLReturnModel()
        model.fit(X, y_ret, y_vol)
        assert model._brier_score is None
        assert model._calibration_error is None
        assert model._hit_calibrator is not None
        assert model.predict(_feature_dict(X[0])) is not None

    def test_cross_sectional_prior_is_batch_grand_mean(self):
        # Item 5: the shrinkage prior must be the cross-sectional mean of this
        # cycle's RAW ensemble expected returns (James-Stein style), not the
        # hard-coded 0.0 that shrank every mu toward zero.
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol)
        feats = [_feature_dict(X[i]) for i in range(5)]
        model.predict_batch(feats)
        expected = float(
            model._ensemble_predictions(
                np.vstack([model._vectorize(f) for f in feats])
            ).mean(axis=1).mean()
        )
        assert model._cross_sectional_prior == pytest.approx(expected, rel=1e-9)
        assert model._cross_sectional_prior != 0.0

    def test_record_outcome_tracks_live_tail_rate(self):
        # Item: record_outcome must USE tail events (derived from actual < threshold)
        # to track a live realized tail-loss rate.
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol)
        assert model.live_tail_rate() is None                 # no live outcomes yet
        for i in range(25):
            model.record_outcome(predicted=0.0, actual=(-0.10 if i < 5 else 0.01))
        assert model.live_tail_rate() == pytest.approx(5 / 25)
        assert model.tail_calibration_error() is not None
        assert model.tail_calibration_error() >= 0.0

    def test_explicit_tail_event_is_honored(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol)
        for _ in range(25):
            model.record_outcome(predicted=0.0, actual=0.01, tail_event=True)   # forced
        assert model.live_tail_rate() == pytest.approx(1.0)

    def test_tail_miscalibration_triggers_refit(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()           # base tail rate ~ 0 (no big losses)
        model.fit(X, y_ret, y_vol)
        assert model.needs_refit is False
        for _ in range(30):
            model.record_outcome(predicted=0.0, actual=-0.20)   # persistent realized tails
        assert model.needs_refit is True                # live tail rate >> base -> refit

    def test_tail_metrics_in_calibration_report(self):
        model = mrm.MLReturnModel()
        X, y_ret, y_vol = _ml_training_data()
        model.fit(X, y_ret, y_vol)
        report = model.calibration_report()
        for key in ("live_tail_rate", "base_tail_rate", "tail_calibration_error"):
            assert key in report


# ── 4. ML direction filter ───────────────────────────────────────────────────────

class TestMLFilter:

    def _training(self, n: int = 120):
        rng = np.random.default_rng(0)
        dicts, ys = [], []
        for i in range(n):
            up = i % 2 == 0
            dicts.append({
                "ofi_signal": (0.4 if up else -0.4) + rng.normal(0, 0.05),
                "realized_vol_5d": 0.02, "realized_vol_20d": 0.02,
                "spread_bps": 5.0, "adv_ratio": 0.01,
                "correlation_to_portfolio": 0.3,
                "insider_flow_age_days": 10.0, "news_age_minutes": 30.0,
                "freshness_flags": {"price": False, "news": up},
            })
            ys.append(1 if up else -1)
        return dicts, ys

    def test_unfitted_abstains(self):
        assert mlf.MLFilter().predict_direction({}) == ("FLAT", 0.0)

    def test_fitted_predicts_direction(self):
        dicts, ys = self._training()
        f = mlf.MLFilter()
        f.fit(dicts, ys, freshness_keys=["price", "news"])
        assert f.predict_direction(dicts[0])[0] == "BUY"
        assert f.predict_direction(dicts[1])[0] == "SELL"
        assert 0.0 <= f.prob_up(dicts[0]) <= 1.0

    def test_ofi_gate_vetoes_to_flat(self):
        dicts, ys = self._training()
        f = mlf.MLFilter()
        f.fit(dicts, ys, freshness_keys=["price", "news"])
        direction, _ = f.predict_direction(dicts[0], ofi_norm=-0.9)
        assert direction == "FLAT"


# ── 5. FinBERT scorer ────────────────────────────────────────────────────────────

class TestFinBERT:

    def _valid_dict(self, d: dict) -> bool:
        return (
            set(d.keys()) == {"positive", "negative", "neutral"}
            and math.isclose(sum(d.values()), 1.0, abs_tol=1e-6)
            and all(0.0 <= v <= 1.0 for v in d.values())
        )

    def test_temperature_and_raw_softmax_are_valid_dicts(self):
        logits = [2.0, 0.5, 0.1]
        assert self._valid_dict(fb.temperature_scaled_probs(logits))
        assert self._valid_dict(fb.raw_softmax_probs(logits))

    def test_temperature_softens_distribution(self):
        logits = [3.0, 0.0, -1.0]
        temp = fb.temperature_scaled_probs(logits)
        raw = fb.raw_softmax_probs(logits)
        assert max(temp.values()) < max(raw.values())

    def test_offline_lexicon_fallback_is_valid_and_directional(self):
        scorer = fb.FinBERTScorer()
        scorer._available = False   # force offline path (no model download)
        pos = scorer.score("Company beats earnings as profit surges")
        neg = scorer.score("Company faces lawsuit as shares plunge")
        assert self._valid_dict(pos) and self._valid_dict(neg)
        assert pos["positive"] > pos["negative"]
        assert neg["negative"] > neg["positive"]

    def test_empty_headline_is_neutral(self):
        scorer = fb.FinBERTScorer()
        scorer._available = False
        assert scorer.score("   ") == {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def test_batch_defers_under_cpu_load(self):
        busy = fb.FinBERTScorer(cpu_percent_fn=lambda: 95.0)
        busy._available = False
        assert busy.batch_score(["a", "b"]) == []
        idle = fb.FinBERTScorer(cpu_percent_fn=lambda: 5.0)
        idle._available = False
        assert len(idle.batch_score(["Company beats", "Company misses"])) == 2


# ── 6. Sentiment aggregator ──────────────────────────────────────────────────────

class TestSentimentAggregator:

    def test_no_headlines_returns_zero(self):
        assert sa.aggregate("AAPL", []) == 0.0

    def test_out_of_window_excluded(self):
        old = [{"positive": 1.0, "negative": 0.0, "age_minutes": 99}]
        assert sa.aggregate("AAPL", old) == 0.0

    def test_fresh_news_dominates(self):
        heads = [
            {"positive": 0.9, "negative": 0.0, "age_minutes": 1},     # fresh, bullish
            {"positive": 0.0, "negative": 0.9, "age_minutes": 55},    # stale, bearish
        ]
        score = sa.aggregate("AAPL", heads)
        assert score > 0.0          # the fresh bullish headline wins
        assert -1.0 <= score <= 1.0

    def test_symbol_filter(self):
        heads = [{"positive": 1.0, "negative": 0.0, "age_minutes": 1, "symbol": "MSFT"}]
        assert sa.aggregate("AAPL", heads) == 0.0
        assert sa.aggregate("MSFT", heads) > 0.0


# ── 7. Sentiment pipeline (news → scorer → per-symbol sentiment) ─────────────────

class _FakeScorer:
    """Deterministic scorer: 'good' → bullish, 'bad' → bearish, else neutral."""

    def score(self, text: str) -> dict:
        lower = text.lower()
        if "good" in lower:
            return {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
        if "bad" in lower:
            return {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


class TestSentimentPipeline:

    def test_no_news_is_neutral_for_all_symbols(self):
        # No news ⇒ every symbol gets exactly 0.0 and the scorer is never needed.
        assert sp.compute_sentiment_scores([], ["AAPL", "MSFT"]) == {"AAPL": 0.0, "MSFT": 0.0}

    def test_scores_are_directional_and_bounded(self):
        news = [
            {"headline": "good quarter", "age_minutes": 1.0, "symbol": "AAPL"},
            {"headline": "bad outlook", "age_minutes": 1.0, "symbol": "MSFT"},
        ]
        out = sp.compute_sentiment_scores(news, ["AAPL", "MSFT", "GOOG"], scorer=_FakeScorer())
        assert out["AAPL"] > 0.0 > out["MSFT"]
        assert out["GOOG"] == 0.0                      # no relevant news ⇒ neutral
        assert all(-1.0 <= v <= 1.0 for v in out.values())

    def test_symbols_list_metadata_is_respected(self):
        news = [{"text": "good news", "age_minutes": 1.0, "symbols": ["AAPL", "GOOG"]}]
        out = sp.compute_sentiment_scores(news, ["AAPL", "GOOG", "MSFT"], scorer=_FakeScorer())
        assert out["AAPL"] > 0.0 and out["GOOG"] > 0.0
        assert out["MSFT"] == 0.0

    def test_window_minutes_excludes_stale_news(self):
        news = [{"headline": "good", "age_minutes": 30.0, "symbol": "AAPL"}]
        wide = sp.compute_sentiment_scores(news, ["AAPL"], scorer=_FakeScorer())
        narrow = sp.compute_sentiment_scores(news, ["AAPL"], scorer=_FakeScorer(),
                                             window_minutes=10.0)
        assert wide["AAPL"] > 0.0
        assert narrow["AAPL"] == 0.0


# ── 8. Training buffer + real refit (ROADMAP Phase 4 — learning loop) ─────────────

class TestRefitLoop:

    def _features(self, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        return {n: float(v) for n, v in zip(mrm.FEATURE_NAMES, rng.standard_normal(18))}

    def test_training_examples_accumulate_and_cap(self):
        model = mrm.MLReturnModel(training_buffer_max=10)
        for i in range(15):
            model.record_training_example(self._features(i), realized_return=0.01)
        assert model.training_buffer_size == 10           # rolling window, capped

    def test_refit_with_too_few_samples_is_a_safe_noop(self):
        model = mrm.MLReturnModel()
        for i in range(5):
            model.record_training_example(self._features(i), realized_return=0.01)
        assert model.refit() is False
        assert model._fitted is False

    def test_refit_fits_from_the_buffer(self):
        model = mrm.MLReturnModel()
        rng = np.random.default_rng(1)
        for i in range(60):
            f = self._features(i)
            ret = 0.5 * f["signal_score"] * 0.02 + float(rng.normal(0, 0.005))
            model.record_training_example(f, realized_return=ret)
        assert model.ready_for_initial_fit is True
        assert model.refit() is True
        assert model._fitted is True
        assert model.ready_for_initial_fit is False        # now fitted
        assert model.predict(self._features(99)) != mrm.SAFE_FALLBACK

    def test_ready_for_initial_fit_needs_enough_samples(self):
        model = mrm.MLReturnModel()
        for i in range(10):
            model.record_training_example(self._features(i), realized_return=0.0)
        assert model.ready_for_initial_fit is False
