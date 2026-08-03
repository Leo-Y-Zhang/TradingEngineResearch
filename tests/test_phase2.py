"""
Phase 2 Tests — Research Validation and Alpha Factory
======================================================
Covers all test targets from the build spec:

  - PurgedWalkForwardSplitter produces non-overlapping splits with correct embargo gaps
  - leakage_guard() detects known timestamp overlaps
  - selection_rule() returns False correctly when any of the six conditions fails
  - promote_factor() rejects a factor with correlation > 0.80 to an existing live factor
  - SignalOutput dataclass is instantiable with all required fields
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from research.validation import (
    PurgedWalkForwardSplitter,
    ValidationResult,
    leakage_guard,
    evaluate_alpha_stability,
    selection_rule,
)
import research.alpha_factory as af
from research.alpha_factory import (
    SignalOutput,
    orthogonalize_factor,
    promote_factor,
)
from data.data_contracts import normalize_mode


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _date_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """Business-day DatetimeIndex of length n."""
    return pd.bdate_range(start=start, periods=n)


def _passing_result(**overrides) -> ValidationResult:
    """A ValidationResult that passes selection_rule() by default."""
    defaults = dict(
        mean_ic=0.05, mean_rank_ic=0.04, sharpe_net=1.2,
        turnover=0.02, hit_rate=0.55, max_drawdown=-0.05,
        pbo_proxy=0.15, deflated_sharpe_proxy=0.50,
        cost_drag_bps=5.0, stability_score=0.70, deflated_sharpe_ratio=0.99,
        regime_breakdown={"trending": {"sharpe": 0.9, "ic": 0.03}},
        leakage_flags=[],
    )
    return ValidationResult(**(defaults | overrides))


# ── 1. PurgedWalkForwardSplitter ─────────────────────────────────────────────

class TestPurgedWalkForwardSplitter:

    def _make_splitter(self, **kwargs):
        defaults = dict(
            train_size=252, valid_size=63, test_size=63,
            embargo_size=5, label_horizon=5,
        )
        return PurgedWalkForwardSplitter(**(defaults | kwargs))

    def test_purge_is_positional_on_business_day_index(self):
        """SIGNALS-2: purge by BAR count, not calendar days. On a business-day
        index a 5-bar label must leave a >= 5-position gap before the eval window
        (calendar-day purging would under-purge: 5 cal days ~ 3-4 trading days)."""
        label_horizon = 5
        splitter = self._make_splitter(label_horizon=label_horizon, embargo_size=0)
        ts = pd.bdate_range("2018-01-01", periods=700)
        splits = splitter.split(ts)
        assert splits
        for train, valid, _test in splits:
            if len(train) == 0:
                continue
            eval_start_pos = int(valid[0])
            assert int(train.max()) + label_horizon < eval_start_pos, (
                "kept training obs label window reaches into the eval window "
                "(calendar-day under-purge)"
            )

    def test_produces_at_least_one_split(self):
        ts = _date_index(600)
        splits = self._make_splitter().split(ts)
        assert len(splits) >= 1

    def test_split_tuples_have_three_arrays(self):
        ts = _date_index(600)
        for train, valid, test in self._make_splitter().split(ts):
            assert isinstance(train, np.ndarray)
            assert isinstance(valid, np.ndarray)
            assert isinstance(test, np.ndarray)

    def test_no_overlap_between_train_valid_test(self):
        ts = _date_index(700)
        for train, valid, test in self._make_splitter().split(ts):
            assert len(set(train) & set(valid)) == 0, "train and valid overlap"
            assert len(set(train) & set(test))  == 0, "train and test overlap"
            assert len(set(valid) & set(test))  == 0, "valid and test overlap"

    def test_train_comes_before_valid_before_test(self):
        ts = _date_index(700)
        for train, valid, test in self._make_splitter().split(ts):
            if len(train) > 0:
                assert train.max() < valid.min(), "train indices not strictly before valid"
            assert valid.max() < test.min(), "valid indices not strictly before test"

    def test_embargo_removes_bars_before_eval_window(self):
        """After purging, no training index should fall within embargo_size of train_end."""
        embargo_size = 10
        splitter = self._make_splitter(embargo_size=embargo_size)
        ts = _date_index(700)
        splits = splitter.split(ts)
        for train, valid, _test in splits:
            if len(train) == 0:
                continue
            # The embargo zone is the embargo_size bars immediately before valid[0]
            embargo_start = int(valid[0]) - embargo_size
            for idx in train:
                assert idx < embargo_start, (
                    f"Train index {idx} is in embargo zone "
                    f"[{embargo_start}, {int(valid[0])})"
                )

    def test_purging_removes_overlapping_label_windows(self):
        """No training observation's label window should reach into the eval period."""
        label_horizon = 10
        splitter = self._make_splitter(label_horizon=label_horizon, embargo_size=0)
        ts = _date_index(700)
        splits = splitter.split(ts)
        for train, valid, _test in splits:
            eval_start_ts = ts[int(valid[0])]
            horizon_td = pd.Timedelta(days=label_horizon)
            for i in train:
                label_end = ts[i] + horizon_td
                assert label_end < eval_start_ts, (
                    f"Training obs at {ts[i].date()} has label_end={label_end.date()} "
                    f">= eval_start={eval_start_ts.date()} — purging failed"
                )

    def test_valid_and_test_sizes_match_config(self):
        ts = _date_index(700)
        splitter = self._make_splitter(valid_size=40, test_size=30)
        for _train, valid, test in splitter.split(ts):
            assert len(valid) == 40
            assert len(test)  == 30

    def test_insufficient_data_raises(self):
        ts = _date_index(50)   # less than train+valid+test = 378
        with pytest.raises(ValueError, match="Not enough observations"):
            self._make_splitter().split(ts)

    def test_zero_embargo_allowed(self):
        ts = _date_index(600)
        splitter = self._make_splitter(embargo_size=0)
        splits = splitter.split(ts)
        assert len(splits) >= 1

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            PurgedWalkForwardSplitter(
                train_size=0, valid_size=63, test_size=63,
                embargo_size=5, label_horizon=5,
            )
        with pytest.raises(ValueError):
            PurgedWalkForwardSplitter(
                train_size=252, valid_size=63, test_size=63,
                embargo_size=-1, label_horizon=5,
            )


# ── 2. leakage_guard ─────────────────────────────────────────────────────────

class TestLeakageGuard:

    def test_clean_data_returns_empty_flags(self):
        idx = _date_index(100)
        feat_df  = pd.DataFrame({"f": np.random.randn(100)}, index=idx)
        # Labels share the same index — perfect alignment, no features ahead of labels
        label_df = pd.DataFrame({"ret": np.random.randn(100)}, index=idx)
        flags = leakage_guard(feat_df, label_df, label_horizon=1)
        # No FEATURE_AHEAD_OF_LABELS flag
        horizon_flags = [f for f in flags if "FEATURE_AHEAD_OF_LABELS" in f]
        assert len(horizon_flags) == 0

    def test_feature_ahead_of_labels_detected(self):
        idx = _date_index(100)
        feat_df  = pd.DataFrame({"f": np.random.randn(100)}, index=idx)
        # Labels end before features end — features are "ahead"
        label_df = pd.DataFrame({"ret": np.random.randn(50)}, index=idx[:50])
        flags = leakage_guard(feat_df, label_df, label_horizon=1)
        assert any("FEATURE_AHEAD_OF_LABELS" in f for f in flags)

    def test_dense_labels_not_flagged_as_leakage(self):
        # SIGNALS-1 fix: overlapping forward-label windows are the NORMAL property
        # of dense daily sampling with horizon>1 - handled by the splitter's purge,
        # NOT leakage. leakage_guard must NOT brick the promotion gate on a clean,
        # contiguous, properly-aligned panel.
        idx = pd.date_range("2020-01-01", periods=40, freq="1D")
        feat_df  = pd.DataFrame({"f": np.ones(40)}, index=idx)
        label_df = pd.DataFrame({"ret": np.ones(40)}, index=idx)
        flags = leakage_guard(feat_df, label_df, label_horizon=5)
        assert not any("LABEL_WINDOW_OVERLAP" in f for f in flags)
        assert flags == []   # a clean aligned panel has no blocking leakage flags

    def test_label_data_insufficient_flagged(self):
        # The one genuine data-sufficiency failure Check 3 still catches: labels end
        # before even the earliest feature could be labelled.
        idx = pd.date_range("2020-01-01", periods=40, freq="1D")
        feat_df  = pd.DataFrame({"f": np.ones(40)}, index=idx)
        label_df = pd.DataFrame({"ret": np.ones(2)}, index=idx[:2])  # only 2 labels
        flags = leakage_guard(feat_df, label_df, label_horizon=10)
        assert any("LABEL_DATA_INSUFFICIENT" in f for f in flags)

    def test_empty_inputs_returns_flag(self):
        flags = leakage_guard(pd.DataFrame(), pd.DataFrame(), label_horizon=1)
        assert any("EMPTY_INPUT" in f for f in flags)


# ── 3. selection_rule ─────────────────────────────────────────────────────────

class TestSelectionRule:

    def test_passing_result_returns_true(self):
        assert selection_rule(_passing_result()) is True

    def test_low_mean_rank_ic_fails(self):
        assert selection_rule(_passing_result(mean_rank_ic=0.005)) is False

    def test_low_sharpe_net_fails(self):
        assert selection_rule(_passing_result(sharpe_net=0.50)) is False

    def test_low_stability_score_fails(self):
        assert selection_rule(_passing_result(stability_score=0.40)) is False

    def test_low_deflated_sharpe_fails(self):
        assert selection_rule(_passing_result(deflated_sharpe_proxy=0.10)) is False

    def test_leakage_flags_present_fails(self):
        assert selection_rule(
            _passing_result(leakage_flags=["LABEL_WINDOW_OVERLAP"])
        ) is False

    def test_bad_regime_sharpe_fails(self):
        assert selection_rule(
            _passing_result(regime_breakdown={"trending": {"sharpe": -0.60}})
        ) is False

    def test_multiple_failures_still_false(self):
        result = _passing_result(
            mean_rank_ic=0.001,
            sharpe_net=0.1,
            leakage_flags=["OVERLAP"],
        )
        assert selection_rule(result) is False

    def test_exactly_at_boundary_fails(self):
        """Boundaries are strict (> not >=)."""
        assert selection_rule(_passing_result(mean_rank_ic=0.01)) is False
        assert selection_rule(_passing_result(sharpe_net=0.75))   is False


# ── 4. SignalOutput ───────────────────────────────────────────────────────────

class TestSignalOutput:

    def test_buy_signal_instantiates(self):
        s = SignalOutput(
            symbol="AAPL", direction="BUY", raw_score=0.7,
            expected_horizon=5, decay_half_life=10,
            confidence_proxy=0.65, sleeve_name="momentum",
            asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        )
        assert s.direction == "BUY"

    def test_sell_and_flat_directions_valid(self):
        base = dict(
            symbol="MSFT", raw_score=0.0,
            expected_horizon=5, decay_half_life=10,
            confidence_proxy=0.5, sleeve_name="mean_reversion",
            asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
        )
        SignalOutput(direction="SELL", **base)
        SignalOutput(direction="FLAT", **base)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            SignalOutput(
                symbol="AAPL", direction="LONG", raw_score=0.5,
                expected_horizon=5, decay_half_life=10,
                confidence_proxy=0.5, sleeve_name="momentum",
                asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
            )

    def test_raw_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="raw_score"):
            SignalOutput(
                symbol="AAPL", direction="BUY", raw_score=1.5,
                expected_horizon=5, decay_half_life=10,
                confidence_proxy=0.5, sleeve_name="momentum",
                asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence_proxy"):
            SignalOutput(
                symbol="AAPL", direction="BUY", raw_score=0.5,
                expected_horizon=5, decay_half_life=10,
                confidence_proxy=1.5, sleeve_name="momentum",
                asof_timestamp=datetime(2026, 6, 6, tzinfo=timezone.utc),
            )


# ── 5. promote_factor ─────────────────────────────────────────────────────────

class TestPromoteFactor:

    def _make_live_matrix(self, n: int = 200) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, 2))

    def _make_candidate(self, n: int = 200, corr_with: np.ndarray | None = None,
                        target_corr: float = 0.0, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cand = rng.standard_normal(n)
        if corr_with is not None and target_corr > 0:
            noise = rng.standard_normal(n)
            cand = target_corr * corr_with + np.sqrt(1 - target_corr ** 2) * noise
        return cand

    def test_passes_with_good_result_and_low_correlation(self):
        live = self._make_live_matrix()
        cand = self._make_candidate()
        result = _passing_result()
        assert promote_factor("factor_a", result, live, cand) is True

    def test_rejects_high_correlation_with_live_factor(self):
        n = 200
        live = self._make_live_matrix(n)
        # Make candidate highly correlated with first live factor
        cand = self._make_candidate(n, corr_with=live[:, 0], target_corr=0.95)
        result = _passing_result()
        assert promote_factor("factor_b", result, live, cand) is False

    def test_rejects_when_selection_rule_fails(self):
        live = self._make_live_matrix()
        cand = self._make_candidate()
        bad_result = _passing_result(sharpe_net=0.1)
        assert promote_factor("factor_c", bad_result, live, cand) is False

    def test_rejects_negative_sharpe_even_if_rule_would_pass_otherwise(self):
        live = self._make_live_matrix()
        cand = self._make_candidate()
        result = _passing_result(sharpe_net=-0.5)
        assert promote_factor("factor_d", result, live, cand) is False

    def test_empty_live_matrix_skips_correlation_check(self):
        empty_live = np.empty((200, 0))
        cand = self._make_candidate()
        result = _passing_result()
        # Should pass since there are no live factors to correlate against
        assert promote_factor("factor_e", result, empty_live, cand) is True


# ── 6. orthogonalize_factor ───────────────────────────────────────────────────

class TestOrthogonalizeFactor:

    def test_result_is_orthogonal_to_live_factors(self):
        rng = np.random.default_rng(0)
        n = 300
        live = rng.standard_normal((n, 3))
        candidate = rng.standard_normal(n)
        ortho = orthogonalize_factor(candidate, live)
        for j in range(3):
            dot = float(np.dot(ortho, live[:, j]))
            assert abs(dot) < 1e-6, f"Not orthogonal to live factor {j}: dot={dot}"

    def test_empty_live_returns_candidate_unchanged_in_direction(self):
        rng = np.random.default_rng(1)
        cand = rng.standard_normal(100)
        ortho = orthogonalize_factor(cand, np.empty((100, 0)))
        # Direction should be the same (normalised)
        cand_norm = cand / np.std(cand)
        corr = abs(float(np.corrcoef(cand_norm, ortho)[0, 1]))
        assert corr > 0.99

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            orthogonalize_factor(np.ones(10), np.ones((20, 2)))


# ── 7. evaluate_alpha_stability ───────────────────────────────────────────────

class TestEvaluateAlphaStability:

    def test_all_positive_ic_gives_high_stability(self):
        ic = pd.Series([0.03, 0.04, 0.05, 0.03, 0.02])
        score = evaluate_alpha_stability(ic)
        assert score > 0.7

    def test_all_negative_ic_gives_low_stability(self):
        ic = pd.Series([-0.03, -0.04, -0.05, -0.03, -0.02])
        score = evaluate_alpha_stability(ic)
        assert score < 0.3

    def test_mixed_ic_gives_middling_stability(self):
        ic = pd.Series([0.03, -0.02, 0.04, -0.01, 0.05])
        score = evaluate_alpha_stability(ic)
        assert 0.0 <= score <= 1.0

    def test_empty_series_returns_zero(self):
        assert evaluate_alpha_stability(pd.Series([], dtype=float)) == 0.0


# ── 8. normalize_mode (security fix) ─────────────────────────────────────────

class TestNormalizeMode:

    def test_valid_modes_accepted(self):
        for m in ("RESEARCH", "PAPER", "LIVE"):
            assert normalize_mode(m) == m

    def test_case_insensitive(self):
        assert normalize_mode("live") == "LIVE"
        assert normalize_mode("paper") == "PAPER"
        assert normalize_mode("research") == "RESEARCH"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown trading mode"):
            normalize_mode("SHADOW")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Unknown trading mode"):
            normalize_mode("")


# ── Factor-promotion pipeline (ROADMAP Phase 2) ──────────────────────────────────

class TestFactoryPromotion:

    def setup_method(self):
        af.reset_live_factors()

    def teardown_method(self):
        af.reset_live_factors()

    def _returns(self, n: int = 120, k: int = 3, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = _date_index(n)
        return pd.DataFrame(rng.normal(0.0, 0.01, (n, k)), index=idx,
                            columns=[f"S{i}" for i in range(k)])

    def _factor(self, idx, seed: int) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0, 1.0, len(idx)), index=idx)

    def test_junk_factor_not_promoted_real_pipeline(self):
        # End-to-end with the REAL evaluate_factor: random junk has no edge, so it
        # fails selection_rule and is not promoted (the library stays empty).
        ret = self._returns()
        splitter = PurgedWalkForwardSplitter(
            train_size=40, valid_size=15, test_size=15, embargo_size=2, label_horizon=2
        )
        out = af.promote_candidates({"junk": self._factor(ret.index, 1)}, ret, splitter=splitter)
        assert out["junk"].promoted is False
        assert af.get_live_factors()[0] == []

    def test_known_edge_factor_passes_evaluation(self):
        # Dual of junk-rejection (SIGNALS-6): a factor that genuinely predicts the
        # FORWARD market return must be ACCEPTED by evaluate_factor + selection_rule.
        # Near-perfect predictor: factor[t] = forward market return + tiny noise.
        h = 2
        ret = self._returns(n=300, k=3, seed=7)
        fwd = ret.mean(axis=1).rolling(h).sum().shift(-h)
        noise = np.random.default_rng(7).normal(0.0, 1e-4, len(ret))
        factor = (fwd + noise).fillna(0.0)
        splitter = PurgedWalkForwardSplitter(
            train_size=60, valid_size=30, test_size=30, embargo_size=2, label_horizon=h
        )
        result = af.evaluate_factor(factor, ret, splitter=splitter)
        assert result.mean_rank_ic > 0.1, f"real edge should show high forward IC, got {result.mean_rank_ic}"
        assert selection_rule(result) is True, result.leakage_flags

    def test_junk_rejected_across_seeds(self):
        # SIGNALS-6: random junk must NOT validate, robustly across seeds.
        splitter = PurgedWalkForwardSplitter(
            train_size=60, valid_size=30, test_size=30, embargo_size=2, label_horizon=2
        )
        for seed in (1, 2, 3, 4, 5):
            ret = self._returns(n=300, k=3, seed=seed)
            factor = self._factor(ret.index, seed + 100)
            result = af.evaluate_factor(factor, ret, splitter=splitter)
            assert selection_rule(result) is False, f"junk seed={seed} wrongly passed: {result}"

    def test_passing_distinct_factor_is_promoted(self, monkeypatch):
        monkeypatch.setattr(af, "evaluate_factor", lambda *a, **k: _passing_result())
        ret = self._returns()
        out = af.promote_candidates({"A": self._factor(ret.index, 10)}, ret)
        assert out["A"].promoted is True
        names, matrix = af.get_live_factors()
        assert names == ["A"] and matrix.shape[1] == 1

    def test_near_duplicate_rejected_by_correlation(self, monkeypatch):
        monkeypatch.setattr(af, "evaluate_factor", lambda *a, **k: _passing_result())
        ret = self._returns()
        base = self._factor(ret.index, 10)
        dup = base + 1e-6 * self._factor(ret.index, 99)        # |corr| ~ 1.0 with base
        out = af.promote_candidates({"A": base, "Adup": dup}, ret)
        assert out["A"].promoted is True
        assert out["Adup"].promoted is False
        assert af.get_live_factors()[0] == ["A"]

    def test_distinct_second_factor_grows_library(self, monkeypatch):
        monkeypatch.setattr(af, "evaluate_factor", lambda *a, **k: _passing_result())
        ret = self._returns()
        out = af.promote_candidates(
            {"A": self._factor(ret.index, 10), "B": self._factor(ret.index, 20)}, ret
        )
        assert out["A"].promoted and out["B"].promoted
        names, matrix = af.get_live_factors()
        assert set(names) == {"A", "B"} and matrix.shape[1] == 2

    def test_failed_selection_rule_not_promoted(self, monkeypatch):
        monkeypatch.setattr(af, "evaluate_factor", lambda *a, **k: _passing_result(sharpe_net=0.1))
        ret = self._returns()
        out = af.promote_candidates({"A": self._factor(ret.index, 10)}, ret)
        assert out["A"].promoted is False
        assert out["A"].passed_selection_rule is False
        assert af.get_live_factors()[0] == []

    def test_reset_clears_library(self, monkeypatch):
        monkeypatch.setattr(af, "evaluate_factor", lambda *a, **k: _passing_result())
        ret = self._returns()
        af.promote_candidates({"A": self._factor(ret.index, 10)}, ret)
        assert af.get_live_factors()[0] == ["A"]
        af.reset_live_factors()
        assert af.get_live_factors()[0] == []


# ── 5. Deflated Sharpe Ratio & PBO (Bailey & Lopez de Prado) ─────────────────────

class TestDeflatedSharpeAndPBO:

    def test_dsr_high_for_strong_consistent_returns(self):
        from research.validation import deflated_sharpe_ratio
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.01, 250)        # per-period Sharpe ~ 1.0, T=250
        assert deflated_sharpe_ratio(r, n_trials=1) > 0.95

    def test_dsr_low_for_zero_mean_noise(self):
        from research.validation import deflated_sharpe_ratio
        rng = np.random.default_rng(1)
        r = rng.normal(0.0, 0.01, 250)
        assert deflated_sharpe_ratio(r, n_trials=1) < 0.95

    def test_dsr_deflates_with_more_trials(self):
        from research.validation import deflated_sharpe_ratio
        rng = np.random.default_rng(2)
        r = rng.normal(0.002, 0.01, 120)
        assert deflated_sharpe_ratio(r, n_trials=100) < deflated_sharpe_ratio(r, n_trials=1)

    def test_dsr_degenerate_returns_zero(self):
        from research.validation import deflated_sharpe_ratio
        assert deflated_sharpe_ratio([1.0, 1.0, 1.0, 1.0]) == 0.0   # zero variance
        assert deflated_sharpe_ratio([1.0, 2.0]) == 0.0            # too few observations

    def test_pbo_around_half_for_pure_noise(self):
        from research.validation import probability_of_backtest_overfitting
        rng = np.random.default_rng(3)
        M = rng.normal(0.0, 1.0, (120, 20))    # noise configs -> IS-best is random OOS
        assert 0.3 < probability_of_backtest_overfitting(M, n_splits=8) < 0.7

    def test_pbo_low_for_genuine_edge(self):
        from research.validation import probability_of_backtest_overfitting
        rng = np.random.default_rng(4)
        M = rng.normal(0.0, 1.0, (120, 20))
        M[:, 0] += 0.5                          # config 0 has a persistent edge
        assert probability_of_backtest_overfitting(M, n_splits=8) < 0.3
