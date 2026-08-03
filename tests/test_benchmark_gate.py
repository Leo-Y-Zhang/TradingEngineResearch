"""Falsification suite for the registered benchmark-relative criterion (C8 of the
2026-07-28 gate review): the function must pass genuinely-better candidates, fail
genuinely-worse ones, refuse to promote the indistinguishable, report pairing, detect
benchmark shopping, and never promote junk. A committed-series replay cross-checks the
recorded carry-vs-trend statistics when the banked parquets are present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.benchmark_gate import (
    BenchmarkComparison,
    benchmark_relative_rule,
    paired_sharpe_comparison,
)

_PPY = 12  # the programme's sleeve series are monthly


def _months(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _series(values: np.ndarray, n: int | None = None) -> pd.Series:
    n = len(values) if n is None else n
    return pd.Series(np.asarray(values, dtype=float), index=_months(n))


# ── contract: identical window, clean inputs, registered parameters ───────────

def test_window_mismatch_raises() -> None:
    a = _series(np.zeros(120) + 0.001)
    b = pd.Series(np.zeros(120) + 0.001, index=_months(120, start="2001-01-31"))
    with pytest.raises(ValueError, match="IDENTICAL"):
        paired_sharpe_comparison(a, b, _PPY)


def test_nan_raises() -> None:
    a = _series(np.zeros(120) + 0.001)
    b = a.copy()
    b.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        paired_sharpe_comparison(a, b, _PPY)


def test_registered_minimum_draws_enforced() -> None:
    a = _series(np.random.default_rng(0).normal(0.005, 0.03, 120))
    with pytest.raises(ValueError, match="10"):
        paired_sharpe_comparison(a, a * 0.9, _PPY, n_boot=500)


def test_empty_panel_refused() -> None:
    a = _series(np.random.default_rng(0).normal(0.005, 0.03, 120))
    b = _series(np.random.default_rng(1).normal(0.004, 0.03, 120))
    with pytest.raises(ValueError, match="PANEL"):
        benchmark_relative_rule(a, b, {}, _PPY)


# ── verdicts: BEATS / LOSES / UNDETERMINED ────────────────────────────────────

def _bench(seed: int = 1, n: int = 360, mu: float = 0.004, vol: float = 0.03) -> pd.Series:
    return _series(np.random.default_rng(seed).normal(mu, vol, n))


def test_genuinely_better_candidate_beats() -> None:
    bench = _bench()
    cand = bench + 0.004  # constant positive alpha, rho ~ 1: unambiguously better
    cmp_ = paired_sharpe_comparison(cand, bench, _PPY)
    assert cmp_.verdict == "BEATS"
    assert cmp_.sharpe_gap > 0
    assert cmp_.diff_lower_90 >= 0
    assert cmp_.rho > 0.99


def test_genuinely_worse_candidate_loses() -> None:
    bench = _bench()
    cand = bench - 0.004
    cmp_ = paired_sharpe_comparison(cand, bench, _PPY)
    assert cmp_.verdict == "LOSES"
    assert cmp_.diff_upper_90 <= 0


def test_indistinguishable_candidate_is_undetermined_and_not_promotable() -> None:
    rng = np.random.default_rng(3)
    bench = _bench(seed=2, n=120)
    # A tiny alpha buried in independent noise on a short window: real but unprovable.
    cand = bench + 0.0003 + pd.Series(rng.normal(0.0, 0.02, 120), index=bench.index)
    cmp_ = paired_sharpe_comparison(cand, bench, _PPY)
    assert cmp_.verdict == "UNDETERMINED"
    out = benchmark_relative_rule(cand, bench, {"own": bench}, _PPY)
    assert out.promotable is False


def test_pairing_tightens_the_interval() -> None:
    bench = _bench(seed=4)
    rng = np.random.default_rng(5)
    paired = bench + 0.002 + pd.Series(rng.normal(0.0, 0.004, len(bench)), index=bench.index)
    independent = _series(rng.normal(float(bench.mean()) + 0.002, 0.03, len(bench)))
    w_paired = paired_sharpe_comparison(paired, bench, _PPY)
    w_indep = paired_sharpe_comparison(independent, bench, _PPY)
    width_paired = w_paired.diff_upper_90 - w_paired.diff_lower_90
    width_indep = w_indep.diff_upper_90 - w_indep.diff_lower_90
    assert width_paired < width_indep, (
        f"paired CI ({width_paired:.3f}) should be tighter than independent "
        f"({width_indep:.3f}) at high rho ({w_paired.rho:.2f} vs {w_indep.rho:.2f})"
    )


# ── the benchmark-shopping detector (C7) ──────────────────────────────────────

def test_beats_nominated_but_loses_to_panel_member_is_sensitive_not_promotable() -> None:
    weak = _bench(seed=6, mu=0.001)
    strong = weak + 0.008  # a much better registered opportunity cost
    cand = weak + 0.004    # beats weak clearly, loses to strong clearly
    out = benchmark_relative_rule(cand, weak, {"own_ew": weak, "passive_ew": strong}, _PPY)
    assert out.nominated.verdict == "BEATS"
    assert out.panel["passive_ew"].verdict != "BEATS"
    assert out.benchmark_sensitive is True
    assert out.promotable is False


# ── junk control and determinism ──────────────────────────────────────────────

def test_junk_is_never_promotable() -> None:
    bench = _bench(seed=7)
    for seed in (101, 102, 103, 104, 105):
        junk = _series(np.random.default_rng(seed).normal(0.0, 0.03, len(bench)))
        out = benchmark_relative_rule(junk, bench, {"own": bench}, _PPY)
        assert out.promotable is False, f"junk seed={seed} promoted"


def test_deterministic_under_seed() -> None:
    bench = _bench(seed=8)
    cand = bench + 0.002
    a = paired_sharpe_comparison(cand, bench, _PPY, seed=99)
    b = paired_sharpe_comparison(cand, bench, _PPY, seed=99)
    assert a == b
    assert isinstance(a, BenchmarkComparison)


# ── committed-series replay: carry vs the trend reference ─────────────────────

_CARRY = Path("research/sleeves/_carry_output/carry_primary_net_monthly.parquet")
_TREND_REF = Path("research/sleeves/_carry_output/trend_reference_net_monthly.parquet")


@pytest.mark.skipif(not (_CARRY.exists() and _TREND_REF.exists()),
                    reason="banked sleeve parquets not present")
def test_replay_carry_vs_trend_reference_not_beats() -> None:
    """Replays a recorded comparison: carry (net Sharpe ~0.43) against the trend
    reference (~0.455), recorded rho ~ -0.044 on 269 shared months. Carry must NOT
    come out BEATS, and the measured rho must be in the recorded neighbourhood."""
    pytest.importorskip("pyarrow")
    carry = pd.read_parquet(_CARRY).squeeze("columns").dropna()
    trend = pd.read_parquet(_TREND_REF).squeeze("columns").dropna()
    common = carry.index.intersection(trend.index)
    assert len(common) >= 200, f"unexpectedly short common window: {len(common)}"
    cmp_ = paired_sharpe_comparison(carry.loc[common], trend.loc[common], _PPY)
    assert cmp_.verdict != "BEATS", (
        f"recorded MARGINAL carry arm came out BEATS: gap={cmp_.sharpe_gap:.3f}, "
        f"lb90={cmp_.diff_lower_90:.3f}"
    )
    assert abs(cmp_.rho - (-0.0441)) < 0.15, f"rho {cmp_.rho:.4f} far from recorded -0.0441"
