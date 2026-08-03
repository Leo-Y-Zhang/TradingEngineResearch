"""Tests for the Sharadar fundamental-alpha research runner
(``scripts/research_sharadar_alpha.py``). Offline, deterministic, NO network, NO paid data.

Mirrors ``tests/test_signal_learner.py``: the runner must (1) recover an injected real
cross-sectional edge end-to-end (through the Sharadar PIT panel → fundamental features →
forward returns → learned, purged walk-forward combination) and report DEPLOYABLE, and
(2) DEFAULT-DENY pure noise so it can never promote a non-edge (golden rule 5 / SIGNALS-5).
A degenerate (too-short) panel must fail closed. The ``--selftest`` CLI entrypoint is also
exercised, both in-process and via subprocess, and must exit 0.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_sharadar_alpha.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("research_sharadar_alpha", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module          # so @dataclass can resolve module annotations
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# ── End-to-end edge recovery + noise rejection (paired contract) ──────────────────────
def test_recovers_real_edge_and_is_deployable(tmp_path):
    runner.write_synthetic_csvs(tmp_path, seed=7, edge=True)
    sf1, sep = runner.load_panel(tmp_path)
    report = runner.run_research(sf1, sep, label="edge")

    # Sanity: the full library is present and the universe/grid is non-trivial.
    assert set(report.weights) == set(runner.FEATURE_NAMES)
    assert report.n_symbols >= 3 and report.n_rebalances >= 8

    assert report.result.mean_ic > 0.0                       # learned a real cross-sectional edge
    assert report.result.mean_rank_ic > 0.01
    assert report.result.deflated_sharpe_ratio > runner.DSR_CUTOFF   # survives deflation
    assert report.deployable is True                         # clears selection_rule
    assert report.weights["roe"] > 0.0                       # positive loading on the true driver
    # ROE is the dominant signal vs the noise factors it competes with.
    assert report.weights["roe"] >= max(abs(report.weights[f]) for f in runner.FEATURE_NAMES) - 1e-9


def test_noise_is_default_denied(tmp_path):
    runner.write_synthetic_csvs(tmp_path, seed=11, edge=False)
    sf1, sep = runner.load_panel(tmp_path)
    report = runner.run_research(sf1, sep, label="noise")

    assert abs(report.result.mean_ic) < 0.05                 # ~no information
    assert report.result.deflated_sharpe_ratio < runner.DSR_CUTOFF   # fails deflation
    assert report.deployable is False                        # NOT deployable (default-deny)


def test_degenerate_panel_fails_closed(tmp_path):
    # Far too little history → fewer than the learner's 8-date floor → fail closed.
    runner.write_synthetic_csvs(tmp_path, seed=1, edge=True, n_years=1)
    sf1, sep = runner.load_panel(tmp_path)
    report = runner.run_research(sf1, sep, label="degenerate")

    assert report.deployable is False
    assert report.result.leakage_flags == ["insufficient_data"]
    assert all(w == 0.0 for w in report.weights.values())


# ── PIT panel correctness propagates: dropping FUTURE filings cannot change the past ──
def test_pipeline_is_point_in_time(tmp_path):
    runner.write_synthetic_csvs(tmp_path, seed=7, edge=True)
    sf1, sep = runner.load_panel(tmp_path)
    cutoff = sf1["datekey"].quantile(0.6)
    masked = sf1[sf1["datekey"] <= cutoff]                   # hide everything filed after `cutoff`

    full = runner._feature_panels(sf1, sep, runner.DEFAULT_DIMENSION)["roe"]
    trimmed = runner._feature_panels(masked, sep, runner.DEFAULT_DIMENSION)["roe"]
    common_dates = [d for d in trimmed.index if d <= cutoff]
    a = full.reindex(index=common_dates, columns=trimmed.columns)
    b = trimmed.reindex(index=common_dates, columns=trimmed.columns)
    pd.testing.assert_frame_equal(a, b)                      # past features unchanged by future data


# ── CLI / selftest entrypoint ─────────────────────────────────────────────────────────
def test_selftest_main_returns_zero():
    assert runner.main(["--selftest"]) == 0


def test_main_requires_data_dir_without_selftest():
    assert runner.main([]) == 2                              # default-deny: no data, no run


def test_selftest_subprocess_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--selftest"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
