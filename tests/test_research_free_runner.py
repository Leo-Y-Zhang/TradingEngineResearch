"""Tests for the FREE-data fundamental-alpha research runner
(``scripts/research_free_alpha.py``). Offline, deterministic, NO network, NO paid data.

Mirrors ``tests/test_research_sharadar_runner.py``: the runner must (1) recover an injected
real cross-sectional edge end-to-end (through the synthetic EDGAR PIT panel ->
``build_edgar_panel`` -> fundamental features -> forward returns -> learned, purged
walk-forward combination) and report DEPLOYABLE, and (2) DEFAULT-DENY pure noise so it can
never promote a non-edge (golden rule 5 / SIGNALS-5). A degenerate (too-short) panel must
fail closed. The ``--selftest`` CLI entrypoint is exercised both in-process and via
subprocess, and must exit 0. The live network path (``fetch_free_funds`` /
``fetch_free_prices``) is NEVER touched.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_free_alpha.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("research_free_alpha", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module          # so @dataclass can resolve module annotations
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# ── End-to-end edge recovery + noise rejection (paired contract) ──────────────────────
def test_recovers_real_edge_and_is_deployable():
    funds, prices = runner.build_synthetic_panel(seed=7, edge=True)
    report = runner.run_research(funds, prices, label="edge")

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


def test_noise_is_default_denied():
    funds, prices = runner.build_synthetic_panel(seed=11, edge=False)
    report = runner.run_research(funds, prices, label="noise")

    assert abs(report.result.mean_ic) < 0.05                 # ~no information
    assert report.result.deflated_sharpe_ratio < runner.DSR_CUTOFF   # fails deflation
    assert report.deployable is False                        # NOT deployable (default-deny)


def test_degenerate_panel_fails_closed():
    # Far too little history → fewer than the learner's 8-date floor → fail closed.
    funds, prices = runner.build_synthetic_panel(seed=1, edge=True, n_years=1)
    report = runner.run_research(funds, prices, label="degenerate")

    assert report.deployable is False
    assert report.result.leakage_flags == ["insufficient_data"]
    assert all(w == 0.0 for w in report.weights.values())


# ── Universe resolution ───────────────────────────────────────────────────────────────
def test_default_universe_is_a_moderate_breadth_list():
    universe = runner.load_universe()
    assert 120 <= len(universe) <= 160                       # broadened, sector-diversified breadth
    assert len(set(universe)) == len(universe)               # de-duplicated
    assert all(t == t.upper() for t in universe)


def test_tickers_override_and_dedup():
    universe = runner.load_universe(tickers="aapl, msft ,AAPL")
    assert universe == ["AAPL", "MSFT"]                       # upper-cased, de-duplicated, ordered


def test_universe_file_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("AAPL\n# a comment\n\nMSFT  # inline\nGOOGL\n", encoding="utf-8")
    assert runner.load_universe(universe_file=f) == ["AAPL", "MSFT", "GOOGL"]


# ── CLI / selftest entrypoint (NO network) ────────────────────────────────────────────
def test_selftest_main_returns_zero():
    assert runner.main(["--selftest"]) == 0


def test_selftest_subprocess_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--selftest"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout
    assert "SURVIVORSHIP CAVEAT" in proc.stdout              # the disclosure is unmissable


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
