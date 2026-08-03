"""THE TRIAL LEDGER.

The Deflated Sharpe deflates for the number of strategy CONFIGURATIONS the programme has
tried. That count was maintained by hand, in prose, in the running log and in each
study's result markdown — and nothing read it programmatically. So
`research/sleeves/_portfolio/portfolio_decision.json` deflated against **38** while the
ledger already stood at **47**, and a sibling module in the same directory hard-coded
**46**. Six different "cumulative" values were live at once: 9, 34, 36, 38, 46, 47.

The guard below fails if any file hard-codes a trial count that is not registered in
`research.trial_ledger.FROZEN_TRIAL_COUNTS`. A study that has already run keeps the count
it was EVALUATED at — that is history. A NEW study cannot invent one: it must call
`cumulative_trials()` or pass ``n_trials=None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.multiasset.panel import dsr_sharpe_bar
from research.trial_ledger import (
    CHECKPOINTS,
    CUMULATIVE_TRIALS,
    FROZEN_TRIAL_COUNTS,
    PROGRAMME,
    cumulative_trials,
    scan_hardcoded_trial_counts,
    to_dict,
    unregistered_trial_counts,
)
from research.validation import deflated_sharpe_ratio


class TestTheLedgerItself:

    def test_the_cumulative_count_is_the_last_recorded_checkpoint(self):
        assert CUMULATIVE_TRIALS == CHECKPOINTS[-1].cumulative
        assert cumulative_trials() == CUMULATIVE_TRIALS

    def test_the_count_is_47(self):
        """Pinned to the citation: internal research log:1909, 46 -> 47 at iteration 14."""
        assert CUMULATIVE_TRIALS == 47

    def test_the_ledger_never_goes_backwards(self):
        counts = [c.cumulative for c in CHECKPOINTS]
        assert counts == sorted(counts)
        assert len(set(counts)) == len(counts)

    def test_every_checkpoint_carries_a_citation(self):
        for checkpoint in CHECKPOINTS:
            assert checkpoint.source and ":" in checkpoint.source, checkpoint
            assert checkpoint.study

    def test_the_ledger_serialises(self):
        payload = to_dict()
        assert payload["cumulative_trials"] == 47
        assert len(payload["checkpoints"]) == len(CHECKPOINTS)
        assert set(payload["frozen"]) == set(FROZEN_TRIAL_COUNTS)


class TestTheDeflationHelpersReadIt:

    def _returns(self) -> np.ndarray:
        return np.random.default_rng(0).normal(0.01, 0.04, 200)

    def test_dsr_sharpe_bar_reads_the_ledger(self):
        assert dsr_sharpe_bar(17.75, n_trials=None) == dsr_sharpe_bar(
            17.75, n_trials=CUMULATIVE_TRIALS)

    def test_deflated_sharpe_ratio_reads_the_ledger(self):
        r = self._returns()
        assert deflated_sharpe_ratio(r, n_trials=None) == deflated_sharpe_ratio(
            r, n_trials=CUMULATIVE_TRIALS)

    def test_the_ledger_bar_matches_the_value_the_controls_already_recorded(self):
        """`_pair_deflation/controls.json` independently recorded this number."""
        assert dsr_sharpe_bar(17.75, n_trials=None) == pytest.approx(
            0.9443077723019092, abs=1e-12)

    def test_the_pinned_anchors_did_not_move(self):
        """Changing a helper DEFAULT would silently re-deflate every banked result."""
        assert dsr_sharpe_bar(7.0) == pytest.approx(1.488, abs=5e-4)
        assert dsr_sharpe_bar(40.0) == pytest.approx(0.597, abs=5e-4)
        assert deflated_sharpe_ratio(self._returns()) == deflated_sharpe_ratio(
            self._returns(), n_trials=1)

    def test_the_ledger_bar_is_higher_than_the_stale_one(self):
        """The whole point: deflating at 38 understates the bar a study must clear."""
        stale = dsr_sharpe_bar(17.75, n_trials=38)
        honest = dsr_sharpe_bar(17.75, n_trials=None)
        assert honest > stale
        assert honest - stale == pytest.approx(0.0209, abs=5e-4)


class TestNoStudyMayHardcodeATrialCount:
    """THE GUARD."""

    def test_every_hardcoded_count_is_registered(self):
        violations = unregistered_trial_counts()
        assert not violations, (
            "unregistered trial counts -- call trial_ledger.cumulative_trials() or "
            "pass n_trials=None instead: " + "; ".join(str(v) for v in violations))

    def test_the_scan_actually_finds_things(self):
        """A guard that matches nothing guards nothing."""
        found = scan_hardcoded_trial_counts()
        assert len(found) >= 15
        assert "research/sleeves/_portfolio/portfolio_decision.py" in found

    def test_an_unregistered_count_is_caught(self, tmp_path):
        """Mutation check: a brand-new study hard-coding a count must fail."""
        (tmp_path / "research").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "research" / "new_study.py").write_text(
            "N_TRIALS = 51\n", encoding="utf-8")
        violations = unregistered_trial_counts(tmp_path)
        assert [v.path for v in violations] == ["research/new_study.py"]
        assert violations[0].counts == {51}

    def test_a_new_count_in_a_registered_file_is_caught(self, tmp_path):
        path = "research/sleeves/lowvol_retest.py"
        (tmp_path / "research" / "sleeves").mkdir(parents=True)
        (tmp_path / "scripts").mkdir()
        (tmp_path / path).write_text("N_TRIALS = 38\nBAR = f(n_trials=99)\n",
                                     encoding="utf-8")
        violations = unregistered_trial_counts(tmp_path)
        assert len(violations) == 1
        assert violations[0].counts == {99}, "38 is registered; 99 is not"


class TestTheStaleEntryIsDeclared:

    def test_portfolio_decision_is_registered_as_stale(self):
        entry = FROZEN_TRIAL_COUNTS["research/sleeves/_portfolio/portfolio_decision.py"]
        assert entry.stale is True
        assert entry.correct_value == CUMULATIVE_TRIALS
        assert entry.counts == frozenset({38})

    def test_it_is_the_only_declared_stale_entry(self):
        stale = {p for p, f in FROZEN_TRIAL_COUNTS.items() if f.stale}
        assert stale == {"research/sleeves/_portfolio/portfolio_decision.py"}

    def test_no_programme_entry_claims_more_than_the_ledger(self):
        """A study cannot have been evaluated at a count the ledger never reached."""
        for path, entry in FROZEN_TRIAL_COUNTS.items():
            if entry.kind != PROGRAMME:
                continue
            assert max(entry.counts) <= CUMULATIVE_TRIALS, path
