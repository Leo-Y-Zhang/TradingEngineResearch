"""THE TRIAL LEDGER — one machine-readable cumulative trial count.

WHY
===
The Deflated Sharpe Ratio deflates an observed Sharpe for the number of strategy
CONFIGURATIONS the programme has tried. That count is cumulative and monotone: every new
study raises the bar for all of them. It was maintained BY HAND, as prose, in
the internal research log and in each study's result markdown, and nothing
read it programmatically.

The predictable happened. `research/sleeves/_portfolio/portfolio_decision.json` deflated
against **n_trials = 38** when the ledger had already reached **47**, and a sibling module
in the same directory hard-coded **46**. Six different "cumulative" values were live in
the code at once: 9, 34, 36, 38, 46, 47.

THIS MODULE IS THE SOURCE OF TRUTH
==================================
`cumulative_trials()` returns the current count. `research.validation
.deflated_sharpe_ratio` and `research.multiasset.panel.dsr_sharpe_bar` both accept
``n_trials=None`` and read it from here, so a new study never has to know the number.

The recorded defaults of those two helpers are NOT changed: `dsr_sharpe_bar(n_trials=32)`
reproduces the programme's two pinned anchors (1.488 at 7 years, 0.597 at 40) and the
test suite asserts both. Changing a default would silently re-deflate every banked result.

FROZEN CONSTANTS
================
A study that has already run must keep the count it was EVALUATED at — that is history,
not a bug. `FROZEN_TRIAL_COUNTS` records every such hard-coded value in the tree, what
kind of number it is, and whether it was correct at the time.
`tests/test_trial_ledger.py` fails if any file hard-codes a trial count that is not
registered here, which is what stops a NEW study from inventing its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ANCHOR",
    "CHECKPOINTS",
    "CUMULATIVE_TRIALS",
    "FROZEN_TRIAL_COUNTS",
    "HISTORICAL",
    "LADDER",
    "PROGRAMME",
    "REPO",
    "SCANNED_ROOTS",
    "FrozenTrialCounts",
    "TrialCheckpoint",
    "cumulative_trials",
    "scan_hardcoded_trial_counts",
    "to_dict",
    "unregistered_trial_counts",
]

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrialCheckpoint:
    """One recorded movement of the cumulative count, with its citation."""

    cumulative: int
    study: str
    source: str
    note: str = ""


# The multi-asset programme's ledger, transcribed from the running log with the citation
# for each entry. Only movements that were RECORDED are listed; the count is the
# cumulative figure as at that entry, not a derivation from first principles.
CHECKPOINTS: tuple[TrialCheckpoint, ...] = (
    TrialCheckpoint(33, "PEAD re-test",
                    "internal research log:179",
                    "n_trials 32 -> 33"),
    TrialCheckpoint(36, "multi-asset carry",
                    "internal research log:643",
                    "n_trials 34 -> 36"),
    TrialCheckpoint(38, "low-volatility / quality re-test",
                    "internal research log:1023",
                    "36 registered + this study + a concurrent seasonality study"),
    TrialCheckpoint(46, "risk parity, iteration 11",
                    "internal research log:1742",
                    "n_trials 44 -> 46"),
    TrialCheckpoint(47, "breadth expansion, iteration 14",
                    "internal research log:1909",
                    "46 -> 47; derived at "
                    "research/multiasset/breadth_expansion_result.md:441 and re-affirmed "
                    "at research/sleeves/pair_deflation_result.md:11"),
)

#: the current cumulative count. Every NEW study must deflate against this.
CUMULATIVE_TRIALS: int = CHECKPOINTS[-1].cumulative


def cumulative_trials() -> int:
    """The programme's cumulative `n_trials`. The only number a new study should use."""
    return CUMULATIVE_TRIALS


# ── what a hard-coded number is allowed to BE ─────────────────────────────────
#: a claim about the programme's cumulative count at the time the study ran
PROGRAMME = "programme"
#: a fixed reproduction anchor for the DSR formula itself (n=32 at 7 and 40 years),
#: not a claim about how many configurations the programme has tried
ANCHOR = "anchor"
#: a declared sensitivity ladder across several counts
LADDER = "ladder"
#: belongs to the earlier, separate equity-alpha programme, which terminated at 23
HISTORICAL = "historical"


@dataclass(frozen=True)
class FrozenTrialCounts:
    """The trial counts one file is allowed to hard-code, and why."""

    kind: str
    counts: frozenset[int]
    note: str = ""
    #: True when the value did NOT match the ledger at the time the study ran
    stale: bool = False
    correct_value: int | None = None


def _f(kind: str, *counts: int, note: str = "", stale: bool = False,
       correct_value: int | None = None) -> FrozenTrialCounts:
    return FrozenTrialCounts(kind=kind, counts=frozenset(counts), note=note,
                             stale=stale, correct_value=correct_value)


# Every hard-coded trial count in the tree, registered. A file not listed here, or a
# count not listed for its file, fails `tests/test_trial_ledger.py`.
FROZEN_TRIAL_COUNTS: dict[str, FrozenTrialCounts] = {
    # ── the helpers themselves ────────────────────────────────────────────────
    "research/multiasset/panel.py": _f(
        ANCHOR, 32, note="the docstring's pinned anchors: 1.488 at 7y, 0.597 at 40y"),

    # ── THE DEFECT ────────────────────────────────────────────────────────────
    "research/sleeves/_portfolio/portfolio_decision.py": _f(
        PROGRAMME, 38, stale=True, correct_value=47,
        note="deflated against 38 when the ledger already stood at 47; its own "
             "combination-search count of 31 also undercounts a 189-234 configuration "
             "search. Frozen because the banked JSON was produced at these values"),

    # ── correct at the time they ran ──────────────────────────────────────────
    "research/sleeves/_portfolio/portfolio_correlation_v2.py": _f(
        PROGRAMME, 46, note="iteration 11's count, correct when that study ran"),
    "research/sleeves/lowvol_retest.py": _f(
        PROGRAMME, 38, note="36 registered + this study + a concurrent seasonality study"),
    "research/sleeves/multiasset_defensive.py": _f(
        PROGRAMME, 38, note="programme-cumulative at the time, per the mission brief"),
    "research/sleeves/breadth_ladder.py": _f(
        PROGRAMME, 46, note="iteration 11's count, correct when that study ran"),
    "scripts/run_multiasset_carry.py": _f(PROGRAMME, 36),
    "scripts/synthesise_carry_trend.py": _f(PROGRAMME, 36),
    "scripts/run_multiasset_trend.py": _f(
        PROGRAMME, 32, 36, note="declared anchor 32 and honest count 36, both reported"),
    "scripts/run_reversal_retest.py": _f(PROGRAMME, 34),
    "scripts/verify_multiasset_trend.py": _f(PROGRAMME, 36),

    # ── declared sensitivity ladders ──────────────────────────────────────────
    "research/sleeves/riskparity_run.py": _f(LADDER, 32, 46, 56, 304),
    "research/sleeves/multiasset_seasonal.py": _f(ANCHOR, 32),
    "scripts/synthesise_multiasset_longhistory.py": _f(LADDER, 32, 36),

    # ── reproduction anchors for the DSR formula ──────────────────────────────
    "research/sleeves/multiasset_defensive_run.py": _f(ANCHOR, 32),
    "research/sleeves/multiasset_defensive_verify.py": _f(ANCHOR, 32),
    "research/sleeves/multiasset_value_verify.py": _f(ANCHOR, 32),
    "scripts/build_multiasset_panel.py": _f(ANCHOR, 32),

    # ── forensics: deliberately reproduce the low-vol study's own count ───────
    "research/sleeves/_lowvol_verify/attack4_statistics.py": _f(
        PROGRAMME, 38, note="reproduces the published low-vol run bar for bar"),
    "research/sleeves/_lowvol_verify/attack5_structure.py": _f(
        PROGRAMME, 38, note="reproduces the published low-vol run bar for bar"),
    "research/sleeves/_pair_deflation/controls.py": _f(
        LADDER, 32, 38, 47,
        note="the only file that already carried the correct 47; it is the control that "
             "exposed portfolio_decision.json as stale"),

    # ── the earlier equity-alpha programme ────────────────────────────────────
    "scripts/research_insider_alpha.py": _f(
        HISTORICAL, 8, 9,
        note="equity-alpha programme, a separate ledger that terminated at 23"),
    "scripts/synthesise_breadth_sleeve_hunt.py": _f(
        HISTORICAL, 7, note="a years value on the same line, not a trial count"),
}

SCANNED_ROOTS: tuple[str, ...] = ("research", "scripts")

_CONSTANT = re.compile(r"^\s*(N_TRIALS[A-Z_]*)\s*=\s*(.+?)\s*(?:#.*)?$")
_KEYWORD = re.compile(r"n_trials\s*=\s*(\d+)")
_INTEGER = re.compile(r"\d+")


def scan_hardcoded_trial_counts(repo: Path = REPO) -> dict[str, set[int]]:
    """Every literal trial count hard-coded under `SCANNED_ROOTS`, by file.

    Catches both idioms: a module-level ``N_TRIALS*`` constant, and an ``n_trials=<int>``
    keyword passed straight to a deflation helper.
    """
    found: dict[str, set[int]] = {}
    for root in SCANNED_ROOTS:
        for path in sorted((repo / root).rglob("*.py")):
            relative = path.relative_to(repo).as_posix()
            if relative == "research/trial_ledger.py":
                continue                      # the registry itself
            counts: set[int] = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                constant = _CONSTANT.match(line)
                if constant:
                    counts.update(int(v) for v in _INTEGER.findall(constant.group(2)))
                counts.update(int(m.group(1)) for m in _KEYWORD.finditer(line))
            if counts:
                found[relative] = counts
    return found


@dataclass
class LedgerViolation:
    path: str
    counts: set[int] = field(default_factory=set)
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.path}: {sorted(self.counts)} ({self.reason})"


def unregistered_trial_counts(repo: Path = REPO) -> list[LedgerViolation]:
    """Files hard-coding a trial count that `FROZEN_TRIAL_COUNTS` does not authorise."""
    violations: list[LedgerViolation] = []
    for path, counts in scan_hardcoded_trial_counts(repo).items():
        frozen = FROZEN_TRIAL_COUNTS.get(path)
        if frozen is None:
            violations.append(LedgerViolation(
                path, counts,
                "not registered; new work must call trial_ledger.cumulative_trials()"))
            continue
        extra = counts - frozen.counts
        if extra:
            violations.append(LedgerViolation(
                path, extra, f"not among the registered {sorted(frozen.counts)}"))
    return violations


def to_dict() -> dict:
    """The whole ledger as plain data, for anything that wants it as JSON."""
    return {
        "cumulative_trials": CUMULATIVE_TRIALS,
        "checkpoints": [
            {"cumulative": c.cumulative, "study": c.study, "source": c.source,
             "note": c.note}
            for c in CHECKPOINTS
        ],
        "frozen": {
            path: {"kind": f.kind, "counts": sorted(f.counts), "note": f.note,
                   "stale": f.stale, "correct_value": f.correct_value}
            for path, f in sorted(FROZEN_TRIAL_COUNTS.items())
        },
    }
