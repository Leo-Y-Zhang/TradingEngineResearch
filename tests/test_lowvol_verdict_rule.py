"""The pre-committed promotion rule for the low-vol sleeve.

``verdict_for`` implements section 6 of ``lowvol_retest_prereg.md``: it is the thing
that decides whether a sleeve is promoted, and it is pure dict logic, so there is no
excuse for it being untested. A rule that quietly loosened -- promoting on three gates
instead of four, or reading the realistic bound where it should read the conservative
one -- would not fail anything. It would just start saying PROMOTE.

Every branch is exercised, and each test states which registered clause it pins.
"""

from __future__ import annotations

from typing import Any

import pytest

from research.sleeves.lowvol_retest import overall_verdict, verdict_for

GATES = ("gate_excess_pass", "gate_tstat_pass", "gate_dsr_bar_pass",
         "gate_beats_benchmark_dsr")


def _evaluated(*, conservative: dict[str, bool] | None = None,
               realistic: dict[str, bool] | None = None,
               realistic_vm: float = 0.05) -> dict[str, Any]:
    """The slice of an evaluate_band result that the decision rule actually reads."""
    cons = dict.fromkeys(GATES, False)
    cons.update(conservative or {})
    real = dict.fromkeys(GATES, False)
    real.update(realistic or {})
    return {
        "bounds": {
            "conservative": cons,
            "realistic": {**real,
                          "vol_matched": {"vol_matched_active_annual": realistic_vm}},
        }
    }


# ── PROMOTE ───────────────────────────────────────────────────────────────────

def test_promote_requires_all_four_conservative_gates():
    assert verdict_for(_evaluated(conservative=dict.fromkeys(GATES, True))) == "PROMOTE"


@pytest.mark.parametrize("dropped", GATES)
def test_dropping_any_single_conservative_gate_prevents_promotion(dropped):
    """Four gates means four, not three. Each one is load-bearing on its own."""
    gates = dict.fromkeys(GATES, True)
    gates[dropped] = False

    assert verdict_for(_evaluated(conservative=gates)) != "PROMOTE"


def test_promotion_is_decided_on_the_conservative_bound_not_the_realistic_one():
    """The honest-costing discipline: passing everything on the optimistic cost bound
    while failing the conservative one must NOT promote."""
    verdict = verdict_for(_evaluated(
        conservative=dict.fromkeys(GATES, False),
        realistic=dict.fromkeys(GATES, True),
        realistic_vm=0.10,
    ))

    assert verdict != "PROMOTE"


# ── MARGINAL ──────────────────────────────────────────────────────────────────

def test_excess_and_tstat_alone_are_marginal_not_promoted():
    verdict = verdict_for(_evaluated(
        conservative={"gate_excess_pass": True, "gate_tstat_pass": True}))

    assert verdict == "MARGINAL"


def test_marginal_is_reached_before_the_dead_test_so_a_negative_realistic_book_stays_marginal():
    """Clause ordering, pinned deliberately.

    MARGINAL is checked BEFORE the realistic vol-matched test, so a book clearing
    excess and t-stat on the conservative bound is MARGINAL even when the realistic
    bound's vol-matched active is negative. Reordering these clauses would silently
    reclassify books as DEAD.
    """
    verdict = verdict_for(_evaluated(
        conservative={"gate_excess_pass": True, "gate_tstat_pass": True},
        realistic_vm=-0.03,
    ))

    assert verdict == "MARGINAL"


# ── DEAD and UNDETERMINED ─────────────────────────────────────────────────────

def test_a_non_positive_realistic_vol_matched_active_is_dead():
    assert verdict_for(_evaluated(realistic_vm=-0.01)) == "DEAD"
    assert verdict_for(_evaluated(realistic_vm=0.0)) == "DEAD", "zero is not an edge"


def test_excess_on_either_bound_with_a_positive_realistic_book_is_undetermined():
    assert verdict_for(_evaluated(conservative={"gate_excess_pass": True},
                                  realistic_vm=0.02)) == "UNDETERMINED"
    assert verdict_for(_evaluated(realistic={"gate_excess_pass": True},
                                  realistic_vm=0.02)) == "UNDETERMINED"


def test_nothing_passing_is_dead():
    assert verdict_for(_evaluated()) == "DEAD"


def test_a_missing_vol_matched_figure_does_not_read_as_a_dead_book():
    """`.get` with a NaN default: an absent measurement must not compare as <= 0."""
    ev = _evaluated(conservative={"gate_excess_pass": True})
    ev["bounds"]["realistic"]["vol_matched"] = {}

    assert verdict_for(ev) == "UNDETERMINED"


# ── overall_verdict ───────────────────────────────────────────────────────────

def test_the_best_verdict_on_the_capacity_curve_is_the_one_reported():
    promote = _evaluated(conservative=dict.fromkeys(GATES, True))
    marginal = _evaluated(conservative={"gate_excess_pass": True, "gate_tstat_pass": True})
    dead = _evaluated(realistic_vm=-0.01)

    assert overall_verdict([dead, marginal, promote]) == "PROMOTE"
    assert overall_verdict([dead, marginal]) == "MARGINAL"
    assert overall_verdict([dead, dead]) == "DEAD"


def test_the_registered_ranking_is_promote_marginal_undetermined_dead():
    undetermined = _evaluated(conservative={"gate_excess_pass": True}, realistic_vm=0.02)
    dead = _evaluated(realistic_vm=-0.01)

    assert overall_verdict([dead, undetermined]) == "UNDETERMINED"


def test_an_empty_capacity_curve_is_dead_rather_than_an_error():
    assert overall_verdict([]) == "DEAD"
