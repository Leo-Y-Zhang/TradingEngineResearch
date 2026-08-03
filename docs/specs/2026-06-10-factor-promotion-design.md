# Factor-Promotion Pipeline — Design

**Date:** 2026-06-10 · **ROADMAP:** Phase 2, item 2 (`evaluate_factor` + `selection_rule`
promotion gating) · **Status:** approved (design), pre-implementation

## Purpose

Close the research→live factor-promotion loop. The pieces exist but are unconnected:
`evaluate_factor` (purged walk-forward validation → `ValidationResult`) and
`promote_factor` (gates: `selection_rule` + `sharpe_net>0` + |corr|<0.80 vs live +
cluster-diversity) are standalone, and the module-level `_LIVE_FACTOR_MATRIX` /
`_LIVE_FACTOR_NAMES` are **declared but never updated** — there is no orchestration
that runs candidates through evaluate→gate→promote and maintains the live library.
(`selection_rule` is already wired into `adaptive_weights` for sleeve weights; this
item is specifically the *factor* library.)

## Design (additions to `research/alpha_factory.py`)

- **`PromotionOutcome`** dataclass — `name`, `promoted: bool`, `passed_selection_rule: bool`,
  `result: ValidationResult`, `reason: str`.
- **`promote_candidates(candidates: dict[str, pd.Series], returns_df, costs_bps=5.0,
  splitter=None, regime_labels=None) -> dict[str, PromotionOutcome]`** — the end-to-end
  loop. For each candidate (in order): `evaluate_factor(...)` → `ValidationResult`;
  align the candidate to `returns_df.index`; `promote_factor(name, result,
  current_live_matrix, candidate_series)`. On approval, append the aligned column to
  `_LIVE_FACTOR_MATRIX` and the name to `_LIVE_FACTOR_NAMES`, so each later candidate is
  gated for correlation/diversity against the already-promoted set. Returns a per-name
  outcome with the rejection reason.
- **`get_live_factors() -> (list[str], np.ndarray)`** and **`reset_live_factors()`** —
  accessors that make the dead globals a real, inspectable library (matching the
  codebase's `get_*`/`reset_*` convention).

## Behaviour (the loop this closes)

- A strong, *distinct* factor (passes `selection_rule`, low correlation to live) →
  promoted and added to the library.
- A near-duplicate of a promoted factor (|corr| ≥ 0.80, or same cluster) → rejected.
- A weak factor (fails any of `selection_rule`'s six conditions) → rejected.

## Scope / caveats (YAGNI)

- The library accumulates within a consistent time index (one universe). A call whose
  index length differs from the existing library starts a fresh library — `promote_factor`
  already guards the length mismatch; documented, not engineered around now.
- No engine / per-cycle integration: factor promotion is a research-mode activity, not a
  trading-cycle step. (The per-cycle gate is the existing STEP-5 `apply_signal_health`.)

## Tests (TDD)

- Real `evaluate_factor` on random junk → not promoted, library empty (end-to-end path).
- (`evaluate_factor` stubbed to a passing result) distinct factor → promoted, in library;
  near-duplicate → rejected (correlation); second distinct factor → library grows to 2;
  a failing `ValidationResult` → not promoted; `reset_live_factors()` clears the library.
