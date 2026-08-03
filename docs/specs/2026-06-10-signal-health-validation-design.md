# Validation-Driven Signal Health — Design

**Date:** 2026-06-10 · **ROADMAP:** Phase 3 (signal-health gates on a `ValidationResult`,
not this-cycle confidence) · **Status:** approved (design), pre-implementation

## Purpose

Make the engine's STEP-5 signal-health filter gate each sleeve on its **validated
quality** (a persistent `ValidationResult` that passed purged walk-forward +
`selection_rule`), rather than only this-cycle mean confidence. Aligns the per-cycle
signal pipeline with golden rule #5 ("no live promotion without validation").

## The gap

`apply_signal_health(signals, stability_score, validation=None)` already supports the
gate (returns weight 0 — disabled — when a `ValidationResult` is supplied and
`selection_rule` fails). But `engine._step5_signal_health` calls it with
`validation=None` and uses `stability = mean(confidence_proxy this cycle)` — so the
validated-quality gate is never applied.

## Design

- **`research/alpha_factory.py`** — a per-sleeve validation registry (same `get_*`/`reset_*`
  convention as the live factor library): `register_sleeve_validation(sleeve, result)`,
  `get_sleeve_validation(sleeve) -> ValidationResult | None`, `reset_sleeve_validation()`.
- **`engine._step5_signal_health`** — for each sleeve:
  - if a `ValidationResult` is registered → use its `stability_score` and pass it into
    `apply_signal_health` (so a sleeve that fails `selection_rule` is **disabled**);
  - else → fall back to the current behaviour (this-cycle mean confidence, `validation=None`).

## The one decision: fallback for un-validated sleeves

Hard-disabling every un-validated sleeve would zero the signal pipeline, so the gate is
**validation-driven when a result is available, soft-default otherwise** — sleeves get
gated as they are validated (by `adaptive_weights` / research runs that register results),
with **no regression today**.

## Tests (TDD)

- registry `register`/`get`/`reset`.
- STEP 5 **disables** a sleeve carrying a registered *failing* `ValidationResult` (its
  symbols contribute nothing); a second, un-validated sleeve still contributes.
- STEP 5 keeps a sleeve with a *passing* registered `ValidationResult` (regression guard).

## Scope (YAGNI)

Results are registered out-of-band by validation runs; this item only makes STEP 5
*honor* them. No new in-cycle validation computation; backward-compatible default.
