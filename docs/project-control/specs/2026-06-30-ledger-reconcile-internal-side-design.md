# Spec — §17 reconciliation: ledger replay as the authoritative internal side

| Field | Value |
|-------|-------|
| **Purpose** | Make the LIVE `_reconcile` compare the broker against the directive's §17 **authoritative internal-event-ledger** (`replay_ledger_to_positions`), not the lossy weight-derived `current_book`. The reconciliation finish-line of the Phase-3 ledger work. |
| **Owner / role** | Project owner (accepts residual) |
| **Status** | ✅ IMPLEMENTED + adversarially reviewed (2026-06-30). Commits `d9e36e1` (change) + `b532dec` (review fix); 881 tests pass / 1 skip. A 3-lens review unanimously caught a P1 ordering regression (fixed). LIVE stays disabled. |
| **Last verified** | 2026-06-30 |
| **Source evidence** | `ops/run_loop.py::_reconcile` (current weight-derived internal side), `ops/ledger.py:210-232` (`replay_ledger_to_positions`), `ops/reconciliation.py` (`reconcile`), existing recon tests in `tests/test_run_loop.py:243-276,418-428`. |
| **Directive refs** | §2 incumbent-first · §7.5 surface-not-correct · §9 reconstruction · §17 three-way reconciliation (internal event ledger ↔ broker ↔ bank). |

## 1. Problem

`_reconcile` builds the internal position side as `expected_shares = round(weight × nav /
price)` from the engine's weight book `current_book`, then reconciles it against broker
positions. That internal side is a **lossy proxy** (weight→share conversion + rounding +
dependence on broker NAV/prices), and it is **not the directive's §17 leg-1** (the immutable
event ledger). The authoritative internal book is `replay_ledger_to_positions(self.ledger)` —
the signed sum of every recorded `FILL` — which already exists and is tested.

## 2. Decision — REPLACE (operator-confirmed: flat start)

The account starts **flat** (cash only; the engine builds every position via its own orders,
each recorded as a signed `FILL`). So `replay_ledger_to_positions` is **authoritative by
construction**, and `current_book` ≈ the ledger replay in production (same fills). Therefore:

- **Replace** the internal side of `_reconcile` with `replay_ledger_to_positions(self.ledger)`
  (signed shares), compared **directly** against broker positions (shares vs shares — no
  nav/price conversion, no rounding). Not *augment*: a second weight-derived check would be
  redundant and would false-break wherever the two internal views momentarily differ.

## 3. The change (`ops/run_loop.py::_reconcile`)

- Internal positions side := `replay_ledger_to_positions(self.ledger)`.
- Drop the now-unused weight-derived `expected_shares` computation and the unused `prices`
  parameter (update the single call site `self._reconcile(asof, prices)` → `self._reconcile(asof)`).
- **Unchanged:** no-broker no-op (RESEARCH), paper no-op (`is_paper`), broker-`account_state`
  failure → AMBER, `peak_nav` tracking, and **surfacing-only** behaviour (the incumbent
  never-auto-apply rule stands — directive §7.5). LIVE break stays RED via `to_alert`.
- **Scope: positions only.** Cash/NAV reconciliation is deferred — the ledger cash leg
  (`replay_ledger_to_balances`) needs real `COMMISSION/FEE` events recorded first (separate
  Phase-3 work), else it would false-break by accumulated commissions.

## 4. Flat-start assumption (documented)

`replay_ledger_to_positions` is the authoritative internal book **only because the deployment
starts flat** and every position change is a recorded signed `FILL`. A future non-flat
deployment (positions transferred/inherited) MUST first seed an audited one-time
opening-position event into the ledger, or the replay will under-report. This assumption is
stated in `_reconcile` and here.

## 5. Error handling / safety

- Surfacing-only (never auto-corrects the book) — unchanged.
- `replay_ledger_to_positions` is fail-safe: a `FILL` lacking `signed_qty` is skipped with a
  warning (under-report, not mis-sign) — so a malformed event makes the internal side *more*
  conservative (a break), never a silent wrong-direction reconciliation.
- No-broker / paper / broker-failure paths unchanged.

## 6. Testing (golden-master + new)

- **Update** the 2 existing recon tests (`test_reconciliation_surfaces_divergence_alert`,
  `test_no_reconciliation_when_aligned`) to seed signed `FILL`s in the ledger so the internal
  side reflects the held position — the deliberate, reviewable behaviour change (golden-master).
- **New:** (a) clean — ledger replay matches broker → no alert; (b) divergence — ledger replay
  ≠ broker → break surfaced; (c) **missing-fill** — a position the broker holds but the ledger
  never recorded → the internal side under-reports → break (the audit gap §17 exists to catch);
  (d) paper still no-op; (e) the `prices` param removal does not break the call site.
- Then a focused adversarial review (safety surface).

## 7. Incumbent-vs-Candidate decision record (§2.7)

- **Component**: `_reconcile` internal position source.
- **Incumbent**: weight-derived `round(weight × nav / price)` from `current_book` (lossy proxy,
  not the §17 event ledger).
- **Candidate**: `replay_ledger_to_positions(self.ledger)` (exact signed shares, the §17 leg-1).
- **Decision**: **REPLACE** — exact + authoritative + directive-canonical; preserves all guards
  and the surface-only rule. Valid under the operator-confirmed flat-start model.
- **Rollback**: revert the change; `_reconcile` returns to the weight-derived internal side.
- **Residual**: cash/NAV reconciliation (needs `COMMISSION/FEE` recording); the bank/accounting
  third leg; the non-flat-start opening-position seed.

## 8. Adversarial review (§21/§22) — 1 P1 found + fixed

A 3-lens review (reconciliation-correctness, safety-regression, directive-incumbent) produced 5
findings; **3 confirmed** (all the SAME defect, found independently by every lens; 2 refuted):

- **P1 ordering regression (`b532dec`)** — the first cut (`d9e36e1`) read the ledger replay in
  `_reconcile` *before* `record_cycle` appended **this cycle's** FILLs, while the broker already
  reflected those executions. So the internal side lagged the broker by exactly this cycle's fills →
  **every trading cycle false-broke** (RED in LIVE) and a genuine divergence was masked by the lag —
  nullifying the §17 control the moment LIVE runs. (The old weight-derived `current_book` was set from
  `achieved_weights` *before* `_reconcile`, so it was time-aligned; the swap moved the internal side
  one cycle too early.) **Fixed** by reordering `_run_once_impl`: record this cycle's FILL/POSITION to
  the ledger first (`record_cycle` without the recon alert), then `_reconcile` (replay now reflects
  this cycle), then append the `RECONCILIATION` audit event separately — all still fail-soft, ledger
  event order unchanged. Two regression tests added (fill-aligned cycle is clean; a genuine divergence
  still surfaces). Lesson: the existing recon tests pre-seeded the ledger and used a fill-less fake
  engine, so the production reconcile-before-record ordering was never exercised — now it is.
