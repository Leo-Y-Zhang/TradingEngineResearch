# Spec — Async-fill / held-book reconciliation flow

| Field | Value |
|-------|-------|
| **Purpose** | Close the one open correctness residual: a reconnect-resync-discovered disconnect-fill is surfaced but never booked, so the internal book + ledger replay under-report the true position and the symbol freezes with no resolution path. |
| **Owner / role** | Project owner (accepts residual / resolves items) |
| **Status** | ✅ IMPLEMENTED + adversarially reviewed (2026-06-30). 4 slices `b743a74`→`c4cf094`; 870 tests pass / 1 skip. A 4-lens review (13 findings, 7 confirmed) fixed 4 defects incl. a P1 crash-idempotency hole. LIVE stays disabled. |
| **Last verified** | 2026-06-30 |
| **Source evidence** | `execution/order_manager.py:163-174`, `execution/order_lifecycle.py:224-236`, `ops/run_loop.py:65-68,327-349,447-466,491-552,624-651`, `ops/ledger.py:163-232`, `core/engine/engine.py:883-924,956-1024`; Phase 6(c) residual note in `RISK_AND_DEFECT_REGISTER.md`. |
| **Directive refs** | §2 incumbent-first · §4 stop conditions · §7.5 no silent repair · §15 order lifecycle · §16 fail-closed for new risk · §17 reconciliation (visible/assigned/aged/explicitly resolved) · §22 conservative interpretation · §23 testing. |

## 1. Problem

When a LIVE reconnect resync calls `broker.open_orders()` and the broker reports a
`filled_qty` greater than what we locally booked (a fill that landed during a
disconnect), `OrderManager._map_broker_open_orders` calls `reconcile_broker_fill`
(bumps the lifecycle `filled_qty` so the pending-overlay residual stays correct) and
parks the order in `RECONCILIATION_HOLD`. Two gaps follow:

1. **The discovered quantity is booked nowhere.** It is recorded as neither a signed
   `FILL` ledger event (so `replay_ledger_to_positions` — the authoritative *internal*
   side of reconciliation — under-reports the real position) nor a `current_book`
   update (so the next cycle sizes against a book missing the disconnect-fill).
2. **No resolution path.** `RECONCILIATION_HOLD ∈ RESUBMIT_BLOCKING_STATES`, so the
   per-symbol no-stacking block (`engine.py:886-887`) freezes the symbol indefinitely.
   Today only manual state surgery can clear it.

The current behaviour is **safe** (the frozen symbol cannot be double-traded) but
**incomplete**: the internal book is knowingly diverged and a real fill sits in a
dead-end until manual intervention.

## 2. Policy decision (operator-gated, not auto-book)

The run-loop embodies a deliberate incumbent decision (`run_loop.py:65-68,:549`):
*"correcting the book from the broker is a deliberate risk decision … surfaced as an
alert, never auto-applied."* Auto-booking the discovered fill would cross that line.

Per incumbent-first (§2), conservative interpretation (§22), fail-closed for new risk
(§16), and visible/assigned/aged/explicitly-resolved (§17), the chosen design is
**operator-gated booking**:

- The discovered fill becomes a **first-class, durable, aged, audited OPEN
  reconciliation item** (surfaced, symbol stays frozen — fail-closed).
- An **explicit, reason-coded operator action** books it: an explicit audited ledger
  correction + a `current_book` update, then the order transitions out of
  `RECONCILIATION_HOLD` (FILLED / PARTIALLY_FILLED) which **unfreezes the symbol**.

This **HARDENS** the incumbent (adds the missing resolution path) without overriding
its no-auto-correct principle. It mirrors the existing kill-switch latch/reset
machinery exactly (durable state in `LoopState`, operator-only lock-serialised method,
ledger event, persist, refused-in-LIVE on the unauthenticated API, surfaced in
`status()`).

## 3. Components & interfaces

**`execution/order_lifecycle.py`**
- `OrderRecord.booked_qty: float = 0.0` — how much of `filled_qty` has been booked into
  the held book. Serialised in `to_json`/`from_json` (back-compat default 0.0).
- `OrderLifecycle.mark_booked(order_id, qty, timestamp)` — set `booked_qty`
  (clamped ≤ `filled_qty`), append history.
- Unbooked delta = `filled_qty − booked_qty` (the quantity an operator resolution books).

**`execution/order_manager.py`**
- `reconcile_open_orders(...)` returns `(changed, discovered)` where `discovered` is a
  list of `{order_id, symbol, side, delta_qty, broker_filled_qty, ref_price}` for orders
  whose unbooked delta > 0 (a disconnect-fill needing resolution). Detection stays in
  `_map_broker_open_orders` (already computes the over-fill); it now reports the delta
  instead of only parking.

**`core/engine/engine.py`**
- `resync_open_orders(...)` returns the enriched `discovered` list (only the run-loop
  calls it).
- `book_reconciled_fill(order_id, qty, timestamp)` — `lifecycle.mark_booked(...)` then
  transition the order out of `RECONCILIATION_HOLD`: → FILLED if `booked_qty ≥
  approved_qty − eps`, else → PARTIALLY_FILLED (still resting at broker). Idempotent.

**`ops/run_loop.py`**
- `LoopState.open_reconciliations: list[dict]` (durable; back-compat default `[]`). Each
  item: `{id, order_id, symbol, side, delta_qty, broker_filled_qty, ref_price, asof,
  status}` (`asof` for aging).
- `_maybe_resync`: after `resync_open_orders`, for each discovered delta not already an
  OPEN item (idempotent by `order_id` + `broker_filled_qty`), append an OPEN item + a
  `RECONCILIATION` ledger event (`kind="disconnect_fill_discovered"`) + an AMBER alert.
- `resolve_reconciliation(item_id, operator, reason, decision="ACCEPT", timestamp)` —
  operator-only, lock-serialised (mirrors `reset_kill_switch`):
  - **ACCEPT**: append explicit signed `FILL` event
    `{source:"RESYNC_RECONCILED", order_id, symbol, side, qty, signed_qty,
    fill_price: ref_price}` (so the existing replay functions count it) **before** any
    book mutation; if the append fails, abort and leave the item OPEN (no unaudited
    mutation). Then `current_book[symbol] += signed_delta_qty × ref_price / capital`;
    `engine.book_reconciled_fill(...)`; append a `RECONCILIATION` resolve event; mark
    item CLOSED; `_persist()`.
  - **REJECT** (spurious/duplicate broker report): close with reason, no booking, audited.
- `status()` adds `open_reconciliations` (OPEN items, with `asof` for aging).
- `_persist` serialises the new field (already snapshots `LoopState`).

**`ops/api.py`**
- `POST /reconciliation/{id}/resolve?operator=&reason=&decision=` — refused in LIVE on
  the unauthenticated API (mirror `/kill-switch/reset`); PAPER/SHADOW may resolve.
  `GET /status` already exposes the OPEN items.

## 4. Data flow

disconnect → next LIVE cycle `_maybe_resync` → `broker.open_orders()` reports
`filled_qty` > booked → order parked HOLD (symbol frozen) + OPEN reconciliation item
(ledger `RECONCILIATION` + `LoopState` + AMBER alert). Operator sees it in `/status`,
calls `/reconciliation/{id}/resolve` → explicit audited `FILL` correction + `current_book`
update + order → FILLED → symbol unblocks → item CLOSED.

## 5. Safety / error handling

- Booking ONLY on explicit operator action — no auto-mutation (respects the incumbent
  decision; §2/§22).
- Ledger append precedes the book mutation; append failure aborts the booking and leaves
  the item OPEN (no unaudited financial mutation; §7.5).
- `booked_qty` idempotency: a repeated resync or re-resolve never double-books.
- Symbol stays frozen (`RECONCILIATION_HOLD`) until resolution — fail-closed for new
  risk (§16); resumption is human-gated (§4).
- RESEARCH/PAPER untouched (resync is LIVE-only); no-discovery cycles leave the ledger +
  book byte-identical; `LoopState` back-compat (`open_reconciliations` default `[]`).

## 6. Testing (directive §23, three-layer verification)

- **Lifecycle**: `booked_qty` round-trips JSON; `mark_booked` clamps ≤ `filled_qty`;
  unbooked-delta arithmetic.
- **order_manager**: `_map_broker_open_orders` returns the discovered delta; returns
  none when `broker_filled == booked`.
- **engine**: `book_reconciled_fill` HOLD→FILLED (full) / →PARTIALLY_FILLED (partial);
  idempotent; resync returns discovered.
- **run_loop**: fake-broker disconnect-fill → OPEN item recorded (ledger `RECONCILIATION`
  + `status()`), symbol frozen; **resolve** → ledger `FILL[source=resync]` present,
  `replay_ledger_to_positions` includes it, `current_book` updated, order FILLED, a
  subsequent cycle can trade the symbol, item CLOSED; resolve unknown/closed rejected;
  re-resync no duplicate item; ledger-append failure leaves item OPEN.
- **api**: resolve refused in LIVE; works in PAPER/SHADOW; `status` surfaces OPEN items.
- **golden-master**: a normal (no-disconnect) run records a byte-identical ledger + book.

## 7. Slices (resumable; commit + push each)

1. **Lifecycle + manager + engine plumbing** — `booked_qty`, `mark_booked`,
   `reconcile_open_orders` returns `discovered`, `engine.resync_open_orders` returns
   `discovered` + `book_reconciled_fill`. Unit tests.
2. **Run-loop surfacing** — `LoopState.open_reconciliations`, `_maybe_resync` records the
   OPEN item + ledger `RECONCILIATION` + alert, `status()` + persistence. Tests.
3. **Operator resolution** — `resolve_reconciliation` (book + unfreeze + close) + api
   endpoint (refused in LIVE). Tests + golden-master.
4. **Adversarial review** (workflow, §21/§22) → fix findings; update
   `RISK_AND_DEFECT_REGISTER.md`.

## 8. Incumbent-vs-Candidate decision record (§2.7)

- **Component**: reconnect-resync handling of a discovered disconnect-fill.
- **Incumbent**: surface + park `RECONCILIATION_HOLD`, never resolve (dead-end; book
  knowingly diverged; symbol frozen until manual surgery).
- **Candidate**: surface as a first-class durable/aged/audited item + explicit
  operator-gated audited resolution that books it and unfreezes the symbol.
- **Decision**: **HARDEN** — adds the missing resolution path while preserving the
  incumbent's no-auto-correct principle (operator, not automation, applies the book
  change).
- **Rollback**: revert the slices; `RECONCILIATION_HOLD` reverts to today's dead-end
  (no capability lost vs current).
- **Residual**: a true three-way (Flex/exec) confirmation of the discovered fill is a
  later phase; this flow records the operator's reason-coded acceptance, not an
  independent broker-statement match.

## 9. Implementation notes (as-built deltas + adversarial review)

Built in 4 slices `b743a74`→`c4cf094`. The as-built differs from §3 in two honest ways:

- **Idempotency mechanism — stronger than designed.** §3 proposed `OrderRecord.booked_qty`
  for idempotency. The as-built does NOT add `booked_qty`: re-discovery is already prevented
  by `filled_qty` (a re-resync cannot re-discover once `filled_qty == broker_filled`), and
  re-resolution is anchored on the **durable immutable ledger** — the booking `FILL` carries
  `reconciliation_id = item_id`, and `resolve_reconciliation` skips the append if a matching
  `FILL` already exists. This is strictly more crash-safe than a `booked_qty` field (which
  the adversarial review showed could be lost in the same crash that orphans the FILL).
- **`decision=ACCEPT|REJECT`** is implemented as designed (REJECT cancels the parked order out
  via `engine.cancel_reconciled_order`, HOLD→CANCELLED, books nothing).

**Adversarial review (slice 4, §21/§22):** a 4-lens panel (order-state, ledger/accounting,
risk/fail-closed, directive-compliance) produced 13 findings; 7 confirmed real after
refute-by-default verification (6 refuted). Fixed:
- **P1 crash-idempotency** — the `FILL` is fsync'd before the item-CLOSED persist; a crash in
  that window + an operator re-resolve appended a second `FILL` → replay double-count. Fixed by
  the `reconciliation_id` ledger-anchored idempotency above.
- **P2 invalid-price guard** — a non-positive/non-finite `ref_price` would book the share qty
  for zero cost (free shares → permanent book/NAV divergence) then close the item unrecoverably.
  Now refused (AMBER + item left OPEN).
- **P2 REJECT branch** — restored from §3 (was dropped in the first cut).
- **P3 estimated-price** — the booking `FILL` is flagged `price_estimated`; the true-execution-price
  plumbing (broker `avgFillPrice`) is now done (residual (a) below).

**Remaining residuals:** (a) ✅ **DONE (2026-06-30)** — the booked `FILL` now uses the broker's TRUE
avg fill price when reported: `broker.open_orders()` entries carry an optional `avg_fill_price`
(IBKR populates it from `orderStatus.avgFillPrice`), plumbed through the discovered item +
reconciliation item; `resolve_reconciliation` prefers it (`price_estimated=false`) and falls back to
`ref_price` (`price_estimated=true`) only when the broker reports none. The IBKR live value is wired
but unvalidated against a real gateway — the supervised paper session covers it. (b) **Future phase
(§17 finish-line):** the live `_reconcile` still diffs broker vs `current_book`, not the ledger
replay; wiring `replay_ledger_to_positions` in as the authoritative internal side needs the ledger to
be a complete position record from a flat start (the existing recon tests pin the weight-derived
behavior), so it warrants its own focused design.
