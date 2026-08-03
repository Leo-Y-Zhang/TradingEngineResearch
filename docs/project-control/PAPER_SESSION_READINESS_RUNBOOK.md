# Runbook — Supervised IBKR Paper-Account Session (paper/shadow readiness gate)

| Field | Value |
|-------|-------|
| **Purpose** | The operator procedure to validate TradingEngineResearch's real order-submission / reconnect / reconciliation machinery against an **IBKR PAPER (simulated) account** — the things that cannot be validated without a real gateway. This is the verification package + promotion gate for **paper readiness**. |
| **Owner / role** | Project owner / operator (runs it) |
| **Status** | READY TO RUN — needs an operator + a running IBKR TWS/Gateway in PAPER mode. Not yet executed. |
| **Last updated** | 2026-06-30 |

## 0. Safety framing — READ FIRST

- This session connects TradingEngineResearch's **`IBKRBroker`** (the real order path) to an **IBKR PAPER
  account** (a *simulated* account — no real money). Because `IBKRBroker` is only built in `mode=LIVE`
  (`core/config.py::make_broker`), the session runs in **`mode=LIVE` pointed at the IBKR paper
  gateway port**. The account MUST be a paper account — verify this (step 4a) before any order.
- **Real-money LIVE stays DISABLED.** Promoting to a real-money account is a SEPARATE, later,
  explicitly-approved gate. Nothing in this runbook authorises it.
- Transmitting orders — **even to a paper account** — is a standing stop-condition: it requires a
  deliberate operator action. This runbook is that procedure; it does not auto-enable anything.
- **Abort immediately** (disconnect the gateway, leave LIVE off) on ANY P0 in §5. Capital
  preservation and correctness outrank completing the checklist.

## 1. Pre-flight gates (ALL must pass before connecting a gateway)

Run from the repository root:

1. **Suite green** — `python -m pytest tests/` → expect all green (a few environment-dependent skips; the pass count grows over time).
2. **Types + lint** — `python -m mypy` → clean; `python -m ruff check core data execution ops
   broker research strategies learning` → clean.
3. **No-live-path safety net** — `python -m pytest tests/test_safety_no_live_path.py -q` → all pass.
   This is the safety net: no config/code path reaches a real account unless mode is LIVE *and* armed.
   **Treat any failure here as P0 — do not proceed.**
4. **Clean tree** — `git status` clean; HEAD = the pushed commit; GitHub is the source of truth.
5. **No secrets committed** — confirm `secrets/`, `.env*`, vault are gitignored and absent from the
   tree. The IBKR account id + vault passphrase are supplied via env/vault at runtime, never committed.

## 2. Environment setup — pick a broker

Two providers, selected by `ENGINE_BROKER__PROVIDER`. **Alpaca (§2A) is recommended for
convenience** (no desktop gateway, instant free paper signup); IBKR (§2B) is the production target.
Either way TradingEngineResearch runs in `mode=LIVE` (the real order path) against a **paper/simulated**
account — `confirm_live=true` + `audit_log_path` are mandatory, and the on-demand `POST /cycle/run`
is **refused (403)** in LIVE (cycles fire only on the scheduled `run_forever` loop; observe via the
read-only endpoints in §3). The control API binds **loopback** (`127.0.0.1`) by default — keep it so.

Common config (env or a gitignored `.env`; `ENGINE_` prefix, nested via `__`):
```dotenv
ENGINE_MODE=LIVE
ENGINE_CONFIRM_LIVE=true
ENGINE_AUDIT_LOG_PATH=_run\audit.log
ENGINE_PERSISTENCE__STATE_DIR=_run\state
ENGINE_UNIVERSE=["AAPL"]        # JSON list (pydantic-settings parses list env vars as JSON)
ENGINE_CAPITAL_GBP=100000
ENGINE_CYCLE_INTERVAL_SECONDS=300
ENGINE_VAULT__PASSPHRASE=<a-local-passphrase>   # required to open the vault in LIVE
```

### §2A. Alpaca (recommended — free, no gateway)
1. Sign up free at **alpaca.markets**, switch to **Paper Trading**, and generate **paper API keys**.
2. `pip install "alpaca-py>=0.30,<1"` (or `pip install -e ".[brokers]"`).
3. Add to `.env`:
   ```dotenv
   ENGINE_BROKER__PROVIDER=alpaca
   ENGINE_BROKER__ALPACA_KEY_ID=<paper key id>
   ENGINE_BROKER__ALPACA_SECRET_KEY=<paper secret>
   ```
   (or store the vault secrets `alpaca_key_id` / `alpaca_secret_key`). The adapter uses the **paper
   endpoint** (`paper=True`) — even in `mode=LIVE` it never reaches real money; a real-money Alpaca
   account is a separate, later, explicitly-configured gate.

### §2B. IBKR (production target — needs a gateway)
1. Run **IB Gateway or TWS** logged into the **paper** account, API enabled (trusted IP `127.0.0.1`,
   read-write socket). Paper ports: **7497** (TWS) / **4002** (Gateway).
2. `pip install "ib-insync>=0.9.86,<1"`.
3. Add to `.env`:
   ```dotenv
   ENGINE_BROKER__PROVIDER=ibkr        # default
   ENGINE_BROKER__HOST=127.0.0.1
   ENGINE_BROKER__PORT=7497            # 7497 TWS paper / 4002 Gateway paper
   ENGINE_BROKER__ACCOUNT_ID=<paper account id>   # or vault secret 'ibkr_account_id'
   ```

### Pre-flight + launch
```powershell
cd <repo-root>
python scripts\broker_preflight.py      # READ-ONLY: connect, print account+positions, disconnect
python -m ops.run_loop --serve          # scheduled loop + control API (127.0.0.1:8000)
```
`broker_preflight.py` confirms the wiring (and prints the account id to verify it is the **paper**
account) without placing any order. The service refuses to start LIVE if the broker can't connect.

## 3. Read-only observation endpoints (`ops/api.py`)

`GET /status` (mode, broker_connected, kill_switch_latched, **open_reconciliations**) ·
`GET /book` (current_book / achieved) · `GET /monitoring` (snapshot + alerts) ·
`GET /cycle/latest` · `GET /metrics`. The **default session is read-only**; the only state-changing
endpoints are the two operator actions below, both **refused in LIVE on this unauthenticated API**:
`POST /kill-switch/reset` and `POST /reconciliation/resolve` — for LIVE they must be a deliberate,
authenticated console action (auth hardening is the Phase-8 follow-up).

## 4. Validation checklist (action → expected evidence)

> Start READ-ONLY. Only enable the scheduled loop once 4a passes.

**4a. Connectivity + account validation.** Start the service; confirm `GET /status`
`broker_connected=true`. **Verify the connected account is the PAPER account** and the base currency /
permissions / asset scope match the approved scope. A **paper/live or account mismatch is a HARD
STOP** — abort. Confirm LIVE start fails closed if the broker is not connected (it should refuse to
enter the loop disconnected).

**4b. Client-order-id round-trip (LIVE6B-3 — the one unverified broker assumption).** Let the loop
place one small child order. Confirm our economic order_id (`asof|symbol|side|slice_index`)
round-trips as the broker's client tag — **IBKR `orderRef`** (in TWS/Gateway) or **Alpaca
`client_order_id`** (`GET /orders` in the Alpaca dashboard / API) — and that we captured the broker's
own order id as `broker_order_id` (visible after ack/fill). This is the key the resync keys on; it
MUST round-trip.

**4c. Order → fill → ledger → reconcile.** After a fill: confirm the immutable ledger
(`<state_dir>/ledger.jsonl`) has a signed `FILL` with the **real `avgFillPrice`** (not an estimate,
i.e. `price_estimated` absent/false), and that
`GET /status` shows **no reconciliation break** — i.e. `replay_ledger_to_positions` (the internal
event-ledger book) matches IBKR positions. A persistent break here is a P0.

**4d. Disconnect → reconnect → resync.** Interrupt the broker connection briefly while an order is in
flight — IBKR: kill the gateway; Alpaca (REST): drop the network / temporarily invalidate the keys
(the gateway-kill case is cleanest on IBKR). Expect: the order goes **SUBMISSION_UNCERTAIN /
BROKER_UNKNOWN** (NOT rejected); no blind
resubmit; `live_submits_blocked` engages; an AMBER `resync_failed` alert. On reconnect, the per-cycle
resync runs (READ-ONLY `open_orders`). If the broker filled more than we locally booked (a
disconnect-fill), expect an **OPEN reconciliation item** in `GET /status` and the symbol **frozen**
(no new orders on it).

**4e. Operator reconciliation resolve.** For a discovered disconnect-fill, resolve it via the
authenticated console (`resolve_reconciliation`, `decision=ACCEPT`). Confirm: an explicit audited
`FILL` (`source=RESYNC_RECONCILED`, booked at the broker `avgFillPrice`), `current_book` updated, the
order advanced out of `RECONCILIATION_HOLD`, the symbol **unfrozen**, and the item CLOSED. For a
spurious/duplicate broker report, confirm `decision=REJECT` cancels it out and books nothing.

**4f. Kill-switch latch (kill-switch latch).** Engage a hard stop (e.g. a drawdown/limit breach in the
controlled config). Confirm: the latch engages, a RED alert, **every subsequent cycle is halted** (no
orders), the latch **persists across restart**, and only an authenticated operator
`reset_kill_switch` clears it (the unauthenticated API refuses it in LIVE).

**4g. Restart persistence (LIVE6B-2).** Stop and restart the service. Confirm open orders, **open
reconciliation items**, the kill-latch, the book, and peak-NAV all survive; `_needs_resync` forces a
resync before any new submit; `live_submits_blocked` stays on until a clean resync.

## 5. Pass / fail / abort

- **PASS** = 4a–4g all observed as specified, the immutable ledger reconciles against IBKR, and no
  unexplained order/position/cash behaviour.
- **ABORT (disconnect, LIVE off)** on any P0: account/paper mismatch (4a); a persistent
  ledger-vs-broker break (4c); a lost or duplicated fill; a resubmit in unknown state; any new order
  while `live_submits_blocked`; the kill-switch failing to halt; or any `test_safety_no_live_path`
  regression.
- Save evidence (ledger excerpts, `/status` + `/monitoring` snapshots, TWS screenshots) as the
  verification package. Record the outcome by updating this runbook's **Status** row and a dated
  entry in the project's promotion-decision record.

## 6. Sign-off + what this does and does NOT authorise

- A PASS validates **paper readiness** only. **Shadow**, **limited-live**, and **real-money LIVE** are
  separate gates each needing their own evidence + explicit operator (and, where relevant,
  professional) approval. Real-money LIVE stays DISABLED.
- Record: operator, date, gateway/account (paper), config version, commit, evidence location, residual
  risks (e.g. the **cash/NAV** legs are not yet reconciled — positions only — pending
  `COMMISSION/FEE` recording; the bank/accounting third leg is future work).

## 7. Known residuals to watch during the session

- **Cash/NAV reconciliation is NOT wired** — only the positions leg is. Do not infer cash correctness
  from a clean positions reconciliation. (`COMMISSION/FEE` event recording + `replay_ledger_to_balances`
  wiring is the next reconciliation step.)
- **avgFillPrice live value is unvalidated** until 4c — the booking logic + fallback are unit-tested,
  but the real `orderStatus.avgFillPrice` round-trip is exactly what this session confirms.
- **Late commissions / corrections after FILLED** in the legacy `execution_engine` (EXEC-5) — the live
  path uses `order_lifecycle` which accepts late commissions; the legacy state machine is off the
  order-state path.
- **Alpaca path simplifications** — orders are MARKET (not the engine's passive-limit plan), and fills
  carry slippage 0, which pulls the in-run TCA cost coefficients down (per-process only — does not
  leak to IBKR). Run an Alpaca session against a **disposable state dir** and don't read its TCA
  priors as real. A rejected order resolves to REJECTED via the resync's recently-CLOSED query within
  a cycle (so an early rejection may briefly show a WORKING order in `/status` before it clears).
