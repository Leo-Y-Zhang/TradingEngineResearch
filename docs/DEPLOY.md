# TradingEngineResearch — Deployment

Reproducible deploy for the TradingEngineResearch v6 platform (ROADMAP Phase 6, item 6).

## Quick start (Docker Compose)

```bash
cp .env.example .env            # edit: set ENGINE_UNIVERSE, mode, cadence, …
docker compose up --build       # combined loop + API on http://127.0.0.1:8000
```

This runs the **combined** topology: the scheduled run-loop and the observation
API in **one process**, sharing one `EngineService` (and its lock), so there is
exactly one writer to the state directory and an API-triggered cycle can never
overlap a scheduled one.

Check it:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/status
curl -s http://127.0.0.1:8000/metrics/prometheus
```

## Run modes (`scripts/entrypoint.sh`)

| Mode | Command | What it does |
|------|---------|--------------|
| `combined` (default) | `python -m ops.run_loop --serve` | loop + API, one shared service |
| `loop` | `python -m ops.run_loop` | scheduled loop only, no HTTP |
| `api` | `uvicorn --no-proxy-headers --factory ops.api:create_app_from_settings` | API only (own service) |

> `--no-proxy-headers` is **security, not style** (SEC-8). uvicorn defaults
> `proxy_headers=True` with `forwarded_allow_ips=127.0.0.1`, so on a loopback bind
> its own middleware rewrites the client address from a caller-supplied
> `X-Forwarded-For` *before the application runs*. Measured on a real server: a
> request from `127.0.0.1` carrying `X-Forwarded-For: 203.0.113.99, 198.51.100.7`
> was recorded as `client_ip 198.51.100.7`, and a 4,000-character header put 4,000
> attacker-chosen characters in the trail. That bypassed the application's
> trusted-proxy control entirely and let a rotating header mint a fresh rate-limit
> budget per request. `python -m ops.run_loop --serve` sets the same flag in code
> (`ops.api.api_uvicorn_kwargs`). If a real reverse proxy fronts this API, name its
> header in `ENGINE_API_TRUSTED_PROXY_HEADER` and let the app honour it: the
> value is capped, and the true socket peer is still recorded alongside it.

> Do **not** run separate `loop` and `api` containers against the **same** state
> volume — two services would both write it. Use `combined`, or keep `api`
> on a different state dir for pure observation.

## Configuration

All config is environment-driven via `core.config.EngineSettings` (prefix
`ENGINE_`, nested with `__`). See `.env.example` for every key. Highlights:

- `ENGINE_MODE` — `RESEARCH` | `PAPER` | `LIVE`. **LIVE** additionally requires
  `ENGINE_CONFIRM_LIVE=true` and `ENGINE_AUDIT_LOG_PATH` (fail-closed), and a
  broker account id (`ENGINE_BROKER__ACCOUNT_ID` or the vault key
  `ibkr_account_id`). LIVE also needs the `brokers` extra in the image
  (add `,brokers` to the `pip install` extras in the `Dockerfile`).
- `ENGINE_UNIVERSE` — JSON array of symbols. An empty universe fails closed.
- `ENGINE_PERSISTENCE__BACKEND` — `json` (file, default) or `sqlite`
  (`SqlStateStore`; point `DATABASE_URL` at Postgres for the `--profile postgres`
  service). Requires the `persistence` extra (already in the image).

## Secrets

The encrypted vault lives in `secrets/` (gitignored, mounted read-only into the
container). Create/rotate it with the operator CLI:

```bash
python -m core.vault            # interactive create/set/rotate
```

Never bake secrets into the image or commit them; supply
`ENGINE_VAULT__PASSPHRASE` at runtime.

## Security posture

This section previously read "the API is **unauthenticated** by design at this
milestone" — that has been false since Phase 8 landed (2026-07-27) and is
corrected rather than deleted, because reading it was enough to conclude the
control surface was open by design.

- **Auth.** Set `ENGINE_API_TOKEN` and every data-exposing GET and every
  mutating POST requires `Authorization: Bearer <token>` (constant-time compare).
  `/health` stays open for liveness probes; `/` serves the dashboard HTML, whose
  client-side fetches will 401 — it is a loopback convenience, not an
  authenticated UI.
- **Bind.** `ops.api.assert_bind_is_safe` refuses at startup to serve off
  loopback without a token, so the unsafe combination never opens a socket. The
  compose file still publishes to **loopback only** (`127.0.0.1:8000`).
- **Rate limits.** Two budgets:
  `ENGINE_API_READ_RATE_LIMIT_PER_MINUTE` (default 240, the observation
  endpoints) and `ENGINE_API_WRITE_RATE_LIMIT_PER_MINUTE` (default 10,
  shared by `/cycle/run`, `/kill-switch/reset`, `/reconciliation/resolve`).
  Checked before the token, so token guessing is capped too. Over budget is a
  429 with `Retry-After`. `/health` is never throttled; 0 disables a budget.
  Memory is bounded by construction — an LRU capped at 4096 keys, O(1) lookup and
  eviction — so a flood of distinct callers cannot grow the limiter.
- **Whose budget (SEC-4).** The budget is keyed on the caller's **identity**, not
  their IP. In both recommended topologies every caller presents the *same*
  address (loopback, or the docker gateway behind a compose port map), so an
  IP-keyed budget was one global budget any anonymous caller could exhaust —
  measured: two anonymous 401s locked the token holder out. A request bearing the
  valid token is now keyed on a fingerprint of that token; everything else shares
  the anonymous budget for its address.
  `ENGINE_API_TRUSTED_PROXY_HEADER` (unset by default) names a proxy header
  to take the client address from. **Leave it unset unless the API is unreachable
  except through a proxy that sets that header itself** — it is caller-supplied,
  so trusting it otherwise lets anyone mint a fresh budget per request. The
  rightmost value is used (what the adjacent proxy saw).
- **Generated docs are gated (SEC-5).** `/openapi.json`, `/docs` and `/redoc`
  carry the same read budget and bearer token as `/status`. FastAPI mounts them
  unauthenticated by default, and `/openapi.json` enumerates every route and every
  parameter name — including the kill-switch `operator`/`reason` fields. They are
  gated rather than deleted: the operator still needs the reference, and a
  disabled-by-flag surface tends to come back on.
- **Telemetry.** The `tradingengineresearch.api.security` logger emits one structured event
  per request (request id, client IP, peer IP, route, outcome, duration) plus
  distinct `auth_failed` and `rate_limited` events. The token is never logged.
  Responses carry `X-Request-ID` for correlation.
- **Detection is PARTIAL, not a pass (SEC-6).** Those events are written to
  `{state_dir}/security.jsonl` — one JSON object per line, rotated at 5 MB x 5, so
  the trail is durable and greppable instead of dying with the console. (Under
  `uvicorn --factory` that logger had no handler at all and the root logger's
  `lastResort` is WARNING, so INFO-level `request` events were dropped outright.)
  Override with `ENGINE_API_SECURITY_LOG_PATH`; disable with
  `ENGINE_API_SECURITY_LOG_ENABLED=false`. **Nothing alerts on it.** No one is
  paged by an `auth_failed` burst; someone has to go and look. Under standards
  §4.10 that is partial detection and is recorded as partial, not scored a pass.
- **The evidence survives the attack that made it (SEC-9).** Measured: a request
  costs **590 bytes** of `security.jsonl`, so **53,327 requests** rotate the whole
  30 MB away — including the `auth_failed` lines proving the attack that did it, and
  it is the attacker's own INFO `request` lines that do the erasing. The
  WARNING-and-above events therefore also go to **`{state_dir}/security-alerts.jsonl`**
  (`ENGINE_API_SECURITY_ALERT_LOG_PATH`), which is rotated **daily and kept for
  30 days** rather than by size, and whose repeated lines are aggregated to at most
  one per `(event, reason)` per minute. Time-based retention only holds if the write
  rate is bounded, and the aggregation is what bounds it: the same 200 requests that
  wrote 400 lines to the full trail wrote **2 lines / 638 bytes** to the alert trail,
  and the suppressed count is carried on the next line (`repeat_suppressed`, with a
  sample of `repeat_clients`). Worst case ~2.9 MB/day, ~85 MB over the 30 days. No
  volume of requests can shorten the retention.

  ```bash
  grep auth_failed  state/security-alerts.jsonl | tail -50  # who failed a credential check
  grep repeat_suppressed state/security-alerts.jsonl        # how big was the burst
  grep -c rate_limited state/security.jsonl                 # every event, until it rolls
  ```
- **Mode gates are unchanged.** `POST /cycle/run`, `/kill-switch/reset` and
  `/reconciliation/resolve` are all refused in LIVE regardless of the token;
  re-enabling them is an operator decision, not a side effect of adding auth.

## State & durability

`ENGINE_PERSISTENCE__STATE_DIR` (default `/app/state`, a named volume) holds:

- `state.json` / the SQL DB — model registry + performance tracker (learning state)
- `loop_state.json` — the durable book + cycle counters (restart-safe)
- `alerts.jsonl` — the durable alert trail (when the sink is `jsonl`/`both`)
- `security.jsonl` — the durable API security trail (auth failures, throttling,
  one line per request), rotated at 5 MB x 5

These survive container restarts via the `engine-state` volume.

## Note on local verification

The image build is not run in CI here (no Docker daemon in the dev environment);
the Dockerfile/compose are conventional and inspected. Validate `docker build`
and a `docker compose up` smoke (hit `/health`) in a real Docker environment
before relying on the image.
