#!/usr/bin/env bash
# TradingEngineResearch container entrypoint (ROADMAP Phase 6, item 6). Dispatches to one of
# three run modes; anything else is exec'd verbatim (e.g. `bash`, `python -m core.vault`).
#
#   combined  (default) — scheduled loop + API in ONE process, sharing one service
#                         (one writer to the state dir; the recommended topology).
#   loop                — the scheduled run-loop only (no HTTP surface).
#   api                 — the API only (observe; runs its own service — do NOT point
#                         this at the same state volume as a `loop`/`combined`).
set -euo pipefail

cmd="${1:-combined}"
case "${cmd}" in
  combined)
    exec python -m ops.run_loop --serve
    ;;
  loop)
    exec python -m ops.run_loop
    ;;
  api)
    # --no-proxy-headers is a SECURITY control, not a preference. uvicorn defaults
    # proxy_headers on with forwarded_allow_ips=127.0.0.1, so its own middleware
    # rewrites the client address from a caller-supplied X-Forwarded-For BEFORE the
    # app sees it: the application-level trusted-proxy check in ops/api.py is then
    # moot, the security trail records a forged address as fact, and a rotating
    # header mints a fresh rate-limit budget per request. If a real reverse proxy
    # fronts this API, name its header in ENGINE_API_TRUSTED_PROXY_HEADER and
    # let the app honour it (capped, and with the true socket peer still recorded).
    # Kept in step with ops.api.api_uvicorn_kwargs by tests/test_api_proxy_headers.py.
    exec uvicorn --factory ops.api:create_app_from_settings \
      --no-proxy-headers \
      --host "${ENGINE_API_HOST:-0.0.0.0}" \
      --port "${ENGINE_API_PORT:-8000}"
    ;;
  *)
    exec "$@"
    ;;
esac
