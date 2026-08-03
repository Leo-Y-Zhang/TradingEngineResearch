"""
scripts/broker_preflight.py — verify the configured broker connection + account, READ-ONLY.

Run this BEFORE a supervised paper session to confirm the gateway/API + credentials + account are
wired correctly, WITHOUT the engine ever placing an order. It uses the same settings + safety gates
as the live service (``core.config.make_broker``), so it works for either provider:

    python scripts/broker_preflight.py        # reads ENGINE_* env / .env

Exit code 0 = connected + account read OK. Non-zero = a wiring problem to fix first. It NEVER
submits an order (read-only: connect -> account_state -> open_orders -> disconnect).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable when run directly (`python scripts/broker_preflight.py`) even
# without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from core.config import get_settings, load_vault, make_broker

    settings = get_settings()
    provider = getattr(settings.broker, "provider", "ibkr")
    print(f"mode={settings.mode} provider={provider}")

    if settings.mode == "RESEARCH":
        print("RESEARCH mode builds no broker (no orders are ever planned). "
              "Set ENGINE_MODE=PAPER or LIVE to pre-flight a broker.")
        return 0

    vault = None
    if settings.mode == "LIVE":
        try:
            vault = load_vault(settings)
        except Exception as exc:  # noqa: BLE001 - report the wiring problem clearly
            print(f"FAIL: vault could not be opened ({exc}). "
                  "LIVE requires ENGINE_VAULT__PASSPHRASE.")
            return 3

    try:
        broker = make_broker(settings, vault)
    except Exception as exc:  # noqa: BLE001 - a config/arming problem
        print(f"FAIL: make_broker refused ({exc}).")
        return 4
    if broker is None:
        print("No broker built for this mode.")
        return 1

    connect = getattr(broker, "connect", None)
    if callable(connect):
        connect()
    if not bool(getattr(broker, "connected", False)):
        print("FAIL: could not connect — check the gateway/API is running and the credentials.")
        return 2

    asof = datetime.now(timezone.utc)
    bs = broker.account_state(asof)
    print(f"connected=True account_id={bs.account_id}")
    print(f"nav={bs.nav_gbp} cash={bs.cash_gbp}  (broker base currency)")
    print(f"positions={dict(bs.positions)}")
    print(f"open_orders={len(broker.open_orders(asof))}")

    disconnect = getattr(broker, "disconnect", None)
    if callable(disconnect):
        disconnect()

    print("OK - connection + account verified (READ-ONLY; no orders placed).")
    print("REMINDER: confirm the account_id above is your PAPER account before running the session.")
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(main())
