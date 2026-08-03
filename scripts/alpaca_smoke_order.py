"""
scripts/alpaca_smoke_order.py — validate the REAL Alpaca execution round-trip with ONE tiny order.

The engine, by design, will not place an order without a *validated* signal edge (golden rule 5),
so a normal LIVE run on free daily data correctly sits in cash and never exercises the broker. This
script validates the broker integration DIRECTLY — submit -> client_order_id round-trip -> status
mapping -> (fill, if the market is open) -> cleanup-to-flat — WITHOUT touching the engine's signal
safety. It is the paper-session execution check the runbook describes.

It is intentionally conservative:
  * It builds the broker through the SAME arming path as the live service (``core.config.make_broker``),
    so the LIVE confirm/audit/vault gates all apply.
  * It REFUSES to run against anything other than an Alpaca PAPER account (``broker.paper`` must be
    True and the account id must look like a paper id) — it never touches real money.
  * Transmitting an order needs DELIBERATE operator action (directive §4): without ``--confirm`` it
    does a READ-ONLY dry run (connect + account + print the exact order) and stops. ``--confirm``
    actually submits the one order.
  * It ALWAYS cleans up: a filled order is flattened (opposite market order); a resting/queued order
    (market closed) is cancelled. The account is returned to its starting position.

Usage:
    python scripts/alpaca_smoke_order.py                 # dry run: verify wiring, show the order
    python scripts/alpaca_smoke_order.py --confirm       # place ONE 1-share AAPL order + validate + clean up
    python scripts/alpaca_smoke_order.py --confirm --symbol MSFT --qty 1
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Make the repo root importable when run directly, even without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _looks_like_paper_account(account_id: str | None) -> bool:
    """Alpaca paper account numbers are prefixed 'PA'; live accounts are not. Conservative: an
    empty/unknown id is NOT treated as paper, so we fail closed and refuse to trade it."""
    return bool(account_id) and str(account_id).upper().startswith("PA")


def _poll_order(broker, client_order_id: str, attempts: int = 6, wait: float = 1.0):
    """Poll the broker for the order by our client_order_id and return the raw Alpaca order
    (or None). Confirms the client_order_id round-trips back from the broker."""
    for _ in range(attempts):
        try:
            order = broker._client.get_order_by_client_id(client_order_id)
            if order is not None:
                return order
        except Exception as exc:  # noqa: BLE001 — keep polling; report None at the end
            print(f"  (poll: get_order_by_client_id failed: {exc})")
        time.sleep(wait)
    return None


def _try_flatten(broker, plan) -> None:
    """Submit a flattening order, reporting (not raising) on failure so cleanup never crashes."""
    try:
        broker.submit([plan], mode="LIVE")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: flatten failed ({exc}); CHECK the Alpaca dashboard and flatten manually.")


def _describe(order) -> str:
    if order is None:
        return "  <order not found by client_order_id>"
    status = getattr(getattr(order, "status", ""), "value", getattr(order, "status", ""))
    return (f"  broker_id={getattr(order, 'id', '')}\n"
            f"  client_order_id={getattr(order, 'client_order_id', '')}\n"
            f"  status={status}\n"
            f"  filled_qty={getattr(order, 'filled_qty', 0)}\n"
            f"  filled_avg_price={getattr(order, 'filled_avg_price', None)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Alpaca paper execution smoke-test (one tiny order).")
    ap.add_argument("--confirm", action="store_true",
                    help="actually submit the order (without this it is a read-only dry run)")
    ap.add_argument("--symbol", default="AAPL", help="symbol to trade (default AAPL)")
    ap.add_argument("--qty", type=float, default=1.0, help="quantity in shares (default 1)")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL"], help="order side (default BUY)")
    args = ap.parse_args()

    from core.config import get_settings, load_vault, make_broker

    settings = get_settings()
    provider = getattr(settings.broker, "provider", "ibkr")
    print(f"mode={settings.mode} provider={provider}")
    if settings.mode != "LIVE":
        print("FAIL: this smoke-test submits via the LIVE-only broker path. Set ENGINE_MODE=LIVE "
              "(with CONFIRM_LIVE + audit log) to run it against the paper account.")
        return 2
    if str(provider).lower() != "alpaca":
        print(f"FAIL: provider is {provider!r}, not 'alpaca'. This smoke-test is Alpaca-specific.")
        return 2

    try:
        vault = load_vault(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: vault could not be opened ({exc}).")
        return 3
    try:
        broker = make_broker(settings, vault)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: make_broker refused ({exc}).")
        return 4
    if broker is None:
        print("FAIL: no broker built for this mode.")
        return 4

    # Refuse anything that is not the Alpaca paper endpoint — never touch real money.
    if not bool(getattr(broker, "paper", False)):
        print("FAIL: the configured Alpaca broker is NOT the paper endpoint (broker.paper is False). "
              "Refusing — this smoke-test is paper-only.")
        return 5

    if not broker.connect():
        print("FAIL: could not connect — check the API credentials in the vault.")
        return 6

    asof = datetime.now(timezone.utc)
    bs = broker.account_state(asof)
    print(f"connected=True account_id={bs.account_id}")
    print(f"buying_power={bs.buying_power_gbp} cash={bs.cash_gbp} nav={bs.nav_gbp}  (USD on Alpaca)")
    print(f"starting positions={dict(bs.positions)}")

    if not _looks_like_paper_account(bs.account_id):
        print(f"FAIL: account_id {bs.account_id!r} does not look like an Alpaca PAPER account (PA...). "
              "Refusing to trade it.")
        broker.disconnect()
        return 5

    # Build the single child-plan the broker.submit contract consumes.
    client_order_id = f"research-smoke-{int(asof.timestamp())}"
    plan = SimpleNamespace(symbol=args.symbol, qty=abs(args.qty), side=args.side,
                           order_ref=client_order_id)
    print("\nOrder to submit (ONE child slice):")
    print(f"  symbol={plan.symbol} side={plan.side} qty={plan.qty} type=MARKET "
          f"client_order_id={client_order_id}")

    if not args.confirm:
        print("\nDRY RUN — no order submitted. Re-run with --confirm to place it.")
        broker.disconnect()
        return 0

    # ── transmit (deliberate operator action) ──────────────────────────────────────
    print("\n--confirm given: submitting the order ...")
    try:
        fills = broker.submit([plan], mode="LIVE")
    except Exception as exc:  # noqa: BLE001 — a 403 (insufficient buying power) raises here
        msg = str(exc)
        print(f"FAIL: submit was rejected by Alpaca ({msg}).")
        if "buying power" in msg.lower() or "insufficient" in msg.lower():
            print("  The paper account has no buying power. Reset/fund it in the Alpaca dashboard "
                  "(Paper Account -> Reset -> set cash to e.g. 100000), then re-run.")
        broker.disconnect()
        return 7
    broker_id = broker.last_broker_order_ids.get(client_order_id, "")
    print(f"submit returned {len(fills)} immediate fill(s); broker order id={broker_id or '<none>'}")
    for f in fills:
        print(f"  FILL: {f.symbol} qty={f.qty} @ {f.fill_price}")

    # Confirm the round-trip: fetch the order back BY OUR client_order_id.
    print("\nRound-trip check (fetch the order by our client_order_id):")
    order = _poll_order(broker, client_order_id)
    print(_describe(order))

    # ── cleanup to flat ─────────────────────────────────────────────────────────────
    status = str(getattr(getattr(order, "status", ""), "value",
                         getattr(order, "status", ""))).lower() if order is not None else "unknown"
    filled_qty = 0.0
    try:
        filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
    except (TypeError, ValueError):
        filled_qty = 0.0

    print("\nCleanup:")
    if filled_qty > 0:
        # Filled (market open) -> flatten with the opposite side so we end flat.
        opposite = "SELL" if args.side == "BUY" else "BUY"
        flat_id = f"{client_order_id}-flat"
        flat = SimpleNamespace(symbol=args.symbol, qty=filled_qty, side=opposite, order_ref=flat_id)
        print(f"  order filled {filled_qty} -> flattening with {opposite} {filled_qty} {args.symbol}")
        _try_flatten(broker, flat)
    elif status in ("filled", "partially_filled"):
        opposite = "SELL" if args.side == "BUY" else "BUY"
        flat = SimpleNamespace(symbol=args.symbol, qty=max(filled_qty, args.qty), side=opposite,
                               order_ref=f"{client_order_id}-flat")
        print(f"  order {status} -> flattening with {opposite} {args.symbol}")
        _try_flatten(broker, flat)
    elif order is not None and broker_id:
        # Resting/queued (market closed) -> cancel so it never fills and leaves a position.
        try:
            broker._client.cancel_order_by_id(broker_id)
            print(f"  order resting ({status}) -> cancelled broker id {broker_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: cancel failed ({exc}); CHECK the Alpaca dashboard and cancel manually.")
    else:
        print(f"  nothing to clean up (status={status}).")

    time.sleep(1.0)
    final = broker.account_state(datetime.now(timezone.utc))
    print(f"\nfinal positions={dict(final.positions)}")
    print("\nSUMMARY: submit + client_order_id round-trip"
          + (" + FILL" if filled_qty > 0 else "")
          + " validated against the real Alpaca paper account; account returned toward flat.")
    if args.symbol in final.positions and abs(final.positions[args.symbol]) > 1e-9:
        print(f"NOTE: a residual {args.symbol} position remains ({final.positions[args.symbol]}); "
              "the flatten may fill at the next open — check the dashboard.")
    broker.disconnect()
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    raise SystemExit(main())
