"""
TradingEngineResearch — Encrypted Secrets Vault
===================================
Credential store with encryption at rest (master prompt Part 21):

  ``secrets/vault_meta.json``  public KDF parameters only (salt, scrypt n/r/p,
                               format version) — contains no secret material.
  ``secrets/vault.enc``        Fernet token (AES-128-CBC + HMAC-SHA256,
                               authenticated encryption) over a UTF-8 JSON
                               ``{name: value}`` map.

Both artifacts are generated, live in ``secrets/`` and are gitignored. The
*code* lives here in ``core/`` rather than the spec's ``secrets/vault.py``
because a ``secrets`` package on the project root would shadow the stdlib
``secrets`` module under ``pythonpath=["."]`` — see DECISIONS.md.

Fail-closed posture:

  • The Fernet key is derived from a master passphrase via scrypt (memory-hard;
    default n=2**17, r=8, p=1 — OWASP interactive minimum). The passphrase is
    supplied via ``ENGINE_VAULT__PASSPHRASE`` or an interactive prompt and
    is never written to disk.
  • The KDF parameters in ``vault_meta.json`` are plaintext and NOT covered by
    Fernet's HMAC, so they are bounds-checked before use: a tampered/corrupted
    ``n`` can neither trigger an unbounded scrypt allocation (OOM/DoS at open)
    nor silently weaken the KDF — out-of-range params raise ``VaultError``.
  • A wrong passphrase and a tampered ciphertext are indistinguishable by
    design (Fernet's HMAC) and both raise ``VaultAuthError``.
  • ``get`` raises ``KeyError`` for a missing name — no silent defaults.
  • Single saves are atomic (write tmp, then ``os.replace``): a crash mid-save
    leaves the previous vault intact. ``rotate`` changes two coupled files
    (new salt in meta + re-encrypted enc); it is made crash-safe across that
    boundary by snapshotting the prior pair to ``*.bak`` first — if the process
    dies mid-rotate, the next ``open()`` rolls back to the pre-rotation state
    rather than leaving a vault that NEITHER passphrase can open.
  • ``create`` refuses to overwrite an existing vault and cleans up an orphaned
    half-written pair from a previous crash.

``cryptography`` is imported lazily (install via ``pip install tradingengineresearch[vault]``)
so the core platform runs without the extra — the same pattern as ib-insync in
``broker/ibkr.py``.

Operator CLI (secret values never pass through argv / shell history):

    python -m core.vault init   [--dir secrets] [--scrypt-n N]
    python -m core.vault set    NAME [--dir secrets]
    python -m core.vault get    NAME [--dir secrets]
    python -m core.vault delete NAME [--dir secrets]
    python -m core.vault list   [--dir secrets]
    python -m core.vault rotate [--dir secrets]
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["Vault", "VaultError", "VaultAuthError", "main"]

_FORMAT_VERSION = 1
_ENC_NAME = "vault.enc"
_META_NAME = "vault_meta.json"
_BAK_SUFFIX = ".bak"
_MIN_PASSPHRASE_LEN = 8
_DEFAULT_KDF: dict[str, int] = {"n": 2**17, "r": 8, "p": 1}
_SALT_BYTES = 16
_PASSPHRASE_ENV = "ENGINE_VAULT__PASSPHRASE"

# Sanity bounds for KDF parameters read from the (unauthenticated) meta file.
# n must be a power of two within [_KDF_N_MIN, _KDF_N_MAX]: the upper bound caps
# scrypt's ~128*n*r memory cost (a tampered huge n would OOM the process at
# open); the lower bound rejects a downgraded/garbage n. r and p are bounded
# likewise. Tests pass a small (but in-range) n to stay fast.
_KDF_N_MIN = 2**12
_KDF_N_MAX = 2**22
_KDF_R_MIN, _KDF_R_MAX = 1, 32
_KDF_P_MIN, _KDF_P_MAX = 1, 16


class VaultError(Exception):
    """Vault failure: missing/malformed artifacts, overwrite refusal, bad params."""


class VaultAuthError(VaultError):
    """Wrong passphrase or tampered ciphertext (indistinguishable by design)."""


def _require_cryptography() -> tuple[Any, Any, Any]:
    """Lazy import; the platform must work without the ``vault`` extra."""
    try:
        from cryptography.fernet import Fernet, InvalidToken
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:  # pragma: no cover — exercised only without the extra
        raise VaultError(
            "The encrypted vault requires the 'cryptography' package. "
            "Install it with: pip install tradingengineresearch[vault]"
        ) from exc
    return Fernet, InvalidToken, Scrypt


def _validate_kdf_params(n: int, r: int, p: int, salt: bytes) -> None:
    """Fail closed on out-of-range KDF parameters before they reach scrypt.

    ``vault_meta.json`` is plaintext and not authenticated by Fernet, so a
    tampered or corrupted ``n`` could otherwise drive scrypt to allocate
    unbounded memory (OOM/DoS at open) or be a non-power-of-two that surfaces as
    a raw ``ValueError`` instead of a ``VaultError``. Bound everything here.
    """
    if n < _KDF_N_MIN or n > _KDF_N_MAX or (n & (n - 1)) != 0:
        raise VaultError(
            f"vault: scrypt n={n!r} is not a power of two in "
            f"[{_KDF_N_MIN}, {_KDF_N_MAX}] — refusing (corrupt or tampered meta)."
        )
    if not (_KDF_R_MIN <= r <= _KDF_R_MAX):
        raise VaultError(f"vault: scrypt r={r!r} out of range [{_KDF_R_MIN}, {_KDF_R_MAX}].")
    if not (_KDF_P_MIN <= p <= _KDF_P_MAX):
        raise VaultError(f"vault: scrypt p={p!r} out of range [{_KDF_P_MIN}, {_KDF_P_MAX}].")
    if len(salt) != _SALT_BYTES:
        raise VaultError(f"vault: salt must be {_SALT_BYTES} bytes, got {len(salt)}.")


def _derive_key(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    """scrypt(passphrase) → 32 bytes → urlsafe-b64 (the Fernet key format)."""
    _, _, scrypt_cls = _require_cryptography()
    kdf = scrypt_cls(salt=salt, length=32, n=n, r=r, p=p)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically (tmp + fsync + ``os.replace``).

    A per-process tmp name avoids two writers clobbering the same scratch file;
    the tmp is removed on any failure so a crashed write leaves no stray sibling.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class Vault:
    """An open (decrypted-in-memory) secrets vault bound to its directory."""

    def __init__(self, directory: Path, fernet: Any, secrets: dict[str, str],
                 meta: dict[str, Any]) -> None:
        # Internal — construct via Vault.create() / Vault.open().
        self._directory = Path(directory)
        self._fernet = fernet
        self._secrets = secrets
        self._meta = meta

    # ── paths ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _paths(directory: Path) -> tuple[Path, Path, Path, Path]:
        """(enc, meta, enc.bak, meta.bak) for ``directory``."""
        return (
            directory / _ENC_NAME,
            directory / _META_NAME,
            directory / (_ENC_NAME + _BAK_SUFFIX),
            directory / (_META_NAME + _BAK_SUFFIX),
        )

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def create(cls, passphrase: str, directory: str | Path = "secrets", *,
               kdf_params: Optional[dict[str, int]] = None) -> "Vault":
        """Create a brand-new empty vault. Refuses to overwrite an existing one."""
        fernet_cls, _, _ = _require_cryptography()
        if len(passphrase) < _MIN_PASSPHRASE_LEN:
            raise VaultError(f"Passphrase must be at least {_MIN_PASSPHRASE_LEN} characters.")
        directory = Path(directory)
        enc_path, meta_path, _, _ = cls._paths(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # An orphaned half-written vault (exactly one of the pair, e.g. from a
        # crash during a previous create) holds no usable secret — open() needs
        # both files and create() refuses if either exists, which would wedge the
        # operator. Clean it up. A COMPLETE pair is a real vault and is untouched.
        if enc_path.exists() != meta_path.exists():
            enc_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        if enc_path.exists() or meta_path.exists():
            raise VaultError(
                f"A vault already exists in {directory} — refusing to overwrite. "
                "Use Vault.open() (or rotate() to change the passphrase)."
            )
        params = {**_DEFAULT_KDF, **(kdf_params or {})}
        n, r, p = int(params["n"]), int(params["r"]), int(params["p"])
        salt = os.urandom(_SALT_BYTES)
        _validate_kdf_params(n, r, p, salt)
        meta: dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "kdf": "scrypt",
            "salt_b64": base64.b64encode(salt).decode("ascii"),
            "n": n,
            "r": r,
            "p": p,
        }
        fernet = fernet_cls(_derive_key(passphrase, salt, n, r, p))
        vault = cls(directory, fernet, {}, meta)
        _atomic_write_bytes(meta_path, json.dumps(meta, indent=1).encode("utf-8"))
        try:
            vault._save()
        except Exception:
            # The enc write failed (e.g. disk full) — drop the orphan meta so the
            # operator can retry create() cleanly instead of being wedged.
            meta_path.unlink(missing_ok=True)
            raise
        logger.info("vault: created new vault in %s", directory)
        return vault

    @classmethod
    def open(cls, passphrase: str, directory: str | Path = "secrets") -> "Vault":
        """Open and decrypt an existing vault. Fails closed on any anomaly."""
        fernet_cls, invalid_token, _ = _require_cryptography()
        directory = Path(directory)
        enc_path, meta_path, enc_bak, meta_bak = cls._paths(directory)
        # Interrupted-rotation recovery: a COMPLETE backup pair exists only while a
        # rotate() is in flight or was interrupted. The primary pair may be
        # half-rotated (new salt in meta with old ciphertext, or the converse) and
        # openable by NEITHER passphrase. Roll back to the known-good pre-rotation
        # pair — the rotation simply did not take effect; the operator re-runs it.
        # Idempotent: the .bak is the source of truth until both primaries are
        # restored from it and it is removed.
        if enc_bak.exists() and meta_bak.exists():
            logger.warning(
                "vault: detected an interrupted rotation in %s — rolling back to the "
                "pre-rotation state. Re-run `rotate` once the vault opens cleanly.",
                directory,
            )
            _atomic_write_bytes(enc_path, enc_bak.read_bytes())
            _atomic_write_bytes(meta_path, meta_bak.read_bytes())
            enc_bak.unlink(missing_ok=True)
            meta_bak.unlink(missing_ok=True)
        if not meta_path.exists() or not enc_path.exists():
            raise VaultError(f"No vault found in {directory} (missing {_META_NAME}/{_ENC_NAME}).")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            version = meta["format_version"]
            kdf = meta["kdf"]
            salt = base64.b64decode(meta["salt_b64"])
            n, r, p = int(meta["n"]), int(meta["r"]), int(meta["p"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VaultError(f"Malformed vault metadata in {meta_path}: {exc}") from exc
        if version != _FORMAT_VERSION:
            raise VaultError(f"Unsupported vault format_version={version!r} (expected {_FORMAT_VERSION}).")
        if kdf != "scrypt":
            raise VaultError(f"Unsupported KDF {kdf!r} (expected 'scrypt').")
        _validate_kdf_params(n, r, p, salt)
        fernet = fernet_cls(_derive_key(passphrase, salt, n, r, p))
        try:
            plaintext = fernet.decrypt(enc_path.read_bytes())
        except invalid_token as exc:
            raise VaultAuthError(
                "Vault decryption failed: wrong passphrase or tampered vault.enc."
            ) from exc
        secrets_map = json.loads(plaintext.decode("utf-8"))
        return cls(directory, fernet, dict(secrets_map), meta)

    # ── access ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> str:
        """Return the secret. Raises ``KeyError`` when absent (fail-closed)."""
        if name not in self._secrets:
            raise KeyError(f"vault: no secret named {name!r}")
        return self._secrets[name]

    def set(self, name: str, value: str) -> None:
        """Store/overwrite a secret and persist immediately (atomic)."""
        self._secrets[str(name)] = str(value)
        self._save()

    def delete(self, name: str) -> None:
        """Remove a secret and persist immediately. ``KeyError`` when absent."""
        if name not in self._secrets:
            raise KeyError(f"vault: no secret named {name!r}")
        del self._secrets[name]
        self._save()

    def keys(self) -> list[str]:
        """Sorted secret names (names only — never values)."""
        return sorted(self._secrets)

    def rotate(self, new_passphrase: str) -> None:
        """Re-key the vault: new salt, new derived key, re-encrypted contents.

        Crash-safe across the two-file (meta + enc) boundary. The current
        known-good pair is snapshotted to ``*.bak`` first; the new ciphertext and
        new meta are then written. If the process dies between those two writes —
        or the writes raise — the backup remains and the next ``open()`` rolls the
        vault back to the pre-rotation state, so a rotation is never able to leave
        a vault that neither the old nor the new passphrase can open.
        """
        fernet_cls, _, _ = _require_cryptography()
        if len(new_passphrase) < _MIN_PASSPHRASE_LEN:
            raise VaultError(f"Passphrase must be at least {_MIN_PASSPHRASE_LEN} characters.")
        enc_path, meta_path, enc_bak, meta_bak = self._paths(self._directory)

        # 1. snapshot the current consistent pair so any interruption is recoverable
        _atomic_write_bytes(enc_bak, enc_path.read_bytes())
        _atomic_write_bytes(meta_bak, meta_path.read_bytes())
        try:
            # 2. re-key: derive the new key, re-encrypt, write enc then meta
            meta = dict(self._meta)
            salt = os.urandom(_SALT_BYTES)
            n, r, p = int(meta["n"]), int(meta["r"]), int(meta["p"])
            _validate_kdf_params(n, r, p, salt)
            meta["salt_b64"] = base64.b64encode(salt).decode("ascii")
            new_fernet = fernet_cls(_derive_key(new_passphrase, salt, n, r, p))
            token = new_fernet.encrypt(json.dumps(self._secrets).encode("utf-8"))
            _atomic_write_bytes(enc_path, token)
            _atomic_write_bytes(meta_path, json.dumps(meta, indent=1).encode("utf-8"))
            self._fernet = new_fernet
            self._meta = meta
        except Exception:
            # Leave the backup in place; open()-recovery rolls back from it. The
            # in-memory key is unchanged (still the old one) until both writes win.
            logger.error(
                "vault: rotation interrupted for %s; backup retained for rollback "
                "on next open().", self._directory,
            )
            raise
        # 3. rotation durably complete — drop the backup
        enc_bak.unlink(missing_ok=True)
        meta_bak.unlink(missing_ok=True)
        logger.info("vault: passphrase rotated for %s", self._directory)

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        payload = json.dumps(self._secrets).encode("utf-8")
        token = self._fernet.encrypt(payload)
        _atomic_write_bytes(self._directory / _ENC_NAME, token)


# ── operator CLI ─────────────────────────────────────────────────────────────────


def _cli_passphrase(confirm: bool = False) -> str:
    """Passphrase from the environment, else an interactive prompt (never argv)."""
    env = os.environ.get(_PASSPHRASE_ENV)
    if env:
        return env
    phrase = getpass.getpass("Vault passphrase: ")  # pragma: no cover — interactive
    if confirm and getpass.getpass("Confirm passphrase: ") != phrase:  # pragma: no cover
        raise VaultError("Passphrases do not match.")
    return phrase  # pragma: no cover — interactive


def main(argv: Optional[list[str]] = None) -> int:
    """Vault operator CLI. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(prog="python -m core.vault",
                                     description="TradingEngineResearch encrypted secrets vault")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        p = sub.add_parser(name, **kwargs)
        p.add_argument("--dir", default="secrets", help="vault directory (default: secrets)")
        return p

    p_init = _add("init", help="create a new empty vault")
    p_init.add_argument("--scrypt-n", type=int, default=_DEFAULT_KDF["n"],
                        help="scrypt cost parameter n (default 2**17)")
    _add("set", help="store a secret (value prompted, never argv)").add_argument("name")
    _add("get", help="print one secret value").add_argument("name")
    _add("delete", help="remove a secret").add_argument("name")
    _add("list", help="list secret names")
    _add("rotate", help="change the vault passphrase")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            Vault.create(_cli_passphrase(confirm=True), args.dir,
                         kdf_params={**_DEFAULT_KDF, "n": args.scrypt_n})
            print(f"vault: initialised in {args.dir}")
            return 0
        vault = Vault.open(_cli_passphrase(), args.dir)
        if args.command == "set":
            vault.set(args.name, getpass.getpass(f"Value for {args.name!r}: "))
            print(f"vault: stored {args.name!r}")
        elif args.command == "get":
            print(vault.get(args.name))
        elif args.command == "delete":
            vault.delete(args.name)
            print(f"vault: deleted {args.name!r}")
        elif args.command == "list":
            for name in vault.keys():
                print(name)
        elif args.command == "rotate":
            # The env var (if set) holds the CURRENT passphrase used to open the
            # vault above — the new one must always be prompted explicitly.
            new_phrase = getpass.getpass("New passphrase: ")
            vault.rotate(new_phrase)
            print("vault: passphrase rotated")
        return 0
    except (VaultError, KeyError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover — direct invocation
    raise SystemExit(main())
