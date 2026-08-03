# Central Config + Encrypted Secrets Vault — Design

**Date:** 2026-06-11 · **ROADMAP:** Phase 6, item 1 ·
**Status:** approved (design), pre-implementation

## Purpose

Give the platform one authoritative, validated source of runtime configuration
(pydantic-settings, env-var/.env overridable) and an encrypted at-rest store for
credentials (master prompt Part 21: `vault.enc` + `vault_meta.json`, never
committed). Today every consumer takes constructor params and there is no place
to put an IBKR account id, an Alpaca key, or a DB password without writing it in
plaintext. The upcoming run-loop/API item (Phase 6 item 3) needs a config object
to build the engine + broker from; this item provides that seam.

## Approaches considered

- **A (chosen): `core/config.py` (pydantic-settings) + `core/vault.py`
  (Fernet/scrypt), `secrets/` as a data-only directory.** Follows the baseline
  app's proven `BaseSettings` + cached `get_settings()` pattern, the repo's
  `get_*()/reset_*()` singleton convention, and the master prompt's vault file
  format/location for the *generated artifacts*.
- **B: vault code in `secrets/vault.py` (verbatim master prompt).** Rejected —
  importable only as a `secrets` package, which under pytest's
  `pythonpath=["."]` would shadow the stdlib `secrets` module (a regular package
  on a front-of-path entry wins over stdlib). Breaking `import secrets`
  platform-wide to satisfy a path listing is not acceptable; the deviation is
  recorded in DECISIONS.md. The *data files* stay in `secrets/` exactly as
  specified (already gitignored: `secrets/vault.enc`, `secrets/vault_meta.json`).
- **C: OS keyring / DPAPI.** Rejected — ties secrets to one Windows user
  profile, untestable headless, and the master prompt explicitly specifies a
  portable file vault.

## Components

### `core/config.py` — central settings

- `BrokerSettings` (BaseModel): `host="127.0.0.1"`, `port=7497` (TWS paper),
  `client_id=1`, `account_id: str | None = None`.
- `VaultSettings` (BaseModel): `directory=Path("secrets")`,
  `passphrase: SecretStr | None = None` (env `ENGINE_VAULT__PASSPHRASE`;
  never printable — `SecretStr` redacts in repr/str/dump).
- `PersistenceSettings` (BaseModel): `state_dir=Path("state")`,
  `retention_days=90` (>0).
- `EngineSettings(BaseSettings)`: `env_prefix="ENGINE_"`,
  `env_nested_delimiter="__"`, `env_file=".env"`, `extra="ignore"`. Fields:
  `mode="RESEARCH"` (validated through `normalize_mode` — unknown modes raise,
  default-deny), `capital_gbp=1_000_000.0` (finite, >0),
  `stale_threshold_seconds=300.0` (finite, >0), `audit_log_path: str | None`,
  `confirm_live=False`, plus the three nested sections.
- **LIVE fail-closed validator** (`model_validator(mode="after")`): when
  `mode == "LIVE"`, require `confirm_live=True` (a lone `ENGINE_MODE=LIVE`
  env var must not arm real-money trading) **and** `audit_log_path` set (LIVE
  is never unaudited). Broker-credential completeness is enforced where the
  information lives, in `make_broker`.
- `get_settings()` / `reset_settings()` — repo-convention singleton.
- `make_broker(settings, vault=None)`: RESEARCH → `None`; PAPER →
  `PaperBroker(nav_gbp=settings.capital_gbp)`; LIVE → `IBKRBroker(...)` with
  `account_id` from settings or, failing that, vault key `ibkr_account_id` —
  **raises if neither supplies one** (no anonymous LIVE broker).
- `engine_kwargs(settings)`: the exact `TradingEngine` ctor kwargs (mode,
  capital_gbp, stale_threshold_seconds, audit_log_path). A dict, not a built
  engine — keeps `core/config.py` import-cycle-free w.r.t. the engine.
- `load_vault(settings)`: opens the configured vault with
  `settings.vault.passphrase`; raises `VaultError` when no passphrase is
  configured (the consumer of the `SecretStr` field).

### `core/vault.py` — encrypted secrets vault

- **Format:** `secrets/vault_meta.json` = `{format_version: 1, kdf: "scrypt",
  salt_b64, n, r, p}` — public KDF parameters only, no secret material.
  `secrets/vault.enc` = Fernet token (AES-128-CBC + HMAC-SHA256, authenticated)
  over UTF-8 JSON `{name: value}`. Key = scrypt(passphrase, salt, 32 bytes) →
  urlsafe-b64. Defaults `n=2**17, r=8, p=1` (OWASP interactive minimum);
  params live in meta so they are tunable (tests pass a small `n`).
- **API:** `Vault.create(passphrase, directory)` (refuses to overwrite an
  existing vault; passphrase ≥ 8 chars), `Vault.open(passphrase, directory)`
  (`VaultAuthError` on wrong passphrase or tampered ciphertext — Fernet's HMAC
  authenticates; `VaultError` on missing/malformed/unknown-version meta),
  `get(name)` (KeyError when absent — fail-closed, no silent defaults),
  `set(name, value)` / `delete(name)` (persist immediately, atomic
  write-tmp-then-`os.replace`, same pattern as `ops/persistence.py`),
  `keys()`, `rotate(new_passphrase)` (new salt, re-encrypt).
- `cryptography` is **lazy-imported** with an actionable error
  (`pip install tradingengineresearch[vault]`) so the core platform runs without the extra
  — same pattern as ib-insync in `broker/ibkr.py`. Tests `importorskip` it.
- **Operator CLI:** `python -m core.vault {init,set,get,delete,list,rotate}`.
  Passphrase from `ENGINE_VAULT__PASSPHRASE` or `getpass` prompt; `set`
  reads the value via `getpass` — secret values never appear in `argv` (shell
  history) or logs. Thin `main()`; dispatch unit-tested with monkeypatched
  prompts.

## Wiring & packaging

- `pydantic-settings` moves from the `app` extra to core `[project]`
  dependencies (config is core now); `cryptography` stays in the `vault` extra;
  `constraints.txt` gains `cryptography==47.0.0` (pydantic-settings 2.7.0
  already pinned). No mypy/ruff/coverage config changes needed (`core` is
  already in all three).
- `.gitignore` is **default-deny** for `secrets/`: `secrets/*` is ignored with
  only `secrets/.gitkeep` tracked (covers transient `.tmp`/`.bak` siblings and
  any stray file, not just the two named artifacts), plus `.env`/`.env.*`/`*.env`
  (with `!.env.example`). A root-anchored `.gitignore` at the multi-project repo
  root adds a `/.env*` + `/secrets/` safety net for an unusual CWD. *(Tightened
  from the original exact-filename allowlist after the adversarial review.)*
- DECISIONS.md records the `secrets/vault.py` → `core/vault.py` deviation and
  the LIVE `confirm_live` gate.

## Error handling

Config: pydantic `ValidationError` at construction — a bad config never yields
a half-built settings object. Vault: `VaultError` hierarchy, never a bare
`except`; wrong passphrase and tampering are indistinguishable by design
(Fernet `InvalidToken`) and both surface as `VaultAuthError`. Atomic writes
mean a crash mid-save leaves the previous vault intact; `rotate()` — which
changes two coupled files — is additionally crash-safe via a `*.bak` snapshot
that `open()` rolls back if a rotation was interrupted (so a mid-rotate crash
can never brick the vault). KDF parameters read from the unauthenticated meta
are bounds-checked before use, so a tampered `n` cannot OOM the process at open.

## Testing (`tests/test_config_vault.py`)

Settings: defaults + singleton get/reset; env overrides (top-level + nested
`__` delimiter); mode normalisation (`"paper"` → `"PAPER"`) and unknown-mode
rejection; non-finite/non-positive capital rejected; LIVE without
`confirm_live` rejected; LIVE without `audit_log_path` rejected; valid LIVE
accepted; passphrase redacted from `repr`/`model_dump`. Tests construct with
`_env_file=None` and scrub `ENGINE_*` env vars for hermeticity.

Vault: create→set→get round-trip across reopen; wrong passphrase →
`VaultAuthError`; tampered ciphertext byte → `VaultAuthError`; `create` refuses
existing vault; meta contains no secret material and ciphertext does not
contain plaintext bytes; missing key → `KeyError`; `delete` persists; `rotate`
preserves data, old passphrase fails, new works; hypothesis round-trip property
over unicode names/values; CLI dispatch with monkeypatched `getpass`.

`make_broker`: per-mode mapping; LIVE missing account_id raises; LIVE account
id resolved from vault. `engine_kwargs`: exact ctor-kwarg mapping.
