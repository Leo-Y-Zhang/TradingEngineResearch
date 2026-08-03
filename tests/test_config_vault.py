"""
Config + Vault Tests — ROADMAP Phase 6, item 1
==============================================
Central settings (core/config.py, pydantic-settings) and the encrypted
secrets vault (core/vault.py, Fernet + scrypt; artifacts in secrets/).

Covers (see docs/specs/2026-06-11-config-vault-design.md):
  - settings defaults, env overrides (incl. nested __ delimiter), singleton
  - mode normalisation + default-deny on unknown modes
  - finite/positive validation of capital and staleness threshold
  - LIVE fail-closed: confirm_live AND audit_log_path required
  - vault passphrase never appears in repr/dump (SecretStr)
  - vault create/open round-trip, wrong passphrase, tamper detection,
    no-overwrite, no plaintext at rest, delete/rotate, property round-trip
  - make_broker per-mode mapping + LIVE account-id fail-closed
  - engine_kwargs -> TradingEngine ctor mapping; load_vault seam
"""

from __future__ import annotations

import base64
import json
import os

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from core.config import (
    BrokerSettings,
    EngineSettings,
    engine_kwargs,
    get_settings,
    load_vault,
    make_broker,
    reset_settings,
)
from core.vault import VaultError

# Small-but-valid scrypt params so tests don't pay the production 2**17 cost.
SMALL_KDF = {"n": 2**12, "r": 8, "p": 1}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip ambient ENGINE_* env vars and reset the settings singleton."""
    for key in list(os.environ):
        if key.startswith("ENGINE_"):
            monkeypatch.delenv(key)
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def vault_mod():
    pytest.importorskip("cryptography")
    from core import vault

    return vault


def _settings(**kwargs) -> EngineSettings:
    """Construct settings without reading any developer .env file."""
    return EngineSettings(_env_file=None, **kwargs)


# ── Settings: defaults, env, validation ─────────────────────────────────────────


class TestSettings:
    def test_defaults_are_research_and_sane(self):
        s = _settings()
        assert s.mode == "RESEARCH"
        assert s.capital_gbp == 1_000_000.0
        assert s.stale_threshold_seconds == 300.0
        assert s.audit_log_path is None
        assert s.confirm_live is False
        assert s.broker.host == "127.0.0.1"
        assert s.broker.port == 7497
        assert s.broker.client_id == 1
        assert s.broker.account_id is None
        assert str(s.vault.directory) == "secrets"
        assert s.vault.passphrase is None
        assert s.persistence.retention_days == 90

    def test_env_overrides_top_level_and_nested(self, monkeypatch):
        monkeypatch.setenv("ENGINE_MODE", "paper")
        monkeypatch.setenv("ENGINE_CAPITAL_GBP", "250000")
        monkeypatch.setenv("ENGINE_BROKER__PORT", "4002")
        monkeypatch.setenv("ENGINE_PERSISTENCE__RETENTION_DAYS", "30")
        s = _settings()
        assert s.mode == "PAPER"
        assert s.capital_gbp == 250_000.0
        assert s.broker.port == 4002
        assert s.persistence.retention_days == 30

    def test_mode_is_normalised(self):
        assert _settings(mode=" live ", confirm_live=True, audit_log_path="a.md").mode == "LIVE"
        assert _settings(mode="research").mode == "RESEARCH"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValidationError):
            _settings(mode="YOLO")

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_capital_must_be_positive_finite(self, bad):
        with pytest.raises(ValidationError):
            _settings(capital_gbp=bad)

    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
    def test_stale_threshold_must_be_positive_finite(self, bad):
        with pytest.raises(ValidationError):
            _settings(stale_threshold_seconds=bad)

    def test_live_without_confirm_rejected(self):
        with pytest.raises(ValidationError):
            _settings(mode="LIVE", audit_log_path="audit.md")

    def test_live_without_audit_path_rejected(self):
        with pytest.raises(ValidationError):
            _settings(mode="LIVE", confirm_live=True)

    def test_live_fully_armed_accepted(self):
        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="audit.md")
        assert s.mode == "LIVE"

    def test_passphrase_never_leaks_in_repr_or_dump(self):
        s = _settings(vault={"passphrase": "hunter2-super-secret"})
        assert "hunter2-super-secret" not in repr(s)
        assert "hunter2-super-secret" not in str(s.model_dump())

    def test_singleton_get_and_reset(self):
        a = get_settings()
        assert get_settings() is a
        reset_settings()
        assert get_settings() is not a


# ── Vault ───────────────────────────────────────────────────────────────────────


class TestVault:
    def test_create_set_get_roundtrip_across_reopen(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("alpaca_key", "AK-123-SECRET")
        v2 = vault_mod.Vault.open("pass-phrase-1", tmp_path)
        assert v2.get("alpaca_key") == "AK-123-SECRET"
        assert v2.keys() == ["alpaca_key"]

    def test_wrong_passphrase_is_auth_error(self, tmp_path, vault_mod):
        vault_mod.Vault.create("correct-horse-battery", tmp_path, kdf_params=SMALL_KDF)
        with pytest.raises(vault_mod.VaultAuthError):
            vault_mod.Vault.open("wrong-passphrase!", tmp_path)

    def test_tampered_ciphertext_is_auth_error(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("k", "v")
        enc = tmp_path / "vault.enc"
        raw = bytearray(enc.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        enc.write_bytes(bytes(raw))
        with pytest.raises(vault_mod.VaultAuthError):
            vault_mod.Vault.open("pass-phrase-1", tmp_path)

    def test_create_refuses_existing_vault(self, tmp_path, vault_mod):
        vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        with pytest.raises(vault_mod.VaultError):
            vault_mod.Vault.create("pass-phrase-2", tmp_path, kdf_params=SMALL_KDF)

    def test_short_passphrase_rejected(self, tmp_path, vault_mod):
        with pytest.raises(vault_mod.VaultError):
            vault_mod.Vault.create("short", tmp_path, kdf_params=SMALL_KDF)

    def test_nothing_secret_at_rest(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("ibkr_account_id", "DU7654321")
        meta_text = (tmp_path / "vault_meta.json").read_text(encoding="utf-8")
        assert "DU7654321" not in meta_text
        assert "pass-phrase-1" not in meta_text
        meta = json.loads(meta_text)
        assert meta["format_version"] == 1
        assert meta["kdf"] == "scrypt"
        enc_bytes = (tmp_path / "vault.enc").read_bytes()
        assert b"DU7654321" not in enc_bytes
        assert b"ibkr_account_id" not in enc_bytes

    def test_get_missing_key_raises(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        with pytest.raises(KeyError):
            v.get("nope")

    def test_delete_persists(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("k", "v")
        v.delete("k")
        v2 = vault_mod.Vault.open("pass-phrase-1", tmp_path)
        with pytest.raises(KeyError):
            v2.get("k")

    def test_rotate_changes_passphrase_keeps_data(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("old-passphrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("k", "v")
        v.rotate("new-passphrase-2")
        with pytest.raises(vault_mod.VaultAuthError):
            vault_mod.Vault.open("old-passphrase-1", tmp_path)
        assert vault_mod.Vault.open("new-passphrase-2", tmp_path).get("k") == "v"

    @hyp_settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(name=st.text(min_size=1, max_size=40), value=st.text(max_size=200))
    def test_roundtrip_property(self, tmp_path, vault_mod, name, value):
        # tmp_path is constant across hypothesis examples: create once, reopen after.
        try:
            v = vault_mod.Vault.open("property-pass-1", tmp_path)
        except vault_mod.VaultError:
            v = vault_mod.Vault.create("property-pass-1", tmp_path, kdf_params=SMALL_KDF)
        v.set(name, value)
        assert v.get(name) == value


# ── CLI ─────────────────────────────────────────────────────────────────────────


class TestVaultCli:
    def test_init_set_get_list(self, tmp_path, monkeypatch, capsys, vault_mod):
        monkeypatch.setenv("ENGINE_VAULT__PASSPHRASE", "cli-pass-123")
        assert vault_mod.main(["init", "--dir", str(tmp_path), "--scrypt-n", str(SMALL_KDF["n"])]) == 0
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "CLI-SECRET-VALUE")
        assert vault_mod.main(["set", "alpaca_key", "--dir", str(tmp_path)]) == 0
        assert vault_mod.main(["get", "alpaca_key", "--dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "CLI-SECRET-VALUE" in out
        assert vault_mod.main(["list", "--dir", str(tmp_path)]) == 0
        assert "alpaca_key" in capsys.readouterr().out

    def test_wrong_passphrase_exits_nonzero(self, tmp_path, monkeypatch, vault_mod):
        monkeypatch.setenv("ENGINE_VAULT__PASSPHRASE", "cli-pass-123")
        vault_mod.main(["init", "--dir", str(tmp_path), "--scrypt-n", str(SMALL_KDF["n"])])
        monkeypatch.setenv("ENGINE_VAULT__PASSPHRASE", "not-the-passphrase")
        assert vault_mod.main(["list", "--dir", str(tmp_path)]) != 0


# ── Factories: make_broker / engine_kwargs / load_vault ─────────────────────────


class _StubVault:
    def __init__(self, data):
        self._data = data

    def get(self, name: str) -> str:
        return self._data[name]


class TestFactories:
    def test_make_broker_research_is_none(self):
        assert make_broker(_settings(mode="RESEARCH")) is None

    def test_make_broker_paper(self):
        from broker.paper import PaperBroker

        b = make_broker(_settings(mode="PAPER", capital_gbp=500_000.0))
        assert isinstance(b, PaperBroker)
        assert b.nav_gbp == 500_000.0

    def test_make_broker_live_requires_account_id(self):
        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="a.md")
        with pytest.raises(ValueError):
            make_broker(s)

    def test_make_broker_live_from_settings(self):
        from broker.ibkr import IBKRBroker

        s = _settings(
            mode="LIVE",
            confirm_live=True,
            audit_log_path="a.md",
            broker={"host": "10.0.0.5", "port": 4001, "client_id": 7, "account_id": "U111"},
        )
        b = make_broker(s)
        assert isinstance(b, IBKRBroker)
        assert (b.host, b.port, b.client_id, b.account_id) == ("10.0.0.5", 4001, 7, "U111")

    def test_make_broker_live_account_from_vault(self):
        from broker.ibkr import IBKRBroker

        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="a.md")
        b = make_broker(s, vault=_StubVault({"ibkr_account_id": "DU999"}))
        assert isinstance(b, IBKRBroker)
        assert b.account_id == "DU999"

    def test_make_broker_live_alpaca_from_settings(self):
        pytest.importorskip("alpaca")
        from broker.alpaca import AlpacaBroker

        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="a.md",
                      broker={"provider": "alpaca", "alpaca_key_id": "PK123",
                              "alpaca_secret_key": "sek", "account_id": "DU1"})
        b = make_broker(s)
        assert isinstance(b, AlpacaBroker)
        assert b.paper is True and b.account_id == "DU1"   # paper endpoint; real-money Alpaca is a separate gate

    def test_make_broker_live_alpaca_keys_from_vault(self):
        pytest.importorskip("alpaca")
        from broker.alpaca import AlpacaBroker

        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="a.md", broker={"provider": "alpaca"})
        b = make_broker(s, vault=_StubVault({"alpaca_key_id": "PK", "alpaca_secret_key": "SEK"}))
        assert isinstance(b, AlpacaBroker)

    def test_make_broker_alpaca_requires_keys(self):
        s = _settings(mode="LIVE", confirm_live=True, audit_log_path="a.md", broker={"provider": "alpaca"})
        with pytest.raises(ValueError):
            make_broker(s)                                 # no keys (settings or vault) -> refuse

    def test_make_broker_alpaca_still_gated_by_confirm_live(self):
        # the provider switch must NOT bypass the LIVE arming gate (model_construct skips the validator)
        s = EngineSettings.model_construct(
            mode="LIVE", confirm_live=False, audit_log_path=None,
            broker=BrokerSettings(provider="alpaca", alpaca_key_id="PK", alpaca_secret_key="SEK"),
        )
        with pytest.raises(ValueError):
            make_broker(s)

    def test_engine_kwargs_match_engine_ctor(self):
        from core.engine.engine import TradingEngine

        s = _settings(
            mode="PAPER",
            capital_gbp=500_000.0,
            stale_threshold_seconds=120.0,
            audit_log_path="audit.md",
        )
        kwargs = engine_kwargs(s)
        assert kwargs == {
            "mode": "PAPER",
            "capital_gbp": 500_000.0,
            "stale_threshold_seconds": 120.0,
            "enforce_per_feature_freshness": True,
            "audit_log_path": "audit.md",
            "baseline_deploy_enabled": True,
            "baseline_in_crisis": False,
            "target_vol": 0.22,
            "max_gross_leverage": 2.0,
            "max_position_weight": 0.20,
            "cvar_limit": 0.12,
            "signal_tilt_strength": 3e-3,
            "max_lever_up_step": None,   # OPT-1 leverage ramp: ships OFF
        }
        engine = TradingEngine(**kwargs)
        assert engine.mode == "PAPER"
        assert engine.capital_gbp == 500_000.0

    def test_load_vault_requires_passphrase(self):
        with pytest.raises(VaultError):
            load_vault(_settings())

    def test_load_vault_opens_configured_vault(self, tmp_path, vault_mod):
        vault_mod.Vault.create("load-pass-123", tmp_path, kdf_params=SMALL_KDF).set("k", "v")
        s = _settings(vault={"directory": str(tmp_path), "passphrase": "load-pass-123"})
        assert load_vault(s).get("k") == "v"


# ── Hardening (post-review) ───────────────────────────────────────────────────────


class TestConfigHardening:
    def test_assignment_revalidates_and_blocks_unarmed_live(self):
        # validate_assignment: flipping mode to LIVE post-construction without
        # confirm_live + audit_log_path must fail closed, not silently arm money.
        s = _settings(mode="PAPER")
        with pytest.raises(ValidationError):
            s.mode = "LIVE"

    def test_make_broker_independently_rechecks_live_invariant(self):
        # An object that skipped the construction validator (model_construct) must
        # still not be able to build a real-money broker — make_broker re-asserts.
        s = EngineSettings.model_construct(
            mode="LIVE", confirm_live=False, audit_log_path=None,
            broker=BrokerSettings(account_id="U1"),
        )
        with pytest.raises(ValueError):
            make_broker(s)


class TestVaultHardening:
    def test_open_rejects_oversized_scrypt_n(self, tmp_path, vault_mod):
        # A tampered/corrupted huge n would OOM if handed to scrypt — reject first.
        vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        meta_path = tmp_path / "vault_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["n"] = 2**30
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(vault_mod.VaultError):
            vault_mod.Vault.open("pass-phrase-1", tmp_path)

    def test_open_rejects_non_power_of_two_n(self, tmp_path, vault_mod):
        # Surfaces as VaultError, not a raw ValueError leaking out of scrypt.
        vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        meta_path = tmp_path / "vault_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["n"] = 3
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(vault_mod.VaultError):
            vault_mod.Vault.open("pass-phrase-1", tmp_path)

    def test_create_rejects_too_weak_scrypt_n(self, tmp_path, vault_mod):
        with pytest.raises(vault_mod.VaultError):
            vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params={"n": 2**6, "r": 8, "p": 1})

    def test_rotate_interruption_rolls_back_and_preserves_secrets(self, tmp_path, vault_mod):
        # Simulate a rotation interrupted mid-flight: a valid backup pair on disk
        # plus a primary meta clobbered with a fresh (mismatched) salt, so the
        # primary opens with NEITHER passphrase. open() must roll back from .bak.
        v = vault_mod.Vault.create("old-pass-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("ibkr_account_id", "DU123")
        enc, meta = tmp_path / "vault.enc", tmp_path / "vault_meta.json"
        (tmp_path / "vault.enc.bak").write_bytes(enc.read_bytes())
        (tmp_path / "vault_meta.json.bak").write_bytes(meta.read_bytes())
        m = json.loads(meta.read_text())
        m["salt_b64"] = base64.b64encode(os.urandom(16)).decode("ascii")
        meta.write_text(json.dumps(m))
        v2 = vault_mod.Vault.open("old-pass-1", tmp_path)
        assert v2.get("ibkr_account_id") == "DU123"
        assert not (tmp_path / "vault.enc.bak").exists()
        assert not (tmp_path / "vault_meta.json.bak").exists()

    def test_rotate_happy_path_clears_backup(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("old-pass-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("k", "v")
        v.rotate("new-pass-2")
        assert not (tmp_path / "vault.enc.bak").exists()
        assert not (tmp_path / "vault_meta.json.bak").exists()
        assert vault_mod.Vault.open("new-pass-2", tmp_path).get("k") == "v"

    def test_create_recovers_from_orphaned_meta(self, tmp_path, vault_mod):
        # An orphan meta from a crashed prior create must not wedge a fresh create.
        (tmp_path / "vault_meta.json").write_text(
            '{"format_version":1,"kdf":"scrypt","salt_b64":"AAAAAAAAAAAAAAAAAAAAAA==","n":4096,"r":8,"p":1}'
        )
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("k", "v")
        assert vault_mod.Vault.open("pass-phrase-1", tmp_path).get("k") == "v"

    def test_keys_returns_sorted_names(self, tmp_path, vault_mod):
        v = vault_mod.Vault.create("pass-phrase-1", tmp_path, kdf_params=SMALL_KDF)
        v.set("zeta", "1")
        v.set("alpha", "2")
        v.set("mu", "3")
        assert v.keys() == ["alpha", "mu", "zeta"]
