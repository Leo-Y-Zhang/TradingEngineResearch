"""
TradingEngineResearch — Central Configuration (pydantic-settings)
=====================================================
The single authoritative source of runtime configuration. Every value can be
overridden by environment variables (prefix ``ENGINE_``, nested sections via
``__`` — e.g. ``ENGINE_BROKER__PORT=4001``) or a local ``.env`` file.

Fail-closed posture (golden rules 1 & 2):

  • ``mode`` is validated through ``normalize_mode`` — an unknown mode raises,
    it is never coerced to something runnable (default-deny).
  • LIVE must be armed explicitly: ``confirm_live=True`` is required, so a lone
    ``ENGINE_MODE=LIVE`` env var cannot reach real money. LIVE also requires
    ``audit_log_path`` — a LIVE engine is never unaudited.
  • The vault passphrase is a ``SecretStr``: redacted in ``repr``/``str``/dumps.

Factories bridge the settings to the platform:

  • ``engine_kwargs(settings)``   → ``TradingEngine`` constructor kwargs.
  • ``make_broker(settings)``     → None (RESEARCH) / PaperBroker (PAPER) / a LIVE broker
                                    by ``broker.provider``: IBKRBroker (default; account id
                                    required) or AlpacaBroker (paper endpoint; keys required).
  • ``load_vault(settings)``      → the opened ``core.vault.Vault``.

Singleton per repo convention: ``get_settings()`` / ``reset_settings()``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.vault import Vault, VaultError
from data.data_contracts import normalize_mode

__all__ = [
    "BrokerSettings",
    "VaultSettings",
    "PersistenceSettings",
    "AlertingSettings",
    "EngineSettings",
    "get_settings",
    "reset_settings",
    "engine_kwargs",
    "make_broker",
    "load_vault",
    "make_state_store",
    "make_alert_sink",
]


class BrokerSettings(BaseModel):
    """Broker connectivity. ``provider="ibkr"`` (production target; TWS/Gateway, consumed by
    ``broker.ibkr.IBKRBroker``) or ``provider="alpaca"`` (the convenient FREE paper-validation
    broker; REST, no gateway — ``broker.alpaca.AlpacaBroker``, PAPER endpoint)."""

    host: str = "127.0.0.1"
    port: int = Field(default=7497, ge=1, le=65535)  # 7497 TWS paper; 7496 live; 4001/4002 gateway
    client_id: int = Field(default=1, ge=0)
    account_id: Optional[str] = None
    provider: Literal["ibkr", "alpaca"] = "ibkr"
    # Alpaca credentials (provider="alpaca"); a SecretStr is redacted in repr/dumps. May instead be
    # stored as the vault secrets 'alpaca_key_id' / 'alpaca_secret_key'.
    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[SecretStr] = None


class VaultSettings(BaseModel):
    """Where the encrypted vault lives and how to open it."""

    directory: Path = Path("secrets")
    passphrase: Optional[SecretStr] = None  # env: ENGINE_VAULT__PASSPHRASE


class PersistenceSettings(BaseModel):
    """Durability layer (``ops/persistence.py`` + ``ops/state_store.py``)."""

    state_dir: Path = Path("state")
    retention_days: int = Field(default=90, gt=0)
    backend: Literal["json", "sqlite"] = "json"  # json default: deterministic, dep-free
    database_url: Optional[str] = None  # sqlite default derived from state_dir


class AlertingSettings(BaseModel):
    """Where computed alerts go (``ops/observability.py``). ``logging`` is the safe
    default; ``jsonl``/``both`` also write a durable append-only trail."""

    sink: Literal["logging", "jsonl", "both", "null"] = "logging"
    alert_log_path: Optional[Path] = None  # default {state_dir}/alerts.jsonl for jsonl/both


class EngineSettings(BaseSettings):
    """The platform's central, validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Re-run validators on attribute assignment so a post-construction
        # `settings.mode = "LIVE"` cannot bypass the fail-closed gate below
        # (a lone mode flip must never arm real-money trading).
        validate_assignment=True,
    )

    mode: str = "RESEARCH"
    capital_gbp: float = 1_000_000.0
    stale_threshold_seconds: float = 300.0
    # Per-feature LIVE staleness guard. The FEATURE_FRESHNESS_THRESHOLDS (30s–300s) are tuned for an
    # INTRADAY feed; on a DAILY feed (e.g. yfinance) the latest bar is ~1 day old and always violates
    # them. Set false for a daily-data run/session so only the row-level stale_threshold_seconds
    # applies. Keep true for a real intraday LIVE deployment.
    enforce_per_feature_freshness: bool = True
    audit_log_path: Optional[str] = None
    confirm_live: bool = False

    # Run-loop (ROADMAP Phase 6 item 3): the traded symbol set and the default
    # cadence for `ops.run_loop.run_forever`. An empty universe is allowed here
    # (config may defer it) but `EngineService` fails closed on it — refusing to
    # run blind. Override via ENGINE_UNIVERSE / ENGINE_CYCLE_INTERVAL_SECONDS.
    universe: list[str] = Field(default_factory=list)
    cycle_interval_seconds: float = 86_400.0  # daily by default

    # Long-biased deployment: when ML conviction is weak (nothing admitted), deploy
    # the optimizer's CAPM-equilibrium prior tilted by the validated signal sleeves
    # at the vol target, instead of sitting in cash. All risk protections still run.
    # `baseline_in_crisis=False` keeps capital in cash when a crisis is detected.
    baseline_deploy_enabled: bool = True
    baseline_in_crisis: bool = False

    # Risk budget / aggressiveness (returns vs. drawdown trade-off). Defaults are the
    # "robust-aggressive" sweet spot from real-data backtests: ~20%/yr, Sharpe ~1.23,
    # ~16% max-dd (beats an equal-weight benchmark on BOTH return and Sharpe). Raise
    # target_vol / max_gross_leverage for more absolute return — but past ~2x the
    # Sharpe degrades toward benchmark and drawdowns deepen (diminishing returns).
    # Every constraint (CVaR, caps, crisis tightening, the fail-closed risk gate) still
    # applies. max_gross_leverage=1.0 disables leverage (long-only unlevered).
    target_vol: float = 0.22
    max_gross_leverage: float = 2.0
    max_position_weight: float = 0.20
    cvar_limit: float = 0.12
    # OPT-1 leverage ramp. None (default) = today's behaviour: the vol-target
    # scaler may jump straight to max_gross_leverage in a single rebalance. That
    # scaler is procyclical — it reads TRAILING realised vol, so it reaches full
    # leverage exactly when vol has been LOWEST, and at a monthly cadence it
    # cannot pre-empt the gap that often follows. Set e.g. 0.25 to cap each
    # rebalance's gross at 1.25x the previous book's gross, so leverage ramps in
    # over several cycles; DE-levering stays instant and unbounded either way.
    # This is a risk-appetite decision (it trades upside capture for less gap
    # exposure), which is why it ships OFF. See RISK_AND_DEFECT_REGISTER (OPT-1).
    max_lever_up_step: Optional[float] = None
    # Phase-8 control-API auth. When set, /status, /book, /monitoring, /metrics,
    # /cycle/latest and every mutating POST require Authorization: Bearer <token>.
    # /health stays open for liveness probes. Serving the API on a non-loopback
    # host WITHOUT this set is refused at startup (ops.api.assert_bind_is_safe) -
    # the API can trigger cycles, reset the kill switch and resolve reconciliation
    # items, so "reachable off-host and unauthenticated" is not a valid state.
    # A token does NOT unlock LIVE mutations; those gates are unchanged.
    api_token: Optional[str] = None
    # Per-client-IP request budgets for the control API (ops/api_security.py).
    # Reads cover the observation endpoints - the browser dashboard polls five of
    # them every 3s (~100/min), so 240 leaves room for a couple of tabs while still
    # stopping an enumeration flood. Writes are a separate, far tighter budget shared
    # by /cycle/run, /kill-switch/reset and /reconciliation/resolve, because a
    # repeated kill-switch reset is a financial-safety event, not a nuisance.
    # Either at 0 disables that budget. /health is never throttled.
    api_read_rate_limit_per_minute: int = Field(default=240, ge=0)
    api_write_rate_limit_per_minute: int = Field(default=10, ge=0)
    # SEC-4. In the recommended topologies (loopback bind, or a compose port map
    # where the peer is the docker gateway) every caller presents the SAME client
    # IP, so an IP-keyed budget is one global budget an anonymous caller can
    # exhaust. Requests bearing the valid api_token are therefore keyed on that
    # identity instead. Set api_trusted_proxy_header ONLY if the API is
    # unreachable except through a proxy that sets the header itself - it is
    # caller-supplied, so trusting it by default would let anyone mint a fresh
    # budget per request. The RIGHTMOST value is used (what the adjacent proxy
    # saw; the leftmost is whatever the client claimed).
    api_trusted_proxy_header: Optional[str] = None
    # SEC-6 detection. Where the tradingengineresearch.api.security events are written so
    # they outlive the console - under `uvicorn --factory` that logger has no
    # handler and root falls back to lastResort (WARNING), so INFO request events
    # were dropped outright. Default {state_dir}/security.jsonl; size-capped and
    # rotated by the handler. This is durable logging, NOT alerting: nobody is
    # paged, so detection is partial by design and is recorded that way.
    api_security_log_path: Optional[Path] = None
    # SEC-9. The trail above is size-capped (5 MB x 5) and dominated by INFO
    # `request` lines at ~580 bytes each, so ~52,000 requests rotate the whole
    # thing away - an attacker erases the record of their own attack simply by
    # continuing it. The WARNING-and-above events (auth_failed, rate_limited,
    # request_failed, denied/5xx requests) therefore get their OWN file with its
    # own retention: 30 daily generations, with repeated lines aggregated so that
    # no request volume can shorten it. Default {state_dir}/security-alerts.jsonl.
    api_security_alert_log_path: Optional[Path] = None
    api_security_log_enabled: bool = True
    # Strength of the validated signal-sleeve tilt on the optimiser's expected return.
    # Backtest-tuned to ~3e-3 (the sweep optimum: Sharpe 1.27 vs 1.23 at the legacy
    # 5e-4), beyond which extra tilt just adds turnover/noise. The active sleeve alpha
    # on a small large-cap universe is modest; a broader universe + predictive ML is
    # where larger, safe alpha comes from.
    signal_tilt_strength: float = 3e-3

    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        return normalize_mode(value)  # raises on unknown modes (default-deny)

    @field_validator("capital_gbp", "stale_threshold_seconds", "cycle_interval_seconds",
                     "target_vol", "max_gross_leverage", "max_position_weight", "cvar_limit",
                     "signal_tilt_strength")
    @classmethod
    def _finite_positive(cls, value: float, info: Any) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{info.field_name} must be positive and finite, got {value!r}")
        return float(value)

    @model_validator(mode="after")
    def _live_fails_closed(self) -> "EngineSettings":
        if self.mode == "LIVE":
            if not self.confirm_live:
                raise ValueError(
                    "mode=LIVE requires confirm_live=True (ENGINE_CONFIRM_LIVE=true). "
                    "A lone ENGINE_MODE=LIVE must not arm real-money trading."
                )
            if not self.audit_log_path:
                raise ValueError("mode=LIVE requires audit_log_path — LIVE is never unaudited.")
        return self


# ── singleton (repo convention: get_*/reset_*) ──────────────────────────────────

_SETTINGS: Optional[EngineSettings] = None


def get_settings() -> EngineSettings:
    """The process-wide settings singleton (constructed on first use)."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = EngineSettings()
    return _SETTINGS


def reset_settings() -> None:
    """Drop the singleton so the next ``get_settings()`` re-reads env/.env."""
    global _SETTINGS
    _SETTINGS = None


# ── factories ───────────────────────────────────────────────────────────────────


def engine_kwargs(settings: EngineSettings) -> dict[str, Any]:
    """The ``TradingEngine`` constructor kwargs for this configuration.

    Returns a dict rather than a built engine to keep this module free of an
    engine import (the engine is the composition root's concern).
    """
    return {
        "mode": settings.mode,
        "capital_gbp": settings.capital_gbp,
        "stale_threshold_seconds": settings.stale_threshold_seconds,
        "enforce_per_feature_freshness": settings.enforce_per_feature_freshness,
        "audit_log_path": settings.audit_log_path,
        "baseline_deploy_enabled": settings.baseline_deploy_enabled,
        "baseline_in_crisis": settings.baseline_in_crisis,
        "target_vol": settings.target_vol,
        "max_gross_leverage": settings.max_gross_leverage,
        "max_position_weight": settings.max_position_weight,
        "cvar_limit": settings.cvar_limit,
        "max_lever_up_step": settings.max_lever_up_step,
        "signal_tilt_strength": settings.signal_tilt_strength,
    }


def make_broker(settings: EngineSettings, vault: Optional[Any] = None) -> Optional[Any]:
    """Build the mode-appropriate broker (golden rule 1: mode is explicit).

    RESEARCH → ``None`` (no orders are ever planned, let alone submitted);
    PAPER → ``PaperBroker`` seeded with the configured capital;
    LIVE → ``IBKRBroker`` — the account id must come from settings or the vault
    key ``ibkr_account_id``; a LIVE broker is never built anonymously.
    """
    mode = normalize_mode(settings.mode)
    if mode == "RESEARCH":
        return None
    if mode == "PAPER":
        from broker.paper import PaperBroker

        return PaperBroker(nav_gbp=settings.capital_gbp)

    # Defense-in-depth: re-assert the fail-closed invariant at the money boundary.
    # The construction-time validator normally guarantees this, but the action that
    # actually reaches a real-money broker must not trust that it ran (e.g. a
    # model_construct'd or otherwise unvalidated settings object).
    if not settings.confirm_live or not settings.audit_log_path:
        raise ValueError(
            "LIVE broker requires confirm_live=True and audit_log_path — refusing to "
            "build a real-money broker from an unconfirmed or unaudited config."
        )

    # Provider switch (still behind the LIVE-arming gate above). "alpaca" = the convenient free
    # PAPER-validation broker — it uses the Alpaca PAPER endpoint (paper=True), so even in
    # TradingEngineResearch LIVE mode it never reaches real money; real-money Alpaca is a separate later gate.
    provider = str(settings.broker.provider or "ibkr").lower()
    if provider == "alpaca":
        key = settings.broker.alpaca_key_id
        secret = (settings.broker.alpaca_secret_key.get_secret_value()
                  if settings.broker.alpaca_secret_key is not None else None)
        if (not key or not secret) and vault is not None:
            try:
                key = key or vault.get("alpaca_key_id")
                secret = secret or vault.get("alpaca_secret_key")
            except KeyError:
                pass
        if not key or not secret:
            raise ValueError(
                "Alpaca broker requires alpaca_key_id + alpaca_secret_key — set "
                "ENGINE_BROKER__ALPACA_KEY_ID / ENGINE_BROKER__ALPACA_SECRET_KEY "
                "or store the vault secrets 'alpaca_key_id' / 'alpaca_secret_key'."
            )
        from broker.alpaca import AlpacaBroker

        return AlpacaBroker(key, secret, paper=True, account_id=settings.broker.account_id)

    # provider == "ibkr" (default — the production target)
    account_id = settings.broker.account_id
    if account_id is None and vault is not None:
        try:
            account_id = vault.get("ibkr_account_id")
        except KeyError:
            account_id = None
    if isinstance(account_id, str):
        account_id = account_id.strip() or None
    if not account_id:
        raise ValueError(
            "LIVE broker requires an account id — set ENGINE_BROKER__ACCOUNT_ID "
            "or store the vault secret 'ibkr_account_id'."
        )
    from broker.ibkr import IBKRBroker

    return IBKRBroker(
        host=settings.broker.host,
        port=settings.broker.port,
        client_id=settings.broker.client_id,
        account_id=account_id,
    )


def load_vault(settings: EngineSettings) -> Vault:
    """Open the configured vault. Fails closed when no passphrase is configured."""
    if settings.vault.passphrase is None:
        raise VaultError(
            "No vault passphrase configured — set ENGINE_VAULT__PASSPHRASE "
            "(or settings.vault.passphrase)."
        )
    return Vault.open(
        settings.vault.passphrase.get_secret_value(), settings.vault.directory
    )


def make_state_store(settings: EngineSettings) -> Any:
    """Build the configured persistence backend (ROADMAP Phase 6 item 2).

    ``backend="json"`` (default) → ``JsonStateStore`` at ``{state_dir}/state.json``;
    ``backend="sqlite"`` → ``SqlStateStore`` at ``database_url`` (default
    ``sqlite:///{state_dir}/tradingengineresearch.db``). The JSON default keeps RESEARCH and
    tests on the deterministic, dependency-free path. Lazy imports keep this
    module free of a SQLAlchemy dependency (the same pattern as ``make_broker``).
    """
    ps = settings.persistence
    if ps.backend == "sqlite":
        from ops.state_store import SqlStateStore

        url = ps.database_url or f"sqlite:///{(ps.state_dir / 'tradingengineresearch.db').as_posix()}"
        return SqlStateStore(url)

    from ops.state_store import JsonStateStore

    return JsonStateStore(ps.state_dir / "state.json")


def make_alert_sink(settings: EngineSettings) -> Any:
    """Build the configured alert sink (ROADMAP Phase 6 item 5).

    ``logging`` (default) → ``LoggingAlertSink``; ``jsonl`` → ``JsonlAlertSink`` at
    ``alert_log_path`` (default ``{state_dir}/alerts.jsonl``); ``both`` → a
    ``CompositeAlertSink`` of the two; ``null`` → ``NullAlertSink``. Lazy import
    keeps this module free of an ops dependency (same pattern as the others)."""
    from ops.observability import (
        CompositeAlertSink,
        JsonlAlertSink,
        LoggingAlertSink,
        NullAlertSink,
    )

    kind = settings.alerting.sink
    if kind == "null":
        return NullAlertSink()
    if kind == "logging":
        return LoggingAlertSink()

    path = settings.alerting.alert_log_path or (settings.persistence.state_dir / "alerts.jsonl")
    if kind == "jsonl":
        return JsonlAlertSink(path)
    # "both"
    return CompositeAlertSink([LoggingAlertSink(), JsonlAlertSink(path)])
