"""
TradingEngineResearch — Data Contracts
==========================
Pydantic v2 models for all data crossing module boundaries.

Every field that carries a timestamp MUST include asof_timestamp.
In LIVE mode, a missing asof_timestamp raises ValueError.
In LIVE mode, stale_flag=True on critical inputs blocks new risk-taking.
All decision-time joins must enforce asof_timestamp <= decision_time.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


def to_aware_utc(dt: datetime) -> datetime:
    """Normalise a datetime to **tz-aware UTC** — the engine's canonical timestamp convention in
    LIVE (the feature store fail-closes on a naive asof: 'naive boundary is ambiguous'; directive
    §10 wants UTC). A naive input is assumed UTC and localised; an aware input is converted to UTC.
    Use this at the timestamp ENTRY boundaries (the price-data index, broker fill times) so the
    whole pipeline stays consistently aware and never mixes naive/aware in aggregation/sizing."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── Allowed trading modes ─────────────────────────────────────────────────────

TradingMode = Literal["RESEARCH", "PAPER", "LIVE"]

_KNOWN_MODES: frozenset[str] = frozenset({"RESEARCH", "PAPER", "LIVE"})
_LIVE_MODES:  frozenset[str] = frozenset({"LIVE"})


def normalize_mode(mode: str) -> str:
    """
    Normalise and validate a trading mode string.

    Raises ValueError for unknown modes. Treats anything that is not
    provably PAPER or RESEARCH as LIVE (default-deny).
    """
    normalised = mode.strip().upper()
    if normalised not in _KNOWN_MODES:
        raise ValueError(
            f"Unknown trading mode: {mode!r}. "
            f"Must be one of {sorted(_KNOWN_MODES)}. "
            "Refusing to continue — an unrecognised mode defaults to LIVE "
            "treatment for safety."
        )
    return normalised


def _require_asof(asof_timestamp: datetime | None, mode: str) -> datetime | None:
    """Raise ValueError in LIVE mode when asof_timestamp is absent."""
    mode = normalize_mode(mode)
    if mode in _LIVE_MODES and asof_timestamp is None:
        raise ValueError(
            "asof_timestamp is required in LIVE mode. "
            "Missing asof_timestamp means the data provenance cannot be verified."
        )
    return asof_timestamp


def _block_if_stale(stale_flag: bool, mode: str, field: str) -> None:
    """Raise ValueError in LIVE mode when a critical input is stale."""
    mode = normalize_mode(mode)
    if mode in _LIVE_MODES and stale_flag:
        raise ValueError(
            f"{field}.stale_flag=True in LIVE mode: stale data cannot be used "
            "for risk-taking decisions. Refresh the data or switch to PAPER mode."
        )


# ── 1. MarketBar ──────────────────────────────────────────────────────────────

class _FiniteContract(BaseModel):
    """Base for every data contract: reject non-finite (NaN/±inf) floats at parse
    time, for all float fields including ``dict[str, float]`` values.

    Fail-closed (golden rules): the original validators used ``< 0`` / ``<= 0`` /
    ``== 0`` comparisons, which NaN and ±inf silently evade (every comparison with
    NaN is False; inf passes a ``<= 0`` test). A NaN price, qty, weight, or
    probability slipping into the risk/optimiser/TCA math is a real hazard — the
    R5d security pass closed this for PortfolioState/BrokerState individually; this
    base closes it uniformly for the whole contract surface. ``allow_inf_nan=False``
    makes pydantic raise before any field validator runs, so non-finite never enters
    a contract regardless of which per-field check exists."""

    model_config = ConfigDict(allow_inf_nan=False)


class MarketBar(_FiniteContract):
    """OHLCV bar for a single symbol at a point in time."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    event_timestamp: datetime          # bar close time in market
    ingest_timestamp: datetime         # when we received it
    asof_timestamp: Optional[datetime] = None  # None only allowed in RESEARCH/PAPER mode
    source: str
    freshness_seconds: float
    stale_flag: bool

    @field_validator("close", "open", "high", "low")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Prices must be positive, got {v}")
        return v

    @field_validator("volume")
    @classmethod
    def _non_negative_volume(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Volume cannot be negative, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)
        _block_if_stale(self.stale_flag, mode, "MarketBar")


# ── 2. QuoteSnapshot ──────────────────────────────────────────────────────────

class QuoteSnapshot(_FiniteContract):
    """Best bid/ask snapshot for a symbol."""

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    event_timestamp: datetime
    asof_timestamp: Optional[datetime] = None
    freshness_seconds: float
    stale_flag: bool

    @field_validator("bid", "ask")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Prices must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _bid_below_ask(self) -> QuoteSnapshot:
        if self.bid >= self.ask:
            raise ValueError(
                f"bid ({self.bid}) must be less than ask ({self.ask})"
            )
        return self

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)
        _block_if_stale(self.stale_flag, mode, "QuoteSnapshot")


# ── 3. NewsEvent ──────────────────────────────────────────────────────────────

class NewsEvent(_FiniteContract):
    """A news headline and its metadata."""

    headline: str
    symbols_mentioned: list[str]
    source: str
    event_timestamp: datetime
    ingest_timestamp: datetime
    age_minutes: float
    stale_flag: bool

    @field_validator("age_minutes")
    @classmethod
    def _non_negative_age(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"age_minutes cannot be negative, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        # NewsEvent has no asof_timestamp (not a PIT field) but stale check still applies
        _block_if_stale(self.stale_flag, mode, "NewsEvent")


# ── 4. InsiderEvent ───────────────────────────────────────────────────────────

class InsiderEvent(_FiniteContract):
    """A public insider disclosure (SEC Form 4 or STOCK Act filing)."""

    symbol: str
    insider_name: str
    transaction_code: str              # e.g. "P" buy, "S" sale, "A" grant
    amount_usd: float
    event_timestamp: datetime          # disclosure date (point-in-time safe)
    age_days: float
    stale_flag: bool

    @field_validator("amount_usd")
    @classmethod
    def _positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"amount_usd must be positive, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        _block_if_stale(self.stale_flag, mode, "InsiderEvent")


# ── 5. FeatureRow ─────────────────────────────────────────────────────────────

class FeatureRow(_FiniteContract):
    """A row of ML features for a symbol, versioned and PIT-safe."""

    symbol: str
    asof_timestamp: Optional[datetime] = None  # None only allowed in RESEARCH/PAPER mode
    feature_schema_version: str        # e.g. "v6.0"
    features: dict[str, float]
    freshness_flags: dict[str, bool]   # True = stale for that feature
    missing_count: int

    @field_validator("missing_count")
    @classmethod
    def _non_negative_missing(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"missing_count cannot be negative, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)
        # Any stale feature in LIVE mode is a hard block
        if mode in _LIVE_MODES:
            stale_features = [k for k, v in self.freshness_flags.items() if v]
            if stale_features:
                raise ValueError(
                    f"FeatureRow has stale features in LIVE mode: {stale_features}. "
                    "Refresh the feature store before making risk decisions."
                )


# ── 6. PredictionRow ─────────────────────────────────────────────────────────

class PredictionRow(_FiniteContract):
    """ML model output for a symbol: the 5-tuple prediction."""

    symbol: str
    asof_timestamp: Optional[datetime] = None
    model_version: str
    expected_return: float
    risk_estimate: float
    p_positive: float                  # probability return > 0
    p_tail_loss: float                 # probability of tail-loss event
    confidence: float                  # [0, 1]

    @field_validator("p_positive", "p_tail_loss", "confidence")
    @classmethod
    def _probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Probability fields must be in [0, 1], got {v}")
        return v

    @field_validator("risk_estimate")
    @classmethod
    def _positive_risk(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"risk_estimate must be non-negative, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)


# ── 7. OrderIntent ────────────────────────────────────────────────────────────

class OrderIntent(_FiniteContract):
    """A proposed order before execution — carries full decision audit trail."""

    symbol: str
    direction: Literal["BUY", "SELL"]
    target_weight: float               # fractional portfolio weight (–1 to 1)
    expected_cost_bps: float
    urgency: Literal["NORMAL", "URGENT_DERISK", "REBALANCE"]
    alpha_half_life_minutes: int
    decision_timestamp: datetime
    model_version: str
    regime_state: str
    risk_approved: bool

    @field_validator("target_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"target_weight must be in [-1, 1], got {v}")
        return v

    @field_validator("expected_cost_bps")
    @classmethod
    def _non_negative_cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"expected_cost_bps cannot be negative, got {v}")
        return v

    def validate_for_mode(self, mode: str) -> None:
        if mode in _LIVE_MODES and not self.risk_approved:
            raise ValueError(
                "OrderIntent.risk_approved must be True before reaching LIVE execution. "
                "All orders must pass the pre-trade risk gate."
            )


# ── 8. FillEvent ──────────────────────────────────────────────────────────────

class FillEvent(_FiniteContract):
    """The execution outcome of an order, used for TCA and learning."""

    order_id: str
    symbol: str
    qty: float
    fill_price: float
    decision_price: float              # price at decision time (for slippage calc)
    arrival_price: float               # price at order arrival (VWAP benchmark)
    slippage_bps: float
    fill_timestamp: datetime
    # §17 cash leg: the broker-REPORTED commission for this fill (a non-negative cost in the
    # account currency). OPTIONAL — None when the broker reports none (e.g. Alpaca equities
    # paper); never inferred or estimated. ops.ledger.record_cycle appends it to the immutable
    # trail as a COMMISSION event so the ledger-replayed cash leg reconciles against the broker.
    commission: Optional[float] = None

    @field_validator("fill_price", "decision_price", "arrival_price")
    @classmethod
    def _positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Prices must be positive, got {v}")
        return v

    @field_validator("qty")
    @classmethod
    def _non_zero_qty(cls, v: float) -> float:
        if v == 0:
            raise ValueError("Fill qty cannot be zero")
        return v

    @field_validator("commission")
    @classmethod
    def _non_negative_commission(cls, v: Optional[float]) -> Optional[float]:
        # A negative "cost" would flip the sign of the replayed cash leg (the ledger treats
        # COMMISSION amounts as costs); a broker rebate is not representable here — refused.
        if v is not None and v < 0:
            raise ValueError(f"commission cannot be negative, got {v}")
        return v


# ── 9. RiskEvent ─────────────────────────────────────────────────────────────

class RiskEvent(_FiniteContract):
    """A risk management event — kill switch trigger, drawdown warning, etc."""

    event_type: str                    # e.g. "KILL_SWITCH", "DRAWDOWN_WARNING", "STALENESS"
    severity: Literal["INFO", "WARNING", "AMBER", "RED"]
    description: str
    timestamp: datetime
    auto_action: str | None = None     # e.g. "HALT_TRADING", "REDUCE_EXPOSURE_30PCT"

    @field_validator("description")
    @classmethod
    def _non_empty_description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("RiskEvent.description cannot be empty")
        return v


# ── 10. PortfolioState ───────────────────────────────────────────────────────

class PortfolioState(_FiniteContract):
    """The internally-tracked book: positions, weights, NAV. The system of
    record the engine trades against (``CycleInputs.current_weights`` should
    derive from this; reconciled against ``BrokerState`` before LIVE cycles)."""

    asof_timestamp: Optional[datetime] = None   # None only allowed off-LIVE
    nav_gbp: float
    cash_gbp: float
    positions: dict[str, float] = {}            # symbol → signed quantity (shares)
    weights: dict[str, float] = {}              # symbol → signed weight of NAV
    stale_flag: bool = False

    @field_validator("nav_gbp", "cash_gbp")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("PortfolioState monetary fields must be finite")
        return v

    @field_validator("positions", "weights")
    @classmethod
    def _finite_entries(cls, d: dict[str, float]) -> dict[str, float]:
        for k, v in d.items():
            if not math.isfinite(v):
                raise ValueError(f"PortfolioState requires finite values; got {k}={v}")
        return d

    @property
    def gross_exposure(self) -> float:
        return float(sum(abs(w) for w in self.weights.values()))

    @property
    def net_exposure(self) -> float:
        return float(sum(self.weights.values()))

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)
        _block_if_stale(self.stale_flag, mode, "PortfolioState")
        if mode in _LIVE_MODES and self.nav_gbp <= 0.0:
            raise ValueError(
                f"PortfolioState NAV must be positive in LIVE mode (got {self.nav_gbp}). "
                "A non-positive NAV means the account state is broken — halt trading."
            )


# ── 11. BrokerState ──────────────────────────────────────────────────────────

class BrokerState(_FiniteContract):
    """The broker's view of the account: connectivity, NAV, positions. LIVE
    trading against a disconnected or stale broker is forbidden (fail-closed)."""

    broker: str                                  # e.g. "IBKR"
    connected: bool
    account_id: Optional[str] = None
    asof_timestamp: Optional[datetime] = None
    nav_gbp: Optional[float] = None
    cash_gbp: Optional[float] = None
    buying_power_gbp: Optional[float] = None
    positions: dict[str, float] = {}             # symbol → signed quantity (shares)
    stale_flag: bool = False

    @field_validator("nav_gbp", "cash_gbp", "buying_power_gbp")
    @classmethod
    def _finite_optional(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not math.isfinite(v):
            raise ValueError("BrokerState monetary fields must be finite")
        return v

    @field_validator("positions")
    @classmethod
    def _finite_positions(cls, d: dict[str, float]) -> dict[str, float]:
        for k, v in d.items():
            if not math.isfinite(v):
                raise ValueError(f"BrokerState requires finite positions; got {k}={v}")
        return d

    def validate_for_mode(self, mode: str) -> None:
        _require_asof(self.asof_timestamp, mode)
        _block_if_stale(self.stale_flag, mode, "BrokerState")
        if mode in _LIVE_MODES:
            # A bogus broker reply must fail CLOSED: disconnected, missing/zero
            # NAV, or negative buying power all mean the account state cannot be
            # trusted enough to take new risk.
            if not self.connected:
                raise ValueError(
                    "BrokerState.connected is False in LIVE mode. "
                    "Refusing to trade against a disconnected broker."
                )
            if self.nav_gbp is None or self.nav_gbp <= 0.0:
                raise ValueError(
                    f"BrokerState NAV must be positive in LIVE mode (got {self.nav_gbp})."
                )
            if self.buying_power_gbp is not None and self.buying_power_gbp < 0.0:
                raise ValueError(
                    "BrokerState.buying_power_gbp cannot be negative in LIVE mode."
                )


def position_divergence(
    internal: dict[str, float],
    broker: dict[str, float],
    tolerance: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Per-symbol divergence between the internal book and the broker's view
    (the LIVE reconciliation primitive). Symbols whose absolute quantity gap
    exceeds ``tolerance`` are reported as ``{internal, broker}``. Fails closed:
    a non-finite quantity on either side is ALWAYS divergent (NaN compares
    false against any threshold, which would otherwise silently pass), and a
    non-finite or negative tolerance is rejected."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and >= 0, got {tolerance}")
    out: dict[str, dict[str, float]] = {}
    for symbol in sorted(set(internal) | set(broker)):
        ours = float(internal.get(symbol, 0.0))
        theirs = float(broker.get(symbol, 0.0))
        if not (math.isfinite(ours) and math.isfinite(theirs)) or abs(ours - theirs) > tolerance:
            out[symbol] = {"internal": ours, "broker": theirs}
    return out


# ── 12. Order/reconciliation dict contracts (typed; cross broker→execution→ops) ─────────
# These flow as plain JSON-native dicts (built as literals, persisted in LoopState, accessed
# with .get()), so they are TypedDicts rather than Pydantic models — minimal + JSON-native,
# while still giving mypy boundary checking at the construction sites so a key-name drift
# across the four modules they traverse is caught instead of silently returning None.

class BrokerOpenOrder(TypedDict):
    """One resting/open order from ``broker.open_orders()`` (reconnect resync, LIVE6B-3).
    ``avg_fill_price`` (optional) is the broker's TRUE avg fill price for the filled portion,
    preferred over ref_price when a disconnect-fill is booked."""
    order_ref: Optional[str]
    broker_order_id: Optional[str]
    status: str
    symbol: str
    filled_qty: float
    avg_fill_price: NotRequired[Optional[float]]


class DiscoveredFill(TypedDict):
    """A reconnect-resync-discovered disconnect-fill (broker filled MORE than locally booked),
    surfaced on the OrderManager outbox for operator-gated booking (held-book flow)."""
    order_id: str
    symbol: str
    side: str
    delta_qty: float
    broker_filled_qty: float
    ref_price: float
    avg_fill_price: NotRequired[Optional[float]]    # broker avg price when reported; else absent/None


class ReconciliationItem(TypedDict):
    """A durable OPEN/CLOSED reconciliation item (held-book flow), persisted in ``LoopState`` and
    surfaced in ``status()``. The resolution fields are added by ``resolve_reconciliation``."""
    id: str
    order_id: str
    symbol: str
    side: str
    delta_qty: float
    broker_filled_qty: float
    ref_price: float
    avg_fill_price: NotRequired[Optional[float]]
    asof: str
    status: str                         # "OPEN" | "CLOSED"
    operator: NotRequired[str]
    reason: NotRequired[str]
    decision: NotRequired[str]          # "ACCEPT" | "REJECT"
    resolved_asof: NotRequired[str]
