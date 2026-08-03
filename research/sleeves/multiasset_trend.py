"""Sleeve: MULTI-ASSET TIME-SERIES MOMENTUM (trend) on the long-history panel.

Pre-registered in ``research/sleeves/multiasset_trend_prereg.md``. Run ONCE, no tuning.
Read that file before this one; every choice here is fixed there and nothing is searched.

The whole point of this sleeve is that the two things that killed twelve prior studies --
a short sample (which raises the DSR bar) and 117-236bps round-trip costs (which ate the
gross edge breadth produced) -- are both relaxed here: ~60 years of history on instruments
that trade at 1-10bps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.book_scaler import (  # noqa: F401  -- NO_ESTIMATE_FLAT is the declared repair
    NO_ESTIMATE_FLAT,
    REGISTERED_NO_ESTIMATE,
    book_scaler,
)

__all__ = [
    "PRIMARY_UNIVERSE",
    "BLOCKS",
    "LOOKBACKS",
    "TrendConfig",
    "TrendResult",
    "load_excess_panel",
    "build_signals",
    "inverse_vol",
    "run_trend",
    "newey_west_tstat",
    "effective_n",
    "max_drawdown",
    "annual_sharpe",
    "active_report",
    "decade_sharpe",
    "kelly_report",
    "concentration",
    "MONTHS",
    "VOL_TARGETS",
    "COST_BRACKETS",
    "SPLIT_DATE",
]

# ── Pre-registered constants (prereg §1, §3, §4, §5) ──────────────────────────

BLOCKS: dict[str, tuple[str, ...]] = {
    "equity": ("SPX", "NASDAQ", "FTSE100", "N225", "DAX", "HSI", "ASX200"),
    "rates": ("US5Y_TR", "US10Y_TR", "US30Y_TR"),
    "commodity": ("GOLD_F", "WTI_F", "SILVER_F", "COPPER_F"),
    "fx": ("USDX", "EURUSD", "GBPUSD", "JPYUSD"),
}
PRIMARY_UNIVERSE: tuple[str, ...] = tuple(k for keys in BLOCKS.values() for k in keys)

# Series that are USD TOTAL returns (cash + excess) and therefore need the bill rate
# subtracted to become a futures-equivalent excess return. Everything else is a price
# / futures / spot return which already is an excess return (prereg §2).
CASH_SUBTRACTED: frozenset[str] = frozenset({"US5Y_TR", "US10Y_TR", "US30Y_TR"})

LOOKBACKS: tuple[int, ...] = (1, 3, 6, 12)
VOL_WINDOW = 36            # months, instrument volatility
VOL_MIN_OBS = 24
ELIGIBLE_MIN_OBS = 36      # months of history before an instrument may be traded
MIN_INSTRUMENTS = 3        # book is off below this
BOOK_VOL_WINDOW = 36
BOOK_VOL_MIN = 12
UNIT_VOL = 0.10            # per-instrument vol unit; cancels in the book scaler
GROSS_CAP = 10.0           # x book equity
VOL_TARGETS: tuple[float, ...] = (0.10, 0.20, 0.40, 0.60)
COST_BRACKETS: dict[str, float] = {"2bps": 0.0002, "10bps": 0.0010}
NW_LAG = 6
MONTHS = 12
SPLIT_DATE = "2009-01-01"

_DATA = Path("_data/multiasset")


# ── Statistics ────────────────────────────────────────────────────────────────

def newey_west_tstat(x: pd.Series, lag: int = NW_LAG) -> float:
    """t-stat of the mean of ``x`` with a Newey-West HAC standard error.

    Monthly strategy returns are autocorrelated (vol targeting alone induces it), so an
    iid standard error would overstate significance. Bartlett kernel, ``lag`` lags.
    """
    a = np.asarray(x.dropna(), dtype=float)
    n = a.size
    if n < 8:
        return float("nan")
    mu = a.mean()
    e = a - mu
    s = float(e @ e) / n
    for lag_i in range(1, min(lag, n - 1) + 1):
        g = float(e[lag_i:] @ e[:-lag_i]) / n
        s += 2.0 * (1.0 - lag_i / (lag + 1.0)) * g
    if s <= 0:
        return float("nan")
    return float(mu / math.sqrt(s / n))


def annual_sharpe(x: pd.Series) -> float:
    a = x.dropna()
    if len(a) < 8 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * math.sqrt(MONTHS))


def max_drawdown(x: pd.Series) -> float:
    """Worst peak-to-trough on the compounded path of monthly returns (negative)."""
    a = x.dropna()
    if a.empty:
        return float("nan")
    curve = (1.0 + a).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def effective_n(corr: pd.DataFrame) -> float:
    """Participation ratio of the eigenvalues: (sum L)^2 / sum L^2.

    Four FX pairs that are all the dollar are not four bets. This is the number that
    says so.
    """
    c = corr.dropna(how="all").dropna(axis=1, how="all")
    if c.empty:
        return float("nan")
    vals = np.linalg.eigvalsh(np.nan_to_num(c.to_numpy(dtype=float), nan=0.0))
    vals = np.clip(vals, 0.0, None)
    denom = float((vals ** 2).sum())
    return float(vals.sum() ** 2 / denom) if denom > 0 else float("nan")


# ── Panel loading (prereg §2) ─────────────────────────────────────────────────

def load_excess_panel(
    *,
    unscreened: bool = False,
    universe: tuple[str, ...] = PRIMARY_UNIVERSE,
    data_dir: Path = _DATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(excess_returns, was_null)`` for ``universe`` on the month-end panel.

    ``was_null`` marks INTERIOR nulls, which the prereg amendment treats as a zero return
    with no position held. Leading nulls stay NaN so eligibility can see them.
    """
    name = "returns_monthly_unscreened.parquet" if unscreened else "returns_monthly.parquet"
    rets = pd.read_parquet(data_dir / name)
    cash = pd.read_parquet(data_dir / "cash_monthly.parquet")["US_CASH_13W"]

    missing = [k for k in universe if k not in rets.columns]
    if missing:
        raise KeyError(f"panel is missing pre-registered instruments: {missing}")

    x = rets.loc[:, list(universe)].copy()
    for key in universe:
        if key in CASH_SUBTRACTED:
            x[key] = x[key] - cash.reindex(x.index)

    interior = pd.DataFrame(False, index=x.index, columns=x.columns)
    for key in universe:
        first = x[key].first_valid_index()
        if first is not None:
            interior.loc[first:, key] = x.loc[first:, key].isna()
    x = x.mask(interior, 0.0)
    return x, interior


# ── Signal and sizing (prereg §3, §4) ─────────────────────────────────────────

def build_signals(x: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Composite trend signal and the four single-lookback signals.

    ``S = mean_L sign(sum of the last L monthly excess returns)`` over L in 1/3/6/12.
    """
    per_lb: dict[int, pd.DataFrame] = {}
    for lb in LOOKBACKS:
        per_lb[lb] = np.sign(x.rolling(lb, min_periods=lb).sum())
    composite = sum(per_lb.values()) / float(len(LOOKBACKS))
    return composite, per_lb


def inverse_vol(x: pd.DataFrame) -> pd.DataFrame:
    """Annualised trailing volatility, 36m window, min 24 obs. Causal by construction."""
    return x.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std(ddof=1) * math.sqrt(MONTHS)


def _eligibility(x: pd.DataFrame, sigma: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    counted = x.notna().cumsum()
    return (counted >= ELIGIBLE_MIN_OBS) & sigma.notna() & signal.notna() & (sigma > 0)


# ── The book ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrendConfig:
    name: str = "PRIMARY"
    block_risk_parity: bool = False
    unscreened: bool = False
    randomise_seed: int | None = None
    lookbacks: tuple[int, ...] = LOOKBACKS


@dataclass
class TrendResult:
    config: str
    gross: pd.Series
    net: dict[str, pd.Series]
    bench_gross: pd.Series
    bench_net: dict[str, pd.Series]
    weights: pd.DataFrame
    turnover: pd.Series
    gross_leverage: pd.Series
    cap_binding: pd.Series
    net_exposure: pd.Series
    eligible_count: pd.Series
    pnl: pd.DataFrame
    stats: dict = field(default_factory=dict)
    #: months with NO volatility estimate, where the gross cap alone set the leverage.
    #: `cap_binding` EXCLUDES these by construction, which is how they stayed invisible.
    #: DECISION-dated like `cap_binding` in this sleeve: the scale is applied the
    #: FOLLOWING month, so `gross_leverage` shows the cap one month later.
    no_vol_estimate: pd.Series = field(default_factory=pd.Series)
    no_vol_estimate_policy: str = REGISTERED_NO_ESTIMATE
    #: VERIFY-2(a): the book-vol scaler k(t), DECISION-dated -- the value at t is
    #: computed from book returns realised at or before t and scales the weights HELD
    #: during t+1. Same convention as ``DefensiveResult.scaler``. Full panel index
    #: (NaN-free once the book is live; the gross cap sets k before a vol estimate
    #: exists). Additive: nothing existing reads it.
    scaler: pd.Series = field(default_factory=pd.Series)


def _positions(
    x: pd.DataFrame,
    cfg: TrendConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Pre-scaler notionals ``n(i,t)`` decided AT month-end t, plus eligibility."""
    signals = np.sign(x.rolling(1, min_periods=1).sum()) * 0.0
    per_lb = {lb: np.sign(x.rolling(lb, min_periods=lb).sum()) for lb in cfg.lookbacks}
    signals = sum(per_lb.values()) / float(len(cfg.lookbacks))

    sigma = inverse_vol(x)
    eligible = _eligibility(x, sigma, signals)

    if cfg.randomise_seed is not None:
        rng = np.random.default_rng(cfg.randomise_seed)
        flips = pd.DataFrame(
            rng.choice([-1.0, 1.0], size=signals.shape), index=signals.index,
            columns=signals.columns,
        )
        signals = signals * flips

    n = signals.where(eligible, 0.0) * (UNIT_VOL / sigma.where(eligible))
    n = n.fillna(0.0)

    if cfg.block_risk_parity:
        scaled = pd.DataFrame(0.0, index=n.index, columns=n.columns)
        live_blocks = pd.Series(0.0, index=n.index)
        for _, keys in BLOCKS.items():
            cols = [k for k in keys if k in n.columns]
            m = eligible[cols].sum(axis=1)
            live_blocks += (m > 0).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                scaled[cols] = n[cols].div(m.replace(0, np.nan), axis=0)
        scaled = scaled.fillna(0.0)
        n = scaled.mul(len(BLOCKS) / live_blocks.replace(0, np.nan), axis=0).fillna(0.0)

    count = eligible.sum(axis=1)
    n = n.where(count >= MIN_INSTRUMENTS, 0.0)
    return n, eligible, count


def run_trend(
    cfg: TrendConfig,
    *,
    vol_target: float = 0.10,
    data_dir: Path = _DATA,
    x: pd.DataFrame | None = None,
    interior: pd.DataFrame | None = None,
    no_vol_estimate: str = REGISTERED_NO_ESTIMATE,
) -> TrendResult:
    """Run the pre-registered sleeve once at one volatility target.

    `no_vol_estimate` defaults to the REGISTERED behaviour, which reproduces every banked
    number bit-for-bit: for the book's first BOOK_VOL_MIN months there is no volatility
    estimate and the GROSS CAP alone sets the leverage. `NO_ESTIMATE_FLAT` is the repair.
    """
    if x is None or interior is None:
        x, interior = load_excess_panel(unscreened=cfg.unscreened, data_dir=data_dir)

    n, eligible, count = _positions(x, cfg)
    held = interior.reindex_like(n).fillna(False)
    n = n.mask(held, 0.0)                       # no position in a nulled month

    # Raw (unscaled) book, held during t from the decision at t-1.
    pos = n.shift(1).fillna(0.0)
    xz = x.fillna(0.0)
    b = (pos * xz).sum(axis=1)
    b = b.where(pos.abs().sum(axis=1) > 0)

    # Causal book-vol estimate at t, applied to t+1.
    sig_b = b.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)
    sig_b = sig_b.replace(0.0, np.nan)
    k_raw = (vol_target / sig_b)
    gross_unit = n.abs().sum(axis=1)
    k_cap = (GROSS_CAP / gross_unit.replace(0.0, np.nan))
    # `min` skips NaN, so before BOOK_VOL_MIN months of book history exist k_raw is NaN
    # and the GROSS CAP alone sets the scale -- the book runs at FULL leverage with no
    # volatility estimate behind it. `no_vol_estimate` is the REGISTERED behaviour and
    # the default; NO_ESTIMATE_FLAT is the repair. See `research.book_scaler`.
    scaler = book_scaler(k_raw, k_cap, no_estimate=no_vol_estimate,
                         live=gross_unit > 0)
    k, cap_binding = scaler.k, scaler.cap_binding

    w = n.mul(k, axis=0).shift(1).fillna(0.0)   # weights HELD during month t
    live = w.abs().sum(axis=1) > 0

    pnl = w * xz
    gross = pnl.sum(axis=1).where(live)
    turnover = w.diff().abs().sum(axis=1).where(live)
    gross_lev = w.abs().sum(axis=1).where(live)
    net_exposure = w.sum(axis=1).where(live)

    # Benchmark: equal-weight LONG-ONLY over the same eligible set, same convention.
    elig_shift = eligible.shift(1).astype(float).fillna(0.0).astype(bool)
    elig_shift = elig_shift & ~held
    nb = elig_shift.sum(axis=1)
    wb = elig_shift.astype(float).div(nb.replace(0, np.nan), axis=0).fillna(0.0)
    wb = wb.where(live.reindex(wb.index).fillna(False), 0.0)
    bench_gross = (wb * xz).sum(axis=1).where(live)
    bench_turnover = wb.diff().abs().sum(axis=1).where(live)

    net: dict[str, pd.Series] = {}
    bench_net: dict[str, pd.Series] = {}
    for label, c in COST_BRACKETS.items():
        net[label] = gross - 0.5 * c * turnover
        bench_net[label] = bench_gross - 0.5 * c * bench_turnover

    return TrendResult(
        config=cfg.name,
        gross=gross.dropna(),
        net={k_: v.dropna() for k_, v in net.items()},
        bench_gross=bench_gross.dropna(),
        bench_net={k_: v.dropna() for k_, v in bench_net.items()},
        weights=w,
        turnover=turnover.dropna(),
        gross_leverage=gross_lev.dropna(),
        cap_binding=cap_binding.reindex(gross.index).fillna(False),
        no_vol_estimate=scaler.no_estimate.reindex(gross.index).fillna(False).astype(bool),
        no_vol_estimate_policy=no_vol_estimate,
        net_exposure=net_exposure.dropna(),
        eligible_count=count,
        pnl=pnl,
        scaler=k,
    )


# ── Reporting helpers ─────────────────────────────────────────────────────────

def active_report(strat: pd.Series, bench: pd.Series) -> dict[str, float]:
    """Arithmetic active return + t-stat, Jensen alpha, and the variance-drag identity.

    The variance-drag line is the whole reason this function exists: geometric excess
    equals arithmetic active MINUS (var_s - var_b)/2, so a lower-volatility strategy shows
    a positive geometric excess with no alpha at all. That illusion killed a prior result.
    """
    a, b = strat.align(bench, join="inner")
    d = a - b
    var_s = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))

    X = np.column_stack([np.ones(len(b)), b.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, a.to_numpy(dtype=float), rcond=None)
    resid = a.to_numpy(dtype=float) - X @ coef
    alpha_t = newey_west_tstat(pd.Series(resid + coef[0], index=a.index))

    sd_s, sd_b = math.sqrt(var_s), math.sqrt(var_b)
    scale = (sd_b / sd_s) if sd_s > 0 else float("nan")
    vm = a * scale - b

    geo_s = float(np.expm1(np.log1p(a).mean() * MONTHS))
    geo_b = float(np.expm1(np.log1p(b).mean() * MONTHS))

    return {
        "months": int(len(d)),
        "arith_active_annual": float(d.mean() * MONTHS),
        "arith_active_tstat": newey_west_tstat(d),
        "jensen_alpha_annual": float(coef[0] * MONTHS),
        "jensen_beta": float(coef[1]),
        "jensen_alpha_tstat": alpha_t,
        "volmatched_active_annual": float(vm.mean() * MONTHS),
        "volmatched_active_tstat": newey_west_tstat(vm),
        "geometric_excess_annual": geo_s - geo_b,
        "variance_drag_annual": float((var_s - var_b) / 2.0 * MONTHS),
        "strat_vol": sd_s * math.sqrt(MONTHS),
        "bench_vol": sd_b * math.sqrt(MONTHS),
        "strat_sharpe": annual_sharpe(a),
        "bench_sharpe": annual_sharpe(b),
    }


def decade_sharpe(x: pd.Series) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dec, grp in x.groupby((x.index.year // 10) * 10):
        out[f"{int(dec)}s"] = {
            "months": int(len(grp)),
            "sharpe": annual_sharpe(grp),
            "mean_annual": float(grp.mean() * MONTHS),
        }
    return out


def kelly_report(sharpe: float) -> dict[str, float]:
    """Half-Kelly reachable compound return and the volatility it implies."""
    if not np.isfinite(sharpe):
        return {"half_kelly_growth": float("nan"), "implied_vol": float("nan"),
                "full_kelly_growth": float("nan")}
    return {
        "half_kelly_growth": 3.0 * sharpe ** 2 / 8.0,
        "full_kelly_growth": sharpe ** 2 / 2.0,
        "implied_vol": sharpe / 2.0,
    }


def concentration(pnl: pd.DataFrame) -> dict[str, Any]:
    flat = pnl.stack()
    total = float(flat.sum())
    if total == 0:
        return {"top_cell_share": float("nan"), "top_instrument_share": float("nan")}
    by_inst = pnl.sum(axis=0)
    idx = flat.abs().idxmax()
    return {
        "top_cell_share": float(flat.loc[idx] / total),
        "top_cell": f"{idx[1]} {pd.Timestamp(idx[0]).strftime('%Y-%m')}",
        "top_cell_abs_share": float(flat.abs().max() / flat.abs().sum()),
        "top_instrument_share": float(by_inst.max() / total),
        "top_instrument": str(by_inst.idxmax()),
    }
