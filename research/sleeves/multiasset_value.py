"""Sleeve: CROSS-ASSET VALUE on the long-history multi-asset panel.

Pre-registered in ``research/sleeves/multiasset_value_prereg.md``. Run ONCE, no tuning.
Read that file before this one; every choice here is fixed there and nothing is searched.

Why this sleeve exists (prereg §0). The portfolio arithmetic
``S = s*sqrt(N/(1+(N-1)*rho))`` makes ``rho`` -- not ``s`` -- the binding quantity once a
programme has several sleeves. Cross-asset value is documented as NEGATIVELY correlated to
cross-asset momentum (Asness, Moskowitz & Pedersen, *Value and Momentum Everywhere*, JF
68(3), 2013). **The headline deliverable is therefore the correlation to the trend sleeve,
not the standalone Sharpe.**

Three things this module is careful about
=========================================
1. **Units.** A 5-year log return and a term-spread deviation are not comparable numbers.
   Scores are ranked WITHIN block and the resulting block books are combined at equal risk,
   which is AMP's own construction. No cross-block comparison of raw scores ever happens.
2. **Causality.** The rates score uses an EXPANDING mean of the instrument's own term
   spread; the vol estimates use trailing windows; the book scaler ``k(t)`` is estimated
   from book returns realised at or before ``t``; positions decided at ``t`` are held during
   ``t+1``. Nothing reads forward.
3. **Convention parity with the trend sleeve.** The excess-return panel is loaded through
   ``research.sleeves.multiasset_trend.load_excess_panel`` and the reporting statistics are
   the same functions. A correlation between two sleeves computed on two different return
   conventions is not a correlation between two sleeves.
"""

from __future__ import annotations

import json
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
from research.multiasset.panel import dsr_sharpe_bar

# Deliberately shared with the trend sleeve so the two books are measured identically.
from research.sleeves.multiasset_trend import (
    MONTHS,
    active_report,
    annual_sharpe,
    concentration,
    decade_sharpe,
    effective_n,
    kelly_report,
    load_excess_panel,
    max_drawdown,
    newey_west_tstat,
)

__all__ = [
    "BLOCKS",
    "VALUE_UNIVERSE",
    "RATE_YIELD",
    "ValueConfig",
    "ValueResult",
    "load_yield_spreads",
    "reversal_score",
    "rates_value_score",
    "value_scores",
    "rank_weights",
    "trailing_vol",
    "run_value",
    "combined_sharpe_equal_risk",
    "combined_sharpe_optimal",
    "year_concentration",
    "VOL_TARGETS",
    "COST_BRACKETS",
    "REVERSAL_MONTHS",
    "MIN_PER_BLOCK",
    "MIN_BLOCKS",
    "SPLIT_DATE",
]

# ── Pre-registered constants (prereg §1, §3, §4, §5) ──────────────────────────

BLOCKS: dict[str, tuple[str, ...]] = {
    "equity": ("SPX", "NASDAQ", "FTSE100", "N225", "DAX", "HSI", "ASX200"),
    "rates": ("US5Y_TR", "US10Y_TR", "US30Y_TR"),
    "commodity": ("GOLD_F", "WTI_F", "SILVER_F", "COPPER_F"),
}
VALUE_UNIVERSE: tuple[str, ...] = tuple(k for keys in BLOCKS.values() for k in keys)

# FX is absent by decision, not by oversight: the value measure for a currency is the
# deviation from long-run PPP, and this panel contains NO price-level series for any
# country. A nominal 5-year spot change is a different signal, so none is substituted.

RATE_YIELD: dict[str, str] = {
    "US5Y_TR": "US5Y_YLD",
    "US10Y_TR": "US10Y_YLD",
    "US30Y_TR": "US30Y_YLD",
}
BILL_YIELD = "US13W_YLD"

REVERSAL_MONTHS = 60       # AMP's 5-year window; ends at t, no skip (prereg §3a)
SKIP_MONTHS = 12           # D3 diagnostic only
SPREAD_MIN_OBS = 60        # expanding-mean minimum for the rates score
VOL_WINDOW = 36            # months, instrument volatility
VOL_MIN_OBS = 24
MIN_PER_BLOCK = 3          # a cross-section needs a top and a bottom that differ
MIN_BLOCKS = 2             # a one-block book is not cross-asset value
BOOK_VOL_WINDOW = 36
BOOK_VOL_MIN = 12
UNIT_VOL = 0.10            # per-instrument vol unit; cancels in the book scaler
GROSS_CAP = 10.0           # x book equity
VOL_TARGETS: tuple[float, ...] = (0.10, 0.20, 0.40)
COST_BRACKETS: dict[str, float] = {"2bps": 0.0002, "10bps": 0.0010}
NW_LAG = 6
SPLIT_DATE = "2009-01-01"

_DATA = Path("_data/multiasset")
_OUT = Path("research/sleeves/_value")
_TREND_SERIES = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")


# ── Signals (prereg §3) ───────────────────────────────────────────────────────

def load_yield_spreads(index: pd.DatetimeIndex, *, data_dir: Path = _DATA) -> pd.DataFrame:
    """Term spreads ``y_maturity - y_13w`` for the three bonds, on ``index``.

    Yields are already in DECIMAL in the panel (``yields_monthly.parquet``). Reindexing
    onto the returns index before the expanding mean is safe because both indices are
    normalised calendar month-ends; the assertion below refuses to proceed if they are not.
    """
    y = pd.read_parquet(data_dir / "yields_monthly.parquet")
    missing = [c for c in (*RATE_YIELD.values(), BILL_YIELD) if c not in y.columns]
    if missing:
        raise KeyError(f"yields panel is missing: {missing}")
    y = y.reindex(index)
    if not y[BILL_YIELD].notna().any():
        raise ValueError("yield panel does not align with the returns index")
    return pd.DataFrame(
        {key: y[col] - y[BILL_YIELD] for key, col in RATE_YIELD.items()}, index=index
    )


def reversal_score(x: pd.DataFrame, *, months: int = REVERSAL_MONTHS, skip: int = 0) -> pd.DataFrame:
    """``-(cumulative log excess return)`` over the trailing window ending at ``t-skip``.

    Log rather than simple: the object AMP use is ``log(P_{t-5y}/P_t)``, and compounding 60
    simple returns makes the score's scale depend on its own sign. ``skip`` is the D3
    diagnostic (drop the most recent 12 months, i.e. the trend sleeve's own window) and is
    0 in the pre-registered PRIMARY. Either way the instrument must have ``months`` of
    history, so eligibility is unchanged by the diagnostic.
    """
    if skip < 0 or skip >= months:
        raise ValueError("skip must be in [0, months)")
    lg = np.log1p(x)
    win = months - skip
    s = lg.rolling(win, min_periods=win).sum()
    if skip:
        s = s.shift(skip)
    return -s


def rates_value_score(spreads: pd.DataFrame, *, min_obs: int = SPREAD_MIN_OBS) -> pd.DataFrame:
    """Term spread minus its OWN expanding-window long-run mean. High = cheap = long.

    Real yield is not constructible here -- the panel contains no inflation series -- so the
    brief's stated alternative is used. The subtraction of the instrument's own long-run mean
    is what makes this a VALUE signal rather than the bond CARRY signal, which is the level
    of the same spread. Expanding (not rolling) because "long-run average" means all history
    to date, and it is causal by construction.
    """
    return spreads - spreads.expanding(min_periods=int(min_obs)).mean()


def value_scores(
    x: pd.DataFrame,
    spreads: pd.DataFrame,
    *,
    skip: int = 0,
    uniform_rates: bool = False,
) -> pd.DataFrame:
    """The full value-score panel: reversal everywhere, term-spread deviation for bonds.

    ``uniform_rates`` is the D4 diagnostic -- bonds scored by the same 5-year reversal as
    everything else, removing the term-spread signal entirely.
    """
    v = reversal_score(x, skip=skip)
    if not uniform_rates:
        rate_v = rates_value_score(spreads)
        for key in RATE_YIELD:
            if key in v.columns:
                # Keep the 60-month return-history requirement as well, so eligibility is
                # identical across blocks and a bond cannot trade on a spread alone.
                v[key] = rate_v[key].where(v[key].notna())
    return v


def trailing_vol(x: pd.DataFrame) -> pd.DataFrame:
    """Annualised trailing volatility, 36m window, min 24 obs. Causal by construction."""
    return x.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std(ddof=1) * math.sqrt(MONTHS)


# ── Sizing (prereg §4) ────────────────────────────────────────────────────────

def rank_weights(v: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """AMP rank weighting inside one block: ``u = (rank - (N+1)/2) / sum|rank - (N+1)/2|``.

    Dollar-neutral (``sum u = 0``) and unit gross (``sum |u| = 1``) on every row with at
    least ``MIN_PER_BLOCK`` eligible instruments; all zeros otherwise. Ties take average
    ranks, which preserves both properties.
    """
    masked = v.where(eligible)
    ranks = masked.rank(axis=1, method="average")
    n = ranks.notna().sum(axis=1)
    d = ranks.sub((n + 1.0) / 2.0, axis=0)
    denom = d.abs().sum(axis=1)
    u = d.div(denom.replace(0.0, np.nan), axis=0)
    u = u.where(n >= MIN_PER_BLOCK)
    return u.fillna(0.0)


@dataclass(frozen=True)
class ValueConfig:
    """One pre-registered configuration. PRIMARY is the default; the rest are diagnostics."""

    name: str = "PRIMARY"
    skip_months: int = 0                      # D3
    uniform_rates: bool = False               # D4
    blocks: tuple[str, ...] = tuple(BLOCKS)   # D2 sub-books
    min_blocks: int = MIN_BLOCKS              # D2 relaxes this to 1; PRIMARY never does
    unscreened: bool = False                  # D6
    randomise_seed: int | None = None         # D1


@dataclass
class ValueResult:
    config: str
    gross: pd.Series
    net: dict[str, pd.Series]
    bench_gross: pd.Series
    bench_net: dict[str, pd.Series]
    weights: pd.DataFrame
    turnover: pd.Series
    bench_turnover: pd.Series
    gross_leverage: pd.Series
    cap_binding: pd.Series
    net_exposure: pd.Series
    eligible_count: pd.Series
    live_blocks: pd.Series
    pnl: pd.DataFrame
    x: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: months with NO volatility estimate, where the gross cap alone set the leverage.
    #: `cap_binding` EXCLUDES these by construction, which is how they stayed invisible.
    #: DECISION-dated like `cap_binding` in this sleeve: the scale is applied the
    #: FOLLOWING month, so `gross_leverage` shows the cap one month later.
    no_vol_estimate: pd.Series = field(default_factory=pd.Series)
    no_vol_estimate_policy: str = REGISTERED_NO_ESTIMATE


def _notionals(
    x: pd.DataFrame,
    spreads: pd.DataFrame,
    cfg: ValueConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Pre-scaler notionals ``n(i,t)`` decided AT month-end ``t``, plus tradability."""
    v = value_scores(x, spreads, skip=cfg.skip_months, uniform_rates=cfg.uniform_rates)
    sigma = trailing_vol(x)
    eligible = v.notna() & sigma.notna() & (sigma > 0.0)

    blocks = {b: [k for k in BLOCKS[b] if k in x.columns] for b in cfg.blocks}
    u = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    live = pd.Series(0.0, index=x.index)
    tradable = pd.DataFrame(False, index=x.index, columns=x.columns)

    for _, cols in blocks.items():
        if not cols:
            continue
        u_b = rank_weights(v[cols], eligible[cols])
        block_live = eligible[cols].sum(axis=1) >= MIN_PER_BLOCK
        u[cols] = u_b
        live += block_live.astype(float)
        tradable[cols] = eligible[cols].mul(block_live, axis=0).astype(bool)

    if cfg.randomise_seed is not None:
        rng = np.random.default_rng(cfg.randomise_seed)
        flips = pd.DataFrame(
            rng.choice([-1.0, 1.0], size=u.shape), index=u.index, columns=u.columns
        )
        u = u * flips

    book_on = live >= float(cfg.min_blocks)
    u = u.where(book_on, 0.0)
    tradable = tradable.mul(book_on, axis=0).astype(bool)

    # Equal risk across live blocks: 7/3/4 instruments would otherwise hand the equity
    # block 54% of the gross book purely by count.
    n = (u * (UNIT_VOL / sigma)).fillna(0.0)
    n = n.div(live.where(book_on).replace(0.0, np.nan), axis=0).fillna(0.0)
    return n, tradable, tradable.sum(axis=1), live


def run_value(
    cfg: ValueConfig = ValueConfig(),
    *,
    vol_target: float = 0.20,
    data_dir: Path = _DATA,
    x: pd.DataFrame | None = None,
    spreads: pd.DataFrame | None = None,
    no_vol_estimate: str = REGISTERED_NO_ESTIMATE,
) -> ValueResult:
    """Run the pre-registered sleeve once at one volatility target.

    `no_vol_estimate` defaults to the REGISTERED behaviour, which reproduces every banked
    number bit-for-bit: for the book's first BOOK_VOL_MIN months there is no volatility
    estimate and the GROSS CAP alone sets the leverage. `NO_ESTIMATE_FLAT` is the repair.
    """
    if x is None:
        x, _interior = load_excess_panel(
            unscreened=cfg.unscreened, universe=VALUE_UNIVERSE, data_dir=data_dir
        )
    if spreads is None:
        spreads = load_yield_spreads(x.index, data_dir=data_dir)

    n, tradable, count, live = _notionals(x, spreads, cfg)

    # Raw (unscaled) book, held during t from the decision at t-1.
    pos = n.shift(1).fillna(0.0)
    xz = x.fillna(0.0)
    b = (pos * xz).sum(axis=1)
    b = b.where(pos.abs().sum(axis=1) > 0)

    # Causal book-vol estimate at t, applied to t+1.
    sig_b = b.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)
    sig_b = sig_b.replace(0.0, np.nan)
    k_raw = vol_target / sig_b
    gross_unit = n.abs().sum(axis=1)
    k_cap = GROSS_CAP / gross_unit.replace(0.0, np.nan)
    # `min` skips NaN, so before BOOK_VOL_MIN months of book history exist k_raw is NaN
    # and the GROSS CAP alone sets the scale -- FULL leverage with no volatility estimate
    # behind it, and `cap_binding` excludes exactly those months. See
    # `research.book_scaler`; this sleeve is a byte-identical clone of the trend one.
    scaler = book_scaler(k_raw, k_cap, no_estimate=no_vol_estimate, live=gross_unit > 0)
    k, cap_binding = scaler.k, scaler.cap_binding

    w = n.mul(k, axis=0).shift(1).fillna(0.0)   # weights HELD during month t
    is_live = w.abs().sum(axis=1) > 0

    pnl = w * xz
    gross = pnl.sum(axis=1).where(is_live)
    turnover = w.diff().abs().sum(axis=1).where(is_live)
    gross_lev = w.abs().sum(axis=1).where(is_live)
    net_exposure = w.sum(axis=1).where(is_live)

    # Benchmark: equal-weight LONG-ONLY over exactly the tradable set, same convention.
    tb = tradable.shift(1).astype(float).fillna(0.0).astype(bool)
    nb = tb.sum(axis=1)
    wb = tb.astype(float).div(nb.replace(0, np.nan), axis=0).fillna(0.0)
    wb = wb.where(is_live.reindex(wb.index).fillna(False), 0.0)
    bench_gross = (wb * xz).sum(axis=1).where(is_live)
    bench_turnover = wb.diff().abs().sum(axis=1).where(is_live)

    net: dict[str, pd.Series] = {}
    bench_net: dict[str, pd.Series] = {}
    for label, c in COST_BRACKETS.items():
        net[label] = gross - 0.5 * c * turnover
        bench_net[label] = bench_gross - 0.5 * c * bench_turnover

    return ValueResult(
        config=cfg.name,
        gross=gross.dropna(),
        net={k_: v_.dropna() for k_, v_ in net.items()},
        bench_gross=bench_gross.dropna(),
        bench_net={k_: v_.dropna() for k_, v_ in bench_net.items()},
        weights=w,
        turnover=turnover.dropna(),
        bench_turnover=bench_turnover.where(is_live).dropna(),
        gross_leverage=gross_lev.dropna(),
        cap_binding=cap_binding.reindex(gross.index).fillna(False),
        net_exposure=net_exposure.dropna(),
        eligible_count=count,
        live_blocks=live,
        pnl=pnl.loc[gross.dropna().index] if not gross.dropna().empty else pnl,
        x=x,
        no_vol_estimate=scaler.no_estimate.reindex(gross.index).fillna(False).astype(bool),
        no_vol_estimate_policy=no_vol_estimate,
    )


# ── Portfolio arithmetic (prereg §7) ──────────────────────────────────────────

def combined_sharpe_equal_risk(s1: float, s2: float, rho: float) -> float:
    """Two sleeves at EQUAL risk weight: ``(s1 + s2) / sqrt(2 + 2*rho)``."""
    denom = 2.0 + 2.0 * rho
    if denom <= 0:
        return float("nan")
    return float((s1 + s2) / math.sqrt(denom))


def combined_sharpe_optimal(s1: float, s2: float, rho: float) -> float:
    """Two sleeves at the mean-variance optimal weight.

    ``S* = sqrt( (s1^2 + s2^2 - 2*rho*s1*s2) / (1 - rho^2) )``. Reported alongside the
    equal-risk number because the optimal weights are in-sample and can be negative; the
    equal-risk figure is the deployable one.
    """
    if abs(rho) >= 1.0:
        return float("nan")
    num = s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2
    if num <= 0:
        return float("nan")
    return float(math.sqrt(num / (1.0 - rho * rho)))


def year_concentration(pnl: pd.DataFrame) -> dict[str, Any]:
    """Share of total P&L contributed by the single best calendar year."""
    by_year = pnl.sum(axis=1).groupby(pnl.index.year).sum()
    total = float(by_year.sum())
    if total == 0 or by_year.empty:
        return {"top_year": None, "top_year_share": float("nan")}
    return {
        "top_year": int(by_year.idxmax()),
        "top_year_share": float(by_year.max() / total),
        "top_year_abs_share": float(by_year.abs().max() / by_year.abs().sum()),
    }


# ── Study runner ──────────────────────────────────────────────────────────────

def _series_stats(r: pd.Series) -> dict[str, float]:
    return {
        "months": int(len(r)),
        "sharpe": annual_sharpe(r),
        "mean_annual": float(r.mean() * MONTHS),
        "vol_annual": float(r.std(ddof=1) * math.sqrt(MONTHS)),
        "tstat": newey_west_tstat(r),
        "max_drawdown": max_drawdown(r),
        "worst_month": float(r.min()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


def _read_sibling_series(path: Path) -> pd.Series | None:
    """Load another sleeve's monthly net-of-cost series if it is on disk.

    The sibling sleeves are written by concurrently running workflows and do not share an
    output format, so both CSV and parquet are handled and the net column is looked up by
    name rather than by position.
    """
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    for col in ("net_10bps", "net10bps", "net", "gross"):
        if col in df.columns:
            return df[col].dropna()
    return None


def _find_carry_series() -> tuple[str, pd.Series] | None:
    """Locate the carry sleeve's monthly net series, whatever it called its output dir."""
    root = Path("research/sleeves")
    seen: set[Path] = set()
    for d in sorted(root.glob("_*carry*")) + sorted(root.glob("_carry*")):
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        cands = [f for f in sorted(d.glob("*net*monthly*")) if f.suffix in (".csv", ".parquet")]
        cands += [f for f in sorted(d.glob("*monthly*")) if f.suffix in (".csv", ".parquet")]
        for f in cands:
            if "trend" in f.name or "two_sleeve" in f.name:
                continue                      # that is a reference copy, not the carry book
            s = _read_sibling_series(f)
            if s is not None and len(s) > 24:
                return str(f).replace("\\", "/"), s
    return None


def run_study(out_dir: Path = _OUT, data_dir: Path = _DATA) -> dict:
    """Execute the pre-registered study exactly once and write the receipts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x, _ = load_excess_panel(universe=VALUE_UNIVERSE, data_dir=data_dir)
    spreads = load_yield_spreads(x.index, data_dir=data_dir)

    # D6 -- the quarantine touches only EURUSD/JPYUSD, both excluded here.
    xu, _ = load_excess_panel(unscreened=True, universe=VALUE_UNIVERSE, data_dir=data_dir)
    unscreened_identical = bool(x.equals(xu))

    primary = ValueConfig()
    res = {t: run_value(primary, vol_target=t, x=x, spreads=spreads) for t in VOL_TARGETS}
    ref = res[0.20]

    out: dict = {
        "universe": list(VALUE_UNIVERSE),
        "blocks": {b: list(k) for b, k in BLOCKS.items()},
        "fx_excluded_reason": (
            "PPP deviation is not constructible from price data alone; the panel contains "
            "no inflation or price-level series for any country."
        ),
        "sample": {
            "start": str(ref.gross.index.min().date()),
            "end": str(ref.gross.index.max().date()),
            "months": int(len(ref.gross)),
            "years": round(len(ref.gross) / 12.0, 2),
        },
        "unscreened_panel_identical_on_this_universe": unscreened_identical,
    }

    years = len(ref.gross) / 12.0
    out["dsr_bar"] = {
        f"n_trials_{n}": round(dsr_sharpe_bar(years, n_trials=n), 4) for n in (32, 40, 48)
    }

    # Vol targets
    out["vol_targets"] = {}
    for t, r in res.items():
        row = {
            "gross_sharpe": annual_sharpe(r.gross),
            "cap_binding_pct": float(100.0 * r.cap_binding.mean()),
            "gross_leverage_mean": float(r.gross_leverage.mean()),
            "gross_leverage_p95": float(r.gross_leverage.quantile(0.95)),
            "turnover_per_year": float(r.turnover.mean() * MONTHS),
            "net_exposure_mean": float(r.net_exposure.mean()),
        }
        for label in COST_BRACKETS:
            row[f"net_sharpe_{label}"] = annual_sharpe(r.net[label])
            row[f"net_mean_annual_{label}"] = float(r.net[label].mean() * MONTHS)
            row[f"net_vol_annual_{label}"] = float(r.net[label].std(ddof=1) * math.sqrt(MONTHS))
        out["vol_targets"][f"{int(t*100)}pct"] = row

    # Primary detail at 20%
    out["primary_20pct"] = {
        "gross": _series_stats(ref.gross),
        **{f"net_{lab}": _series_stats(ref.net[lab]) for lab in COST_BRACKETS},
        **{f"bench_{lab}": _series_stats(ref.bench_net[lab]) for lab in COST_BRACKETS},
        "bench_gross": _series_stats(ref.bench_gross),
        "bench_turnover_per_year": float(ref.bench_turnover.mean() * MONTHS),
    }
    out["active"] = {
        lab: active_report(ref.net[lab], ref.bench_net[lab]) for lab in COST_BRACKETS
    }
    out["decades"] = {
        lab: decade_sharpe(ref.net[lab]) for lab in COST_BRACKETS
    }
    out["decades_bench"] = decade_sharpe(ref.bench_net["10bps"])

    # D5 split
    out["split_2009"] = {}
    for lab in COST_BRACKETS:
        s = ref.net[lab]
        bmk = ref.bench_net[lab]
        pre, post = s[s.index < SPLIT_DATE], s[s.index >= SPLIT_DATE]
        bpre, bpost = bmk[bmk.index < SPLIT_DATE], bmk[bmk.index >= SPLIT_DATE]
        out["split_2009"][lab] = {
            "pre": {**_series_stats(pre), "active": active_report(pre, bpre)},
            "post": {**_series_stats(post), "active": active_report(post, bpost)},
        }

    # Concentration and breadth
    out["concentration"] = {**concentration(ref.pnl), **year_concentration(ref.pnl)}

    # Structural tilt. The 6 price indices exclude dividends and DAX does not, and the
    # dividend yield differs by market (UK/AU ~4%, US ~2%), so a 5-year RETURN score
    # mechanically ranks high-yield markets cheap. If the book is a static long-FTSE /
    # short-DAX position that is a data convention, not a value effect -- this is the
    # table that shows it.
    live_w = ref.weights.loc[ref.gross.index]
    out["structural_tilt"] = {
        "mean_weight": {c: float(live_w[c].mean()) for c in live_w.columns},
        "mean_abs_weight": {c: float(live_w[c].abs().mean()) for c in live_w.columns},
        "pnl_by_instrument": {c: float(ref.pnl[c].sum()) for c in ref.pnl.columns},
        "pct_months_long": {
            c: (float(100.0 * (live_w[c] > 0).sum() / max((live_w[c] != 0).sum(), 1)))
            for c in live_w.columns
        },
        "mean_weight_over_mean_abs_weight": {
            c: (float(live_w[c].mean() / live_w[c].abs().mean())
                if live_w[c].abs().mean() > 0 else float("nan"))
            for c in live_w.columns
        },
    }
    live_x = x.loc[ref.gross.index]
    out["breadth"] = {
        "mean_tradable_instruments": float(ref.eligible_count.loc[ref.gross.index].mean()),
        "min_tradable_instruments": int(ref.eligible_count.loc[ref.gross.index].min()),
        "max_tradable_instruments": int(ref.eligible_count.loc[ref.gross.index].max()),
        "mean_live_blocks": float(ref.live_blocks.loc[ref.gross.index].mean()),
        "effective_n": effective_n(live_x.corr()),
    }

    # D2 -- per-block sub-books (each needs >=2 blocks, so a single block is run by
    # relaxing MIN_BLOCKS is NOT done; instead each block is run against itself with the
    # book-on rule satisfied by construction below).
    out["per_block"] = {}
    for b in BLOCKS:
        sub = _run_single_block(b, x, spreads, vol_target=0.20)
        if sub is None:
            out["per_block"][b] = None
            continue
        out["per_block"][b] = {
            "months": int(len(sub.net["10bps"])),
            "start": str(sub.net["10bps"].index.min().date()),
            "net_sharpe_10bps": annual_sharpe(sub.net["10bps"]),
            "net_sharpe_2bps": annual_sharpe(sub.net["2bps"]),
            "gross_sharpe": annual_sharpe(sub.gross),
            "tstat_net_10bps": newey_west_tstat(sub.net["10bps"]),
            "corr_to_primary": _corr(sub.net["10bps"], ref.net["10bps"]),
        }

    # D3 -- skip-12m reversal
    d3 = run_value(ValueConfig(name="D3_SKIP12", skip_months=SKIP_MONTHS),
                   vol_target=0.20, x=x, spreads=spreads)
    out["d3_skip12"] = {
        "months": int(len(d3.net["10bps"])),
        "net_sharpe_10bps": annual_sharpe(d3.net["10bps"]),
        "gross_sharpe": annual_sharpe(d3.gross),
        "active_10bps": active_report(d3.net["10bps"], d3.bench_net["10bps"]),
        "corr_to_primary": _corr(d3.net["10bps"], ref.net["10bps"]),
    }

    # D4 -- uniform reversal for bonds too
    d4 = run_value(ValueConfig(name="D4_UNIFORM_RATES", uniform_rates=True),
                   vol_target=0.20, x=x, spreads=spreads)
    out["d4_uniform_rates"] = {
        "months": int(len(d4.net["10bps"])),
        "net_sharpe_10bps": annual_sharpe(d4.net["10bps"]),
        "gross_sharpe": annual_sharpe(d4.gross),
        "active_10bps": active_report(d4.net["10bps"], d4.bench_net["10bps"]),
        "corr_to_primary": _corr(d4.net["10bps"], ref.net["10bps"]),
    }

    # D1 -- negative control
    controls = []
    for seed in range(8):
        c = run_value(ValueConfig(name=f"D1_SEED{seed}", randomise_seed=seed),
                      vol_target=0.20, x=x, spreads=spreads)
        controls.append(annual_sharpe(c.net["10bps"]))
    arr = np.array([v for v in controls if np.isfinite(v)], dtype=float)
    live_sharpe = annual_sharpe(ref.net["10bps"])
    out["negative_control"] = {
        "sharpes": [round(float(v), 4) for v in controls],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "live_z_vs_control": (
            float((live_sharpe - arr.mean()) / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else float("nan")
        ),
    }

    # Kelly
    out["kelly"] = {
        lab: {**kelly_report(annual_sharpe(ref.net[lab]))} for lab in COST_BRACKETS
    }

    # ── The headline: correlation to the other sleeves ────────────────────────
    corr_block: dict = {}
    trend = _read_sibling_series(_TREND_SERIES)
    if trend is not None:
        mine = ref.net["10bps"]
        a, bser = mine.align(trend, join="inner")
        s_v, s_t = annual_sharpe(a), annual_sharpe(bser)
        rho = float(a.corr(bser))
        eq = combined_sharpe_equal_risk(s_v, s_t, rho)
        opt = combined_sharpe_optimal(s_v, s_t, rho)
        corr_block["trend"] = {
            "source": str(_TREND_SERIES).replace("\\", "/"),
            "overlap_months": int(len(a)),
            "overlap_start": str(a.index.min().date()),
            "overlap_end": str(a.index.max().date()),
            "correlation": rho,
            "value_sharpe_on_overlap": s_v,
            "trend_sharpe_on_overlap": s_t,
            "combined_equal_risk_sharpe": eq,
            "combined_optimal_sharpe": opt,
            "combined_equal_risk_half_kelly": 3.0 * eq ** 2 / 8.0 if np.isfinite(eq) else float("nan"),
            "trend_alone_half_kelly": 3.0 * s_t ** 2 / 8.0 if np.isfinite(s_t) else float("nan"),
            "corr_of_d3_skip12_to_trend": _corr(d3.net["10bps"], trend),
            "corr_of_gross_to_trend": _corr(ref.gross, trend),
        }
    carry = _find_carry_series()
    if carry is not None:
        path, cs = carry
        a, bser = ref.net["10bps"].align(cs, join="inner")
        rho = float(a.corr(bser))
        rates_block = (_run_single_block("rates", x, spreads, vol_target=0.20)
                       if out["per_block"].get("rates") else None)
        corr_block["carry"] = {
            "source": path,
            "overlap_months": int(len(a)),
            "correlation": rho,
            "value_sharpe_on_overlap": annual_sharpe(a),
            "carry_sharpe_on_overlap": annual_sharpe(bser),
            "rates_block_corr_to_carry": (
                _corr(rates_block.net["10bps"], cs) if rates_block is not None
                else float("nan")
            ),
        }
    out["sleeve_correlations"] = corr_block

    # Receipts
    frame = pd.DataFrame(
        {
            "net_10bps": ref.net["10bps"],
            "net_2bps": ref.net["2bps"],
            "gross": ref.gross,
            "bench_net_10bps": ref.bench_net["10bps"],
        }
    )
    frame.index.name = "date"
    frame.to_csv(out_dir / "primary_20pct_monthly.csv")
    (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def _corr(a: pd.Series, b: pd.Series) -> float:
    x1, x2 = a.align(b, join="inner")
    if len(x1) < 12:
        return float("nan")
    return float(x1.corr(x2))


def _run_single_block(
    block: str, x: pd.DataFrame, spreads: pd.DataFrame, *, vol_target: float
) -> ValueResult | None:
    """D2 helper: run one block alone.

    The two-block rule would switch a one-block book off, so the sub-book relaxes
    ``min_blocks`` to 1 -- this is a DIAGNOSTIC and can never become the headline
    (prereg §8).
    """
    cols = [k for k in BLOCKS[block] if k in x.columns]
    if len(cols) < MIN_PER_BLOCK:
        return None
    r = run_value(
        ValueConfig(name=f"D2_{block.upper()}", blocks=(block,), min_blocks=1),
        vol_target=vol_target,
        x=x,
        spreads=spreads,
    )
    return r if len(r.net["10bps"]) > 24 else None


if __name__ == "__main__":  # pragma: no cover
    result = run_study()
    print(json.dumps(result, indent=2, default=str))
