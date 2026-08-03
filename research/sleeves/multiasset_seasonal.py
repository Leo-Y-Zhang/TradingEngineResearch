"""Sleeve: CALENDAR SEASONALITY on the long-history multi-asset panel.

Pre-registered in ``research/sleeves/multiasset_seasonal_prereg.md``, which was committed
BEFORE this file existed. Read it first. Every window, universe, constant and statistic here
is fixed there; nothing is searched, and no effect discovered in this panel may be reported.

Three pre-specified, cited effects:

* **E1 turn-of-the-month** — long on the last business day of the month and the first three
  of the next (Ariel 1987; Lakonishok & Smidt 1988).
* **E2 Halloween / sell-in-May** — long November-April (Bouman & Jacobsen 2002).
* **E3 January** — long the equity block in January (Rozeff & Kinney 1976; Keim 1983).
* **E4** — the equal-risk composite of the three. This is the sleeve headline.

The sizing machinery is imported UNCHANGED from ``multiasset_trend`` so that the only
difference between this sleeve and that one is the signal.

The signal is a function of the DATE ALONE. That is what licenses the use of the daily panel
despite ``data_integrity.md`` §10.1 (see prereg §2a): the session-overlap lookahead is about
signals that read a same-day price, and this one reads no price at all.
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
from research.sleeves.multiasset_trend import (
    BLOCKS,
    CASH_SUBTRACTED,
    PRIMARY_UNIVERSE,
    concentration,
    decade_sharpe,
    kelly_report,
    load_excess_panel,
    max_drawdown,
    newey_west_tstat,
)

__all__ = [
    "EFFECTS",
    "PUBLICATION_YEAR",
    "SeasonalResult",
    "build_signal",
    "load_daily_excess_panel",
    "run_effect",
    "run_composite",
    "main",
]

# ── Pre-registered constants (prereg §4, §5, §8) ──────────────────────────────
# Copied from multiasset_trend.py. NOT re-tuned.

VOL_WINDOW = 36            # months, instrument volatility
VOL_MIN_OBS = 24
ELIGIBLE_MIN_OBS = 36
MIN_INSTRUMENTS = 3
BOOK_VOL_WINDOW = 36
BOOK_VOL_MIN = 12
UNIT_VOL = 0.10
GROSS_CAP = 10.0
VOL_TARGETS: tuple[float, ...] = (0.10, 0.20, 0.40)
COST_BRACKETS: dict[str, float] = {"2bps": 0.0002, "10bps": 0.0010}
NW_LAG = 6
MONTHS = 12

# prereg §1: the three effects and the publication year each is split at (§7.2)
EFFECTS: tuple[str, ...] = ("E1_TOM", "E2_HALLOWEEN", "E3_JANUARY")
PUBLICATION_YEAR: dict[str, int] = {"E1_TOM": 1987, "E2_HALLOWEEN": 2002, "E3_JANUARY": 1976}

HALLOWEEN_MONTHS = frozenset({11, 12, 1, 2, 3, 4})
TOM_TAIL = 1               # last N business days of the month
TOM_HEAD = 3               # first N business days of the month
PLACEBO_DAYS = (10, 11, 12, 13)   # prereg §7.11 — the mid-month interior placebo

# trial counts, prereg §8
N_TRIALS_TABLE: tuple[int, ...] = (32, 44, 56, 304)

_DATA = Path("_data/multiasset")
_OUT = Path("research/sleeves/_seasonal")


# ── Statistics ────────────────────────────────────────────────────────────────

def sharpe(x: pd.Series, periods: int = MONTHS) -> float:
    a = pd.Series(x).dropna()
    if len(a) < 8 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * math.sqrt(periods))


def geometric_annual(x: pd.Series, periods: int = MONTHS) -> float:
    a = pd.Series(x).dropna()
    if a.empty:
        return float("nan")
    return float(np.expm1(np.log1p(a).mean() * periods))


def to_monthly(daily: pd.Series) -> pd.Series:
    """Compound a daily return series to calendar month end (exact, not approximate)."""
    a = pd.Series(daily).dropna()
    if a.empty:
        return a
    out = (1.0 + a).groupby(a.index.to_period("M")).prod() - 1.0
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out


def active_report(strat: pd.Series, bench: pd.Series) -> dict[str, float]:
    """Statistics A / B / C of prereg §6, plus the variance-drag identity.

    C is the mandated comparison: the benchmark LEVERED to the strategy's own volatility,
    ``strat - bench * sd_s/sd_b``. Its t-stat is identical to the equivalent form that
    scales the strategy down instead, which is exactly why it is the statistic that decides
    — unlike the raw arithmetic active t-stat, which the carry study measured to be a pure
    leverage dial.
    """
    a, b = pd.Series(strat).align(pd.Series(bench), join="inner")
    a, b = a.dropna(), b.dropna()
    a, b = a.align(b, join="inner")
    if len(a) < 12:
        return {"months": int(len(a))}
    d = a - b
    var_s, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    sd_s, sd_b = math.sqrt(var_s), math.sqrt(var_b)
    lever = (sd_s / sd_b) if sd_b > 0 else float("nan")
    vm = a - b * lever                       # benchmark levered TO the strategy's vol

    X = np.column_stack([np.ones(len(b)), b.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, a.to_numpy(dtype=float), rcond=None)
    resid = a.to_numpy(dtype=float) - X @ coef

    return {
        "months": int(len(d)),
        "geometric_excess_annual": geometric_annual(a) - geometric_annual(b),
        "arith_active_annual": float(d.mean() * MONTHS),
        "arith_active_tstat": newey_west_tstat(d, NW_LAG),
        "volmatched_active_annual": float(vm.mean() * MONTHS),
        "volmatched_active_tstat": newey_west_tstat(vm, NW_LAG),
        "bench_leverage_applied": lever,
        "variance_drag_annual": float((var_s - var_b) / 2.0 * MONTHS),
        "jensen_alpha_annual": float(coef[0] * MONTHS),
        "jensen_beta": float(coef[1]),
        "jensen_alpha_tstat": newey_west_tstat(pd.Series(resid + coef[0], index=a.index), NW_LAG),
        "strat_vol": sd_s * math.sqrt(MONTHS),
        "bench_vol": sd_b * math.sqrt(MONTHS),
        "strat_sharpe": sharpe(a),
        "bench_sharpe": sharpe(b),
    }


def era_split(x: pd.Series, year: int) -> dict[str, Any]:
    """Pre-publication vs post-publication Sharpe (prereg §7.2)."""
    a = pd.Series(x).dropna()
    pre = a[a.index.year < year]
    post = a[a.index.year >= year]
    out = {
        "split_year": int(year),
        "pre_months": int(len(pre)), "post_months": int(len(post)),
        "pre_sharpe": sharpe(pre), "post_sharpe": sharpe(post),
        "pre_mean_annual": float(pre.mean() * MONTHS) if len(pre) else float("nan"),
        "post_mean_annual": float(post.mean() * MONTHS) if len(post) else float("nan"),
        "pre_tstat": newey_west_tstat(pre, NW_LAG) if len(pre) >= 8 else float("nan"),
        "post_tstat": newey_west_tstat(post, NW_LAG) if len(post) >= 8 else float("nan"),
    }
    if np.isfinite(out["pre_sharpe"]) and np.isfinite(out["post_sharpe"]):
        out["decay_sharpe_points"] = out["post_sharpe"] - out["pre_sharpe"]
        out["survived_publication"] = bool(out["post_sharpe"] >= 0.5 * out["pre_sharpe"])
    return out


# ── Panel loading ─────────────────────────────────────────────────────────────

def load_daily_excess_panel(
    *, unscreened: bool = False, universe: tuple[str, ...] = PRIMARY_UNIVERSE,
    data_dir: Path = _DATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily excess returns for ``universe``, plus the interior-null mask.

    Same convention as ``load_excess_panel``: the three bond total-return series have the
    13-week bill accrual subtracted, everything else is already an excess return. Interior
    nulls (an instrument that exists but did not print that day) become a **zero return**
    with the position held through — you cannot trade a closed market. Leading nulls stay
    NaN so eligibility can see them.
    """
    name = "returns_daily_unscreened.parquet" if unscreened else "returns_daily.parquet"
    rets = pd.read_parquet(data_dir / name)
    cash = pd.read_parquet(data_dir / "cash_daily.parquet")["US_CASH_13W"]

    missing = [k for k in universe if k not in rets.columns]
    if missing:
        raise KeyError(f"daily panel is missing pre-registered instruments: {missing}")

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


# ── The signal: a function of the DATE ALONE (prereg §3) ──────────────────────

def build_signal(
    effect: str, index: pd.DatetimeIndex, columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """``{0,1}`` long-flat indicator. Reads no price, no return and no volatility.

    The turn-of-month window is defined on the pure Monday-Friday ``bdate_range`` grid
    (prereg §2b) — no holiday knowledge, no data knowledge, so it is unambiguously ex ante.
    """
    idx = pd.DatetimeIndex(index)
    cols = list(columns)

    if effect in ("E1_TOM", "PLACEBO_MIDMONTH", "E1_TOM_TRADINGDAY"):
        if effect == "E1_TOM_TRADINGDAY":
            raise ValueError("the observed-trading-day variant is built by _tom_trading_day")
        grid = pd.bdate_range(idx.min(), idx.max())
        rank = pd.Series(grid, index=grid).groupby(grid.to_period("M")).rank(method="first")
        size = pd.Series(1, index=grid).groupby(grid.to_period("M")).transform("size")
        if effect == "E1_TOM":
            on = (rank <= TOM_HEAD) | (rank > size - TOM_TAIL)
        else:
            on = rank.isin(PLACEBO_DAYS)
        flag = on.reindex(idx).fillna(False).astype(float)
        return pd.DataFrame({c: flag for c in cols}, index=idx)

    if effect == "E2_HALLOWEEN":
        flag = pd.Series(idx.month, index=idx).isin(HALLOWEEN_MONTHS).astype(float)
        return pd.DataFrame({c: flag for c in cols}, index=idx)

    if effect == "E3_JANUARY":
        flag = pd.Series(idx.month == 1, index=idx).astype(float)
        eq = set(BLOCKS["equity"])
        return pd.DataFrame(
            {c: (flag if c in eq else pd.Series(0.0, index=idx)) for c in cols}, index=idx,
        )

    raise ValueError(f"unknown effect {effect!r}")


def _tom_trading_day(x: pd.DataFrame) -> pd.DataFrame:
    """Secondary S3: the TOM window on each instrument's OWN observed trading days."""
    out = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    for c in x.columns:
        d = x[c].dropna().index
        if len(d) == 0:
            continue
        r = pd.Series(d, index=d).groupby(d.to_period("M")).rank(method="first")
        n = pd.Series(1, index=d).groupby(d.to_period("M")).transform("size")
        on = ((r <= TOM_HEAD) | (r > n - TOM_TAIL)).astype(float)
        out[c] = on.reindex(x.index).fillna(0.0)
    return out


# ── Sizing (prereg §4) ────────────────────────────────────────────────────────

def _monthly_risk(xm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lagged instrument vol and eligibility, both observable at the previous month end."""
    sigma = xm.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std(ddof=1) * math.sqrt(MONTHS)
    counted = xm.notna().cumsum()
    eligible = ((counted >= ELIGIBLE_MIN_OBS) & sigma.notna() & (sigma > 0)).astype(float)
    return sigma.shift(1), eligible.shift(1).fillna(0.0)


def _to_days(monthly: pd.DataFrame | pd.Series, days: pd.DatetimeIndex):
    """Broadcast a month-end-indexed frame onto the daily grid by calendar month."""
    m = monthly.copy()
    m.index = pd.DatetimeIndex(m.index).to_period("M")
    return m.reindex(days.to_period("M")).set_axis(days)


def _daily_risk(xm: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    sigma_lag, elig_lag = _monthly_risk(xm)
    sigma_d = _to_days(sigma_lag, days)
    elig_d = _to_days(elig_lag, days).fillna(0.0) > 0.5
    return sigma_d, elig_d


def _positions(sig: pd.DataFrame, sigma_d: pd.DataFrame, elig_d: pd.DataFrame) -> pd.DataFrame:
    """Unscaled notionals ``n_i(d) = sig_i(d) * (0.10 / sigma_i)`` on eligible names only."""
    n = sig.where(elig_d, 0.0) * (UNIT_VOL / sigma_d.where(elig_d))
    return n.fillna(0.0).where(elig_d.sum(axis=1) >= MIN_INSTRUMENTS, 0.0)


def _book_scaler(b_daily: pd.Series, gross_unit_d: pd.Series, vol_target: float,
                 no_vol_estimate: str = REGISTERED_NO_ESTIMATE):
    """Causal book-vol scaler: trailing 36 MONTHS of the unscaled book, applied next month.

    ``min`` skips NaN, so before 12 months of book history exist the GROSS CAP alone sets
    the scale. That is the trend sleeve's behaviour, reproduced deliberately rather than
    changed, and both counts are reported. It is now the shared
    `research.book_scaler`, so the fall-through can be SWITCHED as well as counted;
    `REGISTERED_NO_ESTIMATE` is the default and reproduces the banked run bit-for-bit.
    """
    bm = to_monthly(b_daily)
    sig_b = bm.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)
    sig_b = sig_b.replace(0.0, np.nan).shift(1)
    k_raw = vol_target / sig_b
    k_raw_d = _to_days(pd.DataFrame({"k": k_raw}), b_daily.index)["k"]
    k_cap = GROSS_CAP / gross_unit_d.replace(0.0, np.nan)
    scaler = book_scaler(k_raw_d, k_cap, no_estimate=no_vol_estimate)
    return scaler.k, scaler.cap_binding, scaler.no_estimate


# ── One effect, one vol target ────────────────────────────────────────────────

@dataclass
class SeasonalResult:
    effect: str
    vol_target: float
    gross_d: pd.Series
    net_d: dict[str, pd.Series]
    bench_gross_d: pd.Series
    bench_net_d: dict[str, pd.Series]
    turnover_d: pd.Series
    weights: pd.DataFrame
    pnl_d: pd.DataFrame
    days_in_market: float
    cap_binding_days: int
    gross_leverage: pd.Series
    stats: dict = field(default_factory=dict)

    @property
    def gross_m(self) -> pd.Series:
        return to_monthly(self.gross_d)

    def net_m(self, label: str) -> pd.Series:
        return to_monthly(self.net_d[label])

    def bench_net_m(self, label: str) -> pd.Series:
        return to_monthly(self.bench_net_d[label])


def _assemble(
    n: pd.DataFrame, xd: pd.DataFrame, elig_d: pd.DataFrame, sigma_d: pd.DataFrame,
    vol_target: float, effect: str,
) -> SeasonalResult:
    """Apply the book scaler, the benchmark and the cost bracket to a position matrix.

    **A flat day is a ZERO, not a missing observation.** A seasonal sleeve is out of the
    market most of the time; dropping those days would annualise the Sharpe of the days it
    happens to be in, which for the January leg would inflate it by sqrt(12). The sample
    window therefore runs from the first day the book carries risk to the last complete
    calendar month, and every day inside it counts — zero included. The benchmark is held
    on EVERY day of that same window (prereg §6), not only on the days the sleeve trades.
    """
    tradable = elig_d.sum(axis=1) >= MIN_INSTRUMENTS
    gross_unit = n.abs().sum(axis=1)
    b_raw = (n * xd).sum(axis=1).where(tradable)          # flat days contribute exactly 0
    k, binding, undefined = _book_scaler(b_raw, gross_unit, vol_target)

    w = n.mul(k, axis=0).fillna(0.0)
    carries_risk = tradable & (w.abs().sum(axis=1) > 0)
    if not bool(carries_risk.any()):
        raise ValueError(f"{effect}: the book never carries risk")
    first = carries_risk.idxmax()
    start = (pd.Timestamp(first) + pd.offsets.MonthBegin(1)).normalize()
    last_full = (pd.Timestamp(xd.index.max()).normalize()
                 - pd.offsets.MonthBegin(1) + pd.offsets.MonthEnd(0))
    window = tradable & (xd.index >= start) & (xd.index <= last_full)

    pnl = w * xd
    gross = pnl.sum(axis=1).where(window)
    turnover = w.diff().fillna(w).abs().sum(axis=1).where(window)
    gross_lev = w.abs().sum(axis=1).where(window)

    # Benchmark: equal-weight LONG-ONLY over the same eligible set, held EVERY day of the
    # window, rebalanced monthly. Levered by a full-sample constant to the strategy's own
    # vol (prereg §6). The leverage is in-sample and disclosed; it is applied to the
    # BENCHMARK, so it cannot manufacture alpha for the strategy.
    nb_count = elig_d.sum(axis=1)
    wb_unit = elig_d.astype(float).div(nb_count.replace(0, np.nan), axis=0).fillna(0.0)
    wb_unit = wb_unit.where(tradable, 0.0)
    bench_unit = (wb_unit * xd).sum(axis=1).where(window)
    bench_to_unit = wb_unit.diff().fillna(wb_unit).abs().sum(axis=1).where(window)

    sm = to_monthly(gross.dropna())
    bm = to_monthly(bench_unit.dropna())
    a, bb = sm.align(bm, join="inner")
    lever = float(a.std(ddof=1) / bb.std(ddof=1)) if bb.std(ddof=1) > 0 else 1.0
    bench_gross = bench_unit * lever
    bench_to = bench_to_unit * lever

    net, bench_net = {}, {}
    for label, c in COST_BRACKETS.items():
        net[label] = gross - 0.5 * c * turnover
        bench_net[label] = bench_gross - 0.5 * c * bench_to

    return SeasonalResult(
        effect=effect, vol_target=vol_target,
        gross_d=gross.dropna(), net_d={k_: v.dropna() for k_, v in net.items()},
        bench_gross_d=bench_gross.dropna(),
        bench_net_d={k_: v.dropna() for k_, v in bench_net.items()},
        turnover_d=turnover.dropna(), weights=w, pnl_d=pnl.where(window).dropna(how="all"),
        days_in_market=float((gross_lev.dropna() > 0).mean()),
        cap_binding_days=int(binding.sum()),
        gross_leverage=gross_lev.dropna(),
        stats={"bench_leverage": lever,
               "cap_only_days_no_bookvol": int(undefined.sum())},
    )


def run_effect(
    effect: str, *, vol_target: float = 0.20,
    xd: pd.DataFrame, xm: pd.DataFrame, long_short: bool = False,
    signal_override: pd.DataFrame | None = None,
) -> SeasonalResult:
    sigma_d, elig_d = _daily_risk(xm, xd.index)

    sig = signal_override if signal_override is not None else build_signal(
        effect, xd.index, list(xd.columns))
    if long_short:
        sig = 2.0 * sig - 1.0
        if effect == "E3_JANUARY":              # only the equity block is ever traded
            for c in xd.columns:
                if c not in set(BLOCKS["equity"]):
                    sig[c] = 0.0

    n = _positions(sig, sigma_d, elig_d)
    return _assemble(n, xd, elig_d, sigma_d, vol_target, effect)


def run_composite(
    *, vol_target: float = 0.20, xd: pd.DataFrame, xm: pd.DataFrame,
    long_short: bool = False,
) -> SeasonalResult:
    """E4: equal-RISK composite. Each leg is divided by its own causal trailing book vol."""
    sigma_d, elig_d = _daily_risk(xm, xd.index)

    legs = []
    for eff in EFFECTS:
        sig = build_signal(eff, xd.index, list(xd.columns))
        if long_short:
            sig = 2.0 * sig - 1.0
            if eff == "E3_JANUARY":
                for c in xd.columns:
                    if c not in set(BLOCKS["equity"]):
                        sig[c] = 0.0
        n_j = _positions(sig, sigma_d, elig_d)
        b_j = (n_j * xd).sum(axis=1)
        s_j = to_monthly(b_j).rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN)
        s_j = (s_j.std(ddof=1) * math.sqrt(MONTHS)).replace(0.0, np.nan).shift(1)
        s_jd = _to_days(pd.DataFrame({"s": s_j}), xd.index)["s"]
        legs.append(n_j.mul((UNIT_VOL / s_jd), axis=0).fillna(0.0))

    n = sum(legs) / float(len(legs))
    return _assemble(n, xd, elig_d, sigma_d, vol_target, "E4_COMPOSITE")


# ── Reporting ─────────────────────────────────────────────────────────────────

def summarise(res: SeasonalResult) -> dict:
    out: dict = {
        "effect": res.effect, "vol_target": res.vol_target,
        "days": int(len(res.gross_d)),
        "days_in_market_pct": round(100.0 * res.days_in_market, 2),
        "cap_binding_days": res.cap_binding_days,
        "mean_gross_leverage": float(res.gross_leverage.mean()),
        "bench_leverage": res.stats.get("bench_leverage"),
        "turnover_per_year": float(res.turnover_d.mean() * 252.0),
    }
    gm = res.gross_m
    out["months"] = int(len(gm))
    out["years"] = round(len(gm) / 12.0, 2)
    out["gross"] = {
        "sharpe": sharpe(gm), "vol": float(gm.std(ddof=1) * math.sqrt(MONTHS)),
        "mean_annual": float(gm.mean() * MONTHS), "geo_annual": geometric_annual(gm),
        "tstat": newey_west_tstat(gm, NW_LAG),
    }
    for label in COST_BRACKETS:
        nm = res.net_m(label)
        bm = res.bench_net_m(label)
        out[f"net_{label}"] = {
            "sharpe": sharpe(nm), "vol": float(nm.std(ddof=1) * math.sqrt(MONTHS)),
            "mean_annual": float(nm.mean() * MONTHS), "geo_annual": geometric_annual(nm),
            "tstat": newey_west_tstat(nm, NW_LAG), "max_drawdown": max_drawdown(nm),
            "skew": float(nm.skew()), "worst_month": float(nm.min()),
            "bench_sharpe": sharpe(bm),
            "active": active_report(nm, bm),
            "decades": decade_sharpe(nm),
            "kelly": kelly_report(sharpe(nm)),
        }
    # Breakeven round-trip cost (prereg §5)
    mg, mt = float(res.gross_d.mean()), float(res.turnover_d.mean())
    out["breakeven_roundtrip_bps"] = float(2.0 * mg / mt * 1e4) if mt > 0 else float("inf")
    if res.effect in PUBLICATION_YEAR:
        yr = PUBLICATION_YEAR[res.effect]
        nm, bm = res.net_m("10bps"), res.bench_net_m("10bps")
        a, b = nm.align(bm, join="inner")
        lev = float(a.std(ddof=1) / b.std(ddof=1)) if b.std(ddof=1) > 0 else 1.0
        out["publication_split"] = era_split(nm, yr)
        # The raw split confounds the calendar edge with the market drift of the era. The
        # split of the VOL-MATCHED ACTIVE series (statistic C, prereg §6) does not: it is
        # the sleeve minus its own levered long-only universe, era by era. Both are
        # reported; the active one is the one that answers the question.
        out["publication_split_active"] = era_split(a - b * lev, yr)
        out["publication_split_benchmark"] = era_split(b, yr)
    conc_src = res.pnl_d.groupby(res.pnl_d.index.to_period("M")).sum()
    conc_src.index = conc_src.index.to_timestamp(how="end").normalize()
    out["concentration"] = concentration(conc_src)
    by_inst = res.pnl_d.sum(axis=0)
    tot = float(by_inst.sum())
    out["per_instrument_pnl_share"] = (
        {k: round(float(v / tot), 4) for k, v in by_inst.items()} if tot != 0 else {})
    out["per_block_pnl_share"] = (
        {blk: round(float(by_inst.reindex([c for c in keys if c in by_inst.index]).sum() / tot), 4)
         for blk, keys in BLOCKS.items()} if tot != 0 else {})
    return out


def _combo(series: dict[str, pd.Series], point_in_time: bool = False) -> dict:
    """Equal-risk combination of monthly sleeve returns, on the common overlap."""
    df = pd.concat(series, axis=1).dropna()
    if len(df) < 24:
        return {"months": int(len(df))}
    if point_in_time:
        vol = df.rolling(36, min_periods=12).std(ddof=1).shift(1)
        wts = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
        live = wts.notna().sum(axis=1)
        comb = ((df * wts).sum(axis=1) / live.replace(0, np.nan)).dropna()
    else:
        comb = (df / df.std(ddof=1)).mean(axis=1)
    # The combination is in risk units, so its level is arbitrary. Rescale to a 10% annual
    # volatility before any path statistic — a drawdown on an unscaled risk-unit series is
    # a meaningless number, not a small one.
    comb = comb * (0.10 / (comb.std(ddof=1) * math.sqrt(MONTHS)))
    s = sharpe(comb)
    return {
        "months": int(len(comb)), "sharpe": s,
        "tstat": newey_west_tstat(comb, NW_LAG),
        "max_drawdown_at_10pct_vol": max_drawdown(comb),
        "half_kelly_growth": 3.0 * s ** 2 / 8.0,
        "decades": {k: v["sharpe"] for k, v in decade_sharpe(comb).items()},
        "start": str(comb.index.min().date()), "end": str(comb.index.max().date()),
    }


def _formula_combo(s: float, rhos: list[float]) -> float:
    """S = s*sqrt(N/(1+(N-1)*rho_bar)) — the brief's check formula, N from len(rhos)+1."""
    n = len(rhos) + 1
    rho = float(np.mean(rhos))
    denom = 1.0 + (n - 1) * rho
    return float(s * math.sqrt(n / denom)) if denom > 0 else float("nan")


# ── Verification: positive controls that prove the machinery, prereg §7.11 ────

def verify(xd: pd.DataFrame, xm: pd.DataFrame) -> dict:
    v: dict = {}

    # 1. DSR bar reproduces both recorded anchors.
    v["dsr_anchor_7yr"] = round(dsr_sharpe_bar(7.0, n_trials=32), 4)
    v["dsr_anchor_40yr"] = round(dsr_sharpe_bar(40.0, n_trials=32), 4)
    v["dsr_anchors_ok"] = bool(abs(v["dsr_anchor_7yr"] - 1.4881) < 5e-4
                               and abs(v["dsr_anchor_40yr"] - 0.5971) < 5e-4)

    rng = np.random.default_rng(20260728)
    sd, ed = _daily_risk(xm, xd.index)

    # 2. The POSITION MATRIX does not depend on any return. Rebuilt against a panel of pure
    #    noise (same index, same columns) it must be bit-identical: positions are a function
    #    of (calendar, lagged monthly volatility) and nothing else.
    noise = pd.DataFrame(rng.normal(0.0, 0.01, xd.shape), index=xd.index, columns=xd.columns)
    same = all(
        _positions(build_signal(e, xd.index, list(xd.columns)), sd, ed).equals(
            _positions(build_signal(e, noise.index, list(noise.columns)), sd, ed))
        for e in EFFECTS)
    v["positions_independent_of_returns"] = bool(same)

    # 3. Perfect-foresight positive control on the identical pipeline: replace the calendar
    #    indicator with the sign of the day's realised return. If the pipeline cannot make
    #    money with tomorrow's newspaper, a negative result would be a bug, not a finding.
    oracle = pd.DataFrame(np.sign(xd.to_numpy()), index=xd.index, columns=xd.columns)
    n_or = _positions(oracle, sd, ed)
    r_or = _assemble(n_or, xd, ed, sd, 0.20, "ORACLE")
    v["oracle_gross_sharpe"] = round(sharpe(r_or.gross_m), 3)
    v["oracle_net_10bps_sharpe"] = round(sharpe(r_or.net_m("10bps")), 3)
    v["oracle_control_ok"] = bool(v["oracle_gross_sharpe"] > 3.0)

    # 4. Daily -> monthly compounding is exact.
    probe = pd.Series(rng.normal(0, 0.01, 500),
                      index=pd.bdate_range("2000-01-03", periods=500))
    direct = float((1.0 + probe["2000-01-01":"2000-01-31"]).prod() - 1.0)
    v["compound_exact"] = bool(abs(to_monthly(probe).iloc[0] - direct) < 1e-12)

    # 5. The TOM window really has the pre-registered shape: exactly 4 flagged Mon-Fri days
    #    per month on the pure business-day grid, in every month of the sample.
    grid = pd.bdate_range("1965-01-01", "2026-06-30")
    s = build_signal("E1_TOM", grid, ["X"])["X"]
    per_month = s.groupby(grid.to_period("M")).sum()
    v["tom_days_per_month_min"] = int(per_month.min())
    v["tom_days_per_month_max"] = int(per_month.max())
    v["tom_window_shape_ok"] = bool(per_month.min() == 4 and per_month.max() == 4)

    # 6. Halloween really flags 6 months a year and January 1.
    hs = build_signal("E2_HALLOWEEN", grid, ["X"])["X"]
    v["halloween_months_per_year"] = int(round(
        hs.groupby(grid.year).apply(lambda z: len(z[z > 0].index.to_period("M").unique())).mean()))
    v["halloween_shape_ok"] = bool(v["halloween_months_per_year"] == 6)

    # 7. DATE-SCRAMBLE control. Permute the mapping between calendar dates and daily
    #    returns (positions are untouched, since they do not read returns). This destroys
    #    every calendar effect while preserving the return distribution and the sleeve's
    #    long-only, part-time-in-market shape. What survives is the market DRIFT the window
    #    would have earned with no calendar information at all — which is the right baseline
    #    for "did the calendar add anything", strictly stronger than a zero baseline.
    scram: dict[str, list[float]] = {e: [] for e in ("E1_TOM", "E2_HALLOWEEN", "E4_COMPOSITE")}
    for seed in (11, 22, 33, 44):
        r2 = np.random.default_rng(seed)
        xs = xd.copy()
        for c in xs.columns:                    # permute WITHIN each column's own live dates,
            live_idx = xs.index[xs[c].notna()]  # so every instrument's availability window
            vals = xs.loc[live_idx, c].to_numpy()   # and NaN pattern are preserved exactly
            xs.loc[live_idx, c] = vals[r2.permutation(len(vals))]
        for e in ("E1_TOM", "E2_HALLOWEEN"):
            n_s = _positions(build_signal(e, xd.index, list(xd.columns)), sd, ed)
            scram[e].append(sharpe(_assemble(n_s, xs, ed, sd, 0.20, e).gross_m))
        scram["E4_COMPOSITE"].append(sharpe(run_composite(vol_target=0.20, xd=xs, xm=xm).gross_m))
    v["date_scramble_drift_baseline"] = {
        k: {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4),
            "runs": [round(float(z), 4) for z in vals]} for k, vals in scram.items()}

    v["all_ok"] = bool(v["dsr_anchors_ok"] and v["positions_independent_of_returns"]
                       and v["oracle_control_ok"] and v["compound_exact"]
                       and v["tom_window_shape_ok"] and v["halloween_shape_ok"])
    return v


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    _OUT.mkdir(parents=True, exist_ok=True)
    xm, _ = load_excess_panel()
    xd, _ = load_daily_excess_panel()

    report: dict = {"universe": list(PRIMARY_UNIVERSE),
                    "daily_rows": int(len(xd)), "monthly_rows": int(len(xm))}
    report["verification"] = verify(xd, xm)

    # ---- primaries at three vol targets -------------------------------------
    results: dict[str, dict[float, SeasonalResult]] = {}
    for eff in EFFECTS:
        results[eff] = {vt: run_effect(eff, vol_target=vt, xd=xd, xm=xm) for vt in VOL_TARGETS}
    results["E4_COMPOSITE"] = {vt: run_composite(vol_target=vt, xd=xd, xm=xm)
                               for vt in VOL_TARGETS}
    report["primaries"] = {e: {f"{vt:.0%}": summarise(r) for vt, r in d.items()}
                           for e, d in results.items()}

    head = results["E4_COMPOSITE"][0.20]
    years = len(head.gross_m) / 12.0
    report["sample_years"] = round(years, 2)
    report["dsr_bars"] = {str(n): round(dsr_sharpe_bar(years, n_trials=n), 4)
                          for n in N_TRIALS_TABLE}

    # The pre/post-publication confound, quantified rather than waved at: the eligible
    # universe is far smaller in the early eras, so an era comparison is not ceteris paribus.
    _, elig_lag = _monthly_risk(xm)
    ec = elig_lag.sum(axis=1)
    report["eligible_instruments_by_era"] = {
        f"{d}s": round(float(ec[(ec.index.year // 10) * 10 == d].mean()), 2)
        for d in sorted({(y // 10) * 10 for y in ec.index.year})}
    report["eligible_instruments_at_split"] = {
        e: {"pre_mean": round(float(ec[ec.index.year < PUBLICATION_YEAR[e]].mean()), 2),
            "post_mean": round(float(ec[ec.index.year >= PUBLICATION_YEAR[e]].mean()), 2)}
        for e in EFFECTS}
    report["benchmark_dsr"] = {
        "bench_net_10bps_sharpe": sharpe(head.bench_net_m("10bps")),
        "clears_bar_at_n_trials": {
            str(n): bool(sharpe(head.bench_net_m("10bps")) >= dsr_sharpe_bar(years, n_trials=n))
            for n in N_TRIALS_TABLE},
    }

    # ---- negative control: mid-month placebo --------------------------------
    plc = run_effect("PLACEBO_MIDMONTH", vol_target=0.20, xd=xd, xm=xm,
                     signal_override=build_signal("PLACEBO_MIDMONTH", xd.index, list(xd.columns)))
    report["placebo_midmonth"] = {
        "gross_sharpe": sharpe(plc.gross_m), "net_10bps_sharpe": sharpe(plc.net_m("10bps")),
        "live_E1_gross_sharpe": sharpe(results["E1_TOM"][0.20].gross_m),
        "live_E1_net_10bps_sharpe": sharpe(results["E1_TOM"][0.20].net_m("10bps")),
    }

    # ---- correlations to the real trend and carry sleeves -------------------
    trend = pd.read_csv("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv",
                        parse_dates=["date"]).set_index("date")["net_10bps"]
    carry = pd.read_parquet(
        "research/sleeves/_carry_output/carry_primary_net_monthly.parquet")["net"]
    seas = head.net_m("10bps")

    def _rho(a: pd.Series, b: pd.Series) -> dict:
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        return {"rho": float(j["a"].corr(j["b"])), "months": int(len(j)),
                "start": str(j.index.min().date()) if len(j) else None,
                "end": str(j.index.max().date()) if len(j) else None}

    report["correlations"] = {
        "seasonal_vs_trend": _rho(seas, trend),
        "seasonal_vs_carry": _rho(seas, carry),
        "trend_vs_carry": _rho(trend, carry),
        "per_effect_vs_trend": {e: _rho(results[e][0.20].net_m("10bps"), trend)
                                for e in EFFECTS},
        "per_effect_vs_carry": {e: _rho(results[e][0.20].net_m("10bps"), carry)
                                for e in EFFECTS},
    }

    # ---- portfolio arithmetic ------------------------------------------------
    report["portfolio"] = {
        "trend_carry_equal_risk": _combo({"trend": trend, "carry": carry}),
        "trend_carry_seasonal_equal_risk": _combo(
            {"trend": trend, "carry": carry, "seasonal": seas}),
        "trend_carry_seasonal_pit_risk_parity": _combo(
            {"trend": trend, "carry": carry, "seasonal": seas}, point_in_time=True),
        "trend_seasonal_equal_risk": _combo({"trend": trend, "seasonal": seas}),
        "formula_check_3_sleeve": _formula_combo(
            float(np.mean([sharpe(trend), sharpe(carry), sharpe(seas)])),
            [report["correlations"]["seasonal_vs_trend"]["rho"],
             report["correlations"]["seasonal_vs_carry"]["rho"],
             report["correlations"]["trend_vs_carry"]["rho"]]),
        "sleeve_sharpes": {"trend": sharpe(trend), "carry": sharpe(carry),
                           "seasonal": sharpe(seas)},
    }

    # ---- secondaries ---------------------------------------------------------
    eq = tuple(BLOCKS["equity"])
    xm_eq, _ = load_excess_panel(universe=eq)
    xd_eq, _ = load_daily_excess_panel(universe=eq)
    s1 = {e: summarise(run_effect(e, vol_target=0.20, xd=xd_eq, xm=xm_eq))
          for e in ("E1_TOM", "E2_HALLOWEEN")}

    s2 = {e: summarise(run_effect(e, vol_target=0.20, xd=xd, xm=xm, long_short=True))
          for e in EFFECTS}
    s2["E4_COMPOSITE"] = summarise(run_composite(vol_target=0.20, xd=xd, xm=xm,
                                                 long_short=True))
    ls_seas = run_composite(vol_target=0.20, xd=xd, xm=xm, long_short=True).net_m("10bps")
    s2["rho_vs_trend"] = _rho(ls_seas, trend)
    s2["rho_vs_carry"] = _rho(ls_seas, carry)

    s3 = summarise(run_effect("E1_TOM", vol_target=0.20, xd=xd, xm=xm,
                              signal_override=_tom_trading_day(
                                  pd.read_parquet(_DATA / "returns_daily.parquet")[
                                      list(PRIMARY_UNIVERSE)])))
    try:
        xm_u, _ = load_excess_panel(unscreened=True)
        xd_u, _ = load_daily_excess_panel(unscreened=True)
        s4 = summarise(run_composite(vol_target=0.20, xd=xd_u, xm=xm_u))
    except Exception as exc:                       # noqa: BLE001 - reported, not swallowed
        s4 = {"error": f"{type(exc).__name__}: {exc}"}

    report["secondaries"] = {"S1_equity_only": s1, "S2_long_short": s2,
                             "S3_trading_day_calendar": s3, "S4_unscreened": s4}

    # ---- artefacts -----------------------------------------------------------
    out_m = pd.DataFrame({
        "seasonal_net_10bps": head.net_m("10bps"),
        "seasonal_net_2bps": head.net_m("2bps"),
        "seasonal_gross": head.gross_m,
        "bench_net_10bps": head.bench_net_m("10bps"),
    })
    out_m.to_parquet(_OUT / "seasonal_composite_20pct_monthly.parquet")
    for e in EFFECTS:
        pd.DataFrame({"net_10bps": results[e][0.20].net_m("10bps")}).to_parquet(
            _OUT / f"{e.lower()}_20pct_monthly.parquet")

    (_OUT / "result.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps(r["verification"], indent=2))
    print("\nsample_years", r["sample_years"], "dsr_bars", r["dsr_bars"])
    for eff, d in r["primaries"].items():
        s = d["20%"]
        print(f"\n=== {eff} @20% vol ===")
        print("  months", s["months"], "days_in_mkt%", s["days_in_market_pct"],
              "turnover/yr", round(s["turnover_per_year"], 2),
              "breakeven_bps", round(s["breakeven_roundtrip_bps"], 1))
        print("  gross S", round(s["gross"]["sharpe"], 4))
        for lab in ("2bps", "10bps"):
            n = s[f"net_{lab}"]
            a = n["active"]
            print(f"  net {lab}: S={n['sharpe']:.4f} bench_S={n['bench_sharpe']:.4f} "
                  f"volmatched_active={a.get('volmatched_active_annual', float('nan')):+.4%} "
                  f"t={a.get('volmatched_active_tstat', float('nan')):+.3f}")
        if "publication_split" in s:
            p = s["publication_split"]
            print(f"  pub split {p['split_year']}: pre S={p['pre_sharpe']:.4f} "
                  f"({p['pre_months']}m)  post S={p['post_sharpe']:.4f} ({p['post_months']}m)")
    print("\ncorrelations", json.dumps(r["correlations"], indent=2, default=str))
    print("\nportfolio", json.dumps(r["portfolio"], indent=2, default=str))
    print("\nplacebo", json.dumps(r["placebo_midmonth"], indent=2, default=str))
