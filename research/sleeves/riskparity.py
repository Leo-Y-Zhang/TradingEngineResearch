"""Sleeve: RISK PARITY on the long-history multi-asset panel.

Pre-registered in ``research/sleeves/riskparity_prereg.md`` (commit ``d895110``, written
BEFORE this file existed). Run ONCE, no tuning.

The question is not "is there alpha here" -- there is no signal in a risk-parity book, only
a sizing rule. The question the prereg fixes is:

    what is the highest COMPOUND RETURN achievable at a MAX DRAWDOWN the account survives?

Everything in this module exists to answer that honestly: an explicit financing charge on
the levered notional, drawdown reported in the same row as the growth rate it took to get
there, and a bond-bull-market exclusion because a 40-year bond bull flatters any bond-heavy
risk-parity book.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.sleeves.multiasset_trend import (
    BLOCKS,
    BOOK_VOL_MIN,
    BOOK_VOL_WINDOW,
    ELIGIBLE_MIN_OBS,
    GROSS_CAP,
    MIN_INSTRUMENTS,
    MONTHS,
    annual_sharpe,
    inverse_vol,
)

__all__ = [
    "VOL_TARGETS",
    "COSTS",
    "FINANCING",
    "LEGACY_FLAT",
    "BOND_BULL",
    "Book",
    "build_book",
    "weights_ew",
    "weights_rp_naive",
    "weights_rp_bucket",
    "levered",
    "drawdown_report",
    "ladder",
    "weight_concentration",
]

# ── Pre-registered constants (prereg §4, §5) ──────────────────────────────────

VOL_TARGETS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
COSTS: dict[str, float] = {"2bps": 0.0002, "10bps": 0.0010}
FINANCING: dict[str, float] = {          # spread over the 13-week bill, annual
    "primary_bill_plus_150bp": 0.0150,
    "optimistic_bill_plus_50bp": 0.0050,
    "retail_bill_plus_300bp": 0.0300,
}
LEGACY_FLAT = 0.06                       # flat all-in borrow rate used by prior repo work
BOND_BULL = ("1981-10-01", "2021-12-31")
RATES_KEYS: tuple[str, ...] = BLOCKS["rates"]

_DATA = Path("_data/multiasset")
_OUT = Path("research/sleeves/_riskparity")


# ── Eligibility and weights (prereg §4) ───────────────────────────────────────

def eligibility(x: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """Eligible at decision time ``t``: >=36 observations AND a positive trailing vol."""
    counted = x.notna().cumsum()
    return (counted >= ELIGIBLE_MIN_OBS) & sigma.notna() & (sigma > 0)


def _renorm(w: pd.DataFrame) -> pd.DataFrame:
    s = w.sum(axis=1)
    return w.div(s.replace(0.0, np.nan), axis=0).fillna(0.0)


def weights_ew(elig: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """W0 -- equal weight over the eligible set."""
    return _renorm(elig.astype(float))


def weights_rp_naive(elig: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    """W1 -- ``w_i`` proportional to ``1/sigma_i``, long only, sums to 1."""
    inv = (1.0 / sigma).where(elig)
    return _renorm(inv.fillna(0.0))


def weights_rp_bucket(
    elig: pd.DataFrame,
    sigma: pd.DataFrame,
    x: pd.DataFrame,
    blocks: dict[str, tuple[str, ...]] = BLOCKS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """W2 -- two-level risk parity: inverse-vol inside a block, inverse-vol across blocks.

    Returns ``(weights, block_weights, block_returns)``. The block volatility is estimated
    from the block sub-portfolio's OWN realised history, accumulated point-in-time: the
    weights ``u`` decided at ``t-1`` are applied to the returns of ``t``, exactly as the
    real book would have been held.
    """
    cols_by_block = {b: [k for k in keys if k in x.columns] for b, keys in blocks.items()}
    u = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    live_b = pd.DataFrame(False, index=x.index, columns=list(blocks))
    for b, cols in cols_by_block.items():
        if not cols:
            continue
        inv = (1.0 / sigma[cols]).where(elig[cols]).fillna(0.0)
        u[cols] = _renorm(inv)
        live_b[b] = elig[cols].sum(axis=1) > 0

    xz = x.fillna(0.0)
    rb = pd.DataFrame(index=x.index, columns=list(blocks), dtype=float)
    for b, cols in cols_by_block.items():
        if not cols:
            continue
        r = (u[cols].shift(1) * xz[cols]).sum(axis=1)
        rb[b] = r.where(live_b[b].shift(1).fillna(False))

    sig_b = rb.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)

    inv_b = (1.0 / sig_b.replace(0.0, np.nan)).where(live_b)
    n_live = live_b.sum(axis=1)
    # Pre-registered fallback: if ANY live block has no volatility estimate yet, all live
    # blocks are weighted equally that month.
    have_all = (inv_b.notna().sum(axis=1) == n_live) & (n_live > 0)
    eq = live_b.astype(float).div(n_live.replace(0, np.nan), axis=0)
    W = _renorm(inv_b.fillna(0.0)).where(have_all, eq).fillna(0.0)

    w = pd.DataFrame(0.0, index=x.index, columns=x.columns)
    for b, cols in cols_by_block.items():
        if cols:
            w[cols] = u[cols].mul(W[b], axis=0)
    return _renorm(w), W, rb


# ── The unlevered book ────────────────────────────────────────────────────────

@dataclass
class Book:
    name: str
    w: pd.DataFrame          # weights DECIDED at t (row t), sum to 1 when live
    x: pd.DataFrame          # excess returns
    excess: pd.Series        # unlevered book excess return earned DURING t
    live: pd.Series
    sigma_book: pd.Series    # trailing 36m annualised vol of ``excess`` through t
    elig_count: pd.Series
    extra: dict


def build_book(
    name: str,
    x: pd.DataFrame,
    interior: pd.DataFrame,
    scheme: str,
    blocks: dict[str, tuple[str, ...]] = BLOCKS,
) -> Book:
    """Build one unlevered long-only book. ``scheme`` in {'ew', 'rp_naive', 'rp_bucket'}."""
    sigma = inverse_vol(x)
    elig = eligibility(x, sigma)
    # No position decided into a month whose return is an interior null (trend convention).
    elig = elig & ~interior.reindex_like(elig).fillna(False)
    elig = elig.where(elig.sum(axis=1) >= MIN_INSTRUMENTS, False)

    extra: dict = {}
    if scheme == "ew":
        w = weights_ew(elig, sigma)
    elif scheme == "rp_naive":
        w = weights_rp_naive(elig, sigma)
    elif scheme == "rp_bucket":
        w, W, rb = weights_rp_bucket(elig, sigma, x, blocks)
        extra["block_weights"] = W
        extra["block_returns"] = rb
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    xz = x.fillna(0.0)
    held = w.shift(1).fillna(0.0)
    live = held.abs().sum(axis=1) > 0
    excess = (held * xz).sum(axis=1).where(live)
    sigma_book = (
        excess.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)
    )
    return Book(name, w, x, excess, live, sigma_book, elig.sum(axis=1), extra)


# ── Leverage, financing and costs (prereg §5) ─────────────────────────────────

def levered(
    bk: Book,
    cash: pd.Series,
    *,
    tau: float,
    cost: float,
    spread: float | None,
    flat_rate: float | None = None,
    k_override: pd.Series | None = None,
    cap: float = GROSS_CAP,
) -> dict:
    """Lever ``bk`` to ``tau`` and charge trading costs and financing explicitly.

    ``spread`` is an annual spread over the bill rate charged on ``max(k-1, 0)``; the bill
    itself is already embedded because the panel holds EXCESS returns. ``flat_rate``
    instead charges a flat all-in borrow rate, which is what prior work in this repo did:
    the adjustment is ``(flat/12 - cash_t)`` per unit of levered notional, and it is a
    SUBSIDY whenever the bill rate exceeded ``flat`` (the 1970s and 1980s).
    """
    if k_override is not None:
        k_raw = k_override.reindex(bk.x.index)
        k = k_raw.copy()
        cap_binding = pd.Series(False, index=bk.x.index)
    else:
        k_raw = tau / bk.sigma_book.replace(0.0, np.nan)
        k = k_raw.clip(upper=cap)
        cap_binding = (k_raw > cap).fillna(False)

    wl = bk.w.mul(k, axis=0)
    held = wl.shift(1).fillna(0.0)
    live = held.abs().sum(axis=1) > 0
    pnl = held * bk.x.fillna(0.0)
    gross_excess = pnl.sum(axis=1).where(live)

    turnover = held.diff().abs().sum(axis=1).where(live)
    cost_s = 0.5 * cost * turnover

    k_held = k.shift(1).where(live)
    borrowed = (k_held - 1.0).clip(lower=0.0)
    cash_a = cash.reindex(bk.x.index).astype(float)
    if flat_rate is not None:
        fin = (flat_rate / MONTHS - cash_a) * borrowed
    else:
        fin = (float(spread or 0.0) / MONTHS) * borrowed

    net_excess = (gross_excess - cost_s - fin).dropna()
    total = (net_excess + cash_a.reindex(net_excess.index)).dropna()
    return {
        "gross_excess": gross_excess.dropna(),
        "net_excess": net_excess,
        "total": total,
        "k": k_held.dropna(),
        "turnover": turnover.dropna(),
        "cost": cost_s.dropna(),
        "financing": fin.dropna(),
        "cap_binding": cap_binding.reindex(net_excess.index).fillna(False),
        "pnl": pnl.reindex(net_excess.index),
        "weights_held": held.reindex(net_excess.index),
    }


# ── Drawdown, recovery, compounding ───────────────────────────────────────────

def _months_between(idx: pd.Index, a, b) -> int:
    return int(idx.get_loc(b) - idx.get_loc(a))


def drawdown_report(total: pd.Series) -> dict:
    """Max drawdown on the compounded TOTAL-return path, plus time to recover.

    Ruin is reported, never silently compounded through: a monthly return of -100% or worse
    ends the account, and no later gain undoes it.
    """
    r = total.dropna()
    if r.empty:
        return {"max_drawdown": float("nan")}
    ruin = bool((r <= -1.0).any())
    ruin_date = str(r.index[(r <= -1.0).to_numpy().argmax()].date()) if ruin else None
    if ruin:
        r = r.loc[: r.index[(r <= -1.0).to_numpy().argmax()]]

    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    mdd = float(dd.min())
    trough = dd.idxmin()
    pk = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    hit = after[after >= float(curve.loc[pk])]
    rec = hit.index[0] if len(hit) else None

    underwater = (dd < -1e-12).astype(int)
    runs, cur = [], 0
    for v in underwater.to_numpy():
        cur = cur + 1 if v else 0
        runs.append(cur)
    n = len(r)
    yrs = n / MONTHS
    comp = -1.0 if ruin else float(curve.iloc[-1] ** (1.0 / yrs) - 1.0)
    return {
        "months": n,
        "years": yrs,
        "compound_annual": comp,
        "vol_annual": float(r.std(ddof=1) * math.sqrt(MONTHS)),
        "max_drawdown": mdd,
        "dd_peak": str(pd.Timestamp(pk).date()),
        "dd_trough": str(pd.Timestamp(trough).date()),
        "dd_recovery": str(pd.Timestamp(rec).date()) if rec is not None else None,
        "months_peak_to_trough": _months_between(curve.index, pk, trough),
        "months_trough_to_recovery": (
            _months_between(curve.index, trough, rec) if rec is not None else None
        ),
        "months_underwater": (
            _months_between(curve.index, pk, rec) if rec is not None
            else int(_months_between(curve.index, pk, curve.index[-1]))
        ),
        "recovered": rec is not None,
        "longest_underwater_months": int(max(runs) if runs else 0),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
        "ruin": ruin,
        "ruin_date": ruin_date,
    }


# ── Concentration (prereg §7.5) ───────────────────────────────────────────────

def weight_concentration(w: pd.DataFrame, live: pd.Series) -> dict:
    ww = w.loc[live.reindex(w.index).fillna(False)]
    ww = ww.loc[ww.abs().sum(axis=1) > 0]
    if ww.empty:
        return {}
    arr = np.sort(ww.to_numpy(), axis=1)[:, ::-1]
    top1 = arr[:, 0]
    top3 = arr[:, :3].sum(axis=1)
    top5 = arr[:, :5].sum(axis=1)
    eff_n = 1.0 / (ww.to_numpy() ** 2).sum(axis=1)
    blocks = {
        b: float(ww[[k for k in keys if k in ww.columns]].sum(axis=1).mean())
        for b, keys in BLOCKS.items()
    }
    return {
        "top1_mean": float(top1.mean()), "top1_max": float(top1.max()),
        "top3_mean": float(top3.mean()), "top3_max": float(top3.max()),
        "top5_mean": float(top5.mean()),
        "effective_n_mean": float(eff_n.mean()), "effective_n_min": float(eff_n.min()),
        "block_share_mean": blocks,
        "mean_instrument_weight_by_key": {
            str(c): float(ww[c].mean()) for c in ww.columns
        },
        "months": int(len(ww)),
    }


# ── The ladder (prereg §6) ────────────────────────────────────────────────────

def ladder(bk: Book, cash: pd.Series, *, cost_label: str, financing_label: str) -> dict:
    cost = COSTS[cost_label]
    spread = FINANCING[financing_label]
    out: dict = {}
    for tau in VOL_TARGETS:
        L = levered(bk, cash, tau=tau, cost=cost, spread=spread)
        rep = drawdown_report(L["total"])
        rep.update({
            "target_vol": tau,
            "sharpe_net": annual_sharpe(L["net_excess"]),
            "sharpe_gross": annual_sharpe(L["gross_excess"]),
            "mean_leverage": float(L["k"].mean()),
            "max_leverage": float(L["k"].max()),
            "cap_binding_months": int(L["cap_binding"].sum()),
            "turnover_per_year": float(L["turnover"].mean() * MONTHS),
            "cost_drag_annual": float(L["cost"].mean() * MONTHS),
            "financing_drag_annual": float(L["financing"].mean() * MONTHS),
        })
        out[f"{tau:.2f}"] = rep
    return out
