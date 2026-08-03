"""Sleeve: DEFENSIVE / BETTING-AGAINST-BETA on the long-history multi-asset panel.

Pre-registered in ``research/sleeves/multiasset_defensive_prereg.md``. Run ONCE, no
tuning. Read that file before this one; every choice here is fixed there and nothing is
searched.

Frazzini & Pedersen, *Betting Against Beta*, JFE 111(1), 2014. Leverage-constrained
investors bid up high-beta assets, flattening the security market line, so a book that is
long low-beta and short high-beta with each leg scaled to unit beta earns the difference.

Three things this module is careful about
=========================================
1. **The beta neutralisation IS the strategy.** A naive long-low / short-high book is a
   short position in the panel proxy wearing a costume. The hedge ratio ``rho`` is
   computed from the realised leg betas every month, and where it is clipped the clip is
   COUNTED and the book's REALISED beta is reported rather than assumed to be zero.
2. **Convention parity with trend and carry.** The excess-return panel is loaded through
   ``research.sleeves.multiasset_trend.load_excess_panel`` and every reporting statistic
   is that module's own function. A correlation between two sleeves computed on two
   different return conventions is not a correlation between two sleeves.
3. **Mechanical vs economic correlation.** The 36-month beta window CONTAINS all four of
   trend's lookbacks. The value sleeve's headline diversification was destroyed by exactly
   this, so arm S3 re-estimates every position input on months ``t-47..t-12`` — nothing in
   that book has seen the 12 months trend's signal is computed from — and the change in
   correlation is the study's headline number.
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
from research.multiasset.panel import dsr_sharpe_bar
from research.validation import deflated_sharpe_ratio

# Deliberately shared with the trend sleeve so the three books are measured identically.
from research.sleeves.multiasset_trend import (
    BLOCKS,
    MONTHS,
    PRIMARY_UNIVERSE,
    annual_sharpe,
    kelly_report,
    load_excess_panel,
    max_drawdown,
    newey_west_tstat,
)

__all__ = [
    "BETA_WINDOW",
    "BETA_MIN_OBS",
    "COST_BRACKETS",
    "DefensiveConfig",
    "DefensiveResult",
    "GROSS_CAP",
    "MIN_BETA_SPREAD",
    "MIN_INSTRUMENTS",
    "N_TRIALS",
    "VOL_TARGETS",
    "panel_proxy",
    "rolling_beta",
    "trailing_vol",
    "run_defensive",
    "realised_beta",
    "combined_sharpe",
    "kelly_reality",
    "year_concentration",
    "describe",
]

# ── Pre-registered constants (prereg §2, §3) ──────────────────────────────────

BETA_WINDOW = 36           # months, trailing beta estimate
BETA_MIN_OBS = 24
VOL_WINDOW = 36            # months, instrument volatility (inverse-vol sizing)
VOL_MIN_OBS = 24
HISTORY_MIN_OBS = 36       # months of own history before an instrument is eligible
MIN_INSTRUMENTS = 6        # panel-wide book is OFF below this
MIN_BLOCK_INSTRUMENTS = 3  # within-block arm
MIN_BETA_SPREAD = 0.10     # betaH - betaL below this ⇒ no bet
MIN_HIGH_BETA = 0.05       # a "high-beta" leg with no beta is not a leg
RHO_CAP = 3.0
BOOK_VOL_WINDOW = 36
BOOK_VOL_MIN = 12
GROSS_CAP = 10.0           # x book equity
VOL_TARGETS: tuple[float, ...] = (0.10, 0.20, 0.40)
COST_BRACKETS: dict[str, float] = {"2bps": 0.0002, "10bps": 0.0010}
OVERLAP_LAG = 12           # months excised in arm S3 (= trend's longest lookback)
PLACEBO_SEED = 20260728
N_TRIALS = 38              # programme-cumulative, per the mission brief

_DATA = Path("_data/multiasset")


# ── Signal inputs ─────────────────────────────────────────────────────────────

def _history_count(x: pd.DataFrame) -> pd.DataFrame:
    """Months of OBSERVED history for each instrument as of each month (causal)."""
    return x.notna().cumsum()


def panel_proxy(x: pd.DataFrame, *, min_history: int = HISTORY_MIN_OBS) -> pd.Series:
    """Equal-weight return of every instrument with enough history, month by month.

    Point-in-time by construction: membership in month ``s`` depends only on whether the
    instrument had a return in ``s`` and how many months it had accumulated BY ``s``. The
    series is never revised, so a beta estimated against it in 1974 is the beta an
    investor could have estimated in 1974.
    """
    ok = x.notna() & (_history_count(x) >= int(min_history))
    masked = x.where(ok)
    n = ok.sum(axis=1)
    return masked.mean(axis=1).where(n > 0)


def rolling_beta(
    x: pd.DataFrame,
    proxy: pd.Series,
    *,
    window: int = BETA_WINDOW,
    min_obs: int = BETA_MIN_OBS,
) -> pd.DataFrame:
    """OLS slope of each instrument on ``proxy`` over the trailing ``window`` months.

    ``rolling(window)`` on month ``t`` covers months ``t-window+1 .. t`` INCLUSIVE, and
    positions built from it are held during ``t+1``, so nothing reads forward. Computed
    from pairwise-complete observations; fewer than ``min_obs`` pairs ⇒ NaN.
    """
    p = proxy.reindex(x.index)
    both = x.notna() & p.notna().to_numpy()[:, None]
    xv = x.where(both)
    pv = pd.DataFrame({c: p for c in x.columns}, index=x.index).where(both)

    n = both.rolling(window, min_periods=1).sum()
    sx = xv.rolling(window, min_periods=1).sum()
    sp = pv.rolling(window, min_periods=1).sum()
    sxp = (xv * pv).rolling(window, min_periods=1).sum()
    spp = (pv * pv).rolling(window, min_periods=1).sum()

    cov = sxp / n - (sx / n) * (sp / n)
    var = spp / n - (sp / n) ** 2
    beta = cov / var.where(var > 0)
    return beta.where(n >= int(min_obs))


def trailing_vol(
    x: pd.DataFrame, *, window: int = VOL_WINDOW, min_obs: int = VOL_MIN_OBS
) -> pd.DataFrame:
    """Annualised trailing volatility. Causal by construction."""
    return x.rolling(window, min_periods=min_obs).std(ddof=1) * math.sqrt(MONTHS)


# ── The book (prereg §3) ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class DefensiveConfig:
    name: str = "PRIMARY"
    within_block: bool = False
    hedged: bool = False
    overlap_lag: int = 0          # months to lag every position input (arm S3)
    unscreened: bool = False
    placebo_seed: int | None = None
    foresight: bool = False       # verification only: rank by NEXT month's return
    naive: bool = False           # verification only: no beta neutralisation (rho = 1)


@dataclass
class DefensiveResult:
    config: str
    gross: pd.Series
    net: dict[str, pd.Series]
    bench_gross: pd.Series
    bench_net: dict[str, pd.Series]
    weights: pd.DataFrame
    unscaled: pd.DataFrame
    turnover: pd.Series
    gross_leverage: pd.Series
    scaler: pd.Series
    cap_binding: pd.Series
    no_vol_estimate: pd.Series
    pnl: pd.DataFrame
    proxy: pd.Series
    beta: pd.DataFrame
    diagnostics: pd.DataFrame
    eligible: pd.DataFrame
    stats: dict = field(default_factory=dict)


def _rank_legs(
    rank_row: pd.Series, sigma_row: pd.Series, rng: np.random.Generator | None
) -> tuple[pd.Series, pd.Series]:
    """Frazzini-Pedersen rank weights, inverse-vol scaled inside the ranking.

    ``wL_i ∝ max(0, zbar - z_i)/sigma_i``, ``wH_i ∝ max(0, z_i - zbar)/sigma_i``, each
    normalised to sum to 1. The extremes still carry the most weight, but a 60%-vol
    commodity does not swamp a 5%-vol bond leg.
    """
    n = len(rank_row)
    z = rank_row.rank(method="first").astype(float)
    if rng is not None:                                   # arm S5: shuffle the ranks
        z = pd.Series(rng.permutation(z.to_numpy()), index=z.index)
    zbar = (n + 1) / 2.0
    inv = 1.0 / sigma_row

    lo = np.maximum(0.0, zbar - z) * inv
    hi = np.maximum(0.0, z - zbar) * inv
    lo_sum, hi_sum = float(lo.sum()), float(hi.sum())
    if lo_sum <= 0 or hi_sum <= 0:
        empty = pd.Series(0.0, index=rank_row.index)
        return empty, empty
    return lo / lo_sum, hi / hi_sum


def _bab_book(
    beta: pd.DataFrame,
    sigma: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    min_instruments: int,
    rng: np.random.Generator | None,
    rank_key: pd.DataFrame | None = None,
    naive: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One BAB book over ``keys``. Returns ``(unscaled weights, per-month diagnostics)``.

    ``rank_key`` replaces beta as the SORT key only (leg betas are still the real ones) --
    used by the verification script's perfect-foresight positive control. ``naive`` drops
    the beta neutralisation entirely (``rho = 1``), which is the straw man the
    pre-registration says is not this sleeve.
    """
    cols = [k for k in keys if k in beta.columns]
    rk = beta if rank_key is None else rank_key
    u = pd.DataFrame(0.0, index=beta.index, columns=beta.columns)
    diag = pd.DataFrame(
        np.nan, index=beta.index,
        columns=["n_elig", "betaL", "betaH", "rho", "rho_clipped_low",
                 "rho_clipped_high", "off_spread", "off_count", "book_beta"],
    )

    for t in beta.index:
        elig = eligible.loc[t, cols]
        live = [c for c in cols if bool(elig[c])]
        diag.loc[t, "n_elig"] = len(live)
        if len(live) < min_instruments:
            diag.loc[t, "off_count"] = 1.0
            continue
        b = beta.loc[t, live]
        s = sigma.loc[t, live]
        key = rk.loc[t, live]
        if key.isna().any():
            diag.loc[t, "off_count"] = 1.0
            continue
        wl, wh = _rank_legs(key, s, rng)
        if wl.sum() <= 0:
            diag.loc[t, "off_count"] = 1.0
            continue
        bl = float((wl * b).sum())
        bh = float((wh * b).sum())
        diag.loc[t, ["betaL", "betaH"]] = [bl, bh]
        if (bh - bl) < MIN_BETA_SPREAD or bh <= MIN_HIGH_BETA:
            diag.loc[t, "off_spread"] = 1.0
            continue
        raw = bl / bh
        rho = 1.0 if naive else float(min(max(raw, 0.0), RHO_CAP))
        diag.loc[t, "rho"] = rho
        diag.loc[t, "rho_clipped_low"] = float(raw < 0.0)
        diag.loc[t, "rho_clipped_high"] = float(raw > RHO_CAP)
        row = wl - rho * wh
        u.loc[t, live] = row.to_numpy()
        diag.loc[t, "book_beta"] = float((row * b).sum())

    return u, diag


def _positions(
    x: pd.DataFrame, cfg: DefensiveConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Unscaled book ``u`` decided AT month-end ``t``, plus every diagnostic it produced."""
    proxy = panel_proxy(x)
    beta = rolling_beta(x, proxy)
    sigma = trailing_vol(x)

    if cfg.overlap_lag:                       # arm S3 — nothing here has seen the last L months
        beta = beta.shift(cfg.overlap_lag)
        sigma = sigma.shift(cfg.overlap_lag)

    eligible = (
        (_history_count(x) >= HISTORY_MIN_OBS)
        & beta.notna() & sigma.notna() & (sigma > 0)
    )
    rng = np.random.default_rng(cfg.placebo_seed) if cfg.placebo_seed is not None else None
    # Verification-only control: rank by -(next month's return) so the "low-beta" leg is
    # by construction next month's winners. Leg betas stay real, so the machinery is
    # untouched and only the signal is replaced.
    rank_key = -x.shift(-1) if cfg.foresight else None

    if not cfg.within_block:
        u, diag = _bab_book(beta, sigma, eligible, keys=PRIMARY_UNIVERSE,
                            min_instruments=MIN_INSTRUMENTS, rng=rng,
                            rank_key=rank_key, naive=cfg.naive)
    else:
        # Each block's own BAB book, each scaled to unit trailing vol, equally weighted.
        parts: dict[str, pd.DataFrame] = {}
        # Per-block diagnostics are kept SEPARATE. Summing a beta across four blocks,
        # or averaging one measured against four different proxies, is not a statistic.
        blk_diags: list[pd.DataFrame] = []
        for name, keys in BLOCKS.items():
            cols = [k for k in keys if k in x.columns]
            bproxy = panel_proxy(x[cols])
            bbeta = rolling_beta(x[cols], bproxy).reindex(columns=x.columns)
            if cfg.overlap_lag:
                bbeta = bbeta.shift(cfg.overlap_lag)
            belig = eligible & bbeta.notna()
            ub, db = _bab_book(bbeta, sigma, belig, keys=tuple(cols),
                               min_instruments=MIN_BLOCK_INSTRUMENTS, rng=rng,
                               rank_key=rank_key, naive=cfg.naive)
            parts[name] = ub
            blk_diags.append(db.add_prefix(f"{name}_"))
        diag = pd.concat(blk_diags, axis=1)
        u = pd.DataFrame(0.0, index=x.index, columns=x.columns)
        live_blocks = pd.Series(0.0, index=x.index)
        xz = x.fillna(0.0)
        for name, ub in parts.items():
            rb = (ub.shift(1) * xz).sum(axis=1)
            vb = rb.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1)
            vb = vb.replace(0.0, np.nan)
            on = (ub.abs().sum(axis=1) > 0) & vb.notna()
            live_blocks += on.astype(float)
            u = u.add(ub.div(vb, axis=0).where(on, 0.0).fillna(0.0), fill_value=0.0)
        u = u.div(live_blocks.replace(0.0, np.nan), axis=0).fillna(0.0)
        diag["n_blocks_live"] = live_blocks

    if cfg.hedged:                            # arm S2 — force ex-ante zero beta
        bb = (u * beta.reindex_like(u)).sum(axis=1)
        pw = eligible.astype(float)
        pw = pw.div(pw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        u = u.sub(pw.mul(bb, axis=0), fill_value=0.0)

    return u, beta, sigma, proxy, eligible, diag


def run_defensive(
    cfg: DefensiveConfig,
    *,
    vol_target: float = 0.20,
    data_dir: Path = _DATA,
    x: pd.DataFrame | None = None,
    interior: pd.DataFrame | None = None,
    no_vol_estimate: str = REGISTERED_NO_ESTIMATE,
) -> DefensiveResult:
    """Run the pre-registered sleeve once, at one volatility target.

    `no_vol_estimate` defaults to the REGISTERED behaviour, which reproduces every banked
    number bit-for-bit. `NO_ESTIMATE_FLAT` is the repair for the book's first
    BOOK_VOL_MIN months, which otherwise run at the full gross cap.
    """
    if x is None or interior is None:
        x, interior = load_excess_panel(
            unscreened=cfg.unscreened, universe=PRIMARY_UNIVERSE, data_dir=data_dir
        )

    u, beta, sigma, proxy, eligible, diag = _positions(x, cfg)
    held = interior.reindex_like(u).fillna(False)
    u = u.mask(held, 0.0)                      # no position in a nulled month

    xz = x.fillna(0.0)
    pos = u.shift(1).fillna(0.0)
    b = (pos * xz).sum(axis=1)
    b = b.where(pos.abs().sum(axis=1) > 0)

    # Causal book-vol estimate at t, applied to t+1.
    sig_b = b.rolling(BOOK_VOL_WINDOW, min_periods=BOOK_VOL_MIN).std(ddof=1) * math.sqrt(MONTHS)
    sig_b = sig_b.replace(0.0, np.nan)
    k_raw = vol_target / sig_b
    gross_unit = u.abs().sum(axis=1)
    k_cap = GROSS_CAP / gross_unit.replace(0.0, np.nan)
    # k is set by the CAP whenever the vol-targeted scaler exceeds it -- and also in the
    # book's first BOOK_VOL_MIN months, when there is no volatility estimate at all and
    # k_raw is NaN so the min() silently falls through to the cap. That second case runs
    # the book at full GROSS_CAP leverage. This sleeve has always counted it; the shared
    # `research.book_scaler` now makes it switchable as well as visible, and this sleeve
    # keeps its own convention of folding it into `cap_binding`.
    scaler = book_scaler(k_raw, k_cap, no_estimate=no_vol_estimate, live=gross_unit > 0)
    k = scaler.k
    no_vol_est = scaler.no_estimate
    cap_binding = scaler.cap_or_no_estimate

    w = u.mul(k, axis=0).shift(1).fillna(0.0)  # weights HELD during month t
    live = w.abs().sum(axis=1) > 0

    pnl = w * xz
    gross = pnl.sum(axis=1).where(live)
    turnover = w.diff().abs().sum(axis=1).where(live)
    gross_lev = w.abs().sum(axis=1).where(live)

    # Benchmark: equal-weight LONG-ONLY over the same eligible set, same convention.
    elig_shift = eligible.shift(1).astype(float).fillna(0.0).astype(bool) & ~held
    nb = elig_shift.sum(axis=1)
    wb = elig_shift.astype(float).div(nb.replace(0, np.nan), axis=0).fillna(0.0)
    wb = wb.where(live.reindex(wb.index).fillna(False), 0.0)
    bench_gross = (wb * xz).sum(axis=1).where(live)
    bench_turnover = wb.diff().abs().sum(axis=1).where(live)

    net, bench_net = {}, {}
    for label, c in COST_BRACKETS.items():
        net[label] = gross - 0.5 * c * turnover
        bench_net[label] = bench_gross - 0.5 * c * bench_turnover

    return DefensiveResult(
        config=cfg.name,
        gross=gross.dropna(),
        net={k_: v.dropna() for k_, v in net.items()},
        bench_gross=bench_gross.dropna(),
        bench_net={k_: v.dropna() for k_, v in bench_net.items()},
        weights=w,
        unscaled=u,
        turnover=turnover.dropna(),
        gross_leverage=gross_lev.dropna(),
        scaler=k,
        # `shift(fill_value=False)` keeps the bool dtype; `.shift().fillna()` promoted it
        # to object and tripped a pandas downcasting FutureWarning. Same values.
        cap_binding=cap_binding.shift(1, fill_value=False).reindex(
            gross.index, fill_value=False).astype(bool),
        no_vol_estimate=no_vol_est.shift(1, fill_value=False).reindex(
            gross.index, fill_value=False).astype(bool),
        pnl=pnl,
        proxy=proxy,
        beta=beta,
        diagnostics=diag,
        eligible=eligible,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def realised_beta(strategy: pd.Series, proxy: pd.Series) -> dict[str, float]:
    """OLS beta of the realised book on the realised proxy. Claimed neutrality, audited.

    ``rho`` is clipped at zero when the low-beta leg's own beta is negative, so the book
    is NOT guaranteed neutral ex ante and is certainly not guaranteed neutral ex post.
    This is the number that says how far off it is.
    """
    a, p = strategy.align(proxy.dropna(), join="inner")
    if len(a) < 8:
        return {"beta": float("nan"), "alpha_annual": float("nan"), "r2": float("nan")}
    X = np.column_stack([np.ones(len(p)), p.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, a.to_numpy(dtype=float), rcond=None)
    resid = a.to_numpy(dtype=float) - X @ coef
    ss_tot = float(((a - a.mean()) ** 2).sum())
    dof = len(a) - 2
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se_beta = math.sqrt(s2 * float(xtx_inv[1, 1])) if np.isfinite(s2) else float("nan")
    return {
        "months": int(len(a)),
        "beta": float(coef[1]),
        "beta_se": se_beta,
        "beta_tstat_ols": float(coef[1] / se_beta) if se_beta > 0 else float("nan"),
        "alpha_annual": float(coef[0] * MONTHS),
        "alpha_tstat": newey_west_tstat(pd.Series(resid + coef[0], index=a.index)),
        "r2": float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
    }


def combined_sharpe(sharpes: dict[str, float], corr: pd.DataFrame) -> dict[str, Any]:
    """Equal-risk multi-sleeve Sharpe, plus a comparator that can actually disagree.

    ``formula_sharpe`` and ``exact_equal_risk_sharpe`` ARE THE SAME NUMBER, always. This
    docstring used to say they were reported "so the approximation error is visible rather
    than assumed away"; that was wrong, because the comparison detects nothing. For an
    equal-risk book the quadratic form sees only the mean correlation::

        rho_bar = mean of the off-diagonal entries of C
        1'C1    = n + (sum of ALL off-diagonals) = n(1 + (n-1)*rho_bar)   for ANY C
        exact   = sum(s)/sqrt(1'C1)                = sum(s)/sqrt(n(1 + (n-1)rho_bar))
        approx  = s_bar*sqrt(n/(1 + (n-1)rho_bar)) = sum(s)/sqrt(n(1 + (n-1)rho_bar))

    Neither documented assumption (equal Sharpes, equal pairwise correlations) is needed:
    the mean Sharpe and the sqrt(n) cancel. Both fields are KEPT -- each is the correct
    equal-risk Sharpe and published result files quote them -- but their agreement must
    never be cited as evidence that the brief's approximation was checked. Pinned in
    ``tests/test_defensive_combination.py``.

    ``optimal_sharpe`` is the real comparator (added 2026-08-01, closing that finding). It
    is the mean-variance optimal combination ``sqrt(s' C^-1 s)``, which reads the FULL
    correlation matrix instead of collapsing it to a mean, so it genuinely differs from the
    equal-risk figure -- by exactly what equal-risk weighting costs. Same convention as
    ``multiasset_value.combined_sharpe_optimal``, and the same caveat applies: **the optimal
    weights are in-sample and can be negative, so the equal-risk figure is the deployable
    one.** It is NaN when C is singular or ill-conditioned rather than a number built on a
    near-singular inverse.

    ``sharpe_dispersion`` and ``corr_dispersion`` report whether the brief's two assumptions
    actually hold on this input; both are exactly 0.0 when they do. Those are the
    diagnostics the vacuous comparison was standing in for.
    """
    names = [n for n in sharpes if n in corr.columns]
    s = np.array([sharpes[n] for n in names], dtype=float)
    C = corr.loc[names, names].to_numpy(dtype=float)
    n = len(names)
    off = C[np.triu_indices(n, 1)]
    rho_bar = float(off.mean()) if off.size else 0.0
    s_bar = float(s.mean())
    ones = np.ones(n)
    exact = float(s.sum() / math.sqrt(float(ones @ C @ ones)))
    approx = float(s_bar * math.sqrt(n / (1.0 + (n - 1) * rho_bar)))
    return {
        "sleeves": names,
        "mean_sharpe": s_bar,
        "mean_pairwise_corr": rho_bar,
        "formula_sharpe": approx,
        "exact_equal_risk_sharpe": exact,
        "optimal_sharpe": _optimal_sharpe(s, C),
        "sharpe_dispersion": float(s.std(ddof=0)) if s.size else 0.0,
        "corr_dispersion": float(off.std(ddof=0)) if off.size else 0.0,
        "half_kelly_growth_formula": 3.0 * approx ** 2 / 8.0,
        "half_kelly_growth_exact": 3.0 * exact ** 2 / 8.0,
    }


def _optimal_sharpe(s: np.ndarray, C: np.ndarray) -> float:
    """Mean-variance optimal combined Sharpe, ``sqrt(s' C^-1 s)``.

    NaN rather than a number whenever the answer would rest on a near-singular inverse: a
    perfectly correlated pair makes ``C^-1`` explode, and the "optimal" book is then an
    arbitrarily large offsetting position nobody can hold. Returning NaN keeps an
    unreachable number out of a result file, which is the same reason the equal-risk figure
    is the deployable one.
    """
    if s.size == 0:
        return float("nan")
    try:
        if not np.all(np.isfinite(C)) or np.linalg.cond(C) > 1e10:
            return float("nan")
        quad = float(s @ np.linalg.solve(C, s))
    except np.linalg.LinAlgError:
        return float("nan")
    if not np.isfinite(quad) or quad <= 0:
        return float("nan")
    return float(math.sqrt(quad))


def kelly_reality(returns: pd.Series) -> dict[str, float]:
    """Half-Kelly growth WITH the volatility, leverage and drawdown it actually requires.

    Standing rule adopted by the programme 2026-07-28 (internal research log correction): the bare
    ``g = 3S^2/8`` is correct arithmetic and misleading as a deployable number, because it
    silently assumes running at ``sigma = S/2``. A growth rate whose required leverage
    implies a drawdown beyond roughly 60% is not a reachable return, it is arithmetic.

    Drawdown is scaled LINEARLY with leverage, which is optimistic for a fat-tailed
    path-dependent series -- so the implied drawdown below is itself a floor.
    """
    a = returns.dropna()
    s = annual_sharpe(a)
    own_vol = float(a.std(ddof=1) * math.sqrt(MONTHS))
    own_dd = max_drawdown(a)
    if not np.isfinite(s) or own_vol <= 0:
        return {}
    req_vol = s / 2.0
    lev = req_vol / own_vol
    implied_dd = own_dd * lev
    return {
        "sharpe": s,
        "half_kelly_growth": 3.0 * s * s / 8.0,
        "required_vol": req_vol,
        "own_vol": own_vol,
        "leverage_on_own_vol": lev,
        "measured_max_drawdown": own_dd,
        "implied_max_drawdown_at_that_leverage": implied_dd,
        "survivable": bool(implied_dd > -0.60),
    }


def year_concentration(pnl: pd.DataFrame) -> dict[str, float]:
    by_year = pnl.sum(axis=1).groupby(pnl.index.year).sum()
    total = float(by_year.sum())
    if total == 0:
        return {"top_year_share": float("nan")}
    return {
        "top_year": int(by_year.idxmax()),
        "top_year_share": float(by_year.max() / total),
        "top_year_abs_share": float(by_year.abs().max() / by_year.abs().sum()),
        "n_positive_years": int((by_year > 0).sum()),
        "n_years": int(len(by_year)),
    }


def describe(r: pd.Series) -> dict[str, Any]:
    a = r.dropna()
    if len(a) < 8:
        return {}
    years = len(a) / MONTHS
    return {
        "months": int(len(a)),
        "years": round(years, 2),
        "first": str(a.index.min().date()),
        "last": str(a.index.max().date()),
        "arith_annual": float(a.mean() * MONTHS),
        "geom_annual": float(np.expm1(np.log1p(a).mean() * MONTHS)),
        "vol_annual": float(a.std(ddof=1) * math.sqrt(MONTHS)),
        "sharpe": annual_sharpe(a),
        "tstat_newey_west": newey_west_tstat(a),
        "max_drawdown": max_drawdown(a),
        "skew": float(a.skew()),
        "kurtosis": float(a.kurtosis()),
        "hit_rate": float((a > 0).mean()),
        "worst_month": float(a.min()),
        "best_month": float(a.max()),
        "dsr": deflated_sharpe_ratio(a.to_numpy(), n_trials=N_TRIALS),
        "dsr_bar_sharpe": dsr_sharpe_bar(years, n_trials=N_TRIALS),
        **kelly_report(annual_sharpe(a)),
    }
