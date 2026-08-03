"""Cross-asset CARRY: signals, sizing, backtest and the statistics that judge it.

Pure functions only — no network, no file IO, no globals — so every claim the study makes
is reproducible offline and testable. The network fetch lives in
``scripts/build_carry_inputs.py`` and the single registered run in
``scripts/run_multiasset_carry.py``.

Carry is defined once and applied to every asset class the same way: **the annualised
return a position earns if prices do not move.**

* rates  — constant-maturity yield minus the 13-week bill yield
* FX     — foreign 3-month interbank rate minus the US 3-month interbank rate
* equity — trailing 12-month *realised* dividend yield minus the 13-week bill yield

The three traps this module exists to avoid
===========================================
1. **FX spot is not FX carry.** A long-foreign forward earns the spot move *plus* the
   interest differential. Ranking on the differential while paying only the spot move
   measures the opposite of the strategy. ``fx_excess_returns`` adds the differential,
   lagged, and says so.
2. **Geometric excess lies.** geometric = arithmetic − (σ²_a − σ²_b)/2, so a
   lower-volatility book shows a fake positive excess. Every statistic here is
   arithmetic, with a Newey–West t-statistic.
3. **A carry book can be all accrual.** ``decompose_pnl`` splits gross P&L into the
   deterministic carry accrual and the price move, because a sleeve whose accrual is
   exactly cancelled by its price leg has found nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FRED_SHORT_RATES",
    "OECD_DATAFLOW",
    "OECD_OVERNIGHT_MEASURE",
    "OECD_SHORT_MEASURE",
    "OECD_SHORT_RATES",
    "FX_INSTRUMENTS",
    "FxInstrument",
    "TREND_EXCLUDE",
    "backtest",
    "benchmark_positions",
    "carry_positions",
    "decompose_pnl",
    "drawdown_curve",
    "fx_excess_returns",
    "newey_west_tstat",
    "ols_alpha",
    "performance",
    "rank_weights",
    "realised_dividend_yield",
    "scan_quarantine_candidates",
    "sharpe_by_decade",
    "trailing_vol",
    "trend_positions",
    "vol_matched_active",
]

MONTHS_PER_YEAR = 12
VOL_WINDOW = 36
VOL_MIN_OBS = 24
MIN_INSTRUMENTS = 6
INSTRUMENT_VOL_TARGET = 0.10
NW_LAGS = 4


# ── registries ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FxInstrument:
    """One FX spot series, with the quote convention made explicit.

    ``invert`` is True when the Yahoo quote is FOREIGN per USD, so a long-foreign
    position gains as the quote FALLS. Getting this wrong flips the sign of half the
    universe and would still produce a plausible-looking backtest.
    """

    key: str
    ticker: str
    ccy: str
    invert: bool
    notes: str = ""


FX_INSTRUMENTS: tuple[FxInstrument, ...] = (
    FxInstrument("FX_EUR", "EURUSD=X", "EZ", False, "USD per EUR; rising quote = long-EUR gain"),
    FxInstrument("FX_GBP", "GBPUSD=X", "GB", False, "USD per GBP"),
    FxInstrument("FX_JPY", "JPY=X", "JP", True, "JPY per USD — inverted"),
    FxInstrument("FX_AUD", "AUDUSD=X", "AU", False, "USD per AUD"),
    FxInstrument("FX_NZD", "NZDUSD=X", "NZ", False, "USD per NZD"),
    FxInstrument("FX_CAD", "CAD=X", "CA", True, "CAD per USD — inverted"),
    FxInstrument("FX_CHF", "CHF=X", "CH", True, "CHF per USD — inverted"),
    FxInstrument("FX_SEK", "SEK=X", "SE", True, "SEK per USD — inverted"),
    FxInstrument("FX_NOK", "NOK=X", "NO", True, "NOK per USD — inverted"),
)

# OECD 3-month interbank rate, one family for every country including the US so that no
# maturity or basis is mixed inside a differential. Free, keyless, monthly, in percent.
FRED_SHORT_RATES: dict[str, str] = {
    "US": "IR3TIB01USM156N",
    "EZ": "IR3TIB01EZM156N",
    "GB": "IR3TIB01GBM156N",
    "JP": "IR3TIB01JPM156N",
    "AU": "IR3TIB01AUM156N",
    "NZ": "IR3TIB01NZM156N",
    "CA": "IR3TIB01CAM156N",
    "CH": "IR3TIB01CHM156N",
    "SE": "IR3TIB01SEM156N",
    "NO": "IR3TIB01NOM156N",
}

# The SAME series from their publisher. OECD publishes this family; FRED only mirrors it.
# FRED became IP-blocked from this machine on 2026-07-31 (Akamai edge: every path times out,
# including the site root, while the TLS handshake succeeds; browser-realistic headers do
# not lift it), so `build_carry_inputs` now prefers OECD and falls back to FRED.
#
# A transport change, not a source change, and PROVEN rather than asserted: OECD's IR3TIB is
# byte-identical to the FRED-sourced parquet for GB, JP and US (max |diff| = 0.000e+00 over
# 481 / 290 / 744 months) and within 1.24e-04 for EZ, which is precision/revision noise.
# See the amendment in `fx_residual_prereg.md` and `transport_crosscheck_3m` in
# `_fx_residual/fx_residual.json`.
OECD_DATAFLOW: str = "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0"
OECD_SHORT_MEASURE: str = "IR3TIB"        # 3-month interbank, the FRED IR3TIB01 family
OECD_OVERNIGHT_MEASURE: str = "IRSTCI"    # immediate/call money, the tenor counterpart

#: Panel currency code -> OECD reference area. Every one verified present 2026-07-31.
OECD_SHORT_RATES: dict[str, str] = {
    "US": "USA", "EZ": "EA20", "GB": "GBR", "JP": "JPN", "AU": "AUS",
    "NZ": "NZL", "CA": "CAN", "CH": "CHE", "SE": "SWE", "NO": "NOR",
}

# Excluded from the TREND reference universe: NATGAS_F is roll-contaminated (integrity
# report §6a); the other four are explicit ETF duplicates of series already present.
TREND_EXCLUDE: tuple[str, ...] = ("NATGAS_F", "SPY", "GLD", "IEF", "TLT")


# ── cleaning: the panel's published criterion, applied mechanically ───────────

def scan_quarantine_candidates(
    levels_by_key: dict[str, pd.Series],
    returns_by_key: dict[str, pd.Series],
    invert_by_key: dict[str, bool],
) -> list[dict[str, object]]:
    """Apply the panel's ALREADY-PUBLISHED corrupt-close criterion to new FX series.

    The criterion is not re-derived and not tuned. It is verbatim the one recorded in
    ``research/multiasset/instruments.py::QUARANTINE``:

        (a) the 8th or 9th of a month **in 2008**,
        (b) ``|return| > 5%``,
        (c) dropping the close leaves a two-day return under 2.5% in magnitude.

    Every bar meeting (a) and (b) is returned whether or not it is admitted, with the
    numbers that decided it, so the list is auditable rather than trusted.
    """
    rows: list[dict[str, object]] = []
    for key, ret in returns_by_key.items():
        series = pd.Series(ret).dropna()
        if series.empty:
            continue
        idx = pd.DatetimeIndex(series.index)
        mask = (idx.year == 2008) & np.isin(idx.day, (8, 9)) & (series.abs() > 0.05).to_numpy()
        for stamp in idx[mask]:
            pos = int(series.index.get_loc(stamp))
            r_now = float(series.iloc[pos])
            r_next = float(series.iloc[pos + 1]) if pos + 1 < len(series) else float("nan")
            round_trip = (1.0 + r_now) * (1.0 + r_next) - 1.0 if np.isfinite(r_next) else float("nan")
            admitted = bool(np.isfinite(round_trip) and abs(round_trip) < 0.025)
            rows.append({
                "key": key,
                "date": stamp.date().isoformat(),
                "ret": round(r_now, 6),
                "next_ret": round(r_next, 6) if np.isfinite(r_next) else None,
                "round_trip": round(round_trip, 6) if np.isfinite(round_trip) else None,
                "admitted": admitted,
                "inverted_quote": bool(invert_by_key.get(key, False)),
            })
    rows.sort(key=lambda r: (r["key"], r["date"]))
    return rows


# ── signals ───────────────────────────────────────────────────────────────────

def fx_excess_returns(
    spot_returns: pd.DataFrame,
    short_rates: pd.DataFrame,
    instruments: tuple[FxInstrument, ...],
    *,
    base: str = "US",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spot returns → USD-funded excess returns, and the carry that goes with them.

    Under covered interest parity the return to a fully-collateralised long-foreign /
    short-USD forward is::

        r_t = spot_return_t + (r3m_foreign_{t-1} - r3m_US_{t-1}) / 12

    The differential is taken from the PREVIOUS month end — you earn the rate you
    contracted at, which is also what makes it point-in-time safe.

    Returns ``(excess_returns, carry)`` where ``carry`` is the annualised differential
    observable AT each month end (not lagged), i.e. the signal that sets the next
    month's position.

    Assumes CIP. Post-2008 cross-currency basis deviations of order 10-50 bps/yr are a
    real error term this construction cannot see; it is disclosed, not hidden.
    """
    idx = spot_returns.index
    rates = short_rates.reindex(idx)
    carry = pd.DataFrame(index=idx, dtype=float)
    excess = pd.DataFrame(index=idx, dtype=float)
    for inst in instruments:
        if inst.key not in spot_returns.columns or inst.ccy not in rates.columns:
            continue
        diff = rates[inst.ccy] - rates[base]
        carry[inst.key] = diff
        excess[inst.key] = spot_returns[inst.key] + diff.shift(1) / MONTHS_PER_YEAR
    return excess, carry


def realised_dividend_yield(
    total_return: pd.Series,
    price_return: pd.Series,
    *,
    window: int = 12,
) -> pd.Series:
    """Trailing realised dividend yield from a total-return / price-return pair.

    ``(Π(1+TR) / Π(1+PR)) - 1`` over ``window`` months. Backward-looking by construction,
    so it is point-in-time safe. The panel's integrity report measures this gap at
    1.95%/yr for SPY vs SPX over 8,427 overlapping days, which is the S&P dividend yield —
    that measurement is what licenses using it as the equity carry proxy.
    """
    tr = pd.Series(total_return).astype(float)
    pr = pd.Series(price_return).astype(float)
    both = pd.concat([tr, pr], axis=1).dropna()
    if both.empty:
        return pd.Series(dtype=float, index=tr.index)
    gross_tr = (1.0 + both.iloc[:, 0]).rolling(window).apply(np.prod, raw=True)
    gross_pr = (1.0 + both.iloc[:, 1]).rolling(window).apply(np.prod, raw=True)
    return (gross_tr / gross_pr - 1.0).reindex(tr.index)


def trailing_vol(
    returns: pd.DataFrame,
    *,
    window: int = VOL_WINDOW,
    min_obs: int = VOL_MIN_OBS,
) -> pd.DataFrame:
    """Annualised trailing standard deviation, strictly backward-looking.

    ``rolling(window).std()`` on month ``t`` uses months ``t-window+1 .. t`` inclusive,
    which is information available AT month end ``t`` — the position it scales is held
    over ``t+1``. NaN until ``min_obs`` observations exist.
    """
    return returns.rolling(window, min_periods=min_obs).std() * np.sqrt(MONTHS_PER_YEAR)


# ── position construction ─────────────────────────────────────────────────────

def rank_weights(scores: pd.Series) -> pd.Series:
    """Koijen-Moskowitz-Pedersen rank weights: dollar-neutral, extremes weighted most.

    ``a_i = rank_i - (N+1)/2``, ``w_i = a_i / Σ|a_j|`` so ``Σ|w| = 1`` and ``Σw = 0``.
    An all-equal score vector produces all-zero weights (average rank for everyone),
    which is the correct refusal to bet, not a division by zero.
    """
    valid = pd.Series(scores).dropna()
    out = pd.Series(0.0, index=pd.Series(scores).index, dtype=float)
    n = len(valid)
    if n < 2:
        return out
    ranks = valid.rank(method="average")
    a = ranks - (n + 1) / 2.0
    denom = a.abs().sum()
    if denom <= 0:
        return out
    out.loc[valid.index] = a / denom
    return out


def _risk_scaled(weights: pd.Series, vol: pd.Series) -> pd.Series:
    """Scale each weight to a common volatility reference; drop non-positive vols.

    Note what this does to neutrality. The rank weights satisfy ``Σw = 0`` (dollar
    neutral), but after dividing by each instrument's own volatility the NOTIONAL sum is
    no longer zero — what is conserved is ``Σ(pos_i · σ_i) = 0.10 · Σw_i = 0``, i.e. the
    book carries equal risk long and short. That is the intended property for a
    cross-asset book where a 4%-vol bond and a 12%-vol currency are not comparable
    notionals, and it is asserted in the test suite.
    """
    scaled = weights * (INSTRUMENT_VOL_TARGET / vol)
    return scaled.where(np.isfinite(scaled) & (vol > 0), 0.0)


def carry_positions(
    carry: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    vol_window: int = VOL_WINDOW,
    vol_min_obs: int = VOL_MIN_OBS,
    min_instruments: int = MIN_INSTRUMENTS,
    permute_seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Carry + returns → month-end positions. Returns ``(positions, vol, n_eligible)``.

    Eligibility at ``t``: a non-missing carry, a non-missing return in month ``t`` (the
    instrument actually traded), and at least ``vol_min_obs`` returns inside the trailing
    ``vol_window``. Fewer than ``min_instruments`` eligible ⇒ no position that month.

    ``permute_seed`` shuffles the eligible carry scores ACROSS instruments within each
    date — the negative control. It destroys the cross-sectional information while
    leaving the universe, the volatilities, the weighting and the costs untouched, so a
    surviving Sharpe would be an artefact of the machinery rather than the signal.
    """
    idx = returns.index
    carry = carry.reindex(index=idx, columns=returns.columns)
    vol = trailing_vol(returns, window=vol_window, min_obs=vol_min_obs)
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None

    positions = pd.DataFrame(0.0, index=idx, columns=returns.columns, dtype=float)
    n_eligible = pd.Series(0, index=idx, dtype=int)
    for stamp in idx:
        eligible = (carry.loc[stamp].notna() & vol.loc[stamp].notna()
                    & (vol.loc[stamp] > 0) & returns.loc[stamp].notna())
        keys = list(returns.columns[eligible])
        n_eligible.loc[stamp] = len(keys)
        if len(keys) < min_instruments:
            continue
        score = carry.loc[stamp, keys] / vol.loc[stamp, keys]
        if rng is not None:
            score = pd.Series(rng.permutation(score.to_numpy()), index=score.index)
        weights = rank_weights(score)
        positions.loc[stamp, keys] = _risk_scaled(weights, vol.loc[stamp, keys]).to_numpy()
    return positions, vol, n_eligible


def benchmark_positions(
    returns: pd.DataFrame,
    vol: pd.DataFrame,
    n_eligible: pd.Series,
    *,
    min_instruments: int = MIN_INSTRUMENTS,
) -> pd.DataFrame:
    """The own-universe benchmark: equal-RISK long-only ownership of the same names.

    ``pos_i = (1/N) × (0.10 / σ_i)`` over exactly the instruments the sleeve was allowed
    to trade that month. Benchmarking a long/short book against a cap-weighted index it
    never held is how a positive raw return gets mistaken for an edge.
    """
    positions = pd.DataFrame(0.0, index=returns.index, columns=returns.columns, dtype=float)
    for stamp in returns.index:
        eligible = (vol.loc[stamp].notna() & (vol.loc[stamp] > 0) & returns.loc[stamp].notna())
        keys = list(returns.columns[eligible])
        if len(keys) < min_instruments or n_eligible.loc[stamp] < min_instruments:
            continue
        equal = pd.Series(1.0 / len(keys), index=keys)
        positions.loc[stamp, keys] = _risk_scaled(equal, vol.loc[stamp, keys]).to_numpy()
    return positions


def trend_positions(
    returns: pd.DataFrame,
    *,
    lookback: int = 12,
    vol_window: int = VOL_WINDOW,
    vol_min_obs: int = VOL_MIN_OBS,
    min_instruments: int = MIN_INSTRUMENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """The trend REFERENCE: 12-month time-series momentum, same sizing as carry.

    ``pos_i = sign(trailing 12m compounded return) × (0.10/σ_i) / N``. Deliberately the
    textbook specification with no tuning — it exists to give the carry sleeve something
    to be correlated against, not to be a verdict on trend.
    """
    idx = returns.index
    vol = trailing_vol(returns, window=vol_window, min_obs=vol_min_obs)
    momentum = (1.0 + returns).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0

    positions = pd.DataFrame(0.0, index=idx, columns=returns.columns, dtype=float)
    n_eligible = pd.Series(0, index=idx, dtype=int)
    for stamp in idx:
        eligible = (momentum.loc[stamp].notna() & vol.loc[stamp].notna()
                    & (vol.loc[stamp] > 0) & returns.loc[stamp].notna())
        keys = list(returns.columns[eligible])
        n_eligible.loc[stamp] = len(keys)
        if len(keys) < min_instruments:
            continue
        signs = np.sign(momentum.loc[stamp, keys])
        weights = signs / len(keys)
        positions.loc[stamp, keys] = _risk_scaled(weights, vol.loc[stamp, keys]).to_numpy()
    return positions, vol, n_eligible


# ── backtest ──────────────────────────────────────────────────────────────────

def backtest(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    round_trip_bps: float,
) -> dict[str, object]:
    """Month-end positions + monthly returns → gross/net series, turnover, P&L matrix.

    Position ``pos_t`` is set at the END of month ``t`` from information through ``t``
    and earns ``r_{t+1}``. Turnover compares the new position against the PREVIOUS one
    after it has drifted with that month's return — a position left alone still changes
    size, and charging for that drift would invent turnover that never traded.

    The rebalance cost is charged to the month the position is entered for, so no return
    is booked before the trade that produced it is paid for.
    """
    pos = positions.reindex(columns=returns.columns).fillna(0.0)
    ret = returns.reindex(columns=pos.columns)
    one_way = float(round_trip_bps) / 2.0 / 10000.0

    ret_filled = ret.fillna(0.0)
    n_missing = int((ret.isna() & (pos != 0.0)).to_numpy().sum())

    pnl = pos.shift(1) * ret_filled            # P&L booked in month t from pos set at t-1
    gross = pnl.sum(axis=1)

    drifted = pos.shift(1) * (1.0 + ret_filled)
    turnover = (pos - drifted.fillna(0.0)).abs().sum(axis=1)
    cost = turnover * one_way
    net = gross - cost.shift(1).fillna(0.0)

    active = pos.abs().sum(axis=1) > 0
    first = active.idxmax() if active.any() else None
    live = ret.index > first if first is not None else pd.Series(False, index=ret.index)

    return {
        "gross": gross[live],
        "net": net[live],
        "turnover": turnover[live],
        "cost": cost.shift(1).fillna(0.0)[live],
        "pnl": pnl.loc[live],
        "positions": pos,
        "n_missing_return_cells": n_missing,
        "first_position_date": first,
    }


def decompose_pnl(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    carry: pd.DataFrame,
) -> dict[str, float]:
    """Split gross P&L into the deterministic carry accrual and the price move.

    ``accrual_i,t = pos_i,t-1 × carry_i,t-1 / 12`` is what the position earns if nothing
    moves; the remainder is the price leg. A carry sleeve whose accrual is large and
    whose price leg cancels it has found nothing — that check is the whole reason this
    function exists.
    """
    pos = positions.reindex(columns=returns.columns).fillna(0.0)
    ret = returns.reindex(columns=pos.columns).fillna(0.0)
    car = carry.reindex(index=returns.index, columns=pos.columns).fillna(0.0)

    accrual = (pos * car / MONTHS_PER_YEAR).shift(1)
    total = (pos.shift(1) * ret)
    price = total - accrual

    # nansum, not sum: the first row is NaN by construction (no prior position), and a
    # plain sum would propagate that NaN through every headline in the decomposition.
    tot = float(np.nansum(total.to_numpy()))
    acc = float(np.nansum(accrual.to_numpy()))
    pri = float(np.nansum(price.to_numpy()))
    return {
        "total_gross_pnl": tot,
        "accrual_pnl": acc,
        "price_pnl": pri,
        "accrual_share": acc / tot if tot != 0 else float("nan"),
        "price_share": pri / tot if tot != 0 else float("nan"),
    }


# ── statistics ────────────────────────────────────────────────────────────────

def newey_west_tstat(x: pd.Series, *, lags: int = NW_LAGS) -> tuple[float, float, float]:
    """``(mean, se, t)`` of a series with a Newey-West / Bartlett HAC standard error.

    Monthly strategy returns are mildly autocorrelated (positions persist), so an iid
    standard error overstates significance. Lags default to 4 and are declared in the
    pre-registration rather than chosen after seeing the answer.
    """
    v = pd.Series(x).dropna().astype(float).to_numpy()
    n = len(v)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    dev = v - mean
    gamma0 = float(dev @ dev) / n
    var = gamma0
    for lag in range(1, min(int(lags), n - 1) + 1):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    var = max(var, 1e-18)
    se = float(np.sqrt(var / n))
    return mean, se, mean / se


def ols_alpha(strategy: pd.Series, benchmark: pd.Series, *, lags: int = NW_LAGS) -> dict[str, float]:
    """Annualised OLS alpha of ``strategy`` on ``benchmark`` with a HAC t-statistic.

    The test of whether a long/short book is disguised beta. The alpha t-statistic is
    computed on the residual series' mean, which is the same estimator as ``a`` in
    ``y = a + b·x + e`` once ``b`` is fixed at its OLS value.
    """
    both = pd.concat([pd.Series(strategy), pd.Series(benchmark)], axis=1).dropna()
    if len(both) < 12:
        return {"alpha_annual": float("nan"), "beta": float("nan"), "t_alpha": float("nan"),
                "n_months": int(len(both))}
    y = both.iloc[:, 0].to_numpy(dtype=float)
    x = both.iloc[:, 1].to_numpy(dtype=float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    beta = float(xc @ (y - y.mean())) / denom if denom > 0 else 0.0
    resid = pd.Series(y - beta * x, index=both.index)
    mean, _se, t = newey_west_tstat(resid, lags=lags)
    return {"alpha_annual": mean * MONTHS_PER_YEAR, "beta": beta, "t_alpha": t,
            "n_months": int(len(both))}


def vol_matched_active(strategy: pd.Series, benchmark: pd.Series,
                       *, lags: int = NW_LAGS) -> dict[str, float]:
    """Arithmetic active return after scaling the benchmark to the strategy's OWN volatility.

    The variance-drag trap has two sides and this programme has now been bitten by both.
    PEAD faked a positive GEOMETRIC excess by running at LOWER volatility than its
    benchmark. The multi-asset trend sleeve faked a positive ARITHMETIC active return by
    running at HIGHER volatility: differencing two streams at different volatilities does
    not compare skill, it compares leverage, so the raw active t-statistic rises with the
    volatility target while the strategy's own t-statistic does not.

    Scaling the benchmark by a single FULL-SAMPLE constant removes exactly that and
    nothing else — a rolling scale factor would smuggle in volatility timing. The
    resulting statistic is invariant to levering the strategy, which is asserted in the
    test suite and is the property that makes it trustworthy.
    """
    both = pd.concat([pd.Series(strategy).rename("s"),
                      pd.Series(benchmark).rename("b")], axis=1).dropna()
    if len(both) < 12:
        return {}
    s, b = both["s"], both["b"]
    sd_b = float(b.std())
    k = float(s.std()) / sd_b if sd_b > 0 else float("nan")
    mean, _se, t = newey_west_tstat(s - b * k, lags=lags)
    raw_mean, _rse, raw_t = newey_west_tstat(s - b, lags=lags)
    return {
        "n_months": int(len(both)),
        "strategy_vol": float(s.std()) * np.sqrt(MONTHS_PER_YEAR),
        "benchmark_vol": sd_b * np.sqrt(MONTHS_PER_YEAR),
        "benchmark_scale_factor": k,
        "raw_active_annual": raw_mean * MONTHS_PER_YEAR,
        "raw_active_tstat": raw_t,
        "vol_matched_active_annual": mean * MONTHS_PER_YEAR,
        "vol_matched_active_tstat": t,
        "strategy_sharpe": float(s.mean()) / float(s.std()) * np.sqrt(MONTHS_PER_YEAR),
        "benchmark_sharpe": float(b.mean()) / sd_b * np.sqrt(MONTHS_PER_YEAR) if sd_b > 0 else float("nan"),
    }


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Compounded drawdown from the running peak."""
    equity = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return equity / equity.cummax() - 1.0


def performance(returns: pd.Series, *, lags: int = NW_LAGS) -> dict[str, float]:
    """Every headline statistic for one return stream — all ARITHMETIC where it matters."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 3:
        return {"n_months": int(len(r))}
    mean, se, t = newey_west_tstat(r, lags=lags)
    vol_m = float(r.std())
    ann_vol = vol_m * np.sqrt(MONTHS_PER_YEAR)
    ann_arith = mean * MONTHS_PER_YEAR
    sharpe = ann_arith / ann_vol if ann_vol > 0 else float("nan")
    equity = float((1.0 + r).prod())
    years = len(r) / MONTHS_PER_YEAR
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else float("nan")
    dd = drawdown_curve(r)
    return {
        "n_months": int(len(r)),
        "years": round(years, 2),
        "arithmetic_annual": ann_arith,
        "geometric_annual": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "t_stat": t,
        "se_monthly": se,
        "max_drawdown": float(dd.min()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "worst_month": float(r.min()),
        "best_month": float(r.max()),
        "hit_rate": float((r > 0).mean()),
    }


def sharpe_by_decade(returns: pd.Series) -> dict[str, dict[str, float]]:
    """Annualised arithmetic Sharpe per calendar decade. A full-sample pass carried by
    one era is not deployable, so this is reported for every stream, always."""
    r = pd.Series(returns).dropna().astype(float)
    out: dict[str, dict[str, float]] = {}
    if r.empty:
        return out
    decade = (pd.DatetimeIndex(r.index).year // 10) * 10
    for dec, chunk in r.groupby(decade):
        if len(chunk) < 12:
            out[f"{int(dec)}s"] = {"n_months": int(len(chunk)), "sharpe": float("nan"),
                                   "arithmetic_annual": float("nan")}
            continue
        ann = float(chunk.mean()) * MONTHS_PER_YEAR
        vol = float(chunk.std()) * np.sqrt(MONTHS_PER_YEAR)
        out[f"{int(dec)}s"] = {
            "n_months": int(len(chunk)),
            "arithmetic_annual": ann,
            "annual_vol": vol,
            "sharpe": ann / vol if vol > 0 else float("nan"),
        }
    return out
