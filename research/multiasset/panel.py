"""Panel construction and integrity primitives for the long-history multi-asset study.

Pure functions only — no network, no file IO, no globals. Every function here is
exercised offline by ``tests/test_multiasset_panel.py``; the network fetch lives
in ``scripts/build_multiasset_panel.py``.

The three things this module exists to get right
================================================
1. **Chronological order.** Every consecutive-bar calculation sorts and dedupes
   first (``clean_levels``). The programme has already paid for this once.
2. **Yields are not prices.** ``par_bond_total_return`` converts a constant-maturity
   yield into the total return of the bond by REPRICING it. ``pct_change`` on a
   yield series is not merely inaccurate — it has the wrong sign.
3. **A ratio needs two positive numbers.** WTI front-month settled NEGATIVE on
   2020-04-20, so ``P_t / P_{t-1} - 1`` is undefined on that bar and the one after
   it. Those bars are nulled, counted and reported, never silently compounded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "GAP_NULL_DAYS",
    "apply_quarantine",
    "dsr_sharpe_bar",
    "bill_cash_return",
    "clean_levels",
    "day_of_month_signature",
    "coverage_row",
    "flag_extreme_returns",
    "gap_report",
    "monthly_last",
    "monthly_returns",
    "par_bond_total_return",
    "simple_returns",
    "wide_panel",
]

# A bar labelled "one day" that actually spans more than this many calendar days is
# not a daily return; compounding it into a daily volatility estimate corrupts it.
# Such bars are nulled and COUNTED (never dropped silently). Three days covers a
# normal weekend; 15 covers every exchange holiday cluster including Golden Week.
GAP_NULL_DAYS = 15

_DAYS_PER_YEAR = 365.0


# ── Statistical power of a sample length ──────────────────────────────────────

def dsr_sharpe_bar(
    years: float,
    *,
    n_trials: int | None = 32,
    periods_per_year: int = 12,
    target: float = 0.95,
) -> float:
    """The ANNUAL Sharpe a strategy needs to reach ``DSR >= target`` on ``years`` of data.

    This is the whole reason the study wants long history: the bar falls with sample
    length. Inverting Bailey & Lopez de Prado's Deflated Sharpe (the same formulation
    as ``research.validation.deflated_sharpe_ratio``) under Gaussian returns::

        SR* = sigma * [(1-gamma)*Z(1-1/N) + gamma*Z(1-1/(Ne))]
        DSR = Phi[(SR - SR*)/sigma] = target
        =>  SR = sigma * (bracket + Z(target)),   sigma = sqrt((1 + SR^2/2)/(T-1))

    solved by fixed-point iteration because sigma depends on SR. Reproduces the two
    anchors recorded by the programme EXACTLY — 1.488 at 7 years and 0.597 at 40
    years, both at ``n_trials=32`` on monthly returns — which is what pins the
    frequency convention; the test suite asserts both.

    Gaussian by construction: real returns are skewed and fat-tailed, which RAISES the
    bar, so this is a floor, not a promise.

    ``n_trials=None`` reads the programme's cumulative count from
    `research.trial_ledger`, which is the single source of truth. The default of 32 is
    UNCHANGED: it is the anchor the frequency convention is pinned by and the one the
    test suite asserts, so moving it would silently re-deflate every banked result.
    """
    if n_trials is None:
        from research.trial_ledger import cumulative_trials
        n_trials = cumulative_trials()
    T = int(round(years * periods_per_year))
    if T < 4:
        raise ValueError("sample too short for a DSR bar")
    n = max(int(n_trials), 1)
    gamma = 0.5772156649015329                     # Euler-Mascheroni
    if n > 1:
        bracket = ((1.0 - gamma) * float(norm.ppf(1.0 - 1.0 / n))
                   + gamma * float(norm.ppf(1.0 - 1.0 / (n * math.e))))
    else:
        bracket = 0.0
    k = bracket + float(norm.ppf(target))

    sr = 0.1
    for _ in range(200):                            # converges in <20; 200 is free
        sr = k * math.sqrt((1.0 + 0.5 * sr * sr) / (T - 1))
    return float(sr * math.sqrt(periods_per_year))


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_levels(raw: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Sort chronologically, dedupe, drop non-finite. Returns ``(series, stats)``.

    Duplicated timestamps keep the LAST occurrence (a restatement supersedes the
    earlier print). The returned index is guaranteed strictly increasing and unique,
    which is the precondition every other function in this module assumes.
    """
    s = pd.Series(raw).copy()
    n_raw = int(len(s))

    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()

    s = pd.to_numeric(s, errors="coerce")
    n_nonfinite = int((~np.isfinite(s.to_numpy(dtype=float))).sum())
    s = s[np.isfinite(s.to_numpy(dtype=float))]

    n_before_dedupe = int(len(s))
    s = s[~s.index.duplicated(keep="last")]
    n_dupes = n_before_dedupe - int(len(s))

    was_sorted = bool(s.index.is_monotonic_increasing)
    s = s.sort_index()

    stats = {
        "n_raw": n_raw,
        "n_nonfinite_dropped": n_nonfinite,
        "n_duplicate_dates_dropped": n_dupes,
        "was_already_sorted": int(was_sorted),
        "n_clean": int(len(s)),
    }
    return s, stats


def apply_quarantine(
    levels_by_key: dict[str, pd.Series],
    quarantine: tuple[tuple[str, str, str], ...],
) -> tuple[dict[str, pd.Series], list[dict[str, object]]]:
    """Drop individually evidenced corrupt CLOSES before any return is computed.

    Dropping the level (rather than nulling the two returns that straddle it) is
    what preserves the truth: the surrounding observations then form a single valid
    two-day return, so the genuine move across the corrupt print is kept and only
    the fabricated spike-and-reversal pair is removed.

    Returns the new mapping plus one audit record per entry, including entries that
    matched nothing — a quarantine list that silently stops matching after a data
    refresh is how a stale exclusion becomes an invisible one.
    """
    out = {k: s.copy() for k, s in levels_by_key.items()}
    audit: list[dict[str, object]] = []
    for key, date_str, reason in quarantine:
        stamp = pd.Timestamp(date_str)
        present = key in out and stamp in out[key].index
        if present:
            out[key] = out[key].drop(index=stamp)
        audit.append({"key": key, "date": date_str, "reason": reason,
                      "matched": bool(present)})
    return out, audit


def day_of_month_signature(returns: pd.Series, *, top_n: int = 10) -> dict[str, float]:
    """Do an instrument's largest moves cluster on a day of the month?

    A market event has no opinion about the calendar. A vendor defect does. This
    returns the modal day-of-month among the ``top_n`` largest absolute returns, how
    many of the top_n share it, and the base rate of that day — the test that
    identified the 2008 EURUSD/JPYUSD corrupt closes.
    """
    r = pd.Series(returns).dropna()
    if len(r) < top_n:
        return {"modal_day": float("nan"), "n_of_top": 0, "base_rate_pct": float("nan"),
                "lift": float("nan")}
    top = r.reindex(r.abs().sort_values(ascending=False).index).head(top_n)
    days = pd.Series([d.day for d in top.index])
    modal_day = int(days.mode().iloc[0])
    n_of_top = int((days == modal_day).sum())
    base = float((pd.DatetimeIndex(r.index).day == modal_day).mean())
    return {
        "modal_day": modal_day,
        "n_of_top": n_of_top,
        "base_rate_pct": round(100.0 * base, 2),
        "lift": round((n_of_top / top_n) / base, 2) if base > 0 else float("nan"),
    }


# ── Returns: prices ───────────────────────────────────────────────────────────

def simple_returns(
    levels: pd.Series,
    *,
    invert: bool = False,
    max_gap_days: int | None = GAP_NULL_DAYS,
) -> tuple[pd.Series, dict[str, int]]:
    """Level series → simple returns, with the two undefined cases handled explicitly.

    ``invert`` reciprocates the level first — for a quote like ``JPY=X`` (JPY per
    USD) where the position we want gains as the quote falls.

    A return is NULLED (not dropped, not zeroed) when either endpoint level is
    non-positive — the ratio is undefined, and pretending otherwise produced a
    -306% "return" on WTI in April 2020 — or when the bar spans more than
    ``max_gap_days`` calendar days. Both are counted in the returned stats.
    """
    s = levels.astype(float)
    prev = s.shift(1)

    bad_level = (s <= 0.0) | (prev <= 0.0)
    base = 1.0 / s if invert else s
    prev_base = base.shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = base / prev_base - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    ret[bad_level] = np.nan

    n_gap_nulled = 0
    if max_gap_days is not None and len(s) > 1:
        gap_days = pd.Series(s.index, index=s.index).diff().dt.days
        too_long = gap_days > float(max_gap_days)
        n_gap_nulled = int((too_long & ret.notna()).sum())
        ret[too_long.fillna(False)] = np.nan

    stats = {
        "n_nonpositive_level_bars": int((s <= 0.0).sum()),
        "n_returns_nulled_nonpositive": int((bad_level & prev.notna()).sum()),
        "n_returns_nulled_long_gap": n_gap_nulled,
    }
    return ret, stats


# ── Returns: yields ───────────────────────────────────────────────────────────

def par_bond_total_return(
    yield_pct: pd.Series,
    maturity_years: float,
    *,
    coupons_per_year: int = 2,
    max_gap_days: int = GAP_NULL_DAYS,
) -> pd.Series:
    """Constant-maturity YIELD series → the total return of holding the bond.

    Method (exact given the stated assumptions, not a duration approximation).
    At ``t-1`` we buy a par bond of the stated maturity: its coupon rate is set to
    that day's yield ``y_{t-1}``, so it is worth exactly 100. One bar later
    ``dt = (date_t - date_{t-1}) / 365`` years have passed, leaving ``n`` coupon
    periods of which the first arrives in ``f = 1 - dt*m`` periods. Repricing the
    same cash flows at the new yield ``y_t``::

        v = 1 / (1 + y_t/m)
        dirty = v**f * [ (100*y_{t-1}/m) * (1 - v**n)/(1 - v) + 100 * v**(n-1) ]
        return = dirty/100 - 1

    Both legs of the bond return are therefore present and correctly signed: the
    coupon accrues with ``dt`` (a Friday→Monday bar earns three days of carry,
    which is why the day count is ACT/365 calendar and not 1/252), and the capital
    leg moves OPPOSITE to the yield. Set ``y_t = y_{t-1}`` and the result collapses
    to pure carry; that identity is asserted in the test suite.

    What it assumes, and therefore what it cannot capture: the constant-maturity
    par-bond convention (no specific on-the-run bond, no financing, no bid-ask),
    and a flat curve at ``y_t`` for discounting. The output is validated against
    IEF/TLT/IEI total returns in the integrity report rather than taken on faith.

    Returns a series aligned to the input index; the first bar and any bar spanning
    more than ``max_gap_days`` calendar days are NaN.
    """
    if maturity_years <= 0:
        raise ValueError("maturity_years must be positive")
    if coupons_per_year <= 0:
        raise ValueError("coupons_per_year must be positive")

    m = float(coupons_per_year)
    n = int(round(maturity_years * m))
    if n < 1:
        raise ValueError("maturity_years * coupons_per_year must be at least 1")

    y = pd.Series(yield_pct).astype(float) / 100.0
    idx = y.index
    y_prev = y.shift(1).to_numpy(dtype=float)
    y_now = y.to_numpy(dtype=float)

    gap_days = pd.Series(idx, index=idx).diff().dt.days.to_numpy(dtype=float)
    dt = gap_days / _DAYS_PER_YEAR
    f = 1.0 - dt * m

    valid = (
        np.isfinite(y_prev) & np.isfinite(y_now) & np.isfinite(dt)
        & (gap_days > 0) & (gap_days <= float(max_gap_days))
        & (f > 0.0)
        & (y_now > -m) & (y_prev > -m)          # discount factor must stay positive
    )

    y_now_s = np.where(valid, y_now, 0.0)
    y_prev_s = np.where(valid, y_prev, 0.0)
    f_s = np.where(valid, f, 1.0)

    v = 1.0 / (1.0 + y_now_s / m)
    a = np.power(v, f_s)
    one_minus_v = 1.0 - v
    # y_t == 0 ⇒ v == 1 ⇒ the annuity factor is its limit, n.
    annuity = np.where(
        np.abs(one_minus_v) < 1e-15,
        float(n),
        (1.0 - np.power(v, n)) / np.where(np.abs(one_minus_v) < 1e-15, 1.0, one_minus_v),
    )
    coupon = 100.0 * y_prev_s / m
    dirty = a * (coupon * annuity + 100.0 * np.power(v, n - 1))

    ret = np.where(valid, dirty / 100.0 - 1.0, np.nan)
    return pd.Series(ret, index=idx)


def bill_cash_return(
    discount_pct: pd.Series,
    *,
    bill_days: int = 91,
    max_gap_days: int = GAP_NULL_DAYS,
) -> pd.Series:
    """13-week T-bill DISCOUNT rate → the daily return on cash.

    ``^IRX`` is quoted on a bank-discount basis (ACT/360 against face), which is not
    a yield you can accrue. Converted to the bond-equivalent yield::

        BEY = 365 * d / (360 - d * bill_days)

    then accrued over the ACTUAL calendar days in the bar, using the rate observed
    at ``t-1`` — you earn the rate you bought at, which also keeps it point-in-time
    safe. The discount-to-BEY step is worth a few basis points at low rates and
    tens of basis points at 1980s rates; skipping it would understate cash.
    """
    d = pd.Series(discount_pct).astype(float) / 100.0
    idx = d.index

    denom = 360.0 - d * float(bill_days)
    bey = (365.0 * d / denom).where(denom > 0.0)

    gap_days = pd.Series(idx, index=idx).diff().dt.days.astype(float)
    ok = (gap_days > 0) & (gap_days <= float(max_gap_days))

    ret = bey.shift(1) * gap_days / _DAYS_PER_YEAR
    return ret.where(ok & ret.notna())


# ── Panels ────────────────────────────────────────────────────────────────────

def wide_panel(series_by_key: dict[str, pd.Series]) -> pd.DataFrame:
    """Align per-instrument series onto one sorted union DatetimeIndex."""
    if not series_by_key:
        return pd.DataFrame()
    frame = pd.DataFrame({k: pd.Series(v).astype(float) for k, v in series_by_key.items()})
    frame = frame.sort_index()
    frame.index.name = "date"
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    return frame


def monthly_returns(daily: pd.DataFrame, *, min_obs: int = 5) -> pd.DataFrame:
    """Daily simple returns → month-end compounded returns.

    Indexed at CALENDAR month end so instruments on different exchange calendars
    line up. A month is NaN for an instrument with fewer than ``min_obs`` daily
    observations in it, so a two-day stub is never presented as a month. The final
    calendar month is dropped unless the data reaches its last business day.
    """
    if daily.empty:
        return daily.copy()

    period = daily.index.to_period("M")
    grouped = (1.0 + daily).groupby(period)
    compounded = grouped.prod(min_count=1) - 1.0
    counts = daily.groupby(period).count()
    monthly = compounded.where(counts >= int(min_obs))
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "date"

    last_date = daily.index.max()
    last_bday = (last_date + pd.offsets.MonthEnd(0)) - pd.offsets.BDay(0)
    month_end = (last_date + pd.offsets.MonthEnd(0)).normalize()
    if last_date < pd.Timestamp(last_bday).normalize():
        monthly = monthly[monthly.index < month_end]
    return monthly


def monthly_last(daily_levels: pd.DataFrame) -> pd.DataFrame:
    """Daily levels/yields → the last observation in each calendar month."""
    if daily_levels.empty:
        return daily_levels.copy()
    period = daily_levels.index.to_period("M")
    out = daily_levels.groupby(period).last()
    out.index = out.index.to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out


# ── Integrity reporting ───────────────────────────────────────────────────────

def gap_report(index: pd.DatetimeIndex) -> dict[str, float]:
    """Calendar-gap statistics for one instrument's observation dates."""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return {"max_gap_days": float("nan"), "n_gaps_gt_5d": 0, "n_gaps_gt_15d": 0,
                "n_gaps_gt_30d": 0, "bday_coverage_pct": float("nan")}
    gaps = pd.Series(idx).diff().dt.days.dropna()
    n_bdays = int(len(pd.bdate_range(idx.min(), idx.max())))
    return {
        "max_gap_days": float(gaps.max()),
        "n_gaps_gt_5d": int((gaps > 5).sum()),
        "n_gaps_gt_15d": int((gaps > 15).sum()),
        "n_gaps_gt_30d": int((gaps > 30).sum()),
        "bday_coverage_pct": 100.0 * len(idx) / n_bdays if n_bdays else float("nan"),
    }


def coverage_row(key: str, levels: pd.Series) -> dict[str, object]:
    """First date, last date, observation count, missing percent, gap stats."""
    s = pd.Series(levels).dropna()
    if s.empty:
        return {"key": key, "first_date": None, "last_date": None, "n_obs": 0,
                "years": 0.0, "pct_missing_vs_bdays": float("nan")}
    gaps = gap_report(pd.DatetimeIndex(s.index))
    first, last = s.index.min(), s.index.max()
    years = (last - first).days / 365.25
    cov = gaps["bday_coverage_pct"]
    return {
        "key": key,
        "first_date": first.date().isoformat(),
        "last_date": last.date().isoformat(),
        "n_obs": int(len(s)),
        "years": round(float(years), 2),
        "pct_missing_vs_bdays": round(float(100.0 - cov), 2) if np.isfinite(cov) else float("nan"),
        **{k: gaps[k] for k in ("max_gap_days", "n_gaps_gt_5d", "n_gaps_gt_15d", "n_gaps_gt_30d")},
    }


def flag_extreme_returns(returns: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """Every bar with ``|return| > threshold``, long format, worst first.

    These are not dropped by this function. Each one is INSPECTED and given an
    explicit disposition in the integrity report — a prior study compounded a
    +9,900% print worth 13% of its total P&L because nothing looked.
    """
    if returns.empty:
        return pd.DataFrame(columns=["key", "date", "ret", "gap_days"])
    stacked = returns.stack(future_stack=True).rename("ret").reset_index()
    stacked.columns = ["date", "key", "ret"]
    hits = stacked[stacked["ret"].abs() > float(threshold)].copy()
    if hits.empty:
        return pd.DataFrame(columns=["key", "date", "ret", "gap_days"])

    gap_lookup: dict[str, pd.Series] = {}
    for key in hits["key"].unique():
        col = returns[key].dropna()
        gap_lookup[key] = pd.Series(col.index, index=col.index).diff().dt.days

    hits["gap_days"] = [
        float(gap_lookup[k].get(d, float("nan"))) for k, d in zip(hits["key"], hits["date"])
    ]
    hits = hits.reindex(columns=["key", "date", "ret", "gap_days"])
    return hits.sort_values("ret", key=lambda c: c.abs(), ascending=False).reset_index(drop=True)
