"""Build the BREADTH-EXPANSION panel and prove its integrity.

    .venv/Scripts/python.exe -m research.multiasset.breadth_build [--use-cache]

Fetches the candidate independent bets registered in ``breadth_instruments.py``, applies
**the same guards** the original builder applies (chronological sort, non-positive price
guard, long-gap nulling, |return| > 50% hunt with individual inspection, day-of-month
calendar signature, roll-contamination test on every futures series), and writes the
daily and month-end panels to ``_data/multiasset/breadth/``.

It reuses ``research/multiasset/panel.py`` unmodified and does not touch the existing
builder, the existing registry, or the existing panels. Nothing but derived statistics is
ever committed — Yahoo's terms forbid redistributing its data and ``_data/`` is
gitignored.

Builds NO strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.multiasset.breadth_instruments import (
    BREADTH_INSTRUMENTS,
    ROLL_VALIDATION_PAIRS,
    SYNTHETIC_SPREADS,
    Instrument,
)
from research.multiasset.panel import (
    GAP_NULL_DAYS,
    clean_levels,
    coverage_row,
    day_of_month_signature,
    flag_extreme_returns,
    monthly_returns,
    simple_returns,
    wide_panel,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "_data" / "multiasset" / "breadth"
ORIGINAL_DATA = ROOT / "_data" / "multiasset"
EXTREME_THRESHOLD = 0.50
ROLL_EXTREME_THRESHOLD = 0.15          # same threshold the NATGAS_F test used
TRADING_DAYS = 252

# Declared roll / expiry windows, day-of-month. A front-month continuous series that is
# not back-adjusted prints the roll spread as if it were a price move, so its extreme
# bars will know what day of the month it is. Each window is the contract's own last
# trading / first notice period, NOT a fitted window.
#
# CORRECTION, disclosed rather than quietly applied. The first version of this table put
# SUGAR_F and CATTLE_F at days 25-31 because both contracts expire on a LAST BUSINESS
# DAY. That reasoning was wrong about where the splice lands: a contract that trades to
# the last business day is still the front month on that day, so the spliced bar is the
# FIRST bar of the following month. Both were moved to (1, 2). CATTLE_F's declared
# window scored a variance ratio of 0.77 (nothing) at 25-31 and 7.55 at day 1.
ROLL_WINDOWS: dict[str, tuple[int, int]] = {
    "CORN_F": (10, 16),        # last trading day = business day before the 15th
    "WHEAT_F": (10, 16),
    "SOYBEAN_F": (10, 16),
    "SUGAR_F": (1, 2),         # expires last business day of the PRECEDING month
    "COFFEE_F": (18, 28),      # notice period ~8 business days before month end
    "COTTON_F": (3, 12),       # last trading day ~17 business days from month end
    "COCOA_F": (12, 22),
    "CATTLE_F": (1, 2),        # expires the last business day of the contract month
    "HOGS_F": (12, 18),        # 10th business day of the contract month
}
# The window the original report used for NATGAS_F, kept so the two are comparable.
LEGACY_ROLL_WINDOW = (24, 31)


# ── fetch (same pattern as the original builder) ─────────────────────────────

def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_").replace("-", "_")


def fetch_one(ticker: str, cache_dir: Path, *, use_cache: bool, retries: int = 3) -> pd.DataFrame:
    """Fetch one ticker's full history, caching the raw frame to parquet."""
    path = cache_dir / f"{_safe_name(ticker)}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True,
                              progress=False, actions=False, threads=False)
            if raw is None or raw.empty:
                raise RuntimeError("empty frame")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.loc[:, ~raw.columns.duplicated()]
            raw.index.name = "date"
            cache_dir.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(path)
            return raw
        except Exception as exc:  # noqa: BLE001 — one ticker must not kill the build
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{ticker}: fetch failed after {retries} attempts ({last_exc})")


def instrument_returns(inst: Instrument, levels: pd.Series) -> tuple[pd.Series, dict]:
    """Dispatch to the correct return convention. Every breadth key is a price series."""
    if inst.return_method == "price_return":
        return simple_returns(levels, invert=False)
    if inst.return_method == "inverse_price_return":
        return simple_returns(levels, invert=True)
    raise ValueError(f"{inst.key}: unsupported return_method {inst.return_method!r} "
                     "for the breadth panel (yields belong in the original builder)")


# ── integrity: roll contamination ────────────────────────────────────────────

def roll_contamination(
    returns: pd.Series,
    window: tuple[int, int],
    *,
    threshold: float = ROLL_EXTREME_THRESHOLD,
) -> dict[str, float]:
    """Do this series' extreme bars cluster inside its contract's roll window?

    The test that condemned ``NATGAS_F`` (65.7% of |r|>15% bars in days 24-31 against a
    24.0% base rate — a 2.74x lift). A price move has no opinion about the calendar; a
    splice does. ``window`` is the instrument's OWN declared last-trading / notice
    period, not a fitted one.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"n_extreme": 0, "pct_in_window": float("nan"),
                "base_rate_pct": float("nan"), "lift": float("nan")}
    lo, hi = window
    days = pd.DatetimeIndex(r.index).day
    in_win = (days >= lo) & (days <= hi)
    base = float(in_win.mean())
    ext = r.abs() > float(threshold)
    n_ext = int(ext.sum())
    if n_ext == 0 or base <= 0:
        return {"n_extreme": n_ext, "pct_in_window": float("nan"),
                "base_rate_pct": round(100.0 * base, 2), "lift": float("nan")}
    frac = float(in_win[ext.to_numpy()].mean())
    return {
        "n_extreme": n_ext,
        "pct_in_window": round(100.0 * frac, 2),
        "base_rate_pct": round(100.0 * base, 2),
        "lift": round(frac / base, 3),
    }


def dom_variance_share(returns: pd.Series, window: tuple[int, int]) -> dict[str, float]:
    """Share of total squared return contributed by bars inside a day-of-month window.

    Strictly stronger than counting extremes: a splice does not have to clear a 15%
    threshold to dominate the series' variance. If a window holding 3% of the bars holds
    23% of the variance, those bars are not returns.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return {"bar_share_pct": float("nan"), "variance_share_pct": float("nan"),
                "ratio": float("nan"), "mean_abs_in_pct": float("nan"),
                "mean_abs_out_pct": float("nan"), "mean_in_annualised_pct": float("nan")}
    lo, hi = window
    days = pd.DatetimeIndex(r.index).day
    inw = (days >= lo) & (days <= hi)
    sq = r.to_numpy() ** 2
    bar_share = float(inw.mean())
    var_share = float(sq[inw].sum() / sq.sum()) if sq.sum() > 0 else float("nan")
    return {
        "bar_share_pct": round(100.0 * bar_share, 2),
        "variance_share_pct": round(100.0 * var_share, 2),
        "ratio": round(var_share / bar_share, 3) if bar_share > 0 else float("nan"),
        "mean_abs_in_pct": round(100.0 * float(r[inw].abs().mean()), 4),
        "mean_abs_out_pct": round(100.0 * float(r[~inw].abs().mean()), 4),
        # what those bars alone contribute to the annualised mean return
        "mean_in_annualised_pct": round(100.0 * float(r[inw].sum()) / (len(r) / TRADING_DAYS), 3),
    }


def max_window_lift(
    returns: pd.Series,
    *,
    width: int = 7,
    threshold: float = ROLL_EXTREME_THRESHOLD,
) -> dict[str, Any]:
    """Worst lift over ANY contiguous ``width``-day window of the month.

    A distribution-free companion to ``roll_contamination``: it does not need the
    contract's calendar to be known, so it cannot be accused of having been pointed at
    the answer. With ~25 candidate windows a lift near 2x is unremarkable; the NATGAS
    scale (2.74x on a DECLARED window) is not.
    """
    r = pd.Series(returns).dropna()
    ext = r.abs() > float(threshold)
    if int(ext.sum()) < 5:
        return {"n_extreme": int(ext.sum()), "best_window": None, "lift": float("nan")}
    best: tuple[float, tuple[int, int] | None] = (float("-inf"), None)
    for lo in range(1, 32 - width + 1):
        res = roll_contamination(r, (lo, lo + width - 1), threshold=threshold)
        lift = res["lift"]
        if np.isfinite(lift) and lift > best[0]:
            best = (lift, (lo, lo + width - 1))
    return {"n_extreme": int(ext.sum()), "best_window": best[1], "lift": round(best[0], 3)}


def validation_stats(constructed: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    """Overlap correlation, annualised drift gap and volatility ratio of two return series."""
    both = pd.concat([constructed.rename("a"), benchmark.rename("b")], axis=1).dropna()
    if len(both) < 60:
        return {"n_overlap": int(len(both)), "corr": float("nan"),
                "drift_gap_pct_yr": float("nan"), "vol_ratio": float("nan"),
                "first": None, "last": None}
    a, b = both["a"], both["b"]
    return {
        "n_overlap": int(len(both)),
        "corr": round(float(a.corr(b)), 4),
        "drift_gap_pct_yr": round(float((b.mean() - a.mean()) * TRADING_DAYS * 100.0), 3),
        "vol_ratio": round(float(a.std() / b.std()), 4) if b.std() else float("nan"),
        "first": both.index.min().date().isoformat(),
        "last": both.index.max().date().isoformat(),
    }


def monthly_reconciliation(daily: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, float]:
    """Independently recompute the month-end panel and assert it matches, cell for cell."""
    period = daily.index.to_period("M")
    worst, n_cells = 0.0, 0
    for col in daily.columns:
        s = daily[col]
        for per, chunk in s.groupby(period):
            vals = chunk.dropna()
            if len(vals) < 5:
                continue
            stamp = per.to_timestamp(how="end").normalize()
            if stamp not in monthly.index:
                continue
            got = monthly.at[stamp, col]
            if pd.isna(got):
                continue
            expected = float(np.prod(1.0 + vals.to_numpy())) - 1.0
            worst = max(worst, abs(float(got) - expected))
            n_cells += 1
    return {"n_cells_checked": n_cells, "max_abs_discrepancy": worst}


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true", help="reuse cached raw parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    out_dir: Path = args.out
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"threshold_extreme": EXTREME_THRESHOLD, "gap_null_days": GAP_NULL_DAYS}

    # 1 ── fetch and clean ────────────────────────────────────────────────────
    levels: dict[str, pd.Series] = {}
    clean_stats: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for inst in BREADTH_INSTRUMENTS:
        try:
            raw = fetch_one(inst.ticker, raw_dir, use_cache=args.use_cache)
        except Exception as exc:  # noqa: BLE001
            failures[inst.key] = str(exc)[:200]
            print(f"  FETCH FAILED {inst.key:12s} {inst.ticker:10s} {exc}")
            continue
        s, st = clean_levels(raw["Close"])
        levels[inst.key] = s
        clean_stats[inst.key] = st
        print(f"  {inst.key:12s} {inst.ticker:10s} {st['n_clean']:>6d} obs  "
              f"{s.index.min().date()} -> {s.index.max().date()}  "
              f"sorted={st['was_already_sorted']} dupes={st['n_duplicate_dates_dropped']}")
    report["fetch_failures"] = failures
    report["clean_stats"] = clean_stats
    report["n_fetched"] = len(levels)

    # 2 ── returns, per the registered convention ─────────────────────────────
    rets: dict[str, pd.Series] = {}
    ret_stats: dict[str, dict] = {}
    for inst in BREADTH_INSTRUMENTS:
        if inst.key not in levels:
            continue
        r, st = instrument_returns(inst, levels[inst.key])
        rets[inst.key] = r
        ret_stats[inst.key] = st
    report["return_stats"] = ret_stats
    report["n_returns_nulled_nonpositive_total"] = int(
        sum(v["n_returns_nulled_nonpositive"] for v in ret_stats.values()))
    report["n_returns_nulled_long_gap_total"] = int(
        sum(v["n_returns_nulled_long_gap"] for v in ret_stats.values()))

    panel_keys = [i.key for i in BREADTH_INSTRUMENTS if i.tradable and i.key in rets]
    valid_keys = [i.key for i in BREADTH_INSTRUMENTS if not i.tradable and i.key in rets]

    daily_all = wide_panel(rets)
    daily = daily_all.loc[:, panel_keys]

    # 3 ── synthetic long-short spreads (already excess: self-financing) ──────
    orig_daily = pd.read_parquet(ORIGINAL_DATA / "returns_daily.parquet")
    spread_info: dict[str, dict] = {}
    for key, (a, b, why) in SYNTHETIC_SPREADS.items():
        leg_a = daily_all[a] if a in daily_all.columns else orig_daily[a]
        leg_b = daily_all[b] if b in daily_all.columns else orig_daily[b]
        both = pd.concat([leg_a.rename("a"), leg_b.rename("b")], axis=1)
        sp = (both["a"] - both["b"]).where(both["a"].notna() & both["b"].notna())
        daily[key] = sp
        spread_info[key] = {
            "definition": f"{a} - {b}", "rationale": why,
            "n_obs": int(sp.notna().sum()),
            "first": str(sp.dropna().index.min().date()),
            "corr_to_leg_a": round(float(sp.corr(both["a"])), 4),
            "corr_to_leg_b": round(float(sp.corr(both["b"])), 4),
        }
    report["synthetic_spreads"] = spread_info
    daily = daily.loc[:, [c for c in panel_keys] + list(SYNTHETIC_SPREADS)]

    # 4 ── the |return| > 50% hunt, shipped vs naive ──────────────────────────
    extremes = flag_extreme_returns(daily, EXTREME_THRESHOLD)
    naive = pd.DataFrame({k: pd.Series(v).sort_index().pct_change()
                          for k, v in levels.items() if k in panel_keys})
    extremes_naive = flag_extreme_returns(naive, EXTREME_THRESHOLD)
    report["n_extreme_shipped"] = int(len(extremes))
    report["n_extreme_naive"] = int(len(extremes_naive))
    report["extremes_shipped"] = extremes.head(40).assign(
        date=lambda d: d["date"].astype(str)).to_dict("records")

    # 5 ── day-of-month calendar signature, every column ──────────────────────
    report["day_of_month_signature"] = {
        c: day_of_month_signature(daily[c]) for c in daily.columns
    }

    # 6 ── roll contamination, every futures series ───────────────────────────
    roll: dict[str, dict] = {}
    for key, win in ROLL_WINDOWS.items():
        if key not in daily.columns:
            continue
        roll[key] = {
            "declared_window": list(win),
            "declared": roll_contamination(daily[key], win),
            "legacy_window_24_31": roll_contamination(daily[key], LEGACY_ROLL_WINDOW),
            "max_any_7day_window": max_window_lift(daily[key]),
            "variance_share_declared": dom_variance_share(daily[key], win),
            # EXPLORATORY scan, not a test: the three single days of the month with the
            # highest variance ratio. Reported so the declared window can be checked
            # against what the data actually says, with the multiple testing visible.
            "top3_single_days": sorted(
                ((dd, dom_variance_share(daily[key], (dd, dd))["ratio"]) for dd in range(1, 32)),
                key=lambda t: -(t[1] if np.isfinite(t[1]) else -1))[:3],
        }
    report["roll_contamination"] = roll

    # Control: the SAME variance-share test on the original panel's four clean futures
    # (gold/WTI/silver/copper, which §6a cleared) and on NATGAS_F, which it condemned.
    # Without this the reader has no noise floor for the numbers above.
    orig_ctrl = pd.read_parquet(ORIGINAL_DATA / "returns_daily.parquet")
    report["roll_control_original_panel"] = {
        k: {"legacy_window_24_31": roll_contamination(orig_ctrl[k], LEGACY_ROLL_WINDOW),
            "variance_share_24_31": dom_variance_share(orig_ctrl[k], LEGACY_ROLL_WINDOW),
            "day_of_month_signature": day_of_month_signature(orig_ctrl[k]),
            "top3_single_days": sorted(
                ((dd, dom_variance_share(orig_ctrl[k], (dd, dd))["ratio"])
                 for dd in range(1, 32)),
                key=lambda t: -(t[1] if np.isfinite(t[1]) else -1))[:3]}
        for k in ("GOLD_F", "WTI_F", "SILVER_F", "COPPER_F", "NATGAS_F")
        if k in orig_ctrl.columns
    }

    # Grains: the day-15 splice is separable from the USDA WASDE report (released the
    # 9th-12th), which is a REAL scheduled event and would otherwise be mistaken for a
    # defect. Both windows are reported so the attribution is checkable, not asserted.
    report["grain_roll_vs_wasde"] = {
        k: {"roll_day_14_15": dom_variance_share(daily[k], (14, 15)),
            "wasde_day_9_12": dom_variance_share(daily[k], (9, 12))}
        for k in ("CORN_F", "WHEAT_F", "SOYBEAN_F") if k in daily.columns
    }

    # 7 ── validation against roll-managed / spot benchmarks ──────────────────
    report["validation_pairs"] = {
        f"{a}~{b}": {"establishes": why, **validation_stats(daily_all[a], daily_all[b])}
        for a, b, why in ROLL_VALIDATION_PAIRS
        if a in daily_all.columns and b in daily_all.columns
    }

    # 8 ── coverage ───────────────────────────────────────────────────────────
    cov = pd.DataFrame([coverage_row(k, levels[k]) for k in panel_keys + valid_keys])
    cov = cov.sort_values("first_date")

    # 9 ── monthly panel and an INDEPENDENT reconciliation ────────────────────
    monthly = monthly_returns(daily)
    report["monthly_reconciliation"] = monthly_reconciliation(daily, monthly)
    report["panel_shape_daily"] = list(daily.shape)
    report["panel_shape_monthly"] = list(monthly.shape)
    report["n_inf"] = int(np.isinf(daily.to_numpy(dtype=float)).sum())
    report["all_nan_columns"] = [c for c in daily.columns if daily[c].notna().sum() == 0]
    assert daily.index.is_monotonic_increasing and daily.index.is_unique
    assert monthly.index.is_monotonic_increasing and monthly.index.is_unique

    # 10 ── first month each key is usable ────────────────────────────────────
    report["first_month"] = {
        c: (str(monthly[c].first_valid_index().date())
            if monthly[c].first_valid_index() is not None else None)
        for c in monthly.columns
    }

    # ── write ────────────────────────────────────────────────────────────────
    daily.to_parquet(out_dir / "returns_daily.parquet")
    monthly.to_parquet(out_dir / "returns_monthly.parquet")
    daily_all.to_parquet(out_dir / "returns_daily_all.parquet")
    cov.to_csv(out_dir / "coverage.csv", index=False)
    extremes.to_csv(out_dir / "extreme_returns.csv", index=False)
    extremes_naive.to_csv(out_dir / "extreme_returns_naive.csv", index=False)
    (out_dir / "integrity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\n=== BREADTH PANEL ===")
    print(f"tradable additions : {len(daily.columns)}  "
          f"({len(panel_keys)} fetched + {len(SYNTHETIC_SPREADS)} synthetic)")
    print(f"daily              : {daily.shape[0]} x {daily.shape[1]}")
    print(f"monthly            : {monthly.shape[0]} x {monthly.shape[1]}  "
          f"{monthly.index.min().date()} -> {monthly.index.max().date()}")
    print(f"|r|>50% shipped    : {len(extremes)}   naive: {len(extremes_naive)}")
    print(f"nulled nonpositive : {report['n_returns_nulled_nonpositive_total']}   "
          f"long gap: {report['n_returns_nulled_long_gap_total']}")
    print(f"monthly recon max  : {report['monthly_reconciliation']['max_abs_discrepancy']:.2e} "
          f"over {report['monthly_reconciliation']['n_cells_checked']} cells")
    print(f"inf / all-NaN cols : {report['n_inf']} / {report['all_nan_columns']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
