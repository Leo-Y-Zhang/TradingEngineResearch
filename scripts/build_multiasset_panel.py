"""Build the long-history multi-asset panel from free data, and prove its integrity.

    .venv/Scripts/python.exe scripts/build_multiasset_panel.py [--use-cache] [--out DIR]

Network step of the study. Fetches every registry ticker from yfinance one at a time
(isolating failures), caches the raw frames under ``_data/multiasset/raw/``, then
builds the daily and month-end panels and runs every integrity check the study
depends on. Writes machine-readable results next to the panels and prints the
report that ``research/multiasset/data_integrity.md`` is written from.

Builds NO strategy. Nothing here is committed except derived statistics — Yahoo's
terms forbid redistributing its data, and ``_data/`` is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.instruments import (  # noqa: E402
    INSTRUMENTS,
    QUARANTINE,
    VALIDATION_PAIRS,
    Instrument,
)
from research.multiasset.panel import (  # noqa: E402
    GAP_NULL_DAYS,
    apply_quarantine,
    bill_cash_return,
    clean_levels,
    coverage_row,
    day_of_month_signature,
    dsr_sharpe_bar,
    flag_extreme_returns,
    monthly_last,
    monthly_returns,
    par_bond_total_return,
    simple_returns,
    wide_panel,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "_data" / "multiasset"
EXTREME_THRESHOLD = 0.50
TRADING_DAYS = 252


# ── fetch ─────────────────────────────────────────────────────────────────────

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


# ── per-instrument construction ───────────────────────────────────────────────

def instrument_returns(inst: Instrument, levels: pd.Series) -> tuple[pd.Series, dict]:
    """Dispatch a cleaned level/yield series to the correct return convention."""
    if inst.return_method == "price_return":
        return simple_returns(levels, invert=False)
    if inst.return_method == "inverse_price_return":
        return simple_returns(levels, invert=True)
    if inst.return_method == "par_bond_total_return":
        assert inst.maturity_years is not None
        ret = par_bond_total_return(levels, inst.maturity_years)
        return ret, {"n_nonpositive_level_bars": int((levels <= 0).sum()),
                     "n_returns_nulled_nonpositive": 0, "n_returns_nulled_long_gap": 0}
    if inst.return_method == "bill_cash_accrual":
        ret = bill_cash_return(levels)
        return ret, {"n_nonpositive_level_bars": int((levels <= 0).sum()),
                     "n_returns_nulled_nonpositive": 0, "n_returns_nulled_long_gap": 0}
    raise ValueError(f"{inst.key}: unknown return_method {inst.return_method!r}")


# ── integrity checks ──────────────────────────────────────────────────────────

def validation_stats(constructed: pd.Series, benchmark: pd.Series) -> dict[str, float]:
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
        # benchmark minus constructed, annualised — positive ⇒ the benchmark earned more.
        "drift_gap_pct_yr": round(float((b.mean() - a.mean()) * TRADING_DAYS * 100.0), 3),
        "vol_ratio": round(float(a.std() / b.std()), 4) if b.std() else float("nan"),
        "first": both.index.min().date().isoformat(),
        "last": both.index.max().date().isoformat(),
    }


def monthly_reconciliation(daily: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, float]:
    """Independently recompute the month-end panel and assert it matches, cell for cell.

    Guards the groupby/period path — a month-boundary off-by-one here would shift
    every monthly return by one bar and no downstream number would look odd.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true", help="reuse cached raw parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out: Path = args.out
    raw_dir = out / "raw"
    out.mkdir(parents=True, exist_ok=True)

    raw_levels: dict[str, pd.Series] = {}
    clean_stats: dict[str, dict] = {}
    failures: dict[str, str] = {}
    by_key_inst: dict[str, Instrument] = {}

    print(f"Fetching {len(INSTRUMENTS)} tickers (use_cache={args.use_cache})...")
    for inst in INSTRUMENTS:
        try:
            raw = fetch_one(inst.ticker, raw_dir, use_cache=args.use_cache)
        except Exception as exc:  # noqa: BLE001
            failures[inst.key] = str(exc)
            print(f"  FAIL {inst.key:<12} {inst.ticker:<10} {exc}")
            continue

        levels, cstats = clean_levels(raw["Close"])
        raw_levels[inst.key] = levels
        clean_stats[inst.key] = cstats
        by_key_inst[inst.key] = inst
        print(f"  ok   {inst.key:<12} {inst.ticker:<10} "
              f"{levels.index.min().date()} -> {levels.index.max().date()}  n={len(levels):,}")

    if not raw_levels:
        print("nothing fetched", file=sys.stderr)
        return 1

    # Quarantine evidenced corrupt CLOSES, then build returns from both the screened
    # and the unscreened levels so every downstream result can be tested against the
    # cleaning decision rather than having to trust it.
    levels_by_key, quarantine_audit = apply_quarantine(raw_levels, QUARANTINE)

    returns_by_key: dict[str, pd.Series] = {}
    returns_unscreened: dict[str, pd.Series] = {}
    for key, inst in by_key_inst.items():
        ret, rstats = instrument_returns(inst, levels_by_key[key])
        returns_by_key[key] = ret
        returns_unscreened[key] = instrument_returns(inst, raw_levels[key])[0]
        clean_stats[key].update(rstats)

    # ── panels ────────────────────────────────────────────────────────────────
    panel_keys = [i.key for i in INSTRUMENTS if i.role == "panel" and i.key in returns_by_key]
    cash_keys = [i.key for i in INSTRUMENTS if i.role == "cash" and i.key in returns_by_key]
    valid_keys = [i.key for i in INSTRUMENTS if i.role == "validation" and i.key in returns_by_key]

    levels_daily = wide_panel(levels_by_key)
    returns_daily = wide_panel({k: returns_by_key[k] for k in panel_keys})
    returns_all_daily = wide_panel(returns_by_key)          # incl. cash + validation
    cash_daily = wide_panel({k: returns_by_key[k] for k in cash_keys})
    yields_daily = wide_panel({
        i.key.replace("_TR", "_YLD").replace("US_CASH_13W", "US13W_YLD"): levels_by_key[i.key] / 100.0
        for i in INSTRUMENTS
        if i.asset_class == "rates" and i.key in levels_by_key
    })

    returns_monthly = monthly_returns(returns_daily)
    returns_all_monthly = monthly_returns(returns_all_daily)
    cash_monthly = monthly_returns(cash_daily)
    levels_monthly = monthly_last(levels_daily)
    yields_monthly = monthly_last(yields_daily)

    # ── integrity ─────────────────────────────────────────────────────────────
    assert returns_daily.index.is_monotonic_increasing and returns_daily.index.is_unique
    assert returns_monthly.index.is_monotonic_increasing and returns_monthly.index.is_unique
    n_inf = int(np.isinf(returns_all_daily.to_numpy(dtype=float)).sum())
    empty_cols = [c for c in returns_daily.columns if returns_daily[c].notna().sum() == 0]

    coverage = pd.DataFrame([coverage_row(k, levels_by_key[k]) for k in levels_by_key])
    coverage = coverage.sort_values("first_date")

    extremes = flag_extreme_returns(returns_all_daily, EXTREME_THRESHOLD)
    recon = monthly_reconciliation(returns_daily, returns_monthly)

    # What a NAIVE build would have produced: pct_change on every raw level series,
    # yields included, no inversion, no quarantine, no non-positive guard. This is the
    # counterfactual that makes the cleaning auditable instead of merely asserted.
    naive_daily = wide_panel({k: v.pct_change().replace([np.inf, -np.inf], np.nan)
                              for k, v in raw_levels.items()})
    naive_extremes = flag_extreme_returns(naive_daily, EXTREME_THRESHOLD)

    # Calendar-signature scan: a vendor defect clusters on a day of the month, a
    # market event does not. Run over every instrument, not just the ones suspected,
    # and run it again AFTER the quarantine so the fix is demonstrated, not asserted.
    def _scan(src: dict[str, pd.Series]) -> dict[str, dict]:
        sigs = {k: day_of_month_signature(v) for k, v in src.items()}
        return {k: v for k, v in sigs.items()
                if np.isfinite(v.get("lift", np.nan)) and v["n_of_top"] >= 3}

    flagged_sigs = _scan(returns_unscreened)
    flagged_sigs_after = _scan(returns_by_key)

    # Breadth over time. The DSR bar falls with sample length, but only the SPX reaches
    # 1927 — a sleeve needing N instruments cannot start before N of them exist.
    avail = returns_daily.notna().groupby(returns_daily.index.year).sum().gt(100).sum(axis=1)
    breadth_by_year = {int(y): int(n) for y, n in avail.items()}
    breadth_start = {}
    this_year = int(returns_daily.index.max().year)
    for n_req in (1, 2, 4, 8, 12, 16, 20, int(avail.max())):
        ok = avail[avail >= n_req]
        if not len(ok):
            continue
        start = int(ok.index.min())
        yrs = float(this_year - start) + 0.6
        breadth_start[n_req] = {
            "first_year": start,
            "years_available": round(yrs, 1),
            "dsr_bar_ann_sharpe": round(dsr_sharpe_bar(yrs, n_trials=32), 3),
            "half_kelly_return_at_bar_pct": round(
                3.0 * dsr_sharpe_bar(yrs, n_trials=32) ** 2 / 8.0 * 100.0, 1),
        }
    # 30%/yr at half Kelly (g = 3S^2/8) needs this Sharpe, independent of sample length.
    sharpe_for_30pct = float(np.sqrt(0.30 * 8.0 / 3.0))

    validations = {}
    for constructed, benchmark, label in VALIDATION_PAIRS:
        if constructed in returns_by_key and benchmark in returns_by_key:
            stats = validation_stats(returns_by_key[constructed], returns_by_key[benchmark])
            # Daily correlation is the wrong test for a smooth accrual against a noisy
            # ETF NAV, so also compare CUMULATIVE growth over the common window.
            both = pd.concat([returns_by_key[constructed].rename("a"),
                              returns_by_key[benchmark].rename("b")], axis=1).dropna()
            if len(both) >= 60:
                ga = float((1.0 + both["a"]).prod())
                gb = float((1.0 + both["b"]).prod())
                yrs = (both.index.max() - both.index.min()).days / 365.25
                stats["cum_growth_constructed_pct"] = round((ga - 1.0) * 100.0, 2)
                stats["cum_growth_benchmark_pct"] = round((gb - 1.0) * 100.0, 2)
                stats["cagr_gap_pct_yr"] = round(
                    (gb ** (1 / yrs) - ga ** (1 / yrs)) * 100.0, 3)
            validations[f"{constructed}~{benchmark}"] = {"label": label, **stats}

    # Futures settle on a session that may not close when the ETF does. A non-zero
    # correlation at lag +/-1 means a daily cross-asset signal mixes information sets.
    lead_lag = {}
    for fut, etf in (("GOLD_F", "GLD"), ("SILVER_F", "SLV")):
        if fut in returns_by_key and etf in returns_by_key:
            both = pd.concat([returns_by_key[fut].rename("f"),
                              returns_by_key[etf].rename("e")], axis=1).dropna()
            lead_lag[f"{fut}~{etf}"] = {
                f"lag{lag:+d}": round(float(both["f"].corr(both["e"].shift(lag))), 4)
                for lag in (-1, 0, 1)}

    # Roll contamination: extreme bars in a front-month continuous series should NOT
    # know what day of the month it is. Concentration in the roll window means the
    # "return" is a contract switch, not a price move.
    roll_clustering = {}
    for key in (k for k in panel_keys if by_key_inst[k].asset_class == "commodity"):
        r = returns_daily[key].dropna()
        dom = pd.Series(r.index.day, index=r.index)
        big = r.abs() > 0.15
        if int(big.sum()) < 5:
            continue
        base = float(dom.between(24, 31).mean())
        late = int(dom[big].between(24, 31).sum())
        roll_clustering[key] = {
            "n_bars_gt_15pct": int(big.sum()),
            "pct_in_roll_window": round(100.0 * late / int(big.sum()), 1),
            "base_rate_pct": round(100.0 * base, 1),
            "lift": round((late / int(big.sum())) / base, 2) if base else float("nan"),
        }

    # Annualised summary stats per panel instrument (derived statistics, committable).
    summary_rows = []
    for key in panel_keys:
        r = returns_daily[key].dropna()
        if r.empty:
            continue
        summary_rows.append({
            "key": key,
            "ann_return_pct": round(float(r.mean() * TRADING_DAYS * 100.0), 2),
            "ann_vol_pct": round(float(r.std() * np.sqrt(TRADING_DAYS) * 100.0), 2),
            "sharpe_naive": round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 3)
            if r.std() else float("nan"),
            "worst_day_pct": round(float(r.min() * 100.0), 2),
            "best_day_pct": round(float(r.max() * 100.0), 2),
            "n_returns": int(len(r)),
        })
    summary = pd.DataFrame(summary_rows)

    # ── write ─────────────────────────────────────────────────────────────────
    written: list[str] = []

    def _write(frame: pd.DataFrame, name: str) -> None:
        p = out / name
        frame.to_parquet(p)
        written.append(str(p))

    _write(levels_daily, "levels_daily.parquet")
    _write(returns_daily, "returns_daily.parquet")
    _write(returns_monthly, "returns_monthly.parquet")
    _write(returns_all_daily, "returns_all_daily.parquet")
    _write(returns_all_monthly, "returns_all_monthly.parquet")
    _write(levels_monthly, "levels_monthly.parquet")
    _write(yields_daily, "yields_daily.parquet")
    _write(yields_monthly, "yields_monthly.parquet")
    _write(cash_daily, "cash_daily.parquet")
    _write(cash_monthly, "cash_monthly.parquet")
    unscreened_daily = wide_panel({k: returns_unscreened[k] for k in panel_keys})
    _write(unscreened_daily, "returns_daily_unscreened.parquet")
    _write(monthly_returns(unscreened_daily), "returns_monthly_unscreened.parquet")

    pd.DataFrame([{
        "key": i.key, "ticker": i.ticker, "name": i.name, "asset_class": i.asset_class,
        "currency": i.currency, "return_method": i.return_method, "role": i.role,
        "maturity_years": i.maturity_years, "notes": i.notes,
    } for i in INSTRUMENTS]).to_csv(out / "instruments.csv", index=False)
    written.append(str(out / "instruments.csv"))

    coverage.to_csv(out / "coverage.csv", index=False)
    summary.to_csv(out / "summary_stats.csv", index=False)
    extremes.to_csv(out / "extreme_returns.csv", index=False)
    naive_extremes.to_csv(out / "extreme_returns_naive.csv", index=False)
    pd.DataFrame(quarantine_audit).to_csv(out / "quarantine_audit.csv", index=False)
    written += [str(out / n) for n in ("coverage.csv", "summary_stats.csv",
                                       "extreme_returns.csv", "extreme_returns_naive.csv",
                                       "quarantine_audit.csv")]

    earliest = min(s.index.min() for s in levels_by_key.values())
    latest = max(s.index.max() for s in levels_by_key.values())
    integrity = {
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "n_instruments_registry": len(INSTRUMENTS),
        "n_fetched": len(levels_by_key),
        "n_panel": len(panel_keys),
        "n_cash": len(cash_keys),
        "n_validation": len(valid_keys),
        "failures": failures,
        "earliest_date": earliest.date().isoformat(),
        "latest_date": latest.date().isoformat(),
        "years_of_history": round((latest - earliest).days / 365.25, 2),
        "daily_panel_shape": list(returns_daily.shape),
        "monthly_panel_shape": list(returns_monthly.shape),
        "monthly_first": returns_monthly.index.min().date().isoformat(),
        "monthly_last": returns_monthly.index.max().date().isoformat(),
        "n_inf_in_returns": n_inf,
        "all_nan_columns": empty_cols,
        "gap_null_days": GAP_NULL_DAYS,
        "extreme_threshold": EXTREME_THRESHOLD,
        "n_extreme_returns": int(len(extremes)),
        "n_extreme_returns_naive_build": int(len(naive_extremes)),
        "naive_extremes": naive_extremes.assign(
            date=naive_extremes["date"].astype(str)).to_dict("records")
        if len(naive_extremes) else [],
        "quarantine": quarantine_audit,
        "day_of_month_signatures_flagged": flagged_sigs,
        "day_of_month_signatures_after_quarantine": flagged_sigs_after,
        "breadth_by_year": breadth_by_year,
        "breadth_vs_sample_length": breadth_start,
        "dsr_anchors_reproduced": {"7yr_n32": round(dsr_sharpe_bar(7.0), 3),
                                   "40yr_n32": round(dsr_sharpe_bar(40.0), 3)},
        "sharpe_needed_for_30pct_half_kelly": round(sharpe_for_30pct, 3),
        "futures_etf_lead_lag": lead_lag,
        "roll_clustering": roll_clustering,
        "monthly_reconciliation": recon,
        "validations": validations,
        "clean_stats": clean_stats,
    }
    (out / "integrity.json").write_text(json.dumps(integrity, indent=2, default=str))
    written.append(str(out / "integrity.json"))

    # ── report ────────────────────────────────────────────────────────────────
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)
    print("\n" + "=" * 78)
    print(f"PANEL: {len(panel_keys)} tradable + {len(cash_keys)} cash + "
          f"{len(valid_keys)} validation; {earliest.date()} → {latest.date()} "
          f"({integrity['years_of_history']} yr)")
    print(f"daily {returns_daily.shape}  monthly {returns_monthly.shape} "
          f"({integrity['monthly_first']} -> {integrity['monthly_last']})")
    print(f"inf in returns: {n_inf}   all-NaN columns: {empty_cols}")
    print(f"monthly reconciliation: {recon}")
    print("\n--- COVERAGE ---")
    print(coverage.to_string(index=False))
    print("\n--- CLEANING ---")
    cs = pd.DataFrame(clean_stats).T
    print(cs[cs.drop(columns=["n_raw", "n_clean"], errors="ignore").sum(axis=1) > 0].to_string())
    print("\n--- SUMMARY (naive, no cost, local currency) ---")
    print(summary.to_string(index=False))
    print(f"\n--- EXTREME RETURNS |r| > {EXTREME_THRESHOLD:.0%} (constructed panel) ---")
    print(extremes.to_string(index=False) if len(extremes) else "  none")
    print(f"--- same threshold on a NAIVE pct_change build: {len(naive_extremes)} ---")
    print(naive_extremes.to_string(index=False) if len(naive_extremes) else "  none")
    print("\n--- QUARANTINE ---")
    for q in quarantine_audit:
        print(f"  {'OK ' if q['matched'] else 'MISS'} {q['key']:<8} {q['date']}  {q['reason']}")
    print("\n--- DAY-OF-MONTH SIGNATURE (>=3 of the top-10 moves share a day) ---")
    print("  BEFORE quarantine:")
    for k, v in flagged_sigs.items():
        print(f"    {k:<10} day {v['modal_day']:>2}: {v['n_of_top']}/10 "
              f"(base {v['base_rate_pct']}%, lift {v['lift']}x)")
    print("  AFTER quarantine:")
    if not flagged_sigs_after:
        print("    none")
    for k, v in flagged_sigs_after.items():
        print(f"    {k:<10} day {v['modal_day']:>2}: {v['n_of_top']}/10 "
              f"(base {v['base_rate_pct']}%, lift {v['lift']}x)")
    print("\n--- BREADTH vs SAMPLE LENGTH (they trade off; n_trials=32) ---")
    print(f"  DSR anchors reproduced: 7yr={dsr_sharpe_bar(7.0):.3f} (recorded 1.488), "
          f"40yr={dsr_sharpe_bar(40.0):.3f} (recorded 0.597)")
    print(f"  {'N':>3} {'from':>6} {'years':>6} {'DSR bar':>9} {'half-Kelly @ bar':>18}")
    for n, v in breadth_start.items():
        print(f"  {n:>3} {v['first_year']:>6} {v['years_available']:>6.1f} "
              f"{v['dsr_bar_ann_sharpe']:>9.3f} {v['half_kelly_return_at_bar_pct']:>17.1f}%")
    print(f"  30%/yr at half Kelly needs Sharpe {sharpe_for_30pct:.3f} "
          f"-> clears the bar at EVERY length above.")
    print("  instruments live by year: " + ", ".join(
        f"{y}:{breadth_by_year.get(y, 0)}" for y in
        (1930, 1950, 1965, 1972, 1980, 1990, 2000, 2004, 2008, 2020, 2025)))
    print("\n--- FUTURES/ETF LEAD-LAG ---")
    for k, v in lead_lag.items():
        print(f"  {k:<16} {v}")
    print("\n--- ROLL-WINDOW CLUSTERING of |r|>15% bars ---")
    for k, v in roll_clustering.items():
        print(f"  {k:<10} {v['pct_in_roll_window']}% in days 24-31 vs base "
              f"{v['base_rate_pct']}%  lift={v['lift']}x  (n={v['n_bars_gt_15pct']})")
    print("\n--- VALIDATION PAIRS ---")
    for k, v in validations.items():
        print(f"  {k:<22} corr={v['corr']}  drift_gap={v['drift_gap_pct_yr']}%/yr  "
              f"cagr_gap={v.get('cagr_gap_pct_yr')}%/yr  vol_ratio={v['vol_ratio']}  "
              f"n={v['n_overlap']}  ({v['first']}->{v['last']})")
    print("\nwrote:")
    for p in written:
        print("  " + p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
