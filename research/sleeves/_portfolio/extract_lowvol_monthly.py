"""Extract the LOW-VOL / QUALITY band-B2 monthly net return series.

The registered low-vol re-test (`research/sleeves/lowvol_retest_run.py`) persisted only
BAND-LEVEL SUMMARY STATISTICS to `lowvol_retest_result.json`. It never wrote a per-month
return series to disk. Every other sleeve did.

This script does NOT reconstruct the series approximately. It re-executes the registered
code path -- `load_universe` -> `attach_spread_bounds` -> `build_signal` -> `run_band` --
with the registered defaults, and lifts the monthly arrays straight off the returned
`BandBooks` dataclass, which already carries them (`months`, `gross`, `cost_conservative`,
`cost_realistic`, `benchmark`).

It then REFUSES TO PERSIST unless the regenerated series reproduces the registered
summary statistics in `lowvol_retest_result.json` to 1e-9. That is the proof that the
series is the registered one and not a lookalike.

Writes only into `research/sleeves/_portfolio/` (concurrency scope).

    .venv/Scripts/python.exe -m research.sleeves._portfolio.extract_lowvol_monthly
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.capacity_panel import DEV_CUTOFF, PANEL_DIR
from research.sleeves.low_vol_quality import build_signal
from research.sleeves.lowvol_retest import attach_spread_bounds, run_band
from research.sleeves.lowvol_retest_data import QUALITY_CACHE, load_universe

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "research" / "sleeves" / "_portfolio"
RESULT_JSON = REPO / "research" / "sleeves" / "lowvol_retest_result.json"
RISK_CACHE = PANEL_DIR / "risk_features_dev.parquet"

BAND = "B2_200k_1M"
MONTHS_PER_YEAR = 12
TOL = 1e-9

log = logging.getLogger("extract_lowvol")


def _sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    registered = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    band_ref = next(b for b in registered["bands"] if b["band"] == BAND)

    universe = load_universe()
    if universe["date"].max() > DEV_CUTOFF:
        raise ValueError("confirmation window leaked into the universe")
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    risk = pd.read_parquet(RISK_CACHE)
    quality = pd.read_parquet(QUALITY_CACHE)

    merged = (
        universe
        .merge(risk, on=["ticker", "date"], how="left")
        .merge(quality, on=["ticker", "date"], how="left")
    )
    merged = attach_spread_bounds(merged)
    merged = build_signal(merged)

    log.info("running registered band %s", BAND)
    books = run_band(merged, BAND, delistings)
    if books is None:
        raise RuntimeError(f"{BAND} produced no book")

    gross = np.asarray(books.gross, dtype=float)
    net_cons = gross - np.asarray(books.cost_conservative, dtype=float)
    net_real = gross - np.asarray(books.cost_realistic, dtype=float)
    bench = np.asarray(books.benchmark, dtype=float)

    # ---- REPRODUCTION GATE -------------------------------------------------------
    checks: list[tuple[str, float, float]] = [
        ("n_months", float(len(gross)), float(band_ref["n_months"])),
        ("gross_sharpe", _sharpe(gross), band_ref["gross"]["sharpe"]),
        ("gross_vol", float(np.std(gross, ddof=1) * np.sqrt(MONTHS_PER_YEAR)),
         band_ref["gross"]["volatility"]),
        ("bench_sharpe", _sharpe(bench), band_ref["benchmark"]["sharpe"]),
        ("bench_vol", float(np.std(bench, ddof=1) * np.sqrt(MONTHS_PER_YEAR)),
         band_ref["benchmark"]["volatility"]),
        ("net_cons_sharpe", _sharpe(net_cons),
         band_ref["bounds"]["conservative"]["net"]["sharpe"]),
        ("net_real_sharpe", _sharpe(net_real),
         band_ref["bounds"]["realistic"]["net"]["sharpe"]),
    ]
    failed = []
    for name, got, want in checks:
        delta = abs(got - want)
        ok = delta <= TOL * max(1.0, abs(want))
        log.info("  %-16s got %.12f  registered %.12f  %s",
                 name, got, want, "OK" if ok else "MISMATCH")
        if not ok:
            failed.append((name, got, want, delta))
    if failed:
        for name, got, want, delta in failed:
            log.error("REPRODUCTION FAILED %s: %.12f vs %.12f (delta %.3e)",
                      name, got, want, delta)
        log.error("refusing to persist a series that is not the registered one")
        return 1

    # `books.date_convention` is FORMATION here (run_band's registered default), so this
    # index is the month the SIGNAL was formed and each row holds the FOLLOWING month's
    # return -- ONE MONTH EARLY. It is recorded in the provenance below, compensated for
    # by `portfolio_correlation_v2.NEEDS_MONTH_SHIFT`, declared in
    # `research.sleeve_registry` and pinned by `tests/test_dating_alignment.py`. Do NOT
    # join this file to anything by date without shifting it +1 month.
    index = pd.PeriodIndex(books.months, freq="M").to_timestamp(how="end").normalize()
    frame = pd.DataFrame(
        {
            "gross": gross,
            "net_conservative": net_cons,
            "net_realistic": net_real,
            "benchmark": bench,
            "benchmark_rankable": np.asarray(books.benchmark_rankable, dtype=float),
        },
        index=pd.DatetimeIndex(index, name="date"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lowvol_b2_net_monthly.parquet"
    frame.to_parquet(out)
    log.info("wrote %s  (%d months, %s -> %s)", out, len(frame),
             frame.index.min().date(), frame.index.max().date())

    meta = {
        "source": "re-run of the registered lowvol_retest code path (run_band defaults)",
        "band": BAND,
        "reproduction_gate": "PASSED",
        "tolerance": TOL,
        "checks": [{"name": n, "regenerated": g, "registered": w} for n, g, w in checks],
        "n_months": int(len(frame)),
        "first_month": str(frame.index.min().date()),
        "last_month": str(frame.index.max().date()),
        "index_convention": books.date_convention,
        "index_convention_note": (
            "FORMATION means the label is the month the SIGNAL was formed and the row "
            "holds the FOLLOWING month's return; shift +1 month before joining by date."
        ),
    }
    (OUT_DIR / "lowvol_b2_provenance.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
