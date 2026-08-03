"""Extract the CORRECTED low-vol / quality band-B2 monthly series.

The inherited portfolio study used the REGISTERED low-vol book (net Sharpe 0.8779).
Iteration 10's independent adversarial verification
(`research/sleeves/lowvol_retest_verification.md`) established that three of the published
headline levels are WRONG, and that the corrected book is:

    gross 14.51%/yr, net 9.89%/yr, net Sharpe 0.614, benchmark 5.71% / 0.231,
    vol-matched active +6.18%/yr at NW t +2.12, DSR 0.586 vs a 0.9234 bar.

The three corrections, in the order the verification stacked them:
  1. the ACTIONS delisting window is off by one (strict `<` against a delisting dated ON
     the last traded bar) and rejected all 9 bankruptcies the book held;
  2. 21.4% of legs traded were charged zero transaction cost (uncharged exits);
  3. execution is moved to the next trading day's close.

This script re-executes exactly that corrected code path -- the SAME
`_lowvol_verify.attack5_structure.combined_repair` and the SAME next-trading-day frame
construction as `_lowvol_verify.attack7_final` -- and lifts the monthly arrays off the
returned `BandBooks`. It is NOT an approximate reconstruction.

It REFUSES TO PERSIST unless the regenerated series reproduces the verification's
corrected headline. That is the proof the series is the corrected one.

Writes only into `research/sleeves/_portfolio/` (concurrency scope).

    .venv/Scripts/python.exe -m research.sleeves._portfolio.extract_lowvol_corrected_monthly
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR, load_prices
from research.multiasset.carry import vol_matched_active
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.attack5_structure import combined_repair
from research.sleeves._lowvol_verify.build_frame import build
from research.validation import deflated_sharpe_ratio

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "research" / "sleeves" / "_portfolio"
BAND = "B2_200k_1M"
MPY = 12

# The verification's corrected headline, `lowvol_retest_verification.md` section 10,
# row "+ next-trading-day execution". Tolerances are the display precision of that table.
CORRECTED_TARGET = {
    "gross_annual": (0.1451, 5e-5),
    "net_annual": (0.0989, 5e-5),
    "net_sharpe": (0.614, 5e-4),
    "bench_annual": (0.0571, 5e-5),
    "bench_sharpe": (0.231, 5e-4),
    "vol_matched_active_annual": (0.0618, 5e-5),
    "vol_matched_active_tstat": (2.12, 5e-3),
    "dsr": (0.586, 5e-4),
}

log = logging.getLogger("extract_lowvol_corrected")


def _sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(MPY))


def next_trading_day_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """Move execution to the NEXT trading day's close -- verbatim from attack7_final."""
    prices = load_prices()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["row"] = np.arange(len(prices))
    full = pd.read_parquet(PANEL_DIR / "monthly_panel_dev.parquet",
                           columns=["ticker", "date", "closeadj"])
    key = prices.set_index(["ticker", "date"])["row"]
    full["row"] = [key.get((t, d), -1) for t, d in zip(full["ticker"], full["date"])]
    closeadj = prices["closeadj"].to_numpy()
    code = pd.factorize(prices["ticker"])[0]
    row = full["row"].to_numpy()
    nxt = row + 1
    ok = (row >= 0) & (nxt < len(prices))
    same = np.zeros(len(full), dtype=bool)
    same[ok] = code[nxt[ok]] == code[row[ok]]
    ex = np.full(len(full), np.nan)
    ex[same] = closeadj[nxt[same]]
    full["exec_price"] = ex
    full = full.sort_values(["ticker", "date"]).reset_index(drop=True)
    full["fwd_nextday"] = (full.groupby("ticker")["exec_price"].shift(-1)
                           / full["exec_price"] - 1.0)
    nextday = merged.merge(full[["ticker", "date", "fwd_nextday"]],
                           on=["ticker", "date"], how="left")
    nextday["forward_return"] = nextday["fwd_nextday"].where(
        nextday["fwd_nextday"].notna(), nextday["forward_return"])
    return nextday.drop(columns=["fwd_nextday"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")

    log.info("building the next-trading-day execution frame")
    nextday = next_trading_day_frame(merged)

    log.info("running the fully corrected band %s", BAND)
    books = combined_repair(nextday, BAND, delistings,
                            repair_delisting=True, charge_free_exits=True)

    gross = np.asarray(books.gross, dtype=float)
    net_cons = np.maximum(gross - np.asarray(books.cost_conservative, dtype=float), -1.0)
    net_real = np.maximum(gross - np.asarray(books.cost_realistic, dtype=float), -1.0)
    bench = np.asarray(books.benchmark, dtype=float)

    vm = vol_matched_active(pd.Series(net_cons), pd.Series(bench))
    got = {
        "gross_annual": INS.annual(gross),
        "net_annual": INS.annual(net_cons),
        "net_sharpe": _sharpe(net_cons),
        "bench_annual": INS.annual(bench),
        "bench_sharpe": _sharpe(bench),
        "vol_matched_active_annual": vm["vol_matched_active_annual"],
        "vol_matched_active_tstat": vm["vol_matched_active_tstat"],
        "dsr": float(deflated_sharpe_ratio(net_cons, n_trials=LV.N_TRIALS)),
    }

    failed = []
    for name, (want, tol) in CORRECTED_TARGET.items():
        delta = abs(got[name] - want)
        ok = delta <= tol
        log.info("  %-26s got %+.6f  verification %+.6f  delta %.2e  %s",
                 name, got[name], want, delta, "OK" if ok else "MISMATCH")
        if not ok:
            failed.append(name)
    if failed:
        log.error("REPRODUCTION FAILED for %s -- refusing to persist", failed)
        return 1

    # `books.date_convention` is FORMATION here (run_band's registered default), so this
    # index is the month the SIGNAL was formed and each row holds the FOLLOWING month's
    # return -- ONE MONTH EARLY. Recorded in the provenance below, compensated for by
    # `portfolio_correlation_v2.NEEDS_MONTH_SHIFT`, declared in
    # `research.sleeve_registry`, pinned by `tests/test_dating_alignment.py`. Do NOT join
    # this file to anything by date without shifting it +1 month.
    index = pd.PeriodIndex(books.months, freq="M").to_timestamp(how="end").normalize()
    frame = pd.DataFrame(
        {
            "gross": gross,
            "net_conservative": net_cons,
            "net_realistic": net_real,
            "benchmark": bench,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "lowvol_b2_corrected_monthly.parquet"
    frame.to_parquet(out)
    log.info("wrote %s  (%d months, %s -> %s)", out, len(frame),
             frame.index.min().date(), frame.index.max().date())

    (OUT_DIR / "lowvol_b2_corrected_provenance.json").write_text(
        json.dumps({
            "source": ("_lowvol_verify.attack5_structure.combined_repair on the "
                       "next-trading-day execution frame of _lowvol_verify.attack7_final"),
            "corrections": ["delisting ACTIONS window off-by-one repaired",
                            "previously-free exit legs charged",
                            "execution moved to next trading day's close"],
            "band": BAND,
            "reproduction_gate": "PASSED",
            "targets": {k: v[0] for k, v in CORRECTED_TARGET.items()},
            "measured": got,
            "n_months": int(len(frame)),
            "first_month": str(frame.index.min().date()),
            "last_month": str(frame.index.max().date()),
            "index_convention": books.date_convention,
            "index_convention_note": (
                "FORMATION means the label is the month the SIGNAL was formed and the "
                "row holds the FOLLOWING month's return; shift +1 month before joining "
                "by date."
            ),
            "return_convention": ("TOTAL return, not excess over cash. The registered "
                                  "low-vol convention carries no risk-free deduction; the "
                                  "multi-asset sleeves are EXCESS returns over the 13-week "
                                  "bill. Do not mix them without converting."),
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
