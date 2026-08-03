"""Emit k(t) into the banked trend CSV -- additive, byte-preserving, gated.

    .venv/Scripts/python.exe -m scripts.verify2a_emit_trend_scaler

Unblocks register item VERIFY-2(a) (`docs/project-control/RISK_AND_DEFECT_REGISTER.md`)
WITHOUT re-running the full pre-registered study: it recomputes ONLY the PRIMARY @ 20%
trend book -- the exact book `primary_20pct_monthly.csv` holds -- and appends its
book-vol scaler k(t) as a new final `scaler` column. DECISION-dated: the value on row t
was computed at month-end t and scales the weights held during t+1 (the same convention
as `DefensiveResult.scaler`).

Safety properties, in order:

1. REPRODUCTION GATE. The recomputed net_10bps / gross / bench_net_10bps must match the
   banked CSV within 1e-12 on every overlapping row, else NOTHING is written and the
   exit code is 3. (Measured drift on 2026-08-03: ~1e-16, pure float noise.)
2. BYTE-PRESERVING. Existing lines are kept as TEXT and only ",<value>" is appended to
   each, so every banked digit survives untouched. (A pandas rewrite would truncate the
   banked 17-significant-digit values to 16 -- measured -- which is why this script does
   not round-trip the frame.)
3. NOTHING ELSE. `result.json` and every other receipt are untouched. A full re-run of
   `scripts.run_multiasset_trend` also persists the column now, but rewrites every
   receipt with current-library float formatting; this script exists so the column can
   land without that.

After it has run, compute the measurement itself with
`.venv/Scripts/python.exe -m scripts.verify2a_scaler_correlation`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from research.sleeves.multiasset_trend import TrendConfig, load_excess_panel, run_trend

TREND_CSV = Path("research/sleeves/_multiasset_trend/primary_20pct_monthly.csv")
REPRODUCTION_TOL = 1e-12
VOL_TARGET = 0.20


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=TREND_CSV,
                    help="CSV to verify against and append to (default: the banked one)")
    args = ap.parse_args(argv)
    csv_path: Path = args.csv

    banked = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    if "scaler" in banked.columns:
        print(f"NOTHING TO DO: {csv_path} already has a `scaler` column.")
        return 0

    x, interior = load_excess_panel()
    ref = run_trend(TrendConfig(name="PRIMARY"), vol_target=VOL_TARGET,
                    x=x, interior=interior)

    recomputed = ref.net["10bps"].to_frame("net_10bps").assign(
        gross=ref.gross, bench_net_10bps=ref.bench_net["10bps"])
    overlap = banked.index.intersection(recomputed.index)
    if len(overlap) < len(banked.index):
        missing = len(banked.index) - len(overlap)
        print(f"REPRODUCTION FAILED: {missing} banked rows have no recomputed "
              "counterpart. Nothing written.")
        return 3
    drift = (banked.loc[overlap, ["net_10bps", "gross", "bench_net_10bps"]]
             - recomputed.loc[overlap]).abs().max().max()
    if not drift <= REPRODUCTION_TOL:
        print(f"REPRODUCTION FAILED: max |drift| {drift:.3e} exceeds "
              f"{REPRODUCTION_TOL:.0e}. The recomputed book is NOT the banked book -- "
              "nothing written. Investigate before emitting anything.")
        return 3

    scaler = ref.scaler.reindex(banked.index)
    if scaler.isna().any():
        n = int(scaler.isna().sum())
        print(f"REPRODUCTION FAILED: k(t) is missing on {n} banked rows. "
              "Nothing written.")
        return 3

    # Textual append: banked bytes survive exactly; only ",<k>" is added per line.
    with open(csv_path, "r", encoding="ascii", newline="") as fh:
        lines = fh.readlines()
    header = lines[0].rstrip("\r\n")
    eol = lines[0][len(header):] or "\n"
    out = [f"{header},scaler{eol}"]
    by_date = {d.strftime("%Y-%m-%d"): float(v) for d, v in scaler.items()}
    for line in lines[1:]:
        body = line.rstrip("\r\n")
        if not body:
            out.append(line)
            continue
        date_key = body.split(",", 1)[0]
        out.append(f"{body},{by_date[date_key]!r}{eol}")
    with open(csv_path, "w", encoding="ascii", newline="") as fh:
        fh.writelines(out)

    print(f"OK: appended `scaler` to {csv_path} "
          f"({len(out) - 1} rows, reproduction drift {drift:.3e} <= "
          f"{REPRODUCTION_TOL:.0e}, banked columns byte-preserved).")
    print("Next: .venv/Scripts/python.exe -m scripts.verify2a_scaler_correlation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
