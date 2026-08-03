"""Licence-compliance purge of the raw Sharadar export (Nasdaq Personal Use terms §5.3:
all copies of the Data must be purged within 30 days of subscription termination).

Removes RAW Sharadar data only — the full SF1/SEP export CSVs, sample CSVs, leftover
export ZIPs, any parquet checkpoints derived row-for-row from them, and the API key.
It does NOT touch Derived Data, which we own outright (§6.2): study verdicts, learned
weights, validation statistics and the banked result documents in
``research/medallion_style_alpha_search/``.

Dry-run by default; pass ``--confirm`` to delete. Writes a dated purge record next to
the banked results so the compliance trail is part of the repo history.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "_data" / "sharadar"
RECORD = REPO / "research" / "medallion_style_alpha_search" / "sharadar_purge_record.md"

# Everything under _data/sharadar is either raw Data, a row-level derivative of it,
# the API key, or run scaffolding (logs/batch files) — all safe to remove. Banked
# Derived Data lives in research/, never here. The dev/ and confirm/ parquet extracts
# are row-level COPIES of the Data and are covered by the purge obligation.
# NOTE the recursive patterns. An earlier version globbed only the data root, which
# silently MISSED `panel/*.parquet` -- gigabytes of row-level derivatives of the licensed
# Data, sitting in a subdirectory the glob never reached. A purge that leaves the Data on
# disk is a compliance failure that reports success, so these are now recursive and a
# post-purge sweep below fails loudly on any survivor.
PURGE_GLOBS = ("**/*.csv", "**/*.zip", "**/*.parquet", "**/*.log", "**/*.err",
               "**/*.bat", "**/*.marker", "**/*.json", "ndl_api_key.txt")

# The download manifest holds no Data — only SHA-256 digests, row counts and vendor
# snapshot times. That is Derived Data (§6.2, ours outright), and it is the only proof of
# which data vintage a banked study ran against, so it must survive the purge. It is
# RESCUED into research/ rather than deleted, which also leaves _data/sharadar genuinely
# empty so the record can honestly report zero files remaining.
MANIFEST = DATA_DIR / "download_manifest.json"
MANIFEST_ARCHIVE = (REPO / "research" / "medallion_style_alpha_search"
                    / "sharadar_download_manifest.json")


def targets() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in PURGE_GLOBS:
        for p in sorted(DATA_DIR.glob(pattern)):
            if p.is_file():
                seen[p] = None
    return list(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm", action="store_true",
                    help="Actually delete (default: dry-run listing).")
    args = ap.parse_args()

    files = targets()
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    for p in files:
        print(f"{'DELETE' if args.confirm else 'would delete'}  "
              f"{p.relative_to(REPO)}  ({p.stat().st_size/1e6:,.1f} MB)")
    print(f"{len(files)} file(s), {total_mb:,.0f} MB total")

    if not args.confirm:
        print("\nDry-run only. Re-run with --confirm to purge.")
        return 0

    rescued = False
    if MANIFEST.exists():
        MANIFEST_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(MANIFEST), str(MANIFEST_ARCHIVE))
        rescued = True

    for p in files:
        # missing_ok: the manifest is in `files` but was already moved out by the rescue
        # above, and a crashed prior run may have removed others.
        p.unlink(missing_ok=True)
    leftovers = [p for p in DATA_DIR.rglob("*") if p.is_file()]
    if leftovers:
        # Fail loudly rather than writing a record that claims a clean purge.
        print("\nERROR: files remain under _data/sharadar after purge:",
              file=sys.stderr)
        for leftover in leftovers:
            print(f"  {leftover.relative_to(REPO)}", file=sys.stderr)
        print("The purge is INCOMPLETE. Do not treat this as licence-compliant.",
              file=sys.stderr)
        return 1
    lines = [
        "",
        "---",
        "",
        f"## Purge — {date.today().isoformat()}",
        "",
        "Per Nasdaq Data Link Personal Use terms §5.3 (purge within 30 days of",
        "termination), the raw Sharadar export was deleted from this machine:",
        "",
        *[f"- `{f.relative_to(REPO)}`" for f in files],
        "",
        f"Files remaining under `_data/sharadar/` after purge: {len(leftovers)}",
        "",
        "Derived Data retained under §6.2 (owned outright): the banked study verdict,",
        "validation statistics and learned weights in research/medallion_style_alpha_search/.",
    ]
    if rescued:
        lines += [
            "",
            f"Provenance manifest rescued to `{MANIFEST_ARCHIVE.relative_to(REPO)}` "
            "(hashes and row counts only, no Data).",
        ]
    # Append, never overwrite: a compliance trail should accumulate. A subscription can
    # be taken out and purged more than once, and each cycle is part of the record.
    header = "" if RECORD.exists() else "# Sharadar raw-data purge record\n"
    with RECORD.open("a", encoding="utf-8") as handle:
        handle.write(header + "\n".join(lines) + "\n")
    print(f"\nPurged. Record appended to {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
