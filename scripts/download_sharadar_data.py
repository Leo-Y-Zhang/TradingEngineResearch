"""Re-export the raw Sharadar tables from Nasdaq Data Link into ``_data/sharadar/``.

The counterpart to ``purge_sharadar_data.py``. That script deletes the licensed raw
Data (and must be re-run within 30 days of subscription termination); this one fetches
it back while the subscription is live. Neither touches Derived Data in
``research/medallion_style_alpha_search/``, which we own outright under §6.2.

Uses the bulk-export endpoint (``qopts.export=true``), which asks Nasdaq to build a
zipped snapshot of the whole table, then polls until the snapshot is ``fresh`` and
streams it down. This is the documented path for full-table access and is far cheaper
on their side (and ours) than paginating millions of rows.

The API key is read from ``_data/sharadar/ndl_api_key.txt`` (gitignored) and is never
printed, logged, or written into the manifest.

Provenance: every downloaded table is recorded in ``download_manifest.json`` with its
row count, byte size, SHA-256, the vendor's snapshot time and our fetch time. A study
that cites this data can therefore prove which vintage it ran against -- and a re-run
after a vendor restatement will show up as a hash change rather than a silent drift.

Usage::

    python scripts/download_sharadar_data.py                  # all tables
    python scripts/download_sharadar_data.py --tables SEP SF1 # a subset
    python scripts/download_sharadar_data.py --force          # re-fetch existing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "_data" / "sharadar"
KEY_FILE = DATA_DIR / "ndl_api_key.txt"
MANIFEST = DATA_DIR / "download_manifest.json"

# The ``.json`` suffix is required: the datatables endpoint 404s without an explicit
# format, and the HTML 404 body makes that look like an auth failure if you skip it.
BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"
USER_AGENT = "tradingengine-research/1.0"

# Ordered smallest-first so a wrong key or a lapsed subscription fails in seconds
# rather than after a multi-gigabyte transfer.
ALL_TABLES = ("ACTIONS", "TICKERS", "SF2", "DAILY", "SF3", "SF1", "SEP")

POLL_SECONDS = 15
POLL_TIMEOUT_SECONDS = 3600
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A table could not be exported or fetched."""


def load_key() -> str:
    """Prefer the NDL_API_KEY environment variable over the on-disk file.

    The file handoff has a leak the gitignore cannot fix: a key pasted into an on-disk
    template can be captured by editors, sync tools, or session tooling regardless of
    what git ignores. An environment variable is read without the value ever being
    displayed. Set it with:

        setx NDL_API_KEY "<key>"      (new shells pick it up)

    The file remains supported so existing setups keep working.
    """
    from_env = os.environ.get("NDL_API_KEY", "").strip()
    if from_env:
        return from_env

    if not KEY_FILE.exists():
        raise DownloadError(
            f"No API key at {KEY_FILE}. Create it with the key from "
            "data.nasdaq.com/account/profile (the file is gitignored)."
        )
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("PASTE_"):
            return candidate
    raise DownloadError(
        f"No API key in $NDL_API_KEY or {KEY_FILE}. Prefer the environment variable: "
        'setx NDL_API_KEY "<key>"'
    )


def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400].replace("\n", " ")
        # Never let the querystring (which carries the key) reach the message.
        raise DownloadError(f"HTTP {exc.code} from Nasdaq Data Link: {detail}") from None


def request_export(table: str, key: str) -> dict:
    """Ask for a bulk snapshot and poll until it is ``fresh``.

    Nasdaq answers ``creating``/``regenerating`` while the snapshot is being built.
    Returning the stale link early would silently hand back a previous vintage, so we
    wait for ``fresh`` and fail loudly on timeout instead.
    """
    query = urllib.parse.urlencode({"qopts.export": "true", "api_key": key})
    url = f"{BASE.format(table=table)}?{query}"
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    waited = 0

    while True:
        payload = _get_json(url)
        bulk = payload.get("datatable_bulk_download", {})
        file_info = bulk.get("file", {})
        status = str(file_info.get("status", "")).lower()

        if status == "fresh" and file_info.get("link"):
            return {
                "link": file_info["link"],
                "snapshot_time": file_info.get("data_snapshot_time"),
                "last_refreshed": bulk.get("datatable", {}).get("last_refreshed_time"),
            }
        if status not in {"creating", "regenerating"}:
            raise DownloadError(
                f"{table}: unexpected export status {status!r} (no download link)."
            )
        if time.monotonic() > deadline:
            raise DownloadError(
                f"{table}: snapshot still {status!r} after "
                f"{POLL_TIMEOUT_SECONDS // 60} minutes; try again later."
            )

        waited += POLL_SECONDS
        print(f"    {table}: snapshot {status}, waited {waited}s...", flush=True)
        time.sleep(POLL_SECONDS)


def download(link: str, destination: Path) -> None:
    req = urllib.request.Request(link, headers={"User-Agent": USER_AGENT})
    tmp = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        last_pct = -5
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            seen += len(block)
            if total:
                pct = int(100 * seen / total)
                if pct >= last_pct + 5:
                    last_pct = pct
                    print(f"    {seen / 1e6:,.0f} / {total / 1e6:,.0f} MB ({pct}%)",
                          flush=True)
    tmp.replace(destination)


def extract(zip_path: Path, table: str) -> Path:
    """Unzip the single CSV Nasdaq packs per table, naming it ``<TABLE>.csv``."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise DownloadError(
                f"{table}: expected exactly one CSV in the archive, found {members!r}"
            )
        target = DATA_DIR / f"{table}.csv"
        with archive.open(members[0]) as source, target.open("wb") as handle:
            shutil.copyfileobj(source, handle, CHUNK)
    return target


def digest_and_count(path: Path) -> tuple[str, int]:
    """SHA-256 and data-row count (header excluded) in a single streaming pass."""
    sha = hashlib.sha256()
    newlines = 0
    trailing_newline = True
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            sha.update(block)
            newlines += block.count(b"\n")
            trailing_newline = block.endswith(b"\n")
    lines = newlines + (0 if trailing_newline else 1)
    return sha.hexdigest(), max(lines - 1, 0)


def read_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("  (manifest unreadable; starting a fresh one)")
    return {}


def fetch_table(table: str, key: str, manifest: dict, force: bool) -> None:
    csv_path = DATA_DIR / f"{table}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0 and not force:
        print(f"  {table}: already present ({csv_path.stat().st_size / 1e6:,.0f} MB) "
              "- skipping (--force to re-fetch)")
        return

    print(f"  {table}: requesting bulk export...", flush=True)
    export = request_export(table, key)

    zip_path = DATA_DIR / f"{table}.zip"
    print(f"  {table}: downloading...", flush=True)
    download(export["link"], zip_path)
    print(f"  {table}: extracting...", flush=True)
    csv_path = extract(zip_path, table)
    zip_path.unlink()

    sha, rows = digest_and_count(csv_path)
    size_mb = csv_path.stat().st_size / 1e6
    manifest[table] = {
        "file": csv_path.name,
        "rows": rows,
        "bytes": csv_path.stat().st_size,
        "sha256": sha,
        "vendor_snapshot_time": export["snapshot_time"],
        "vendor_last_refreshed": export["last_refreshed"],
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"  {table}: DONE - {rows:,} rows, {size_mb:,.0f} MB, sha256 {sha[:16]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", nargs="+", default=list(ALL_TABLES),
                        choices=list(ALL_TABLES),
                        help="Subset of tables to fetch (default: all).")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch tables that are already on disk.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        key = load_key()
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = read_manifest()
    # Preserve the smallest-first ordering regardless of how --tables was given.
    ordered = [t for t in ALL_TABLES if t in set(args.tables)]
    print(f"Fetching {len(ordered)} table(s) into {DATA_DIR}\n")

    failures: list[str] = []
    for table in ordered:
        try:
            fetch_table(table, key, manifest, args.force)
        except DownloadError as exc:
            print(f"  {table}: FAILED - {exc}", file=sys.stderr)
            failures.append(table)
        except KeyboardInterrupt:
            print("\nInterrupted. Partial files are left as .partial and are skipped "
                  "on the next run; re-run to resume.", file=sys.stderr)
            return 130

    total_mb = sum(v["bytes"] for v in manifest.values()) / 1e6
    print(f"\n{len(ordered) - len(failures)}/{len(ordered)} table(s) fetched. "
          f"Corpus on disk: {total_mb:,.0f} MB across {len(manifest)} table(s).")
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"Manifest: {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
