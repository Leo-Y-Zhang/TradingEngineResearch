"""Fetch and build the two free inputs the carry sleeve needs that the panel lacks.

    .venv/Scripts/python.exe scripts/build_carry_inputs.py [--use-cache] [--out DIR]

The existing multi-asset panel has FX **spot** and US yields. A carry sleeve needs two
things it does not have:

1. **Seven more FX spot series** (AUD, NZD, CAD, CHF, SEK, NOK on top of the panel's
   EUR/GBP/JPY), because three currencies is not a cross-section.
2. **Foreign short rates**, because the interest differential — which *is* FX carry — is
   not in a spot series. Free, keyless, from FRED's OECD 3-month interbank family
   (``IR3TIB01*M156N``), one family for every country including the US so no maturity or
   basis is mixed inside a differential.

Cleaning is NOT re-invented here. ``clean_levels`` and ``simple_returns`` are imported
unchanged from ``research.multiasset.panel``, and the panel's already-published quarantine
criterion for 2008 corrupt FX closes — 8th/9th of a month in 2008, |return| > 5%, and
dropping the close leaves a two-day return under 2.5% — is applied MECHANICALLY to the new
series by ``research.multiasset.carry.scan_quarantine_candidates``. Every admission is
printed with its numbers.

Writes derived series only, to a gitignored directory. No raw vendor rows are committed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.carry import (  # noqa: E402
    FRED_SHORT_RATES,
    FX_INSTRUMENTS,
    OECD_DATAFLOW,
    OECD_SHORT_MEASURE,
    OECD_SHORT_RATES,
    scan_quarantine_candidates,
)
from research.multiasset.panel import (  # noqa: E402
    clean_levels,
    monthly_returns,
    simple_returns,
    wide_panel,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "_data" / "carry"
YF_CACHE = REPO / "_data" / "multiasset" / "raw"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
OECD_URL = ("https://sdmx.oecd.org/public/rest/data/{flow}/"
            "{areas}.M.{measure}.PA.....?format=csvfile")
#: A rebuild must agree with the existing panel to this tolerance or it is refused. Set at
#: 1e-3 because OECD-direct and the FRED mirror were measured to differ by at most 1.24e-04
#: (EZ, precision/revision noise) and by exactly 0.0 for GB/JP/US.
RATE_DRIFT_LIMIT = 1e-3


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace(".", "_").replace("-", "_")


def fetch_yf(ticker: str, cache_dir: Path, *, use_cache: bool, retries: int = 3) -> pd.DataFrame:
    """Fetch one ticker's full daily history, caching the raw frame to parquet."""
    path = cache_dir / f"{_safe_name(ticker)}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf

    last: Exception | None = None
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
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{ticker}: fetch failed after {retries} attempts ({last})")


def fetch_fred(series_id: str, cache_dir: Path, *, use_cache: bool,
               retries: int = 3) -> pd.Series:
    """One FRED series as a float Series indexed by observation date, percent units."""
    path = cache_dir / f"FRED_{series_id}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path).iloc[:, 0]

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                FRED_URL.format(sid=series_id),
                headers={"User-Agent": "Mozilla/5.0 (research; private)"},
            )
            with urllib.request.urlopen(req, timeout=40) as fh:  # noqa: S310
                payload = fh.read()
            frame = pd.read_csv(io.BytesIO(payload))
            if frame.shape[1] != 2:
                raise RuntimeError(f"unexpected shape {frame.shape}")
            frame.columns = ["date", "value"]
            frame["date"] = pd.to_datetime(frame["date"])
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            out = frame.dropna().set_index("date")["value"].sort_index()
            if out.empty:
                raise RuntimeError("empty after parse")
            cache_dir.mkdir(parents=True, exist_ok=True)
            out.to_frame(series_id).to_parquet(path)
            return out
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{series_id}: FRED fetch failed after {retries} ({last})")


def fetch_oecd_rates(cache_dir: Path, *, use_cache: bool,
                     retries: int = 3) -> dict[str, pd.Series]:
    """All ten 3-month interbank series from OECD, as month-end DECIMALS.

    One request for every reference area. OECD publishes this family and FRED mirrors it,
    so this is a transport change rather than a source change -- proven, not assumed: see
    the note on `carry.OECD_SHORT_RATES`. Two conversions are explicit because both have
    bitten this repo: OECD publishes PERCENT while the panel holds DECIMALS, and monthly
    observations carry a period label ("1994-01") while the panel is stamped MONTH END.
    """
    path = cache_dir / f"OECD_{OECD_SHORT_MEASURE}_rates.parquet"
    if use_cache and path.exists():
        frame = pd.read_parquet(path)
    else:
        url = OECD_URL.format(flow=OECD_DATAFLOW, measure=OECD_SHORT_MEASURE,
                              areas="+".join(OECD_SHORT_RATES.values()))
        last: Exception | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                raw = urllib.request.urlopen(req, timeout=90).read().decode("utf-8", "replace")
                break
            except Exception as exc:  # noqa: BLE001 - retried, then raised to the caller
                last = exc
                if attempt < retries - 1:
                    time.sleep(5.0 * (attempt + 1))
        else:
            raise RuntimeError(f"OECD {OECD_SHORT_MEASURE}: failed after {retries} ({last})")

        rows = list(csv.DictReader(io.StringIO(raw)))
        cols: dict[str, pd.Series] = {}
        for ccy, area in OECD_SHORT_RATES.items():
            pairs = {r["TIME_PERIOD"]: float(r["OBS_VALUE"]) for r in rows
                     if r["REF_AREA"] == area and r["OBS_VALUE"] not in ("", "NaN")}
            if not pairs:
                raise RuntimeError(f"OECD returned no rows for {ccy} ({area})")
            cols[ccy] = pd.Series(pairs).sort_index()
        frame = pd.DataFrame(cols).sort_index()
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    out: dict[str, pd.Series] = {}
    for ccy in OECD_SHORT_RATES:
        s = frame[ccy].dropna()
        s.index = pd.PeriodIndex(s.index, freq="M").to_timestamp(how="end").normalize()
        out[ccy] = s[~s.index.duplicated(keep="last")].sort_index() / 100.0
    return out


def compare_to_existing(new: pd.DataFrame, old: pd.DataFrame) -> dict[str, dict]:
    """Per-column agreement between a rebuilt rate panel and the one already on disk."""
    report: dict[str, dict] = {}
    for col in new.columns:
        if col not in old.columns:
            report[col] = {"n_overlap": 0, "max_abs_diff": 0.0, "note": "new column"}
            continue
        both = pd.concat([new[col].rename("n"), old[col].rename("o")], axis=1).dropna()
        d = (both["n"] - both["o"]).abs()
        report[col] = {"n_overlap": int(len(both)),
                       "max_abs_diff": float(d.max()) if len(d) else 0.0,
                       "mean_abs_diff": float(d.mean()) if len(d) else 0.0}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    audit: dict[str, object] = {}

    # ── 1. FX spot ────────────────────────────────────────────────────────────
    print(f"Fetching {len(FX_INSTRUMENTS)} FX spot series (use_cache={args.use_cache})...")
    levels: dict[str, pd.Series] = {}
    clean_stats: dict[str, dict] = {}
    for inst in FX_INSTRUMENTS:
        frame = fetch_yf(inst.ticker, YF_CACHE, use_cache=args.use_cache)
        series, stats = clean_levels(frame["Close"])
        levels[inst.key] = series
        clean_stats[inst.key] = stats
        print(f"  {inst.key:8s} {inst.ticker:10s} {series.index.min().date()} -> "
              f"{series.index.max().date()}  n={len(series)}")
    audit["fx_clean_stats"] = clean_stats

    # ── 2. Quarantine: the panel's published criterion, applied mechanically ──
    print("\nQuarantine scan (8th/9th of a month in 2008, |r|>5%, round trip <2.5%):")
    provisional = {
        inst.key: simple_returns(levels[inst.key], invert=inst.invert)[0]
        for inst in FX_INSTRUMENTS
    }
    candidates = scan_quarantine_candidates(levels, provisional,
                                            {i.key: i.invert for i in FX_INSTRUMENTS})
    for row in candidates:
        verdict = "QUARANTINE" if row["admitted"] else "kept"
        print(f"  {row['key']:8s} {row['date']}  r={row['ret']:+.4f}  "
              f"next={row['next_ret']:+.4f}  round_trip={row['round_trip']:+.4f}  -> {verdict}")
    if not candidates:
        print("  (no candidate bars in any series)")
    audit["quarantine_candidates"] = candidates

    admitted = {(row["key"], row["date"]) for row in candidates if row["admitted"]}
    for key, date_str in sorted(admitted):
        stamp = pd.Timestamp(date_str)
        if stamp in levels[key].index:
            levels[key] = levels[key].drop(index=stamp)
    audit["n_quarantined"] = len(admitted)

    # ── 3. FX returns, daily → month-end ─────────────────────────────────────
    ret_stats: dict[str, dict] = {}
    daily: dict[str, pd.Series] = {}
    daily_unscreened: dict[str, pd.Series] = {}
    for inst in FX_INSTRUMENTS:
        series, stats = simple_returns(levels[inst.key], invert=inst.invert)
        daily[inst.key] = series
        ret_stats[inst.key] = stats
        daily_unscreened[inst.key] = provisional[inst.key]
    audit["fx_return_stats"] = ret_stats

    fx_daily = wide_panel(daily)
    fx_monthly = monthly_returns(fx_daily)
    fx_monthly_unscreened = monthly_returns(wide_panel(daily_unscreened))
    fx_daily.to_parquet(out / "fx_spot_returns_daily.parquet")
    fx_monthly.to_parquet(out / "fx_spot_returns_monthly.parquet")
    fx_monthly_unscreened.to_parquet(out / "fx_spot_returns_monthly_unscreened.parquet")
    print(f"\nFX spot monthly panel: {fx_monthly.shape[0]} rows x {fx_monthly.shape[1]} cols, "
          f"{fx_monthly.index.min().date()} -> {fx_monthly.index.max().date()}")

    # ── 4. Short rates ───────────────────────────────────────────────────────
    # OECD publishes this family; FRED only mirrors it, and FRED is IP-blocked from this
    # machine (see `carry.OECD_SHORT_RATES`). Try the publisher first, fall back to the
    # mirror, and record which route was actually used.
    rates: dict[str, pd.Series] = {}
    spans: dict[str, dict] = {}
    route = "oecd"
    try:
        print(f"\nFetching {len(OECD_SHORT_RATES)} short-rate series from OECD...")
        rates = fetch_oecd_rates(raw_dir, use_cache=args.use_cache)
    except Exception as exc:  # noqa: BLE001 - fall back to the mirror, loudly
        route = "fred_fallback"
        print(f"  OECD fetch failed ({type(exc).__name__}: {exc}) - falling back to FRED")
        for ccy, sid in FRED_SHORT_RATES.items():
            series = fetch_fred(sid, raw_dir, use_cache=args.use_cache)
            # Monthly series are stamped at the first of the month; the value is the
            # month's average, known by the month's end. Stamp at CALENDAR MONTH END so it
            # aligns with the panel and the one-month holding lag is unambiguous.
            series.index = (pd.DatetimeIndex(series.index).to_period("M")
                            .to_timestamp(how="end").normalize())
            rates[ccy] = series[~series.index.duplicated(keep="last")].sort_index() / 100.0

    ids = OECD_SHORT_RATES if route == "oecd" else FRED_SHORT_RATES
    for ccy, series in rates.items():
        spans[ccy] = {"source": route, "series_id": ids[ccy],
                      "first": str(series.index.min().date()),
                      "last": str(series.index.max().date()), "n": int(len(series))}
        print(f"  {ccy:4s} {ids[ccy]:18s} {series.index.min().date()} -> "
              f"{series.index.max().date()}  n={len(series)}")
    audit["short_rate_spans"] = spans
    audit["short_rate_route"] = route

    rate_panel = wide_panel(rates)

    # GATE: never silently replace a panel the rest of the repo already depends on. If a
    # previous build exists, the new one must agree with it on their overlap.
    existing_path = out / "short_rates_monthly.parquet"
    if existing_path.exists():
        drift = compare_to_existing(rate_panel, pd.read_parquet(existing_path))
        audit["short_rate_drift_vs_existing"] = drift
        worst = max((d["max_abs_diff"] for d in drift.values()), default=0.0)
        print(f"\n  drift vs existing parquet: worst |diff| = {worst:.3e} "
              f"({'OK' if worst <= RATE_DRIFT_LIMIT else 'ABOVE LIMIT'})")
        for ccy, d in sorted(drift.items()):
            if d["max_abs_diff"] > 0.0:
                print(f"    {ccy}: n={d['n_overlap']} max|diff|={d['max_abs_diff']:.3e}")
        if worst > RATE_DRIFT_LIMIT:
            raise SystemExit(
                f"REFUSING TO OVERWRITE: short rates drifted by {worst:.3e} against the "
                f"existing panel (limit {RATE_DRIFT_LIMIT}). Investigate before rebuilding.")

    rate_panel.to_parquet(existing_path)
    print(f"\nShort-rate panel: {rate_panel.shape[0]} rows x {rate_panel.shape[1]} cols")

    (out / "carry_inputs_audit.json").write_text(json.dumps(audit, indent=2, default=str),
                                                 encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
