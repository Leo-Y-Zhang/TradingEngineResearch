"""Fetch the total-return and carry sources the convention repair needs.

    .venv/Scripts/python.exe scripts/build_convention_inputs.py [--use-cache] [--out DIR]

Network step of the repair pre-registered in
``research/multiasset/convention_repair_prereg.md`` §2. Fetches only what that
document names, reusing ``build_multiasset_panel.fetch_one`` so the raw cache, the
cleaning and the month-end convention are identical to the panel it must line up with.

Builds NO strategy and corrects nothing — it only assembles inputs. Nothing here is
committed except derived statistics; ``_data/`` is gitignored and Yahoo's terms forbid
redistributing its rows.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.multiasset.panel import (  # noqa: E402
    clean_levels,
    monthly_returns,
    simple_returns,
    wide_panel,
)
from scripts.build_multiasset_panel import fetch_one  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "_data" / "multiasset" / "convention"
PANEL_RAW = Path(__file__).resolve().parents[1] / "_data" / "multiasset" / "raw"
CARRY_RATES = (Path(__file__).resolve().parents[1] / "_data" / "carry"
               / "short_rates_monthly.parquet")

FF_MONTHLY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                  "F-F_Research_Data_Factors_CSV.zip")
UA = {"User-Agent": "Mozilla/5.0"}

#: Total-return references. key -> (ticker, what it is measured against, note).
#: ``pair_with`` names the PANEL key whose price return this series is the
#: total-return counterpart of; ``fx_with`` names the panel FX key needed to put the
#: two in the same currency (the ETFs quote in USD, the indices in local currency).
TR_REFERENCES: dict[str, dict[str, str | None]] = {
    "SP500TR": {"ticker": "^SP500TR", "pair_with": "SPX", "fx_with": None,
                "note": "S&P 500 total-return index; second read on the US correction."},
    "QQQ": {"ticker": "QQQ", "pair_with": "NASDAQ", "fx_with": None,
            "note": "Nasdaq-100 TR ETF. Pairs with ^NDX, NOT with the Composite -- the "
                    "composition difference is carried as bias, see prereg 4E."},
    "NDX": {"ticker": "^NDX", "pair_with": None, "fx_with": None,
            "note": "Nasdaq-100 PRICE index, the clean partner for QQQ."},
    "EWU": {"ticker": "EWU", "pair_with": "FTSE100", "fx_with": "GBPUSD",
            "note": "MSCI UK TR ETF in USD."},
    "EWJ": {"ticker": "EWJ", "pair_with": "N225", "fx_with": "JPYUSD",
            "note": "MSCI Japan TR ETF in USD."},
    "EWG": {"ticker": "EWG", "pair_with": "DAX", "fx_with": "EURUSD",
            "note": "MSCI Germany TR ETF in USD. Control D: this gap must come out "
                    "NEAR ZERO because the DAX already contains its dividends."},
    "EWH": {"ticker": "EWH", "pair_with": "HSI", "fx_with": None,
            "note": "MSCI Hong Kong TR ETF in USD. HKD is pegged to USD, so no FX leg "
                    "is applied; the peg band is a disclosed residual."},
    "EWA": {"ticker": "EWA", "pair_with": "ASX200", "fx_with": "AUDUSD",
            "note": "MSCI Australia TR ETF in USD. AUDUSD is fetched for this purpose "
                    "only and never enters the tradable panel."},
}

#: Currency-deposit ETFs: spot PLUS accrued foreign interest, so a total return.
#: These are Control C's independent check on the FX correction.
FX_REFERENCES: dict[str, dict[str, str]] = {
    "FXY": {"ticker": "FXY", "pair_with": "JPYUSD",
            "note": "JPY deposits. The discriminating leg -- JPY rates sat far below "
                    "USD for the whole window, so the omission is one-signed."},
    "FXB": {"ticker": "FXB", "pair_with": "GBPUSD", "note": "GBP deposits."},
    "FXE": {"ticker": "FXE", "pair_with": "EURUSD", "note": "EUR deposits."},
}

#: Commodity references. NOT applied -- they bracket the uncorrected roll error.
COMMODITY_REFERENCES: dict[str, dict[str, str]] = {
    "GSG": {"ticker": "GSG", "pair_with": "", "note": "GSCI total-return commodity ETF."},
    "USO": {"ticker": "USO", "pair_with": "WTI_F", "note": "Rolled WTI reference."},
}

#: Fetched only to put a USD-quoted ETF back into its index's currency.
FX_SUPPORT: dict[str, str] = {"AUDUSD": "AUDUSD=X"}


def fetch_french_monthly() -> pd.DataFrame:
    """Monthly Fama-French factors -> DataFrame with mkt_rf and rf as DECIMALS.

    ``mkt_rf + rf`` is the CRSP value-weighted US equity TOTAL return, which is the
    only source in this repair that covers the whole 1965-2026 sample.
    """
    resp = requests.get(FF_MONTHLY_URL, headers=UA, timeout=120)
    resp.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    text = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()

    rows: list[tuple[str, float, float]] = []
    started = False
    for line in text:
        stripped = line.strip()
        head = stripped[:6]
        if len(head) == 6 and head.isdigit():
            started = True
            parts = [p for p in stripped.replace(",", " ").split() if p]
            if len(parts) < 5:
                break
            try:
                rows.append((parts[0], float(parts[1]) / 100.0, float(parts[4]) / 100.0))
            except ValueError:
                break
        elif started:
            # The monthly block is followed by an annual block under its own header;
            # the first non-YYYYMM line after the data starts ends the monthly block.
            break

    if not rows:
        raise RuntimeError("Fama-French monthly file parsed to zero rows")
    frame = pd.DataFrame(rows, columns=["ym", "mkt_rf", "rf"])
    frame["date"] = pd.to_datetime(frame["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    out = frame.set_index("date")[["mkt_rf", "rf"]].astype(float)
    out.index.name = "date"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true", help="reuse cached raw parquet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    every: dict[str, dict[str, str | None]] = {
        **TR_REFERENCES,
        **{k: dict(v) for k, v in FX_REFERENCES.items()},
        **{k: dict(v) for k, v in COMMODITY_REFERENCES.items()},
        **{k: {"ticker": t, "pair_with": None, "note": "FX support leg, never tradable."}
           for k, t in FX_SUPPORT.items()},
    }

    levels: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    audit: dict[str, dict] = {}

    print(f"Fetching {len(every)} reference series (use_cache={args.use_cache})...")
    for key, spec in every.items():
        ticker = str(spec["ticker"])
        try:
            raw = fetch_one(ticker, PANEL_RAW, use_cache=args.use_cache)
        except Exception as exc:  # noqa: BLE001 - one ticker must not kill the build
            failures[key] = str(exc)
            print(f"  FAIL {key:<10} {ticker:<10} {exc}")
            continue
        clean, cstats = clean_levels(raw["Close"])
        # AUDUSD=X quotes USD per AUD, which is already the direction a long-AUD
        # position gains in, so no inversion. Same for every ETF.
        ret, rstats = simple_returns(clean)
        levels[key] = clean
        returns[key] = ret
        audit[key] = {"ticker": ticker, "pair_with": spec.get("pair_with"),
                      "note": spec.get("note"),
                      "first": str(clean.index.min().date()),
                      "last": str(clean.index.max().date()),
                      "n_daily": int(len(clean)), **cstats, **rstats}
        print(f"  ok   {key:<10} {ticker:<10} {clean.index.min().date()} -> "
              f"{clean.index.max().date()}  n={len(clean):,}")

    if not returns:
        print("nothing fetched", file=sys.stderr)
        return 1

    daily = wide_panel(returns)
    monthly = monthly_returns(daily)

    print("\nFetching the Fama-French monthly factors...")
    french = fetch_french_monthly()
    print(f"  ok   french     {french.index.min().date()} -> {french.index.max().date()}"
          f"  n={len(french):,}")

    if not CARRY_RATES.exists():
        raise FileNotFoundError(
            f"{CARRY_RATES} is missing - run scripts/build_carry_inputs.py first. "
            "The repair reuses the OECD short rates this repo already built rather "
            "than re-fetching them.")
    rates = pd.read_parquet(CARRY_RATES)

    monthly.to_parquet(out / "reference_returns_monthly.parquet")
    daily.to_parquet(out / "reference_returns_daily.parquet")
    french.to_parquet(out / "french_monthly.parquet")

    summary = {
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "n_references": len(returns),
        "failures": failures,
        "monthly_shape": list(monthly.shape),
        "monthly_first": str(monthly.index.min().date()),
        "monthly_last": str(monthly.index.max().date()),
        "french": {"first": str(french.index.min().date()),
                   "last": str(french.index.max().date()), "n": int(len(french)),
                   "us_total_ann_pct_1965on": round(float(
                       (french.loc["1965":, "mkt_rf"] + french.loc["1965":, "rf"]
                        ).mean() * 12 * 100), 3),
                   "rf_ann_pct_1965on": round(float(
                       french.loc["1965":, "rf"].mean() * 12 * 100), 3)},
        "short_rates": {"source": str(CARRY_RATES), "shape": list(rates.shape),
                        "first": str(rates.index.min().date()),
                        "last": str(rates.index.max().date()),
                        "months_per_ccy": {c: int(rates[c].notna().sum())
                                           for c in rates.columns}},
        "references": audit,
    }
    (out / "convention_inputs_audit.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nmonthly reference panel {monthly.shape} "
          f"({summary['monthly_first']} -> {summary['monthly_last']})")
    print(f"short rates {list(rates.shape)} reused from {CARRY_RATES}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
