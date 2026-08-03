"""
TradingEngineResearch — Fama-French factor ingestion (Stage B: richer free data)
=======================================================================
Point-in-time-safe ingestion of the Fama-French research factors (free, from Ken
French's data library) and the features derived from them. The audit established
that price-only signals carry no robust alpha; richer data is the documented path
to a *credible* cross-sectional edge. This module supplies two kinds of feature:

  • factor TIMING features (market-wide): trailing cumulative factor returns + factor
    vol — regime/timing inputs, one value per date (broadcast to all symbols).
  • factor LOADINGS (cross-sectional): each stock's trailing betas to MKT/SMB/HML —
    a genuinely per-stock value/size/market-exposure signal to rank names on.

All accessors are point-in-time: only data with ``index <= asof`` is ever used, so
nothing can leak future information into an earlier decision (golden rule 3).

Live download (`fetch_fama_french_factors`) hits the network and is excluded from
coverage; the test suite runs against the committed offline fixture
``tests/fixtures/fama_french_daily_sample.csv`` and never touches the network.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "load_fama_french",
    "fetch_fama_french_factors",
    "factor_momentum_features",
    "factor_loadings",
    "FF_FACTORS",
]

FF_FACTORS = ("mkt_rf", "smb", "hml")
_FF_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def load_fama_french(path: str | Path) -> pd.DataFrame:
    """Load a Fama-French daily factor CSV (``date,mkt_rf,smb,hml,rf``, decimals) into
    a DataFrame indexed by a sorted ``DatetimeIndex``. The canonical offline fixture is
    ``tests/fixtures/fama_french_daily_sample.csv``."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    expected = ["mkt_rf", "smb", "hml", "rf"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Fama-French CSV missing columns {missing}; got {list(df.columns)}")
    return df[expected].astype(float)


def fetch_fama_french_factors() -> pd.DataFrame:  # pragma: no cover - network
    """Download + parse the Fama-French 3-factor *daily* file (percent → decimal).
    Network-only; the suite uses the committed fixture instead."""
    import urllib.request

    req = urllib.request.Request(_FF_DAILY_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    name = zipfile.ZipFile(io.BytesIO(raw)).namelist()[0]
    text = zipfile.ZipFile(io.BytesIO(raw)).read(name).decode("utf-8", "replace").splitlines()
    hdr = next(i for i, line in enumerate(text) if "Mkt-RF" in line)
    rows = []
    for line in text[hdr + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
            try:
                vals = [float(x) / 100.0 for x in parts[1:5]]
            except ValueError:
                continue
            d = parts[0]
            rows.append((pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}"), *vals))
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"]).set_index("date")
    return df.sort_index()


def factor_momentum_features(
    ff: pd.DataFrame, asof, windows: tuple[int, ...] = (21, 63, 252)
) -> dict[str, float]:
    """PIT-safe market-wide factor-timing features as of ``asof``: trailing cumulative
    return of each factor over each window, plus trailing market-factor volatility.
    Uses only ``ff.index <= asof``."""
    asof_ts = pd.Timestamp(asof)
    hist = ff[ff.index <= asof_ts]
    feats: dict[str, float] = {}
    for w in windows:
        win = hist.tail(w)
        if len(win) < max(5, w // 2):
            continue
        for col in FF_FACTORS:
            feats[f"ff_{col}_cumret_{w}d"] = float((1.0 + win[col]).prod() - 1.0)
        feats[f"ff_mkt_rf_vol_{w}d"] = float(win["mkt_rf"].std(ddof=1)) if len(win) > 1 else 0.0
    return feats


def factor_loadings(
    stock_returns: pd.DataFrame,
    ff: pd.DataFrame,
    asof,
    window: int = 126,
    min_obs: int = 60,
) -> dict[str, dict[str, float]]:
    """PIT-safe per-stock Fama-French betas as of ``asof``.

    Regress each stock's *excess* return (``ret - rf``) on a constant + MKT/SMB/HML
    over the trailing ``window`` trading days ending at ``asof`` (only data
    ``<= asof``). Returns ``{symbol: {"beta_mkt", "beta_smb", "beta_hml"}}``. A stock
    with fewer than ``min_obs`` aligned finite observations is skipped (fail-closed:
    no fabricated exposure)."""
    asof_ts = pd.Timestamp(asof)
    ff_w = ff[ff.index <= asof_ts].tail(window)
    sr_w = stock_returns[stock_returns.index <= asof_ts].tail(window)
    common = ff_w.index.intersection(sr_w.index)
    if len(common) < min_obs:
        return {}
    ff_w = ff_w.loc[common]
    sr_w = sr_w.loc[common]
    design = np.column_stack([
        np.ones(len(common)),
        ff_w["mkt_rf"].to_numpy(float),
        ff_w["smb"].to_numpy(float),
        ff_w["hml"].to_numpy(float),
    ])
    rf = ff_w["rf"].to_numpy(float)
    out: dict[str, dict[str, float]] = {}
    for sym in sr_w.columns:
        y = sr_w[sym].to_numpy(float) - rf
        mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
        if int(mask.sum()) < min_obs:
            continue
        beta, *_ = np.linalg.lstsq(design[mask], y[mask], rcond=None)
        out[str(sym)] = {
            "beta_mkt": float(beta[1]),
            "beta_smb": float(beta[2]),
            "beta_hml": float(beta[3]),
        }
    return out


def _default_fixture() -> Optional[Path]:  # pragma: no cover - convenience
    p = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fama_french_daily_sample.csv"
    return p if p.exists() else None
