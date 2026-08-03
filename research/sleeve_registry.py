"""THE REGISTRY of every dated return series this programme has written to disk.

WHY
===
The one-month dating defect in `research.sleeves.lowvol_retest` (see
`research.alignment`) was found only when the series was joined to another one, months
after it was banked and after an independent bit-for-bit verification had signed it off.
Nothing in the repo recorded what a sleeve's index MEANT, so nothing could check it.

This module is that record. Every entry declares the convention its index uses:

    REALISATION  the label is the month the return was EARNED — joinable
    FORMATION    the label is the month the SIGNAL was formed and the slot holds the
                 FOLLOWING month's return — one month early, must be shifted +1 before
                 it is joined to anything

`tests/test_dating_alignment.py` measures every entry against a correctly-dated reference
and fails if the measurement disagrees with the declaration. That makes the defect
impossible to reintroduce silently: a sleeve that starts dating its output differently
breaks the test, and so does a consumer that drops a compensating shift.

`market_exposed` says whether the equity reference should have power over the series. A
market-neutral long/short book correlates with nothing at any lag, so the probe cannot
prove it aligned — that is recorded honestly rather than counted as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.alignment import (
    FORMATION,
    REALISATION,
    AlignmentProbe,
    probe_alignment,
    shift_months,
)

__all__ = [
    "REFERENCE_KEY",
    "REGISTRY",
    "REPO",
    "SPX_PATH",
    "DatedSeries",
    "audit",
    "by_key",
    "load_reference",
    "load_spx",
]

REPO = Path(__file__).resolve().parents[1]
#: the external anchor. Gitignored (vendor licence), so treat its absence as "skip".
SPX_PATH = REPO / "_data" / "multiasset" / "returns_monthly.parquet"


@dataclass(frozen=True)
class DatedSeries:
    """One column of one on-disk artefact, and what its index means."""

    key: str
    path: str                 # repo-relative
    column: str
    convention: str           # REALISATION | FORMATION
    market_exposed: bool      # should an equity reference have power over it?
    note: str = ""

    @property
    def full_path(self) -> Path:
        return REPO / self.path

    def load(self) -> pd.Series:
        """The series exactly AS STORED, dates untouched."""
        path = self.full_path
        if path.suffix == ".csv":
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            frame = pd.read_parquet(path)
        return frame[self.column].astype(float).dropna().rename(self.key)

    def shift_to_realisation(self) -> int:
        """Months to add to the stored index to make it mean 'the month earned'."""
        return 1 if self.convention == FORMATION else 0

    def load_realisation_dated(self) -> pd.Series:
        """The series with its declared correction applied — safe to join by date."""
        return shift_months(self.load(), self.shift_to_realisation())


# The multi-asset passive benchmark: equal-weight, monthly-rebalanced, 1965-2026, and
# strongly market-exposed (rho 0.827 against SPX at lag 0, argmax lag 0). It is TRACKED,
# unlike the vendor panel, so the audit runs on a bare clone. `load_spx` re-anchors it
# against an external index wherever `_data` is present.
REFERENCE_KEY = "trend.bench_net_10bps"

REGISTRY: tuple[DatedSeries, ...] = (
    # ── low-vol / quality — the sleeve the defect was found in ────────────────
    DatedSeries("lowvol_registered.benchmark",
                "research/sleeves/_portfolio/lowvol_b2_net_monthly.parquet",
                "benchmark", FORMATION, True,
                "run_band labelled the slot by the FORMATION month; the extractor put "
                "that straight on disk"),
    DatedSeries("lowvol_registered.net_conservative",
                "research/sleeves/_portfolio/lowvol_b2_net_monthly.parquet",
                "net_conservative", FORMATION, True),
    DatedSeries("lowvol_corrected.benchmark",
                "research/sleeves/_portfolio/lowvol_b2_corrected_monthly.parquet",
                "benchmark", FORMATION, True),
    DatedSeries("lowvol_corrected.net_conservative",
                "research/sleeves/_portfolio/lowvol_b2_corrected_monthly.parquet",
                "net_conservative", FORMATION, True),
    # ── multi-asset trend ─────────────────────────────────────────────────────
    DatedSeries("trend.net_10bps",
                "research/sleeves/_multiasset_trend/primary_20pct_monthly.csv",
                "net_10bps", REALISATION, False, "long/short: no market exposure"),
    DatedSeries("trend.gross",
                "research/sleeves/_multiasset_trend/primary_20pct_monthly.csv",
                "gross", REALISATION, False, "long/short: no market exposure"),
    DatedSeries(REFERENCE_KEY,
                "research/sleeves/_multiasset_trend/primary_20pct_monthly.csv",
                "bench_net_10bps", REALISATION, True, "the audit reference"),
    # ── carry ─────────────────────────────────────────────────────────────────
    DatedSeries("carry.net",
                "research/sleeves/_carry_output/carry_primary_net_monthly.parquet",
                "net", REALISATION, False),
    DatedSeries("carry.gross",
                "research/sleeves/_carry_output/carry_primary_gross_monthly.parquet",
                "gross", REALISATION, False),
    DatedSeries("carry.trend_reference",
                "research/sleeves/_carry_output/trend_reference_net_monthly.parquet",
                "net", REALISATION, False),
    DatedSeries("carry.two_sleeve_risk_parity",
                "research/sleeves/_carry_output/two_sleeve_risk_parity_monthly.parquet",
                "net", REALISATION, False),
    # ── seasonal ──────────────────────────────────────────────────────────────
    DatedSeries("seasonal.composite",
                "research/sleeves/_seasonal/seasonal_composite_20pct_monthly.parquet",
                "seasonal_net_10bps", REALISATION, True),
    DatedSeries("seasonal.bench_net_10bps",
                "research/sleeves/_seasonal/seasonal_composite_20pct_monthly.parquet",
                "bench_net_10bps", REALISATION, True),
    DatedSeries("seasonal.e1_tom",
                "research/sleeves/_seasonal/e1_tom_20pct_monthly.parquet",
                "net_10bps", REALISATION, True),
    DatedSeries("seasonal.e2_halloween",
                "research/sleeves/_seasonal/e2_halloween_20pct_monthly.parquet",
                "net_10bps", REALISATION, True),
    DatedSeries("seasonal.e3_january",
                "research/sleeves/_seasonal/e3_january_20pct_monthly.parquet",
                "net_10bps", REALISATION, False,
                "market-exposed but only marginally (|rho| ~0.21-0.25); power not asserted"),
    # ── defensive ─────────────────────────────────────────────────────────────
    DatedSeries("defensive.net",
                "research/sleeves/_defensive/defensive_primary_net_monthly.parquet",
                "net", REALISATION, False),
    DatedSeries("defensive.within_block",
                "research/sleeves/_defensive/defensive_within_block_net_monthly.parquet",
                "net", REALISATION, False),
    DatedSeries("defensive.gross",
                "research/sleeves/_defensive/primary_20pct_monthly.csv",
                "gross", REALISATION, False),
    DatedSeries("defensive.bench_net_10bps",
                "research/sleeves/_defensive/primary_20pct_monthly.csv",
                "bench_net_10bps", REALISATION, True),
    # ── value ─────────────────────────────────────────────────────────────────
    DatedSeries("value.net_10bps",
                "research/sleeves/_value/primary_20pct_monthly.csv",
                "net_10bps", REALISATION, False),
    DatedSeries("value.bench_net_10bps",
                "research/sleeves/_value/primary_20pct_monthly.csv",
                "bench_net_10bps", REALISATION, True),
    # ── reversal re-test ──────────────────────────────────────────────────────
    DatedSeries("reversal.decile_long_only",
                "research/sleeves/_reversal_retest/net_returns_top_decile_monthly.parquet",
                "conservative__long_only", REALISATION, True),
    DatedSeries("reversal.decile_long_short",
                "research/sleeves/_reversal_retest/net_returns_top_decile_monthly.parquet",
                "conservative__long_short", REALISATION, True),
    DatedSeries("reversal.quintile_long_only",
                "research/sleeves/_reversal_retest/net_returns_top_quintile_monthly.parquet",
                "conservative__long_only", REALISATION, True),
    DatedSeries("reversal.quintile_long_short",
                "research/sleeves/_reversal_retest/net_returns_top_quintile_monthly.parquet",
                "conservative__long_short", REALISATION, True),
)


def by_key(key: str) -> DatedSeries:
    for entry in REGISTRY:
        if entry.key == key:
            return entry
    raise KeyError(key)


def load_reference() -> pd.Series:
    """The tracked, correctly-dated reference the audit measures everything against."""
    return by_key(REFERENCE_KEY).load()


def load_spx() -> pd.Series | None:
    """Correctly-dated SPX, or None when the vendor panel is not on this machine."""
    if not SPX_PATH.exists():
        return None
    return pd.read_parquet(SPX_PATH)["SPX"].astype(float).dropna()


def audit(reference: pd.Series | None = None,
          *, as_stored: bool = False) -> dict[str, AlignmentProbe]:
    """Probe every registered series against ``reference``.

    ``as_stored=False`` (the default) applies each entry's declared correction first, so
    a clean audit means every series in the programme can be joined by date. ``True``
    probes the raw on-disk dates, which is what pins each declared convention.
    """
    ref = load_reference() if reference is None else reference
    out: dict[str, AlignmentProbe] = {}
    for entry in REGISTRY:
        if not entry.full_path.exists():
            continue
        series = entry.load() if as_stored else entry.load_realisation_dated()
        out[entry.key] = probe_alignment(series, ref, name=entry.key,
                                         reference_name="reference")
    return out
