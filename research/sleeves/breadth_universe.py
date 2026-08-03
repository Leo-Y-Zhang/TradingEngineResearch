"""The combined ORIGINAL + BREADTH excess-return universe.

One job: hand every downstream study a month-end excess-return panel whose conventions
are IDENTICAL to the one iteration 11 used, so that any difference in the numbers is a
difference in breadth and not a difference in bookkeeping.

It therefore replicates ``research.sleeves.multiasset_trend.load_excess_panel`` exactly
— same cash subtraction for funded positions, same interior-null convention, same
month-end index — rather than inventing a second loader, and it reads both panels
read-only. Nothing in ``research/multiasset/`` or the original ``_data/multiasset/*``
files is modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.multiasset.breadth_instruments import (
    BREADTH_BLOCKS,
    CASH_SUBTRACTED_NEW,
    ROLL_MANAGED_SUBSTITUTE,
    TIERS,
)
from research.sleeves.multiasset_trend import (
    BLOCKS,
    CASH_SUBTRACTED,
    PRIMARY_UNIVERSE,
)

__all__ = [
    "ORIGINAL_18",
    "BREADTH_ADDITIONS",
    "EXPANDED",
    "ALL_BLOCKS",
    "UNIVERSES",
    "load_combined_panel",
]

_DATA = Path("_data/multiasset")
_BREADTH = _DATA / "breadth"

ORIGINAL_18: tuple[str, ...] = PRIMARY_UNIVERSE
BREADTH_ADDITIONS: tuple[str, ...] = tuple(
    k for keys in BREADTH_BLOCKS.values() for k in keys
)
EXPANDED: tuple[str, ...] = ORIGINAL_18 + BREADTH_ADDITIONS
ALL_BLOCKS: dict[str, tuple[str, ...]] = {**BLOCKS, **BREADTH_BLOCKS}

# Every funded position needs the bill subtracted to become futures-equivalent: the
# three par-bond series the original panel already handles, plus every ETF added here.
_CASH_SUBTRACTED_ALL: frozenset[str] = CASH_SUBTRACTED | CASH_SUBTRACTED_NEW

# The universes each study reports on. Each one exists to answer a specific question,
# and every one of them is declared here rather than assembled at the call site.
UNIVERSES: dict[str, tuple[str, ...]] = {
    # the control: must reproduce iteration 11 exactly
    "original_18": ORIGINAL_18,
    # the headline: everything free data can supply
    "expanded_37": EXPANDED,
    # per-block marginal contribution: original + one block
    **{f"orig_plus_{b}": ORIGINAL_18 + keys for b, keys in BREADTH_BLOCKS.items()},
    # the blocks alone
    **{f"block_{b}": keys for b, keys in BREADTH_BLOCKS.items()},
    # honesty variants
    "expanded_no_livestock": tuple(
        k for k in EXPANDED if k not in BREADTH_BLOCKS["livestock"]),
    "expanded_no_rollcontam": tuple(
        k for k in EXPANDED
        if k not in ("CATTLE_F", "HOGS_F", "CORN_F", "SOYBEAN_F", "SUGAR_F")),
    "expanded_no_vol": tuple(k for k in EXPANDED if k != "VIX_ETF"),
    "expanded_long_history_only": ORIGINAL_18 + TIERS["T1_long"] + TIERS["T2_mid"],
}


def load_combined_panel(
    universe: tuple[str, ...] = EXPANDED,
    *,
    substitute_roll_managed: bool = False,
    data_dir: Path = _DATA,
    breadth_dir: Path = _BREADTH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(excess_returns, interior_nulls)`` for ``universe``, month-end.

    ``substitute_roll_managed`` swaps each front-month continuous agricultural or
    livestock series for its roll-managed, actually-holdable ETF/ETN, where one exists.
    That is a SENSITIVITY, not the primary panel: the substitutes start 2009-2018 and
    carry ~1%/yr of expense, so they trade contamination for sample length.
    """
    orig = pd.read_parquet(data_dir / "returns_monthly.parquet")
    extra = pd.read_parquet(breadth_dir / "returns_monthly.parquet")
    if substitute_roll_managed:
        val = pd.read_parquet(breadth_dir / "returns_daily_all.parquet")
        sub_m = _to_monthly(val[[c for c in set(ROLL_MANAGED_SUBSTITUTE.values())
                                 if c in val.columns]])
        for fut, etf in ROLL_MANAGED_SUBSTITUTE.items():
            if fut in extra.columns and etf in sub_m.columns:
                extra[fut] = sub_m[etf].reindex(extra.index)

    dup = [c for c in extra.columns if c in orig.columns]
    rets = pd.concat([orig, extra.drop(columns=dup)], axis=1).sort_index()
    cash = pd.read_parquet(data_dir / "cash_monthly.parquet")["US_CASH_13W"]

    missing = [k for k in universe if k not in rets.columns]
    if missing:
        raise KeyError(f"panel is missing instruments: {missing}")

    x = rets.loc[:, list(universe)].copy()
    for key in universe:
        if key in _CASH_SUBTRACTED_ALL:
            x[key] = x[key] - cash.reindex(x.index)

    interior = pd.DataFrame(False, index=x.index, columns=x.columns)
    for key in universe:
        first = x[key].first_valid_index()
        if first is not None:
            interior.loc[first:, key] = x.loc[first:, key].isna()
    x = x.mask(interior, 0.0)
    return x, interior


def _to_monthly(daily: pd.DataFrame, *, min_obs: int = 5) -> pd.DataFrame:
    """Local copy of the month-end compounding convention, for the substitute series."""
    from research.multiasset.panel import monthly_returns
    return monthly_returns(daily, min_obs=min_obs)
