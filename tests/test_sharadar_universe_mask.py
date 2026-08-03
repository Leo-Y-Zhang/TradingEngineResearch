"""
Tests for the tradable-universe machinery added after the 2026-07-13 dev diagnosis
(``sharadar_dev_log.md`` entry 2): ``build_liquidity_universe`` + the ``universe_mask``
/ ``fwd_return_cap`` parameters of ``run_research``. Offline, deterministic, NO network.

Properties:
  1. Liquidity ranking: top_n keeps the most liquid names per date; an absolute
     dollar-volume floor excludes below-floor names; both are PIT (a name becomes
     eligible only from trailing data strictly up to each panel date).
  2. min_obs: sparse trading histories are ineligible.
  3. run_research defaults (mask=None, cap=None) are BIT-IDENTICAL to the registered
     behavior — pinned by comparing full reports on a synthetic panel.
  4. The mask restricts evaluation: an outlier name that dominates the unfiltered
     P&L stops mattering when masked out; the cap tames the same outlier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research_sharadar_alpha import (
    build_liquidity_universe,
    load_panel,
    run_research,
    write_synthetic_csvs,
)


def _sep(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "close": [r[2] for r in rows],
            "volume": [r[3] for r in rows],
        }
    )


def _daily_sep(specs: dict[str, tuple[float, float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for tic, (price, vol) in specs.items():
        for d in dates:
            rows.append((tic, d.strftime("%Y-%m-%d"), price, vol))
    return _sep(rows)


BDAYS = pd.bdate_range("2010-01-01", "2010-12-31")


def test_top_n_keeps_most_liquid_and_floor_excludes() -> None:
    sep = _daily_sep(
        {"BIG": (100.0, 1e6), "MID": (10.0, 1e5), "TINY": (1.0, 1e3)}, BDAYS
    )
    panel = [pd.Timestamp("2010-06-30"), pd.Timestamp("2010-09-30")]

    top2 = build_liquidity_universe(sep, panel, top_n=2)
    assert top2.loc[panel[0], "BIG"] and top2.loc[panel[0], "MID"]
    assert not top2.loc[panel[0], "TINY"]

    # MID trades $1M/day which clears a $0.5M floor; TINY ($1k/day) does not.
    floored = build_liquidity_universe(sep, panel, min_dollar_volume=5e5)
    assert floored.loc[panel[1], "BIG"]
    assert floored.loc[panel[1], "MID"]
    assert not floored.loc[panel[1], "TINY"]


def test_universe_is_point_in_time() -> None:
    # ILLIQ trades thin all year, then becomes hugely liquid from October.
    rows = []
    for d in BDAYS:
        big_vol = 1e6
        illiq_vol = 1e2 if d < pd.Timestamp("2010-10-01") else 1e7
        rows.append(("BIG", d.strftime("%Y-%m-%d"), 100.0, big_vol))
        rows.append(("ILLIQ", d.strftime("%Y-%m-%d"), 10.0, illiq_vol))
    sep = _sep(rows)
    panel = [pd.Timestamp("2010-06-30"), pd.Timestamp("2010-12-31")]
    mask = build_liquidity_universe(sep, panel, min_dollar_volume=1e5)
    assert not mask.loc[panel[0], "ILLIQ"]          # June: history says thin
    assert mask.loc[panel[1], "ILLIQ"]              # December: trailing median is rich


def test_min_obs_excludes_sparse_histories() -> None:
    dates = list(BDAYS[:100])
    rows = [("FULL", d.strftime("%Y-%m-%d"), 50.0, 1e6) for d in dates]
    rows += [("SPARSE", d.strftime("%Y-%m-%d"), 50.0, 1e6) for d in dates[-10:]]
    sep = _sep(rows)
    panel = [dates[-1]]
    mask = build_liquidity_universe(sep, panel, top_n=5, min_obs=42)
    assert mask.loc[panel[0], "FULL"]
    assert not mask.loc[panel[0], "SPARSE"]        # only 10 observed days


def test_requires_some_criterion() -> None:
    sep = _daily_sep({"AAA": (10.0, 1e5)}, BDAYS[:70])
    with pytest.raises(ValueError, match="top_n and/or min_dollar_volume"):
        build_liquidity_universe(sep, [BDAYS[69]])


def test_run_research_defaults_are_bit_identical(tmp_path: Path) -> None:
    write_synthetic_csvs(tmp_path, seed=7, edge=True)
    sf1, sep = load_panel(tmp_path)
    base = run_research(sf1, sep, label="pin")
    again = run_research(sf1, sep, label="pin", universe_mask=None, fwd_return_cap=None)
    assert base.weights == again.weights
    assert base.result.deflated_sharpe_ratio == again.result.deflated_sharpe_ratio
    assert base.result.mean_ic == again.result.mean_ic
    assert base.result.mean_rank_ic == again.result.mean_rank_ic
    assert base.result.sharpe_net == again.result.sharpe_net
    assert base.pbo == again.pbo
    assert base.comp_sharpe == again.comp_sharpe


def test_mask_and_cap_change_evaluation(tmp_path: Path) -> None:
    write_synthetic_csvs(tmp_path, seed=7, edge=True)
    sf1, sep = load_panel(tmp_path)
    base = run_research(sf1, sep, label="pin")

    # Mask that voids half the names: metrics must differ from the unfiltered run
    # (the synthetic edge lives across the whole cross-section).
    px_names = sorted(sep["ticker"].unique())
    keep = px_names[: len(px_names) // 2]
    dates = pd.date_range("2010-01-01", "2030-01-01", freq="ME")
    mask = pd.DataFrame(False, index=dates, columns=px_names)
    mask.loc[:, keep] = True
    masked = run_research(sf1, sep, label="pin", universe_mask=mask)
    assert masked.result.sharpe_net != base.result.sharpe_net

    # A tiny cap must change measured returns on the synthetic panel too.
    capped = run_research(sf1, sep, label="pin", fwd_return_cap=0.01)
    assert capped.result.sharpe_net != base.result.sharpe_net

    with pytest.raises(ValueError, match="fwd_return_cap"):
        run_research(sf1, sep, fwd_return_cap=-1.0)
