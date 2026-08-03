"""
Sharadar (Nasdaq Data Link) ingestion tests — offline, deterministic, NO network.

Fixtures are synthetic SF1 + SEP CSVs written to a pytest ``tmp_path`` (a few tickers,
one of them delisted, a couple of filings each). They prove the three properties the
loader exists to guarantee:

  (1) POINT-IN-TIME — a filing is NOT visible before its ``datekey`` (and crucially, the
      loader keys on ``datekey``, not ``calendardate``: a figure whose accounting period
      has ended but whose filing has not yet been published must stay invisible).
  (2) SURVIVORSHIP-FREEDOM — a delisted ticker's price history is retained in the panel.
  (3) The merged panel's shape / keys / PIT forward-fill are correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.sharadar_ingestion import (
    SF1_FUNDAMENTAL_COLUMNS,
    build_panel,
    load_sep,
    load_sf1,
    pit_fundamentals,
    pit_value,
)

# ── Synthetic fixture builders ───────────────────────────────────────────────────────

# AAA / BBB stay listed all the way through; ZZZ is delisted after 2020-06 (no later
# prices). Each filing's `datekey` (publication) is WEEKS after its `calendardate`
# (period end) — that gap is what the PIT / no-calendardate-leak tests exploit.
_SF1_ROWS = [
    # ticker, dimension, datekey,     calendardate, revenue, netinc, equity, assets
    ("AAA", "ARQ", "2020-05-01", "2020-03-31", 100.0, 10.0, 200.0, 500.0),
    ("AAA", "ARQ", "2020-08-01", "2020-06-30", 120.0, 12.0, 210.0, 520.0),
    # A restated MRQ view for AAA Q1 — must be excluded when dimension="ARQ".
    ("AAA", "MRQ", "2020-05-01", "2020-03-31", 999.0, 99.0, 999.0, 999.0),
    ("BBB", "ARQ", "2020-05-05", "2020-03-31", 300.0, 30.0, 400.0, 900.0),
    ("BBB", "ARQ", "2020-08-05", "2020-06-30", 330.0, 33.0, 410.0, 920.0),
    ("ZZZ", "ARQ", "2020-05-01", "2020-03-31", 50.0, 5.0, 80.0, 150.0),
]

_SEP_DATES = ["2020-04-15", "2020-05-15", "2020-06-15", "2020-07-15", "2020-08-15", "2020-09-15"]
# ZZZ is delisted after June: only the first three dates exist for it.
_SEP_TICKER_DATES = {
    "AAA": _SEP_DATES,
    "BBB": _SEP_DATES,
    "ZZZ": _SEP_DATES[:3],
}


def _write_sf1(path: Path) -> Path:
    """Write a full-width SF1 CSV (all documented fundamental columns present)."""
    rng = np.random.default_rng(0)
    records = []
    for tic, dim, datekey, cal, rev, ni, eq, assets in _SF1_ROWS:
        row = {
            "ticker": tic,
            "dimension": dim,
            "datekey": datekey,
            "calendardate": cal,
            "revenue": rev,
            "netinc": ni,
            "equity": eq,
            "assets": assets,
            "liabilities": assets - eq,
            "eps": round(ni / 100.0, 4),
            "ebit": ni * 1.3,
            "ebitda": ni * 1.6,
            "gp": rev * 0.4,
            "ncfo": ni * 1.1,
            "debt": assets * 0.2,
            "sharesbas": 1000.0 + float(rng.integers(0, 10)),
        }
        records.append(row)
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def _write_sep(path: Path) -> Path:
    rng = np.random.default_rng(1)
    records = []
    for tic, dates in _SEP_TICKER_DATES.items():
        for d in dates:
            close = 10.0 + float(rng.uniform(0.0, 5.0))
            records.append({
                "ticker": tic,
                "date": d,
                "close": close,
                "closeadj": close * 0.95,   # distinct from raw close so we can tell them apart
                "volume": float(rng.integers(1_000, 5_000)),
            })
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


# ── Loading ────────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_sf1_loads_and_parses(self, tmp_path):
        sf1 = load_sf1(_write_sf1(tmp_path / "sf1.csv"))
        assert {"ticker", "dimension", "datekey", "calendardate"}.issubset(sf1.columns)
        assert sf1["datekey"].dtype.kind == "M" and sf1["calendardate"].dtype.kind == "M"
        for col in SF1_FUNDAMENTAL_COLUMNS:
            assert col in sf1.columns
        assert {"AAA", "BBB", "ZZZ"}.issubset(set(sf1["ticker"]))

    def test_sf1_dimension_filter_excludes_restated(self, tmp_path):
        # With dimension="ARQ" the restated MRQ row (revenue 999) must be gone.
        arq = load_sf1(_write_sf1(tmp_path / "sf1.csv"), dimension="ARQ")
        assert set(arq["dimension"]) == {"ARQ"}
        assert 999.0 not in set(arq["revenue"])

    def test_sf1_loads_from_directory(self, tmp_path):
        # Split the SF1 rows across two CSVs in a directory; the loader concatenates them.
        d = tmp_path / "sf1dir"
        d.mkdir()
        full = pd.DataFrame.from_records([
            {"ticker": t, "dimension": dim, "datekey": dk, "calendardate": cal,
             "revenue": rev, "netinc": ni, "equity": eq, "assets": a}
            for (t, dim, dk, cal, rev, ni, eq, a) in _SF1_ROWS
        ])
        full.iloc[:3].to_csv(d / "part_a.csv", index=False)
        full.iloc[3:].to_csv(d / "part_b.csv", index=False)
        sf1 = load_sf1(d)
        assert len(sf1) == len(_SF1_ROWS)
        assert {"AAA", "BBB", "ZZZ"}.issubset(set(sf1["ticker"]))

    def test_sep_prefers_adjusted_close(self, tmp_path):
        sep_path = _write_sep(tmp_path / "sep.csv")
        adj = load_sep(sep_path, use_adjusted=True)
        raw = load_sep(sep_path, use_adjusted=False)
        assert list(adj.columns) == ["ticker", "date", "close", "volume"]
        # closeadj == 0.95 * close in the fixture, so the two loads must differ.
        merged = adj.merge(raw, on=["ticker", "date"], suffixes=("_adj", "_raw"))
        assert np.allclose(merged["close_adj"], merged["close_raw"] * 0.95)

    def test_sep_missing_volume_is_all_nan_float(self, tmp_path):
        # Regression: a SEP export with NO 'volume' column must not crash (pandas 2.2
        # cannot fill a float64 buffer with pd.NA) and must yield an all-NaN float64 column.
        path = tmp_path / "sep_novol.csv"
        pd.DataFrame(
            {"ticker": ["AAA", "AAA"], "date": ["2020-01-31", "2020-02-28"],
             "close": [10.0, 11.0]}
        ).to_csv(path, index=False)
        sep = load_sep(path)
        assert "volume" in sep.columns
        assert sep["volume"].dtype == np.dtype("float64")
        assert sep["volume"].isna().all()
        assert len(sep) == 2                       # rows are still loaded, just no volume


# ── (1) Point-in-time correctness ─────────────────────────────────────────────────────

class TestPointInTime:
    def _sf1(self, tmp_path):
        return load_sf1(_write_sf1(tmp_path / "sf1.csv"), dimension="ARQ")

    def test_not_visible_before_datekey(self, tmp_path):
        sf1 = self._sf1(tmp_path)
        # First AAA filing has datekey 2020-05-01; nothing is knowable on 2020-04-15.
        assert pit_fundamentals(sf1, "AAA", "2020-04-15") is None
        assert pit_value(sf1, "AAA", "revenue", "2020-04-15") is None
        # The day on/after the filing date it becomes visible.
        assert pit_value(sf1, "AAA", "revenue", "2020-05-15") == 100.0

    def test_datekey_not_calendardate(self, tmp_path):
        # THE leak test. AAA's Q2 filing has calendardate 2020-06-30 but datekey
        # 2020-08-01. On 2020-07-15 the *period* has ended but the *filing* has not been
        # published — a calendardate-based loader would leak revenue 120 here.
        sf1 = self._sf1(tmp_path)
        assert pit_value(sf1, "AAA", "revenue", "2020-07-15") == 100.0   # still Q1
        assert pit_value(sf1, "AAA", "revenue", "2020-08-15") == 120.0   # Q2 now known

    def test_dropping_future_filings_is_a_noop(self, tmp_path):
        # Explicit PIT property: removing rows filed AFTER asof cannot change the answer.
        sf1 = self._sf1(tmp_path)
        asof = pd.Timestamp("2020-07-15")
        masked = sf1[sf1["datekey"] <= asof]
        assert pit_value(sf1, "AAA", "revenue", asof) == pit_value(masked, "AAA", "revenue", asof)

    def test_default_path_returns_as_reported_not_restated(self, tmp_path):
        # FAIL-CLOSED default: AAA has an ARQ row (revenue 100) AND a restated MRQ row
        # (revenue 999) sharing datekey 2020-05-01 AND calendardate 2020-03-31. Loaded
        # WITHOUT a dimension filter (both rows present), the DEFAULT accessor path (no
        # explicit dimension) must return the As-Reported value, NEVER the restated one.
        sf1_all = load_sf1(_write_sf1(tmp_path / "sf1.csv"))     # keep every dimension
        assert {"ARQ", "MRQ"}.issubset(set(sf1_all["dimension"]))
        # No `dimension=` argument anywhere -> the ARQ default applies.
        assert pit_value(sf1_all, "AAA", "revenue", "2020-05-15") == 100.0
        row = pit_fundamentals(sf1_all, "AAA", "2020-05-15")
        assert row is not None and row["dimension"] == "ARQ" and row["revenue"] == 100.0

    def test_opt_out_tie_break_still_prefers_as_reported(self, tmp_path):
        # Even when the caller explicitly opts out of the fail-closed default (dimension
        # =None / "ALL", so BOTH the ARQ and restated MRQ row for the tied datekey are in
        # play), the datekey tie-break must prefer the As-Reported row over the restatement.
        sf1_all = load_sf1(_write_sf1(tmp_path / "sf1.csv"))
        for opt_out in (None, "ALL"):
            assert pit_value(sf1_all, "AAA", "revenue", "2020-05-15", dimension=opt_out) == 100.0
            row = pit_fundamentals(sf1_all, "AAA", "2020-05-15", dimension=opt_out)
            assert row is not None and row["dimension"] == "ARQ"


# ── (2) Survivorship-freedom ───────────────────────────────────────────────────────────

class TestSurvivorship:
    def test_delisted_ticker_history_retained(self, tmp_path):
        sf1 = load_sf1(_write_sf1(tmp_path / "sf1.csv"), dimension="ARQ")
        sep = load_sep(_write_sep(tmp_path / "sep.csv"))
        panel = build_panel(sf1, sep)
        # ZZZ is delisted (no prices past June) yet its history is NOT dropped.
        assert "ZZZ" in set(panel["ticker"])
        zzz = panel[panel["ticker"] == "ZZZ"]
        assert len(zzz) == 3
        assert zzz["date"].max() == pd.Timestamp("2020-06-15")
        # Its known fundamentals (filed 2020-05-01) are present from 2020-05-15 on.
        zzz_late = zzz[zzz["date"] == pd.Timestamp("2020-06-15")].iloc[0]
        assert zzz_late["revenue"] == 50.0


# ── (3) Merged panel shape / keys / PIT forward-fill ──────────────────────────────────

class TestPanel:
    def _panel(self, tmp_path):
        sf1 = load_sf1(_write_sf1(tmp_path / "sf1.csv"), dimension="ARQ")
        sep = load_sep(_write_sep(tmp_path / "sep.csv"))
        return build_panel(sf1, sep)

    def test_shape_and_keys(self, tmp_path):
        panel = self._panel(tmp_path)
        # 6 + 6 + 3 = 15 price rows; one panel row each, keyed uniquely by (ticker, date).
        assert len(panel) == 15
        assert not panel.duplicated(subset=["ticker", "date"]).any()
        for col in ("ticker", "date", "close", "volume", "revenue",
                    "filed_datekey", "fundamental_calendardate"):
            assert col in panel.columns
        assert set(panel["ticker"]) == {"AAA", "BBB", "ZZZ"}

    def test_forward_fill_is_pit_safe(self, tmp_path):
        panel = self._panel(tmp_path)
        aaa = panel[panel["ticker"] == "AAA"].set_index("date")
        # Before the first filing was known -> NaN fundamentals (nothing fabricated).
        assert pd.isna(aaa.loc[pd.Timestamp("2020-04-15"), "revenue"])
        assert pd.isna(aaa.loc[pd.Timestamp("2020-04-15"), "filed_datekey"])
        # Q1 known from May; Q2 not until its 2020-08-01 datekey (NOT its 06-30 period end).
        assert aaa.loc[pd.Timestamp("2020-05-15"), "revenue"] == 100.0
        assert aaa.loc[pd.Timestamp("2020-07-15"), "revenue"] == 100.0
        assert aaa.loc[pd.Timestamp("2020-08-15"), "revenue"] == 120.0
        # The audit column reflects the actual publication date used for the fill.
        assert aaa.loc[pd.Timestamp("2020-08-15"), "filed_datekey"] == pd.Timestamp("2020-08-01")

    def test_build_panel_agrees_with_pit_on_tied_datekey_when_scrambled(self, tmp_path):
        # Determinism: two AAA ARQ filings share datekey 2020-06-01 but have DIFFERENT
        # calendardates (Q1 rev 10 vs Q2 rev 20). build_panel sorts the right frame by
        # (datekey, calendardate) so the tie resolves identically to pit_fundamentals
        # (latest calendardate wins) REGARDLESS of the caller's input row order.
        tied = pd.DataFrame.from_records([
            {"ticker": "AAA", "dimension": "ARQ", "datekey": "2020-06-01",
             "calendardate": "2020-03-31", "revenue": 10.0, "netinc": 1.0,
             "equity": 50.0, "assets": 100.0},
            {"ticker": "AAA", "dimension": "ARQ", "datekey": "2020-06-01",
             "calendardate": "2020-06-30", "revenue": 20.0, "netinc": 2.0,
             "equity": 60.0, "assets": 110.0},
        ])
        sep = pd.DataFrame({"ticker": ["AAA", "AAA"],
                            "date": ["2020-05-15", "2020-06-15"],
                            "close": [10.0, 11.0], "closeadj": [10.0, 11.0]})
        sep_path = tmp_path / "sep_tied.csv"
        sep.to_csv(sep_path, index=False)

        results = []
        for order in ([0, 1], [1, 0]):                          # both input orderings
            sf1_path = tmp_path / f"sf1_tied_{order[0]}{order[1]}.csv"
            tied.iloc[order].to_csv(sf1_path, index=False)
            sf1 = load_sf1(sf1_path, dimension="ARQ")
            sep_loaded = load_sep(sep_path)
            panel = build_panel(sf1, sep_loaded)
            asof = pd.Timestamp("2020-06-15")
            panel_rev = panel.loc[
                (panel["ticker"] == "AAA") & (panel["date"] == asof), "revenue"
            ].iloc[0]
            pit_rev = pit_fundamentals(sf1, "AAA", asof)["revenue"]
            # build_panel and pit_fundamentals must agree, and pick the latest-calendardate
            # (Q2, revenue 20) filing — not the order-dependent first row.
            assert panel_rev == pit_rev == 20.0
            results.append(panel_rev)
        assert results[0] == results[1]                         # order-independent (deterministic)


# --------------------------------------------------------------------------- #
# Column-selective reads (full exports are multi-GB; only needed columns load)
# --------------------------------------------------------------------------- #
def test_loaders_ignore_extra_export_columns(tmp_path: Path) -> None:
    """The real SF1 export has ~110 columns and SEP ~10; the loaders must return
    IDENTICAL frames whether or not the extras exist — and must not require them."""
    sf1_plain = _write_sf1(tmp_path / "plain_SF1.csv")
    sep_plain = _write_sep(tmp_path / "plain_SEP.csv")

    fat_sf1 = pd.read_csv(sf1_plain)
    for i in range(60):
        fat_sf1[f"junkcol{i}"] = "x" * 50            # bulky never-used strings
    fat_sf1_path = tmp_path / "fat_SF1.csv"
    fat_sf1.to_csv(fat_sf1_path, index=False)

    fat_sep = pd.read_csv(sep_plain)
    for extra in ("open", "high", "low", "closeunadj", "lastupdated"):
        fat_sep[extra] = "9999.99"
    fat_sep_path = tmp_path / "fat_SEP.csv"
    fat_sep.to_csv(fat_sep_path, index=False)

    pd.testing.assert_frame_equal(load_sf1(sf1_plain, dimension="ARQ"),
                                  load_sf1(fat_sf1_path, dimension="ARQ"))
    pd.testing.assert_frame_equal(load_sep(sep_plain), load_sep(fat_sep_path))


def test_read_frames_usecols_prunes_at_read_time(tmp_path: Path) -> None:
    from data.sharadar_ingestion import _read_frames

    p = tmp_path / "t.csv"
    pd.DataFrame({"Ticker": ["A"], "date": ["2020-01-01"], "JUNK": ["z"]}).to_csv(
        p, index=False)
    frame = _read_frames(p, usecols=("ticker", "date"))
    assert [c.strip().lower() for c in frame.columns] == ["ticker", "date"]
