"""THE DATING-ALIGNMENT GUARD.

`research/sleeves/lowvol_retest.py::run_band` dated every monthly slot by the FORMATION
month while filling it with the FOLLOWING month's return. Mean, volatility, Sharpe,
drawdown, Newey-West t and the vol-matched active return are ALL invariant to that shift,
so an independent bit-for-bit re-implementation reproduced the series exactly and still
did not see it. It only surfaced when the series was joined to another one by date.

These tests are the thing that would have caught it:

  1. the probe itself, on synthetic data where the true answer is known by construction;
  2. `lowvol_retest.run_band` driven end to end on a synthetic panel with a planted
     market factor — the defect is reproduced under the registered convention and shown
     repaired under the corrected one, with the return arrays proven identical;
  3. every dated series the programme has written to disk, measured against a tracked
     correctly-dated reference and checked against its DECLARED convention;
  4. the reference itself re-anchored on an external index, where the vendor panel is
     present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.alignment import (
    ALIGNED,
    FORMATION,
    INSUFFICIENT_OVERLAP,
    MISALIGNED,
    REALISATION,
    UNINFORMATIVE,
    MisalignedSeriesError,
    assert_aligned,
    lag_correlations,
    month_end_index,
    probe_alignment,
    shift_months,
    to_month_end,
)
from research.sleeve_registry import REFERENCE_KEY, REGISTRY, audit, load_reference, load_spx

MONTHS = 240


def _market(seed: int = 20260728, n: int = MONTHS) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.period_range("1990-01", periods=n, freq="M").to_timestamp(how="end")
    return pd.Series(rng.normal(0.006, 0.042, n), index=index, name="market")


def _exposed(market: pd.Series, beta: float = 0.8, noise: float = 0.02,
             seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(beta * market.to_numpy() + rng.normal(0.0, noise, len(market)),
                     index=market.index, name="book")


# ── 1. the probe ──────────────────────────────────────────────────────────────
class TestAlignmentProbe:

    def test_correctly_dated_series_is_aligned_with_power(self):
        market = _market()
        probe = probe_alignment(_exposed(market), market)
        assert probe.verdict == ALIGNED
        assert probe.best_lag == 0
        assert probe.has_power
        assert probe.suggested_shift_months == 0

    def test_a_series_dated_one_month_early_is_caught(self):
        """The exact lowvol_retest defect: slot t holds month t+1's return."""
        market = _market()
        book = _exposed(market)
        early = shift_months(book, -1)          # relabel each return one month early
        probe = probe_alignment(early, market)
        assert probe.verdict == MISALIGNED
        assert probe.best_lag == +1
        assert probe.suggested_shift_months == +1
        assert probe.max_abs_rho > abs(probe.rho_at_zero)

    def test_a_series_dated_one_month_late_is_caught(self):
        market = _market()
        late = shift_months(_exposed(market), +1)
        probe = probe_alignment(late, market)
        assert probe.verdict == MISALIGNED
        assert probe.best_lag == -1
        assert probe.suggested_shift_months == -1

    @pytest.mark.parametrize("wrong_by", [-1, +1])
    def test_the_suggested_shift_repairs_the_series(self, wrong_by: int):
        market = _market()
        book = _exposed(market)
        broken = shift_months(book, -wrong_by)
        probe = probe_alignment(broken, market)
        assert probe.suggested_shift_months == wrong_by
        repaired = shift_months(broken, probe.suggested_shift_months)
        assert probe_alignment(repaired, market).verdict == ALIGNED
        # and the repair is a pure relabelling: the values are untouched
        joined = pd.concat({"a": to_month_end(book), "b": repaired}, axis=1).dropna()
        assert len(joined) >= MONTHS - 2
        assert np.allclose(joined["a"], joined["b"])

    def test_a_market_neutral_series_is_uninformative_not_a_pass(self):
        """A book with no market exposure cannot be proven aligned by this probe."""
        market = _market()
        rng = np.random.default_rng(99)
        neutral = pd.Series(rng.normal(0.0, 0.03, len(market)), index=market.index)
        probe = probe_alignment(neutral, market)
        assert probe.verdict in (ALIGNED, UNINFORMATIVE)
        assert not probe.has_power
        assert not probe.is_misaligned

    def test_require_power_refuses_an_unprovable_pass(self):
        market = _market()
        rng = np.random.default_rng(4)
        neutral = pd.Series(rng.normal(0.0, 0.03, len(market)), index=market.index)
        assert_aligned(neutral, market)                       # tolerated by default
        with pytest.raises(MisalignedSeriesError, match="no power"):
            assert_aligned(neutral, market, require_power=True)

    def test_assert_aligned_raises_on_a_misaligned_series(self):
        market = _market()
        early = shift_months(_exposed(market), -1)
        with pytest.raises(MisalignedSeriesError, match="misaligned|month"):
            assert_aligned(early, market, name="early")

    def test_gaps_do_not_corrupt_the_measurement(self):
        """A positional `.shift` would smear every later month; calendar labels cannot."""
        market = _market()
        book = _exposed(market)
        holed = book.drop(book.index[[40, 41, 90, 150]])
        assert probe_alignment(holed, market).verdict == ALIGNED
        assert probe_alignment(shift_months(holed, -1), market).best_lag == +1

    def test_month_start_and_month_end_labels_join(self):
        market = _market()
        book = _exposed(market)
        month_start = book.copy()
        month_start.index = book.index.to_period("M").to_timestamp(how="start")
        assert probe_alignment(month_start, market).verdict == ALIGNED
        assert list(month_end_index(month_start.index)) == list(month_end_index(book.index))

    def test_duplicate_months_are_refused(self):
        market = _market()
        doubled = pd.concat([market, market])
        with pytest.raises(ValueError, match="duplicate months"):
            to_month_end(doubled)

    def test_too_little_overlap_is_reported_not_guessed(self):
        market = _market()
        short = _exposed(market).iloc[:10]
        probe = probe_alignment(short, market)
        assert probe.verdict == INSUFFICIENT_OVERLAP
        assert not probe.has_power

    def test_lag_zero_must_be_probed(self):
        market = _market()
        with pytest.raises(ValueError, match="lags must include 0"):
            probe_alignment(_exposed(market), market, lags=(-1, 1))

    def test_lag_correlations_report_their_overlap(self):
        market = _market()
        rho, n = lag_correlations(_exposed(market), market)
        assert set(rho) == {-1, 0, 1}
        assert n[0] == MONTHS
        assert n[-1] == n[1] == MONTHS - 1
        assert rho[0] > rho[-1] and rho[0] > rho[1]

    def test_probe_is_serialisable(self):
        market = _market()
        payload = probe_alignment(_exposed(market), market, name="book").to_dict()
        assert payload["verdict"] == ALIGNED
        assert payload["suggested_shift_months"] == 0
        assert set(payload["rho"]) == {"-1", "0", "1"}


# ── 2. the sleeve that had the defect, end to end ─────────────────────────────
def _synthetic_lowvol_panel(n_months: int = 60, n_names: int = 80,
                            seed: int = 20260728) -> tuple[pd.DataFrame, pd.Series]:
    """A panel with a planted market factor, in the shape `run_band` expects.

    The row dated month ``m`` carries ``forward_return`` — the return EARNED in month
    ``m+1``. The returned reference series is the market factor dated by the month it was
    earned, so a correctly-dated book must peak against it at lag 0.
    """
    from research.sleeves.low_vol_quality import MIN_CROSS_SECTION, N_POSITIONS

    assert n_names > MIN_CROSS_SECTION >= N_POSITIONS
    rng = np.random.default_rng(seed)
    months = pd.period_range("1990-01", periods=n_months + 1, freq="M")
    market = rng.normal(0.007, 0.045, n_months + 1)

    rows = []
    for i, month in enumerate(months[:-1]):
        # forward_return is EARNED in month i+1
        earned = market[i + 1] + rng.normal(0.0, 0.012, n_names)
        rows.append(pd.DataFrame({
            "ticker": [f"T{j:03d}" for j in range(n_names)],
            "date": month.to_timestamp(how="end").normalize(),
            "band_group": "B2_200k_1M",
            "forward_return": earned,
            "signal": rng.normal(0.0, 1.0, n_names),
            "median_dollar_volume": 5.0e5,
            "close": 25.0,
            "realised_vol": 0.02,
            "spread_regime": "measured",
            "spread_conservative": 0.004,
            "spread_realistic": 0.002,
        }))
    panel = pd.concat(rows, ignore_index=True)
    reference = pd.Series(
        market[1:],
        index=pd.PeriodIndex(months[1:], freq="M").to_timestamp(how="end").normalize(),
        name="market",
    )
    return panel, reference


@pytest.fixture(scope="module")
def synthetic_panel():
    return _synthetic_lowvol_panel()


@pytest.fixture(scope="module")
def empty_delistings():
    return pd.DataFrame({"ticker": pd.Series(dtype="object"),
                         "date": pd.Series(dtype="datetime64[ns]"),
                         "terminal_return": pd.Series(dtype="float")})


class TestRunBandDating:
    """Drive the real `run_band` and prove the defect and its repair."""

    def test_registered_convention_reproduces_the_defect(self, synthetic_panel,
                                                         empty_delistings):
        from research.sleeves import lowvol_retest as LV

        panel, reference = synthetic_panel
        books = LV.run_band(panel, "B2_200k_1M", empty_delistings)
        assert books is not None
        assert books.date_convention == FORMATION
        series = pd.Series(books.gross,
                           index=pd.PeriodIndex(books.months, freq="M")
                           .to_timestamp(how="end").normalize())
        probe = probe_alignment(series, reference, name="run_band[formation]")
        assert probe.verdict == MISALIGNED, probe.describe()
        assert probe.best_lag == +1, probe.describe()

    def test_realisation_convention_is_aligned(self, synthetic_panel, empty_delistings):
        from research.sleeves import lowvol_retest as LV

        panel, reference = synthetic_panel
        books = LV.run_band(panel, "B2_200k_1M", empty_delistings,
                            date_convention=REALISATION)
        assert books is not None
        assert books.date_convention == REALISATION
        series = pd.Series(books.gross,
                           index=pd.PeriodIndex(books.months, freq="M")
                           .to_timestamp(how="end").normalize())
        assert_aligned(series, reference, name="run_band[realisation]", require_power=True)

    def test_the_switch_moves_only_the_labels(self, synthetic_panel, empty_delistings):
        """The registered default must stay bit-for-bit; only the index may move."""
        from research.sleeves import lowvol_retest as LV

        panel, _reference = synthetic_panel
        registered = LV.run_band(panel, "B2_200k_1M", empty_delistings)
        corrected = LV.run_band(panel, "B2_200k_1M", empty_delistings,
                                date_convention=REALISATION)
        assert registered is not None and corrected is not None
        for field in ("gross", "benchmark", "benchmark_rankable", "cost_conservative",
                      "cost_realistic", "commission_cost"):
            a = np.asarray(getattr(registered, field), dtype=float)
            b = np.asarray(getattr(corrected, field), dtype=float)
            assert np.array_equal(a, b, equal_nan=True), field
        assert [m + 1 for m in registered.months] == list(corrected.months)
        assert [(t, m + 1, v) for t, m, v in registered.pnl_by_name_month] == \
            corrected.pnl_by_name_month

    def test_an_unknown_convention_is_refused(self, synthetic_panel, empty_delistings):
        from research.sleeves import lowvol_retest as LV

        panel, _ = synthetic_panel
        with pytest.raises(ValueError, match="date_convention"):
            LV.run_band(panel, "B2_200k_1M", empty_delistings,
                        date_convention="whatever")


# ── 3. every dated artefact on disk ───────────────────────────────────────────
class TestRegisteredSeriesAreDatedAsDeclared:

    def test_registry_keys_are_unique(self):
        keys = [e.key for e in REGISTRY]
        assert len(keys) == len(set(keys))

    def test_every_registered_artefact_exists(self):
        missing = [e.key for e in REGISTRY if not e.full_path.exists()]
        assert not missing, f"registered artefacts missing from disk: {missing}"

    def test_the_reference_is_registered_and_realisation_dated(self):
        from research.sleeve_registry import by_key

        reference = by_key(REFERENCE_KEY)
        assert reference.convention == REALISATION
        assert reference.market_exposed

    def test_nothing_is_misaligned_once_its_declared_shift_is_applied(self):
        """THE GUARD. Fails if any sleeve's returns are off by +/-1 month."""
        probes = audit()
        broken = {k: p.describe() for k, p in probes.items() if p.is_misaligned}
        assert not broken, "misaligned series: " + "; ".join(broken.values())

    def test_market_exposed_series_actually_prove_alignment(self):
        from research.sleeve_registry import by_key

        probes = audit()
        for key, probe in probes.items():
            if not by_key(key).market_exposed:
                continue
            assert probe.verdict == ALIGNED, probe.describe()
            assert probe.has_power, probe.describe()

    def test_the_declared_convention_matches_the_raw_dates(self):
        """Pins each artefact's ON-DISK dating, so a producer cannot change it quietly."""
        from research.sleeve_registry import by_key

        probes = audit(as_stored=True)
        for key, probe in probes.items():
            entry = by_key(key)
            if not entry.market_exposed:
                continue                      # the probe has no power to pin anything
            if entry.convention == REALISATION:
                assert probe.verdict == ALIGNED and probe.best_lag == 0, probe.describe()
            else:
                assert probe.verdict == MISALIGNED, probe.describe()
                assert probe.best_lag == entry.shift_to_realisation(), probe.describe()

    def test_the_portfolio_study_shifts_exactly_the_formation_dated_sources(self):
        """A consumer that drops the compensating shift must break this test."""
        from research.sleeves._portfolio import portfolio_correlation_v2 as V2

        formation_paths = {e.path for e in REGISTRY if e.convention == FORMATION}
        for key, (path, _col, _conv) in V2.SOURCES.items():
            relative = path.relative_to(V2.REPO).as_posix()
            if relative in formation_paths:
                assert key in V2.NEEDS_MONTH_SHIFT, (
                    f"{key} reads a FORMATION-dated artefact but is not shifted")
            else:
                assert key not in V2.NEEDS_MONTH_SHIFT, (
                    f"{key} reads a REALISATION-dated artefact but is being shifted")


# ── 4. anchor the reference on something external ─────────────────────────────
class TestReferenceAnchor:

    def test_reference_is_aligned_against_spx(self):
        spx = load_spx()
        if spx is None:
            pytest.skip("_data/multiasset/returns_monthly.parquet is not on this machine")
        probe = assert_aligned(load_reference(), spx, name=REFERENCE_KEY,
                               reference_name="SPX", require_power=True)
        assert probe.max_abs_rho > 0.5, probe.describe()

    def test_the_full_audit_agrees_against_spx(self):
        spx = load_spx()
        if spx is None:
            pytest.skip("_data/multiasset/returns_monthly.parquet is not on this machine")
        broken = {k: p.describe() for k, p in audit(spx).items() if p.is_misaligned}
        assert not broken, "misaligned against SPX: " + "; ".join(broken.values())
