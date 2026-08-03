"""
Phase 3 — reconciliation engine tests (positions / cash / NAV; surfacing only).
"""

from __future__ import annotations

from ops.reconciliation import reconcile

ASOF = "2026-01-01T00:00:00"


class TestPositions:
    def test_aligned_is_clean(self):
        r = reconcile({"positions": {"AAPL": 100}}, {"positions": {"AAPL": 100}},
                      asof=ASOF, share_tol=1.0)
        assert r.clean and r.to_alert() is None

    def test_within_tolerance_clean(self):
        r = reconcile({"positions": {"AAPL": 100}}, {"positions": {"AAPL": 100.5}},
                      asof=ASOF, share_tol=1.0)
        assert r.clean

    def test_break_reported(self):
        r = reconcile({"positions": {"AAPL": 100}}, {"positions": {"AAPL": 50}},
                      asof=ASOF, share_tol=1.0)
        assert not r.clean
        b = [x for x in r.breaks if x.dimension == "position"][0]
        assert b.key == "AAPL" and b.diff == 50.0 and b.severity == "BREAK"

    def test_broker_only_symbol_breaks(self):
        r = reconcile({"positions": {}}, {"positions": {"MSFT": 10}}, asof=ASOF, share_tol=1.0)
        assert any(b.key == "MSFT" for b in r.breaks)


class TestCashAndNav:
    def test_cash_break(self):
        r = reconcile({"cash": {"USD": 1000.0}}, {"cash": {"USD": 500.0}}, asof=ASOF, cash_tol=1.0)
        assert any(b.dimension == "cash" and b.key == "USD" for b in r.breaks)

    def test_nav_relative_tolerance(self):
        assert reconcile({"nav": 1_000_000}, {"nav": 1_001_000}, asof=ASOF, nav_tol_pct=0.005).clean
        assert not reconcile({"nav": 1_000_000}, {"nav": 1_010_000}, asof=ASOF, nav_tol_pct=0.005).clean

    def test_nav_only_compared_when_both_present(self):
        assert reconcile({"nav": 1e6}, {}, asof=ASOF).clean       # broker NAV absent -> not compared


class TestFailClosedAndOutputs:
    def test_nonfinite_fails_closed(self):
        r = reconcile({"positions": {"AAPL": float("nan")}}, {"positions": {"AAPL": 100}}, asof=ASOF)
        assert any(b.severity == "BREAK" for b in r.breaks)

    def test_alert_shape_and_severity(self):
        r = reconcile({"positions": {"AAPL": 100}}, {"positions": {"AAPL": 0}}, asof=ASOF)
        a_paper, a_live = r.to_alert(mode="PAPER"), r.to_alert(mode="LIVE")
        assert a_paper["kind"] == "reconciliation" and a_paper["severity"] == "WARNING"
        assert a_live["severity"] == "RED"
        assert isinstance(a_paper["detail"], list) and a_paper["detail"]

    def test_payload(self):
        r = reconcile({"positions": {"AAPL": 100}}, {"positions": {"AAPL": 0}}, asof=ASOF)
        p = r.to_payload()
        assert p["clean"] is False and p["n_breaks"] == 1 and p["asof"] == ASOF

    def test_multi_dimension(self):
        r = reconcile(
            {"positions": {"AAPL": 100}, "cash": {"USD": 1000}, "nav": 1_000_000},
            {"positions": {"AAPL": 0}, "cash": {"USD": 0}, "nav": 1_500_000},
            asof=ASOF, share_tol=1.0, cash_tol=1.0, nav_tol_pct=0.005,
        )
        dims = {b.dimension for b in r.breaks}
        assert dims == {"position", "cash", "nav"}
