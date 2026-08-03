"""
Phase 2 Tests — Property-based invariants for the numeric kernels
=================================================================
Hypothesis property tests for the four kernels called the building spec flags as
correctness-critical: OFI (microstructure), vol_ratio (volatility), CVaR
(optimiser) and TCA cost. Each test asserts an invariant that PROVABLY holds for
all valid in-domain inputs; counterexamples indicate real bugs. Two such bugs were
surfaced here and fixed (constant-magnitude HAR-RV crash; OFI NaN propagation).

Invariants deliberately NOT asserted (would flag correct code) are documented at
the kernel that owns them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import kurtosis, norm, skew

from core.engine.microstructure import _OFI_REJECT_THRESHOLD, compute_ofi, ofi_filter_gate
from core.engine.optimizer import _gaussian_cvar, compute_portfolio_cvar_cf, portfolio_cvar
from execution.tca import ex_ante_cost_model, reset_tca_model
from strategies.volatility_model import vol_ratio_current

_F = dict(allow_nan=False, allow_infinity=False)
_finite = st.floats(min_value=-1e12, max_value=1e12, **_F)
_nonneg = st.floats(min_value=0.0, max_value=1e12, **_F)
_deltas = st.lists(_finite, min_size=1, max_size=5)
_volume = st.floats(min_value=1e-6, max_value=1e12, **_F)
_l2 = st.builds(
    lambda b, a, v: {"delta_bid_sizes": b, "delta_ask_sizes": a, "total_volume_5min": v},
    _deltas, _deltas, _volume,
)
_l1 = st.builds(
    lambda bc, ac, bs, a_s: {"bid_size_change": bc, "ask_size_change": ac,
                             "bid_size": bs, "ask_size": a_s},
    _finite, _finite, _nonneg, _nonneg,
)


# ── 1. OFI ───────────────────────────────────────────────────────────────────────

class TestOFIProperties:

    @settings(max_examples=300, deadline=None)
    @given(d=st.one_of(_l2, _l1))
    def test_bounded_and_finite(self, d):
        out = compute_ofi(d)
        assert isinstance(out, float)
        assert not math.isnan(out)
        assert -1.0 <= out <= 1.0

    @settings(max_examples=200, deadline=None)
    @given(b=_deltas, a=_deltas, v=_volume)
    def test_l2_antisymmetric(self, b, a, v):
        o1 = compute_ofi({"delta_bid_sizes": b, "delta_ask_sizes": a, "total_volume_5min": v})
        o2 = compute_ofi({"delta_bid_sizes": a, "delta_ask_sizes": b, "total_volume_5min": v})
        assert math.isclose(o1, -o2, abs_tol=1e-9)

    @settings(max_examples=200, deadline=None)
    @given(b=_deltas, a=_deltas, v=_volume, bump=st.floats(min_value=0.0, max_value=1e9, **_F))
    def test_l2_monotone_in_bid(self, b, a, v, bump):
        base = compute_ofi({"delta_bid_sizes": b, "delta_ask_sizes": a, "total_volume_5min": v})
        b2 = [b[0] + bump, *b[1:]]
        more = compute_ofi({"delta_bid_sizes": b2, "delta_ask_sizes": a, "total_volume_5min": v})
        assert more >= base - 1e-9

    @settings(max_examples=150, deadline=None)
    @given(b=_deltas, v=_volume)
    def test_l2_symmetric_is_zero(self, b, v):
        assert compute_ofi({"delta_bid_sizes": b, "delta_ask_sizes": b, "total_volume_5min": v}) == 0.0

    def test_one_sided_saturates(self):
        big, zero = [1e9] * 5, [0.0] * 5
        assert compute_ofi({"delta_bid_sizes": big, "delta_ask_sizes": zero, "total_volume_5min": 1.0}) == 1.0
        assert compute_ofi({"delta_bid_sizes": zero, "delta_ask_sizes": big, "total_volume_5min": 1.0}) == -1.0

    @given(x=st.one_of(st.none(), st.integers(), st.text(), st.just({})))
    def test_unavailable_is_zero(self, x):
        assert compute_ofi(x) == 0.0

    @settings(max_examples=300, deadline=None)
    @given(direction=st.sampled_from(["BUY", "SELL", "HOLD", "FLAT", ""]),
           ofi=st.floats(allow_nan=True, allow_infinity=True))
    def test_gate_returns_bool_neutral_passes(self, direction, ofi):
        result = ofi_filter_gate(direction, ofi)
        assert isinstance(result, bool)
        if direction not in ("BUY", "SELL"):
            assert result is True

    @settings(max_examples=300, deadline=None)
    @given(ofi=st.floats(**_F))
    def test_gate_threshold_semantics(self, ofi):
        t = _OFI_REJECT_THRESHOLD
        assert ofi_filter_gate("BUY", ofi) == (not ofi < -t)
        assert ofi_filter_gate("SELL", ofi) == (not ofi > t)

    @settings(max_examples=150, deadline=None)
    @given(bid=st.lists(st.one_of(st.just(float("nan")), _finite), min_size=1, max_size=5),
           ask=_deltas, v=_volume)
    def test_nan_deltas_are_neutral_not_nan(self, bid, ask, v):
        # A bad tick (NaN in an L2 delta) must NOT produce a NaN OFI — it stays
        # bounded/neutral (the module contract: "never an exception", "[-1, 1]").
        out = compute_ofi({"delta_bid_sizes": bid, "delta_ask_sizes": ask, "total_volume_5min": v})
        assert not math.isnan(out)
        assert -1.0 <= out <= 1.0


# ── 2. vol_ratio_current ─────────────────────────────────────────────────────────

class TestVolRatioProperties:

    @settings(max_examples=120, deadline=None)
    @given(arrays(np.float64, st.integers(0, 160), elements=st.floats(-0.5, 0.5, **_F)))
    def test_finite_and_nonneg(self, r):
        v = vol_ratio_current(r)
        assert math.isfinite(v)
        assert v >= 0.0

    @given(arrays(np.float64, st.integers(0, 4), elements=st.floats(-0.5, 0.5, **_F)))
    def test_insufficient_history_is_unit(self, r):
        assert vol_ratio_current(r) == 1.0

    @given(v=st.floats(-0.5, 0.5, **_F), n=st.integers(5, 200))
    def test_constant_input_is_unit(self, v, n):
        assert vol_ratio_current(np.full(n, v, dtype=np.float64)) == 1.0

    @settings(max_examples=150, deadline=None)
    @given(arrays(np.float64, st.integers(5, 29), elements=st.floats(-0.5, 0.5, **_F)))
    def test_rolling_std_regime_is_unit(self, r):
        assert vol_ratio_current(r) == pytest.approx(1.0, abs=1e-9)

    @settings(max_examples=100, deadline=None)
    @given(r=arrays(np.float64, st.integers(5, 29), elements=st.floats(-0.4, 0.4, **_F)),
           c=st.floats(0.1, 100.0, **_F))
    def test_scale_invariant_small_n(self, r, c):
        # Exact only in the fallback regime (n<30); the arch MLE makes it merely
        # approximate for n>=30, so that is not asserted here.
        assert vol_ratio_current(r) == pytest.approx(vol_ratio_current(r * c), abs=1e-9)

    def test_constant_magnitude_long_history_does_not_crash(self):
        # Constant-magnitude returns make the HAR-RV realised-variance regressors
        # collinear; the kernel must degrade to 1.0, not raise (it previously
        # raised ValueError: not enough values to unpack).
        v = vol_ratio_current([0.01, -0.01] * 40)
        assert math.isfinite(v) and v >= 0.0


# ── 3. CVaR ──────────────────────────────────────────────────────────────────────

@st.composite
def _returns_long_only(draw, min_t, max_t, lo=-0.05, hi=0.05):
    n = draw(st.integers(1, 5))
    t = draw(st.integers(min_t, max_t))
    elems = st.floats(lo, hi, **_F)
    r = np.array(draw(st.lists(elems, min_size=t * n, max_size=t * n)), dtype=float).reshape(t, n)
    raw = np.array(draw(st.lists(_nonneg, min_size=n, max_size=n)), dtype=float)
    w = raw + 1e-6
    return r, w / w.sum()


class TestCVaRProperties:

    @settings(max_examples=200, deadline=None)
    @given(_returns_long_only(min_t=2, max_t=29))
    def test_gaussian_path_nonneg(self, rw):
        r, w = rw
        assert portfolio_cvar(w, r) >= -1e-12

    @settings(max_examples=200, deadline=None)
    @given(_returns_long_only(min_t=0, max_t=200))
    def test_gaussian_cvar_nonneg_any_T(self, rw):
        r, w = rw
        assert _gaussian_cvar(w, r) >= 0.0

    @settings(max_examples=150, deadline=None)
    @given(_returns_long_only(min_t=2, max_t=120), st.floats(0.1, 10.0, **_F))
    def test_positive_homogeneity(self, rw, s):
        # CVaR is positively homogeneous of degree 1 in gross exposure (both the
        # LP and Gaussian paths). NOTE: LP-path CVaR is a SIGNED tail expectation,
        # so "CVaR >= 0" is NOT asserted on the LP path (negative for gain tails).
        # s is bounded away from 0: for sub-1e-6 scales both sides fall below the
        # HiGHS LP solver's absolute resolution (~1e-9) — the invariant holds but is
        # numerically unresolvable there. The s=0 edge is exact (covered below).
        r, w = rw
        assert np.isclose(portfolio_cvar(s * w, r), s * portfolio_cvar(w, r), rtol=1e-5, atol=1e-8)

    def test_zero_scale_cvar_is_zero(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.01, (60, 3))
        w = np.full(3, 1.0 / 3.0)
        assert portfolio_cvar(0.0 * w, r) == pytest.approx(0.0, abs=1e-9)

    @settings(max_examples=200, deadline=None)
    @given(_returns_long_only(min_t=60, max_t=300))
    def test_cf_floored_at_gaussian(self, rw):
        # Item-8 floor — holds on the cornish_fisher branch only (the historical
        # fallback for |skew|>3 or kurtosis>20 applies no floor).
        r, w = rw
        port = (r @ w)[-252:]
        assume(port.size > 3)
        assume(abs(float(skew(port))) <= 3.0 and float(kurtosis(port, fisher=True)) <= 20.0)
        out = compute_portfolio_cvar_cf(w, r)
        assume(out["method"] == "cornish_fisher")
        mu_p, sig_p = float(np.mean(port)), float(np.std(port, ddof=1))
        z = float(norm.ppf(0.95))
        cvar_gauss = -mu_p + sig_p * float(norm.pdf(z)) / 0.05
        assert out["cvar"] >= cvar_gauss - 1e-9 * max(1.0, abs(cvar_gauss))


# ── 4. TCA cost ──────────────────────────────────────────────────────────────────

_SPREAD = st.floats(0.0, 100.0, **_F)
_VOL = st.floats(0.0, 5.0, **_F)
_ADV = st.floats(1.0, 1e9, **_F)
_PART = st.floats(0.0, 50.0, **_F)
_FEE = st.floats(0.0, 5.0, **_F)


class TestTCAProperties:

    def setup_method(self):
        reset_tca_model()                  # positive default coefficients; no prior updates

    def teardown_method(self):
        reset_tca_model()

    @settings(max_examples=300, deadline=None)
    @given(_SPREAD, _VOL, _ADV, _PART, _FEE)
    def test_cost_non_negative_and_finite(self, spread, vol, adv, part, fee):
        c = ex_ante_cost_model("X", 1.0, "BUY", spread, vol, adv, part, fee_bps=fee)
        assert math.isfinite(c)
        assert c >= 0.0

    @settings(max_examples=300, deadline=None)
    @given(_SPREAD, _VOL, _ADV, _PART, _FEE, st.floats(0.0, 50.0, **_F))
    def test_cost_monotone_in_participation(self, spread, vol, adv, part, fee, delta):
        lo = ex_ante_cost_model("X", 1.0, "BUY", spread, vol, adv, part, fee_bps=fee)
        hi = ex_ante_cost_model("X", 1.0, "BUY", spread, vol, adv, part + delta, fee_bps=fee)
        assert hi >= lo - 1e-9

    @settings(max_examples=200, deadline=None)
    @given(_SPREAD, _VOL, _ADV, _PART, _FEE)
    def test_zero_participation_is_floor(self, spread, vol, adv, part, fee):
        floor = ex_ante_cost_model("X", 1.0, "BUY", spread, vol, adv, 0.0, fee_bps=fee)
        assert floor == pytest.approx(spread / 2.0 + fee, abs=1e-9)
        assert ex_ante_cost_model("X", 1.0, "BUY", spread, vol, adv, part, fee_bps=fee) >= floor - 1e-9
