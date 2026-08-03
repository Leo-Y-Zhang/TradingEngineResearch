"""Alpha-slice: tests for the learned, DSR/PBO-gated cross-sectional signal combiner
(``research.alpha_factory.learn_signal_weights``).

The learner must (1) recover a real cross-sectional edge and pass ``selection_rule``, and
(2) DEFAULT-DENY noise — a combination that does not survive deflation must fail the gate
so it can never replace the incumbent equal-weight blend (golden rule 5 / SIGNALS-5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.alpha_factory import learn_signal_weights
from research.validation import selection_rule

SYMBOLS = [f"S{i}" for i in range(12)]


def _panel(rng, n_dates: int = 240) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n_dates)
    return pd.DataFrame(rng.standard_normal((n_dates, len(SYMBOLS))), index=idx, columns=SYMBOLS)


def test_recovers_real_edge_and_passes_gate():
    rng = np.random.default_rng(7)
    sig_a = _panel(rng)                       # the predictive signal
    sig_b = _panel(rng)                       # pure noise
    # Forward return is driven by signal A (clear cross-sectional edge) + modest noise.
    fwd = 0.03 * sig_a + 0.02 * _panel(rng)

    weights, result = learn_signal_weights({"alpha": sig_a, "noise": sig_b}, fwd, n_trials=8)

    assert weights["alpha"] > 0.0                       # learned a positive loading on A
    assert weights["alpha"] > abs(weights["noise"])     # and weights A above the noise signal
    assert result.mean_ic > 0.0
    assert result.deflated_sharpe_ratio > 0.95          # survives deflation
    assert selection_rule(result) is True               # deployable


def test_noise_is_default_denied():
    rng = np.random.default_rng(11)
    sig_a = _panel(rng)
    sig_b = _panel(rng)
    fwd = _panel(rng)                                    # independent of both signals → no edge

    _weights, result = learn_signal_weights({"a": sig_a, "b": sig_b}, fwd, n_trials=8)

    assert abs(result.mean_ic) < 0.05                   # ~no information
    assert result.deflated_sharpe_ratio < 0.95          # fails deflation
    assert selection_rule(result) is False              # NOT deployable (default-deny)


def test_degenerate_input_fails_closed():
    rng = np.random.default_rng(1)
    # empty panel
    w, r = learn_signal_weights({}, _panel(rng))
    assert w == {} and selection_rule(r) is False
    # too few dates
    short = pd.bdate_range("2020-01-01", periods=5)
    sig = pd.DataFrame(rng.standard_normal((5, len(SYMBOLS))), index=short, columns=SYMBOLS)
    w2, r2 = learn_signal_weights({"a": sig}, sig)
    assert w2 == {"a": 0.0} and selection_rule(r2) is False
