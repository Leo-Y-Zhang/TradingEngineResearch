"""THE BOOK SCALER — and the 12 months every sleeve ran at FULL leverage.

THE DEFECT
==========
The volatility-targeted sleeves size the book by taking the smaller of two scalers::

    k_raw = vol_target / trailing_book_vol      # what the vol target asks for
    k_cap = GROSS_CAP  / gross_unit             # what the leverage cap allows
    k     = pd.concat([k_raw, k_cap], axis=1).min(axis=1)

`DataFrame.min(axis=1)` defaults to ``skipna=True``. The trailing book vol is a rolling
window with ``min_periods = BOOK_VOL_MIN = 12``, so for a book's first **12 months**
`k_raw` is NaN, `min` silently drops it, and `k` becomes `k_cap` alone — the book runs at
exactly ``GROSS_CAP`` (10x) with no volatility estimate behind it. There is no
``if not np.isfinite(sigma)`` branch anywhere; the fall-through is implicit.

Worth **0.050 of Sharpe** where it was measured.

AND THE DIAGNOSTIC HID IT
=========================
The standard cap-binding count is::

    cap_binding = (k_raw > k_cap) & k_raw.notna() & k_cap.notna()

`k_raw.notna()` **excludes exactly the months in which the cap is the only thing setting
leverage**. The diagnostic reports 0% binding for the 12 months that are 100% cap-driven.

THE TWO BEHAVIOURS
==================
``NO_ESTIMATE_CAP``   the registered behaviour: fall through to the gross cap. Kept as
                      the DEFAULT so every banked number reproduces bit-for-bit.
``NO_ESTIMATE_FLAT``  the repair: no volatility estimate means no position. This is
                      what `riskparity.levered` does by accident (``Series.clip``
                      propagates NaN where ``DataFrame.min`` skips it) and what
                      `tsmom_multitimeframe` does on purpose with an explicit
                      ``isfinite`` branch that goes flat. Both were immune.

`no_estimate` is returned as its own mask under BOTH settings, so the count can never be
hidden again.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "NO_ESTIMATE_CAP",
    "NO_ESTIMATE_FLAT",
    "NO_ESTIMATE_POLICIES",
    "REGISTERED_NO_ESTIMATE",
    "BookScaler",
    "book_scaler",
]

#: registered: with no volatility estimate the gross cap alone sets the scale (FULL 10x)
NO_ESTIMATE_CAP = "cap"
#: the repair: with no volatility estimate the book is flat
NO_ESTIMATE_FLAT = "flat"

NO_ESTIMATE_POLICIES = (NO_ESTIMATE_CAP, NO_ESTIMATE_FLAT)
REGISTERED_NO_ESTIMATE = NO_ESTIMATE_CAP


@dataclass(frozen=True)
class BookScaler:
    """The scaler, plus the two masks that say WHY it took the value it did."""

    #: the applied scaler. NaN under NO_ESTIMATE_FLAT wherever the vol estimate is missing
    k: pd.Series
    #: months where the vol target asked for MORE leverage than the cap allows
    cap_binding: pd.Series
    #: months with NO volatility estimate at all — the 12-month hole
    no_estimate: pd.Series
    policy: str

    @property
    def cap_or_no_estimate(self) -> pd.Series:
        """Every month whose leverage was set by the cap, for whichever reason."""
        return (self.cap_binding | self.no_estimate).astype(bool)


def book_scaler(k_raw: pd.Series, k_cap: pd.Series, *,
                no_estimate: str = REGISTERED_NO_ESTIMATE,
                live: pd.Series | None = None) -> BookScaler:
    """Combine the vol-target and gross-cap scalers, declaring the no-estimate policy.

    ``live`` optionally restricts both masks to months in which the book actually holds
    something, which is what stops an all-flat month being counted as cap-driven.
    """
    if no_estimate not in NO_ESTIMATE_POLICIES:
        raise ValueError(f"no_estimate must be one of {NO_ESTIMATE_POLICIES}")

    k = pd.concat([k_raw, k_cap], axis=1).min(axis=1)
    on = pd.Series(True, index=k.index) if live is None else live.reindex(k.index).fillna(False)
    on = on.astype(bool)

    missing = (k_raw.isna() & k_cap.notna() & on).astype(bool)
    binding = ((k_raw > k_cap) & k_raw.notna() & k_cap.notna() & on).astype(bool)

    if no_estimate == NO_ESTIMATE_FLAT:
        # NaN, not 0.0: every caller multiplies notionals by `k` and then fills the
        # result, so NaN propagates to a flat book exactly as `riskparity` does.
        k = k.where(~missing)

    return BookScaler(k=k, cap_binding=binding, no_estimate=missing, policy=no_estimate)
