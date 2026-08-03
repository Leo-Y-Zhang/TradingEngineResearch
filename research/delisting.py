"""THE DELISTING WINDOW — and the off-by-one that made it almost never fire.

THE DEFECT
==========
A terminal (delisting) return must be booked only if the delisting is what CLOSED the
position — asking merely "did this ticker ever delist?" charges a 2012 bankruptcy against
a 2003 exit. Every sleeve therefore gates the terminal return on a date window running
from the position's exit bar.

Every one of those windows was written with a STRICT lower edge::

    at < delisted_on <= at + 62 days          # rejects delisted_on == at

but Sharadar's ACTIONS table dates a delisting **ON the ticker's last traded SEP bar** —
median gap **0 days**. The strict `<` therefore rejects the MODAL case. Measured on the
low-vol band B2..B5 universe: the window fired **39 times out of 3,018**, and **6,322**
last-observation cells carry a delisting record whose median terminal return is **-1.00**
that was never booked. `delisting_drag_annual = 0.0` was a dead code path, not a finding.

The whole repair is one character — `<` becomes `<=` — expressed here as a day offset so
it can be switched rather than silently changed::

    REGISTERED_WINDOW = (1, 62)     # the defect; reproduces every banked number
    CORRECTED_WINDOW  = (0, 62)     # a delisting dated ON the last bar counts

WHY IT IS A PARAMETER AND NOT A FIX
===================================
Every banked sleeve result was produced under `REGISTERED_WINDOW`, so flipping the
default would rewrite published numbers with no record of the before. Each call site
takes the window as an argument defaulting to `REGISTERED_WINDOW`; the corrected run is
declared, measured and reported separately. `tests/test_delisting_window.py` proves the
two differ exactly on the gap-0 case and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "CORRECTED_WINDOW",
    "DELISTING_WINDOW_DAYS",
    "REGISTERED_WINDOW",
    "in_window",
    "in_window_mask",
    "window_for",
]

#: the registered holding-period grace: a position closed at `at` is exposed to a
#: delisting for the following 62 calendar days (~2 monthly rebalances)
DELISTING_WINDOW_DAYS = 62

#: (lower_offset_days, upper_offset_days) around the exit bar, both INCLUSIVE.
#: `(1, 62)` reproduces `at < delisted_on <= at + 62 days` exactly, because these dates
#: carry day resolution.
REGISTERED_WINDOW: tuple[int, int] = (1, DELISTING_WINDOW_DAYS)
#: the repair: a delisting dated ON the last traded bar closed the position
CORRECTED_WINDOW: tuple[int, int] = (0, DELISTING_WINDOW_DAYS)


def window_for(corrected: bool = False,
               window_days: int = DELISTING_WINDOW_DAYS) -> tuple[int, int]:
    """The (lower, upper) day offsets for the registered or the corrected window."""
    return (0 if corrected else 1, int(window_days))


def in_window(exit_date: pd.Timestamp, delist_date: pd.Timestamp | None,
              window: tuple[int, int] = REGISTERED_WINDOW) -> bool:
    """Did ``delist_date`` fall inside the exposure window around ``exit_date``?"""
    if delist_date is None or pd.isna(delist_date):
        return False
    low, high = window
    at = pd.Timestamp(exit_date)
    on = pd.Timestamp(delist_date)
    return bool(at + pd.Timedelta(days=low) <= on <= at + pd.Timedelta(days=high))


def in_window_mask(exit_dates: pd.Series, delist_dates: pd.Series,
                   window: tuple[int, int] = REGISTERED_WINDOW) -> pd.Series:
    """Vectorised `in_window`. NaT on either side is False, never NaN."""
    low, high = window
    at = pd.to_datetime(exit_dates, errors="coerce")
    on = pd.to_datetime(np.asarray(delist_dates), errors="coerce")
    on = pd.Series(on, index=at.index)
    mask = (on.notna()
            & at.notna()
            & (on >= at + pd.Timedelta(days=low))
            & (on <= at + pd.Timedelta(days=high)))
    return mask.fillna(False).astype(bool)
