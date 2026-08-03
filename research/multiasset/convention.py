"""Return-convention repair for the long-history multi-asset panel.

Pre-registered in ``research/multiasset/convention_repair_prereg.md``. Read that first;
every source, window, bound and tolerance here is fixed there.

The panel this repairs treats fifteen of its eighteen instruments as though they were
already excess returns. They are not: seven are equity index levels, four are FX spot,
four are unadjusted front-month futures. This module turns the ones that can be
corrected into genuine **USD-funded excess returns**, and labels every cell it could not
correct rather than quietly leaving it looking like the ones it could.

Pure functions only -- no network, no file IO, no globals. The fetch lives in
``scripts/build_convention_inputs.py`` and the single registered run in
``scripts/run_convention_repair.py``.

The four traps this module exists to avoid
==========================================
1. **A price return is not an excess return.** ``price - risk_free`` understates the
   holder's return by the dividend yield. Correcting means adding a *measured* dividend
   yield back, not assuming one.
2. **A measured correction and an assumed one must never look alike.** Every corrected
   cell carries a ``Provenance`` label, and the fraction of the book that rests on
   measurement rather than assumption is reported as a headline, not a footnote.
3. **A bracket that can cross is not a bracket.** ``assert_bracket_ordering`` checks
   ``conservative <= central <= realistic`` elementwise over every cell, so a bound that
   overtakes another fails the build instead of producing a quietly wrong range.
4. **A repair that improves everything is measuring its own wishes.** The three rates
   series are the block the panel already converts correctly; ``correct_panel`` must
   return them byte-identical, and ``rates_block_unchanged`` is the assertion that says
   so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from research.multiasset.carry import realised_dividend_yield

__all__ = [
    "BIAS_BUDGET_DEFAULT",
    "BRACKET_BOUNDS",
    "EQUITY_CORRECTIONS",
    "EquityCorrection",
    "Provenance",
    "assert_bracket_ordering",
    "bracket_dividend_yields",
    "correct_equity",
    "correct_panel",
    "local_total_return",
    "measured_dividend_yield",
    "measured_fraction",
    "provenance_frame",
    "rates_block_unchanged",
]

MONTHS_PER_YEAR = 12
YIELD_WINDOW = 12

#: The three bracket bounds, ordered from harshest to kindest (prereg 3b).
BRACKET_BOUNDS: tuple[str, str, str] = ("conservative", "central", "realistic")

#: Fee + dividend-withholding + index-composition drag carried by a country
#: total-return ETF, as an annual decimal. The registered run MEASURES this against the
#: US pair, where the true answer is independently known from the French library
#: (prereg 4E); this constant is only the fallback for offline use and tests.
BIAS_BUDGET_DEFAULT = 0.008


class Provenance(str, Enum):
    """What a corrected cell actually rests on. Never inferred at read time."""

    #: The correction came from a total-return source overlapping this month.
    MEASURED = "MEASURED"
    #: No source overlaps this month; the correction is the registered bracket.
    BRACKETED = "BRACKETED"
    #: No correction is due -- the series is already a total return (DAX) and the
    #: charge is the full risk-free rate with no dividend credit.
    EXEMPT = "EXEMPT"
    #: The series was already an excess return and was not touched (the rates block).
    ALREADY_EXCESS = "ALREADY_EXCESS"
    #: A known convention error this repair cannot fix from free data (commodity roll).
    UNCORRECTED = "UNCORRECTED"


@dataclass(frozen=True)
class EquityCorrection:
    """How one equity instrument's dividend yield is obtained.

    ``reference`` is the key of the total-return series in the reference panel;
    ``fx_leg`` is the panel FX key needed to put a USD-quoted reference back into the
    index's own currency, or ``None`` when both are already in the same currency.
    ``total_return_index`` marks an index that ALREADY contains its dividends, which
    therefore gets no dividend credit at all.
    """

    key: str
    reference: str | None
    fx_leg: str | None = None
    total_return_index: bool = False
    price_partner: str | None = None
    #: True when the reference tracks the SAME index as the price series (so the pair
    #: has zero composition risk and the measured gap is a clean dividend yield). False
    #: for a country ETF measured against a local headline index -- MSCI Japan against
    #: the Nikkei 225 is not the same basket, and the drift between them can be larger
    #: than the dividend being measured.
    index_matched: bool = False
    note: str = ""


#: Fixed by prereg 1 and 3. DAX is the one instrument corrected by definition rather
#: than by measurement, and prereg control D tests that definition against data.
EQUITY_CORRECTIONS: tuple[EquityCorrection, ...] = (
    EquityCorrection(
        "SPX", "FRENCH_US", None, False, None, False,
        "CRSP value-weighted US total return minus the S&P 500 price return. The only "
        "equity instrument measured over the WHOLE sample."),
    EquityCorrection(
        "NASDAQ", "QQQ", None, False, "NDX", True,
        "Nasdaq-100 TR ETF against the Nasdaq-100 PRICE index. Paired index-to-index so "
        "the yield is clean; the Composite-vs-100 composition gap is carried as bias."),
    EquityCorrection(
        "FTSE100", "EWU", "GBPUSD", False, None, False,
        "MSCI UK TR ETF, USD, de-dollarised."),
    EquityCorrection(
        "N225", "EWJ", "JPYUSD", False, None, False,
        "MSCI Japan TR ETF, USD, de-dollarised. 31 of its 61 years predate the ETF -- "
        "the largest single BRACKETED quantity in the repair."),
    EquityCorrection(
        "DAX", None, None, True, None, False,
        "DAX Performance-Index: dividends are already inside it, so the dividend credit "
        "is ZERO and the charge is the full bill."),
    EquityCorrection(
        "HSI", "EWH", None, False, None, False,
        "MSCI Hong Kong TR ETF. HKD is pegged to USD so no FX leg is applied; the peg "
        "band is a disclosed residual."),
    EquityCorrection(
        "ASX200", "EWA", "AUDUSD", False, None, False,
        "MSCI Australia TR ETF, USD."),
)


# -- measurement ---------------------------------------------------------------

def local_total_return(
    reference_usd: pd.Series,
    fx_return: pd.Series | None,
) -> pd.Series:
    """Put a USD-quoted total return back into the index's own currency.

    ``(1 + usd) = (1 + local) * (1 + fx)`` where ``fx`` is the return of a long-local
    position measured in USD, so ``local = (1 + usd) / (1 + fx) - 1``. With no FX leg
    the two are already in the same currency and the series passes through.
    """
    usd = pd.Series(reference_usd).astype(float)
    if fx_return is None:
        return usd
    fx = pd.Series(fx_return).astype(float).reindex(usd.index)
    return (1.0 + usd) / (1.0 + fx) - 1.0


def measured_dividend_yield(
    reference_usd: pd.Series,
    price_return: pd.Series,
    fx_return: pd.Series | None = None,
    *,
    window: int = YIELD_WINDOW,
) -> pd.Series:
    """Trailing realised dividend yield implied by a total-return / price pair.

    Backward-looking by construction, so it is point-in-time safe. Returns an ANNUAL
    decimal on the index of ``price_return``; months with no overlapping reference are
    NaN and are exactly the months the bracket has to cover.
    """
    local = local_total_return(reference_usd, fx_return)
    price = pd.Series(price_return).astype(float)
    q = realised_dividend_yield(local, price, window=window)
    return q.reindex(price.index)


# -- the bracket (prereg 3b, as amended 2026-07-31 before any run) --------------

def bracket_dividend_yields(
    measured: pd.Series,
    us_era_path: pd.Series,
    *,
    bias_budget: float = BIAS_BUDGET_DEFAULT,
) -> dict[str, pd.Series]:
    """Extend a partially-measured dividend yield over the whole sample, three ways.

    ``us_era_path`` is the US dividend yield measured over the FULL sample from the
    French library; it is the only thing in this repair that knows what era yields
    looked like before the country ETFs existed, and all three bounds inherit their
    time variation from it.

    * ``conservative`` -- the measurement where it exists, and **zero** before it. The
      instrument pays the full bill with no dividend credit. A pass here is REAL.
    * ``central`` -- the measurement where it exists; before it, the country's mean
      measured yield scaled by the US era path. The reported headline.
    * ``realistic`` -- the measurement plus the ETF's fee/withholding/composition bias
      where it exists; before it, the same gross-up scaled by the US era path. A fail
      here is DEAD.

    All three share the era path, which is what keeps them from crossing: an era in
    which US yields ran 1.8x their modern level lifts ``central`` and ``realistic``
    together instead of lifting only one of them.
    """
    q = pd.Series(measured).astype(float)
    us = pd.Series(us_era_path).astype(float).reindex(q.index)
    have = q.notna()

    if not bool(have.any()):
        zero = pd.Series(0.0, index=q.index)
        return {"conservative": zero.copy(), "central": zero.copy(),
                "realistic": zero.copy()}

    mean_q = float(q[have].mean())
    mean_us = float(us[have].mean()) if bool(us[have].notna().any()) else float("nan")
    if np.isfinite(mean_us) and mean_us > 0:
        ratio = (us / mean_us).clip(lower=0.0)
    else:
        ratio = pd.Series(1.0, index=q.index)
    ratio = ratio.fillna(1.0)

    conservative = q.where(have, 0.0)
    central = q.where(have, mean_q * ratio)
    realistic = (q + bias_budget).where(have, (mean_q + bias_budget) * ratio)
    return {"conservative": conservative.astype(float),
            "central": central.astype(float),
            "realistic": realistic.astype(float)}


def assert_bracket_ordering(bounds: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Verify ``conservative <= central <= realistic`` elementwise, over every cell.

    Method rule 10 exists because a bracket that can cross is not a bracket. This is the
    assertion, not the argument: it returns the number of cells checked and raises on the
    first violation with the offending location, so a bound that overtakes another fails
    the build instead of producing a quietly wrong range.
    """
    missing = [b for b in BRACKET_BOUNDS if b not in bounds]
    if missing:
        raise KeyError(f"bracket is missing bounds {missing}")
    low, mid, high = (bounds[b] for b in BRACKET_BOUNDS)
    if not (low.shape == mid.shape == high.shape):
        raise ValueError(f"bracket bounds disagree in shape: "
                         f"{low.shape} / {mid.shape} / {high.shape}")

    checked = 0
    for lower, upper, names in ((low, mid, "conservative <= central"),
                                (mid, high, "central <= realistic")):
        both = lower.notna() & upper.notna()
        checked += int(both.to_numpy().sum())
        bad = both & (lower > upper + 1e-12)
        if bool(bad.to_numpy().any()):
            where = [(str(idx.date()), col)
                     for col in bad.columns for idx in bad.index[bad[col]]][:5]
            raise AssertionError(
                f"bracket ordering violated ({names}) in {int(bad.to_numpy().sum())} "
                f"cells; first: {where}")
    return {"pairs_checked": checked, "ordered": True}


# -- applying the correction ---------------------------------------------------

def correct_equity(
    price_return: pd.Series,
    risk_free: pd.Series,
    dividend_yield: pd.Series | float,
) -> pd.Series:
    """``price_return - risk_free + dividend_yield/12`` -- a USD-funded excess return.

    ``dividend_yield`` is an ANNUAL decimal (a series or a constant); it is divided by
    twelve here so no caller has to remember to. A total-return index passes 0.0 and so
    pays the full bill, which is the whole point of the DAX case.
    """
    price = pd.Series(price_return).astype(float)
    rf = pd.Series(risk_free).astype(float).reindex(price.index)
    if isinstance(dividend_yield, (int, float)):
        q = pd.Series(float(dividend_yield), index=price.index)
    else:
        q = pd.Series(dividend_yield).astype(float).reindex(price.index).fillna(0.0)
    return price - rf + q / MONTHS_PER_YEAR


def correct_panel(
    panel: pd.DataFrame,
    risk_free: pd.Series,
    dividend_yields: dict[str, pd.Series],
    fx_excess: pd.DataFrame | None = None,
    *,
    already_excess: tuple[str, ...] = (),
    uncorrected: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply the whole correction to a panel, leaving untouched what must stay untouched.

    ``already_excess`` names the columns that were converted correctly in the first place
    (the rates block) -- they are copied through **byte-identical**, and
    ``rates_block_unchanged`` is the check that proves it. ``uncorrected`` names columns
    with a known convention error this repair cannot fix (the commodity roll); they are
    also copied through, but labelled differently, because "we checked and it is fine" and
    "we could not check" must never look the same in an output.
    """
    out = panel.copy()
    for key, q in dividend_yields.items():
        if key in out.columns:
            out[key] = correct_equity(panel[key], risk_free, q)
    if fx_excess is not None:
        for key in fx_excess.columns:
            if key in out.columns:
                out[key] = fx_excess[key].reindex(out.index)
    for key in (*already_excess, *uncorrected):
        if key in out.columns:
            out[key] = panel[key]
    return out


def provenance_frame(
    panel: pd.DataFrame,
    measured: dict[str, pd.Series],
    *,
    exempt: tuple[str, ...] = (),
    already_excess: tuple[str, ...] = (),
    uncorrected: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One label per instrument-month saying what that corrected cell rests on."""
    out = pd.DataFrame(Provenance.BRACKETED.value, index=panel.index,
                       columns=panel.columns, dtype=object)
    for key, q in measured.items():
        if key in out.columns:
            have = pd.Series(q).reindex(panel.index).notna()
            out.loc[have, key] = Provenance.MEASURED.value
    for group, label in ((exempt, Provenance.EXEMPT),
                         (already_excess, Provenance.ALREADY_EXCESS),
                         (uncorrected, Provenance.UNCORRECTED)):
        for key in group:
            if key in out.columns:
                out[key] = label.value
    return out


def measured_fraction(provenance: pd.DataFrame, live: pd.DataFrame) -> dict[str, float]:
    """Share of the LIVE panel that rests on measurement rather than assumption.

    ``live`` marks the cells an instrument actually contributes (it is unbalanced -- an
    instrument enters when it becomes eligible), so a series that did not exist yet
    cannot dilute the number in either direction.
    """
    mask = live.notna()
    total = int(mask.to_numpy().sum())
    if total == 0:
        return {"n_live_cells": 0}
    counts: dict[str, float] = {"n_live_cells": total}
    for label in Provenance:
        n = int(((provenance == label.value) & mask).to_numpy().sum())
        counts[f"frac_{label.value.lower()}"] = round(n / total, 6)
        counts[f"n_{label.value.lower()}"] = n
    return counts


def rates_block_unchanged(
    old_panel: pd.DataFrame,
    new_panel: pd.DataFrame,
    keys: tuple[str, ...],
) -> dict[str, float]:
    """The anti-rigging control: the rates block must be byte-identical (prereg 4B).

    A repair that improves every block is a repair that is measuring its own wishes.
    Returns the worst absolute difference found, which must be exactly zero.
    """
    worst = 0.0
    cells = 0
    for key in keys:
        if key not in old_panel.columns or key not in new_panel.columns:
            raise KeyError(f"rates key {key!r} is absent from one of the panels")
        a, b = old_panel[key], new_panel[key]
        if not a.index.equals(b.index):
            raise ValueError(f"{key}: panels are not on the same index")
        if not a.isna().equals(b.isna()):
            raise AssertionError(f"{key}: null pattern changed")
        both = a.notna() & b.notna()
        cells += int(both.sum())
        if bool(both.any()):
            worst = max(worst, float((a[both] - b[both]).abs().max()))
    if worst != 0.0:
        raise AssertionError(
            f"rates block moved by {worst:.3e}; it must be byte-identical (prereg 4B)")
    return {"keys": len(keys), "cells_compared": cells, "max_abs_diff": worst}
