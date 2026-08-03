"""Registry for the BREADTH-EXPANSION panel — candidate INDEPENDENT bets.

Iteration 12 measured the binding constraint: the existing 18-instrument panel behaves
like **5.26 independent bets** (correlation-effective N), and since the growth ceiling is
``S^2/2`` with ``S = s * sqrt(N_eff)``, the only lever left that can move the ceiling is
more independent bets. This file lists the candidates.

It **does not modify** ``research/multiasset/instruments.py``. It reuses that module's
``Instrument`` dataclass and every return-convention primitive in
``research/multiasset/panel.py``, so the conventions are identical by construction rather
than by assertion.

Conventions carried over unchanged
==================================
* ``price_return`` on a futures front-month continuous series is already an EXCESS
  return (you post margin, not principal), exactly as ``GOLD_F``/``WTI_F`` are treated.
* ``price_return`` on an ``auto_adjust=True`` ETF is a TOTAL return of a FUNDED position.
  To be futures-equivalent it needs the bill rate subtracted — the same treatment the
  original panel applies to the three par-bond series. Every ETF key here is therefore
  listed in ``CASH_SUBTRACTED_NEW``.
* Non-USD listings return LOCAL-currency returns and are not FX-converted, exactly as the
  original panel treats ``^FTSE``/``^N225``/``^GDAXI``/``^HSI``/``^AXJO`` (§10.6 of
  ``data_integrity.md``).

What is deliberately NOT tradable
=================================
``^VIX`` is an index level, not an instrument: there is no spot VIX position, so a return
on the index is not earnable by anyone. It is fetched at ``role="validation"`` only, to
quantify how much of the tradable proxy's return is futures roll rather than volatility.
"""

from __future__ import annotations

from research.multiasset.instruments import Instrument

__all__ = [
    "BREADTH_INSTRUMENTS",
    "BREADTH_BLOCKS",
    "CASH_SUBTRACTED_NEW",
    "SYNTHETIC_SPREADS",
    "ROLL_VALIDATION_PAIRS",
    "TIERS",
    "breadth_panel_instruments",
    "breadth_validation_instruments",
    "breadth_tickers",
]


# ── Agriculture: front-month continuous futures, roll NOT back-adjusted ───────
# The single most plausible source of genuinely independent bets: weather and
# acreage are not macro factors. Every one of these is subjected to the same
# roll-contamination test that condemned NATGAS_F.
_AGRICULTURE = [
    Instrument("CORN_F", "ZC=F", "Corn front-month future", "agriculture", "USD",
               "price_return",
               notes="CBOT corn. Front-month continuous, roll gaps NOT back-adjusted. "
                     "Last trading day is the business day before the 15th of the "
                     "delivery month, so the roll window is mid-month, not month-end."),
    Instrument("WHEAT_F", "ZW=F", "Chicago wheat front-month future", "agriculture", "USD",
               "price_return",
               notes="CBOT SRW wheat. Same mid-month roll window as corn."),
    Instrument("SOYBEAN_F", "ZS=F", "Soybean front-month future", "agriculture", "USD",
               "price_return",
               notes="CBOT soybeans. Same mid-month roll window as corn."),
    Instrument("SUGAR_F", "SB=F", "Sugar No.11 front-month future", "agriculture", "USD",
               "price_return",
               notes="ICE sugar #11. Last trading day is the last business day of the "
                     "month PRECEDING delivery, so its roll window is month-end."),
    Instrument("COFFEE_F", "KC=F", "Coffee C front-month future", "agriculture", "USD",
               "price_return",
               notes="ICE arabica. Notice period starts ~8 business days before month end."),
    Instrument("COTTON_F", "CT=F", "Cotton No.2 front-month future", "agriculture", "USD",
               "price_return",
               notes="ICE cotton #2. Last trading day is 17 business days from month end."),
    Instrument("COCOA_F", "CC=F", "Cocoa front-month future", "agriculture", "USD",
               "price_return",
               notes="ICE cocoa. Roll window is mid-to-late month."),
]

# ── Livestock: the least financialised block available for free ───────────────
_LIVESTOCK = [
    Instrument("CATTLE_F", "LE=F", "Live cattle front-month future", "livestock", "USD",
               "price_return",
               notes="CME live cattle. Physically delivered; last trading day is the last "
                     "business day of the contract month. Roll window is month-end."),
    Instrument("HOGS_F", "HE=F", "Lean hogs front-month future", "livestock", "USD",
               "price_return",
               notes="CME lean hogs. CASH-SETTLED against the CME Lean Hog Index; last "
                     "trading day is the 10th business day of the contract month."),
]

# ── Volatility as an asset ────────────────────────────────────────────────────
# ^VIX is NOT tradable. VIXY (ProShares VIX Short-Term Futures ETF, 2011-01) is the
# longest continuously-listed free tradable proxy: VXX's Yahoo history only starts
# 2018-01 because the original iPath ETN matured and the Series B replaced it.
_VOLATILITY = [
    Instrument("VIX_ETF", "VIXY", "ProShares VIX Short-Term Futures ETF", "volatility",
               "USD", "price_return",
               notes="TRADABLE proxy for VIX exposure. Holds a constant-maturity roll of "
                     "the front two VIX futures, so in contango it BLEEDS. Its return is "
                     "roll plus volatility, not volatility. auto_adjust total return, so "
                     "the bill rate is subtracted to make it futures-equivalent."),
    Instrument("VIX_SPOT", "^VIX", "CBOE Volatility Index (NOT TRADABLE)", "volatility",
               "USD", "price_return", role="validation",
               notes="An index level, not an instrument — there is no spot VIX position. "
                     "Fetched ONLY to measure how much of VIXY's return is futures roll."),
]

# ── Credit ────────────────────────────────────────────────────────────────────
_CREDIT = [
    Instrument("HYG", "HYG", "iShares iBoxx High Yield Corporate Bond ETF", "credit",
               "USD", "price_return",
               notes="Total return, dividend-adjusted. Funded position: bill subtracted."),
    Instrument("LQD", "LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "credit",
               "USD", "price_return",
               notes="Total return, dividend-adjusted. Funded position: bill subtracted."),
]

# ── Non-US sovereigns, under DIFFERENT central banks ──────────────────────────
# The point of this block is monetary-policy independence: a Bund is priced by the ECB,
# a gilt by the BoE, a JGB by the BoJ. None of these is a US Treasury.
_FOREIGN_SOVEREIGN = [
    Instrument("GILT_ETF", "IGLT.L", "iShares Core UK Gilts UCITS ETF", "foreign_sovereign",
               "GBP", "price_return",
               notes="UK gilts, all maturities, LSE-listed, GBP. LOCAL-CURRENCY return — "
                     "a USD investor's return differs by GBPUSD, which the panel already "
                     "carries as a separate instrument. Bill subtracted (funded)."),
    Instrument("BUND_ETF", "EXX6.DE", "iShares eb.rexx Government Germany 5.5-10.5yr",
               "foreign_sovereign", "EUR", "price_return",
               notes="GERMAN federal government bonds, 5.5-10.5yr — the Bund proxy. "
                     "Xetra-listed, EUR, local-currency return. Bill subtracted."),
    Instrument("JGB_ETF", "1482.T", "iShares Core Japan Government Bond ETF",
               "foreign_sovereign", "JPY", "price_return",
               notes="JAPANESE government bonds, TSE-listed, JPY. Starts 2016-05, so it "
                     "is the shortest series in the panel and is reported separately. "
                     "Bill subtracted."),
]

# ── Real assets and other ─────────────────────────────────────────────────────
_REAL_ASSETS = [
    Instrument("REIT", "VNQ", "Vanguard Real Estate ETF", "real_assets", "USD",
               "price_return",
               notes="US equity REITs, total return. Bill subtracted (funded)."),
    Instrument("FREIGHT", "BDRY", "Breakwave Dry Bulk Shipping ETF", "real_assets", "USD",
               "price_return",
               notes="Dry bulk freight futures (Capesize/Panamax/Supramax). Starts "
                     "2018-03 — too short to carry weight, included to MEASURE that."),
    Instrument("CARBON", "KRBN", "KraneShares Global Carbon Strategy ETF", "real_assets",
               "USD", "price_return",
               notes="EU/UK/California carbon allowance futures. Starts 2020-07 — the "
                     "shortest series here; included to measure, not to rely on."),
]

# ── Validation-only: never strategy inputs ───────────────────────────────────
# Roll-managed ETFs for the agricultural futures, the same role GLD/SLV play for the
# metals. NIB is an iPath ETN that was DELISTED in 2023 — it is kept deliberately,
# because it is the panel's only direct evidence that this asset class delists.
_VALIDATION = [
    Instrument("CORN_ETF", "CORN", "Teucrium Corn Fund", "agriculture", "USD",
               "price_return", role="validation",
               notes="Roll-managed corn futures (2nd/3rd/Dec deferred). Benchmark for ZC=F."),
    Instrument("WHEAT_ETF", "WEAT", "Teucrium Wheat Fund", "agriculture", "USD",
               "price_return", role="validation", notes="Roll-managed. Benchmark for ZW=F."),
    Instrument("SOYB_ETF", "SOYB", "Teucrium Soybean Fund", "agriculture", "USD",
               "price_return", role="validation", notes="Roll-managed. Benchmark for ZS=F."),
    Instrument("CANE_ETF", "CANE", "Teucrium Sugar Fund", "agriculture", "USD",
               "price_return", role="validation", notes="Roll-managed. Benchmark for SB=F."),
    Instrument("COCOA_ETN", "NIB", "iPath Bloomberg Cocoa Subindex ETN", "agriculture",
               "USD", "price_return", role="validation",
               notes="DELISTED 2023-07. Benchmark for CC=F over 2009-2023, and the direct "
                     "evidence that instruments in this asset class do disappear."),
    Instrument("LIVESTOCK_ETN", "COW", "iPath Bloomberg Livestock Subindex ETN",
               "livestock", "USD", "price_return", role="validation",
               notes="DELISTED 2023-07; Yahoo carries the Series B from 2018. Roll-managed "
                     "cattle+hogs benchmark for LE=F and HE=F."),
    *(_VOLATILITY[1:]),
]

BREADTH_INSTRUMENTS: tuple[Instrument, ...] = tuple(
    _AGRICULTURE + _LIVESTOCK + _VOLATILITY[:1] + _CREDIT + _FOREIGN_SOVEREIGN
    + _REAL_ASSETS + _VALIDATION[:6] + _VOLATILITY[1:]
)

# Roll-MANAGED, actually-holdable substitutes for the front-month continuous series.
# A Teucrium fund holds deferred contracts and rolls on a published schedule, so it is
# the return a person could really have earned. Short and expense-bearing, which is why
# it is a SENSITIVITY, not the primary panel.
ROLL_MANAGED_SUBSTITUTE: dict[str, str] = {
    "CORN_F": "CORN_ETF",
    "WHEAT_F": "WHEAT_ETF",
    "SOYBEAN_F": "SOYB_ETF",
    "SUGAR_F": "CANE_ETF",
    "COCOA_F": "COCOA_ETN",
    "CATTLE_F": "LIVESTOCK_ETN",
    "HOGS_F": "LIVESTOCK_ETN",
}

# ── Blocks, for per-block effective-N attribution ─────────────────────────────
BREADTH_BLOCKS: dict[str, tuple[str, ...]] = {
    "agriculture": ("CORN_F", "WHEAT_F", "SOYBEAN_F", "SUGAR_F", "COFFEE_F",
                    "COTTON_F", "COCOA_F"),
    "livestock": ("CATTLE_F", "HOGS_F"),
    "volatility": ("VIX_ETF",),
    "credit": ("HYG", "LQD", "CREDIT_SPREAD"),
    "foreign_sovereign": ("GILT_ETF", "BUND_ETF", "JGB_ETF"),
    "real_assets": ("REIT", "FREIGHT", "CARBON"),
}

# Funded positions: an auto_adjust ETF total return minus the bill is the
# futures-equivalent excess return. Futures keys are NOT in this set.
CASH_SUBTRACTED_NEW: frozenset[str] = frozenset({
    "VIX_ETF", "HYG", "LQD", "GILT_ETF", "BUND_ETF", "JGB_ETF",
    "REIT", "FREIGHT", "CARBON",
})

# Long-short synthetic bets. ``a - b`` on TOTAL returns is self-financing, so the
# result is ALREADY an excess return and must not have the bill subtracted again.
# HYG-minus-IEF is credit risk with the Treasury duration hedged out — a distinct bet
# from either leg, which is exactly why it is carried separately.
SYNTHETIC_SPREADS: dict[str, tuple[str, str, str]] = {
    "CREDIT_SPREAD": ("HYG", "IEF", "high-yield minus 7-10y Treasury: credit risk with "
                                    "duration hedged out"),
}

# (futures key, roll-managed or spot benchmark key, what the comparison establishes)
ROLL_VALIDATION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("CORN_F", "CORN_ETF", "front-month corn vs roll-managed corn fund"),
    ("WHEAT_F", "WHEAT_ETF", "front-month wheat vs roll-managed wheat fund"),
    ("SOYBEAN_F", "SOYB_ETF", "front-month soybeans vs roll-managed soybean fund"),
    ("SUGAR_F", "CANE_ETF", "front-month sugar vs roll-managed sugar fund"),
    ("COCOA_F", "COCOA_ETN", "front-month cocoa vs cocoa subindex ETN (delisted 2023)"),
    ("CATTLE_F", "LIVESTOCK_ETN", "front-month live cattle vs roll-managed livestock ETN"),
    ("HOGS_F", "LIVESTOCK_ETN", "front-month lean hogs vs roll-managed livestock ETN"),
    ("VIX_ETF", "VIX_SPOT", "tradable VIX futures ETF vs the NON-TRADABLE spot index"),
)

# Tiers exist because breadth and sample length trade off directly (data_integrity.md
# §1). Each tier states the price in years of the breadth it buys.
TIERS: dict[str, tuple[str, ...]] = {
    "T1_long": ("CORN_F", "WHEAT_F", "SOYBEAN_F", "SUGAR_F", "COFFEE_F", "COTTON_F",
                "COCOA_F", "CATTLE_F", "HOGS_F", "LQD"),
    "T2_mid": ("HYG", "CREDIT_SPREAD", "REIT", "GILT_ETF", "BUND_ETF"),
    "T3_short": ("VIX_ETF",),
    "T4_verysh": ("JGB_ETF", "FREIGHT", "CARBON"),
}


def breadth_panel_instruments() -> tuple[Instrument, ...]:
    """The tradable additions — those that go into the expanded returns panel."""
    return tuple(i for i in BREADTH_INSTRUMENTS if i.role == "panel")


def breadth_validation_instruments() -> tuple[Instrument, ...]:
    """Fetched only to test the additions; never a strategy input."""
    return tuple(i for i in BREADTH_INSTRUMENTS if i.role == "validation")


def breadth_tickers() -> tuple[str, ...]:
    """Every Yahoo ticker the breadth build needs, in registry order."""
    return tuple(i.ticker for i in BREADTH_INSTRUMENTS)
