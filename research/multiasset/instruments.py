"""The instrument registry for the long-history multi-asset panel.

Every entry states EXACTLY what a "return" means for that series, because the
instruments do not share a convention and silently mixing them is how a panel
lies:

* ``price_return``      — level ratio. What it excludes (dividends, roll yield,
                          interest differential) is stated per instrument.
* ``inverse_price_return`` — the Yahoo quote is the RECIPROCAL of the position we
                          want (``JPY=X`` is JPY per USD; a long-JPY position
                          earns when that number FALLS).
* ``par_bond_total_return`` — the series is a YIELD, not a price. A naive
                          ``pct_change`` on ``^TNX`` is meaningless (it is the
                          return on the yield, and it has the WRONG SIGN versus
                          the bond). Converted by repricing a constant-maturity
                          par bond; see ``panel.par_bond_total_return``.
* ``bill_cash_accrual`` — ``^IRX`` is a 13-week bill DISCOUNT rate. Converted to
                          a bond-equivalent yield and accrued ACT/365 as cash.
* ``none``              — carried as a level/yield for signal construction only,
                          never as a return.

``role``:
  ``panel``      — goes into the tradable returns panel.
  ``cash``       — the risk-free accrual used to form excess returns.
  ``validation`` — fetched ONLY to test the panel against an independent
                   instrument that should track it. Never a strategy input.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "INSTRUMENTS",
    "Instrument",
    "by_key",
    "panel_instruments",
    "tickers",
    "validation_instruments",
]


@dataclass(frozen=True)
class Instrument:
    """One series, with the semantics of its return spelled out."""

    key: str
    ticker: str
    name: str
    asset_class: str
    currency: str
    return_method: str
    role: str = "panel"
    maturity_years: float | None = None
    notes: str = ""

    @property
    def tradable(self) -> bool:
        """True when the key belongs in the tradable returns panel."""
        return self.role == "panel"


# ── Equity indices ────────────────────────────────────────────────────────────
# All PRICE indices (dividends EXCLUDED) except ^GDAXI, which is the DAX
# Performance-Index and DOES include reinvested dividends. That asymmetry is
# roughly 2-3%/yr of drift and must not be read as German equity outperformance.
_EQUITY = [
    Instrument("SPX", "^GSPC", "S&P 500", "equity_index", "USD", "price_return",
               notes="PRICE index; dividends excluded (~1.8-4%/yr depending on era)."),
    Instrument("DJIA", "^DJI", "Dow Jones Industrial Average", "equity_index", "USD",
               "price_return", notes="PRICE index; price-weighted; dividends excluded."),
    Instrument("NASDAQ", "^IXIC", "Nasdaq Composite", "equity_index", "USD", "price_return",
               notes="PRICE index; dividends excluded."),
    Instrument("FTSE100", "^FTSE", "FTSE 100", "equity_index", "GBP", "price_return",
               notes="PRICE index, GBP. Dividends excluded; UK yield is high (~3-4%/yr)."),
    Instrument("N225", "^N225", "Nikkei 225", "equity_index", "JPY", "price_return",
               notes="PRICE index, JPY, price-weighted. Dividends excluded."),
    Instrument("DAX", "^GDAXI", "DAX 40", "equity_index", "EUR", "price_return",
               notes="TOTAL-RETURN index (DAX Performance-Index) — NOT comparable to the "
                     "other equity indices, which are price-only."),
    Instrument("HSI", "^HSI", "Hang Seng", "equity_index", "HKD", "price_return",
               notes="PRICE index, HKD. Dividends excluded."),
    Instrument("ASX200", "^AXJO", "S&P/ASX 200", "equity_index", "AUD", "price_return",
               notes="PRICE index, AUD. Dividends excluded; AU yield is high (~4%/yr)."),
]

# ── Rates: YIELDS, converted by repricing, never by pct_change ────────────────
_RATES = [
    Instrument("US5Y_TR", "^FVX", "US 5y constant-maturity par bond", "rates", "USD",
               "par_bond_total_return", maturity_years=5.0,
               notes="^FVX is the 5y CMT yield in percent. Converted to a total return by "
                     "repricing a par bond; the yield itself is kept in the yields panel."),
    Instrument("US10Y_TR", "^TNX", "US 10y constant-maturity par bond", "rates", "USD",
               "par_bond_total_return", maturity_years=10.0,
               notes="^TNX is the 10y CMT yield in percent. Validated against IEF."),
    Instrument("US30Y_TR", "^TYX", "US 30y constant-maturity par bond", "rates", "USD",
               "par_bond_total_return", maturity_years=30.0,
               notes="^TYX is the 30y CMT yield in percent. Validated against TLT."),
    Instrument("US_CASH_13W", "^IRX", "US 13-week T-bill cash", "rates", "USD",
               "bill_cash_accrual", role="cash", maturity_years=0.25,
               notes="^IRX is a DISCOUNT rate. Converted to bond-equivalent yield and "
                     "accrued ACT/365. This is the risk-free leg, not a risk asset."),
]

# ── Commodities: Yahoo front-month CONTINUOUS futures ─────────────────────────
# These splice contracts at expiry WITHOUT back-adjusting, so the return series
# contains the roll spread as if it were a price move. Quantified against the
# physical ETFs where one exists (GLD, SLV).
_COMMODITIES = [
    Instrument("GOLD_F", "GC=F", "Gold front-month future", "commodity", "USD", "price_return",
               notes="Front-month continuous; roll gaps NOT back-adjusted. Checked vs GLD."),
    Instrument("WTI_F", "CL=F", "WTI crude front-month future", "commodity", "USD",
               "price_return",
               notes="Front-month continuous; went NEGATIVE 2020-04-20, so the level ratio "
                     "is undefined there. Roll gaps NOT back-adjusted."),
    Instrument("SILVER_F", "SI=F", "Silver front-month future", "commodity", "USD",
               "price_return", notes="Front-month continuous; roll gaps. Checked vs SLV."),
    Instrument("COPPER_F", "HG=F", "Copper front-month future", "commodity", "USD",
               "price_return", notes="Front-month continuous; roll gaps NOT back-adjusted."),
    Instrument("NATGAS_F", "NG=F", "Natural gas front-month future", "commodity", "USD",
               "price_return",
               notes="Front-month continuous; the steepest roll yield of the set — a "
                     "front-month series overstates the return to a held position."),
]

# ── FX: SPOT only. The interest differential (the carry) is NOT in these. ─────
_FX = [
    Instrument("USDX", "DX-Y.NYB", "US Dollar Index (long USD)", "fx", "USD", "price_return",
               notes="Long-USD basket, spot only; no interest differential."),
    Instrument("EURUSD", "EURUSD=X", "Long EUR vs USD", "fx", "USD", "price_return",
               notes="Quoted USD per EUR, so a rising quote IS a long-EUR gain. Spot only."),
    Instrument("GBPUSD", "GBPUSD=X", "Long GBP vs USD", "fx", "USD", "price_return",
               notes="Quoted USD per GBP. Spot only; no interest differential."),
    Instrument("JPYUSD", "JPY=X", "Long JPY vs USD", "fx", "USD", "inverse_price_return",
               notes="JPY=X is JPY per USD — the RECIPROCAL of a long-JPY position. "
                     "Inverted here so the sign matches the other FX keys. Spot only."),
]

# ── Modern ETF proxies: auto_adjust=True ⇒ dividend-adjusted TOTAL return ──────
_ETFS = [
    Instrument("SPY", "SPY", "SPDR S&P 500 ETF", "etf", "USD", "price_return",
               notes="auto_adjust=True ⇒ dividends reinvested (TOTAL return)."),
    Instrument("TLT", "TLT", "iShares 20+yr Treasury ETF", "etf", "USD", "price_return",
               notes="Dividend-adjusted total return."),
    Instrument("GLD", "GLD", "SPDR Gold Shares", "etf", "USD", "price_return",
               notes="Physically backed; the holdable gold benchmark for GC=F."),
    Instrument("DBC", "DBC", "Invesco DB Commodity Index Fund", "etf", "USD", "price_return",
               notes="Broad commodity futures, roll-managed. Total return."),
    Instrument("EFA", "EFA", "iShares MSCI EAFE ETF", "etf", "USD", "price_return",
               notes="Developed ex-US, USD-denominated total return."),
    Instrument("EEM", "EEM", "iShares MSCI Emerging Markets ETF", "etf", "USD", "price_return",
               notes="Emerging markets, USD-denominated total return."),
    Instrument("IEF", "IEF", "iShares 7-10yr Treasury ETF", "etf", "USD", "price_return",
               notes="Dividend-adjusted total return; the benchmark for the ^TNX proxy."),
]

# ── Validation-only: never strategy inputs ────────────────────────────────────
_VALIDATION = [
    Instrument("BIL", "BIL", "SPDR 1-3 Month T-Bill ETF", "etf", "USD", "price_return",
               role="validation", notes="Independent check on the ^IRX cash accrual."),
    Instrument("IEI", "IEI", "iShares 3-7yr Treasury ETF", "etf", "USD", "price_return",
               role="validation", notes="Independent check on the ^FVX par-bond proxy."),
    Instrument("SLV", "SLV", "iShares Silver Trust", "etf", "USD", "price_return",
               role="validation", notes="Physical silver; roll check for SI=F."),
]

INSTRUMENTS: tuple[Instrument, ...] = tuple(
    _EQUITY + _RATES + _COMMODITIES + _FX + _ETFS + _VALIDATION
)

# Pairs used to prove a constructed series against an independent instrument that
# should track it: (constructed_key, benchmark_key, what the check establishes).
VALIDATION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("US10Y_TR", "IEF", "10y par-bond proxy vs 7-10y Treasury ETF"),
    ("US30Y_TR", "TLT", "30y par-bond proxy vs 20+yr Treasury ETF"),
    ("US5Y_TR", "IEI", "5y par-bond proxy vs 3-7yr Treasury ETF"),
    ("US_CASH_13W", "BIL", "13-week bill accrual vs 1-3mo T-bill ETF"),
    ("GOLD_F", "GLD", "front-month gold future vs physical gold ETF"),
    ("SILVER_F", "SLV", "front-month silver future vs physical silver ETF"),
    ("SPX", "SPY", "S&P 500 PRICE index vs its TOTAL-return ETF (dividend drag)"),
)


# ── Quarantined observations ──────────────────────────────────────────────────
# Individually evidenced CORRUPT CLOSES, not a statistical screen. Each entry is a
# single level observation dropped before returns are computed, so the surrounding
# bars form one valid two-day return instead of a fake spike and a fake reversal.
#
# Why these and nothing else. A generic "large move that reverses" rule was built,
# measured and REJECTED: it removes Black Tuesday 1929, the 2020-03-12 crash and the
# 2024-08-05 yen-carry unwind, because real V-shaped crashes reverse too. What
# actually identifies these bars is a calendar signature no market event can have —
# 9 of EURUSD's 10 largest daily moves and 7 of JPYUSD's 10 land on the 8th or 9th of
# a month (6.6% base rate), every one of them in 2008 — combined with cross-instrument
# contradiction: on 2008-12-08 the dollar allegedly fell 17.3% against EUR while
# rising 17.7% against JPY, with GBPUSD +1.2% and the dollar index -1.8%.
#
# Admission criterion, applied uniformly: (a) the 8th/9th of a month in 2008, (b)
# |return| > 5%, (c) dropping the close leaves a two-day return under 2.5% in
# magnitude. JPYUSD 2008-05-08 (-3.2%) fails (b) and is deliberately KEPT.
QUARANTINE: tuple[tuple[str, str, str], ...] = (
    ("EURUSD", "2008-01-08", "corrupt close: +6.05% then -5.82%, round trip -0.13%"),
    ("EURUSD", "2008-02-08", "corrupt close: +7.54% then -6.87%, round trip +0.16%"),
    ("EURUSD", "2008-09-08", "corrupt close: +5.43% then -5.99%, round trip -0.89%"),
    ("EURUSD", "2008-10-08", "corrupt close: +10.08% then -9.09%, round trip +0.07%"),
    ("EURUSD", "2008-12-08", "corrupt close: +17.31% then -13.35%, round trip +1.64%"),
    ("JPYUSD", "2008-04-08", "corrupt close: -5.31% then +6.24%, round trip +0.61%"),
    ("JPYUSD", "2008-10-08", "corrupt close: -7.84% then +10.63%, round trip +1.96%"),
    ("JPYUSD", "2008-12-08", "corrupt close: -15.03% then +18.35%, round trip +0.56%"),
)


def by_key(key: str) -> Instrument:
    """Look an instrument up by panel key. Raises ``KeyError`` if unknown."""
    for inst in INSTRUMENTS:
        if inst.key == key:
            return inst
    raise KeyError(f"unknown instrument key: {key!r}")


def panel_instruments() -> tuple[Instrument, ...]:
    """The tradable instruments — those that go into the returns panel."""
    return tuple(i for i in INSTRUMENTS if i.role == "panel")


def validation_instruments() -> tuple[Instrument, ...]:
    """Fetched only to test the panel; never a strategy input."""
    return tuple(i for i in INSTRUMENTS if i.role == "validation")


def tickers() -> tuple[str, ...]:
    """Every Yahoo ticker the build needs, in registry order."""
    return tuple(i.ticker for i in INSTRUMENTS)
