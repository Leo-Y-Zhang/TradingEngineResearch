"""Long-history multi-asset price panel (free data only).

Assembles the longest honest daily price history obtainable from free sources
(yfinance) across equity indices, rates, commodities, FX and modern ETF proxies,
and proves its integrity before any strategy is built on it.

Nothing in this package builds a strategy. It builds a panel and a receipt.

Raw rows are cached to ``_data/multiasset/`` which is gitignored — Yahoo's terms
forbid redistributing its data, so only DERIVED STATISTICS (coverage counts,
correlations, anomaly dispositions) are ever committed.
"""

from research.multiasset.instruments import (
    INSTRUMENTS,
    Instrument,
    by_key,
    panel_instruments,
    tickers,
)

__all__ = ["INSTRUMENTS", "Instrument", "by_key", "panel_instruments", "tickers"]
