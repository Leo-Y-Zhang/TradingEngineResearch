# Real Feature Ingestion — Design

**Date:** 2026-06-10 · **ROADMAP:** Phase 2, item 3 (real feature ingestion + recorded
fixture) · **Status:** approved (design), pre-implementation

## Purpose

Replace synthetic feature data with **real, price-derived features** flowing through the
existing PIT-safe `feature_store`, plus a **committed offline fixture** so tests run on
real data without network.

## Source

`yfinance` (the user's chosen source), **upgraded 0.2.44 → 1.4.1** — the old pin was
broken by Yahoo API changes (verified: 0.2.44 returns empty; 1.4.1 fetches cleanly).
Update `constraints.txt` (`yfinance==1.4.1`) and the `ingestion` extra (`>=1.0,<2`).

## Reality of "the 18 features"

Most model features need sources yfinance does not provide (insider flow → SEC,
sentiment → news, OFI → L2, earnings proximity → calendar, engine_expected_return → a
model). "Real ingestion" therefore computes the **price-derivable subset** from real
prices; the rest fall to the feature store's existing deterministic imputation. Derivable
subset (all trailing / PIT-safe):

| feature | formula (trailing) |
|---|---|
| `momentum_12_1` | `close.pct_change(252) − close.pct_change(21)` |
| `reversal_5d` | `−close.pct_change(5)` |
| `volume_ratio` | `volume / volume.rolling(20).mean()` |
| `adv_ratio` | `vol.rolling(20).mean() / vol.rolling(60).mean()` |
| `overnight_gap_mean` | `(open/close.shift(1) − 1).rolling(20).mean()` |
| `idiosyncratic_vol` | `close.pct_change().rolling(20).std() · √252` |

## Components — `data/price_ingestion.py` (feeds `data/feature_store.py`)

- `fetch_prices(symbols, start, end) -> tidy DataFrame` — `yfinance` wrapper (lazy import;
  network-only, `# pragma: no cover`).
- `load_prices(path) -> DataFrame` — read the committed CSV fixture (offline).
- `compute_price_features(prices) -> dict[(symbol, Timestamp), dict[str, float]]` — the
  derivable subset, per symbol/date, all trailing windows (no future leak); NaN warm-up
  values dropped.
- `ingest_prices(prices, mode="RESEARCH") -> int` — build PIT-safe `FeatureRow`s
  (`asof_timestamp` = bar date, `feature_schema_version` = v6.0, `freshness_flags` all
  fresh, `missing_count` = unfilled price features) and register them via
  `feature_store._register_features`, so `get_features` returns **real** values.

## Fixture

`tests/fixtures/prices_sample.csv` — 5 liquid symbols (AAPL, MSFT, GOOG, AMZN, JPM) ×
2 years daily (2022–2023), columns `date,symbol,open,high,low,close,volume`. ~2.5k rows.
Fetched once; committed; small; for tests only (proprietary repo, not redistributed).

## Tests (TDD, offline)

- fixture loads offline with the expected schema / 5 symbols.
- `compute_price_features`: all values finite; `volume_ratio>0`; `idiosyncratic_vol≥0`;
  `momentum_12_1` present for late dates (252-day lookback satisfied).
- **PIT-safety**: a feature at date `t` is identical whether computed on full history or
  on history truncated at `t` (trailing windows only).
- `ingest_prices` → `get_features` returns the real computed values (not imputed defaults)
  for a queried symbol/date; future rows do not leak at an earlier `asof`.

The live `fetch_prices` is **not** in the suite (validated once when recording the
fixture) so tests never hit the network.

## Scope (YAGNI)

Price-derived features only; daily bars; no SEC/news/L2/earnings sources; no scheduled /
incremental refresh. The non-price features remain imputed by the feature store.
