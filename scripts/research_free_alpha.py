"""FREE-data alpha research runner — does a LEARNED combination of PIT-safe FUNDAMENTAL
factors, built from ZERO-COST public sources (SEC EDGAR companyfacts + yfinance prices),
carry a deflation-surviving edge at moderate breadth?

This is the free-data sibling of ``scripts/research_sharadar_alpha.py``. It tests the SAME
hypothesis — that the richer 14-factor fundamental library crosses the deflation bar — but
on data anyone can obtain for free, to learn whether the (paid) Sharadar read is buying a
real research advantage or merely cleaner plumbing.

Pipeline (mirrors the Sharadar runner's structure, honesty and gating exactly, but sourced
from free EDGAR fundamentals + yfinance prices instead of the paid bulk export):

  1. Resolve a configurable universe of liquid US large/mid-cap names (``DEFAULT_UNIVERSE``,
     or ``--tickers`` / ``--universe-file``). Free data is CURRENT-LISTED only — see the
     explicit SURVIVORSHIP caveat printed on every run.
  2. Pull each name's ENTIRE XBRL fact set from SEC EDGAR's *companyfacts* API in ONE request
     per company (``data.edgar_ingestion.fetch_company_facts``), CACHED to a gitignored dir
     (``--cache-dir``, default ``_data/edgar_cache/``) so reruns are instant and SEC is not
     hammered; live requests are politely rate-limited (<= ~8 req/s). Pull free yfinance
     prices (monthly by default) for the same names.
  3. ``data.edgar_ingestion.build_edgar_panel`` forward-fills each fundamental onto the price
     grid by its FILING date (strict point-in-time), then
     ``research.fundamental_features.compute_features`` produces the 14 cross-sectionally
     normalized factors (value / quality / growth / investment / earnings-quality / leverage
     + a 12-1 momentum control). ``marketcap`` is reconstructed PIT-safely as
     ``price * sharesbas`` (SEC publishes no prices), enabling the value ratios.
  4. Build forward returns PIT-safely: ``fwd.loc[t, s] = price[t+1]/price[t] - 1`` stored at
     the rebalance date ``t`` (the return earned AFTER ``t`` — no look-ahead).
  5. Learn a regularised cross-sectional combination with a PURGED walk-forward via
     ``research.alpha_factory.learn_signal_weights``.
  6. Gate it with ``research.validation.selection_rule`` (the real Bailey-Lopez de Prado
     Deflated Sharpe Ratio cutoff + rank-IC / net-Sharpe / stability checks).
  7. Print an HONEST report: n_names, n_dates, period, OOS mean/rank IC, net Sharpe, Deflated
     Sharpe vs the 0.95 cutoff, PBO (CSCV), and a clear DEPLOYABLE / NOT-DEPLOYABLE verdict.
     DEFAULT-DENY: anything that does not clear the gate is reported as no robust edge.

A missing standardized factor for a name is treated as cross-sectionally NEUTRAL (0.0) at the
COMBINATION layer only; raw fundamentals are never imputed (golden rule 3). This keeps the
cross-section usable when some names lack a field (common on a broad real-world universe).

SURVIVORSHIP CAVEAT (read this): the universe is whatever is CURRENTLY listed/known to the
ticker->CIK map and to yfinance. Free data CANNOT provide a survivorship-free historical
universe (delisted/acquired/bankrupt names are absent), so any positive read here is
optimistically biased and is NOT a substitute for the survivorship-free Sharadar study.

Run on free data:   python scripts/research_free_alpha.py --tickers AAPL,MSFT,...   (network)
Offline self-test:  python scripts/research_free_alpha.py --selftest   (no network; CI-safe)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.edgar_ingestion import (  # noqa: E402
    build_edgar_panel,
    extract_company_facts,
    fetch_company_facts,
    ticker_to_cik_map,
)
from research.alpha_factory import learn_signal_weights  # noqa: E402
from research.fundamental_features import FEATURE_NAMES, compute_features  # noqa: E402
from research.validation import (  # noqa: E402
    PurgedWalkForwardSplitter,
    ValidationResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    selection_rule,
)

logger = logging.getLogger(__name__)

# ── Defaults / constants ─────────────────────────────────────────────────────────────
DSR_CUTOFF = 0.95                 # the real Bailey-Lopez de Prado DSR gate in selection_rule
DEFAULT_WARMUP_DAYS = 400         # > 1y so YoY growth + 12-1 momentum are defined at rebalance 0
DEFAULT_COST_BPS = 10.0           # per-rebalance cost drag subtracted from OOS returns
PERIODS_PER_YEAR = 12             # monthly rebalances
DEFAULT_CACHE_DIR = Path("_data/edgar_cache")
DEFAULT_PRICE_INTERVAL = "1mo"    # yfinance interval; monthly suits a monthly rebalance
DEFAULT_START = "2005-01-01"      # yfinance price history start (EDGAR facts carry full history)
_MIN_REQUEST_INTERVAL_S = 0.13    # polite SEC rate-limit (<= ~8 req/s)

# An explicit, unmissable disclosure printed on every run (the whole point of a free study).
SURVIVORSHIP_CAVEAT = (
    "SURVIVORSHIP CAVEAT (current-listed + winner-selected; read this):\n"
    "  (a) CURRENT-LISTED ONLY: the universe is whatever is currently listed and resolvable via "
    "the SEC ticker->CIK map and yfinance; delisted / acquired / bankrupt names are absent, so "
    "any positive read is optimistically biased.\n"
    "  (b) WINNER-SELECTED DEFAULT: the built-in DEFAULT_UNIVERSE is a hand-chosen basket of "
    "today's surviving US large/mid-caps -- a current-winner selection layered on top of (a). "
    "Pass --tickers / --universe-file for a set you control.\n"
    "  (c) TICKER REASSIGNMENT: tickers are recycled over time, so today's ticker->CIK map can "
    "mis-attribute a now-defunct issuer's filing history to the company that currently holds the "
    "ticker.\n"
    "  Free data CANNOT give a survivorship-free historical universe; this is NOT a substitute "
    "for the survivorship-free (paid Sharadar) study."
)

# ~140 liquid US large/mid-cap names, sector-diversified across all 11 GICS sectors.
# CURRENT-LISTED and current-WINNER-selected (see the SURVIVORSHIP CAVEAT above) — a free,
# reproducible default a reader can re-pull without a paid subscription; pass --tickers /
# --universe-file for a set you control. Broadening reduces (it does NOT remove) the selection
# bias of a hand-picked mega-cap basket.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    # Information technology (26)
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN", "INTC",
    "AMD", "QCOM", "TXN", "IBM", "INTU", "NOW", "AMAT", "MU", "ADI", "LRCX",
    "KLAC", "SNPS", "CDNS", "PANW", "NXPI", "MCHP",
    # Communication services (9)
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    # Consumer discretionary (14)
    "AMZN", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "TGT", "GM",
    "F", "ORLY", "AZO", "MAR",
    # Consumer staples (10)
    "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "PM", "KMB",
    # Health care (18)
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT", "ISRG", "ELV", "VRTX", "REGN",
    # Financials (17)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "SPGI",
    "CB", "PNC", "USB", "TFC", "COF", "MMC", "ICE",
    # Industrials (17)
    "CAT", "BA", "HON", "GE", "UNP", "UPS", "RTX", "LMT", "DE", "MMM",
    "EMR", "FDX", "NSC", "ETN", "CSX", "GD", "NOC",
    # Energy (9)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    # Materials (8)
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DD", "NUE",
    # Utilities (7)
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    # Real estate (6)
    "AMT", "PLD", "EQIX", "SPG", "O", "CCI",
)


# ── Report container ─────────────────────────────────────────────────────────────────
@dataclass
class ResearchReport:
    """Everything the honest report prints (and the tests assert on)."""

    label: str
    n_symbols: int
    n_rebalances: int
    test_size: int
    n_trials: int
    date_start: pd.Timestamp | None
    date_end: pd.Timestamp | None
    weights: dict[str, float]
    result: ValidationResult
    pbo: float
    comp_sharpe: float
    comp_dsr: float
    # Cost actually applied to the OOS returns (the report label must never hardcode
    # the default -- a 2026-07 review finding).
    cost_bps: float = DEFAULT_COST_BPS

    @property
    def deployable(self) -> bool:
        """DEFAULT-DENY verdict: deployable only if the statistical gate passes."""
        return selection_rule(self.result)


# ── Panel / grid helpers ─────────────────────────────────────────────────────────────
def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide ``(date × ticker)`` price matrix from a tidy ``(ticker, date, price)`` frame."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    px = p.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
    return px.sort_index()


def _rebalance_dates(px: pd.DataFrame, warmup_days: int) -> list[pd.Timestamp]:
    """Last actual trading day of each month, after a warm-up so lookback factors exist."""
    idx = px.index
    if len(idx) == 0:
        return []
    by_month = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last()
    floor = idx[0] + pd.Timedelta(days=warmup_days)
    return [pd.Timestamp(d) for d in by_month if pd.Timestamp(d) >= floor]


def _feature_panels(funds: pd.DataFrame, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """EDGAR PIT panel → per-date cross-sectionally normalized factors → wide
    ``(date × ticker)`` frame per feature (one DataFrame per name in :data:`FEATURE_NAMES`).

    ``marketcap`` is reconstructed PIT-safely as ``price * sharesbas`` (the price is known on
    ``date`` and the share count is the latest filed by ``date``), which is more current than
    any stale filing-date figure and unlocks the value ratios SEC cannot price directly.

    NOTE (mild value-factor coupling): the value ratios divide by this ``marketcap``, whose
    ``price[t]`` is the SAME price :func:`run_research` uses as the BASE of the forward return
    ``price[t+1]/price[t] - 1``. A shared ``price[t]`` thus appears in both the value factor and
    the return denominator — a mild mechanical coupling (NOT a look-ahead: both use only data
    known at ``t``) worth bearing in mind when reading value-factor results."""
    panel = build_edgar_panel(funds, prices)
    if "sharesbas" in panel.columns:
        panel = panel.copy()
        panel["marketcap"] = panel["price"] * panel["sharesbas"]
    features = compute_features(panel)
    return {
        f: features.pivot_table(index="date", columns="ticker", values=f, aggfunc="last")
        for f in FEATURE_NAMES
    }


def _single_factor_returns(sig: pd.Series, y: pd.Series, cost_bps: float) -> float | None:
    """One date's dollar-neutral, gross-1 long-short net return for a single signal."""
    s = sig.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    mask = np.isfinite(s) & np.isfinite(yv)
    if int(mask.sum()) < 3:
        return None
    w = s[mask] - s[mask].mean()
    denom = float(np.abs(w).sum())
    if denom <= 0.0:
        return None
    return float((w / denom) @ yv[mask]) - cost_bps / 1e4


# ── Core research flow ───────────────────────────────────────────────────────────────
def run_research(
    funds: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    label: str = "EDGAR+yfinance",
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
) -> ResearchReport:
    """Build features + forward returns, learn the gated combination, return a report.

    Honest + default-deny by construction: degenerate / no-edge inputs flow through
    ``learn_signal_weights`` (which fails closed) into a NOT-DEPLOYABLE verdict.

    Parameters
    ----------
    funds : tidy ``(ticker, tag, filed, period_end, value)`` frame with CANONICAL tag names
        (the output of :func:`data.edgar_ingestion.extract_company_facts`).
    prices : tidy ``(ticker, date, price)`` frame (free yfinance closes, or synthetic).
    """
    px = _price_matrix(prices)
    syms = list(px.columns)
    rebal = _rebalance_dates(px, warmup_days)
    panel_dates = rebal[:-1]
    n_dates = len(panel_dates)

    wide = _feature_panels(funds, prices)
    grid = {
        f: wide[f].reindex(index=panel_dates, columns=syms).astype(float) for f in FEATURE_NAMES
    }
    # Learner inputs: neutral-fill residual NaNs (missing factor = cross-sectionally neutral).
    panel = {f: grid[f].fillna(0.0) for f in FEATURE_NAMES}

    fwd = pd.DataFrame(index=panel_dates, columns=syms, dtype=float)
    for i, t in enumerate(panel_dates):
        t1 = rebal[i + 1]
        fwd.loc[t] = (px.loc[t1] / px.loc[t] - 1.0).reindex(syms).to_numpy(dtype=float)

    # Naive equal-weight composite from the RAW (un-filled) standardized factors.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comp_arr = (
            np.nanmean(np.stack([grid[f].to_numpy(dtype=float) for f in FEATURE_NAMES]), axis=0)
            if n_dates
            else np.empty((0, len(syms)))
        )
    comp_df = pd.DataFrame(comp_arr, index=panel_dates, columns=syms)

    test = max(3, n_dates // 5)
    splitter = PurgedWalkForwardSplitter(
        train_size=max(6, n_dates - 3 * test),
        valid_size=test,
        test_size=test,
        embargo_size=1,
        label_horizon=1,
    )
    n_trials = len(FEATURE_NAMES) + 2     # the 14 factors tried + naive composite + learned config

    weights, result = learn_signal_weights(
        panel,
        fwd,
        splitter=splitter,
        n_trials=n_trials,
        cost_drag_bps=cost_bps,
        periods_per_year=PERIODS_PER_YEAR,
    )

    # Naive composite baseline: per-date dollar-neutral long-short, net of cost, then DSR.
    comp_ret = [
        r
        for t in panel_dates
        if (r := _single_factor_returns(comp_df.loc[t], fwd.loc[t], cost_bps)) is not None
    ]
    comp_a = np.asarray(comp_ret, dtype=float)
    comp_dsr = float(deflated_sharpe_ratio(comp_a, n_trials=n_trials)) if comp_a.size >= 4 else 0.0
    comp_sharpe = (
        float(comp_a.mean() / comp_a.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        if comp_a.size > 1 and comp_a.std(ddof=1) > 0
        else 0.0
    )

    # PBO (CSCV) across the single-factor library — an honest selection-overfitting diagnostic
    # (NOTE: selection_rule does NOT consume PBO; the real DSR gate is the binding constraint).
    cols = []
    for f in FEATURE_NAMES:
        col = [_single_factor_returns(grid[f].loc[t], fwd.loc[t], cost_bps) or 0.0 for t in panel_dates]
        cols.append(col)
    perf = np.asarray(cols, dtype=float).T
    pbo = float(probability_of_backtest_overfitting(perf)) if perf.shape[0] >= 4 else 0.5

    return ResearchReport(
        label=label,
        n_symbols=len(syms),
        n_rebalances=n_dates,
        test_size=test,
        n_trials=n_trials,
        date_start=panel_dates[0] if panel_dates else None,
        date_end=panel_dates[-1] if panel_dates else None,
        weights=weights,
        result=result,
        pbo=pbo,
        comp_sharpe=comp_sharpe,
        comp_dsr=comp_dsr,
        cost_bps=cost_bps,
    )


def print_report(report: ResearchReport) -> None:
    """Print the honest, default-deny research report (with the survivorship caveat)."""
    res = report.result
    d0 = report.date_start.date() if report.date_start is not None else "n/a"
    d1 = report.date_end.date() if report.date_end is not None else "n/a"
    print(f"\n===========  LEARNED fundamental alpha - {report.label} "
          f"(FREE data, net of {report.cost_bps:.0f}bps, monthly)  ===========")
    print(f"universe: {report.n_symbols} names | {d0}..{d1} | "
          f"rebalances: {report.n_rebalances} | walk-forward test={report.test_size} | "
          f"DSR trials={report.n_trials}")
    print(f"features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    print("-" * 92)
    print("LEARNED ridge combination (out-of-sample, purged walk-forward):")
    print(f"  OOS IC={res.mean_ic:+.4f}  rank-IC={res.mean_rank_ic:+.4f}  "
          f"net_Sharpe={res.sharpe_net:.2f}  stability={res.stability_score:.2f}")
    print(f"  Deflated Sharpe={res.deflated_sharpe_ratio:.3f}  (cutoff {DSR_CUTOFF:.2f})   "
          f"PBO={report.pbo:.2f}  (CSCV, lower is better)")
    if res.leakage_flags:
        print(f"  leakage_flags: {res.leakage_flags}")
    print("  weights: " + ", ".join(f"{k}={report.weights.get(k, 0.0):+.3f}" for k in FEATURE_NAMES))
    print(f"  selection_rule -> {'PASS' if report.deployable else 'FAIL (default-deny)'}")
    print("-" * 92)
    print(f"NAIVE equal-weight composite:  net_Sharpe={report.comp_sharpe:.2f}  "
          f"DSR={report.comp_dsr:.3f}")
    print("-" * 92)
    if report.deployable:
        print("VERDICT: DEPLOYABLE - learned fundamental combination carries a "
              "deflation-surviving edge.")
    else:
        print("VERDICT: NOT-DEPLOYABLE (default-deny) - no robust edge survives deflation "
              "(honest: consistent with the no-easy-alpha prior).")
    print("-" * 92)
    print(SURVIVORSHIP_CAVEAT)
    print("=" * 92)


# ── Universe resolution ───────────────────────────────────────────────────────────────
def load_universe(
    tickers: str | None = None,
    universe_file: Path | None = None,
) -> list[str]:
    """Resolve the research universe.

    Precedence: explicit ``--tickers`` (comma-separated) > ``--universe-file`` (one ticker per
    line; ``#`` comments and blanks ignored) > :data:`DEFAULT_UNIVERSE`. Tickers are
    upper-cased and de-duplicated while preserving first-seen order."""
    raw: list[str]
    if tickers:
        raw = tickers.split(",")
    elif universe_file is not None:
        lines = Path(universe_file).read_text(encoding="utf-8").splitlines()
        raw = [ln.split("#", 1)[0] for ln in lines]
    else:
        raw = list(DEFAULT_UNIVERSE)
    seen: dict[str, None] = {}
    for tk in raw:
        t = tk.strip().upper()
        if t:
            seen.setdefault(t, None)
    return list(seen)


# ── Free network fetchers (NOT unit-tested; the suite uses the synthetic path) ────────
def fetch_free_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
    interval: str = DEFAULT_PRICE_INTERVAL,
) -> pd.DataFrame:  # pragma: no cover - network
    """Free yfinance adjusted closes → tidy ``(ticker, date, price)`` frame.

    ``auto_adjust=True`` so ``Close`` is split/dividend-adjusted (the correct series for
    total-return factor research). Names yfinance cannot serve simply contribute no rows."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):                      # single-ticker shape
        close = close.to_frame(tickers[0])
    tidy = (
        close.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="price")
    )
    tidy["ticker"] = tidy["ticker"].astype(str).str.upper()
    tidy["date"] = pd.to_datetime(tidy["date"])
    tidy = tidy.dropna(subset=["price"])
    return tidy.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_or_fetch_facts(
    cik: int, cache_dir: Path, *, force: bool = False
) -> tuple[dict, bool]:  # pragma: no cover - network
    """Return ``(companyfacts_json, hit_network)`` for ``cik``, caching raw JSON under
    ``cache_dir`` (gitignored). A cache hit performs NO request (so SEC is not hammered on
    reruns); ``force`` bypasses the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"CIK{int(cik):010d}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8")), False
    facts = fetch_company_facts(cik)
    path.write_text(json.dumps(facts), encoding="utf-8")
    return facts, True


def fetch_free_funds(
    tickers: list[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
    min_interval_s: float = _MIN_REQUEST_INTERVAL_S,
) -> pd.DataFrame:  # pragma: no cover - network
    """Free SEC EDGAR companyfacts for ``tickers`` → tidy canonical-tag fundamentals frame.

    One cached *companyfacts* request per resolvable name; live requests are spaced by
    ``min_interval_s`` (>= ~0.125s ⇒ <= ~8 req/s, SEC-polite). Unknown tickers and names
    reporting none of the wanted concepts are skipped (never fabricated)."""
    cik_map = ticker_to_cik_map()
    frames: list[pd.DataFrame] = []
    for tk in tickers:
        cik = cik_map.get(tk.upper())
        if cik is None:
            logger.warning("fetch_free_funds: no CIK for %s; skipping.", tk)
            continue
        facts, hit_network = _load_or_fetch_facts(cik, cache_dir, force=force)
        if hit_network:
            time.sleep(min_interval_s)               # rate-limit only on real requests
        frame = extract_company_facts(facts, tk.upper())
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["ticker", "tag", "filed", "period_end", "value"])
    return pd.concat(frames, ignore_index=True)


# ── Synthetic data (offline self-test + tests; NO network) ────────────────────────────
def build_synthetic_panel(
    *,
    seed: int = 7,
    edge: bool = True,
    n_tickers: int = 24,
    n_years: int = 8,
    start: str = "2012-01-06",
    sigma: float = 0.020,
    edge_amp: float = 0.006,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic SYNTHETIC EDGAR funds + yfinance-shaped prices (no network, seeded).

    Returns ``(funds, prices)`` in EXACTLY the free-path shapes: ``funds`` is tidy
    ``(ticker, tag, filed, period_end, value)`` with CANONICAL tag names (what
    :func:`data.edgar_ingestion.extract_company_facts` emits and
    :func:`data.edgar_ingestion.build_edgar_panel` consumes); ``prices`` is tidy
    ``(ticker, date, price)``.

    Prices are emitted WEEKLY (a faithful, CI-cheap stand-in for the real grid — monthly
    rebalances and the YoY / 12-1 lookback tolerances all resolve on a weekly grid). Every one
    of the 14 factors is given genuine cross-sectional variance. When ``edge=True`` a clean
    edge is injected: each name's price drift is proportional to its (constant) ROE, so ROE —
    and, since prices accumulate that drift, 12-1 momentum — predict the forward return. When
    ``edge=False`` all drifts are zero (pure random walk) so no factor predicts returns and the
    gate must DEFAULT-DENY.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]

    # ROE ramp (the edge driver) + independent, distinct levels/growths for every other ratio.
    roe = np.linspace(-0.05, 0.30, n_tickers)
    z_roe = (roe - roe.mean()) / roe.std()
    e0 = rng.uniform(500.0, 1500.0, n_tickers)
    a0 = rng.uniform(2000.0, 8000.0, n_tickers)
    r0 = rng.uniform(800.0, 3000.0, n_tickers)
    s0 = rng.uniform(50.0, 500.0, n_tickers)
    g_e = rng.uniform(0.00, 0.04, n_tickers)
    g_a = rng.uniform(0.00, 0.05, n_tickers)
    g_r = rng.uniform(-0.01, 0.06, n_tickers)
    g_s = rng.uniform(-0.005, 0.03, n_tickers)
    gpm = rng.uniform(0.20, 0.60, n_tickers)
    opm = rng.uniform(0.05, 0.25, n_tickers)
    accr = rng.uniform(-0.20, 0.20, n_tickers)
    de = rng.uniform(0.10, 1.50, n_tickers)

    # ── Weekly prices (tidy ticker, date, price) ──
    dates = pd.date_range(start=start, periods=n_years * 52, freq="W-FRI")
    mu = edge_amp * z_roe if edge else np.zeros(n_tickers)
    eps_noise = rng.standard_normal((len(dates), n_tickers))
    weekly = mu[None, :] + sigma * eps_noise
    prices_arr = 100.0 * np.cumprod(1.0 + weekly, axis=0)
    price_rows: list[dict[str, object]] = []
    for j, tk in enumerate(tickers):
        for di, d in enumerate(dates):
            price_rows.append({"ticker": tk, "date": d, "price": float(prices_arr[di, j])})
    prices = pd.DataFrame.from_records(price_rows)

    # ── Quarterly fundamentals (filed lags period_end by 45d → PIT exercised) ──
    quarter_ends = pd.date_range(start=dates[0], end=dates[-1], freq="QE")
    fund_rows: list[dict[str, object]] = []
    for q, ce in enumerate(quarter_ends):
        filed = ce + pd.Timedelta(days=45)
        for j, tk in enumerate(tickers):
            equity = e0[j] * (1.0 + g_e[j]) ** q
            assets = a0[j] * (1.0 + g_a[j]) ** q
            revenue = r0[j] * (1.0 + g_r[j]) ** q
            shares = s0[j] * (1.0 + g_s[j]) ** q
            netinc = roe[j] * equity                       # roe = netinc/equity = roe[j]
            values: dict[str, float] = {
                "netinc": netinc,
                "equity": equity,
                "assets": assets,
                "revenue": revenue,
                "gp": gpm[j] * revenue,
                "ebit": opm[j] * revenue,
                "ncfo": netinc - accr[j] * assets,         # accruals = (netinc-ncfo)/assets
                "debt": de[j] * equity,
                "sharesbas": shares,
                "eps": netinc / shares,
            }
            for tag, val in values.items():
                fund_rows.append({
                    "ticker": tk, "tag": tag, "filed": filed,
                    "period_end": ce, "value": float(val),
                })
    funds = pd.DataFrame.from_records(fund_rows)
    return funds, prices


def selftest() -> int:
    """Run the WHOLE free-data flow end-to-end on a small synthetic panel, NO network.

    Proves the pipeline (a) recovers an injected edge and reports DEPLOYABLE, and (b) honestly
    DEFAULT-DENIES pure noise. Returns 0 on success, 1 on any failed expectation."""
    print("[selftest] building synthetic EDGAR+price panel (deterministic, no network)...")
    ok = True

    funds, prices = build_synthetic_panel(seed=7, edge=True)
    edge_report = run_research(funds, prices, label="SELFTEST edge")
    print_report(edge_report)
    if not edge_report.deployable:
        print("[selftest] FAIL: injected edge was not found DEPLOYABLE.")
        ok = False
    if edge_report.result.deflated_sharpe_ratio < DSR_CUTOFF:
        print("[selftest] FAIL: injected edge did not clear the DSR cutoff.")
        ok = False

    funds_n, prices_n = build_synthetic_panel(seed=11, edge=False)
    noise_report = run_research(funds_n, prices_n, label="SELFTEST noise")
    print_report(noise_report)
    if noise_report.deployable:
        print("[selftest] FAIL: pure noise was (wrongly) found DEPLOYABLE.")
        ok = False

    print(f"\n[selftest] {'PASS - pipeline recovers edge and default-denies noise.' if ok else 'FAILED.'}")
    return 0 if ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default=None,
                   help="Comma-separated universe (overrides --universe-file and the default).")
    p.add_argument("--universe-file", type=Path, default=None,
                   help="File with one ticker per line ('#' comments / blanks ignored).")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="Gitignored dir for cached EDGAR companyfacts JSON (default _data/edgar_cache/).")
    p.add_argument("--start", default=DEFAULT_START, help="yfinance price history start date.")
    p.add_argument("--end", default=None, help="yfinance price history end date (default: today).")
    p.add_argument("--interval", default=DEFAULT_PRICE_INTERVAL,
                   help="yfinance bar interval (e.g. 1mo, 1wk). Default 1mo.")
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                   help="History before the first rebalance (>365 so YoY + momentum exist).")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="Per-rebalance cost drag (bps) subtracted from OOS returns.")
    p.add_argument("--force-refresh", action="store_true",
                   help="Bypass the EDGAR cache and re-fetch every name.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the whole flow on synthetic data (no network); exit 0 on success.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()

    universe = load_universe(args.tickers, args.universe_file)
    print(SURVIVORSHIP_CAVEAT)
    print(f"resolving {len(universe)} names | EDGAR cache: {args.cache_dir} | "
          f"prices: yfinance {args.interval} from {args.start}")
    funds = fetch_free_funds(universe, args.cache_dir, force=args.force_refresh)  # pragma: no cover - network
    prices = fetch_free_prices(universe, start=args.start, end=args.end,           # pragma: no cover - network
                               interval=args.interval)
    if funds.empty or prices.empty:                                                # pragma: no cover - network
        print("error: no free fundamentals/prices fetched for the requested universe.",
              file=sys.stderr)
        return 2
    report = run_research(funds, prices, label="EDGAR+yfinance",                   # pragma: no cover - network
                          warmup_days=args.warmup_days, cost_bps=args.cost_bps)
    print_report(report)                                                           # pragma: no cover - network
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
