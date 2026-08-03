"""Alpha research runner — does a LEARNED combination of PIT-safe FUNDAMENTAL factors,
built on survivorship-free Sharadar data, carry a deflation-surviving edge?

Pipeline (mirrors ``scripts/research_learned_alpha.py`` structure + honesty, but sourced
from the paid Sharadar bulk export instead of yfinance/EDGAR):

  1. Load the point-in-time, survivorship-free panel via ``data.sharadar_ingestion``
     (SF1 fundamentals + SEP prices → a tidy ``(ticker, date)`` panel forward-filled on
     ``datekey`` — the publication date, never ``calendardate``). ``--data-dir`` is
     configurable; SF1 / SEP files inside it are located by glob.
  2. Compute the 14 cross-sectionally normalized fundamental factors via
     ``research.fundamental_features.compute_features`` (value / quality / growth /
     investment / earnings-quality / leverage + a 12-1 momentum control).
  3. Build forward returns PIT-safely: ``fwd.loc[t, s] = price[t+1]/price[t] - 1`` stored at
     the rebalance date ``t`` (the return earned AFTER ``t`` — no look-ahead).
  4. Learn a regularised cross-sectional combination with a PURGED walk-forward over a
     moderate universe + long history via ``research.alpha_factory.learn_signal_weights``.
  5. Gate it with ``research.validation.selection_rule`` (the real Bailey-Lopez de Prado
     Deflated Sharpe Ratio cutoff + rank-IC / net-Sharpe / stability checks).
  6. Print an HONEST report: OOS IC / rank-IC, net Sharpe, Deflated Sharpe vs the 0.95
     cutoff, PBO (CSCV, an auxiliary overfitting diagnostic across the factor library), and
     a clear DEPLOYABLE / NOT-DEPLOYABLE verdict. DEFAULT-DENY: anything that does not clear
     the gate is reported as no robust edge — never promoted.

Survivorship note: Sharadar SEP retains delisted/acquired names, so (unlike the large-cap
yfinance runner) this read is NOT survivorship-biased — a genuine advantage of the paid data.

A missing standardized factor for a name is treated as cross-sectionally NEUTRAL (0.0) at the
COMBINATION layer only; raw fundamentals are never imputed (golden rule 3). This keeps the
cross-section usable when some names lack a field (common on a broad real-world universe).

Run on real data:   python scripts/research_sharadar_alpha.py --data-dir /path/to/sharadar
Offline self-test:   python scripts/research_sharadar_alpha.py --selftest   (no paid data; CI-safe)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.sharadar_ingestion import build_panel, load_sep, load_sf1  # noqa: E402
from research.alpha_factory import learn_signal_weights  # noqa: E402
from research.fundamental_features import FEATURE_NAMES, compute_features  # noqa: E402
from research.validation import (  # noqa: E402
    PurgedWalkForwardSplitter,
    ValidationResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    selection_rule,
)

# ── Defaults / constants ─────────────────────────────────────────────────────────────
DSR_CUTOFF = 0.95                 # the real Bailey-Lopez de Prado DSR gate in selection_rule
DEFAULT_DIMENSION = "ARQ"         # As-Reported Quarterly — PIT-safe (MR* views restate history)
DEFAULT_WARMUP_DAYS = 400         # > 1y so YoY growth + 12-1 momentum are defined at rebalance 0
DEFAULT_COST_BPS = 10.0           # per-rebalance cost drag subtracted from OOS returns
PERIODS_PER_YEAR = 12             # monthly rebalances
DEFAULT_SF1_GLOB = "*SF1*.csv"
DEFAULT_SEP_GLOB = "*SEP*.csv"


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

    @property
    def deployable(self) -> bool:
        """DEFAULT-DENY verdict: deployable only if the statistical gate passes."""
        return selection_rule(self.result)


# ── Panel / grid helpers ─────────────────────────────────────────────────────────────
def _price_matrix(sep: pd.DataFrame) -> pd.DataFrame:
    """Wide ``(date × ticker)`` adjusted-close matrix from the tidy SEP frame."""
    px = sep.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    return px.sort_index()


def _rebalance_dates(px: pd.DataFrame, warmup_days: int) -> list[pd.Timestamp]:
    """Last actual trading day of each month, after a warm-up so lookback factors exist."""
    idx = px.index
    if len(idx) == 0:
        return []
    by_month = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last()
    floor = idx[0] + pd.Timedelta(days=warmup_days)
    return [pd.Timestamp(d) for d in by_month if pd.Timestamp(d) >= floor]


def _feature_panels(sf1: pd.DataFrame, sep: pd.DataFrame, dimension: str) -> dict[str, pd.DataFrame]:
    """PIT panel → per-date cross-sectionally normalized factors → wide ``(date × ticker)``
    frame per feature (one DataFrame per name in :data:`FEATURE_NAMES`)."""
    panel = build_panel(sf1, sep, dimension=dimension)
    feat_panel = panel.rename(columns={"close": "price"})
    if "sharesbas" in feat_panel.columns:
        # Market cap on each PRICE date (price known on `date` × shares filed by `date`):
        # PIT-safe and more current than SF1's stale filing-date marketcap.
        feat_panel["marketcap"] = feat_panel["price"] * feat_panel["sharesbas"]
    features = compute_features(feat_panel)
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
def build_liquidity_universe(
    sep: pd.DataFrame,
    panel_dates: "pd.DatetimeIndex | list[pd.Timestamp]",
    *,
    top_n: int | None = None,
    min_dollar_volume: float | None = None,
    window: int = 63,
    min_obs: int = 42,
) -> pd.DataFrame:
    """PIT-safe tradable-universe mask: for each panel date, rank names by the MEDIAN
    daily dollar volume (``close * volume``) over the trailing ``window`` TRADING days
    strictly up to that date, requiring at least ``min_obs`` observed days. ``top_n``
    keeps the N most liquid names; ``min_dollar_volume`` is an absolute floor (they
    combine when both are given). Returns a boolean (panel_dates x tickers) frame.

    Motivated by the 2026-07-13 dev diagnosis (sharadar_dev_log.md entry 2): the
    unfiltered universe's P&L lives in untradeable micro-caps. CAVEAT (recorded there):
    ``close`` is the split/dividend-ADJUSTED price while volume is as-traded, so names
    with large future splits have understated historical dollar volume — a conservative
    bias (would-be winners are excluded early, never snuck in)."""
    if top_n is None and min_dollar_volume is None:
        raise ValueError("liquidity universe needs top_n and/or min_dollar_volume")
    dv = (
        sep.assign(_dv=sep["close"] * sep["volume"])
        .pivot_table(index="date", columns="ticker", values="_dv", aggfunc="last")
        .sort_index()
    )
    mask = pd.DataFrame(False, index=pd.DatetimeIndex(panel_dates), columns=dv.columns)
    for t in mask.index:
        trailing = dv.loc[:t].tail(window)
        med = trailing.median(skipna=True).where(trailing.notna().sum() >= min_obs)
        eligible = med.dropna()
        if min_dollar_volume is not None:
            eligible = eligible[eligible >= min_dollar_volume]
        if top_n is not None:
            eligible = eligible.nlargest(top_n)
        mask.loc[t, eligible.index] = True
    return mask


def run_research(
    sf1: pd.DataFrame,
    sep: pd.DataFrame,
    *,
    label: str = "Sharadar",
    dimension: str = DEFAULT_DIMENSION,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
    universe_mask: pd.DataFrame | None = None,
    fwd_return_cap: float | None = None,
) -> ResearchReport:
    """Build features + forward returns, learn the gated combination, return a report.

    Honest + default-deny by construction: degenerate / no-edge inputs flow through
    ``learn_signal_weights`` (which fails closed) into a NOT-DEPLOYABLE verdict.

    ``universe_mask`` (boolean dates x tickers — see :func:`build_liquidity_universe`)
    restricts EVALUATION to the masked names by voiding forward returns outside it;
    features stay cross-sectionally intact and every consumer (learner, naive composite,
    PBO) inherits the restriction through its own finite-pair masking.
    ``fwd_return_cap`` clips forward returns to ±cap (dev-QA guard: single unrealizable
    penny-stock prints must not dominate a verdict). Defaults (``None``) leave the
    registered behavior bit-identical."""
    px = _price_matrix(sep)
    syms = list(px.columns)
    rebal = _rebalance_dates(px, warmup_days)
    panel_dates = rebal[:-1]
    n_dates = len(panel_dates)

    wide = _feature_panels(sf1, sep, dimension)
    grid = {
        f: wide[f].reindex(index=panel_dates, columns=syms).astype(float) for f in FEATURE_NAMES
    }
    # Learner inputs: neutral-fill residual NaNs (missing factor = cross-sectionally neutral).
    panel = {f: grid[f].fillna(0.0) for f in FEATURE_NAMES}

    fwd = pd.DataFrame(index=panel_dates, columns=syms, dtype=float)
    for i, t in enumerate(panel_dates):
        t1 = rebal[i + 1]
        fwd.loc[t] = (px.loc[t1] / px.loc[t] - 1.0).reindex(syms).to_numpy(dtype=float)
    if fwd_return_cap is not None:
        if fwd_return_cap <= 0:
            raise ValueError(f"fwd_return_cap must be positive, got {fwd_return_cap!r}")
        fwd = fwd.clip(lower=-fwd_return_cap, upper=fwd_return_cap)
    if universe_mask is not None:
        aligned = universe_mask.astype(bool).reindex(
            index=panel_dates, columns=syms, fill_value=False
        )
        fwd = fwd.where(aligned)

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
    )


def print_report(report: ResearchReport) -> None:
    """Print the honest, default-deny research report."""
    res = report.result
    d0 = report.date_start.date() if report.date_start is not None else "n/a"
    d1 = report.date_end.date() if report.date_end is not None else "n/a"
    print(f"\n===========  LEARNED fundamental alpha - {report.label} "
          f"(survivorship-free, net of {DEFAULT_COST_BPS:.0f}bps, monthly)  ===========")
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
    print("=" * 92)


# ── Synthetic data (offline self-test + tests) ───────────────────────────────────────
def write_synthetic_csvs(
    out_dir: Path,
    *,
    seed: int = 7,
    edge: bool = True,
    n_tickers: int = 24,
    n_years: int = 8,
    start: str = "2012-01-06",
    sigma: float = 0.020,
    edge_amp: float = 0.006,
) -> tuple[Path, Path]:
    """Write deterministic SYNTHETIC Sharadar SF1 + SEP CSVs (no network, seeded).

    Prices are emitted WEEKLY (a faithful, CI-cheap stand-in for daily SEP — monthly
    rebalances and the YoY / 12-1 lookback tolerances all resolve on a weekly grid). Every
    one of the 14 factors is given genuine cross-sectional variance so the learner is not
    starved. When ``edge=True`` a clean cross-sectional edge is injected: the price drift of
    each name is proportional to its (constant) ROE, so ROE — and, since prices accumulate
    that drift, 12-1 momentum — predict the forward return. When ``edge=False`` all drifts are
    zero (pure random walk) so no factor predicts returns and the gate must DEFAULT-DENY.
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

    # ── SEP weekly prices ──
    dates = pd.date_range(start=start, periods=n_years * 52, freq="W-FRI")
    mu = edge_amp * z_roe if edge else np.zeros(n_tickers)
    eps = rng.standard_normal((len(dates), n_tickers))
    weekly = mu[None, :] + sigma * eps
    prices = 100.0 * np.cumprod(1.0 + weekly, axis=0)
    sep_rows = []
    for j, tk in enumerate(tickers):
        for di, d in enumerate(dates):
            px = float(prices[di, j])
            sep_rows.append({"ticker": tk, "date": d.strftime("%Y-%m-%d"),
                             "close": px, "closeadj": px, "volume": 1_000_000.0})
    sep_path = out_dir / "SHARADAR_SEP_synthetic.csv"
    pd.DataFrame.from_records(sep_rows).to_csv(sep_path, index=False)

    # ── SF1 quarterly fundamentals (datekey lags calendardate by 45d → PIT exercised) ──
    quarter_ends = pd.date_range(start=dates[0], end=dates[-1], freq="QE")
    sf1_rows = []
    for q, ce in enumerate(quarter_ends):
        datekey = ce + pd.Timedelta(days=45)
        for j, tk in enumerate(tickers):
            equity = e0[j] * (1.0 + g_e[j]) ** q
            assets = a0[j] * (1.0 + g_a[j]) ** q
            revenue = r0[j] * (1.0 + g_r[j]) ** q
            shares = s0[j] * (1.0 + g_s[j]) ** q
            netinc = roe[j] * equity                       # roe = netinc/equity = roe[j]
            gp = gpm[j] * revenue
            ebit = opm[j] * revenue
            ncfo = netinc - accr[j] * assets               # accruals = (netinc-ncfo)/assets
            debt = de[j] * equity
            sf1_rows.append({
                "ticker": tk, "dimension": "ARQ",
                "datekey": datekey.strftime("%Y-%m-%d"), "calendardate": ce.strftime("%Y-%m-%d"),
                "revenue": revenue, "netinc": netinc, "equity": equity, "assets": assets,
                "liabilities": assets - equity, "eps": netinc / shares,
                "ebit": ebit, "ebitda": ebit * 1.3, "gp": gp, "ncfo": ncfo,
                "debt": debt, "sharesbas": shares,
            })
    sf1_path = out_dir / "SHARADAR_SF1_synthetic.csv"
    pd.DataFrame.from_records(sf1_rows).to_csv(sf1_path, index=False)
    return sf1_path, sep_path


def load_panel(
    data_dir: Path,
    *,
    sf1_glob: str = DEFAULT_SF1_GLOB,
    sep_glob: str = DEFAULT_SEP_GLOB,
    dimension: str = DEFAULT_DIMENSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate + load the Sharadar SF1 / SEP files inside ``data_dir`` (glob-matched)."""
    data_dir = Path(data_dir)
    sf1_files = sorted(data_dir.glob(sf1_glob))
    sep_files = sorted(data_dir.glob(sep_glob))
    if not sf1_files:
        raise FileNotFoundError(f"No SF1 files matching {sf1_glob!r} under {data_dir}")
    if not sep_files:
        raise FileNotFoundError(f"No SEP files matching {sep_glob!r} under {data_dir}")
    sf1 = load_sf1(sf1_files, dimension=dimension)
    sep = load_sep(sep_files)
    return sf1, sep


def selftest() -> int:
    """Run the WHOLE flow end-to-end on a small synthetic panel, with NO network or paid data.

    Proves the pipeline (a) recovers an injected edge and reports DEPLOYABLE, and (b) honestly
    DEFAULT-DENIES pure noise. Returns 0 on success, 1 on any failed expectation."""
    print("[selftest] building synthetic survivorship-free Sharadar panel (deterministic)...")
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        edge_dir = tmp / "edge"
        edge_dir.mkdir()
        write_synthetic_csvs(edge_dir, seed=7, edge=True)
        sf1, sep = load_panel(edge_dir)
        edge_report = run_research(sf1, sep, label="SELFTEST edge")
        print_report(edge_report)
        if not edge_report.deployable:
            print("[selftest] FAIL: injected edge was not found DEPLOYABLE.")
            ok = False
        if edge_report.result.deflated_sharpe_ratio < DSR_CUTOFF:
            print("[selftest] FAIL: injected edge did not clear the DSR cutoff.")
            ok = False

        noise_dir = tmp / "noise"
        noise_dir.mkdir()
        write_synthetic_csvs(noise_dir, seed=11, edge=False)
        sf1n, sepn = load_panel(noise_dir)
        noise_report = run_research(sf1n, sepn, label="SELFTEST noise")
        print_report(noise_report)
        if noise_report.deployable:
            print("[selftest] FAIL: pure noise was (wrongly) found DEPLOYABLE.")
            ok = False

    print(f"\n[selftest] {'PASS - pipeline recovers edge and default-denies noise.' if ok else 'FAILED.'}")
    return 0 if ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Directory holding the Sharadar SF1 + SEP CSV export.")
    p.add_argument("--sf1-glob", default=DEFAULT_SF1_GLOB, help="Glob for the SF1 file(s) in --data-dir.")
    p.add_argument("--sep-glob", default=DEFAULT_SEP_GLOB, help="Glob for the SEP file(s) in --data-dir.")
    p.add_argument("--dimension", default=DEFAULT_DIMENSION,
                   help="SF1 dimension (PIT-safe As-Reported: ARQ/ART/ARY). Default ARQ.")
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                   help="History before the first rebalance (>365 so YoY + momentum exist).")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="Per-rebalance cost drag (bps) subtracted from OOS returns.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the whole flow on synthetic data (no paid data); exit 0 on success.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if args.data_dir is None:
        print("error: --data-dir is required (or pass --selftest for the offline check).",
              file=sys.stderr)
        return 2
    sf1, sep = load_panel(args.data_dir, sf1_glob=args.sf1_glob, sep_glob=args.sep_glob,
                          dimension=args.dimension)
    report = run_research(sf1, sep, label=str(args.data_dir), dimension=args.dimension,
                          warmup_days=args.warmup_days, cost_bps=args.cost_bps)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
