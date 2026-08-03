"""INSIDER-TRANSACTIONS alpha research runner — does a LEARNED combination of PIT-safe
SEC Form 4 insider-trading features carry a deflation-surviving edge at moderate breadth?

This is the insider-data sibling of ``scripts/research_free_alpha.py`` and mirrors its
spine EXACTLY: universe -> free yfinance monthly prices -> monthly rebalance dates ->
feature grid -> forward returns (t to t+1, no look-ahead) -> purged walk-forward ->
``learn_signal_weights`` -> ``selection_rule`` gate -> honest DEFAULT-DENY verdict.

The study is PRE-REGISTERED: hypothesis, the FIXED 5-feature set, universe, dates,
gate and decision rule were written down BEFORE any real-data run in
``research/medallion_style_alpha_search/insider_study_prereg.md``. No features may be
added after seeing results; any FAIL is banked NOT-DEPLOYABLE.

Pipeline specifics:
  1. Universe: ``--tickers`` / ``--universe-file`` or the free runner's 141-name
     ``DEFAULT_UNIVERSE`` (no larger curated list exists in this repo; reusing it keeps
     the study comparable with the banked fundamentals results; the prereg's "140-name"
     was a mis-count — see its erratum).
  2. Insider data: quarterly SEC ``form345`` ZIPs under ``--raw-dir``
     (``data.insider_ingestion``; as-filed Form 4 only, parquet-cached). The ONLY
     availability timestamp is FILING_DATE; features apply filing_date + 1 business day.
     Transactions join the universe RENAME-SAFELY by issuer CIK
     (``research.insider_universe`` — the first run's as-filed-ticker join lost ~22% of
     matched rows across renames like GOOG→GOOGL/FB→META).
  3. Features: ``research.insider_features`` — 5 pre-registered monthly panels,
     per-date cross-sectionally z-scored; missing = NaN, neutral-filled 0.0 at the
     combination layer only (matching research_free_alpha conventions).
  4. Gate: ``learn_signal_weights`` (ridge, purged walk-forward) + ``selection_rule``
     (Bailey-Lopez de Prado Deflated Sharpe >= 0.95 cutoff) + PBO (CSCV) diagnostic.
     n_trials = 8, counted HONESTLY: 5 pre-registered features + the naive equal-weight
     composite + the learned ridge combination + this runner's single configuration.

SURVIVORSHIP CAVEAT (read this): prices are CURRENT-LISTED yfinance closes — delisted /
acquired / bankrupt names are absent. Insider-purchase signals are strongest exactly in
the small/distressed names most likely to have delisted, so the bias here is OPTIMISTIC
(it helps the signal). Therefore a FAIL on this study is robust evidence of no edge,
while a PASS would be provisional and would REQUIRE survivorship-free prices before any
deployment decision.

Run (real data):    python scripts/research_insider_alpha.py            (network: yfinance)
Offline self-test:  python scripts/research_insider_alpha.py --selftest (no network)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
import zipfile
import zlib
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.insider_ingestion import load_insider_transactions  # noqa: E402
from research.alpha_factory import learn_signal_weights  # noqa: E402
from research.insider_features import (  # noqa: E402
    INSIDER_FEATURES,
    compute_insider_features,
    raw_insider_features,
)
from research.insider_universe import (  # noqa: E402
    build_universe_cik_map,
    map_transactions_to_universe,
)
from research.validation import (  # noqa: E402
    PurgedWalkForwardSplitter,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from scripts.research_free_alpha import (  # noqa: E402
    DEFAULT_UNIVERSE,
    ResearchReport,
    _price_matrix,
    _rebalance_dates,
    _single_factor_returns,
    fetch_free_prices,
    load_universe,
)

# ── Defaults / constants ─────────────────────────────────────────────────────────────
DSR_CUTOFF = 0.95                  # the real Bailey-Lopez de Prado DSR gate in selection_rule
DEFAULT_COST_BPS = 10.0            # per-rebalance cost drag subtracted from OOS returns
PERIODS_PER_YEAR = 12              # monthly rebalances
DEFAULT_RAW_DIR = Path("_data/insider/raw")
DEFAULT_START = "2006-01-01"       # 2006 is BURN-IN for the trailing 6m/3-year windows...
DEFAULT_WARMUP_DAYS = 335          # ...so the first rebalance lands 2007-01 (pre-registered)
DEFAULT_PRICE_INTERVAL = "1mo"
DEFAULT_RESULT_PATH = Path("research/medallion_style_alpha_search/insider_alpha_result.md")

# HONEST trial count for DSR deflation: 5 pre-registered features + naive equal-weight
# composite + learned ridge combination + this runner's one configuration = 8, PLUS the
# 2026-07-11 first run whose results were seen before the join-defect correction
# (ticker-rename join loss; see the prereg erratum) = 9. Any further configuration or
# corrected re-run must raise this again.
N_TRIALS = 9

# Audited overrides for the rename-safe universe mapping, filled from the
# build_universe_cik_map audit against the rebuilt parquet (2026-07-13; evidence =
# issuer names + filing-date ranges in the audit CSV written next to the verdict).
#
# extra: cross-CIK reorg predecessors the symbol bridge cannot reach. Google Inc
# (CIK 1288776, 75,600 rows, last filing 2015-10-05, only ever symbol GOOG) hands off
# to Alphabet Inc (CIK 1652044, first filing 2015-10-08) at the October-2015 reorg.
AUDITED_EXTRA_CIKS: dict[str, list[str]] = {"GOOGL": ["1288776"]}
# exclude: CIKs that filed under a universe symbol but are NOT the listed issuer
# (recycled/coincident tickers, mis-filed symbols, unlisted subsidiaries, or a
# predecessor security with no continuous price lineage). Each verified by issuer
# name + date range in the audit table:
AUDITED_EXCLUDE_CIKS: list[str] = [
    "1387156",  # AirXpanders Inc (AXP.AX)         -> not American Express
    "1043325",  # Centerline Holding / CharterMac  -> not Citigroup
    "1108967",  # CUI Global / Orbital Energy      -> not Citigroup
    "1201135",  # Credit One Financial (COFI)      -> not Capital One
    "712537",   # First Commonwealth Fin. (FCF)    -> not Ford
    "1022321",  # Genesis Energy LP (GEL, 'GE:')   -> not General Electric
    "1023052",  # Linens 'n Things                 -> not Linde
    "1575571",  # LIN Media LLC                    -> not Linde
    "61398",    # Magellan Petroleum / Tellurian   -> not Marathon Petroleum
    "1364856",  # Monarch Financial (MNRK)         -> not Merck
    "1108924",  # OPNET Technologies (OPNT)        -> not Realty Income
    "836267",   # Taiwan Greater China Fund        -> not Truist
    "946115",   # Target Receivables Corp (SPV)    -> unlisted Target subsidiary
    "1097609",  # T-Mobile USA Inc (subsidiary)    -> not the listed T-Mobile US
    "40730",    # General Motors CORP (pre-2009 bankruptcy; no continuous GM price
                #   lineage - new GM CIK 1467858 IPO'd 2010-11)
]
# Documented merger-seam imperfections (kept: excluding the CIK would lose far more
# correct history than the seam adds): Merck CIK 310158 carries Schering-Plough's
# 2006-09 rows (as-filed SGP); Prologis CIK 1045609 carries AMB's pre-2011 rows.

INSIDER_SURVIVORSHIP_CAVEAT = (
    "SURVIVORSHIP CAVEAT (current-listed prices; read this):\n"
    "  Prices are CURRENT-LISTED yfinance closes; delisted/acquired/bankrupt names are\n"
    "  absent. Insider purchases predict best in exactly the small/distressed names most\n"
    "  likely to have delisted, so this bias is OPTIMISTIC (it helps the signal):\n"
    "  a FAIL here is robust; a PASS is provisional and requires survivorship-free\n"
    "  prices before any deployment."
)


def _to_month_end_labels(px: pd.DataFrame) -> pd.DataFrame:
    """Relabel a monthly price matrix's index to each month's calendar month-end.

    yfinance ``interval='1mo'`` bars are LABELLED at the first of the month while their
    Close is the month-END price (empirically: the 2020-01-01 bar's Close equals the
    2020-01-31 daily close). The insider feature bucketing keys a filing's month off the
    panel-date LABEL (:func:`research.insider_features._effective_periods`), so a
    month-START label would push essentially every mid-month filing an extra month forward
    -- adding ~1 month of lag beyond the pre-registered ``filing_date + 1bd -> that month's
    month-end`` rule, attenuating a decaying insider signal and biasing the verdict toward
    a false NOT-DEPLOYABLE. Mapping every label to its calendar month-end makes the label
    agree with the price it already carries. Idempotent for an index that is already
    month-end (the synthetic ``freq='ME'`` grids used in tests are unchanged); raises on the
    monthly-uniqueness assumption being violated so a mixed/daily index cannot pass silently.
    """
    idx = pd.DatetimeIndex(px.index)
    month_end = idx + pd.offsets.MonthEnd(0)
    if pd.PeriodIndex(idx, freq="M").duplicated().any():
        raise ValueError(
            "insider price matrix must carry at most one bar per calendar month "
            "(monthly bars expected); got multiple rows in a month"
        )
    out = px.copy()
    out.index = month_end
    return out.sort_index()


def _drop_partial_last_bar(px: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop a trailing month bar whose month-end label lies in the future of ``asof``
    (default: today). yfinance serves the CURRENT month as a partial bar mid-month;
    after :func:`_to_month_end_labels` it is labelled at the month-END, so the final
    forward return would silently span only the elapsed fraction of the month (the
    2026-07-11 run's last OOS point was a mislabeled ~7-trading-day July return)."""
    cutoff = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof)
    if len(px.index) and pd.Timestamp(px.index[-1]) > cutoff:
        return px.iloc[:-1]
    return px


# ── Core research flow (mirrors research_free_alpha.run_research) ────────────────────
def run_research(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    label: str = "SEC Form 4 + yfinance",
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
    asof: pd.Timestamp | None = None,
) -> ResearchReport:
    """Build insider features + forward returns, learn the gated combination, report.

    Honest + default-deny by construction: degenerate / no-edge inputs flow through
    ``learn_signal_weights`` (which fails closed) into a NOT-DEPLOYABLE verdict.
    Transactions are joined to the price universe RENAME-SAFELY by issuer CIK
    (``research.insider_universe``); a trailing partial-month price bar is dropped.

    Parameters
    ----------
    transactions : tidy Form-4 frame from :func:`data.insider_ingestion.load_insider_transactions`.
    prices : tidy ``(ticker, date, price)`` frame (free yfinance closes, or synthetic).
    asof : run date for the partial-bar guard (default today; injectable for tests).
    """
    px = _drop_partial_last_bar(_to_month_end_labels(_price_matrix(prices)), asof)
    syms = list(px.columns)
    transactions = map_transactions_to_universe(
        transactions, syms,
        extra_ciks=AUDITED_EXTRA_CIKS, exclude_ciks=AUDITED_EXCLUDE_CIKS,
    )
    rebal = _rebalance_dates(px, warmup_days)
    panel_dates = rebal[:-1]
    n_dates = len(panel_dates)

    features = (
        compute_insider_features(transactions, panel_dates, syms)
        if n_dates
        else pd.DataFrame(columns=["ticker", "date", *INSIDER_FEATURES])
    )
    grid = {
        f: features.pivot_table(index="date", columns="ticker", values=f, aggfunc="last")
        .reindex(index=panel_dates, columns=syms)
        .astype(float)
        for f in INSIDER_FEATURES
    }
    # Learner inputs: neutral-fill residual NaNs (no insider activity = neutral).
    panel = {f: grid[f].fillna(0.0) for f in INSIDER_FEATURES}

    fwd = pd.DataFrame(index=panel_dates, columns=syms, dtype=float)
    for i, t in enumerate(panel_dates):
        t1 = rebal[i + 1]
        fwd.loc[t] = (px.loc[t1] / px.loc[t] - 1.0).reindex(syms).to_numpy(dtype=float)

    test = max(3, n_dates // 5)
    splitter = PurgedWalkForwardSplitter(
        train_size=max(6, n_dates - 3 * test),
        valid_size=test,
        test_size=test,
        embargo_size=1,
        label_horizon=1,
    )
    weights, result = learn_signal_weights(
        panel,
        fwd,
        splitter=splitter,
        n_trials=N_TRIALS,
        cost_drag_bps=cost_bps,
        periods_per_year=PERIODS_PER_YEAR,
    )

    # Naive equal-weight composite of the RAW (un-filled) standardized features.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comp_arr = (
            np.nanmean(
                np.stack([grid[f].to_numpy(dtype=float) for f in INSIDER_FEATURES]), axis=0
            )
            if n_dates
            else np.empty((0, len(syms)))
        )
    comp_df = pd.DataFrame(comp_arr, index=panel_dates, columns=syms)
    comp_ret = [
        r
        for t in panel_dates
        if (r := _single_factor_returns(comp_df.loc[t].fillna(0.0), fwd.loc[t], cost_bps))
        is not None
    ]
    comp_a = np.asarray(comp_ret, dtype=float)
    comp_dsr = float(deflated_sharpe_ratio(comp_a, n_trials=N_TRIALS)) if comp_a.size >= 4 else 0.0
    comp_sharpe = (
        float(comp_a.mean() / comp_a.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        if comp_a.size > 1 and comp_a.std(ddof=1) > 0
        else 0.0
    )

    # PBO (CSCV) across the pre-registered single-feature library.
    cols = []
    for f in INSIDER_FEATURES:
        col = [
            _single_factor_returns(panel[f].loc[t], fwd.loc[t], cost_bps) or 0.0
            for t in panel_dates
        ]
        cols.append(col)
    perf = np.asarray(cols, dtype=float).T
    pbo = float(probability_of_backtest_overfitting(perf)) if perf.shape[0] >= 4 else 0.5

    return ResearchReport(
        label=label,
        n_symbols=len(syms),
        n_rebalances=n_dates,
        test_size=test,
        n_trials=N_TRIALS,
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
    print(f"\n===========  LEARNED insider alpha - {report.label} "
          f"(net of {report.cost_bps:.0f}bps, monthly)  ===========")
    print(f"universe: {report.n_symbols} names | {d0}..{d1} | "
          f"rebalances: {report.n_rebalances} | walk-forward test={report.test_size} | "
          f"DSR trials={report.n_trials}")
    print(f"features ({len(INSIDER_FEATURES)}, pre-registered): {', '.join(INSIDER_FEATURES)}")
    print("-" * 92)
    print("LEARNED ridge combination (out-of-sample, purged walk-forward):")
    print(f"  OOS IC={res.mean_ic:+.4f}  rank-IC={res.mean_rank_ic:+.4f}  "
          f"net_Sharpe={res.sharpe_net:.2f}  stability={res.stability_score:.2f}")
    print(f"  Deflated Sharpe={res.deflated_sharpe_ratio:.3f}  (cutoff {DSR_CUTOFF:.2f})   "
          f"PBO={report.pbo:.2f}  (CSCV, lower is better)")
    if res.leakage_flags:
        print(f"  leakage_flags: {res.leakage_flags}")
    print("  weights: " + ", ".join(f"{k}={report.weights.get(k, 0.0):+.3f}"
                                    for k in INSIDER_FEATURES))
    print(f"  selection_rule -> {'PASS' if report.deployable else 'FAIL (default-deny)'}")
    print("-" * 92)
    print(f"NAIVE equal-weight composite:  net_Sharpe={report.comp_sharpe:.2f}  "
          f"DSR={report.comp_dsr:.3f}")
    print("-" * 92)
    if report.deployable:
        print("VERDICT: DEPLOYABLE (provisional) - passes the gate on current-listed "
              "prices; survivorship-free replication is REQUIRED before deployment.")
    else:
        print("VERDICT: NOT-DEPLOYABLE (default-deny) - no robust edge survives deflation.")
    print("-" * 92)
    print(INSIDER_SURVIVORSHIP_CAVEAT)
    print("=" * 92)


def write_result_markdown(report: ResearchReport, path: Path) -> None:
    """Write the verdict document (same shape as the prior result docs in
    ``research/medallion_style_alpha_search/``)."""
    res = report.result
    d0 = report.date_start.date() if report.date_start is not None else "n/a"
    d1 = report.date_end.date() if report.date_end is not None else "n/a"
    verdict = ("DEPLOYABLE (provisional - survivorship-free replication required)"
               if report.deployable else "NOT-DEPLOYABLE")
    rule = "PASS" if report.deployable else "FAIL (default-deny)"
    weights = ", ".join(f"{k}={report.weights.get(k, 0.0):+.3f}" for k in INSIDER_FEATURES)
    body = f"""# Insider (SEC Form 4) transactions alpha study — real-data result ({_date.today().isoformat()})

> **SURVIVORSHIP NOTE (header, by design):** prices are CURRENT-LISTED yfinance closes —
> an OPTIMISTIC bias that *helps* the signal (insider buying predicts best in the small /
> distressed names most likely to have delisted). A **FAIL below is therefore robust**;
> a PASS would be provisional and would require survivorship-free prices before any
> deployment.

**Question:** do PIT-safe SEC Form 4 insider-transaction features (net buying, clustered
buying, opportunistic i.e. non-routine buying) carry a deflation-surviving 1-month
cross-sectional edge? Pre-registered BEFORE this run in `insider_study_prereg.md`
(fixed 5-feature set, gate, decision rule — no post-hoc feature additions).

**Method:** `scripts/research_insider_alpha.py` -> `research.alpha_factory.learn_signal_weights`
(ridge, purged walk-forward) gated by `research.validation.selection_rule`
(Deflated Sharpe >= {DSR_CUTOFF:.2f}) + PBO (CSCV) diagnostic; net of
{report.cost_bps:.0f} bps per monthly rebalance.

- **Universe:** {report.n_symbols} current-listed names, {d0} -> {d1},
  **{report.n_rebalances} monthly rebalances**, walk-forward test window {report.test_size}
  per fold, DSR deflated for **{report.n_trials} trials** (5 pre-registered features +
  naive composite + learned combination + 1 runner configuration + the seen-then-corrected
  2026-07-11 first run; see the prereg erratum).
- **Join:** transactions matched to the universe RENAME-SAFELY by issuer CIK
  (`research.insider_universe`; the 2026-07-11 first run's as-filed-ticker join silently
  lost ~22% of matched rows across renames — GOOG/FB/UTX/PCLN/WLP/MHP/FPL/KFT and
  pre-2009 parenthesized symbols).
- **Features (5, fixed a priori):** {", ".join(INSIDER_FEATURES)}. Multi-owner filings
  contribute dollar value once (accession dedup); counts stay per reporting owner.
- **PIT discipline:** availability = FILING_DATE + 1 business day; as-filed Form 4 only
  (4/A amendments excluded); month-end panel assignment proven leak-free by `--selftest`;
  a trailing partial-month price bar is dropped (no mislabeled short forward return).

## Result

| Combination | OOS IC | rank-IC | net Sharpe | stability | **DSR** | PBO | `selection_rule` |
|---|---|---|---|---|---|---|---|
| **Learned ridge** | {res.mean_ic:+.4f} | {res.mean_rank_ic:+.4f} | {res.sharpe_net:.2f} | {res.stability_score:.2f} | **{res.deflated_sharpe_ratio:.3f}** | {report.pbo:.2f} | **{rule}** |
| Naive equal-weight | — | — | {report.comp_sharpe:.2f} | — | {report.comp_dsr:.3f} | — | — |

Learned weights: {weights}

**VERDICT: {verdict}**

## Decision rule (pre-registered)

Any FAIL = banked NOT-DEPLOYABLE — no feature additions, no window re-tuning, no
universe swaps after seeing this table. A PASS is provisional until replicated on
survivorship-free prices (the current-listed bias above is in the signal's favour).

## Interpretation caveats (registered from the 2026-07 adversarial review, verdict-independent)

1. **Power:** at ~92 OOS months and this trial count, DSR >= {DSR_CUTOFF:.2f} requires an
   OBSERVED annualized Sharpe of ~1.1; simulated power for the pre-registered realistic
   effect size (~2-5%/yr long-tilt) is ~1-30%. A FAIL is therefore **"cannot certify a
   deployable edge"** — it is NOT evidence the (small) literature effect is absent.
2. **Scope:** on this mega-cap universe insiders overwhelmingly sell (~97% of qualifying
   events), so the ratio features are mostly binary (69% of finite cells pinned at -1)
   and the FAIL generalizes to *insider net-buy indicators on ~140 mega-caps*, not to
   insider signals on richer cross-sections (small caps / breadth).
3. **Opportunistic channel not discriminated:** the Cohen-Malloy-Pomorski routine strip
   fires on ~0.05% of events here (mega-cap insiders almost never buy on a fixed annual
   schedule), so `opportunistic_buy_6m` is a near-duplicate of `net_buy_ratio_6m`; the
   CMP non-routine hypothesis was NOT independently tested on this universe.
4. **OOS window:** the walk-forward verdict rests on the LAST ~92 months only (2 folds);
   earlier history is training-only. The learner's 46-month validation block between
   train and test is computed but unused (fixed l2) — a harness inefficiency for future
   studies, bounded here by the staleness-free naive composite failing independently.
5. The naive-composite DSR row is an in-sample full-length statistic (T={report.n_rebalances}),
   not an OOS one; it is a control, not a second candidate.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ── Synthetic data + self-test (offline, no network) ─────────────────────────────────
_SUB_HEADER = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB",
               "NO_SECURITIES_OWNED", "NOT_SUBJECT_SEC16", "FORM3_HOLDINGS_REPORTED",
               "FORM4_TRANS_REPORTED", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
               "ISSUERTRADINGSYMBOL", "REMARKS"]
_OWN_HEADER = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP",
               "RPTOWNER_TITLE", "RPTOWNER_TXT", "RPTOWNER_STREET1", "RPTOWNER_STREET2",
               "RPTOWNER_CITY", "RPTOWNER_STATE", "RPTOWNER_ZIPCODE", "RPTOWNER_STATE_DESC",
               "FILE_NUMBER"]
_TRN_HEADER = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "SECURITY_TITLE_FN",
               "TRANS_DATE", "TRANS_DATE_FN", "DEEMED_EXECUTION_DATE",
               "DEEMED_EXECUTION_DATE_FN", "TRANS_FORM_TYPE", "TRANS_CODE",
               "EQUITY_SWAP_INVOLVED", "EQUITY_SWAP_TRANS_CD_FN", "TRANS_TIMELINESS",
               "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN",
               "TRANS_PRICEPERSHARE", "TRANS_PRICEPERSHARE_FN", "TRANS_ACQUIRED_DISP_CD",
               "TRANS_ACQUIRED_DISP_CD_FN", "SHRS_OWND_FOLWNG_TRANS",
               "SHRS_OWND_FOLWNG_TRANS_FN", "VALU_OWND_FOLWNG_TRANS",
               "VALU_OWND_FOLWNG_TRANS_FN", "DIRECT_INDIRECT_OWNERSHIP",
               "DIRECT_INDIRECT_OWNERSHIP_FN", "NATURE_OF_OWNERSHIP",
               "NATURE_OF_OWNERSHIP_FN"]


def _sec_date(d: pd.Timestamp) -> str:
    return d.strftime("%d-%b-%Y").upper()


def write_fake_quarter_zip(path: Path, filings: list[dict[str, object]]) -> Path:
    """A synthetic SEC ``form345`` quarterly ZIP with the EXACT TSV headers. Each filing
    dict: ticker, filing_date (Timestamp), code ('P'/'S'), owner, relationship, shares,
    price (optional), doc_type (optional, default '4')."""
    subs, owners, trans = [], [], []
    for i, f in enumerate(filings):
        acc = f"0000000000-00-{i:06d}"
        fd = _sec_date(pd.Timestamp(str(f["filing_date"])))
        sub = {c: "" for c in _SUB_HEADER}
        # Per-ticker issuer CIK (stable, deterministic): the CIK-bridged universe join
        # must see distinct synthetic issuers, not one CIK shared by every ticker.
        cik = str(zlib.crc32(str(f["ticker"]).encode()) % 10**9 + 1).zfill(10)
        sub.update(ACCESSION_NUMBER=acc, FILING_DATE=fd, PERIOD_OF_REPORT=fd,
                   DOCUMENT_TYPE=str(f.get("doc_type", "4")), ISSUERCIK=cik,
                   ISSUERNAME="Synthetic Issuer", ISSUERTRADINGSYMBOL=str(f["ticker"]))
        own = {c: "" for c in _OWN_HEADER}
        own.update(ACCESSION_NUMBER=acc, RPTOWNERCIK=str(f["owner"]),
                   RPTOWNERNAME="Synthetic Insider",
                   RPTOWNER_RELATIONSHIP=str(f.get("relationship", "Officer")))
        trn = {c: "" for c in _TRN_HEADER}
        trn.update(ACCESSION_NUMBER=acc, NONDERIV_TRANS_SK=str(i),
                   SECURITY_TITLE="Common Stock", TRANS_DATE=fd, TRANS_FORM_TYPE="4",
                   TRANS_CODE=str(f["code"]), EQUITY_SWAP_INVOLVED="0",
                   TRANS_SHARES=str(f.get("shares", 100.0)),
                   TRANS_PRICEPERSHARE=str(f.get("price", 10.0)),
                   TRANS_ACQUIRED_DISP_CD="A" if f["code"] == "P" else "D",
                   SHRS_OWND_FOLWNG_TRANS="1000.0", DIRECT_INDIRECT_OWNERSHIP="D")
        subs.append(sub)
        owners.append(own)
        trans.append(trn)

    def _tsv(header: list[str], rows: list[dict[str, str]]) -> str:
        out = ["\t".join(header)]
        out += ["\t".join(r[c] for c in header) for r in rows]
        return "\n".join(out) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("SUBMISSION.tsv", _tsv(_SUB_HEADER, subs))
        z.writestr("REPORTINGOWNER.tsv", _tsv(_OWN_HEADER, owners))
        z.writestr("NONDERIV_TRANS.tsv", _tsv(_TRN_HEADER, trans))
    return path


def _selftest_pit(tmp: Path) -> bool:
    """A filing dated ON month-end t must NOT influence the t signal. Constructed so a
    leak would FLIP the measured payoff: the bought name jumps +20% the month AFTER the
    filing and -20% the month after that — a leaked signal harvests the +20%, the correct
    signal (first active one month later) harvests only the -20%."""
    print("[selftest:pit] filing on 2020-06-30 (month-end) -> usable at July, not June")
    raw_dir = tmp / "pit" / "raw"
    write_fake_quarter_zip(raw_dir / "2020q2_form345.zip", [
        {"ticker": "AAA", "filing_date": "2020-06-30", "code": "P", "owner": "11"},
        {"ticker": "BBB", "filing_date": "2020-06-30", "code": "S", "owner": "22"},
    ])
    txns = load_insider_transactions(raw_dir)

    dates = list(pd.date_range("2020-01-31", "2020-12-31", freq="ME"))
    grid = raw_insider_features(txns, dates, ["AAA", "BBB", "CCC"])
    june = grid[grid["date"] == pd.Timestamp("2020-06-30")].set_index("ticker")
    july = grid[grid["date"] == pd.Timestamp("2020-07-31")].set_index("ticker")
    ok = True
    if not np.isnan(june.loc["AAA", "net_buy_ratio_6m"]):
        print("[selftest:pit] FAIL: month-end filing visible in its own month (LEAK).")
        ok = False
    if july.loc["AAA", "net_buy_ratio_6m"] != 1.0 or july.loc["BBB", "net_buy_ratio_6m"] != -1.0:
        print("[selftest:pit] FAIL: filing not visible the following month.")
        ok = False

    # Payoff sign check: +20%/-20% for AAA (and mirrored BBB) around the filing.
    ret = {t: {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0} for t in dates[:-1]}
    ret[pd.Timestamp("2020-06-30")] = {"AAA": 0.20, "BBB": -0.20, "CCC": 0.0}
    ret[pd.Timestamp("2020-07-31")] = {"AAA": -0.20, "BBB": 0.20, "CCC": 0.0}
    rows = []
    lvl = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    for i, t in enumerate(dates):
        for s in ("AAA", "BBB", "CCC"):
            if i > 0:
                lvl[s] *= 1.0 + ret[dates[i - 1]][s]
            rows.append({"ticker": s, "date": t, "price": lvl[s]})
    px = _price_matrix(pd.DataFrame(rows))
    total = 0.0
    for i, t in enumerate(dates[:-1]):
        sig = grid[grid["date"] == t].set_index("ticker")["net_buy_ratio_6m"]
        sig = sig.reindex(["AAA", "BBB", "CCC"]).fillna(0.0)
        fwd = (px.loc[dates[i + 1]] / px.loc[t] - 1.0).reindex(["AAA", "BBB", "CCC"])
        r = _single_factor_returns(sig, fwd, 0.0)
        total += r if r is not None else 0.0
    if total >= -0.1:
        print(f"[selftest:pit] FAIL: long-short payoff {total:+.3f} not negative -- "
              "a leak would push this to ~0 or positive.")
        ok = False
    else:
        print(f"[selftest:pit] OK: payoff {total:+.3f} (correct pipeline harvests only "
              "the post-availability -20%).")
    return ok


def _selftest_routine(tmp: Path) -> bool:
    """Routine stripping: same owner bought each March 2017-2019 -> the March-2020 buy is
    routine and must vanish from opportunistic_buy_6m while net_buy_ratio_6m keeps it."""
    print("[selftest:routine] 4th consecutive same-calendar-month purchase is stripped")
    raw_dir = tmp / "routine" / "raw"
    filings: list[dict[str, object]] = [
        {"ticker": "TTT", "filing_date": f"{y}-03-16", "code": "P", "owner": "9001"}
        for y in (2017, 2018, 2019, 2020)
    ]
    write_fake_quarter_zip(raw_dir / "2020q1_form345.zip", filings)
    txns = load_insider_transactions(raw_dir)
    dates = list(pd.date_range("2019-06-30", "2020-12-31", freq="ME"))
    grid = raw_insider_features(txns, dates, ["TTT"])
    at = grid[grid["date"] == pd.Timestamp("2020-03-31")].iloc[0]
    ok = True
    if at["net_buy_ratio_6m"] != 1.0:
        print("[selftest:routine] FAIL: net_buy_ratio_6m should still see the buy.")
        ok = False
    if not np.isnan(at["opportunistic_buy_6m"]):
        print("[selftest:routine] FAIL: routine buy leaked into opportunistic_buy_6m.")
        ok = False
    if ok:
        print("[selftest:routine] OK: net_buy_ratio=+1.0, opportunistic=NaN (stripped).")
    return ok


def _build_synthetic_study(
    tmp: Path, *, seed: int, edge: bool, n_symbols: int = 24, n_months: int = 96
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(transactions, prices) for the gate check, ROUND-TRIPPED through a fake quarterly
    ZIP (full ingestion -> features -> gate wiring). With ``edge=True`` insiders of
    high-drift names buy and low-drift names sell (a real, learnable signal); with
    ``edge=False`` prices are pure noise so the SAME transactions carry no information."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i:03d}" for i in range(n_symbols)]
    z = np.linspace(-1.0, 1.0, n_symbols)
    z = (z - z.mean()) / z.std()
    mu = 0.012 * z if edge else np.zeros(n_symbols)
    dates = pd.date_range("2012-01-31", periods=n_months, freq="ME")

    monthly = mu[None, :] + 0.04 * rng.standard_normal((n_months, n_symbols))
    lvl = 100.0 * np.cumprod(1.0 + monthly, axis=0)
    price_rows = [
        {"ticker": s, "date": d, "price": float(lvl[i, j])}
        for i, d in enumerate(dates)
        for j, s in enumerate(symbols)
    ]

    order = np.argsort(z)
    sellers, buyers = order[:6], order[-6:]
    filings: list[dict[str, object]] = []
    for i, d in enumerate(dates):
        filing_day = d.replace(day=10)
        for j in buyers:
            filings.append({"ticker": symbols[j], "filing_date": filing_day, "code": "P",
                            "owner": f"{j}01", "relationship": "Officer"})
            if i % 2 == 0:   # a second DISTINCT buyer on alternate months (cluster path)
                filings.append({"ticker": symbols[j], "filing_date": filing_day,
                                "code": "P", "owner": f"{j}02",
                                "relationship": "Director"})
        for j in sellers:
            filings.append({"ticker": symbols[j], "filing_date": filing_day, "code": "S",
                            "owner": f"{j}01", "relationship": "Officer"})

    raw_dir = tmp / ("edge" if edge else "noise") / "raw"
    write_fake_quarter_zip(raw_dir / "2012q1_form345.zip", filings)
    return load_insider_transactions(raw_dir), pd.DataFrame(price_rows)


def _selftest_gate(tmp: Path) -> bool:
    """End-to-end gate wiring: a planted insider edge PASSES; pure noise is DENIED; the
    verdict markdown is written."""
    print("[selftest:gate] planted-edge and pure-noise runs through the FULL pipeline")
    ok = True

    txns, prices = _build_synthetic_study(tmp, seed=7, edge=True)
    edge_report = run_research(txns, prices, label="SELFTEST edge", warmup_days=200)
    print_report(edge_report)
    if not edge_report.deployable or edge_report.result.deflated_sharpe_ratio < DSR_CUTOFF:
        print("[selftest:gate] FAIL: planted insider edge was not recovered as DEPLOYABLE.")
        ok = False

    txns_n, prices_n = _build_synthetic_study(tmp, seed=11, edge=False)
    noise_report = run_research(txns_n, prices_n, label="SELFTEST noise", warmup_days=200)
    print_report(noise_report)
    if noise_report.deployable:
        print("[selftest:gate] FAIL: pure noise was (wrongly) found DEPLOYABLE.")
        ok = False

    out = tmp / "insider_alpha_result_selftest.md"
    write_result_markdown(noise_report, out)
    text = out.read_text(encoding="utf-8")
    if "NOT-DEPLOYABLE" not in text or "SURVIVORSHIP" not in text:
        print("[selftest:gate] FAIL: verdict markdown missing verdict/caveat.")
        ok = False
    else:
        print(f"[selftest:gate] OK: verdict markdown written ({out.name}).")
    return ok


def selftest() -> int:
    """Offline proof of the four properties the adversarial review will attack:
    PIT lag, routine stripping, end-to-end gate wiring, and noise denial."""
    print("[selftest] synthetic quarterly ZIPs -> ingestion -> features -> gate "
          "(deterministic, no network)...")
    ok = True
    with tempfile.TemporaryDirectory(prefix="insider_selftest_") as td:
        tmp = Path(td)
        ok &= _selftest_pit(tmp)
        ok &= _selftest_routine(tmp)
        ok &= _selftest_gate(tmp)
    outcome = ("PASS - PIT lag, routine stripping, gate wiring and noise denial "
               "all verified." if ok else "FAILED.")
    print(f"\n[selftest] {outcome}")
    return 0 if ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", default=None,
                   help="Comma-separated universe (overrides --universe-file and the default).")
    p.add_argument("--universe-file", type=Path, default=None,
                   help="File with one ticker per line ('#' comments / blanks ignored).")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Directory of quarterly SEC form345 ZIPs (default _data/insider/raw).")
    p.add_argument("--start", default=DEFAULT_START,
                   help="yfinance price history start (2006 = burn-in; panel starts 2007-01).")
    p.add_argument("--end", default=None, help="yfinance price history end (default: today).")
    p.add_argument("--interval", default=DEFAULT_PRICE_INTERVAL,
                   help="yfinance bar interval (default 1mo).")
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS,
                   help="History before the first rebalance (default lands it at 2007-01).")
    p.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                   help="Per-rebalance cost drag (bps) subtracted from OOS returns.")
    p.add_argument("--out", type=Path, default=DEFAULT_RESULT_PATH,
                   help="Verdict markdown path (default the medallion_style_alpha_search doc).")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild the insider parquet cache from the raw ZIPs.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the offline synthetic self-test (no network); exit 0 on success.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()

    universe = load_universe(args.tickers, args.universe_file)   # pragma: no cover - network
    print(INSIDER_SURVIVORSHIP_CAVEAT)
    print(f"universe: {len(universe)} names (default = research_free_alpha DEFAULT_UNIVERSE, "
          f"{len(DEFAULT_UNIVERSE)} names) | insider raw: {args.raw_dir}")
    transactions = load_insider_transactions(args.raw_dir, force=args.force_rebuild)
    if transactions.empty:                                       # pragma: no cover - network
        print("error: no insider transactions parsed; check --raw-dir.", file=sys.stderr)
        return 2
    n_naive = int(transactions["ticker"].isin(set(universe)).sum())
    mapped = map_transactions_to_universe(                       # pragma: no cover - network
        transactions, universe,
        extra_ciks=AUDITED_EXTRA_CIKS, exclude_ciks=AUDITED_EXCLUDE_CIKS)
    print(f"insider rows: {len(transactions):,} total | CIK-bridged universe match: "
          f"{len(mapped):,} (naive exact-symbol join: {n_naive:,}; "
          f"recovered {len(mapped) - n_naive:+,})")
    audit = build_universe_cik_map(                              # pragma: no cover - network
        transactions, universe,
        extra_ciks=AUDITED_EXTRA_CIKS, exclude_ciks=AUDITED_EXCLUDE_CIKS)
    audit_path = args.out.parent / "insider_universe_cik_audit.csv"
    audit.to_csv(audit_path, index=False)                        # pragma: no cover - network
    print(f"universe-CIK audit table written to {audit_path} "
          f"({len(audit)} ticker-CIK pairs; review before banking)")
    prices = fetch_free_prices(universe, start=args.start, end=args.end,   # pragma: no cover
                               interval=args.interval)
    if prices.empty:                                             # pragma: no cover - network
        print("error: no prices fetched for the requested universe.", file=sys.stderr)
        return 2
    report = run_research(transactions, prices,                  # pragma: no cover - network
                          warmup_days=args.warmup_days, cost_bps=args.cost_bps)
    print_report(report)                                         # pragma: no cover - network
    write_result_markdown(report, args.out)                      # pragma: no cover - network
    print(f"verdict written to {args.out}")                      # pragma: no cover - network
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
