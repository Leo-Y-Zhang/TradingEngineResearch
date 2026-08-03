"""THE LONG-HISTORY BOOKS -- the only combinations with an honest sample behind them.

Every combination containing low-vol is capped at 213 months and is entirely inside
low-vol's in-sample DEV window. The combinations that EXCLUDE low-vol run for up to 738
months (61.5 years). If a 61.5-year book clears Sharpe 0.894, that is a far stronger claim
than a 17.75-year one clearing it, because the DSR bar falls with sample length.

This measures the Kelly / leverage / drawdown block for those books, on the same
financing-charged engine, plus the bootstrap drawdown that says how much of the observed
maximum was luck.

    .venv/Scripts/python.exe -m research.sleeves._portfolio.portfolio_longhistory_books
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.multiasset.panel import dsr_sharpe_bar
from research.validation import deflated_sharpe_ratio

from research.sleeves._portfolio.portfolio_correlation_v2 import (
    CASH_PATH, FINANCING, MPY, N_TRIALS_PROGRAMME, SCHEMES, SOURCES, _load, load_source,
    block_bootstrap_sharpe, kelly_block,
)
from research.sleeves._portfolio.portfolio_window_control import (
    dd_bootstrap, solve_leverage_for_dd,
)

OUT_DIR = Path(__file__).resolve().parent
SPREAD = FINANCING["primary_bill_plus_150bp"]
N_SUBSETS_SEARCHED = 58

BOOKS = [
    (["trend", "passive_monthly"], "inverse_vol"),
    (["trend", "passive_monthly"], "erc"),
    (["trend", "passive_monthly"], "equal_weight"),
    (["trend", "seasonal", "passive_monthly"], "erc"),
    (["trend", "seasonal", "defensive", "passive_monthly"], "erc"),
    (["trend", "defensive", "passive_monthly"], "inverse_variance"),
    (["passive_monthly"], "equal_weight"),
    (["trend"], "equal_weight"),
    # the best low-vol books, for a like-for-like comparison on the same engine
    (["lowvol", "trend"], "equal_weight"),
    (["lowvol", "trend", "defensive"], "erc"),
    (["lowvol", "trend", "carry", "defensive"], "equal_weight"),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    cash = _load(CASH_PATH, "US_CASH_13W")
    raw = {k: load_source(k) for k in SOURCES}
    series = {k: ((s - cash.reindex(s.index)).dropna() if SOURCES[k][2] == "total" else s)
              for k, s in raw.items()}

    rows = []
    for combo, scheme in BOOKS:
        f = pd.concat({c: series[c] for c in combo}, axis=1).dropna()
        w = SCHEMES[scheme](f) if len(combo) > 1 else np.array([1.0])
        port = f.to_numpy() @ w
        cs = cash.reindex(f.index).to_numpy()
        blk = kelly_block(port, cs, f"{'+'.join(combo)} [{scheme}]", SPREAD)
        blk["weights"] = {c: float(x) for c, x in zip(f.columns, w)}
        blk["first"], blk["last"] = str(f.index.min().date()), str(f.index.max().date())
        lo, hi = block_bootstrap_sharpe(port)
        blk["sharpe_ci95_block_boot"] = [lo, hi]
        n = len(port)
        se = float(np.sqrt((1 + 0.5 * (blk["sharpe_excess"] / np.sqrt(MPY)) ** 2) / n)
                   * np.sqrt(MPY))
        blk["sharpe_se_analytic"] = se
        blk["sharpe_ci95_analytic"] = [blk["sharpe_excess"] - 1.96 * se,
                                       blk["sharpe_excess"] + 1.96 * se]
        blk["dsr_bar_n46"] = dsr_sharpe_bar(blk["years"], n_trials=N_TRIALS_PROGRAMME)
        blk["dsr_bar_incl_search"] = dsr_sharpe_bar(
            blk["years"], n_trials=N_TRIALS_PROGRAMME + N_SUBSETS_SEARCHED)
        blk["dsr_n46"] = float(deflated_sharpe_ratio(port, n_trials=N_TRIALS_PROGRAMME))
        blk["dsr_incl_search"] = float(deflated_sharpe_ratio(
            port, n_trials=N_TRIALS_PROGRAMME + N_SUBSETS_SEARCHED))
        blk["dd50_observed"] = solve_leverage_for_dd(port, cs, 0.50, use_bootstrap=False)
        blk["dd50_bootstrap"] = solve_leverage_for_dd(port, cs, 0.50, use_bootstrap=True)
        blk["dd60_observed"] = solve_leverage_for_dd(port, cs, 0.60, use_bootstrap=False)
        blk["dd60_bootstrap"] = solve_leverage_for_dd(port, cs, 0.60, use_bootstrap=True)
        blk["half_kelly_dd_bootstrap"] = dd_bootstrap(
            np.asarray(port) * blk["half_kelly_leverage"]
            - max(blk["half_kelly_leverage"] - 1, 0) * SPREAD / MPY + cs)
        # Deliberately empty: the full-sample rung is not computed here. The label
        # tuple is typed rather than the call deleted, so the intended shape stays
        # on record and the comprehension still never runs.
        fs_labels: tuple[str, ...] = ()
        blk["_fs_placeholder"] = {
            lab: solve_leverage_for_dd(port, cs, 0.50, use_bootstrap=False)
            for lab in fs_labels}
        rows.append(blk)

        print(f"  done {blk['label']}")

    (OUT_DIR / "portfolio_longhistory_books.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 128)
    print("LONG-HISTORY vs SHORT-HISTORY BOOKS -- measured covariance, financing at bill+150bp")
    print("=" * 128)
    print(f"{'book':>48} {'n':>5} {'yrs':>6} {'S':>7} {'CI95 analytic':>16} "
          f"{'DSR bar':>8} {'bar+srch':>9} {'DSR':>6} {'>=.894':>7}")
    for b in rows:
        print(f"{b['label']:>48} {b['n_months']:>5} {b['years']:>6.1f} "
              f"{b['sharpe_excess']:>+7.4f} "
              f"[{b['sharpe_ci95_analytic'][0]:>+5.2f},{b['sharpe_ci95_analytic'][1]:>+5.2f}] "
              f"{b['dsr_bar_n46']:>8.4f} {b['dsr_bar_incl_search']:>9.4f} "
              f"{b['dsr_n46']:>6.3f} {'YES' if b['clears_0894'] else 'no':>7}")

    print("\n" + "=" * 128)
    print("WHAT EACH BOOK ACTUALLY COMPOUNDS AT, AND THE DRAWDOWN IT COSTS")
    print("=" * 128)
    print(f"{'book':>48} {'hK lev':>7} {'hK CAGR':>8} {'hK DD obs':>10} {'hK DD p95':>10} "
          f"{'DD<=50 obs':>16} {'DD<=50 boot':>16}")
    for b in rows:
        print(f"{b['label']:>48} {b['half_kelly_leverage']:>7.2f} "
              f"{b['half_kelly_cagr_measured']:>+8.2%} {b['half_kelly_dd_measured']:>+10.1%} "
              f"{b['half_kelly_dd_bootstrap']['boot_p95']:>+10.1%} "
              f"{b['dd50_observed']['cagr']:>+9.2%}@{b['dd50_observed']['leverage']:>5.2f}x "
              f"{b['dd50_bootstrap']['cagr']:>+9.2%}@{b['dd50_bootstrap']['leverage']:>5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
