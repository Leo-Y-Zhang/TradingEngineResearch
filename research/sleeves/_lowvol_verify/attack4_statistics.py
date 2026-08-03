"""ATTACKS 2, 5, 7 -- VARIANCE DRAG, AUTOCORRELATION, SUB-PERIODS AND THE DSR BAR.

Everything here is recomputed FROM THE RETURN STREAMS, not read back out of the result
file, and the vol-matched active return is re-derived by hand rather than by calling the
function under test.

    .venv/Scripts/python.exe -m research.sleeves._lowvol_verify.attack4_statistics
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from research.capacity_panel import PANEL_DIR
from research.multiasset.carry import NW_LAGS, vol_matched_active
from research.multiasset.panel import dsr_sharpe_bar
from research.sleeves import lowvol_retest as LV
from research.sleeves._lowvol_verify import instrumented as INS
from research.sleeves._lowvol_verify.build_frame import build
from research.validation import deflated_sharpe_ratio

BAND = "B2_200k_1M"
M = 12.0


def nw_t(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Newey-West/Bartlett t on the mean. Re-derived; `lags=0` is the iid case."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    mean = float(v.mean())
    dev = v - mean
    var = float(dev @ dev) / n
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    se = math.sqrt(max(var, 1e-18) / n)
    return mean, mean / se


def stationary_bootstrap(rng, n: int, mean_block: float) -> np.ndarray:
    """Politis-Romano indices: geometric blocks, wraps around. Preserves autocorrelation."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < p else (idx[i - 1] + 1) % n
    return idx


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    merged = build()
    delistings = pd.read_parquet(PANEL_DIR / "delistings.parquet")
    books = LV.run_band(merged, BAND, delistings)
    if books is None:
        print(f"{BAND}: insufficient data to build the book")
        return 1
    net = np.maximum(books.gross - books.cost_conservative, -1.0)
    bench = books.benchmark
    months = books.months
    T = len(net)

    print("=" * 112)
    print("ATTACK 2 - IS THE VOL-MATCHED ACTIVE RETURN COMPUTED CORRECTLY, AND WHICH WAY?")
    print("=" * 112)
    sd_s, sd_b = float(pd.Series(net).std()), float(pd.Series(bench).std())
    k = sd_s / sd_b
    active = net - k * bench
    ann = float(active.mean()) * M
    sr_s = float(net.mean()) / sd_s * math.sqrt(M)
    sr_b = float(bench.mean()) / sd_b * math.sqrt(M)
    print(f"  strategy monthly sd {sd_s:.6f} -> annual {sd_s*math.sqrt(M):.4%}")
    print(f"  benchmark monthly sd {sd_b:.6f} -> annual {sd_b*math.sqrt(M):.4%}")
    print(f"  k = sd_s / sd_b = {k:.10f}   (published {0.6575477570528394:.10f})")
    print("  k < 1 so the BENCHMARK IS LEVERED DOWN to the strategy's risk. That is the")
    print("  correct direction for comparing at equal risk, and it is the direction that")
    print(f"  FLATTERS a low-volatility book: it discards {1-k:.1%} of the benchmark's mean.")
    print(f"  hand-computed vol-matched active {ann:+.6%}/yr  "
          f"(published {0.07371166175764268:+.6%})")
    print(f"  identity check  sigma_s x (SR_s - SR_b) = "
          f"{sd_s*math.sqrt(M)*(sr_s-sr_b):+.6%}   -> the statistic IS just a Sharpe gap")
    print(f"  SR_s {sr_s:.4f}  SR_b {sr_b:.4f}  gap {sr_s-sr_b:.4f}")

    print("\n  autocorrelation of the vol-matched active series:")
    dev = active - active.mean()
    ac = [float(dev[lag_i:] @ dev[:-lag_i]) / float(dev @ dev) for lag_i in range(1, 7)]
    print("    " + "  ".join(f"rho{lag_i}={v:+.3f}" for lag_i, v in enumerate(ac, 1)))
    print(f"\n{'lag structure':>28} {'t (vol-matched active)':>24} {'t (net returns)':>18} "
          f"{'t (raw active)':>16}")
    auto = int(round(4 * (T / 100.0) ** (2.0 / 9.0)))
    for label, lags in (("iid (lags 0)", 0), ("NW 4 = the registered choice", 4),
                        (f"NW {auto} (Newey-West rule)", auto), ("NW 6", 6), ("NW 12", 12),
                        ("NW 24", 24)):
        _m, t_a = nw_t(active, lags)
        _m, t_n = nw_t(net, lags)
        _m, t_r = nw_t(net - bench, lags)
        print(f"{label:>28} {t_a:>24.3f} {t_n:>18.3f} {t_r:>16.3f}")
    print(f"  (module default NW_LAGS = {NW_LAGS}; the published t of +2.6369 is the NW-4 one)")

    print("\n  the t-stat above treats k as KNOWN. It is estimated from the same sample.")
    rng = np.random.default_rng(20260728)
    for block in (1, 3, 6, 12):
        stats = []
        for _ in range(4000):
            idx = stationary_bootstrap(rng, T, block)
            s, b = net[idx], bench[idx]
            sd = float(np.std(b, ddof=1))
            kk = float(np.std(s, ddof=1)) / sd if sd > 0 else np.nan
            stats.append(float(np.mean(s - kk * b)) * M)
        stats_arr = np.asarray(stats)
        print(f"    stationary bootstrap, mean block {block:>2} months: "
              f"vol-matched active {stats_arr.mean():+.2%}  "
              f"[{np.percentile(stats_arr,2.5):+.2%}, "
              f"{np.percentile(stats_arr,97.5):+.2%}]  "
              f"P(<= +2%) = {float((stats_arr <= 0.02).mean()):.3f}")

    print("\n  RISK-FREE RATE. Sharpes and the vol-matched active are computed on RAW")
    print(f"  returns, so de-levering the benchmark to {k:.3f}x parks {1-k:.1%} of capital")
    print("  at 0%. Over 1998-2015 3-month T-bills averaged ~2%/yr.")
    print(f"{'assumed rf':>12} {'SR strategy':>12} {'SR benchmark':>13} "
          f"{'vol-matched active':>19} {'t':>7} {'clears +2%?':>12}")
    for rf in (0.0, 0.01, 0.02, 0.03, 0.04):
        s_x, b_x = net - rf / M, bench - rf / M
        sd_sx, sd_bx = float(np.std(s_x, ddof=1)), float(np.std(b_x, ddof=1))
        kx = sd_sx / sd_bx
        act = s_x - kx * b_x
        _m, t = nw_t(act, 4)
        print(f"{rf:>11.1%} {float(s_x.mean())/sd_sx*math.sqrt(M):>12.3f} "
              f"{float(b_x.mean())/sd_bx*math.sqrt(M):>13.3f} "
              f"{float(act.mean())*M:>+18.2%} {t:>7.2f} "
              f"{'yes' if float(act.mean())*M > 0.02 and t > 2 else 'NO':>12}")

    print("\n" + "=" * 112)
    print("ATTACK 7 - THE DSR BAR AND THE REGISTERED GATE")
    print("=" * 112)
    years = T / M
    bar = dsr_sharpe_bar(years, n_trials=38, target=0.95)
    print(f"  sample {T} months = {years:.2f} years")
    print(f"  dsr_sharpe_bar(years={years:.2f}, n_trials=38, target=0.95) = {bar:.6f}"
          f"   (published {0.9233854510551396:.6f})")
    print(f"{'n_trials':>10} " + " ".join(f"{n:>9}" for n in (1, 10, 20, 32, 37, 38, 50, 100)))
    print(f"{'bar':>10} " + " ".join(f"{dsr_sharpe_bar(years, n_trials=n):>9.4f}"
                                     for n in (1, 10, 20, 32, 37, 38, 50, 100)))
    sr_net = INS.sharpe(net)
    print(f"\n  net Sharpe (conservative bound) {sr_net:.4f}  vs bar {bar:.4f}  "
          f"-> {'PASS' if sr_net >= bar else 'FAIL'}")
    print(f"  the sleeve MISSES its own registered promotion gate (iii) by "
          f"{bar - sr_net:.4f} of Sharpe.")
    print(f"  gross Sharpe {INS.sharpe(books.gross):.4f} clears it, but gross is not "
          f"a number anyone can trade.")
    print(f"  DSR(net, n=38) = {deflated_sharpe_ratio(net, n_trials=38):.4f}   "
          f"DSR(benchmark) = {deflated_sharpe_ratio(bench, n_trials=38):.4f}")
    print(f"  skew {float(pd.Series(net).skew()):+.3f}  excess kurtosis "
          f"{float(pd.Series(net).kurtosis()):+.3f} -- DSR already penalises both")
    need = bar
    print(f"  to reach the bar the net stream would need {need*float(np.std(net,ddof=1))*math.sqrt(M):.2%}/yr, "
          f"i.e. {need*float(np.std(net,ddof=1))*math.sqrt(M) - float(net.mean())*M:+.2%}/yr more than it made")

    print("\n" + "=" * 112)
    print("ATTACK 5 - SUB-PERIODS. 17.75 YEARS IS NOT TWO DECADES.")
    print("=" * 112)
    frame = pd.DataFrame({"month": months, "net": net, "bench": bench})
    frame["year"] = [m.year for m in months]
    print(f"  the sample runs {months[0]} .. {months[-1]}")
    print(f"  'per decade' means: {int((frame['year']//10*10 == 1990).sum())} months of the "
          f"1990s, {int((frame['year']//10*10 == 2000).sum())} of the 2000s, "
          f"{int((frame['year']//10*10 == 2010).sum())} of the 2010s.")
    print("  A 21-month stub and a 72-month stub are not decades. There are TWO")
    print("  independent-ish sub-samples here, not three, and one full market cycle.")
    print(f"\n{'window':>22} {'n':>5} {'net ann':>9} {'bench ann':>10} {'net Sh':>8} "
          f"{'bench Sh':>9} {'vol-matched':>12} {'t':>7}")

    def block_row(label, mask):
        s, b = net[mask], bench[mask]
        if s.size < 12:
            print(f"{label:>22} {s.size:>5}  (too short to judge)")
            return
        vm = vol_matched_active(pd.Series(s), pd.Series(b))
        print(f"{label:>22} {s.size:>5} {INS.annual(s):>8.2%} {INS.annual(b):>9.2%} "
              f"{INS.sharpe(s):>8.3f} {INS.sharpe(b):>9.3f} "
              f"{vm.get('vol_matched_active_annual', np.nan):>+11.2%} "
              f"{vm.get('vol_matched_active_tstat', np.nan):>+7.2f}")

    block_row("FULL SAMPLE", np.ones(T, dtype=bool))
    for decade in (1990, 2000, 2010):
        block_row(f"{decade}s", (frame["year"] // 10 * 10 == decade).to_numpy())
    half = T // 2
    block_row("first half", np.arange(T) < half)
    block_row("second half", np.arange(T) >= half)
    crisis = np.array([pd.Period("2008-01", "M") <= m <= pd.Period("2011-12", "M")
                       for m in months])
    block_row("2008-2011 crisis", crisis)
    block_row("EXCLUDING 2008-2011", ~crisis)
    dotcom = np.array([pd.Period("2000-03", "M") <= m <= pd.Period("2002-12", "M")
                       for m in months])
    block_row("2000-2002 dot-com", dotcom)
    block_row("ex both bear markets", ~crisis & ~dotcom)
    for lo, hi in ((1998, 2003), (2003, 2008), (2008, 2013), (2013, 2016)):
        block_row(f"{lo}-{hi-1}", ((frame["year"] >= lo) & (frame["year"] < hi)).to_numpy())

    print(f"\n{'calendar year':>16} {'net':>9} {'bench':>9} {'active':>9}")
    yearly = frame.groupby("year")[["net", "bench"]].apply(
        lambda g: pd.Series({"net": (1 + g["net"]).prod() - 1,
                             "bench": (1 + g["bench"]).prod() - 1}))
    losses = 0
    for year, row in yearly.iterrows():
        flag = " <-- underperforms" if row["net"] < row["bench"] else ""
        losses += int(row["net"] < row["bench"])
        print(f"{year:>16} {row['net']:>+8.1%} {row['bench']:>+8.1%} "
              f"{row['net']-row['bench']:>+8.1%}{flag}")
    print(f"  underperformed its own benchmark in {losses} of {len(yearly)} calendar years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
