"""An INSTRUMENTED re-implementation of `lowvol_retest.run_band`.

It must produce the identical book -- that identity is asserted in `check_repro` -- and it
additionally records, for every rebalance, exactly which names left, why, and what the
book charged them. Without that record the 46.5% forced-exit share cannot be traced to
anything.

Deliberately a SEPARATE implementation of the same rules rather than a monkey-patch of the
original: if the two disagree, one of them is wrong and that is itself a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.capacity_study import (
    COMMISSION_MAX_FRACTION,
    COMMISSION_MIN_PER_ORDER,
    COMMISSION_PER_SHARE,
    FX_COST_EACH_WAY,
    IMPACT_COEFFICIENT_CONSERVATIVE,
    IMPACT_COEFFICIENT_REALISTIC,
    impact_fraction,
)
from research.sleeves.low_vol_quality import (
    DELISTING_WINDOW_DAYS,
    FORWARD_RETURN_CLIP,
    MIN_CROSS_SECTION,
    MONTHS_PER_YEAR,
    N_POSITIONS,
    PARTICIPATION_LIMIT,
)

BOUNDS = ("conservative", "realistic")
_COEF = {"conservative": IMPACT_COEFFICIENT_CONSERVATIVE,
         "realistic": IMPACT_COEFFICIENT_REALISTIC}


def commission_fraction(trade_value: float, price: float) -> float:
    if trade_value <= 0 or price <= 0:
        return FX_COST_EACH_WAY
    shares = trade_value / price
    commission = max(COMMISSION_MIN_PER_ORDER, shares * COMMISSION_PER_SHARE)
    commission = min(commission, trade_value * COMMISSION_MAX_FRACTION)
    return commission / trade_value + FX_COST_EACH_WAY


@dataclass
class Instrumented:
    band: str
    months: list[pd.Period]
    gross: np.ndarray
    cost: dict[str, np.ndarray]
    benchmark: np.ndarray
    benchmark_rankable: np.ndarray
    deployable_capital: float
    position_value: float
    legs_traded: int
    n_rebalances: int
    # exit ledger: one row per name that left the book at a rebalance
    exits: pd.DataFrame
    entries: pd.DataFrame
    holdings_log: pd.DataFrame          # (month, ticker, realised_return, is_last_obs)
    pnl_by_name_month: list
    participation: pd.DataFrame          # every leg traded, with its participation rate
    notes: list[str] = field(default_factory=list)


def prepare(rows: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    dd = {r.ticker: r.date for r in delistings.itertuples()}
    dv = {r.ticker: float(r.terminal_return) for r in delistings.itertuples()}
    forward = rows["forward_return"].clip(-FORWARD_RETURN_CLIP, FORWARD_RETURN_CLIP)
    delist_date = pd.to_datetime(rows["ticker"].map(dd), errors="coerce")
    delist_value = pd.to_numeric(rows["ticker"].map(dv), errors="coerce")
    in_window = (delist_date.notna() & (delist_date > rows["date"])
                 & (delist_date <= rows["date"] + pd.Timedelta(days=DELISTING_WINDOW_DAYS)))
    rows["terminal_on_exit"] = np.where(in_window, delist_value.fillna(0.0), 0.0)
    rows["forward_clipped"] = forward
    rows["realised_return"] = forward.where(forward.notna(), rows["terminal_on_exit"])
    rows["month"] = rows["date"].dt.to_period("M")
    return rows


def run(panel: pd.DataFrame, band: str, delistings: pd.DataFrame) -> Instrumented | None:
    rows = panel[panel["band_group"] == band]
    if rows.empty:
        return None
    rows = prepare(rows, delistings)

    dd = {r.ticker: r.date for r in delistings.itertuples()}
    dv = {r.ticker: float(r.terminal_return) for r in delistings.itertuples()}

    def exit_return(ticker: str, at: pd.Timestamp) -> float:
        on = dd.get(ticker)
        if on is None:
            return 0.0
        if at < on <= at + pd.Timedelta(days=DELISTING_WINDOW_DAYS):
            return float(dv.get(ticker, 0.0))
        return 0.0

    deployable = N_POSITIONS * PARTICIPATION_LIMIT * float(
        rows["median_dollar_volume"].median())
    position_value = deployable / N_POSITIONS

    by_month = {m: f for m, f in rows.groupby("month", sort=True)}
    months = sorted(by_month)

    holdings: set[str] = set()
    last_seen: dict[str, pd.Timestamp] = {}
    equity_conservative = 1.0

    gross: list[float] = []
    cost: dict[str, list[float]] = {b: [] for b in BOUNDS}
    benchmark: list[float] = []
    benchmark_rankable: list[float] = []
    exits: list[dict] = []
    entries: list[dict] = []
    holdings_log: list[dict] = []
    participation: list[dict] = []
    pnl_by_name_month: list = []
    legs_traded_total = 0
    n_rebalances = 0

    for month in months:
        cross_section = by_month[month]
        rankable = cross_section[cross_section["signal"].notna()]
        period = {b: 0.0 for b in BOUNDS}
        period_commission = 0.0

        if len(rankable) >= MIN_CROSS_SECTION:
            n_rebalances += 1
            target = set(rankable.nlargest(N_POSITIONS, "signal")["ticker"])
            traded = target ^ holdings
            still_rankable = set(rankable["ticker"])
            present = set(cross_section["ticker"])
            weight = 1.0 / max(len(target), 1)
            priced = cross_section.set_index("ticker")

            for ticker in sorted(holdings - target):
                if ticker in still_rankable:
                    kind = "discretionary"
                elif ticker in present:
                    kind = "forced_present_unrankable"
                else:
                    kind = "forced_vanished"
                exits.append({
                    "month": month, "ticker": ticker, "kind": kind,
                    "charged": ticker in present,
                    "last_seen": last_seen.get(ticker, pd.NaT),
                })
            for ticker in sorted(target - holdings):
                entries.append({"month": month, "ticker": ticker})

            for ticker in traded:
                if ticker not in priced.index:
                    continue
                row = priced.loc[ticker]
                mdv = float(row["median_dollar_volume"])
                price = float(row["close"])
                vol_raw = float(row["realised_vol"])
                daily_vol = vol_raw if np.isfinite(vol_raw) and vol_raw > 0.0 else None
                period_commission += weight * commission_fraction(position_value, price)
                participation.append({
                    "month": month, "ticker": ticker,
                    "participation": position_value / mdv if mdv > 0 else np.nan,
                    "mdv": mdv, "side": "in" if ticker in target else "out",
                })
                for bound in BOUNDS:
                    half_spread = float(row[f"spread_{bound}"]) / 2.0
                    impact = impact_fraction(position_value, mdv, daily_vol, _COEF[bound])
                    if not np.isfinite(impact):
                        impact = 0.0
                    period[bound] += weight * (half_spread + impact)
            legs_traded_total += len(traded)
            holdings = target

        for bound in BOUNDS:
            cost[bound].append(period[bound] + period_commission)

        universe = cross_section["realised_return"].dropna()
        step = float(universe.mean()) if len(universe) else 0.0
        rankable_universe = rankable["realised_return"].dropna()
        rankable_step = float(rankable_universe.mean()) if len(rankable_universe) else step
        benchmark.append(step)
        benchmark_rankable.append(rankable_step)

        if not holdings:
            gross.append(0.0)
            equity_conservative *= 1.0 - cost["conservative"][-1]
            continue

        indexed = cross_section.set_index("ticker")
        realised: list[float] = []
        contributions: list[tuple] = []
        closing_out: list[str] = []
        for ticker in holdings:
            if ticker not in indexed.index:
                exit_date = last_seen.get(ticker, month.to_timestamp(how="end"))
                value = exit_return(ticker, exit_date)
                realised.append(value)
                contributions.append((ticker, value))
                closing_out.append(ticker)
                holdings_log.append({"month": month, "ticker": ticker,
                                     "realised_return": value, "is_last_obs": True,
                                     "path": "vanished_nonrebalance"})
                continue
            row = indexed.loc[ticker]
            last_seen[ticker] = row["date"]
            value = float(row["realised_return"])
            realised.append(value)
            contributions.append((ticker, value))
            is_last = bool(pd.isna(row["forward_clipped"]))
            holdings_log.append({"month": month, "ticker": ticker,
                                 "realised_return": value, "is_last_obs": is_last,
                                 "path": "last_obs" if is_last else "held"})
            if is_last:
                closing_out.append(ticker)
        holdings.difference_update(closing_out)

        weight = 1.0 / len(realised)
        period_gross = max(float(np.mean(realised)), -1.0)
        gross.append(period_gross)
        for ticker, value in contributions:
            pnl_by_name_month.append((ticker, month, equity_conservative * weight * value))
        equity_conservative *= 1.0 + max(period_gross - cost["conservative"][-1], -1.0)

    if len(gross) < 24:
        return None
    return Instrumented(
        band=band, months=months,
        gross=np.asarray(gross, dtype=float),
        cost={b: np.asarray(cost[b], dtype=float) for b in BOUNDS},
        benchmark=np.asarray(benchmark, dtype=float),
        benchmark_rankable=np.asarray(benchmark_rankable, dtype=float),
        deployable_capital=deployable, position_value=position_value,
        legs_traded=legs_traded_total, n_rebalances=n_rebalances,
        exits=pd.DataFrame(exits), entries=pd.DataFrame(entries),
        holdings_log=pd.DataFrame(holdings_log),
        pnl_by_name_month=pnl_by_name_month,
        participation=pd.DataFrame(participation),
    )


def annual(series: np.ndarray) -> float:
    return float(np.mean(series)) * MONTHS_PER_YEAR


def sharpe(series: np.ndarray) -> float:
    vol = float(np.std(series, ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
    return annual(series) / vol if vol > 0 else float("nan")
