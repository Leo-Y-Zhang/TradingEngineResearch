# TradingEngineResearch — The Three Systems (20 / 24 / 30)

Three risk-tiered configurations of the **same** engine, selected purely via
`core/config.py` settings (no separate codebase, no trading-logic change). Each is set
with `ENGINE_*` environment variables; every risk protection (CVaR limit, per-name &
sector caps, crisis tightening, fail-closed pre-trade gate, drawdown governor) applies to
all three.

> **Numbers are real-data backtest *estimates* (8-name, net of cost) and STARTING points.**
> The exact knobs will be calibrated and **out-of-sample validated** during the audit week
> (roadmap M2) — and the 24% claim honestly confirmed/adjusted. Backtest period is a bull
> market; expect lower live returns and real losing years.

## Profiles

| Setting (`ENGINE_*`) | **20 System** (conservative) | **24 System** (core baseline) | **30 System** (experimental, HIGH RISK) |
|---|---:|---:|---:|
| `TARGET_VOL` | 0.16 | 0.22 | 0.30 |
| `MAX_GROSS_LEVERAGE` | 1.3 | 2.0 | 3.5 |
| `MAX_POSITION_WEIGHT` | 0.12 | 0.20 | 0.30 |
| `CVAR_LIMIT` | 0.09 | 0.12 | 0.20 |
| `SIGNAL_TILT_STRENGTH` | 0.003 | 0.003 | 0.003 |
| **Est. CAGR (net)** | ~16–20% | ~20–24% | ~24–30% |
| **Est. Sharpe** | ~1.25 | ~1.2 | ~1.1 |
| **Est. max drawdown** | ~12–15% | ~16–20% | ~25–35% |
| **Honest label** | safe, durable | core product | **high risk / ruin-risk in a crash** |

## Blended fund allocation (owner's instruction)

| Tier | Capital weight |
|------|---------------:|
| 30 System | 10% |
| 24 System | 60% |
| 20 System | 30% |

Blended target ≈ 0.10·30 + 0.60·24 + 0.30·20 = **~23.4% CAGR**, with risk diversified across
tiers (the small 30-System sleeve adds upside without betting the fund on the high-risk profile).

## Honest constraints
- The 30 System reaches ~30% by taking **real risk** (leverage/concentration), **not** by a
  low-risk alpha edge that does not exist in free price/volume data (proven — see the research write-up).
- The way to lift the *whole* table (more return per unit of risk) is **genuine independent
  alpha** from the research track, not bigger leverage. Until that lands, treat 30% as the
  high-risk tier and size it small (10%).
- No profile may disable a risk control; the 30 System only *widens* the budget within the
  existing fail-closed machinery.

## How to run a system
```bash
# 24 System (core) example
export ENGINE_TARGET_VOL=0.22 ENGINE_MAX_GROSS_LEVERAGE=2.0 \
       ENGINE_MAX_POSITION_WEIGHT=0.20 ENGINE_CVAR_LIMIT=0.12
python -m ops.run_loop            # or measure: python scripts/backtest_real.py
```
