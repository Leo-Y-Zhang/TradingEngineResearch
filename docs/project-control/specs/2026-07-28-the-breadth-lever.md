# The route to high Sharpe is BREADTH, not a better signal

**2026-07-28.** The missing lever behind every failed study in this programme.

## The law

Grinold's Fundamental Law of Active Management:

    IR ~= IC * sqrt(BR)

IC is per-bet skill; BR is the number of INDEPENDENT bets per year. Combined with the
Kelly ceiling recorded in `2026-07-27-the-dsr-sample-length-finding.md`
(max compound growth = Sharpe^2 / 2), the target translates into a breadth requirement,
not a signal-quality requirement.

## Why every study here failed on this axis

| study | rebalances/yr | independent cross-sections | effective BR |
|---|---|---|---|
| capacity curve (2026-07-27) | 4 | 1 | very low |
| Sharadar fundamentals | 12 | 1 | low |
| insider Form-4 | 12 | 1 | low |
| free EDGAR fundamentals | 12 | 1 | low |

Ten studies, all at single-cross-section monthly-or-slower frequency. At that breadth
Sharpe ~1.2 is unreachable **regardless of signal quality**. The programme has spent a
year improving IC while leaving sqrt(BR) at its floor.

A multi-market systematic programme: ~40 futures markets x 3 quasi-independent
timeframes x daily decisions. Even discounting hard for cross-market and cross-timeframe
correlation, that is 2-3 orders of magnitude more independent bets, i.e. a 10-30x Sharpe
multiplier for IDENTICAL per-bet skill.

## Why this programme structurally could not find it

Breadth requires trading cheaply and often. Costs scale LINEARLY with turnover; Sharpe
scales only with sqrt(BR). At the 240bps spreads measured in the smallest capacity band
(`capacity_curve_result.md`), high frequency loses that race by construction. The whole
programme has been hunting in the one region of the market where the breadth lever
cannot be pulled. Liquid futures at 1-3bps is where it can.

This also corrects a claim made earlier in the same session: "trend following delivers
Sharpe 0.5-0.8" is the figure for SLOW, SINGLE-TIMEFRAME trend. Multi-market,
multi-timeframe programmes run higher, and the difference is breadth, not cleverness.

## What the targets actually require

Half-Kelly growth = 3*S^2/8, at volatility S/2:

| target | required Sharpe | implied volatility | honest read |
|---|---|---|---|
| 30%/yr | ~0.90 | ~45% | demanding, arithmetically open |
| 60%/yr | ~1.27 | **~63%** | different risk regime; one bad sequence does not recover |

60% is not a gentler 30%. It is a commitment to ~63% annualised volatility.

## Constraints that are real

- Futures access needs capital and broker account eligibility. ETF proxies work at
  smaller size with materially more drag.
- FCA rules bar crypto derivatives for UK retail, removing perpetual-funding carry.
- IC must SURVIVE at higher frequency; it usually decays. Breadth is not free.
- There is an optimum frequency, not a monotone improvement: linear costs vs sqrt gains.

## Status

Recorded, NOT acted on. Sequenced after the GATE-1/2/3 fixes. Sleeve #2 should be
pre-registered as multi-market, multi-timeframe, explicitly measuring realised breadth
and reporting Sharpe per decade.
