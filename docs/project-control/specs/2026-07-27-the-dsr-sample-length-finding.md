# The way through DSR is sample length, not a lower bar

**2026-07-27.** Supersedes the reasoning behind the rejected portfolio-gate amendment
(`2026-07-27-portfolio-gate-amendment-REVIEW.md`). No gate change is needed.

## The finding

The four-lens review established that `DSR >= 0.95`, not `sharpe_net > 0.75`, is what has
rejected every sleeve for ten studies. What the review did not draw out is that the DSR
requirement is a function of SAMPLE LENGTH. Significance scales as ~1/sqrt(years):

| OOS years | min annual Sharpe for DSR >= 0.95 (n_trials 26) | n_trials 40 | what has this history |
|---|---|---|---|
| **7** | **1.45** | 1.52 | the window every study has used |
| 10 | 1.21 | 1.27 | |
| 15 | 1.00 | 1.05 | |
| 20 | 0.84 | 0.88 | Sharadar full history |
| 30 | 0.69 | 0.72 | futures trend via index/bond proxies |
| **40** | **0.60** | 0.63 | trend following, published record |
| 50 | 0.53 | 0.56 | trend following, academic record |

Computed against this repo's own `research/validation.py::deflated_sharpe_ratio`, with a
sanity check that reproduces the programme's published anchor (n_trials 9, T 92 months:
Sharpe 1.1 -> DSR 0.925, crossing ~1.15, matching the "~1.1" recorded in
`insider_study_prereg.md:123`).

## Why it matters

Every study has demanded Sharpe ~1.45 because every study ran on ~7 years. A strategy
family with a 40-year honest record clears the SAME UNMODIFIED GATE at Sharpe 0.60 --
which is inside the documented range for trend following (0.5-0.8).

The rejected amendment was therefore not merely wrong about which criterion binds; it was
unnecessary. The route through DSR is more years, not a weaker test. Nothing in
`selection_rule` needs to change.

Combined with volatility targeting (the dial the programme has never used, since
return ~= Sharpe x vol): Sharpe 0.65 at 40% vol is about 26%/yr; 0.70 at 45% is about 31%.

## What must be established before believing it

1. **Drawdown reality.** Trend following's historical worst drawdowns run ~25-35% at its
   natural ~15% vol. At 40% vol that is 50-65%, which ends most accounts before recovery.
2. **Regime front-loading.** The 40-year record is not uniform; trend materially
   underperformed ~2009-2019. Measure Sharpe PER DECADE before trusting a full-sample
   number, and treat a full-sample pass with a failing recent decade as a failure.
3. **Data honesty over 40 years.** Free index/bond/commodity proxies reach back far
   enough; continuous futures splicing introduces roll assumptions, and commodity
   contract survivorship is real. The data-integrity work is the study, not a preliminary.
4. **n_trials rises.** A new family is a new pre-registration; the ledger is cumulative
   (currently 26), and the required Sharpe rises slowly with it (0.60 -> 0.63 from 26 to
   40 trials at 40 years). That is affordable; it is not free.

## Status

Finding recorded, NOT acted on. Next step is a pre-registration for a trend-following
sleeve on a 30-50 year proxy history, with per-decade reporting mandatory and the gate
unchanged.
