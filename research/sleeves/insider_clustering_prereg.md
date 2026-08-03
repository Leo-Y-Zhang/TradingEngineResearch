# PRE-REGISTRATION — Sleeve: Insider Transaction Clustering (Sharadar SF2)

**Written 2026-07-28, BEFORE any return, Sharpe or excess figure was computed.**
One configuration. It will be run ONCE. Whatever comes out is the result.

## 0. Why this is not a re-run

The prior Form-4 study (`research/insider_features.py`,
`research/medallion_style_alpha_search/insider_study_prereg.md`) used free scraped SEC
data whose as-filed-ticker join dropped ~22% of rows (Alphabet absent entirely). Its
verdict was **"cannot certify" on POWER grounds** — not "no effect". Sharadar SF2
resolves the issuer entity itself, so the join defect is gone. This is a first genuine
measurement, not a second look at the same numbers.

## 1. Hypothesis (H1)

Names bought on the open market by **several distinct insiders** inside a short window
earn higher forward returns, net of realistic per-name costs, than an equal-weight
buy-and-hold of the same investable universe.

Null: excess return over the own-universe benchmark is <= 0.

## 2. Data and window

- `_data/sharadar/SF2.csv`, filtered to `filingdate <= 2015-12-31` (DEV only).
- **SF2's earliest filing date in this export is 2008-01-02.** The sleeve therefore has
  a materially shorter sample than the price panel: rebalances run
  **2008-04-30 .. 2015-11-30 (92 monthly rebalances, 7.7 years)**. The first rebalance is
  set one full month after the earliest possible 90-day lookback so the trailing window
  is never truncated. The last is the final month-end with a measurable forward return.
- Universe / prices / spreads: `_data/sharadar/panel/monthly_panel_dev.parquet`.
- Delistings: `_data/sharadar/panel/delistings.parquet`.
- Nothing after 2015-12-31 is read. The confirmation window stays unfired.

## 3. Transaction selection (open-market purchases only)

A row qualifies iff `transactioncode == 'P'` **and** `securityadcode == 'NA'`
(non-derivative ACQUISITION). This excludes option exercises (`M`), grants/awards (`A`),
tax withholding (`F`), gifts (`G`), and every derivative leg (`DA`/`DD`) — the signal is
someone spending their own money, not a compensation event.

`securityadcode == 'NA'` must be read with `keep_default_na=False`; pandas otherwise
silently parses the string `"NA"` as missing and the filter matches nothing.

## 4. Deduplication (the fan-out defect)

A joint Form 4 reports ONE economic transaction once per co-reporting owner. Counting
those rows separately inflates both the buyer count and the dollar value — the prior
study measured 41% of purchase-leg value duplicated this way.

Economic-transaction key:
`(ticker, transactiondate, transactionshares, transactionpricepershare, securitytitle,
directorindirect)`.
Rows are sorted by `filingdate` and the **earliest** row per key is kept, so a
`RESTATED - 4` amendment cannot postpone the date the market learned the trade, and
cannot double-count it either.

Measured on the DEV slice before any return work: 393,140 purchase legs collapse to
347,064 (11.7% of rows, **13.2% of dollar value**). Collapsing two genuinely distinct
insiders who bought identical share counts at an identical price on the same day is
possible; it can only UNDERSTATE clustering, which is the conservative direction.

## 5. Signal (fixed now)

At rebalance date `t`, over filings with `t - 90 days <= filingdate < t` (strictly
before `t`, matching the prior study's next-day-availability convention), restricted to
**directors and officers** (`isdirector == 'Y'` or `isofficer == 'Y'`):

- `n_buyers` = count of DISTINCT `ownername` among deduped legs.
- `buy_value` = sum of deduped leg value (`transactionvalue`, falling back to
  `transactionshares * transactionpricepershare`).
- `value_ratio` = `buy_value / (median_dollar_volume * 21)` — the cluster's purchases as
  a fraction of one month of the name's own dollar volume.

The director/officer restriction is chosen **because** 10%-owner fund complexes are the
population that produces fan-out, not because of any effect on returns; 72% of purchase
legs qualify.

**Ranking key (lexicographic, descending): `(n_buyers, value_ratio)`.** No weights, no
fitted combination, no z-scoring.

## 6. Universe (per rebalance date)

Rebalance dates are the maximum panel date within each calendar month (the panel also
carries mid-month stub rows for names that stopped trading; those are never rebalance
dates). A name is eligible at `t` iff, in the panel row at `t`:

1. `band` is not null (median trailing-63d dollar volume >= $50k);
2. `spread_regime == 'measured'` — `upper_bound` and `unmeasurable` names are EXCLUDED,
   never costed at the floor;
3. price >= $2.00 and non-zero volume on >= 90% of the trailing 63 days (both already
   enforced upstream — a `measured` regime implies both);
4. its realised one-month outcome is resolvable per §7 and `|return| <= 1.00`.

## 7. Return for the month, and delistings

`next_date` = the ticker's next panel row date; `t_next` = the next rebalance date.

- **Normal** (`next_date == t_next`): `r = forward_return`.
- **Stopped trading** (`next_date` missing, or `next_date < t_next`): apply the terminal
  return **only if** the delisting event date falls in `(t, t + 62 days]`.
  `r = (1 + forward_return_or_0) * (1 + terminal_return) - 1`.
  If no delisting event lands in that window the outcome is unknown and the name is
  **dropped from the universe at `t`** (counted and reported).
- `r` is then required to satisfy `|r| <= 1.00`; cells outside are dropped (artefact
  filter, applied identically to strategy and benchmark).

A rebuilt-from-scratch book is used at every rebalance, so a delisted name cannot be
re-booked in a later month. Its proceeds go to cash and are redeployed at `t_next`.

## 8. Portfolio

Long-only, **top decile** of the eligible universe by the §5 ranking key, equal weight,
monthly rebalance. Decile size = `ceil(0.10 * N_eligible)`. If fewer names carry
`n_buyers >= 1` than the decile size, the book holds only the names with a signal — it
never pads with zero-signal names.

Book size **$250,000** (a retail book, matching the capacity study's finding that the
smallest band is the only one worth measuring).

## 9. Costs (mandatory, per name, charged on both sides)

Per traded name, on the traded notional:

- **Spread:** `spread / 2` per side, using that name's own EDGE estimate at `t`
  (`measured` regime only, by §6.2).
- **Impact:** `sigma_daily * sqrt(notional / median_dollar_volume)` per side — the
  standard square-root law with the name's OWN trailing-63d daily volatility, rather
  than a flat constant.
- **Commission (IBKR):** `min(max(0.0035 * shares, 0.35), 0.01 * notional)`.

Trades are the difference between target weights and the previous holdings **after
drift**. Names removed by a corporate action pay no exit trading cost (they exit in
cash or worthless); this is a small, stated generosity.

## 10. Benchmark

Equal-weight, monthly-rebalanced, **zero-cost** buy-and-hold of the identical eligible
universe (§6) over the identical dates. Zero cost makes it harder to beat, which is the
conservative direction. **The reported headline is EXCESS over this benchmark.**

## 11. Breadth (rule 7)

Reported three ways:
- naive: `12 * median names held`;
- effective independent bets: `12 * N_eff`, where
  `N_eff = mean per-name residual variance / variance of the equal-weight residual
  portfolio return`, residuals being name return minus the universe equal-weight return
  that month. This is the exact identity for an equal-weight book of correlated names;
- the cross-sectional IC (mean monthly Spearman rank correlation of `n_buyers` against
  realised return) that Grinold's `IR = IC * sqrt(BR)` needs.

## 12. Declared in advance as DIAGNOSTIC, not gate-eligible

Reported for interpretation, never as the headline, never as a basis for re-running:
mean realised return by `n_buyers` bucket (0 / 1 / 2 / 3+), and the clustered
(`n_buyers >= 2`) versus single (`n_buyers == 1`) equal-weight books.

## 13. Decision rule

- **PROMISING** iff net excess over the own-universe benchmark > 0 **and** net Sharpe
  >= 0.75 (the programme's standing gate).
- **MARGINAL** iff net excess > 0 but Sharpe < 0.75.
- **DEAD** iff net excess <= 0.

No re-run with different windows, deciles, cost assumptions or buyer definitions. One
trial. Cumulative programme trial count after this study: 27.

## 14. Errata — defects found and fixed DURING the run

Both were caught because the first output was impossible, not because a test failed.
Neither changed a registered parameter; both are accounting corrections. Recorded
because the corrected numbers are only trustworthy if the corrections are.

1. **The final rebalance kept only the names that delisted.** The "did this name stop
   trading?" test compared the ticker's next panel date against the next *rebalance*
   date, and the rebalance list is truncated at `LAST_REBALANCE`. The last month
   therefore had a synthetic successor 40 days out, every still-listed name looked as
   though it had stopped trading before it, and all of them were dropped as
   unresolvable. Symptom: 17 names out of ~2,200 in 2015-11, and a -52% benchmark month.
   Fixed by taking the successor from the full month-end list.
2. **Every exit was free.** Cost inputs were read from the current month's selected
   names, so any position being SOLD — absent from that frame by definition — fell
   through to a zero-cost branch. Only the buy side was ever charged. Fixed by carrying
   each name's last observed price, spread, volatility and dollar volume forward, so a
   forced liquidation pays a real spread, real impact and a real commission. This
   roughly doubled measured turnover (2.9x -> 5.8x one-way) and the cost drag.

After both fixes the equal-weight benchmark returns 4.0%/yr at 20% volatility over
2008-2015, with a -45% 2008 and a +52% 2009 — which is the check that the accounting is
now sound.
