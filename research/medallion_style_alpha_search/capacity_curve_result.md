# RESULT — The Capacity Curve of the Fundamental Ordering Edge

**Run:** 2026-07-27, DEV window only (1998-04-30 → 2015-12-31). Single run, no variants.
**Registration:** `capacity_curve_prereg.md` + errata 1 and 2 (written before the run).
**Verdict: H1 SUPPORTED — and NOT DEPLOYABLE. Both, and the second matters more.**

---

## 1. Headline

The capacity effect is real: net performance declines monotonically with deployable
capital, and it survives adjustment for the small-cap premium. **And it is worthless,
because the strategy loses to simple buy-and-hold in every single band.**

| band | deployable capital | net return | net vol | net Sharpe | maxDD | benchmark | **excess** | cost drag | measured % |
|---|---|---|---|---|---|---|---|---|---|
| B1 $50k–$200k | $32k | 9.4% | 19.3% | 0.49 | 57.8% | 12.0% | **−2.6%** | 7.1% | 79% |
| B2 $200k–$1M | $127k | 5.8% | 16.0% | 0.36 | 35.2% | 11.3% | **−5.5%** | 5.7% | 57% |
| B3 $1M–$5M | $639k | 3.7% | 16.3% | 0.22 | 54.4% | 10.1% | **−6.4%** | 6.2% | 37% |
| B4 $5M–$25M | $3.1M | 3.4% | 13.3% | 0.25 | 41.4% | 9.6% | **−6.2%** | 6.6% | 28% |
| B5 $25M–$200M | $16.5M | 1.0% | 15.9% | 0.06 | 37.1% | 8.4% | **−7.3%** | 6.6% | 27% |
| B6 >$200M | $99.9M | −7.0% | 22.3% | −0.32 | 91.4% | 8.4% | **−15.5%** | 12.5% | 25% |

**Primary statistic (registered, one trial):** Spearman rho between deployable capital
and net Sharpe = **−0.943**, one-sided permutation p = **0.0080** (100,000 permutations,
6 bands). H1 supported.

**Secondary (declared in advance as non-gate-eligible):** the same test on excess over
each band's own equal-weight buy-and-hold gives rho = **−0.943**, p = **0.0081**. This
matters: raw returns fall with capacity partly because the small-cap premium does, and
the primary statistic alone cannot separate a capacity effect from a size effect. The
secondary shows the ordering is **not** merely the size premium.

## 2. What this does and does not establish

**Established.** The relationship the prior programme never looked for is there. Between
an untradeable $51k/day shell and a $5M/day floor lie two orders of magnitude that had
never been measured, and across the full ladder the ordering is clean and significant in
both the raw and benchmark-adjusted forms. Retail's capacity advantage is a real,
measurable feature of this data.

**Not established, and this is the part that governs.** The advantage is an advantage at
losing less slowly. Every band underperforms passive ownership of the same names: −2.6%
a year at the smallest size, worsening to −15.5% at institutional size. The best band's
net Sharpe is 0.49 against a promotion gate of 0.75, and its *excess* Sharpe is negative.
**Nothing here is deployable, and nothing here goes near the gate.** Per prereg §1, H1
support was declared in advance not to be a deployability finding; this is that case
exactly.

The capacity curve therefore describes how fast this construction destroys value as
capital grows, not how much alpha it earns. That is a genuine measurement and a
publishable one. It is not an edge.

## 3. Why the strategy loses

Cost drag runs 5.7–12.5% a year against benchmark returns of 8.4–12.0%. At roughly
3.0–3.5x annual turnover, quarterly rebalancing with a no-trade band was not enough: the
registered construction still trades too much for a signal this thin, which is the same
verdict the prior programme reached on a different construction. The B6 anomaly (12.5%
cost, the highest, in the *most* liquid band) is impact, not spread: deployable capital
is defined as 1% of median dollar volume per position, so every band trades at its own
capacity and pays ~100bps of square-root impact by construction. B6's higher turnover
then makes it the most expensive.

## 4. Two defects found and fixed during the run

Both were found because the first outputs were impossible, not because a test caught
them. Recorded because the corrected numbers are only trustworthy if the corrections are.

1. **Delisted names were re-booked every month.** A position that left the measurable
   universe had its terminal return applied but was never removed from the book, so a
   bankrupt name booked −100% every month thereafter. Symptom: −112% annualised on a
   long-only book, which cannot lose more than 100%.
2. **Terminal returns ignored the delisting DATE.** `terminal.get(ticker)` asked whether
   a name delisted *ever*, so a 2012 bankruptcy was charged against a 2003 exit. A name
   leaves the measurable universe for many reasons — its band changes, its spread stops
   resolving — and almost none of them are delisting. Symptom: −60%/yr against a
   universe returning +12%/yr.

After both fixes the benchmark reproduces sensible band returns (8.4–12.0%/yr, declining
with size as the size premium implies), which is the check that the accounting is now
sound.

## 5. Honest limitations

- **Coverage falls with liquidity.** Only 25–37% of cells in bands B3–B6 have a genuinely
  measured spread; the rest are floor-bounded and excluded. Those bands' results rest on
  a minority of their names and should be read with more suspicion than B1's 79%.
- **Acquisitions are booked flat**, understating the return to being acquired. This is
  the conservative direction but it is not free of effect: 6,137 of 20,560 resolved
  delistings are acquisitions.
- **The signal is deliberately unfitted** — a fixed equal-weight composite of 14
  sign-aligned factors. A learned combiner might do better, but the prior programme found
  the learned ridge *underperformed* the naive composite in every tradable universe, and
  fitting would spend trials this study cannot afford.
- **Single run, no variants.** No configuration was tried and discarded.

## 6. Trial accounting and status

**Cumulative n_trials: 23 (prior programme) + 3 (this study) = 26.** The three are the
feature set, the construction, and the monotonicity test. The three spread estimators
evaluated in errata 1–2 cost nothing: they were selected by calibration against known
ground truth, never by their effect on a strategy result.

**The 2016+ confirmation window remains UNFIRED**, and correctly so — the prereg permits
firing it only at a model that passes the DEV-side gate, and no band comes close. The
`load_prices` guard in `research/capacity_panel.py` refused DEV-side tools access to it
throughout.

**Do not re-run this hypothesis with adjusted bands, horizons or constructions.** That is
the selection bias the whole apparatus exists to refuse. A further attempt requires a new
pre-registration at a higher trial count and on materially new information.
