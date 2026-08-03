# Registered amendment — a portfolio-marginal criterion for sleeve promotion

**Status: REJECTED by four-lens adversarial review, 2026-07-27. NOT implemented.**
See `2026-07-27-portfolio-gate-amendment-REVIEW.md`. Verdicts: 3 REJECT, 1
APPROVE-WITH-CHANGES. **The core diagnosis in §1 is WRONG**: criterion 2 is not the
binding constraint -- DSR >= 0.95 at the current cumulative `n_trials` implies an
effective standalone Sharpe floor ABOVE the 0.75 this amendment relaxes, so it would
admit zero additional sleeves. The §1 `s*sqrt(N)` table also assumes zero inter-sleeve
correlation; at a realistic 0.2-0.4 the target is unreachable at ANY N. Everything
below is retained unedited as the rejected proposal. `selection_rule` is unchanged.
**Written:** 2026-07-27, **before** any sleeve was measured against it.
**Scope:** `research/validation.py::selection_rule`, criterion 2 only.

---

## 1. The defect

`selection_rule` requires **`sharpe_net > 0.75` standalone** of every sleeve
(`research/validation.py:424`). That criterion is correct for a system that will run
exactly one strategy. It is wrong for a system that intends to run several, and it makes
the project's own stated return objective unreachable by construction.

Portfolio Sharpe from `N` uncorrelated sleeves each of Sharpe `s` is `S = s*sqrt(N)`, and
annual return is approximately `S * sigma`. A 30%/yr objective at a survivable 23%
volatility requires portfolio Sharpe ≈ 1.30:

| per-sleeve Sharpe | uncorrelated sleeves required |
|---|---|
| 0.40 | 11 |
| 0.50 | 7 |
| 0.60 | 5 |
| 0.75 | 3 |

Two consequences follow. First, even at its own threshold the gate needs **three**
passing sleeves; after nine studies the programme has zero, and has been searching for
them one at a time. Second, and more seriously, a sleeve at Sharpe 0.50 that is
genuinely uncorrelated to the existing book is **more valuable** than a sleeve at 0.80
that is 0.7-correlated to it — and the current gate accepts the second and rejects the
first. The criterion is not merely conservative; on the dimension that matters for a
multi-sleeve portfolio it is **inverted**.

## 1a. The gate contradicts the project's own governing plan

This is not a new idea being imposed on the project. The internal week plan (not
part of this repository) states the path to the 30% objective in terms that
could not be clearer:

> "Therefore the path to a *credible, after-cost* 30% is **genuine independent alpha**
> (Medallion-style: many small, **weak**, independent signals; richer data; rigorous OOS
> validation)"

The governing plan mandates *many weak independent* signals. `selection_rule` requires
every signal to be **strong standalone** (`sharpe_net > 0.75`). A gate that rejects weak
signals cannot implement a strategy defined as an aggregation of weak signals. The two
documents have contradicted each other since 2026-06-19, and nine studies have been run
under a promotion rule that structurally forbids the approach the plan prescribes.

The research directory is literally named `medallion_style_alpha_search/`. The intent was
never in doubt; only the gate's implementation of it was wrong.

This reframes the amendment. It is not a relaxation of a standard — it is a correction
that brings the promotion rule into line with the brief it was built to serve. That does
not exempt it from the review in §6; a reviewer should still test whether §2's list of
untouched criteria is complete and honest.

## 2. What is NOT changing

This amendment must not become a way to let disappointing results through. Every
anti-self-deception criterion is untouched and remains default-deny:

- **DSR ≥ 0.95** — the multiple-testing / non-normality deflation. Unchanged.
- **PBO** via combinatorial symmetric cross-validation. Unchanged.
- **mean rank-IC > 0.01** — the cheap orthogonal criterion that caught the micro-cap
  mirage when DSR 1.000 and PBO 0.00 both waved it through. Unchanged.
- **stability > 0.60**, **deflated-Sharpe proxy > 0.25**, **zero leakage flags**, **no
  regime Sharpe < −0.50**. All unchanged.
- Purged walk-forward with embargo remains the only permitted cross-validation.
- Honest cumulative `n_trials` accounting remains in force.

Only criterion 2 is amended.

## 3. The amendment

Criterion 2 becomes conditional on whether the sleeve would run alone.

**(a) No validated sleeve exists yet** — the standalone criterion `sharpe_net > 0.75`
applies, unchanged. A first sleeve *is* the portfolio, so it must stand on its own.

**(b) A validated sleeve set exists** — the candidate must instead satisfy **both**:

1. `sharpe_net > 0.30`, an absolute floor. A sleeve below this contributes too little
   to justify the operational risk of running it, whatever its correlation.
2. **Marginal contribution to portfolio Sharpe ≥ 0.10**, measured
   out-of-sample: `S(existing + candidate) - S(existing) >= 0.10`, where both terms are
   computed on the purged walk-forward OOS return series, with the candidate weighted by
   the same allocation rule that would be used live.

## 4. Why this is a harder test, not a weaker one

The marginal criterion **adds a requirement that does not currently exist**: the
candidate must demonstrate low correlation to the existing sleeves *out-of-sample*.
Correlation is unstable and tends to rise in exactly the drawdowns that matter, so this
is a demanding test and it is easy to fail. A candidate that clears 0.75 standalone but
is highly correlated to the book **fails** criterion (b) where it would have passed
today — the amendment can reject sleeves the current rule accepts.

Three further guards, registered here:

- **Correlation is measured on OOS returns only**, never in-sample.
- **The existing sleeve set is frozen** before the candidate is measured. Re-optimising
  the allocation to flatter a candidate is banned.
- **`n_trials` accrues across the whole programme**, so building a portfolio one sleeve
  at a time does not reset the deflation bar for any of them.

## 5. Honest statement of the risk

The classic way to fool yourself is to relax a gate until something passes. This
amendment is that shape of change, and it deserves the suspicion. Three things separate
it from threshold-chasing, and they should be checked by the reviewer rather than
taken on trust:

1. It is registered **before** any sleeve has been measured against it, and before the
   capacity study runs. No number motivated it.
2. It leaves every statistical anti-overfitting criterion untouched. The thing being
   changed is a *portfolio-construction* criterion, which is a different kind of claim
   from a *statistical-significance* criterion.
3. It is derived from arithmetic (`S = s*sqrt(N)`) that is checkable independently of any
   result in this repository.

If a reviewer judges that (1)–(3) do not hold, the amendment should be rejected and the
standalone criterion retained.

## 6. Review requirement

This amendment requires a four-lens adversarial review, refute-by-default, **before
implementation**, with at least one lens tasked specifically with arguing that it is
threshold-chasing in disguise. Until that review passes, `selection_rule` is unchanged
and the standalone criterion governs every study, including the capacity study.
