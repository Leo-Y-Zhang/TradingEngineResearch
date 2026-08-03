# Four-lens adversarial review — VERDICT: REJECTED

**Reviewed:** `2026-07-27-portfolio-gate-amendment.md`
**Run:** 2026-07-27, four independent refute-by-default lenses.
**Outcome: 3 REJECT, 1 APPROVE-WITH-CHANGES → the amendment is REJECTED and NOT implemented.**
`selection_rule` is unchanged. The standalone `sharpe_net > 0.75` criterion still governs.

---

## 1. The finding that kills it (reached independently by three of four lenses)

**The amendment targets a constraint that is not binding, so it would admit zero
additional sleeves.**

`selection_rule` also requires `DSR >= 0.95` (`research/validation.py:439`), which the
amendment declared untouched. Inverting the DSR formula at the programme's actual
sample length and cumulative trial count gives an *effective* standalone Sharpe floor
**above the 0.75 being relaxed**:

| lens | assumption | implied standalone Sharpe floor |
|---|---|---|
| threshold-chasing | 212-month DEV window, n_trials 26 | **~0.83** |
| statistical validity | 84-month OOS window, n_trials 26 | **~1.45** |
| completeness | criterion 4 (`deflated_sharpe_proxy > 0.25`) alone, n_trials 26 | **~0.52** |

The lenses differ on the exact number because they assume different windows, but they
agree on the direction and it is decisive: **DSR, not criterion 2, is why nine studies
produced zero passing sleeves.** The amendment's §1 diagnosis — that criterion 2 "makes
the objective unreachable by construction" — identifies the wrong line of code. It is
line 439, not line 423.

Worse, §2's list of untouched criteria then functions as a *deferral* rather than a
safeguard: the first use of the amended rule would reveal DSR as the blocker, and §1a
would already have supplied the argument for amending that too. That is a ratchet one
step downstream of where the amendment looked for one.

## 2. The `s*sqrt(N)` arithmetic assumes zero correlation, and that assumption is load-bearing

The general equicorrelation result is `S = s*sqrt(N / (1 + (N-1)*rho))`, with a hard
ceiling of `s/sqrt(rho)` as N grows. Required sleeves for portfolio Sharpe 1.30:

| per-sleeve s | rho=0 | rho=0.1 | rho=0.2 | rho=0.3 | rho=0.4 |
|---|---|---|---|---|---|
| 0.40 | 11 | **impossible** | **impossible** | **impossible** | **impossible** |
| 0.50 | 7 | 19 | **impossible** | **impossible** | **impossible** |
| 0.60 | 5 | 8 | 62 | **impossible** | **impossible** |
| 0.75 | 3 (**arithmetic error — 0.75*sqrt(3)=1.299 < 1.30, answer is 4**) | 4 | 7 | 22 | **impossible** |

**At realistic inter-sleeve correlations of 0.2–0.4, portfolio Sharpe 1.30 is
unreachable at any N** for every per-sleeve Sharpe the amendment contemplates. The
amendment presented the zero-correlation corner as the operating case without labelling
it as such.

## 3. The new criterion is calibrated below its own noise

At the programme's real OOS length (84–92 monthly observations), the standard error of
the marginal-Sharpe statistic is **0.14–0.30** against a **0.10** hurdle. It is a coin
flip with a favourable bias, carries no deflation, and accrues no trials:

- P(false pass | true contribution <= 0) ≈ **13.8%** per candidate
- Family-wise error over the programme's 26 trials ≈ **97.9%**; expected false passes ≈ 3.6
- A Bonferroni-equivalent hurdle at 5% FWER over 26 candidates is **0.86**, not 0.10
- SE scales with *years*, not observations — sampling daily instead of monthly buys
  nothing. Reaching 80% power at a 0.10 effect needs **~100+ years** of OOS data.

## 4. The amendment forbids the very portfolio it exists to enable

A *fixed absolute* 0.10 hurdle against a growing book is a shrinking target. For N equal
uncorrelated sleeves of Sharpe s, the Nth adds `s*(sqrt(N) - sqrt(N-1))`, so the rule
caps N:

| s | max sleeves admissible | max portfolio Sharpe |
|---|---|---|
| 0.40 | 4 | 0.80 — **cannot reach 1.30** |
| 0.50 | 6 | 1.22 — **cannot reach 1.30** |
| 0.60 | 9 | 1.80 |

The amendment's own headline case — eleven 0.40-Sharpe sleeves — **is forbidden by the
amendment's own criterion**; the fifth such sleeve fails the hurdle.

## 5. §4's central defensive claim is false in the operative regime

§4 claimed the marginal test "adds a requirement that does not currently exist" and
would reject a highly-correlated 0.80-Sharpe candidate. Against a DSR-clearing candidate
(s >= 1.2), blending into a 0.75-Sharpe book clears the 0.10 hurdle **at every
correlation up to and including rho = 1.0** — a literal duplicate sleeve passes. The
criterion only bites where the candidate's Sharpe is at or below the book's, and DSR has
already excluded that entire region. A correlation gate also already exists
(`alpha_factory.py:518-560`, `max_correlation=0.80` plus clustering), unlisted by §2.

## 6. GENUINE CODE DEFECTS FOUND — independent of this amendment, and live now

These make the **current** gate more lenient than every study document claims. They are
not hypothetical and they should be fixed on their own merits.

| # | defect | evidence | effect |
|---|---|---|---|
| **GATE-1** | `n_trials` is wired to the **fold count**, not the configuration count | `research/alpha_factory.py:233` (`n_trials = max(len(splits), 1)`) directly contradicts the warning in `research/validation.py:330` | DSR gate is **~0.14 Sharpe too lenient** at 8 folds vs the honest 26 |
| **GATE-2** | `deflated_sharpe_ratio` defaults to **1.0 = pass** | `research/validation.py:45` | Any `ValidationResult` built without the field **silently clears the DSR gate**. Default-allow, while §2 called it default-deny |
| **GATE-3** | **PBO is never checked by `selection_rule`** | no reference to `pbo_proxy` in `validation.py:401-467`; the repo's own scripts say so at `scripts/research_sharadar_alpha.py:277` | §2 listed PBO as an untouched safeguard. It is not in the gate at all, and never has been |
| **GATE-4** | `selection_rule` has **three other live callers** the amendment never mentions | `ops/model_registry.py:88,138`; `learning/adaptive_weights.py:109` | A criterion-2 change would silently widen live model promotion and live sleeve reweighting |
| **GATE-5** | `pbo_proxy` where populated is **not CSCV PBO** | `research/alpha_factory.py:237` computes the fraction of losing periods; `:872` hardcodes `0.0` | The name implies a safeguard the value does not provide |
| **GATE-6** | `leakage_flags` and `regime_breakdown` are **empty by construction** on the learned path | `research/alpha_factory.py:871-874` | Two of the eight advertised criteria are vacuous on the path every study used |

## 7. What survives

The **portfolio-marginal concept is correct and worth keeping.** All four lenses agreed
on that, and on the reality of the doc-vs-code contradiction in §1a: the governing plan
does mandate "many small, weak, independent signals"
(the internal week plan, not part of this repository) while the gate demands strong standalone ones.
The pre-registration timeline also verified — the amendment was committed before the
capacity study ran, so it was not fitted to a seen result.

What fails is the execution: wrong constraint targeted, undeflated selection statistic,
uncalibrated thresholds, and a §2 honesty claim that does not survive contact with the
code.

## 8. Required before any resubmission

1. State the **effective standalone floor implied by DSR** at the current `n_trials`, and
   register a binding commitment that DSR >= 0.95 and cumulative `n_trials` will not
   themselves be amended. Without this the amendment is operationally a no-op.
2. Replace the `s*sqrt(N)` table with the **rho-conditioned** version, state the
   `s/sqrt(rho)` ceiling, and pre-register an estimate of realised inter-sleeve rho.
3. Make the marginal test a **bootstrap lower confidence bound** with explicit
   multiple-testing deflation over candidates — not a point estimate.
4. Make the hurdle **relative** (e.g. proportional to current S), not a fixed absolute
   increment that forbids the target portfolio.
5. Restrict branch (b) to the **sleeve-promotion path only**; pin `model_registry` and
   `adaptive_weights` to the standalone criterion.
6. Register the **allocation rule by name** and use walk-forward weights, never weights
   fitted on the scoring window.
7. Define, hash and persist the **validated sleeve set**; add grandfathering and a
   set-shrinkage re-validation trigger.
8. Fix **GATE-1, GATE-2, GATE-3** as preconditions.
9. **Falsification test before adoption:** replay the nine prior failed studies and the
   junk negative control through the amended rule. If any junk result passes, the
   amendment is refuted on the programme's own evidence.

## 9. Status

**REJECTED. Not implemented. `selection_rule` unchanged.** The multi-sleeve route to a
higher portfolio Sharpe is therefore still structurally blocked — and now known to be
blocked by DSR rather than by criterion 2, which is a different and harder problem than
the amendment assumed.
