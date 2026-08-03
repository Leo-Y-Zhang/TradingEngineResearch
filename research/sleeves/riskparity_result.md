# RESULT — RISK PARITY on the long-history multi-asset panel: **DEAD**

**Pre-registered** in `research/sleeves/riskparity_prereg.md`, committed as `d895110`
**before** `research/sleeves/riskparity.py` existed. Run **once**. No tuning, no second
look, no parameter moved after the fact.

**Code** `research/sleeves/riskparity.py` + `research/sleeves/riskparity_run.py` ·
**artefacts** `research/sleeves/_riskparity/result.json` · re-run is **byte-identical**
(payload md5 `57d2571bd9087b351549a5a34e23dd4e` on three consecutive runs).

**Sample** 738 months, **61.5 years**, 1965-01-31 → 2026-06-30, the **same 18 instruments**
the trend and seasonal sleeves used. Mean eligible instruments 11.30. *Every levered figure
(the whole of §1, §2 and §2b) runs on **726 months, 60.5 years, 1966-01-31 → 2026-06-30**,
because the book-volatility estimate that sets the leverage needs 12 months of book history
first; the Sharpe, DSR, decade and bond-bull figures use the full 738.*
**Trials spent: 2** (W1 naive risk parity, W2 bucketed risk parity) → cumulative **46**.

---

# 1. THE HEADLINE — the survivable-drawdown return

> ## **12.30 %/yr.**
>
> That is the highest compound return this panel delivers at a maximum drawdown of ≤ 50 %,
> net of 10 bps round-trip costs and net of financing at the bill rate + 150 bp.
>
> It takes **1.9× average leverage** (max 5.5×), it draws down **−47.3 %**, and it spends
> **76 months — 6.3 years — under water**, taking **60 months to recover from the trough**.
>
> **And it is not the risk-parity book. It is plain equal weight.** Risk parity's best
> survivable rung is **10.03 %/yr** at −45.4 %. Bucketed risk parity's is **7.89 %** at −47.2 %.

**The 30 % question is answered, and the answer is not close.** The compound return of this
book is **concave in leverage and peaks at 15.83 %/yr** (equal weight, τ = 39 %) — with an
**−87.8 % drawdown**. Beyond that, more leverage makes you *poorer*: at τ = 60 % the compound
return is 9.48 % and the drawdown is −99.6 %. **There is no leverage, survivable or
otherwise, at which a diversified passive book of these 18 instruments compounds at 30 %/yr.**
Not "not survivably" — *not at all*. The Kelly ceiling was never the binding constraint;
variance drag and the financing bill are.

| the honest ladder of answers | compound | max DD | recover |
|---|---:|---:|---:|
| max compound at **DD ≤ 35 %** | **10.56 %/yr** | −33.0 % | 36 mo |
| max compound at **DD ≤ 50 %** | **12.30 %/yr** | −47.3 % | 60 mo |
| max compound at **DD ≤ 60 %** | **13.64 %/yr** | −59.2 % | 104 mo |
| max compound at **any** leverage (τ = 39 %) | 15.83 %/yr | **−87.8 %** | — |
| nearest measured ladder rung (τ = 40 %) | 15.81 %/yr | −88.7 % | 199 mo |
| **required for the programme's target** | **30 %/yr** | — | — |

---

## 2. The full ladder — compound return and max drawdown in the same row, always

Net of **10 bps** round trip and financing at **bill + 150 bp**. "recover" = months from the
worst drawdown's trough back to the prior peak; "u/w" = the full peak-to-recovery span.

### Equal weight (the benchmark) — levered

| τ | compound | vol | Sharpe | **max DD** | trough | recover | u/w | mean lev | fin drag |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 10 % | 10.56 % | 11.0 % | 0.5517 | **−32.99 %** | 2009-02 | 36 mo | 52 mo | 1.25× | 0.42 % |
| **15 %** | **12.30 %** | 16.5 % | 0.5097 | **−47.29 %** | 2009-02 | 60 mo | 76 mo | 1.88× | 1.32 % |
| 20 % | 13.64 % | 21.9 % | 0.4869 | **−59.25 %** | 2009-02 | 104 mo | 120 mo | 2.50× | 2.25 % |
| 25 % | 14.61 % | 27.4 % | 0.4732 | **−69.12 %** | 2009-02 | 106 mo | 122 mo | 3.13× | 3.19 % |
| 30 % | 15.26 % | 32.9 % | 0.4661 | **−77.16 %** | 2009-02 | 144 mo | 160 mo | 3.75× | 4.13 % |
| 40 % | 15.81 % | 43.5 % | 0.4685 | **−88.69 %** | 2009-02 | 199 mo | **215 mo** | 4.97× | 5.95 % |

### Risk parity, naive inverse-vol (W1) — levered

| τ | compound | vol | Sharpe | **max DD** | trough | recover | u/w | mean lev | fin drag |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 10 % | 9.11 % | 11.0 % | 0.4290 | **−31.46 %** | 2009-02 | 26 mo | 42 mo | 1.80× | 1.20 % |
| **15 %** | **10.03 %** | 16.5 % | 0.3841 | **−45.36 %** | 2009-02 | 43 mo | 59 mo | 2.70× | 2.55 % |
| 20 % | 10.72 % | 21.9 % | 0.3668 | **−57.05 %** | 2009-02 | 64 mo | 80 mo | 3.59× | 3.89 % |
| 25 % | 11.28 % | 27.2 % | 0.3641 | **−66.78 %** | 2009-02 | 106 mo | 122 mo | 4.47× | 5.20 % |
| 30 % | 11.52 % | 32.4 % | 0.3628 | **−74.78 %** | 2009-02 | 107 mo | 123 mo | 5.34× | 6.50 % |
| 40 % | 11.11 % | 42.5 % | 0.3650 | **−90.58 %** | 1980-03 | 69 mo | **239 mo** | 7.00× | 9.00 % |

### Risk parity, bucketed / All-Weather style (W2) — levered

| τ | compound | Sharpe | **max DD** | peak → trough | u/w |
|---:|---:|---:|---:|---|---:|
| 10 % | 7.68 % | 0.3137 | **−32.18 %** | 2020-12 → 2022-10 | 57 mo |
| 15 % | 7.89 % | 0.2670 | **−47.17 %** | 2012-09 → 2022-10 | **161 mo** |
| 20 % | 7.78 % | 0.2436 | **−62.82 %** | 1972-12 → 1980-02 | 119 mo |
| 25 % | 7.33 % | 0.2294 | **−76.88 %** | 1972-12 → 1980-02 | 126 mo |
| 30 % | 6.79 % | 0.2260 | **−86.12 %** | 1972-12 → 1980-03 | 142 mo |
| 40 % | 5.75 % | 0.2350 | **−93.68 %** | 1972-12 → 1980-03 | 146 mo |

**The bucketed book's compound return falls monotonically with leverage from τ = 15 % up.**
Its worst drawdown at the lowest rung is the 2020-2022 rate shock and at τ = 15 % it is a
**ten-year decline, 2012-09 to 2022-10, not recovered until 2026-02**. That is the live
risk-parity failure mode, reproduced on this panel.

At **2 bps** instead of 10 bps every compound figure moves by ≤ 0.13 pp and no drawdown by
more than 0.17 pp. **Trading costs are irrelevant to this study's verdict.** Financing is not.

## 2b. Why the ladder saturates — the attribution, measured

Equal weight, 10 bps, bill + 150 bp. Everything below is a measured mean of the realised
monthly series, not an identity assumed in advance:

| τ | cash | levered gross excess | trading cost | **financing** | = arithmetic | − **variance drag** | = **compound** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 % | 4.64 % | 6.49 % | −0.02 % | −0.42 % | 10.69 % | 0.13 % | **10.56 %** |
| 15 % | 4.64 % | 9.74 % | −0.04 % | −1.32 % | 13.03 % | 0.73 % | **12.30 %** |
| 20 % | 4.64 % | 12.99 % | −0.05 % | −2.25 % | 15.32 % | 1.68 % | **13.64 %** |
| 30 % | 4.64 % | 19.52 % | −0.07 % | −4.13 % | 19.97 % | 4.71 % | **15.26 %** |
| 40 % | 4.64 % | 26.43 % | −0.09 % | −5.95 % | 25.03 % | **9.22 %** | **15.81 %** |

Going from τ = 20 % to τ = 40 % **doubles the gross excess return** (13.0 % → 26.4 %) and buys
**2.2 points of compound return**, because financing takes 3.7 points and variance drag takes
7.5. The drawdown meanwhile goes from −59 % to −89 %. That is the whole argument against
leverage as a route to the target, in one table.

**Two caveats that must travel with every compound number above.** First, **4.64 pp of it is
the average 13-week bill rate over 1965-2026** — a rate regime, not a strategy. Second, the
same book compounded at **5.86 %/yr through the 2010s**, when the bill paid 0.56 %:

| decade | EW compound @ τ=15 % | its max DD | RP compound @ τ=15 % | bill rate |
|---|---:|---:|---:|---:|
| 1960s (48 mo) | 1.57 % | −37.4 % | −10.12 % | 5.17 % |
| 1970s | 10.36 % | −39.9 % | 5.35 % | 6.51 % |
| 1980s | **23.00 %** | −22.9 % | 21.94 % | **9.20 %** |
| 1990s | 19.59 % | −18.8 % | 19.15 % | 4.99 % |
| 2000s | 7.88 % | **−47.3 %** | 9.65 % | 2.75 % |
| 2010s | **5.86 %** | −28.0 % | 6.02 % | **0.56 %** |
| 2020s (78 mo) | 12.76 % | −28.9 % | 7.14 % | 2.87 % |

The only decades that got near 20 %/yr are the two with the highest bill rates and the
strongest bond bull. **The headline 12.30 % is a 61-year average that no decade since the
1990s has matched.**

## 2c. Financing is decisive, and the ladder inverts at retail rates

Risk parity (W1), 10 bps, compound / max DD:

| financing | τ=10 % | τ=15 % | τ=20 % | τ=30 % | τ=40 % |
|---|---:|---:|---:|---:|---:|
| bill + 50 bp (institutional) | 9.98 % / −30 % | 11.91 % / −44 % | 13.62 % / −55 % | 16.45 % / −73 % | **17.99 %** / −85 % |
| **bill + 150 bp (pre-registered primary)** | 9.11 % / −32 % | 10.03 % / −45 % | 10.72 % / −57 % | 11.52 % / −75 % | 11.11 % / −91 % |
| **bill + 300 bp (retail margin)** | **7.81 %** / −33 % | 7.26 % / −49 % | 6.50 % / −69 % | 4.47 % / −91 % | **1.46 %** / −98 % |
| legacy flat 6 % (prior repo convention) | 8.64 % / −34 % | 9.39 % / −67 % | 9.91 % / −85 % | 10.31 % / −97 % | 9.68 % / −99.5 % |

**At retail margin rates the ladder inverts: every rung above 10 % vol makes you poorer and
multiplies your drawdown.** Moving the borrow spread from +50 bp to +300 bp — a 250 bp
change — moves the τ = 40 % compound return by **16.5 percentage points**, more than every
strategy decision in this study combined; the +50 bp → +150 bp step alone costs **6.9 pp**.
The mean bill rate over the sample is 4.63 %, so the legacy flat 6 % is a *subsidy*
in the 1970s-80s (bills at 9-15 %) and a *penalty* since 2009 (bills near zero); its
drawdowns are correspondingly distorted, which is why the time-varying charge is the correct
one and is what the headline uses.

---

## 3. Risk parity **loses to equal weight**, and the reason is the leverage it needs

| book | net Sharpe (10 bps) | vol | mean excess | unlevered compound | unlevered max DD |
|---|---:|---:|---:|---:|---:|
| **W0 equal weight** | **0.6678** | 8.80 % | 5.87 % | 10.60 % | −26.6 % |
| W1 risk parity, naive | 0.6483 | 6.21 % | 4.03 % | 8.80 % | −14.6 % |
| W2 risk parity, bucketed | 0.5513 | 5.19 % | 2.86 % | 7.60 % | −12.4 % |

**Vol-matched active return, risk parity minus equal weight, both levered to the same target:**

| τ | vol-matched active | NW t | Jensen α | geometric excess |
|---:|---:|---:|---:|---:|
| 10 % | **−1.35 %/yr** | **−2.67** | −1.01 % (t −2.03) | −1.40 % |
| 15 % | **−2.07 %** | **−2.74** | −1.60 % (t −2.15) | −2.17 % |
| 20 % | **−2.63 %** | **−2.62** | −2.04 % (t −2.08) | −2.80 % |
| 25 % | −2.99 % | −2.33 | −2.26 % (t −1.81) | −3.19 % |
| 30 % | −3.40 % | −2.15 | −2.51 % (t −1.64) | −3.58 % |
| 40 % | −4.50 % | −2.17 | −3.28 % (t −1.64) | −4.51 % |

**Negative at every rung, and significantly so.** There is no breakeven trading cost: the
active return is negative at **zero** trading cost too (−0.15 %/yr unlevered).

**The mechanism, isolated.** Unlevered, the gap is small and insignificant: **−0.17 %/yr,
t = −0.39**. Under a *zero-financing counterfactual* (kept only as a diagnostic — leverage is
never free) the levered gap stays insignificant at −0.55 % to −1.23 %, t between −0.6 and
−1.1. It is the financing charge that makes the gap real:

> Risk parity's unlevered volatility is **6.21 %** against equal weight's **8.80 %**, so
> reaching any given volatility target requires **~44 % more leverage** — a measured 2.70×
> versus 1.88× at τ = 15 %. The interest on that extra notional (1.32 % → 2.55 %/yr) is
> **larger than the entire diversification benefit it buys**. Risk parity's advantage is
> exactly cancelled by the cost of the leverage needed to express it.

## 3b. And the sizing rule itself tilts toward the assets with no risk premium

The bucketed book's blocks, measured (mean notional weight, the block's own realised
sub-portfolio volatility, and what that block actually paid):

| block | **mean weight** | block vol | **block Sharpe** | block mean excess |
|---|---:|---:|---:|---:|
| **fx** | **36.6 %** | 6.10 % | **−0.020** | **−0.14 %/yr** |
| **rates** | **39.3 %** | 6.83 % | 0.193 | +1.39 %/yr |
| equity | 20.4 % | 13.48 % | 0.656 | +9.04 %/yr |
| commodity | **3.7 %** | 18.02 % | 0.683 | +12.73 %/yr* |

\* the commodity block only exists from 2003, so its figure is a short, hindsight-flattered sample.

**75.9 % of the bucketed book sits in the two blocks whose own returns were −0.14 %/yr and
+1.39 %/yr, and 3.7 % sits in the block with the highest Sharpe.** The FX block gets the largest single allocation
*because* it has the lowest volatility, and it has the lowest volatility because it is
internally hedged — long USD via USDX against long EUR, GBP and JPY, mean member volatility
8.87 % collapsing to a 6.10 % block volatility. Two-level risk parity reads that hedge as
diversification and rewards it with a third of the portfolio.

**This is the general failure.** Inverse-volatility sizing is the optimal prior *only if
Sharpe ratios are equal across assets.* On this panel they are emphatically not: spot FX has
no risk premium at all (it is a rate differential the panel deliberately excludes), and bonds
have one only inside the bond bull market (§4). Equal weight, which knows nothing, does not
make that mistake.

---

## 4. **THE BOND BULL MARKET — this is what kills it**

Excluding 1981-10 → 2021-12 (255 months survive: 1965-01→1981-09 and 2022-01→2026-06):

| book | full sample | **excl. bond bull** | inside it | pre-1981-10 | post-2021-12 |
|---|---:|---:|---:|---:|---:|
| **W0 equal weight** | 0.6678 | **0.4387** | 0.7723 | 0.3582 | 0.7038 |
| **W1 risk parity** | 0.6483 | **0.1603** | 0.9003 | 0.0596 | 0.4961 |
| **W2 bucketed** | 0.5513 | **0.1513** | 0.7946 | 0.1175 | 0.2818 |

**Risk parity's Sharpe falls by 0.49 — from 0.648 to 0.160 — when the 40-year bond bull
market is removed. Equal weight's falls by 0.23.** Risk parity is more than twice as
dependent on that single regime, which is exactly what a book holding 46 % of its notional in
three Treasury points should be.

The instruments say it plainly. **US Treasury excess returns, inside versus outside the bond
bull:**

| instrument | inside: Sharpe / mean | **outside: Sharpe / mean** |
|---|---:|---:|
| US 5y | +0.554 / +2.76 %/yr | **−0.379 / −2.15 %/yr** |
| US 10y | +0.540 / +4.27 %/yr | **−0.485 / −3.60 %/yr** |
| US 30y | +0.503 / +6.68 %/yr | **−0.939 / −12.70 %/yr** |

**Outside 1981-2021, holding duration cost between 2 % and 13 % a year over bills.** A sizing
rule that hands duration the largest weight in the book *because it is quiet* is a bet that
1981-2021 repeats.

**Rates removed entirely** (15 instruments, full pipeline re-run, 1974-01 → 2026-06, 630
months) — compared on the identical window against the 18-instrument books:

| book | 18 instruments, same window | **15 instruments, no rates** | cost of removing bonds |
|---|---:|---:|---:|
| equal weight | 0.6896 | 0.6702 | −0.019 |
| risk parity | 0.7173 | 0.6731 | **−0.044** |
| bucketed | 0.5906 | 0.5224 | −0.068 |

Removing bonds costs risk parity more than twice what it costs equal weight — but note that
this 630-month window is itself **77 % inside the bond bull** (483 of 630 months), which is
why the exclusion test above, not this one, is the decisive version.

---

## 5. Concentration — risk parity does concentrate more, as predicted

**Gross notional** (unlevered weights; the levered book multiplies all of these by `k`):

| book | top-1 mean | top-1 max | **top-3 mean** | top-5 mean | eff. N (weights) | min |
|---|---:|---:|---:|---:|---:|---:|
| equal weight | 12.1 % | 33.3 % | 36.2 % | 54.8 % | 11.30 | 3.0 |
| **risk parity naive** | 24.6 % | **58.6 %** | **53.8 %** | 70.6 % | 8.48 | 2.06 |
| **bucketed** | 28.8 % | 52.3 % | **63.7 %** | 81.3 % | 6.34 | 2.56 |

Mean block weights: equal weight = equity 46.8 / rates 30.6 / fx 14.2 / commodity 8.4 %;
risk parity = **rates 46.0** / equity 31.0 / fx 18.8 / commodity 4.2 %; bucketed = rates 39.3 /
**fx 36.6** / equity 20.4 / commodity 3.7 %.

Top-3 share by decade (risk parity): 1960s **95.3 %** · 1970s 79.7 % · 1980s 55.4 % · 1990s
51.1 % · 2000s 39.0 % · 2010s 35.1 % · 2020s 34.6 %, against a mean eligible count of
3.4 / 5.2 / 7.3 / 10.4 / 15.2 / 18.0 / 18.0. **The first two decades of this 61-year sample
are a three-to-five instrument book, not an 18-instrument one**, and they are the decades in
which risk parity is worst (Sharpe −0.367 in the 1960s, +0.287 in the 1970s).

**"18 instruments" is not 18 bets.** On the 1996-onward window where all 18 are live, the
correlation-matrix effective N of the universe is **5.26**. Per block: rates **1.17** (three
Treasury maturities are one bet), fx 1.92, equity 2.03, commodity 2.67.

**P&L concentration is clean.** At τ = 20 %, the largest single (instrument, month) cell is
**0.28 %** of gross absolute P&L for equal weight, 0.18 % for risk parity, 0.22 % for bucketed
— an order of magnitude inside the 3 % alarm, as a passive book should be. Top instrument
share is **N225 at 23.2 % / 20.9 % / 21.7 %** respectively. Block P&L share for equal weight:
equity 76.6 %, commodity 19.7 %, rates 5.0 %, fx −1.3 %.

That N225 figure deserves its own sentence. **The single largest P&L contributor to all three
books is the Nikkei**, which is in the universe because it still exists and still has a
continuous price history back to 1965 — a survivorship criterion, not a forecast. Which
brings us to:

## 5b. The 18 instruments are hindsight-selected survivors — bias direction: **UPWARD**

Stated in the pre-registration before the run and restated here unchanged. This universe
contains no market that closed, no sovereign that defaulted, no exchange that stopped
publishing, no currency that was redenominated. Every number in this document is therefore
an **upper bound** on what was investable ex ante, and it is the largest single bias present.
Two smaller biases push the other way and are also not repaired: the equity legs are
**price-only** (dividends excluded, ≈2-4 %/yr depending on era; DAX alone is total-return) and
the bond legs are constant-maturity par-bond repricings with **no roll-down** (≈0.5 %/yr).
The commodity legs are front-month continuous with roll gaps not back-adjusted.

---

## 6. The DSR gate — applied to the benchmark too, as mandated

Bar at 61.5 years:

| n_trials | DSR ≥ 0.95 bar | W0 equal weight 0.6678 | W1 risk parity 0.6483 | W2 bucketed 0.5513 |
|---:|---:|:--:|:--:|:--:|
| 32 | 0.4801 | clears | clears | clears |
| **46** (honest cumulative) | **0.4988** | **clears** | **clears** | **clears** |
| 56 | 0.5086 | clears | clears | clears |
| **304** (inherited-search bar) | **0.5840** | **clears** | **clears** | **fails** |

**Required for 30 %/yr at half Kelly: 0.894. Nothing here reaches it, and the ladder in §1
shows that even a strategy that did would not survive the leverage it implies.**

---

## 7. Decades

Unlevered net excess Sharpe, 10 bps:

| decade | months | equal weight | risk parity | bucketed | mean eligible |
|---|---:|---:|---:|---:|---:|
| 1960s | 60 | 0.271 | **−0.367** | −0.019 | 3.4 |
| 1970s | 120 | 0.492 | 0.287 | 0.155 | 5.2 |
| 1980s | 120 | 0.947 | 0.913 | 0.863 | 7.3 |
| 1990s | 120 | 0.969 | **1.021** | 0.873 | 10.4 |
| 2000s | 120 | 0.388 | 0.648 | 0.797 | 15.2 |
| 2010s | 120 | 0.611 | 0.784 | 0.548 | 18.0 |
| 2020s | 78 | 0.785 | 0.573 | 0.357 | 18.0 |

Risk parity's two best decades are the 1980s and 1990s — the heart of the bond bull. Its
worst is the 1960s, when the eligible universe was 3.4 instruments and roughly two-thirds
bonds by inverse-vol weight.

---

## 8. Verdict against the pre-registered rule (§9 of the prereg)

`S` = W1 risk parity net 10 bps = **0.6483** (selected mechanically as the better of W1/W2).

| PROMOTABLE AS ALPHA — required | measured | pass |
|---|---|:--:|
| 1. `S ≥ dsr_bar(n=46)` = 0.4988 | 0.6483 | ✅ |
| 2. **vol-matched active vs EW > 0 with t ≥ +2.0** | **−1.35 %/yr, t = −2.67** | ❌ |
| 3. no decade with a negative Sharpe | **1960s = −0.367** | ❌ |
| 4. **Sharpe excl. bond bull > 0 and ≥ 0.50 × full** | **0.160 vs 0.324 required** | ❌ |
| 5. mean top-3 gross notional < 70 % | 53.8 % | ✅ |
| 6. top (instrument, month) P&L cell < 3 % | 0.18 % | ✅ |

| DEPLOYABLE AS BETA — required | measured | pass |
|---|---|:--:|
| 1. `S ≥ dsr_bar(n=46)` | 0.6483 | ✅ |
| 2. Sharpe excl. bond bull > 0 | 0.160 | ✅ |
| 3. **some τ gives compound ≥ 12 %/yr at DD ≤ 50 %** | **best is 10.03 % at −45.4 %** | ❌ |

| ANSWERS THE 30 % QUESTION | measured | pass |
|---|---|:--:|
| some τ gives compound ≥ 30 %/yr at DD ≤ 50 % | max at **any** leverage is **15.83 %** at −87.8 % | ❌ |

## → **DEAD.**

**And the finding underneath the verdict is larger than the verdict.** Equal weight *does*
clear the beta bar this study set for risk parity — 12.30 %/yr at −47.3 %. The benchmark that
killed trend, value, carry and seasonality has now killed risk parity too, on the fifth
attempt, and this time it killed a construction whose entire purpose was to improve it.
**Risk-budgeting the passive book made it worse, at every leverage level, significantly.**

---

## 9. Verification — every control run before the result was read

| control | result |
|---|---|
| point-in-time audit: weights rebuilt from a panel **truncated at t** vs full-sample row t, 10 month-ends × 3 schemes | **max difference 0.0** ✅ |
| **sign-flip control** — flip each instrument's whole return series; volatility is sign-invariant so the weights must be *identical* and the Sharpe must collapse | weights **bit-identical (0.0)**; Sharpe −0.363 / +0.056 / +0.027 / +0.378 across 4 seeds, **mean +0.024** ✅ |
| leverage invariance with the 10× gross cap **lifted** | gross Sharpe identical across all six τ, all three books (max−min < 1e−9) ✅ |
| leverage invariance with the cap **on** | breaks exactly and only where the cap binds (EW 4 months at τ=30 %, 7 at τ=40 %; RP 90 at τ=40 %) — reported, not repaired ✅ |
| financing arithmetic: at k ≡ 1 the charge is exactly zero and `R − cash` equals the unlevered book net of costs | **0.0** and **0.0** ✅ |
| weights sum to 1 whenever the book is live | max error **0.0** for all three schemes ✅ |
| DSR bar reproduces both recorded anchors (1.4881 @ 7 yr, 0.5971 @ 40 yr, n=32) | ✅ |
| re-run byte-identical | payload md5 `57d2571b…` on three consecutive runs ✅ |
| **EW reproduces the recorded 0.7065 ± 0.03** | **FAILED in the monthly construction (0.6678, off by 0.039); PASSES when rebuilt daily (0.6854, off by 0.021)** — see below |
| **headline number rebuilt from scratch** on a second, independent code path that imports none of `riskparity.py` | compound **0.122955**, vol 0.164628, Sharpe 0.509673, max DD **−0.472874**, peak 2007-10, trough 2009-02, recovery 2014-02, 60 mo to recover, 76 mo under water, mean leverage 1.8769, arithmetic total 0.130284 — **identical to the reported figures in every digit** ✅ |

**The one control that failed, and what it means.** The pre-registration required this
pipeline's equal-weight book to reproduce the recorded benchmark Sharpe of 0.7065 within
±0.03. **A monthly-rebalanced equal-weight book returns 0.6678 — it fails by 0.009.** The
cause was identified rather than accommodated: the recorded 0.7065 comes from a
**daily-rebalanced** book (the seasonal sleeve runs on the daily panel). Rebuilding a
daily-rebalanced equal-weight book here, over the same eligible set, gives **0.6854** — inside
tolerance, and confirming that **daily rebalancing is worth ≈ +0.018 of Sharpe** over monthly
on this universe. Two independent checks corroborate the monthly number: the trend sleeve's
own `run_trend` benchmark, computed by code this study did not touch, returns **0.6695
gross / 0.6691 net**, which this pipeline reproduces to 0.002.

**Consequence, stated plainly: the "0.7065 passive benchmark" that four prior studies were
measured against is a daily-rebalanced construction. The monthly-rebalance version of the
same book is 0.668.** That does not change any prior verdict — every one of those studies
compared like with like inside its own pipeline — but the headline number carries a rebalance
convention that was not previously stated, and anyone quoting 0.7065 should quote it as
*daily-rebalanced equal weight*.

---

## 10. Predictions scorecard (prereg §10, scored honestly)

| # | prediction | point / band | measured | verdict |
|---|---|---|---:|---|
| P1 | W1 net 10 bps Sharpe | 0.80 [0.65, 0.95] | **0.648** | ❌ missed, just below |
| P2 | W2 net 10 bps Sharpe | 0.82 [0.65, 1.00] | **0.551** | ❌ missed badly |
| P3 | W0 reproduces the benchmark | 0.70 [0.68, 0.73] | 0.685 daily / **0.668 monthly** | ✅ daily / ❌ monthly |
| P4 | RP vs EW vol-matched active **positive**, t < +2.0 | +1.5 %/yr | **−1.35 %/yr, t −2.67** | **REFUTED — wrong sign AND significant** |
| P5 | max DD at τ = 20 % | −42 % [−30, −60] | **−57.1 %** | ✅ in band |
| P6 | max DD at τ = 40 % | −75 % [−55, −95] | **−90.6 %** | ✅ in band |
| **P7** | **highest compound at DD ≤ 50 %** | **15 % [9, 21], not ≥ 30 %** | **12.30 %** | ✅ **in band, and not ≥ 30 %** |
| P8 | mean top-3 gross notional, W1 | 55 % [40, 70] | **53.8 %** | ✅ near-exact |
| P9 | RP Sharpe excl. bond bull, fall ≥ 0.20 | 0.45 [0.10, 0.75] | **0.160**, fall **0.488** | ✅ in band |
| P10 | RP Sharpe on the 15-instrument universe | 0.65 [0.45, 0.85] | **0.673** | ✅ near-exact |
| P11 | clears DSR at n = 46, fails 0.894 | clears / fails | 0.648 vs 0.499 and 0.894 | ✅ both |
| P12 | τ = 40 % unsurvivable (DD worse than −60 %) | unsurvivable | **−90.6 %** | ✅ |
| P13 | 2022-01→2026-06 RP Sharpe **negative** | −0.30 [−1.00, +0.30] | **+0.496** | **REFUTED** |

**Eight hits, three misses, two refutations.**

The prereg named **P4 as the prediction most likely to be wrong**, on the argument that if
risk parity's mechanism is real the active return could be *significantly positive*. It was
wrong in the **opposite** direction: significantly **negative**. The reasoning that flagged it
was right that the sizing question was the live one, and wrong about which way it cut.

**P13 is the more interesting refutation.** Risk parity was expected to be killed by the
2022 rate shock, and on this panel it was not: 2022-01→2026-06 gives Sharpe +0.496 (equal
weight +0.704). The bucketed book *is* damaged there — its worst drawdown at τ = 10 % is
exactly the 2020-12 → 2022-10 rate shock, and at τ = 15 % it is a ten-year slide ending in
October 2022 — but the naive inverse-vol book recovered inside the sample. **The bond bull
exclusion, not the rate shock, is what exposes the dependence**; a 54-month window is too
short to settle it either way.

---

## 11. What this buys the programme

1. **The 30 % target is arithmetically unreachable on this panel, and now the reason is
   measured rather than argued.** Compound return is concave in leverage and **peaks at
   15.83 %/yr** with an −87.8 % drawdown. Financing (up to 5.95 %/yr) and variance drag (up to
   9.22 %/yr) consume the levered excess return before it ever approaches the target. The
   Kelly ceiling `S²/2` was never the binding constraint — it assumes free leverage and no
   drawdown limit, and both assumptions fail here by large margins.
2. **The honest deliverable is 12.30 %/yr at a −47 % drawdown and 6.3 years under water** —
   from levering plain equal weight 1.9×, at institutional-ish financing, on a
   hindsight-selected universe of survivors, in a sample whose average bill rate was 4.63 %.
   Every one of those qualifiers pushes the realisable number down.
3. **Risk-budgeting the passive book made it worse, and the mechanism generalises.**
   Inverse-volatility sizing is the right prior only when Sharpe ratios are equal across
   assets. Here it put 46 % of the book in duration (Sharpe 0.19 full-sample, **negative
   outside 1981-2021**) and, in bucketed form, 37 % in an internally-hedged FX block with a
   Sharpe of **−0.02**. **Do not use inverse-vol sizing on a universe containing assets with
   no risk premium.** Spot FX with no carry leg is such an asset, and this panel has four of
   them.
4. **Leverage is not free, and the spread matters more than any strategy choice in this
   repo.** Moving the borrow spread from +50 bp to +300 bp changes the τ = 40 % compound
   return by **16.5 percentage points** and inverts the entire ladder. Any future study that
   quotes a levered return without naming its financing rate is quoting a number that does
   not exist.
5. **The benchmark's rebalance convention is now on the record.** 0.7065 is
   *daily-rebalanced* equal weight; the monthly version is 0.668, and the +0.018 gap is a
   measured rebalancing premium, not noise.
6. **Where a real edge would have to come from.** Five constructions have now lost to
   diversified passive. The one thing this study did *not* test, and the only remaining
   structural lever it exposes, is that the universe's correlation effective N is **5.26** —
   the breadth this programme keeps trying to buy with signals is not in these 18 instruments
   at all. Adding genuinely independent return streams raises the passive Sharpe directly;
   nothing tested tonight raises it at all.

---

**One run. No tuning. No second look. Banked as DEAD — and the survivable-drawdown number,
12.30 %/yr at −47.3 %, is the honest ceiling this panel supports.**
