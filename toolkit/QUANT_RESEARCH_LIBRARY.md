# Quant-Trading Research Library — best free resources, mapped to TradingEngineResearch

**Mandate:** don't reinvent — find the best existing, free, PhD-grade resources and
use them flawlessly. TradingEngineResearch already implements the *right concepts* (purged CV,
meta-labeling, Black-Litterman + CVaR, vol-targeting); the audit showed they need
**hardening, not replacement**. Each resource below is tagged with the TradingEngineResearch
gap (from the internal remediation roadmap) it directly helps close.

---

## 1. Platforms & libraries (free, open-source)

| Resource | What it is | Use it for TradingEngineResearch (→ roadmap stage) |
|----------|-----------|-------------------------------------------|
| **Microsoft Qlib** (+ **RD-Agent**) `github.com/microsoft/qlib` | Full AI quant pipeline: data → features (Alpha158/Alpha360) → ML models → backtest → portfolio/exec; RD-Agent automates the research loop. 39k★, active 2026. | **Stage B richer data/alpha**: adopt its Alpha158/360 cross-sectional feature sets + data pipeline as a far richer feature source than TradingEngineResearch's ~6 price features (the documented "price-only has no alpha" gap). Reference pipeline to cross-check our engine. |
| **mlfinlab** (Hudson & Thames) | Reference implementation of López de Prado *Advances in Financial ML*: purged & **combinatorial purged** CV, **meta-labeling**, fractional differentiation, **Deflated Sharpe**, **PBO**, sample weights. | **Stage A1 (SIGNALS-6 keystone)**: replace our `deflated_sharpe_proxy`/`pbo_proxy` placeholders with the *real* Deflated Sharpe + PBO (the exact missing §11 machinery that lets junk pass). Gold-standard reference for our existing `PurgedWalkForwardSplitter` + meta-labeler. |
| **awesome-quant** `github.com/wilsonfreitas/awesome-quant` | The curated meta-list of quant libraries/data/resources. | Start here when you need a tool for any new sub-problem (TCA, options, risk, data). |
| **QuantConnect / Lean** `quantconnect.com` | Open-source backtesting engine + data. | **Cross-validation**: re-run a TradingEngineResearch strategy in Lean to catch methodology/backtest bugs independently (directive's "independent verification"). |
| **pyfolio** | Portfolio/risk tear-sheets. | **Stage F reporting**: automated performance/attribution reports. |
| **DeepDow** | Deep-learning portfolio optimization. | **Stage E experimental** candidate only — compare vs the incumbent BL+CVaR optimizer under identical costs. |

## 2. Methodology / canonical knowledge (free)

- **Marcos López de Prado — *Advances in Financial Machine Learning*** (the bible). Core ideas TradingEngineResearch
  already uses (and must harden): **purged + embargoed CV**, **combinatorial purged CV (CPCV)**,
  **meta-labeling**, **backtest overfitting**, **Deflated Sharpe Ratio**, **Probability of Backtest Overfitting (PBO)**,
  sample uniqueness/weights, fractional differentiation. → directly informs Stage A1 + Stage B validation.
- **Bailey & López de Prado papers (SSRN, free):** *The Deflated Sharpe Ratio*, *The Probability of Backtest
  Overfitting*, *Pseudo-Mathematics and Financial Charlatanism*, *The Sharpe Ratio Efficient Frontier*.
  → the exact formulas to implement for SIGNALS-6 (junk-rejection) and to honestly evaluate any >30% claim.
- **Stefan Jansen — *Machine Learning for Trading*** (book + `github.com/stefan-jansen/machine-learning-for-trading`,
  full code). → practical ML alpha pipelines on free data; a template for Stage B.
- **Georgia Tech CS 7646 — ML for Trading** (free lectures/course materials). → foundations, no-cost.

## 3. Data (free, point-in-time-aware where possible)

| Source | Content | TradingEngineResearch use |
|--------|---------|------------------|
| **Fama-French Data Library** | Factor returns (value, size, momentum, profitability, investment, ...). | Stage B: factor baselines + neutralization; "is it alpha or factor beta?" attribution. |
| **FRED** (St. Louis Fed) | Macro time series. | Stage B: macro/regime features (rates, spreads, vol). |
| **SEC EDGAR** (APIs) | Filings, fundamentals, insider (Form 4), 13F. | Stage B: fundamental + insider features (the richer data the audit says is needed for real alpha). |
| **yfinance** | Educational price/volume. | Already used; fine for research, NOT a commercial data licence. |
| ⚠️ **Survivorship-free PIT universe** | point-in-time index membership + delisted names | **The real gap** (METH-1). Some free (e.g. Wikipedia historical constituents) but clean PIT membership often costs — this is the owner data-budget decision in the roadmap. |

## 4. Notable recent papers (free, arXiv)
- **RiskMiner: Discovering Formulaic Alphas via Risk-Seeking Monte-Carlo Tree Search** (arXiv 2402.07080) —
  automated formulaic-alpha discovery; a candidate technique for the experimental engine.
- **ML-Enhanced Multi-Factor Quantitative Trading: Cross-Sectional Portfolio Optimization with Bias Correction**
  (arXiv 2507.07107) — cross-sectional ML factor construction with bias correction; directly relevant to fixing
  our market-timing IC into a true cross-sectional one (SIGNALS-3/4 follow-up).

---

## 5. How to use this flawlessly (the optimize-don't-reinvent playbook)

1. **Fix the validator first using mlfinlab + Bailey/LdP as the reference** → implement real Deflated Sharpe + PBO
   (closes SIGNALS-6; nothing downstream is trustworthy until junk is robustly rejected).
2. **Then widen the data, not the knobs** → wire Fama-French + FRED + EDGAR (and/or Qlib's Alpha158) as new
   PIT-safe features. The audit *proved* price-only has no robust alpha; richer data is the only credible path
   to (and honest test of) >30%.
3. **Validate every candidate the López de Prado way** → CPCV + Deflated Sharpe + PBO + cost/capacity sensitivity,
   on a survivorship-free universe. Reject anything that dies under modest costs/delays.
4. **Cross-check with an independent engine (Lean/Qlib)** → if TradingEngineResearch and Lean disagree on a backtest, find
   the bug before trusting the number.
5. **Report honestly with pyfolio** → drawdown-aware, net-of-cost tear-sheets; never CAGR alone.

**Bottom line:** TradingEngineResearch's architecture is sound and uses the right ideas. The fastest path to a *credible*
edge is to (a) harden the validator with these reference implementations and (b) feed it richer free data — not
to rebuild the engine. Every item above maps to an existing roadmap stage.

---

## Sources
- Microsoft Qlib — https://github.com/microsoft/qlib · RD-Agent — https://github.com/microsoft/RD-Agent
- awesome-quant — https://github.com/wilsonfreitas/awesome-quant
- mlfinlab (Hudson & Thames) — https://github.com/hudson-and-thames/mlfinlab
- QuantConnect/Lean — https://github.com/QuantConnect/Lean
- Stefan Jansen, ML for Trading — https://github.com/stefan-jansen/machine-learning-for-trading
- López de Prado / Bailey papers — https://ssrn.com (search "Deflated Sharpe Ratio", "Probability of Backtest Overfitting")
- Fama-French Data Library — https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- FRED — https://fred.stlouisfed.org · SEC EDGAR — https://www.sec.gov/edgar
- arXiv: 2402.07080 (RiskMiner), 2507.07107 (cross-sectional ML factors)
