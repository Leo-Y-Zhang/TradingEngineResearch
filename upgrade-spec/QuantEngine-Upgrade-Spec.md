# TradingEngineResearch Quant Engine — Research & Upgrade Specification
**Classification:** Internal R&D  
**Version:** 2.0 Draft  
**Scope:** Research-backed weakness analysis, prioritised proposals, and exact spec language for highest-impact upgrades

---

## PART I — WEAKNESS AUDIT BY COMPONENT

### 1. Black-Scholes PDE (IV Extraction & Put Pricing)

**Current role:** Analytical ATM put pricing for tail-hedge cost surfacing; implied volatility extraction.

**Documented weaknesses:**

Black-Scholes assumes constant volatility and log-normally distributed returns — both empirically refuted. The model cannot reproduce the volatility smile or skew observed across strikes and maturities (Ye, ICSSED 2022). When BS is used to price OTM protective puts specifically, it systematically underprices them because it does not account for the negative skew premium embedded in far-OTM options (Bongaerts et al., 2020). This is compounding: the CrisisManager uses BS put costs to evaluate tail-hedge affordability, meaning it is anchoring on a structurally cheap estimate precisely when hedges are most needed.

Additionally, BS implied volatility inversion via Newton-Raphson fails or is poorly conditioned for deep ITM/OTM options, producing unreliable greeks that feed into the protective-put cost model.

**Key empirical finding:** During volatility spikes, OTM put skew expands from ~3–5 vol points over ATM to 15–25+ vol points (HL Hunt, 2025), meaning a BS-priced put can understate the true market cost by 30–60% in crisis conditions — exactly when CrisisManager is surfacing costs to the risk overlay.

---

### 2. GARCH(1,1) for Realised Volatility

**Current role:** Realised volatility estimation, feeding vol targeting and CrisisManager detectors (5d/60d ratio).

**Documented weaknesses:**

GARCH(1,1) uses only daily close-to-close returns and a single lag structure. Its in-sample fit is reasonable, but out-of-sample forecasting performance is substantially inferior to models that use intraday high-frequency data. Nelson (1992) theoretically demonstrated GARCH's strong in-sample performance but poor out-of-sample predictions. Comparative studies show the HAR-RV model (Corsi, 2009) reduces forecast errors by approximately 35–40% over GARCH(1,1) across all standard evaluation metrics, with no statistically significant incremental gain from adding LSTM on top (ResearchGate, 2024).

The GARCH(1,1) 5d/60d ratio used by CrisisManager to detect "vol explosion" is particularly problematic: GARCH vol estimates are slow to react to sudden volatility jumps because the model is a geometric-decay filter — it spreads the shock over many days. A HAR-RV or realised-variance proxy using high-low range data would react to intraday volatility expansions within the same session.

**Key empirical finding:** Celik & Ergin (2014), Ma et al. (2014), Bergsli et al. (2022), and Sapkota (2022) all confirm HAR-RV superiority for one-day-ahead forecasts. HAR-RV is a consensus benchmark in the academic literature; GARCH(1,1) for forecasting is considered outdated for daily equity volatility.

---

### 3. ML Return Regressor (Gradient-Boosted Trees)

**Current role:** Predicts E[return] and Std[return] per asset; outputs used as BL views.

**Documented weaknesses:**

**A. Cross-validation leakage.** Standard k-fold CV and even walk-forward backtesting are subject to look-ahead bias when labels overlap in time with training features. López de Prado (2018) demonstrated that standard CV yields overly optimistic performance estimates in financial settings due to serial correlation in labels and features. Without purged cross-validation and an embargo window, the GBT model's reported training accuracy is likely upward-biased, meaning shrinkage calibration may be insufficient to correct for genuine overfitting in production.

**B. No distributional uncertainty.** The regressor produces point estimates of E[return] and Std[return]. The Std[return] output is used as a proxy for epistemic uncertainty in the BL Ω matrix, but it is a biased estimator: GBT conditional variance estimation via residuals does not provide coverage-guaranteed prediction intervals. In non-stationary regimes the true predictive uncertainty is substantially higher than the empirical residual variance suggests.

**C. Feature stationarity.** GBT operates on raw price-derived features that are non-stationary time series. Without fractional differentiation (López de Prado, 2018) or explicit stationarity enforcement, the model fits a spurious relationship between level-correlated features and returns, which degrades generalisation across regimes.

---

### 4. Black-Litterman Portfolio Optimisation

**Current role:** μ_BL combines market equilibrium prior with ML views, TradingEngineResearch forecasts, and insider flow scores.

**Documented weaknesses:**

**A. τ calibration is arbitrary.** τ governs the relative weight of the prior vs. views and is theoretically described as "close to zero" (Black & Litterman, 1992) but is notoriously difficult to calibrate in practice. The literature provides no consensus on the right value. Meucci (2010) showed that different τ values can produce radically different portfolio tilts even with identical views, making the system's behaviour sensitive to an unvalidated hyperparameter (Springer, Journal of Asset Management, 2017).

**B. Ω (view uncertainty) is under-specified.** When Ω is derived from He and Litterman's (1999) proportional-to-variance method, it assumes view confidence scales mechanically with asset variance. This breaks down when views come from ML models, where uncertainty is heterogeneous across assets and not proportional to historical variance. The result is that confident ML predictions on high-volatility assets are artificially penalised.

**C. Normal prior is misspecified under fat tails.** The BL formula assumes Gaussian posterior returns. During tail events — precisely when the system needs accurate allocation — the Gaussian assumption collapses and the equilibrium prior (CAPM π = λΣw_mkt) becomes a poor anchor because market-cap weights reflect distress, not equilibrium.

**D. Static rebalancing.** The BL optimisation is run at each pipeline step with a single joint estimate of μ_BL. There is no mechanism for multi-period robustness or turnover penalisation, which inflates transaction costs in regimes of view instability.

---

### 5. CVaR₉₅ Constraint (Iterative ES Enforcement)

**Current role:** Hard ES constraint at 95%, iteratively enforced before final allocation.

**Documented weaknesses:**

**A. Gaussian assumption in iterative enforcement.** If CVaR is computed parametrically under a normal distribution, it underestimates tail losses for non-normal return series. Most equity return distributions exhibit excess kurtosis and negative skew; a parametric Gaussian CVaR will understate true tail exposure by a factor that widens substantially in crisis periods (MetricGate, 2026; Boudt et al., 2008).

**B. Window sensitivity.** Historical-simulation CVaR is sensitive to the lookback window: a single catastrophic observation can dominate the CVaR estimate, and when it exits the rolling window CVaR can jump discontinuously (O'Connell, 2026). This creates a cliff-edge effect where risk limits appear satisfied until a past crisis event rolls off.

**C. CVaR alone is not sufficient for crisis convexity.** CVaR is a coherent risk measure but is linear in the tail — it does not capture the convexity of tail losses. During crash acceleration (a CrisisManager-detected signal), portfolio losses accelerate non-linearly, but a static CVaR limit does not tighten pre-emptively.

---

### 6. CrisisManager Detectors

**Current role:** Four binary threshold detectors; ≥2 fires triggers defensive mode.

**Documented weaknesses:**

**A. All detectors are threshold-based and independent.** The binary voting scheme (≥2 of 4 fires) treats each signal as equally important and ignores the interaction between them. A gradual regime transition can activate only 1 signal persistently and never trigger — but a sudden single shock can fire 3 simultaneously and over-react. There is no probabilistic weighting of signal severity.

**B. Correlation spike detector (>0.70) uses a fixed threshold.** Cross-asset correlation is regime-dependent and non-stationary. A 0.70 threshold may be chronically near-triggered in risk-off environments even without genuine crisis, generating false positives that erode alpha by over-activating defensive mode. Conversely, in a fast cascade like March 2020, correlation can breach 0.70 only after significant losses have already occurred.

**C. No forward-looking regime signal.** All four detectors are backward-looking: they require the signal to have already occurred before activating. There is no mechanism to anticipate regime change, for example via VIX term-structure inversion (VIX > VXV) or credit spread widening, which are leading indicators of crisis.

**D. Drawdown acceleration uses a rolling 5-day window.** This is too short for slow-rolling crises (e.g. 2022 rate-shock bear market) and too slow for V-shaped crashes (e.g. COVID Feb–Mar 2020). A single threshold window cannot be optimal across both stress topologies.

---

### 7. Ledoit-Wolf Covariance Shrinkage

**Current role:** Shrinks sample covariance matrix for use in BL optimisation and vol targeting.

**Documented weaknesses:**

Ledoit & Wolf (2004) linear shrinkage toward a constant-correlation target is an analytical estimator that applies a scalar shrinkage intensity — it shrinks all eigenvalues by the same amount. This is suboptimal: large eigenvalues (concentrated risk factors) are typically overestimated and need more shrinkage; small eigenvalues (diversifying components) are typically underestimated and need less. The linear estimator does not distinguish.

Ledoit & Wolf (2020) developed the first analytical nonlinear shrinkage estimator that applies an oracle-approximating function to each eigenvalue independently, achieving up to 90% of the possible improvement over the sample covariance matrix — substantially better than the linear estimator, especially when the asset universe is large relative to the time-series length (Ledoit & Wolf 2020; Springer, 2022).

The current system's use of `sklearn.covariance.LedoitWolf` (linear shrinkage) is therefore leaving a material improvement in covariance estimation on the table, which propagates into BL portfolio weights and CVaR estimates.

---

### 8. FinBERT Sentiment

**Current role:** News sentiment signal, integrated in alt data layer of the pipeline.

**Documented weaknesses:**

FinBERT (Araci, 2019; ProsusAI) was trained on financial news articles and SEC filings circa 2018. It is an encoder-only BERT-base model fine-tuned on ~10,000 labelled sentences. Its weaknesses in a live trading context are:

**A. Alpha decay.** Sentiment extracted from published news articles is widely available and crowded. The alpha from headline-level sentiment has decayed substantially since 2019 as systematic NLP strategies proliferated. The marginal information content is in the speed of extraction and the specificity of the sentiment to the entity (asset-level, not market-wide).

**B. No uncertainty output.** FinBERT returns class probabilities (positive/negative/neutral) but not calibrated confidence. Poorly-calibrated sentiment scores are combined with other alt-data signals without accounting for their varying reliability across news types (earnings releases vs. general market commentary have very different predictive power).

**C. Outdated model.** FinBERT2 (2025) and other subsequent models demonstrate that encoder-only models of this size underperform more recent architectures on domain-specific discriminative tasks. The model does not capture event-type context (merger, earnings miss, macro surprise) which is the most predictive dimension of financial news sentiment.

**D. No decay model.** Sentiment is used as a contemporaneous signal but financial NLP research shows that the predictive window for headline sentiment is extremely short — often intra-session. A sentiment signal from yesterday's close is stale by the open. Without an exponential decay applied to sentiment scores, stale signals pollute the alt-data integration layer.

---

### 9. Vol Targeting (10% Annualised, Static)

**Current role:** Scales portfolio exposure to maintain 10% annualised vol target.

**Documented weaknesses:**

Research confirms that conventional static volatility targeting (Bongaerts et al., 2020, published in Journal of Financial Markets) "does not consistently improve risk-adjusted performance in international equity markets and can significantly overshoot the volatility target — thereby increasing maximum drawdowns and tail risks." The failure mode occurs during rapid vol regime transitions: the scaling factor uses backward-looking vol estimates that are stale exactly when they matter most, causing the system to either de-lever too late (absorbing crisis losses) or de-lever too aggressively on a vol spike that immediately reverses, creating a buy-high/sell-low whipsaw.

Conditional vol targeting — adjusting exposure only in extreme high-vol or low-vol states — significantly reduces drawdowns and tail risks while improving Sharpe ratios, particularly for momentum-like strategies (Bongaerts et al., 2020).

---

## PART II — PRIORITISED PROPOSALS

The following proposals are ranked by expected impact on risk-adjusted returns and crisis protection, from highest to lowest. All are implementable in Python with standard quant libraries.

| Priority | Proposal | Component Modified | Expected Impact |
|---|---|---|---|
| **P1** | Replace GARCH(1,1) with HAR-RV | Vol estimation | Very High — 35–40% forecast error reduction |
| **P2** | Replace purged k-fold CV on ML regressor | GBT cross-validation | Very High — eliminates structural lookahead bias |
| **P3** | Replace linear Ledoit-Wolf with analytical nonlinear shrinkage | Covariance estimation | High — up to 90% improvement over sample covariance |
| **P4** | Add CrisisComposite score (probabilistic, continuous) to CrisisManager | CrisisManager | High — eliminates binary threshold brittleness |
| **P5** | Replace Gaussian CVaR with Cornish-Fisher modified ES | Risk constraint | High — accurate tail capture under fat tails |
| **P6** | Add volatility skew adjustment to BS put pricing | Put cost surfacing | Medium-High — accurate hedge costs in stress |
| **P7** | Implement conditional vol targeting (extreme-state only de-levering) | Vol target | Medium-High — reduces drawdown overshoot |
| **P8** | Replace τ/Ω fixed calibration with Idzorek confidence-weighted Ω | BL optimisation | Medium — more honest view weighting |
| **P9** | Add conformal prediction intervals to GBT output | ML regressor | Medium — replaces Std[return] with coverage-guaranteed intervals |
| **P10** | Add FinBERT sentiment decay and entity-level scoring | Alt data | Medium — eliminates stale sentiment contamination |

---

## PART III — HIGH-PRIORITY SPEC LANGUAGE

The following are exact updated specifications for P1, P2, P3, P4, and P5, written in master-prompt format consistent with the existing TradingEngineResearch architecture.

---

### SPEC P1 — Replace GARCH(1,1) with HAR-RV Volatility Estimator

**Replaces:** GARCH(1,1) in `estimate_realised_vol(asset, window)` and all downstream consumers: CrisisManager vol-explosion detector, vol targeting scalar, BL covariance diagonal, GBT feature generation.

**Academic basis:** Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility." Confirmed superior by Celik & Ergin (2014), Bergsli et al. (2022), and the S&P 500 benchmark study (ResearchGate, 2024) showing 35–40% RMSE reduction over GARCH(1,1).

**Model definition:**

The HAR-RV model forecasts next-day realised variance as a linear combination of daily, weekly, and monthly aggregated realised variance:

```
RV_{t+1} = α + β_d · RV_t^(d) + β_w · RV_t^(w) + β_m · RV_t^(m) + ε_{t+1}

where:
  RV_t^(d)  = RV_t                              (daily: yesterday's realised variance)
  RV_t^(w)  = (1/5)  · Σ_{i=0}^{4}  RV_{t-i}   (weekly: 5-day mean RV)
  RV_t^(m)  = (1/22) · Σ_{i=0}^{21} RV_{t-i}   (monthly: 22-day mean RV)
```

Realised variance is computed from intraday returns if 5-minute bar data is available; otherwise from daily high-low range via the Parkinson (1980) estimator as a fallback:

```
RV_t^(Parkinson) = (1 / (4 ln 2)) · (ln(H_t / L_t))^2
```

Parameters α, β_d, β_w, β_m are estimated via OLS over a rolling 252-day window, re-estimated weekly.

**Function signature (replaces existing):**

```python
def estimate_realised_vol_har(
    asset: str,
    daily_ohlcv: pd.DataFrame,          # columns: open, high, low, close, volume
    intraday_returns: pd.Series | None,  # optional: 5-min return series
    estimation_window: int = 252,        # OLS training window in days
    forecast_horizon: int = 1           # days ahead (1 for next-day)
) -> dict:
    """
    Returns:
        {
          'rv_forecast': float,      # annualised realised vol forecast (σ, not σ²)
          'rv_d': float,             # daily RV component
          'rv_w': float,             # weekly RV component  
          'rv_m': float,             # monthly RV component
          'har_params': dict,        # {'alpha': float, 'beta_d': float, 'beta_w': float, 'beta_m': float}
          'r_squared': float         # in-sample fit quality
        }
    """
```

**Integration points:**

- `CrisisManager.vol_explosion_detector()`: Replace `garch_vol_5d / garch_vol_60d` ratio with `har_rv_5d_forecast / har_rv_60d_mean`. The 60d mean is the equally-weighted mean of daily RV^(Parkinson) over the prior 60 sessions.
- `vol_targeting_scalar()`: Replace GARCH conditional vol with `har_rv_forecast ** 0.5` (convert variance to vol).
- `ml_feature_engineering()`: Add `har_rv_d`, `har_rv_w`, `har_rv_m` as features, replacing any GARCH-derived vol features.
- `BL_covariance()`: Use HAR-RV forecasts to rescale the diagonal of Σ (see P3 spec below).

**Library requirement:** `statsmodels.regression.linear_model.OLS`, `numpy`, `pandas`. No new dependencies.

---

### SPEC P2 — Purged K-Fold Cross-Validation for the GBT Regressor

**Replaces:** Existing cross-validation scheme for the gradient-boosted tree return regressor.

**Adds to:** `train_gbt_regressor(X, y, feature_names, hyperparams)` training loop.

**Academic basis:** López de Prado (2018), *Advances in Financial Machine Learning*, Chapter 7. "Standard k-fold CV yields overly optimistic performance estimates due to information leakage" in financial time series. Purged CV removes all training observations whose label period overlaps with the test period, and adds an embargo window to prevent look-ahead leakage through lagged features.

**Algorithm definition:**

Let each observation i have a feature window ending at t_i and a label window spanning [t_i, t_i + h] where h is the forecast horizon (e.g. 1 or 5 days). For a test fold covering indices I_test:

```
Purge: Remove from training all observations i such that t_i ∈ [min(t_j) - h, max(t_j) + embargo]
        for any j ∈ I_test

Embargo: e = ceil(embargo_pct · T) observations after each test fold are also removed from training
          where embargo_pct = 0.01 (default: 1% of total sample length)
```

**Function signature (wraps existing training):**

```python
def purged_kfold_cv(
    model,                              # sklearn-compatible estimator (GBT)
    X: pd.DataFrame,                    # features, indexed by date
    y: pd.Series,                       # labels, indexed by date
    t1: pd.Series,                      # Series: index = observation date, value = label end date
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    scoring: str = 'neg_mean_squared_error'
) -> dict:
    """
    Implements López de Prado (2018) purged k-fold CV with embargo.
    
    Returns:
        {
          'cv_scores': List[float],       # per-fold OOS scores
          'mean_score': float,
          'std_score': float,
          'n_training_obs_per_fold': List[int]   # diagnostic: training size after purging
        }
    """
```

**Implementation notes:**

- `t1` is a Series mapping each observation's index date to the end of its forward return window. For a 1-day return label on date t, t1[t] = t + 1 business day.
- Purging is applied before each fold split, not once globally.
- Folds are sequential (no shuffling): fold 1 = earliest 20% of data, fold 5 = most recent 20%.
- The `mlfinlab` library (open-source) provides `PurgedKFold` as a reference implementation compatible with sklearn's cross-validation interface, but can be reimplemented in ~50 lines using `pandas` date arithmetic.
- This replaces any `TimeSeriesSplit`, `KFold`, or `train_test_split` currently used in regressor training.

**Hyperparameter tuning integration:**

Wrap `purged_kfold_cv` inside a grid search over GBT hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`). The best hyperparameter set is selected by mean OOS score across folds. Re-train final model on full dataset using best params.

**Impact on BL:** The correctly cross-validated GBT will produce more conservative (lower magnitude, more shrinkage-appropriate) return predictions, which will manifest as appropriately moderated BL view vectors Q.

---

### SPEC P3 — Analytical Nonlinear Ledoit-Wolf Covariance Shrinkage

**Replaces:** `sklearn.covariance.LedoitWolf` (linear shrinkage) in `estimate_covariance(returns_matrix)`.

**Academic basis:** Ledoit & Wolf (2020), "Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices," *Annals of Statistics*. The nonlinear estimator applies an asset-specific, eigenvalue-dependent shrinkage function rather than a single scalar, achieving close to oracle performance across all concentration ratios p/n, with up to 90% improvement over the sample covariance matrix when p/n is large.

**Model definition:**

Let S = (1/T) X'X be the sample covariance matrix with eigendecomposition S = U Λ U', where Λ = diag(λ_1, ..., λ_p). The nonlinear shrinkage estimator replaces each sample eigenvalue λ_k with a shrunk value φ_k:

```
Σ_NL = U · diag(φ_1, ..., φ_p) · U'

φ_k = λ_k / |1 - (p/T) + (p/T)·λ_k · ĥ(λ_k)|²

where ĥ(λ) is the Hilbert transform of the empirical spectral distribution,
estimated analytically via the Ledoit-Wolf (2020) formula using the sample 
eigenvalues {λ_1, ..., λ_p}.
```

The analytical formula for ĥ(λ) uses Random Matrix Theory and is implemented in the `riskfolio-lib` library (open-source, Python) and in Ledoit & Wolf's own MATLAB reference code, which has been ported to Python.

**Function signature (replaces existing):**

```python
def estimate_covariance_nonlinear(
    returns: pd.DataFrame,              # T × p return matrix (rows = dates, cols = assets)
    frequency: str = 'daily',           # for annualisation scaling
    min_periods: int = 252,             # minimum observations required
    fallback_to_linear: bool = True     # fall back to LW linear if p/T > 0.9
) -> dict:
    """
    Returns:
        {
          'cov_matrix': np.ndarray,      # p × p nonlinear shrinkage covariance matrix
          'corr_matrix': np.ndarray,     # derived correlation matrix
          'shrinkage_intensities': np.ndarray,  # per-eigenvalue shrinkage factor φ_k/λ_k
          'concentration_ratio': float,   # p/T — diagnostic
          'method_used': str             # 'nonlinear_LW' or 'linear_LW_fallback'
        }
    """
```

**Library:** `riskfolio-lib` exposes `rp.RiskFunctions.cov_matrix(returns, method='ledoit_wolf_analytics')` which implements the Ledoit-Wolf (2020) analytical nonlinear estimator. Alternatively, the `nlshrink` PyPI package (Lam, 2016) provides a direct implementation. Either satisfies the requirement.

**Integration points:**

- `BL_optimisation()`: Replace Σ input with `estimate_covariance_nonlinear()['cov_matrix']`.
- `cvar_constraint_check()`: Recalculate portfolio variance σ_p² = w'Σ_NL w using nonlinear Σ.
- `vol_targeting_scalar()`: Use `sqrt(w'Σ_NL w · 252)` for annualised portfolio vol estimate.
- `CrisisManager.correlation_spike_detector()`: Derive rolling pairwise correlations from `corr_matrix` rather than from raw sample correlations.

---

### SPEC P4 — CrisisComposite Probabilistic Score (Replaces Binary Voting)

**Replaces:** The binary 4-detector ≥2-fires voting scheme in `CrisisManager.assess_crisis_level()`.

**Adds:** `CrisisComposite` score ∈ [0, 1], a continuous crisis probability that feeds the defensive-mode transition and allows graduated responses rather than a binary switch.

**Academic basis:** Hidden Markov Model literature (Cube Exchange, 2026; Bergsli et al., 2022; ResearchGate, 2013). Regime-detection research confirms that probabilistic state inference (as in HMM filtered probabilities) substantially outperforms threshold-based binary classifiers for distinguishing market regimes, because it captures the severity and co-movement of signals without requiring simultaneous threshold breaches.

**Architecture:**

Replace the 4-detector binary vote with a **2-layer composite**:

**Layer 1 — Continuous signal scores (replaces hard thresholds):**

Each existing detector is converted from a binary to a continuous score in [0, 1] using a sigmoid transformation:

```
CorrelationScore(t)     = σ((ρ_avg(t) - 0.55) / 0.08)
                          where ρ_avg = mean pairwise cross-asset correlation (10d)
                          sigmoid centred at 0.55, scaled so 0.70 → ~0.84

VolExplosionScore(t)    = σ((RV_ratio(t) - 1.5) / 0.3)
                          where RV_ratio = HAR_RV_5d / HAR_RV_60d_mean
                          (uses HAR-RV from P1, not GARCH)

DrawdownAccelScore(t)   = σ((|DD_5d(t)| - 0.025) / 0.012)
                          where DD_5d = 5-day portfolio drawdown
                          centred at 2.5%, scaled so 5% → ~0.88

BreadthScore(t)         = σ((frac_losing(t) - 0.55) / 0.08)
                          where frac_losing = fraction of positions with negative 5d return

σ(x) = 1 / (1 + exp(-x))   [logistic sigmoid]
```

**Layer 2 — CrisisComposite weighted average:**

```
CrisisComposite(t) = w_cor · CorrelationScore(t) 
                   + w_vol · VolExplosionScore(t) 
                   + w_dd  · DrawdownAccelScore(t) 
                   + w_br  · BreadthScore(t)

Default weights: w_cor = 0.30, w_vol = 0.35, w_dd = 0.25, w_br = 0.10
(vol explosion and correlation get higher weight; breadth is most lagged)

Weights are normalised to sum to 1.0.
```

**Add one forward-looking early-warning signal (new, not in original 4):**

```
VIXTermScore(t) = σ((VIX_spot(t) / VXV_90d(t) - 1.05) / 0.10)
                  where VXV_90d = 90-day implied vol index
                  Inversion of term structure (spot > 90d) is a leading indicator of acute stress

CrisisComposite(t) += w_term · VIXTermScore(t) where w_term = 0.0
                      [disabled if VIX/VXV data unavailable; re-normalise other weights]
```

**Graduated response thresholds (replaces binary switch):**

```
CrisisComposite ∈ [0.0, 0.35):   Normal mode. No modification.
CrisisComposite ∈ [0.35, 0.60):  Elevated mode. Vol target *= 0.80. CVaR limit *= 0.85.
CrisisComposite ∈ [0.60, 0.80):  Defensive mode. Vol target *= 0.60. CVaR limit *= 0.65.
                                   (Equivalent to original CrisisManager triggered state)
CrisisComposite ∈ [0.80, 1.00]:  Crisis mode. Vol target *= 0.50. CVaR limit *= 0.50.
                                   Tail-hedge costs surfaced via put pricing module.
```

**Function signature:**

```python
def compute_crisis_composite(
    correlation_matrix: np.ndarray,     # p × p current rolling correlation
    har_rv_ratio: float,                # HAR_RV_5d / HAR_RV_60d_mean (from P1)
    portfolio_drawdown_5d: float,       # signed 5-day portfolio return (negative = loss)
    fraction_positions_losing: float,   # share of positions with negative 5d return
    vix_spot: float | None = None,      # current VIX level (optional)
    vxv_90d: float | None = None,       # current 90-day IV index (optional)
    weights: dict | None = None         # override default signal weights
) -> dict:
    """
    Returns:
        {
          'composite_score': float,          # CrisisComposite ∈ [0, 1]
          'signal_scores': dict,             # individual signal scores
          'crisis_regime': str,              # 'normal' | 'elevated' | 'defensive' | 'crisis'
          'vol_target_scalar': float,        # multiply base vol target by this
          'cvar_limit_scalar': float,        # multiply base CVaR limit by this
          'tail_hedge_active': bool          # True if crisis mode
        }
    """
```

**Backward-compatibility note:** The existing `CrisisManager` class is modified; the four detector methods are preserved as internal sub-functions feeding `compute_crisis_composite()`. The `≥2 fires` boolean is still computed and logged as a diagnostic field in the returned dict but no longer drives the defensive-mode transition.

---

### SPEC P5 — Cornish-Fisher Modified ES to Replace Gaussian CVaR

**Replaces:** Gaussian parametric CVaR in `compute_portfolio_cvar(weights, returns_history, confidence=0.95)`.

**Academic basis:** Boudt, Peterson & Croux (2008), "Estimation and Decomposition of Downside Risk for Portfolios with Non-Normal Returns." The Cornish-Fisher expansion adjusts the Gaussian quantile using the empirical skewness (γ₁) and excess kurtosis (γ₂) of the return distribution, producing a modified VaR and CVaR that collapse to standard Gaussian estimates when the distribution is normal but substantially improve accuracy for fat-tailed, negatively-skewed financial returns.

**Model definition:**

Standard Gaussian CVaR at confidence level α:

```
CVaR_Gaussian = μ_p - σ_p · φ(z_α) / (1 - α)

where z_α = N⁻¹(α), φ = standard normal PDF
```

Modified Cornish-Fisher CVaR:

```
Step 1: Compute portfolio return moments from rolling window (default: 252 days)
  μ_p  = mean portfolio return
  σ_p  = standard deviation of portfolio return
  γ₁   = skewness (Fisher)
  γ₂   = excess kurtosis

Step 2: Cornish-Fisher adjusted quantile
  z_cf = z_α 
       + (1/6)(z_α² - 1)γ₁ 
       + (1/24)(z_α³ - 3z_α)γ₂ 
       - (1/36)(2z_α³ - 5z_α)γ₁²

Step 3: Modified VaR
  VaR_CF = μ_p + σ_p · z_cf     [note: z_cf is typically negative for loss quantile]

Step 4: Modified CVaR (closed-form via second-order Edgeworth expansion)
  CVaR_CF = -μ_p + σ_p · [φ(z_cf)/(1-α)] 
           · [1 + (1/6)·γ₁·(2z_cf² - 1 - z_cf/z_α)·z_α 
               + (1/24)·γ₂·(3z_cf·z_α - z_cf³/z_α)·z_α]
```

For production, use the `PerformanceAnalytics`-equivalent Python implementation: `scipy.stats` provides `norm.pdf`, `norm.ppf`; moments are computed via `scipy.stats.skew`, `scipy.stats.kurtosis`. The formula above is implementable in ~20 lines of numpy/scipy.

**Function signature (replaces existing):**

```python
def compute_portfolio_cvar_cf(
    weights: np.ndarray,                # p-vector of portfolio weights
    returns_history: pd.DataFrame,      # T × p historical return matrix
    confidence: float = 0.95,           # ES confidence level (0.95 or 0.975)
    window: int = 252,                  # rolling estimation window (days)
    method: str = 'cornish_fisher'      # 'cornish_fisher' | 'gaussian' | 'historical'
) -> dict:
    """
    Returns:
        {
          'cvar': float,                # modified CVaR (positive = loss magnitude)
          'var': float,                 # modified VaR
          'portfolio_skew': float,      # γ₁ diagnostic
          'portfolio_kurtosis': float,  # γ₂ diagnostic
          'cf_quantile_adjustment': float,  # z_cf - z_alpha (non-normality adjustment)
          'method': str
        }
    """
```

**Integration points:**

- `risk_overlay()`: Replace `compute_portfolio_cvar()` call with `compute_portfolio_cvar_cf()` at the risk overlay step (pipeline step 13).
- `CrisisManager`: Defensive mode check now uses CF-adjusted CVaR to evaluate whether tail-risk limits have been breached, not Gaussian CVaR.
- **Iterative enforcement loop:** The existing iterative CVaR enforcement loop iterates on `compute_portfolio_cvar_cf()` using the Cornish-Fisher method. Because CF-adjusted CVaR will be higher than Gaussian CVaR in negatively-skewed regimes, the enforced position sizes will be more conservative — this is the desired behaviour.
- **Regime-aware fallback:** If `abs(γ₁) > 3` or `γ₂ > 20` (extreme moment estimates from thin data), fall back to `method='historical'` (empirical quantile average of the worst `(1-α)·T` observations). Log a warning when this occurs.

---

## PART IV — IMPLEMENTATION PRIORITY SEQUENCING

The following order is recommended for staged rollout, each addition fully testable in isolation:

**Sprint 1 (Highest impact, lowest risk of breaking changes):**
- P1: HAR-RV replaces GARCH(1,1) — isolated to vol estimation module
- P2: Purged CV on GBT — isolated to training pipeline, does not change inference API

**Sprint 2 (Risk framework, requires P1 to be live):**
- P3: Nonlinear LW covariance — replaces one function call, output shape unchanged
- P5: Cornish-Fisher CVaR — replaces one function call, output shape unchanged

**Sprint 3 (CrisisManager overhaul, requires P1 and P5):**
- P4: CrisisComposite score — backward-compatible refactor of CrisisManager

**Sprint 4 (Enhancements):**
- P6: Volatility skew adjustment to BS put pricing
- P7: Conditional vol targeting (state-dependent de-levering)
- P8: Idzorek confidence-weighted Ω in BL
- P9: Conformal prediction intervals on GBT output
- P10: FinBERT sentiment decay and entity-level scoring

---

## REFERENCES

- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv:1908.10063*.
- Bongaerts, D., Kräussl, R., & Lippens, W. (2020). Conditional Volatility Targeting. *Journal of Financial Markets*.
- Boudt, K., Peterson, B., & Croux, C. (2008). Estimation and Decomposition of Downside Risk for Portfolios with Non-Normal Returns. *Journal of Risk*.
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- He, G., & Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios. *SSRN 334304*.
- Ledoit, O., & Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*.
- Ledoit, O., & Wolf, M. (2017). Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks. *Review of Financial Studies*, 30(12), 4349–4388.
- Ledoit, O., & Wolf, M. (2020). Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices. *Annals of Statistics*, 48(5), 3043–3065.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7 (Purged CV), 5 (Fractional Differentiation).
- López de Prado, M. (2019). The 10 Reasons Most Machine Learning Funds Fail. *GARP Risk Professional*.
- Meucci, A. (2010). The Black-Litterman Approach: Original Model and Extensions. *SSRN*.
- Nelson, D.B. (1992). Filtering and Forecasting with Misspecified ARCH Models I. *Journal of Econometrics*.
- Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return. *Journal of Business*, 53(1), 61–65.
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. *NeurIPS*.
- Springer (2017). The Black-Litterman Model: Active Risk Targeting and the Parameter Tau. *Journal of Asset Management*.
