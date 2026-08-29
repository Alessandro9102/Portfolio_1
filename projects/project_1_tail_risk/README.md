# Tail Risk & Regime-Switching Volatility Model

Market risk model for equity portfolios that estimates Value at Risk (VaR) and Expected Shortfall (ES) by combining regime detection, conditional volatility modeling, and Extreme Value Theory. The model is validated with formal statistical backtests and deployed as an interactive Streamlit dashboard.

## Motivation

Standard risk models assume that market volatility is constant and that returns are normally distributed. Neither assumption holds in practice: volatility clusters into calm and turbulent periods, and large losses occur far more often than a normal distribution predicts. This project addresses both issues directly:

- **Volatility clustering** is captured with a regime-switching framework (Hidden Markov Model) and regime-conditional GARCH.
- **Fat tails** are captured with Extreme Value Theory, which models the extreme left tail of the return distribution separately from its center.

The resulting VaR and ES estimates are then backtested against realized returns using the same statistical tests used in banking model validation (Basel framework).

## Methodology

The pipeline has five stages, each implemented as an independent module in `src/`.

### 1. Data (`data_loader.py`)
Daily adjusted close prices for the S&P 500 ETF (SPY) and the CBOE Volatility Index (VIX) are downloaded via `yfinance`, starting January 2000. Log returns are computed as `r_t = ln(P_t / P_{t-1})` because they are time-additive and better behaved for GARCH and HMM estimation than simple returns.

### 2. Regime detection (`features.py`, `regime_model.py`)
A 2-state Gaussian Hidden Markov Model classifies each trading day as **calm** or **turbulent**, based on four engineered, standardized features:

- 21-day annualized realized volatility
- Absolute daily return (a fast-reacting shock proxy)
- 63-day rolling skewness (captures crash-prone vs. symmetric volatility)
- VIX level (market-implied volatility)

The model is fit with the Baum-Welch (EM) algorithm and decoded with the Viterbi algorithm to produce the most likely regime path, along with posterior state probabilities at each date. The calm/turbulent labeling is made deterministic by assigning the state with the lower volatility-feature mean to "calm," regardless of how the HMM initializes internally.

### 3. Conditional volatility (`volatility_model.py`)
A GARCH(1,1) model with Student's t innovations is fit both globally and separately within each detected regime:

```
r_t = ε_t · σ_t
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Fitting GARCH per regime allows each state to have its own long-run variance (ω), shock sensitivity (α), and volatility persistence (β), producing sharper conditional volatility estimates than a single pooled model. A Student's t distribution is used instead of a Normal distribution because financial returns exhibit fat tails that a Gaussian model systematically underestimates.

### 4. Tail modeling with Extreme Value Theory (`tail_model.py`)
Rather than relying on the bulk of the distribution to describe rare, extreme losses, the model applies the Peaks-Over-Threshold (POT) method:

1. A loss threshold `u` is selected at a chosen quantile (default: 95th percentile of losses).
2. Exceedances over the threshold are fit to a Generalized Pareto Distribution (GPD) by maximum likelihood.
3. VaR and ES are computed in closed form from the fitted GPD parameters (McNeil & Frey, 2000):

```
VaR_p = u + (σ/ξ) · [((1-p)/F_u)^(-ξ) - 1]
ES_p  = VaR_p / (1-ξ) + (σ - ξu) / (1-ξ)
```

This approach is theoretically grounded in the Pickands–Balkema–de Haan theorem, which shows that exceedances over a sufficiently high threshold converge to a GPD regardless of the underlying return distribution — making it a principled way to model the tail without assuming normality.

### 5. Risk metrics and backtesting (`risk_metrics.py`, `backtesting.py`)
VaR and ES are computed under four approaches for direct comparison: Historical Simulation, unconditional Normal, GARCH-conditional Normal, and EVT (GPD). Model quality is then assessed with:

- **Kupiec Proportion of Failures test** — a likelihood-ratio test of whether the observed VaR violation rate matches the theoretical rate, distributed as χ²(1) under the null hypothesis.
- **Basel II/III traffic-light classification** — violations over a 250-day window are classified as green (acceptable), amber (requires review), or red (model rejected), following the standard used by banking regulators to validate internal VaR models.

## Project Structure

```
project_1_tail_risk/
├── src/
│   ├── data_loader.py        Data download, log returns, cleaning
│   ├── features.py           HMM feature engineering
│   ├── regime_model.py       Gaussian HMM regime detection
│   ├── volatility_model.py   Global and regime-conditional GARCH(1,1)
│   ├── tail_model.py         EVT / GPD tail fitting, VaR & ES
│   ├── risk_metrics.py       Multi-method VaR/ES comparison table
│   └── backtesting.py        Kupiec test, Basel traffic-light zones
├── app/streamlit_app.py      Interactive dashboard
├── data/                     Cached price data and fitted HMM
└── config.py                 All parameters (tickers, thresholds, VaR levels, etc.)
```

## Running the Project

```bash
cd project_1_tail_risk
pip install -r requirements.txt

# Run any module standalone for a sanity check, e.g.:
python src/regime_model.py

# Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

Configuration (tickers, date range, HMM states, EVT threshold, VaR confidence levels, backtesting window) is centralized in `config.py`; no parameters are hard-coded inside the modeling functions.

## Dashboard

The Streamlit app displays the price series with regime overlays, GARCH conditional volatility, the fitted GPD tail with a QQ-plot diagnostic, the VaR/ES comparison table across methods, and the backtesting results with Kupiec p-values and Basel traffic-light status.

## Interview Talking Points

1. **Why a Hidden Markov Model instead of a fixed-window volatility threshold?** An HMM learns the number and character of regimes and their transition dynamics directly from the data, and provides a probabilistic (not just binary) regime assessment via posterior probabilities.

2. **Why fit GARCH per regime instead of one global model?** A single GARCH model averages over calm and turbulent dynamics, understating volatility persistence in turbulent periods and overstating it in calm ones. Regime-conditional GARCH lets each state have its own parameters.

3. **Why EVT instead of just using a fatter-tailed parametric distribution?** EVT is asymptotically justified for tail behavior regardless of the distribution generating the rest of the data (Pickands–Balkema–de Haan theorem), rather than requiring the analyst to guess the correct global distribution.

4. **Why the Kupiec test rather than just counting violations?** Kupiec formally tests whether the number of violations is statistically consistent with the model's stated confidence level, avoiding subjective judgment about what "close enough" means.

5. **Limitation to be transparent about:** the Kupiec test checks the correct *number* of violations but not their *clustering* in time; a Christoffersen independence test would be the natural next step to check whether violations bunch together during crises.

## Data Sources

- S&P 500 ETF (SPY) and CBOE VIX index, via Yahoo Finance (`yfinance`), 2000–present.
