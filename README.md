# Data Science & Quantitative Finance Portfolio

Author: Jair Alessandro Cupi Olivares

This repository contains applied data science and statistics projects with a focus on quantitative finance and credit risk. Each project follows an end-to-end structure: data acquisition, cleaning and feature engineering, modeling, statistical validation, and an interactive Streamlit application for demonstration purposes.

The goal of this portfolio is to demonstrate not only predictive modeling ability, but also the statistical reasoning, model validation discipline, and domain framing expected in quantitative risk and data science roles.

## Repository Structure

```
Portfolio_1/
├── projects/
│   ├── project_1_tail_risk/        Market risk modeling: regime detection, GARCH volatility, EVT tail risk, VaR backtesting
│   ├── project_2_credit_scoring/   Credit risk classification: logistic regression, XGBoost, stacking ensemble, SHAP explainability
│   └── project_3_xxx/              Reserved for an upcoming project (in progress)
├── shared/                         Shared utilities and plotting helpers used across projects
├── requirements.txt                Consolidated dependencies for the full portfolio
└── README.md
```

## Projects

### 1. Tail Risk & Regime-Switching Volatility Model
Statistical model of equity tail risk that combines a Hidden Markov Model for market regime detection, regime-conditional GARCH(1,1) volatility, and Extreme Value Theory (Generalized Pareto Distribution) for the estimation of Value at Risk and Expected Shortfall. VaR estimates are validated with the Kupiec Proportion of Failures test and classified under the Basel traffic-light framework.

See: [`projects/project_1_tail_risk/README.md`](projects/project_1_tail_risk/README.md)

### 2. Credit Scoring — Give Me Some Credit
End-to-end credit risk pipeline on the Kaggle "Give Me Some Credit" dataset (150,000 borrowers). Includes outlier treatment, class-imbalance handling, feature engineering, and a comparison of Logistic Regression, XGBoost, and a stacking ensemble, evaluated with AUC, Gini, and Kolmogorov-Smirnov statistics, plus SHAP-based explainability.

See: [`projects/project_2_credit_scoring/README.md`](projects/project_2_credit_scoring/README.md)

### 3. Project 3 (in progress)
Placeholder for an upcoming project. Structure is scaffolded but not yet implemented.

## Technical Stack

- **Languages:** Python
- **Data & numerics:** pandas, NumPy, PyArrow
- **Statistics & econometrics:** SciPy, statsmodels, arch (GARCH), hmmlearn (Hidden Markov Models)
- **Machine learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Applications:** Streamlit
- **Data sources:** Yahoo Finance (yfinance), Kaggle

## Running the Projects

Each project is self-contained with its own `requirements.txt`, `config` file, and Streamlit app. General pattern:

```bash
cd projects/<project_name>
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Refer to each project's own README for data setup and pipeline execution order.

## Contact

For questions about methodology or collaboration, feel free to reach out via GitHub.
