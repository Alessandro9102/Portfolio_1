# Credit Scoring — Give Me Some Credit

**Live demo:** https://portfolio1-nxuhzaczdytwlnxwbayhxw.streamlit.app/

End-to-end credit risk classification pipeline built on the Kaggle "Give Me Some Credit" dataset, predicting the probability that a borrower will experience serious delinquency within two years. The project covers data cleaning, feature engineering, model comparison, statistical validation, and explainability, and is deployed as an interactive Streamlit scoring tool.

## Overview

| | |
|---|---|
| Dataset | [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 US borrowers |
| Problem | Binary classification: predict 2-year serious delinquency (`SeriousDlqin2yrs`) |
| Class balance | ~6.7% positive rate (approx. 14:1 imbalance) |
| Models compared | Logistic Regression, XGBoost, Stacking Ensemble |
| Validation AUC | Logistic Regression 0.862 · XGBoost 0.870 · Stacking 0.870 |
| Gini / KS (final model) | Gini 0.739 · KS 0.582 |
| Explainability | SHAP global summary and per-borrower waterfall plots |
| Application | Streamlit scorer with live SHAP explanation |

## Methodology

### Data preprocessing
Raw data goes through outlier capping before imputation, since a handful of extreme or sentinel values (e.g. utilization ratios above 100%, debt ratios above 10, delinquency counts of 96–98 used as Kaggle's "no answer" codes) would otherwise distort both the imputed values and the fitted models. Missing `MonthlyIncome` and `NumberOfDependents` are imputed with the median, which is more robust than the mean under the strong right-skew present in both variables. Borrowers under 18 are dropped as invalid records. All thresholds are configured in `config/config.yaml`, not hard-coded, so the cleaning logic is auditable and reproducible.

### Feature engineering
Beyond the eleven raw Kaggle fields, the pipeline derives:
- Aggregate delinquency features (total delinquencies across all buckets, worst delinquency band, a flag for ever being 90+ days late)
- Debt-servicing features (monthly debt payment, a leakage-safe debt-to-income ratio, utilization per credit line)
- Interaction terms between utilization, delinquency history, and debt ratio, since risk teams commonly observe that these variables compound rather than act independently
- Log transforms (`log1p`) on the most skewed variables (income, debt ratio, utilization) to stabilize their scale for the linear model

The `StandardScaler` is fit strictly on the training split and applied to validation, preventing information leakage from the validation set into feature scaling.

### Modeling
Three models are trained and compared:
- **Logistic Regression** — interpretable linear baseline, `class_weight="balanced"` to correct for the imbalance without altering the sample distribution.
- **XGBoost** — captures non-linear interactions (e.g. utilization × debt ratio) that the linear model cannot, with `scale_pos_weight` set to the approximate negative-to-positive ratio.
- **Stacking Ensemble** — combines both base learners with a logistic-regression meta-learner, letting the ensemble weight each model's strengths rather than relying on either one alone.

Class imbalance is handled through class weighting rather than SMOTE or oversampling, because credit data is naturally time-ordered and synthetic oversampling risks leaking information about future observations into the training set.

### Evaluation
The final model is assessed on the held-out validation set with three metrics standard in credit risk practice:
- **AUC** — probability that the model ranks a random defaulter above a random non-defaulter.
- **Gini coefficient** (`2 × AUC − 1`) — the industry-preferred metric in credit scoring because it is centered at 0 for a random model and 1 for a perfect one, making cross-model comparison more intuitive than AUC alone.
- **Kolmogorov-Smirnov (KS) statistic** — the maximum separation between the cumulative distributions of scores for defaulters and non-defaulters; a common regulatory and scorecard-acceptance benchmark.

A threshold analysis table (precision, recall, F1 across cutoffs from 0.10 to 0.90) is also generated, since the deployment threshold in a real credit process is a business decision — balancing the cost of missed defaults against the cost of rejecting creditworthy applicants — rather than a fixed modeling output.

### Explainability
SHAP values are computed for both a global feature-importance summary and individual borrower-level waterfall explanations, supporting the kind of model transparency required under model risk management frameworks such as the Federal Reserve's SR 11-7 guidance, which requires credit models to be explainable and independently validated.

## Results

| Model | ROC-AUC | Gini |
|-------|---------|------|
| Logistic Regression | 0.862 | 0.723 |
| XGBoost | 0.870 | 0.740 |
| **Stacking Ensemble** | 0.870 | 0.739 |

Final evaluation on the held-out validation set (stacking ensemble): **AUC 0.870, Gini 0.739, KS 0.582**. For context, retail credit scorecards are generally considered acceptable above AUC 0.70–0.80 and KS above 0.30; this model is comfortably above both benchmarks.

## Project Structure

```
project_2_credit_scoring/
├── data/
│   ├── raw/                     cs-training.csv (from Kaggle)
│   └── processed/                cleaned and feature-engineered parquet files
├── src/
│   ├── get_data.py               Kaggle data download
│   ├── data_preprocessing.py     cleaning, outlier capping, imputation, split
│   ├── feature_engineering.py    derived features, interactions, scaling
│   ├── train_model.py            Logistic Regression -> XGBoost -> Stacking
│   ├── evaluate_model.py         AUC/Gini/KS, threshold table, SHAP plots
│   └── utils.py                  config loading, logging, path helpers
├── models/                       persisted model and scaler artifacts (.pkl)
├── reports/                      evaluation report and generated figures
├── app/streamlit_app.py          interactive scoring demo
└── config/config.yaml            single source of truth for all parameters
```

## Running the Project

```bash
cd project_2_credit_scoring
pip install -r requirements.txt

# 1. Get the data (or place cs-training.csv manually in data/raw/)
python src/get_data.py

# 2. Run the pipeline in order
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py

# 3. Launch the interactive scorer
streamlit run app/streamlit_app.py
```

A hosted version of this scorer is available at: https://portfolio1-nxuhzaczdytwlnxwbayhxw.streamlit.app/

## Interview Talking Points

1. **Why Gini over raw AUC in credit risk reporting?** Gini rescales AUC to be centered at 0 for a random model, which is the convention risk teams use to communicate model lift intuitively.

2. **Why not use SMOTE for the class imbalance?** Credit data is inherently time-ordered; synthetic oversampling can leak information across time and inflate validation performance in a way that will not generalize in production. Class weighting corrects the objective function without altering the data.

3. **Why winsorize (cap) instead of dropping or log-transforming outliers directly?** Capping preserves the full sample size and keeps the original scale interpretable, while removing the distortion that a handful of erroneous or sentinel values would otherwise cause in imputation and model fitting.

4. **What does the regulatory angle look like?** Under frameworks like SR 11-7, credit models must be explainable and subject to ongoing validation. This is addressed here through SHAP explanations at both global and individual-prediction level, plus a fully interpretable Logistic Regression benchmark alongside the more complex ensemble.

5. **How would the deployment threshold be chosen in practice?** Not by the model. The threshold is a business decision that trades off the cost of a missed default (credit loss) against the cost of a false positive (rejecting a good customer and losing revenue); the threshold analysis table exists to support that conversation, not to resolve it.

## Data Source

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle competition dataset, 150,000 anonymized US consumer credit records.
