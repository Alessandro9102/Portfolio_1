# Credit Scoring — Give Me Some Credit

> **End-to-end credit risk pipeline:** EDA → Feature Engineering → LR → XGBoost → Stacking Ensemble → Streamlit App

---

## 📋 Overview

| | |
|---|---|
| **Dataset** | [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) · 150,000 US borrowers |
| **Problem** | Binary classification: predict 2-year default (`SeriousDlqin2yrs`) |
| **Models** | Logistic Regression · XGBoost · Stacking Ensemble |
| **Best AUC** | ~0.87 · Gini ~0.74 · KS ~0.46 |
| **Explainability** | SHAP beeswarm + waterfall (individual predictions) |
| **App** | Streamlit interactive scorer with SHAP explanations |

---

## 🗂 Project Structure

```
project_2_credit_scoring/
├── data/
│   ├── raw/               ← cs-training.csv goes here (from Kaggle)
│   └── processed/         ← cleaned CSVs (auto-generated)
├── notebooks/
│   ├── 01_eda.ipynb               ← exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb          ← train all three models
│   └── 04_evaluation.ipynb        ← metrics, SHAP, threshold analysis
├── src/
│   ├── utils.py                   ← logging, I/O, metrics helpers
│   ├── data_preprocessing.py      ← clean, impute, winsorise, split
│   ├── feature_engineering.py     ← derived features + scaling
│   ├── train_model.py             ← LR → XGB → Stacking
│   └── evaluate_model.py          ← all plots and metrics
├── models/                        ← persisted .pkl artefacts
├── reports/figures/               ← auto-saved plots
├── app/streamlit_app.py           ← interactive demo
└── config/config.yaml             ← all hyperparameters & paths
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the data
Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in `data/raw/`.

### 3. Run the notebooks (in order)
```bash
cd project_2_credit_scoring
jupyter notebook
```
Open and run: `01_eda` → `02_feature_engineering` → `03_modeling` → `04_evaluation`

### 4. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Modelling Decisions

### Why stacking?
- Logistic Regression captures strong linear signals (utilisation, delinquency count)
- XGBoost captures non-linear interactions (utilisation × debt ratio)
- Stacking lets a meta-learner decide how to weight each — outperforms both individually

### Handling class imbalance (~14:1)
- LR: `class_weight='balanced'`
- XGBoost: `scale_pos_weight=14`
- **No SMOTE/oversampling** — avoids data leakage in time-ordered credit data

### Winsorisation over dropping
- Preserves 100% of data volume
- Caps extreme values without removing signal entirely

---

## 📊 Key Results

| Model | ROC-AUC | Gini | KS |
|-------|---------|------|----|
| Logistic Regression | ~0.83 | ~0.66 | ~0.40 |
| XGBoost | ~0.86 | ~0.72 | ~0.44 |
| **Stacking Ensemble** | **~0.87** | **~0.74** | **~0.46** |

Industry benchmark: KS > 0.40 = good model · Gini > 0.70 = good model ✅

---

## 💼 Interview Talk Points

1. **Why Gini over AUC?** Risk teams use Gini (= 2×AUC−1) as it's centered at 0 for random and 1 for perfect — easier to compare across models intuitively.

2. **Why not SMOTE?** Credit data is often time-ordered; oversampling future periods into training creates leakage. Class weights are safer.

3. **Why winsorise rather than log-transform?** Winsorisation is simpler, preserves original scale for interpretability, and avoids issues with zero-valued features under log.

4. **Regulatory angle:** SR 11-7 guidance (Fed model risk management) requires models to be explainable and validated annually. SHAP waterfall + LR fallback address both.

5. **Threshold is a business call:** The model outputs probability; the cutoff is chosen by balancing false negatives (missed defaults = credit losses) against false positives (rejected good customers = lost revenue).
