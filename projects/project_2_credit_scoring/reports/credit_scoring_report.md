# Credit Scoring Model -- Evaluation Report

## Core Metrics (Validation Set)

| Metric | Value |
|--------|-------|
| AUC | 0.8695 |
| Gini | 0.7391 |
| KS | 0.5818 |

## Threshold Analysis

| Threshold | Flagged | Flagged% | Precision | Recall | F1 |
|-----------|---------|----------|-----------|--------|----|
| 0.1 | 5597.0 | 18.7% | 0.2582 | 0.7207 | 0.3802 |
| 0.15 | 4025.0 | 13.4% | 0.3173 | 0.6369 | 0.4235 |
| 0.2 | 3116.0 | 10.4% | 0.3652 | 0.5676 | 0.4444 |
| 0.25 | 2559.0 | 8.5% | 0.4033 | 0.5147 | 0.4522 |
| 0.3 | 2127.0 | 7.1% | 0.4311 | 0.4574 | 0.4439 |
| 0.35 | 1666.0 | 5.6% | 0.473 | 0.393 | 0.4293 |
| 0.4 | 1175.0 | 3.9% | 0.5209 | 0.3052 | 0.3849 |
| 0.45 | 667.0 | 2.2% | 0.5802 | 0.193 | 0.2897 |
| 0.5 | 125.0 | 0.4% | 0.672 | 0.0419 | 0.0789 |
| 0.55 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.6 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.65 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.7 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.75 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.8 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.85 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |
| 0.9 | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |

## Figures

- `reports/figures/roc_curve.png`  -- ROC curve
- `reports/figures/ks_plot.png`    -- KS separation plot
- `reports/figures/shap_summary.png`   -- Global SHAP importance
- `reports/figures/shap_waterfall.png` -- Single borrower explanation

## Interpretation

- **AUC 0.8695**: probability the model ranks a defaulter above a non-defaulter. Industry benchmark for retail credit: 0.70-0.80.
- **Gini 0.7391**: normalised AUC. Competitive scorecards typically target Gini > 0.40.
- **KS 0.5818**: maximum separation between score distributions. KS > 0.30 is considered good in credit scoring.