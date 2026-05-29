"""
evaluate_model.py
-----------------
Step 5: Evaluate the stacking ensemble with full credit-risk metrics.

Metrics produced:
  - ROC-AUC
  - Gini coefficient  (= 2 * AUC - 1,  industry standard)
  - KS statistic      (max separation between default / non-default CDFs)
  - Threshold analysis table  (precision, recall, F1 at each cut-off)
  - SHAP summary plot  (global feature importance)
  - SHAP waterfall     (single borrower explanation)
  - ROC curve plot

Usage (from project_2_credit_scoring/):
    python src/evaluate_model.py

Outputs:
    reports/figures/roc_curve.png
    reports/figures/shap_summary.png
    reports/figures/shap_waterfall.png
    reports/figures/ks_plot.png
    reports/credit_scoring_report.md
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend -- safe on Windows
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
)

log = get_logger(__name__)


# =============================================================================
# Load
# =============================================================================

def load_val(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_PROCESSED / "val_fe.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\nRun feature_engineering.py first."
        )
    df = pd.read_parquet(path)
    exclude = {TARGET, "row_id"}
    X = df[[c for c in df.columns if c not in exclude]]
    y = df[TARGET]
    log.info("Loaded val set: %s  (default rate %.2f%%)", df.shape, y.mean() * 100)
    return X, y


def load_model(name: str = "model"):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun train_model.py first."
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded model: %s", path)
    return model


# =============================================================================
# Core metrics
# =============================================================================

def compute_core_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> dict:
    auc  = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    # KS statistic: max distance between default and non-default score CDFs
    df_ks = pd.DataFrame({"prob": y_prob, "label": y_true})
    df_ks = df_ks.sort_values("prob")
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    df_ks["cum_pos"] = (df_ks["label"] == 1).cumsum() / n_pos
    df_ks["cum_neg"] = (df_ks["label"] == 0).cumsum() / n_neg
    ks = (df_ks["cum_pos"] - df_ks["cum_neg"]).abs().max()

    return {"AUC": round(auc, 4), "Gini": round(gini, 4), "KS": round(ks, 4)}


# =============================================================================
# Threshold analysis
# =============================================================================

def threshold_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    lo, hi = cfg["evaluation"]["threshold_range"]
    steps  = cfg["evaluation"]["threshold_steps"]
    thresholds = np.linspace(lo, hi, steps)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        n_flagged = y_pred.sum()
        rows.append({
            "threshold":  round(t, 2),
            "flagged":    int(n_flagged),
            "flagged_pct": round(n_flagged / len(y_true) * 100, 1),
            "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        })

    df = pd.DataFrame(rows)
    log.info("Threshold analysis table (%d rows)", len(df))
    return df


# =============================================================================
# Plots
# =============================================================================

def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    auc: float,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"Stacking Ensemble (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve -- Credit Default Model")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curve -> %s", path)


def plot_ks(y_true: pd.Series, y_prob: np.ndarray) -> None:
    df_ks = pd.DataFrame({"prob": y_prob, "label": y_true})
    df_ks = df_ks.sort_values("prob").reset_index(drop=True)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    cum_pos = (df_ks["label"] == 1).cumsum() / n_pos
    cum_neg = (df_ks["label"] == 0).cumsum() / n_neg

    ks_idx = (cum_pos - cum_neg).abs().idxmax()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df_ks["prob"], cum_pos, color="#16A34A", lw=2, label="Defaults (positive)")
    ax.plot(df_ks["prob"], cum_neg, color="#DC2626", lw=2, label="Non-defaults (negative)")
    ax.axvline(df_ks["prob"].iloc[ks_idx], color="navy", linestyle="--", lw=1.5,
               label=f"KS = {(cum_pos - cum_neg).abs().max():.4f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Cumulative Distribution")
    ax.set_title("KS Plot -- Default vs Non-Default Score Distributions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "ks_plot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved KS plot -> %s", path)


def plot_shap_summary(
    model,
    X_val: pd.DataFrame,
    n_samples: int = 2000,
) -> None:
    try:
        import shap
    except ImportError:
        log.warning("shap not installed -- skipping SHAP plots. pip install shap")
        return

    log.info("Computing SHAP values (n=%d samples) ...", n_samples)

    # Sample for speed
    X_sample = X_val.sample(n=min(n_samples, len(X_val)), random_state=42)

    # Extract XGBoost from stacking ensemble for SHAP
    xgb_step = None
    if hasattr(model, "estimators_"):
        for name, est in model.named_estimators_.items():
            if "xgb" in name.lower():
                xgb_step = est
                break

    if xgb_step is None:
        log.warning("Could not extract XGBoost from ensemble -- using full model.")
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)
        sv = shap_values[..., 1]
    else:
        explainer = shap.TreeExplainer(xgb_step)
        shap_values = explainer.shap_values(X_sample)
        sv = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance (XGBoost base learner)")
    plt.tight_layout()
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary -> %s", path)

    # Waterfall for single high-risk borrower
    try:
        probs = xgb_step.predict_proba(X_sample)[:, 1]
        high_risk_idx = int(np.argmax(probs))

        shap_vals_single = explainer(X_sample.iloc[[high_risk_idx]])

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_vals_single[0], show=False, max_display=12)
        plt.title("SHAP Waterfall -- Highest-Risk Borrower in Validation Set")
        plt.tight_layout()
        path2 = FIGURES_DIR / "shap_waterfall.png"
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved SHAP waterfall -> %s", path2)
    except Exception as exc:
        log.warning("Waterfall plot failed: %s", exc)


# =============================================================================
# Report
# =============================================================================

def save_report(
    metrics: dict,
    threshold_df: pd.DataFrame,
    cfg: dict,
) -> None:
    path = REPORTS_DIR / "credit_scoring_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Credit Scoring Model -- Evaluation Report",
        "",
        "## Core Metrics (Validation Set)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Threshold Analysis",
        "",
        "| Threshold | Flagged | Flagged% | Precision | Recall | F1 |",
        "|-----------|---------|----------|-----------|--------|----|",
    ]
    for _, row in threshold_df.iterrows():
        lines.append(
            f"| {row['threshold']} | {row['flagged']} | {row['flagged_pct']}% "
            f"| {row['precision']} | {row['recall']} | {row['f1']} |"
        )

    lines += [
        "",
        "## Figures",
        "",
        "- `reports/figures/roc_curve.png`  -- ROC curve",
        "- `reports/figures/ks_plot.png`    -- KS separation plot",
        "- `reports/figures/shap_summary.png`   -- Global SHAP importance",
        "- `reports/figures/shap_waterfall.png` -- Single borrower explanation",
        "",
        "## Interpretation",
        "",
        f"- **AUC {metrics['AUC']}**: probability the model ranks a defaulter "
        f"above a non-defaulter. Industry benchmark for retail credit: 0.70-0.80.",
        f"- **Gini {metrics['Gini']}**: normalised AUC. Competitive scorecards "
        f"typically target Gini > 0.40.",
        f"- **KS {metrics['KS']}**: maximum separation between score distributions. "
        f"KS > 0.30 is considered good in credit scoring.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report saved -> %s", path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    log.info("=== Step 5: Model Evaluation ===")
    ensure_dirs()

    cfg = load_config()
    set_seed(cfg["seed"])

    # Load
    X_val, y_val = load_val(cfg)
    model = load_model("model")

    # Predict probabilities
    log.info("Generating predictions ...")
    y_prob = model.predict_proba(X_val)[:, 1]

    # Core metrics
    metrics = compute_core_metrics(y_val, y_prob)
    log.info("AUC=%.4f  Gini=%.4f  KS=%.4f",
             metrics["AUC"], metrics["Gini"], metrics["KS"])

    # Threshold table
    threshold_df = threshold_analysis(y_val, y_prob, cfg)

    # Plots
    plot_roc_curve(y_val, y_prob, metrics["AUC"])
    plot_ks(y_val, y_prob)
    plot_shap_summary(model, X_val)

    # Report
    save_report(metrics, threshold_df, cfg)

    log.info("")
    log.info("Step 5 complete. Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()