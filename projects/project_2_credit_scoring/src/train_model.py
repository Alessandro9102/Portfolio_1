"""
train_model.py
--------------
Step 4: Train three models in sequence.

Stages:
  1. Logistic Regression  -- interpretable baseline
  2. XGBoost              -- gradient boosting, handles imbalance natively
  3. Stacking Ensemble    -- LR + XGB base learners, LR meta-learner

All hyperparameters are read from config/config.yaml.
No magic numbers in this file.

Usage (from project_2_credit_scoring/):
    python src/train_model.py

Outputs:
    models/lr_model.pkl
    models/xgb_model.pkl
    models/stack_model.pkl   <- primary model used by app + evaluate
    models/model.pkl         <- alias to stack_model (used by streamlit app)
    reports/training_summary.txt
"""

import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    DATA_PROCESSED,
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

def load_engineered(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_PROCESSED / "train_fe.parquet"
    val_path   = DATA_PROCESSED / "val_fe.parquet"

    for p in [train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Engineered file not found: {p}\n"
                "Run  python src/feature_engineering.py  first."
            )

    train = pd.read_parquet(train_path)
    val   = pd.read_parquet(val_path)
    log.info("Loaded train %s  val %s", train.shape, val.shape)
    return train, val


def split_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    exclude = {TARGET, "row_id"}
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols], df[TARGET]


# =============================================================================
# Stage 1 -- Logistic Regression baseline
# =============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict,
) -> LogisticRegression:
    log.info("--- Stage 1: Logistic Regression ---")
    params = cfg["models"]["logistic_regression"]

    model = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        solver=params["solver"],
        random_state=cfg["seed"],
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    log.info(
        "LR  |  val AUC: %.4f  |  time: %.1fs",
        val_auc, elapsed,
    )
    return model, val_auc


# =============================================================================
# Stage 2 -- XGBoost
# =============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict,
) -> XGBClassifier:
    log.info("--- Stage 2: XGBoost ---")
    params = cfg["models"]["xgboost"]

    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric=params["eval_metric"],
        early_stopping_rounds=params["early_stopping_rounds"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    best_iter = model.best_iteration
    log.info(
        "XGB |  val AUC: %.4f  |  best iter: %d  |  time: %.1fs",
        val_auc, best_iter, elapsed,
    )
    return model, val_auc


# =============================================================================
# Stage 3 -- Stacking Ensemble
# =============================================================================

def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    lr_model: LogisticRegression,
    xgb_model: XGBClassifier,
    cfg: dict,
) -> StackingClassifier:
    log.info("--- Stage 3: Stacking Ensemble ---")
    stack_cfg = cfg["models"]["stacking"]

    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=cfg["seed"],
    )

    # Re-instantiate base learners with fixed params so stacking
    # does its own internal cross-val cleanly (no fitted state carried over)
    lr_params  = cfg["models"]["logistic_regression"]
    xgb_params = cfg["models"]["xgboost"]

    base_lr = LogisticRegression(
        C=lr_params["C"],
        max_iter=lr_params["max_iter"],
        class_weight=lr_params["class_weight"],
        solver=lr_params["solver"],
        random_state=cfg["seed"],
    )

    base_xgb = XGBClassifier(
        n_estimators=200,           # reduced for stacking CV speed
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        min_child_weight=xgb_params["min_child_weight"],
        scale_pos_weight=xgb_params["scale_pos_weight"],
        eval_metric=xgb_params["eval_metric"],
        random_state=xgb_params["random_state"],
        n_jobs=-1,
    )

    stack = StackingClassifier(
        estimators=[
            ("lr",  base_lr),
            ("xgb", base_xgb),
        ],
        final_estimator=meta_learner,
        cv=stack_cfg["cv"],
        passthrough=stack_cfg["passthrough"],
        n_jobs=-1,
    )

    t0 = time.time()
    stack.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_auc = roc_auc_score(y_val, stack.predict_proba(X_val)[:, 1])
    log.info(
        "Stack|  val AUC: %.4f  |  time: %.1fs",
        val_auc, elapsed,
    )
    return stack, val_auc


# =============================================================================
# Save
# =============================================================================

def save_model(model, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("Saved %s -> %s", name, path)
    return path


def save_training_summary(
    results: dict[str, float],
    feature_cols: list[str],
) -> None:
    path = REPORTS_DIR / "training_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "TRAINING SUMMARY",
        "=" * 60,
        "",
        f"Features used : {len(feature_cols)}",
        "",
        "-- Validation AUC by model --",
    ]
    for model_name, auc in results.items():
        gini = 2 * auc - 1
        lines.append(f"  {model_name:<20} AUC={auc:.4f}  Gini={gini:.4f}")

    best = max(results, key=results.get)
    lines += [
        "",
        f"Best model    : {best}  (AUC={results[best]:.4f})",
        "",
        "-- Features --",
        *[f"  {c}" for c in feature_cols],
        "",
        "=" * 60,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Training summary -> %s", path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    log.info("=== Step 4: Model Training ===")
    ensure_dirs()

    cfg = load_config()
    set_seed(cfg["seed"])

    # Load
    train, val = load_engineered(cfg)
    X_train, y_train = split_xy(train)
    X_val,   y_val   = split_xy(val)

    log.info(
        "Class balance -- train: %.2f%%  val: %.2f%%  default rate",
        y_train.mean() * 100, y_val.mean() * 100,
    )

    results = {}

    # Stage 1 -- Logistic Regression
    lr_model, lr_auc = train_logistic_regression(
        X_train, y_train, X_val, y_val, cfg
    )
    save_model(lr_model, "lr_model")
    results["LogisticRegression"] = lr_auc

    # Stage 2 -- XGBoost
    xgb_model, xgb_auc = train_xgboost(
        X_train, y_train, X_val, y_val, cfg
    )
    save_model(xgb_model, "xgb_model")
    results["XGBoost"] = xgb_auc

    # Stage 3 -- Stacking
    stack_model, stack_auc = train_stacking(
        X_train, y_train, X_val, y_val,
        lr_model, xgb_model, cfg,
    )
    save_model(stack_model, "stack_model")
    results["Stacking"] = stack_auc

    # Save primary model alias used by app + evaluate
    save_model(stack_model, "model")

    # Summary
    save_training_summary(results, list(X_train.columns))

    log.info("")
    log.info("Results:")
    for name, auc in results.items():
        log.info("  %-20s AUC=%.4f  Gini=%.4f", name, auc, 2 * auc - 1)

    log.info("")
    log.info("Step 4 complete. Next: python src/evaluate_model.py")


if __name__ == "__main__":
    main()