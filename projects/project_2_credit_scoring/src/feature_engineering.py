"""

-----------------------
Step 3: Build domain-driven features and scale them.

Pipeline (in order):
  1. Load train.parquet / val.parquet  (output of data_preprocessing.py)
  2. Build delinquency aggregate features
  3. Build debt / income ratio features
  4. Build interaction terms  (defined in config.yaml)
  5. Log-transform skewed columns  (defined in config.yaml)
  6. Fit StandardScaler on TRAIN only  -> apply to both  (no leakage)
  7. Save engineered sets + scaler

Usage (from project_2_credit_scoring/):
    python src/feature_engineering.py

Outputs:
    data/processed/train_fe.parquet
    data/processed/val_fe.parquet
    models/scaler.pkl
    data/processed/feature_list.txt
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -- Make src/ importable when running as a script ----------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    DATA_PROCESSED,
    MODELS_DIR,
    PROJECT_ROOT,
    DELINQUENCY_COLS,
    TARGET,
    ensure_dirs,
    get_logger,
    load_config,
    memory_usage_mb,
    set_seed,
    summarise_df,
)

log = get_logger(__name__)


# =============================================================================
# 1. Load
# =============================================================================

def load_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROJECT_ROOT / cfg["data"]["train_file"]
    val_path   = PROJECT_ROOT / cfg["data"]["val_file"]

    for p in [train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Processed file not found: {p}\n"
                "Run  python src/data_preprocessing.py  first."
            )

    train = pd.read_parquet(train_path)
    val   = pd.read_parquet(val_path)
    log.info("Loaded train %s  val %s", train.shape, val.shape)
    return train, val


# =============================================================================
# 2. Delinquency aggregate features
# =============================================================================

def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the three delinquency bucket columns into summary signals.

    Why:
      - Individual delinquency counts are noisy; totals and max capture
        overall credit behaviour more robustly.
      - 'ever_90_days_late' is a hard binary flag used by credit bureaus.
    """
    cols = [c for c in DELINQUENCY_COLS if c in df.columns]

    df["total_delinquencies"]  = df[cols].sum(axis=1)
    df["max_delinquency_band"] = df[cols].max(axis=1)
    df["ever_90_days_late"]    = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    log.info("Added delinquency features: total, max_band, ever_90_days_late")
    return df


# =============================================================================
# 3. Debt / income ratio features
# =============================================================================

def add_debt_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven ratios used by credit risk analysts.

    Why:
      - DebtRatio alone is ambiguous without income context.
      - monthly_debt_payment gives an absolute dollar-amount intuition.
      - debt_to_income_safe clips extreme income values to avoid inf/NaN.
    """
    # Safe income: replace 0 with NaN then fill with median to avoid div/0
    safe_income = df["MonthlyIncome"].replace(0, np.nan)
    safe_income = safe_income.fillna(safe_income.median())

    df["monthly_debt_payment"]  = df["DebtRatio"] * safe_income
    df["debt_to_income_safe"]   = df["monthly_debt_payment"] / (safe_income + 1)

    # Credit line utilisation per open credit line
    safe_lines = df["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan).fillna(1)
    df["utilisation_per_line"]  = (
        df["RevolvingUtilizationOfUnsecuredLines"] / safe_lines
    )

    # Age-based risk proxy: younger borrowers with high utilisation
    df["age_x_utilisation"]     = df["age"] * df["RevolvingUtilizationOfUnsecuredLines"]

    log.info(
        "Added debt/income features: monthly_debt_payment, debt_to_income_safe, "
        "utilisation_per_line, age_x_utilisation"
    )
    return df


# =============================================================================
# 4. Interaction terms
# =============================================================================

def add_interaction_terms(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Multiply column pairs defined in config.yaml.
    Names are auto-generated as  colA_x_colB.
    """
    interactions = cfg["feature_engineering"].get("interactions", [])
    added = []
    for col_a, col_b in interactions:
        if col_a not in df.columns or col_b not in df.columns:
            log.warning("Interaction skipped -- missing column: %s or %s", col_a, col_b)
            continue
        name = f"{col_a}_x_{col_b}"
        df[name] = df[col_a] * df[col_b]
        added.append(name)

    if added:
        log.info("Added interaction terms: %s", added)
    return df


# =============================================================================
# 5. Log-transform skewed columns
# =============================================================================

def log_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply log1p to heavily right-skewed columns defined in config.yaml.

    log1p(x) = log(1 + x) handles zeros safely.
    We rename columns with a '_log' suffix to make it explicit in SHAP plots.
    """
    cols = cfg["feature_engineering"].get("log_transform", [])
    renamed = {}
    for col in cols:
        if col not in df.columns:
            log.warning("Log-transform skipped -- missing column: %s", col)
            continue
        new_name = f"{col}_log"
        df[new_name] = np.log1p(df[col].clip(lower=0))
        renamed[col] = new_name

    if renamed:
        log.info("Log-transformed: %s", list(renamed.keys()))
    return df


# =============================================================================
# 6. Scale (fit on train only)
# =============================================================================

def fit_and_apply_scaler(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit StandardScaler on TRAIN features only, then transform both sets.

    Critical: fitting on val/full data would leak validation statistics
    into the training process -- a common but serious mistake.
    """
    scaler = StandardScaler()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols]   = scaler.transform(val[feature_cols])

    log.info("Scaler fitted on train (%d features) and applied to val.", len(feature_cols))
    return train, val, scaler


# =============================================================================
# 7. Save
# =============================================================================

def save_engineered(
    train: pd.DataFrame,
    val: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> None:
    train_path = DATA_PROCESSED / "train_fe.parquet"
    val_path   = DATA_PROCESSED / "val_fe.parquet"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_list_path = DATA_PROCESSED / "feature_list.txt"

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path,     index=False)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    feature_list_path.write_text("\n".join(feature_cols), encoding="utf-8")

    log.info("Saved train_fe  -> %s  (%.1f MB)", train_path, memory_usage_mb(train))
    log.info("Saved val_fe    -> %s  (%.1f MB)", val_path,   memory_usage_mb(val))
    log.info("Saved scaler    -> %s", scaler_path)
    log.info("Saved %d feature names -> %s", len(feature_cols), feature_list_path)


# =============================================================================
# Helpers
# =============================================================================

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all model input columns -- everything except target and helper cols.
    """
    exclude = {TARGET, "row_id"}
    return [c for c in df.columns if c not in exclude]


def validate_no_inf_nan(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Catch any inf / NaN introduced during feature engineering."""
    for name, df in [("train", train), ("val", val)]:
        n_nan = df.isnull().sum().sum()
        n_inf = np.isinf(df.select_dtypes(include="number")).sum().sum()
        if n_nan > 0 or n_inf > 0:
            bad = df.columns[df.isnull().any() | np.isinf(df.select_dtypes(include="number")).any()].tolist()
            raise ValueError(
                f"{name} has {n_nan} NaN and {n_inf} inf values "
                f"after feature engineering. Columns: {bad}"
            )
    log.info("Validation OK -- no NaN or inf in engineered features.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    log.info("=== Step 3: Feature Engineering ===")
    ensure_dirs()

    cfg = load_config()
    set_seed(cfg["seed"])

    # 1. Load preprocessed splits
    train, val = load_splits(cfg)

    # 2. Delinquency aggregates
    train = add_delinquency_features(train)
    val   = add_delinquency_features(val)

    # 3. Debt / income ratios
    train = add_debt_income_features(train)
    val   = add_debt_income_features(val)

    # 4. Interaction terms
    train = add_interaction_terms(train, cfg)
    val   = add_interaction_terms(val, cfg)

    # 5. Log transforms
    train = log_transform(train, cfg)
    val   = log_transform(val, cfg)

    # 6. Identify feature columns BEFORE scaling (excludes target + row_id)
    feature_cols = get_feature_cols(train)
    log.info("Total features before scaling: %d", len(feature_cols))

    # 7. Validate no inf / NaN before scaling
    validate_no_inf_nan(train, val)

    # 8. Scale (fit on train only)
    train, val, scaler = fit_and_apply_scaler(train, val, feature_cols, cfg)

    # 9. Final summary
    summarise_df(train, label="train_fe")
    summarise_df(val,   label="val_fe")

    # 10. Save
    save_engineered(train, val, scaler, feature_cols)

    log.info("")
    log.info("Step 3 complete. Next: python src/train_model.py")


if __name__ == "__main__":
    main()