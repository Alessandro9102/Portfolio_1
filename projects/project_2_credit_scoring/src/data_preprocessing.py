"""
data_preprocessing.py
---------------------
Step 2: Clean the raw CSV and produce stratified train / val splits.

Pipeline (in order):
  1. Load raw CSV
  2. Drop junk columns (Kaggle row-index artifact)
  3. Cap outliers  (defined in config.yaml)
  4. Impute missing values (median, per column)
  5. Drop invalid rows (age < 18)
  6. Stratified train / val split
  7. Save to data/processed/ as Parquet (fast, typed, compressed)

Usage (from project_2_credit_scoring/):
    python src/data_preprocessing.py

Outputs:
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/preprocessing_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -- Make src/ importable when running as a script -----------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    DATA_PROCESSED,
    DATA_RAW,
    PROJECT_ROOT,
    check_no_leakage,
    downcast_dtypes,
    ensure_dirs,
    get_logger,
    load_config,
    memory_usage_mb,
    set_seed,
    summarise_df,
)

log = get_logger(__name__)


# ==============================================================================
# 1. Load
# ==============================================================================

def load_raw(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["data"]["raw_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            "Run  python src/get_data.py  first."
        )
    log.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path)
    log.info("Raw shape: %s", df.shape)
    return df


# ==============================================================================
# 2. Drop junk columns
# ==============================================================================

def drop_junk_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols_to_drop = [
        c for c in cfg["preprocessing"]["drop_cols"] if c in df.columns
    ]
    if cols_to_drop:
        log.info("Dropping columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


# ==============================================================================
# 3. Cap outliers
# ==============================================================================

def cap_outliers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Hard-cap extreme values defined in config.yaml.

    Why cap instead of remove?
      - Removing outliers in credit data loses real high-risk borrowers.
      - Values like RevolvingUtilization > 1 are DATA ERRORS, not real signals.
      - Sentinel codes (96, 98 in delinquency cols) must be treated as caps.
    """
    caps: dict = cfg["preprocessing"]["outlier_caps"]
    report_lines = []

    for col, cap_val in caps.items():
        if col not in df.columns:
            log.warning("Cap column '%s' not found -- skipping.", col)
            continue
        n_capped = (df[col] > cap_val).sum()
        if n_capped > 0:
            pct = n_capped / len(df) * 100
            log.info("  Capping %-45s  >%6g  ->  %d rows (%.2f%%)",
                     col, cap_val, n_capped, pct)
            report_lines.append(
                f"{col}: {n_capped} rows capped at {cap_val} ({pct:.2f}%)"
            )
            df[col] = df[col].clip(upper=cap_val)

    return df, report_lines


# ==============================================================================
# 4. Impute missing values
# ==============================================================================

def impute_missing(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fit imputation statistics on TRAIN only, then apply to both sets.
    Prevents data leakage from validation set into imputation values.

    Returns:
        train, val (imputed), impute_stats (dict of {col: value}) for logging.
    """
    strategy: dict = cfg["preprocessing"]["imputation"]
    impute_stats = {}

    for col, method in strategy.items():
        if col not in train.columns:
            log.warning("Imputation column '%s' not found -- skipping.", col)
            continue

        n_train_missing = train[col].isnull().sum()
        n_val_missing   = val[col].isnull().sum()

        if method == "median":
            fill_val = train[col].median()
        elif method == "mean":
            fill_val = train[col].mean()
        elif method == "mode":
            fill_val = train[col].mode()[0]
        else:
            raise ValueError(f"Unknown imputation method '{method}' for '{col}'")

        train[col] = train[col].fillna(fill_val)
        val[col]   = val[col].fillna(fill_val)

        log.info(
            "  Imputed %-30s  method=%-6s  fill=%.2f  "
            "(train missing: %d, val missing: %d)",
            col, method, fill_val, n_train_missing, n_val_missing,
        )
        impute_stats[col] = {"method": method, "fill_value": round(fill_val, 4)}

    return train, val, impute_stats


# ==============================================================================
# 5. Drop invalid rows
# ==============================================================================

def drop_invalid_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Remove rows that are logically impossible (e.g. age < 18)."""
    filters = cfg["preprocessing"]["drop_filters"]

    age_min = filters.get("age_min", 18)
    before = len(df)
    df = df[df["age"] >= age_min].copy()
    removed = before - len(df)
    if removed:
        log.info("  Dropped %d rows with age < %d.", removed, age_min)

    return df


# ==============================================================================
# 6. Stratified split
# ==============================================================================

def stratified_split(
    df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split preserves the ~6.7% default rate in both sets.
    Critical for imbalanced classification -- random split can skew class ratios.
    """
    target    = cfg["data"]["target_col"]
    test_size = cfg["data"]["test_size"]
    seed      = cfg["seed"]

    train, val = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=seed,
    )

    log.info(
        "Split -> train: %d rows (default rate %.2f%%)  |  "
        "val: %d rows (default rate %.2f%%)",
        len(train), train[target].mean() * 100,
        len(val),   val[target].mean()   * 100,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


# ==============================================================================
# 7. Save
# ==============================================================================

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: dict,
) -> None:
    """Save processed splits as Parquet (typed, compressed, fast to read)."""
    train_path = PROJECT_ROOT / cfg["data"]["train_file"]
    val_path   = PROJECT_ROOT / cfg["data"]["val_file"]

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path,   index=False)

    log.info("Saved -> %s  (%.1f MB)", train_path, memory_usage_mb(train))
    log.info("Saved -> %s  (%.1f MB)", val_path,   memory_usage_mb(val))


def save_preprocessing_report(
    impute_stats: dict,
    cap_report: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: dict,
) -> None:
    """Write a plain-text audit trail of all preprocessing decisions."""
    report_path = DATA_PROCESSED / "preprocessing_report.txt"
    target = cfg["data"]["target_col"]

    lines = [
        "=" * 70,
        "PREPROCESSING REPORT",
        "=" * 70,
        "",
        f"Train rows : {len(train):,}",
        f"Val rows   : {len(val):,}",
        f"Features   : {len(train.columns) - 1}",
        f"Target col : {target}",
        "",
        f"Train default rate : {train[target].mean():.4f}",
        f"Val   default rate : {val[target].mean():.4f}",
        "",
        "-- Outlier Caps -------------------------------------",
        *cap_report,
        "",
        "-- Imputation ---------------------------------------",
    ]
    for col, stats in impute_stats.items():
        lines.append(f"  {col}: {stats['method']} -> {stats['fill_value']}")

    lines += [
        "",
        "-- Remaining Missing Values (should be 0) -----------",
    ]
    combined = pd.concat([train, val])
    missing = combined.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("  None OK")
    else:
        for col, n in missing.items():
            lines.append(f"  {col}: {n}")

    lines += ["", "=" * 70]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Preprocessing report saved -> %s", report_path)


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    log.info("=== Step 2: Data Preprocessing ===")
    ensure_dirs()

    cfg = load_config()
    set_seed(cfg["seed"])

    # 1. Load
    df = load_raw(cfg)
    summarise_df(df, label="raw")

    # 2. Drop junk
    df = drop_junk_columns(df, cfg)

    # 3. Cap outliers (on full dataset before split -- caps are domain rules, not statistics)
    df, cap_report = cap_outliers(df, cfg)

    # 4. Drop invalid rows
    df = drop_invalid_rows(df, cfg)

    # Assign stable row ID before split -- used by leakage check
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df))

    # 5. Stratified split
    train, val = stratified_split(df, cfg)

    # 6. Impute (fit on train -> apply to both)
    log.info("Imputing missing values (fit on train only) ...")
    train, val, impute_stats = impute_missing(train, val, cfg)

    # 7. Sanity checks
    check_no_leakage(train, val)

    remaining_missing = pd.concat([train, val]).isnull().sum().sum()
    assert remaining_missing == 0, (
        f"ERROR {remaining_missing} missing values remain after imputation!"
    )
    log.info("OK No missing values remain.")

    # 8. Downcast dtypes to save memory
    train = downcast_dtypes(train)
    val   = downcast_dtypes(val)

    summarise_df(train, label="train_final")
    summarise_df(val,   label="val_final")

    # 9. Save
    save_splits(train, val, cfg)
    save_preprocessing_report(impute_stats, cap_report, train, val, cfg)

    log.info("")
    log.info("OK Step 2 complete. Next: python src/feature_engineering.py")


if __name__ == "__main__":
    main()