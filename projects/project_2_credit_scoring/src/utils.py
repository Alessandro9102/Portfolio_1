"""
utils.py
--------
Shared utility functions used across all pipeline steps.

Covers:
  - YAML config loading
  - Reproducible seeding
  - Logging setup
  - Path resolution helpers
  - DataFrame type-checking helpers
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# -- Paths ----------------------------------------------------------------------

# Resolve project root regardless of where Python is invoked from.
# Assumes utils.py lives at:  project_2_credit_scoring/src/utils.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"
CONFIG_PATH    = PROJECT_ROOT / "config" / "config.yaml"


def get_project_root() -> Path:
    return PROJECT_ROOT


def ensure_dirs() -> None:
    """Create all standard output directories if they don't exist."""
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# -- Logging --------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a consistent format.

    Usage:
        from src.utils import get_logger
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# -- Config ---------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Load the YAML config file and return it as a dict.

    Raises FileNotFoundError with a helpful message if the config is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}.\n"
            "Make sure config/config.yaml exists in the project root."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -- Reproducibility ------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and os environment.
    Call once at the top of each script / notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -- DataFrame helpers ----------------------------------------------------------

def summarise_df(df: pd.DataFrame, label: str = "") -> None:
    """
    Print a quick data quality summary to stdout.
    Useful for logging at the start/end of each pipeline step.
    """
    log = get_logger("utils")
    tag = f"[{label}] " if label else ""
    log.info("%sShape: %s", tag, df.shape)
    log.info("%sDtypes:\n%s", tag, df.dtypes.value_counts().to_string())

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log.info("%sNo missing values.", tag)
    else:
        pct = (missing / len(df) * 100).round(2)
        summary = pd.DataFrame({"missing": missing, "pct": pct})
        log.info("%sMissing values:\n%s", tag, summary.to_string())


def check_no_leakage(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """
    Assert that no row appears in both train and val sets.

    Uses a row-content hash -- index values are NOT used because
    reset_index() makes them overlap by design (0..N in both sets).
    """
    overlap = set(train["row_id"]) & set(val["row_id"])
    assert len(overlap) == 0, f"Leakage: {len(overlap)} rows"


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return approximate in-memory size of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory footprint by downcasting numeric columns.
    Converts float64 -> float32 and int64 -> int32 where safe.
    Returns a copy.
    """
    df = df.copy()
    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="integer").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# -- Feature name constants -----------------------------------------------------
# Single source of truth for column names used across the pipeline.

TARGET = "SeriousDlqin2yrs"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Delinquency columns (used in feature engineering aggregations)
DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]