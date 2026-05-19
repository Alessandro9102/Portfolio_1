"""
get_data.py
-----------
Step 1: Download the Give Me Some Credit dataset from Kaggle.

Usage (from project_2_credit_scoring/):
    python src/get_data.py

Requirements:
    - Kaggle account + API token at ~/.kaggle/kaggle.json
      OR manual download from:
      https://www.kaggle.com/competitions/GiveMeSomeCredit/data
      Place cs-training.csv in data/raw/

Output:
    data/raw/cs-training.csv
    data/raw/cs-test.csv
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # project_2_credit_scoring/
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CSV = RAW_DIR / "cs-training.csv"
TEST_CSV     = RAW_DIR / "cs-test.csv"

KAGGLE_COMPETITION = "GiveMeSomeCredit"


def download_via_kaggle_api() -> bool:
    """Try to pull data using the Kaggle CLI / Python API."""
    try:
        import kaggle  # noqa: F401 — triggers credential check on import
    except ImportError:
        log.warning("kaggle package not installed — skipping API download.")
        return False
    except OSError as exc:
        log.warning("Kaggle credentials not found (%s) — skipping API download.", exc)
        return False

    try:
        import subprocess
        log.info("Downloading via Kaggle API …")
        subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "competitions", "download",
                "-c", KAGGLE_COMPETITION,
                "-p", str(RAW_DIR),
            ],
            check=True,
        )
        _unzip_all(RAW_DIR)
        return True
    except Exception as exc:
        log.warning("Kaggle API download failed: %s", exc)
        return False


def _unzip_all(directory: Path) -> None:
    """Unzip any .zip files found in *directory*."""
    for zip_path in directory.glob("*.zip"):
        log.info("Extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)
        zip_path.unlink()
        log.info("Deleted zip after extraction.")


def check_manual_download() -> bool:
    """Return True if the user already placed the CSVs manually."""
    if TRAINING_CSV.exists():
        log.info("Found %s — manual download detected.", TRAINING_CSV.name)
        return True
    return False


def print_manual_instructions() -> None:
    """Print clear instructions when no automated path works."""
    msg = f"""
╔══════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD REQUIRED                               ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Go to:                                                       ║
║     https://www.kaggle.com/competitions/GiveMeSomeCredit/data   ║
║                                                                  ║
║  2. Sign in (free account) and accept the competition rules.     ║
║                                                                  ║
║  3. Download:  cs-training.csv                                   ║
║                cs-test.csv                                       ║
║                                                                  ║
║  4. Place both files in:                                         ║
║     {str(RAW_DIR):<60}║
║                                                                  ║
║  5. Re-run:  python src/get_data.py                              ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(msg)


def validate_csv() -> None:
    """Quick sanity check on the training CSV."""
    import pandas as pd

    log.info("Validating cs-training.csv …")
    df = pd.read_csv(TRAINING_CSV, nrows=5)
    expected_cols = {
        "SeriousDlqin2yrs",
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
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema — missing columns: {missing}")

    log.info("Schema OK — %d columns detected.", len(df.columns))


def main() -> None:
    log.info("=== Step 1: Get Data ===")
    log.info("Target directory: %s", RAW_DIR)

    # Priority 1: already there
    if check_manual_download():
        validate_csv()
        log.info("✓ Data ready. Proceed to Step 2: python src/data_preprocessing.py")
        return

    # Priority 2: Kaggle API
    if download_via_kaggle_api() and TRAINING_CSV.exists():
        validate_csv()
        log.info("✓ Data ready. Proceed to Step 2: python src/data_preprocessing.py")
        return

    # Fallback: manual instructions
    print_manual_instructions()
    sys.exit(1)


if __name__ == "__main__":
    main()