from pathlib import Path

# =========================
# BASE PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SRC_DIR = BASE_DIR / "src"


# =========================
# DATA CONFIG
# =========================

RAW_DATA_FILE = RAW_DATA_DIR / "data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "data_clean.csv"


# =========================
# MODEL CONFIG
# =========================

RANDOM_STATE = 42

TEST_SIZE = 0.2

MODEL_PARAMS = {
    "random_state": RANDOM_STATE
}