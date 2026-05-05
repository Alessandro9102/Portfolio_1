# ─────────────────────────────────────────────
# config.py  –  Central project configuration
# All tuneable parameters live here so no
# magic numbers are scattered across modules.
# ─────────────────────────────────────────────

from pathlib import Path

# ── Paths ─────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Data download ──────────────────────────────
TICKERS = {
    "equity": "SPY",   # S&P 500 ETF – our main risk asset
    "vix":    "^VIX",  # CBOE Volatility Index
}
START_DATE = "2000-01-01"
END_DATE   = None          # None → today

# ── Hidden Markov Model (regime detection) ─────
HMM_N_STATES       = 2    # 0 = calm / low-vol, 1 = turbulent / high-vol
HMM_N_ITER         = 1000
HMM_COVARIANCE_TYPE = "full"
HMM_RANDOM_STATE   = 42

# ── GARCH volatility model ─────────────────────
GARCH_P = 1   # ARCH lag order
GARCH_Q = 1   # GARCH lag order

# ── Extreme Value Theory (EVT) ─────────────────
EVT_THRESHOLD_QUANTILE = 0.05   # losses below 5th pctile feed the GPD
EVT_MIN_EXCEEDANCES    = 50     # minimum tail observations required

# ── Risk metrics ──────────────────────────────
VAR_LEVELS = [0.95, 0.99]       # confidence levels for VaR / ES
DEFAULT_VAR_LEVEL = 0.99

# ── Backtesting ───────────────────────────────
BACKTEST_WINDOW = 252           # rolling 1-year window (trading days)

# ── Misc ──────────────────────────────────────
RANDOM_SEED = 42
PLOT_STYLE  = "dark_background"  # matplotlib style for all charts