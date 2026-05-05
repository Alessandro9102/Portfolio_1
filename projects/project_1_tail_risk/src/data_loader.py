# ─────────────────────────────────────────────
# src/data_loader.py
#
# Responsibility: download raw OHLCV data from
# Yahoo Finance, compute log returns, clean the
# result, and persist a single Parquet file so
# every downstream module reads the same data.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# project config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR,
    END_DATE,
    START_DATE,
    TICKERS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────
RAW_FILE    = DATA_DIR / "raw_prices.parquet"
CLEAN_FILE  = DATA_DIR / "clean_data.parquet"


# ─────────────────────────────────────────────
# 1. Download
# ─────────────────────────────────────────────

def download_raw(
    tickers: dict[str, str] | None = None,
    start: str | None = None,
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted-close prices from Yahoo Finance.

    Parameters
    ----------
    tickers       : mapping of {label: ticker_symbol}; defaults to config.TICKERS
    start / end   : date strings (ISO format); defaults to config dates
    force_refresh : re-download even if a cached file already exists

    Returns
    -------
    DataFrame with columns = labels (e.g. 'equity', 'vix'), index = Date
    """
    tickers = tickers or TICKERS
    start   = start   or START_DATE
    end     = end     or END_DATE

    if RAW_FILE.exists() and not force_refresh:
        log.info("Loading cached raw prices from %s", RAW_FILE)
        return pd.read_parquet(RAW_FILE)

    log.info("Downloading data for %s (%s → %s) …", list(tickers.values()), start, end or "today")

    frames: dict[str, pd.Series] = {}
    for label, symbol in tickers.items():
        raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

        if raw.empty:
            raise ValueError(f"No data returned for ticker '{symbol}'.")

        # yfinance may return a MultiIndex column – flatten to 'Close'
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        frames[label] = raw["Close"].rename(label)

    prices = pd.concat(frames.values(), axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    prices.to_parquet(RAW_FILE)
    log.info("Raw prices saved → %s  (%d rows)", RAW_FILE, len(prices))
    return prices


# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: r_t = ln(P_t / P_{t-1}).

    Log returns are used throughout because they are:
    - time-additive (multi-period returns sum up correctly)
    - approximately normally distributed for small moves
    - better behaved for GARCH and HMM fitting
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns.columns = [f"{c}_ret" for c in prices.columns]
    return log_returns


# ─────────────────────────────────────────────
# 3. Cleaning
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ANY column is NaN (first row from differencing + any gaps).

    For genuine market-closure gaps we forward-fill prices before computing
    returns (handled upstream in build_dataset), but NaNs that survive into
    the returns frame are simply dropped.
    """
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning("Dropped %d rows with NaN values.", n_dropped)
    return df


# ─────────────────────────────────────────────
# 4. Master builder
# ─────────────────────────────────────────────

def build_dataset(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """End-to-end pipeline: download → forward-fill → log-returns → clean.

    Columns returned
    ----------------
    equity        : SPY adjusted close price
    vix           : VIX index level
    equity_ret    : SPY log return
    vix_ret       : VIX log return (used as a secondary feature)

    The clean Parquet is cached under data/clean_data.parquet.
    """
    if CLEAN_FILE.exists() and not force_refresh:
        log.info("Loading cached clean dataset from %s", CLEAN_FILE)
        return pd.read_parquet(CLEAN_FILE)

    # Step 1 – raw prices
    prices = download_raw(force_refresh=force_refresh)

    # Step 2 – forward-fill missing prices (e.g. VIX on some holidays)
    prices = prices.ffill()

    # Step 3 – log returns
    returns = compute_log_returns(prices)

    # Step 4 – merge prices + returns, drop the first NaN row
    dataset = pd.concat([prices, returns], axis=1)
    dataset = clean(dataset)

    # Step 5 – persist
    dataset.to_parquet(CLEAN_FILE)
    log.info(
        "Clean dataset saved → %s  (%d rows, %s to %s)",
        CLEAN_FILE,
        len(dataset),
        dataset.index[0].date(),
        dataset.index[-1].date(),
    )
    return dataset


# ─────────────────────────────────────────────
# Quick sanity-check when run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset(force_refresh=True)
    print("\n── Dataset head ──")
    print(df.head())
    print("\n── Dataset tail ──")
    print(df.tail())
    print(f"\nShape : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nNaN count:\n{df.isna().sum()}")