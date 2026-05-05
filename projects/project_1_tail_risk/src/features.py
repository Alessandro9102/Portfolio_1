# ─────────────────────────────────────────────
# src/features.py
#
# Responsibility: engineer the feature matrix
# that feeds the HMM regime detector.
#
# The HMM sees a compact set of stationary,
# scaled signals – NOT raw prices.  Every
# feature here has an economic motivation.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Individual feature constructors
# ─────────────────────────────────────────────

def rolling_realized_vol(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Annualised rolling realised volatility.

    σ_t = std(r_{t-window+1} … r_t) × √252

    A 21-day (≈1 month) window captures the medium-term
    volatility regime without being too noisy (5-day) or
    too slow (63-day).
    """
    rv = returns.rolling(window).std() * np.sqrt(252)
    return rv.rename(f"realised_vol_{window}d")


def abs_return(returns: pd.Series) -> pd.Series:
    """Absolute daily log return – a fast-moving vol proxy.

    Useful as a second HMM feature because it reacts
    to single-day shocks faster than rolling vol.
    """
    return returns.abs().rename("abs_return")


def rolling_skewness(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling skewness over ~1 quarter.

    Turbulent regimes tend to exhibit negative skew
    (crash risk).  This helps the HMM distinguish
    a *volatile-but-symmetric* period from a
    *crash-prone* one.
    """
    sk = returns.rolling(window).skew()
    return sk.rename(f"skewness_{window}d")


def vix_level(vix: pd.Series) -> pd.Series:
    """VIX index level (already a market-implied vol signal).

    Included directly rather than VIX returns because
    the *level* of VIX is the economically meaningful
    regime signal (e.g. VIX > 30 → fear regime).
    """
    return vix.rename("vix_level")


# ─────────────────────────────────────────────
# Master feature builder
# ─────────────────────────────────────────────

def build_features(
    dataset: pd.DataFrame,
    vol_window: int = 21,
    skew_window: int = 63,
    scale: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct and optionally standardise the HMM feature matrix.

    Parameters
    ----------
    dataset    : output of data_loader.build_dataset()
    vol_window : look-back for rolling realised vol
    skew_window: look-back for rolling skewness
    scale      : z-score each feature (mean=0, std=1)
                 Required for HMM with 'full' covariance.

    Returns
    -------
    features_raw    : un-scaled DataFrame (for inspection / plotting)
    features_scaled : scaled DataFrame passed to HMM
    """
    equity_ret = dataset["equity_ret"]
    vix        = dataset["vix"]

    raw = pd.concat(
        [
            rolling_realized_vol(equity_ret, vol_window),
            abs_return(equity_ret),
            rolling_skewness(equity_ret, skew_window),
            vix_level(vix),
        ],
        axis=1,
    )

    # Drop rows with NaN introduced by rolling windows
    n_before = len(raw)
    raw = raw.dropna()
    log.info(
        "Feature matrix: %d rows (dropped %d for warm-up)",
        len(raw),
        n_before - len(raw),
    )

    if not scale:
        return raw, raw.copy()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(raw.values)
    features_scaled = pd.DataFrame(
        scaled_values,
        index=raw.index,
        columns=raw.columns,
    )

    log.info("Features scaled (StandardScaler).  Columns: %s", list(raw.columns))
    return raw, features_scaled


# ─────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.data_loader import build_dataset

    ds = build_dataset()
    raw_f, scaled_f = build_features(ds)

    print("\n── Raw features (tail) ──")
    print(raw_f.tail())
    print(f"\nShape: {raw_f.shape}")
    print(f"\nDescriptive stats:\n{raw_f.describe().round(4)}")