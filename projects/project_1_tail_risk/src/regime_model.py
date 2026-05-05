# ─────────────────────────────────────────────
# src/regime_model.py
#
# Responsibility: fit a Gaussian Hidden Markov
# Model to detect market regimes, label every
# trading day, and expose helpers for downstream
# modules (GARCH, risk metrics, dashboard).
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR,
    HMM_COVARIANCE_TYPE,
    HMM_N_ITER,
    HMM_N_STATES,
    HMM_RANDOM_STATE,
)

log = logging.getLogger(__name__)

MODEL_PATH = DATA_DIR / "hmm_model.pkl"


# ─────────────────────────────────────────────
# 1.  Model training
# ─────────────────────────────────────────────

def train_hmm(
    features_scaled: pd.DataFrame,
    n_states: int = HMM_N_STATES,
    n_iter: int = HMM_N_ITER,
    covariance_type: str = HMM_COVARIANCE_TYPE,
    random_state: int = HMM_RANDOM_STATE,
    save_model: bool = True,
) -> GaussianHMM:
    """Fit a Gaussian HMM on the scaled feature matrix.

    Why Gaussian HMM?
    -----------------
    Each hidden state emits observations drawn from a
    multivariate Gaussian.  The EM algorithm (Baum-Welch)
    iterates until the log-likelihood converges, learning:
      - transition matrix  A[i,j]  (prob of state i → j)
      - emission means     μ_i
      - emission covariances  Σ_i

    Parameters
    ----------
    features_scaled : standardised feature matrix (T × d)
    n_states        : number of hidden regimes (default 2)
    n_iter          : max EM iterations
    covariance_type : 'full' allows each state its own Σ
    save_model      : pickle the fitted model for reuse

    Returns
    -------
    Fitted GaussianHMM instance
    """
    log.info(
        "Training HMM  |  states=%d  iter=%d  cov=%s",
        n_states, n_iter, covariance_type,
    )

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(features_scaled.values)

    log.info("HMM converged: %s  |  log-likelihood: %.2f", model.monitor_.converged, model.score(features_scaled.values))

    if save_model:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        log.info("Model saved → %s", MODEL_PATH)

    return model


def load_hmm() -> GaussianHMM:
    """Load a previously saved HMM from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved HMM found at {MODEL_PATH}. Run train_hmm() first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log.info("HMM loaded from %s", MODEL_PATH)
    return model


# ─────────────────────────────────────────────
# 2.  Regime labelling
# ─────────────────────────────────────────────

def _identify_calm_state(model: GaussianHMM, vol_feature_idx: int = 0) -> int:
    """Return the state index that corresponds to low volatility (calm).

    We use the emission mean of the first feature (realised vol).
    The state with the *lower* mean is the calm regime.
    This makes labels deterministic regardless of HMM initialisation.
    """
    means = model.means_[:, vol_feature_idx]  # shape (n_states,)
    calm_state = int(np.argmin(means))
    log.info(
        "State means (vol feature): %s  →  calm state = %d",
        means.round(3),
        calm_state,
    )
    return calm_state


def decode_regimes(
    model: GaussianHMM,
    features_scaled: pd.DataFrame,
) -> pd.DataFrame:
    """Viterbi-decode the most likely state sequence and compute probabilities.

    Returns
    -------
    DataFrame indexed like features_scaled with columns:
      regime       : 0 = calm,  1 = turbulent  (re-mapped from raw state ids)
      regime_raw   : original HMM state id (before re-mapping)
      prob_calm    : posterior probability of the calm state
      prob_turbulent : posterior probability of the turbulent state
    """
    X = features_scaled.values

    # Most-likely state sequence (Viterbi algorithm)
    raw_states = model.predict(X)

    # Posterior probabilities for each state at each time step
    posteriors = model.predict_proba(X)   # shape (T, n_states)

    calm_idx = _identify_calm_state(model)
    turb_idx = 1 - calm_idx              # works for 2-state models

    # Re-map so that 0 = calm, 1 = turbulent always
    regime = np.where(raw_states == calm_idx, 0, 1)

    result = pd.DataFrame(
        {
            "regime":          regime,
            "regime_raw":      raw_states,
            "prob_calm":       posteriors[:, calm_idx],
            "prob_turbulent":  posteriors[:, turb_idx],
        },
        index=features_scaled.index,
    )

    n_calm = (result["regime"] == 0).sum()
    n_turb = (result["regime"] == 1).sum()
    log.info(
        "Regime breakdown  |  calm: %d days (%.1f%%)  |  turbulent: %d days (%.1f%%)",
        n_calm, 100 * n_calm / len(result),
        n_turb, 100 * n_turb / len(result),
    )

    return result


# ─────────────────────────────────────────────
# 3.  Summary statistics per regime
# ─────────────────────────────────────────────

def regime_stats(
    regimes: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive statistics of SPY returns per regime.

    Useful for:
      - Validating the HMM found economically meaningful states
      - Dashboard summary table
    """
    merged = dataset[["equity_ret"]].join(regimes[["regime"]], how="inner")

    stats = (
        merged.groupby("regime")["equity_ret"]
        .agg(
            count="count",
            mean_return=lambda x: x.mean() * 252,        # annualised
            ann_volatility=lambda x: x.std() * np.sqrt(252),
            sharpe=lambda x: (x.mean() / x.std()) * np.sqrt(252),
            min_return="min",
            max_return="max",
            skewness=lambda x: x.skew(),
            kurtosis=lambda x: x.kurt(),
        )
        .rename(index={0: "Calm", 1: "Turbulent"})
    )

    return stats.round(4)


# ─────────────────────────────────────────────
# 4.  Convenience wrapper (full pipeline)
# ─────────────────────────────────────────────

def run_regime_detection(
    features_scaled: pd.DataFrame,
    dataset: pd.DataFrame,
    force_retrain: bool = False,
) -> tuple[GaussianHMM, pd.DataFrame, pd.DataFrame]:
    """Train (or load) HMM, decode regimes, compute stats.

    Returns
    -------
    model    : fitted GaussianHMM
    regimes  : DataFrame with regime labels + probabilities
    stats    : descriptive stats per regime
    """
    if MODEL_PATH.exists() and not force_retrain:
        model = load_hmm()
    else:
        model = train_hmm(features_scaled)

    regimes = decode_regimes(model, features_scaled)
    stats   = regime_stats(regimes, dataset)

    log.info("\nRegime statistics:\n%s", stats.to_string())

    return model, regimes, stats


# ─────────────────────────────────────────────
# Standalone sanity-check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader import build_dataset
    from src.features import build_features

    ds               = build_dataset()
    raw_f, scaled_f  = build_features(ds)

    model, regimes, stats = run_regime_detection(scaled_f, ds, force_retrain=True)

    print("\n── Regime tail ──")
    print(regimes.tail(10))
    print("\n── Regime stats ──")
    print(stats)