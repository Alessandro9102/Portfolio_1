# ─────────────────────────────────────────────
# src/tail_model.py
#
# Responsibility: model the extreme left tail
# of the return distribution using Extreme
# Value Theory (EVT) and the Generalised
# Pareto Distribution (GPD).
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import genpareto

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EVT_MIN_EXCEEDANCES, EVT_THRESHOLD_QUANTILE

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data class to hold GPD fit results cleanly
# ─────────────────────────────────────────────

@dataclass
class GPDFit:
    """Container for a fitted Generalised Pareto Distribution.

    Attributes
    ----------
    threshold  : u  – the loss threshold above which GPD is fitted
    n_total    : total number of observations in the return series
    n_exceed   : number of threshold exceedances (tail observations)
    shape      : ξ (xi) – tail shape parameter
                 ξ > 0 → heavy tail (power law)
                 ξ = 0 → exponential tail
                 ξ < 0 → bounded tail (Weibull)
    scale      : σ (sigma) – tail scale parameter
    loc        : location parameter (fixed at 0 for EVT peaks-over-threshold)
    exceedances: the raw excess losses that were fitted
    """
    threshold:   float
    n_total:     int
    n_exceed:    int
    shape:       float
    scale:       float
    loc:         float
    exceedances: np.ndarray = field(repr=False)

    @property
    def tail_fraction(self) -> float:
        """Fraction of observations in the tail: F_u = n_exceed / n_total"""
        return self.n_exceed / self.n_total

    def __str__(self) -> str:
        return (
            f"GPDFit | threshold={self.threshold:.4f}  "
            f"ξ={self.shape:.4f}  σ={self.scale:.4f}  "
            f"tail_fraction={self.tail_fraction:.4f}  "
            f"n_exceed={self.n_exceed}"
        )


# ─────────────────────────────────────────────
# 1.  Threshold selection
# ─────────────────────────────────────────────

def select_threshold(
    losses: pd.Series,
    quantile: float = EVT_THRESHOLD_QUANTILE,
    min_exceedances: int = EVT_MIN_EXCEEDANCES,
) -> float:
    """Select the GPD threshold via the quantile method.

    We work with *losses* (positive numbers) so the tail is on the
    right side.  The threshold u is set at the (1 - quantile)
    percentile of losses, meaning the worst `quantile` fraction
    of days feed the GPD.

    Why quantile method?
    --------------------
    More sophisticated methods (mean excess plot, Hill estimator)
    require subjective visual inspection.  The quantile approach is
    systematic, reproducible, and common in production risk systems.

    Parameters
    ----------
    losses          : series of losses (positive = loss)
    quantile        : fraction of worst observations to use (e.g. 0.05)
    min_exceedances : safety floor – raise quantile if needed

    Returns
    -------
    threshold : scalar u
    """
    u = np.quantile(losses, 1 - quantile)

    # Safety: ensure at least min_exceedances observations above threshold
    n_exceed = (losses > u).sum()
    if n_exceed < min_exceedances:
        # Lower the threshold (be less strict) to get more tail data
        required_quantile = min_exceedances / len(losses)
        u = np.quantile(losses, 1 - required_quantile)
        log.warning(
            "Threshold raised to %.4f to ensure %d exceedances "
            "(was %d with original threshold).",
            u, min_exceedances, n_exceed,
        )

    log.info(
        "EVT threshold u=%.4f (%.1f-pctile loss)  |  exceedances=%d",
        u, (1 - quantile) * 100, (losses > u).sum(),
    )
    return float(u)


# ─────────────────────────────────────────────
# 2.  GPD fitting
# ─────────────────────────────────────────────

def fit_gpd(
    losses: pd.Series,
    threshold: Optional[float] = None,
    quantile: float = EVT_THRESHOLD_QUANTILE,
) -> GPDFit:
    """Fit a Generalised Pareto Distribution to tail exceedances.

    EVT Peaks-Over-Threshold (POT) method:
    ----------------------------------------
    Given losses X_1, …, X_n, and a threshold u:
      1. Extract exceedances: Y_i = X_i - u  for all X_i > u
      2. Fit GPD(ξ, σ) to {Y_i} via maximum likelihood

    The GPD is theoretically justified by the Pickands-Balkema-de
    Haan theorem: for a large class of distributions, excesses over
    a high threshold converge to the GPD as u → ∞.

    Parameters
    ----------
    losses    : positive loss series (e.g. -returns where returns < 0)
    threshold : if None, computed via select_threshold()
    quantile  : used only when threshold=None

    Returns
    -------
    GPDFit dataclass
    """
    if threshold is None:
        threshold = select_threshold(losses, quantile)

    # Extract exceedances (losses beyond the threshold)
    excess = losses[losses > threshold] - threshold  # shift to zero
    exceedances_arr = excess.values

    if len(exceedances_arr) < 10:
        raise ValueError(
            f"Only {len(exceedances_arr)} exceedances above threshold "
            f"{threshold:.4f} – too few to fit GPD reliably."
        )

    # scipy genpareto: shape=c, loc=0 (fixed), scale=sigma
    shape, loc, scale = genpareto.fit(exceedances_arr, floc=0)

    fit = GPDFit(
        threshold=threshold,
        n_total=len(losses),
        n_exceed=len(exceedances_arr),
        shape=shape,
        scale=scale,
        loc=loc,
        exceedances=exceedances_arr,
    )

    log.info("GPD fit complete: %s", fit)

    if shape > 0.5:
        log.warning(
            "ξ=%.4f > 0.5 – very heavy tail.  "
            "Check for data outliers or use more tail data.",
            shape,
        )

    return fit


# ─────────────────────────────────────────────
# 3.  EVT-based VaR and ES
# ─────────────────────────────────────────────

def evt_var(
    fit: GPDFit,
    confidence: float = 0.99,
) -> float:
    """Compute EVT-based Value at Risk from a fitted GPD.

    Formula (McNeil & Frey, 2000):
    --------------------------------
    VaR_p = u + (σ/ξ) · [((1-p) / F_u)^{-ξ} - 1]

    where:
      u    = threshold
      σ, ξ = GPD parameters
      F_u  = tail fraction (n_exceed / n_total)
      p    = confidence level (e.g. 0.99)

    For ξ = 0 (exponential tail):
      VaR_p = u + σ · ln((1-p) / F_u)   [limiting case]

    Parameters
    ----------
    fit        : fitted GPDFit object
    confidence : VaR confidence level (0 < confidence < 1)

    Returns
    -------
    VaR estimate as a positive loss (e.g. 0.025 = 2.5% loss)
    """
    u   = fit.threshold
    xi  = fit.shape
    sig = fit.scale
    Fu  = fit.tail_fraction
    p   = confidence

    if abs(xi) < 1e-8:
        # Limiting case: exponential (GPD with ξ → 0)
        var_evt = u + sig * np.log((1 - p) / Fu)
    else:
        var_evt = u + (sig / xi) * (((1 - p) / Fu) ** (-xi) - 1)

    log.info("EVT VaR(%.0f%%): %.4f", confidence * 100, var_evt)
    return float(var_evt)


def evt_es(
    fit: GPDFit,
    confidence: float = 0.99,
) -> float:
    """Compute EVT-based Expected Shortfall from a fitted GPD.

    Formula:
    ---------
    ES_p = VaR_p / (1 - ξ) + (σ - ξ·u) / (1 - ξ)

    Requires ξ < 1 (always satisfied for realistic financial data).

    Expected Shortfall (also called CVaR or Tail VaR) answers:
    "Given that a loss exceeds VaR, how large is it on average?"
    It is a coherent risk measure, unlike VaR.

    Parameters
    ----------
    fit        : fitted GPDFit object
    confidence : confidence level matching the VaR

    Returns
    -------
    ES estimate as a positive loss
    """
    xi  = fit.shape
    sig = fit.scale
    u   = fit.threshold

    if xi >= 1:
        raise ValueError(
            f"ES is infinite when ξ ≥ 1 (got ξ={xi:.4f}).  "
            "Check threshold selection or data quality."
        )

    var_p = evt_var(fit, confidence)

    if abs(xi) < 1e-8:
        # Exponential limiting case
        es_evt = var_p + sig
    else:
        es_evt = var_p / (1 - xi) + (sig - xi * u) / (1 - xi)

    log.info("EVT ES(%.0f%%):  %.4f", confidence * 100, es_evt)
    return float(es_evt)


# ─────────────────────────────────────────────
# 4.  Goodness of fit helpers
# ─────────────────────────────────────────────

def gpd_quantile(fit: GPDFit, p: float) -> float:
    """GPD quantile for probability p (of the excess distribution).

    Used for QQ-plot construction: theoretical vs empirical quantiles.
    """
    xi, sig = fit.shape, fit.scale
    if abs(xi) < 1e-8:
        return -sig * np.log(1 - p)
    return (sig / xi) * ((1 - p) ** (-xi) - 1)


def qqplot_data(fit: GPDFit) -> tuple[np.ndarray, np.ndarray]:
    """Return (theoretical_quantiles, empirical_quantiles) for a QQ plot.

    Sorting the exceedances and comparing to GPD theoretical quantiles
    visually confirms whether the GPD fits the tail well.
    """
    n = len(fit.exceedances)
    empirical = np.sort(fit.exceedances)
    probs = (np.arange(1, n + 1) - 0.5) / n       # Hazen plotting positions
    theoretical = np.array([gpd_quantile(fit, p) for p in probs])
    return theoretical, empirical


# ─────────────────────────────────────────────
# 5.  Master pipeline
# ─────────────────────────────────────────────

def run_tail_model(
    dataset: pd.DataFrame,
    var_levels: list[float] | None = None,
) -> dict:
    """Full EVT tail modelling pipeline.

    Steps
    -----
    1. Convert returns to losses (negate negative returns).
    2. Select GPD threshold.
    3. Fit GPD to tail exceedances.
    4. Compute EVT VaR and ES at each confidence level.

    Parameters
    ----------
    dataset    : clean dataset with 'equity_ret' column
    var_levels : list of confidence levels, e.g. [0.95, 0.99]

    Returns
    -------
    dict with keys:
      'gpd_fit'      : GPDFit object
      'losses'       : pd.Series of losses (positive)
      'var_evt'      : dict {confidence: VaR_value}
      'es_evt'       : dict {confidence: ES_value}
      'qq_data'      : tuple (theoretical, empirical) for QQ plot
    """
    if var_levels is None:
        from config import VAR_LEVELS
        var_levels = VAR_LEVELS

    returns = dataset["equity_ret"]

    # Losses = positive numbers representing daily portfolio loss
    losses = (-returns).clip(lower=0)   # only keep loss days; gains → 0
    # For threshold/GPD we want all non-zero losses
    losses_nonzero = losses[losses > 0]

    log.info(
        "Building tail model on %d loss observations "
        "(%.1f%% of trading days)",
        len(losses_nonzero),
        100 * len(losses_nonzero) / len(losses),
    )

    gpd_fit = fit_gpd(losses_nonzero)

    var_evt: dict[float, float] = {}
    es_evt:  dict[float, float] = {}

    for level in var_levels:
        var_evt[level] = evt_var(gpd_fit, confidence=level)
        es_evt[level]  = evt_es(gpd_fit, confidence=level)
        log.info(
            "Level %.0f%%  |  EVT VaR=%.4f  EVT ES=%.4f",
            level * 100, var_evt[level], es_evt[level],
        )

    qq_theoretical, qq_empirical = qqplot_data(gpd_fit)

    return {
        "gpd_fit":   gpd_fit,
        "losses":    losses_nonzero,
        "var_evt":   var_evt,
        "es_evt":    es_evt,
        "qq_data":   (qq_theoretical, qq_empirical),
    }


# ─────────────────────────────────────────────
# Standalone sanity-check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader import build_dataset

    ds  = build_dataset()
    out = run_tail_model(ds)

    print("\n── GPD Fit ──")
    print(out["gpd_fit"])
    print("\n── EVT Risk Metrics ──")
    for lvl in sorted(out["var_evt"]):
        print(
            f"  {lvl*100:.0f}%  VaR={out['var_evt'][lvl]:.4f}  "
            f"ES={out['es_evt'][lvl]:.4f}"
        )
    print("\n── QQ data shape ──")
    print("Theoretical:", out["qq_data"][0][:5].round(4))
    print("Empirical:  ", out["qq_data"][1][:5].round(4))