# ─────────────────────────────────────────────
# src/risk_metrics.py
#
# Responsibility: compute Value at Risk (VaR)
# and Expected Shortfall (ES) using three
# methods, then consolidate into a single
# comparison table for reporting.
#
# Methods
# -------
# 1. Historical Simulation  – empirical quantile, no distribution assumption
# 2. Parametric (Normal)    – Gaussian VaR, industry baseline
# 3. EVT-enhanced           – GPD tail fit from tail_model.py
# ─────────────────────────────────────────────

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_VAR_LEVEL, VAR_LEVELS
from src.tail_model import GPDFit, evt_es, evt_var

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  Historical Simulation VaR / ES
# ─────────────────────────────────────────────

def historical_var(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_LEVEL,
) -> float:
    """VaR via Historical Simulation (HS).

    Simply the empirical quantile of the loss distribution.
    No distribution assumption – the data speaks for itself.

    Limitation: requires enough history to populate the tail.
    At 99% confidence we need >100 observations to see the
    1% tail, >500 for a stable estimate.

    Returns
    -------
    VaR as a positive loss (e.g. 0.023 = 2.3% daily loss)
    """
    losses = -returns
    var_hs = float(np.quantile(losses, confidence))
    log.debug("HS VaR(%.0f%%): %.4f", confidence * 100, var_hs)
    return var_hs


def historical_es(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_LEVEL,
) -> float:
    """ES via Historical Simulation.

    Average of all losses exceeding the HS VaR threshold.
    This is the true sample mean of the tail – no parametric
    assumption, but noisy when the tail is sparsely populated.

    Returns
    -------
    ES as a positive loss
    """
    losses = -returns
    var_hs = historical_var(returns, confidence)
    es_hs  = float(losses[losses > var_hs].mean())
    log.debug("HS ES(%.0f%%): %.4f", confidence * 100, es_hs)
    return es_hs


# ─────────────────────────────────────────────
# 2.  Parametric (Normal) VaR / ES
# ─────────────────────────────────────────────

def normal_var(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_LEVEL,
    conditional_vol: float | None = None,
) -> float:
    """VaR under the Gaussian assumption.

    VaR_p = μ - z_p · σ   (expressed as a loss → positive)

    where z_p is the standard normal quantile for level p.

    If conditional_vol is supplied (from GARCH), we use it
    instead of the unconditional standard deviation.  This
    produces a *dynamic* Gaussian VaR that adapts to the
    current vol regime.

    Parameters
    ----------
    returns        : historical log returns
    confidence     : confidence level
    conditional_vol: GARCH one-step-ahead σ (annualised).
                     Pass daily σ (divide by √252) for daily VaR.

    Returns
    -------
    VaR as a positive loss
    """
    mu = returns.mean()

    if conditional_vol is not None:
        # Convert annualised vol → daily vol for daily VaR
        sigma = conditional_vol / np.sqrt(252)
    else:
        sigma = returns.std()

    z_p   = norm.ppf(1 - confidence)          # negative (left tail)
    var_n = float(-(mu + z_p * sigma))         # positive loss
    log.debug(
        "Normal VaR(%.0f%%) | μ=%.5f  σ=%.5f  z=%.3f → VaR=%.4f",
        confidence * 100, mu, sigma, z_p, var_n,
    )
    return var_n


def normal_es(
    returns: pd.Series,
    confidence: float = DEFAULT_VAR_LEVEL,
    conditional_vol: float | None = None,
) -> float:
    """ES under the Gaussian assumption.

    ES_p = μ + σ · φ(z_p) / (1 - p)

    where φ is the standard normal PDF.

    Returns
    -------
    ES as a positive loss
    """
    mu = returns.mean()

    if conditional_vol is not None:
        sigma = conditional_vol / np.sqrt(252)
    else:
        sigma = returns.std()

    z_p  = norm.ppf(1 - confidence)
    phi  = norm.pdf(z_p)
    es_n = float(-(mu - sigma * phi / (1 - confidence)))
    log.debug("Normal ES(%.0f%%): %.4f", confidence * 100, es_n)
    return es_n


# ─────────────────────────────────────────────
# 3.  Dynamic rolling VaR (time series)
# ─────────────────────────────────────────────

def rolling_normal_var(
    returns: pd.Series,
    cond_vol: pd.Series,
    confidence: float = DEFAULT_VAR_LEVEL,
) -> pd.Series:
    """Compute a daily dynamic VaR series using GARCH conditional vol.

    At each time t, VaR_t = μ_rolling - z_p · σ_t

    where σ_t comes from the GARCH conditional_volatility series
    (already available for every trading day).

    This is the primary output compared against actual returns
    during backtesting.

    Parameters
    ----------
    returns    : full log return series
    cond_vol   : annualised GARCH conditional vol series (same index)
    confidence : VaR confidence level

    Returns
    -------
    pd.Series of daily VaR (positive loss), same index as cond_vol
    """
    aligned = returns.align(cond_vol, join="inner")
    ret, cv  = aligned

    # Rolling 252-day mean as the drift estimate
    mu_roll = ret.rolling(252, min_periods=60).mean().fillna(ret.mean())

    # GARCH vol is annualised → convert to daily
    sigma_daily = cv / np.sqrt(252)

    z_p       = norm.ppf(1 - confidence)
    var_series = -(mu_roll + z_p * sigma_daily)
    var_series.name = f"var_{int(confidence*100)}_normal"

    log.info(
        "Rolling Normal VaR(%.0f%%) computed | mean=%.4f  max=%.4f",
        confidence * 100,
        var_series.mean(),
        var_series.max(),
    )
    return var_series


def rolling_hist_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = DEFAULT_VAR_LEVEL,
) -> pd.Series:
    """Rolling Historical Simulation VaR.

    Uses the past `window` days' empirical quantile as the VaR
    estimate.  Provides a non-parametric rolling benchmark.
    """
    losses    = -returns
    var_hist  = losses.rolling(window, min_periods=60).quantile(confidence)
    var_hist.name = f"var_{int(confidence*100)}_hist"
    return var_hist


# ─────────────────────────────────────────────
# 4.  Summary comparison table
# ─────────────────────────────────────────────

def build_risk_table(
    returns: pd.Series,
    gpd_fit: GPDFit,
    vol_forecast: float,
    var_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Build a clean comparison table of VaR and ES across methods.

    Columns: confidence level
    Rows:    Historical, Normal (unconditional), Normal (GARCH),
             EVT (GPD)

    Parameters
    ----------
    returns      : full equity log-return series
    gpd_fit      : fitted GPDFit from tail_model
    vol_forecast : one-step GARCH vol forecast (annualised)
    var_levels   : list of confidence levels to compute

    Returns
    -------
    pd.DataFrame with MultiIndex (method, metric) x confidence_level
    """
    var_levels = var_levels or VAR_LEVELS

    records = []
    for level in var_levels:
        col = f"{int(level*100)}%"

        # Historical Simulation
        records.append({
            "Method": "Historical Simulation",
            "Metric": "VaR",
            "Level":  col,
            "Value":  historical_var(returns, level),
        })
        records.append({
            "Method": "Historical Simulation",
            "Metric": "ES",
            "Level":  col,
            "Value":  historical_es(returns, level),
        })

        # Normal (unconditional)
        records.append({
            "Method": "Normal (unconditional)",
            "Metric": "VaR",
            "Level":  col,
            "Value":  normal_var(returns, level),
        })
        records.append({
            "Method": "Normal (unconditional)",
            "Metric": "ES",
            "Level":  col,
            "Value":  normal_es(returns, level),
        })

        # Normal + GARCH (dynamic)
        records.append({
            "Method": "Normal + GARCH",
            "Metric": "VaR",
            "Level":  col,
            "Value":  normal_var(returns, level, conditional_vol=vol_forecast),
        })
        records.append({
            "Method": "Normal + GARCH",
            "Metric": "ES",
            "Level":  col,
            "Value":  normal_es(returns, level, conditional_vol=vol_forecast),
        })

        # EVT (GPD)
        records.append({
            "Method": "EVT (GPD)",
            "Metric": "VaR",
            "Level":  col,
            "Value":  evt_var(gpd_fit, level),
        })
        records.append({
            "Method": "EVT (GPD)",
            "Metric": "ES",
            "Level":  col,
            "Value":  evt_es(gpd_fit, level),
        })

    df = pd.DataFrame(records)
    table = df.pivot_table(
        index=["Method", "Metric"],
        columns="Level",
        values="Value",
    ).round(4)

    # Preferred row order
    method_order = [
        "Historical Simulation",
        "Normal (unconditional)",
        "Normal + GARCH",
        "EVT (GPD)",
    ]
    table = table.reindex(method_order, level="Method")

    log.info("\nRisk metric comparison table:\n%s", table.to_string())
    return table


# ─────────────────────────────────────────────
# 5.  Master pipeline
# ─────────────────────────────────────────────

def run_risk_metrics(
    dataset: pd.DataFrame,
    cond_vol: pd.Series,
    vol_forecast: float,
    gpd_fit: GPDFit,
    var_levels: list[float] | None = None,
) -> dict:
    """Full risk metrics pipeline.

    Returns
    -------
    dict with keys:
      'risk_table'         : comparison DataFrame (methods × levels)
      'var_series_normal'  : dict {level: pd.Series} dynamic VaR
      'var_series_hist'    : dict {level: pd.Series} rolling HS VaR
    """
    var_levels = var_levels or VAR_LEVELS
    returns    = dataset["equity_ret"]

    risk_table = build_risk_table(returns, gpd_fit, vol_forecast, var_levels)

    var_series_normal: dict[float, pd.Series] = {}
    var_series_hist:   dict[float, pd.Series] = {}

    for level in var_levels:
        var_series_normal[level] = rolling_normal_var(returns, cond_vol, level)
        var_series_hist[level]   = rolling_hist_var(returns, confidence=level)

    return {
        "risk_table":        risk_table,
        "var_series_normal": var_series_normal,
        "var_series_hist":   var_series_hist,
    }


# ─────────────────────────────────────────────
# Standalone sanity-check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader      import build_dataset
    from src.features         import build_features
    from src.regime_model     import run_regime_detection
    from src.volatility_model import run_volatility_model
    from src.tail_model       import run_tail_model

    ds             = build_dataset()
    _, scaled_f    = build_features(ds)
    _, regimes, _  = run_regime_detection(scaled_f, ds)
    vol_out        = run_volatility_model(ds, regimes)
    tail_out       = run_tail_model(ds)

    metrics = run_risk_metrics(
        dataset      = ds,
        cond_vol     = vol_out["cond_vol"],
        vol_forecast = vol_out["vol_forecast"],
        gpd_fit      = tail_out["gpd_fit"],
    )

    print("\n── Risk Comparison Table ──")
    print(metrics["risk_table"].to_string())

    print("\n── Dynamic VaR 99% (tail) ──")
    print(metrics["var_series_normal"][0.99].dropna().tail(5).round(4))