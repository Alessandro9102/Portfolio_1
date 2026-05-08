# ─────────────────────────────────────────────
# src/volatility_model.py
#
# Responsibility: fit GARCH(1,1) models to
# estimate conditional volatility, optionally
# per regime, and produce one-step-ahead
# volatility forecasts for every trading day.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GARCH_P, GARCH_Q

log = logging.getLogger(__name__)

# arch emits ConvergenceWarning on some sub-samples – suppress for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="arch")


# ─────────────────────────────────────────────
# 1.  Single GARCH fit
# ─────────────────────────────────────────────

def fit_garch(
    returns: pd.Series,
    p: int = GARCH_P,
    q: int = GARCH_Q,
    dist: str = "StudentsT",
    label: str = "full",
) -> ARCHModelResult:
    """Fit a GARCH(p,q) model to a return series.

    Why GARCH(1,1)?
    ---------------
    The GARCH(1,1) model captures two empirical facts about
    financial returns:
      1. Volatility clustering – large moves follow large moves.
      2. Mean reversion – volatility eventually returns to its
         long-run average.

    Model: r_t = ε_t · σ_t
           σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Why Student's t distribution?
    ------------------------------
    Financial returns have fat tails.  A Normal distribution
    underestimates the probability of extreme losses, leading to
    VaR violations.  Student's t adds a degrees-of-freedom
    parameter that is estimated from the data.

    Parameters
    ----------
    returns : log return series (as %, i.e. multiplied by 100)
    p       : ARCH lag order (default 1)
    q       : GARCH lag order (default 1)
    dist    : innovation distribution ('Normal', 'StudentsT', 'SkewStudent')
    label   : name used in log messages

    Returns
    -------
    Fitted ARCHModelResult
    """
    # arch library works best with returns scaled to percentage
    r_pct = returns * 100

    model = arch_model(
        r_pct,
        vol="Garch",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )

    result = model.fit(disp="off", show_warning=False)

    log.info(
        "GARCH(%d,%d) [%s]  |  ω=%.4f  α=%.4f  β=%.4f  "
        "persistence=%.4f  AIC=%.1f",
        p, q, label,
        result.params["omega"],
        result.params["alpha[1]"],
        result.params["beta[1]"],
        result.params["alpha[1]"] + result.params["beta[1]"],
        result.aic,
    )

    # Warn if model is near-integrated (persistence ≥ 0.999)
    persistence = result.params["alpha[1]"] + result.params["beta[1]"]
    if persistence >= 0.999:
        log.warning(
            "GARCH [%s]: persistence %.4f ≥ 0.999 – "
            "model may be near-integrated (check data quality).",
            label, persistence,
        )

    return result


# ─────────────────────────────────────────────
# 2.  Extract conditional volatility series
# ─────────────────────────────────────────────

def extract_conditional_vol(
    result: ARCHModelResult,
    annualise: bool = True,
) -> pd.Series:
    """Pull the fitted conditional volatility from a GARCH result.

    The arch library stores conditional_volatility in *percentage*
    units (matching the scaled input), so we divide by 100 to
    restore decimal scale, then optionally annualise.

    Parameters
    ----------
    result    : fitted ARCHModelResult
    annualise : multiply daily vol by √252 to get annual vol

    Returns
    -------
    pd.Series of conditional volatility, same index as fitted data
    """
    cond_vol = result.conditional_volatility / 100  # back to decimal

    if annualise:
        cond_vol = cond_vol * np.sqrt(252)

    cond_vol.name = "cond_vol"
    return cond_vol


# ─────────────────────────────────────────────
# 3.  One-step-ahead volatility forecast
# ─────────────────────────────────────────────

def forecast_vol(
    result: ARCHModelResult,
    horizon: int = 1,
    annualise: bool = True,
) -> float:
    """Produce a h-step-ahead variance forecast from the fitted model.

    For VaR/ES calculation we only need the 1-step forecast
    (tomorrow's conditional volatility).

    Returns
    -------
    Scalar volatility forecast (annualised if annualise=True)
    """
    fc = result.forecast(horizon=horizon, reindex=False)
    # variance forecast → std dev, undo pct scaling
    vol_forecast = np.sqrt(fc.variance.values[-1, 0]) / 100

    if annualise:
        vol_forecast *= np.sqrt(252)

    return float(vol_forecast)


# ─────────────────────────────────────────────
# 4.  Regime-conditional GARCH
# ─────────────────────────────────────────────

def fit_garch_per_regime(
    returns: pd.Series,
    regimes: pd.DataFrame,
    p: int = GARCH_P,
    q: int = GARCH_Q,
) -> dict[int, ARCHModelResult]:
    """Fit a separate GARCH model for each detected regime.

    Why per-regime GARCH?
    ---------------------
    A single GARCH model must average across calm and turbulent
    periods.  Fitting separate models lets each regime have its
    own ω (long-run variance), α (shock sensitivity), and β
    (vol persistence) – producing sharper volatility estimates
    in each state.

    Parameters
    ----------
    returns : full equity log-return series
    regimes : DataFrame with 'regime' column (0=calm, 1=turbulent)

    Returns
    -------
    dict mapping {regime_id: ARCHModelResult}
    """
    results: dict[int, ARCHModelResult] = {}

    aligned = returns.align(regimes["regime"], join="inner")
    ret_aligned, regime_labels = aligned

    for regime_id in sorted(regime_labels.unique()):
        mask = regime_labels == regime_id
        regime_returns = ret_aligned[mask]

        label = "calm" if regime_id == 0 else "turbulent"
        log.info(
            "Fitting GARCH for regime %d (%s) — %d observations",
            regime_id, label, len(regime_returns),
        )

        if len(regime_returns) < 100:
            log.warning(
                "Regime %d has only %d observations – GARCH may be unreliable.",
                regime_id, len(regime_returns),
            )

        results[regime_id] = fit_garch(regime_returns, p=p, q=q, label=label)

    return results


# ─────────────────────────────────────────────
# 5.  Master pipeline
# ─────────────────────────────────────────────

def run_volatility_model(
    dataset: pd.DataFrame,
    regimes: pd.DataFrame,
    per_regime: bool = True,
) -> dict:
    """Full volatility modelling pipeline.

    Steps
    -----
    1. Fit global GARCH on the full return history.
    2. Optionally fit separate GARCH per regime.
    3. Build a combined conditional volatility series:
       - In calm periods → use calm GARCH σ_t
       - In turbulent periods → use turbulent GARCH σ_t
       - Falls back to global GARCH if per_regime=False
    4. Compute one-step-ahead vol forecast.

    Returns
    -------
    dict with keys:
      'global_result'     : ARCHModelResult (full sample)
      'regime_results'    : dict {0: result, 1: result} or None
      'cond_vol'          : pd.Series  (daily, annualised)
      'vol_forecast'      : float (tomorrow, annualised)
    """
    returns = dataset["equity_ret"]

    # ── Global GARCH ───────────────────────────
    log.info("Fitting global GARCH on %d observations …", len(returns))
    global_result = fit_garch(returns, label="global")
    global_vol    = extract_conditional_vol(global_result, annualise=True)

    # ── Per-regime GARCH ───────────────────────
    regime_results: Optional[dict[int, ARCHModelResult]] = None
    combined_vol = global_vol.copy()

    if per_regime:
        regime_results = fit_garch_per_regime(returns, regimes)

        # Stitch regime-specific volatilities into one series
        combined_pieces = []
        for regime_id, res in regime_results.items():
            rv = extract_conditional_vol(res, annualise=True)
            combined_pieces.append(rv)

        # Outer join, then forward-fill any tiny gaps at boundary
        stitched = pd.concat(combined_pieces, axis=0).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]

        # Reindex to the global series index so we have full coverage
        combined_vol = stitched.reindex(global_vol.index).ffill().bfill()
        combined_vol.name = "cond_vol"

    # ── One-step forecast ─────────────────────
    vol_forecast = forecast_vol(global_result, horizon=1, annualise=True)
    log.info("One-step vol forecast (annualised): %.4f (%.2f%%)", vol_forecast, vol_forecast * 100)

    return {
        "global_result":  global_result,
        "regime_results": regime_results,
        "cond_vol":       combined_vol,
        "vol_forecast":   vol_forecast,
    }


# ─────────────────────────────────────────────
# Standalone sanity-check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.data_loader   import build_dataset
    from src.features      import build_features
    from src.regime_model  import run_regime_detection

    ds              = build_dataset()
    _, scaled_f     = build_features(ds)
    _, regimes, _   = run_regime_detection(scaled_f, ds)

    output = run_volatility_model(ds, regimes, per_regime=True)

    print("\n── Conditional vol (tail) ──")
    print(output["cond_vol"].tail(10).round(4))
    print(f"\nOne-step vol forecast: {output['vol_forecast']:.4f} ({output['vol_forecast']*100:.2f}%)")
    print("\n── Global GARCH summary ──")
    print(output["global_result"].summary())