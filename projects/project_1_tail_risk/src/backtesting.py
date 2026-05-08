# ─────────────────────────────────────────────
# src/backtesting.py
#
# Responsibility: evaluate how well our VaR
# models perform by counting violations and
# running formal statistical tests.
#
# Tests implemented
# -----------------
# 1. VaR violation rate  – simple breach count
# 2. Kupiec POF test     – unconditional coverage (Basel standard)
# 3. Traffic-light zones – Basel II/III green/amber/red classification
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_WINDOW, VAR_LEVELS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data class: results for one VaR series
# ─────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Results of backtesting a single VaR series.

    Attributes
    ----------
    confidence      : VaR confidence level (e.g. 0.99)
    method          : model name string
    n_obs           : number of observations tested
    n_violations    : number of days where |loss| > VaR
    violation_rate  : n_violations / n_obs
    expected_rate   : 1 - confidence (the theoretical rate)
    kupiec_stat     : Kupiec LR test statistic
    kupiec_pvalue   : p-value of the Kupiec test
    kupiec_reject   : True if we reject H0 (model is miscalibrated)
    traffic_light   : 'green' / 'amber' / 'red' (Basel zones)
    violations      : pd.Series of boolean violation flags
    """
    confidence:     float
    method:         str
    n_obs:          int
    n_violations:   int
    violation_rate: float
    expected_rate:  float
    kupiec_stat:    float
    kupiec_pvalue:  float
    kupiec_reject:  bool
    traffic_light:  str
    violations:     pd.Series

    def summary(self) -> str:
        return (
            f"[{self.method}] VaR({self.confidence*100:.0f}%)  "
            f"violations={self.n_violations}/{self.n_obs} "
            f"({self.violation_rate*100:.2f}% vs expected {self.expected_rate*100:.2f}%)  "
            f"Kupiec p={self.kupiec_pvalue:.4f}  "
            f"reject={self.kupiec_reject}  "
            f"traffic={self.traffic_light.upper()}"
        )


# ─────────────────────────────────────────────
# 1.  Violation detection
# ─────────────────────────────────────────────

def detect_violations(
    returns: pd.Series,
    var_series: pd.Series,
) -> pd.Series:
    """Return a boolean series: True where |loss| exceeds VaR.

    A VaR violation (breach) occurs when the realised loss on
    day t+1 exceeds the VaR predicted at the end of day t.

    We use next-day returns aligned with today's VaR forecast
    (the VaR is a forward-looking quantity).

    Parameters
    ----------
    returns    : log return series
    var_series : VaR series (positive values = expected max loss)

    Returns
    -------
    pd.Series of bool (True = violation)
    """
    # Align on common dates
    ret, var = returns.align(var_series, join="inner")

    # Loss exceeds VaR → violation
    violations = (-ret) > var
    violations.name = "violation"
    return violations


# ─────────────────────────────────────────────
# 2.  Kupiec Proportion of Failures (POF) test
# ─────────────────────────────────────────────

def kupiec_test(
    violations: pd.Series,
    confidence: float,
) -> tuple[float, float, bool]:
    """Kupiec (1995) Proportion of Failures test.

    H0: the empirical violation rate equals the theoretical rate (1-p).

    Test statistic (likelihood ratio):
    -----------------------------------
    LR_POF = -2 · ln[ p^(T-N) · (1-p)^N / π^(T-N) · (1-π)^N ]

    where:
      p  = theoretical failure rate = 1 - confidence
      π  = empirical failure rate   = N / T
      N  = number of violations
      T  = total observations

    Under H0: LR_POF ~ χ²(1)

    Reject H0 (model is bad) if p-value < 0.05.

    Why Kupiec?
    -----------
    Basel II/III requires banks to backtest their VaR models.
    Kupiec is the standard unconditional coverage test – it only
    checks whether the *number* of violations is correct, not
    their *timing* (that requires the Christoffersen test).

    Returns
    -------
    (lr_stat, p_value, reject_h0)
    """
    T = len(violations)
    N = violations.sum()
    p = 1 - confidence        # theoretical failure rate
    pi_hat = N / T            # empirical failure rate

    if N == 0 or N == T:
        # Edge case: LR is undefined; treat as extreme failure
        log.warning("Kupiec: N=%d, T=%d – degenerate case, returning stat=inf", N, T)
        return float("inf"), 0.0, True

    # Log-likelihood ratio
    try:
        lr_stat = -2 * (
            N * np.log(p / pi_hat) + (T - N) * np.log((1 - p) / (1 - pi_hat))
        )
    except ZeroDivisionError:
        lr_stat = float("inf")

    p_value   = float(1 - chi2.cdf(lr_stat, df=1))
    reject_h0 = p_value < 0.05

    return float(lr_stat), p_value, reject_h0


# ─────────────────────────────────────────────
# 3.  Basel traffic-light zones
# ─────────────────────────────────────────────

def basel_traffic_light(
    n_violations: int,
    n_obs: int,
    confidence: float = 0.99,
) -> str:
    """Classify model quality using Basel II/III traffic-light framework.

    Based on a 250-day backtest at 99% confidence (Basel standard):
      Green  : 0–4 violations  → model is acceptable
      Amber  : 5–9 violations  → model may be flawed, requires review
      Red    : 10+ violations  → model is rejected; capital add-on required

    We scale the thresholds proportionally for non-standard windows.

    Returns
    -------
    'green', 'amber', or 'red'
    """
    # Scale Basel's 250-day thresholds to actual window length
    scale = n_obs / 250
    green_max = int(np.ceil(4  * scale))
    amber_max = int(np.ceil(9  * scale))

    if n_violations <= green_max:
        return "green"
    elif n_violations <= amber_max:
        return "amber"
    else:
        return "red"


# ─────────────────────────────────────────────
# 4.  Full backtest for one VaR series
# ─────────────────────────────────────────────

def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float,
    method: str = "model",
) -> BacktestResult:
    """Run the full backtest suite on a single VaR series.

    Parameters
    ----------
    returns    : equity log returns
    var_series : daily VaR estimates (positive losses)
    confidence : confidence level of the VaR
    method     : label for reporting

    Returns
    -------
    BacktestResult dataclass
    """
    violations = detect_violations(returns, var_series)
    n_obs      = len(violations)
    n_viol     = int(violations.sum())
    viol_rate  = n_viol / n_obs
    exp_rate   = 1 - confidence

    lr_stat, p_val, reject = kupiec_test(violations, confidence)
    traffic = basel_traffic_light(n_viol, n_obs, confidence)

    result = BacktestResult(
        confidence     = confidence,
        method         = method,
        n_obs          = n_obs,
        n_violations   = n_viol,
        violation_rate = viol_rate,
        expected_rate  = exp_rate,
        kupiec_stat    = lr_stat,
        kupiec_pvalue  = p_val,
        kupiec_reject  = reject,
        traffic_light  = traffic,
        violations     = violations,
    )
    log.info(result.summary())
    return result


# ─────────────────────────────────────────────
# 5.  Compare multiple models
# ─────────────────────────────────────────────

def compare_models(
    returns: pd.Series,
    var_series_dict: dict[str, pd.Series],
    confidence: float,
) -> pd.DataFrame:
    """Backtest multiple VaR series and return a comparison DataFrame.

    Parameters
    ----------
    returns         : equity log returns
    var_series_dict : {model_name: var_series}
    confidence      : VaR confidence level

    Returns
    -------
    DataFrame with one row per model showing all backtest metrics
    """
    rows = []
    for name, var_s in var_series_dict.items():
        res = backtest_var(returns, var_s, confidence, method=name)
        rows.append({
            "Model":            res.method,
            "Observations":     res.n_obs,
            "Violations":       res.n_violations,
            "Violation rate":   f"{res.violation_rate*100:.2f}%",
            "Expected rate":    f"{res.expected_rate*100:.2f}%",
            "Kupiec stat":      round(res.kupiec_stat, 3),
            "Kupiec p-value":   round(res.kupiec_pvalue, 4),
            "Reject H0":        res.kupiec_reject,
            "Basel zone":       res.traffic_light.upper(),
        })

    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────
# 6.  Master pipeline
# ─────────────────────────────────────────────

def run_backtesting(
    dataset: pd.DataFrame,
    var_series_normal: dict[float, pd.Series],
    var_series_hist:   dict[float, pd.Series],
    var_levels: list[float] | None = None,
) -> dict:
    """Full backtesting pipeline across methods and confidence levels.

    Returns
    -------
    dict with keys:
      'results'       : dict {(level, method): BacktestResult}
      'summary_table' : dict {level: comparison DataFrame}
    """
    var_levels = var_levels or VAR_LEVELS
    returns    = dataset["equity_ret"]

    all_results:   dict[tuple, BacktestResult] = {}
    summary_tables: dict[float, pd.DataFrame] = {}

    for level in var_levels:
        models_at_level = {
            "Normal+GARCH": var_series_normal[level],
            "Historical":   var_series_hist[level],
        }

        df = compare_models(returns, models_at_level, confidence=level)
        summary_tables[level] = df

        for name, var_s in models_at_level.items():
            res = backtest_var(returns, var_s, level, method=name)
            all_results[(level, name)] = res

        log.info("\nBacktest summary at %.0f%%:\n%s", level * 100, df.to_string())

    return {
        "results":       all_results,
        "summary_table": summary_tables,
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
    from src.risk_metrics     import run_risk_metrics

    ds            = build_dataset()
    _, scaled_f   = build_features(ds)
    _, regimes, _ = run_regime_detection(scaled_f, ds)
    vol_out       = run_volatility_model(ds, regimes)
    tail_out      = run_tail_model(ds)
    metrics       = run_risk_metrics(
        ds, vol_out["cond_vol"], vol_out["vol_forecast"], tail_out["gpd_fit"]
    )

    bt = run_backtesting(
        dataset           = ds,
        var_series_normal = metrics["var_series_normal"],
        var_series_hist   = metrics["var_series_hist"],
    )

    print("\n── Backtest summary (99%) ──")
    print(bt["summary_table"][0.99].to_string())
    print("\n── Backtest summary (95%) ──")
    print(bt["summary_table"][0.95].to_string())