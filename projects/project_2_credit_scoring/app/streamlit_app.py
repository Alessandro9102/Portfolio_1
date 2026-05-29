"""
streamlit_app.py
----------------
Step 6: Interactive credit scoring demo.

Features:
  - Borrower profile input via sidebar sliders
  - Default probability gauge
  - Risk tier classification (Low / Medium / High / Very High)
  - SHAP waterfall explanation for the prediction
  - Threshold sensitivity table

Usage (from project_2_credit_scoring/):
    streamlit run app/streamlit_app.py

Requirements:
    models/model.pkl
    models/scaler.pkl
    data/processed/feature_list.txt
    Run all pipeline steps first.
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -- Path setup ---------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent          # app/
PROJECT_ROOT = APP_DIR.parent                           # project_2_credit_scoring/
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

MODELS_DIR    = PROJECT_ROOT / "models"
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH   = PROJECT_ROOT / "config" / "config.yaml"

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Load assets (cached so they load once)
# =============================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = MODELS_DIR / "model.pkl"
    if not path.exists():
        st.error(f"Model not found at {path}. Run train_model.py first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        st.error(f"Scaler not found at {path}. Run feature_engineering.py first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_feature_list() -> list[str]:
    path = DATA_DIR / "feature_list.txt"
    if not path.exists():
        st.error(f"Feature list not found at {path}. Run feature_engineering.py first.")
        st.stop()
    return path.read_text(encoding="utf-8").strip().splitlines()


# =============================================================================
# Feature engineering (mirrors feature_engineering.py exactly)
# =============================================================================

def engineer_single_borrower(raw: dict) -> pd.DataFrame:
    """
    Apply the same transformations as feature_engineering.py to a single row.
    Must stay in sync with the training pipeline.
    """
    df = pd.DataFrame([raw])

    # Delinquency aggregates
    delinq_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["total_delinquencies"]  = df[delinq_cols].sum(axis=1)
    df["max_delinquency_band"] = df[delinq_cols].max(axis=1)
    df["ever_90_days_late"]    = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    # Debt / income features
    safe_income = df["MonthlyIncome"].replace(0, np.nan).fillna(1)
    df["monthly_debt_payment"] = df["DebtRatio"] * safe_income
    df["debt_to_income_safe"]  = df["monthly_debt_payment"] / (safe_income + 1)

    safe_lines = df["NumberOfOpenCreditLinesAndLoans"].replace(0, np.nan).fillna(1)
    df["utilisation_per_line"] = df["RevolvingUtilizationOfUnsecuredLines"] / safe_lines
    df["age_x_utilisation"]    = df["age"] * df["RevolvingUtilizationOfUnsecuredLines"]

    # Interaction terms
    df["RevolvingUtilizationOfUnsecuredLines_x_DebtRatio"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]
    )
    df["NumberOfTimes90DaysLate_x_DebtRatio"] = (
        df["NumberOfTimes90DaysLate"] * df["DebtRatio"]
    )

    # Log transforms
    for col in [
        "MonthlyIncome",
        "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
    ]:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    return df


def prepare_input(raw: dict, feature_cols: list[str], scaler) -> pd.DataFrame:
    """Engineer and scale a single borrower row, aligned to training features."""
    df = engineer_single_borrower(raw)

    # Align columns to training feature list (fill any gap with 0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols]
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=feature_cols,
    )
    return df_scaled


# =============================================================================
# Risk tier helper
# =============================================================================

def risk_tier(prob: float) -> tuple[str, str]:
    """Return (label, hex_color) based on default probability."""
    if prob < 0.10:
        return "Low Risk", "#16A34A"
    elif prob < 0.25:
        return "Medium Risk", "#D97706"
    elif prob < 0.50:
        return "High Risk", "#EA580C"
    else:
        return "Very High Risk", "#DC2626"


# =============================================================================
# SHAP waterfall
# =============================================================================

def shap_waterfall(model, X_input: pd.DataFrame) -> plt.Figure | None:
    try:
        import shap
    except ImportError:
        return None

    xgb_step = None
    if hasattr(model, "estimators_"):
        for name, est in model.named_estimators_.items():
            if "xgb" in name.lower():
                xgb_step = est
                break

    if xgb_step is None:
        return None

    try:
        explainer   = shap.TreeExplainer(xgb_step)
        shap_vals   = explainer(X_input)
        fig, ax     = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(shap_vals[0], show=False, max_display=12)
        plt.title("SHAP Explanation -- Feature Contributions to This Prediction")
        plt.tight_layout()
        return fig
    except Exception:
        return None


# =============================================================================
# Sidebar -- borrower inputs
# =============================================================================

def sidebar_inputs() -> dict:
    st.sidebar.header("Borrower Profile")
    st.sidebar.markdown("Adjust the sliders to model a borrower.")

    raw = {}

    raw["RevolvingUtilizationOfUnsecuredLines"] = st.sidebar.slider(
        "Revolving Utilisation (0-1)",
        min_value=0.0, max_value=1.0, value=0.35, step=0.01,
        help="Total balance on credit cards / credit limits. >0.9 is high risk.",
    )
    raw["age"] = st.sidebar.slider(
        "Age",
        min_value=18, max_value=100, value=45,
    )
    raw["NumberOfTime30-59DaysPastDueNotWorse"] = st.sidebar.slider(
        "30-59 Days Past Due (count)",
        min_value=0, max_value=10, value=0,
    )
    raw["DebtRatio"] = st.sidebar.slider(
        "Debt Ratio",
        min_value=0.0, max_value=10.0, value=0.35, step=0.01,
        help="Monthly debt payments / monthly gross income.",
    )
    raw["MonthlyIncome"] = st.sidebar.number_input(
        "Monthly Income ($)",
        min_value=0, max_value=50000, value=5000, step=100,
    )
    raw["NumberOfOpenCreditLinesAndLoans"] = st.sidebar.slider(
        "Open Credit Lines & Loans",
        min_value=0, max_value=40, value=8,
    )
    raw["NumberOfTimes90DaysLate"] = st.sidebar.slider(
        "Times 90+ Days Late",
        min_value=0, max_value=20, value=0,
    )
    raw["NumberRealEstateLoansOrLines"] = st.sidebar.slider(
        "Real Estate Loans / Lines",
        min_value=0, max_value=10, value=1,
    )
    raw["NumberOfTime60-89DaysPastDueNotWorse"] = st.sidebar.slider(
        "60-89 Days Past Due (count)",
        min_value=0, max_value=10, value=0,
    )
    raw["NumberOfDependents"] = st.sidebar.slider(
        "Number of Dependents",
        min_value=0, max_value=10, value=0,
    )

    return raw


# =============================================================================
# Main app
# =============================================================================

def main() -> None:
    # Header
    st.title("Credit Risk Scoring Dashboard")
    st.markdown(
        "Powered by a **Stacking Ensemble** (Logistic Regression + XGBoost). "
        "Trained on 150,000 US borrowers from the *Give Me Some Credit* dataset."
    )
    st.divider()

    # Load assets
    model        = load_model()
    scaler       = load_scaler()
    feature_cols = load_feature_list()

    # Sidebar inputs
    raw = sidebar_inputs()

    # Prepare input
    X_input = prepare_input(raw, feature_cols, scaler)

    # Predict
    prob      = float(model.predict_proba(X_input)[0, 1])
    tier, color = risk_tier(prob)

    # -- Top metrics row ------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Default Probability (2-year)",
            value=f"{prob:.1%}",
            delta=None,
        )

    with col2:
        st.markdown(
            f"<div style='padding:10px; border-radius:8px; "
            f"background:{color}22; border:2px solid {color};'>"
            f"<span style='font-size:1.1rem; font-weight:600; color:{color};'>"
            f"Risk Tier: {tier}</span></div>",
            unsafe_allow_html=True,
        )

    with col3:
        credit_score = int(850 - prob * 550)   # rough FICO-style mapping
        st.metric(
            label="Indicative Credit Score",
            value=credit_score,
            help="Approximate FICO-style score derived from default probability.",
        )

    st.divider()

    # -- Probability gauge (simple horizontal bar) ----------------------------
    st.subheader("Default Probability Gauge")
    gauge_fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([""], [1.0], color="#E5E7EB", height=0.5)
    ax.barh([""], [prob], color=color, height=0.5)
    ax.axvline(0.10, color="#16A34A", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(0.25, color="#D97706", linestyle="--", lw=1, alpha=0.7)
    ax.axvline(0.50, color="#DC2626", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of Default")
    ax.text(0.05, 0.85, "Low", transform=ax.transAxes,
            color="#16A34A", fontsize=8, va="top")
    ax.text(0.27, 0.85, "Medium", transform=ax.transAxes,
            color="#D97706", fontsize=8, va="top")
    ax.text(0.52, 0.85, "High", transform=ax.transAxes,
            color="#EA580C", fontsize=8, va="top")
    ax.text(0.75, 0.85, "Very High", transform=ax.transAxes,
            color="#DC2626", fontsize=8, va="top")
    ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(gauge_fig)
    plt.close(gauge_fig)

    st.divider()

    # -- SHAP waterfall -------------------------------------------------------
    st.subheader("Why this prediction? (SHAP Explanation)")
    with st.spinner("Computing SHAP values..."):
        fig = shap_waterfall(model, X_input)
    if fig:
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Each bar shows a feature's contribution to pushing the prediction "
            "above (red) or below (blue) the baseline default rate."
        )
    else:
        st.info(
            "SHAP explanation unavailable. "
            "Install shap:  pip install shap"
        )

    st.divider()

    # -- Key borrower stats ---------------------------------------------------
    st.subheader("Borrower Summary")
    summary = pd.DataFrame({
        "Feature": [
            "Revolving Utilisation",
            "Age",
            "Monthly Income",
            "Debt Ratio",
            "Times 90+ Days Late",
            "Total Past Due Events",
            "Open Credit Lines",
        ],
        "Value": [
            f"{raw['RevolvingUtilizationOfUnsecuredLines']:.0%}",
            raw["age"],
            f"${raw['MonthlyIncome']:,}",
            f"{raw['DebtRatio']:.2f}",
            raw["NumberOfTimes90DaysLate"],
            (raw["NumberOfTime30-59DaysPastDueNotWorse"]
             + raw["NumberOfTime60-89DaysPastDueNotWorse"]
             + raw["NumberOfTimes90DaysLate"]),
            raw["NumberOfOpenCreditLinesAndLoans"],
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # -- Footer ---------------------------------------------------------------
    st.divider()
    st.caption(
        "Model: Stacking Ensemble (LR + XGBoost) | "
        "Dataset: Give Me Some Credit (Kaggle) | "
        "For portfolio demonstration purposes only."
    )


if __name__ == "__main__":
    main()