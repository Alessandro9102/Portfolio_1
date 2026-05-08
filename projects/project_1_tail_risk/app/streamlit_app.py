# ─────────────────────────────────────────────
# app/streamlit_app.py
#
# Interactive dashboard for the Tail Risk
# Regime-Switching Model.
#
# Run with:  streamlit run app/streamlit_app.py
# ─────────────────────────────────────────────

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── path setup so src/ imports work ───────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import VAR_LEVELS
from src.data_loader      import build_dataset
from src.features         import build_features
from src.regime_model     import run_regime_detection
from src.volatility_model import run_volatility_model
from src.tail_model       import run_tail_model, qqplot_data
from src.risk_metrics     import run_risk_metrics
from src.backtesting      import run_backtesting


# ═══════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title  = "Tail Risk Monitor",
    page_icon   = "📉",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
  }

  /* Alert / info boxes */
  .info-box {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #8b949e;
    margin: 0.5rem 0;
  }
  .warn-box {
    background: #161b22;
    border-left: 3px solid #d29922;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #8b949e;
    margin: 0.5rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# Colour palette (Plotly traces)
# ═══════════════════════════════════════════════

CLR = {
    "price":     "#58a6ff",
    "calm":      "#3fb950",
    "turbulent": "#f85149",
    "vol":       "#d2a8ff",
    "var_norm":  "#ffa657",
    "var_hist":  "#79c0ff",
    "return":    "#8b949e",
    "violation": "#f85149",
    "gpd":       "#d2a8ff",
    "empirical": "#ffa657",
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#c9d1d9",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor = CLR["bg"],
    plot_bgcolor  = CLR["bg"],
    font          = dict(family="IBM Plex Mono, monospace", color=CLR["text"], size=11),
    margin        = dict(l=50, r=20, t=40, b=40),
    xaxis         = dict(gridcolor=CLR["grid"], zeroline=False, showline=False),
    yaxis         = dict(gridcolor=CLR["grid"], zeroline=False, showline=False),
    legend        = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hovermode     = "x unified",
)


# ═══════════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════════

@st.cache_data(show_spinner="Downloading market data …")
def load_data(ticker: str, force: bool = False) -> pd.DataFrame:
    return build_dataset(force_refresh=force)


@st.cache_data(show_spinner="Running regime detection …")
def load_regimes(_dataset: pd.DataFrame):
    _, scaled_f = build_features(_dataset)
    _, regimes, stats = run_regime_detection(scaled_f, _dataset)
    return regimes, stats


@st.cache_data(show_spinner="Fitting GARCH …")
def load_volatility(_dataset: pd.DataFrame, _regimes: pd.DataFrame):
    return run_volatility_model(_dataset, _regimes, per_regime=True)


@st.cache_data(show_spinner="Fitting EVT tail model …")
def load_tail(_dataset: pd.DataFrame):
    return run_tail_model(_dataset)


@st.cache_data(show_spinner="Computing risk metrics …")
def load_metrics(_dataset, _cond_vol, vol_forecast, _gpd_fit):
    return run_risk_metrics(_dataset, _cond_vol, vol_forecast, _gpd_fit)


@st.cache_data(show_spinner="Running backtests …")
def load_backtest(_dataset, _var_normal, _var_hist):
    return run_backtesting(_dataset, _var_normal, _var_hist)


# ═══════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📉 Tail Risk Monitor")
    st.markdown("---")

    st.markdown("### Asset")
    asset = st.selectbox(
        "Equity index",
        options=["SPY", "QQQ", "IWM"],
        index=0,
        help="ETF to analyse. Data sourced from Yahoo Finance (2000–present).",
    )

    st.markdown("### Confidence level")
    confidence = st.select_slider(
        "VaR / ES confidence",
        options=[0.90, 0.95, 0.99],
        value=0.99,
        format_func=lambda x: f"{int(x*100)}%",
    )

    st.markdown("### Options")
    force_refresh = st.checkbox("Force data refresh", value=False)

    st.markdown("---")
    st.markdown(
        "<div class='info-box'>"
        "Model detects high-risk regimes via a Hidden Markov Model, "
        "estimates conditional volatility with GARCH(1,1), and adjusts "
        "tail risk dynamically using Extreme Value Theory (GPD)."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.75rem; color:#484f58; margin-top:1rem;'>"
        "For educational purposes only. Not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════
# Run pipeline
# ═══════════════════════════════════════════════

with st.spinner("Running full pipeline …"):
    dataset  = load_data(asset, force=force_refresh)
    regimes, reg_stats = load_regimes(dataset)
    vol_out  = load_volatility(dataset, regimes)
    tail_out = load_tail(dataset)
    metrics  = load_metrics(
        dataset,
        vol_out["cond_vol"],
        vol_out["vol_forecast"],
        tail_out["gpd_fit"],
    )
    bt = load_backtest(
        dataset,
        metrics["var_series_normal"],
        metrics["var_series_hist"],
    )

returns  = dataset["equity_ret"]
prices   = dataset["equity"]
cond_vol = vol_out["cond_vol"]
gpd_fit  = tail_out["gpd_fit"]

# Align everything to common index
common_idx = (
    returns.index
    .intersection(regimes.index)
    .intersection(cond_vol.index)
)
returns_a  = returns.reindex(common_idx)
prices_a   = prices.reindex(common_idx)
regimes_a  = regimes.reindex(common_idx)
cond_vol_a = cond_vol.reindex(common_idx)
var_norm_a = metrics["var_series_normal"][confidence].reindex(common_idx)
var_hist_a = metrics["var_series_hist"][confidence].reindex(common_idx)


# ═══════════════════════════════════════════════
# Header KPI strip
# ═══════════════════════════════════════════════

st.markdown(f"## {asset} — Dynamic Tail Risk Dashboard")
st.markdown(
    f"<span style='font-family: IBM Plex Mono; font-size:0.8rem; color:#484f58;'>"
    f"{common_idx[0].date()} → {common_idx[-1].date()} · "
    f"{len(common_idx):,} trading days"
    f"</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Current regime
current_regime     = int(regimes_a["regime"].iloc[-1])
current_regime_lbl = "🔴 TURBULENT" if current_regime == 1 else "🟢 CALM"
current_vol        = float(cond_vol_a.iloc[-1])
current_var        = float(var_norm_a.dropna().iloc[-1])
current_prob_turb  = float(regimes_a["prob_turbulent"].iloc[-1])

# EVT metrics at chosen confidence
var_evt_val = tail_out["var_evt"][confidence]
es_evt_val  = tail_out["es_evt"][confidence]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Regime",       current_regime_lbl)
col2.metric("Cond. Vol (ann.)",     f"{current_vol*100:.1f}%")
col3.metric(f"VaR ({int(confidence*100)}%) GARCH", f"{current_var*100:.2f}%")
col4.metric(f"VaR ({int(confidence*100)}%) EVT",   f"{var_evt_val*100:.2f}%")
col5.metric(f"ES  ({int(confidence*100)}%) EVT",   f"{es_evt_val*100:.2f}%")

st.markdown("---")


# ═══════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 REGIMES",
    "🌊 VOLATILITY",
    "📉 VAR & ES",
    "🔬 TAIL FIT",
    "✅ BACKTEST",
])


# ──────────────────────────────────────────────
# TAB 1: Regimes
# ──────────────────────────────────────────────

with tab1:
    st.markdown("### Market regime detection (Hidden Markov Model)")
    st.markdown(
        "<div class='info-box'>"
        "The HMM identifies two latent market states from realised volatility, "
        "absolute returns, skewness, and VIX. Turbulent periods (red) coincide "
        "with the 2008 crisis, 2020 COVID crash, and 2022 rate shock."
        "</div>",
        unsafe_allow_html=True,
    )

    # Price chart with regime shading
    fig = go.Figure()

    # Regime background bands
    calm_mask = regimes_a["regime"] == 0
    turb_mask = regimes_a["regime"] == 1

    for mask, colour, name in [
        (calm_mask,  "rgba(63,185,80,0.08)",  "Calm"),
        (turb_mask,  "rgba(248,81,73,0.12)", "Turbulent"),
    ]:
        in_block = False
        start    = None
        for i, (idx, val) in enumerate(mask.items()):
            if val and not in_block:
                in_block = True
                start    = idx
            elif not val and in_block:
                in_block = False
                fig.add_vrect(
                    x0=start, x1=idx,
                    fillcolor=colour, layer="below", line_width=0,
                    annotation_text="" ,
                )
        if in_block:
            fig.add_vrect(
                x0=start, x1=mask.index[-1],
                fillcolor=colour, layer="below", line_width=0,
            )

    fig.add_trace(go.Scatter(
        x=prices_a.index, y=prices_a,
        mode="lines", name=asset,
        line=dict(color=CLR["price"], width=1.2),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Price history with detected regimes",
        yaxis_title="Price (USD)",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Regime probability
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=regimes_a.index,
        y=regimes_a["prob_turbulent"],
        mode="lines",
        name="P(turbulent)",
        line=dict(color=CLR["turbulent"], width=1),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.15)",
    ))
    fig2.add_hline(y=0.5, line_dash="dot", line_color=CLR["grid"], line_width=1)
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title="Posterior probability of turbulent regime",
        yaxis_title="P(turbulent)",
        yaxis_range=[0, 1],
        height=200,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Regime stats table
    st.markdown("#### Regime statistics")
    st.dataframe(
        reg_stats.style.format("{:.4f}"),
        use_container_width=True,
    )


# ──────────────────────────────────────────────
# TAB 2: Volatility
# ──────────────────────────────────────────────

with tab2:
    st.markdown("### Conditional volatility (GARCH)")
    st.markdown(
        "<div class='info-box'>"
        "GARCH(1,1) with Student-t innovations captures volatility clustering. "
        "Separate models are fitted per regime, producing sharper estimates "
        "during calm vs. turbulent periods."
        "</div>",
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(
        x=cond_vol_a.index, y=cond_vol_a * 100,
        mode="lines", name="GARCH σ_t (ann.%)",
        line=dict(color=CLR["vol"], width=1.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=returns_a.index, y=returns_a * 100,
        mode="lines", name="Daily return %",
        line=dict(color=CLR["return"], width=0.7),
        opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="GARCH conditional volatility vs realised returns",
        height=460,
    )
    fig.update_yaxes(title_text="Ann. vol (%)", row=1, col=1, gridcolor=CLR["grid"])
    fig.update_yaxes(title_text="Return (%)",   row=2, col=1, gridcolor=CLR["grid"])
    st.plotly_chart(fig, use_container_width=True)

    # GARCH parameter table
    st.markdown("#### GARCH model parameters")
    garch_rows = []
    label_map = {0: "Calm regime", 1: "Turbulent regime", "global": "Global (full sample)"}
    for key, res in (vol_out["regime_results"] or {}).items():
        p = res.params
        garch_rows.append({
            "Model":       label_map.get(key, str(key)),
            "ω (omega)":   round(p["omega"], 6),
            "α (alpha)":   round(p["alpha[1]"], 4),
            "β (beta)":    round(p["beta[1]"], 4),
            "Persistence": round(p["alpha[1]"] + p["beta[1]"], 4),
            "AIC":         round(res.aic, 1),
        })
    gp = vol_out["global_result"].params
    garch_rows.append({
        "Model":       "Global (full sample)",
        "ω (omega)":   round(gp["omega"], 6),
        "α (alpha)":   round(gp["alpha[1]"], 4),
        "β (beta)":    round(gp["beta[1]"], 4),
        "Persistence": round(gp["alpha[1]"] + gp["beta[1]"], 4),
        "AIC":         round(vol_out["global_result"].aic, 1),
    })
    st.dataframe(pd.DataFrame(garch_rows).set_index("Model"), use_container_width=True)


# ──────────────────────────────────────────────
# TAB 3: VaR & ES
# ──────────────────────────────────────────────

with tab3:
    st.markdown(f"### Dynamic VaR & Expected Shortfall at {int(confidence*100)}%")
    st.markdown(
        "<div class='info-box'>"
        "The dynamic VaR (orange) uses today's GARCH volatility forecast so it "
        "rises during turbulent regimes and shrinks during calm ones. Bars below "
        "the VaR line are violation days — the model is tested against these."
        "</div>",
        unsafe_allow_html=True,
    )

    # Clip to last 5 years for readability
    cutoff     = common_idx[-1] - pd.DateOffset(years=5)
    idx_5y     = common_idx[common_idx >= cutoff]
    ret_5y     = returns_a.reindex(idx_5y)
    var_norm_5y = var_norm_a.reindex(idx_5y)
    var_hist_5y = var_hist_a.reindex(idx_5y)
    violations_5y = (-ret_5y) > var_norm_5y

    fig = make_subplots(rows=1, cols=1)

    # Returns bars
    fig.add_trace(go.Bar(
        x=ret_5y.index, y=ret_5y * 100,
        name="Daily return",
        marker_color=[
            CLR["violation"] if v else CLR["return"]
            for v in violations_5y
        ],
        marker_line_width=0,
        opacity=0.7,
    ))

    # Dynamic VaR lines (negative = loss threshold)
    fig.add_trace(go.Scatter(
        x=var_norm_5y.index, y=-var_norm_5y * 100,
        mode="lines", name=f"VaR {int(confidence*100)}% GARCH",
        line=dict(color=CLR["var_norm"], width=1.5, dash="solid"),
    ))
    fig.add_trace(go.Scatter(
        x=var_hist_5y.index, y=-var_hist_5y * 100,
        mode="lines", name=f"VaR {int(confidence*100)}% Historical",
        line=dict(color=CLR["var_hist"], width=1, dash="dot"),
    ))

    # EVT static line
    fig.add_hline(
        y=-var_evt_val * 100,
        line_dash="dash",
        line_color=CLR["gpd"],
        line_width=1.2,
        annotation_text=f"EVT VaR {int(confidence*100)}%",
        annotation_position="bottom right",
        annotation_font_color=CLR["gpd"],
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Daily returns vs dynamic VaR — last 5 years (red bars = violations)",
        yaxis_title="Return / VaR (%)",
        height=420,
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk table
    st.markdown("#### Risk metric comparison (point-in-time)")
    risk_df = metrics["risk_table"].copy()
    risk_df_pct = risk_df * 100
    st.dataframe(
        risk_df_pct.style.format("{:.3f}%").highlight_max(
            axis=0, color="#3d1f1f"
        ),
        use_container_width=True,
    )
    st.markdown(
        "<div class='warn-box'>"
        "EVT VaR and ES are higher than the Normal model — the fat tail of "
        "equity returns means extreme losses occur more often than Gaussian "
        "models predict. Use EVT estimates for regulatory capital calculations."
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# TAB 4: Tail fit (GPD)
# ──────────────────────────────────────────────

with tab4:
    st.markdown("### Extreme Value Theory — GPD tail fit")
    st.markdown(
        "<div class='info-box'>"
        "The Peaks-Over-Threshold method fits a Generalised Pareto Distribution "
        "to the worst 5% of loss days. The shape parameter ξ > 0 confirms a "
        "heavy-tailed distribution — consistent with equity crash dynamics."
        "</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        # Loss histogram + GPD density overlay
        losses = tail_out["losses"].values
        thresh = gpd_fit.threshold

        x_range = np.linspace(thresh, losses.max() * 1.05, 300)
        from scipy.stats import genpareto
        gpd_pdf = genpareto.pdf(
            x_range - thresh,
            c    = gpd_fit.shape,
            loc  = gpd_fit.loc,
            scale= gpd_fit.scale,
        )
        # Scale density to match histogram (rough)
        bin_width = (losses.max() - thresh) / 40
        scale_factor = len(losses[losses > thresh]) * bin_width

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=losses * 100,
            nbinsx=60,
            name="Loss distribution",
            marker_color=CLR["return"],
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=x_range * 100,
            y=gpd_pdf * scale_factor,
            mode="lines",
            name="GPD fit",
            line=dict(color=CLR["gpd"], width=2),
        ))
        fig.add_vline(
            x=thresh * 100,
            line_dash="dot", line_color=CLR["var_norm"],
            annotation_text=f"Threshold u={thresh*100:.2f}%",
            annotation_font_color=CLR["var_norm"],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Loss histogram + GPD density",
            xaxis_title="Daily loss (%)",
            yaxis_title="Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # QQ plot
        theo, emp = qqplot_data(gpd_fit)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theo * 100, y=emp * 100,
            mode="markers",
            name="Empirical vs theoretical",
            marker=dict(color=CLR["empirical"], size=5, opacity=0.7),
        ))
        # 45-degree reference line
        diag_max = max(theo.max(), emp.max()) * 100 * 1.05
        fig.add_trace(go.Scatter(
            x=[0, diag_max], y=[0, diag_max],
            mode="lines",
            name="Perfect fit",
            line=dict(color=CLR["gpd"], dash="dot", width=1),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="QQ plot — GPD fit quality",
            xaxis_title="Theoretical quantile (%)",
            yaxis_title="Empirical quantile (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # GPD parameter table
    st.markdown("#### GPD fit parameters")
    gpd_table = pd.DataFrame([{
        "Threshold u":      f"{gpd_fit.threshold*100:.3f}%",
        "Shape ξ (xi)":     round(gpd_fit.shape, 4),
        "Scale σ (sigma)":  round(gpd_fit.scale, 6),
        "Exceedances N":    gpd_fit.n_exceed,
        "Total obs T":      gpd_fit.n_total,
        "Tail fraction":    f"{gpd_fit.tail_fraction*100:.2f}%",
    }])
    st.dataframe(gpd_table, use_container_width=True, hide_index=True)

    interpretation = (
        f"ξ = {gpd_fit.shape:.4f} > 0 → **heavy tail (Fréchet domain)**. "
        f"This means tail quantiles grow as a power law — each additional "
        f"standard deviation of loss is more likely than the normal model predicts."
        if gpd_fit.shape > 0
        else f"ξ = {gpd_fit.shape:.4f} ≈ 0 → **exponential tail (Gumbel domain)**. "
        f"Tail losses decay exponentially."
    )
    st.markdown(
        f"<div class='info-box'>{interpretation}</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# TAB 5: Backtesting
# ──────────────────────────────────────────────

with tab5:
    st.markdown("### Backtesting — VaR model validation")
    st.markdown(
        "<div class='info-box'>"
        "Kupiec Proportion-of-Failures test checks whether the number of VaR "
        "breaches matches the theoretical rate. Basel II/III classifies models "
        "into Green / Amber / Red zones based on breach counts over 250 days."
        "</div>",
        unsafe_allow_html=True,
    )

    # Summary tables
    for level in [0.99, 0.95]:
        if level not in bt["summary_table"]:
            continue
        df = bt["summary_table"][level].copy()
        st.markdown(f"#### {int(level*100)}% VaR backtest")

        def colour_zone(val):
            mapping = {"GREEN": "color: #3fb950", "AMBER": "color: #d29922", "RED": "color: #f85149"}
            return mapping.get(val, "")

        styled = df.style.map(colour_zone, subset=["Basel zone"])
        st.dataframe(styled, use_container_width=True)

    # Violation timeline at chosen confidence
    st.markdown(f"#### Violation timeline — VaR {int(confidence*100)}% GARCH")
    viol_key = (confidence, "Normal+GARCH")
    if viol_key in bt["results"]:
        br = bt["results"][viol_key]
        viol_series = br.violations.astype(int)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=viol_series.index,
            y=viol_series.rolling(63).mean() * 100,
            mode="lines",
            name="63-day rolling violation rate %",
            line=dict(color=CLR["violation"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.15)",
        ))
        expected_pct = (1 - confidence) * 100
        fig.add_hline(
            y=expected_pct,
            line_dash="dot",
            line_color=CLR["var_norm"],
            annotation_text=f"Expected {expected_pct:.1f}%",
            annotation_font_color=CLR["var_norm"],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Rolling 63-day VaR violation rate vs expected {expected_pct:.1f}%",
            yaxis_title="Violation rate (%)",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary verdict
        zone  = br.traffic_light.upper()
        color = {"GREEN": "#3fb950", "AMBER": "#d29922", "RED": "#f85149"}.get(zone, "#fff")
        verdict = {
            "GREEN": "Model passes Basel validation. Violation rate is within acceptable bounds.",
            "AMBER": "Model requires review. Breach count exceeds Basel green-zone threshold.",
            "RED":   "Model rejected under Basel III. Capital add-on would be required.",
        }.get(zone, "")
        st.markdown(
            f"<div style='border-left: 3px solid {color}; background:#161b22; "
            f"padding: 0.75rem 1rem; border-radius:4px; margin:0.5rem 0;'>"
            f"<strong style='color:{color}'>{zone} ZONE</strong> — {verdict}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='font-size:0.75rem; color:#484f58; text-align:center;'>"
    "Tail Risk Regime-Switching Model · HMM + GARCH + EVT · "
    "Built with Python, arch, hmmlearn, scipy, Streamlit · For portfolio demonstration only"
    "</div>",
    unsafe_allow_html=True,
)