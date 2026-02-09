import sys
import pathlib

# Add root folder to PYTHONPATH
root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from components.layout import page_header
from enterprise.charts import (
    confidence_band,
    anomaly_chart,
    cost_chart
)
from utils.config import ROOT


# -------------------------------------------------------
# LOAD HYBRID RESULTS
# -------------------------------------------------------
def load_hybrid():
    df = pd.read_csv(ROOT / "data/processed/hybrid_results.csv")
    df.index = pd.RangeIndex(len(df))
    return df


def app():

    # -----------------------------------------
    # PAGE HEADER
    # -----------------------------------------
    page_header(
        "📊 Forecast & Hybrid Prediction Engine (Enterprise Edition)",
        "Deep Learning + ML Fusion • Confidence Bands • Anomaly Detection • Cost Projection"
    )

    df = load_hybrid()
    actual, hybrid = df["Actual"], df["Hybrid"]

    # -------------------------------------------------------
    # METRICS — ENTERPRISE KPI BOX
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📌 Model Accuracy Overview")

    mae = np.mean(abs(actual - hybrid))
    rmse = np.sqrt(np.mean((actual - hybrid) ** 2))
    r2 = 1 - np.sum((actual - hybrid) ** 2) / np.sum((actual - np.mean(actual)) ** 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE (Mean Abs Error)", round(mae, 4))

    with col2:
        st.metric("RMSE", round(rmse, 4))

    with col3:
        st.metric("R² Score (Accuracy)", f"{r2:.3f}")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # ACTUAL VS HYBRID TREND (500 POINTS)
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🔮 Actual vs Hybrid Prediction (Last 500 Points)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[-500:], y=actual[-500:],
        mode="lines", name="Actual",
        line=dict(color="#001219", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index[-500:], y=hybrid[-500:],
        mode="lines", name="Hybrid",
        line=dict(color="#ee9b00", width=2)
    ))

    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # CONFIDENCE BAND
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📉 Prediction Confidence Band")
    confidence_band(actual.tail(500), hybrid.tail(500))
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # MODEL CONTRIBUTION (Hybrid Weights)
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🧠 Hybrid Model Contribution Breakdown")

    fig_pie = px.pie(
        names=["LightGBM", "XGBoost", "CNN-LSTM"],
        values=[40, 30, 30],
        hole=0.55,
        title="Model Weightage in Hybrid Prediction",
        color=["LightGBM", "XGBoost", "CNN-LSTM"],
        color_discrete_map={
            "LightGBM": "#0a9396",
            "XGBoost": "#ee9b00",
            "CNN-LSTM": "#81B452"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # ANOMALY DETECTION
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🚨 Anomaly Detection Based on Forecast Deviations")
    anomaly_chart(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # VOLATILITY INDICATOR
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🌐 Prediction Volatility Indicator")

    df["volatility"] = df["Hybrid"].diff().abs()
    vol = df["volatility"].tail(200).mean()

    st.info(
        f"**Volatility (Last 200 points): {vol:.4f}** — "
        f"{'Stable' if vol < 0.15 else '⚠ Fluctuating — major appliance switching detected'}"
    )

    st.line_chart(df["volatility"].tail(400))
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # COST FORECAST
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 💰 Cost Projection Trend (Based on Hybrid Prediction)")
    cost_chart(df.tail(500))
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # NEXT 24 HOURS SIMULATION
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🔮 Simulated Next 24-Hour Projection")

    future = []
    last_value = hybrid.iloc[-1]

    for _ in range(24):
        last_value = last_value * (1 + np.random.uniform(-0.05, 0.05))
        future.append(last_value)

    future_df = pd.DataFrame({
        "Hour Ahead": list(range(1, 25)),
        "Predicted Usage (kW)": np.round(future, 3)
    })

    st.dataframe(future_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # INSIGHTS
    # -------------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🧠 Forecast Intelligence Insights")

    insights = []

    if mae < 0.5:
        insights.append("✔ Hybrid model showing excellent low-error performance.")

    if r2 > 0.90:
        insights.append("📈 Very strong correlation between real and predicted usage.")

    if vol > 0.2:
        insights.append("⚠ High volatility — appliance switching pattern detected.")

    if hybrid.tail(50).mean() > actual.tail(50).mean() * 1.15:
        insights.append("📌 Hybrid forecast indicates possible future peak usage period.")

    if not insights:
        insights.append("✔ Model prediction stable and healthy.")

    for i in insights:
        st.info(i)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()