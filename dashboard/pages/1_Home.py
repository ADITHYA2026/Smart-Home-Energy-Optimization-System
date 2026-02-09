import sys
import pathlib

# Add root folder to PYTHONPATH
root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import numpy as np

from components.cards import kpi_card
from components.layout import page_header
from enterprise.charts import (
    appliance_trend,
    monthly_bloom,
    surface_3d,
    weekly_spike,
    cost_chart
)
from utils.config import FEATURE_DATA_PATH


# -------------------------------------------------------
# LOAD PROCESSED FEATURE DATASET
# -------------------------------------------------------
def load_data():
    return pd.read_csv(FEATURE_DATA_PATH, index_col="DateTime", parse_dates=True)


def app():
    # -----------------------------------------
    # PAGE HEADER
    # -----------------------------------------
    page_header(
        "🏠 Smart Energy Home Dashboard (Enterprise Edition)",
        "Live Monitoring • AI Insights • High-Fidelity Visualization"
    )

    df = load_data()
    latest = df.iloc[-1]

    # ---------------------------------------------------
    # TOP KPI SECTION (WITH SECTION BOX)
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## ⚡ Key System Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Current Power Usage (kW)", round(latest["Global_active_power"], 2))

    with col2:
        kpi_card("Voltage Level (V)", round(latest["Voltage"], 2))

    with col3:
        kpi_card("Current Intensity (A)", round(latest["Global_intensity"], 2))

    with col4:
        daily_avg = df["Global_active_power"].tail(288).mean()
        kpi_card("Daily Avg Load (kW)", round(daily_avg, 2))

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # ALERT SYSTEM SECTION
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🚨 System Alerts")

    alerts = []

    if latest["Voltage"] < 225:
        alerts.append("⚠️ Low voltage detected – avoid AC/heater to prevent damage.")

    if latest["Voltage"] > 245:
        alerts.append("⚠️ High voltage spike – stabilizer recommended.")

    if latest["Sub_metering_3"] > 25:
        alerts.append("🔥 HVAC/Heater consumption unusually high right now.")

    if not alerts:
        st.success("✔ No active alerts — System operating normally.")
    else:
        for a in alerts:
            st.error(a)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # 24-HOUR POWER TREND
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📈 24-Hour Power Usage Trend")

    last_24 = df["Global_active_power"].tail(288)
    st.line_chart(last_24)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # APPLIANCE TREND
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🛠 Appliance Consumption Timeline (Last 24 Hours)")
    appliance_trend(df.tail(288))
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # 3D SURFACE ENERGY SIGNATURE
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🌐 Daily Load Signature (3D Energy Surface)")
    surface_3d(df.tail(2000))  # ~7 days window
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # MONTHLY BLOOM HEATMAP
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🌸 Monthly Bloom (Hour × Month Heat Visualization)")
    monthly_bloom(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # WEEKLY SPIKE DETECTOR
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📅 Weekly Spike Detector")
    weekly_spike(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # COST TREND
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 💰 Estimated Cost Trend (INR)")
    cost_chart(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------
    # INTELLIGENT SUMMARY INSIGHTS
    # ---------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🧠 Intelligent Summary Insights")

    insights = []

    if latest["Global_active_power"] > daily_avg * 1.4:
        insights.append("⚡ Current load significantly higher than daily average.")

    if latest["Sub_metering_3"] == df["Sub_metering_3"].max():
        insights.append("❄ HVAC/Heater is currently the top energy consumer.")

    if df["Global_active_power"].tail(288).mean() < df["Global_active_power"].mean():
        insights.append("📉 Last 24-hour average lower than historical — Good efficiency today.")

    if latest["Sub_metering_1"] > latest["Sub_metering_2"] and latest["Sub_metering_1"] > latest["Sub_metering_3"]:
        insights.append("🍳 Kitchen appliances dominating — optimize stove/oven usage.")

    if not insights:
        insights.append("✔ System stable. No unusual patterns detected.")

    for i in insights:
        st.info(i)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()