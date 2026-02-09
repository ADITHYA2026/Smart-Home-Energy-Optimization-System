import sys
import pathlib

# Add root folder to PYTHONPATH
root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import numpy as np

from components.layout import page_header
from components.charts import heatmap
from enterprise.charts import (
    appliance_trend,
    monthly_bloom,
    anomaly_chart,
    surface_3d,
    weekly_spike,
    monthly_boxplot,
    seasonal_overlay,
    cost_chart
)
from utils.config import FEATURE_DATA_PATH


# -------------------------------------------------------
# LOAD FEATURE DATASET
# -------------------------------------------------------
def load_data():
    return pd.read_csv(FEATURE_DATA_PATH, index_col="DateTime", parse_dates=True)


def app():

    # ======================================================
    # ENTERPRISE HEADER
    # ======================================================
    page_header(
        "📈 Energy Analytics Center (Enterprise)",
        "Deep Behavior Analysis • Seasonal Forecast Patterns • Cost Metrics • Appliance Behavior"
    )

    df = load_data()

    # ======================================================
    # TOP KPI BLOCK (4 Cards)
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🔍 High-Level Consumption Summary")

    col1, col2, col3, col4 = st.columns(4)

    daily_avg = df["Global_active_power"].resample("D").mean().mean()
    peak_avg = df["Global_active_power"].between_time("18:00", "22:00").mean()
    night_avg = df["Global_active_power"].between_time("02:00", "05:00").mean()
    overall_max = df["Global_active_power"].max()

    with col1: st.metric("Daily Avg Usage (kW)", round(daily_avg, 3))
    with col2: st.metric("Peak Hour Load (6–10 PM)", round(peak_avg, 3))
    with col3: st.metric("Night Low Load (2–5 AM)", round(night_avg, 3))
    with col4: st.metric("Maximum Load Recorded", round(overall_max, 3))

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # HOURLY HEATMAP
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🔥 Hourly Consumption Heatmap")
    heatmap(df, "Hourly Energy Heatmap")
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # APPLIANCE TREND (Last 3 Days)
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🛠 Appliance Trend Timeline (Last 72 Hours)")
    appliance_trend(df.tail(3 * 288))  # 3 days = 864 records
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # WEEKLY SPIKE DETECTOR
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📅 Weekly Spike Detector")
    weekly_spike(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # SEASONAL OVERLAY
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🍂 Seasonal Load Overlay (Winter → Autumn)")
    seasonal_overlay(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # MONTH × HOUR BLOOM MAP
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🌸 Monthly Bloom Pattern (Month × Hour Visualization)")
    monthly_bloom(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # 3D LOAD SURFACE
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🌐 Daily Load Signature (3D Energy Surface)")
    surface_3d(df.tail(2000))
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # MONTHLY BOX PLOT
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📊 Monthly Load Distribution (Box Plot)")
    monthly_boxplot(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # COST ANALYTICS
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 💰 Cost Trend Analysis (INR)")
    cost_chart(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # ANOMALY DETECTION
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🚨 Anomaly Detection")
    anomaly_chart(df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # AI-GENERATED INSIGHTS (ENTERPRISE)
    # ======================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🧠 Enterprise Analytics Insights")

    insights = []

    if peak_avg > daily_avg * 1.3:
        insights.append("⚡ Evening usage is *significantly* higher than usual — HVAC & kitchen are main contributors.")

    if night_avg < daily_avg * 0.4:
        insights.append("🌙 Night-time efficiency excellent — very low off-peak usage.")

    last_month = df.index[-1].month
    if last_month in [6, 7, 8]:
        insights.append("☀ Summer season detected — expect AC load dominance.")
    elif last_month in [12, 1, 2]:
        insights.append("❄ Winter period — heater spikes expected in evenings.")

    apps = df[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].mean()
    max_app = apps.idxmax()

    if max_app == "Sub_metering_3":
        insights.append("❄ HVAC/AC is currently dominating household power usage.")
    elif max_app == "Sub_metering_2":
        insights.append("🌀 Laundry appliances show consistently high consumption.")
    else:
        insights.append("🍳 Kitchen baseline consumption is higher than other loads.")

    # check for stable pattern
    if df["Global_active_power"].diff().abs().max() < 1.0:
        insights.append("✔ Power consumption stable, no abrupt changes detected.")

    if not insights:
        insights.append("✔ All systems running stable with strong energy efficiency metrics.")

    for tip in insights:
        st.info(tip)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()