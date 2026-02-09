import sys
import pathlib

root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.config import ROOT
from components.layout import page_header


# ========================================================
# SIMULATION ENGINE
# ========================================================
def simulate_energy(consumption_level, season, tariff, appliance_intensity):

    base_profiles = {
        "LOW":     np.random.uniform(0.3, 1.2, 24),
        "MEDIUM":  np.random.uniform(0.8, 2.5, 24),
        "HIGH":    np.random.uniform(2.0, 4.2, 24),
    }

    profile = base_profiles[consumption_level].copy()

    # Season effect
    if season == "Summer":
        profile += np.random.uniform(1.0, 2.0, 24)
    elif season == "Winter":
        profile += np.random.uniform(0.5, 1.5, 24)

    # Tariff-based behavior
    if tariff == "High Tariff":
        profile *= np.random.uniform(0.9, 1.1, 24)
    elif tariff == "Low Tariff":
        profile *= np.random.uniform(0.7, 0.9, 24)

    # Appliance intensity
    if appliance_intensity == "High":
        profile *= 1.35
    elif appliance_intensity == "Low":
        profile *= 0.75

    # Smoothen curve
    smooth = np.convolve(profile, np.ones(3)/3, mode="same")

    return smooth


# ========================================================
# COST CALCULATION
# ========================================================
def calculate_cost(usage):
    return usage * 8.2  # ₹ per kWh


# ========================================================
# AI OPTIMIZATION LOGIC
# ========================================================
def generate_ai_advice(consumption_level, season, intensity):

    advice = []

    # ---- Load level advice ----
    if consumption_level == "HIGH":
        advice.append("⚠ High consumption — reduce parallel heavy appliances.")
    elif consumption_level == "MEDIUM":
        advice.append("ℹ Moderate usage — shift heavy tasks to off-peak hours.")
    else:
        advice.append("✔ Low usage — system efficiently balanced.")

    # ---- Season advice ----
    if season == "Summer":
        advice.append("☀ AC-heavy period — maintain thermostat at 24–26°C.")
    elif season == "Winter":
        advice.append("❄ Heater usage may spike — ensure insulation.")

    # ---- Intensity advice ----
    if intensity == "High":
        advice.append("🔥 Several appliances running — avoid stacking tasks.")
    else:
        advice.append("🟢 Normal appliance intensity.")

    # ---- General tips ----
    advice.append("💡 Use timers for laundry/dishwashers.")
    advice.append("🔌 Unplug idle appliances to reduce phantom load.")
    advice.append("🏠 Keep AC filters clean for efficiency.")

    return advice


# ========================================================
# STREAMLIT PAGE
# ========================================================
def app():

    # ====================================================
    # PAGE HEADER
    # ====================================================
    page_header(
        "🔮 AI-Driven Scenario Simulator (Enterprise)",
        "Simulate Household Conditions • See Energy Impact • Get Optimization Advice"
    )

    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## ⚙ Configure Scenario Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        consumption_level = st.selectbox("Consumption Type", ["LOW", "MEDIUM", "HIGH"])

    with col2:
        season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])

    with col3:
        tariff = st.selectbox("Tariff Type", ["Normal", "High Tariff", "Low Tariff"])

    appliance_intensity = st.radio(
        "Appliance Usage Intensity",
        ["Low", "Normal", "High"],
        horizontal=True
    )

    duration = st.slider("Duration (Hours)", 1, 24, 12)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ====================================================
    # RUN SIMULATION
    # ====================================================
    if st.button("▶ Run Simulation", use_container_width=True):

        usage = simulate_energy(consumption_level, season, tariff, appliance_intensity)
        usage_duration = usage[:duration]

        cost = calculate_cost(usage_duration)
        total_cost = cost.sum()

        # ====================================================
        # SECTION — ENERGY USAGE TREND
        # ====================================================
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## 📈 Simulated Energy Usage Trend")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=usage_duration,
            x=list(range(duration)),
            mode="lines+markers",
            line=dict(color="#0a9396", width=3),
            marker=dict(size=6),
        ))

        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Energy (kWh)",
            template="plotly_white",
            height=380
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # APPLIANCE INTENSITY BAR
        # ====================================================
        # ====================================================

        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## 🔥 Appliance Intensity Factor")

        intensity_raw = {"Low": 0.5, "Normal": 1.0, "High": 1.5}
        value = intensity_raw[appliance_intensity] / 1.5  # normalize to 0–1

        st.progress(value)
        st.markdown(f"**Intensity Level:** {appliance_intensity}")


        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # COST PROJECTION
        # ====================================================
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## 💰 Cost Projection")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=cost,
            x=list(range(duration)),
            marker_color="#ee9b00",
        ))

        fig2.update_layout(
            xaxis_title="Hour",
            yaxis_title="Cost (₹)",
            template="plotly_white",
            height=300
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.metric("Total Projected Cost (₹)", round(total_cost, 2))
        st.metric("Avg Cost Per Hour (₹)", round(total_cost / duration, 2))

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # ENERGY RISK GAUGE
        # ====================================================
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## ⚡ Energy Risk Gauge")

        risk = np.mean(usage_duration) * 22
        risk = min(max(risk, 0), 100)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d00000" if risk > 70 else "#ff8800" if risk > 40 else "#2e7d32"},
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # OPTIMIZATION ADVICE
        # ====================================================
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## 🧠 AI Optimization Advice")

        for tip in generate_ai_advice(consumption_level, season, appliance_intensity):
            st.info(tip)

        st.markdown("</div>", unsafe_allow_html=True)

        # ====================================================
        # DOWNLOAD SECTION
        # ====================================================
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("## 📥 Download Scenario Data")

        scenario_df = pd.DataFrame({
            "Hour": np.arange(duration),
            "Energy (kWh)": usage_duration,
            "Cost (₹)": cost
        })

        st.download_button(
            "Download Simulation CSV",
            scenario_df.to_csv(index=False).encode("utf-8"),
            file_name="scenario_simulation.csv",
        )

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()