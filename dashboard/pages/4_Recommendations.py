import sys
import pathlib

root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.layout import page_header
from components.cards import recommendation_card
from utils.config import ROOT


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
def load_recommendations():
    df = pd.read_csv(ROOT / "data/processed/final_recommendations.csv")
    return df


def app():

    # =================================================================
    # HEADER
    # =================================================================
    page_header(
        "🧠 AI Smart Recommendations (Enterprise Intelligence Suite)",
        "Priority Analysis • Appliance Risk • Cost Optimization • Stability Metrics"
    )

    df = load_recommendations()
    latest = df.iloc[-1]

    # =================================================================
    # SECTION 1 — LATEST INSIGHT
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🔎 Latest AI Insight")

    recommendation_card(latest["Priority"], latest["Recommendation"])
    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 2 — COST IMPACT PANEL
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 💰 Estimated Cost Impact (Based on Latest Prediction)")

    pred_usage = latest["Hybrid"]
    hourly_cost = pred_usage * 8.2
    daily_cost = hourly_cost * 24

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Load (kW)", round(pred_usage, 3))

    with col2:
        st.metric("Hourly Cost (₹)", round(hourly_cost, 2))

    with col3:
        st.metric("Projected Daily Cost (₹)", round(daily_cost, 2))

    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 3 — ENERGY RISK SCORE METER
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## ⚡ Energy Risk Score (AI Weighted)")

    score = 0
    score += 40 if pred_usage > 6 else 20 if pred_usage > 4 else 5
    score += 15 if latest["Sub_metering_3"] > 25 else 0
    score += 15 if latest["Sub_metering_2"] > 18 else 0
    score += 10 if latest["Voltage"] < 225 else 0

    if score > 70:
        status = "HIGH RISK"
        color = "red"
    elif score > 40:
        status = "MEDIUM RISK"
        color = "orange"
    else:
        status = "LOW RISK"
        color = "green"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': f"Energy Risk Score — {status}", 'font': {'size': 22}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#d4edda"},
                {"range": [40, 70], "color": "#ffe5b4"},
                {"range": [70, 100], "color": "#ffcccc"},
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 4 — PRIORITY DISTRIBUTION
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📊 Priority Distribution Overview")

    priority_counts = df["Priority"].value_counts()

    fig = px.pie(
        priority_counts,
        names=priority_counts.index,
        values=priority_counts.values,
        hole=0.55,
        color=priority_counts.index,
        title="High / Medium / Low Recommendation Frequency",
        color_discrete_map={
            "HIGH": "#d00000",
            "MEDIUM": "#ff8800",
            "LOW": "#2e7d32"
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 5 — PRIORITY TIMELINE
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📉 Priority Timeline Trend")

    timeline_df = df.copy()
    timeline_df["Priority_num"] = timeline_df["Priority"].map({"LOW": 1, "MEDIUM": 2, "HIGH": 3})

    st.line_chart(timeline_df["Priority_num"].tail(500))
    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 6 — APPLIANCE ALERT GRID
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🛠 Appliance Alert Status (Latest Record)")

    cols = st.columns(3)
    alert_data = [
        ("Kitchen (Sub1)", latest["Sub_metering_1"], 12),
        ("Laundry (Sub2)", latest["Sub_metering_2"], 18),
        ("Heating/AC (Sub3)", latest["Sub_metering_3"], 25)
    ]

    for col, (name, val, threshold) in zip(cols, alert_data):
        with col:
            if val > threshold:
                st.error(f"⚠ {name}: High — {val} Wh (Threshold {threshold})")
            else:
                st.success(f"✔ {name}: Normal — {val} Wh")

    st.markdown("</div>", unsafe_allow_html=True)

    
    # =================================================================
    # SECTION 7 — APPLIANCE CONTRIBUTION PIE
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 🍽 Appliance Load Contribution Breakdown")

    # Ensure safe values
    s1 = latest["Sub_metering_1"] if latest["Sub_metering_1"] > 0 else 0.01
    s2 = latest["Sub_metering_2"] if latest["Sub_metering_2"] > 0 else 0.01
    s3 = latest["Sub_metering_3"] if latest["Sub_metering_3"] > 0 else 0.01

    # Debug print (remove later)
    # st.write(s1, s2, s3)

    fig_cont = px.pie(
        names=["Kitchen", "Laundry", "Heating/AC"],
        values=[s1, s2, s3],
        color=["Kitchen", "Laundry", "Heating/AC"],
        color_discrete_map={
            "Kitchen": "#0a9396",
            "Laundry": "#ee9b00",
            "Heating/AC": "#81B452"
        }
    )

    st.plotly_chart(fig_cont, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 8 — RECOMMENDATION LOG TABLE
    # =================================================================
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("## 📋 Recommendation Log (Recent 150 Entries)")

    st.dataframe(df.tail(150), use_container_width=True, height=450)

    st.download_button(
        label="📥 Download Full Recommendation Report",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smart_energy_recommendations.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()