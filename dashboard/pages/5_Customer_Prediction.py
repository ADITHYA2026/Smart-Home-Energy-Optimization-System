import sys
import pathlib

root_path = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import tensorflow as tf

from components.layout import page_header
from components.cards import recommendation_card
from utils.config import ROOT


# -------------------------------------------------------
# LOAD MODELS (cached)
# -------------------------------------------------------
@st.cache_resource
def load_models():
    custom_objects = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "MeanSquaredError": tf.keras.losses.MeanSquaredError(),
    }

    lgb = joblib.load(ROOT / "models/lightgbm_model.pkl")
    xgb = joblib.load(ROOT / "models/xgboost_model.pkl")

    lstm = tf.keras.models.load_model(
        ROOT / "models/lstm_model.h5",
        custom_objects=custom_objects,
        compile=False
    )

    cnn_lstm = tf.keras.models.load_model(
        ROOT / "models/cnn_lstm_model.h5",
        custom_objects=custom_objects,
        compile=False
    )

    return lgb, xgb, lstm, cnn_lstm


# -------------------------------------------------------
# GENERATE SEQUENCE FOR LSTM/CNN
# -------------------------------------------------------
def generate_sequence(input_vector):
    seq = np.tile(input_vector, (60, 1))
    noise = np.random.normal(0, 0.01, seq.shape)
    return seq + noise


# -------------------------------------------------------
# BUILD INPUT FEATURE VECTOR
# -------------------------------------------------------
def build_full_feature_vector(voltage, current, sub1, sub2, sub3, hour, weekday, season):

    global_power = (sub1 + sub2 + sub3) / 10 + (current * 0.12)

    day = 15
    month = 5

    is_weekend = 1 if weekday >= 5 else 0

    lag1 = global_power * np.random.uniform(0.9, 1.1)
    lag2 = global_power * np.random.uniform(0.9, 1.1)
    lag3 = global_power * np.random.uniform(0.9, 1.1)

    roll1 = (lag1 + global_power) / 2
    roll3 = (lag1 + lag2 + lag3) / 3
    roll6 = roll3

    total = sub1 + sub2 + sub3 + 1e-6
    r1, r2, r3 = sub1 / total, sub2 / total, sub3 / total

    return np.array([
        global_power, voltage, current, sub1, sub2, sub3,
        hour, day, month, weekday, is_weekend, season,
        lag1, lag2, lag3, roll1, roll3, roll6, r1, r2, r3
    ], dtype=float)


# -------------------------------------------------------
# HYBRID PREDICTOR
# -------------------------------------------------------
def hybrid_predict(features, lgb, xgb, lstm, cnn):
    f = features.reshape(1, -1)

    lgb_p = lgb.predict(f)[0]
    xgb_p = xgb.predict(f)[0]

    seq = generate_sequence(features)
    seq = seq.reshape(1, 60, seq.shape[1])

    lstm_p = lstm.predict(seq).flatten()[0]
    cnn_p = cnn.predict(seq).flatten()[0]

    final = 0.4 * lgb_p + 0.3 * xgb_p + 0.3 * cnn_p

    return final, lgb_p, xgb_p, lstm_p, cnn_p


# -------------------------------------------------------
# RECOMMENDATION LOGIC
# -------------------------------------------------------
def generate_custom_recommendation(pred, sub1, sub2, sub3, voltage, hour, season):
    rec = []
    priority = "LOW"

    if pred > 6:
        priority = "HIGH"
        rec.append("Very high predicted load — Reduce AC/heater and avoid laundry right now.")
    elif pred > 4:
        priority = "MEDIUM"
        rec.append("Moderate load — Prefer scheduling non-essential tasks later.")
    else:
        rec.append("Low load — Good time for washing/cooking/heavy usage.")

    if sub3 > 25:
        rec.append("HVAC usage is high — Adjust thermostat by +2°C.")

    if sub2 > 18:
        rec.append("Laundry load high — Schedule after 10 PM.")

    if sub1 > 12:
        rec.append("Kitchen load spike — Avoid parallel usage of oven + induction.")

    if voltage < 225:
        rec.append("Low voltage — Avoid sensitive devices.")

    if 18 <= hour <= 22:
        rec.append("Peak hour — Try reducing heavy appliances.")

    if season == 3:
        rec.append("Summer — AC consumption expected high.")

    rec.append("Switch to LED & efficient appliances.")

    return priority, " | ".join(rec)


# -------------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------------
def app():
    page_header(
        "🧪 Custom Prediction (Enterprise AI Lab)",
        "Real-time Hybrid Prediction • Multi-Model Analysis • Cost Impact • Smart Recommendation"
    )

    lgb, xgb, lstm, cnn = load_models()

    # -------------------------------------------------------
    # INPUT PANEL
    # -------------------------------------------------------
    st.markdown("### 🔧 Enter Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        voltage = st.number_input("Voltage (V)", 200.0, 260.0, 235.0)
        current = st.number_input("Intensity (A)", 0.0, 40.0, 12.0)
        sub1 = st.number_input("Kitchen Load (SubMeter-1)", 0.0, 50.0, 6.0)

    with col2:
        sub2 = st.number_input("Laundry Load (SubMeter-2)", 0.0, 50.0, 3.0)
        sub3 = st.number_input("HVAC Load (SubMeter-3)", 0.0, 50.0, 15.0)
        hour = st.slider("Hour of Day", 0, 23, 12)

    weekday = st.selectbox("Weekday", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    season = st.selectbox("Season", [1,2,3,4], format_func=lambda x: ["Winter","Spring","Summer","Autumn"][x - 1])


    # Build Input Vector
    features = build_full_feature_vector(voltage, current, sub1, sub2, sub3, hour, weekday, season)

    st.markdown("---")

    if st.button("Predict"):
        # -------------------------------------------------------
        # RUN HYBRID PREDICTION
        # -------------------------------------------------------
        pred, lgb_p, xgb_p, lstm_p, cnn_p = hybrid_predict(features, lgb, xgb, lstm, cnn)

        st.success(f"### 🔮 Predicted Power Usage: **{pred:.3f} kW**")

        # -------------------------------------------------------
        # Model Contribution Chart
        # -------------------------------------------------------
        st.markdown("### 📊 Model Contribution Breakdown")

        fig = go.Figure(data=[go.Bar(
            x=["LightGBM", "XGBoost", "LSTM", "CNN-LSTM"],
            y=[lgb_p, xgb_p, lstm_p, cnn_p],
            marker_color=["#0a9396", "#ee9b00", "#005f73", "#001219"]
        )])

        fig.update_layout(
            title="Model Output Distribution",
            yaxis_title="Prediction (kW)",
            xaxis_title="Model"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # Cost Projection
        # -------------------------------------------------------
        st.markdown("### 💰 Cost Projection")

        hourly_cost = pred * 8.2
        daily_cost = hourly_cost * 24

        col1, col2 = st.columns(2)
        col1.metric("Hourly Cost (₹)", round(hourly_cost, 2))
        col2.metric("Estimated Daily Cost (₹)", round(daily_cost, 2))

        # -------------------------------------------------------
        # AI Recommendation
        # -------------------------------------------------------
        priority, rec_text = generate_custom_recommendation(pred, sub1, sub2, sub3, voltage, hour, season)

        st.markdown("### 🧠 AI Recommendation")
        recommendation_card(priority, rec_text)

        # -------------------------------------------------------
        # Efficiency Score
        # -------------------------------------------------------
        st.markdown("### 🟩 Energy Efficiency Score")

        efficiency = max(0, 100 - (pred * 12))  # simple scoring
        st.progress(int(efficiency))
        st.info(f"Efficiency Score: **{efficiency:.1f}/100**")

        # -------------------------------------------------------
        # Confidence Band (Enterprise Grade)
        # -------------------------------------------------------
        st.markdown("### 📉 Confidence Band Estimate")

        upper = pred + 0.4
        lower = pred - 0.4

        # Create wide X-axis to avoid compression
        x_range = [-1, 1]   # provides width
        y_pred = [pred, pred]
        y_upper = [upper, upper]
        y_lower = [lower, lower]

        fig2 = go.Figure()

        # Prediction line
        fig2.add_trace(go.Scatter(
            x=x_range, y=y_pred,
            mode="lines",
            name="Prediction",
            line=dict(color="#ee9b00", width=4)
        ))

        # Confidence band fill
        fig2.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=y_upper + y_lower[::-1],
            fill="toself",
            fillcolor="rgba(148,210,189,0.4)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Band"
        ))

        fig2.update_layout(
            title="Prediction Confidence Range",
            xaxis_title="Range",
            yaxis_title="kW",
            height=300,
            template="plotly_white",
        )

        st.plotly_chart(fig2, use_container_width=True)



if __name__ == "__main__":
    app()