import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd


# AUTO DETECT POWER COLUMN
def get_power_column(df):
    if "Global_active_power" in df.columns:
        return "Global_active_power"
    elif "Hybrid" in df.columns:
        return "Hybrid"
    elif "Actual" in df.columns:
        return "Actual"
    else:
        return None

# ============================================================
# 1️⃣  3D HOUR × DAY × POWER SURFACE
# ============================================================
def surface_3d(df):
    col = get_power_column(df)
    if col is None:
        st.error("No valid column found for 3D surface plot.")
        return

    df_s = df.copy()
    df_s["day"] = df_s.index.day
    df_s["hour"] = df_s.index.hour

    pivot = df_s.pivot_table(
        index="day",
        columns="hour",
        values=col,
        aggfunc="mean"
    )

    fig = go.Figure(data=[
        go.Surface(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Teal",
            showscale=True
        )
    ])

    fig.update_layout(
        title=f"3D Energy Surface ({col})",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2️⃣  MONTHLY BLOOM CHART (Circular Heatmap)
# ============================================================
def monthly_bloom(df):
    col = get_power_column(df)
    if col is None:
        st.error("No valid column found for Monthly Bloom chart.")
        return

    df_b = df.copy()
    df_b["month"] = df_b.index.month
    df_b["hour"] = df_b.index.hour

    pivot = df_b.pivot_table(
        index="month",
        columns="hour",
        values=col,
        aggfunc="mean"
    )

    fig = px.imshow(
        pivot, aspect="auto",
        title=f"Monthly Energy Bloom ({col})"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 3️⃣  ANOMALY DETECTION CHART
# ============================================================
def anomaly_chart(df):
    df_a = df.copy()

    # Identify which column to use (Forecast page OR Analytics page)
    if "Global_active_power" in df_a.columns:
        col = "Global_active_power"
    elif "Actual" in df_a.columns:
        col = "Actual"
    else:
        st.error("No valid power column found for anomaly detection.")
        return

    # Rolling mean for anomaly detection
    df_a["rolling"] = df_a[col].rolling(48).mean()

    # Calculate difference for anomaly detection
    df_a["diff"] = abs(df_a[col] - df_a["rolling"])
    threshold = df_a["diff"].mean() + 2 * df_a["diff"].std()

    fig = go.Figure()

    # Normal data
    fig.add_trace(go.Scatter(
        x=df_a.index,
        y=df_a[col],
        mode="lines",
        name="Power Usage",
        line=dict(color="#005f73")
    ))

    # Anomalies
    anomalies = df_a[df_a["diff"] > threshold]

    fig.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies[col],
        mode="markers",
        name="Anomaly",
        marker=dict(size=10, color="red")
    ))

    fig.update_layout(
        title="Anomaly Detection",
        height=500,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)



# ============================================================
# 4️⃣  SEASONAL OVERLAY CHART
# ============================================================
def seasonal_overlay(df):
    col = get_power_column(df)
    if col is None:
        st.error("No valid column found for Seasonal Overlay.")
        return

    df_s = df.copy()
    df_s["month"] = df_s.index.month

    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }

    fig = go.Figure()

    for name, months in seasons.items():
        sub = df_s[df_s["month"].isin(months)]
        fig.add_trace(go.Scatter(
            x=sub.index,
            y=sub[col],
            mode="lines",
            name=name
        ))

    fig.update_layout(
        title=f"Seasonal Overlay ({col})",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5️⃣  APPLIANCE TREND COMPARISON
# ============================================================
def appliance_trend(df):
    fig = go.Figure()

    for col, color in zip(
        ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        ["#94d2bd", "#ee9b00", "#001219"]
    ):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            name=col.replace("_", " ").title(),
            line=dict(color=color)
        ))

    fig.update_layout(
        title="Appliance-wise Power Consumption Trend",
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 6️⃣  PREDICTION CONFIDENCE BAND
# ============================================================
def confidence_band(actual, pred):
    upper = pred + 0.3
    lower = pred - 0.3

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual,
        mode="lines",
        name="Actual",
        line=dict(color="#001219")
    ))

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=upper,
        mode="lines",
        name="Upper CI",
        line=dict(color="#94d2bd", dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=lower,
        mode="lines",
        name="Lower CI",
        line=dict(color="#94d2bd", dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=pred,
        mode="lines",
        name="Prediction",
        line=dict(color="#ee9b00")
    ))

    fig.update_layout(
        title="Prediction Confidence Band",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 7️⃣  WEEKLY SPIKE DETECTOR
# ============================================================
def weekly_spike(df):
    col = get_power_column(df)
    if col is None:
        st.error("No valid column found for Weekly Spike Detector.")
        return

    df_w = df.copy()
    df_w["weekday"] = df_w.index.weekday

    spike = df_w.groupby("weekday")[col].max()

    fig = px.bar(
        spike,
        title=f"Weekly Spike Detection ({col})",
        labels={"value": "Peak Power"}
    )

    st.plotly_chart(fig, use_container_width=True)



# ============================================================
# 8️⃣  MONTHLY BOX PLOT
# ============================================================
def monthly_boxplot(df):
    col = get_power_column(df)
    if col is None:
        st.error("No valid column found for Monthly Box Plot.")
        return

    df_b = df.copy()
    df_b["month"] = df_b.index.month

    fig = px.box(
        df_b,
        x="month",
        y=col,
        title=f"Monthly Power Distribution ({col})",
        color="month"
    )

    st.plotly_chart(fig, use_container_width=True)



# ============================================================
# 9️⃣  COST ESTIMATION CHART
# ============================================================
def cost_chart(df, rate=7.5):
    df_c = df.copy()

    # Determine which column exists
    if "Global_active_power" in df_c.columns:
        col = "Global_active_power"
    elif "Hybrid" in df_c.columns:
        col = "Hybrid"
    elif "Actual" in df_c.columns:
        col = "Actual"
    else:
        st.error("No valid column found for cost estimation.")
        return

    # Cost calculation
    df_c["Cost"] = df_c[col] * rate

    fig = px.line(
        df_c,
        y="Cost",
        title=f"Estimated Cost Trend (INR) — using {col}",
        labels={"Cost": "Cost (₹)"}
    )

    st.plotly_chart(fig, use_container_width=True)