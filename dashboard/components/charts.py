import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def line_chart(df, title):
    fig = px.line(df, x=df.index, y=df.columns, title=title)
    st.plotly_chart(fig, use_container_width=True)


def comparison_chart(df):
    fig = go.Figure()

    for col in ["Actual", "Hybrid"]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col
        ))

    fig.update_layout(
        title="Actual vs Hybrid Prediction",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def heatmap(df, title):
    # Heatmap using actual real power usage
    if "Global_active_power" not in df.columns:
        st.error("Column 'Global_active_power' not found in DataFrame.")
        return
    
    df_h = df.copy()
    df_h["date"] = df_h.index.date
    df_h["hour"] = df_h.index.hour

    pivot_df = df_h.pivot_table(
        index="date",
        columns="hour",
        values="Global_active_power",
        aggfunc="mean"
    )

    fig = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale="Teal",
        title=title,
        labels=dict(color="kW")
    )

    st.plotly_chart(fig, use_container_width=True)
