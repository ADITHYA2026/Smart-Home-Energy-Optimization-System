import streamlit as st
import plotly.graph_objects as go


def energy_gauge(label, value, min_val=0, max_val=10):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"<b>{label}</b>", "font": {"size": 20}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#0a9396"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e9ecef",
            "steps": [
                {"range": [0, max_val * 0.4], "color": "#d4edda"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#ffe5b4"},
                {"range": [max_val * 0.7, max_val], "color": "#ffcccc"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)