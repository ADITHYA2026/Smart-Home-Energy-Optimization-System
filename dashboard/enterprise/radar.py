import plotly.graph_objects as go
import streamlit as st


def appliance_radar(values_dict):
    labels = list(values_dict.keys())
    values = list(values_dict.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color='#0a9396'),
        name='Appliance Usage'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)