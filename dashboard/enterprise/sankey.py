import streamlit as st
import plotly.graph_objects as go


def energy_flow_sankey(sub1, sub2, sub3):
    labels = ["Total Energy", "Kitchen", "Laundry", "Heating/AC"]

    source = [0, 0, 0]
    target = [1, 2, 3]
    values = [sub1, sub2, sub3]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.4),
            label=labels,
            color=["#005f73", "#94d2bd", "#e9d8a6", "#ee9b00"]
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=["#94d2bd", "#e9d8a6", "#ee9b00"]
        )
    )])

    fig.update_layout(title_text="Energy Consumption Flow", font_size=14)
    st.plotly_chart(fig, use_container_width=True)