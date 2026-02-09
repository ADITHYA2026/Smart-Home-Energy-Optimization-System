import streamlit as st


def kpi_card(title, value):
    st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
        </div>
    """, unsafe_allow_html=True)


def recommendation_card(priority, text):
    cls = (
        "reco-high" if priority == "HIGH" else 
        "reco-medium" if priority == "MEDIUM" else
        "reco-low"
    )

    st.markdown(f"""
        <div class='reco-card {cls}'>
            <strong>{priority} PRIORITY</strong><br>
            {text}
        </div>
    """, unsafe_allow_html=True)