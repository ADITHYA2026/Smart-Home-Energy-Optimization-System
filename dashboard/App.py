import streamlit as st
from pathlib import Path

# --------------------------------------------------------------------------------
# APP CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Energy Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# LOAD CSS THEME
# --------------------------------------------------------------------------------
css_path = Path(__file__).parent / "assets/theme.css"
with open(css_path, "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# SIDEBAR BRANDING
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class='sidebar-title'>
            ⚡ Smart Home Energy Optimization
        </div>
        <div class='sidebar-subtitle'>
            Powered by ML + Deep Learning + Hybrid Intelligence
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("Use the sidebar to navigate between pages.")
    st.markdown("---")

# --------------------------------------------------------------------------------
# HOME PAGE MESSAGE (MAIN LANDING)
# --------------------------------------------------------------------------------
st.markdown("""
<div class='header'>
    <h1>Welcome to Smart Energy Optimization Dashboard</h1>
    <h3>Your Intelligent Energy Consumption Assistant</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='body-text'>
This dashboard provides:
<li>Real-time insights into household energy consumption</li>
<li>Deep Learning–powered future predictions</li>
<li>AI-driven recommendations to reduce energy wastage</li>
<li>Hourly, weekly & seasonal analytics</li>
<li>Hybrid model combining ML + DL for better prediction accuracy</li>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.info("Use the Menu on the left to explore each section!")