import streamlit as st

def power_alert(voltage):
    if voltage < 215:
        st.error("⚠️ **Voltage Sag Detected** — Risk to sensitive appliances.")
    elif voltage > 250:
        st.warning("⚠️ **Voltage Swell Detected** — Overvoltage danger.")
    else:
        st.success("✅ Voltage level stable.")