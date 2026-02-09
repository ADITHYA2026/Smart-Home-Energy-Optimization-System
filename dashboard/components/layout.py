import streamlit as st


def page_header(title, subtitle=None):
    st.markdown(f"<h1 style='color:#003566;font-weight:800;'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='color:#0a9396;font-weight:500;'>{subtitle}</h3>", unsafe_allow_html=True)
    st.markdown("---")