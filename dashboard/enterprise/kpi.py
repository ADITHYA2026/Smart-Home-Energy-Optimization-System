import streamlit as st

def premium_kpi(title, value, icon=None, color="#005f73"):
    st.markdown(f"""
        <div style="
            background: white;
            padding: 18px;
            border-radius: 16px;
            box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
            border-left: 6px solid {color};
            transition: 0.25s;
        " class="kpi-hover">
            <div style="font-size:15px; color:#5e6972; font-weight:600;">
                {icon if icon else ""} {title}
            </div>
            <div style="font-size:32px; color:{color}; font-weight:900;">
                {value}
            </div>
        </div>
    """, unsafe_allow_html=True)