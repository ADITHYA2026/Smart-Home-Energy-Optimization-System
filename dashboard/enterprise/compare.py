import streamlit as st

def compare_card(title, left_label, left_value, right_label, right_value):
    st.markdown(f"""
        <div style="
            display:flex;
            justify-content:space-between;
            padding:20px;
            background:white;
            border-radius:16px;
            box-shadow:0 4px 16px rgba(0,0,0,0.1);
            margin-bottom:20px;
        ">
            <div>
                <h4 style="color:#005f73; margin:0;">{left_label}</h4>
                <h2 style="font-weight:900; margin-top:4px;">{left_value}</h2>
            </div>
            <div style="text-align:right">
                <h4 style="color:#9b2226; margin:0;">{right_label}</h4>
                <h2 style="font-weight:900; margin-top:4px;">{right_value}</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)