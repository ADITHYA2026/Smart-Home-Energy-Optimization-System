import streamlit as st

def recommendation_block(priority, title, points, color):
    st.markdown(f"""
        <div style="
            padding:22px;
            border-radius:16px;
            background: linear-gradient(135deg, {color[0]}, {color[1]});
            box-shadow: 0 6px 20px rgba(0,0,0,0.18);
            margin-bottom: 20px;
            color:white;
        ">
            <h3 style="margin-top:0; font-weight:900;">{priority} PRIORITY</h3>
            <h4 style="margin-top:-10px; margin-bottom:12px;">{title}</h4>
            <ul>
                {''.join([f"<li style='margin:8px 0; font-size:17px;'>{p}</li>" for p in points])}
            </ul>
        </div>
    """, unsafe_allow_html=True)