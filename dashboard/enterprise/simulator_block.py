import streamlit as st

def simulator_block(title, content):
    st.markdown(f"""
        <div class="sim-block">
            <h3 style="font-weight:900; margin-top:0;">{title}</h3>
            <div style="font-size:18px; line-height:1.7;">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)