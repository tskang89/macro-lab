"""거시 분석 플랫폼 진입점.

    streamlit run app.py

메뉴는 modules/ 폴더를 자동으로 읽어 만든다(core/registry.py).
새 분석을 추가하려면 modules/<이름>/module.py 에 META 와 render() 를 만들면 된다.
"""
import os

import streamlit as st

from core.registry import discover, navigation

st.set_page_config(page_title="거시 분석 플랫폼", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stSidebarNav"] a span { font-size: 0.95rem; }
.block-container { padding-top: 2.2rem; }
</style>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
st.logo(os.path.join(ROOT, "assets", "logo.svg"), size="large")
with st.sidebar:
    st.caption("산업연관 · 국제산업연관 · 잠재성장률 · (예정) DSGE · HANK")

navigation(discover()).run()
