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
/* 레이아웃 */
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1360px; }
h1 { letter-spacing: -0.025em; margin-bottom: 0.1rem !important; }
h2, h3 { letter-spacing: -0.015em; }
[data-testid="stCaptionContainer"] { line-height: 1.55; }

/* 지표 카드: 얇은 테두리 + 옅은 면 */
[data-testid="stMetric"] {
  border: 1px solid rgba(128,128,128,0.20); border-radius: 12px;
  padding: 0.85rem 1rem 0.8rem; background: rgba(128,128,128,0.035);
  height: 100%;
}
[data-testid="stMetricLabel"] p { font-size: 0.82rem !important; opacity: 0.78; font-weight: 500;
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.35; }
[data-testid="stMetricLabel"] > div { overflow: visible !important; }
[data-testid="stMetricValue"] { letter-spacing: -0.02em; }
[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  line-height: 1.2; }

/* 차트 카드 */
[data-testid="stPlotlyChart"] {
  border: 1px solid rgba(128,128,128,0.18); border-radius: 12px; padding: 0.6rem 0.75rem 0.3rem;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid rgba(128,128,128,0.22); }
.stTabs [data-baseweb="tab"] { padding: 0.55rem 0.1rem; }
.stTabs [data-baseweb="tab"] p { font-size: 0.97rem; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { height: 2.5px; }

/* 펼침 상자·테두리 상자 */
[data-testid="stExpander"] details { border-radius: 10px; }
[data-testid="stExpander"] summary p { font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }

/* 사이드바 */
[data-testid="stSidebarNav"] a span { font-size: 0.95rem; }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] {
  font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.6; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
st.logo(os.path.join(ROOT, "assets", "logo.svg"), size="large")
with st.sidebar:
    st.caption("산업연관 · 국제산업연관 · 잠재성장률 · (예정) DSGE · HANK")

navigation(discover()).run()
