"""앱 공통 껍데기: 페이지 설정·공통 CSS·로고·메뉴.

진입 파일마다 run() 을 부른다.
    app.py            전체 플랫폼(모든 메뉴)
    app_potential.py  잠재성장률 전용
    app_neutral.py    중립금리 전용
    app_io.py         산업연관표 전용(국내·기본부문·국제)
Streamlit Community Cloud 에서는 같은 저장소로 앱을 여러 개 만들고
'Main file path' 만 위 파일 중 하나로 다르게 준다.
"""
from __future__ import annotations

import os

import streamlit as st

from core.registry import discover, navigation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
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
"""


def run(slugs: list[str] | None = None, *, page_title: str = "거시 분석 플랫폼",
        page_icon: str = ":material/insights:", caption: str | None = None) -> None:
    """slugs 가 None 이면 모든 메뉴, 주어지면 그 메뉴만(주어진 순서대로) 싣는다.

    메뉴가 하나뿐이면 왼쪽 메뉴를 숨기고 사이드바를 접어 그 화면만 보이게 한다.
    """
    single = slugs is not None and len(slugs) == 1
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide",
                       initial_sidebar_state="collapsed" if single else "expanded")

    mods = discover()
    if slugs is not None:
        by_slug = {m["slug"]: m for m in mods}
        missing = [s for s in slugs if s not in by_slug]
        if missing:
            raise KeyError(f"없는 메뉴: {missing}")
        mods = [dict(by_slug[s], default=(i == 0)) for i, s in enumerate(slugs)]

    st.markdown(CSS, unsafe_allow_html=True)
    st.logo(os.path.join(ROOT, "assets", "logo.svg"), size="large")
    if caption and not single:
        with st.sidebar:
            st.caption(caption)

    navigation(mods, position="hidden" if single else "sidebar").run()
