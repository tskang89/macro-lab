"""기본부문(380부문) 산업연관 분석 — 한국은행 실측표 기본부문 투입산출표.

화면·계산은 '산업연관표 분석'(대·중분류)과 같은 함수를 쓰고 부문 분류만 기본부문으로 바꾼다.
기본부문표는 실측표 연도(2020년)에만 있으므로 최신 연도 분석은 기존 메뉴로 안내한다.
"""
from __future__ import annotations

import streamlit as st

from core.registry import PAGES
from core.ui import footer, page_header
from engines.io import domestic as D
from modules.b_io_domestic.module import demand_tab, price_tab, structure_tab, supply_tab

META = {"title": "기본부문 산업연관 분석", "group": "산업연관 분석", "order": 15,
        "icon": ":material/grid_view:", "status": "ready", "slug": "io-basic"}


@st.cache_resource(show_spinner="기본부문 380부문 표와 역행렬을 불러오는 중…")
def _table(year: str) -> D.KrIO:
    t = D.load_table(D.BASIC, year)
    _ = (t.Ad, t.e, t.w, t.Fd, t.dom_share)        # 역행렬 관련 계수를 한 번만 계산해 둠
    return t


def render():
    page_header("기본부문 산업연관 분석",
                "한국은행 산업연관표 기본부문(380부문)으로 세부 품목의 수요·비용·공급 충격 파급효과를 계산합니다.")
    years = D.available_years(D.BASIC)
    if not years:
        st.error("기본부문 자료가 없습니다. ECOS 에서 실측표 기본부문 파일을 받아 build/build_io_basic.py 를 실행하세요.")
        return
    with st.container(border=True):
        st.markdown(
            "**언제 이 메뉴를 쓰나요?** 기본부문표는 5년마다 작성되는 **실측표 연도(현재 2020년)에만** 있습니다. "
            "반도체(집적회로)·전기승용차·LNG 처럼 **세부 품목**의 파급효과가 필요할 때 이 메뉴를 쓰세요. "
            "**최신 연도**(2021–2024년 연장표)의 산업 구조로 분석하려면 대·중분류 표를 쓰는 ‘산업연관표 분석’ 메뉴가 맞습니다.")
        if "io-domestic" in PAGES:
            st.page_link(PAGES["io-domestic"], label="산업연관표 분석(2015–2024년, 33·83부문)으로 가기",
                         icon=":material/arrow_forward:")
    c1, c3 = st.columns([1, 4])
    year = c1.selectbox("자료 연도(실측표)", years[::-1], index=0, key="basic_year")
    t = _table(year)
    c3.markdown(f"<div style='padding-top:1.9rem;color:#666'>{year}년 실측표 · 기본부문 {t.n}부문 "
                f"(→ 소분류 165 · 중분류 83 · 대분류 33) · 총산출 {t.x.sum() / 1e6:,.0f}조원 · "
                f"총부가가치 {t.total_va / 1e6:,.0f}조원</div>", unsafe_allow_html=True)
    st.caption("부문 칸에서 글자를 쳐서 찾을 수 있습니다(예: ‘집적회로’, ‘전기승용차’, ‘천연가스’). "
               "2020년 분류에는 메모리반도체가 따로 없고 ‘3102 집적회로’에 들어 있습니다.")
    tabs = st.tabs(["수요 파급", "가격 파급", "공급 차질", "연관 구조"])
    with tabs[0]:
        demand_tab(t, D.BASIC)
    with tabs[1]:
        price_tab(t, D.BASIC)
    with tabs[2]:
        supply_tab(t, D.BASIC)
    with tabs[3]:
        structure_tab(t, D.BASIC)
    footer()
