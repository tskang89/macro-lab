"""준비 중인 모듈의 공통 안내 화면."""
import streamlit as st

from core.ui import footer


def planned_page(title: str, body: str, plan: list[str]):
    st.title(title)
    st.info("이 분석은 준비 중입니다. 플랫폼 구조(메뉴·결과 형식·엑셀 출력)는 이미 갖춰져 있어 "
            "계산 엔진만 붙이면 바로 열립니다.")
    st.markdown(body)
    st.markdown("**계획**")
    for p in plan:
        st.markdown(f"- {p}")
    footer()
