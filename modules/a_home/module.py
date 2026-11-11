import streamlit as st

from core.registry import PAGES
from core.ui import footer

META = {"title": "처음 화면", "group": "", "order": 0, "icon": ":material/home:", "status": "ready",
        "slug": "home", "default": True}


def _go(slug: str, label: str):
    if slug in PAGES:
        st.page_link(PAGES[slug], label=label, icon=":material/arrow_forward:")


def render():
    st.title("거시 분석 플랫폼")
    st.markdown(
        "산업연관표와 거시 모형으로 **시나리오를 넣고 파급효과를 계산해 엑셀로 내려받는** 분석 도구입니다. "
        "왼쪽 메뉴에서 분석을 고르세요.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("산업연관표 분석")
            st.markdown(
                "한국은행 산업연관표(2015–2024년, 33·83부문)\n\n"
                "- **수요 파급**: 소비·투자·수출 충격 → 생산·부가가치·수입·취업\n"
                "- **가격 파급**: 환율·유가·수입물가·임금 → 산업별 가격, 소비자물가\n"
                "- **공급 차질**: 특정 산업 생산 차질 → 전방·후방 파급\n"
                "- **연관 구조**: 유발계수, 영향력·감응도계수, 가설추출, 연도 비교\n"
                "- **기본부문(380)**: 세부 품목(예: 집적회로) 분석, 중분류 집계 함께 표시")
            _go("io-domestic", "산업연관표 분석 열기")
            _go("io-basic", "기본부문(380부문) 분석 열기 — 세부 품목, 2020년 실측표")
    with c2:
        with st.container(border=True):
            st.subheader("국제 산업연관표 분석")
            st.markdown(
                "OECD 국제산업연관표(ICIO, 77개 경제권 × 45개 산업)\n\n"
                "- **국가 간 수요 파급**: 한 나라의 수요 충격 → 각국 GDP\n"
                "- **관세 시나리오**: 관세 부과 → 교역 감소·대체, 각국 GDP·물가\n"
                "- **공급망 차질**: 특정 국가·산업 차질 → 하류·상류 파급\n"
                "- **가격 파급**: 에너지 등 가격 상승 → 각국 생산자·소비자 물가\n"
                "- **GVC·부가가치무역**: 총수출 부가가치 분해(KWW), 전방·후방 참여도, 부가가치 기준 교역")
            _go("io-international", "국제 산업연관표 분석 열기")
    with st.container(border=True):
        st.subheader("잠재성장률 추정")
        st.markdown(
            "한국 실질 GDP·고용·근로시간·자본서비스·물가·금리(ECOS 등) 또는 직접 올린 다른 나라 자료 · 한국은행 추정 방법론 참고\n\n"
            "- **통계모형 7가지**: 선형 추세, HP(양방향·실시간), 대역통과(CF), Hamilton 회귀, Beveridge-Nelson, 비관측 요인 모형\n"
            "- **준구조모형 3가지**: LW형 준구조모형(IS·필립스·오쿤), 다변량 HP 필터, 구조 VAR\n"
            "- **생산함수모형 2가지**: 생산함수 접근(인구·참가율·자연실업률·근로시간, 자본서비스), 생산함수 다변량 필터(FRB/US형)\n"
            "- **비교·분해**: 실제 성장률·잠재성장률·GDP 갭을 방법별·모형군별로, TFP·자본·노동 요인 기여도, 자연실업률·중립금리\n"
            "- **끝점 보정**: 한국은행 경제전망(분기 경로·반기·연간)으로 자료를 연장해 추정")
        _go("potential-growth", "잠재성장률 추정 열기")
    with st.container(border=True):
        st.subheader("중립금리 추정")
        st.markdown(
            "기준금리·기대인플레이션·국고채 금리·잠재성장률로 실질·명목 중립금리와 통화정책 기조를 추정\n\n"
            "- **6가지 방법**: HP 추세, 비관측 요인 모형, LW형 준구조모형, VAR 장기 예측(Lubik-Matthes형), "
            "시장금리(5년 후 5년 선도), 잠재성장률 연계(오일러 방정식)\n"
            "- **정책 기조**: 실질 정책금리 − 중립금리(플러스 = 긴축적), 한국은행 기존 추정 범위와 비교")
        _go("neutral-rate", "중립금리 추정 열기")
    st.subheader("추가 예정")
    c4, c5 = st.columns(2)
    c4.info("**DSGE 모형**\n\n신케인즈 3방정식·소규모개방경제 등. 모수·정책규칙을 바꿔 충격반응 계산")
    c5.info("**HANK 모형**\n\n한국 소규모개방경제 2자산 HANK. 소득분위별 소비 반응")
    st.subheader("이용 방법")
    st.markdown(
        "1. 왼쪽 메뉴에서 분석을 고릅니다.\n"
        "2. 자료 연도·부문 분류를 정하고, 충격(시나리오)을 입력 줄에 고릅니다(▾ 를 누르면 목록이 열리고, 줄 추가·✕ 로 늘리고 지웁니다). 자주 쓰는 시나리오는 **예시 불러오기**로 채울 수 있습니다.\n"
        "3. **분석 실행**을 눌러야 계산되며, 입력을 바꾸면 다시 눌러야 합니다. 결과로 요약 지표, 차트, 상세 표가 나옵니다.\n"
        "4. **엑셀로 내려받기**: 요약·입력조건·결과표·차트·방법론 시트가 담긴 파일을 받습니다.\n"
        "5. 입력한 시나리오는 JSON 파일로 저장해 두었다가 다시 불러올 수 있습니다.")
    footer()
