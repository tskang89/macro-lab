from modules._planned import planned_page

META = {"title": "잠재성장률 추정", "group": "성장 분석", "order": 10, "icon": ":material/trending_up:",
        "status": "planned", "slug": "potential-growth"}


def render():
    planned_page("잠재성장률 추정",
                 "한국·유로지역 등의 잠재성장률과 산출갭을 여러 방법으로 추정해 비교합니다.",
                 ["통계적 필터: HP, 밴드패스(Christiano-Fitzgerald), Hamilton 회귀 필터",
                  "비관측 요인 모형(칼만필터): 필립스곡선을 포함한 다변량 모형",
                  "생산함수 접근: 노동·자본·총요소생산성 분해(EC 방식 참고)",
                  "자료: ECOS·Eurostat·OECD 자동 수집, 방법 간 비교 차트와 엑셀"])
