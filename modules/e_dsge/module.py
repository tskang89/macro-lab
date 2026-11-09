from modules._planned import planned_page

META = {"title": "DSGE 모형", "group": "구조 모형", "order": 10, "icon": ":material/hub:",
        "status": "planned", "slug": "dsge"}


def render():
    planned_page("DSGE 모형",
                 "신케인즈 DSGE 모형으로 통화·재정·대외 충격의 거시 파급경로를 계산합니다.",
                 ["모형: 3방정식 신케인즈, 소규모개방경제(중형), 금융마찰(BGG) 등 교재 예제에서 선별",
                  "추정: 사무소 PC의 Dynare 7.1·Octave 로 베이지안 추정 → 사후 모수를 웹에 탑재",
                  "웹 계산: Python 선형 합리적기대 해법(QZ)으로 모수·정책규칙을 바꿔 즉시 충격반응 계산",
                  "검증: 기본 모수에서 Dynare 충격반응과 자동 대조"])
