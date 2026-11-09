from modules._planned import planned_page

META = {"title": "HANK 모형", "group": "구조 모형", "order": 20, "icon": ":material/groups:",
        "status": "planned", "slug": "hank"}


def render():
    planned_page("HANK 모형",
                 "이질적 가계 신케인즈(HANK) 모형으로 정책 충격이 소득·자산 계층별로 어떻게 다르게 전달되는지 봅니다.",
                 ["모형: 한국 소규모개방경제 연속시간 2자산 HANK (Kaplan–Moll–Violante형)",
                  "사전 계산: 정상상태·가계 야코비안(약 6분)을 오프라인으로 계산해 탑재",
                  "웹 계산: Sequence-Space Jacobian 으로 테일러 준칙·재정 조달 방식·충격을 바꿔 즉시 계산",
                  "결과: 거시 충격반응, 소비 경로 분해(직접·간접), 소득분위별 반응, 엑셀"])
