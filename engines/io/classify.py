"""한국은행 산업연관표 부문 분류 연계표.

중분류(83) → 대분류(33): 2015·2020년 기준 모두 같은 83부문 분류. 총산출 합계로 검증함.
대분류(33) → 6대 산업군: 결과 화면·엑셀 요약용.
"""
from __future__ import annotations

MID_TO_LARGE_RANGES = {
    "A": (1, 5), "B": (6, 7), "C01": (8, 10), "C02": (11, 12), "C03": (13, 15),
    "C04": (16, 16), "C05": (17, 24), "C06": (25, 26), "C07": (27, 29), "C08": (30, 30),
    "C09": (31, 36), "C10": (37, 37), "C11": (38, 39), "C12": (40, 42), "C13": (43, 43),
    "C14": (44, 44), "D": (45, 46), "E": (47, 49), "F": (50, 51), "G": (52, 52),
    "H": (53, 57), "I": (58, 58), "J": (59, 64), "K": (65, 67), "L": (68, 69),
    "M": (70, 72), "N": (73, 74), "O": (75, 75), "P": (76, 76), "Q": (77, 78),
    "R": (79, 80), "S": (81, 82), "T": (83, 83),
}

MID_TO_LARGE = {f"{i:02d}": large for large, (lo, hi) in MID_TO_LARGE_RANGES.items()
                for i in range(lo, hi + 1)}

GROUPS = ["농림어업", "광업", "제조업", "전기·가스·수도", "건설", "서비스"]


def large_to_group(code: str) -> str:
    if code == "A":
        return "농림어업"
    if code == "B":
        return "광업"
    if code.startswith("C"):
        return "제조업"
    if code in ("D", "E"):
        return "전기·가스·수도"
    if code == "F":
        return "건설"
    return "서비스"


def to_group(code: str) -> str:
    """대분류·중분류 코드 모두 받아 6대 산업군으로."""
    if code in MID_TO_LARGE:
        code = MID_TO_LARGE[code]
    return large_to_group(code)


def to_large(code: str) -> str:
    return MID_TO_LARGE.get(code, code)


# 프리셋용 부문 (분류별 코드)
ENERGY_IMPORT = {"대분류": ["B", "C04"], "중분류": ["06", "16"],
                 # 기본부문: 원유·천연가스(LNG)·석유정제제품 (석탄·석탄제품 제외)
                 "기본부문": ["0621", "0622", "1621", "1622", "1623", "1624", "1625", "1626", "1627", "1628", "1631", "1639"]}
