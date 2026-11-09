"""한국은행 산업연관표(ECOS) 원자료 → 표준화 자료(data/processed/io_kr/*.npz).

ECOS 표 구조
  - ITEM_CODE1 = 열(수요부문), ITEM_CODE2 = 행(투입부문)
  - 부문 코드는 숫자/영문 2~3자리, 9로 시작하는 4자리는 합계·최종수요·부가가치 항목
  - 2015년 기준(2015~2019), 2020년 기준(2020~2024) 두 개 계열

산출 (연도·분류마다 한 파일)
  codes, names           : 부문 코드·이름
  Z                      : 총거래 중간거래 (n×n, 백만원) — 국산+수입
  F                      : 총거래 최종수요 (n×k, 백만원), FD_CODES 순서
  imports                : 부문별 수입(+생산물세(수입)) (n, 백만원)
  x                      : 총산출 (n)
  V                      : 부가가치 항목 (4×n), VA_CODES 순서
  L_bok                  : 한국은행 생산유발계수 (I-Ad)^-1 (n×n)
  Lm_bok, Lv_bok, Le_bok, Lw_bok : 수입·부가가치·취업·고용 유발계수 (n×n 또는 열합)
실행:  python build/build_io_kr.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "ecos")
OUT = os.path.join(ROOT, "data", "processed", "io_kr")

FD_CODES = ["9111", "9112", "9121", "9122", "9131", "9132", "9140"]
FD_NAMES = ["민간소비", "정부소비", "민간고정자본형성", "정부고정자본형성", "재고증감", "귀중품순취득", "수출"]
VA_CODES = ["9610", "9621", "9622", "9630"]
VA_NAMES = ["피용자보수", "영업잉여", "고정자본소모", "순생산물세"]

# (기준, 분류) → ECOS 통계표 코드
TABLES = {
    ("2020", "대분류"): dict(tot="271Y110", L="271Y112", M="271Y113", V="271Y114", E="271Y115", W="271Y116"),
    ("2020", "중분류"): dict(tot="271Y120", L="271Y122", M="271Y123", V="271Y124", E="271Y125", W="271Y126"),
    ("2015", "대분류"): dict(tot="271Y008", L="271Y010", M="271Y011", V="271Y012", E="271Y013", W="271Y014"),
    ("2015", "중분류"): dict(tot="271Y068", L="271Y070", M="271Y071", V="271Y072", E="271Y073", W="271Y074"),
}


def is_sector(code: str) -> bool:
    return not (len(code) == 4 and code.startswith("9"))


def load(code: str):
    """연도 → {(행코드, 열코드): 값}, 코드→이름."""
    path = os.path.join(RAW, f"{code}.json")
    rows = json.load(open(path, encoding="utf-8"))
    data = defaultdict(dict)
    names = {}
    for r in rows:
        v = r["DATA_VALUE"]
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        data[r["TIME"]][(r["ITEM_CODE2"], r["ITEM_CODE1"])] = v
        names[r["ITEM_CODE1"]] = r["ITEM_NAME1"].replace("\n", " ").strip()
        names[r["ITEM_CODE2"]] = r["ITEM_NAME2"].replace("\n", " ").strip()
    return data, names


def matrix(d, rows, cols):
    return np.array([[d.get((i, j), 0.0) for j in cols] for i in rows], dtype=float)


def build(basis: str, level: str):
    t = TABLES[(basis, level)]
    if not os.path.exists(os.path.join(RAW, t["tot"] + ".json")):
        print("missing", t["tot"])
        return []
    tot, names = load(t["tot"])
    others = {k: load(t[k])[0] for k in ("L", "M", "V", "E", "W")
              if os.path.exists(os.path.join(RAW, t[k] + ".json"))}
    years = sorted(tot)
    made = []
    for y in years:
        d = tot[y]
        row_codes = {i for (i, j) in d}
        col_codes = {j for (i, j) in d}
        sectors = sorted(c for c in row_codes & col_codes if is_sector(c))
        Z = matrix(d, sectors, sectors)
        F = matrix(d, sectors, FD_CODES)
        imp = matrix(d, sectors, ["9329"])[:, 0]  # 수입계(수입+생산물세(수입))
        x = matrix(d, sectors, ["9310"])[:, 0]
        V = matrix(d, VA_CODES, sectors)
        x_col = matrix(d, ["9790"], sectors)[0]
        out = dict(codes=np.array(sectors), names=np.array([names[c] for c in sectors]),
                   Z=Z, F=F, imports=imp, x=x, x_col=x_col, V=V,
                   fd_names=np.array(FD_NAMES), va_names=np.array(VA_NAMES))
        for k, key in (("L", "L_bok"), ("M", "Lm_bok"), ("V", "Lv_bok"), ("E", "Le_bok"), ("W", "Lw_bok")):
            if k in others and y in others[k]:
                out[key] = matrix(others[k][y], sectors, sectors)
        os.makedirs(OUT, exist_ok=True)
        fn = os.path.join(OUT, f"io_kr_{level}_{y}.npz")
        np.savez_compressed(fn, basis=basis, year=y, level=level, **out)
        made.append((level, y, len(sectors)))
    return made


if __name__ == "__main__":
    for key in TABLES:
        for m in build(*key):
            print("built", m)
