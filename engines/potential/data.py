"""잠재성장률 추정용 자료 읽기.

표준 형식: 분기 PeriodIndex(freq=Q) DataFrame
  gdp      실질 GDP 수준 (계절조정)            — 필수
  urate    실업률 (%)                          — 다변량 필터·SVAR·생산함수
  core     근원 소비자물가 지수                — 다변량 필터(필립스곡선)
  emp, lf, pop15  취업자·경제활동인구·15세 이상 인구 — 생산함수
  hours    취업자 1인당 평균 근로시간           — 생산함수(없으면 취업자 수 기준)
  ksvc     자본서비스 물량지수                  — 생산함수(없으면 capital)
  capital  실질 자본스톡 (분기말)              — 생산함수(자본서비스 대안)
  rate     정책금리 (%)                         — 준구조모형(LW)
  pie      기대인플레이션 (%)                   — 준구조모형·생산함수 다변량 필터(없으면 근원물가 과거 평균)
"""
from __future__ import annotations

import io
import json
import os
from functools import lru_cache

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "data", "processed", "potential")

COLUMNS = {
    "gdp": "실질 GDP (계절조정, 십억원)",
    "emp": "취업자 (계절조정, 천명)",
    "lf": "경제활동인구 (계절조정, 천명)",
    "pop15": "15세 이상 인구 (천명)",
    "urate": "실업률 (계절조정, %)",
    "cpi": "소비자물가 (2020=100)",
    "core": "근원물가: 농산물·석유류 제외 (2020=100)",
    "capu": "제조업 평균가동률 (%)",
    "capital": "실질 생산자본스톡 (분기말 보간, 십억원)",
    "ksvc": "자본서비스 물량지수 (전산업, 2020=100, 분기 보간)",
    "hours": "취업자 1인당 연간 실근로시간 (시간, 분기 보간)",
    "rate": "한국은행 기준금리 (분기 평균, %)",
    "pie": "기대인플레이션: 소비자 향후 1년 (분기 평균, %)",
    "y3": "국고채 3년 (분기 평균, %)",
    "y5": "국고채 5년 (분기 평균, %)",
    "y10": "국고채 10년 (분기 평균, %)",
}


@lru_cache(maxsize=1)
def load_korea() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(os.path.join(DIR, "kr_quarterly.csv"), index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="Q")
    meta = json.load(open(os.path.join(DIR, "kr_meta.json"), encoding="utf-8"))
    return df, meta


def parse_upload(raw: bytes) -> pd.DataFrame:
    """사용자 CSV/엑셀 → 표준 형식. 첫 열은 분기(2024Q1, 2024-Q1, 2024.1, 2024-03-31 등)."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_excel(io.BytesIO(raw))
    df.columns = [str(c).strip().lower() for c in df.columns]
    first = df.columns[0]
    idx = []
    for v in df[first].astype(str):
        v = v.strip().upper().replace(" ", "")
        if "Q" in v:
            y, q = v.replace("-", "").split("Q")
            idx.append(pd.Period(f"{int(y)}Q{int(q)}", freq="Q"))
        elif "." in v and len(v.split(".")[1]) == 1:
            y, q = v.split(".")
            idx.append(pd.Period(f"{int(y)}Q{int(q)}", freq="Q"))
        else:
            idx.append(pd.Period(pd.Timestamp(v), freq="Q"))
    df.index = pd.PeriodIndex(idx, freq="Q")
    df = df.drop(columns=[first])
    keep = [c for c in COLUMNS if c in df.columns]
    if "gdp" not in keep:
        raise ValueError("'gdp' 열이 필요합니다.")
    df = df[keep].apply(pd.to_numeric, errors="coerce").sort_index()
    return df


def label(p: pd.Period) -> str:
    """분기 라벨: 2026Q3 → '26.3Q'"""
    return f"{str(p.year)[2:]}.{p.quarter}Q"


NEEDS = {
    "lw": "실업률(urate)·근원물가(core)·정책금리(rate)",
    "mv": "실업률(urate)·근원물가(core)",
    "svar": "실업률(urate)",
    "pf": "고용(emp·lf·pop15)·실업률(urate)·자본(ksvc 또는 capital)",
    "pfmv": "고용(emp·lf·pop15)·실업률(urate)·근원물가(core)·자본(ksvc 또는 capital)",
}


def available_methods(df: pd.DataFrame) -> dict[str, bool]:
    has = lambda *c: all(x in df.columns and df[x].notna().sum() > 40 for x in c)  # noqa: E731
    cap = has("ksvc") or has("capital")
    return {
        "lw": has("gdp", "urate", "core", "rate"),
        "mv": has("gdp", "urate", "core"),
        "svar": has("gdp", "urate"),
        "pf": has("gdp", "emp", "lf", "pop15", "urate") and cap,
        "pfmv": has("gdp", "emp", "lf", "pop15", "urate", "core") and cap,
    }


def template_csv() -> bytes:
    cols = "quarter,gdp,urate,core,emp,lf,pop15,hours,ksvc,capital,capu,rate,pie"
    blank = "," * cols.count(",")
    rows = [cols, "2000Q1" + blank, "2000Q2" + blank]
    return ("\n".join(rows) + "\n").encode("utf-8-sig")
