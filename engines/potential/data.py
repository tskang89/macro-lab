"""잠재성장률 추정용 자료 읽기.

표준 형식: 분기 PeriodIndex(freq=Q) DataFrame
  gdp      실질 GDP 수준 (계절조정)            — 필수
  urate    실업률 (%)                          — 다변량 필터·SVAR·생산함수
  core     근원 소비자물가 지수                — 다변량 필터(필립스곡선)
  emp, lf, pop15  취업자·경제활동인구·15세 이상 인구 — 생산함수
  capital  실질 자본스톡 (분기말)              — 생산함수
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


def available_methods(df: pd.DataFrame) -> dict[str, bool]:
    has = lambda *c: all(x in df.columns and df[x].notna().sum() > 40 for x in c)
    return {
        "mv": has("gdp", "urate", "core"),
        "pf": has("gdp", "emp", "lf", "pop15", "urate", "capital"),
        "svar": has("gdp", "urate"),
    }


def template_csv() -> bytes:
    rows = ["quarter,gdp,urate,core,emp,lf,pop15,capital",
            "2000Q1,,,,,,,", "2000Q2,,,,,,,"]
    return ("\n".join(rows) + "\n").encode("utf-8-sig")
