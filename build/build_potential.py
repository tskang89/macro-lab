"""잠재성장률 추정용 한국 거시 자료 (ECOS) → data/processed/potential/kr_quarterly.csv

분기 자료 (계절조정)
  gdp        실질 GDP (계절조정, 연쇄가격, 십억원)                200Y104 / 1400
  emp        취업자 (계절조정, 천명)                              901Y027 / I61BA · I28B
  lf         경제활동인구 (계절조정, 천명)                        901Y027 / I61B  · I28B
  pop15      15세 이상 인구 (천명)                                901Y027 / I61A  · I28B(없으면 원계열)
  urate      실업률 (계절조정, %)                                 901Y027 / I61BC · I28B
  cpi        소비자물가 총지수 (2020=100)                          901Y009 / 0
  core       농산물및석유류제외 소비자물가 (2020=100)              901Y010 / QB
  capu       제조업 평균가동률 (%)                                 901Y025 / I31A
연간 자료 → 분기 보간
  capital    실질 생산자본스톡 (고정자산, 연말, 십억원)            291Y504 / 101
             분기값은 연말 잔액을 로그 선형 보간(분기말 기준). 마지막 연말 이후는 직전 3년 평균 증가율로 연장.

실행:  python build/build_potential.py      (키: 환경변수 ECOS_API_KEY 또는 euro-chartpack/.env)
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed", "potential")


def _key() -> str:
    k = os.environ.get("ECOS_API_KEY")
    if k:
        return k
    env = r"D:\Claude\Projects\euro-chartpack\.env"
    return [l.split("=", 1)[1].strip() for l in open(env, encoding="utf-8") if l.startswith("ECOS_API_KEY=")][0]


def fetch(stat: str, cycle: str, start: str, end: str, item1: str, item2: str | None = None) -> pd.Series:
    key = _key()
    path = "/".join(x for x in [item1, item2] if x)
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10000/{stat}/{cycle}/{start}/{end}/{path}"
    d = json.load(urllib.request.urlopen(url, timeout=120))
    if "StatisticSearch" not in d:
        raise RuntimeError(f"{stat}/{path}: {d}")
    rows = d["StatisticSearch"]["row"]
    s = pd.Series({r["TIME"]: float(r["DATA_VALUE"]) for r in rows if r["DATA_VALUE"] not in (None, "")})
    return s.sort_index()


def q_index(s: pd.Series) -> pd.Series:
    s.index = pd.PeriodIndex([f"{t[:4]}Q{t[-1]}" for t in s.index], freq="Q")
    return s


def interp_capital(annual: pd.Series, qidx: pd.PeriodIndex) -> pd.Series:
    """연말 잔액 → 분기말 잔액 (로그 선형). 이후 구간은 최근 3년 평균 성장률로 연장."""
    a = annual.copy()
    a.index = pd.PeriodIndex([f"{y}Q4" for y in a.index], freq="Q")
    g = np.log(a).diff().iloc[-3:].mean() / 4
    last = a.index[-1]
    full = pd.Series(np.nan, index=pd.period_range(a.index[0], max(qidx[-1], last), freq="Q"))
    full.loc[a.index] = np.log(a.values)
    full = full.interpolate(limit_area="inside")
    for p in full.index[full.index > last]:
        full.loc[p] = full.loc[p - 1] + g
    return np.exp(full).reindex(qidx)


def main():
    os.makedirs(OUT, exist_ok=True)
    gdp = q_index(fetch("200Y104", "Q", "1960Q1", "2030Q4", "1400"))
    lab = {}
    for name, item in [("emp", "I61BA"), ("lf", "I61B"), ("pop15", "I61A"), ("urate", "I61BC")]:
        try:
            lab[name] = q_index(fetch("901Y027", "Q", "1999Q1", "2030Q4", item, "I28B"))
        except RuntimeError:
            lab[name] = q_index(fetch("901Y027", "Q", "1999Q1", "2030Q4", item, "I28A"))
    cpi = q_index(fetch("901Y009", "Q", "1965Q1", "2030Q4", "0"))
    core = q_index(fetch("901Y010", "Q", "1975Q1", "2030Q4", "QB"))
    capu = q_index(fetch("901Y025", "Q", "1980Q1", "2030Q4", "I31A"))
    capa = fetch("291Y504", "A", "1970", "2030", "101")
    capa.index = capa.index.astype(int)

    df = pd.DataFrame({"gdp": gdp})
    for k, s in lab.items():
        df[k] = s
    df["cpi"], df["core"], df["capu"] = cpi, core, capu
    df = df.loc[df.index <= gdp.index[-1]]
    df["capital"] = interp_capital(capa, df.index)
    df.index = df.index.astype(str)
    df.index.name = "quarter"
    df.to_csv(os.path.join(OUT, "kr_quarterly.csv"), float_format="%.6g")
    meta = {"last_gdp": str(gdp.index[-1]), "capital_last_year": int(capa.index[-1]),
            "built": pd.Timestamp.now().strftime("%Y-%m-%d")}
    json.dump(meta, open(os.path.join(OUT, "kr_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(df.tail(6).to_string())
    print(df.notna().sum().to_string())
    print(meta)


if __name__ == "__main__":
    main()
