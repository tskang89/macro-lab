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
  rate       한국은행 기준금리 (분기 평균, 연%)                    722Y001 / 0101000
  pie        일반인 향후 1년 기대인플레이션 (분기 평균, %)          511Y003 / FMB   (2002Q1~)
  y3,y5,y10  국고채 3·5·10년 수익률 (분기 평균, 연%)               721Y001 / 5020000·5040000·5050000
연간 자료 → 분기 보간
  capital    실질 생산자본스톡 (고정자산, 연말, 십억원)            291Y504 / 101
             분기값은 연말 잔액을 로그 선형 보간(분기말 기준). 마지막 연말 이후는 직전 3년 평균 증가율로 연장.
  ksvc       자본서비스물량지수 (전산업, 연쇄 통크비스트, 2020=100)  291Y522 / IND00
             연간 흐름 지표이므로 Denton 방식(분기 평균 = 연간값, 분기 간 변화 최소)으로 분기화.
             마지막 공표 연도 이후는 직전 3년 평균 증가율로 연장.
  hours      취업자 1인당 연간 실근로시간                          OECD(2008~) + PWT 증가율로 과거 연장
             OECD: Average annual hours actually worked per worker (DSD_HW@DF_AVG_ANN_HRS_WKD, 전체 취업자)
             PWT : FRED AVHWPEKRA065NRUG (Penn World Table avh) — 2008년 이전은 PWT 증가율로 거슬러 연장
             공표 연도 이후(당해 연도)는 고용노동부 사업체노동력조사 월 근로시간(901Y148·901Y052 / I72BD)의
             연초 이후 전년동기비로 연장. 연간 → 분기는 Denton 방식.

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


def denton(annual: pd.Series, first: int, last: int) -> pd.Series:
    """연간 평균 → 분기: 분기 평균이 연간값과 같고 분기 간 차분 제곱합이 최소인 계열 (Denton 가법, 1차 차분)."""
    a = annual.loc[first:last].astype(float)
    n = 4 * len(a)
    D = np.diff(np.eye(n), axis=0)
    C = np.kron(np.eye(len(a)), np.full((1, 4), 0.25))
    K = np.block([[2 * D.T @ D, C.T], [C, np.zeros((len(a), len(a)))]])
    rhs = np.concatenate([np.zeros(n), a.to_numpy()])
    q = np.linalg.solve(K, rhs)[:n]
    idx = pd.period_range(f"{first}Q1", periods=n, freq="Q")
    return pd.Series(q, index=idx)


def extend_annual(a: pd.Series, to_year: int, years: int = 3) -> pd.Series:
    a = a.copy()
    g = np.log(a).diff().iloc[-years:].mean()
    while a.index[-1] < to_year:
        a.loc[a.index[-1] + 1] = a.iloc[-1] * np.exp(g)
    return a


def _csv(url: str) -> pd.DataFrame:
    import io
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (macro-lab data build)"})
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=180).read()))


def oecd_hours() -> pd.Series:
    url = ("https://sdmx.oecd.org/public/rest/data/OECD.ELS.SAE,DSD_HW@DF_AVG_ANN_HRS_WKD,1.0/"
           "KOR.HW.H_Y_PS._Z._Z.EMP.A.ACTUAL._T._Z.MEAN._Z._T?format=csvfile")
    df = _csv(url)
    return df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float).sort_index()


def pwt_hours() -> pd.Series:
    df = _csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=AVHWPEKRA065NRUG")
    df["y"] = df.iloc[:, 0].str[:4].astype(int)
    return df.set_index("y").iloc[:, 1].astype(float)


def hours_annual(last_q: pd.Period) -> tuple[pd.Series, dict]:
    o, p = oecd_hours(), pwt_hours()
    y0 = int(o.index[0])
    back = (p / p.loc[y0] * o.loc[y0]).loc[:y0 - 1]
    h = pd.concat([back, o]).sort_index()
    h.index = h.index.astype(int)
    info = {"hours_oecd": f"{y0}~{int(o.index[-1])}", "hours_pwt_back": f"{int(back.index[0])}~{y0 - 1}"}
    # 당해 연도: 사업체노동력조사 월 근로시간의 연초 이후 전년동기비
    yr = last_q.year
    if h.index[-1] < yr:
        cur = fetch("901Y148", "M", f"{yr}01", f"{yr}12", "A", "I72BD")
        prev = fetch("901Y052", "M", f"{yr - 1}01", f"{yr - 1}12", "A", "I72BD")
        months = [m[-2:] for m in cur.index]
        prev = prev[[m[-2:] in months for m in prev.index]]
        r = cur.sum() / prev.sum()
        h.loc[yr] = h.loc[yr - 1] * r
        info["hours_current_year"] = f"{yr}: 사업체노동력조사 1~{int(months[-1])}월 전년동기비 {100 * (r - 1):+.2f}%"
    return h, info


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
    ksvc_a = fetch("291Y522", "A", "1970", "2030", "IND00")
    ksvc_a.index = ksvc_a.index.astype(int)
    rate = q_index(fetch("722Y001", "Q", "1999Q1", "2030Q4", "0101000"))
    pie_m = fetch("511Y003", "M", "200201", "203012", "FMB")
    pie_m.index = pd.PeriodIndex([f"{t[:4]}-{t[4:]}" for t in pie_m.index], freq="M")
    pie = pie_m.groupby(pie_m.index.asfreq("Q")).mean()

    df = pd.DataFrame({"gdp": gdp})
    for k, s in lab.items():
        df[k] = s
    df["cpi"], df["core"], df["capu"] = cpi, core, capu
    df = df.loc[df.index <= gdp.index[-1]]
    df["capital"] = interp_capital(capa, df.index)
    df["rate"], df["pie"] = rate, pie
    for col, item in (("y3", "5020000"), ("y5", "5040000"), ("y10", "5050000")):
        df[col] = q_index(fetch("721Y001", "Q", "1995Q1", "2030Q4", item))
    last_q = gdp.index[-1]
    ks = extend_annual(ksvc_a, last_q.year)
    df["ksvc"] = denton(ks, int(ks.index[0]), int(ks.index[-1])).reindex(df.index)
    h_a, hinfo = hours_annual(last_q)
    df["hours"] = denton(h_a, int(h_a.index[0]), int(h_a.index[-1])).reindex(df.index)
    df.index = df.index.astype(str)
    df.index.name = "quarter"
    df.to_csv(os.path.join(OUT, "kr_quarterly.csv"), float_format="%.6g")
    meta = {"last_gdp": str(gdp.index[-1]), "capital_last_year": int(capa.index[-1]),
            "ksvc_last_year": int(ksvc_a.index[-1]), **hinfo,
            "built": pd.Timestamp.now().strftime("%Y-%m-%d")}
    json.dump(meta, open(os.path.join(OUT, "kr_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(df.tail(6).to_string())
    print(df.notna().sum().to_string())
    print(meta)


if __name__ == "__main__":
    main()
