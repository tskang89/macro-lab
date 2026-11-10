"""끝점 보정: 한국은행 경제전망(성장률 등)으로 자료를 전망 시계 끝까지 연장.

한국은행(권지호 외 2019, 이은경 외 2024)은 필터의 끝점 문제를 줄이려고 GDP 를 자체 전망치로 연장한 뒤 잠재성장률을 추정한다.
경제전망보고서는 당해·다음 해의 반기(상반·하반, 전년동기비)·연간 성장률과 향후 1년의 분기 경로를 발표한다.

연장 규칙 (마지막 실적 분기 다음부터 전망 시계 끝 분기까지)
  실질 GDP   ① 분기 경로가 있는 분기: 기본(yoy)은 전년동기 수준 × (1 + 전년동기비 전망) — 발표된 반기·연간 성장률을 재현.
                선택(qoq)은 직전 분기 수준 × (1 + 전기비 전망) — 실적이 전망과 달라진 만큼의 수준 차이가 그대로 이어짐
             ② 두 분기가 모두 비어 있는 반기: 반기 합계의 전년동기비 = 전망치가 되도록, 두 분기에 같은 전기비 성장을 줌
             ③ 한 분기만 비어 있는 반기: 그 분기의 전년동기비 = 반기 전망치
             ④ 반기 전망이 없으면 연간 전망으로 같은 방식
  근원물가·소비자물가  분기 경로(전년동기비) → 반기 전망 → 연간 전망 순으로 그 분기의 전년동기비를 맞춤
  취업자     전년동기 수준 + 반기(없으면 연간) 취업자수 증감 전망
  실업률     반기(없으면 연간) 실업률 전망(계절조정)
  경제활동인구 = 취업자 / (1 − 실업률)
  15세 이상 인구·평균 근로시간·자본서비스·자본스톡  최근 3년(12분기) 평균 증가율로 추세 연장
  가동률     최근 4분기 평균 유지,  기준금리·기대인플레이션  마지막 값 유지
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from engines.potential.data import DIR

FC_COL = "fc"            # 1 = 전망·가정으로 채운 분기


def load_bok() -> dict | None:
    p = os.path.join(DIR, "bok_forecast.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _half(q: pd.Period) -> str:
    return f"{q.year}H{1 if q.quarter <= 2 else 2}"


def _half_quarters(h: str) -> list[pd.Period]:
    y, k = int(h[:4]), int(h[-1])
    return [pd.Period(f"{y}Q{2 * k - 1}", freq="Q"), pd.Period(f"{y}Q{2 * k}", freq="Q")]


def _year_quarters(y: int) -> list[pd.Period]:
    return [pd.Period(f"{y}Q{i}", freq="Q") for i in range(1, 5)]


def _solve_equal_growth(prev: float, n: int, target_sum: float) -> float:
    """prev·(1+r) + … + prev·(1+r)^n = target_sum 을 만족하는 r (이분법)."""
    lo, hi = -0.2, 0.2
    f = lambda r: sum(prev * (1 + r) ** (i + 1) for i in range(n)) - target_sum  # noqa: E731
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _extend_gdp(s: pd.Series, fc: dict, quarters: list[pd.Period], mode: str = "yoy") -> tuple[pd.Series, dict]:
    """mode='yoy': 분기 경로의 전년동기비를 그대로 맞춤(발표된 반기·연간 성장률을 최대한 재현).
    mode='qoq': 분기 경로의 전기비를 마지막 실적 수준에 이어 붙임(실적이 전망과 달라진 수준 차이가 이어짐)."""
    s = s.copy()
    how = {}
    qq = fc.get("q_qoq", {}) if mode == "qoq" else fc.get("q_yoy", {})
    hh = fc.get("h_yoy", {})
    aa = fc.get("a_yoy", {})
    for q in quarters:
        if pd.notna(s.get(q, np.nan)):
            continue
        if str(q) in qq:
            if mode == "qoq":
                s[q] = s[q - 1] * (1 + qq[str(q)] / 100)
                how[q] = f"분기 경로 전기비 {qq[str(q)]:.1f}%"
            else:
                s[q] = s[q - 4] * (1 + qq[str(q)] / 100)
                how[q] = f"분기 경로 전년동기비 {qq[str(q)]:.1f}%"
            continue
        # 반기 또는 연간 제약
        for key, members, tab, name in ((_half(q), _half_quarters(_half(q)), hh, "반기"),
                                        (str(q.year), _year_quarters(q.year), aa, "연간")):
            if key not in tab:
                continue
            g = tab[key] / 100
            free = [m for m in members if pd.isna(s.get(m, np.nan)) and str(m) not in qq]
            known = [m for m in members if m not in free]
            prior = [m - 4 for m in members]
            if any(pd.isna(s.get(m, np.nan)) for m in prior):
                continue
            if not known and free[0] == q:
                r = _solve_equal_growth(s[q - 1], len(free), (1 + g) * sum(s[m] for m in prior))
                lev = s[q - 1]
                for m in free:
                    lev *= 1 + r
                    s[m] = lev
                    how[m] = f"{name} 전망 {tab[key]:.1f}%(전년동기비)에 맞춘 균등 성장(전기비 {100 * r:.2f}%)"
            else:
                s[q] = s[q - 4] * (1 + g)
                how[q] = f"{name} 전망 {tab[key]:.1f}%를 이 분기 전년동기비로 적용"
            break
        if pd.isna(s.get(q, np.nan)):              # 전망이 없으면 직전 4분기 평균 성장
            s[q] = s[q - 1] * np.exp(np.log(s[q - 1] / s[q - 5]) / 4)
            how[q] = "전망 없음: 최근 1년 평균 성장"
    return s, how


def _yoy_path(s: pd.Series, fc: dict, quarters: list[pd.Period]) -> tuple[pd.Series, dict]:
    """지수 계열을 전년동기비 전망(분기 → 반기 → 연간)으로 연장."""
    s = s.copy()
    how = {}
    for q in quarters:
        if pd.notna(s.get(q, np.nan)):
            continue
        for tab, key, name in ((fc.get("q_yoy", {}), str(q), "분기 경로"), (fc.get("h_yoy", {}), _half(q), "반기"),
                               (fc.get("a_yoy", {}), str(q.year), "연간")):
            if key in tab:
                s[q] = s[q - 4] * (1 + tab[key] / 100)
                how[q] = f"{name} 전망 {tab[key]:.1f}%(전년동기비)"
                break
        else:
            s[q] = s[q - 4] * s[q - 1] / s[q - 5]
            how[q] = "전망 없음: 직전 상승률 유지"
    return s, how


def _level_path(s: pd.Series, fc: dict, quarters: list[pd.Period], change: bool, scale: float = 1.0):
    s = s.copy()
    how = {}
    for q in quarters:
        if pd.notna(s.get(q, np.nan)):
            continue
        for tab, key, name in ((fc.get("h", {}), _half(q), "반기"), (fc.get("a", {}), str(q.year), "연간")):
            if key in tab:
                v = tab[key]
                s[q] = s[q - 4] + v * scale if change else v
                how[q] = f"{name} 전망 {v:g}" + ("(전년동기 대비 증감)" if change else "")
                break
        else:
            s[q] = s[q - 1]
            how[q] = "전망 없음: 직전 값 유지"
    return s, how


def _trend(s: pd.Series, quarters: list[pd.Period], n: int = 12) -> pd.Series:
    s = s.copy()
    obs = s.dropna()
    if len(obs) < n + 1:
        return s
    r = np.log(obs.iloc[-1] / obs.iloc[-1 - n]) / n
    for q in quarters:
        if pd.isna(s.get(q, np.nan)):
            s[q] = s[q - 1] * np.exp(r)
    return s


def extend(df: pd.DataFrame, fc: dict, mode: str = "yoy") -> tuple[pd.DataFrame, pd.DataFrame]:
    """df 를 전망 시계 끝까지 연장. 돌려주는 값: (연장된 df, 연장 방법 표)."""
    last = df["gdp"].dropna().index[-1]
    end = pd.Period(fc["horizon"], freq="Q")
    if end <= last:
        out = df.copy()
        out[FC_COL] = 0.0
        return out, pd.DataFrame()
    quarters = list(pd.period_range(last + 1, end, freq="Q"))
    out = df.reindex(pd.period_range(df.index[0], end, freq="Q")).copy()
    # 실적 이후 구간의 기존 값(가정치)은 지우지 않음: 자본서비스·근로시간은 자료 구축 때 연장한 값이 있음
    rows = {q: {} for q in quarters}

    out["gdp"], how = _extend_gdp(out["gdp"], fc["gdp"], quarters, mode)
    for q, h in how.items():
        rows[q]["실질 GDP"] = h
    for col, key in (("core", "core"), ("cpi", "cpi")):
        if col in out and key in fc and out[col].notna().any():
            out[col], how = _yoy_path(out[col], fc[key], quarters)
            for q, h in how.items():
                rows[q][{"core": "근원물가", "cpi": "소비자물가"}[col]] = h
    if "urate" in out and "urate" in fc:
        out["urate"], how = _level_path(out["urate"], fc["urate"], quarters, change=False)
        for q, h in how.items():
            rows[q]["실업률"] = h + "%"
    if "emp" in out and "emp_change" in fc and out["emp"].notna().any():
        out["emp"], how = _level_path(out["emp"], fc["emp_change"], quarters, change=True, scale=10.0)
        for q, h in how.items():
            rows[q]["취업자"] = h.replace("증감)", "증감, 만명)")
    if {"emp", "urate", "lf"} <= set(out.columns):
        m = [q for q in quarters if pd.isna(df["lf"].get(q, np.nan))]
        out.loc[m, "lf"] = out.loc[m, "emp"] / (1 - out.loc[m, "urate"] / 100)
        for q in m:
            rows[q]["경제활동인구"] = "취업자 / (1 − 실업률)"
    for col, name in (("pop15", "15세 이상 인구"), ("hours", "평균 근로시간"), ("ksvc", "자본서비스"),
                      ("capital", "자본스톡")):
        if col in out and out[col].notna().sum() > 12:
            filled = [q for q in quarters if pd.isna(out[col].get(q, np.nan))]
            out[col] = _trend(out[col], quarters)
            for q in filled:
                rows[q][name] = "최근 3년 평균 증가율로 추세 연장"
    if "capu" in out and out["capu"].notna().any():
        v = out["capu"].dropna().iloc[-4:].mean()
        for q in quarters:
            if pd.isna(out["capu"].get(q, np.nan)):
                out.loc[q, "capu"] = v
                rows[q]["가동률"] = f"최근 4분기 평균 {v:.1f}% 유지"
    for col, name in (("rate", "기준금리"), ("pie", "기대인플레이션")):
        if col in out and out[col].notna().any():
            v = out[col].dropna().iloc[-1]
            for q in quarters:
                if pd.isna(out[col].get(q, np.nan)):
                    out.loc[q, col] = v
                    rows[q][name] = f"마지막 값 {v:.2f}% 유지"
    out[FC_COL] = 0.0
    out.loc[quarters, FC_COL] = 1.0
    tab = pd.DataFrame.from_dict(rows, orient="index")
    tab.index = pd.PeriodIndex(tab.index, freq="Q")
    return out, tab


TAIL = 2.0               # FC_COL 값: 전망 시계 이후 중기 가정으로 연장한 분기(필터 끝점용, 화면에는 표시 안 함)


def add_tail(df: pd.DataFrame, growth: float, n: int = 12) -> pd.DataFrame:
    """전망 시계 뒤를 중기 성장률 가정으로 n분기 연장 (AR 예측 연장 대신).

    AR 연장은 표본(2008년 이후) 평균 성장률(약 2.7%)로 되돌아가 끝점 추세를 끌어올린다. 대신 실질 GDP 는 연 growth% 로,
    15세 이상 인구·근로시간·자본서비스·자본스톡은 직전 분기 증가율로, 취업자는 고용률 유지(인구와 같은 증가율),
    실업률·가동률·금리·기대인플레이션은 유지, 물가는 2% 목표 상승률로 연장한다.
    """
    idx = pd.period_range(df.index[-1] + 1, periods=n, freq="Q")
    e = df.reindex(df.index.append(idx)).copy()
    g = (1 + growth / 100) ** 0.25
    for q in idx:
        e.loc[q, "gdp"] = e.loc[q - 1, "gdp"] * g
        for c in ("pop15", "hours", "ksvc", "capital"):
            if c in e and pd.notna(e.loc[q - 2, c]):
                e.loc[q, c] = e.loc[q - 1, c] ** 2 / e.loc[q - 2, c]
        for c in ("urate", "capu", "rate", "pie"):
            if c in e:
                e.loc[q, c] = e.loc[q - 1, c]
        if {"emp", "pop15"} <= set(e.columns) and pd.notna(e.loc[q - 4, "emp"]):
            e.loc[q, "emp"] = e.loc[q - 4, "emp"] * e.loc[q, "pop15"] / e.loc[q - 4, "pop15"]
            if "lf" in e and "urate" in e:
                e.loc[q, "lf"] = e.loc[q, "emp"] / (1 - e.loc[q, "urate"] / 100)
        for c in ("core", "cpi"):
            if c in e and pd.notna(e.loc[q - 4, c]):
                e.loc[q, c] = e.loc[q - 4, c] * 1.02
    e.loc[idx, FC_COL] = TAIL
    return e


def shown(df: pd.DataFrame) -> pd.DataFrame:
    """화면·표에 보일 구간(실적 + 전망, 중기 가정 꼬리는 제외)."""
    if FC_COL in df.columns:
        return df[df[FC_COL] < TAIL]
    return df


def last_actual(df: pd.DataFrame) -> pd.Period:
    if FC_COL in df.columns and (df[FC_COL] == 0).any():
        return df.index[(df[FC_COL] == 0) & df["gdp"].notna()][-1]
    return df["gdp"].dropna().index[-1]


def first_forecast(df: pd.DataFrame) -> pd.Period | None:
    if FC_COL in df.columns and (df[FC_COL] == 1).any():
        return df.index[df[FC_COL] == 1][0]
    return None


def compare(df_ext: pd.DataFrame, fc: dict) -> pd.DataFrame:
    """연장한 GDP 경로의 반기·연간 성장률과 한국은행 전망 비교."""
    g = df_ext["gdp"]
    rows = []
    for key, val in list(fc["gdp"].get("h_yoy", {}).items()) + list(fc["gdp"].get("a_yoy", {}).items()):
        mem = _half_quarters(key) if "H" in key else _year_quarters(int(key))
        if any(pd.isna(g.get(m, np.nan)) or pd.isna(g.get(m - 4, np.nan)) for m in mem):
            continue
        path = (sum(g[m] for m in mem) / sum(g[m - 4] for m in mem) - 1) * 100
        n_fc = int(sum(df_ext[FC_COL].get(m, 0) for m in mem))
        label = f"{key[:4]}년 {'상반기' if key.endswith('H1') else '하반기'}" if "H" in key else f"{key}년"
        rows.append({"기간": label, "한국은행 전망(%)": val, "연장 경로(%)": round(path, 2),
                     "전망으로 채운 분기": f"{n_fc}/{len(mem)}"})
    return pd.DataFrame(rows)
