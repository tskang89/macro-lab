"""중립금리(자연이자율, r*) 추정 — 여러 모형.

실질 중립금리 r*: 경제가 잠재 수준에서 물가 안정 상태일 때 실질 정책금리. 명목 중립금리 = r* + 물가목표(2%).
실질 정책금리 r = 기준금리 − 기대인플레이션. 기본은 근원물가 상승률의 직전 4분기 평균(Holston·Laubach·Williams 2017 방식);
소비자동향조사 일반인 기대(향후 1년)는 2010년대 이후 실제 물가를 1%p 안팎 웃돌아 실질금리를 낮게 잡으므로 선택 사항으로 둠.

모형군
  통계모형     hp      실질 정책금리의 HP 추세 (λ=1600, AR 연장으로 끝점 보정)
              uc      비관측 요인 모형: r = r* + 순환,  r* 확률보행 · 순환 AR(1)  (칼만 평활, 최우추정)
  준구조모형   lw      Laubach-Williams(2003)형: IS곡선·필립스곡선·오쿤 + r* = 잠재성장률 + z  (잠재성장률 모듈과 같은 모형)
  시계열모형   var     Lubik·Matthes(2015) 방식 단순화: 이동 창 VAR(성장률·물가·실질금리)의 5년 뒤 실질금리 예측
  시장·성장   market  국고채 5년·10년으로 구한 5년 후 5년 선도금리 − 물가목표 − 기간 프리미엄 (Christensen·Rudebusch 2019 의 생각을 단순화)
              growth  잠재성장률 연계: r* = σ·잠재성장률 + ρ (오일러 방정식, 2001~2019 회귀로 σ·ρ 결정)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engines.potential import methods as M
from engines.potential import statespace as SS

METHODS = {
    "hp": "HP 추세(실질 정책금리)",
    "uc": "비관측 요인 모형(UC)",
    "lw": "준구조모형(LW형)",
    "var": "VAR 장기 예측(Lubik-Matthes형)",
    "market": "시장금리(5년 후 5년 선도)",
    "growth": "잠재성장률 연계",
}
GROUPS = {"stat": "통계모형", "semi": "준구조모형", "ts": "시계열모형", "mkt": "시장·성장 연계"}
GROUP_OF = {"hp": "stat", "uc": "stat", "lw": "semi", "var": "ts", "market": "mkt", "growth": "mkt"}
TARGET_NOTE = "물가목표: 2016년 이후 2%, 그 이전 3%(2004~15년 목표 범위의 중심, 2001~03년 3±1%)."


@dataclass
class RStar:
    key: str
    name: str
    rstar: pd.Series                         # 실질 중립금리, %
    params: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)


def inflation_target(idx: pd.PeriodIndex) -> pd.Series:
    return pd.Series(np.where(idx.year >= 2016, 2.0, 3.0), index=idx)


def real_rate(df: pd.DataFrame) -> pd.Series:
    return (df["rate"] - M.expected_inflation(df)).dropna()


# ---------------------------------------------------------------------------
def hp_trend(df: pd.DataFrame, lam: float = 1600, extend: int = 12) -> RStar:
    r = real_rate(df)
    tr = M.hp_series(r, lam, extend)
    return RStar("hp", METHODS["hp"], tr, {"λ": lam, "끝점 연장(분기)": extend},
                 [f"실질 정책금리의 HP 추세(λ={lam:g}). 끝점은 AR(4) 예측으로 {extend}분기 연장 후 필터. 경제 이론 정보 없이 "
                  "실질금리의 저주파 움직임을 중립금리로 봄 — 정책금리 자체가 오래 중립에서 벗어나 있으면 그만큼 끌려감."])


def uc(df: pd.DataFrame, s_star_max: float = 0.2, rho_max: float = 0.9) -> RStar:
    r = real_rate(df)
    y = r.to_numpy()[:, None]
    hp0 = M.hp_array(r.to_numpy(), 1600)

    def build(x):
        phi = x[0]
        sc, se, ss = np.exp(x[1:4])
        T = np.array([[1.0, 0.0], [0.0, phi]])
        Q = np.diag([ss ** 2, sc ** 2])
        Z = np.array([[1.0, 1.0]])
        return SS.SSM(Z, np.zeros((len(y), 1)), np.array([[se ** 2]]), T, np.zeros(2), Q,
                      np.array([hp0[0], y[0, 0] - hp0[0]]), np.diag([1.0, 1.0]))

    x0 = np.array([0.85, np.log(0.5), np.log(0.1), np.log(0.1)])
    bounds = [(0.0, rho_max), (-6, 2), (-6, 0), (-6, np.log(s_star_max))]
    xh, ll, _ = SS.fit(y, build, x0, bounds)
    st, _, _ = SS.ksmooth(y, build(xh))
    rs = pd.Series(st[:, 0], index=r.index)
    sd = np.exp(xh[1:4])
    return RStar("uc", METHODS["uc"], rs,
                 {"순환 AR(1)": xh[0], "순환 충격 표준편차": sd[0], "관측 오차 표준편차": sd[1], "r* 충격 표준편차": sd[2],
                  "r* 충격 표준편차 상한": s_star_max, "순환 AR 상한": rho_max, "로그우도": ll, "표본": f"{r.index[0]}~{r.index[-1]}"},
                 ["실질 정책금리 = 중립금리(확률보행) + 순환(AR(1)) + 관측 오차. 칼만 필터 최우추정 후 평활기로 중립금리를 뽑음. "
                  f"최우추정이 금리 변동을 중립금리 충격으로 돌리지 않도록 분기 충격 표준편차 상한을 {s_star_max:g}%p 로, "
                  f"순환이 추세를 흡수하지 않도록 순환 AR 계수 상한을 {rho_max:g} 로 둠."])


def lw(df: pd.DataFrame, fit_end: str | None = None, **kw) -> RStar:
    e = M.lw(df, fit_end=fit_end, **kw)
    t = e.extra["중립금리·자연실업률"]
    return RStar("lw", METHODS["lw"], t["중립금리 추정(%)"], dict(e.params),
                 ["Laubach·Williams(2003)형 준구조모형(한국은행 권지호 외 2019 의 준구조모형과 같은 구조, 잠재성장률 화면의 모형과 같음). "
                  "IS곡선(실질금리 갭 → GDP 갭)·필립스곡선·오쿤의 관계식을 상태공간모형으로 묶어 r* = 잠재성장률(연율) + 기타 요인 z 를 추정."]
                 + e.notes[1:],
                 {"잠재성장률·z": pd.DataFrame({"잠재성장률(연율, %)": t["잠재성장률(연율, %)"],
                                               "기타 요인 z(%)": t["중립금리 추정(%)"] - t["잠재성장률(연율, %)"]})})


def var_forecast(df: pd.DataFrame, window: int = 40, p: int = 2, horizon: int = 20) -> RStar:
    """각 시점까지 window 분기 VAR(p)(GDP 성장률·근원물가·실질 정책금리)를 맞추고 horizon 분기 뒤 실질금리 예측을 r* 로."""
    g = (df["gdp"] / df["gdp"].shift(4) - 1) * 100
    pi = M._inflation(df["core"])
    r = df["rate"] - M.expected_inflation(df)
    X = pd.DataFrame({"g": g, "pi": pi, "r": r}).dropna()
    out, roots = {}, {}
    prev = np.nan
    for i in range(window, len(X) + 1):
        w = X.iloc[i - window:i].to_numpy()
        n, k = w.shape
        Y = w[p:]
        Z = np.column_stack([np.ones(n - p)] + [w[p - j - 1:n - j - 1] for j in range(p)])
        B = np.linalg.lstsq(Z, Y, rcond=None)[0]               # (1+kp) × k
        A = np.zeros((k * p, k * p))
        A[:k] = np.hstack([B[1 + j * k:1 + (j + 1) * k].T for j in range(p)])
        if p > 1:
            A[k:, :-k] = np.eye(k * (p - 1))
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        t = X.index[i - 1]
        roots[t] = rho
        if rho >= 0.995:                                         # 폭발적이면 직전 값 유지
            out[t] = prev
            continue
        hist = [w[-j - 1] for j in range(p)]
        for _ in range(horizon):
            nxt = B[0] + sum(B[1 + j * k:1 + (j + 1) * k].T @ hist[j] for j in range(p))
            hist = [nxt] + hist[:-1]
        prev = float(hist[0][2])
        out[t] = prev
    rs = pd.Series(out).sort_index()
    rs.index = pd.PeriodIndex(rs.index, freq="Q")
    rs = rs.rolling(4, min_periods=1).mean()                    # 창 이동에 따른 잡음 완화
    return RStar("var", METHODS["var"], rs,
                 {"VAR 시차": p, "이동 창(분기)": window, "예측 시계(분기)": horizon,
                  "최근 최대 고유값": roots[max(roots)] if roots else np.nan},
                 [f"Lubik·Matthes(2015)의 시변 모수 VAR 대신 {window}분기 이동 창 VAR({p})(GDP 성장률·근원물가 상승률·실질 정책금리)로 "
                  f"매 분기 {horizon}분기(5년) 뒤 실질금리를 예측해 중립금리로 봄(충격이 사라진 뒤 경제가 돌아가는 실질금리). "
                  "창 이동에 따른 잡음을 줄이려 4분기 이동평균. 추정 VAR 이 폭발적이면 직전 값을 씀."])


def market(df: pd.DataFrame, term_premium: float = 0.8) -> RStar:
    f = (10 * df["y10"] - 5 * df["y5"]) / 5                    # 5년 후 5년 선도 명목금리(만기수익률 근사)
    f = f.dropna()
    tgt = inflation_target(f.index)
    rs = f - tgt - term_premium
    return RStar("market", METHODS["market"], rs, {"기간 프리미엄 가정(%p)": term_premium},
                 ["국고채 5년·10년 금리로 5년 후 5년 선도금리를 구하고(10×y10 − 5×y5)/5, 장기 기대인플레이션(물가목표)과 "
                  f"기간 프리미엄({term_premium:g}%p, 가정)을 빼서 시장이 내다보는 장기 실질 단기금리를 구함. " + TARGET_NOTE,
                  "기간 프리미엄을 모형으로 추정하지 않아 수준은 가정에 좌우됨(가정을 1%p 바꾸면 그대로 1%p 이동). 추세와 변화 방향을 "
                  "보는 지표로 쓰는 것이 적절. 만기수익률을 영 쿠폰 금리로 근사함."],
                 {"선도금리": pd.DataFrame({"5년 후 5년 선도금리(%)": f, "물가목표(%)": tgt})})


def growth_linked(df: pd.DataFrame, gpot: pd.Series, base: tuple[str, str] = ("2001Q1", "2019Q4")) -> RStar:
    """오일러 방정식 r* = σ·g + ρ 의 σ·ρ 를 실질 정책금리를 잠재성장률에 회귀해 정함(평균적으로 정책이 중립이었다고 보는 가정)."""
    r = real_rate(df)
    g = gpot.reindex(r.index)
    x = pd.DataFrame({"r": r, "g": g}).loc[base[0]:base[1]].dropna()
    sigma, rho = np.polyfit(x["g"], x["r"], 1)
    rs = (rho + sigma * gpot).dropna()
    rs = rs.loc[r.index[0]:r.index[-1]]
    return RStar("growth", METHODS["growth"], rs,
                 {"성장률 계수 σ": sigma, "상수 ρ(%)": rho, "회귀 기간": f"{base[0]}~{base[1]}"},
                 ["오일러 방정식(r* = σ·g + ρ): 중립금리가 잠재성장률과 함께 움직인다고 보고, 실질 정책금리를 잠재성장률(잠재성장률 화면의 "
                  f"한국은행 4개 모형 평균, 전년동기비)에 {base[0]}~{base[1]} 회귀해 σ={sigma:.2f}, ρ={rho:+.2f} 를 얻음(그 기간 정책이 "
                  "평균적으로 중립이었다는 가정). 인구구조·안전자산 수요 등 성장률 밖 요인의 변화는 반영하지 못함."])


def ensemble(ests: dict[str, RStar]) -> pd.DataFrame:
    D = pd.concat({e.name: e.rstar for e in ests.values()}, axis=1).sort_index()
    out = D.copy()
    out["평균"] = D.mean(axis=1, skipna=True)
    out["최소"] = D.min(axis=1)
    out["최대"] = D.max(axis=1)
    return out
