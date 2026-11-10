"""잠재 GDP·GDP 갭 추정 방법들 — 통계모형·준구조모형·생산함수모형 3분류.

모든 방법은 y = 100·log(실질 GDP) 를 받아 잠재 GDP(같은 단위의 로그 수준)와 GDP 갭(%)을 돌려준다.
    gap_t = y_t - y*_t   (실제가 잠재보다 1% 높으면 +1)
잠재성장률은 잠재 GDP 수준에서 계산한다(전년동기비 또는 연간).

통계모형 (GDP 시계열의 통계적 성질만 이용)
  linear   선형 추세(구조변화 시점에서 기울기만 바뀌는 연속 꺾은선)
  hp       Hodrick-Prescott 필터 (끝점 보정: AR 예측으로 연장 후 필터)
  hp1      단방향 HP 필터 (각 시점까지의 자료만 쓰는 실시간 추정)
  cf       Christiano-Fitzgerald 대역통과 필터
  hamilton Hamilton(2018) 회귀 필터
  bn       Beveridge-Nelson 필터 (Kamber-Morley-Wong 2018 방식)
  uc       비관측 요인 모형 (평활 추세 + AR(2) 순환, 칼만 필터)
준구조모형 (GDP 갭과 물가·고용·금리의 관계식을 함께 추정)
  lw       Laubach-Williams(2003)형 준구조모형 (IS곡선·필립스곡선·오쿤의 관계식, 칼만 필터 최우추정)
           — 한국은행 권지호 외(2019) 준구조모형을 따름
  mv       다변량 HP 필터 (필립스곡선·오쿤의 법칙; Laxton-Tetlow 1992)
  svar     구조 VAR (Blanchard-Quah 장기 제약)
생산함수모형 (총요소생산성·노동·자본 투입으로 잠재 GDP 구성)
  pf       생산함수 접근 — 한국은행 권지호 외(2019)·이은경 외(2024) 방식
           잠재노동 = 15세이상인구 × 경제활동참가율 추세 × (1-자연실업률) × 평균 근로시간 추세
           자본투입 = 자본서비스 물량지수(추세로 간주), 총요소생산성 추세는 가동률 효과를 걸러낸 뒤 HP
  pfmv     생산함수 다변량 필터 — 한국은행 다변량필터링모형(미 연준 Fleischman-Roberts 2011 수정형)
           생산함수에 필립스곡선·오쿤의 관계식(참가율·고용률·근로시간 갭)을 결합해 상태공간모형으로 동시 추정
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from engines.potential import statespace as SS

METHODS = {
    "linear": "선형 추세(구조변화)",
    "hp": "HP 필터",
    "hp1": "단방향 HP 필터(실시간)",
    "cf": "대역통과 필터(CF)",
    "hamilton": "Hamilton 회귀 필터",
    "bn": "Beveridge-Nelson 필터",
    "uc": "비관측 요인 모형(UC)",
    "lw": "준구조모형(LW형: IS·필립스·오쿤)",
    "mv": "다변량 HP 필터(필립스·오쿤)",
    "svar": "구조 VAR(Blanchard-Quah)",
    "pf": "생산함수 접근",
    "pfmv": "생산함수 다변량 필터(FRB/US형)",
}
GROUPS = {"stat": "통계모형", "semi": "준구조모형", "pf": "생산함수모형"}
GROUP_OF = {"linear": "stat", "hp": "stat", "hp1": "stat", "cf": "stat", "hamilton": "stat", "bn": "stat", "uc": "stat",
            "lw": "semi", "mv": "semi", "svar": "semi", "pf": "pf", "pfmv": "pf"}
GROUP = {k: GROUPS[g] for k, g in GROUP_OF.items()}
AVG_KEYS = {"avg": "방법 평균", "avg_stat": "통계모형 평균", "avg_semi": "준구조모형 평균", "avg_pf": "생산함수모형 평균"}


@dataclass
class Estimate:
    key: str
    name: str
    trend: pd.Series                 # 잠재 GDP, 100·log
    gap: pd.Series                   # GDP 갭, %
    params: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------------------
def log100(s: pd.Series) -> pd.Series:
    return 100 * np.log(s.astype(float))


def _d2(n: int):
    return sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))


def hp_array(x: np.ndarray, lam: float) -> np.ndarray:
    n = len(x)
    if n < 4:
        return x.copy()
    D = _d2(n)
    A = sparse.eye(n) + lam * (D.T @ D)
    return spsolve(A.tocsc(), x)


def ar_extend(x: pd.Series, h: int, p: int = 4, window: int = 80) -> pd.Series:
    """로그 수준 x 의 차분에 AR(p)+상수를 최근 window 분기로 맞춰 h 분기 연장."""
    if h <= 0:
        return x
    dx = x.diff().dropna().iloc[-window:].to_numpy()
    X = np.column_stack([np.ones(len(dx) - p)] + [dx[p - j - 1: len(dx) - j - 1] for j in range(p)])
    b = np.linalg.lstsq(X, dx[p:], rcond=None)[0]
    hist = list(dx[-p:])
    lev = x.iloc[-1]
    out = []
    for _ in range(h):
        d = b[0] + sum(b[j + 1] * hist[-j - 1] for j in range(p))
        hist.append(d)
        lev += d
        out.append(lev)
    idx = pd.period_range(x.index[-1] + 1, periods=h, freq=x.index.freq)
    return pd.concat([x, pd.Series(out, index=idx)])


def hp_series(x: pd.Series, lam: float = 1600, extend: int = 0) -> pd.Series:
    s = x.dropna()
    ext = ar_extend(s, extend) if extend else s
    tr = pd.Series(hp_array(ext.to_numpy(), lam), index=ext.index)
    return tr.reindex(s.index).reindex(x.index)


def _finish(key: str, y: pd.Series, trend: pd.Series, **kw) -> Estimate:
    trend = trend.reindex(y.index)
    return Estimate(key, METHODS[key], trend, (y - trend), **kw)


# ---------------------------------------------------------------------------
# 통계모형
# ---------------------------------------------------------------------------
def linear(y: pd.Series, breaks: list[str] | None = None) -> Estimate:
    s = y.dropna()
    t = np.arange(len(s), dtype=float)
    cols = [np.ones_like(t), t]
    used = []
    for b in breaks or []:
        p = pd.Period(b, freq="Q")
        if s.index[0] < p < s.index[-1]:
            tb = float(s.index.get_loc(p))
            cols.append(np.maximum(t - tb, 0))
            used.append(str(p))
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, s.to_numpy(), rcond=None)[0]
    tr = pd.Series(X @ beta, index=s.index)
    slopes = np.cumsum(beta[1:]) * 4
    seg = ", ".join(f"{a}~ {g:.1f}%" for a, g in zip([str(s.index[0])] + used, slopes))
    return _finish("linear", y, tr, params={"구조변화 시점": ", ".join(used) or "없음", "구간별 추세 성장률(연율)": seg},
                   notes=["로그 GDP 를 시간 추세에 회귀. 구조변화 시점에서 기울기만 바뀌고 수준은 이어지는 꺾은선 추세."])


def hp(y: pd.Series, lam: float = 1600, extend: int = 12) -> Estimate:
    tr = hp_series(y, lam, extend)
    note = [f"HP 필터 λ={lam:g}."]
    if extend:
        note.append(f"끝점 편의를 줄이려고 AR(4) 모형으로 {extend}분기를 연장한 뒤 필터를 적용하고 연장분은 버림.")
    return _finish("hp", y, tr, params={"λ": lam, "끝점 연장(분기)": extend}, notes=note)


def hp_onesided(y: pd.Series, lam: float = 1600, min_obs: int = 20) -> Estimate:
    s = y.dropna()
    x = s.to_numpy()
    out = np.full(len(x), np.nan)
    for t in range(min_obs - 1, len(x)):
        out[t] = hp_array(x[: t + 1], lam)[-1]
    tr = pd.Series(out, index=s.index)
    return _finish("hp1", y, tr, params={"λ": lam},
                   notes=["각 분기의 추세를 그 분기까지의 자료로만 HP 필터해 끝값을 취함(당시 시점에서 알 수 있었던 실시간 추정). "
                          "양방향 HP 보다 갭이 실제 GDP 를 늦게 따라감."])


def cf(y: pd.Series, high: int = 32) -> Estimate:
    from statsmodels.tsa.filters.cf_filter import cffilter
    s = y.dropna()
    cyc, _ = cffilter(s.to_numpy(), low=2, high=high, drift=True)
    tr = pd.Series(s.to_numpy() - np.asarray(cyc), index=s.index)
    return _finish("cf", y, tr, params={"잠재로 보는 주기": f"{high}분기 초과"},
                   notes=[f"Christiano-Fitzgerald 비대칭 대역통과 필터(추세 제거 후). 주기 2~{high}분기 변동을 갭으로, "
                          f"{high}분기({high / 4:g}년)를 넘는 저주파 변동을 잠재 GDP 로 봄."])


def hamilton(y: pd.Series, h: int = 8, p: int = 4) -> Estimate:
    s = y.dropna()
    x = s.to_numpy()
    n = len(x)
    rows = range(p - 1, n - h)
    X = np.array([[1.0] + [x[t - j] for j in range(p)] for t in rows])
    Y = np.array([x[t + h] for t in rows])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    fit = np.full(n, np.nan)
    fit[p - 1 + h:] = X @ b
    tr = pd.Series(fit, index=s.index)
    return _finish("hamilton", y, tr, params={"예측 시차 h": h, "설명 시차 p": p},
                   notes=[f"Hamilton(2018): y(t+{h}) 를 상수와 y(t)…y(t-{p - 1}) 에 회귀. 예측치를 추세, 예측오차를 갭으로 봄. "
                          f"추세가 매끄럽지 않아 잠재성장률이 출렁이며, 처음 {h + p - 1}분기는 값이 없음."])


def _bn_given_rho(x: np.ndarray, p: int, rho: float):
    n = len(x)
    T = np.arange(p, n)
    Z = np.column_stack([x[T - j] - x[T - p] for j in range(1, p)]) if p > 1 else np.empty((len(T), 0))
    target = x[T] - rho * x[T - p]
    phi_ = np.linalg.lstsq(Z, target, rcond=None)[0] if p > 1 else np.array([])
    phi = np.append(phi_, rho - phi_.sum())

    F = np.zeros((p, p))
    F[0] = phi
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    M = F @ np.linalg.inv(np.eye(p) - F)
    cyc = np.full(n, np.nan)
    for t in range(p - 1, n):
        Xt = x[t - np.arange(p)]
        cyc[t] = -(M[0] @ Xt)
    ok = np.max(np.abs(np.linalg.eigvals(F))) < 1
    return cyc, phi, ok


def bn(y: pd.Series, p: int = 8, window: int = 40, delta: float = 0.2) -> Estimate:
    s = y.dropna()
    dy = s.diff()
    mu = dy.rolling(window, min_periods=12).mean().bfill()
    x = (dy - mu).dropna()
    rho = 1 - 1 / np.sqrt(delta)
    cyc, phi, ok = _bn_given_rho(x.to_numpy(), p, rho)
    gap = pd.Series(cyc, index=x.index).reindex(s.index)
    tr = s - gap
    notes = ["Kamber·Morley·Wong(2018) BN 필터: GDP 성장률(이동평균 제거)에 AR 모형을 맞추되, 추세 충격과 성장률 충격의 "
             "분산비(신호대잡음비 δ = 1/(1-AR계수합)²)를 고정. δ 가 작을수록 추세가 매끄럽고 갭이 커짐. "
             "갭 = 앞으로 예상되는 성장률이 평균을 웃도는(밑도는) 누적분의 음(-)의 값.",
             f"성장률 평균은 직전 {window}분기 이동평균(추세 성장률의 완만한 하락 반영). 원논문의 베이지안 축소 추정·δ 자동 선택 대신 "
             "δ 를 직접 고르는 제약 최소제곱을 씀."]
    if not ok:
        notes.append("주의: 이 δ 에서 추정된 AR 모형이 안정적이지 않음. δ 나 차수를 바꿔 보세요.")
    return _finish("bn", y, tr, params={"AR 차수": p, "신호대잡음비 δ": delta, "AR 계수 합": float(rho),
                                        "이동평균 기간(분기)": window}, notes=notes)


def uc(y: pd.Series) -> Estimate:
    import statsmodels.api as sm
    s = y.dropna()
    mod = sm.tsa.UnobservedComponents(s.to_numpy(), level=True, stochastic_level=False, trend=True,
                                      stochastic_trend=True, autoregressive=2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(method="powell", disp=False, maxiter=2000)
        res = mod.fit(start_params=res.params, method="lbfgs", disp=False, maxiter=500)
    level = res.level["smoothed"]
    tr = pd.Series(level, index=s.index)
    pn = dict(zip(mod.param_names, res.params))
    ar1, ar2 = pn.get("ar.L1", np.nan), pn.get("ar.L2", np.nan)
    return _finish("uc", y, tr, params={"추세 기울기 충격 표준편차": float(np.sqrt(pn.get("sigma2.trend", np.nan))),
                                        "순환 AR(1)": float(ar1), "순환 AR(2)": float(ar2),
                                        "순환 충격 표준편차": float(np.sqrt(pn.get("sigma2.ar", np.nan))),
                                        "로그우도": float(res.llf)},
                   notes=["로그 GDP = 잠재 GDP + 순환. 잠재 GDP 는 기울기(추세 성장률)가 확률보행하는 평활 추세, 순환은 AR(2). "
                          "칼만 평활기로 전 기간 자료를 써서 추정(최우추정)."])


# ---------------------------------------------------------------------------
# 준구조모형
# ---------------------------------------------------------------------------
def _n_fit(idx: pd.PeriodIndex, fit_end: str | None) -> int:
    """모수 추정에 쓸 관측 수: fit_end(마지막 실적 분기)까지. 그 뒤 전망 구간은 평활에만 씀."""
    if fit_end is None:
        return len(idx)
    return int((idx <= pd.Period(fit_end, freq="Q")).sum())


def _sample(idx: pd.PeriodIndex, nf: int) -> str:
    s = f"{idx[0]}~{idx[nf - 1]}"
    return s if nf == len(idx) else s + f" (모수 추정), {idx[nf]}~{idx[-1]} 전망 구간은 평활에만 사용"


def _inflation(core: pd.Series) -> pd.Series:
    return (core / core.shift(4) - 1) * 100


def mv(y: pd.Series, u: pd.Series, core: pd.Series, lam: float = 1600, lam_u: float = 6400,
       weight: float = 1.0, max_iter: int = 100) -> Estimate:
    """다변량 HP 필터 (Laxton-Tetlow 계열).

    최소화: Σ(y-y*)² + λΣ(Δ²y*)² + wπ Σ e_π² + wu Σ e_u² + λu Σ(Δ²u*)²
      필립스곡선  π_t = a·π_{t-1} + (1-a)·πᵉ_t + κ·gap_{t-1} + e_π   (πᵉ: t-2~t-5 평균, 근원물가 전년동기비)
      오쿤의 법칙 u_t = u*_t − θ·gap_t + e_u
    계수(a, κ, θ)와 가중치(잔차 분산의 역수)는 추정치가 수렴할 때까지 번갈아 갱신.
    """
    pi = _inflation(core)
    pie = (pi.shift(2) + pi.shift(3) + pi.shift(4) + pi.shift(5)) / 4
    df = pd.DataFrame({"y": y, "u": u, "pi": pi, "pil": pi.shift(1), "pie": pie}).dropna()
    n = len(df)
    yv, uv = df["y"].to_numpy(), df["u"].to_numpy()
    piv, pilv, piev = df["pi"].to_numpy(), df["pil"].to_numpy(), df["pie"].to_numpy()
    D = _d2(n).toarray()
    I = np.eye(n)
    S = np.eye(n)[:-1]                   # gap_{t-1} 선택 (t=1..n-1)
    ystar = hp_array(yv, lam)
    ustar = hp_array(uv, lam_u)
    a = kappa = theta = np.nan
    for it in range(max_iter):
        g = yv - ystar
        # 필립스곡선: (π-πᵉ) = a(π₋₁-πᵉ) + κ g₋₁
        Xp = np.column_stack([(pilv - piev)[1:], g[:-1]])
        a, kappa = np.linalg.lstsq(Xp, (piv - piev)[1:], rcond=None)[0]
        a = float(np.clip(a, 0.0, 0.99))
        kappa = float(max(kappa, 0.01))
        e_pi = (piv - piev)[1:] - a * (pilv - piev)[1:] - kappa * g[:-1]
        # 오쿤: (u-u*) = -θ g
        theta = float(max(-np.linalg.lstsq(g[:, None], uv - ustar, rcond=None)[0][0], 0.01))
        e_u = uv - ustar + theta * g
        w_pi = weight * g.var() / max(e_pi.var(), 1e-8)
        w_u = weight * g.var() / max(e_u.var(), 1e-8)
        pit = (piv - a * pilv - (1 - a) * piev)[1:]
        Z = np.zeros_like(D)
        A = np.vstack([
            np.hstack([I, np.zeros((n, n))]),
            np.hstack([np.sqrt(lam) * D, Z]),
            np.sqrt(w_pi) * np.hstack([kappa * S, np.zeros((n - 1, n))]),
            np.sqrt(w_u) * np.hstack([theta * I, I]),
            np.hstack([Z, np.sqrt(lam_u) * D]),
        ])
        b = np.concatenate([yv, np.zeros(n - 2), np.sqrt(w_pi) * (kappa * (S @ yv) - pit),
                            np.sqrt(w_u) * (uv + theta * yv), np.zeros(n - 2)])
        z = np.linalg.solve(A.T @ A, A.T @ b)
        new_y, new_u = z[:n], z[n:]
        diff = max(np.abs(new_y - ystar).max(), np.abs(new_u - ustar).max())
        ystar, ustar = new_y, new_u
        if diff < 1e-7:
            break
    tr = pd.Series(ystar, index=df.index)
    nairu = pd.Series(ustar, index=df.index)
    extra = pd.DataFrame({"실업률(%)": df["u"], "자연실업률 추정(%)": nairu, "근원물가 상승률(%)": df["pi"]})
    return _finish("mv", y, tr, params={"필립스곡선 물가 관성 a": a, "필립스곡선 갭 계수 κ": kappa, "오쿤 계수 θ": theta,
                                        "λ(GDP)": lam, "λ(자연실업률)": lam_u, "물가·고용 가중 배수": weight,
                                        "반복 횟수": it + 1},
                   notes=["다변량 필터: HP 필터의 평활 조건에 필립스곡선(근원물가)과 오쿤의 법칙(실업률) 잔차를 함께 최소화해 "
                          "잠재 GDP 와 자연실업률을 동시에 추정. 물가가 오르는 시기에는 갭이 플러스 쪽으로, 실업률이 자연실업률보다 높으면 "
                          "마이너스 쪽으로 당겨짐.",
                          "계수는 추정 갭을 넣은 최소제곱으로 갱신하며, 이론과 부호가 반대면 0.01 로 하한을 둠. 가중치는 각 식 잔차 분산의 역수 비율."],
                   extra={"자연실업률": extra})


def expected_inflation(df: pd.DataFrame) -> pd.Series:
    """기대인플레이션: 일반인 향후 1년 기대인플레이션(pie)이 있으면 그것, 없으면 근원물가 상승률의 직전 4분기 평균."""
    pi = _inflation(df["core"])
    back = pi.shift(1).rolling(4).mean()
    if "pie" in df.columns and df["pie"].notna().any():
        return df["pie"].combine_first(back)
    return back


def lw(df: pd.DataFrame, start: str = "2000Q1", lam_g: float = 0.06, lam_z: float = 0.03, c: float = 1.0,
       sy_max: float = 0.25, fit_end: str | None = None) -> Estimate:
    """Laubach-Williams(2003)형 준구조모형 (한국은행 권지호 외 2019 준구조모형).

      IS곡선     ỹ_t = a1 ỹ_{t-1} + a2 ỹ_{t-2} + (a_r/2) Σ_{j=1,2} (r_{t-j} − r*_{t-j}) + ε1
      필립스곡선 π_t = b_π π_{t-1} + (1−b_π) πᵉ_t + b_y ỹ_{t-1} + ε2
      오쿤       u_t = u*_t − θ ỹ_t + ε3
      잠재 GDP   y*_t = y*_{t-1} + g_{t-1} + ε4 ,  g_t = g_{t-1} + ε5 ,  σ5 = λ_g σ4
      중립금리   r*_t = c·4g_t + z_t ,  z_t = z_{t-1} + ε6 ,  σ6 = λ_z σ1 / |a_r|
      자연실업률 u*_t = u*_{t-1} + ε7
    r = 기준금리 − 기대인플레이션(실질 기준금리), π = 근원물가 상승률(전년동기비).
    λ_g·λ_z 는 최우추정의 쏠림(pile-up) 문제를 피하려고 고정(LW 2003 의 중위불편 추정 방식 대신).
    """
    y = log100(df["gdp"])
    pi = _inflation(df["core"])
    pe = expected_inflation(df)
    r = df["rate"] - pe
    u = df["urate"]
    d = pd.DataFrame({"y": y, "y1": y.shift(1), "y2": y.shift(2), "r1": r.shift(1), "r2": r.shift(2),
                      "pi": pi, "pi1": pi.shift(1), "pe": pe, "u": u}).loc[start:].dropna()
    Y = d[["y", "pi", "u"]].to_numpy()
    nf = _n_fit(d.index, fit_end)
    hp_y = hp_array(d["y"].to_numpy()[:nf], 1600)
    g0 = float(np.mean(np.diff(hp_y[:8])))
    z0 = float(d["r1"].iloc[:8].mean() - c * 4 * g0)
    us0 = float(hp_array(d["u"].to_numpy()[:nf], 6400)[0])
    m = 10
    y1, y2, r1, r2 = (d[c_].to_numpy() for c_ in ("y1", "y2", "r1", "r2"))
    pi1, pev, yv = d["pi1"].to_numpy(), d["pe"].to_numpy(), d["y"].to_numpy()

    def build(x):
        a1, a2, ar, bpi, by, th = x[:6]
        sIS, spi, sys_, su, sus = np.exp(x[6:11])
        T = np.zeros((m, m))
        T[0, 0] = T[0, 3] = 1
        T[1, 0] = 1
        T[2, 1] = 1
        T[3, 3] = 1
        T[4, 3] = 1
        T[5, 4] = 1
        T[6, 6] = 1
        T[7, 6] = 1
        T[8, 7] = 1
        T[9, 9] = 1
        Q = np.zeros((m, m))
        Q[0, 0] = sys_ ** 2
        Q[3, 3] = (lam_g * sys_) ** 2
        Q[6, 6] = (lam_z * sIS / max(abs(ar), 1e-3)) ** 2
        Q[9, 9] = sus ** 2
        Z = np.zeros((3, m))
        Z[0, 0], Z[0, 1], Z[0, 2] = 1, -a1, -a2
        Z[0, 4] = Z[0, 5] = -(ar / 2) * c * 4
        Z[0, 7] = Z[0, 8] = -(ar / 2)
        Z[1, 1] = -by
        Z[2, 9], Z[2, 0] = 1, th
        D = np.column_stack([a1 * y1 + a2 * y2 + (ar / 2) * (r1 + r2),
                             bpi * pi1 + (1 - bpi) * pev + by * y1,
                             -th * yv])
        H = np.diag([sIS ** 2, spi ** 2, su ** 2])
        a0 = np.array([hp_y[0], hp_y[0] - g0, hp_y[0] - 2 * g0, g0, g0, g0, z0, z0, z0, us0])
        P0 = np.diag([0.5, 0.5, 0.5, 0.15, 0.15, 0.15, 1.0, 1.0, 1.0, 0.5]) ** 2
        return SS.SSM(Z, D, H, T, np.zeros(m), Q, a0, P0)

    def build_s(x):
        if x[0] + x[1] >= 0.98:            # 순환의 정상성
            raise ValueError
        return build(x)

    x0 = np.array([1.2, -0.3, -0.05, 0.8, 0.1, 0.2, np.log(0.5), np.log(0.3), np.log(0.8 * sy_max), np.log(0.2),
                   np.log(0.1)])
    bounds = ([(-0.5, 1.7), (-0.95, 0.5), (-1.0, -0.0025), (0.0, 0.99), (0.025, 1.0), (0.01, 1.0)] + [(-6, 2)] * 2
              + [(-6, np.log(sy_max))] + [(-6, 2)] * 2)
    xh, ll, _ = SS.fit(Y[:nf], build_s, x0, bounds)
    st, _, _ = SS.ksmooth(Y, build(xh))
    idx = d.index
    ystar = pd.Series(st[:, 0], index=idx)
    g = pd.Series(st[:, 3], index=idx)
    rstar = c * 4 * g + pd.Series(st[:, 6], index=idx)
    ustar = pd.Series(st[:, 9], index=idx)
    est = _finish("lw", y, ystar)
    a1, a2, ar, bpi, by, th = xh[:6]
    sd = np.exp(xh[6:11])
    est.params = {"IS: 갭 AR(1) a1": a1, "IS: 갭 AR(2) a2": a2, "IS: 실질금리 갭 계수 a_r": ar,
                  "필립스: 물가 관성 b_π": bpi, "필립스: 갭 계수 b_y": by, "오쿤 계수 θ": th,
                  "IS 충격 표준편차": sd[0], "물가 충격 표준편차": sd[1], "잠재 GDP 충격 표준편차": sd[2],
                  "λ_g(고정)": lam_g, "λ_z(고정)": lam_z, "중립금리의 성장률 계수 c(고정)": c,
                  "잠재 GDP 수준 충격 표준편차 상한": sy_max,
                  "표본": _sample(idx, nf), "로그우도": ll}
    est.notes = ["Laubach·Williams(2003) 준구조모형을 한국은행 권지호 외(2019)처럼 오쿤의 관계식을 더해 수정한 모형. "
                 "IS곡선(실질금리 갭 → GDP 갭), 필립스곡선(GDP 갭 → 근원물가), 오쿤의 관계식(GDP 갭 → 실업률)을 상태공간모형으로 묶어 "
                 "칼만 필터 최우추정 후 평활기로 잠재 GDP·잠재성장률·중립금리·자연실업률을 동시에 추정.",
                 "실질금리 = 한국은행 기준금리 − 기대인플레이션(한국은행 소비자동향조사 향후 1년, 2002년 이전은 근원물가 상승률 직전 4분기 평균). "
                 "물가는 농산물·석유류 제외 근원물가 전년동기비. 원논문의 전문가 5년 기대인플레이션 대신 일반인 1년 기대를 씀.",
                 f"잠재성장률 충격과 중립금리 충격의 상대 크기(λ_g={lam_g:g}, λ_z={lam_z:g})는 고정. 값을 키우면 잠재성장률·중립금리가 더 자주 변함. "
                 f"최우추정이 GDP 변동을 잠재 GDP 수준 충격으로 돌리는 경향이 있어 그 표준편차 상한을 {sy_max:g}%로 둠(상한을 높이면 잠재성장률이 출렁임)."]
    est.extra = {"중립금리·자연실업률": pd.DataFrame({
        "실질 기준금리(%)": r.reindex(idx), "중립금리 추정(%)": rstar, "잠재성장률(연율, %)": 4 * g,
        "실업률(%)": u.reindex(idx), "자연실업률 추정(%)": ustar,
        "근원물가 상승률(%)": pi.reindex(idx), "기대인플레이션(%)": pe.reindex(idx)})}
    return est


def svar(y: pd.Series, u: pd.Series, p: int = 4) -> Estimate:
    from statsmodels.tsa.api import VAR
    d = pd.DataFrame({"y": y, "u": u}).dropna()
    dy = d["y"].diff()
    d = pd.DataFrame({"dy": dy, "u": d["u"], "y": d["y"]}).dropna()
    t = np.arange(len(d), dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    detr = {}
    for c in ["dy", "u"]:
        b = np.linalg.lstsq(X, d[c].to_numpy(), rcond=None)[0]
        detr[c] = d[c].to_numpy() - X @ b
    Z = np.column_stack([detr["dy"], detr["u"]])
    res = VAR(Z).fit(p)
    A = res.coefs                                   # (p, 2, 2)
    A1 = np.eye(2) - A.sum(axis=0)
    C1 = np.linalg.inv(A1)
    Sig = res.sigma_u
    LR = np.linalg.cholesky(C1 @ Sig @ C1.T)
    B0 = A1 @ LR                                    # 충격 1: 공급(영구), 충격 2: 수요(일시)
    e = np.linalg.solve(B0, res.resid.T).T          # 구조 충격
    n = len(e)
    xd = np.zeros((n + p, 2))
    for t_ in range(n):
        xd[t_ + p] = sum(A[i] @ xd[t_ + p - i - 1] for i in range(p)) + B0[:, 1] * e[t_, 1]
    dem_dy = xd[p:, 0]
    idx = d.index[p:]
    gap = pd.Series(np.cumsum(dem_dy), index=idx)
    gap = gap - gap.mean()                          # 초기값 의존 제거: 표본 평균 0
    tr = (d["y"].reindex(idx) - gap)
    lr_dem = float((C1 @ B0)[0, 1])
    return _finish("svar", y, tr, params={"VAR 시차": p, "표본": f"{idx[0]}~{idx[-1]}",
                                          "수요 충격의 GDP 장기 효과(0 이어야 함)": lr_dem},
                   notes=["Blanchard-Quah(1989): GDP 성장률과 실업률(각각 선형 추세 제거)의 VAR(4). 수요 충격은 GDP 수준에 장기 영향이 없다는 "
                          "제약으로 공급·수요 충격을 식별. 수요 충격이 만든 GDP 변동의 누적을 갭으로, 나머지를 잠재 GDP 로 봄.",
                          "갭은 표본 평균이 0 이 되도록 맞춤(초기값 영향 제거)."])


# ---------------------------------------------------------------------------
# 생산함수모형
# ---------------------------------------------------------------------------
def _capital(df: pd.DataFrame, kind: str) -> tuple[pd.Series, str]:
    has = lambda c: c in df.columns and df[c].notna().sum() > 40  # noqa: E731
    if kind == "ksvc" and has("ksvc"):
        return log100(df["ksvc"]), "자본서비스 물량지수(전산업, 연간 → 분기 Denton 보간)"
    if has("capital"):
        return log100(df["capital"]).shift(1), "실질 생산자본스톡(직전 분기말)"
    return log100(df["ksvc"]), "자본서비스 물량지수"


def _hours(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    if "hours" in df.columns and df["hours"].notna().sum() > 40:
        return log100(df["hours"]), True
    return pd.Series(0.0, index=df.index), False


HOURS_NOTE = ("평균 근로시간: OECD 취업자 1인당 연간 실근로시간(2008년~), 그 이전은 Penn World Table 증가율로 연장하고, 당해 연도는 "
              "사업체노동력조사 근로시간 증가율로 연장한 뒤 Denton 방식으로 분기화(한국은행 2024년은 경제활동인구조사 미시자료로 구성).")


def pf(df: pd.DataFrame, labshare: float = 0.61, lam: float = 1600, extend: int = 12,
       nairu: pd.Series | None = None, capital: str = "ksvc", labour: str = "lfpr", cu_adjust: bool = True,
       tfp_cut: str | None = None) -> Estimate:
    """생산함수 접근 (한국은행 권지호 외 2019, 이은경 외 2024).

      Y = A · L^s · K^(1-s)      (s: 노동소득분배율)
      잠재 GDP = 총요소생산성 추세 + s·잠재노동 + (1-s)·자본서비스        (로그)
      잠재노동 = 15세이상인구 × 경제활동참가율 추세 × (1−자연실업률) × 평균 근로시간 추세   (labour="lfpr", 2019년 방식)
               = 15세이상인구 × 고용률 추세 × 평균 근로시간 추세                       (labour="emprate", 2024년 방식)
      자본서비스 물량지수는 전부 추세로 간주(CBO 2001).
      총요소생산성 = 솔로 잔차 (실제 노동 = 취업자 × 평균 근로시간). 추세는 ① HP → ② 순환을 자기 시차·가동률(현재·직전)에 회귀
      → ③ 가동률 효과를 뺀 TFP → ④ 다시 HP (Chalaux·Guillemette 2019).
    """
    alpha = 1 - labshare
    k, ksrc = _capital(df, capital)
    lh, has_h = _hours(df)
    d = pd.DataFrame({"gdp": df["gdp"], "emp": df["emp"], "lf": df["lf"], "pop15": df["pop15"],
                      "urate": df["urate"], "k": k, "lh": lh}).dropna()
    y = log100(d["gdp"])
    k = d["k"]
    pop = log100(d["pop15"])
    l = log100(d["emp"]) + d["lh"]
    tfp = y - alpha * k - labshare * l
    tfp_adj, cu_info = tfp, "가동률 조정 없음"
    if cu_adjust and "capu" in df.columns and df["capu"].reindex(d.index).notna().sum() > 40:
        cyc = tfp - hp_series(tfp, lam, extend)
        cu = df["capu"].reindex(d.index)
        cu = cu - cu.mean()
        R = pd.DataFrame({"c": cyc, "c1": cyc.shift(1), "cu": cu, "cu1": cu.shift(1)}).dropna()
        X = np.column_stack([np.ones(len(R)), R["c1"], R["cu"], R["cu1"]])
        b = np.linalg.lstsq(X, R["c"].to_numpy(), rcond=None)[0]
        eff = (b[2] * cu + b[3] * cu.shift(1)).fillna(b[2] * cu).fillna(0)
        tfp_adj = tfp - eff
        cu_info = f"가동률 계수(현재 {b[2]:.3f}, 직전 {b[3]:.3f})"
    cut = pd.Period(tfp_cut, freq="Q") if tfp_cut else None
    if cut is not None and cut < d.index[-1]:
        # 전망 구간의 성장(반도체 호황 등)을 순환으로 보고 TFP 추세는 실적 구간까지만 추정한 뒤 최근 3년 기울기로 연장
        part = hp_series(tfp_adj.loc[:cut], lam, max(extend, 12))
        sl = float(part.diff().iloc[-12:].mean())
        after = d.index[d.index > cut]
        tfp_t = pd.concat([part, pd.Series(part.iloc[-1] + sl * np.arange(1, len(after) + 1), index=after)])
    else:
        tfp_t = hp_series(tfp_adj, lam, extend)
    h_t = hp_series(d["lh"], lam, extend) if has_h else d["lh"]
    comp_lab = {}
    u_t = er_t = part_t = None
    if labour == "emprate":
        er = d["emp"] / d["pop15"] * 100
        er_t = hp_series(er, lam, extend)
        l_t = pop + log100(er_t / 100) + h_t
        comp_lab["노동: 고용률"] = log100(er_t / 100)
        nsrc = "고용률(취업자/15세이상인구)의 HP 추세"
    else:
        part = d["lf"] / d["pop15"] * 100
        part_t = hp_series(part, lam, extend)
        if nairu is None:
            u_t = hp_series(d["urate"], lam * 4, extend)
            nsrc = f"실업률의 HP 추세(λ={lam * 4:g})"
        else:
            u_t = nairu.reindex(d.index).interpolate(limit_direction="both")
            nsrc = "필립스곡선을 넣은 모형의 자연실업률"
        l_t = pop + log100(part_t / 100) + log100(1 - u_t / 100) + h_t
        comp_lab["노동: 경제활동참가율"] = log100(part_t / 100)
        comp_lab["노동: 1-자연실업률"] = log100(1 - u_t / 100)
    ystar = tfp_t + alpha * k + labshare * l_t
    est = _finish("pf", log100(df["gdp"]), ystar)
    comp = pd.DataFrame({"총요소생산성": tfp_t.diff(4), "자본": alpha * k.diff(4),
                         "노동: 15세이상인구": labshare * pop.diff(4)})
    for name, s_ in comp_lab.items():
        comp[name] = labshare * s_.diff(4)
    comp["노동: 근로시간"] = labshare * h_t.diff(4)
    comp["잠재성장률(로그 근사)"] = comp.sum(axis=1, min_count=len(comp.columns))
    inputs = pd.DataFrame({"실제 TFP(로그×100)": tfp, "가동률 조정 TFP": tfp_adj, "추세 TFP": tfp_t,
                           "자본투입(로그×100)": k, "실제 노동투입(로그×100)": l, "잠재노동(로그×100)": l_t,
                           "경제활동참가율(%)": d["lf"] / d["pop15"] * 100,
                           "고용률(%)": d["emp"] / d["pop15"] * 100,
                           "실업률(%)": d["urate"]})
    if has_h:
        inputs["평균 근로시간(연간, 시간)"] = np.exp(d["lh"] / 100)
        inputs["평균 근로시간 추세"] = np.exp(h_t / 100)
    if labour == "emprate":
        inputs["고용률 추세(%)"] = er_t
    else:
        inputs["경제활동참가율 추세(%)"] = part_t
        inputs["자연실업률(%)"] = u_t
    est.params = {"노동소득분배율": labshare, "HP λ": lam, "끝점 연장(분기)": extend, "자본투입": ksrc,
                  "잠재노동 정의": "15세이상인구×고용률 추세×근로시간 추세" if labour == "emprate"
                  else "15세이상인구×참가율 추세×(1-자연실업률)×근로시간 추세",
                  "자연실업률·고용률": nsrc, "TFP 가동률 조정": cu_info,
                  "TFP 추세": f"{cut} 까지 추정 후 최근 3년 기울기로 연장(전망 구간 성장은 순환으로 간주)"
                  if cut is not None and cut < d.index[-1] else "전 구간 HP"}
    lab_note = ("잠재노동 = 15세이상인구 × 고용률 추세 × 평균 근로시간 추세(한국은행 2024년 방식). " if labour == "emprate" else
                f"잠재노동 = 15세이상인구 × 경제활동참가율 추세 × (1-자연실업률) × 평균 근로시간 추세(한국은행 2019년 방식). 자연실업률: {nsrc}. ")
    est.notes = [f"콥-더글러스 생산함수 Y = A·L^s·K^(1-s), 노동소득분배율 s={labshare:g}(한국은행 2019년 논고와 같은 값). "
                 "총요소생산성(TFP)은 솔로 잔차이며 실제 노동투입 = 취업자 × 평균 근로시간(총근로시간).",
                 lab_note + "15세이상인구는 구조적 요인으로 보아 그대로 추세로 씀. 노동의 질(성·연령·학력 구성)은 미시자료가 필요해 넣지 않았고 "
                 "그 효과는 TFP 에 섞여 있음.",
                 f"자본투입: {ksrc}. 자본서비스는 자산별 생산자본스톡을 사용자비용으로 가중한 물량(한국은행 국민대차대조표, 전산업 연쇄 통크비스트)이며 "
                 "전부 추세로 간주. 마지막 공표 연도 이후는 최근 3년 평균 증가율로 연장.",
                 HOURS_NOTE if has_h else "근로시간 자료가 없어 취업자 수 기준으로 계산함.",
                 f"TFP 추세: HP 필터 순환 부분 중 제조업 가동률(현재·직전 분기)로 설명되는 부분을 걷어낸 뒤 다시 HP 필터({cu_info})."
                 + (f" 전망 구간의 성장은 순환(수요 압력)으로 보아 TFP 추세는 {cut}까지 실적으로만 추정하고 그 뒤는 최근 3년 추세 기울기로 연장."
                    if cut is not None and cut < d.index[-1] else "")]
    est.extra = {"잠재성장률 기여도": comp, "생산함수 투입": inputs}
    return est


def _ar2_var(f1: float, f2: float) -> float:
    """AR(2) 무조건 분산 / 충격 분산."""
    den = (1 + f2) * ((1 - f2) ** 2 - f1 ** 2)
    return float((1 - f2) / den) if den > 1e-6 else 1e6


def pfmv(df: pd.DataFrame, labshare: float = 0.61, lam: float = 1600, start: str = "2000Q1",
         capital: str = "ksvc", rho_max: float = 0.9, fit_end: str | None = None,
         tfp_cut: str | None = None) -> Estimate:
    """생산함수 다변량 필터 (한국은행 다변량필터링모형; 미 연준 Fleischman-Roberts 2011 의 수정형).

    관측식 (모두 100·log, 갭 ỹ 는 AR(2))
      y − (1-s)k − s·n = a* + s(p* + e* + h*) + ỹ + ε1          (생산함수: n=15세이상인구, k=자본서비스)
      p  = p* + β_p ỹ + ε2      (경제활동참가율)
      e  = e* + β_e ỹ + ε3      (고용률 = 1−실업률)
      h  = h* + β_h ỹ + ε4      (평균 근로시간)
      π  = b_π π_{-1} + (1−b_π) πᵉ + κ ỹ_{-1} + ε5    (필립스곡선)
    상태식: a*·p*·h* 는 기울기가 확률보행하는 평활 추세(기울기 충격 = 해당 식 충격/√λ), e* 는 확률보행.
    """
    s = labshare
    k, ksrc = _capital(df, capital)
    lh, has_h = _hours(df)
    pi = _inflation(df["core"])
    pe = expected_inflation(df)
    d = pd.DataFrame({"y": log100(df["gdp"]), "k": k, "n": log100(df["pop15"]),
                      "p": log100(df["lf"] / df["pop15"]), "e": log100(1 - df["urate"] / 100), "h": lh,
                      "pi": pi, "pi1": pi.shift(1), "pe": pe}).loc[start:].dropna()
    n_ = len(d)
    x1 = d["y"] - (1 - s) * d["k"] - s * d["n"]
    Y = np.column_stack([x1, d["p"], d["e"], d["h"], d["pi"]])
    if not has_h:
        Y[:, 3] = np.nan
    nf = _n_fit(d.index, fit_end)
    hp_p, hp_e, hp_h = (hp_array(d[c].to_numpy()[:nf], lam) for c in ("p", "e", "h"))
    a_init = hp_array((x1 - s * (d["p"] + d["e"] + d["h"])).to_numpy()[:nf], lam)
    m = 9
    pi1, pev = d["pi1"].to_numpy(), d["pe"].to_numpy()

    def build(x):
        f1, f2, bp, be, bh, kap, bpi = x[:7]
        sc, s1, sp, se, sh, spi, ses, sgp, sgh = np.exp(x[7:16])
        T = np.zeros((m, m))
        T[0, 0], T[0, 1], T[1, 0] = f1, f2, 1
        T[2, 2] = T[2, 3] = T[3, 3] = 1
        T[4, 4] = T[4, 5] = T[5, 5] = 1
        T[6, 6] = 1
        T[7, 7] = T[7, 8] = T[8, 8] = 1
        g0 = _ar2_var(f1, f2) * sc ** 2                 # 갭의 무조건 분산
        Q = np.diag([sc ** 2, 0, 0, g0 / lam, 0, sgp ** 2, ses ** 2, 0, sgh ** 2])
        Z = np.zeros((5, m))
        Z[0, 0], Z[0, 2], Z[0, 4], Z[0, 6], Z[0, 7] = 1, 1, s, s, s
        Z[1, 4], Z[1, 0] = 1, bp
        Z[2, 6], Z[2, 0] = 1, be
        Z[3, 7], Z[3, 0] = 1, bh
        Z[4, 1] = kap
        D = np.zeros((n_, 5))
        D[:, 4] = bpi * pi1 + (1 - bpi) * pev
        H = np.diag([s1 ** 2, sp ** 2, se ** 2, sh ** 2, spi ** 2])
        a0 = np.array([0, 0, a_init[0], a_init[1] - a_init[0], hp_p[0], hp_p[1] - hp_p[0], hp_e[0], hp_h[0],
                       hp_h[1] - hp_h[0]])
        P0 = np.diag([1.0, 1.0, 1.0, 0.2, 0.5, 0.05, 0.3, 0.5, 0.05]) ** 2
        return SS.SSM(Z, D, H, T, np.zeros(m), Q, a0, P0)

    def build_s(x):
        if x[0] + x[1] >= rho_max or x[1] - x[0] >= 0.98:
            raise ValueError
        return build(x)

    x0 = np.array([1.0, -0.2, 0.2, 0.2, 0.1, 0.1, 0.8] + list(np.log([0.6, 0.3, 0.3, 0.2, 0.5, 0.3, 0.1, 0.02, 0.02])))
    bounds = ([(-0.5, 1.8), (-0.95, 0.5), (0.0, 1.5), (0.0, 1.5), (0.0, 1.5), (0.01, 1.0), (0.0, 0.99)] + [(-7, 2)] * 6 + [(-7, np.log(0.08))]
              + [(np.log(0.002), np.log(0.04))] * 2)
    xh, ll, _ = SS.fit(Y[:nf], build_s, x0, bounds)
    Ys = Y.copy()
    if tfp_cut:
        # 전망 구간의 산출(생산함수 관측식)은 평활에 쓰지 않음 → TFP 추세는 실적 구간의 기울기로 이어지고 전망 성장은 갭으로
        Ys[int((d.index <= pd.Period(tfp_cut, freq="Q")).sum()):, 0] = np.nan
    st, _, _ = SS.ksmooth(Ys, build(xh))
    idx = d.index
    S = lambda j: pd.Series(st[:, j], index=idx)  # noqa: E731
    a_, p_, e_, h_ = S(2), S(4), S(6), S(7)
    ystar = a_ + (1 - s) * d["k"] + s * (d["n"] + p_ + e_ + h_)
    est = _finish("pfmv", log100(df["gdp"]), ystar)
    comp = pd.DataFrame({"총요소생산성": a_.diff(4), "자본": (1 - s) * d["k"].diff(4),
                         "노동: 15세이상인구": s * d["n"].diff(4), "노동: 경제활동참가율": s * p_.diff(4),
                         "노동: 1-자연실업률": s * e_.diff(4), "노동: 근로시간": s * h_.diff(4)})
    comp["잠재성장률(로그 근사)"] = comp.sum(axis=1, min_count=len(comp.columns))
    f1, f2, bp, be, bh, kap, bpi = xh[:7]
    sd = np.exp(xh[7:16])
    est.params = {"노동소득분배율": s, "갭 AR(1)": f1, "갭 AR(2)": f2, "오쿤: 참가율 β_p": bp, "오쿤: 고용률 β_e": be,
                  "오쿤: 근로시간 β_h": bh, "필립스: 갭 계수 κ": kap, "필립스: 물가 관성 b_π": bpi,
                  "갭 충격 표준편차": sd[0], "고용률 추세 충격 표준편차": sd[6],
                  "참가율 추세 기울기 충격 표준편차": sd[7], "근로시간 추세 기울기 충격 표준편차": sd[8],
                  "TFP 추세 평활 λ": lam, "갭 지속성 상한(AR 계수 합)": rho_max,
                  "자본투입": ksrc, "표본": _sample(idx, nf), "로그우도": ll}
    est.notes = ["한국은행 다변량필터링모형(권지호 외 2019; 미 연준 FRB/US 공급부문의 Fleischman·Roberts 2011 모형 수정형)을 따름. "
                 "생산함수 접근과 같이 잠재 GDP = TFP 추세 + 노동소득분배율×(15세이상인구 + 참가율 추세 + 고용률 추세 + 근로시간 추세) "
                 "+ 자본몫×자본서비스 이나, 투입요소별 추세·순환을 따로 HP 필터하지 않고 하나의 상태공간모형으로 동시에 추정(칼만 필터 최우추정).",
                 "오쿤의 관계식: 경제활동참가율·고용률(1-실업률)·평균 근로시간의 순환 부분이 GDP 갭에 비례. 필립스곡선: 근원물가 상승률이 "
                 "직전 분기 GDP 갭에 반응(기대인플레이션은 소비자동향조사 향후 1년, 2002년 이전은 근원물가 직전 4분기 평균).",
                 f"TFP 추세 기울기 충격 분산은 GDP 갭 분산의 1/λ(λ={lam:g})로 고정해 HP 필터와 같은 평활도를 주고, 참가율·근로시간 추세의 기울기 충격은 "
                 f"최우추정. 갭이 단위근에 가깝게 추정돼 추세 변화를 흡수하지 않도록 AR 계수 합을 {rho_max:g} 이하로 제한. "
                 "원논문의 공급충격 변수와 노동의 질은 넣지 않음.",
                 f"자본투입: {ksrc}(추세로 간주). " + (HOURS_NOTE if has_h else "근로시간 자료가 없어 근로시간 식은 뺌.")]
    if tfp_cut and pd.Period(tfp_cut, freq="Q") < idx[-1]:
        est.params["TFP 추세"] = f"{tfp_cut} 이후 산출 관측은 평활에 쓰지 않음(그 뒤 성장은 순환으로 간주)"
        est.notes.append(f"TFP 추세: {tfp_cut} 이후의 산출(생산함수 관측식)은 평활에 쓰지 않아, TFP 추세는 그때까지의 기울기로 이어지고 "
                         "그 뒤 성장(반도체 호황 등)은 GDP 갭으로 잡힘. 노동 투입 추세는 전망 자료를 그대로 씀.")
    ustar = 100 * (1 - np.exp(e_ / 100))
    est.extra = {"잠재성장률 기여도": comp,
                 "자연실업률": pd.DataFrame({"실업률(%)": df["urate"].reindex(idx), "자연실업률 추정(%)": ustar,
                                        "근원물가 상승률(%)": pi.reindex(idx)})}
    return est


# ---------------------------------------------------------------------------
# 결합·지표
# ---------------------------------------------------------------------------
def ensemble(y: pd.Series, ests: list[Estimate], key: str = "avg", note: str | None = None) -> Estimate:
    G = pd.concat([e.gap for e in ests], axis=1)
    gap = G.mean(axis=1, skipna=True)
    tr = y - gap
    return Estimate(key, AVG_KEYS[key], tr, gap, params={"포함 방법": ", ".join(x.name for x in ests)},
                    notes=[note or "포함된 방법들의 GDP 갭을 단순 평균하고, 잠재 GDP = 실제 GDP − 평균 갭 으로 계산."])


def yoy(x100: pd.Series) -> pd.Series:
    return (np.exp((x100 - x100.shift(4)) / 100) - 1) * 100


def annual_growth(x100: pd.Series) -> pd.Series:
    lev = np.exp(x100 / 100)
    yr = lev.groupby(lev.index.year)
    tot = yr.sum(min_count=4)
    tot[yr.count() < 4] = np.nan
    return (tot / tot.shift(1) - 1) * 100


def annual_mean(s: pd.Series) -> pd.Series:
    g = s.groupby(s.index.year)
    m = g.mean()
    m[g.count() < 4] = np.nan
    return m
