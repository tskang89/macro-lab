"""잠재 GDP·GDP 갭 추정 방법들.

모든 방법은 y = 100·log(실질 GDP) 를 받아 잠재 GDP(같은 단위의 로그 수준)와 GDP 갭(%)을 돌려준다.
    gap_t = y_t - y*_t   (실제가 잠재보다 1% 높으면 +1)
잠재성장률은 잠재 GDP 수준에서 계산한다(전년동기비 또는 연간).

방법
  linear   선형 추세(구조변화 시점에서 기울기만 바뀌는 연속 꺾은선)
  hp       Hodrick-Prescott 필터 (끝점 보정: AR 예측으로 연장 후 필터)
  hp1      단방향 HP 필터 (각 시점까지의 자료만 쓰는 실시간 추정)
  cf       Christiano-Fitzgerald 대역통과 필터 (주기 32분기 넘는 변동만 잠재로)
  hamilton Hamilton(2018) 회귀 필터 (8분기 뒤 GDP 를 현재·과거 4분기로 예측, 예측치 = 추세)
  bn       Beveridge-Nelson 필터 (Kamber-Morley-Wong 2018 방식, 신호대잡음비 제약)
  uc       비관측 요인 모형 (평활 추세 + AR(2) 순환, 칼만 필터·최우추정)
  mv       다변량 필터 (HP + 필립스곡선 + 오쿤의 법칙; 잠재 GDP 와 자연실업률 동시 추정)
  pf       생산함수 접근 (콥-더글러스: 총요소생산성·자본·잠재 노동)
  svar     구조 VAR (Blanchard-Quah 장기 제약: 수요 충격은 GDP 수준에 장기 영향 없음)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

METHODS = {
    "linear": "선형 추세(구조변화)",
    "hp": "HP 필터",
    "hp1": "단방향 HP 필터(실시간)",
    "cf": "대역통과 필터(CF)",
    "hamilton": "Hamilton 회귀 필터",
    "bn": "Beveridge-Nelson 필터",
    "uc": "비관측 요인 모형(UC)",
    "mv": "다변량 필터(필립스·오쿤)",
    "pf": "생산함수 접근",
    "svar": "구조 VAR(Blanchard-Quah)",
}
GROUP = {
    "linear": "통계적 추세", "hp": "통계적 추세", "hp1": "통계적 추세", "cf": "통계적 추세",
    "hamilton": "통계적 추세", "bn": "시계열 모형", "uc": "시계열 모형",
    "mv": "경제 구조 반영", "pf": "경제 구조 반영", "svar": "경제 구조 반영",
}


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
# 통계적 추세
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


# ---------------------------------------------------------------------------
# 시계열 모형
# ---------------------------------------------------------------------------
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
# 경제 구조 반영
# ---------------------------------------------------------------------------
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


def pf(df: pd.DataFrame, alpha: float = 0.35, lam: float = 1600, extend: int = 12,
       nairu: pd.Series | None = None) -> Estimate:
    """콥-더글러스 생산함수: Y = A·K^α·L^(1-α).

    잠재 GDP = 추세 TFP · 실제 자본 · 잠재 노동
    잠재 노동 = 추세 15세 이상 인구 × 추세 경제활동참가율 × (1 − 자연실업률)
    """
    need = ["gdp", "emp", "lf", "pop15", "urate", "capital"]
    d = df[need].copy()
    d["k"] = log100(d["capital"]).shift(1)          # 분기초 자본 = 직전 분기말
    d = d.dropna()
    y = log100(d["gdp"])
    k = d["k"]
    l = log100(d["emp"])
    tfp = y - alpha * k - (1 - alpha) * l
    tfp_t = hp_series(tfp, lam, extend)
    pop_t = hp_series(log100(d["pop15"]), lam, extend)
    part = d["lf"] / d["pop15"] * 100
    part_t = hp_series(part, lam, extend)
    if nairu is None:
        u_t = hp_series(d["urate"], lam * 4, extend)
        nsrc = f"실업률의 HP 추세(λ={lam * 4:g})"
    else:
        u_t = nairu.reindex(d.index).interpolate(limit_direction="both")
        nsrc = "다변량 필터의 자연실업률"
    l_t = pop_t + log100(part_t / 100) + log100(1 - u_t / 100)
    ystar = tfp_t + alpha * k + (1 - alpha) * l_t
    yy = log100(df["gdp"])
    est = _finish("pf", yy, ystar)
    # 잠재성장률 기여도 (전년동기비 로그 차분, %p)
    comp = pd.DataFrame({
        "총요소생산성": tfp_t.diff(4),
        "자본": alpha * k.diff(4),
        "노동: 인구": (1 - alpha) * pop_t.diff(4),
        "노동: 경제활동참가율": (1 - alpha) * log100(part_t / 100).diff(4),
        "노동: 고용률(1-자연실업률)": (1 - alpha) * log100(1 - u_t / 100).diff(4),
    })
    comp["잠재성장률(로그 근사)"] = comp.sum(axis=1, min_count=5)
    inputs = pd.DataFrame({"실제 TFP(로그×100)": tfp, "추세 TFP": tfp_t, "자본(로그×100)": k,
                           "실제 취업자(로그×100)": l, "잠재 취업자(로그×100)": l_t,
                           "경제활동참가율(%)": part, "추세 참가율(%)": part_t,
                           "실업률(%)": d["urate"], "자연실업률(%)": u_t})
    est.params = {"자본소득분배율 α": alpha, "HP λ": lam, "끝점 연장(분기)": extend, "자연실업률": nsrc}
    est.notes = [f"콥-더글러스 생산함수 Y = A·K^α·L^(1-α), α={alpha:g}(자본소득 몫). 총요소생산성(TFP)은 솔로 잔차.",
                 "잠재 GDP = 추세 TFP + α·자본 + (1-α)·잠재 노동 (로그). 자본은 완전가동으로 보고 실제 스톡(직전 분기말)을 그대로 씀.",
                 f"잠재 노동 = 추세 15세 이상 인구 × 추세 경제활동참가율 × (1-자연실업률). 자연실업률: {nsrc}.",
                 "근로시간 자료는 쓰지 않아(취업자 수 기준) 근로시간 감소 효과는 TFP 에 섞여 있음.",
                 "자본스톡은 연말 실질 생산자본스톡(한국은행)을 분기로 로그 선형 보간했고, 마지막 공표 연도 이후는 최근 3년 평균 증가율로 연장."]
    est.extra = {"잠재성장률 기여도": comp, "생산함수 투입": inputs}
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
# 결합·지표
# ---------------------------------------------------------------------------
def ensemble(y: pd.Series, ests: list[Estimate]) -> Estimate:
    G = pd.concat([e.gap for e in ests], axis=1)
    gap = G.mean(axis=1, skipna=True)
    tr = y - gap
    e = Estimate("avg", "방법 평균", tr, gap,
                 params={"포함 방법": ", ".join(x.name for x in ests)},
                 notes=["선택한 방법들의 GDP 갭을 단순 평균하고, 잠재 GDP = 실제 GDP − 평균 갭 으로 계산."])
    return e


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
