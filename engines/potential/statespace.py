"""작은 선형 가우시안 상태공간 모형용 칼만 필터·평활기 (결측 관측 허용).

  관측식  y_t = Z a_t + d_t + e_t ,  e_t ~ N(0, H)
  상태식  a_{t+1} = T a_t + c + η_t , η_t ~ N(0, Q)

d_t 는 시점마다 다른 관측 절편(시차 관측치 등 외생 항)을 담는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SSM:
    Z: np.ndarray        # (p, m)
    d: np.ndarray        # (n, p)
    H: np.ndarray        # (p, p)
    T: np.ndarray        # (m, m)
    c: np.ndarray        # (m,)
    Q: np.ndarray        # (m, m)
    a0: np.ndarray       # (m,)  첫 시점 상태의 사전 평균
    P0: np.ndarray       # (m, m)


def kfilter(y: np.ndarray, M: SSM, burn: int = 0):
    """칼만 필터. y: (n, p), 결측은 NaN. 반환: (로그우도, 예측 a, 예측 P, 필터 a, 필터 P)."""
    n, p = y.shape
    m = M.T.shape[0]
    a, P = M.a0.copy(), M.P0.copy()
    ap = np.zeros((n, m)); Pp = np.zeros((n, m, m))
    af = np.zeros((n, m)); Pf = np.zeros((n, m, m))
    ll = 0.0
    for t in range(n):
        ap[t], Pp[t] = a, P
        obs = ~np.isnan(y[t])
        if obs.any():
            Z = M.Z[obs]
            v = y[t, obs] - Z @ a - M.d[t, obs]
            F = Z @ P @ Z.T + M.H[np.ix_(obs, obs)]
            try:
                Fi = np.linalg.inv(F)
                sign, logdet = np.linalg.slogdet(F)
            except np.linalg.LinAlgError:
                return -1e12, ap, Pp, af, Pf
            if sign <= 0:
                return -1e12, ap, Pp, af, Pf
            K = P @ Z.T @ Fi
            a = a + K @ v
            P = P - K @ Z @ P
            if t >= burn:
                ll += -0.5 * (obs.sum() * np.log(2 * np.pi) + logdet + v @ Fi @ v)
        af[t], Pf[t] = a, P
        a = M.T @ a + M.c
        P = M.T @ P @ M.T.T + M.Q
        P = (P + P.T) / 2
    return ll, ap, Pp, af, Pf


def ksmooth(y: np.ndarray, M: SSM):
    """Rauch-Tung-Striebel 평활. 반환: (평활 상태 (n, m), 평활 분산 (n, m, m), 로그우도)."""
    ll, ap, Pp, af, Pf = kfilter(y, M)
    n, m = af.shape
    as_ = af.copy(); Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        Pn = Pp[t + 1]
        J = Pf[t] @ M.T.T @ np.linalg.pinv(Pn)
        as_[t] = af[t] + J @ (as_[t + 1] - ap[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pn) @ J.T
    return as_, Ps, ll


def fit(y: np.ndarray, build, x0: np.ndarray, bounds, burn: int = 4, maxiter: int = 400, rounds: int = 4):
    """최우추정: build(x) → SSM.

    L-BFGS-B 는 목적함수가 크면 상대 감소 기준으로 일찍 멈추고도 '성공'을 돌려주는 일이 있어(생산함수 다변량 필터에서 2회 반복 후 정지),
    L-BFGS-B → Powell 을 번갈아 돌려 로그우도가 더 오르지 않을 때까지(0.001 미만) 다듬는다. 반환 결과의 ``rounds`` 에 반복 횟수,
    ``at_bounds`` 에 경계에 걸린 모수 번호를 담는다.
    """
    def nll(x):
        try:
            v = -kfilter(y, build(x), burn)[0]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return 1e12
        return v if np.isfinite(v) else 1e12

    best = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-12})
    r = 0
    for r in range(1, rounds + 1):
        prev = best.fun
        alt = minimize(nll, best.x, method="Powell", bounds=bounds, options={"maxiter": 6000, "xtol": 1e-5, "ftol": 1e-9})
        if alt.fun < best.fun:
            best = alt
        alt = minimize(nll, best.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-12})
        if alt.fun < best.fun:
            best = alt
        if prev - best.fun < 1e-3:
            break
    best.rounds = r
    best.at_bounds = at_bounds(best.x, bounds)
    return best.x, -best.fun, best


def at_bounds(x: np.ndarray, bounds, tol: float = 1e-4) -> list[int]:
    """경계(상·하한)에 걸린 모수 번호."""
    out = []
    for i, (v, (lo, hi)) in enumerate(zip(x, bounds)):
        span = (hi - lo) if (lo is not None and hi is not None) else 1.0
        if (lo is not None and v - lo <= tol * max(span, 1.0)) or (hi is not None and hi - v <= tol * max(span, 1.0)):
            out.append(i)
    return out


def _nll_fn(y: np.ndarray, build, burn: int):
    def nll(x):
        try:
            v = -kfilter(y, build(x), burn)[0]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.nan
        return v if np.isfinite(v) and v < 1e11 else np.nan
    return nll


def hessian(f, x: np.ndarray, idx: list[int], rel: float = 1e-3) -> np.ndarray:
    """f 의 x 에서의 헤시안(idx 모수만, 중앙 차분)."""
    k = len(idx)
    h = np.array([rel * max(1.0, abs(x[i])) for i in idx])
    H = np.zeros((k, k))
    f0 = f(x)

    def fx(steps):
        z = x.copy()
        for i, s in steps:
            z[idx[i]] += s
        return f(z)

    for a in range(k):
        H[a, a] = (fx([(a, h[a])]) - 2 * f0 + fx([(a, -h[a])])) / h[a] ** 2
        for b in range(a + 1, k):
            v = (fx([(a, h[a]), (b, h[b])]) - fx([(a, h[a]), (b, -h[b])])
                 - fx([(a, -h[a]), (b, h[b])]) + fx([(a, -h[a]), (b, -h[b])])) / (4 * h[a] * h[b])
            H[a, b] = H[b, a] = v
    return H


def uncertainty(y_fit: np.ndarray, y_all: np.ndarray, build, xh: np.ndarray, bounds, W: np.ndarray,
                n_draws: int = 200, burn: int = 4, seed: int = 0) -> dict:
    """평활 추정치(상태의 선형결합 W)의 표준오차.

    filter : 모수를 추정치로 아는 것으로 본 칼만 평활 오차   sqrt(w' P_t|n w)
    total  : 모수 불확실성 포함(Hamilton 1986). 최우추정량의 점근 분포 N(θ̂, H⁻¹)에서 모수를 뽑아 평활을 되풀이하고
             분산 = 추정 모수의 평활 분산 + 분산(추출별 평활 평균). 경계(및 정상성 제약)에 걸린 모수는 고정.
             헤시안이 양정치가 아니면 total = None.
    W: (m, k) → k 개 결합. 반환 sd 는 (n, k).
    """
    W = np.asarray(W, float)
    a0, P0, _ = ksmooth(y_all, build(xh))
    filt = np.sqrt(np.maximum(np.einsum("tij,ik,jk->tk", P0, W, W), 0))
    out = {"filter": filt, "total": None, "draws": 0, "free": [], "note": ""}
    edge = set(at_bounds(xh, bounds))
    free = [i for i in range(len(xh)) if i not in edge]
    out["free"] = free
    nll = _nll_fn(y_fit, build, burn)
    # 경계 밖 제약(정상성 등)에 닿아 있는 모수: 조금만 움직여도 우도가 정의되지 않음 → 고정
    f0 = nll(xh)
    for i in list(free):
        hi_ = 1e-3 * max(1.0, abs(xh[i]))
        z1, z2 = xh.copy(), xh.copy()
        z1[i] += hi_
        z2[i] -= hi_
        if not (np.isfinite(f0) and np.isfinite(nll(z1)) and np.isfinite(nll(z2))):
            free.remove(i)
    try:
        for _ in range(4):
            H = hessian(nll, xh, free)
            bad = ~np.isfinite(H)
            if not bad.any():
                break
            # 두 모수를 함께 움직일 때만 제약(예: a1 + a2 < 0.98)에 닿는 경우 → 가장 많이 걸리는 모수를 고정하고 다시
            drop = free[int(np.argmax(bad.sum(axis=1)))]
            free = [i for i in free if i != drop]
        else:
            raise np.linalg.LinAlgError
        H = (H + H.T) / 2
        # 우도가 평평한(식별되지 않는) 모수는 고정: 예) 기대인플레이션 = 직전 분기 물가이면 필립스곡선의 물가 관성 b_π
        flat = np.diag(H) <= 1e-6 * max(np.abs(np.diag(H)).max(), 1e-12)
        if flat.any():
            keep = ~flat
            free = [i for i, k_ in zip(free, keep) if k_]
            H = H[np.ix_(keep, keep)]
        out["free"] = free
        ev = np.linalg.eigvalsh(H)
        if ev.min() <= 0:
            out["note"] = "헤시안이 양정치가 아님(모수 식별이 약함)"
            return out
        C = np.linalg.inv(H)
        Lc = np.linalg.cholesky((C + C.T) / 2)
    except np.linalg.LinAlgError:
        out["note"] = "헤시안 계산 실패"
        return out
    rng = np.random.default_rng(seed)
    means, varis = [], []
    tries = 0
    while len(means) < n_draws and tries < 4 * n_draws:
        tries += 1
        x = xh.copy()
        x[free] = xh[free] + Lc @ rng.standard_normal(len(free))
        if any((lo is not None and x[i] < lo) or (hi is not None and x[i] > hi)
               for i, (lo, hi) in enumerate(bounds)):
            continue
        try:
            a_, P_, _ = ksmooth(y_all, build(x))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        mu = a_ @ W
        v = np.einsum("tij,ik,jk->tk", P_, W, W)
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(v))):
            continue
        means.append(mu)
        varis.append(v)
    if len(means) < 30:
        out["note"] = "모수 모의 추출이 경계·정상성 제약에 너무 많이 걸림"
        return out
    means, varis = np.array(means), np.array(varis)
    # 총분산 = 추정 모수에서의 평활 분산 + 모수 추출에 따른 평활 평균의 분산 (평활 분산을 추출 평균으로 바꾸면
    # 추출 모수에 따라 평활 분산이 작아지는 경우 '모수 불확실성을 더했는데 구간이 좁아지는' 일이 생김)
    out["total"] = np.sqrt(np.maximum(filt ** 2 + means.var(0), 0))
    out["draws"] = len(means)
    out["accept"] = len(means) / tries
    return out


def unc_note(u: dict) -> str:
    if u["total"] is not None:
        return (f"표준오차: 칼만 평활 오차 + 모수 추정 오차(Hamilton 1986 방식, 최우추정량 점근분포에서 {u['draws']}회 추출, "
                "경계·정상성 제약에 걸린 모수는 고정). 그래프의 음영은 90% 구간(±1.645 표준오차).")
    return (f"표준오차: 모수를 아는 것으로 본 칼만 평활 오차만 반영({u['note']}). 모수 추정 오차를 더하면 구간이 더 넓어짐. "
            "그래프의 음영은 90% 구간(±1.645 표준오차).")
