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
