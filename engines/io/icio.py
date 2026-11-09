"""국제 산업연관 분석 엔진 (OECD ICIO, 기초가격, 백만 달러).

N = 국가수 × 산업수. 행·열은 국가별 산업 블록 순서.
  x = A x + y          A = Z / x   (국가 간 중간거래 포함)
  L = (I - A)^-1       LU 분해 한 번으로 L·b, L'·c 를 모두 푼다 (역행렬은 만들지 않음)

분석
  demand_effect   : 최종수요 충격 → 각국·산업 산출·부가가치
  tariff_effect   : 관세 → 수입 감소·대체(수요 이동) + 수입국 비용·물가 상승
  supply_effect   : 국가·산업 생산 차질 → 전방(Ghosh)·후방(레온티에프)
  price_effect    : 특정 국가·산업 가격을 외생으로 올림 → 나머지 가격(비용 전가), 각국 물가
  gvc_indicators  : 수출의 부가가치 원천(DVA·FVA), 전방 참여(DVX), 부가가치 기준 교역
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

from engines.io.icio_meta import COUNTRY_GROUPS, COUNTRY_KO, INDUSTRY_KO, industry_group

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "processed", "icio")
FD_CODES = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]
FD_ALL = "ALL"


def available() -> dict[str, list[str]]:
    """판 → 연도 목록."""
    out: dict[str, list[str]] = {}
    for f in glob.glob(os.path.join(DATA, "icio_*_*.npz")):
        _, ed, yr = os.path.basename(f)[:-4].split("_")
        out.setdefault(ed, []).append(yr)
    return {k: sorted(v) for k, v in sorted(out.items())}


@dataclass
class ICIO:
    edition: str
    year: str
    countries: list[str]
    industries: list[str]
    Z: np.ndarray
    Y: np.ndarray
    va: np.ndarray
    tls: np.ndarray
    x: np.ndarray

    # ---------- 기본 ----------
    @property
    def C(self) -> int:
        return len(self.countries)

    @property
    def S(self) -> int:
        return len(self.industries)

    @property
    def N(self) -> int:
        return self.C * self.S

    @cached_property
    def ci(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.countries)}

    @cached_property
    def si(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.industries)}

    def rows(self, country: str) -> slice:
        i = self.ci[country]
        return slice(i * self.S, (i + 1) * self.S)

    def idx(self, country: str, industry: str) -> int:
        return self.ci[country] * self.S + self.si[industry]

    @cached_property
    def row_country(self) -> np.ndarray:
        return np.repeat(np.arange(self.C), self.S)

    @cached_property
    def row_industry(self) -> np.ndarray:
        return np.tile(np.arange(self.S), self.C)

    @cached_property
    def xs(self) -> np.ndarray:
        """나눗셈용 산출 (0 → 1)."""
        return np.where(self.x > 0, self.x, 1.0)

    @cached_property
    def A(self) -> np.ndarray:
        return self.Z / self.xs[None, :]

    @cached_property
    def v(self) -> np.ndarray:
        return np.where(self.x > 0, self.va / self.xs, 0.0)

    @cached_property
    def lu(self):
        return lu_factor(np.eye(self.N) - self.A, overwrite_a=True, check_finite=False)

    def Lmul(self, b: np.ndarray) -> np.ndarray:
        """L @ b."""
        return lu_solve(self.lu, b, check_finite=False)

    def LTmul(self, c: np.ndarray) -> np.ndarray:
        """L' @ c  (행벡터 c'L 의 전치)."""
        return lu_solve(self.lu, c, trans=1, check_finite=False)

    @cached_property
    def Y3(self) -> np.ndarray:
        """최종수요 (N × 국가 × 항목)."""
        return self.Y.reshape(self.N, self.C, len(FD_CODES))

    @cached_property
    def Yd(self) -> np.ndarray:
        """최종수요 (N × 수요국)."""
        return self.Y3.sum(2)

    @cached_property
    def va_country(self) -> np.ndarray:
        return np.bincount(self.row_country, weights=self.va, minlength=self.C)

    @cached_property
    def exports(self) -> np.ndarray:
        """국가별 산업 총수출 (N): 다른 나라로 가는 중간재 + 최종재."""
        Zc = self.Z.reshape(self.N, self.C, self.S).sum(2)          # N × 구매국
        to = Zc + self.Yd
        own = to[np.arange(self.N), self.row_country]
        return to.sum(1) - own

    @cached_property
    def export_matrix(self) -> np.ndarray:
        """E (N × 국가): 열 k 는 k 국 산업들의 총수출(나머지 0)."""
        E = np.zeros((self.N, self.C))
        E[np.arange(self.N), self.row_country] = self.exports
        return E

    @cached_property
    def bilateral_gross(self) -> np.ndarray:
        """총수출 C×C [수출국, 수입국]."""
        Zc = self.Z.reshape(self.N, self.C, self.S).sum(2) + self.Yd
        M = np.zeros((self.C, self.C))
        np.add.at(M, self.row_country, Zc)
        np.fill_diagonal(M, 0)
        return M

    # ---------- 표 꾸미기 ----------
    def cell_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "국가코드": np.repeat(self.countries, self.S),
            "국가": [COUNTRY_KO.get(c, c) for c in np.repeat(self.countries, self.S)],
            "산업코드": np.tile(self.industries, self.C),
            "산업": [INDUSTRY_KO.get(s, s) for s in np.tile(self.industries, self.C)],
        })

    def country_sum(self, vec: np.ndarray) -> np.ndarray:
        return np.bincount(self.row_country, weights=vec, minlength=self.C)

    def country_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"국가코드": self.countries, "국가": [COUNTRY_KO.get(c, c) for c in self.countries]})

    def expand(self, sel: list[str] | str) -> list[str]:
        """국가 코드·묶음 이름·'전 세계' → 국가 코드 목록."""
        if isinstance(sel, str):
            sel = [sel]
        out: list[str] = []
        for s in sel:
            if s == WORLD:
                out += self.countries
            elif s == WORLD_EX:
                continue
            elif s in COUNTRY_GROUPS:
                out += [c for c in COUNTRY_GROUPS[s] if c in self.ci]
            elif s in self.ci:
                out.append(s)
        return list(dict.fromkeys(out))


WORLD = "전 세계"
WORLD_EX = "(수입국 제외)"


@lru_cache(maxsize=2)
def load(edition: str, year: str) -> ICIO:
    d = np.load(os.path.join(DATA, f"icio_{edition}_{year}.npz"))
    t = ICIO(edition=str(d["edition"]), year=str(d["year"]), countries=list(d["countries"].astype(str)),
             industries=list(d["industries"].astype(str)), Z=d["Z"].astype(np.float64),
             Y=d["Y"].astype(np.float64), va=d["va"], tls=d["tls"], x=d["x"])
    return t


# =====================================================================
# 공통: 국가별 결과 표
# =====================================================================
def country_table(t: ICIO, dva: np.ndarray, dx: np.ndarray, extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    df = t.country_frame()
    dvac = t.country_sum(dva)
    df["부가가치 변화(백만 달러)"] = dvac
    df["GDP 대비(%)"] = np.where(t.va_country > 0, dvac / t.va_country * 100, 0)
    df["산출 변화(백만 달러)"] = t.country_sum(dx)
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def group_rows(t: ICIO, df: pd.DataFrame) -> pd.DataFrame:
    """국가 묶음·세계 합계 행."""
    rows = []
    for name, members in list(COUNTRY_GROUPS.items()) + [("세계", t.countries)]:
        m = df["국가코드"].isin(members)
        if not m.any():
            continue
        vac = t.va_country[[t.ci[c] for c in df.loc[m, "국가코드"]]].sum()
        r = {"국가코드": name, "국가": name}
        for col in df.columns[2:]:
            if col == "GDP 대비(%)":
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and "(%" not in col:
                r[col] = df.loc[m, col].sum()
        r["GDP 대비(%)"] = r["부가가치 변화(백만 달러)"] / vac * 100 if vac else 0
        rows.append(r)
    return pd.DataFrame(rows)[[c for c in df.columns if c in rows[0]]] if rows else pd.DataFrame()


def industry_table(t: ICIO, country: str, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    sl = t.rows(country)
    df = pd.DataFrame({"산업코드": t.industries, "산업": [INDUSTRY_KO.get(s, s) for s in t.industries],
                       "산업군": [industry_group(s) for s in t.industries]})
    for k, v in cols.items():
        df[k] = v[sl]
    return df


def heat_matrix(t: ICIO, vec: np.ndarray, countries: list[str], pct: bool = True) -> pd.DataFrame:
    """국가 × 산업군 행렬 (부가가치 변화, 국가 산업군 부가가치 대비 % 또는 금액)."""
    groups = ["농림어업", "광업", "제조업", "전기·가스·수도", "건설", "서비스"]
    gi = np.array([groups.index(industry_group(s)) for s in t.industries])
    out = []
    for c in countries:
        sl = t.rows(c)
        num = np.bincount(gi, weights=vec[sl], minlength=len(groups))
        if pct:
            den = np.bincount(gi, weights=t.va[sl], minlength=len(groups))
            num = np.where(den > 0, num / np.where(den > 0, den, 1) * 100, 0)
        out.append(num)
    return pd.DataFrame(out, index=[COUNTRY_KO.get(c, c) for c in countries], columns=groups)


# =====================================================================
# 1. 최종수요 충격
# =====================================================================
def build_demand(t: ICIO, shocks: list[dict]) -> np.ndarray:
    """shock = {"country": 코드|묶음, "component": FD 코드|ALL, "origin": "전체"|"국산"|"수입",
                "industries": [코드]|[] (빈 목록=전 산업), "pct": float}"""
    dy = np.zeros(t.N)
    for sh in shocks:
        comps = list(range(len(FD_CODES))) if sh["component"] == FD_ALL else [FD_CODES.index(sh["component"])]
        ind_mask = np.ones(t.N, bool)
        if sh.get("industries"):
            ind_mask = np.isin(t.row_industry, [t.si[s] for s in sh["industries"] if s in t.si])
        for c in t.expand(sh["country"]):
            k = t.ci[c]
            base = t.Y3[:, k, comps].sum(1)
            org = np.ones(t.N, bool)
            if sh.get("origin") == "국산":
                org = t.row_country == k
            elif sh.get("origin") == "수입":
                org = t.row_country != k
            dy += base * (org & ind_mask) * sh["pct"] / 100.0
    return dy


def demand_effect(t: ICIO, dy: np.ndarray):
    dx = t.Lmul(dy)
    dva = t.v * dx
    cells = t.cell_frame()
    cells["최종수요 변화"] = dy
    cells["산출 변화"] = dx
    cells["부가가치 변화"] = dva
    ctab = country_table(t, dva, dx, {"최종수요 변화(백만 달러)": t.country_sum(dy)})
    summ = {
        "최종수요 변화(10억 달러)": dy.sum() / 1e3,
        "세계 부가가치 변화(10억 달러)": dva.sum() / 1e3,
        "세계 GDP 대비(%)": dva.sum() / t.va.sum() * 100,
    }
    return ctab, cells, dva, dx, summ


# =====================================================================
# 2. 관세
# =====================================================================
def tariff_effect(t: ICIO, importers: list[str], exporters: list[str], industries: list[str],
                  tariff: float, elasticity: float, sub_domestic: float, sub_third: float,
                  passthrough: float = 1.0):
    """관세 시나리오 (1차 근사).

    수량: 수입국 I 가 수출국 E 의 대상 산업 제품(중간재+최종재) 수입을 r = 1-(1+τ·전가율)^(-ε) 만큼 줄인다(등탄력 수입수요).
          줄어든 수입액 중 sub_domestic 은 I 의 같은 산업 국내 생산으로, sub_third 는 제3국(관세 비대상) 같은 산업
          공급자로 (I 시장 내 기존 점유율대로) 옮겨가고, 나머지는 수요 소멸.
          → 수요 이동 Δd 를 L 로 파급: Δx = L Δd.
    가격: I 의 산업이 사들이는 대상 수입 중간재 비용이 τ·전가율 만큼 오름 → Δp' = Δc' L (국제 파급 포함).
          I 의 소비자물가 = 가계소비(원산지별) 가중; 대상 수입 소비재는 τ·전가율 직접 상승.
    """
    tau = tariff / 100.0
    r = 1.0 - (1.0 + tau * passthrough) ** (-elasticity)   # 등탄력 수입수요: M1/M0 = (1+τ)^(-ε)
    I = t.expand(importers)
    E = [c for c in t.expand(exporters) if c not in I]
    ind = np.isin(t.row_industry, [t.si[s] for s in industries]) if industries else np.ones(t.N, bool)
    e_rows = np.isin(t.row_country, [t.ci[c] for c in E]) & ind
    i_cols = np.isin(np.repeat(np.arange(t.C), t.S), [t.ci[c] for c in I])   # Z 열(구매 산업)이 수입국
    i_idx = [t.ci[c] for c in I]

    # 대상 교역액: 수출국 산업행 → 수입국 (중간재 열 + 최종수요 열)
    Zt = t.Z[np.ix_(e_rows, i_cols)]                     # (대상 행) × (수입국 산업들)
    Yt = t.Yd[np.ix_(e_rows, i_idx)]                     # (대상 행) × 수입국
    lost_row = -r * (Zt.sum(1) + Yt.sum(1))              # 수출국 산업별 수요 감소
    dd = np.zeros(t.N)
    dd[e_rows] += lost_row
    lost_total = -lost_row.sum()

    # 대체: 수입국별·산업별로 나눠서 배분
    e_ind = t.row_industry[e_rows]
    for c in I:
        k = t.ci[c]
        cols_k = np.zeros(t.N, bool)
        cols_k[t.rows(c)] = True
        lost_ck_by_ind = np.bincount(
            e_ind, weights=r * (t.Z[np.ix_(e_rows, cols_k)].sum(1) + t.Yd[e_rows, k]), minlength=t.S)
        # 국내 대체
        dd[t.rows(c)] += sub_domestic * lost_ck_by_ind
        # 제3국 대체: I 시장에서 E·I 가 아닌 나라들의 산업별 점유율
        if sub_third > 0:
            third = ~np.isin(t.row_country, [t.ci[x] for x in E + [c]])
            supply = (t.Z[:, cols_k].sum(1) + t.Yd[:, k]) * third
            tot_by_ind = np.bincount(t.row_industry, weights=supply, minlength=t.S)
            share = np.where(tot_by_ind[t.row_industry] > 0, supply / np.where(tot_by_ind[t.row_industry] > 0, tot_by_ind[t.row_industry], 1), 0)
            dd += sub_third * lost_ck_by_ind[t.row_industry] * share
    dx = t.Lmul(dd)
    dva = t.v * dx

    # 가격: 수입국 산업의 대상 수입 중간재 비용 증가
    dc = np.zeros(t.N)
    dc[i_cols] = tau * passthrough * (t.Z[np.ix_(e_rows, i_cols)].sum(0) / t.xs[i_cols])
    dp = t.LTmul(dc)
    cpi = {}
    ppi = {}
    for c in I:
        k = t.ci[c]
        w = t.Y3[:, k, 0]                                # 가계소비 (원산지별)
        direct = np.where(e_rows, tau * passthrough, 0.0)
        cpi[c] = float((w * (dp + direct)).sum() / w.sum() * 100) if w.sum() > 0 else 0.0
        sl = t.rows(c)
        ppi[c] = float((t.x[sl] * dp[sl]).sum() / t.x[sl].sum() * 100)
    ctab = country_table(t, dva, dx, {"수요 이동(백만 달러)": t.country_sum(dd)})
    ctab["소비자물가(%)"] = [cpi.get(c, np.nan) for c in t.countries]
    ctab["생산자가격(%)"] = [ppi.get(c, float((t.x[t.rows(c)] * dp[t.rows(c)]).sum() / t.x[t.rows(c)].sum() * 100))
                          for c in t.countries]
    cells = t.cell_frame()
    cells["수요 이동"] = dd
    cells["산출 변화"] = dx
    cells["부가가치 변화"] = dva
    cells["생산자가격 변화(%)"] = dp * 100
    summ = {
        "수입 감소율(%)": -r * 100,
        "대상 교역액(10억 달러)": (Zt.sum() + Yt.sum()) / 1e3,
        "수입 감소액(10억 달러)": -lost_total / 1e3,
        "세계 부가가치 변화(%)": dva.sum() / t.va.sum() * 100,
    }
    if len(I) == 1:
        summ[f"{COUNTRY_KO.get(I[0], I[0])} 소비자물가(%)"] = cpi[I[0]]
        summ[f"{COUNTRY_KO.get(I[0], I[0])} GDP(%)"] = float(ctab.loc[ctab["국가코드"] == I[0], "GDP 대비(%)"].iloc[0])
    return ctab, cells, dva, dx, summ, {"importers": I, "exporters": E, "r": r}


# =====================================================================
# 3. 공급 차질
# =====================================================================
@lru_cache(maxsize=2)
def _ghosh_lu(edition: str, year: str):
    t = load(edition, year)
    B = t.Z / t.xs[:, None]
    return lu_factor(np.eye(t.N) - B, overwrite_a=True, check_finite=False)


def supply_effect(t: ICIO, cuts: dict[int, float]):
    """cuts = {행 인덱스: 차질률(%)}."""
    idx = np.array(sorted(cuts))
    cut = np.array([cuts[i] for i in idx]) / 100.0
    dx0 = -cut * t.x[idx]
    glu = _ghosh_lu(t.edition, t.year)
    # G 의 해당 행들: e_i' (I-B)^-1  → (I-B)' g = e_i
    Eb = np.zeros((t.N, len(idx)))
    Eb[idx, np.arange(len(idx))] = 1.0
    Grows = lu_solve(glu, Eb, trans=1, check_finite=False).T      # len(idx) × N
    dv = np.linalg.solve(Grows[:, idx].T, dx0)
    fwd = dv @ Grows
    dz = t.Z[:, idx] @ (dx0 / t.xs[idx])
    bwd = t.Lmul(dz)
    bwd[idx] += dx0
    cells = t.cell_frame()
    cells["직접 차질"] = 0.0
    cells.loc[idx, "직접 차질"] = dx0
    cells["전방 산출 변화"] = fwd
    cells["후방 산출 변화"] = bwd
    cells["전방 부가가치 변화"] = t.v * fwd
    cells["후방 부가가치 변화"] = t.v * bwd
    ctab = t.country_frame()
    for nm, vec in (("전방", fwd), ("후방", bwd)):
        vac = t.country_sum(t.v * vec)
        ctab[f"{nm} 부가가치 변화(백만 달러)"] = vac
        ctab[f"{nm} GDP 대비(%)"] = np.where(t.va_country > 0, vac / t.va_country * 100, 0)
    summ = {
        "직접 차질(10억 달러)": dx0.sum() / 1e3,
        "전방 세계 부가가치(10억 달러)": (t.v * fwd).sum() / 1e3,
        "후방 세계 부가가치(10억 달러)": (t.v * bwd).sum() / 1e3,
    }
    return ctab, cells, fwd, bwd, summ


# =====================================================================
# 4. 가격 파급 (외생 가격)
# =====================================================================
def price_effect(t: ICIO, fixed: dict[int, float]):
    """fixed = {행 인덱스: 가격 변화율(%)}: 해당 국가·산업 가격을 외생으로 고정.
    나머지 가격: Δp_r' = Δp_f' A_fr (I - A_rr)^-1 (비용 전가, 이윤·임금 불변)."""
    f = np.array(sorted(fixed))
    dpf = np.array([fixed[i] for i in f]) / 100.0
    rmask = np.ones(t.N, bool)
    rmask[f] = False
    Arr = t.A[np.ix_(rmask, rmask)]
    rhs = t.A[np.ix_(f, rmask)].T @ dpf
    dpr = np.linalg.solve((np.eye(rmask.sum()) - Arr).T, rhs)
    dp = np.zeros(t.N)
    dp[rmask] = dpr
    dp[f] = dpf
    ctab = t.country_frame()
    cpi, ppi = [], []
    for c in t.countries:
        k = t.ci[c]
        w = t.Y3[:, k, 0]
        cpi.append(float((w * dp).sum() / w.sum() * 100) if w.sum() > 0 else 0.0)
        sl = t.rows(c)
        ppi.append(float((t.x[sl] * dp[sl]).sum() / t.x[sl].sum() * 100))
    ctab["소비자물가(%)"] = cpi
    ctab["생산자가격(%)"] = ppi
    cells = t.cell_frame()
    cells["가격 변화(%)"] = dp * 100
    cells["외생 여부"] = np.where(~rmask, "외생", "")
    summ = {"세계 생산자가격(산출 가중, %)": (t.x @ dp) / t.x.sum() * 100,
            "세계 소비자물가(가계소비 가중, %)": float((t.Y3[:, :, 0].sum(1) @ dp) / t.Y3[:, :, 0].sum() * 100)}
    return ctab, cells, dp, summ


# =====================================================================
# 5. GVC·부가가치 무역
# =====================================================================
@lru_cache(maxsize=2)
def _vax(edition: str, year: str):
    """VAX[o, k] = o 국 부가가치 중 k 국 총수출에 체화된 것.  VAFD[o, d] = o 국 부가가치 중 d 국 최종수요에 흡수된 것."""
    t = load(edition, year)
    LE = t.Lmul(t.export_matrix)               # N × C
    VE = t.v[:, None] * LE
    VAX = np.zeros((t.C, t.C))
    np.add.at(VAX, t.row_country, VE)
    LY = t.Lmul(t.Yd)
    VY = t.v[:, None] * LY
    VAFD = np.zeros((t.C, t.C))
    np.add.at(VAFD, t.row_country, VY)
    return VAX, VAFD, LE


def gvc_indicators(t: ICIO) -> pd.DataFrame:
    VAX, VAFD, _ = _vax(t.edition, t.year)
    gross = t.country_sum(t.exports)
    dva = np.diag(VAX)
    fva = VAX.sum(0) - dva
    dvx = VAX.sum(1) - dva
    df = t.country_frame()
    df["총수출(10억 달러)"] = gross / 1e3
    df["국내 부가가치(DVA) 비중(%)"] = np.where(gross > 0, dva / gross * 100, 0)
    df["해외 부가가치(FVA) 비중(%)"] = np.where(gross > 0, fva / gross * 100, 0)
    df["전방 참여(DVX) 비중(%)"] = np.where(gross > 0, dvx / gross * 100, 0)
    df["GVC 참여도(%)"] = df["해외 부가가치(FVA) 비중(%)"] + df["전방 참여(DVX) 비중(%)"]
    df["부가가치 수출(10억 달러)"] = (VAFD.sum(1) - np.diag(VAFD)) / 1e3
    df["부가가치 수출/총수출(%)"] = np.where(gross > 0, df["부가가치 수출(10억 달러)"] * 1e3 / gross * 100, 0)
    df["GDP(10억 달러)"] = t.va_country / 1e3
    df["수출/GDP(%)"] = gross / t.va_country * 100
    return df


def gvc_country(t: ICIO, country: str):
    """한 나라의 수출 부가가치 원천(국가별·산업별)과 상대국별 총·부가가치 교역."""
    VAX, VAFD, LE = _vax(t.edition, t.year)
    k = t.ci[country]
    origin = t.v * LE[:, k]                               # N: c 국 수출에 들어간 부가가치의 원천 셀
    src = t.country_frame()
    src["수출 내 부가가치(백만 달러)"] = t.country_sum(origin)
    src["비중(%)"] = src["수출 내 부가가치(백만 달러)"] / origin.sum() * 100
    src = src.sort_values("수출 내 부가가치(백만 달러)", ascending=False).reset_index(drop=True)
    # 산업별: c 국 산업 수출의 FVA 비중
    sl = t.rows(country)
    ex = t.exports[sl]
    # 산업 j 수출 1단위의 부가가치 원천 = v ⊙ L[:, j]
    Eb = np.zeros((t.N, t.S))
    Eb[np.arange(sl.start, sl.stop), np.arange(t.S)] = 1.0
    Lcols = t.Lmul(Eb)                                    # N × S (해당국 산업 열)
    vb = t.v[:, None] * Lcols
    dom = vb[sl].sum(0)
    ind = pd.DataFrame({"산업코드": t.industries, "산업": [INDUSTRY_KO.get(s, s) for s in t.industries],
                        "총수출(백만 달러)": ex, "국내 부가가치 비중(%)": dom * 100, "해외 부가가치 비중(%)": (1 - dom) * 100})
    for c2 in ["CHN", "USA", "JPN", "DEU"]:
        if c2 in t.ci and c2 != country:
            ind[f"  {COUNTRY_KO[c2]} 부가가치(%)"] = vb[t.rows(c2)].sum(0) * 100
    # 상대국별
    part = t.country_frame()
    part["총수출(백만 달러)"] = t.bilateral_gross[k]
    part["총수입(백만 달러)"] = t.bilateral_gross[:, k]
    part["총교역수지(백만 달러)"] = part["총수출(백만 달러)"] - part["총수입(백만 달러)"]
    part["부가가치 수출(백만 달러)"] = VAFD[k]
    part["부가가치 수입(백만 달러)"] = VAFD[:, k]
    part.loc[part["국가코드"] == country, ["부가가치 수출(백만 달러)", "부가가치 수입(백만 달러)"]] = 0
    part["부가가치 교역수지(백만 달러)"] = part["부가가치 수출(백만 달러)"] - part["부가가치 수입(백만 달러)"]
    part = part[part["국가코드"] != country].sort_values("총수출(백만 달러)", ascending=False).reset_index(drop=True)
    return src, ind, part
