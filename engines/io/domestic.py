"""국내 산업연관 분석 엔진 (한국은행 투입산출표, 생산자가격).

모형
  수량모형  x = Ad x + fd             → Δx = L Δfd,  L = (I-Ad)^-1
  가격모형  p' = p' Ad + π' Am + w'   → Δp' = (Δπ' Am + Δw') L
  Ghosh    x' = x' B + v'             → Δx' = Δv' G, G = (I-B)^-1, B = diag(x)^-1 Zd
  가설추출  부문 j의 행·열을 제거했을 때 총산출 감소분

자료 처리
  L        : 한국은행 공표 생산유발계수표를 그대로 사용 (국산거래표 기준).
  Ad       : I - L^-1 (공표 유발계수와 정확히 일치하도록 역산).
  Am       : A - Ad,  A = Z/x (총거래표).
  최종수요의 국산 비율 s_i : 최종수요용 수입(= 총수입 - 중간재 수입 Am x)이
             수출을 제외한 최종수요 항목에 같은 비율로 들어간다고 가정.
  취업·고용계수 : 공표 취업(고용)유발계수표 = diag(e) L 에서 e 를 역산 (명/십억원).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from engines.io.classify import large_to_group, to_group, to_large

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "processed", "io_kr")

FD_NAMES = ["민간소비", "정부소비", "민간고정자본형성", "정부고정자본형성", "재고증감", "귀중품순취득", "수출"]
VA_NAMES = ["피용자보수", "영업잉여", "고정자본소모", "순생산물세"]
EXPORT = FD_NAMES.index("수출")
LEVELS = ["대분류", "중분류"]
BASIC = "기본부문"          # 실측표 연도에만 있는 380부문 (build/build_io_basic.py)


def available_years(level: str = "대분류") -> list[str]:
    files = glob.glob(os.path.join(DATA, f"io_kr_{level}_*.npz"))
    return sorted(os.path.basename(f).split("_")[-1][:4] for f in files)


@dataclass
class KrIO:
    year: str
    level: str
    basis: str
    codes: np.ndarray
    names: np.ndarray
    Z: np.ndarray
    F: np.ndarray
    imports: np.ndarray
    x: np.ndarray
    V: np.ndarray
    L: np.ndarray
    Le: np.ndarray
    Lw: np.ndarray
    extra: dict = field(default_factory=dict)

    # ---------- 기본 계수 ----------
    @property
    def n(self) -> int:
        return len(self.x)

    @cached_property
    def labels(self) -> list[str]:
        return [f"{c} {nm}" for c, nm in zip(self.codes, self.names)]

    @cached_property
    def xs(self) -> np.ndarray:
        """나눗셈용 산출 (산출 0 인 기본부문 → 1)."""
        return np.where(self.x > 0, self.x, 1.0)

    @cached_property
    def A(self) -> np.ndarray:
        return self.Z / self.xs[None, :]

    @cached_property
    def Ad(self) -> np.ndarray:
        return np.eye(self.n) - np.linalg.inv(self.L)

    @cached_property
    def Am(self) -> np.ndarray:
        return self.A - self.Ad

    @cached_property
    def vcomp(self) -> np.ndarray:
        """부가가치 항목별 계수 (4×n)."""
        return self.V / self.xs[None, :]

    @cached_property
    def v(self) -> np.ndarray:
        return self.vcomp.sum(0)

    @cached_property
    def m(self) -> np.ndarray:
        """중간투입 수입계수 (열합)."""
        return self.Am.sum(0)

    @cached_property
    def e(self) -> np.ndarray:
        """취업계수 (명/십억원)."""
        return np.diag(self.Le @ np.linalg.inv(self.L)).copy()

    @cached_property
    def w(self) -> np.ndarray:
        """고용(임금근로자)계수 (명/십억원)."""
        return np.diag(self.Lw @ np.linalg.inv(self.L)).copy()

    @cached_property
    def import_final(self) -> np.ndarray:
        """최종수요용 수입 (백만원)."""
        return self.imports - self.Am @ self.x

    @cached_property
    def s(self) -> np.ndarray:
        """수출 외 최종수요의 국산 비율."""
        fn = self.F[:, :EXPORT].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = 1.0 - np.where(fn > 0, self.import_final / fn, 0.0)
        return np.clip(s, 0.0, 1.0)

    @cached_property
    def dom_share(self) -> np.ndarray:
        """최종수요 항목별 국산 비율 (n×k). 기본부문표는 국산거래표의 실측 비율."""
        if "Fd" in self.extra:
            F, Fd = self.F, self.extra["Fd"]
            with np.errstate(divide="ignore", invalid="ignore"):
                sh = np.where(np.abs(F) > 1.0, Fd / F, 1.0)
            return np.clip(sh, 0.0, 1.0)
        sh = np.repeat(self.s[:, None], len(FD_NAMES), axis=1)
        sh[:, EXPORT] = 1.0
        return sh

    @cached_property
    def Fd(self) -> np.ndarray:
        """국산 최종수요 (n×k). 국산비율 클리핑으로 생기는 오차는 재고증감에 흡수해
        행 합계가 (I-Ad)x 와 정확히 일치하도록 한다. 기본부문표는 국산거래표 값을 그대로 쓴다."""
        if "Fd" in self.extra:
            return self.extra["Fd"]
        fd = self.F * self.dom_share
        resid = (np.eye(self.n) - self.Ad) @ self.x - fd.sum(1)
        fd[:, FD_NAMES.index("재고증감")] += resid
        return fd

    @cached_property
    def total_va(self) -> float:
        return float(self.V.sum())

    @cached_property
    def employment(self) -> np.ndarray:
        """부문별 취업자 수 (명) = e × 산출(십억원)."""
        return self.e * self.x / 1e3

    def sector_frame(self) -> pd.DataFrame:
        if "large" in self.extra:
            return pd.DataFrame({"코드": self.codes, "부문": self.names,
                                 "산업군": [large_to_group(c) for c in self.extra["large"]]})
        return pd.DataFrame({"코드": self.codes, "부문": self.names,
                             "산업군": [to_group(c) for c in self.codes]})

    def class_map(self, by: str) -> tuple[np.ndarray, dict]:
        """기본부문 → 소·중·대분류 코드와 코드→이름 (기본부문표만)."""
        k = {"소분류": "small", "중분류": "mid", "대분류": "large"}[by]
        codes = self.extra[k]
        return codes, dict(zip(codes, self.extra[k + "_nm"]))


@lru_cache(maxsize=32)
def load_table(level: str, year: str) -> KrIO:
    d = np.load(os.path.join(DATA, f"io_kr_{level}_{year}.npz"), allow_pickle=False)
    extra = {"Lv_bok": d["Lv_bok"], "Lm_bok": d["Lm_bok"]}
    if "Fd" in d.files:                      # 기본부문표: 국산 최종수요 실측치와 분류 연계
        extra["Fd"] = d["Fd"]
        for k in ("small", "mid", "large"):
            extra[k] = d[k].astype(str)
            extra[k + "_nm"] = d[k + "_nm"].astype(str)
    return KrIO(year=str(d["year"]), level=str(d["level"]), basis=str(d["basis"]),
                codes=d["codes"].astype(str), names=d["names"].astype(str),
                Z=d["Z"], F=d["F"], imports=d["imports"], x=d["x"], V=d["V"],
                L=d["L_bok"], Le=d["Le_bok"], Lw=d["Lw_bok"], extra=extra)


# =====================================================================
# 1. 최종수요 충격 (수량모형)
# =====================================================================
ALL = "전 부문"
FD_ALL = "최종수요 전체"


def build_demand_shock(t: KrIO, shocks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """충격 목록 → (총 최종수요 변화, 국산 최종수요 변화), 백만원.

    shock = {"sector": 코드 | ALL, "component": FD_NAMES 중 하나 | FD_ALL,
             "unit": "%" | "십억원", "value": float}
    % : 해당 부문·항목 최종수요(국산+수입)의 % 변화.
    십억원 : 해당 부문 생산물에 대한 최종수요(국산+수입) 증감액. 국산 비율만큼 국내생산 수요.
             '전 부문'이면 그 항목의 부문 구성비대로 나눈다.
    """
    k = len(FD_NAMES)
    dF = np.zeros((t.n, k))
    for sh in shocks:
        comps = list(range(k)) if sh["component"] == FD_ALL else [FD_NAMES.index(sh["component"])]
        rows = np.arange(t.n) if sh["sector"] == ALL else np.where(t.codes == sh["sector"])[0]
        if len(rows) == 0:
            continue
        base = np.zeros((t.n, k))
        base[np.ix_(rows, comps)] = t.F[np.ix_(rows, comps)]
        if sh["unit"] == "%":
            dF += base * sh["value"] / 100.0
        else:
            amt = sh["value"] * 1e3  # 십억원 → 백만원
            tot = base.sum()
            if len(rows) == 1 and len(comps) == 1:
                dF[rows[0], comps[0]] += amt
            elif abs(tot) > 0:
                dF += base / tot * amt
    return dF, dF * t.dom_share


def demand_effect(t: KrIO, dF: np.ndarray, dFd: np.ndarray) -> pd.DataFrame:
    """부문별 파급효과 (금액: 십억원, 인원: 명)."""
    dfd = dFd.sum(1)
    dx = t.L @ dfd
    dva = t.v * dx
    dva_comp = t.vcomp * dx[None, :]
    dm_inter = t.Am @ dx                 # 중간재 수입 (품목별)
    dm_final = (dF - dFd).sum(1)         # 최종재 수입
    out = t.sector_frame()
    out["최종수요 변화"] = dF.sum(1) / 1e3
    out["국산 최종수요 변화"] = dfd / 1e3
    out["생산 변화"] = dx / 1e3
    out["  직접(국산 최종수요)"] = dfd / 1e3
    out["  간접(중간재 파급)"] = (dx - dfd) / 1e3
    out["부가가치 변화"] = dva / 1e3
    for i, nm in enumerate(VA_NAMES):
        out[f"  {nm}"] = dva_comp[i] / 1e3
    out["수입 변화(품목별)"] = (dm_inter + dm_final) / 1e3
    out["취업자 변화(명)"] = t.e * dx / 1e3
    out["임금근로자 변화(명)"] = t.w * dx / 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        out["생산 변화율(%)"] = np.where(t.x > 0, dx / t.xs * 100, 0.0)
    return out


def demand_summary(t: KrIO, tab: pd.DataFrame) -> dict:
    dva = tab["부가가치 변화"].sum()
    return {
        "최종수요 변화(조원)": tab["최종수요 변화"].sum() / 1e3,
        "생산 변화(조원)": tab["생산 변화"].sum() / 1e3,
        "부가가치 변화(조원)": dva / 1e3,
        "부가가치 변화율(%)": dva * 1e3 / t.total_va * 100,
        "수입 변화(조원)": tab["수입 변화(품목별)"].sum() / 1e3,
        "취업자 변화(만 명)": tab["취업자 변화(명)"].sum() / 1e4,
        "부가가치 승수": (dva / tab["최종수요 변화"].sum()) if abs(tab["최종수요 변화"].sum()) > 1e-12 else np.nan,
    }


# =====================================================================
# 2. 비용 충격 (가격모형)
# =====================================================================
def price_effect(t: KrIO, import_pct: np.ndarray, va_pct: dict[str, np.ndarray] | None = None,
                 passthrough: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """수입가격·부가가치(단위당) 변화 → 부문별 생산자가격 변화.

    import_pct : 부문(품목)별 수입가격 변화율(%) (n)
    va_pct     : {부가가치 항목: 부문별 단위당 변화율(%)}
    passthrough: 비용 상승분 가격 전가율 (0~1). 레온티에프 가격모형은 1.
    """
    dpi = np.asarray(import_pct, float) / 100.0
    dw = np.zeros(t.n)
    for nm, pct in (va_pct or {}).items():
        dw += t.vcomp[VA_NAMES.index(nm)] * np.asarray(pct, float) / 100.0
    push = (dpi @ t.Am + dw) * passthrough
    dp = push @ t.L
    tab = t.sector_frame()
    tab["생산자가격 변화(%)"] = dp * 100
    tab["  직접 비용 상승(%)"] = push * 100
    tab["  간접 파급(%)"] = (dp - push) * 100
    tab["  수입 중간재 기여(%)"] = (dpi @ t.Am * passthrough) @ t.L * 100
    tab["  부가가치 기여(%)"] = (dw * passthrough) @ t.L * 100
    tab["수입품 가격 변화(%)"] = dpi * 100

    # 지출 항목별 물가: 국산품(Δp)과 수입품(Δπ)을 국산비율로 가중
    def basket(col: int, exports: bool = False):
        wgt = t.F[:, col].clip(min=0)
        sh = np.ones(t.n) if exports else t.s
        pc = sh * dp + (1 - sh) * dpi * passthrough
        return wgt, pc

    w_c, p_c = basket(0)
    w_g, p_g = basket(1)
    inv_w = t.F[:, 2:4].sum(1).clip(min=0)
    p_i = t.s * dp + (1 - t.s) * dpi * passthrough
    w_x, p_x = basket(EXPORT, exports=True)
    c_contrib = w_c * p_c / w_c.sum() * 100
    tab["소비자물가 기여(%p)"] = c_contrib
    summ = {
        "소비자물가(가계소비 가중, %)": c_contrib.sum(),
        "정부소비 디플레이터(%)": (w_g @ p_g) / w_g.sum() * 100,
        "투자재 가격(%)": (inv_w @ p_i) / inv_w.sum() * 100,
        "수출가격(%)": (w_x @ p_x) / w_x.sum() * 100,
        "국내 생산자가격(산출 가중, %)": (t.x @ dp) / t.x.sum() * 100,
    }
    return tab, summ


# =====================================================================
# 3. 공급 차질 (Ghosh + 후방 수요 감소)
# =====================================================================
def supply_effect(t: KrIO, cut_pct: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """부문별 생산 차질률(%) → 전방(공급측, Ghosh)·후방(수요측) 파급.

    전방: 차질 부문의 산출이 줄어 이를 중간재로 쓰는 하류 부문 산출이 Ghosh 배분계수만큼 감소.
    후방: 차질 부문이 덜 사들이는 중간재 → 상류 부문 생산 감소 (레온티에프).
    두 효과는 서로 다른 극단 가정이므로 합산하지 않고 나란히 보인다.
    """
    cut = np.asarray(cut_pct, float) / 100.0
    dx0 = -cut * t.x                                  # 직접 차질
    Zd = t.Ad * t.x[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(t.x[:, None] > 0, Zd / t.xs[:, None], 0.0)
    G = np.linalg.inv(np.eye(t.n) - B)
    # 전방: 차질 부문의 본원적 투입이 x 감소분만큼 사라진다고 보고 Ghosh 전파
    # Δv_j 를 정해 Δx_j(자기 부문) = dx0_j 가 되도록 맞춤: Δv' G 의 j 성분 = dx0_j (차질 부문 집합에 대해 풀기)
    idx = np.where(cut != 0)[0]
    fwd = np.zeros(t.n)
    if len(idx):
        Gs = G[np.ix_(idx, idx)]
        dv = np.linalg.solve(Gs.T, dx0[idx])
        fwd = dv @ G[idx, :]
    # 후방: 차질 부문의 중간재 구매 감소 → 상류 생산 감소
    dz = Zd[:, idx] @ (dx0[idx] / t.xs[idx]) if len(idx) else np.zeros(t.n)
    bwd = t.L @ dz
    bwd[idx] += dx0[idx]
    tab = t.sector_frame()
    tab["직접 생산 차질"] = dx0 / 1e3
    tab["전방 파급 후 생산 변화"] = fwd / 1e3
    tab["후방 파급 후 생산 변화"] = bwd / 1e3
    tab["전방 부가가치 변화"] = t.v * fwd / 1e3
    tab["후방 부가가치 변화"] = t.v * bwd / 1e3
    with np.errstate(divide="ignore", invalid="ignore"):
        tab["전방 생산 변화율(%)"] = np.where(t.x > 0, fwd / t.xs * 100, 0)
        tab["후방 생산 변화율(%)"] = np.where(t.x > 0, bwd / t.xs * 100, 0)
    summ = {
        "직접 차질(조원)": dx0.sum() / 1e6,
        "전방 파급 부가가치(조원)": (t.v * fwd).sum() / 1e6,
        "전방 부가가치 변화율(%)": (t.v * fwd).sum() / t.total_va * 100,
        "후방 파급 부가가치(조원)": (t.v * bwd).sum() / 1e6,
        "후방 부가가치 변화율(%)": (t.v * bwd).sum() / t.total_va * 100,
    }
    return tab, summ


# =====================================================================
# 4. 연관 구조
# =====================================================================
def hypothetical_extraction(t: KrIO) -> np.ndarray:
    """부문별 가설추출 손실 (총산출 대비 %)."""
    fd = t.Fd.sum(1)
    base = t.x.sum()
    out = np.zeros(t.n)
    for j in range(t.n):
        keep = np.ones(t.n, bool)
        keep[j] = False
        Ak = t.Ad[np.ix_(keep, keep)]
        xk = np.linalg.solve(np.eye(t.n - 1) - Ak, fd[keep])
        out[j] = (base - xk.sum() - t.x[j]) / base * 100   # 다른 부문에서 사라지는 산출
    return out


def multipliers(t: KrIO, extraction: bool = True) -> pd.DataFrame:
    L = t.L
    col = L.sum(0)
    row = L.sum(1)
    tab = t.sector_frame()
    tab["생산유발계수"] = col
    tab["부가가치유발계수"] = t.v @ L
    tab["수입유발계수"] = t.m @ L
    tab["취업유발계수(명/10억원)"] = t.e @ L
    tab["고용유발계수(명/10억원)"] = t.w @ L
    tab["영향력계수"] = col / col.mean()
    tab["감응도계수"] = row / row.mean()
    tab["유형"] = np.select(
        [(tab["영향력계수"] >= 1) & (tab["감응도계수"] >= 1),
         (tab["영향력계수"] >= 1) & (tab["감응도계수"] < 1),
         (tab["영향력계수"] < 1) & (tab["감응도계수"] >= 1)],
        ["Ⅰ 핵심(후방·전방 모두 큼)", "Ⅱ 후방 연관 큼", "Ⅲ 전방 연관 큼"],
        "Ⅳ 연관 작음")
    tab["총산출(조원)"] = t.x / 1e6
    tab["부가가치(조원)"] = t.V.sum(0) / 1e6
    tab["부가가치율(%)"] = t.v * 100
    tab["중간투입 수입의존도(%)"] = t.m * 100
    tab["수출 비중(%)"] = np.where(t.x > 0, t.F[:, EXPORT] / t.xs * 100, 0)
    tab["취업자(만 명)"] = t.employment / 1e4
    if extraction:
        tab["가설추출 손실(총산출 대비 %)"] = hypothetical_extraction(t)
    return tab


def multipliers_over_time(level: str, code: str) -> pd.DataFrame:
    rows = []
    for y in available_years(level):
        t = load_table(level, y)
        j = int(np.where(t.codes == code)[0][0])
        m = multipliers(t, extraction=False).iloc[j]
        rows.append({"연도": y, **{k: m[k] for k in
                     ["생산유발계수", "부가가치유발계수", "수입유발계수", "취업유발계수(명/10억원)",
                      "영향력계수", "감응도계수", "부가가치율(%)", "중간투입 수입의존도(%)", "수출 비중(%)"]}})
    return pd.DataFrame(rows)


def aggregate(tab: pd.DataFrame, by: str, value_cols: list[str], t: KrIO | None = None) -> pd.DataFrame:
    """결과 표를 산업군·대분류(기본부문표는 소·중분류도)로 합산. 기본부문표는 t 를 넘겨 분류 연계를 씀."""
    df = tab.copy()
    if t is not None and "mid" in t.extra and by in ("소분류", "중분류", "대분류"):
        codes, _ = t.class_map(by)
        df[by] = pd.Series(codes, index=t.codes).reindex(df["코드"]).to_numpy()
    elif by == "대분류":
        df["대분류"] = [to_large(c) for c in df["코드"]]
    g = df.groupby(by, sort=False)[value_cols].sum().reset_index()
    return g
