"""여러 방법을 한 번에 돌리고 비교 표를 만든다 (화면·엑셀 공용)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engines.potential import methods as M
from engines.potential.data import NEEDS, available_methods, label

DEFAULT_CFG = {
    "methods": list(M.METHODS),
    "start": "1991Q1",
    "lam": 1600.0,
    "extend": 12,
    "breaks": ["1998Q1", "2008Q4"],
    "cf_high": 32,
    "ham_h": 8,
    "ham_p": 4,
    "bn_p": 8,
    "bn_delta": 0.2,
    "bn_window": 40,
    "mv_lam_u": 6400.0,
    "mv_weight": 1.0,
    "svar_p": 4,
    "lw_lam_g": 0.06,
    "lw_lam_z": 0.03,
    "lw_sy_max": 0.25,
    "pf_labshare": 0.61,
    "pf_capital": "ksvc",
    "pf_labour": "lfpr",
    "pf_nairu": "mv",
    "pf_cu": True,
    "pfmv_rho": 0.9,
    "avg_mode": "group",
}
NEEDS_OTHER = ("lw", "mv", "svar", "pf", "pfmv")


def run(df: pd.DataFrame, cfg: dict) -> tuple[dict[str, M.Estimate], dict[str, str]]:
    c = {**DEFAULT_CFG, **cfg}
    y_full = M.log100(df["gdp"])
    y = y_full.loc[c["start"]:]
    av = available_methods(df)
    ests: dict[str, M.Estimate] = {}
    errs: dict[str, str] = {}
    want = [k for k in M.METHODS if k in c["methods"]]
    # 생산함수 접근이 다른 모형의 자연실업률을 쓰면 그 모형을 먼저 돌린다
    first = {"mv": 0, "lw": 0, "pfmv": 0}
    order = sorted(want, key=lambda k: first.get(k, 1))
    nairu_src = {"mv": ("mv", "자연실업률"), "lw": ("lw", "중립금리·자연실업률"), "pfmv": ("pfmv", "자연실업률")}

    def nairu_of(src):
        key, tab = nairu_src[src]
        if key not in ests:
            e_ = _one(key)
            if e_ is None:
                return None
            ests_tmp[key] = e_
            return e_.extra[tab]["자연실업률 추정(%)"]
        return ests[key].extra[tab]["자연실업률 추정(%)"]

    ests_tmp: dict[str, M.Estimate] = {}

    def _one(k):
        if k in ests_tmp:
            return ests_tmp[k]
        if k == "linear":
            return M.linear(y, c["breaks"])
        if k == "hp":
            return M.hp(y, c["lam"], c["extend"])
        if k == "hp1":
            return M.hp_onesided(y, c["lam"])
        if k == "cf":
            return M.cf(y, c["cf_high"])
        if k == "hamilton":
            return M.hamilton(y, c["ham_h"], c["ham_p"])
        if k == "bn":
            return M.bn(y, c["bn_p"], c["bn_window"], c["bn_delta"])
        if k == "uc":
            return M.uc(y)
        if k == "mv":
            return M.mv(y_full, df["urate"], df["core"], c["lam"], c["mv_lam_u"], c["mv_weight"])
        if k == "lw":
            return M.lw(df, lam_g=c["lw_lam_g"], lam_z=c["lw_lam_z"], sy_max=c["lw_sy_max"])
        if k == "svar":
            return M.svar(y_full, df["urate"], c["svar_p"])
        if k == "pfmv":
            return M.pfmv(df, c["pf_labshare"], c["lam"], capital=c["pf_capital"], rho_max=c["pfmv_rho"])
        if k == "pf":
            nairu = None
            if c["pf_labour"] == "lfpr" and c["pf_nairu"] in nairu_src and av.get(c["pf_nairu"], False):
                nairu = nairu_of(c["pf_nairu"])
            return M.pf(df, c["pf_labshare"], c["lam"], c["extend"], nairu, c["pf_capital"], c["pf_labour"], c["pf_cu"])
        return None

    for k in order:
        try:
            if k in NEEDS_OTHER and not av[k]:
                errs[k] = f"필요한 자료({NEEDS[k]})가 없어 건너뜀"
                continue
            e = _one(k)
            if e is None:
                continue
            e.trend = e.trend.reindex(y_full.index)
            e.gap = e.gap.reindex(y_full.index)
            ests[k] = e
        except Exception as ex:  # noqa: BLE001
            errs[k] = f"{type(ex).__name__}: {ex}"
    ests = {k: ests[k] for k in M.METHODS if k in ests}
    add_averages(y_full, ests, c["avg_mode"])
    return ests, errs


def add_averages(y_full: pd.Series, ests: dict[str, M.Estimate], mode: str = "group") -> None:
    """모형군 평균과 전체 평균을 ests 에 더한다. mode='group': 전체 = 모형군 평균들의 평균(모형군마다 같은 비중),
    'all': 전체 = 모든 방법의 단순 평균."""
    base = [k for k in ests if not k.startswith("avg")]
    groups = {}
    for g in M.GROUPS:
        mem = [ests[k] for k in base if M.GROUP_OF[k] == g]
        if mem:
            groups[g] = M.ensemble(y_full, mem, f"avg_{g}",
                                   f"{M.GROUPS[g]}에 속한 방법들의 GDP 갭 단순 평균.")
    if len(groups) >= 2:
        for g, e in groups.items():
            ests[e.key] = e
    if len(base) >= 2:
        if mode == "group" and len(groups) >= 2:
            avg = M.ensemble(y_full, list(groups.values()), "avg",
                             "통계모형·준구조모형·생산함수모형 평균의 평균(모형군마다 같은 비중). 잠재 GDP = 실제 GDP − 평균 갭.")
            avg.params = {"포함 방법": ", ".join(ests[k].name for k in base), "가중": "모형군별 동일 비중"}
        else:
            avg = M.ensemble(y_full, [ests[k] for k in base], "avg")
        ests["avg"] = avg


def _names(ests):
    return {k: e.name for k, e in ests.items()}


def quarterly_tables(df: pd.DataFrame, ests: dict[str, M.Estimate], start: str):
    y = M.log100(df["gdp"])
    idx = y.loc[start:].index
    lab = [label(p) for p in idx]
    g = pd.DataFrame({"분기": idx.astype(str), "라벨": lab, "실제 성장률": M.yoy(y).reindex(idx).to_numpy()})
    gap = pd.DataFrame({"분기": idx.astype(str), "라벨": lab})
    lev = pd.DataFrame({"분기": idx.astype(str), "라벨": lab, "실제 GDP": df["gdp"].reindex(idx).to_numpy()})
    for k, e in ests.items():
        g[e.name] = M.yoy(e.trend).reindex(idx).to_numpy()
        gap[e.name] = e.gap.reindex(idx).to_numpy()
        lev[e.name] = np.exp(e.trend.reindex(idx).to_numpy() / 100)
    names = [e.name for k, e in ests.items() if not k.startswith("avg")]
    if names:
        gap["최소"] = gap[names].min(axis=1)
        gap["최대"] = gap[names].max(axis=1)
        g["최소"] = g[names].min(axis=1)
        g["최대"] = g[names].max(axis=1)
    return g, gap, lev


def annual_tables(df: pd.DataFrame, ests: dict[str, M.Estimate], start: str):
    y = M.log100(df["gdp"])
    y0 = int(start[:4])
    ga = pd.DataFrame({"실제 성장률": M.annual_growth(y)})
    gp = pd.DataFrame(index=ga.index)
    for k, e in ests.items():
        ga[e.name] = M.annual_growth(e.trend)
        gp[e.name] = M.annual_mean(e.gap)
    ga = ga.loc[y0:].dropna(how="all")
    gp = gp.loc[y0:].dropna(how="all")
    ga.index.name = gp.index.name = "연도"
    return ga.reset_index(), gp.reset_index()


def method_table(df: pd.DataFrame, e: M.Estimate, start: str):
    y = M.log100(df["gdp"])
    idx = y.loc[start:].index
    q = pd.DataFrame({
        "분기": idx.astype(str), "라벨": [label(p) for p in idx],
        "실제 GDP(십억원)": df["gdp"].reindex(idx).to_numpy(),
        "잠재 GDP(십억원)": np.exp(e.trend.reindex(idx).to_numpy() / 100),
        "실제 성장률(전년동기비, %)": M.yoy(y).reindex(idx).to_numpy(),
        "잠재성장률(전년동기비, %)": M.yoy(e.trend).reindex(idx).to_numpy(),
        "GDP 갭(%)": e.gap.reindex(idx).to_numpy(),
    })
    a = pd.DataFrame({"실제 성장률(%)": M.annual_growth(y), "잠재성장률(%)": M.annual_growth(e.trend),
                      "GDP 갭(연평균, %)": M.annual_mean(e.gap)})
    a = a.loc[int(start[:4]):].dropna(how="all")
    a.index.name = "연도"
    return q, a.reset_index()


def latest(df: pd.DataFrame, ests: dict[str, M.Estimate]) -> dict:
    y = M.log100(df["gdp"])
    last = y.dropna().index[-1]
    out = {"last": last, "actual_yoy": float(M.yoy(y).loc[last])}
    gaps = {k: float(e.gap.get(last, np.nan)) for k, e in ests.items()}
    pots = {k: float(M.yoy(e.trend).get(last, np.nan)) for k, e in ests.items()}
    out["gaps"], out["pots"] = gaps, pots
    return out
