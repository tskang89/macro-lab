"""여러 방법을 한 번에 돌리고 비교 표를 만든다 (화면·엑셀 공용)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engines.potential import methods as M
from engines.potential.data import available_methods, label

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
    "pf_alpha": 0.35,
    "pf_nairu": "mv",
    "svar_p": 4,
}


def run(df: pd.DataFrame, cfg: dict) -> tuple[dict[str, M.Estimate], dict[str, str]]:
    c = {**DEFAULT_CFG, **cfg}
    y_full = M.log100(df["gdp"])
    y = y_full.loc[c["start"]:]
    av = available_methods(df)
    ests: dict[str, M.Estimate] = {}
    errs: dict[str, str] = {}
    want = [k for k in M.METHODS if k in c["methods"]]
    # 생산함수가 다변량 필터의 자연실업률을 쓰면 다변량 필터를 먼저 돌린다
    order = sorted(want, key=lambda k: 0 if k == "mv" else 1)
    mv_nairu = None
    for k in order:
        try:
            if k in ("mv", "pf", "svar") and not av[k]:
                errs[k] = "필요한 자료(실업률·근원물가·고용·자본스톡)가 없어 건너뜀"
                continue
            if k == "linear":
                e = M.linear(y, c["breaks"])
            elif k == "hp":
                e = M.hp(y, c["lam"], c["extend"])
            elif k == "hp1":
                e = M.hp_onesided(y, c["lam"])
            elif k == "cf":
                e = M.cf(y, c["cf_high"])
            elif k == "hamilton":
                e = M.hamilton(y, c["ham_h"], c["ham_p"])
            elif k == "bn":
                e = M.bn(y, c["bn_p"], c["bn_window"], c["bn_delta"])
            elif k == "uc":
                e = M.uc(y)
            elif k == "mv":
                e = M.mv(y_full, df["urate"], df["core"], c["lam"], c["mv_lam_u"], c["mv_weight"])
                mv_nairu = e.extra["자연실업률"]["자연실업률 추정(%)"]
            elif k == "pf":
                nairu = mv_nairu if c["pf_nairu"] == "mv" else None
                if c["pf_nairu"] == "mv" and nairu is None:
                    nairu = M.mv(y_full, df["urate"], df["core"], c["lam"], c["mv_lam_u"],
                                 c["mv_weight"]).extra["자연실업률"]["자연실업률 추정(%)"]
                e = M.pf(df, c["pf_alpha"], c["lam"], c["extend"], nairu)
            elif k == "svar":
                e = M.svar(y_full, df["urate"], c["svar_p"])
            else:
                continue
            e.trend = e.trend.reindex(y_full.index)
            e.gap = e.gap.reindex(y_full.index)
            ests[k] = e
        except Exception as ex:  # noqa: BLE001
            errs[k] = f"{type(ex).__name__}: {ex}"
    ests = {k: ests[k] for k in M.METHODS if k in ests}
    if len(ests) >= 2:
        ests["avg"] = M.ensemble(y_full, list(ests.values()))
    return ests, errs


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
    names = [e.name for k, e in ests.items() if k != "avg"]
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
