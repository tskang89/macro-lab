"""민감도 분석 — 핵심 가정을 하나씩 바꿔 그 가정을 쓰는 방법만 다시 추정하고, 방법 평균의 최근 GDP 갭·잠재성장률
(잠재성장률 화면)과 실질 중립금리·정책 기조(중립금리 화면)가 얼마나 움직이는지 표로 만든다.

여기서는 '무엇을 바꿔 볼지'(변형 목록)와 '결과 요약'만 정한다. 실제 추정은 화면이 넘겨 주는 runner 로 하므로
화면의 디스크 캐시(data/cache)를 그대로 쓰고, 기본 설정의 민감도는 build/precompute.py 로 미리 계산해 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engines.potential import methods as M


@dataclass
class Variant:
    group: str                   # 가정 이름
    label: str                   # 바꾼 값
    over: dict = field(default_factory=dict)     # 설정 덮어쓰기
    data: str | None = None      # 자료 변형: None | "pe_core" | "pe_survey" | "nofc" | "fc:<mode>:<tail>"
    methods: tuple = ()          # 다시 추정할 방법(빈 튜플 = 선택한 방법 전부)


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------------------
# 잠재성장률
# ---------------------------------------------------------------------------
def potential_variants(cfg: dict, has_pie: bool, fc: dict | None) -> tuple[list[Variant], dict[str, str]]:
    """(변형 목록, 가정별 기준값). fc: 끝점 보정 정보 {"mode", "tail"} 또는 None(보정 안 함)."""
    def nairu_users(*ks):
        s = set(ks)
        if cfg.get("pf_labour") == "lfpr" and cfg.get("pf_nairu") in s:
            s.add("pf")
        return tuple(sorted(s))

    V: list[Variant] = []
    base: dict[str, str] = {}

    def add(group, base_val, alts, key, methods=()):
        base[group] = _fmt(base_val)
        for a in alts:
            if a != base_val:
                V.append(Variant(group, _fmt(a), {key: a}, None, methods))

    us = cfg.get("ustar_sd", M.USTAR_SD)
    add("자연실업률 변동 크기(분기 %p)", us, [0.02, 0.08] if us not in (0.02, 0.08) else [M.USTAR_SD], "ustar_sd",
        nairu_users("lw", "pfmv"))
    add("LW: 잠재 GDP 수준 충격 상한(%)", cfg["lw_sy_max"], [0.15, 0.5], "lw_sy_max", nairu_users("lw"))
    ls = cfg["pf_labshare"]
    add("노동소득분배율", ls, [round(ls - 0.06, 2), round(ls + 0.06, 2)], "pf_labshare", ("pf", "pfmv"))
    add("다변량 필터: 갭 지속성 상한", cfg["pfmv_rho"], [0.85, 0.95] if cfg["pfmv_rho"] == 0.9 else [0.9], "pfmv_rho",
        nairu_users("pfmv"))
    cut = cfg.get("tfp_cut")
    base["TFP 추세 고정 시점"] = cut or "고정 안 함"
    V.append(Variant("TFP 추세 고정 시점", "고정 안 함" if cut else "2025Q4", {"tfp_cut": None if cut else "2025Q4"},
                     None, ("pf", "pfmv")))
    lam = cfg["lam"]
    add("HP λ(통계·생산함수 추세)", lam, [lam / 2, lam * 2], "lam", tuple(sorted({"hp", "mv", "pf", "pfmv"})))
    if has_pie:
        base["기대인플레이션(준구조·다변량 필터)"] = "소비자 기대(1년)"
        V.append(Variant("기대인플레이션(준구조·다변량 필터)", "근원물가 직전 4분기 평균 상승률", {}, "pe_core",
                         nairu_users("lw", "pfmv")))
    if fc is not None:
        base["끝점 보정"] = f"한국은행 전망 연장({'전기비' if fc['mode'] == 'qoq' else '전년동기비'}), 중기 {fc['tail']:g}%"
        V.append(Variant("끝점 보정", "안 함(실적까지만, AR 예측 12분기 연장)", {"extend": 12}, "nofc"))
        other = "yoy" if fc["mode"] == "qoq" else "qoq"
        V.append(Variant("끝점 보정", f"연장 방식: {'전년동기비 그대로' if other == 'yoy' else '전기비 이어 붙임'}", {},
                         f"fc:{other}:{fc['tail']}"))
        for t in (fc["tail"] - 0.5, fc["tail"] + 0.5):
            V.append(Variant("끝점 보정", f"전망 이후 중기 성장률 {t:g}%", {}, f"fc:{fc['mode']}:{t}"))
    return V, base


def potential_summary(df: pd.DataFrame, ests: dict[str, M.Estimate], years: list[int]) -> dict:
    """방법 평균('avg')과 개별 방법의 최근 실적 분기 갭, 연간 잠재성장률."""
    from engines.potential.forecast import last_actual
    last = last_actual(df)
    out = {}
    for k, e in ests.items():
        ag = M.annual_growth(e.trend)
        out[k] = {"gap": float(e.gap.get(last, np.nan)), **{y: float(ag.get(y, np.nan)) for y in years}}
    return out


def combine(base: dict[str, M.Estimate], new: dict[str, M.Estimate], redo: list[str], replace_all: bool,
            y_full: pd.Series, avg_mode: str) -> dict[str, M.Estimate]:
    from engines.potential.analysis import add_averages
    if replace_all:
        out = {k: e for k, e in new.items() if not k.startswith("avg")}
    else:
        out = {k: e for k, e in base.items() if not k.startswith("avg")}
        for k in redo:
            if k in new:
                out[k] = new[k]
    out = {k: out[k] for k in M.METHODS if k in out}
    add_averages(y_full, out, avg_mode)
    return out


def potential_table(base_sum: dict, rows: list[tuple[Variant, list[str], dict | None, str | None]], years: list[int],
                    lab: str, names: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rows: (변형, 다시 추정한 방법, 요약 또는 None, 오류). 반환: (평균 기준 요약표, 방법별 표)."""
    b = base_sum.get("avg", {})
    head = []
    for v, redo, s, err in rows:
        a = (s or {}).get("avg", {})
        r = {"가정": v.group, "바꾼 값": v.label, "다시 추정한 방법": ", ".join(names.get(k, k) for k in redo) or "전부"}
        r[f"GDP 갭 {lab}(%)"] = a.get("gap", np.nan)
        r["갭 변화(%p)"] = a.get("gap", np.nan) - b.get("gap", np.nan)
        for y in years:
            r[f"잠재성장률 {y}(%)"] = a.get(y, np.nan)
            r[f"{y} 변화(%p)"] = a.get(y, np.nan) - b.get(y, np.nan)
        if err:
            r["비고"] = err
        head.append(r)
    det = []
    for v, redo, s, err in rows:
        if not s:
            continue
        for k in redo or [k for k in s if not k.startswith("avg")]:
            if k not in s or k not in base_sum:
                continue
            r = {"가정": v.group, "바꾼 값": v.label, "방법": names.get(k, k),
                 f"GDP 갭 {lab}(%)": s[k]["gap"], "갭 변화(%p)": s[k]["gap"] - base_sum[k]["gap"]}
            for y in years:
                r[f"잠재성장률 {y}(%)"] = s[k][y]
                r[f"{y} 변화(%p)"] = s[k][y] - base_sum[k][y]
            det.append(r)
    return pd.DataFrame(head), pd.DataFrame(det)


# ---------------------------------------------------------------------------
# 중립금리
# ---------------------------------------------------------------------------
NEUTRAL_ALL_PE = ("hp", "uc", "lw", "var", "growth")     # 기대인플레이션(실질금리)을 쓰는 방법


def neutral_variants(cfg: dict, fc_avail: bool) -> tuple[list[Variant], dict[str, str]]:
    V: list[Variant] = []
    base: dict[str, str] = {}

    def add(group, key, alts, methods, fmt=_fmt):
        cur = cfg[key]
        base[group] = fmt(cur)
        for a in alts:
            if a != cur:
                V.append(Variant(group, fmt(a), {key: a}, None, methods))

    other = "survey" if cfg["pie"] == "core" else "core"
    pe_name = {"core": "근원물가 직전 4분기 평균 상승률", "survey": "소비자 기대인플레이션(1년)"}
    base["기대인플레이션"] = pe_name[cfg["pie"]]
    V.append(Variant("기대인플레이션", pe_name[other], {"pie": other}, None, NEUTRAL_ALL_PE))
    if fc_avail:
        base["끝점 보정"] = "한국은행 전망 연장" if cfg["fc"] else "안 함"
        V.append(Variant("끝점 보정", "안 함" if cfg["fc"] else "한국은행 전망 연장", {"fc": not cfg["fc"]},
                         "nofc" if cfg["fc"] else "fc", ("lw", "growth")))
    tp = cfg["tp"]
    add("시장금리: 기간 프리미엄(%p)", "tp", [round(tp - 0.4, 1), round(tp + 0.4, 1)], ("market",))
    add("UC: 중립금리 충격 상한(%p)", "uc_smax", [0.1, 0.3] if cfg["uc_smax"] == 0.2 else [0.2], ("uc",))
    add("UC: 순환 AR 상한", "uc_rho", [0.8, 0.95] if cfg["uc_rho"] == 0.9 else [0.9], ("uc",))
    add("LW: 자연실업률 변동(분기 %p)", "ustar_sd", [0.02, 0.08] if cfg["ustar_sd"] not in (0.02, 0.08) else [M.USTAR_SD],
        ("lw",))
    add("LW: λ_z(기타 요인 z 변동)", "lw_lam_z", [0.01, 0.05] if cfg["lw_lam_z"] == 0.03 else [0.03], ("lw",))
    add("HP λ", "lam", [cfg["lam"] / 2, cfg["lam"] * 2], ("hp",))
    add("VAR: 이동 창(분기)", "var_window", [32, 60] if cfg["var_window"] == 40 else [40], ("var",))
    gb = tuple(cfg.get("g_base", ("2001Q1", "2019Q4")))
    alt = ("2010Q1", "2019Q4") if gb != ("2010Q1", "2019Q4") else ("2001Q1", "2019Q4")
    base["잠재성장률 연계: 회귀 기간"] = f"{gb[0]}~{gb[1]}"
    V.append(Variant("잠재성장률 연계: 회귀 기간", f"{alt[0]}~{alt[1]}", {"g_base": list(alt)}, None, ("growth",)))
    return V, base


def neutral_summary(ests: dict, real_rate: pd.Series, last) -> dict:
    """마지막 분기의 방법별 r*, 평균, 실질 정책금리, 정책 기조."""
    vals = {k: float(e.rstar.get(last, np.nan)) for k, e in ests.items()}
    ok = [v for v in vals.values() if np.isfinite(v)]
    avg = float(np.mean(ok)) if ok else np.nan
    rr = float(real_rate.get(last, np.nan))
    return {"methods": vals, "avg": avg, "rr": rr, "stance": rr - avg}


def neutral_table(base_sum: dict, rows: list[tuple[Variant, list[str], dict | None, str | None]], lab: str,
                  names: dict[str, str]) -> pd.DataFrame:
    out = []
    for v, redo, s, err in rows:
        s = s or {"methods": {}, "avg": np.nan, "rr": np.nan, "stance": np.nan}
        mv = [f"{names.get(k, k)} {s['methods'].get(k, np.nan):.2f} ({s['methods'].get(k, np.nan) - base_sum['methods'].get(k, np.nan):+.2f})"
              for k in redo if k in s["methods"]]
        r = {"가정": v.group, "바꾼 값": v.label, "다시 추정한 방법: r*(변화)": "; ".join(mv),
             f"실질 중립금리 평균 {lab}(%)": s["avg"], "평균 변화(%p)": s["avg"] - base_sum["avg"],
             "정책 기조(%p)": s["stance"], "기조 변화(%p)": s["stance"] - base_sum["stance"]}
        if err:
            r["비고"] = err
        out.append(r)
    return pd.DataFrame(out)
