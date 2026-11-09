"""잠재성장률 추정 — 여러 방법으로 잠재 GDP·잠재성장률·GDP 갭을 추정해 비교."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core import charts
from core.result import ExcelChart, Result
from core.ui import excel_button, footer, notes_box, page_header, table
from engines.potential import analysis as A
from engines.potential import data as D
from engines.potential import methods as M

META = {"title": "잠재성장률 추정", "group": "성장 분석", "order": 10, "icon": ":material/trending_up:",
        "status": "ready", "slug": "potential-growth"}

MODULE = "잠재성장률 추정"
SOURCES_KR = [
    "한국은행 ECOS: 실질 GDP(경제활동별 GDP, 계절조정·연쇄가격, 200Y104), 실질 생산자본스톡(291Y504), 제조업 평균가동률(901Y025)",
    "통계청 경제활동인구조사(ECOS 901Y027, 계절조정): 취업자·경제활동인구·15세 이상 인구·실업률",
    "통계청 소비자물가지수(ECOS 901Y009·901Y010): 총지수, 농산물 및 석유류 제외 지수(근원물가)",
]
COMMON_NOTES = [
    "GDP 갭 = (실제 GDP − 잠재 GDP) / 잠재 GDP (로그 차이 ×100, %). 플러스면 경기 과열, 마이너스면 경기 부진.",
    "잠재성장률은 추정한 잠재 GDP 수준의 증가율. 분기는 전년동기비, 연간은 4개 분기 합계의 전년 대비 증가율.",
    "모든 필터는 최신 시점에서 추정이 불안정함(끝점 문제). 자료가 추가·개정되면 최근 몇 년의 잠재성장률과 갭이 바뀔 수 있음.",
    "2020년 코로나19 충격은 별도 처리 없이 그대로 들어가 있음. 방법에 따라 그 전후 잠재성장률을 끌어내리거나 갭을 키울 수 있음.",
]

HELP = {
    "linear": ("통계적 추세", "로그 GDP 에 시간 추세를 맞추고, 지정한 구조변화 시점에서 기울기를 바꿈.",
               "단순하고 해석이 쉬움.", "구조변화 시점을 사람이 정해야 하고, 구간 안에서는 잠재성장률이 일정하다고 봄."),
    "hp": ("통계적 추세", "실제 GDP 와의 차이와 추세의 굴곡을 함께 최소화하는 매끄러운 추세(λ=1600 이 분기 표준).",
           "가장 널리 쓰여 비교 기준이 됨.", "끝점에서 최근 자료에 끌려감(이 도구는 AR 예측으로 연장해 완화). 경제 이론 정보는 없음."),
    "hp1": ("통계적 추세", "각 분기까지의 자료로만 HP 필터를 돌려 그 분기의 끝값을 잇는 실시간 추정.",
            "당시 정책 판단에 쓸 수 있었던 갭을 재현.", "양방향보다 갭이 늦게 반응하고 사후 수정 폭이 큼."),
    "cf": ("통계적 추세", "주파수 영역에서 경기순환 주기(2~32분기) 변동을 걸러내고 나머지를 잠재로 봄.",
           "경기순환 주기를 명시적으로 정의.", "주기 범위 선택에 따라 결과가 달라지고 끝점이 불안정."),
    "hamilton": ("통계적 추세", "8분기 뒤 GDP 를 현재·과거 GDP 로 예측해, 예측 가능한 부분을 추세, 예측오차를 갭으로 봄.",
                 "HP 필터의 가짜 순환 문제를 피함(Hamilton 2018).", "추세가 거칠어 잠재성장률이 크게 출렁임. 처음 11분기는 값이 없음."),
    "bn": ("시계열 모형", "성장률의 AR 모형으로 장기 예측 수준(추세)을 계산하는 BN 분해. 신호대잡음비를 제약해 갭 크기를 안정화.",
           "실시간 추정이 사후 수정에 강하다고 알려짐(Kamber·Morley·Wong 2018).", "δ·AR 차수 선택에 민감."),
    "uc": ("시계열 모형", "GDP 를 '기울기가 천천히 변하는 추세'와 'AR(2) 순환'의 합으로 보고 칼만 필터로 추정.",
           "모형의 확률 구조가 명시적이고 모수를 자료로 추정.", "추세와 순환의 식별이 약하면 결과가 HP 와 비슷하거나 불안정."),
    "mv": ("경제 구조 반영", "HP 필터에 필립스곡선(근원물가)과 오쿤의 법칙(실업률)을 더해 잠재 GDP 와 자연실업률을 함께 추정.",
           "물가·고용 정보를 써서 '인플레이션을 자극하지 않는 산출'이라는 정의에 가까움.", "관계식 계수가 약하거나 불안정하면 HP 와 비슷해짐."),
    "pf": ("경제 구조 반영", "잠재 GDP = 추세 총요소생산성 × 자본 × 잠재 노동(인구·참가율·자연실업률).",
           "잠재성장률을 노동·자본·생산성 기여도로 분해 가능. 중앙은행·EU 집행위가 주로 쓰는 방식.",
           "분배율·자본스톡·자연실업률 가정에 좌우. TFP 추세에 다시 필터를 써서 끝점 문제가 남음."),
    "svar": ("경제 구조 반영", "GDP 성장률·실업률 VAR 에서 '수요 충격은 GDP 에 장기 영향이 없다'는 제약으로 공급·수요 충격을 식별.",
             "경제적 식별 조건에 기반.", "갭이 수요 충격만으로 정의돼 다른 방법과 시점이 엇갈릴 수 있고, 한국처럼 오쿤 관계가 약하면 불안정."),
}


# ---------------------------------------------------------------------------
# 계산 (캐시)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="잠재 GDP 를 추정하고 있습니다…", max_entries=16)
def _run(df: pd.DataFrame, cfg_json: str):
    return A.run(df, json.loads(cfg_json))


def _step_q(n: int) -> int:
    return 4 if n <= 52 else 8 if n <= 104 else 12


def _step_a(n: int) -> int:
    return 1 if n <= 16 else 2


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def _data_source():
    src = st.radio("자료", ["한국 (ECOS)", "직접 올리기 (다른 나라·다른 자료)"], horizontal=True, key="pg_src")
    if src.startswith("한국"):
        df, meta = D.load_korea()
        st.caption(f"실질 GDP {meta['last_gdp']}까지 · 자본스톡 {meta['capital_last_year']}년까지 공표분 · 자료 갱신 {meta['built']}")
        return df, "KR", SOURCES_KR
    up = st.file_uploader("분기 자료 CSV·엑셀", type=["csv", "xlsx"], key="pg_upload",
                          help="첫 열은 분기(2024Q1 형식), 다음 열 이름: gdp(필수, 계절조정 실질), urate, core, emp, lf, pop15, capital")
    st.download_button("양식 내려받기 (CSV)", data=D.template_csv(), file_name="potential_template.csv",
                       mime="text/csv", key="pg_tmpl", on_click="ignore")
    if up is None:
        st.info("자료를 올리면 추정이 시작됩니다. gdp 열만 있어도 통계적 추세·시계열 모형 7가지를 쓸 수 있고, "
                "실업률(urate)·근원물가(core)를 더하면 다변량 필터와 구조 VAR, 고용(emp·lf·pop15)과 자본스톡(capital)까지 "
                "더하면 생산함수 접근도 쓸 수 있습니다.")
        return None, None, []
    try:
        df = D.parse_upload(up.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"자료를 읽지 못했습니다: {e}")
        return None, None, []
    st.caption(f"{df.index[0]}~{df.index[-1]}, 열: {', '.join(df.columns)}")
    return df, f"UP{up.file_id}", [f"사용자 제공 자료 ({up.name})"]


def _settings(df: pd.DataFrame) -> tuple[dict, str]:
    av = D.available_methods(df)
    usable = [k for k in M.METHODS if k not in ("mv", "pf", "svar") or av[k]]
    quarters = [str(p) for p in df["gdp"].dropna().index]
    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("추정 방법", usable, default=usable, format_func=lambda k: M.METHODS[k], key="pg_methods")
    with c2:
        cc = st.columns(2)
        start_def = "1991Q1" if "1991Q1" in quarters else quarters[0]
        start = cc[0].selectbox("추정 시작 분기", quarters[: max(1, len(quarters) - 40)],
                                index=quarters.index(start_def) if start_def in quarters[: len(quarters) - 40] else 0,
                                key="pg_start", help="통계적 추세·시계열 모형의 추정 표본 시작. 다변량 필터·생산함수·구조 VAR 은 실업률·고용 자료가 있는 기간 전체를 씀.")
        disp_def = "2005Q1" if "2005Q1" in quarters else quarters[min(len(quarters) - 1, 20)]
        show = cc[1].selectbox("표시 시작 분기", quarters[:-8], index=quarters.index(disp_def) if disp_def in quarters[:-8] else 0,
                               key="pg_show")
    cfg = dict(A.DEFAULT_CFG)
    cfg["methods"], cfg["start"] = chosen, start
    with st.expander("방법별 세부 설정"):
        a, b, c, d = st.columns(4)
        with a:
            st.markdown("**HP·공통**")
            cfg["lam"] = st.number_input("HP λ (분기)", 100.0, 100000.0, 1600.0, step=100.0, key="pg_lam")
            cfg["extend"] = st.slider("끝점 연장(분기, AR 예측)", 0, 20, 12, key="pg_ext",
                                      help="HP·생산함수 추세에 적용. 0이면 연장하지 않음.")
            st.markdown("**선형 추세**")
            cand = [q for q in quarters if q.endswith("Q1") or q.endswith("Q4")]
            cfg["breaks"] = st.multiselect("구조변화 시점", cand, default=[b_ for b_ in ["1998Q1", "2008Q4"] if b_ in cand],
                                           key="pg_breaks")
        with b:
            st.markdown("**대역통과·Hamilton**")
            cfg["cf_high"] = st.slider("CF: 잠재로 볼 최소 주기(분기)", 24, 48, 32, step=4, key="pg_cf")
            cfg["ham_h"] = st.slider("Hamilton: 예측 시차 h", 4, 12, 8, key="pg_h")
            cfg["ham_p"] = st.slider("Hamilton: 설명 시차 p", 2, 8, 4, key="pg_p")
        with c:
            st.markdown("**BN 필터**")
            cfg["bn_delta"] = st.slider("신호대잡음비 δ", 0.05, 1.0, 0.2, step=0.05, key="pg_bnd",
                                        help="작을수록 추세가 매끄럽고 갭이 큼.")
            cfg["bn_p"] = st.slider("AR 차수", 2, 12, 8, key="pg_bnp")
            cfg["bn_window"] = st.slider("성장률 평균 이동 기간(분기)", 20, 80, 40, step=4, key="pg_bnw")
        with d:
            st.markdown("**다변량·생산함수·VAR**")
            cfg["mv_weight"] = st.select_slider("다변량: 물가·고용 정보 가중 배수", [0.25, 0.5, 1.0, 2.0, 4.0], 1.0, key="pg_mvw")
            cfg["mv_lam_u"] = st.number_input("다변량: 자연실업률 λ", 400.0, 100000.0, 6400.0, step=400.0, key="pg_mvl")
            cfg["pf_alpha"] = st.slider("생산함수: 자본소득분배율 α", 0.25, 0.45, 0.35, step=0.01, key="pg_alpha")
            cfg["pf_nairu"] = st.radio("생산함수: 자연실업률", ["mv", "hp"], horizontal=True, key="pg_nairu",
                                       format_func=lambda x: "다변량 필터" if x == "mv" else "실업률 HP 추세")
            cfg["svar_p"] = st.slider("구조 VAR 시차", 1, 8, 4, key="pg_svp")
    return cfg, show


def _params_table(ests: dict[str, M.Estimate]) -> pd.DataFrame:
    rows = []
    for e in ests.values():
        for k, v in e.params.items():
            rows.append({"방법": e.name, "항목": k, "값": float(v) if isinstance(v, (int, float, np.floating)) else str(v)})
    return pd.DataFrame(rows, columns=["방법", "항목", "값"])


def _inputs(cfg: dict, show: str) -> dict:
    return {"추정 방법": ", ".join(M.METHODS[k] for k in cfg["methods"]), "추정 시작": cfg["start"], "표시 시작": show,
            "HP λ": cfg["lam"], "끝점 연장(분기)": cfg["extend"], "선형 추세 구조변화": ", ".join(cfg["breaks"]) or "없음",
            "CF 최소 주기": cfg["cf_high"], "Hamilton h·p": f"{cfg['ham_h']}·{cfg['ham_p']}",
            "BN δ·차수·이동기간": f"{cfg['bn_delta']}·{cfg['bn_p']}·{cfg['bn_window']}",
            "다변량 가중 배수·자연실업률 λ": f"{cfg['mv_weight']}·{cfg['mv_lam_u']:g}",
            "생산함수 α·자연실업률": f"{cfg['pf_alpha']}·{'다변량 필터' if cfg['pf_nairu'] == 'mv' else 'HP 추세'}",
            "구조 VAR 시차": cfg["svar_p"]}


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
def _cards(df, ests):
    L = A.latest(df, ests)
    lab = D.label(L["last"])
    gaps = {k: v for k, v in L["gaps"].items() if k != "avg" and np.isfinite(v)}
    pots = {k: v for k, v in L["pots"].items() if k != "avg" and np.isfinite(v)}
    ga, _ = A.annual_tables(df, ests, str(df.index[0]))
    ga = ga.set_index("연도")
    full = ga.index[ga["실제 성장률"].notna()][-1] if ga["실제 성장률"].notna().any() else None
    cols = st.columns(5)
    cols[0].metric(f"실제 성장률 ({lab}, 전년동기비)", f"{L['actual_yoy']:.2f}%")
    if "avg" in ests:
        cols[1].metric(f"잠재성장률 ({lab}, 방법 평균)", f"{L['pots']['avg']:.2f}%",
                       help="범위: " + (f"{min(pots.values()):.2f} ~ {max(pots.values()):.2f}%" if pots else "–"))
        cols[2].metric(f"GDP 갭 ({lab}, 방법 평균)", f"{L['gaps']['avg']:+.2f}%")
        cols[3].metric("GDP 갭 범위 (방법 간)", f"{min(gaps.values()):+.2f} ~ {max(gaps.values()):+.2f}%" if gaps else "–")
        if full is not None:
            row = ga.loc[full].drop(labels=["실제 성장률", "방법 평균"], errors="ignore").dropna()
            cols[4].metric(f"{full}년 잠재성장률 (방법 평균)", f"{ga.loc[full, '방법 평균']:.2f}%",
                           help=f"범위 {row.min():.2f} ~ {row.max():.2f}% · 실제 {ga.loc[full, '실제 성장률']:.2f}%")
    else:
        e = next(iter(ests.values()))
        cols[1].metric(f"잠재성장률 ({lab})", f"{L['pots'][e.key]:.2f}%")
        cols[2].metric(f"GDP 갭 ({lab})", f"{L['gaps'][e.key]:+.2f}%")


def compare_tab(df, ests, cfg, show, sources):
    names = [e.name for k, e in ests.items() if k != "avg"]
    has_avg = "avg" in ests
    c1, c2 = st.columns([2, 3])
    basis = c1.radio("성장률 기준", ["분기 (전년동기비)", "연간"], horizontal=True, key="pg_cmp_basis")
    all_lines = c2.toggle("개별 방법 선 모두 표시", value=not has_avg, key="pg_cmp_all",
                          help="끄면 방법 평균·방법 간 범위·실제값만 표시")
    g, gap, lev = A.quarterly_tables(df, ests, show)
    ga, gpa = A.annual_tables(df, ests, show)
    avg = "방법 평균" if has_avg else None
    band = ("최소", "최대") if has_avg and len(names) >= 2 else None
    lines = (names + ([avg] if avg else [])) if all_lines or not avg else [avg]
    if basis.startswith("분기"):
        step = _step_q(len(g))
        x = "라벨"
        gtab, ptab = g, gap
    else:
        step = _step_a(len(ga))
        x = "연도"
        gtab = ga.assign(연도=ga["연도"].astype(str))
        ptab = gpa.assign(연도=gpa["연도"].astype(str))
        if band:
            gtab["최소"], gtab["최대"] = ga[names].min(axis=1), ga[names].max(axis=1)
            ptab["최소"], ptab["최대"] = gpa[names].min(axis=1), gpa[names].max(axis=1)
    st.plotly_chart(charts.ts_lines(gtab, x, ["실제 성장률"] + lines, "실제 성장률과 잠재성장률", "%", 440, step,
                                    emphasize=avg, dashed=("실제 성장률",), band=band), width="stretch")
    st.plotly_chart(charts.ts_lines(ptab, x, lines, "GDP 갭 (실제 GDP 가 잠재 GDP 를 웃도는 정도)", "%", 400, step,
                                    emphasize=avg, band=band), width="stretch")
    st.markdown("**연간 잠재성장률 (%)**")
    table(ga.assign(연도=ga["연도"].astype(str)))
    st.markdown("**연간 GDP 갭 (연평균, %)**")
    table(gpa.assign(연도=gpa["연도"].astype(str)))
    qskip = _step_q(len(g))
    ex_lines = [avg, "최소", "최대"] if band else names[:6]
    res = Result(
        module=MODULE, analysis="방법 비교", inputs=_inputs(cfg, show),
        summary=_summary(df, ests),
        tables={"연간 잠재성장률": ga, "연간 GDP갭": gpa, "분기 잠재성장률(전년동기비)": g, "분기 GDP갭": gap,
                "분기 잠재GDP(십억원)": lev, "방법별 모수": _params_table(ests)},
        notes=COMMON_NOTES + [f"[{e.name}] " + " ".join(e.notes) for e in ests.values()],
        sources=sources,
        charts=[ExcelChart("분기 GDP갭", "라벨", ex_lines, "GDP 갭 (%)", kind="line", top=None, label_skip=qskip),
                ExcelChart("분기 잠재성장률(전년동기비)", "라벨", ["실제 성장률"] + ex_lines, "실제·잠재성장률 (전년동기비, %)",
                           kind="line", top=None, label_skip=qskip),
                ExcelChart("연간 잠재성장률", "연도", ["실제 성장률"] + ([avg] if avg else names[:4]), "연간 실제·잠재성장률 (%)",
                           kind="col", top=None)])
    excel_button(res, "pg_x_cmp")
    notes_box(res)


def _summary(df, ests) -> dict:
    L = A.latest(df, ests)
    lab = D.label(L["last"])
    out = {f"실제 성장률 {lab}(전년동기비, %)": L["actual_yoy"]}
    for k, e in ests.items():
        out[f"{e.name} 잠재성장률 {lab}(%)"] = L["pots"][k]
        out[f"{e.name} GDP 갭 {lab}(%)"] = L["gaps"][k]
    return out


def detail_tab(df, ests, cfg, show, sources):
    keys = list(ests)
    k = st.selectbox("방법", keys, format_func=lambda x: ests[x].name, key="pg_det_m",
                     index=keys.index("hp") if "hp" in keys else 0)
    e = ests[k]
    if k in HELP:
        grp, idea, pro, con = HELP[k]
        st.caption(f"{grp} · {idea}")
    q, a = A.method_table(df, e, show)
    step = _step_q(len(q))
    L = A.latest(df, {k: e})
    lab = D.label(L["last"])
    cols = st.columns(4)
    cols[0].metric(f"잠재성장률 ({lab})", f"{L['pots'][k]:.2f}%")
    cols[1].metric(f"GDP 갭 ({lab})", f"{L['gaps'][k]:+.2f}%")
    an = a.dropna(subset=["실제 성장률(%)"])
    if len(an):
        last = an.iloc[-1]
        cols[2].metric(f"{int(last['연도'])}년 잠재성장률", f"{last['잠재성장률(%)']:.2f}%")
        cols[3].metric(f"{int(last['연도'])}년 GDP 갭", f"{last['GDP 갭(연평균, %)']:+.2f}%")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.ts_lines(q, "라벨", ["실제 성장률(전년동기비, %)", "잠재성장률(전년동기비, %)"],
                                    "실제 성장률과 잠재성장률 (전년동기비)", "%", 400, step,
                                    dashed=("실제 성장률(전년동기비, %)",)), width="stretch")
    c2.plotly_chart(charts.ts_bar_line(q, "라벨", "GDP 갭(%)", [], "GDP 갭", "%", 400, step), width="stretch")
    ac = a.assign(연도=a["연도"].astype(str))
    st.plotly_chart(charts.ts_lines(ac, "연도", ["실제 성장률(%)", "잠재성장률(%)", "GDP 갭(연평균, %)"],
                                    "연간 실제 성장률·잠재성장률·GDP 갭", "%", 380, _step_a(len(ac)),
                                    dashed=("실제 성장률(%)",)), width="stretch")
    with st.expander("연간·분기 표", expanded=False):
        table(ac)
        table(q.drop(columns=["라벨"]))
    pt = _params_table({k: e})
    if len(pt):
        st.markdown("**추정 모수**")
        table(pt.drop(columns=["방법"]))
    if k in HELP:
        st.markdown(f"**장점** {HELP[k][2]}  \n**한계** {HELP[k][3]}")
    tables = {"분기": q, "연간": a, "모수": pt}
    for name, t in e.extra.items():
        tables[name] = _qframe(t, show) if isinstance(t.index, pd.PeriodIndex) else t
    res = Result(module=MODULE, analysis=e.name, inputs=_inputs(cfg, show),
                 summary={f"잠재성장률 {lab}(%)": L["pots"][k], f"GDP 갭 {lab}(%)": L["gaps"][k]},
                 tables=tables, notes=COMMON_NOTES + e.notes, sources=sources,
                 charts=[ExcelChart("분기", "라벨", ["실제 성장률(전년동기비, %)", "잠재성장률(전년동기비, %)"],
                                    "실제·잠재성장률 (전년동기비, %)", kind="line", top=None, label_skip=step),
                         ExcelChart("분기", "라벨", ["GDP 갭(%)"], "GDP 갭 (%)", kind="col", top=None, label_skip=step),
                         ExcelChart("연간", "연도", ["실제 성장률(%)", "잠재성장률(%)"], "연간 실제·잠재성장률 (%)",
                                    kind="col", top=None)])
    excel_button(res, f"pg_x_det_{k}")
    notes_box(res)


def _qframe(t: pd.DataFrame, show: str) -> pd.DataFrame:
    t = t.loc[show:]
    out = t.reset_index(drop=True)
    out.insert(0, "라벨", [D.label(p) for p in t.index])
    out.insert(0, "분기", t.index.astype(str))
    return out


def pf_tab(df, ests, cfg, show, sources):
    if "pf" not in ests:
        st.info("생산함수 접근을 선택하면(자료: 고용·자본스톡 필요) 잠재성장률을 노동·자본·총요소생산성 기여도로 나눠 보여 줍니다.")
        return
    e = ests["pf"]
    comp = e.extra["잠재성장률 기여도"]
    parts = ["총요소생산성", "자본", "노동: 인구", "노동: 경제활동참가율", "노동: 고용률(1-자연실업률)"]
    ann = comp.groupby(comp.index.year).mean()
    cnt = comp.groupby(comp.index.year)["자본"].count()
    ann = ann[cnt == 4].loc[int(show[:4]):]
    ann.index.name = "연도"
    ann = ann.reset_index()
    ann["연도"] = ann["연도"].astype(str)
    st.plotly_chart(charts.ts_stacked(ann, "연도", parts, "잠재성장률(로그 근사)", "잠재성장률 기여도 (연평균, %p)", "%p", 440,
                                      _step_a(len(ann))), width="stretch")
    qc = _qframe(comp, show)
    st.plotly_chart(charts.ts_stacked(qc, "라벨", parts, "잠재성장률(로그 근사)", "잠재성장률 기여도 (분기, 전년동기비, %p)",
                                      "%p", 420, _step_q(len(qc))), width="stretch")
    inp = _qframe(e.extra["생산함수 투입"], show)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.ts_lines(inp, "라벨", ["경제활동참가율(%)", "추세 참가율(%)"], "경제활동참가율", "%", 340,
                                    _step_q(len(inp)), zero=False), width="stretch")
    c2.plotly_chart(charts.ts_lines(inp, "라벨", ["실업률(%)", "자연실업률(%)"], "실업률과 자연실업률", "%", 340,
                                    _step_q(len(inp)), zero=False), width="stretch")
    st.markdown("**연간 기여도 (%p)**")
    table(ann)
    res = Result(module=MODULE, analysis="생산함수 기여도 분해", inputs=_inputs(cfg, show),
                 summary={f"{r['연도']}년 {c}(%p)": r[c] for _, r in ann.tail(1).iterrows()
                          for c in parts + ["잠재성장률(로그 근사)"]},
                 tables={"연간 기여도": ann, "분기 기여도": qc, "생산함수 투입": inp},
                 notes=COMMON_NOTES + e.notes + ["기여도는 로그 차분(전년동기비)이라 합이 잠재성장률의 근사치이며, 연간은 4개 분기 평균."],
                 sources=sources,
                 charts=[ExcelChart("연간 기여도", "연도", parts, "잠재성장률 기여도 (%p)", kind="col", top=None)])
    excel_button(res, "pg_x_pf")
    notes_box(res)


def nairu_tab(df, ests, cfg, show, sources):
    if "mv" not in ests:
        st.info("다변량 필터를 선택하면(자료: 실업률·근원물가 필요) 자연실업률과 필립스곡선 추정 결과를 보여 줍니다.")
        return
    e = ests["mv"]
    t = _qframe(e.extra["자연실업률"], show)
    t["실업률 갭(%p)"] = t["실업률(%)"] - t["자연실업률 추정(%)"]
    gap = e.gap.loc[show:]
    t["GDP 갭(%)"] = gap.reindex(e.extra["자연실업률"].loc[show:].index).to_numpy()
    step = _step_q(len(t))
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.ts_lines(t, "라벨", ["실업률(%)", "자연실업률 추정(%)"], "실업률과 자연실업률", "%", 380, step,
                                    zero=False), width="stretch")
    c2.plotly_chart(charts.ts_lines(t, "라벨", ["GDP 갭(%)", "실업률 갭(%p)"], "GDP 갭과 실업률 갭 (오쿤의 법칙)", "%", 380, step),
                    width="stretch")
    st.plotly_chart(charts.ts_lines(t, "라벨", ["근원물가 상승률(%)", "GDP 갭(%)"], "근원물가 상승률과 GDP 갭 (필립스곡선)", "%",
                                    380, step), width="stretch")
    table(_params_table({"mv": e}).drop(columns=["방법"]))
    res = Result(module=MODULE, analysis="자연실업률·필립스곡선", inputs=_inputs(cfg, show),
                 summary={f"자연실업률 {t['라벨'].iloc[-1]}(%)": t["자연실업률 추정(%)"].iloc[-1],
                          f"실업률 갭 {t['라벨'].iloc[-1]}(%p)": t["실업률 갭(%p)"].iloc[-1]},
                 tables={"자연실업률": t, "모수": _params_table({"mv": e})}, notes=COMMON_NOTES + e.notes, sources=sources,
                 charts=[ExcelChart("자연실업률", "라벨", ["실업률(%)", "자연실업률 추정(%)"], "실업률과 자연실업률 (%)",
                                    kind="line", top=None, label_skip=step)])
    excel_button(res, "pg_x_mv")
    notes_box(res)


def data_tab(df, show, sources):
    t = _qframe(df, show).rename(columns=D.COLUMNS)
    cols = [c for c in t.columns if c not in ("분기", "라벨")]
    pick = st.selectbox("그래프로 볼 자료", cols, key="pg_data_pick")
    st.plotly_chart(charts.ts_lines(t, "라벨", [pick], pick, "", 360, _step_q(len(t)), zero=False), width="stretch")
    table(t.drop(columns=["라벨"]))
    res = Result(module=MODULE, analysis="입력 자료", tables={"분기 자료": t}, sources=sources,
                 notes=["자본스톡은 연말 잔액을 분기말로 로그 선형 보간(마지막 공표 연도 이후는 최근 3년 평균 증가율로 연장)."])
    excel_button(res, "pg_x_data")
    for s in sources:
        st.caption(s)


def help_tab():
    st.markdown("GDP 갭과 잠재성장률은 직접 관측되지 않으므로 방법마다 결과가 다릅니다. 한 방법에 기대기보다 "
                "여러 방법의 **범위**와 **평균**을 함께 보는 것이 일반적입니다(한국은행·IMF·OECD 도 여러 방법을 병행).")
    rows = [{"방법": M.METHODS[k], "분류": g, "아이디어": i, "장점": p, "한계": c} for k, (g, i, p, c) in HELP.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=430)
    st.markdown("**참고 문헌**  \n"
                "Hodrick·Prescott(1997), Christiano·Fitzgerald(2003), Hamilton(2018, REStat), "
                "Kamber·Morley·Wong(2018, REStat), Harvey(1985)·Clark(1987), Laxton·Tetlow(1992, 다변량 필터), "
                "Blanchard·Quah(1989, AER), 유럽연합 집행위 생산함수 방법(Havik 외 2014).")


# ---------------------------------------------------------------------------
def render():
    page_header("잠재성장률 추정",
                "통계적 필터·시계열 모형·경제 구조 모형 등 10가지 방법으로 잠재 GDP 를 추정해 "
                "실제 성장률·잠재성장률·GDP 갭을 비교합니다.")
    with st.container(border=True):
        df, data_key, sources = _data_source()
        if df is None:
            footer()
            return
        cfg, show = _settings(df)
    if not cfg["methods"]:
        st.warning("추정 방법을 하나 이상 고르세요.")
        return
    ests, errs = _run(df, json.dumps(cfg, sort_keys=True))
    for k, msg in errs.items():
        st.warning(f"{M.METHODS.get(k, k)}: {msg}")
    if not ests:
        st.error("추정 결과가 없습니다.")
        return
    _cards(df, ests)
    tabs = st.tabs(["방법 비교", "방법별 상세", "생산함수 기여도", "자연실업률·필립스곡선", "입력 자료", "방법 설명"])
    with tabs[0]:
        compare_tab(df, ests, cfg, show, sources)
    with tabs[1]:
        detail_tab(df, ests, cfg, show, sources)
    with tabs[2]:
        pf_tab(df, ests, cfg, show, sources)
    with tabs[3]:
        nairu_tab(df, ests, cfg, show, sources)
    with tabs[4]:
        data_tab(df, show, sources)
    with tabs[5]:
        help_tab()
    footer()
