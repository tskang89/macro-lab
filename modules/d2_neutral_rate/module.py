"""중립금리 추정 — 통계·준구조·시계열·시장/성장 연계 모형으로 실질·명목 중립금리와 통화정책 기조를 추정."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core import charts
from core.result import ExcelChart, Result
from core.ui import excel_button, footer, notes_box, page_header, table
from core.diskcache import disk_cache
from engines import neutral as N
from engines.potential import analysis as A
from engines.potential import data as D
from engines.potential import forecast as F
from engines.potential import methods as M
from engines.potential import sensitivity as S

META = {"title": "중립금리 추정", "group": "성장 분석", "order": 20, "icon": ":material/balance:",
        "status": "ready", "slug": "neutral-rate"}

MODULE = "중립금리 추정"
SLOT = {"hp": 0, "lw": 1, "uc": 2, "var": 6, "market": 4, "growth": 5}
BOK_RANGE = (1.8, 3.3)            # 한국은행 「한국의 중립금리 추정」 명목 중립금리 범위(언론 보도 기준)
SOURCES = [
    "한국은행 ECOS: 기준금리(722Y001), 국고채 5·10년 수익률(721Y001), 실질 GDP(200Y104), 근원물가(901Y010), "
    "소비자동향조사 기대인플레이션(511Y003), 고용·자본서비스 등(잠재성장률 화면과 같은 자료)",
    "한국은행 경제전망보고서(2026년 8월): 끝점 보정(준구조모형·잠재성장률 연계)",
]
REFS = ("Laubach·Williams(2003, REStat), Holston·Laubach·Williams(2017, JIE), Lubik·Matthes(2015, Richmond Fed Economic Brief), "
        "Christensen·Rudebusch(2019, REStat), Del Negro 외(2017, BPEA)  \n"
        "권지호·김도완·지정구·김건·노경서(2019), 「우리나라의 잠재성장률 추정」, 조사통계월보 2019년 8월호  \n"
        "한국은행, 「한국의 중립금리 추정」 — 언론 보도 기준 명목 중립금리 1.8~3.3%(중간값 2.55%). 한국은행은 2026년 9월 "
        "통화신용정책보고서 설명회에서 잠재성장률과 함께 중립금리를 다시 추정 중이라고 밝힘.")
HELP = {
    "hp": ("실질 정책금리의 HP 추세를 중립금리로 봄.", "단순하고 투명.", "정책금리가 오래 중립에서 벗어나 있으면 추세도 끌려감. 경제 이론 정보 없음."),
    "uc": ("실질 정책금리 = 중립금리(확률보행) + 순환(AR(1)) 으로 나누는 칼만 필터 모형.", "확률 구조가 명시적.",
           "중립과 순환의 식별이 약해 충격 크기 상한 등 가정에 좌우."),
    "lw": ("IS곡선·필립스곡선·오쿤의 관계식으로 GDP 갭과 함께 r* = 잠재성장률 + z 를 추정(Laubach·Williams 2003).",
           "중앙은행 표준. 물가·경기와 일관된 중립금리와 잠재성장률을 함께 얻음.",
           "IS곡선의 금리 민감도가 작게 추정되면 z 가 잘 식별되지 않아 불확실성이 큼."),
    "var": ("성장률·물가·실질금리 VAR 로 5년 뒤 실질금리를 예측(충격이 사라진 뒤 돌아가는 수준, Lubik·Matthes 2015 단순화).",
            "모형 제약이 적고 자료가 말하는 장기 수준을 봄.", "이동 창 길이·시차에 민감하고 구조 변화 직후 불안정."),
    "market": ("국고채 5년·10년 금리로 5년 후 5년 선도금리를 구하고 물가목표와 기간 프리미엄을 뺌.",
               "시장 참가자의 장기 기대를 실시간 반영.", "기간 프리미엄을 가정으로 빼므로 수준은 가정에 좌우. 해외 금리 영향도 섞임."),
    "growth": ("오일러 방정식 r* = σ·g + ρ: 잠재성장률(잠재성장률 화면의 4개 모형 평균)에 과거 관계를 적용.",
               "잠재성장률 추정과 바로 연결돼 해석이 쉬움.", "성장률 밖 요인(인구구조·안전자산 수요·해외 금리) 변화는 못 잡음."),
}


# ---------------------------------------------------------------------------
@disk_cache("neutral_potential", code=("engines/potential",))
def _potential_disk(df: pd.DataFrame) -> pd.Series:
    ests, _ = A.run(df, {"methods": ["hp", "lw", "pfmv", "pf"], "extend": 0 if (df.get(F.FC_COL, pd.Series([0])) == F.TAIL).any()
                         else 12, "avg_mode": "all", "tfp_cut": "2025Q4"})
    return M.yoy(ests["avg"].trend)


@disk_cache("neutral_run", code=("engines/potential", "engines/neutral.py", "modules/d2_neutral_rate/module.py"))
def _run_disk(df: pd.DataFrame, dfx: pd.DataFrame, cfg_json: str) -> tuple[dict, dict]:
    c = json.loads(cfg_json)
    base = df if c["pie"] == "survey" else df.drop(columns=["pie"])
    basex = dfx if c["pie"] == "survey" else dfx.drop(columns=["pie"])
    last = str(F.last_actual(dfx))
    out, errs = {}, {}
    gpot = _potential_disk(dfx) if "growth" in c["methods"] else None
    for k in N.METHODS:
        if k not in c["methods"]:
            continue
        try:
            if k == "hp":
                e = N.hp_trend(base, c["lam"])
            elif k == "uc":
                e = N.uc(base, c["uc_smax"], c.get("uc_rho", 0.9))
            elif k == "lw":
                e = N.lw(basex, fit_end=last if F.FC_COL in dfx.columns else None, lam_g=c["lw_lam_g"], lam_z=c["lw_lam_z"],
                         su_star=c.get("ustar_sd", M.USTAR_SD))
            elif k == "var":
                e = N.var_forecast(base, c["var_window"], c["var_p"], c["var_h"])
            elif k == "market":
                e = N.market(base, c["tp"])
            else:
                e = N.growth_linked(base, gpot, tuple(c.get("g_base", ("2001Q1", "2019Q4"))))
            e.rstar = e.rstar.loc[:last]
            if e.sd is not None:
                e.sd = e.sd.loc[:last]
            for n_, t in list(e.extra.items()):
                e.extra[n_] = t.loc[:last]
            out[k] = e
        except Exception as ex:  # noqa: BLE001
            errs[k] = f"{type(ex).__name__}: {ex}"
    return out, errs


@st.cache_data(show_spinner="잠재성장률을 추정하고 있습니다(중립금리의 잠재성장률 연계용)…", max_entries=4)
def _potential(df: pd.DataFrame) -> pd.Series:
    return _potential_disk(df)


@st.cache_data(show_spinner="중립금리를 추정하고 있습니다(칼만 필터 최우추정 포함)…", max_entries=8)
def _run(df: pd.DataFrame, dfx: pd.DataFrame, cfg_json: str) -> tuple[dict, dict]:
    return _run_disk(df, dfx, cfg_json)


def _lab(p) -> str:
    return D.label(p)


def _qframe(t: pd.DataFrame, show: str) -> pd.DataFrame:
    t = t.loc[show:]
    out = t.reset_index(drop=True)
    out.insert(0, "라벨", [_lab(p) for p in t.index])
    out.insert(0, "분기", t.index.astype(str))
    return out


def _step(n: int) -> int:
    return 4 if n <= 52 else 8 if n <= 104 else 12


# ---------------------------------------------------------------------------
def _settings(df: pd.DataFrame) -> tuple[dict, str, pd.DataFrame]:
    fc = F.load_bok()
    c1, c2 = st.columns([1, 1])
    pie = c1.radio("기대인플레이션(실질금리 계산)", ["core", "survey"], horizontal=True, key="nr_pie",
                   format_func=lambda x: "근원물가 직전 4분기 평균 상승률" if x == "core" else "소비자 기대인플레이션(1년)",
                   help="실질 정책금리 = 기준금리 − 기대인플레이션. 기본은 Holston·Laubach·Williams(2017)처럼 근원물가 상승률의 직전 4분기 "
                        "평균. 소비자동향조사 일반인 기대는 2010년대 이후 실제 물가를 1%p 안팎 웃돌아 실질금리와 중립금리를 낮게 잡음.")
    fc_on = c2.toggle(f"끝점 보정: 한국은행 전망으로 {_lab(pd.Period(fc['horizon'], freq='Q'))}까지 연장" if fc else "끝점 보정",
                      value=fc is not None, disabled=fc is None, key="nr_fc",
                      help="준구조모형(LW형)과 잠재성장률 연계에 한국은행 경제전망을 써서 끝점 문제를 줄임(잠재성장률 화면과 같은 방식, "
                           "전망 이후는 중기 성장률 2% 가정). 결과는 마지막 실적 분기까지만 보임.")
    dfx = df
    if fc_on and fc:
        dfx = F.add_tail(F.extend(df, fc)[0], 2.0)
    st.markdown("**추정 방법**")
    gcols = st.columns(len(N.GROUPS))
    chosen = []
    for col, (g, gname) in zip(gcols, N.GROUPS.items()):
        opts = [k for k in N.METHODS if N.GROUP_OF[k] == g]
        keys = {k: f"nr_m_{k}" for k in opts}
        for k in opts:
            st.session_state.setdefault(keys[k], True)
        n_sel = sum(bool(st.session_state.get(keys[k])) for k in opts)
        with col:
            st.markdown(f"<div style='font-size:0.875rem;margin-bottom:0.25rem'>{gname}</div>", unsafe_allow_html=True)
            with st.popover(f"{n_sel}/{len(opts)}개 선택", width="stretch"):
                for k in opts:
                    st.checkbox(N.METHODS[k], key=keys[k], help=HELP[k][0])
            sel = [k for k in opts if st.session_state.get(keys[k])]
            chosen += sel
            st.caption(", ".join(N.METHODS[k] for k in sel) if sel else "선택 없음")
    quarters = [str(p) for p in df["rate"].dropna().index]
    cfg = {"methods": chosen, "pie": pie, "fc": bool(fc_on and fc)}
    with st.expander("방법별 세부 설정"):
        a, b, c = st.columns(3)
        with a:
            show = st.selectbox("표시 시작 분기", quarters[:-12], index=quarters.index("2003Q1") if "2003Q1" in quarters else 0,
                                key="nr_show")
            cfg["lam"] = st.number_input("HP λ", 400.0, 100000.0, 1600.0, step=400.0, key="nr_lam")
            cfg["uc_smax"] = st.select_slider("UC: 중립금리 분기 충격 표준편차 상한(%p)", [0.1, 0.15, 0.2, 0.3, 0.5], 0.2, key="nr_uc")
            cfg["uc_rho"] = st.select_slider("UC: 순환 AR 계수 상한", [0.8, 0.85, 0.9, 0.95], 0.9, key="nr_ucrho",
                                             help="순환이 오래 지속될 수 있게 하면 실질금리의 느린 움직임이 순환으로 잡혀 중립금리가 평평해짐.")
        with b:
            cfg["lw_lam_g"] = st.select_slider("LW: λ_g", [0.02, 0.04, 0.06, 0.08, 0.1], 0.06, key="nr_lwg")
            cfg["lw_lam_z"] = st.select_slider("LW: λ_z (기타 요인 z 변동)", [0.0, 0.01, 0.03, 0.05, 0.1], 0.03, key="nr_lwz")
            cfg["ustar_sd"] = st.select_slider("LW: 자연실업률 변동 크기(분기 %p)", [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12],
                                               M.USTAR_SD, key="nr_ustar",
                                               help="잠재성장률 화면과 같은 값이 기본. LW 중립금리는 이 값에 민감합니다: 키우면 자연실업률이 "
                                                    "실제 실업률을 따라가고 오쿤 계수가 작아지며, 중립금리의 기타 요인 z 가 낮아지는 쪽으로 움직입니다.")
            cfg["tp"] = st.slider("시장금리: 기간 프리미엄 가정(%p)", 0.0, 2.0, 0.8, step=0.1, key="nr_tp",
                                  help="5년 후 5년 선도금리에 들어 있는 기간 프리미엄. 모형으로 추정하지 않은 가정치이며 수준을 그대로 옮김.")
        with c:
            cfg["var_window"] = st.slider("VAR: 이동 창(분기)", 32, 64, 40, step=4, key="nr_vw")
            cfg["var_p"] = st.slider("VAR: 시차", 1, 4, 2, key="nr_vp")
            cfg["var_h"] = st.slider("VAR: 예측 시계(분기)", 12, 40, 20, step=4, key="nr_vh")
            g0 = st.selectbox("잠재성장률 연계: 회귀 시작", ["2001Q1", "2005Q1", "2010Q1"], key="nr_gb",
                              help="이 분기부터 2019Q4 까지 실질 정책금리를 잠재성장률에 회귀(그 기간 정책이 평균적으로 중립이었다고 가정).")
            cfg["g_base"] = [g0, "2019Q4"]
    return cfg, show, dfx


def _inputs(cfg: dict, show: str) -> dict:
    return {"추정 방법": ", ".join(N.METHODS[k] for k in cfg["methods"]),
            "기대인플레이션": "근원물가 직전 4분기 평균 상승률(전년동기비 t−1)" if cfg["pie"] == "core" else "소비자 기대인플레이션(1년)",
            "끝점 보정": "한국은행 경제전망" if cfg["fc"] else "없음", "표시 시작": show, "HP λ": cfg["lam"],
            "UC 충격 상한·순환 AR 상한": f"{cfg['uc_smax']}·{cfg['uc_rho']}",
            "잠재성장률 연계 회귀 기간": "~".join(cfg["g_base"]), "LW λ_g·λ_z·자연실업률 변동": f"{cfg['lw_lam_g']}·{cfg['lw_lam_z']}·{cfg['ustar_sd']}",
            "기간 프리미엄 가정(%p)": cfg["tp"], "VAR 창·시차·시계": f"{cfg['var_window']}·{cfg['var_p']}·{cfg['var_h']}"}


def _frame(df: pd.DataFrame, ests: dict, cfg: dict) -> pd.DataFrame:
    base = df if cfg["pie"] == "survey" else df.drop(columns=["pie"])
    E = N.ensemble(ests)
    r = N.real_rate(base)
    tgt = N.inflation_target(E.index)
    E["실질 정책금리"] = r.reindex(E.index)
    E["기준금리"] = df["rate"].reindex(E.index)
    E["명목 중립금리(평균)"] = E["평균"] + tgt
    E["명목 최소"] = E["최소"] + tgt
    E["명목 최대"] = E["최대"] + tgt
    E["정책 기조(실질 정책금리 − 중립금리)"] = E["실질 정책금리"] - E["평균"]
    return E


def _cards(E: pd.DataFrame, ests: dict):
    last = E["평균"].dropna().index[-1]
    row = E.loc[last]
    lab = _lab(last)
    stance = row["정책 기조(실질 정책금리 − 중립금리)"]
    word = "긴축적" if stance > 0.25 else "완화적" if stance < -0.25 else "중립 부근"
    cols = st.columns(5)
    spark = E.loc[:last].tail(20)
    cols[0].metric(f"실질 중립금리 ({lab}, 방법 평균)", f"{row['평균']:.2f}%",
                   help=f"방법 간 {row['최소']:.2f} ~ {row['최대']:.2f}% · 작은 그래프: 최근 5년",
                   chart_data=spark["평균"].round(3).tolist(), chart_type="line")
    cols[1].metric(f"명목 중립금리 ({lab})", f"{row['명목 중립금리(평균)']:.2f}%",
                   help=f"실질 + 물가목표 2%. 범위 {row['명목 최소']:.2f} ~ {row['명목 최대']:.2f}% · "
                        f"한국은행 기존 추정 {BOK_RANGE[0]}~{BOK_RANGE[1]}%",
                   chart_data=spark["명목 중립금리(평균)"].round(3).tolist(), chart_type="line")
    cols[2].metric(f"기준금리 ({lab}, 분기 평균)", f"{row['기준금리']:.2f}%",
                   chart_data=spark["기준금리"].round(3).tolist(), chart_type="line")
    cols[3].metric(f"실질 정책금리 ({lab})", f"{row['실질 정책금리']:.2f}%",
                   chart_data=spark["실질 정책금리"].round(3).tolist(), chart_type="line")
    cols[4].metric("통화정책 기조 (실질 정책금리 − 중립금리)", f"{stance:+.2f}%p", delta=word, delta_color="off",
                   delta_arrow="off", chart_data=spark["정책 기조(실질 정책금리 − 중립금리)"].round(3).tolist(), chart_type="bar",
                   help="플러스면 정책금리가 중립보다 높아 긴축적, 마이너스면 완화적. ±0.25%p 안은 중립 부근으로 표시. "
                        "기조는 실질 기준 비교라 '기준금리 − 명목 중립금리'와 다를 수 있음: 명목 중립금리는 실질 + 물가목표(2%)이고 "
                        "실질 정책금리는 기준금리 − 기대인플레이션이므로, 둘의 차이 = 기대인플레이션 − 물가목표.")


def compare_tab(E, ests, cfg, show, sources):
    names = [e.name for e in ests.values()]
    basis = st.radio("기준", ["실질", "명목(+물가목표)"], horizontal=True, key="nr_basis")
    T = E.loc[show:].copy()
    if basis.startswith("명목"):
        tgt = N.inflation_target(T.index)
        for c in names + ["평균", "최소", "최대"]:
            T[c] = T[c] + tgt
        pol = "기준금리(명목)"
        T[pol] = T["기준금리"]
    else:
        pol = "실질 정책금리 "
        T[pol] = T["실질 정책금리"]
    q = _qframe(T, show)
    slots = {ests[k].name: SLOT[k] for k in ests}
    band = ("최소", "최대") if len(names) >= 2 else None
    st.plotly_chart(charts.ts_lines(q, "라벨", names + (["평균"] if len(names) > 1 else []) + [pol],
                                    f"{basis.split('(')[0]} 중립금리 추정", "%", 460, _step(len(q)), emphasize="평균",
                                    dashed=(pol,), band=band, slots=slots), width="stretch")
    st.caption(("명목 = 실질 + 물가목표(2016년 이후 2%, 이전 3%). 점선: 기준금리. " if basis.startswith("명목") else
                "점선: 실질 정책금리(기준금리 − 기대인플레이션). ") + "음영: 방법 간 최소~최대(통계적 신뢰구간이 아님).")
    ut = _unc_table(ests)
    if ut is not None:
        st.markdown("**추정 불확실성 (상태공간 모형)**")
        table(ut)
        st.caption("표준오차 = 칼만 평활 오차 + 모수 추정 오차(Hamilton 1986: 최우추정량 점근분포에서 모수를 200회 뽑아 평활을 되풀이). "
                   "기대인플레이션·충격 크기 상한 등 고정한 가정의 불확실성은 들어가지 않으므로 '민감도' 탭과 함께 볼 것. "
                   "Laubach·Williams(2003)·Holston·Laubach·Williams(2017)도 r* 의 신뢰구간이 매우 넓다고 보고함.")
    ann = T[names + ["평균", pol]].groupby(T.index.year).mean().round(2)
    ann.index.name = "연도"
    ann = ann.reset_index()
    ann["연도"] = ann["연도"].astype(str)
    st.markdown(f"**연평균 {basis.split('(')[0]} 중립금리 (%)**")
    table(ann)
    res = Result(module=MODULE, analysis=f"방법 비교({basis})", inputs=_inputs(cfg, show),
                 summary={k: float(v) for k, v in E[["평균", "명목 중립금리(평균)", "실질 정책금리",
                                                    "정책 기조(실질 정책금리 − 중립금리)"]].dropna().iloc[-1].items()},
                 tables={"연평균": ann, "분기": q.drop(columns=["라벨"])},
                 notes=[f"[{e.name}] " + " ".join(e.notes) for e in ests.values()], sources=sources,
                 charts=[ExcelChart("분기", "분기", names + ["평균"], f"{basis} 중립금리 (%)", kind="line", top=None,
                                    label_skip=_step(len(q)))])
    excel_button(res, "nr_x_cmp")
    notes_box(res)


def _unc_table(ests: dict) -> pd.DataFrame | None:
    rows = []
    for k, e in ests.items():
        if e.sd is None:
            continue
        last = e.rstar.dropna().index[-1]
        v, sd = float(e.rstar.loc[last]), float(e.sd.loc[last])
        rows.append({"방법": e.name, "분기": _lab(last), "실질 중립금리(%)": round(v, 2), "표준오차(%p)": round(sd, 2),
                     "90% 구간(%)": f"{v - 1.645 * sd:.2f} ~ {v + 1.645 * sd:.2f}",
                     "계산": e.params.get("표준오차 계산", "")})
    return pd.DataFrame(rows) if rows else None


def stance_tab(E, ests, cfg, show, sources):
    T = E.loc[show:]
    q = _qframe(T, show)
    st.plotly_chart(charts.ts_lines(q, "라벨", ["평균", "실질 정책금리"], "실질 정책금리와 중립금리(방법 평균)", "%", 400,
                                    _step(len(q)), emphasize="평균", band=("최소", "최대")), width="stretch")
    st.plotly_chart(charts.ts_bar_line(q, "라벨", "정책 기조(실질 정책금리 − 중립금리)", [],
                                       "통화정책 기조: 실질 정책금리 − 중립금리 (플러스 = 긴축적)", "%p", 380, _step(len(q))),
                    width="stretch")
    st.caption("중립금리 추정의 불확실성(방법 간 범위)이 커서 기조 판단은 부호와 방향 위주로 보는 것이 적절함.")
    t = q[["분기", "라벨", "기준금리", "실질 정책금리", "평균", "최소", "최대", "정책 기조(실질 정책금리 − 중립금리)"]]
    res = Result(module=MODULE, analysis="통화정책 기조", inputs=_inputs(cfg, show), tables={"정책 기조": t.drop(columns=["라벨"])},
                 notes=["정책 기조 = 실질 정책금리 − 실질 중립금리(방법 평균). 플러스면 긴축적, 마이너스면 완화적.",
                        "실질 기준 비교이므로 '기준금리 − 명목 중립금리(실질 + 물가목표)'와는 기대인플레이션 − 물가목표 만큼 다름.",
                        "방법 평균은 그 분기에 값이 있는 방법만 평균함. 시장금리 방법은 2000Q4, VAR 은 2009Q1 부터 들어오므로 "
                        "그 시점에 평균이 계단처럼 움직일 수 있음."], sources=sources,
                 charts=[ExcelChart("정책 기조", "분기", ["정책 기조(실질 정책금리 − 중립금리)"], "정책 기조 (%p)", kind="col",
                                    top=None, label_skip=_step(len(t)))])
    excel_button(res, "nr_x_stance")


def detail_tab(E, ests, cfg, show, sources):
    keys = list(ests)
    k = st.selectbox("방법", keys, format_func=lambda x: f"[{N.GROUPS[N.GROUP_OF[x]]}] {N.METHODS[x]}", key="nr_det")
    e = ests[k]
    st.caption(HELP[k][0])
    T = pd.DataFrame({"실질 중립금리": e.rstar, "실질 정책금리": E["실질 정책금리"]}).loc[show:]
    for n_, t in e.extra.items():
        for c in t.columns:
            T[c] = t[c].reindex(T.index)
    band = None
    if e.sd is not None:
        T["90% 하단"] = e.rstar - 1.645 * e.sd
        T["90% 상단"] = e.rstar + 1.645 * e.sd
        T["표준오차(%p)"] = e.sd
        band = ("90% 하단", "90% 상단")
    q = _qframe(T, show)
    ys = [c for c in q.columns if c not in ("분기", "라벨", "90% 하단", "90% 상단", "표준오차(%p)")]
    st.plotly_chart(charts.ts_lines(q, "라벨", ys, f"{e.name}", "%", 420, _step(len(q)), emphasize="실질 중립금리",
                                    dashed=("실질 정책금리",), band=band, band_name="90% 구간(평활·모수 오차)"), width="stretch")
    if e.sd is not None:
        lv, ls = float(e.rstar.dropna().iloc[-1]), float(e.sd.dropna().iloc[-1])
        st.caption(f"음영: 90% 구간. {_lab(e.rstar.dropna().index[-1])} 실질 중립금리 {lv:.2f}%, 표준오차 {ls:.2f}%p "
                   f"→ 90% 구간 {lv - 1.645 * ls:.2f} ~ {lv + 1.645 * ls:.2f}%. " + e.notes[-1].split("표준오차:")[-1].split("그래프")[0].strip())
    st.markdown(f"**장점** {HELP[k][1]}  \n**한계** {HELP[k][2]}")
    pt = pd.DataFrame([{"항목": a, "값": f"{float(b):,.4g}" if isinstance(b, (int, float, np.floating)) else str(b)}
                       for a, b in e.params.items()])
    if len(pt):
        st.markdown("**추정 모수**")
        table(pt)
    res = Result(module=MODULE, analysis=e.name, inputs=_inputs(cfg, show), tables={"분기": q.drop(columns=["라벨"]), "모수": pt},
                 notes=e.notes, sources=sources,
                 charts=[ExcelChart("분기", "분기", ["실질 중립금리", "실질 정책금리"], f"{e.name} (%)", kind="line", top=None,
                                    label_skip=_step(len(q)))])
    excel_button(res, f"nr_x_det_{k}")
    notes_box(res)


def _sens_rows(df, dfx_fc, ests, cfg):
    """(변형, 다시 추정한 방법, 실행 인자) 목록."""
    variants, base = S.neutral_variants(cfg, dfx_fc is not None)
    jobs = []
    for v in variants:
        redo = [k for k in v.methods if k in cfg["methods"]]
        if not redo:
            continue
        c = {**cfg, **v.over, "methods": redo}
        dfx_v = df if v.data == "nofc" else dfx_fc if v.data == "fc" else (dfx_fc if cfg["fc"] else df)
        jobs.append((v, redo, (df, dfx_v, json.dumps(c, sort_keys=True))))
    return jobs, base


def sens_tab(df, dfx_fc, E, ests, cfg, sources):
    st.markdown("핵심 가정을 하나씩 바꿔 그 가정을 쓰는 방법만 다시 추정하고, 나머지 방법은 그대로 둔 채 마지막 분기의 "
                "**방법 평균 실질 중립금리**와 **정책 기조**가 얼마나 움직이는지 봅니다. 표준오차(방법 비교 탭)가 담지 못하는 "
                "'가정의 불확실성'을 보여 줍니다.")
    jobs, base = _sens_rows(df, dfx_fc, ests, cfg)
    if not jobs:
        st.info("바꿔 볼 가정을 쓰는 방법이 선택되지 않았습니다.")
        return
    todo = [j for j in jobs if not _run_disk.is_cached(*j[2])]
    sig = json.dumps(cfg, sort_keys=True, default=str)
    if todo and st.session_state.get("nr_sens_go") != sig:
        st.info(f"현재 설정의 민감도 결과가 저장돼 있지 않습니다. 변형 {len(todo)}개를 다시 추정해야 합니다"
                "(준구조모형이 들어간 변형은 하나에 수십 초). 기본 설정은 미리 계산해 두어 바로 뜹니다.")
        if st.button("민감도 분석 실행", key="nr_sens_btn", type="primary"):
            st.session_state["nr_sens_go"] = sig          # 이 설정에 대해서만 실행(설정을 바꾸면 다시 묻기)
            st.rerun()
        return
    last = E["평균"].dropna().index[-1]
    lab = _lab(last)
    rbase = N.real_rate(df if cfg["pie"] == "survey" else df.drop(columns=["pie"]))
    base_sum = S.neutral_summary(ests, rbase, last)
    rows = []
    bar = st.progress(0.0, text="민감도 분석 중…") if todo else None
    for i, (v, redo, args) in enumerate(jobs):
        try:
            new, errs = _run_disk(*args)
            comb = {k: e for k, e in ests.items()}
            comb.update(new)
            pie = v.over.get("pie", cfg["pie"])
            rr = N.real_rate(df if pie == "survey" else df.drop(columns=["pie"]))
            rows.append((v, redo, S.neutral_summary(comb, rr, last), "; ".join(errs.values()) or None))
        except Exception as ex:  # noqa: BLE001
            rows.append((v, redo, None, f"{type(ex).__name__}: {ex}"))
        if bar:
            bar.progress((i + 1) / len(jobs), text=f"민감도 분석 중… {i + 1}/{len(jobs)}")
    if bar:
        bar.empty()
    names = {k: N.METHODS[k] for k in N.METHODS}
    T = S.neutral_table(base_sum, rows, lab, names)
    T = T[[c for c in T.columns if not c.startswith("다시")] + [c for c in T.columns if c.startswith("다시")]]
    st.markdown(f"**기준**: 실질 중립금리 평균 {base_sum['avg']:.2f}%, 정책 기조 {base_sum['stance']:+.2f}%p ({lab})  \n"
                + " · ".join(f"{g}: {b}" for g, b in base.items()))
    num = [c for c in T.columns if T[c].dtype.kind == "f"]
    table(T.round({c: 2 for c in num}))
    sw = T.groupby("가정")["평균 변화(%p)"].agg(lambda x: x.abs().max()).sort_values(ascending=False)
    st.caption("평균에 가장 크게 영향을 주는 가정: " + ", ".join(f"{g} ±{v:.2f}%p" for g, v in sw.head(3).items())
               + ". 한 방법만 다시 추정하므로 평균 변화는 그 방법 변화 ÷ 방법 수 입니다.")
    res = Result(module=MODULE, analysis="민감도 분석", inputs=_inputs(cfg, "—"), tables={"민감도": T},
                 notes=["가정을 하나씩 바꾸고 그 가정을 쓰는 방법만 다시 추정, 나머지 방법은 기준 결과 유지.",
                        "기준: " + " · ".join(f"{g}: {b}" for g, b in base.items())], sources=sources)
    excel_button(res, "nr_x_sens")


def data_tab(df, dfx, cfg, show):
    base = df if cfg["pie"] == "survey" else df.drop(columns=["pie"])
    t = pd.DataFrame({"기준금리(%)": df["rate"], "기대인플레이션(%)": M.expected_inflation(base),
                      "실질 정책금리(%)": N.real_rate(base), "근원물가 상승률(%)": M._inflation(df["core"]),
                      "국고채 5년(%)": df.get("y5"), "국고채 10년(%)": df.get("y10"),
                      "5년 후 5년 선도금리(%)": (10 * df["y10"] - 5 * df["y5"]) / 5 if "y10" in df else np.nan,
                      "실질 GDP 성장률(전년동기비, %)": M.yoy(M.log100(df["gdp"]))}).loc[show:]
    q = _qframe(t, show)
    pick = st.selectbox("그래프로 볼 자료", list(t.columns), index=2, key="nr_data_pick")
    st.plotly_chart(charts.ts_lines(q, "라벨", [pick], pick, "%", 340, _step(len(q))), width="stretch")
    table(q.drop(columns=["라벨"]))
    excel_button(Result(module=MODULE, analysis="입력 자료", tables={"분기 자료": q.drop(columns=["라벨"])}, sources=SOURCES),
                 "nr_x_data")


def help_tab():
    st.markdown("중립금리(자연이자율)는 경제가 잠재 수준에서 물가 안정을 이룰 때의 실질 정책금리입니다. 직접 관측되지 않아 모형마다 "
                "값이 크게 다르므로, 여러 방법의 **범위**와 **평균**, 그리고 **변화 방향**을 함께 보는 것이 일반적입니다. "
                "명목 중립금리는 실질 중립금리에 물가목표(2%)를 더한 값입니다.")
    rows = [{"분류": N.GROUPS[N.GROUP_OF[k]], "방법": N.METHODS[k], "아이디어": i, "장점": p, "한계": c} for k, (i, p, c) in HELP.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=270)
    st.markdown(f"**한국은행 추정과 비교**  \n한국은행 「한국의 중립금리 추정」의 명목 중립금리 범위는 {BOK_RANGE[0]}~{BOK_RANGE[1]}%"
                "(중간값 2.55%, 언론 보도 기준)이며, 한국은행은 2026년 9월 현재 기준금리(3.00%)를 그 범위의 상단으로 평가하면서 "
                "잠재성장률과 함께 중립금리를 다시 추정하고 있다고 밝혔습니다. 이 도구의 DSGE 기반 추정은 아직 없습니다.")
    st.markdown("**불확실성과 민감도**  \n"
                "- UC 와 준구조모형(LW형)은 중립금리의 표준오차를 함께 냅니다(칼만 평활 오차 + 모수 추정 오차, Hamilton 1986). "
                "'방법 비교' 탭의 표와 '방법별 상세' 그래프의 90% 구간으로 보여 줍니다. 한국 자료에서는 IS곡선의 금리 민감도가 작게 "
                "추정돼 LW형 중립금리의 구간이 특히 넓습니다.  \n"
                "- 기대인플레이션 선택, 기간 프리미엄, 충격 크기 상한 같은 가정의 영향은 '민감도' 탭에서 하나씩 바꿔 봅니다.")
    st.markdown("**참고 문헌**  \n" + REFS)


# ---------------------------------------------------------------------------
def render():
    page_header("중립금리 추정",
                "통계·준구조·시계열·시장/성장 연계 모형 6가지로 실질·명목 중립금리를 추정하고 통화정책 기조(실질 정책금리 − 중립금리)를 "
                "평가합니다. 잠재성장률 화면의 추정과 연결됩니다.")
    df, meta = D.load_korea()
    with st.container(border=True):
        st.caption(f"한국 자료 · 기준금리 {df['rate'].dropna().index[-1]}까지 · 국고채 {df['y10'].dropna().index[-1]}까지 · "
                   f"실질 GDP {meta['last_gdp']}까지 · 자료 갱신 {meta['built']}")
        cfg, show, dfx = _settings(df)
    if not cfg["methods"]:
        st.warning("추정 방법을 하나 이상 고르세요.")
        return
    ests, errs = _run(df, dfx, json.dumps({k: v for k, v in cfg.items()}, sort_keys=True))
    for k, m in errs.items():
        st.warning(f"{N.METHODS[k]}: {m}")
    if not ests:
        st.error("추정 결과가 없습니다.")
        return
    E = _frame(df, ests, cfg)
    _cards(E, ests)
    tabs = st.tabs(["방법 비교", "통화정책 기조", "방법별 상세", "민감도", "입력 자료", "방법 설명"])
    with tabs[0]:
        compare_tab(E, ests, cfg, show, SOURCES)
    with tabs[1]:
        stance_tab(E, ests, cfg, show, SOURCES)
    with tabs[2]:
        detail_tab(E, ests, cfg, show, SOURCES)
    with tabs[3]:
        fc = F.load_bok()
        sens_tab(df, F.add_tail(F.extend(df, fc)[0], 2.0) if fc else None, E, ests, cfg, SOURCES)
    with tabs[4]:
        data_tab(df, dfx, cfg, show)
    with tabs[5]:
        help_tab()
    footer()
