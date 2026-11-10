"""산업연관표 분석 (한국은행 투입산출표)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import charts
from core.result import ExcelChart, Result
from core.ui import editor_hint, excel_button, footer, notes_box, num_col, page_header, pick_col, scenario_io, summary_cards, table
from engines.io import domestic as D
from engines.io.classify import ENERGY_IMPORT

META = {"title": "산업연관표 분석", "group": "산업연관 분석", "order": 10,
        "icon": ":material/account_tree:", "status": "ready", "slug": "io-domestic"}

MODULE = "산업연관표 분석"
SOURCE = ("한국은행, 산업연관표(투입산출표, 생산자가격) — 2015년·2020년 실측표 기준 계열, ECOS "
          "(총거래표·생산유발·수입유발·부가가치유발·취업유발·고용유발계수표)")
LEVEL_LABEL = {"대분류": "대분류 (33부문)", "중분류": "중분류 (83부문)"}
BENCH = {"2015", "2020"}

COMMON_NOTES = [
    "생산자가격 투입산출표 기준. 국산 투입계수는 한국은행 생산유발계수표 (I-Ad)^-1 에서 역산해 공표치와 정확히 일치시킴.",
    "최종수요의 국산·수입 구분: 최종수요용 수입(총수입 - 중간재 수입)이 수출을 뺀 최종수요 항목에 같은 비율로 들어간다고 가정. "
    "수출은 모두 국산으로 봄. 국산비율 계산 오차는 재고증감에 흡수.",
    "취업·고용계수는 공표 취업(고용)유발계수표 = diag(e)·L 에서 역산(명/십억원). 해당 연도 표의 계수이므로 이후 생산성 변화는 반영하지 않음.",
    "산업연관 모형은 고정 투입계수·유휴 생산능력·가격 불변(수량모형)을 가정하는 1차 파급 계산임. "
    "소득→소비의 2차 효과, 가격·금리·환율의 반응, 공급 제약은 들어 있지 않음.",
]


def _labels(t: D.KrIO) -> tuple[list[str], dict[str, str]]:
    labels = list(t.labels)
    return labels, {lab: c for lab, c in zip(labels, t.codes)}


def _code(lab: str, lab2code: dict) -> str:
    if lab == D.ALL:
        return D.ALL
    return lab2code.get(lab, lab.split(" ", 1)[0])


def _aggregate_view(tab: pd.DataFrame, level: str, value_cols: list[str], key: str) -> tuple[pd.DataFrame, str]:
    opts = ["부문", "산업군"] if level == "대분류" else ["부문", "대분류", "산업군"]
    by = st.radio("표시 단위", opts, horizontal=True, key=key)
    if by == "부문":
        return tab, "부문"
    g = D.aggregate(tab, by, value_cols)
    if by == "대분류":
        names = dict(zip(D.load_table("대분류", "2020").codes, D.load_table("대분류", "2020").names))
        g.insert(1, "부문", [names.get(c, c) for c in g["대분류"]])
    return g, by


# ---------------------------------------------------------------------------
# 수요 파급
# ---------------------------------------------------------------------------
DEMAND_PRESETS = {
    "수출 전반 10% 감소": [(D.ALL, "수출", "%", -10.0)],
    "반도체·전자 수출 10% 감소": [({"대분류": "C09", "중분류": "31"}, "수출", "%", -10.0)],
    "자동차 수출 10% 감소(관세 등)": [({"대분류": "C12", "중분류": "40"}, "수출", "%", -10.0)],
    "건설투자 5% 감소": [({"대분류": "F", "중분류": "50"}, "민간고정자본형성", "%", -5.0),
                    ({"대분류": "F", "중분류": "51"}, "정부고정자본형성", "%", -5.0)],
    "정부소비 10조원 확대": [(D.ALL, "정부소비", "십억원", 10000.0)],
    "민간소비 1% 증가": [(D.ALL, "민간소비", "%", 1.0)],
}


def _preset_rows(name: str, level: str, labels: list[str], code2lab: dict) -> list[dict]:
    rows = []
    for sec, comp, unit, val in DEMAND_PRESETS[name]:
        if isinstance(sec, dict):
            sec = sec[level]
            # 건설: 대분류는 F 하나, 중분류는 50/51 로 나뉨 → 대분류에서 두 항목 모두 F 로
        rows.append({"부문": code2lab.get(sec, D.ALL), "최종수요 항목": comp, "단위": unit, "크기": val})
    return rows


def demand_tab(t: D.KrIO, level: str):
    labels, lab2code = _labels(t)
    code2lab = {v: k for k, v in lab2code.items()}
    key = f"dom_demand_{level}"
    if key not in st.session_state:
        st.session_state[key] = _preset_rows("수출 전반 10% 감소", level, labels, code2lab)

    st.markdown("##### 최종수요 충격")
    c1, c2 = st.columns([3, 1])
    preset = c1.selectbox("예시 시나리오", ["(직접 입력)"] + list(DEMAND_PRESETS), key=f"{key}_preset")
    if c2.button("예시 불러오기", key=f"{key}_load", disabled=preset == "(직접 입력)", width="stretch"):
        st.session_state[key] = _preset_rows(preset, level, labels, code2lab)
        st.session_state.pop(f"{key}_editor", None)
        st.rerun()
    st.caption("% : 해당 부문·항목 최종수요(국산+수입)의 증감률.  십억원 : 증감액(국산+수입). "
               "‘전 부문’ + 십억원이면 그 항목의 부문 구성비대로 나눕니다. 여러 줄은 합산됩니다.")
    df_in = pd.DataFrame(st.session_state[key], columns=["부문", "최종수요 항목", "단위", "크기"])
    editor_hint()
    edited = st.data_editor(
        df_in, num_rows="dynamic", width="stretch", key=f"{key}_editor",
        column_config={
            "부문": pick_col("부문", options=[D.ALL] + labels, required=True, width="large"),
            "최종수요 항목": pick_col("최종수요 항목", options=[D.FD_ALL] + D.FD_NAMES, required=True),
            "단위": pick_col("단위", options=["%", "십억원"], required=True),
            "크기": num_col("크기", format="%.2f", required=True),
        })
    edited = edited.dropna()
    scenario_io(edited.to_dict("records"), key, (f"{key}_editor",), "충격")
    shocks = [{"sector": _code(r["부문"], lab2code), "component": r["최종수요 항목"], "unit": r["단위"],
               "value": float(r["크기"])} for r in edited.to_dict("records")]
    if not shocks:
        st.info("충격을 한 줄 이상 입력하세요.")
        return

    dF, dFd = D.build_demand_shock(t, shocks)
    tab = D.demand_effect(t, dF, dFd)
    summ = D.demand_summary(t, tab)

    st.markdown("##### 결과")
    summary_cards(summ)
    tab["표시명"] = [f"{c} {n}" for c, n in zip(tab["코드"], tab["부문"])]
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.bar_top(tab, "표시명", "부가가치 변화", 15, "부가가치 변화 상위 부문", "십억원"),
                    width="stretch")
    g2.plotly_chart(charts.bar_stacked(tab, "표시명", ["  직접(국산 최종수요)", "  간접(중간재 파급)"],
                                       "생산 변화: 직접·간접 (십억원)", 15, sort_by="생산 변화"),
                    width="stretch")
    value_cols = [c for c in tab.columns if c not in ("코드", "부문", "산업군", "표시명", "생산 변화율(%)")]
    view, by = _aggregate_view(tab.drop(columns=["표시명"]), level, value_cols, f"{key}_agg")
    st.caption("금액: 십억원(생산자가격) · 인원: 명 · 들여쓴 항목은 바로 위 항목의 내역")
    table(view)

    group = D.aggregate(tab, "산업군", value_cols)
    res = Result(
        module=MODULE, analysis="수요 파급",
        inputs={"자료": f"{t.year}년 산업연관표 ({'실측표' if t.year in BENCH else '연장표'}, {t.basis}년 기준)",
                "부문 분류": LEVEL_LABEL[level], "금액 단위": "십억원 (요약은 조원)", "인원 단위": "명",
                "충격 목록": edited.reset_index(drop=True)},
        summary=summ,
        tables={"부문별 결과": tab.drop(columns=["표시명"]), "산업군별 결과": group},
        notes=COMMON_NOTES + ["생산 변화 = L × 국산 최종수요 변화. 부가가치 변화 = 부가가치율 × 생산 변화. "
                              "수입 변화 = 중간재 수입(Am × 생산 변화) + 최종재 수입. 부가가치 변화율은 해당 연도 총부가가치(기초가격) 대비."],
        sources=[SOURCE],
        charts=[ExcelChart("부문별 결과", "부문", ["부가가치 변화"], "부가가치 변화 (십억원)", "bar", 20),
                ExcelChart("산업군별 결과", "산업군", ["생산 변화", "부가가치 변화"], "산업군별 변화 (십억원)", "col", None)],
    )
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 가격 파급
# ---------------------------------------------------------------------------
def price_tab(t: D.KrIO, level: str):
    labels, lab2code = _labels(t)
    key = f"dom_price_{level}"
    st.markdown("##### 비용 충격")
    c1, c2, c3 = st.columns(3)
    fx = c1.number_input("원화 환율 상승률(%, 원화 약세 +)", value=10.0, step=1.0, key=f"{key}_fx")
    fx_pt = c1.slider("환율 → 수입물가 전가율(%)", 0, 100, 100, key=f"{key}_fxpt")
    oil = c2.number_input("국제유가(원유·석유제품 수입가격) 상승률(%)", value=0.0, step=5.0, key=f"{key}_oil")
    energy = ENERGY_IMPORT[level]
    c2.caption("적용 부문: " + ", ".join(lab for lab in labels if lab.split(" ")[0] in energy)
               + (" — 대분류 ‘광산품’은 석탄·철광석 등도 포함하므로 중분류가 더 정확합니다." if level == "대분류" else ""))
    wage = c3.number_input("단위당 피용자보수(임금) 상승률(%, 전 부문)", value=0.0, step=1.0, key=f"{key}_wage")
    prof = c3.number_input("단위당 영업잉여 상승률(%, 전 부문)", value=0.0, step=1.0, key=f"{key}_prof")
    tax = c3.number_input("순생산물세 상승률(%, 전 부문)", value=0.0, step=1.0, key=f"{key}_tax")
    with st.expander("부문별 수입가격·부가가치 개별 충격 (선택)"):
        st.caption("여기 넣은 값은 위의 공통 충격에 더해집니다.")
        editor_hint()
        ed = st.data_editor(pd.DataFrame(columns=["부문", "항목", "변화율(%)"]), num_rows="dynamic",
                            width="stretch", key=f"{key}_editor",
                            column_config={
                                "부문": pick_col("부문", options=labels, required=True, width="large"),
                                "항목": pick_col("항목", options=["수입가격"] + D.VA_NAMES, required=True),
                                "변화율(%)": num_col("변화율(%)", format="%.2f", required=True)})
    passthrough = st.slider("비용 상승의 국내 가격 전가율(%)", 0, 100, 100, key=f"{key}_pt",
                            help="레온티에프 가격모형은 100%(완전 전가)를 가정합니다. 낮추면 결과가 비례해 줄어듭니다.") / 100

    imp = np.full(t.n, fx * fx_pt / 100)
    for c in energy:
        imp[t.codes == c] += oil
    va = {"피용자보수": np.full(t.n, wage), "영업잉여": np.full(t.n, prof), "순생산물세": np.full(t.n, tax)}
    for r in ed.dropna().to_dict("records"):
        j = t.codes == _code(r["부문"], lab2code)
        if r["항목"] == "수입가격":
            imp[j] += r["변화율(%)"]
        else:
            va.setdefault(r["항목"], np.zeros(t.n))
            va[r["항목"]][j] += r["변화율(%)"]
    if not np.any(imp) and not any(np.any(v) for v in va.values()):
        st.info("충격을 하나 이상 입력하세요.")
        return
    tab, summ = D.price_effect(t, imp, va, passthrough)

    st.markdown("##### 결과")
    summary_cards(summ, per_row=5)
    tab["표시명"] = [f"{c} {n}" for c, n in zip(tab["코드"], tab["부문"])]
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.bar_stacked(tab, "표시명", ["  수입 중간재 기여(%)", "  부가가치 기여(%)"],
                                       "생산자가격 상승률 상위 부문 (%)", 15, sort_by="생산자가격 변화(%)"),
                    width="stretch")
    g2.plotly_chart(charts.bar_top(tab, "표시명", "소비자물가 기여(%p)", 15, "소비자물가 상승 기여 상위 부문", "%p"),
                    width="stretch")
    show = tab.drop(columns=["표시명"])
    table(show)
    inputs = {"자료": f"{t.year}년 산업연관표 ({t.basis}년 기준)", "부문 분류": LEVEL_LABEL[level],
              "환율 상승률(%)": fx, "환율→수입물가 전가율(%)": fx_pt, "국제유가 상승률(%)": oil,
              "유가 적용 부문": ", ".join(energy), "임금(단위당) 상승률(%)": wage,
              "영업잉여(단위당) 상승률(%)": prof, "순생산물세 상승률(%)": tax, "국내 가격 전가율(%)": passthrough * 100,
              "개별 충격": ed.dropna().reset_index(drop=True)}
    res = Result(
        module=MODULE, analysis="가격 파급", inputs=inputs, summary=summ,
        tables={"부문별 가격 변화": show},
        notes=COMMON_NOTES[:1] + [
            "레온티에프 가격모형: Δp' = (Δπ'·Am + Δw')·L. 수입품 가격(Δπ)과 단위당 부가가치(Δw) 변화가 국산 중간재를 거쳐 전부 전가된다고 가정(전가율로 비례 조정).",
            "소비자물가 = 가계소비 지출 구성(국산+수입, 생산자가격) 가중. 국산품은 Δp, 수입 소비재는 Δπ 가 오른다고 봄. "
            "유통마진·세금·주거비 귀속임대료 등 CPI 바스켓과의 차이 때문에 실제 CPI 영향과 다를 수 있음.",
            "임금 상승은 ‘산출 1단위당’ 피용자보수 증가로 해석(생산성 불변 가정). 수요 감소·대체효과는 반영하지 않음.",
        ],
        sources=[SOURCE],
        charts=[ExcelChart("부문별 가격 변화", "부문", ["생산자가격 변화(%)"], "생산자가격 변화 (%)", "bar", 20)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 공급 차질
# ---------------------------------------------------------------------------
def supply_tab(t: D.KrIO, level: str):
    labels, lab2code = _labels(t)
    code2lab = {v: k for k, v in lab2code.items()}
    key = f"dom_supply_{level}"
    st.markdown("##### 생산 차질")
    default = {"대분류": "C09", "중분류": "31"}[level]
    editor_hint()
    ed = st.data_editor(pd.DataFrame([{"부문": code2lab[default], "생산 차질률(%)": 10.0}]), num_rows="dynamic",
                        width="stretch", key=f"{key}_editor",
                        column_config={
                            "부문": pick_col("부문", options=labels, required=True, width="large"),
                            "생산 차질률(%)": num_col("생산 차질률(%)", min_value=0.0, max_value=100.0, format="%.1f")})
    cut = np.zeros(t.n)
    for r in ed.dropna().to_dict("records"):
        cut[t.codes == _code(r["부문"], lab2code)] = r["생산 차질률(%)"]
    if not cut.any():
        st.info("차질 부문을 한 줄 이상 입력하세요.")
        return
    tab, summ = D.supply_effect(t, cut)
    st.markdown("##### 결과")
    summary_cards(summ, per_row=5)
    tab["표시명"] = [f"{c} {n}" for c, n in zip(tab["코드"], tab["부문"])]
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.bar_top(tab, "표시명", "전방 부가가치 변화", 15, "전방(공급망 하류) 파급: 부가가치", "십억원"),
                    width="stretch")
    g2.plotly_chart(charts.bar_top(tab, "표시명", "후방 부가가치 변화", 15, "후방(공급망 상류) 파급: 부가가치", "십억원"),
                    width="stretch")
    show = tab.drop(columns=["표시명"])
    st.caption("금액: 십억원(생산자가격)")
    table(show)
    res = Result(
        module=MODULE, analysis="공급 차질",
        inputs={"자료": f"{t.year}년 산업연관표", "부문 분류": LEVEL_LABEL[level], "차질 부문": ed.dropna().reset_index(drop=True)},
        summary=summ, tables={"부문별 결과": show},
        notes=COMMON_NOTES[:1] + [
            "전방 파급(Ghosh 공급모형): 차질 부문 산출이 줄면 이를 중간재로 쓰는 하류 부문 산출이 배분계수만큼 감소한다고 가정(대체 불가·재고 없음 → 상한).",
            "후방 파급(레온티에프): 차질 부문이 사들이던 국산 중간재 수요가 줄어 상류 부문 생산이 감소.",
            "두 결과는 서로 다른 극단 가정이므로 더하지 않고 나란히 제시. 실제 영향은 대체·재고·수입 대체 정도에 따라 그 사이에 위치하는 경우가 많음.",
        ],
        sources=[SOURCE],
        charts=[ExcelChart("부문별 결과", "부문", ["전방 부가가치 변화", "후방 부가가치 변화"], "부가가치 변화 (십억원)", "bar", 20)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 연관 구조
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="연관 지표를 계산하는 중…")
def _multipliers(level: str, year: str) -> pd.DataFrame:
    return D.multipliers(D.load_table(level, year))


@st.cache_data
def _over_time(level: str, code: str) -> pd.DataFrame:
    return D.multipliers_over_time(level, code)


def structure_tab(t: D.KrIO, level: str):
    labels, lab2code = _labels(t)
    key = f"dom_struct_{level}"
    m = _multipliers(level, t.year)
    tot = {
        "총산출(조원)": t.x.sum() / 1e6, "총부가가치(조원)": t.total_va / 1e6,
        "평균 생산유발계수": float(m["생산유발계수"].mean()),
        "부가가치유발계수(최종수요 가중)": float((t.v @ t.L) @ t.Fd.sum(1) / t.Fd.sum()),
        "중간투입 수입의존도(%)": float((t.Am @ t.x).sum() / t.Z.sum() * 100),
    }
    summary_cards(tot, per_row=5, signed=False)
    m2 = m.copy()
    m2["표시명"] = m2["코드"]
    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(charts.scatter_quadrant(m2, "영향력계수", "감응도계수", "표시명", "총산출(조원)",
                                            "영향력계수(후방) × 감응도계수(전방) — 원 크기 = 총산출"),
                    width="stretch")
    with g2:
        st.markdown("**네 유형**")
        st.markdown("- Ⅰ 핵심: 다른 산업에서 많이 사고(후방) 많이 팔림(전방)\n- Ⅱ 후방 연관 큼: 최종재 성격\n"
                    "- Ⅲ 전방 연관 큼: 중간재·기반 산업\n- Ⅳ 연관 작음")
        st.dataframe(m["유형"].value_counts().rename("부문 수"), width="stretch")
    table(m)

    st.markdown("##### 연도별 추이 (2015~2024년)")
    pick = st.selectbox("부문", labels, index=0, key=f"{key}_pick")
    ts = _over_time(level, _code(pick, lab2code))
    cols = st.multiselect("지표", [c for c in ts.columns if c != "연도"],
                          default=["생산유발계수", "부가가치유발계수", "수입유발계수"], key=f"{key}_cols")
    if cols:
        st.plotly_chart(charts.lines(ts, "연도", cols, pick), width="stretch")
    st.caption("2015–2019년은 2015년 기준, 2020–2024년은 2020년 기준 계열입니다. 실측표는 2015·2020년, 나머지는 연장표입니다.")
    res = Result(
        module=MODULE, analysis="연관 구조",
        inputs={"자료": f"{t.year}년 산업연관표", "부문 분류": LEVEL_LABEL[level], "추이 부문": pick},
        summary=tot, tables={"유발계수·연관지표": m, f"추이_{_code(pick, lab2code)}": ts},
        notes=COMMON_NOTES[:1] + [
            "생산유발계수 = L 의 열합. 부가가치·수입유발계수 = 부가가치율·수입계수 × L (둘의 합 = 1). 취업·고용유발계수 = 취업·고용계수 × L.",
            "영향력계수 = 열합 / 열합 평균(후방 연관), 감응도계수 = 행합 / 행합 평균(전방 연관). 한국은행 공표 방식과 같음.",
            "가설추출 손실 = 해당 부문의 행·열을 국산 투입계수에서 지웠을 때 ‘다른 부문’ 총산출 감소분(총산출 대비 %). 국산 최종수요는 그대로 둠.",
        ],
        sources=[SOURCE],
        charts=[ExcelChart("유발계수·연관지표", "부문", ["생산유발계수"], "생산유발계수", "bar", 33)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


def render():
    page_header("산업연관표 분석", "한국은행 산업연관표로 수요·비용·공급 충격의 산업 간 파급효과를 계산합니다.")
    years = D.available_years("대분류")
    c1, c2, c3 = st.columns([1, 1.3, 3])
    year = c1.selectbox("자료 연도", years[::-1], index=0, key="dom_year")
    level = c2.selectbox("부문 분류", D.LEVELS, format_func=lambda x: LEVEL_LABEL[x], key="dom_level")
    t = D.load_table(level, year)
    kind = "실측표" if year in BENCH else "연장표"
    c3.markdown(f"<div style='padding-top:1.9rem;color:#666'>{year}년 {kind} · {t.basis}년 기준 · {t.n}부문 · "
                f"총산출 {t.x.sum() / 1e6:,.0f}조원 · 총부가가치 {t.total_va / 1e6:,.0f}조원</div>",
                unsafe_allow_html=True)
    tabs = st.tabs(["수요 파급", "가격 파급", "공급 차질", "연관 구조"])
    with tabs[0]:
        demand_tab(t, level)
    with tabs[1]:
        price_tab(t, level)
    with tabs[2]:
        supply_tab(t, level)
    with tabs[3]:
        structure_tab(t, level)
    footer()
