"""국제 산업연관표 분석 (OECD ICIO)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import charts
from core.result import ExcelChart, Result
from core.ui import excel_button, footer, notes_box, page_header, row_editor, run_gate, scenario_io, summary_cards, table
from engines.io import icio as I
from engines.io.icio_meta import COUNTRY_GROUPS, COUNTRY_KO, FD_KO, INDUSTRY_KO

META = {"title": "국제 산업연관표 분석", "group": "산업연관 분석", "order": 20,
        "icon": ":material/public:", "status": "ready", "slug": "io-international"}

MODULE = "국제 산업연관표 분석"
SOURCE_TMPL = "OECD (2023), Inter-Country Input-Output (ICIO) Tables, {ed}년판, https://oe.cd/icio — 기초가격, 백만 달러"
GOODS = "상품 전체(농림어업·광업·제조업)"
ALL_IND = "전 산업"
COMMON_NOTES = [
    "OECD 국제산업연관표(ICIO)의 국가 간 투입계수 A = Z/x 와 레온티에프 역행렬 L = (I-A)^-1 (LU 분해로 계산) 사용.",
    "고정 투입계수·유휴 생산능력·가격 불변의 1차 파급 계산. 소득→소비 2차 효과, 환율·금리·가격의 반응, 재고 조정은 반영하지 않음.",
    "GDP 대비 = 부가가치(기초가격) 변화 / 해당 연도 국가별 부가가치 합계. 금액은 해당 연도 경상 달러.",
]


@st.cache_resource(show_spinner="국제산업연관표를 불러오는 중…(처음 한 번 수 초)", max_entries=2)
def _load(ed: str, yr: str) -> I.ICIO:
    t = I.load(ed, yr)
    _ = t.lu
    return t


def _country_opts(t: I.ICIO, groups: bool = True, world: bool = True) -> list[str]:
    opts = ([I.WORLD] if world else []) + (list(COUNTRY_GROUPS) if groups else [])
    return opts + sorted(t.countries, key=lambda c: (c == "ROW", COUNTRY_KO.get(c, c)))


def _clab(c: str) -> str:
    return f"{COUNTRY_KO[c]} ({c})" if c in COUNTRY_KO else c


def _ilab(s: str) -> str:
    return f"{INDUSTRY_KO.get(s, s)} ({s})" if s in INDUSTRY_KO else s


def _goods(t: I.ICIO) -> list[str]:
    return [s for s in t.industries if s[0] in "ABC"]


def _with_groups(t: I.ICIO, ctab: pd.DataFrame) -> pd.DataFrame:
    g = I.group_rows(t, ctab)
    return pd.concat([ctab, g], ignore_index=True) if len(g) else ctab


def _country_charts(t, ctab, col, title, focus, unit="%"):
    d = ctab[ctab["국가코드"] != "ROW"].copy()
    d["표시"] = d["국가"]
    g1, g2 = st.columns([1, 1])
    g1.plotly_chart(charts.bar_top(d, "표시", col, 20, title, unit), width="stretch")
    return g2


def _focus_industry_chart(container, t, focus, vec, title, unit="백만 달러", n=15):
    df = I.industry_table(t, focus, {"v": vec})
    df["표시"] = df["산업"]
    container.plotly_chart(charts.bar_top(df, "표시", "v", n, title, unit), width="stretch")


def _heat(t, ctab, vec, n=20, title="부가가치 변화율 (국가 × 산업군, %)"):
    top = ctab[ctab["국가코드"] != "ROW"].reindex(
        ctab[ctab["국가코드"] != "ROW"]["GDP 대비(%)"].abs().sort_values(ascending=False).index[:n])
    st.plotly_chart(charts.heatmap(I.heat_matrix(t, vec, list(top["국가코드"])), title, "%"), width="stretch")


# ---------------------------------------------------------------------------
# 1. 국가 간 수요 파급
# ---------------------------------------------------------------------------
DEMAND_PRESETS = {
    "중국 투자 5% 감소": [("CHN", "GFCF", "전체", ALL_IND, -5.0)],
    "중국 내수(가계소비·투자) 3% 감소": [("CHN", "HFCE", "전체", ALL_IND, -3.0), ("CHN", "GFCF", "전체", ALL_IND, -3.0)],
    "미국 가계소비 2% 감소": [("USA", "HFCE", "전체", ALL_IND, -2.0)],
    "유로지역 최종수요 1% 감소": [("유로지역(20)", I.FD_ALL, "전체", ALL_IND, -1.0)],
    "미국 IT기기 수입수요 10% 감소": [("USA", I.FD_ALL, "수입", "C26", -10.0)],
}
FD_OPTS = [I.FD_ALL] + I.FD_CODES
FD_LABEL = {I.FD_ALL: "최종수요 전체", **FD_KO}


def demand_tab(t: I.ICIO, focus: str):
    key = "icio_demand"
    copts = _country_opts(t, world=True)
    if key not in st.session_state:
        st.session_state[key] = DEMAND_PRESETS["중국 투자 5% 감소"]
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    preset = c1.selectbox("예시 시나리오", ["(직접 입력)"] + list(DEMAND_PRESETS), key=f"{key}_preset")
    if c2.button("예시 불러오기", key=f"{key}_load", disabled=preset == "(직접 입력)", width="stretch"):
        st.session_state[key] = DEMAND_PRESETS[preset]
        st.session_state.pop(f"{key}_editor", None)
        st.rerun()
    st.caption("최종수요를 낸 나라(수요국)와 항목을 고르고 변화율을 넣습니다. 원산지: 전체 / 국산(자국 생산품) / 수입(외국 생산품). "
               "산업을 고르면 그 산업 생산품에 대한 수요만 바뀝니다.")
    df_in = pd.DataFrame([{"수요국": _clab(c), "항목": FD_LABEL[f], "원산지": o, "산업": _ilab(s) if s != ALL_IND else ALL_IND, "변화율(%)": v}
                          for c, f, o, s, v in st.session_state[key]])
    ed = row_editor(f"{key}_editor", df_in, {
        "수요국": {"options": [_clab(c) for c in copts], "w": 2.5},
        "항목": {"options": [FD_LABEL[f] for f in FD_OPTS], "w": 2.2},
        "원산지": {"options": ["전체", "국산", "수입"], "w": 1.2},
        "산업": {"options": [ALL_IND] + [_ilab(s) for s in t.industries], "w": 3},
        "변화율(%)": {"num": True, "fmt": "%.2f", "step": 1.0, "w": 1.4},
    })
    ran = run_gate(key, ed.to_dict("records"))
    lab2c = {_clab(c): c for c in copts}
    lab2f = {v: k for k, v in FD_LABEL.items()}
    lab2s0 = {_ilab(s): s for s in t.industries}
    scenario_io([[lab2c.get(r["수요국"], r["수요국"]), lab2f.get(r["항목"], I.FD_ALL), r["원산지"],
                  lab2s0.get(r["산업"], ALL_IND), r["변화율(%)"]] for r in ed.dropna().to_dict("records")],
                key, (f"{key}_editor",), "충격")
    if ran is None:
        return
    lab2s = {_ilab(s): s for s in t.industries}
    shocks = []
    for r in ran:
        if r["변화율(%)"] == 0:
            continue
        s = lab2s.get(r["산업"])
        shocks.append({"country": lab2c.get(r["수요국"], r["수요국"]), "component": lab2f.get(r["항목"], I.FD_ALL),
                       "origin": r["원산지"], "industries": [s] if s else [], "pct": float(r["변화율(%)"])})
    if not shocks:
        st.info("충격을 한 줄 이상 입력하세요.")
        return
    dy = I.build_demand(t, shocks)
    ctab, cells, dva, dx, summ = I.demand_effect(t, dy)
    frow = ctab[ctab["국가코드"] == focus].iloc[0]
    summ[f"{COUNTRY_KO[focus]} GDP 대비(%)"] = frow["GDP 대비(%)"]
    summ[f"{COUNTRY_KO[focus]} 부가가치(10억 달러)"] = frow["부가가치 변화(백만 달러)"] / 1e3
    summary_cards(summ, per_row=5)
    g2 = _country_charts(t, ctab, "GDP 대비(%)", "GDP 영향이 큰 나라 (부가가치 변화, GDP 대비 %)", focus)
    _focus_industry_chart(g2, t, focus, dva, f"{COUNTRY_KO[focus]}: 산업별 부가가치 변화 (백만 달러)")
    _heat(t, ctab, dva)
    full = _with_groups(t, ctab)
    table(full.sort_values("GDP 대비(%)").reset_index(drop=True))
    ind = I.industry_table(t, focus, {"최종수요 변화(백만 달러)": dy, "산출 변화(백만 달러)": dx, "부가가치 변화(백만 달러)": dva})
    res = Result(
        module=MODULE, analysis="국가 간 수요 파급",
        inputs={"자료": f"OECD ICIO {t.edition}년판 {t.year}년 표", "관심 국가": _clab(focus), "단위": "백만 달러(요약은 10억 달러)",
                "충격 목록": pd.DataFrame(ran)},
        summary=summ,
        tables={"국가별 결과": full, f"{COUNTRY_KO[focus]} 산업별": ind,
                "국가×산업 전체": cells[cells[["최종수요 변화", "산출 변화", "부가가치 변화"]].abs().sum(1) > 1e-9]},
        notes=COMMON_NOTES + ["Δx = L·Δy, Δ부가가치 = v ⊙ Δx. 충격은 수요국의 원산지·산업별 최종수요(ICIO 최종수요 행렬)에 변화율을 곱해 만듦."],
        sources=[SOURCE_TMPL.format(ed=t.edition)],
        charts=[ExcelChart("국가별 결과", "국가", ["GDP 대비(%)"], "GDP 대비 부가가치 변화 (%)", "bar", 20)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 2. 관세
# ---------------------------------------------------------------------------
TARIFF_PRESETS = {
    "미국 보편관세 10%": dict(imp=["USA"], exp=[I.WORLD], ind=GOODS, tau=10.0),
    "미국 → 중국 상품 60%": dict(imp=["USA"], exp=["CHN"], ind=GOODS, tau=60.0),
    "미국 → 한국 상품 25%": dict(imp=["USA"], exp=["KOR"], ind=GOODS, tau=25.0),
    "미국 → EU 자동차 25%": dict(imp=["USA"], exp=["EU(27)"], ind=["C29"], tau=25.0),
    "미국 → 전 세계 반도체·전자 25%": dict(imp=["USA"], exp=[I.WORLD], ind=["C26"], tau=25.0),
    "중국 → 미국 상품 34%(보복)": dict(imp=["CHN"], exp=["USA"], ind=GOODS, tau=34.0),
}


def _apply_tariff_preset():
    p = TARIFF_PRESETS.get(st.session_state.get("tar_preset"))
    if not p:
        return
    st.session_state["tar_imp"] = [_clab(c) for c in p["imp"]]
    st.session_state["tar_exp"] = [_clab(c) for c in p["exp"]]
    st.session_state["tar_ind"] = [GOODS] if p["ind"] == GOODS else [_ilab(s) for s in p["ind"]]
    st.session_state["tar_tau"] = p["tau"]


def tariff_tab(t: I.ICIO, focus: str):
    copts = _country_opts(t, world=True)
    st.session_state.setdefault("tar_imp", [_clab("USA")])
    st.session_state.setdefault("tar_exp", [_clab(I.WORLD)])
    st.session_state.setdefault("tar_ind", [GOODS])
    st.session_state.setdefault("tar_tau", 10.0)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.selectbox("예시 시나리오", ["(직접 입력)"] + list(TARIFF_PRESETS), key="tar_preset")
    c2.button("예시 불러오기", on_click=_apply_tariff_preset, width="stretch", key="tar_load")
    a, b, c = st.columns(3)
    imp = a.multiselect("관세 부과국(수입국)", [_clab(x) for x in copts if x != I.WORLD], key="tar_imp")
    exp = b.multiselect("대상 수출국", [_clab(x) for x in copts], key="tar_exp",
                        help="‘전 세계’를 고르면 수입국 자신을 뺀 모든 나라")
    ind = c.multiselect("대상 산업", [GOODS] + [_ilab(s) for s in t.industries], key="tar_ind",
                        help="관세는 상품에만 붙으므로 기본값은 농림어업·광업·제조업 전체")
    d1, d2, d3, d4, d5 = st.columns(5)
    tau = d1.number_input("관세율(%p 인상)", min_value=0.0, max_value=300.0, step=5.0, key="tar_tau")
    eps = d2.number_input("수입 탄력성", value=2.0, min_value=0.0, max_value=10.0, step=0.5, key="tar_eps",
                          help="수입수요의 가격탄력성 ε. 수입 감소율 = 1-(1+관세율×전가율)^(-ε). 무역 탄력성 실증치는 대체로 2~5")
    sd = d3.slider("국내 대체(%)", 0, 100, 30, key="tar_sd", help="줄어든 수입 중 수입국 국내 생산으로 바뀌는 몫")
    s3 = d4.slider("제3국 대체(%)", 0, 100 - sd, min(30, 100 - sd), key="tar_s3",
                   help="줄어든 수입 중 관세 비대상 국가의 같은 산업 제품으로 바뀌는 몫(기존 점유율대로)")
    pt = d5.slider("가격 전가율(%)", 0, 100, 100, key="tar_pt", help="관세가 수입가격에 반영되는 비율")
    lab2c = {_clab(x): x for x in copts}
    imp_c = [lab2c[x] for x in imp]
    exp_c = [lab2c[x] for x in exp]
    inds = _goods(t) if GOODS in ind else [x.rsplit("(", 1)[1][:-1] for x in ind]
    if not imp_c or not exp_c or not inds:
        st.info("수입국·수출국·산업을 고르세요.")
        return
    p = run_gate("tar", {"imp": imp_c, "exp": exp_c, "inds": inds, "tau": tau, "eps": eps, "sd": sd, "s3": s3, "pt": pt})
    if p is None:
        return
    imp_c, exp_c, inds, tau, eps, sd, s3, pt = (p[k] for k in ("imp", "exp", "inds", "tau", "eps", "sd", "s3", "pt"))
    ctab, cells, dva, dx, summ, info = I.tariff_effect(t, imp_c, exp_c, inds, tau, eps, sd / 100, s3 / 100, pt / 100)
    frow = ctab[ctab["국가코드"] == focus].iloc[0]
    summ[f"{COUNTRY_KO[focus]} GDP 대비(%)"] = frow["GDP 대비(%)"]
    summary_cards(summ, per_row=4)
    g2 = _country_charts(t, ctab, "GDP 대비(%)", "GDP 영향 (부가가치 변화, GDP 대비 %)", focus)
    _focus_industry_chart(g2, t, focus, dva, f"{COUNTRY_KO[focus]}: 산업별 부가가치 변화 (백만 달러)")
    _heat(t, ctab, dva)
    full = _with_groups(t, ctab)
    table(full.sort_values("GDP 대비(%)").reset_index(drop=True))
    ind_tab = I.industry_table(t, focus, {"수요 이동(백만 달러)": cells["수요 이동"].to_numpy(),
                                          "산출 변화(백만 달러)": dx, "부가가치 변화(백만 달러)": dva,
                                          "생산자가격 변화(%)": cells["생산자가격 변화(%)"].to_numpy()})
    res = Result(
        module=MODULE, analysis="관세 시나리오",
        inputs={"자료": f"OECD ICIO {t.edition}년판 {t.year}년 표", "관세 부과국": ", ".join(imp), "대상 수출국": ", ".join(exp),
                "대상 산업": ", ".join(ind), "관세율(%p)": tau, "수입 탄력성": eps, "수입 감소율(%)": info["r"] * 100,
                "국내 대체(%)": sd, "제3국 대체(%)": s3, "가격 전가율(%)": pt, "관심 국가": _clab(focus)},
        summary=summ,
        tables={"국가별 결과": full, f"{COUNTRY_KO[focus]} 산업별": ind_tab},
        notes=COMMON_NOTES + [
            "수량: 대상 수입(중간재+최종재)이 등탄력 수입수요 1-(1+관세율×전가율)^(-탄력성) 만큼 줄고, 그 일부가 수입국 국내 생산·제3국 공급으로 옮겨간다고 보고 "
            "수요 이동을 L 로 파급(1차 근사). 관세로 인한 수입국 실질소득 감소, 환율 조정, 보복·우회 수출은 넣지 않음.",
            "가격: 수입국 산업이 쓰는 대상 수입 중간재 비용이 관세만큼 오르고 국제 가격모형(Δp' = Δc'·L)으로 전가. "
            "소비자물가는 수입국 가계소비(원산지별) 가중이며 대상 수입 소비재는 관세만큼 직접 오른다고 가정.",
            "국내 대체가 크면 관세 부과국 GDP 가 늘어나는 것으로 계산될 수 있음. 이는 가격 상승에 따른 수요 위축을 뺀 1차 효과이므로 해석에 주의.",
        ],
        sources=[SOURCE_TMPL.format(ed=t.edition)],
        charts=[ExcelChart("국가별 결과", "국가", ["GDP 대비(%)"], "GDP 대비 부가가치 변화 (%)", "bar", 20)])
    excel_button(res, "tar_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 3. 공급망 차질
# ---------------------------------------------------------------------------
SUPPLY_PRESETS = {
    "대만 반도체·전자 생산 50% 차질": [("TWN", "C26", 50.0)],
    "중국 화학·1차금속 20% 차질": [("CHN", "C20", 20.0), ("CHN", "C24", 20.0)],
    "독일 자동차 생산 20% 차질": [("DEU", "C29", 20.0)],
    "한국 반도체·전자 생산 20% 차질": [("KOR", "C26", 20.0)],
    "러시아 에너지 공급 50% 차질": [("RUS", "B05_06", 50.0)],
}


def supply_tab(t: I.ICIO, focus: str):
    key = "icio_supply"
    clist = [c for c in _country_opts(t, groups=False, world=False)]
    if key not in st.session_state:
        st.session_state[key] = SUPPLY_PRESETS["대만 반도체·전자 생산 50% 차질"]
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    preset = c1.selectbox("예시 시나리오", ["(직접 입력)"] + list(SUPPLY_PRESETS), key=f"{key}_preset")
    if c2.button("예시 불러오기", key=f"{key}_load", disabled=preset == "(직접 입력)", width="stretch"):
        st.session_state[key] = [p for p in SUPPLY_PRESETS[preset] if p[1] in t.si]
        st.session_state.pop(f"{key}_editor", None)
        st.rerun()
    df_in = pd.DataFrame([{"국가": _clab(c), "산업": _ilab(s), "생산 차질률(%)": v} for c, s, v in st.session_state[key]])
    ed = row_editor(f"{key}_editor", df_in, {
        "국가": {"options": [_clab(c) for c in clist], "w": 2.5},
        "산업": {"options": [_ilab(s) for s in t.industries], "w": 4},
        "생산 차질률(%)": {"num": True, "min": 0.0, "max": 100.0, "fmt": "%.1f", "step": 5.0, "w": 1.6},
    })
    ran = run_gate(key, ed.to_dict("records"))
    if ran is None:
        return
    lab2c = {_clab(c): c for c in clist}
    lab2s = {_ilab(s): s for s in t.industries}
    cuts = {}
    for r in ran:
        if r["생산 차질률(%)"] > 0:
            cuts[t.idx(lab2c[r["국가"]], lab2s[r["산업"]])] = float(r["생산 차질률(%)"])
    if not cuts:
        st.info("차질 국가·산업을 한 줄 이상 입력하세요.")
        return
    with st.spinner("공급 측 역행렬을 계산하는 중…"):
        ctab, cells, fwd, bwd, summ = I.supply_effect(t, cuts)
    fr = ctab[ctab["국가코드"] == focus].iloc[0]
    summ[f"{COUNTRY_KO[focus]} 전방 GDP(%)"] = fr["전방 GDP 대비(%)"]
    summ[f"{COUNTRY_KO[focus]} 후방 GDP(%)"] = fr["후방 GDP 대비(%)"]
    summary_cards(summ, per_row=5)
    d = ctab[ctab["국가코드"] != "ROW"].copy()
    d["표시"] = d["국가"]
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.bar_top(d, "표시", "전방 GDP 대비(%)", 20, "전방(하류) 파급: GDP 대비 %", "%"), width="stretch")
    g2.plotly_chart(charts.bar_top(d, "표시", "후방 GDP 대비(%)", 20, "후방(상류) 파급: GDP 대비 %", "%"), width="stretch")
    table(ctab.sort_values("전방 GDP 대비(%)").reset_index(drop=True))
    ind = I.industry_table(t, focus, {"전방 부가가치 변화(백만 달러)": t.v * fwd, "후방 부가가치 변화(백만 달러)": t.v * bwd})
    res = Result(
        module=MODULE, analysis="공급망 차질",
        inputs={"자료": f"OECD ICIO {t.edition}년판 {t.year}년 표", "관심 국가": _clab(focus), "차질 목록": pd.DataFrame(ran)},
        summary=summ, tables={"국가별 결과": ctab, f"{COUNTRY_KO[focus]} 산업별": ind},
        notes=COMMON_NOTES + [
            "전방(Ghosh 공급모형): 차질 산업의 산출 감소가 그 제품을 중간재로 쓰는 국내외 하류 산업으로 배분계수만큼 전파된다고 가정. 대체·재고가 없을 때의 상한.",
            "후방(레온티에프): 차질 산업이 사들이던 국내외 중간재 수요가 줄어 상류 산업 생산이 감소.",
            "두 결과는 서로 다른 극단 가정이므로 더하지 않음.",
        ],
        sources=[SOURCE_TMPL.format(ed=t.edition)],
        charts=[ExcelChart("국가별 결과", "국가", ["전방 GDP 대비(%)", "후방 GDP 대비(%)"], "GDP 대비 (%)", "bar", 20)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 4. 가격 파급
# ---------------------------------------------------------------------------
PRICE_PRESETS = {
    "국제 에너지(원유·가스) 가격 30% 상승": [(I.WORLD, "B05_06", 30.0)],
    "중국산 공산품 가격 10% 상승": [("CHN", "제조업 전체", 10.0)],
    "국제 농산물 가격 20% 상승": [(I.WORLD, "A01_02", 20.0)],
    "세계 반도체·전자 가격 15% 상승": [(I.WORLD, "C26", 15.0)],
}
MANUF = "제조업 전체"


def price_tab(t: I.ICIO, focus: str):
    key = "icio_price"
    copts = _country_opts(t, world=True)
    if key not in st.session_state:
        st.session_state[key] = PRICE_PRESETS["국제 에너지(원유·가스) 가격 30% 상승"]
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    preset = c1.selectbox("예시 시나리오", ["(직접 입력)"] + list(PRICE_PRESETS), key=f"{key}_preset")
    if c2.button("예시 불러오기", key=f"{key}_load", disabled=preset == "(직접 입력)", width="stretch"):
        st.session_state[key] = PRICE_PRESETS[preset]
        st.session_state.pop(f"{key}_editor", None)
        st.rerun()
    st.caption("고른 국가·산업의 생산자가격을 외생으로 올리면, 그 제품을 직·간접으로 쓰는 다른 산업의 가격이 비용 전가로 오릅니다.")
    df_in = pd.DataFrame([{"국가": _clab(c), "산업": s if s == MANUF else _ilab(s), "가격 변화율(%)": v} for c, s, v in st.session_state[key]])
    ed = row_editor(f"{key}_editor", df_in, {
        "국가": {"options": [_clab(c) for c in copts], "w": 2.5},
        "산업": {"options": [MANUF] + [_ilab(s) for s in t.industries], "w": 4},
        "가격 변화율(%)": {"num": True, "fmt": "%.1f", "step": 5.0, "w": 1.6},
    })
    ran = run_gate(key, ed.to_dict("records"))
    if ran is None:
        return
    lab2c = {_clab(c): c for c in copts}
    lab2s = {_ilab(s): s for s in t.industries}
    fixed = {}
    for r in ran:
        inds = [s for s in t.industries if s[0] == "C"] if r["산업"] == MANUF else [lab2s[r["산업"]]]
        for c in t.expand(lab2c[r["국가"]]):
            for s in inds:
                fixed[t.idx(c, s)] = float(r["가격 변화율(%)"])
    fixed = {k: v for k, v in fixed.items() if t.x[k] > 0}
    if not fixed:
        st.info("가격 충격을 한 줄 이상 입력하세요.")
        return
    with st.spinner("가격 파급을 계산하는 중…"):
        ctab, cells, dp, summ = I.price_effect(t, fixed)
    fr = ctab[ctab["국가코드"] == focus].iloc[0]
    summ[f"{COUNTRY_KO[focus]} 소비자물가(%)"] = fr["소비자물가(%)"]
    summ[f"{COUNTRY_KO[focus]} 생산자가격(%)"] = fr["생산자가격(%)"]
    summary_cards(summ, per_row=4)
    d = ctab[ctab["국가코드"] != "ROW"].copy()
    d["표시"] = d["국가"]
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.bar_top(d, "표시", "소비자물가(%)", 25, "소비자물가 상승률 (가계소비 가중, %)", "%"), width="stretch")
    _focus_industry_chart(g2, t, focus, dp * 100, f"{COUNTRY_KO[focus]}: 산업별 생산자가격 (%)", "%")
    full = ctab.copy()
    table(full.sort_values("소비자물가(%)", ascending=False).reset_index(drop=True))
    ind = I.industry_table(t, focus, {"생산자가격 변화(%)": dp * 100})
    res = Result(
        module=MODULE, analysis="가격 파급",
        inputs={"자료": f"OECD ICIO {t.edition}년판 {t.year}년 표", "관심 국가": _clab(focus), "가격 충격": pd.DataFrame(ran)},
        summary=summ, tables={"국가별 결과": full, f"{COUNTRY_KO[focus]} 산업별": ind},
        notes=COMMON_NOTES[:1] + [
            "외생 가격 모형: 충격 국가·산업의 가격을 고정하고, 나머지 가격은 Δp_r' = Δp_f'·A_fr·(I - A_rr)^-1 로 결정(비용 완전 전가, 임금·이윤·환율 불변).",
            "소비자물가 = 각국 가계소비(원산지·산업별, 기초가격) 가중 평균. 유통마진·세금·에너지 직접소비 비중 차이로 실제 CPI 영향과 다를 수 있음.",
        ],
        sources=[SOURCE_TMPL.format(ed=t.edition)],
        charts=[ExcelChart("국가별 결과", "국가", ["소비자물가(%)"], "소비자물가 (%)", "bar", 25)])
    excel_button(res, f"{key}_xlsx")
    notes_box(res)


# ---------------------------------------------------------------------------
# 5. GVC·부가가치 무역
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="GVC 지표를 계산하는 중…")
def _gvc_all(ed: str, yr: str) -> pd.DataFrame:
    return I.gvc_indicators(_load(ed, yr))


@st.cache_data(show_spinner="GVC 지표 추이를 계산하는 중…(연도마다 1~2초)")
def _gvc_series(ed: str, years: tuple[str, ...], country: str) -> pd.DataFrame:
    rows = []
    for y in years:
        g = I.gvc_indicators(I.load(ed, y))
        r = g[g["국가코드"] == country].iloc[0].to_dict()
        r["연도"] = y
        rows.append(r)
    return pd.DataFrame(rows)


def gvc_tab(t: I.ICIO, focus: str, years: list[str]):
    g = _gvc_all(t.edition, t.year)
    fr = g[g["국가코드"] == focus].iloc[0]
    summary_cards({k: fr[k] for k in ["총수출(10억 달러)", "국내 부가가치(DVA) 비중(%)", "해외 부가가치(FVA) 비중(%)",
                                      "전방 참여(DVX) 비중(%)", "GVC 참여도(%)"]}, per_row=5, signed=False)
    src, ind, part = I.gvc_country(t, focus)
    g1, g2 = st.columns(2)
    s2 = src[src["국가코드"] != focus].head(15).copy()
    s2["표시"] = s2["국가"]
    g1.plotly_chart(charts.bar_top(s2, "표시", "비중(%)", 15, f"{COUNTRY_KO[focus]} 수출에 들어간 해외 부가가치의 원천 (수출 내 부가가치 대비 %)", "%"),
                    width="stretch")
    i2 = ind.copy()
    i2["표시"] = i2["산업"]
    i2 = i2[i2["총수출(백만 달러)"] > i2["총수출(백만 달러)"].sum() * 0.005]
    g2.plotly_chart(charts.bar_top(i2, "표시", "해외 부가가치 비중(%)", 15, "산업별 수출의 해외 부가가치 비중 (%, 수출 0.5% 이상 산업)", "%"),
                    width="stretch")
    st.markdown("##### 상대국별 총교역과 부가가치 교역")
    table(part.head(30))
    st.markdown("##### 국가 비교")
    cmp = st.multiselect("비교 국가", [_clab(c) for c in t.countries],
                         default=[_clab(c) for c in ["KOR", "CHN", "JPN", "DEU", "USA", "TWN", "VNM", "MEX"] if c in t.ci],
                         key="gvc_cmp")
    sel = g[g["국가코드"].isin([x.rsplit("(", 1)[1][:-1] for x in cmp])]
    if len(sel):
        st.plotly_chart(charts.bar_stacked(sel.assign(표시=sel["국가"]), "표시",
                                           ["해외 부가가치(FVA) 비중(%)", "전방 참여(DVX) 비중(%)"],
                                           "GVC 참여도 = 후방(FVA) + 전방(DVX), 총수출 대비 %", n=len(sel), sort_by="GVC 참여도(%)"),
                        width="stretch")
    table(g.sort_values("GVC 참여도(%)", ascending=False).reset_index(drop=True))
    ts = None
    if st.toggle(f"{COUNTRY_KO[focus]} 연도별 추이 계산 ({years[0]}–{years[-1]}년)", key="gvc_ts"):
        ts = _gvc_series(t.edition, tuple(years), focus)
        st.plotly_chart(charts.lines(ts, "연도", ["국내 부가가치(DVA) 비중(%)", "해외 부가가치(FVA) 비중(%)", "전방 참여(DVX) 비중(%)"],
                                     f"{COUNTRY_KO[focus]} GVC 지표 추이 (%)"), width="stretch")
    tables = {"국가별 GVC 지표": g, f"{COUNTRY_KO[focus]} 부가가치 원천": src, f"{COUNTRY_KO[focus]} 산업별": ind,
              f"{COUNTRY_KO[focus]} 상대국별": part}
    if ts is not None:
        tables[f"{COUNTRY_KO[focus]} 추이"] = ts
    res = Result(
        module=MODULE, analysis="GVC·부가가치무역",
        inputs={"자료": f"OECD ICIO {t.edition}년판 {t.year}년 표", "관심 국가": _clab(focus)},
        summary={k: fr[k] for k in ["총수출(10억 달러)", "국내 부가가치(DVA) 비중(%)", "해외 부가가치(FVA) 비중(%)",
                                    "전방 참여(DVX) 비중(%)", "GVC 참여도(%)", "부가가치 수출/총수출(%)"]},
        tables=tables,
        notes=COMMON_NOTES[:1] + [
            "수출의 부가가치 원천: VAX = diag(v)·L·E (E = 국가별 총수출). DVA = 자국 부가가치, FVA = 외국 부가가치(후방 참여).",
            "DVX(전방 참여) = 자국 부가가치 중 다른 나라 수출에 체화된 부분. GVC 참여도 = (FVA + DVX)/총수출 (Koopman-Wang-Wei 계열의 간이 지표, 일부 중복계산 포함).",
            "부가가치 수출 = 자국 부가가치 중 외국 최종수요에 흡수된 부분 (TiVA 의 FD_VA 개념). 상대국별 부가가치 교역수지는 최종수요 기준.",
            "생산물세(TLS)는 부가가치에 포함하지 않으므로 DVA+FVA 는 총수출보다 약간 작음.",
        ],
        sources=[SOURCE_TMPL.format(ed=t.edition)],
        charts=[ExcelChart("국가별 GVC 지표", "국가", ["해외 부가가치(FVA) 비중(%)", "전방 참여(DVX) 비중(%)"], "GVC 참여 (총수출 대비 %)", "bar", 30)])
    excel_button(res, "gvc_xlsx")
    notes_box(res)


def render():
    page_header("국제 산업연관표 분석",
                "OECD 국제산업연관표(ICIO)로 국가 간 수요·관세·공급망·가격 충격의 파급과 글로벌 가치사슬(GVC)을 계산합니다.")
    av = I.available()
    if not av:
        st.error("국제산업연관표 자료가 없습니다. build/build_icio.py 를 먼저 실행하세요.")
        return
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 2.5])
    ed = c1.selectbox("ICIO 판", list(av)[::-1], key="icio_ed")
    years = av[ed]
    yr = c2.selectbox("자료 연도", years[::-1], key="icio_year")
    t = _load(ed, yr)
    focus = c3.selectbox("관심 국가(결과 상세)", sorted([c for c in t.countries if c != "ROW"], key=lambda c: COUNTRY_KO.get(c, c)),
                         index=sorted([c for c in t.countries if c != "ROW"], key=lambda c: COUNTRY_KO.get(c, c)).index("KOR"),
                         format_func=_clab, key="icio_focus")
    c4.markdown(f"<div style='padding-top:1.9rem;color:#666'>{t.C - 1}개국+기타 지역 · {t.S}개 산업 · "
                f"세계 부가가치 {t.va.sum() / 1e6:,.1f}조 달러 · {t.year}년</div>", unsafe_allow_html=True)
    tabs = st.tabs(["국가 간 수요 파급", "관세 시나리오", "공급망 차질", "가격 파급", "GVC·부가가치무역"])
    with tabs[0]:
        demand_tab(t, focus)
    with tabs[1]:
        tariff_tab(t, focus)
    with tabs[2]:
        supply_tab(t, focus)
    with tabs[3]:
        price_tab(t, focus)
    with tabs[4]:
        gvc_tab(t, focus, years)
    footer()
