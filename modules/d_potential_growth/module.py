"""잠재성장률 추정 — 통계모형·준구조모형·생산함수모형으로 잠재 GDP·잠재성장률·GDP 갭을 추정해 비교."""
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
from engines.potential import forecast as F
from engines.potential import methods as M

META = {"title": "잠재성장률 추정", "group": "성장 분석", "order": 10, "icon": ":material/trending_up:",
        "status": "ready", "slug": "potential-growth"}

MODULE = "잠재성장률 추정"
SOURCES_KR = [
    "한국은행 ECOS: 실질 GDP(계절조정·연쇄가격, 200Y104), 자본서비스 물량지수(경제활동별 연쇄 통크비스트 전산업, 291Y522), "
    "실질 생산자본스톡(291Y504), 제조업 평균가동률(901Y025), 기준금리(722Y001), 소비자동향조사 기대인플레이션(511Y003)",
    "통계청 경제활동인구조사(ECOS 901Y027, 계절조정): 취업자·경제활동인구·15세 이상 인구·실업률",
    "통계청 소비자물가지수(ECOS 901Y009·901Y010): 총지수, 농산물 및 석유류 제외 지수(근원물가)",
    "평균 근로시간: OECD Average annual hours actually worked per worker(2008년~), Penn World Table avh(FRED AVHWPEKRA065NRUG, "
    "2007년 이전 증가율), 고용노동부 사업체노동력조사 근로시간(ECOS 901Y148·901Y052, 당해 연도 증가율)",
]
REFS = ("권지호·김도완·지정구·김건·노경서(2019), 「우리나라의 잠재성장률 추정」, 조사통계월보 2019년 8월호, 한국은행  \n"
        "이은경·천동민·김정욱·이동재(2024), 「우리 경제의 잠재성장률과 향후 전망」, BOK 이슈노트 제2024-33호, 한국은행  \n"
        "정원석·장준호·김철주(2021), 「코로나19를 감안한 우리경제의 잠재성장률 재추정」, BOK 이슈노트 제2021-22호  \n"
        "Laubach·Williams(2003, REStat), Fleischman·Roberts(2011, FEDS 2011-46), Chalaux·Guillemette(2019, OECD), "
        "Havik 외(2014, EU 집행위 생산함수 방법), CBO(2001), Laxton·Tetlow(1992), Blanchard·Quah(1989, AER), "
        "Hodrick·Prescott(1997), Christiano·Fitzgerald(2003), Hamilton(2018, REStat), Kamber·Morley·Wong(2018, REStat), "
        "Harvey(1985)·Clark(1987)")
COMMON_NOTES = [
    "GDP 갭 = (실제 GDP − 잠재 GDP) / 잠재 GDP (로그 차이 ×100, %). 플러스면 경기 과열, 마이너스면 경기 부진.",
    "잠재성장률은 추정한 잠재 GDP 수준의 증가율. 분기는 전년동기비, 연간은 4개 분기 합계의 전년 대비 증가율.",
    "모형 분류는 한국은행(권지호 외 2019)을 따름: 통계모형(GDP 시계열만 이용), 준구조모형(GDP 갭과 물가·고용·금리의 관계식 이용), "
    "생산함수모형(총요소생산성·노동·자본 투입으로 잠재 GDP 구성).",
    "모든 필터는 최신 시점에서 추정이 불안정함(끝점 문제). 자료가 추가·개정되면 최근 몇 년의 잠재성장률과 갭이 바뀔 수 있음.",
    "2020년 코로나19 충격은 별도 더미 없이 그대로 들어가 있음(한국은행 2024년 재추정도 더미를 없앰).",
]

HELP = {
    "linear": ("로그 GDP 에 시간 추세를 맞추고, 지정한 구조변화 시점에서 기울기를 바꿈.",
               "단순하고 해석이 쉬움.", "구조변화 시점을 사람이 정해야 하고, 구간 안에서는 잠재성장률이 일정하다고 봄."),
    "hp": ("실제 GDP 와의 차이와 추세의 굴곡을 함께 최소화하는 매끄러운 추세(λ=1600 이 분기 표준). 한국은행 4개 모형 중 하나.",
           "가장 널리 쓰여 비교 기준이 됨.", "끝점에서 최근 자료에 끌려감(이 도구는 AR 예측으로 연장해 완화). 경제 이론 정보는 없음."),
    "hp1": ("각 분기까지의 자료로만 HP 필터를 돌려 그 분기의 끝값을 잇는 실시간 추정.",
            "당시 정책 판단에 쓸 수 있었던 갭을 재현.", "양방향보다 갭이 늦게 반응하고 사후 수정 폭이 큼."),
    "cf": ("주파수 영역에서 경기순환 주기(2~32분기) 변동을 걸러내고 나머지를 잠재로 봄.",
           "경기순환 주기를 명시적으로 정의.", "주기 범위 선택에 따라 결과가 달라지고 끝점이 불안정."),
    "hamilton": ("8분기 뒤 GDP 를 현재·과거 GDP 로 예측해, 예측 가능한 부분을 추세, 예측오차를 갭으로 봄.",
                 "HP 필터의 가짜 순환 문제를 피함(Hamilton 2018).", "추세가 거칠어 잠재성장률이 크게 출렁임. 처음 11분기는 값이 없음."),
    "bn": ("성장률의 AR 모형으로 장기 예측 수준(추세)을 계산하는 BN 분해. 신호대잡음비를 제약해 갭 크기를 안정화.",
           "실시간 추정이 사후 수정에 강하다고 알려짐(Kamber·Morley·Wong 2018).", "δ·AR 차수 선택에 민감."),
    "uc": ("GDP 를 '기울기가 천천히 변하는 추세'와 'AR(2) 순환'의 합으로 보고 칼만 필터로 추정.",
           "모형의 확률 구조가 명시적이고 모수를 자료로 추정.", "추세와 순환의 식별이 약하면 결과가 HP 와 비슷하거나 불안정."),
    "lw": ("Laubach·Williams(2003) 모형에 오쿤의 관계식을 더한 한국은행 준구조모형. IS곡선(실질금리 갭→GDP 갭)·필립스곡선(GDP 갭→근원물가)·"
           "오쿤(GDP 갭→실업률)을 상태공간모형으로 묶어 잠재 GDP·잠재성장률·중립금리·자연실업률을 동시 추정.",
           "물가·고용·금리의 이론적 관계를 갭 추정에 반영하고 중립금리도 얻음.",
           "투입요소별 분해 불가. 최우추정이 불안정해 몇몇 충격 크기를 고정해야 함(쏠림 문제)."),
    "mv": ("HP 필터에 필립스곡선(근원물가)과 오쿤의 법칙(실업률)을 더해 잠재 GDP 와 자연실업률을 함께 추정(Laxton·Tetlow 1992).",
           "물가·고용 정보를 쓰면서도 계산이 단순하고 안정적.", "관계식 계수가 약하거나 불안정하면 HP 와 비슷해짐."),
    "svar": ("GDP 성장률·실업률 VAR 에서 '수요 충격은 GDP 에 장기 영향이 없다'는 제약으로 공급·수요 충격을 식별.",
             "경제적 식별 조건에 기반.", "갭이 수요 충격만으로 정의돼 다른 방법과 시점이 엇갈릴 수 있고, 오쿤 관계가 약하면 불안정."),
    "pf": ("잠재 GDP = TFP 추세 × 자본서비스 × 잠재노동. 잠재노동 = 15세이상인구 × 참가율 추세 × (1−자연실업률) × 근로시간 추세"
           "(또는 고용률 추세). 한국은행 생산함수모형.",
           "잠재성장률을 TFP·자본·노동(인구·참가율·실업률·근로시간) 기여도로 분해. 중앙은행·국제기구 표준.",
           "분배율·자연실업률·근로시간 자료 가정에 좌우. TFP 추세에 다시 필터를 써서 끝점 문제가 남음."),
    "pfmv": ("생산함수에 필립스곡선과 오쿤의 관계식(참가율·고용률·근로시간 갭)을 결합해 투입요소별 추세와 GDP 갭을 상태공간모형으로 "
             "동시 추정. 한국은행 다변량필터링모형(미 연준 FRB/US 공급부문 수정형).",
             "생산함수의 요인 분해와 물가·고용 정보를 함께 씀.", "모형이 복잡하고 추정할 모수가 많아 불확실성이 큼."),
}


# ---------------------------------------------------------------------------
# 계산 (캐시)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="잠재 GDP 를 추정하고 있습니다(상태공간모형 최우추정 포함, 십여 초)…", max_entries=16)
def _run(df: pd.DataFrame, cfg_json: str):
    return A.run(df, json.loads(cfg_json))


def _step_q(n: int) -> int:
    return 4 if n <= 52 else 8 if n <= 104 else 12


def _step_a(n: int) -> int:
    return 1 if n <= 16 else 2


def _gname(k: str) -> str:
    return f"[{M.GROUP[k]}] {M.METHODS[k]}" if k in M.GROUP else M.AVG_KEYS.get(k, k)


BOK4 = ("hp", "lw", "pfmv", "pf")


def _trim(df, ests):
    """중기 가정 꼬리를 화면에서 뺌(추정에는 썼음)."""
    d = F.shown(df)
    if len(d) == len(df):
        return df, ests
    end = d.index[-1]
    out = {}
    for k, e in ests.items():
        e2 = M.Estimate(e.key, e.name, e.trend.loc[:end], e.gap.loc[:end], dict(e.params), list(e.notes),
                        {n: (t.loc[:end] if isinstance(t.index, pd.PeriodIndex) else t) for n, t in e.extra.items()})
        out[k] = e2
    return d, out


def _sq(df) -> str | None:
    """분기 차트 음영 시작 라벨(첫 전망 분기)."""
    p = F.first_forecast(df)
    return D.label(p) if p is not None else None


def _sa(df) -> str | None:
    """연간 차트 음영 시작 라벨(전망 분기가 들어 있는 첫 해)."""
    p = F.first_forecast(df)
    return str(p.year) if p is not None else None


def _fc_notes() -> list[str]:
    info = st.session_state.get("pg_fc_info")
    if not info:
        return ["끝점 보정(전망 연장)을 쓰지 않음: 마지막 실적 분기까지의 자료로만 추정."]
    fc = info["fc"]
    how = ("발표된 분기 경로의 전년동기비를 그대로 맞추고, 분기 경로가 없는 반기는 반기 성장률(전년동기비)이 전망치가 되도록 두 분기에 "
           "같은 전기비 성장을 줌" if info["mode"] == "yoy" else
           "마지막 실적 분기 수준에 분기 경로의 전기비를 이어 붙이고, 그 뒤 반기는 반기 성장률 전망에 맞춤")
    return [f"끝점 보정: {fc['title']} 전망으로 자료를 {fc['horizon']}까지 연장해 추정(한국은행 잠재성장률 추정 방식). "
            f"실질 GDP 는 {how}.",
            "생산함수·준구조모형 입력: 취업자 = 전년동기 + 취업자수 증감 전망, 실업률 = 반기 전망(계절조정), 경제활동인구 = 취업자/(1−실업률), "
            "근원물가·소비자물가 = 분기 경로·반기 상승률 전망(한국은행 근원물가는 식료품·에너지 제외 기준이나 상승률을 농산물·석유류 제외 지수에 적용), "
            "15세 이상 인구·평균 근로시간·자본서비스·자본스톡 = 최근 3년 평균 증가율로 추세 연장, 가동률 = 최근 4분기 평균, "
            "기준금리·기대인플레이션 = 마지막 값 유지.",
            f"전망 시계 뒤로는 3년간 실질 GDP 를 연 {info.get('tail', 2.0)}%(중기 성장률 가정)로, 인구·근로시간·자본은 추세로, 고용률·실업률은 "
            "유지, 물가는 2%로 이어 붙여 필터의 끝점을 처리하고 화면에는 전망 시계까지만 보임(AR 예측 연장은 2008년 이후 평균 "
            "성장률로 되돌아가 끝점 추세를 끌어올리므로 쓰지 않음).",
            "전망 구간(차트 음영, 표의 '전망')의 잠재성장률·GDP 갭은 전망을 전제로 한 값. 연간 표의 '실제 성장률'도 전망 포함 연도는 전망치."]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def _data_source():
    st.session_state["pg_fc_info"] = None
    src =st.radio("자료", ["한국 (ECOS)", "직접 올리기 (다른 나라·다른 자료)"], horizontal=True, key="pg_src")
    if src.startswith("한국"):
        df, meta = D.load_korea()
        st.caption(f"실질 GDP {meta['last_gdp']}까지 · 자본서비스·자본스톡 {meta.get('ksvc_last_year', meta['capital_last_year'])}년까지 공표분 · "
                   f"근로시간 {meta.get('hours_current_year', '')} · 자료 갱신 {meta['built']}")
        fc = F.load_bok()
        if fc is None:
            return df, "KR", SOURCES_KR
        c1, c2, c3 = st.columns([5, 4, 3])
        on = c1.toggle(f"끝점 보정: 한국은행 전망으로 {D.label(pd.Period(fc['horizon'], freq='Q'))}까지 연장", value=True,
                       key="pg_fc_on",
                       help=f"{fc['title']}의 GDP 성장률(분기 경로·반기·연간)로 실질 GDP 를 연장하고, 생산함수·준구조모형 입력 변수도 "
                            "전망(취업자·실업률·물가) 또는 추세 연장으로 같은 시계에 맞춘 뒤 추정합니다(한국은행 잠재성장률 추정 방식). "
                            "필터가 최근 자료에 끌려가는 끝점 문제를 줄입니다.")
        mode = c2.radio("GDP 연장 방식", ["yoy", "qoq"], horizontal=True, key="pg_fc_mode", disabled=not on,
                        format_func=lambda x: "전년동기비 전망 그대로" if x == "yoy" else "전기비 경로를 실적에 이어 붙임",
                        help="전년동기비: 발표된 분기·반기·연간 성장률을 그대로 재현. 전기비: 마지막 실적 수준에서 전망 경로의 전기비 "
                             "성장을 이어 붙여, 전망 이후 실적이 예상과 달라진 만큼의 수준 차이가 남습니다.")
        tail = c3.number_input("전망 이후 중기 성장률 가정(%)", 0.0, 5.0, 2.0, step=0.1, key="pg_fc_tail", disabled=not on,
                               help="전망 시계 뒤 3년을 이 성장률로 이어 붙여 필터의 끝점을 처리합니다(화면에는 전망 시계까지만 표시). "
                                    "AR 예측 연장은 2008년 이후 평균 성장률(약 2.7%)로 되돌아가 끝점 잠재성장률을 끌어올리므로 대신 씀. "
                                    "기본 2.0%는 한국은행 이은경 외(2024)의 2024~26년 잠재성장률 추정치이자 이 도구의 실적 기준 2025년 추정치 수준.")
        if not on:
            return df, "KR", SOURCES_KR
        dfx, how = F.extend(df, fc, mode)
        st.session_state["pg_fc_info"] = {"fc": fc, "how": how, "cmp": F.compare(dfx, fc), "mode": mode, "tail": tail}
        dfx = F.add_tail(dfx, tail)
        return dfx, f"KR_fc_{fc['published']}_{mode}_{tail}", SOURCES_KR + [
            f"{fc['title']} — {fc['pages']} ({fc['url']})"]
    up = st.file_uploader("분기 자료 CSV·엑셀", type=["csv", "xlsx"], key="pg_upload",
                          help="첫 열은 분기(2024Q1 형식), 다음 열 이름: gdp(필수, 계절조정 실질), urate, core, emp, lf, pop15, "
                               "hours, ksvc, capital, capu, rate, pie")
    st.download_button("양식 내려받기 (CSV)", data=D.template_csv(), file_name="potential_template.csv",
                       mime="text/csv", key="pg_tmpl", on_click="ignore")
    if up is None:
        st.info("자료를 올리면 추정이 시작됩니다. gdp 열만 있어도 통계모형 7가지를 쓸 수 있습니다. "
                "실업률(urate)·근원물가(core)를 더하면 다변량 HP 필터·구조 VAR, 정책금리(rate)까지 더하면 LW형 준구조모형, "
                "고용(emp·lf·pop15)과 자본(ksvc 자본서비스 또는 capital 자본스톡)을 더하면 생산함수모형을 쓸 수 있습니다. "
                "근로시간(hours)·가동률(capu)·기대인플레이션(pie)은 있으면 씁니다.")
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
    usable = [k for k in M.METHODS if k not in av or av[k]]
    quarters = [str(p) for p in F.shown(df)["gdp"].dropna().index]
    has_tail = F.FC_COL in df.columns and (df[F.FC_COL] == F.TAIL).any()
    preset = st.radio("**추정 방법**", ["bok4", "all"], horizontal=True, key="pg_preset",
                      format_func=lambda x: "한국은행 4개 모형 (HP·준구조·생산함수 다변량 필터·생산함수)" if x == "bok4"
                      else "전체 12가지 방법",
                      help="한국은행은 HP 필터·준구조모형(LW형)·다변량필터링모형·생산함수모형 4개의 평균을 발표합니다(권지호 외 2019, "
                           "이은경 외 2024). 선형 추세·CF·Hamilton·구조 VAR 등은 끝점에서 출렁임이 커 기본 평균에서 뺐습니다.")
    gcols = st.columns(3)
    chosen = []
    for col, (g, gname) in zip(gcols, M.GROUPS.items()):
        opts = [k for k in M.METHODS if M.GROUP_OF[k] == g]
        ok = [k for k in opts if k in usable]
        dflt = [k for k in ok if k in BOK4] if preset == "bok4" else ok
        with col:
            chosen += st.multiselect(gname, ok, default=dflt, format_func=lambda k: M.METHODS[k], key=f"pg_methods_{g}_{preset}",
                                     placeholder="자료가 없어 쓸 수 없음" if not ok else "방법 선택")
            miss = [k for k in opts if k not in usable]
            if miss:
                st.caption("자료 부족: " + ", ".join(f"{M.METHODS[k]}({D.NEEDS[k]})" for k in miss))
    cc = st.columns([1, 1, 2])
    start_def = "1991Q1" if "1991Q1" in quarters else quarters[0]
    start = cc[0].selectbox("통계모형 추정 시작", quarters[: max(1, len(quarters) - 40)],
                            index=quarters.index(start_def) if start_def in quarters[: len(quarters) - 40] else 0,
                            key="pg_start", help="통계모형의 추정 표본 시작. 준구조·생산함수모형은 고용·물가·금리 자료가 있는 2000년 이후(한국)를 씀.")
    disp_def = "2005Q1" if "2005Q1" in quarters else quarters[min(len(quarters) - 1, 20)]
    show = cc[1].selectbox("표시 시작 분기", quarters[:-8], index=quarters.index(disp_def) if disp_def in quarters[:-8] else 0,
                           key="pg_show")
    cfg = dict(A.DEFAULT_CFG)
    cfg["methods"], cfg["start"] = chosen, start
    cfg["avg_mode"] = cc[2].radio("방법 평균", ["group", "all"], horizontal=True, key=f"pg_avgmode_{preset}",
                                  index=1 if preset == "bok4" else 0,
                                  format_func=lambda x: "모형군 평균의 평균" if x == "group" else "모든 방법 단순 평균",
                                  help="모형군 평균의 평균: 통계·준구조·생산함수 모형군에 같은 비중(방법 수가 많은 통계모형에 쏠리지 않음). "
                                       "한국은행 4개 모형은 단순 평균이 기본(한국은행 발표 방식).")
    with st.expander("방법별 세부 설정"):
        t1, t2, t3 = st.tabs(["통계모형", "준구조모형", "생산함수모형"])
        with t1:
            a, b, c = st.columns(3)
            with a:
                cfg["lam"] = st.number_input("HP λ (분기, 생산함수 추세에도 적용)", 100.0, 100000.0, 1600.0, step=100.0, key="pg_lam")
                cfg["extend"] = st.slider("끝점 연장(분기, AR 예측)", 0, 20, 12, key="pg_ext", disabled=has_tail,
                                          help="HP·생산함수 추세에 적용. 0이면 연장하지 않음. 끝점 보정(전망 연장)을 쓰면 "
                                               "전망 이후는 중기 성장률 가정으로 잇기 때문에 쓰지 않음.")
                if has_tail:
                    cfg["extend"] = 0
                cand = [q for q in quarters if q.endswith("Q1") or q.endswith("Q4")]
                cfg["breaks"] = st.multiselect("선형 추세 구조변화 시점", cand,
                                               default=[b_ for b_ in ["1998Q1", "2008Q4"] if b_ in cand], key="pg_breaks")
            with b:
                cfg["cf_high"] = st.slider("CF: 잠재로 볼 최소 주기(분기)", 24, 48, 32, step=4, key="pg_cf")
                cfg["ham_h"] = st.slider("Hamilton: 예측 시차 h", 4, 12, 8, key="pg_h")
                cfg["ham_p"] = st.slider("Hamilton: 설명 시차 p", 2, 8, 4, key="pg_p")
            with c:
                cfg["bn_delta"] = st.slider("BN: 신호대잡음비 δ", 0.05, 1.0, 0.2, step=0.05, key="pg_bnd",
                                            help="작을수록 추세가 매끄럽고 갭이 큼.")
                cfg["bn_p"] = st.slider("BN: AR 차수", 2, 12, 8, key="pg_bnp")
                cfg["bn_window"] = st.slider("BN: 성장률 평균 이동 기간(분기)", 20, 80, 40, step=4, key="pg_bnw")
        with t2:
            a, b, c = st.columns(3)
            with a:
                st.markdown("**LW형 준구조모형**")
                cfg["lw_lam_g"] = st.select_slider("λ_g (잠재성장률 변동 크기)", [0.02, 0.04, 0.06, 0.08, 0.1, 0.15], 0.06,
                                                   key="pg_lwg")
                cfg["lw_lam_z"] = st.select_slider("λ_z (중립금리 기타 요인 변동)", [0.0, 0.01, 0.03, 0.05, 0.1], 0.03,
                                                   key="pg_lwz")
                cfg["lw_sy_max"] = st.select_slider("잠재 GDP 수준 충격 표준편차 상한(%)", [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 1.0],
                                                    0.25, key="pg_lwsy",
                                                    help="높일수록 GDP 변동이 잠재 GDP 로 흡수돼 잠재성장률이 출렁이고 갭이 작아짐.")
            with b:
                st.markdown("**다변량 HP 필터**")
                cfg["mv_weight"] = st.select_slider("물가·고용 정보 가중 배수", [0.25, 0.5, 1.0, 2.0, 4.0], 1.0, key="pg_mvw")
                cfg["mv_lam_u"] = st.number_input("자연실업률 λ", 400.0, 100000.0, 6400.0, step=400.0, key="pg_mvl")
            with c:
                st.markdown("**구조 VAR**")
                cfg["svar_p"] = st.slider("VAR 시차", 1, 8, 4, key="pg_svp")
        with t3:
            a, b, c = st.columns(3)
            with a:
                cfg["pf_labshare"] = st.slider("노동소득분배율", 0.50, 0.75, 0.61, step=0.01, key="pg_ls",
                                               help="한국은행(2019) 0.61. 자본소득분배율 = 1 − 노동소득분배율.")
                cfg["pf_capital"] = st.radio("자본투입", ["ksvc", "stock"], horizontal=True, key="pg_cap",
                                             format_func=lambda x: "자본서비스" if x == "ksvc" else "생산자본스톡")
            with b:
                cfg["pf_labour"] = st.radio("잠재노동 정의", ["lfpr", "emprate"], key="pg_lab",
                                            format_func=lambda x: "인구×참가율×(1−자연실업률)×근로시간 (2019)" if x == "lfpr"
                                            else "인구×고용률×근로시간 (2024)")
                nopts = [k for k in ("mv", "lw", "pfmv") if av.get(k)] + ["hp"]
                cfg["pf_nairu"] = st.selectbox("자연실업률", nopts, key="pg_nairu", disabled=cfg["pf_labour"] != "lfpr",
                                               format_func=lambda x: {"mv": "다변량 HP 필터", "lw": "LW형 준구조모형",
                                                                      "pfmv": "생산함수 다변량 필터", "hp": "실업률 HP 추세"}[x])
            with c:
                cut_opts = ["none"] + [q for q in quarters if q >= "2015Q1"]
                cfg["tfp_cut"] = st.selectbox(
                    "TFP 추세 고정 시점", cut_opts, index=cut_opts.index("2025Q4") if "2025Q4" in cut_opts else 0,
                    key="pg_tfpcut", format_func=lambda x: "고정 안 함(전 구간 HP)" if x == "none" else f"{x} (이후 성장은 순환으로)",
                    help="이 분기까지의 자료로만 TFP 추세를 추정하고 그 뒤는 최근 3년 추세 기울기로 연장합니다(두 생산함수모형에 적용). "
                         "2026년 이후 반도체 호황 성장을 잠재 성장이 아니라 수요 압력(플러스 GDP 갭)으로 보는 판단이며, "
                         "한국은행 2026년 8월 경제전망의 해석(수요측 압력 확대)과 맞춤. 노동·자본 투입은 그대로 전망 자료를 씀.")
                if cfg["tfp_cut"] == "none":
                    cfg["tfp_cut"] = None
                cfg["pf_cu"] = st.toggle("TFP 추세에서 가동률 효과 제거", value=True, key="pg_cu",
                                         help="TFP 순환 중 제조업 가동률로 설명되는 부분을 걷어낸 뒤 추세를 구함(한국은행 2019).")
                cfg["pfmv_rho"] = st.select_slider("다변량 필터: 갭 지속성 상한(AR 계수 합)", [0.8, 0.85, 0.9, 0.95], 0.9,
                                                   key="pg_pfrho",
                                                   help="높이면 갭이 오래 지속될 수 있어 추세 변화의 일부가 갭으로 잡힐 수 있음.")
    return cfg, show


def _params_table(ests: dict[str, M.Estimate]) -> pd.DataFrame:
    rows = []
    for e in ests.values():
        for k, v in e.params.items():
            rows.append({"방법": e.name, "항목": k, "값": float(v) if isinstance(v, (int, float, np.floating)) else str(v)})
    return pd.DataFrame(rows, columns=["방법", "항목", "값"])


def _inputs(cfg: dict, show: str) -> dict:
    nsrc = {"mv": "다변량 HP 필터", "lw": "LW형 준구조모형", "pfmv": "생산함수 다변량 필터", "hp": "실업률 HP 추세"}
    out = {}
    for g, gname in M.GROUPS.items():
        out[f"추정 방법: {gname}"] = ", ".join(M.METHODS[k] for k in cfg["methods"] if M.GROUP_OF[k] == g) or "없음"
    out.update({
        "방법 평균": "모형군 평균의 평균" if cfg["avg_mode"] == "group" else "모든 방법 단순 평균",
        "통계모형 추정 시작": cfg["start"], "표시 시작": show,
        "HP λ": cfg["lam"], "끝점 연장(분기)": cfg["extend"], "선형 추세 구조변화": ", ".join(cfg["breaks"]) or "없음",
        "CF 최소 주기": cfg["cf_high"], "Hamilton h·p": f"{cfg['ham_h']}·{cfg['ham_p']}",
        "BN δ·차수·이동기간": f"{cfg['bn_delta']}·{cfg['bn_p']}·{cfg['bn_window']}",
        "LW λ_g·λ_z·수준충격 상한": f"{cfg['lw_lam_g']}·{cfg['lw_lam_z']}·{cfg['lw_sy_max']}",
        "다변량 HP 가중 배수·자연실업률 λ": f"{cfg['mv_weight']}·{cfg['mv_lam_u']:g}",
        "구조 VAR 시차": cfg["svar_p"],
        "생산함수 노동소득분배율": cfg["pf_labshare"],
        "생산함수 자본투입": "자본서비스" if cfg["pf_capital"] == "ksvc" else "생산자본스톡",
        "생산함수 잠재노동": "인구×참가율×(1−자연실업률)×근로시간" if cfg["pf_labour"] == "lfpr" else "인구×고용률×근로시간",
        "생산함수 자연실업률": nsrc[cfg["pf_nairu"]] if cfg["pf_labour"] == "lfpr" else "–",
        "TFP 가동률 조정": "예" if cfg["pf_cu"] else "아니오",
        "TFP 추세 고정 시점": cfg.get("tfp_cut") or "고정 안 함",
        "끝점 보정": (lambda i: f"{i['fc']['title']}, {'전년동기비' if i['mode'] == 'yoy' else '전기비'} 방식, 이후 중기 성장률 {i['tail']}%"
                  if i else "쓰지 않음")(st.session_state.get("pg_fc_info")),
        "생산함수 다변량 필터 갭 지속성 상한": cfg["pfmv_rho"],
    })
    return out


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
def _cards(df, ests):
    L = A.latest(df, ests)
    lab = D.label(L["last"])
    gaps = {k: v for k, v in L["gaps"].items() if not k.startswith("avg") and np.isfinite(v)}
    pots = {k: v for k, v in L["pots"].items() if not k.startswith("avg") and np.isfinite(v)}
    ga, _ = A.annual_tables(df, ests, str(df.index[0]))
    ga = ga.set_index("연도")
    act = ga["실제 성장률"].notna() & (ga["구분"] == "실적")
    full = ga.index[act][-1] if act.any() else None
    cols = st.columns(5)
    cols[0].metric(f"실제 성장률 ({lab}, 전년동기비)", f"{L['actual_yoy']:.2f}%")
    if "avg" in ests:
        cols[1].metric(f"잠재성장률 ({lab}, 방법 평균)", f"{L['pots']['avg']:.2f}%",
                       help="범위: " + (f"{min(pots.values()):.2f} ~ {max(pots.values()):.2f}%" if pots else "–"))
        cols[2].metric(f"GDP 갭 ({lab}, 방법 평균)", f"{L['gaps']['avg']:+.2f}%")
        cols[3].metric("GDP 갭 범위 (방법 간)", f"{min(gaps.values()):+.2f} ~ {max(gaps.values()):+.2f}%" if gaps else "–")
        if full is not None:
            base = [ests[k].name for k in ests if not k.startswith("avg")]
            row = ga.loc[full, [c for c in base if c in ga.columns]].dropna()
            cols[4].metric(f"{full}년 잠재성장률 (방법 평균)", f"{ga.loc[full, '방법 평균']:.2f}%",
                           help=f"범위 {row.min():.2f} ~ {row.max():.2f}% · 실제 {ga.loc[full, '실제 성장률']:.2f}%")
        gk = [k for k in ("avg_stat", "avg_semi", "avg_pf") if k in ests]
        if gk and full is not None:
            gc = st.columns(len(gk))
            for col, k in zip(gc, gk):
                nm = ests[k].name
                col.metric(f"{full}년 잠재성장률 · {nm}", f"{ga.loc[full, nm]:.2f}%",
                           help=f"{lab} GDP 갭 {L['gaps'][k]:+.2f}%")
        fys = [y for y in ga.index if ga.loc[y, "구분"] != "실적" and pd.notna(ga.loc[y, "방법 평균"])]
        if fys:
            _, gpa = A.annual_tables(df, ests, str(df.index[0]))
            gpa = gpa.set_index("연도")
            fc = (st.session_state.get("pg_fc_info") or {}).get("fc", {})
            bok = fc.get("gdp", {}).get("a_yoy", {})
            fcols = st.columns(2 * len(fys))
            for i, y in enumerate(fys):
                row = ga.loc[y, [ests[k].name for k in ests if not k.startswith("avg") and ests[k].name in ga.columns]].dropna()
                grp = " · ".join(f"{ests[k].name.replace(' 평균', '')} {ga.loc[y, ests[k].name]:.2f}"
                                 for k in ("avg_stat", "avg_semi", "avg_pf") if k in ests)
                fcols[2 * i].metric(f"{y}년 잠재성장률 (전망 반영)", f"{ga.loc[y, '방법 평균']:.2f}%",
                                    help=f"방법 간 {row.min():.2f} ~ {row.max():.2f}% · {grp}")
                fcols[2 * i + 1].metric(f"{y}년 GDP 갭 (전망 반영)", f"{gpa.loc[y, '방법 평균']:+.2f}%",
                                        help=f"성장률 {ga.loc[y, '실제 성장률']:.2f}%"
                                             + (f" (한국은행 전망 {bok[str(y)]:.1f}%)" if str(y) in bok else ""))
    else:
        e = next(iter(ests.values()))
        cols[1].metric(f"잠재성장률 ({lab})", f"{L['pots'][e.key]:.2f}%")
        cols[2].metric(f"GDP 갭 ({lab})", f"{L['gaps'][e.key]:+.2f}%")


def compare_tab(df, ests, cfg, show, sources):
    base = [k for k in ests if not k.startswith("avg")]
    gavg = [k for k in ("avg_stat", "avg_semi", "avg_pf") if k in ests]
    has_avg = "avg" in ests
    c1, c2 = st.columns([2, 3])
    basis = c1.radio("성장률 기준", ["분기 (전년동기비)", "연간"], horizontal=True, key="pg_cmp_basis")
    views = (["모형군 평균"] if gavg else []) + ["모든 방법"] + [M.GROUPS[g] for g in M.GROUPS
                                                          if any(M.GROUP_OF[k] == g for k in base)]
    view = c2.radio("표시", views, horizontal=True, key="pg_cmp_view",
                    help="모형군 평균: 통계·준구조·생산함수모형 평균과 전체 평균. 모형군을 고르면 그 안의 개별 방법.")
    avg = "방법 평균" if has_avg else None
    if view == "모형군 평균":
        lines = [ests[k].name for k in gavg] + ([avg] if avg else [])
        members = base
    elif view == "모든 방법":
        lines = [ests[k].name for k in base] + ([avg] if avg else [])
        members = base
    else:
        g = next(k for k, v in M.GROUPS.items() if v == view)
        members = [k for k in base if M.GROUP_OF[k] == g]
        lines = [ests[k].name for k in members] + ([ests[f"avg_{g}"].name] if f"avg_{g}" in ests else [])
    names = [ests[k].name for k in members]
    emph = avg if view in ("모형군 평균", "모든 방법") else (lines[-1] if len(members) > 1 else None)
    g, gap, lev = A.quarterly_tables(df, ests, show)
    ga, gpa = A.annual_tables(df, ests, show)
    band = ("최소", "최대") if len(names) >= 2 else None
    if basis.startswith("분기"):
        step, x, sh = _step_q(len(g)), "라벨", _sq(df)
        gtab, ptab = g.copy(), gap.copy()
        if band:
            gtab["최소"], gtab["최대"] = g[names].min(axis=1), g[names].max(axis=1)
            ptab["최소"], ptab["최대"] = gap[names].min(axis=1), gap[names].max(axis=1)
    else:
        step, x, sh = _step_a(len(ga)), "연도", _sa(df)
        gtab = ga.assign(연도=ga["연도"].astype(str))
        ptab = gpa.assign(연도=gpa["연도"].astype(str))
        if band:
            gtab["최소"], gtab["최대"] = ga[names].min(axis=1), ga[names].max(axis=1)
            ptab["최소"], ptab["최대"] = gpa[names].min(axis=1), gpa[names].max(axis=1)
    rng = "음영: 표시 방법 간 최소~최대" if band else ""
    st.plotly_chart(charts.ts_lines(gtab, x, ["실제 성장률"] + lines, "실제 성장률과 잠재성장률", "%", 440, step,
                                    emphasize=emph, dashed=("실제 성장률",), band=band, shade=sh), width="stretch")
    st.plotly_chart(charts.ts_lines(ptab, x, lines, "GDP 갭 (실제 GDP 가 잠재 GDP 를 웃도는 정도)", "%", 400, step,
                                    emphasize=emph, band=band, shade=sh), width="stretch")
    if rng:
        st.caption(rng)
    order = ["연도", "구분", "실제 성장률"] + [ests[k].name for k in base] + [ests[k].name for k in gavg] + ([avg] if avg else [])
    ga, gpa = ga[[c for c in order if c in ga.columns]], gpa[[c for c in order if c in gpa.columns and c != "실제 성장률"]]
    st.markdown("**연간 잠재성장률 (%)**")
    table(ga.assign(연도=ga["연도"].astype(str)))
    st.markdown("**연간 GDP 갭 (연평균, %)**")
    table(gpa.assign(연도=gpa["연도"].astype(str)))
    per = _period_table(df, ests)
    if per is not None:
        per = per[[c for c in ["기간"] + order[2:] if c in per.columns]]
        st.markdown("**기간별 연평균 잠재성장률 (%)** — 한국은행 발표 방식(5년 평균)과 비교용"
                    + (" · e) 전망 포함 연도가 들어간 기간" if _sa(df) else ""))
        table(per)
    qskip = _step_q(len(g))
    ex_lines = [ests[k].name for k in gavg] + ([avg] if avg else []) if gavg else names[:6]
    tables = {"연간 잠재성장률": ga, "연간 GDP갭": gpa}
    if per is not None:
        tables["기간별 잠재성장률"] = per
    tables.update({"분기 잠재성장률(전년동기비)": g, "분기 GDP갭": gap, "분기 잠재GDP(십억원)": lev,
                   "방법별 모수": _params_table(ests)})
    res = Result(
        module=MODULE, analysis="방법 비교", inputs=_inputs(cfg, show),
        summary=_summary(df, ests),
        tables=tables,
        notes=COMMON_NOTES + _fc_notes() + [f"[{e.name}] " + " ".join(e.notes) for e in ests.values()],
        sources=sources,
        charts=[ExcelChart("분기 GDP갭", "라벨", ex_lines, "GDP 갭 (%)", kind="line", top=None, label_skip=qskip),
                ExcelChart("분기 잠재성장률(전년동기비)", "라벨", ["실제 성장률"] + ex_lines, "실제·잠재성장률 (전년동기비, %)",
                           kind="line", top=None, label_skip=qskip),
                ExcelChart("연간 잠재성장률", "연도", ["실제 성장률"] + (ex_lines if gavg else ([avg] if avg else names[:4])),
                           "연간 실제·잠재성장률 (%)", kind="col", top=None)])
    excel_button(res, "pg_x_cmp")
    notes_box(res)


def _bins(years) -> list[tuple[int, int]]:
    """한국은행 발표처럼 2001~05, 2006~10 … 5년 구간 (2001 기준)."""
    ys = sorted(int(y) for y in years)
    if not ys:
        return []
    a = ys[0] - ((ys[0] - 2001) % 5)
    out = []
    while a <= ys[-1]:
        sub = [y for y in ys if a <= y <= a + 4]
        if sub:
            out.append((sub[0], sub[-1]))
        a += 5
    return out


def _plabel(s: int, e: int) -> str:
    return f"{s}~{str(e)[2:]}" if e > s else str(s)


def _period_table(df, ests) -> pd.DataFrame | None:
    ga, _ = A.annual_tables(df, ests, "2001Q1")
    t = ga.set_index("연도")
    t = t[t["실제 성장률"].notna()]
    if len(t) < 5:
        return None
    fy = A.fc_years(df)
    rows = []
    for s_, e_ in _bins(t.index):
        r = t.loc[s_:e_].drop(columns=["구분"]).mean()
        r.name = _plabel(s_, e_) + ("e)" if any(y in fy for y in range(s_, e_ + 1)) else "")
        rows.append(r)
    out = pd.DataFrame(rows).round(2)
    out.index.name = "기간"
    return out.reset_index()


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
    k = st.selectbox("방법", keys, format_func=_gname, key="pg_det_m",
                     index=keys.index("pf") if "pf" in keys else 0)
    e = ests[k]
    if k in HELP:
        st.caption(f"{M.GROUP[k]} · {HELP[k][0]}")
    q, a = A.method_table(df, e, show)
    step = _step_q(len(q))
    L = A.latest(df, {k: e})
    lab = D.label(L["last"])
    cols = st.columns(4)
    cols[0].metric(f"잠재성장률 ({lab})", f"{L['pots'][k]:.2f}%")
    cols[1].metric(f"GDP 갭 ({lab})", f"{L['gaps'][k]:+.2f}%")
    an = a[a["구분"] == "실적"].dropna(subset=["실제 성장률(%)"])
    if len(an):
        last = an.iloc[-1]
        cols[2].metric(f"{int(last['연도'])}년 잠재성장률", f"{last['잠재성장률(%)']:.2f}%")
        cols[3].metric(f"{int(last['연도'])}년 GDP 갭", f"{last['GDP 갭(연평균, %)']:+.2f}%")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.ts_lines(q, "라벨", ["실제 성장률(전년동기비, %)", "잠재성장률(전년동기비, %)"],
                                    "실제 성장률과 잠재성장률 (전년동기비)", "%", 400, step,
                                    dashed=("실제 성장률(전년동기비, %)",), shade=_sq(df)), width="stretch")
    c2.plotly_chart(charts.ts_bar_line(q, "라벨", "GDP 갭(%)", [], "GDP 갭", "%", 400, step, shade=_sq(df)), width="stretch")
    ac = a.assign(연도=a["연도"].astype(str))
    st.plotly_chart(charts.ts_lines(ac, "연도", ["실제 성장률(%)", "잠재성장률(%)", "GDP 갭(연평균, %)"],
                                    "연간 실제 성장률·잠재성장률·GDP 갭", "%", 380, _step_a(len(ac)),
                                    dashed=("실제 성장률(%)",), shade=_sa(df)), width="stretch")
    with st.expander("연간·분기 표", expanded=False):
        table(ac)
        table(q.drop(columns=["라벨"]))
    pt = _params_table({k: e})
    if len(pt):
        st.markdown("**추정 모수**")
        table(pt.drop(columns=["방법"]))
    if k in HELP:
        st.markdown(f"**장점** {HELP[k][1]}  \n**한계** {HELP[k][2]}")
    tables = {"분기": q, "연간": a, "모수": pt}
    for name, t in e.extra.items():
        tables[name] = _qframe(t, show) if isinstance(t.index, pd.PeriodIndex) else t
    res = Result(module=MODULE, analysis=e.name, inputs=_inputs(cfg, show),
                 summary={f"잠재성장률 {lab}(%)": L["pots"][k], f"GDP 갭 {lab}(%)": L["gaps"][k]},
                 tables=tables, notes=COMMON_NOTES + _fc_notes() + e.notes, sources=sources,
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
    avail = [k for k in ("pf", "pfmv") if k in ests]
    if not avail:
        st.info("생산함수모형(생산함수 접근, 생산함수 다변량 필터)을 선택하면(자료: 고용·실업률·자본 필요) "
                "잠재성장률을 총요소생산성·자본·노동(인구·참가율·실업률·근로시간) 기여도로 나눠 보여 줍니다.")
        return
    opts = avail + (["both"] if len(avail) == 2 else [])
    pick = st.radio("모형", opts, horizontal=True, key="pg_pf_pick",
                    format_func=lambda x: "두 모형 평균(한국은행 기여도 분석 방식)" if x == "both" else M.METHODS[x])
    if pick == "both":
        cs = [ests[k].extra["잠재성장률 기여도"] for k in avail]
        cols_ = [c for c in cs[0].columns if all(c in x.columns for x in cs)]
        comp = sum(x[cols_] for x in cs) / len(cs)
        e_notes = ["두 생산함수모형(생산함수 접근, 생산함수 다변량 필터)의 기여도 평균(한국은행 권지호 외 2019·이은경 외 2024 방식)."]
        e_notes += ests["pf"].notes + ests["pfmv"].notes
        label_ = "두 모형 평균"
    else:
        comp = ests[pick].extra["잠재성장률 기여도"]
        e_notes = ests[pick].notes
        label_ = M.METHODS[pick]
    parts = [c for c in comp.columns if c != "잠재성장률(로그 근사)"]
    ann = comp.groupby(comp.index.year).mean()
    cnt = comp.groupby(comp.index.year)["자본"].count()
    ann = ann[cnt == 4].loc[int(show[:4]):]
    ann.index.name = "연도"
    ann = ann.reset_index()
    ann["연도"] = ann["연도"].astype(str)
    st.plotly_chart(charts.ts_stacked(ann, "연도", parts, "잠재성장률(로그 근사)", f"잠재성장률 기여도 (연평균, %p) — {label_}",
                                      "%p", 440, _step_a(len(ann)), shade=_sa(df)), width="stretch")
    per = _period_contrib(comp, parts, A.fc_years(df))
    if per is not None:
        st.plotly_chart(charts.ts_stacked(per, "기간", parts, "잠재성장률(로그 근사)", "기간별 기여도 (연평균, %p)", "%p", 380, 1),
                        width="stretch")
    qc = _qframe(comp, show)
    st.plotly_chart(charts.ts_stacked(qc, "라벨", parts, "잠재성장률(로그 근사)", "잠재성장률 기여도 (분기, 전년동기비, %p)",
                                      "%p", 420, _step_q(len(qc)), shade=_sq(df)), width="stretch")
    tables = {"연간 기여도": ann}
    if per is not None:
        tables["기간별 기여도"] = per
    tables["분기 기여도"] = qc
    if "pf" in ests and pick in ("pf", "both"):
        inp = _qframe(ests["pf"].extra["생산함수 투입"], show)
        stp = _step_q(len(inp))
        c1, c2 = st.columns(2)
        if "경제활동참가율 추세(%)" in inp:
            c1.plotly_chart(charts.ts_lines(inp, "라벨", ["경제활동참가율(%)", "경제활동참가율 추세(%)"], "경제활동참가율", "%", 320,
                                            stp, zero=False, shade=_sq(df)), width="stretch")
            c2.plotly_chart(charts.ts_lines(inp, "라벨", ["실업률(%)", "자연실업률(%)"], "실업률과 자연실업률", "%", 320, stp,
                                            zero=False, shade=_sq(df)), width="stretch")
        else:
            c1.plotly_chart(charts.ts_lines(inp, "라벨", ["고용률(%)", "고용률 추세(%)"], "고용률(취업자/15세이상인구)", "%", 320,
                                            stp, zero=False, shade=_sq(df)), width="stretch")
            c2.plotly_chart(charts.ts_lines(inp, "라벨", ["실업률(%)"], "실업률", "%", 320, stp, zero=False, shade=_sq(df)), width="stretch")
        c3, c4 = st.columns(2)
        if "평균 근로시간 추세" in inp:
            c3.plotly_chart(charts.ts_lines(inp, "라벨", ["평균 근로시간(연간, 시간)", "평균 근로시간 추세"],
                                            "취업자 1인당 연간 근로시간", "시간", 320, stp, zero=False, shade=_sq(df)), width="stretch")
        c4.plotly_chart(charts.ts_lines(inp, "라벨", ["실제 TFP(로그×100)", "가동률 조정 TFP", "추세 TFP"],
                                        "총요소생산성 (로그×100)", "", 320, stp, zero=False, shade=_sq(df)), width="stretch")
        kg = _capital_growth(df, show)
        if kg is not None:
            st.plotly_chart(charts.ts_lines(kg, "연도", list(kg.columns[1:]), "자본서비스와 생산자본스톡 증가율 (연간)", "%", 320,
                                            _step_a(len(kg)), shade=_sa(df)), width="stretch")
            tables["자본 증가율"] = kg
        tables["생산함수 투입"] = inp
    st.markdown("**연간 기여도 (%p)**")
    table(ann)
    res = Result(module=MODULE, analysis=f"생산함수 기여도 분해 ({label_})", inputs=_inputs(cfg, show),
                 summary={f"{r['연도']}년 {c}(%p)": r[c] for _, r in ann.tail(1).iterrows()
                          for c in parts + ["잠재성장률(로그 근사)"]},
                 tables=tables,
                 notes=COMMON_NOTES + _fc_notes() + e_notes + ["기여도는 로그 차분(전년동기비)이라 합이 잠재성장률의 근사치이며, 연간은 4개 분기 평균."],
                 sources=sources,
                 charts=[ExcelChart("연간 기여도", "연도", parts, "잠재성장률 기여도 (%p)", kind="col", top=None)])
    excel_button(res, "pg_x_pf")
    notes_box(res)


def _period_contrib(comp: pd.DataFrame, parts: list[str], fy=frozenset()) -> pd.DataFrame | None:
    a = comp.groupby(comp.index.year).mean()
    a = a[comp.groupby(comp.index.year)["자본"].count() == 4].dropna()
    if len(a) < 5:
        return None
    rows = []
    for s_, e_ in _bins(a.index):
        r = a.loc[s_:e_, parts + ["잠재성장률(로그 근사)"]].mean()
        r.name = _plabel(s_, e_) + ("e)" if any(y in fy for y in range(s_, e_ + 1)) else "")
        rows.append(r)
    out = pd.DataFrame(rows).round(2)
    out.index.name = "기간"
    return out.reset_index()


def _capital_growth(df: pd.DataFrame, show: str) -> pd.DataFrame | None:
    cols = {}
    if "ksvc" in df and df["ksvc"].notna().sum() > 40:
        cols["자본서비스 증가율(%)"] = M.annual_growth(M.log100(df["ksvc"]))
    if "capital" in df and df["capital"].notna().sum() > 40:
        q4 = df["capital"][df.index.quarter == 4]
        g = q4.pct_change() * 100
        g.index = g.index.year
        cols["생산자본스톡 증가율(%, 연말)"] = g
    if not cols:
        return None
    t = pd.DataFrame(cols).loc[int(show[:4]):].dropna(how="all")
    t.index.name = "연도"
    t = t.reset_index()
    t["연도"] = t["연도"].astype(str)
    return t


def nairu_tab(df, ests, cfg, show, sources):
    avail = [k for k in ("lw", "mv", "pfmv") if k in ests]
    if not avail:
        st.info("필립스곡선을 넣은 모형(LW형 준구조모형, 다변량 HP 필터, 생산함수 다변량 필터)을 선택하면 "
                "자연실업률·중립금리와 필립스곡선 추정 결과를 보여 줍니다.")
        return
    k = st.radio("모형", avail, horizontal=True, key="pg_nairu_pick", format_func=lambda x: M.METHODS[x])
    e = ests[k]
    src = e.extra["중립금리·자연실업률"] if k == "lw" else e.extra["자연실업률"]
    t = _qframe(src, show)
    t["실업률 갭(%p)"] = t["실업률(%)"] - t["자연실업률 추정(%)"]
    t["GDP 갭(%)"] = e.gap.reindex(src.loc[show:].index).to_numpy()
    step = _step_q(len(t))
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.ts_lines(t, "라벨", ["실업률(%)", "자연실업률 추정(%)"], "실업률과 자연실업률", "%", 380, step,
                                    zero=False, shade=_sq(df)), width="stretch")
    c2.plotly_chart(charts.ts_lines(t, "라벨", ["GDP 갭(%)", "실업률 갭(%p)"], "GDP 갭과 실업률 갭 (오쿤의 관계식)", "%", 380, step, shade=_sq(df)),
                    width="stretch")
    st.plotly_chart(charts.ts_lines(t, "라벨", ["근원물가 상승률(%)", "GDP 갭(%)"], "근원물가 상승률과 GDP 갭 (필립스곡선)", "%",
                                    380, step, shade=_sq(df)), width="stretch")
    charts_x = [ExcelChart("자연실업률", "라벨", ["실업률(%)", "자연실업률 추정(%)"], "실업률과 자연실업률 (%)",
                           kind="line", top=None, label_skip=step)]
    la = D.label(F.last_actual(df))
    r_ = t[t["라벨"] == la].iloc[-1] if (t["라벨"] == la).any() else t.iloc[-1]
    summ = {f"자연실업률 {r_['라벨']}(%)": r_["자연실업률 추정(%)"], f"실업률 갭 {r_['라벨']}(%p)": r_["실업률 갭(%p)"]}
    if k == "lw":
        st.plotly_chart(charts.ts_lines(t, "라벨", ["실질 기준금리(%)", "중립금리 추정(%)", "잠재성장률(연율, %)"],
                                        "실질 기준금리와 중립금리 (LW형 준구조모형)", "%", 380, step,
                                        shade=_sq(df)), width="stretch")
        st.caption("실질 기준금리 = 기준금리 − 기대인플레이션. 중립금리 = 잠재성장률(연율) + 기타 요인(z). "
                   "실질금리가 중립금리보다 높으면 긴축적, 낮으면 완화적.")
        charts_x.append(ExcelChart("자연실업률", "라벨", ["실질 기준금리(%)", "중립금리 추정(%)"], "실질 기준금리와 중립금리 (%)",
                                   kind="line", top=None, label_skip=step))
        summ[f"중립금리 {r_['라벨']}(%)"] = r_["중립금리 추정(%)"]
    table(_params_table({k: e}).drop(columns=["방법"]))
    res = Result(module=MODULE, analysis=f"자연실업률·중립금리 ({e.name})", inputs=_inputs(cfg, show), summary=summ,
                 tables={"자연실업률": t, "모수": _params_table({k: e})}, notes=COMMON_NOTES + _fc_notes() + e.notes, sources=sources,
                 charts=charts_x)
    excel_button(res, f"pg_x_nairu_{k}")
    notes_box(res)


def data_tab(df, show, sources):
    t = _qframe(df.drop(columns=[F.FC_COL], errors="ignore"), show).rename(columns=D.COLUMNS)
    if F.FC_COL in df.columns:
        t.insert(2, "구분", ["전망" if v else "실적" for v in df[F.FC_COL].loc[show:].to_numpy()])
    cols = [c for c in t.columns if c not in ("분기", "라벨", "구분")]
    pick = st.selectbox("그래프로 볼 자료", cols, key="pg_data_pick")
    st.plotly_chart(charts.ts_lines(t, "라벨", [pick], pick, "", 360, _step_q(len(t)), zero=False, shade=_sq(df)),
                    width="stretch")
    tables = {"분기 자료": t}
    info = st.session_state.get("pg_fc_info")
    if info:
        fc = info["fc"]
        st.markdown(f"**끝점 보정에 쓴 전망·가정** — {fc['title']} ({fc['published']} 발표)")
        cmp_ = info["cmp"]
        if len(cmp_):
            st.caption("실질 GDP 성장률: 한국은행 전망과 연장 경로 비교(전년동기비). 실적이 이미 나온 분기는 실적을 씀.")
            table(cmp_)
            tables["GDP 전망 비교"] = cmp_
        how = info["how"].copy()
        how.insert(0, "라벨", [D.label(p) for p in how.index])
        how.insert(0, "분기", how.index.astype(str))
        st.caption("분기별 연장 방법")
        table(how.reset_index(drop=True))
        tables["연장 방법"] = how.reset_index(drop=True)
        st.caption(f"출처: {fc['url']} ({fc['pages']})")
    table(t.drop(columns=["라벨"]))
    res = Result(module=MODULE, analysis="입력 자료", tables=tables, sources=sources,
                 notes=["자본스톡은 연말 잔액을 분기말로 로그 선형 보간(마지막 공표 연도 이후는 최근 3년 평균 증가율로 연장).",
                        "자본서비스 물량지수와 평균 근로시간은 연간 자료를 Denton 방식(분기 평균 = 연간값, 분기 간 변화 최소)으로 분기화.",
                        "평균 근로시간은 OECD(2008년~)에 Penn World Table 증가율을 거슬러 이었고, 당해 연도는 사업체노동력조사 증가율로 연장."]
                 + _fc_notes())
    excel_button(res, "pg_x_data")
    for s in sources:
        st.caption(s)


def help_tab():
    st.markdown("GDP 갭과 잠재성장률은 직접 관측되지 않으므로 방법마다 결과가 다릅니다. 한 방법에 기대기보다 "
                "여러 방법의 **범위**와 **평균**을 함께 보는 것이 일반적입니다. 한국은행은 생산함수모형·다변량필터링모형·준구조모형·"
                "HP필터 4개 모형의 평균을 발표합니다(권지호 외 2019, 이은경 외 2024).")
    st.markdown(
        "- **통계모형**: 실질 GDP 시계열의 통계적 성질만으로 추세를 뽑음. 간단하지만 경제 이론 정보가 없음.\n"
        "- **준구조모형**: GDP 갭과 물가(필립스곡선)·고용(오쿤의 관계식)·금리(IS곡선) 사이의 관계를 함께 추정해, "
        "'물가를 자극하지 않는 산출'이라는 잠재 GDP 정의에 가깝게 갭을 구함.\n"
        "- **생산함수모형**: 잠재 GDP 를 총요소생산성·노동·자본의 잠재 수준으로 구성. 잠재성장률을 요인별로 분해할 수 있음. "
        "노동은 15세이상인구·경제활동참가율·자연실업률·평균 근로시간, 자본은 자본서비스 물량을 씀.")
    st.markdown("**끝점 보정(한국은행 전망 연장)**  \n"
                "HP 필터·칼만 평활기 등은 표본 끝에서 최근 실적에 끌려가 추세가 흔들립니다(끝점 문제). 한국은행은 자체 GDP 전망으로 "
                "자료를 연장한 뒤 잠재성장률을 추정합니다. 이 도구도 한국은행 경제전망보고서의 분기 경로·반기(상반·하반, 전년동기비)·"
                "연간 성장률로 실질 GDP 를 전망 시계 끝(다음 해 4분기)까지 연장하고, 생산함수·준구조모형의 입력 변수도 같은 시계로 맞춥니다: "
                "취업자·실업률·물가는 전망치, 15세 이상 인구·근로시간·자본서비스는 최근 3년 추세 연장, 가동률은 최근 평균, "
                "기준금리·기대인플레이션은 마지막 값 유지. 전망 구간은 차트에 음영으로, 표에 '전망'으로 표시합니다. "
                "새 경제전망(2·5·8·11월)이 나오면 전망 숫자를 바꿔 다시 추정합니다.")
    rows = [{"분류": M.GROUP[k], "방법": M.METHODS[k], "아이디어": i, "장점": p, "한계": c} for k, (i, p, c) in HELP.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=470)
    st.markdown("**한국은행 방식과의 차이**  \n"
                "- 노동의 질(성·연령·학력 구성) 지수는 미시자료가 필요해 넣지 않았음 → 그 효과는 총요소생산성에 포함.  \n"
                "- 근로시간은 OECD·PWT·사업체노동력조사로 구성(한국은행 2024년은 경제활동인구조사 미시자료로 집단별 구성).  \n"
                "- 준구조모형의 기대인플레이션은 일반인 1년 기대(원논문은 전문가 5년 기대), 팬데믹 기간 이분산 처리는 하지 않음.  \n"
                "- 집단별(성·연령·학력 20개 집단) 고용률·근로시간 추세 대신 집계 자료의 추세를 씀.")
    st.markdown("**참고 문헌**  \n" + REFS)


# ---------------------------------------------------------------------------
def render():
    page_header("잠재성장률 추정",
                "통계모형·준구조모형·생산함수모형 3개 모형군 12가지 방법으로 잠재 GDP 를 추정해 "
                "실제 성장률·잠재성장률·GDP 갭을 비교합니다(한국은행 잠재성장률 추정 방법론 참고).")
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
    df, ests = _trim(df, ests)
    for k, msg in errs.items():
        st.warning(f"{M.METHODS.get(k, k)}: {msg}")
    if not ests:
        st.error("추정 결과가 없습니다.")
        return
    _cards(df, ests)
    tabs = st.tabs(["방법 비교", "방법별 상세", "생산함수 기여도", "자연실업률·중립금리", "입력 자료", "방법 설명"])
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
