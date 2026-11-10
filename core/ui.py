"""공통 화면 요소: 결과 카드, 표, 엑셀 다운로드, 시나리오 저장·불러오기."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core.excel import to_excel
from core.result import Result

DISCLAIMER = ("개인 연구용 분석 도구입니다. 결과는 모형 가정에 크게 좌우되며 어떤 기관의 공식 견해도 아닙니다.")


def page_header(title: str, caption: str = ""):
    st.title(title)
    if caption:
        st.caption(caption)


def fmt_value(k: str, v, signed: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, str):
        return v
    sg = "+" if signed else ""
    if "%" in k:
        return f"{v:{sg}.2f}%" if "%p" not in k else f"{v:{sg}.2f}%p"
    if "명" in k:
        return f"{v:{sg},.2f}"
    if abs(v) >= 100:
        return f"{v:{sg},.1f}"
    return f"{v:{sg},.3f}"


LEVEL_WORDS = ("승수", "교역액", "총수출", "총산출", "총부가가치")


def summary_cards(summary: dict, per_row: int = 4, signed: bool = True):
    """signed=False: 수준(level) 지표. True: 변화 지표(부호 표시)."""
    items = list(summary.items())
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (k, v) in zip(cols, items[i:i + per_row]):
            col.metric(k, fmt_value(k, v, signed and not any(w in k for w in LEVEL_WORDS)))


def table(df: pd.DataFrame, height: int | None = None, key: str | None = None):
    num = df.select_dtypes("number").columns
    cfg = {}
    for c in num:
        if "(%" in c or "%p" in c or "변화율" in c:
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
        elif "명/" in c:
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
        elif "(명" in c:
            cfg[c] = st.column_config.NumberColumn(c, format="%,.0f")
        elif "계수" in c or "승수" in c:
            cfg[c] = st.column_config.NumberColumn(c, format="%.4f")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%,.2f")
    st.dataframe(df, width="stretch", hide_index=True, column_config=cfg,
                 height=height or min(38 * (len(df) + 1) + 4, 560), key=key)


def excel_button(res: Result, key: str):
    st.download_button("⬇ 엑셀로 내려받기", data=lambda: to_excel(res), file_name=res.filename(),
                       on_click="ignore",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=key, type="primary", width="content")


def notes_box(res: Result):
    with st.expander("방법론·가정·자료 출처"):
        for n in res.notes:
            st.markdown(f"- {n}")
        if res.sources:
            st.markdown("**자료 출처**")
            for s in res.sources:
                st.markdown(f"- {s}")
        st.caption(DISCLAIMER)


def scenario_io(current, load_into: str, reset_keys: tuple[str, ...] = (), label: str = "시나리오"):
    """현재 시나리오(current: list/dict)를 JSON 으로 저장하거나, 올린 파일을 session_state[load_into] 에 넣는다."""
    with st.expander(f"{label} 저장·불러오기"):
        st.download_button("현재 설정을 파일로 저장(JSON)",
                           data=json.dumps(current, ensure_ascii=False, indent=1, default=str),
                           file_name=f"{load_into}.json", mime="application/json", key=f"{load_into}_savejson",
                           on_click="ignore")
        up = st.file_uploader("저장한 설정 파일 불러오기", type="json", key=f"{load_into}_upload")
        if up is not None and st.session_state.get(f"{load_into}_loaded_id") != up.file_id:
            try:
                st.session_state[load_into] = json.loads(up.getvalue().decode("utf-8"))
                st.session_state[f"{load_into}_loaded_id"] = up.file_id
                for k in reset_keys:
                    st.session_state.pop(k, None)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"파일을 읽지 못했습니다: {e}")


def footer():
    st.divider()
    st.caption(DISCLAIMER)


PICK_HELP = "칸을 두 번 누르면 목록이 펼쳐집니다. 글자를 쳐서 찾을 수도 있습니다."
NUM_HELP = "칸을 두 번 누르고 숫자를 입력합니다."


def pick_col(label: str, **kw):
    """data_editor 의 목록 선택 열. 표 안에서는 화살표가 안 보이므로 머리글에 ▼ 를 붙인다."""
    return st.column_config.SelectboxColumn(f"{label} ▼", help=kw.pop("help", PICK_HELP), **kw)


def num_col(label: str, **kw):
    """data_editor 의 숫자 입력 열. 머리글에 ✎ 를 붙인다."""
    return st.column_config.NumberColumn(f"{label} ✎", help=kw.pop("help", NUM_HELP), **kw)


def editor_hint():
    st.caption("▼ 열은 목록에서 고르고, ✎ 열은 숫자를 입력합니다(칸을 두 번 누름). "
               "맨 아래 빈 줄을 누르면 줄이 추가되고, 줄 왼쪽을 골라 Delete 키로 지웁니다.")
