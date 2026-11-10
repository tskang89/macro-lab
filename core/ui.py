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



def row_editor(key: str, df_init: pd.DataFrame, cols: dict, note: str = "") -> pd.DataFrame:
    """시나리오 입력 줄 편집기. 표(data_editor) 대신 줄마다 드롭다운·숫자 칸을 둔다(한 번 누르면 열림).

    cols = {열 이름: {"options": [...]} (목록) 또는 {"num": True, "min", "max", "step", "fmt"}, "w": 폭}
    st.session_state[key] 를 지우면 df_init 으로 다시 채운다(예시 불러오기·파일 불러오기).
    """
    nk = f"{key}__nid"
    if not isinstance(st.session_state.get(key), list):
        nid = st.session_state.get(nk, 0)
        rows = []
        for r in df_init.to_dict("records"):
            rows.append({"_id": nid, **r})
            nid += 1
        st.session_state[key] = rows
        st.session_state[nk] = nid
    rows = st.session_state[key]
    names = list(cols)
    widths = [cols[n].get("w", 2) for n in names] + [0.45]
    hdr = st.columns(widths, gap="small")
    for c, n in zip(hdr, names):
        c.markdown(f"<div style='font-size:0.85rem;font-weight:600;opacity:0.75'>{n}</div>", unsafe_allow_html=True)
    out, drop = [], None
    for row in rows:
        cs = st.columns(widths, gap="small", vertical_alignment="center")
        rec = {}
        for c, n in zip(cs, names):
            sp, wk = cols[n], f"{key}__{row['_id']}__{n}"
            if "options" in sp:
                opts = sp["options"]
                v = row.get(n)
                val = c.selectbox(n, opts, index=opts.index(v) if v in opts else 0, key=wk,
                                  label_visibility="collapsed")
            else:
                lo, hi = sp.get("min"), sp.get("max")
                v = row.get(n)
                v = 0.0 if v is None or pd.isna(v) else float(v)
                v = max(lo, v) if lo is not None else v
                v = min(hi, v) if hi is not None else v
                val = c.number_input(n, value=v, min_value=lo, max_value=hi, step=sp.get("step", 1.0),
                                     format=sp.get("fmt", "%.2f"), key=wk, label_visibility="collapsed")
            rec[n] = val
            row[n] = val
        if cs[-1].button("", icon=":material/close:", key=f"{key}__{row['_id']}__del", help="이 줄 지우기"):
            drop = row["_id"]
        out.append(rec)
    if drop is not None:
        st.session_state[key] = [r for r in rows if r["_id"] != drop]
        st.rerun()
    a, b = st.columns([1, 4], vertical_alignment="center")
    if a.button("줄 추가", icon=":material/add:", key=f"{key}__add"):
        new = {n: (rows[-1][n] if rows else (cols[n]["options"][0] if "options" in cols[n] else 0.0)) for n in names}
        st.session_state[key] = rows + [{"_id": st.session_state[nk], **new}]
        st.session_state[nk] += 1
        st.rerun()
    if note:
        b.caption(note)
    return pd.DataFrame(out, columns=names)


def run_gate(key: str, params):
    """분석 실행 버튼. 누른 시점의 입력(params)을 기억해 돌려준다. 아직 안 눌렀으면 None."""
    sk = f"{key}__ran"
    if st.button("분석 실행", type="primary", icon=":material/play_arrow:", key=f"{key}__run"):
        st.session_state[sk] = params
    ran = st.session_state.get(sk)
    if ran is None:
        st.info("입력을 정한 뒤 **분석 실행**을 누르세요.")
        return None
    if ran != params:
        st.warning("입력이 바뀌었습니다. 아래 결과는 직전 실행 기준입니다. **분석 실행**을 눌러야 새 입력이 반영됩니다.")
    return ran
