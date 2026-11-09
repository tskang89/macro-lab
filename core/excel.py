"""Result → 서식 갖춘 엑셀 파일(bytes).

시트 구성: 요약 · 입력조건 · (결과 표들) · 방법론·주석
  - 머리글 색, 틀 고정, 자동 필터, 열 너비, 숫자 서식
  - ExcelChart 지정 시 해당 표 시트 옆에 편집 가능한 Excel 기본 차트
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from core.result import Result

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=14, color="1F3864")
SUB_FONT = Font(bold=True, size=11, color="1F3864")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(bottom=THIN)


def _sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\\/*?:\[\]]", "", name)[:31] or "표"
    nm, k = base, 2
    while nm in used:
        suffix = f"_{k}"
        nm = base[: 31 - len(suffix)] + suffix
        k += 1
    used.add(nm)
    return nm


def _num_format(col: str, s: pd.Series) -> str | None:
    if not pd.api.types.is_numeric_dtype(s):
        return None
    if "(%" in col or "%p" in col or col.endswith("%") or "변화율" in col:
        return "0.00"
    if "명/" in col:
        return "0.00"
    if "(명" in col:
        return "#,##0"
    if "계수" in col or "승수" in col:
        return "0.0000"
    vals = s.replace([np.inf, -np.inf], np.nan).dropna().abs()
    if len(vals) and vals.max() >= 1000:
        return "#,##0"
    return "#,##0.000"


def _write_table(ws, df: pd.DataFrame, start_row: int = 1):
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(row=start_row, column=j, value=str(col).strip())
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    fmts = [_num_format(str(col), df[col]) for col in df.columns]
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.floating, float)) and not np.isfinite(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            c = ws.cell(row=i, column=j, value=v)
            if fmts[j - 1]:
                c.number_format = fmts[j - 1]
    for j, col in enumerate(df.columns, 1):
        width = max([len(str(col).strip()) * 1.6] +
                    [len(str(v)) * (1.6 if isinstance(v, str) else 1.0) for v in df[col].head(200)])
        ws.column_dimensions[get_column_letter(j)].width = min(max(8, width + 2), 42)
    ws.row_dimensions[start_row].height = 32
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)
    if len(df):
        ws.auto_filter.ref = f"A{start_row}:{get_column_letter(len(df.columns))}{start_row + len(df)}"


def _add_chart(wb, ws, df: pd.DataFrame, spec):
    if spec.category not in df.columns or not all(v in df.columns for v in spec.values):
        return
    d = df[[spec.category] + spec.values].copy()
    if spec.top and len(d) > spec.top:
        d = d.reindex(d[spec.values[0]].abs().sort_values(ascending=False).index[: spec.top])
    # 차트용 보조 표: 원 표 오른쪽에 둔다
    col0 = len(df.columns) + 3
    ws.cell(row=1, column=col0, value="차트 자료").font = SUB_FONT
    for j, name in enumerate([spec.category] + spec.values):
        ws.cell(row=2, column=col0 + j, value=name).font = Font(bold=True)
    for i, row in enumerate(d.itertuples(index=False), 3):
        for j, v in enumerate(row):
            ws.cell(row=i, column=col0 + j, value=v.item() if isinstance(v, np.generic) else v)
    n = len(d)
    if spec.kind == "line":
        ch = LineChart()
    else:
        ch = BarChart()
        ch.type = "bar" if spec.kind == "bar" else "col"
        if spec.kind == "bar":
            ch.y_axis.scaling.orientation = "minMax"
            ch.x_axis.scaling.orientation = "maxMin"
    ch.title = spec.title or spec.values[0]
    ch.height, ch.width = max(7.5, 0.45 * n + 2), 18
    data = Reference(ws, min_col=col0 + 1, max_col=col0 + len(spec.values), min_row=2, max_row=2 + n)
    cats = Reference(ws, min_col=col0, min_row=3, max_row=2 + n)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.legend.position = "b"
    ws.add_chart(ch, ws.cell(row=2, column=col0 + len(spec.values) + 2).coordinate)


def to_excel(res: Result) -> bytes:
    wb = Workbook()
    used: set[str] = set()

    # 요약
    ws = wb.active
    ws.title = _sheet_name("요약", used)
    ws["A1"] = res.title
    ws["A1"].font = TITLE_FONT
    ws["A2"] = f"작성: {res.created:%Y-%m-%d %H:%M}  ·  거시 분석 플랫폼(macro-lab)"
    ws["A2"].font = Font(color="7F7F7F")
    r = 4
    ws.cell(row=r, column=1, value="핵심 결과").font = SUB_FONT
    r += 1
    for k, v in res.summary.items():
        ws.cell(row=r, column=1, value=k)
        c = ws.cell(row=r, column=2, value=float(v) if isinstance(v, (int, float, np.number)) and np.isfinite(v) else (None if isinstance(v, float) else v))
        c.number_format = "#,##0.000"
        c.border = BORDER
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="표 목록").font = SUB_FONT
    r += 1
    for name in res.tables:
        ws.cell(row=r, column=1, value=name)
        r += 1
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 18

    # 입력 조건
    ws = wb.create_sheet(_sheet_name("입력조건", used))
    rows = []
    for k, v in res.inputs.items():
        if isinstance(v, pd.DataFrame):
            continue
        rows.append({"항목": k, "값": v if isinstance(v, (int, float, str)) else str(v)})
    _write_table(ws, pd.DataFrame(rows, columns=["항목", "값"]))
    r = len(rows) + 3
    for k, v in res.inputs.items():
        if isinstance(v, pd.DataFrame) and len(v):
            ws.cell(row=r, column=1, value=k).font = SUB_FONT
            _write_table(ws, v.reset_index(drop=True), start_row=r + 1)
            ws.freeze_panes = "A2"
            r += len(v) + 3

    # 결과 표
    sheet_of = {}
    for name, df in res.tables.items():
        ws = wb.create_sheet(_sheet_name(name, used))
        sheet_of[name] = ws
        _write_table(ws, df.reset_index(drop=True))
    for spec in res.charts:
        if spec.table in sheet_of:
            _add_chart(wb, sheet_of[spec.table], res.tables[spec.table].reset_index(drop=True), spec)

    # 방법론·주석
    ws = wb.create_sheet(_sheet_name("방법론·주석", used))
    ws["A1"] = "방법론·가정"
    ws["A1"].font = SUB_FONT
    r = 2
    for note in res.notes:
        ws.cell(row=r, column=1, value=f"• {note}").alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="자료 출처").font = SUB_FONT
    r += 1
    for s in res.sources:
        ws.cell(row=r, column=1, value=f"• {s}").alignment = Alignment(wrap_text=True)
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="이 결과는 개인 연구용 분석 도구의 산출물이며 어떤 기관의 공식 견해도 아닙니다.").font = Font(italic=True, color="7F7F7F")
    ws.column_dimensions["A"].width = 120

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
