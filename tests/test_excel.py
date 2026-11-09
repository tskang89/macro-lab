"""엑셀 출력기: 시트 구성·차트·숫자 확인."""
import io

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from core.excel import to_excel
from core.result import ExcelChart, Result
from engines.io import domestic as D


def test_excel_roundtrip():
    t = D.load_table("대분류", "2022")
    dF, dFd = D.build_demand_shock(t, [{"sector": D.ALL, "component": "수출", "unit": "%", "value": -10}])
    tab = D.demand_effect(t, dF, dFd)
    res = Result(module="산업연관표 분석", analysis="수요 파급",
                 inputs={"자료": "2022", "충격 목록": pd.DataFrame([{"부문": "전 부문", "크기": -10}])},
                 summary=D.demand_summary(t, tab), tables={"부문별 결과": tab},
                 notes=["테스트"], sources=["한국은행"],
                 charts=[ExcelChart("부문별 결과", "부문", ["부가가치 변화"], "부가가치", "bar", 10)])
    wb = load_workbook(io.BytesIO(to_excel(res)))
    assert wb.sheetnames == ["요약", "입력조건", "부문별 결과", "방법론·주석"]
    ws = wb["부문별 결과"]
    assert ws.max_row == t.n + 1 or ws.max_row >= t.n + 1
    assert len(ws._charts) == 1
    header = [c.value for c in ws[1]]
    j = header.index("부가가치 변화") + 1
    total = sum(ws.cell(row=i, column=j).value for i in range(2, t.n + 2))
    assert np.isclose(total, tab["부가가치 변화"].sum())
