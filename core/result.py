"""모든 분석 모듈이 돌려주는 공통 결과 형식.

화면(core.ui)과 엑셀(core.excel)이 이 형식만 알면 되므로, 새 모듈(잠재성장률·DSGE·HANK)은
Result 를 채워 돌려주기만 하면 요약 카드·표·엑셀 다운로드가 자동으로 붙는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class ExcelChart:
    """엑셀에 넣을 기본 차트 (편집 가능한 Excel 네이티브 차트)."""
    table: str                 # Result.tables 의 키
    category: str              # 범주 열 이름
    values: list[str]          # 값 열 이름들
    title: str = ""
    kind: str = "bar"          # bar | col | line
    top: int | None = 20       # 절댓값 기준 상위 n개만 (None 이면 전부)


@dataclass
class Result:
    module: str                                   # 모듈 이름 (예: "산업연관표 분석")
    analysis: str                                 # 분석 유형 (예: "수요 파급")
    inputs: dict = field(default_factory=dict)    # 입력 조건 {항목: 값}
    summary: dict = field(default_factory=dict)   # 핵심 지표 {이름: 숫자}
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    charts: list[ExcelChart] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)

    @property
    def title(self) -> str:
        return f"{self.module} — {self.analysis}"

    def filename(self) -> str:
        stamp = self.created.strftime("%Y%m%d_%H%M")
        safe = "".join(ch for ch in f"{self.module}_{self.analysis}" if ch not in '\\/:*?"<>| ')
        return f"{safe}_{stamp}.xlsx"
