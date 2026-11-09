"""잠재성장률 엔진 검증."""
import io

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from core.excel import to_excel
from core.result import ExcelChart, Result
from engines.potential import analysis as A
from engines.potential import data as D
from engines.potential import methods as M

IDX = pd.period_range("1990Q1", periods=140, freq="Q")


def test_hp_keeps_linear_trend():
    x = pd.Series(100 + 0.5 * np.arange(len(IDX)), index=IDX)
    e = M.hp(x, 1600, extend=0)
    assert np.allclose(e.trend, x, atol=1e-8)
    assert np.allclose(e.gap, 0, atol=1e-8)


def test_linear_breaks_recover_slopes():
    t = np.arange(len(IDX))
    x = pd.Series(np.where(t < 60, 1.0 * t, 60 + 0.5 * (t - 60)), index=IDX)
    e = M.linear(x, [str(IDX[60])])
    assert np.allclose(e.gap, 0, atol=1e-8)


def test_hamilton_deterministic():
    x = pd.Series(0.7 * np.arange(len(IDX)), index=IDX)
    e = M.hamilton(x)
    assert np.nanmax(np.abs(e.gap.to_numpy())) < 1e-8
    assert e.gap.isna().sum() == 8 + 4 - 1


def test_growth_helpers():
    x = pd.Series(100 * np.log(1.01) * np.arange(len(IDX)), index=IDX)   # 분기 1%씩 성장
    assert np.allclose(M.yoy(x).dropna(), (1.01 ** 4 - 1) * 100)
    assert np.allclose(M.annual_growth(x).dropna(), (1.01 ** 4 - 1) * 100)


@pytest.fixture(scope="module")
def kr():
    df, meta = D.load_korea()
    return df


@pytest.fixture(scope="module")
def ests(kr):
    e, errs = A.run(kr, {})
    assert not errs, errs
    return e


def test_all_methods_run(ests):
    assert set(M.METHODS) | {"avg"} == set(ests)


def test_recent_potential_growth_plausible(kr, ests):
    ga, gp = A.annual_tables(kr, ests, "2015Q1")
    row = ga.set_index("연도").loc[2024]
    for k, e in ests.items():
        if k in ("linear", "hamilton", "svar"):     # 추세가 거칠거나 구간 고정인 방법 제외
            continue
        assert 1.0 < row[e.name] < 3.5, (e.name, row[e.name])
    # 코로나 충격 2020년 갭은 대부분 방법에서 마이너스
    neg = (gp.set_index("연도").loc[2020].drop(labels=["방법 평균"]) < 0).mean()
    assert neg >= 0.8


def test_mv_signs(ests):
    p = ests["mv"].params
    assert p["필립스곡선 갭 계수 κ"] > 0 and p["오쿤 계수 θ"] > 0
    nairu = ests["mv"].extra["자연실업률"]["자연실업률 추정(%)"]
    assert 2 < nairu.iloc[-1] < 5
    # HP 갭과 같은 방향
    both = pd.concat([ests["mv"].gap, ests["hp"].gap], axis=1).dropna()
    assert np.corrcoef(both.T)[0, 1] > 0.7


def test_pf_contributions_add_up(ests):
    e = ests["pf"]
    comp = e.extra["잠재성장률 기여도"].dropna()
    dtr = e.trend.diff(4).reindex(comp.index)
    assert np.allclose(comp["잠재성장률(로그 근사)"], dtr, atol=1e-8)


def test_svar_long_run_restriction(ests):
    assert abs(ests["svar"].params["수요 충격의 GDP 장기 효과(0 이어야 함)"]) < 1e-8


def test_ensemble_is_mean(ests):
    keys = [k for k in ests if k != "avg"]
    G = pd.concat([ests[k].gap for k in keys], axis=1).mean(axis=1)
    assert np.allclose(G.dropna(), ests["avg"].gap.dropna())


def test_upload_parse_and_univariate_only():
    csv = "quarter,gdp\n" + "\n".join(f"{p.year}Q{p.quarter},{100 * 1.01 ** i:.4f}" for i, p in enumerate(IDX))
    df = D.parse_upload(csv.encode())
    assert df.index.freqstr.startswith("Q") and len(df) == len(IDX)
    e, errs = A.run(df, {"start": "1990Q1"})
    assert {"mv", "pf", "svar"} <= set(errs)
    assert "hp" in e and "avg" in e


def test_excel_time_series_chart(kr, ests):
    g, gap, lev = A.quarterly_tables(kr, ests, "2010Q1")
    res = Result(module="잠재성장률 추정", analysis="테스트", tables={"GDP갭": gap},
                 charts=[ExcelChart("GDP갭", "라벨", ["HP 필터", "방법 평균"], kind="line", top=None, label_skip=4)])
    wb = load_workbook(io.BytesIO(to_excel(res)))
    ws = wb["GDP갭"]
    assert len(ws._charts) == 1
    ch = ws._charts[0]
    assert ch.x_axis.tickLblSkip == 4
    # 차트 자료의 마지막 라벨 = 마지막 관측 분기, 첫 라벨부터 4칸 간격으로 닿는다
    col = len(gap.columns) + 3
    labels = [c.value for c in next(ws.iter_cols(min_col=col, max_col=col, min_row=3))]
    labels = [v for v in labels if v]
    assert labels[-1] == gap["라벨"].iloc[-1] and (len(labels) - 1) % 4 == 0
