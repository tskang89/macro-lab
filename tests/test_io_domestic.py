"""국내 산업연관 엔진 검증: 한국은행 공표 유발계수 대조 + 모형 항등식."""
import numpy as np
import pytest

from engines.io import domestic as D

CASES = [(lv, y) for lv in D.LEVELS for y in D.available_years(lv)]


@pytest.mark.parametrize("level,year", CASES)
def test_bok_published_coefficients(level, year):
    t = D.load_table(level, year)
    # 행·열 항등식
    assert np.allclose(t.Z.sum(1) + t.F.sum(1) - t.imports, t.x, rtol=1e-6, atol=5)
    # 부가가치·수입 유발계수 (열합) = 공표치
    assert np.allclose(t.v @ t.L, t.extra["Lv_bok"].sum(0), atol=5e-5)
    assert np.allclose(t.m @ t.L, t.extra["Lm_bok"].sum(0), atol=5e-5)
    # 부가가치 + 수입 유발 = 1
    assert np.allclose(t.v @ t.L + t.m @ t.L, 1.0, atol=1e-4)
    # 취업유발 = diag(e) L
    assert np.allclose(np.diag(t.e) @ t.L, t.Le, atol=1e-3)


@pytest.mark.parametrize("level,year", [("대분류", "2022"), ("중분류", "2020")])
def test_domestic_final_demand_consistent(level, year):
    t = D.load_table(level, year)
    fd = (np.eye(t.n) - t.Ad) @ t.x
    # 국산 최종수요 합계가 (I-Ad)x 와 맞는지 (클리핑 영향 1% 이내)
    assert np.allclose(t.Fd.sum(1), fd, atol=1)
    # L fd = x
    assert np.allclose(t.L @ t.Fd.sum(1), t.x, rtol=1e-6, atol=10)


def test_price_homogeneity():
    t = D.load_table("대분류", "2022")
    ones = np.full(t.n, 10.0)
    tab, summ = D.price_effect(t, ones, {nm: ones for nm in D.VA_NAMES})
    assert np.allclose(tab["생산자가격 변화(%)"], 10.0, atol=1e-3)
    assert abs(summ["소비자물가(가계소비 가중, %)"] - 10.0) < 1e-3


def test_demand_linear_and_va_identity():
    t = D.load_table("대분류", "2022")
    dF, dFd = D.build_demand_shock(t, [{"sector": D.ALL, "component": "수출", "unit": "%", "value": 10}])
    tab = D.demand_effect(t, dF, dFd)
    # 수출 10% → 국산 최종수요 변화 = 수출의 10%
    assert np.isclose(tab["국산 최종수요 변화"].sum() * 1e3, 0.1 * t.F[:, D.EXPORT].sum())
    # 부가가치 + 중간재 수입 = 국산 최종수요 (수출은 전부 국산)
    assert np.isclose(tab["부가가치 변화"].sum() + tab["수입 변화(품목별)"].sum(), tab["최종수요 변화"].sum(), rtol=1e-4)
    # 금액 입력: 반도체 아닌 대분류 C09 수출 1조원
    dF, dFd = D.build_demand_shock(t, [{"sector": "C09", "component": "수출", "unit": "십억원", "value": 1000}])
    tab = D.demand_effect(t, dF, dFd)
    assert np.isclose(tab["생산 변화"].sum(), t.L[:, list(t.codes).index("C09")].sum() * 1000)


def test_supply_shapes():
    t = D.load_table("대분류", "2022")
    cut = np.zeros(t.n)
    cut[list(t.codes).index("C09")] = 10
    tab, summ = D.supply_effect(t, cut)
    j = list(t.codes).index("C09")
    assert np.isclose(tab["전방 파급 후 생산 변화"].iloc[j], tab["직접 생산 차질"].iloc[j])
    assert summ["전방 파급 부가가치(조원)"] < 0 and summ["후방 파급 부가가치(조원)"] < 0


def test_influence_matches_bok():
    """영향력계수: 열합/열합평균 — 공표 계수표의 열합 행에서 계산한 값과 일치."""
    t = D.load_table("대분류", "2022")
    m = D.multipliers(t, extraction=False)
    col = t.L.sum(0)
    assert np.allclose(m["영향력계수"], col / col.mean())
