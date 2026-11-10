"""중립금리 엔진 검증."""
import numpy as np
import pandas as pd
import pytest

from engines import neutral as N
from engines.potential import data as D
from engines.potential import methods as M


@pytest.fixture(scope="module")
def kr():
    df, _ = D.load_korea()
    return df.drop(columns=["pie"])          # 기본: 근원물가 직전 4분기 평균


def test_real_rate_definition(kr):
    r = N.real_rate(kr)
    pe = M.expected_inflation(kr)
    q = r.index[-1]
    assert np.isclose(r[q], kr.loc[q, "rate"] - pe[q])


def test_market_forward_formula(kr):
    e = N.market(kr, term_premium=0.5)
    q = e.rstar.index[-1]
    f = (10 * kr.loc[q, "y10"] - 5 * kr.loc[q, "y5"]) / 5
    assert np.isclose(e.rstar[q], f - 2.0 - 0.5)
    assert N.inflation_target(pd.PeriodIndex(["2010Q1"], freq="Q")).iloc[0] == 3.0


def test_growth_linked_is_regression(kr):
    g = pd.Series(np.linspace(5, 2, len(kr)), index=kr.index)
    e = N.growth_linked(kr, g)
    s, rho = e.params["성장률 계수 σ"], e.params["상수 ρ(%)"]
    q = e.rstar.index[-1]
    assert np.isclose(e.rstar[q], rho + s * g[q])


def test_all_methods_plausible(kr):
    ests = {"hp": N.hp_trend(kr), "uc": N.uc(kr), "lw": N.lw(kr), "var": N.var_forecast(kr), "market": N.market(kr)}
    E = N.ensemble(ests)
    last = E["평균"].dropna().iloc[-8:]
    assert (last > -2.5).all() and (last < 3).all()          # 실질 중립금리 최근 2년이 그럴듯한 범위
    assert (E["최소"] <= E["평균"] + 1e-9).all() and (E["평균"] <= E["최대"] + 1e-9).all()
    assert ests["uc"].params["순환 AR(1)"] <= 0.9 + 1e-9
    early = E.loc["2003Q1":"2007Q4", "평균"].mean()
    late = E.loc["2016Q1":"2021Q4", "평균"].mean()
    assert early > late                                        # 중립금리 하락 추세


def test_var_uses_window(kr):
    e = N.var_forecast(kr, window=40)
    first = kr["rate"].dropna().index[0]
    assert e.rstar.index[0] >= first + 39
    assert e.rstar.notna().all()
